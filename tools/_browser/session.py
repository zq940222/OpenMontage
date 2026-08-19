"""Persistent Chromium session shared by browser-driven provider tools.

A ``BrowserSession`` opens the provider's own Chromium profile — the one the
user logged into once via ``python -m tools._browser login <provider>`` — and
hands back a Playwright page. Because the profile carries the login cookies,
generation runs on the user's subscription with no API key involved.

Design constraints this module exists to satisfy:

* **Lazy Playwright import.** ``registry.discover()`` imports every module under
  ``tools/``; a top-level ``import playwright`` would break discovery on a
  machine that doesn't have it.
* **One process at a time per profile.** Chromium refuses to open the same
  ``user_data_dir`` twice, and OpenMontage generates scenes from parallel
  threads. A thread lock plus a cross-process file lock serializes access
  instead of crashing halfway through a batch.
* **Debuggable failures.** Web UIs drift. Every failure can dump a screenshot
  and the page HTML so a broken selector is a five-minute fix, not a mystery.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tools._browser.selectors import selectors_for

__all__ = [
    "BrowserSession",
    "LoginState",
    "debug_dir_for",
    "login_state",
    "profile_dir_for",
    "selectors_for",
    "PROVIDER_URLS",
]


# Home page each provider's login CLI opens, and which a tool navigates to.
PROVIDER_URLS: dict[str, str] = {
    "gemini": "https://gemini.google.com/app",
    "suno": "https://suno.com/create",
}

_LOGIN_MARKER = "openmontage_login.json"
_LOCK_FILE = "openmontage.lock"
# A held lock older than this is treated as abandoned (crashed run).
_LOCK_STALE_SECONDS = 1800.0

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _base_dir() -> Path:
    """Root for browser profiles — outside the repo, since it holds cookies."""
    override = os.environ.get("OPENMONTAGE_BROWSER_PROFILE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".openmontage" / "browser"


def profile_dir_for(provider: str) -> Path:
    return _base_dir() / provider


def debug_dir_for(provider: str) -> Path:
    return _base_dir() / "_debug" / provider


@dataclass
class LoginState:
    """What we can tell about a provider's login without opening a browser."""

    provider: str
    profile_exists: bool
    marker_exists: bool
    logged_in_at: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.profile_exists and self.marker_exists


def login_state(provider: str) -> LoginState:
    """Inspect the on-disk login markers for a provider.

    Cookies can still be expired — this only reports that a login once
    happened. Tools must also detect the logged-out DOM at execute() time.
    """
    profile = profile_dir_for(provider)
    marker = profile / _LOGIN_MARKER
    logged_in_at = None
    if marker.is_file():
        try:
            logged_in_at = json.loads(marker.read_text(encoding="utf-8")).get("logged_in_at")
        except (OSError, ValueError):
            logged_in_at = None
    return LoginState(
        provider=provider,
        profile_exists=profile.is_dir(),
        marker_exists=marker.is_file(),
        logged_in_at=logged_in_at,
    )


def write_login_marker(provider: str) -> Path:
    """Record that an interactive login completed for this provider."""
    profile = profile_dir_for(provider)
    profile.mkdir(parents=True, exist_ok=True)
    marker = profile / _LOGIN_MARKER
    marker.write_text(
        json.dumps(
            {
                "provider": provider,
                "logged_in_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "url": PROVIDER_URLS.get(provider, ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return marker


def playwright_available() -> bool:
    try:
        import playwright  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def _pid_alive(pid: int) -> bool:
    """Whether a process id is still running, without signalling it.

    ``os.kill(pid, 0)`` is not usable here: on Windows Python implements it as
    TerminateProcess, which would kill the very process we are asking about.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes  # noqa: PLC0415

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True  # can't tell — assume alive rather than steal the lock
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


class BrowserLockTimeout(RuntimeError):
    """Another run is using this provider's browser profile."""


class _ProfileLock:
    """Cross-process guard so only one Chromium opens a given profile."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        self._path = profile_dir_for(provider) / _LOCK_FILE
        self._timeout = timeout_seconds
        self._acquired = False

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._break_if_stale():
                    continue
                if time.monotonic() >= deadline:
                    raise BrowserLockTimeout(
                        f"another OpenMontage run has been holding {self._path} for "
                        f"more than {self._timeout:.0f}s. Wait for it to finish, or "
                        f"delete that file if no browser is actually running."
                    )
                time.sleep(1.0)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "since": time.time()}))
            self._acquired = True
            return

    def _break_if_stale(self) -> bool:
        """Drop the lock if its holder is gone, or if it is simply too old.

        A window closed with the X button never runs the release path, so
        without the liveness check a crashed login would block the next run for
        the full staleness window.
        """
        try:
            age = time.time() - self._path.stat().st_mtime
            holder = json.loads(self._path.read_text(encoding="utf-8"))
        except OSError:
            return True  # vanished between checks — retry the create
        except ValueError:
            holder = {}

        pid = holder.get("pid")
        abandoned = isinstance(pid, int) and not _pid_alive(pid)
        if not abandoned and age < _LOCK_STALE_SECONDS:
            return False
        try:
            self._path.unlink()
        except OSError:
            return False
        return True

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self._path.unlink()
        except OSError:
            pass
        self._acquired = False


def _thread_lock(provider: str) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(provider, threading.Lock())


class BrowserSession:
    """Context manager yielding a Playwright page on the provider's profile.

    ``with BrowserSession("gemini") as page:`` — the page is already logged in
    if the user completed ``python -m tools._browser login gemini``.
    """

    def __init__(
        self,
        provider: str,
        *,
        headless: bool = True,
        default_timeout_ms: int = 60_000,
        lock_timeout_seconds: float = 900.0,
        viewport: tuple[int, int] = (1440, 900),
    ) -> None:
        self.provider = provider
        self.headless = headless
        self.default_timeout_ms = default_timeout_ms
        self.viewport = viewport
        self.selectors = selectors_for(provider)
        self._lock = _ProfileLock(provider, lock_timeout_seconds)
        self._thread_lock = _thread_lock(provider)
        self._thread_lock_held = False
        self._playwright: Any = None
        self._context: Any = None
        self.page: Any = None

    # ---- lifecycle ----

    def __enter__(self) -> Any:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        self._thread_lock.acquire()
        self._thread_lock_held = True
        try:
            self._lock.acquire()
            self._playwright = sync_playwright().start()
            self._context = self._launch_context()
            self.page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self.page.set_default_timeout(self.default_timeout_ms)
        except BaseException:
            self.close()
            raise
        return self.page

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _launch_context(self) -> Any:
        """Open the persistent profile, preferring a real Chrome/Edge install.

        Consumer web apps treat the bundled Chromium build with more suspicion
        than a stock Chrome, so try the installed channels first and fall back
        to whatever Playwright shipped.
        """
        profile = profile_dir_for(self.provider)
        profile.mkdir(parents=True, exist_ok=True)
        channels = [c for c in (os.environ.get("OPENMONTAGE_BROWSER_CHANNEL"), "chrome", "msedge") if c]
        errors: list[str] = []
        for channel in [*channels, None]:
            kwargs: dict[str, Any] = {
                "user_data_dir": str(profile),
                "headless": self.headless,
                "viewport": {"width": self.viewport[0], "height": self.viewport[1]},
                "args": ["--disable-blink-features=AutomationControlled"],
            }
            if channel:
                kwargs["channel"] = channel
            try:
                return self._playwright.chromium.launch_persistent_context(**kwargs)
            except Exception as launch_error:  # noqa: BLE001 — try next channel
                errors.append(f"{channel or 'bundled chromium'}: {launch_error}")
        raise RuntimeError(
            "could not launch a Chromium profile. Tried "
            + "; ".join(errors)
            + ". Install browsers with: python -m playwright install chromium"
        )

    def close(self) -> None:
        for closer in (
            lambda: self._context and self._context.close(),
            lambda: self._playwright and self._playwright.stop(),
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001 — teardown must not mask the real error
                pass
        self._context = None
        self._playwright = None
        self.page = None
        self._lock.release()
        if self._thread_lock_held:
            self._thread_lock.release()
            self._thread_lock_held = False

    # ---- page helpers ----

    def first_locator(self, key: str, *, timeout_ms: int = 15_000) -> Any:
        """Return the first selector under ``key`` that actually resolves."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        candidates = self.selectors.get(key, [])
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            for selector in candidates:
                try:
                    locator = self.page.locator(selector).first
                    if locator.count() > 0:
                        return locator
                except Exception as exc:  # noqa: BLE001 — bad selector, try next
                    last_error = exc
            time.sleep(0.5)
        raise LookupError(
            f"no selector for {self.provider}.{key} matched: {candidates}. "
            f"The site's DOM probably changed — override it in "
            f"{_base_dir() / 'selectors.json'} or update tools/_browser/selectors.py."
            + (f" Last error: {last_error}" if last_error else "")
        )

    def is_logged_out(self) -> bool:
        """Best-effort logged-out detection: sign-in URL or sign-in controls."""
        try:
            url = (self.page.url or "").lower()
        except Exception:  # noqa: BLE001
            url = ""
        if "accounts.google.com" in url or "/signin" in url or "/login" in url:
            return True
        for selector in self.selectors.get("logged_out", []):
            try:
                if self.page.locator(selector).first.is_visible(timeout=1000):
                    return True
            except Exception:  # noqa: BLE001 — absent selector means "not that one"
                continue
        return False

    def logged_out_error(self) -> str:
        return (
            f"{self.provider} session is not logged in (auth issue — not a prompt "
            f"or tool bug). Fix: run `python -m tools._browser login {self.provider}`, "
            f"sign in with the account that holds your subscription, then retry."
        )

    def dump_debug(self, label: str) -> Optional[str]:
        """Save a screenshot + HTML snapshot; returns the directory."""
        if self.page is None:
            return None
        target = debug_dir_for(self.provider) / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
        try:
            target.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(target / "page.png"), full_page=True)
            (target / "page.html").write_text(self.page.content(), encoding="utf-8")
        except Exception:  # noqa: BLE001 — debug capture must never mask the error
            return None
        return str(target)
