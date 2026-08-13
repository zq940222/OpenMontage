"""One-time interactive login for browser-driven providers.

    python -m tools._browser login gemini     # opens a real browser window
    python -m tools._browser status           # who is logged in
    python -m tools._browser logout gemini    # forget the session

The login command opens the provider's page in a visible Chromium using the
same persistent profile the tools use, then watches the page until the signed-in
app appears and records the login by itself — no keypress to come back for.
Credentials are never read, typed, or stored by OpenMontage — you type them
into the browser yourself.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

from tools._browser.session import (
    PROVIDER_URLS,
    BrowserSession,
    login_state,
    playwright_available,
    profile_dir_for,
    write_login_marker,
)


def _login(provider: str, timeout_seconds: int = 600) -> int:
    url = PROVIDER_URLS.get(provider)
    if url is None:
        print(f"Unknown provider: {provider}. Known: {', '.join(sorted(PROVIDER_URLS))}")
        return 2
    if not playwright_available():
        print(
            "Playwright is not installed. Install it first:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium"
        )
        return 3

    print(f"Opening {url} in a browser window using profile {profile_dir_for(provider)}")
    print("Sign in with the account that holds your subscription.")
    print("Leave this running — the login is detected automatically; no keypress needed.")
    session = BrowserSession(provider, headless=False, default_timeout_ms=120_000)
    try:
        page = session.__enter__()
    except Exception as exc:  # noqa: BLE001 — surface launch problems verbatim
        print(f"Could not open the browser: {exc}")
        return 4
    try:
        page.goto(url, wait_until="domcontentloaded")
        if not wait_for_signed_in(session, timeout_seconds=timeout_seconds):
            print(
                f"\nStill not signed in after {timeout_seconds // 60} minutes. "
                "Nothing was recorded — run the command again when you're ready."
            )
            return 5
        marker = write_login_marker(provider)
        print(f"\nSigned in. Login recorded: {marker}")
        print("Verify anytime with: python -m tools._browser status")
        return 0
    finally:
        session.close()


def wait_for_signed_in(
    session, *, timeout_seconds: int = 600, poll_seconds: float = 2.0,
    now=time.monotonic, sleep=time.sleep, report=print,
) -> bool:
    """Poll until the provider's signed-in app is on screen.

    Absence of a sign-in link is not enough — a blank or still-loading page has
    none either. Confirmation requires the app's own prompt input to exist,
    which only renders for a signed-in session.
    """
    deadline = now() + timeout_seconds
    last_heartbeat = 0.0
    while now() < deadline:
        if not session.is_logged_out():
            try:
                session.first_locator("prompt_input", timeout_ms=1500)
                return True
            except Exception:  # noqa: BLE001 — app not ready yet; keep waiting
                pass
        remaining = int(deadline - now())
        if remaining and (last_heartbeat - remaining >= 15 or not last_heartbeat):
            report(f"  waiting for sign-in... ({remaining}s left)")
            last_heartbeat = remaining
        sleep(poll_seconds)
    return False


def _status() -> int:
    print(f"Playwright installed: {'yes' if playwright_available() else 'no'}")
    for provider in sorted(PROVIDER_URLS):
        state = login_state(provider)
        if state.ready:
            detail = f"logged in (recorded {state.logged_in_at or 'unknown time'})"
        elif state.profile_exists:
            detail = "profile exists but login was never confirmed"
        else:
            detail = f"not set up — run: python -m tools._browser login {provider}"
        print(f"  {provider:10s} {detail}")
    return 0


def _logout(provider: str) -> int:
    profile = profile_dir_for(provider)
    if not profile.exists():
        print(f"No profile for {provider} — nothing to remove.")
        return 0
    shutil.rmtree(profile, ignore_errors=True)
    print(f"Removed {profile}. Run `login {provider}` to sign in again.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools._browser",
        description="Manage logged-in browser sessions for subscription-backed tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    login_parser = sub.add_parser("login", help="sign in to a provider interactively")
    login_parser.add_argument("provider", choices=sorted(PROVIDER_URLS))
    login_parser.add_argument(
        "--timeout", type=int, default=600, metavar="SECONDS",
        help="how long to wait for you to finish signing in (default 600)",
    )

    sub.add_parser("status", help="show login state for every provider")

    logout_parser = sub.add_parser("logout", help="delete a provider's browser profile")
    logout_parser.add_argument("provider", choices=sorted(PROVIDER_URLS))

    args = parser.parse_args(argv)
    if args.command == "login":
        return _login(args.provider, timeout_seconds=args.timeout)
    if args.command == "logout":
        return _logout(args.provider)
    return _status()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
