"""Suno music generation driven through a logged-in browser session.

Uses the user's own Suno subscription instead of an API key: a persistent
Chromium profile carries the login, the tool types a style prompt into
suno.com/create, waits for the clip to render, and saves the audio. One-time
setup:

    pip install playwright && python -m playwright install chromium
    python -m tools._browser login suno

Suno has no self-serve public API — the only way to spend a Suno subscription
programmatically is the web app. Audio bytes are captured from the network
response the page itself fetches (Suno streams clips from ``cdn*.suno.ai``),
with a DOM ``<audio src>`` fallback. That ordering is deliberate: the download
menu is the most drift-prone part of any web UI, while the audio stream URL is
what playback itself depends on.

One generation yields two candidate clips. The tool saves the first by default
and both when ``download_all`` is set, because picking between candidates is a
creative judgement the agent should make with the files in hand.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_PROVIDER = "suno"
# Suno streams finished clips from its own CDN. Matching the host is what makes
# the capture version-agnostic: the undocumented JSON endpoints move between
# releases, the playback CDN does not.
_AUDIO_HOST_MARKERS = ("cdn1.suno.ai", "cdn2.suno.ai", "cdn.suno.ai", "audiopipe.suno.ai")
_AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav", ".ogg", ".flac")
# A floor, not an identifier. Suno's UI click-sounds and silent placeholder
# clips are a few KB; a real 30s+ track is comfortably past this.
_MIN_AUDIO_BYTES = 150_000
# Clip ids in Suno URLs are UUIDs.
_CLIP_ID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I
)


class SunoWebMusic(BaseTool):
    name = "suno_web_music"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "music_generation"
    provider = "suno_web"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.BROWSER

    dependencies = ["python:playwright"]
    install_instructions = (
        "Use your Suno subscription instead of an API key:\n"
        "  1. pip install playwright\n"
        "  2. python -m playwright install chromium\n"
        "  3. python -m tools._browser login suno   (sign in once, in the window that opens)\n"
        "Check anytime with: python -m tools._browser status\n"
        "No API key needed — generation consumes Suno subscription credits."
    )
    agent_skills = ["suno-web-music", "music"]

    capabilities = [
        "generate_music",
        "instrumental",
        "text_to_music",
        "style_prompt",
    ]
    supports = {
        "instrumental": True,
        "vocals": True,
        "lyrics": True,
        "style_prompt": True,
        "exact_duration": False,   # Suno picks the length; trim downstream
        "seed": False,             # the web UI exposes no seed control
        "stems": False,
        "subscription_billing": True,
    }
    best_for = [
        "free-with-subscription music generation (no API key, no per-track USD cost)",
        "instrumental beds for narrated video where vocals would fight the voiceover",
        "mood-specific BGM described in plain language (genre + instrumentation + tempo)",
    ]
    not_good_for = [
        "exact-duration tracks — Suno decides the length, so trim or loop downstream",
        "unattended batch jobs (a web session can require re-login at any time)",
        "seed-reproducible generation",
        "parallel generation — one browser profile runs one request at a time",
    ]
    fallback_tools = ["google_music", "pixabay_music", "suno_music"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "maxLength": 1000,
                "description": (
                    "Style description. English works best. Name genre + mood + "
                    "instrumentation + tempo, e.g. 'Cinematic tension underscore, "
                    "dark strings and pulsing sub-bass, building dread, instrumental, "
                    "80 BPM'. In custom mode this fills the Styles field."
                ),
            },
            "instrumental": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Suppress vocals. Keep true for anything backing narration — "
                    "vocals compete with the voiceover for the same frequencies."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["simple", "custom"],
                "default": "simple",
                "description": (
                    "'simple' uses the Song Description box (fewest selectors, most "
                    "robust). 'custom' fills Styles/Title/Lyrics separately — more "
                    "control, more DOM surface that can drift."
                ),
            },
            "title": {
                "type": "string",
                "description": "Track title. Custom mode only.",
            },
            "lyrics": {
                "type": "string",
                "description": (
                    "Lyrics for a vocal track. Custom mode only, and ignored when "
                    "instrumental is true."
                ),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Where to write the audio. Extension is normalized to the "
                    "format Suno actually served (usually .mp3)."
                ),
            },
            "download_all": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Suno renders two candidates per generation. False saves the "
                    "first; true saves both as <name>.1.mp3 / <name>.2.mp3 so you "
                    "can audition and pick."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 60,
                "default": 420,
                "description": "How long to wait for a clip. Music takes 1-3 minutes.",
            },
            "headless": {
                "type": "boolean",
                "default": True,
                "description": "Set false to watch the browser work (useful when debugging selectors).",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=1024, vram_mb=0, disk_mb=400, network_required=True
    )
    # Regeneration spends subscription credits, so never retry silently — the
    # agent decides, the same contract dreamina_video uses.
    retry_policy = RetryPolicy(max_retries=0, backoff_seconds=0.0, retryable_errors=[])
    idempotency_key_fields = ["prompt", "instrumental", "mode", "title", "lyrics"]
    side_effects = [
        "writes audio file(s) to output_path",
        "drives a logged-in Suno web session in a real browser",
        "consumes Suno subscription credits per generation",
    ]
    # Set by _set_instrumental(): False means the switch could not be found and
    # the request rode on prompt wording alone, so the track may contain vocals.
    _instrumental_toggle_found: bool = False

    user_visible_verification = [
        "Listen to the track for mood fit against the scene it backs",
        "Confirm no vocals when instrumental was requested (vocals bury narration)",
        "Check the reported duration covers the video, or plan a loop point",
    ]

    # ---- Status ----

    def get_status(self) -> ToolStatus:
        from tools._browser.session import login_state, playwright_available  # noqa: PLC0415

        if not playwright_available():
            return ToolStatus.UNAVAILABLE
        state = login_state(_PROVIDER)
        if state.ready:
            # Cookies may still have expired — execute() detects that for real.
            return ToolStatus.AVAILABLE
        if state.profile_exists:
            return ToolStatus.DEGRADED
        return ToolStatus.UNAVAILABLE

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        if self.get_status() != ToolStatus.AVAILABLE:
            info["setup_offer"] = {
                "kind": "one_time_login",
                "fix_complexity": "2-minute browser login (no API key)",
                "command": "python -m tools._browser login suno",
                "health_check": "python -m tools._browser status",
                "what_it_unlocks": [
                    "music generation on your existing Suno subscription",
                    "instrumental BGM beds for narrated video",
                    "no per-track USD cost",
                ],
            }
        return info

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Billed against the Suno subscription, not per call.
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 150.0

    # ---- Execution ----

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        from tools._browser.session import (  # noqa: PLC0415
            PROVIDER_URLS,
            BrowserSession,
            login_state,
            playwright_available,
        )

        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(success=False, error="prompt is required.")

        if not playwright_available():
            return ToolResult(
                success=False,
                error="Playwright is not installed. " + self.install_instructions,
            )
        if not login_state(_PROVIDER).ready:
            return ToolResult(
                success=False,
                error=(
                    "No confirmed Suno browser login (auth issue — not a prompt or "
                    "tool bug). Fix: run `python -m tools._browser login suno`, sign "
                    "in with the account that holds your subscription, then retry."
                ),
            )

        mode = inputs.get("mode", "simple")
        instrumental = bool(inputs.get("instrumental", True))
        timeout_seconds = int(inputs.get("timeout_seconds", 420))

        start = time.time()
        session = BrowserSession(
            _PROVIDER,
            headless=bool(inputs.get("headless", True)),
            default_timeout_ms=min(timeout_seconds, 120) * 1000,
        )

        try:
            page = session.__enter__()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Could not open the browser: {exc}")

        try:
            # Record audio URLs the page streams, then fetch them ourselves.
            # The handler must not call response.body(): a blocking round-trip
            # inside a sync-API event callback can deadlock.
            seen_urls: list[str] = []
            page.on("response", lambda response: self._record_audio_url(response, seen_urls))

            page.goto(PROVIDER_URLS[_PROVIDER], wait_until="domcontentloaded")
            if session.is_logged_out():
                return ToolResult(success=False, error=session.logged_out_error())

            if mode == "custom":
                self._enter_custom_mode(session)
            self._set_instrumental(session, instrumental)
            self._fill_prompt(session, inputs, mode=mode, instrumental=instrumental)

            # Anything recorded before submit is UI chrome or a previously
            # generated track already in the user's library.
            before = len(seen_urls)
            self._submit(session)
            clips = self._await_clips(
                session,
                seen_urls,
                since_index=before,
                deadline=time.time() + timeout_seconds,
                timeout_seconds=timeout_seconds,
                want=2 if inputs.get("download_all") else 1,
            )
            # Fetch bytes while the session is alive — the CDN URLs are signed
            # and the request must carry the page's cookies.
            fetched = self._fetch_audio(page, clips)
        except _QuotaExhausted as exc:
            return ToolResult(
                success=False,
                error=(
                    f"Suno reports no generation credits left ({exc}). This is a "
                    "quota issue, not a prompt or tool bug — it will not succeed on "
                    "retry. Top up or wait for the monthly reset, or fall back to "
                    "pixabay_music / google_music."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — every failure gets a debug dump
            debug = session.dump_debug("failure")
            hint = f" Debug snapshot (screenshot + HTML): {debug}" if debug else ""
            return ToolResult(
                success=False,
                error=(
                    f"Suno web music generation failed: {exc}.{hint} If a selector "
                    "no longer matches, override it in "
                    "~/.openmontage/browser/selectors.json under the \"suno\" key."
                ),
            )
        finally:
            session.close()

        if not fetched:
            return ToolResult(
                success=False,
                error="Suno produced no usable audio (every candidate was too small).",
            )

        written = self._write_tracks(inputs, fetched)
        primary = written[0]

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "route": "suno_web_session",
                "billing": "suno_subscription_credits",
                "prompt": prompt,
                "mode": mode,
                "instrumental": instrumental,
                # False = the switch was not found, so "no vocals" rode on prompt
                # wording alone. Listen before trusting it under narration.
                "instrumental_toggle_applied": self._instrumental_toggle_found,
                "output": primary["path"],
                "outputs": [w["path"] for w in written],
                "candidates_saved": len(written),
                "clip_ids": [w["clip_id"] for w in written if w["clip_id"]],
                "audio_urls": [w["url"] for w in written],
                "format": primary["format"],
                **primary["probe"],
            },
            artifacts=[w["path"] for w in written],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model="suno-web",
        )

    # ---- Page driving ----

    @staticmethod
    def _record_audio_url(response: Any, sink: list[str]) -> None:
        """Note any URL that looks like generated audio. Never blocks."""
        try:
            url = response.url or ""
            if not url:
                return
            lowered = url.lower().split("?")[0]
            is_audio_host = any(marker in lowered for marker in _AUDIO_HOST_MARKERS)
            has_audio_ext = lowered.endswith(_AUDIO_EXTENSIONS)
            if not (is_audio_host and has_audio_ext) and not (
                has_audio_ext and "suno" in lowered
            ):
                return
            if url not in sink:
                sink.append(url)
        except Exception:  # noqa: BLE001 — a listener must never break the run
            return

    def _enter_custom_mode(self, session: Any) -> None:
        try:
            session.first_locator("custom_mode_toggle", timeout_ms=8000).click()
            session.page.wait_for_timeout(800)
        except LookupError as exc:
            raise RuntimeError(
                "could not find the Custom-mode switch. Use mode='simple', or "
                f"repair the 'custom_mode_toggle' selector. ({exc})"
            ) from exc

    def _set_instrumental(self, session: Any, instrumental: bool) -> None:
        """Toggle the Instrumental switch to the requested state.

        A missing switch is not fatal in simple mode: the prompt itself says
        'instrumental', which Suno honours most of the time. It IS reported, so
        the caller knows to check the track for vocals.
        """
        try:
            toggle = session.first_locator("instrumental_toggle", timeout_ms=6000)
        except LookupError:
            self._instrumental_toggle_found = False
            return
        self._instrumental_toggle_found = True
        try:
            state = (toggle.get_attribute("aria-checked") or "").lower()
            already_on = state == "true"
            if already_on != instrumental:
                toggle.click()
                session.page.wait_for_timeout(400)
        except Exception:  # noqa: BLE001 — fall back to prompt-level wording
            self._instrumental_toggle_found = False

    def _fill_prompt(
        self, session: Any, inputs: dict[str, Any], *, mode: str, instrumental: bool
    ) -> None:
        prompt = (inputs.get("prompt") or "").strip()
        if mode == "custom":
            self._type_into(session, "style_input", prompt)
            title = (inputs.get("title") or "").strip()
            if title:
                self._type_into(session, "title_input", title, optional=True)
            lyrics = (inputs.get("lyrics") or "").strip()
            if lyrics and not instrumental:
                self._type_into(session, "lyrics_input", lyrics, optional=True)
            return
        # Simple mode: one description box. Say "instrumental" in the text as
        # well as flipping the switch — belt and braces, since the switch is
        # the selector most likely to have drifted.
        text = prompt
        if instrumental and "instrumental" not in text.lower():
            text = f"{text}, instrumental, no vocals"
        self._type_into(session, "prompt_input", text)

    def _type_into(
        self, session: Any, key: str, text: str, *, optional: bool = False
    ) -> None:
        try:
            field = session.first_locator(key, timeout_ms=15_000)
        except LookupError:
            if optional:
                return
            raise
        field.click()
        try:
            field.fill(text)
        except Exception:  # noqa: BLE001 — contenteditable rejects fill()
            session.page.keyboard.type(text, delay=8)

    def _submit(self, session: Any) -> None:
        try:
            button = session.first_locator("create_button", timeout_ms=15_000)
        except LookupError as exc:
            raise RuntimeError(
                f"could not find the Create button — repair 'create_button'. ({exc})"
            ) from exc
        button.click()
        session.page.wait_for_timeout(1500)
        self._raise_if_out_of_credits(session)

    def _raise_if_out_of_credits(self, session: Any) -> None:
        for selector in session.selectors.get("quota_exhausted", []):
            try:
                node = session.page.locator(selector).first
                if node.is_visible(timeout=800):
                    raise _QuotaExhausted((node.inner_text() or selector).strip()[:200])
            except _QuotaExhausted:
                raise
            except Exception:  # noqa: BLE001 — selector absent means "not that one"
                continue

    def _await_clips(
        self,
        session: Any,
        seen_urls: list[str],
        *,
        since_index: int,
        deadline: float,
        timeout_seconds: int,
        want: int,
    ) -> list[str]:
        """Wait for generated clip URLs to show up on the network.

        Falls back to the DOM ``<audio src>`` when nothing was sniffed — some
        builds hand playback a blob instead of a direct CDN request.
        """
        last_credit_check = 0.0
        while time.time() < deadline:
            fresh = self._dedupe_clips(seen_urls[since_index:])
            if len(fresh) >= want:
                return fresh[:want]
            # Cheap periodic check: a queued job can fail on credits after submit.
            if time.time() - last_credit_check > 15:
                self._raise_if_out_of_credits(session)
                last_credit_check = time.time()
            session.page.wait_for_timeout(2000)

        fresh = self._dedupe_clips(seen_urls[since_index:])
        if fresh:
            return fresh[:want]

        dom_url = self._audio_src_from_dom(session)
        if dom_url:
            return [dom_url]
        raise TimeoutError(
            f"no finished clip after {timeout_seconds}s — Suno was still rendering, "
            "or the clip never started streaming. Raise timeout_seconds, or re-run "
            "with headless=false to watch what the page is doing"
        )

    @staticmethod
    def _dedupe_clips(urls: list[str]) -> list[str]:
        """One URL per clip id, preserving arrival order.

        Suno re-requests the same clip for the waveform and for playback, and
        range requests repeat the URL with different query strings.
        """
        out: list[str] = []
        seen_ids: set[str] = set()
        for url in urls:
            match = _CLIP_ID_RE.search(url)
            key = match.group(1).lower() if match else url.split("?")[0]
            if key in seen_ids:
                continue
            seen_ids.add(key)
            out.append(url)
        return out

    def _audio_src_from_dom(self, session: Any) -> Optional[str]:
        try:
            node = session.first_locator("audio_element", timeout_ms=4000)
            src = node.get_attribute("src") or ""
        except Exception:  # noqa: BLE001
            return None
        return src if src.startswith("http") else None

    def _fetch_audio(self, page: Any, urls: list[str]) -> list[dict[str, Any]]:
        """Download each clip through the page context so cookies apply."""
        out: list[dict[str, Any]] = []
        for url in urls:
            try:
                response = page.request.get(url, timeout=120_000)
                if not response.ok:
                    continue
                body = response.body()
            except Exception:  # noqa: BLE001 — try the next candidate
                continue
            if len(body) < _MIN_AUDIO_BYTES:
                continue
            match = _CLIP_ID_RE.search(url)
            out.append({
                "url": url,
                "bytes": body,
                "clip_id": match.group(1) if match else None,
                "format": self._sniff_format(body, url),
            })
        return out

    @staticmethod
    def _sniff_format(body: bytes, url: str) -> str:
        if body[:3] == b"ID3" or body[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return "mp3"
        if body[:4] == b"RIFF":
            return "wav"
        if body[4:8] == b"ftyp":
            return "m4a"
        if body[:4] == b"OggS":
            return "ogg"
        if body[:4] == b"fLaC":
            return "flac"
        suffix = url.lower().split("?")[0].rsplit(".", 1)
        return suffix[-1] if len(suffix) == 2 and suffix[-1] in {
            "mp3", "wav", "m4a", "ogg", "flac"
        } else "mp3"

    # ---- Output ----

    def _write_tracks(
        self, inputs: dict[str, Any], fetched: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        base = Path(inputs.get("output_path") or "suno_web_music.mp3")
        base.parent.mkdir(parents=True, exist_ok=True)
        written: list[dict[str, Any]] = []
        for index, item in enumerate(fetched, start=1):
            if len(fetched) == 1:
                target = base.with_suffix(f".{item['format']}")
            else:
                # Build the name by hand: with_suffix() would treat the ".1"
                # candidate index as the extension and replace it, collapsing
                # every candidate onto the same path.
                target = base.parent / f"{base.stem}.{index}.{item['format']}"
            target.write_bytes(item["bytes"])
            written.append({
                "path": str(target),
                "url": item["url"],
                "clip_id": item["clip_id"],
                "format": item["format"],
                "probe": self._probe_audio(target),
            })
        return written

    @staticmethod
    def _probe_audio(path: Path) -> dict[str, Any]:
        info: dict[str, Any] = {
            "file_size_bytes": path.stat().st_size,
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        }
        if not shutil.which("ffprobe"):
            return info
        try:
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", str(path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
            if proc.returncode != 0:
                return info
            probe = json.loads(proc.stdout)
            fmt = probe.get("format", {})
            if fmt.get("duration"):
                info["duration_seconds"] = round(float(fmt["duration"]), 2)
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "audio":
                    info["audio_codec"] = stream.get("codec_name", "")
                    info["audio_channels"] = stream.get("channels")
                    info["sample_rate"] = stream.get("sample_rate")
                    break
        except Exception:  # noqa: BLE001 — probing is best-effort metadata
            pass
        return info


class _QuotaExhausted(RuntimeError):
    """Suno reports no credits left — terminal, never worth a retry."""
