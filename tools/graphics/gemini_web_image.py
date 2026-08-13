"""Gemini web-app image generation driven through a logged-in browser session.

Uses the user's Gemini (Google One AI / Gemini Pro) subscription instead of an
API key: a persistent Chromium profile carries the login, the tool types the
prompt into gemini.google.com, waits for the generated image, and saves the
bytes. One-time setup:

    pip install playwright && python -m playwright install chromium
    python -m tools._browser login gemini

Image bytes are captured from the network response the page itself fetches
(the download button is the most drift-prone part of any web UI), with a DOM
``img`` fallback. Generated images carry a visible Gemini watermark in the
bottom-right corner; it is removed by default with ffmpeg, because these
images are typically fed to a video model as reference frames, which would
otherwise reproduce the watermark into the footage.
"""

from __future__ import annotations

import subprocess
import shutil
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

_PROVIDER = "gemini"
# A floor only — never a way to tell chrome from output. The sparkle logo the
# app loads is a 60KB PNG, so size alone would happily pass it through.
_MIN_IMAGE_BYTES = 40_000
# Observed on the live page: generated content is served from a /rd-gg-dl/ or
# /gg/ path, or rendered from a blob the page built.
_GENERATED_URL_MARKERS = ("/rd-gg-dl/", "/gg/", "blob:")
# UI chrome that any size- or host-based filter would misclassify as output:
# gstatic art, the Google bar avatar, and the signed-in user's profile photo.
_UI_IMAGE_MARKERS = ("gstatic.com", "/ogw/", "/rd-ogw/", "googleusercontent.com/a/")
_ASPECT_RATIOS = ["1:1", "3:4", "4:3", "9:16", "16:9", "21:9", "2:3", "3:2"]
# Plain-language quality of each remover, reported back so the caller can decide
# whether the result is good enough to hand to a video model.

# Watermark geometry, measured on a real 1024x572 Gemini output: a 15x12 px
# sparkle glyph 31px from the right edge and 42px from the bottom. Scaled by
# image width for other output sizes.
_WATERMARK_REFERENCE_WIDTH = 1024
_WATERMARK_GLYPH = (15, 12)     # (width, height) in reference pixels
_WATERMARK_MARGIN = (31, 42)    # (from right edge, from bottom edge)
_WATERMARK_PAD = 6              # breathing room so anti-aliased edges go too

_WATERMARK_QUALITY = {
    "fsr": (
        "excellent — frequency-selective reconstruction continues the surrounding "
        "texture through the patch; verified invisible at 100% on rippled water"
    ),
    "lama": "excellent — LaMa deep inpainting; best on large or structured holes",
    "telea": "fair — diffusion fill; leaves a smooth flat blob on textured areas",
    "delogo": "basic — ffmpeg blur patch; visible smudge on detailed backgrounds",
    "crop": "flawless pixels, but the bottom strip is gone and the aspect ratio changed",
}
# Context around the hole that patch-based algorithms learn texture from.
# Also keeps them fast: FSR_BEST is ~0.7s on a 100px window vs minutes on a
# full 1024px frame.
_INPAINT_CONTEXT_PX = 48


class GeminiWebImage(BaseTool):
    name = "gemini_web_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "gemini_web"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.BROWSER

    dependencies = ["python:playwright"]
    install_instructions = (
        "Use your Gemini subscription instead of an API key:\n"
        "  1. pip install playwright\n"
        "  2. python -m playwright install chromium\n"
        "  3. python -m tools._browser login gemini   (sign in once, in the window that opens)\n"
        "Check anytime with: python -m tools._browser status\n"
        "Watermark cleanup needs `ffmpeg` on PATH (optional but recommended)."
    )
    agent_skills = ["gemini-web-image"]

    capabilities = [
        "generate_image", "text_to_image", "image_edit", "reference_image",
    ]
    supports = {
        "negative_prompt": False,
        "seed": False,              # the web UI exposes no seed control
        "custom_size": False,
        "aspect_ratio": True,       # best-effort: expressed in the prompt text
        "image_edit": True,
        "reference_image": True,
        "subscription_billing": True,
    }
    best_for = [
        "free-with-subscription image generation (no API key, no per-image USD cost)",
        "character and scene reference sheets kept consistent across a shoot",
        "conversational edits of an image you already generated",
        "keyframes that feed video models (first/last frame, multi-keyframe modes)",
    ]
    not_good_for = [
        "unattended batch jobs (a web session can require re-login at any time)",
        "seed-reproducible generation",
        "exact pixel dimensions",
        "parallel generation — one browser profile runs one request at a time",
    ]
    fallback_tools = ["google_imagen", "openai_image", "flux_image"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Image description. English prompts work best. State the "
                    "framing and aspect ratio explicitly — there is no ratio "
                    "control in the web UI, only what the prompt asks for."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": _ASPECT_RATIOS,
                "default": "16:9",
                "description": "Appended to the prompt as a request. Best-effort, not enforced.",
            },
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Local reference images to attach (character sheets, style "
                    "refs, an image to edit). Order matters for prompts that "
                    "refer to 'the first image'."
                ),
            },
            "image_path": {
                "type": "string",
                "description": "Single local reference image — shorthand for image_paths.",
            },
            "remove_watermark": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Erase the bottom-right Gemini watermark with ffmpeg. Keep "
                    "this on for images used as video reference frames."
                ),
            },
            "watermark_mode": {
                "type": "string",
                "enum": ["auto", "fsr", "lama", "telea", "delogo", "crop"],
                "default": "auto",
                "description": (
                    "How to erase the corner watermark, best quality first: "
                    "'fsr' = frequency-selective reconstruction, rebuilds the "
                    "surrounding texture through the patch (needs "
                    "opencv-contrib-python); 'lama' = LaMa deep inpainting (needs "
                    "simple-lama-inpainting + torch); 'telea' = plain diffusion "
                    "fill, leaves a flat blob on texture (needs opencv-python); "
                    "'delogo' = ffmpeg blur patch, always available but smudged; "
                    "'crop' = cut the bottom strip, flawless pixels but loses "
                    "image area and changes the aspect ratio. "
                    "'auto' picks the best installed."
                ),
            },
            "keep_raw": {
                "type": "boolean",
                "default": False,
                "description": "Also keep the un-cleaned image next to the output as <name>.raw.<ext>.",
            },
            "watermark_box": {
                "type": "object",
                "description": (
                    "Override the watermark box in pixels {x, y, width, height}. "
                    "Only needed if Gemini moves or resizes the mark — the "
                    "default geometry was measured on a real output."
                ),
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                },
                "required": ["x", "y", "width", "height"],
            },
            "headless": {
                "type": "boolean",
                "default": True,
                "description": "Set false to watch the browser work (useful when debugging selectors).",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 30,
                "default": 240,
                "description": "How long to wait for the image to appear.",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=1500, vram_mb=0, disk_mb=300, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=1,
        backoff_seconds=10.0,
        retryable_errors=["timeout"],
    )
    idempotency_key_fields = ["prompt", "aspect_ratio", "image_paths", "image_path"]
    side_effects = [
        "writes image file to output_path",
        "drives a logged-in Gemini browser session",
        "consumes your Gemini subscription quota",
    ]
    user_visible_verification = [
        "Open the image and confirm the subject, framing and style match the prompt",
        "Confirm the bottom-right watermark is gone before using it as a video reference frame",
    ]

    # ---- Availability ----

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
                "command": "python -m tools._browser login gemini",
                "health_check": "python -m tools._browser status",
                "what_it_unlocks": [
                    "image generation on your existing Gemini subscription",
                    "reference-image editing and character-consistent sheets",
                    "no per-image USD cost",
                ],
            }
        return info

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Billed against the Gemini subscription, not per call.
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 45.0

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
                    "No confirmed Gemini browser login (auth issue — not a prompt "
                    "or tool bug). Fix: run `python -m tools._browser login gemini`, "
                    "sign in with your Gemini Pro account, then retry."
                ),
            )

        references = self._reference_paths(inputs)
        missing = [p for p in references if not Path(p).is_file()]
        if missing:
            return ToolResult(
                success=False,
                error=f"Reference image(s) not found: {', '.join(missing)}",
            )

        start = time.time()
        timeout_seconds = int(inputs.get("timeout_seconds", 240))
        session = BrowserSession(
            _PROVIDER,
            headless=bool(inputs.get("headless", True)),
            default_timeout_ms=timeout_seconds * 1000,
        )

        try:
            page = session.__enter__()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Could not open the browser: {exc}")

        try:
            # Record image URLs the page loads, then fetch them ourselves after
            # generation. The handler must not call response.body(): a blocking
            # round-trip inside a sync-API event callback can deadlock.
            seen_urls: list[str] = []
            page.on("response", lambda response: self._record_image_url(response, seen_urls))

            self._goto_with_retries(page, PROVIDER_URLS[_PROVIDER])
            if session.is_logged_out():
                return ToolResult(success=False, error=session.logged_out_error())

            if references:
                self._attach_references(session, references)

            full_prompt = self._compose_prompt(prompt, inputs.get("aspect_ratio", "16:9"))
            self._send_prompt(session, full_prompt)

            # Everything recorded so far is UI chrome or the echo of our own
            # uploaded references — only images that arrive after the prompt is
            # sent can be the generated one.
            image_bytes = self._await_image(
                session,
                seen_urls,
                since_index=len(seen_urls),
                deadline=time.time() + timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 — every failure gets a debug dump
            debug = session.dump_debug("failure")
            hint = f" Debug snapshot: {debug}" if debug else ""
            return ToolResult(
                success=False,
                error=f"Gemini web image generation failed: {exc}.{hint}",
            )
        finally:
            session.close()

        output_path = Path(inputs.get("output_path") or "gemini_web_image.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_format = self._sniff_format(image_bytes)
        stored_format = self._write_image(output_path, image_bytes, source_format)

        cleanup = self._clean_watermark(output_path, inputs)

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "route": "gemini_web_session",
                "model": "gemini-web-app",
                "prompt": prompt,
                "aspect_ratio_requested": inputs.get("aspect_ratio", "16:9"),
                "reference_images": references,
                "output": str(output_path),
                "format": stored_format,
                "source_format": source_format,
                "file_size_bytes": output_path.stat().st_size,
                "billing": "gemini_subscription",
                "watermark_cleanup": cleanup,
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model="gemini-web-app",
        )

    # ---- Page interaction ----

    @staticmethod
    def _reference_paths(inputs: dict[str, Any]) -> list[str]:
        paths = [str(p) for p in (inputs.get("image_paths") or [])]
        single = inputs.get("image_path")
        if single and str(single) not in paths:
            paths.append(str(single))
        return paths

    @staticmethod
    def _compose_prompt(prompt: str, aspect_ratio: str) -> str:
        """State the ratio the UI can't set, on a single line.

        Newlines matter: the composer submits on Enter, so a multi-line prompt
        would be sent in fragments and generate from the first line alone.
        """
        parts = [" ".join(prompt.split())]
        if aspect_ratio:
            parts.append(f"Aspect ratio: {aspect_ratio}.")
        return " ".join(parts)

    @staticmethod
    def _goto_with_retries(page: Any, url: str, *, attempts: int = 3) -> None:
        """Load the app, retrying transport-level failures.

        Reaching Google over a VPN or proxy drops connections intermittently
        (ERR_CONNECTION_CLOSED / ERR_CONNECTION_TIMED_OUT). One retry turns a
        dead run into a slow one.
        """
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                page.goto(url, wait_until="domcontentloaded")
                return
            except Exception as exc:  # noqa: BLE001 — retry transport errors
                last_error = exc
                if "net::" not in str(exc):
                    raise
                if attempt < attempts - 1:
                    time.sleep(3 * (attempt + 1))
        raise RuntimeError(
            f"could not load {url} after {attempts} attempts — the connection to "
            f"Google keeps dropping, which is a network/proxy problem rather than "
            f"a tool or prompt problem. Last error: "
            f"{str(last_error).splitlines()[0] if last_error else 'unknown'}"
        )

    def _attach_references(self, session: Any, references: list[str]) -> None:
        file_input = session.first_locator("file_input", timeout_ms=20_000)
        file_input.set_input_files(references)
        # Uploads have to register before the prompt is sent, or they're dropped.
        time.sleep(3)

    def _send_prompt(self, session: Any, prompt: str) -> None:
        editor = session.first_locator("prompt_input", timeout_ms=30_000)
        editor.click()
        # press_sequentially replaced type() in Playwright 1.42; keep both so the
        # tool works across the versions a user might already have installed.
        typist = getattr(editor, "press_sequentially", None) or editor.type
        typist(prompt, delay=5)
        try:
            session.page.keyboard.press("Enter")
        except Exception:  # noqa: BLE001 — fall back to the send button
            session.first_locator("send_button", timeout_ms=10_000).click()

    @staticmethod
    def _record_image_url(response: Any, sink: list[str]) -> None:
        """Note image URLs the page loads. Reads only the event payload.

        Calling a blocking Playwright method (like ``response.body()``) inside a
        sync-API event handler can deadlock, so this records the URL and leaves
        the fetching to :meth:`_fetch_bytes` on the main flow.
        """
        try:
            content_type = (response.headers or {}).get("content-type", "")
            url = response.url or ""
        except Exception:  # noqa: BLE001 — malformed event; nothing to record
            return
        if not content_type.startswith("image/") or "svg" in content_type:
            return
        if not url.startswith("http") or url in sink:
            return
        sink.append(url)

    def _await_image(
        self, session: Any, seen_urls: list[str], *, since_index: int, deadline: float
    ) -> bytes:
        """Wait for the generated image to appear in the response element.

        Identification is structural, never by size or host: the page also
        loads a 512px sparkle logo from gstatic and the signed-in user's
        avatar from googleusercontent, both of which pass any byte-size test.
        Sniffed URLs are only a fallback, and only those recorded at or after
        ``since_index`` — earlier ones include our own uploaded references.
        """
        page = session.page
        # Diagnostics: without these, a page.evaluate() that keeps failing on a
        # flaky connection is indistinguishable from "the model produced nothing".
        errors: list[str] = []
        saw_element = False
        while time.time() < deadline:
            found, note = self._image_from_dom(session)
            if found:
                return found
            if note:
                (errors if note.startswith("error:") else []).append(note)
                saw_element = saw_element or note == "element-present"
            for url in reversed(seen_urls[since_index:]):
                if not self._looks_generated(url):
                    continue
                body = self._fetch_bytes(page, url)
                if body and len(body) >= _MIN_IMAGE_BYTES:
                    return body
            if session.is_logged_out():
                raise RuntimeError(session.logged_out_error())
            page.wait_for_timeout(2000)

        detail = (
            "the response element was present but its image never became "
            "readable — likely a dropped connection while fetching it"
            if saw_element
            else "no image element ever appeared in the response"
        )
        if errors:
            detail += f". Last page error: {errors[-1][:200]}"
        raise TimeoutError(
            f"no generated image appeared before the timeout ({detail}). "
            "Check the debug snapshot for a refusal, a quota notice, or a "
            "changed layout"
        )

    @staticmethod
    def _looks_generated(url: str) -> bool:
        """Whether a sniffed URL is generated content rather than UI chrome."""
        if not url:
            return False
        lowered = url.lower()
        if any(marker in lowered for marker in _UI_IMAGE_MARKERS):
            return False
        return any(marker in lowered for marker in _GENERATED_URL_MARKERS)

    def _image_from_dom(self, session: Any) -> tuple[Optional[bytes], Optional[str]]:
        """Fetch the newest fully-loaded image inside the response element.

        The ``src`` is typically a ``blob:`` URL created by the page, so the
        fetch has to happen in the page context. Returns (bytes, note) where
        note explains a miss — the caller turns it into a real error message
        instead of an unexplained timeout.
        """
        selectors = ", ".join(session.selectors.get("response_image", []))
        if not selectors:
            return None, "error: no response_image selectors configured"
        try:
            report = session.page.evaluate(
                """(selectors) => {
                    const all = Array.from(document.querySelectorAll(selectors));
                    const ready = all.filter(el => el.complete
                        && el.naturalWidth >= 256 && el.naturalHeight >= 256);
                    const el = ready[ready.length - 1];
                    return {
                        total: all.length,
                        ready: ready.length,
                        src: el ? (el.currentSrc || el.src || '') : '',
                        width: el ? el.naturalWidth : 0,
                        height: el ? el.naturalHeight : 0,
                    };
                }""",
                selectors,
            )
        except Exception as exc:  # noqa: BLE001 — page busy/navigating; report it
            return None, f"error: evaluate failed: {str(exc).splitlines()[0]}"

        if not report or not report.get("total"):
            return None, None
        if not report.get("src"):
            return None, "element-present"
        body = self._fetch_bytes(session.page, report["src"])
        if body and len(body) >= _MIN_IMAGE_BYTES:
            return body, None
        return None, "element-present"

    @staticmethod
    def _fetch_bytes(page: Any, url: str) -> Optional[bytes]:
        """Re-download an image from inside the page, so its cookies apply.

        ``blob:`` is accepted deliberately — Gemini renders the generated image
        from a blob the page created, and only the page can read it.
        """
        if not url or not url.startswith(("http://", "https://", "blob:")):
            return None
        try:
            encoded = page.evaluate(
                """async (url) => {
                    const response = await fetch(url);
                    if (!response.ok) return null;
                    const buffer = await response.arrayBuffer();
                    const bytes = new Uint8Array(buffer);
                    let binary = '';
                    for (let i = 0; i < bytes.length; i++) {
                        binary += String.fromCharCode(bytes[i]);
                    }
                    return btoa(binary);
                }""",
                url,
            )
        except Exception:  # noqa: BLE001 — expired URL or navigation; skip it
            return None
        if not encoded:
            return None

        import base64  # noqa: PLC0415

        try:
            return base64.b64decode(encoded)
        except ValueError:
            return None

    # ---- Saving ----

    @staticmethod
    def _sniff_format(data: bytes) -> str:
        """Identify the real container. Gemini serves JPEG, whatever we name it."""
        if data[:3] == b"\xff\xd8\xff":
            return "jpeg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp"
        return "unknown"

    @staticmethod
    def _write_image(path: Path, data: bytes, source_format: str) -> str:
        """Write the bytes, converting when the requested suffix disagrees.

        A caller asking for ``scene.png`` and getting JPEG bytes under that name
        breaks anything that trusts the extension, so re-encode instead of
        lying about it.
        """
        suffix = path.suffix.lower().lstrip(".")
        wanted = {"jpg": "jpeg"}.get(suffix, suffix)
        if not wanted or wanted == source_format or source_format == "unknown":
            path.write_bytes(data)
            return source_format if source_format != "unknown" else (wanted or "unknown")

        try:
            import io  # noqa: PLC0415

            from PIL import Image  # noqa: PLC0415

            with Image.open(io.BytesIO(data)) as img:
                converted = img.convert("RGB") if wanted in ("jpeg", "bmp") else img
                converted.save(path)
            return wanted
        except Exception:  # noqa: BLE001 — keep the real bytes over failing
            path.write_bytes(data)
            return source_format

    # ---- Watermark cleanup ----

    def _clean_watermark(self, path: Path, inputs: dict[str, Any]) -> dict[str, Any]:
        """Erase the bottom-right Gemini watermark. Never fails the generation."""
        if not inputs.get("remove_watermark", True):
            return {"applied": False, "reason": "disabled by remove_watermark=false"}

        requested = inputs.get("watermark_mode", "auto")
        mode = self._resolve_watermark_mode(requested)
        if mode is None:
            return {
                "applied": False,
                "requested_mode": requested,
                "reason": (
                    "no watermark remover available — the Gemini watermark is "
                    "still in the bottom-right corner. Install one: "
                    "`pip install opencv-python` (good) or "
                    "`pip install simple-lama-inpainting` (best), or put ffmpeg "
                    "on PATH (basic). Do not use this image as a video reference "
                    "frame until the watermark is gone."
                ),
            }

        region = self._watermark_region(path, inputs)
        if inputs.get("keep_raw", False):
            shutil.copy2(path, path.with_suffix(f".raw{path.suffix or '.png'}"))
        try:
            if mode == "lama":
                self._inpaint_lama(path, region)
            elif mode in ("fsr", "telea"):
                self._inpaint_opencv(path, region, mode)
            elif mode == "crop":
                self._crop_bottom(path, region)
            else:
                self._delogo_ffmpeg(path, region)
        except Exception as exc:  # noqa: BLE001 — keep the raw image, report why
            return {
                "applied": False,
                "mode": mode,
                "requested_mode": requested,
                "reason": f"{mode} failed: {str(exc).splitlines()[0][:300]}",
            }

        return {
            "applied": True,
            "mode": mode,
            "requested_mode": requested,
            "quality": _WATERMARK_QUALITY[mode],
            "region": region,
            "verify": "Open the image and confirm the bottom-right corner is clean.",
        }

    @staticmethod
    def _resolve_watermark_mode(requested: str) -> Optional[str]:
        """Pick the best available remover, honouring an explicit request."""
        def available(mode: str) -> bool:
            if mode == "lama":
                try:
                    import simple_lama_inpainting  # noqa: F401,PLC0415
                except Exception:  # noqa: BLE001 — the torch stack may be broken too
                    return False
                return True
            if mode == "fsr":
                try:
                    import cv2  # noqa: PLC0415

                    return hasattr(cv2, "xphoto")  # contrib build only
                except Exception:  # noqa: BLE001
                    return False
            if mode == "telea":
                try:
                    import cv2  # noqa: F401,PLC0415
                except Exception:  # noqa: BLE001
                    return False
                return True
            return bool(shutil.which("ffmpeg"))  # delogo and crop

        if requested != "auto":
            return requested if available(requested) else None
        # fsr first: verified against the real watermark, no torch, sub-second.
        for mode in ("fsr", "lama", "telea", "delogo"):
            if available(mode):
                return mode
        return None

    def _watermark_region(self, path: Path, inputs: dict[str, Any]) -> dict[str, int]:
        """Pixel box covering the bottom-right watermark.

        Mask size is what decides whether the repair is invisible. Measured on
        a real 1024x572 Gemini output, the mark is a **15x12 px sparkle glyph**
        sitting 31px from the right edge and 42px from the bottom — so the
        earlier 22%-of-width box repainted ~45x more area than needed and left
        an obvious flat rectangle.

        Geometry is fixed rather than detected on purpose: content-based
        detection kept latching onto water ripples and cloud edges, which is
        worse than a known-good box. ``watermark_box`` overrides it if Gemini
        ever moves the mark.

        The box stays 1px inside the frame: ffmpeg's delogo interpolates from
        the pixels around the box and errors out (exit -22) at an edge.
        """
        try:
            from PIL import Image  # noqa: PLC0415

            with Image.open(path) as img:
                width, height = img.size
        except Exception:  # noqa: BLE001 — fall back to a common Gemini size
            width, height = 1024, 572

        override = inputs.get("watermark_box")
        if isinstance(override, dict) and {"x", "y", "width", "height"} <= set(override):
            box_x, box_y = int(override["x"]), int(override["y"])
            box_w, box_h = int(override["width"]), int(override["height"])
            source = "caller_override"
        else:
            scale = max(0.25, width / _WATERMARK_REFERENCE_WIDTH)
            pad = max(4, round(_WATERMARK_PAD * scale))
            glyph_w = round(_WATERMARK_GLYPH[0] * scale)
            glyph_h = round(_WATERMARK_GLYPH[1] * scale)
            box_w = glyph_w + pad * 2
            box_h = glyph_h + pad * 2
            box_x = width - round(_WATERMARK_MARGIN[0] * scale) - glyph_w - pad
            box_y = height - round(_WATERMARK_MARGIN[1] * scale) - glyph_h - pad
            source = "fixed_geometry"

        box_w = max(4, min(box_w, width - 2))
        box_h = max(4, min(box_h, height - 2))
        return {
            "x": max(1, min(box_x, width - box_w - 1)),
            "y": max(1, min(box_y, height - box_h - 1)),
            "width": box_w,
            "height": box_h,
            "image_width": width,
            "image_height": height,
            "corner": "bottom_right",
            "source": source,
        }

    @staticmethod
    def _mask_for(region: dict[str, int]):
        """White-on-black mask marking the watermark box."""
        from PIL import Image, ImageDraw  # noqa: PLC0415

        mask = Image.new("L", (region["image_width"], region["image_height"]), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(
            [region["x"], region["y"],
             region["x"] + region["width"], region["y"] + region["height"]],
            fill=255,
        )
        return mask

    def _inpaint_lama(self, path: Path, region: dict[str, int]) -> None:
        """LaMa deep inpainting — the closest thing here to content-aware fill."""
        from PIL import Image  # noqa: PLC0415
        from simple_lama_inpainting import SimpleLama  # noqa: PLC0415

        with Image.open(path) as img:
            source = img.convert("RGB")
        result = SimpleLama()(source, self._mask_for(region))
        result.save(path)

    def _inpaint_opencv(self, path: Path, region: dict[str, int], mode: str) -> None:
        """OpenCV inpainting, applied to a window around the mark.

        Windowing is not just a speed trick (FSR_BEST goes from minutes on a
        full frame to under a second): a tight window is also the texture the
        algorithm reconstructs from, so the patch matches its surroundings.
        """
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read {path}")
        height, width = image.shape[:2]

        wx0 = max(0, region["x"] - _INPAINT_CONTEXT_PX)
        wy0 = max(0, region["y"] - _INPAINT_CONTEXT_PX)
        wx1 = min(width, region["x"] + region["width"] + _INPAINT_CONTEXT_PX)
        wy1 = min(height, region["y"] + region["height"] + _INPAINT_CONTEXT_PX)
        window = image[wy0:wy1, wx0:wx1].copy()

        hole = np.zeros(window.shape[:2], dtype="uint8")
        hole[
            region["y"] - wy0:region["y"] - wy0 + region["height"],
            region["x"] - wx0:region["x"] - wx0 + region["width"],
        ] = 255

        if mode == "fsr":
            repaired = np.zeros_like(window)
            # xphoto marks KNOWN pixels as 255 — the inverse of cv2.inpaint.
            cv2.xphoto.inpaint(
                window, cv2.bitwise_not(hole), repaired, cv2.xphoto.INPAINT_FSR_BEST
            )
        else:
            repaired = cv2.inpaint(window, hole, 5, cv2.INPAINT_TELEA)

        image[wy0:wy1, wx0:wx1] = repaired
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"could not write {path}")

    def _delogo_ffmpeg(self, path: Path, region: dict[str, int]) -> None:
        """ffmpeg delogo — always available, but a visible blur patch."""
        self._run_ffmpeg_filter(
            path,
            f"delogo=x={region['x']}:y={region['y']}"
            f":w={region['width']}:h={region['height']}",
        )

    def _crop_bottom(self, path: Path, region: dict[str, int]) -> None:
        """Cut the watermark strip off — flawless, but loses image area."""
        keep = max(16, region["image_height"] - region["height"] - 2)
        self._run_ffmpeg_filter(path, f"crop={region['image_width']}:{keep}:0:0")

    @staticmethod
    def _run_ffmpeg_filter(path: Path, video_filter: str) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found on PATH")
        cleaned = path.with_suffix(f".cleaned{path.suffix or '.png'}")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(path), "-vf", video_filter, str(cleaned)],
                capture_output=True, text=True, timeout=120, check=True,
            )
        except subprocess.CalledProcessError as exc:
            cleaned.unlink(missing_ok=True)
            raise RuntimeError((exc.stderr or "").strip() or f"exit {exc.returncode}") from exc
        except Exception:
            cleaned.unlink(missing_ok=True)
            raise
        cleaned.replace(path)
