"""Contract tests for the subscription-backed Gemini web image tool.

These verify the BaseTool contract, availability reporting, watermark
cleanup, and the shared browser-session infrastructure without launching a
browser, logging in, or consuming any Gemini quota.

Run: pytest tests/contracts/test_gemini_web_image.py -v
"""

import json
import shutil
import time

import pytest

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.graphics.gemini_web_image import GeminiWebImage


@pytest.fixture
def browser_home(tmp_path, monkeypatch):
    """Redirect browser profiles to a temp dir so tests never touch a real login."""
    monkeypatch.setenv("OPENMONTAGE_BROWSER_PROFILE_DIR", str(tmp_path / "browser"))
    monkeypatch.delenv("OPENMONTAGE_BROWSER_SELECTORS", raising=False)
    return tmp_path / "browser"


@pytest.fixture
def logged_in(browser_home):
    from tools._browser.session import write_login_marker
    write_login_marker("gemini")
    return browser_home / "gemini"


# ------------------------------------------------------------------
# Contract compliance
# ------------------------------------------------------------------

class TestContract:

    def test_inherits_base_tool(self):
        assert issubclass(GeminiWebImage, BaseTool)

    def test_has_required_identity(self):
        tool = GeminiWebImage()
        assert tool.name == "gemini_web_image"
        assert tool.version
        assert tool.provider == "gemini_web"
        assert tool.capability == "image_generation"
        assert tool.tier == ToolTier.GENERATE
        assert tool.runtime == ToolRuntime.BROWSER
        assert tool.stability == ToolStability.EXPERIMENTAL

    def test_execution_mode_is_sync(self):
        assert GeminiWebImage().execution_mode == ExecutionMode.SYNC

    def test_determinism_is_stochastic(self):
        assert GeminiWebImage().determinism == Determinism.STOCHASTIC

    def test_has_input_schema(self):
        schema = GeminiWebImage().input_schema
        assert schema["required"] == ["prompt"]
        assert "prompt" in schema["properties"]

    def test_does_not_claim_seed_or_negative_prompt(self):
        """The web UI has neither — claiming otherwise misleads the scorer."""
        supports = GeminiWebImage().supports
        assert supports["seed"] is False
        assert supports["negative_prompt"] is False

    def test_declares_image_edit_for_selector_routing(self):
        """image_selector filters edit briefs on this supports key."""
        assert GeminiWebImage().supports["image_edit"] is True

    def test_has_agent_skills(self):
        assert "gemini-web-image" in GeminiWebImage().agent_skills

    def test_has_fallbacks(self):
        assert "google_imagen" in GeminiWebImage().fallback_tools

    def test_install_instructions_mention_login_and_playwright(self):
        text = GeminiWebImage().install_instructions
        assert "python -m tools._browser login gemini" in text
        assert "playwright" in text

    def test_declares_playwright_dependency(self):
        assert "python:playwright" in GeminiWebImage().dependencies

    def test_cost_is_zero_usd_because_billing_is_subscription(self):
        assert GeminiWebImage().estimate_cost({"prompt": "x"}) == 0.0

    def test_not_good_for_flags_serialization(self):
        assert any("parallel" in s for s in GeminiWebImage().not_good_for)

    def test_side_effects_mention_subscription_quota(self):
        assert any("quota" in s for s in GeminiWebImage().side_effects)

    def test_verification_mentions_watermark(self):
        assert any(
            "watermark" in s for s in GeminiWebImage().user_visible_verification
        )

    def test_get_info_returns_dict(self):
        info = GeminiWebImage().get_info()
        assert info["name"] == "gemini_web_image"
        assert info["runtime"] == "browser"


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------

class TestStatus:

    def test_unavailable_without_playwright(self, monkeypatch, browser_home):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: False
        )
        assert GeminiWebImage().get_status() == ToolStatus.UNAVAILABLE

    def test_unavailable_without_any_profile(self, monkeypatch, browser_home):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        assert GeminiWebImage().get_status() == ToolStatus.UNAVAILABLE

    def test_degraded_when_profile_exists_but_login_unconfirmed(
        self, monkeypatch, browser_home
    ):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        (browser_home / "gemini").mkdir(parents=True)
        assert GeminiWebImage().get_status() == ToolStatus.DEGRADED

    def test_available_after_confirmed_login(self, monkeypatch, logged_in):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        assert GeminiWebImage().get_status() == ToolStatus.AVAILABLE

    def test_setup_offer_names_the_login_command(self, monkeypatch, browser_home):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        offer = GeminiWebImage().get_info()["setup_offer"]
        assert offer["command"] == "python -m tools._browser login gemini"
        assert offer["kind"] == "one_time_login"

    def test_no_setup_offer_when_ready(self, monkeypatch, logged_in):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        assert "setup_offer" not in GeminiWebImage().get_info()


# ------------------------------------------------------------------
# Execute guards — these must not open a browser
# ------------------------------------------------------------------

class TestExecuteGuards:

    def test_empty_prompt_rejected(self, browser_home):
        result = GeminiWebImage().execute({"prompt": "  "})
        assert result.success is False
        assert "prompt is required" in result.error

    def test_missing_playwright_reported_as_setup_issue(self, monkeypatch, browser_home):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: False
        )
        result = GeminiWebImage().execute({"prompt": "a cat"})
        assert result.success is False
        assert "Playwright is not installed" in result.error

    def test_missing_login_classified_as_auth_not_prompt_failure(
        self, monkeypatch, browser_home
    ):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        result = GeminiWebImage().execute({"prompt": "a cat"})
        assert result.success is False
        assert "auth issue" in result.error
        assert "python -m tools._browser login gemini" in result.error

    def test_missing_reference_image_rejected_before_launch(
        self, monkeypatch, logged_in, tmp_path
    ):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        result = GeminiWebImage().execute({
            "prompt": "a cat", "image_paths": [str(tmp_path / "absent.png")],
        })
        assert result.success is False
        assert "not found" in result.error


# ------------------------------------------------------------------
# Prompt composition and reference handling
# ------------------------------------------------------------------

class TestPromptComposition:

    def test_aspect_ratio_is_written_into_the_prompt(self):
        composed = GeminiWebImage._compose_prompt("a cat", "9:16")
        assert "a cat" in composed
        assert "9:16" in composed

    def test_no_ratio_leaves_prompt_untouched(self):
        assert GeminiWebImage._compose_prompt("a cat", "") == "a cat"

    def test_newlines_are_collapsed(self):
        """Enter submits in the composer — a newline would send a partial prompt."""
        composed = GeminiWebImage._compose_prompt("line one\nline two\n\nline three", "")
        assert "\n" not in composed
        assert composed == "line one line two line three"

    def test_indentation_is_collapsed(self):
        composed = GeminiWebImage._compose_prompt("  a   cat  ", "")
        assert composed == "a cat"

    def test_single_image_path_folds_into_the_list(self):
        paths = GeminiWebImage._reference_paths({"image_path": "a.png"})
        assert paths == ["a.png"]

    def test_duplicate_reference_is_not_added_twice(self):
        paths = GeminiWebImage._reference_paths(
            {"image_paths": ["a.png"], "image_path": "a.png"}
        )
        assert paths == ["a.png"]

    def test_reference_order_is_preserved(self):
        paths = GeminiWebImage._reference_paths(
            {"image_paths": ["a.png", "b.png"], "image_path": "c.png"}
        )
        assert paths == ["a.png", "b.png", "c.png"]


# ------------------------------------------------------------------
# Response capture
# ------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, content_type, url="https://lh3.googleusercontent.com/img"):
        self.headers = {"content-type": content_type}
        self.url = url

    def body(self):  # pragma: no cover — must never be called from the handler
        raise AssertionError(
            "response.body() inside a sync-API event handler can deadlock"
        )


class TestImageUrlRecording:

    def test_records_image_urls(self):
        sink = []
        GeminiWebImage._record_image_url(_FakeResponse("image/png"), sink)
        assert sink == ["https://lh3.googleusercontent.com/img"]

    def test_handler_never_blocks_on_the_response_body(self):
        """Regression: fetching bytes in the handler deadlocks the sync API."""
        sink = []
        GeminiWebImage._record_image_url(_FakeResponse("image/png"), sink)  # would raise

    def test_ignores_non_image_responses(self):
        sink = []
        GeminiWebImage._record_image_url(_FakeResponse("application/json"), sink)
        assert sink == []

    def test_ignores_svg_ui_chrome(self):
        sink = []
        GeminiWebImage._record_image_url(_FakeResponse("image/svg+xml"), sink)
        assert sink == []

    def test_ignores_non_http_urls(self):
        sink = []
        GeminiWebImage._record_image_url(_FakeResponse("image/png", "data:image/png;base64,x"), sink)
        assert sink == []

    def test_does_not_record_duplicates(self):
        sink = []
        for _ in range(3):
            GeminiWebImage._record_image_url(_FakeResponse("image/png"), sink)
        assert len(sink) == 1

    def test_malformed_event_does_not_raise(self):
        class Broken:
            @property
            def headers(self):
                raise RuntimeError("event payload gone")

        sink = []
        GeminiWebImage._record_image_url(Broken(), sink)
        assert sink == []


class _FakePage:
    """Minimal page double for _await_image: fetch by URL, no DOM matches."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.waits = 0

    def evaluate(self, script, url):
        import base64

        body = self.bodies.get(url)
        return base64.b64encode(body).decode() if body else None

    def locator(self, selector):
        raise AssertionError("DOM fallback must not run before generation finishes")

    def wait_for_timeout(self, ms):
        self.waits += 1


class _FakeSession:
    def __init__(self, page):
        self.page = page
        self.selectors = {"generating": [], "response_image": []}

    def is_logged_out(self):
        return False

    def logged_out_error(self):
        return "gemini session is not logged in (auth issue)"


class TestAwaitImageIgnoresPreSubmitImages:
    """Regression: the uploaded reference must never be returned as the output."""

    def test_reference_echo_recorded_before_submit_is_ignored(self):
        reference = "https://lh3.googleusercontent.com/rd-gg-dl/uploaded-reference"
        generated = "https://lh3.googleusercontent.com/rd-gg-dl/generated"
        page = _FakePage({reference: b"R" * 200_000, generated: b"G" * 200_000})
        seen = [reference]

        # The generated image lands on the second poll.
        original_wait = page.wait_for_timeout

        def wait(ms):
            original_wait(ms)
            if page.waits == 1:
                seen.append(generated)

        page.wait_for_timeout = wait

        body = GeminiWebImage()._await_image(
            _FakeSession(page), seen, since_index=1, deadline=time.time() + 10
        )
        assert body == b"G" * 200_000

    def test_times_out_rather_than_returning_a_pre_submit_image(self):
        reference = "https://lh3.googleusercontent.com/rd-gg-dl/uploaded-reference"
        page = _FakePage({reference: b"R" * 200_000})
        with pytest.raises(TimeoutError):
            GeminiWebImage()._await_image(
                _FakeSession(page), [reference], since_index=1,
                deadline=time.time() + 1,
            )

    def test_logged_out_mid_wait_is_reported_as_auth(self):
        class LoggedOut(_FakeSession):
            def is_logged_out(self):
                return True

        with pytest.raises(RuntimeError, match="not logged in"):
            GeminiWebImage()._await_image(
                LoggedOut(_FakePage({})), [], since_index=0,
                deadline=time.time() + 5,
            )


class TestFetchBytes:

    def test_rejects_non_http_urls_without_touching_the_page(self):
        class Page:
            def evaluate(self, *args):
                raise AssertionError("must not evaluate for a non-http url")

        assert GeminiWebImage._fetch_bytes(Page(), "data:image/png;base64,x") is None
        assert GeminiWebImage._fetch_bytes(Page(), "") is None

    def test_decodes_base64_payload(self):
        import base64

        class Page:
            def evaluate(self, script, url):
                return base64.b64encode(b"image-bytes").decode()

        assert GeminiWebImage._fetch_bytes(Page(), "https://x/y.png") == b"image-bytes"

    def test_failed_fetch_returns_none(self):
        class Page:
            def evaluate(self, script, url):
                return None

        assert GeminiWebImage._fetch_bytes(Page(), "https://x/y.png") is None

    def test_page_error_returns_none(self):
        class Page:
            def evaluate(self, script, url):
                raise RuntimeError("navigated away")

        assert GeminiWebImage._fetch_bytes(Page(), "https://x/y.png") is None


# ------------------------------------------------------------------
# Watermark cleanup
# ------------------------------------------------------------------

class TestWatermarkCleanup:

    def test_skipped_when_disabled(self, tmp_path):
        path = tmp_path / "img.png"
        path.write_bytes(b"data")
        result = GeminiWebImage()._clean_watermark(path, {"remove_watermark": False})
        assert result["applied"] is False
        assert path.read_bytes() == b"data"

    def test_no_remover_available_warns_instead_of_failing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            GeminiWebImage, "_resolve_watermark_mode", staticmethod(lambda requested: None)
        )
        path = tmp_path / "img.png"
        path.write_bytes(b"data")
        result = GeminiWebImage()._clean_watermark(path, {})
        assert result["applied"] is False
        assert "still in the bottom-right corner" in result["reason"]
        assert "pip install" in result["reason"], "tell the user how to fix it"
        assert path.exists(), "the raw image must survive a failed cleanup"

    def test_unreadable_image_reports_the_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        path = tmp_path / "img.png"
        path.write_bytes(b"not an image")
        result = GeminiWebImage()._clean_watermark(path, {"watermark_mode": "telea"})
        assert result["applied"] is False
        assert path.read_bytes() == b"not an image"

    def test_ffmpeg_failure_keeps_the_original(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg")

        def boom(*args, **kwargs):
            raise RuntimeError("ffmpeg exploded")

        monkeypatch.setattr("subprocess.run", boom)
        path = tmp_path / "img.png"
        path.write_bytes(b"data")
        result = GeminiWebImage()._clean_watermark(path, {})
        assert result["applied"] is False
        assert path.read_bytes() == b"data"

    def test_delogo_replaces_the_image_in_place(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg")
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            # ffmpeg writes the last argument
            open(cmd[-1], "wb").write(b"cleaned")

            class Done:
                returncode = 0
            return Done()

        monkeypatch.setattr("subprocess.run", fake_run)
        path = tmp_path / "img.png"
        path.write_bytes(b"raw")
        result = GeminiWebImage()._clean_watermark(path, {"watermark_mode": "delogo"})
        assert result["applied"] is True
        assert result["mode"] == "delogo"
        assert path.read_bytes() == b"cleaned"
        assert "delogo=" in " ".join(captured["cmd"])

    def test_delogo_box_stays_inside_the_frame(self, tmp_path, monkeypatch):
        """ffmpeg's delogo exits -22 when the box touches an image edge."""
        monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg")
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            open(cmd[-1], "wb").write(b"cleaned")

            class Done:
                returncode = 0
            return Done()

        monkeypatch.setattr("subprocess.run", fake_run)
        path = tmp_path / "img.png"
        path.write_bytes(b"raw")
        region = {"x": 972, "y": 512, "width": 27, "height": 24,
                  "image_width": 1024, "image_height": 572}
        monkeypatch.setattr(
            GeminiWebImage, "_watermark_region", lambda self, p, i: region
        )
        GeminiWebImage()._clean_watermark(path, {"watermark_mode": "delogo"})
        filter_arg = [c for c in captured["cmd"] if c.startswith("delogo=")][0]
        assert filter_arg == "delogo=x=972:y=512:w=27:h=24"
        assert region["x"] + region["width"] < region["image_width"]
        assert region["y"] + region["height"] < region["image_height"]

    def test_crop_mode_uses_crop_filter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg")
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            open(cmd[-1], "wb").write(b"cleaned")

            class Done:
                returncode = 0
            return Done()

        monkeypatch.setattr("subprocess.run", fake_run)
        path = tmp_path / "img.png"
        path.write_bytes(b"raw")
        GeminiWebImage()._clean_watermark(path, {"watermark_mode": "crop"})
        assert "crop=" in " ".join(captured["cmd"])

    def test_keep_raw_preserves_the_original_alongside(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg")

        def fake_run(cmd, **kwargs):
            open(cmd[-1], "wb").write(b"cleaned")

            class Done:
                returncode = 0
            return Done()

        monkeypatch.setattr("subprocess.run", fake_run)
        path = tmp_path / "img.png"
        path.write_bytes(b"raw")
        GeminiWebImage()._clean_watermark(
            path, {"keep_raw": True, "watermark_mode": "delogo"}
        )
        assert path.read_bytes() == b"cleaned"
        assert (tmp_path / "img.raw.png").read_bytes() == b"raw"


class TestWatermarkModeSelection:

    def test_explicit_mode_is_honoured_when_available(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg")
        assert GeminiWebImage._resolve_watermark_mode("delogo") == "delogo"

    def test_explicit_unavailable_mode_is_not_silently_swapped(self, monkeypatch):
        """A silent downgrade would hand back a worse repair than asked for."""
        monkeypatch.setattr(shutil, "which", lambda name: None)
        assert GeminiWebImage._resolve_watermark_mode("delogo") is None

    def test_auto_prefers_texture_reconstruction(self):
        """fsr beats telea on texture, so auto must not settle for telea."""
        pytest.importorskip("cv2")
        import cv2

        if not hasattr(cv2, "xphoto"):
            pytest.skip("opencv-contrib not installed")
        assert GeminiWebImage._resolve_watermark_mode("auto") == "fsr"

    def test_every_mode_has_a_documented_quality(self):
        from tools.graphics.gemini_web_image import _WATERMARK_QUALITY

        modes = set(GeminiWebImage().input_schema["properties"]["watermark_mode"]["enum"])
        assert modes - {"auto"} == set(_WATERMARK_QUALITY)


class TestWatermarkRegion:
    """The mask must hug the 15x12px sparkle, not repaint a fifth of the frame."""

    def test_box_is_small_and_in_the_bottom_right(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image

        path = tmp_path / "img.png"
        Image.new("RGB", (1024, 572), "white").save(path)
        region = GeminiWebImage()._watermark_region(path, {})
        assert region["source"] == "fixed_geometry"
        # Measured mark: 15x12 at (978,518). The box must contain it with pad.
        assert region["x"] <= 978 and region["y"] <= 518
        assert region["x"] + region["width"] >= 993
        assert region["y"] + region["height"] >= 530
        area = region["width"] * region["height"]
        assert area / (1024 * 572) < 0.005, "a big box leaves a visible patch"

    def test_box_never_touches_the_frame_edge(self, tmp_path):
        from PIL import Image

        for size in ((1024, 572), (768, 1024), (512, 512), (2048, 1152)):
            path = tmp_path / f"img_{size[0]}x{size[1]}.png"
            Image.new("RGB", size, "white").save(path)
            region = GeminiWebImage()._watermark_region(path, {})
            assert region["x"] >= 1
            assert region["y"] >= 1
            assert region["x"] + region["width"] < size[0]
            assert region["y"] + region["height"] < size[1]

    def test_caller_override_wins(self, tmp_path):
        from PIL import Image

        path = tmp_path / "img.png"
        Image.new("RGB", (1024, 572), "white").save(path)
        region = GeminiWebImage()._watermark_region(
            path, {"watermark_box": {"x": 10, "y": 20, "width": 30, "height": 40}}
        )
        assert region["source"] == "caller_override"
        assert (region["x"], region["y"], region["width"], region["height"]) == (10, 20, 30, 40)

    def test_unreadable_image_falls_back_without_raising(self, tmp_path):
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")
        region = GeminiWebImage()._watermark_region(path, {})
        assert region["width"] > 0 and region["height"] > 0


class TestInpaintQuality:
    """Does the repair actually remove the mark and keep the texture?"""

    def _textured_image_with_mark(self, path, region):
        import cv2
        import numpy as np

        rng = np.random.default_rng(7)
        # Build in int16 — numpy 2 refuses to add a negative int to a uint8 array.
        base = rng.integers(60, 90, size=(572, 1024, 3)).astype(np.int16)
        # Horizontal ripples, like the water this was verified against.
        ripple = (25 * np.sin(np.arange(572) / 3.0)).astype(np.int16)
        base += ripple[:, None, None]
        base = np.clip(base, 0, 255).astype(np.uint8)
        marked = base.copy()
        marked[region["y"]:region["y"] + region["height"],
               region["x"]:region["x"] + region["width"]] = 235
        cv2.imwrite(str(path), marked)
        return base, marked

    def test_fsr_removes_the_mark_and_matches_the_surroundings(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        if not hasattr(cv2, "xphoto"):
            pytest.skip("opencv-contrib not installed")
        import numpy as np

        path = tmp_path / "marked.png"
        region = {"x": 972, "y": 512, "width": 27, "height": 24,
                  "image_width": 1024, "image_height": 572}
        base, marked = self._textured_image_with_mark(path, region)

        GeminiWebImage()._inpaint_opencv(path, region, "fsr")
        repaired = cv2.imread(str(path), cv2.IMREAD_COLOR)

        def patch(img):
            return img[region["y"]:region["y"] + region["height"],
                       region["x"]:region["x"] + region["width"]].astype(float)

        # The bright mark is gone: the patch is no longer far brighter than the
        # texture it sits in.
        neighbourhood = base[region["y"] - 40:region["y"], region["x"]:region["x"] + region["width"]]
        assert patch(marked).mean() - neighbourhood.mean() > 120, "fixture sanity"
        assert abs(patch(repaired).mean() - neighbourhood.mean()) < 40

    def test_only_the_masked_region_changes(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        path = tmp_path / "marked.png"
        region = {"x": 972, "y": 512, "width": 27, "height": 24,
                  "image_width": 1024, "image_height": 572}
        self._textured_image_with_mark(path, region)
        before = cv2.imread(str(path), cv2.IMREAD_COLOR).copy()

        GeminiWebImage()._inpaint_opencv(path, region, "telea")
        after = cv2.imread(str(path), cv2.IMREAD_COLOR)

        outside_before = before.copy()
        outside_after = after.copy()
        for img in (outside_before, outside_after):
            img[region["y"]:region["y"] + region["height"],
                region["x"]:region["x"] + region["width"]] = 0
        changed = int(np.count_nonzero(cv2.absdiff(outside_before, outside_after) > 2))
        # Telea diffuses a little into the border; a rewrite of the whole frame
        # (e.g. a re-encode bug) would change orders of magnitude more.
        assert changed < 2000, f"{changed} pixels outside the mask changed"


class TestImageFormat:
    """Gemini serves JPEG — the file must not claim to be something it isn't."""

    def test_sniffs_jpeg(self):
        assert GeminiWebImage._sniff_format(b"\xff\xd8\xff\xe0rest") == "jpeg"

    def test_sniffs_png(self):
        assert GeminiWebImage._sniff_format(b"\x89PNG\r\n\x1a\nrest") == "png"

    def test_sniffs_webp(self):
        assert GeminiWebImage._sniff_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"

    def test_unknown_bytes(self):
        assert GeminiWebImage._sniff_format(b"nonsense") == "unknown"

    def test_jpeg_bytes_written_to_a_png_path_are_converted(self, tmp_path):
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), "red").save(buffer, format="JPEG")
        data = buffer.getvalue()
        assert GeminiWebImage._sniff_format(data) == "jpeg"

        path = tmp_path / "out.png"
        stored = GeminiWebImage._write_image(path, data, "jpeg")
        assert stored == "png"
        assert GeminiWebImage._sniff_format(path.read_bytes()) == "png"

    def test_matching_suffix_is_written_untouched(self, tmp_path):
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buffer, format="JPEG")
        data = buffer.getvalue()
        path = tmp_path / "out.jpg"
        assert GeminiWebImage._write_image(path, data, "jpeg") == "jpeg"
        assert path.read_bytes() == data

    def test_undecodable_bytes_are_kept_verbatim(self, tmp_path):
        path = tmp_path / "out.png"
        assert GeminiWebImage._write_image(path, b"garbage", "unknown") == "png"
        assert path.read_bytes() == b"garbage"


# ------------------------------------------------------------------
# Shared browser-session infrastructure
# ------------------------------------------------------------------

class TestBrowserSession:

    def test_profile_dir_lives_outside_the_repo_by_default(self, monkeypatch):
        monkeypatch.delenv("OPENMONTAGE_BROWSER_PROFILE_DIR", raising=False)
        from pathlib import Path

        from tools._browser.session import profile_dir_for

        path = profile_dir_for("gemini")
        assert Path.home() in path.parents, "session cookies must not live in the repo"

    def test_profile_dir_honors_the_env_override(self, browser_home):
        from tools._browser.session import profile_dir_for
        assert profile_dir_for("gemini") == browser_home / "gemini"

    def test_login_state_reports_not_set_up(self, browser_home):
        from tools._browser.session import login_state
        state = login_state("gemini")
        assert state.ready is False
        assert state.profile_exists is False

    def test_login_marker_makes_state_ready(self, logged_in):
        from tools._browser.session import login_state
        state = login_state("gemini")
        assert state.ready is True
        assert state.logged_in_at

    def test_corrupt_marker_does_not_raise(self, browser_home):
        from tools._browser.session import login_state
        profile = browser_home / "gemini"
        profile.mkdir(parents=True)
        (profile / "openmontage_login.json").write_text("not json", encoding="utf-8")
        state = login_state("gemini")
        assert state.ready is True
        assert state.logged_in_at is None

    def test_profile_lock_is_exclusive(self, browser_home):
        from tools._browser.session import BrowserLockTimeout, _ProfileLock

        first = _ProfileLock("gemini", timeout_seconds=0.5)
        first.acquire()
        try:
            with pytest.raises(BrowserLockTimeout):
                _ProfileLock("gemini", timeout_seconds=0.5).acquire()
        finally:
            first.release()

    def test_lock_is_reusable_after_release(self, browser_home):
        from tools._browser.session import _ProfileLock

        first = _ProfileLock("gemini", timeout_seconds=0.5)
        first.acquire()
        first.release()
        second = _ProfileLock("gemini", timeout_seconds=0.5)
        second.acquire()
        second.release()

    def test_lock_from_a_dead_process_is_broken_immediately(self, browser_home):
        """A window closed with the X never releases — don't block the next run."""
        import json

        from tools._browser import session as session_module

        profile = browser_home / "gemini"
        profile.mkdir(parents=True)
        lock_path = profile / "openmontage.lock"
        # A pid that cannot be running: the max pid value is never assigned.
        lock_path.write_text(json.dumps({"pid": 0x7FFFFFFF, "since": 0}), encoding="utf-8")

        lock = session_module._ProfileLock("gemini", timeout_seconds=5)
        lock.acquire()  # must not wait out the staleness window
        lock.release()

    def test_lock_held_by_a_live_process_is_respected(self, browser_home):
        import json
        import os

        from tools._browser import session as session_module

        profile = browser_home / "gemini"
        profile.mkdir(parents=True)
        (profile / "openmontage.lock").write_text(
            json.dumps({"pid": os.getpid(), "since": 0}), encoding="utf-8"
        )

        with pytest.raises(session_module.BrowserLockTimeout):
            session_module._ProfileLock("gemini", timeout_seconds=0.5).acquire()

    def test_pid_alive_reports_this_process(self):
        import os

        from tools._browser.session import _pid_alive

        assert _pid_alive(os.getpid()) is True
        assert _pid_alive(0x7FFFFFFF) is False
        assert _pid_alive(0) is False

    def test_stale_lock_is_broken(self, browser_home, monkeypatch):
        import os
        import time

        from tools._browser import session as session_module

        first = session_module._ProfileLock("gemini", timeout_seconds=0.5)
        first.acquire()
        lock_path = browser_home / "gemini" / "openmontage.lock"
        stale = time.time() - session_module._LOCK_STALE_SECONDS - 60
        os.utime(lock_path, (stale, stale))

        second = session_module._ProfileLock("gemini", timeout_seconds=2)
        second.acquire()  # must not raise — the old holder is presumed dead
        second.release()


class _LoginSession:
    """Session double for the login waiter: signed in after N polls."""

    def __init__(self, signed_in_after=0, ever_signs_in=True):
        self.polls = 0
        self.signed_in_after = signed_in_after
        self.ever_signs_in = ever_signs_in

    def is_logged_out(self):
        self.polls += 1
        if not self.ever_signs_in:
            return True
        return self.polls <= self.signed_in_after

    def first_locator(self, key, timeout_ms=0):
        if self.app_ready:
            return object()
        raise LookupError("app not rendered yet")

    @property
    def app_ready(self):
        return self.ever_signs_in and self.polls > self.signed_in_after


class TestLoginWaiter:
    """The login must complete on its own — a keypress is too easy to miss."""

    def _clock(self):
        state = {"t": 0.0}

        def now():
            return state["t"]

        def sleep(seconds):
            state["t"] += seconds

        return now, sleep

    def test_detects_an_already_signed_in_session_immediately(self):
        from tools._browser.cli import wait_for_signed_in

        now, sleep = self._clock()
        session = _LoginSession(signed_in_after=0)
        assert wait_for_signed_in(
            session, timeout_seconds=60, now=now, sleep=sleep, report=lambda *a: None
        ) is True

    def test_detects_sign_in_that_happens_while_waiting(self):
        from tools._browser.cli import wait_for_signed_in

        now, sleep = self._clock()
        session = _LoginSession(signed_in_after=5)
        assert wait_for_signed_in(
            session, timeout_seconds=600, now=now, sleep=sleep, report=lambda *a: None
        ) is True
        assert session.polls > 5

    def test_gives_up_after_the_timeout(self):
        from tools._browser.cli import wait_for_signed_in

        now, sleep = self._clock()
        session = _LoginSession(ever_signs_in=False)
        assert wait_for_signed_in(
            session, timeout_seconds=30, now=now, sleep=sleep, report=lambda *a: None
        ) is False

    def test_a_loading_page_is_not_mistaken_for_signed_in(self):
        """Absence of a sign-in link is not proof — the app must have rendered."""
        from tools._browser.cli import wait_for_signed_in

        class BlankPage(_LoginSession):
            def is_logged_out(self):
                self.polls += 1
                return False  # no sign-in link on a blank page either

            def first_locator(self, key, timeout_ms=0):
                raise LookupError("composer never rendered")

        now, sleep = self._clock()
        assert wait_for_signed_in(
            BlankPage(), timeout_seconds=30, now=now, sleep=sleep, report=lambda *a: None
        ) is False

    def test_reports_progress_while_waiting(self):
        from tools._browser.cli import wait_for_signed_in

        now, sleep = self._clock()
        lines = []
        wait_for_signed_in(
            _LoginSession(ever_signs_in=False), timeout_seconds=120,
            now=now, sleep=sleep, report=lines.append,
        )
        assert lines, "a long wait with no output looks like a hang"


class TestSelectors:

    def test_gemini_selectors_present(self):
        from tools._browser.selectors import selectors_for
        selectors = selectors_for("gemini")
        for key in ("prompt_input", "file_input", "logged_out", "response_image"):
            assert selectors[key], f"missing selector list: {key}"

    def test_unknown_provider_returns_empty(self):
        from tools._browser.selectors import selectors_for
        assert selectors_for("nope") == {}

    def test_user_override_replaces_a_key(self, tmp_path, monkeypatch):
        override = tmp_path / "selectors.json"
        override.write_text(
            json.dumps({"gemini": {"prompt_input": ["div.custom"]}}), encoding="utf-8"
        )
        monkeypatch.setenv("OPENMONTAGE_BROWSER_SELECTORS", str(override))
        from tools._browser.selectors import selectors_for
        selectors = selectors_for("gemini")
        assert selectors["prompt_input"] == ["div.custom"]
        assert selectors["file_input"], "unrelated keys must survive an override"

    def test_string_override_is_accepted(self, tmp_path, monkeypatch):
        override = tmp_path / "selectors.json"
        override.write_text(
            json.dumps({"gemini": {"prompt_input": "div.single"}}), encoding="utf-8"
        )
        monkeypatch.setenv("OPENMONTAGE_BROWSER_SELECTORS", str(override))
        from tools._browser.selectors import selectors_for
        assert selectors_for("gemini")["prompt_input"] == ["div.single"]

    def test_broken_override_file_is_ignored(self, tmp_path, monkeypatch):
        override = tmp_path / "selectors.json"
        override.write_text("{ not json", encoding="utf-8")
        monkeypatch.setenv("OPENMONTAGE_BROWSER_SELECTORS", str(override))
        from tools._browser.selectors import selectors_for
        assert selectors_for("gemini")["prompt_input"]

    def test_base_table_is_not_mutated_by_overrides(self, tmp_path, monkeypatch):
        override = tmp_path / "selectors.json"
        override.write_text(
            json.dumps({"gemini": {"prompt_input": ["div.custom"]}}), encoding="utf-8"
        )
        monkeypatch.setenv("OPENMONTAGE_BROWSER_SELECTORS", str(override))
        from tools._browser.selectors import SELECTORS, selectors_for
        selectors_for("gemini")
        monkeypatch.delenv("OPENMONTAGE_BROWSER_SELECTORS")
        assert selectors_for("gemini")["prompt_input"] == SELECTORS["gemini"]["prompt_input"]


# ------------------------------------------------------------------
# Import safety and registry discovery
# ------------------------------------------------------------------

class TestImportSafety:

    def test_module_does_not_import_playwright_at_module_level(self):
        """registry.discover() imports every tool module — playwright may be absent."""
        import ast
        import inspect

        import tools.graphics.gemini_web_image as module

        tree = ast.parse(inspect.getsource(module))
        for node in tree.body:  # module level only
            if isinstance(node, ast.Import):
                assert all("playwright" not in a.name for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert "playwright" not in (node.module or "")

    def test_session_module_does_not_import_playwright_at_module_level(self):
        import ast
        import inspect

        import tools._browser.session as module

        tree = ast.parse(inspect.getsource(module))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all("playwright" not in a.name for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert "playwright" not in (node.module or "")

    def test_status_works_without_playwright_installed(self, monkeypatch, browser_home):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("playwright"):
                raise ImportError("no playwright here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert GeminiWebImage().get_status() == ToolStatus.UNAVAILABLE


class TestRegistryDiscovery:

    def test_discoverable(self):
        from tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        registry.discover()
        assert "gemini_web_image" in {t.name for t in registry._tools.values()}

    def test_browser_runtime_is_a_valid_runtime_value(self):
        assert ToolRuntime.BROWSER.value == "browser"

    def test_registered_under_image_generation(self):
        from tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        registry.discover()
        names = {t.name for t in registry.get_by_capability("image_generation")}
        assert "gemini_web_image" in names


class TestSchemaValidation:

    def test_valid_payload_passes_schema(self):
        import jsonschema
        jsonschema.validate(
            {
                "prompt": "a cat", "aspect_ratio": "9:16",
                "image_paths": ["a.png"], "remove_watermark": True,
            },
            GeminiWebImage().input_schema,
        )

    def test_prompt_is_required(self):
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"aspect_ratio": "16:9"}, GeminiWebImage().input_schema)

    def test_unsupported_watermark_mode_rejected(self):
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {"prompt": "x", "watermark_mode": "blur"},
                GeminiWebImage().input_schema,
            )
