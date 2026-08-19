"""Contract tests for the Suno web (subscription-backed) music tool.

These tests never launch a browser, never sign in, and never spend a Suno
credit. Page interaction is exercised through fakes so the parsing, dedupe,
quota, and output logic is covered without a live session.

Run: pytest tests/contracts/test_suno_web_music.py -v
"""

import json
from pathlib import Path

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
from tools.audio.suno_web_music import SunoWebMusic, _QuotaExhausted


CLIP_A = "https://cdn1.suno.ai/11111111-2222-3333-4444-555555555555.mp3"
CLIP_B = "https://cdn1.suno.ai/66666666-7777-8888-9999-000000000000.mp3"

# Smallest thing that both sniffs as mp3 and clears the size floor.
MP3_BYTES = b"ID3" + b"\x00" * 200_000


@pytest.fixture(autouse=True)
def browser_home(tmp_path, monkeypatch):
    """Redirect the browser profile root so tests never touch the real one.

    autouse on purpose. With the wrong env var name this fixture silently let
    a test write a login marker into the user's REAL profile, which made the
    login guard pass and launched an actual browser against suno.com. Applying
    it to every test in the module removes that whole failure mode.
    """
    monkeypatch.setenv("OPENMONTAGE_BROWSER_PROFILE_DIR", str(tmp_path / "browser"))
    monkeypatch.delenv("OPENMONTAGE_BROWSER_SELECTORS", raising=False)
    return tmp_path / "browser"


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch):
    """Hard stop: a contract test must never launch Chromium."""
    def explode(*args, **kwargs):
        raise AssertionError(
            "a contract test tried to open a real browser session — "
            "check the login guards and fakes"
        )
    monkeypatch.setattr("tools._browser.session.BrowserSession.__enter__", explode)


# ------------------------------------------------------------------
# Contract compliance
# ------------------------------------------------------------------

class TestContract:

    def test_inherits_base_tool(self):
        assert issubclass(SunoWebMusic, BaseTool)

    def test_has_required_identity(self):
        tool = SunoWebMusic()
        assert tool.name == "suno_web_music"
        assert tool.version
        assert tool.provider == "suno_web"
        assert tool.capability == "music_generation"
        assert tool.tier == ToolTier.GENERATE
        assert tool.stability == ToolStability.EXPERIMENTAL

    def test_runtime_is_browser(self):
        """The whole point: subscription session, not an API key."""
        assert SunoWebMusic().runtime == ToolRuntime.BROWSER

    def test_execution_mode_is_sync(self):
        assert SunoWebMusic().execution_mode == ExecutionMode.SYNC

    def test_determinism_is_stochastic(self):
        assert SunoWebMusic().determinism == Determinism.STOCHASTIC

    def test_declares_no_api_key_dependency(self):
        """A browser tool must not demand an env var — that's the whole point."""
        deps = SunoWebMusic().dependencies
        assert not [d for d in deps if d.startswith("env:")]
        assert "python:playwright" in deps

    def test_install_instructions_name_the_login_command(self):
        text = SunoWebMusic().install_instructions
        assert "python -m tools._browser login suno" in text
        assert "playwright" in text

    def test_has_input_schema(self):
        schema = SunoWebMusic().input_schema
        assert schema.get("type") == "object"
        assert schema.get("required") == ["prompt"]
        assert "prompt" in schema["properties"]

    def test_has_agent_skills(self):
        assert "suno-web-music" in SunoWebMusic().agent_skills

    def test_has_fallbacks(self):
        fallbacks = SunoWebMusic().fallback_tools
        assert "pixabay_music" in fallbacks
        assert "google_music" in fallbacks

    def test_supports_declares_no_seed(self):
        """The web UI exposes no seed — scoring/idempotency read this field."""
        assert SunoWebMusic().supports["seed"] is False

    def test_supports_declares_subscription_billing(self):
        assert SunoWebMusic().supports["subscription_billing"] is True

    def test_supports_declares_no_exact_duration(self):
        """Suno picks the length; callers must plan to trim or loop."""
        assert SunoWebMusic().supports["exact_duration"] is False

    def test_cost_is_zero(self):
        """Billed to the subscription, so USD cost must be 0 — not a guess."""
        assert SunoWebMusic().estimate_cost({"prompt": "x"}) == 0.0

    def test_estimate_runtime_is_positive(self):
        assert SunoWebMusic().estimate_runtime({"prompt": "x"}) > 0

    def test_no_silent_retries(self):
        """Regeneration spends credits — the agent decides, not the tool."""
        assert SunoWebMusic().retry_policy.max_retries == 0

    def test_has_side_effects_naming_credits(self):
        side = SunoWebMusic().side_effects
        assert any("credit" in s.lower() for s in side)

    def test_has_user_visible_verification(self):
        assert len(SunoWebMusic().user_visible_verification) > 0

    def test_resource_profile_requires_network(self):
        assert SunoWebMusic().resource_profile.network_required is True

    def test_dry_run_returns_dict(self):
        result = SunoWebMusic().dry_run({"prompt": "test"})
        assert result["tool"] == "suno_web_music"


# ------------------------------------------------------------------
# Lazy imports — registry.discover() must work without Playwright
# ------------------------------------------------------------------

class TestLazyImports:

    def test_module_imports_without_playwright(self, monkeypatch):
        """discover() walks every module; a top-level playwright import breaks it."""
        import importlib
        import sys
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        importlib.reload(sys.modules["tools.audio.suno_web_music"])
        # Reload again with the stubs gone so later tests see a clean module.
        monkeypatch.undo()
        importlib.reload(sys.modules["tools.audio.suno_web_music"])

    def test_source_has_no_top_level_playwright_import(self):
        source = Path("tools/audio/suno_web_music.py").read_text(encoding="utf-8")
        head = source.split("class SunoWebMusic")[0]
        assert "import playwright" not in head
        assert "from playwright" not in head


# ------------------------------------------------------------------
# Status reporting
# ------------------------------------------------------------------

class TestStatus:

    def test_unavailable_without_playwright(self, monkeypatch):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: False
        )
        assert SunoWebMusic().get_status() == ToolStatus.UNAVAILABLE

    def test_unavailable_without_a_profile(self, browser_home, monkeypatch):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        assert SunoWebMusic().get_status() == ToolStatus.UNAVAILABLE

    def test_degraded_when_profile_exists_but_login_unconfirmed(
        self, browser_home, monkeypatch
    ):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        (browser_home / "suno").mkdir(parents=True)
        assert SunoWebMusic().get_status() == ToolStatus.DEGRADED

    def test_available_after_login_marker(self, browser_home, monkeypatch):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        from tools._browser.session import write_login_marker
        write_login_marker("suno")
        assert SunoWebMusic().get_status() == ToolStatus.AVAILABLE

    def test_setup_offer_present_when_unavailable(self, browser_home, monkeypatch):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        info = SunoWebMusic().get_info()
        offer = info.get("setup_offer")
        assert offer and offer["kind"] == "one_time_login"
        assert offer["command"] == "python -m tools._browser login suno"

    def test_no_setup_offer_when_available(self, browser_home, monkeypatch):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        from tools._browser.session import write_login_marker
        write_login_marker("suno")
        assert "setup_offer" not in SunoWebMusic().get_info()


# ------------------------------------------------------------------
# Guard rails before any browser work
# ------------------------------------------------------------------

class TestPreflightGuards:

    def test_empty_prompt_rejected(self):
        result = SunoWebMusic().execute({"prompt": "   "})
        assert result.success is False
        assert "prompt is required" in result.error

    def test_missing_prompt_rejected(self):
        result = SunoWebMusic().execute({})
        assert result.success is False

    def test_no_playwright_returns_actionable_error(self, monkeypatch):
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: False
        )
        result = SunoWebMusic().execute({"prompt": "ambient bed"})
        assert result.success is False
        assert "Playwright is not installed" in result.error

    def test_not_logged_in_error_is_classified_as_auth(
        self, browser_home, monkeypatch
    ):
        """Escalation protocol needs auth failures to name themselves."""
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        result = SunoWebMusic().execute({"prompt": "ambient bed"})
        assert result.success is False
        assert "auth issue" in result.error
        assert "python -m tools._browser login suno" in result.error


# ------------------------------------------------------------------
# Audio URL sniffing
# ------------------------------------------------------------------

class FakeResponse:
    def __init__(self, url):
        self.url = url


class TestAudioUrlRecording:

    def test_records_suno_cdn_mp3(self):
        sink = []
        SunoWebMusic._record_audio_url(FakeResponse(CLIP_A), sink)
        assert sink == [CLIP_A]

    def test_records_url_with_query_string(self):
        sink = []
        SunoWebMusic._record_audio_url(FakeResponse(CLIP_A + "?token=abc"), sink)
        assert len(sink) == 1

    def test_ignores_non_audio_assets(self):
        sink = []
        for url in (
            "https://suno.com/_next/static/chunk.js",
            "https://cdn1.suno.ai/image_abc.png",
            "https://suno.com/api/feed/",
            "https://fonts.googleapis.com/css",
        ):
            SunoWebMusic._record_audio_url(FakeResponse(url), sink)
        assert sink == []

    def test_ignores_unrelated_host_mp3(self):
        """A random mp3 from another host is not our generated clip."""
        sink = []
        SunoWebMusic._record_audio_url(
            FakeResponse("https://example.com/notification.mp3"), sink
        )
        assert sink == []

    def test_deduplicates_identical_url(self):
        sink = []
        SunoWebMusic._record_audio_url(FakeResponse(CLIP_A), sink)
        SunoWebMusic._record_audio_url(FakeResponse(CLIP_A), sink)
        assert len(sink) == 1

    def test_listener_never_raises(self):
        class Exploding:
            @property
            def url(self):
                raise RuntimeError("boom")

        sink = []
        SunoWebMusic._record_audio_url(Exploding(), sink)  # must not raise
        assert sink == []


class TestClipDedupe:

    def test_collapses_range_requests_for_one_clip(self):
        """Playback re-requests the same clip id with different query strings."""
        urls = [CLIP_A, CLIP_A + "?range=1", CLIP_A + "?range=2"]
        assert SunoWebMusic._dedupe_clips(urls) == [CLIP_A]

    def test_keeps_distinct_clips_in_arrival_order(self):
        assert SunoWebMusic._dedupe_clips([CLIP_A, CLIP_B]) == [CLIP_A, CLIP_B]

    def test_two_candidates_survive_interleaved_requests(self):
        urls = [CLIP_A, CLIP_B, CLIP_A + "?x=1", CLIP_B + "?x=2"]
        assert SunoWebMusic._dedupe_clips(urls) == [CLIP_A, CLIP_B]

    def test_falls_back_to_path_when_no_uuid(self):
        urls = ["https://cdn1.suno.ai/track.mp3", "https://cdn1.suno.ai/track.mp3?a=1"]
        assert SunoWebMusic._dedupe_clips(urls) == ["https://cdn1.suno.ai/track.mp3"]

    def test_empty_input(self):
        assert SunoWebMusic._dedupe_clips([]) == []


# ------------------------------------------------------------------
# Format sniffing
# ------------------------------------------------------------------

class TestFormatSniffing:

    def test_detects_mp3_by_id3(self):
        assert SunoWebMusic._sniff_format(b"ID3\x04\x00", "x") == "mp3"

    def test_detects_mp3_by_frame_sync(self):
        assert SunoWebMusic._sniff_format(b"\xff\xfb\x90\x00", "x") == "mp3"

    def test_detects_wav(self):
        assert SunoWebMusic._sniff_format(b"RIFF....WAVE", "x") == "wav"

    def test_detects_m4a(self):
        assert SunoWebMusic._sniff_format(b"\x00\x00\x00\x20ftypM4A ", "x") == "m4a"

    def test_detects_ogg(self):
        assert SunoWebMusic._sniff_format(b"OggS\x00\x02", "x") == "ogg"

    def test_detects_flac(self):
        assert SunoWebMusic._sniff_format(b"fLaC\x00", "x") == "flac"

    def test_falls_back_to_url_extension(self):
        assert SunoWebMusic._sniff_format(b"\x00\x01\x02\x03", "http://x/a.wav") == "wav"

    def test_defaults_to_mp3_when_unknowable(self):
        assert SunoWebMusic._sniff_format(b"\x00\x01\x02\x03", "http://x/a") == "mp3"


# ------------------------------------------------------------------
# Quota exhaustion is terminal
# ------------------------------------------------------------------

class FakeLocator:
    def __init__(self, *, visible=False, text="", attrs=None):
        self._visible = visible
        self._text = text
        self._attrs = attrs or {}
        self.clicked = 0
        self.filled = None

    @property
    def first(self):
        return self

    def is_visible(self, timeout=None):
        return self._visible

    def inner_text(self):
        return self._text

    def get_attribute(self, name):
        return self._attrs.get(name)

    def count(self):
        return 1 if self._visible else 0

    def click(self):
        self.clicked += 1

    def fill(self, text):
        self.filled = text


class FakeSession:
    def __init__(self, selectors, locators=None):
        self.selectors = selectors
        self._locators = locators or {}
        self.page = self

    # page surface
    def locator(self, selector):
        return self._locators.get(selector, FakeLocator(visible=False))

    def wait_for_timeout(self, ms):
        return None

    def first_locator(self, key, timeout_ms=None):
        if key in self._locators:
            return self._locators[key]
        raise LookupError(f"no selector for {key}")


class TestQuotaExhaustion:

    def test_raises_when_out_of_credits_banner_visible(self):
        banner = FakeLocator(visible=True, text="You are out of credits")
        session = FakeSession(
            {"quota_exhausted": ["#banner"]}, {"#banner": banner}
        )
        # Resolve the class off the live module: TestLazyImports reloads it,
        # which rebinds _QuotaExhausted to a new object. The module-level
        # import in this file would then be a stale, non-matching class.
        from tools.audio import suno_web_music as mod
        with pytest.raises(mod._QuotaExhausted, match="out of credits"):
            mod.SunoWebMusic()._raise_if_out_of_credits(session)

    def test_silent_when_no_banner(self):
        session = FakeSession({"quota_exhausted": ["#banner"]}, {})
        SunoWebMusic()._raise_if_out_of_credits(session)  # must not raise

    def test_quota_error_message_says_do_not_retry(self, browser_home, monkeypatch):
        """A terminal error must tell the agent not to burn another attempt."""
        monkeypatch.setattr(
            "tools._browser.session.playwright_available", lambda: True
        )
        from tools._browser.session import write_login_marker
        write_login_marker("suno")

        from tools.audio import suno_web_music as mod
        tool = mod.SunoWebMusic()
        quota_error = mod._QuotaExhausted

        # Drive the quota branch directly: replace BrowserSession with a fake
        # whose first page action raises the quota error.
        class Session:
            def __init__(self, *a, **k):
                self.selectors = {}
                self.page = self

            def __enter__(self):
                return self

            def on(self, *a, **k):
                pass

            def goto(self, *a, **k):
                raise quota_error("no credits remaining")

            def is_logged_out(self):
                return False

            def dump_debug(self, label):
                return None

            def close(self):
                pass

        monkeypatch.setattr("tools._browser.session.BrowserSession", Session)
        result = tool.execute({"prompt": "ambient"})
        assert result.success is False
        assert "quota issue" in result.error
        assert "not succeed on retry" in result.error
        assert "pixabay_music" in result.error


# ------------------------------------------------------------------
# Instrumental toggle
# ------------------------------------------------------------------

class TestInstrumentalToggle:

    def test_clicks_when_switch_is_off_and_instrumental_wanted(self):
        toggle = FakeLocator(visible=True, attrs={"aria-checked": "false"})
        session = FakeSession({}, {"instrumental_toggle": toggle})
        tool = SunoWebMusic()
        tool._set_instrumental(session, True)
        assert toggle.clicked == 1
        assert tool._instrumental_toggle_found is True

    def test_leaves_switch_alone_when_already_on(self):
        toggle = FakeLocator(visible=True, attrs={"aria-checked": "true"})
        session = FakeSession({}, {"instrumental_toggle": toggle})
        tool = SunoWebMusic()
        tool._set_instrumental(session, True)
        assert toggle.clicked == 0

    def test_turns_off_when_vocals_wanted(self):
        toggle = FakeLocator(visible=True, attrs={"aria-checked": "true"})
        session = FakeSession({}, {"instrumental_toggle": toggle})
        SunoWebMusic()._set_instrumental(session, False)
        assert toggle.clicked == 1

    def test_missing_switch_is_not_fatal_but_is_recorded(self):
        """Prompt wording still asks for instrumental — but flag the doubt."""
        session = FakeSession({}, {})
        tool = SunoWebMusic()
        tool._set_instrumental(session, True)
        assert tool._instrumental_toggle_found is False

    def test_flag_defaults_false_before_any_call(self):
        assert SunoWebMusic()._instrumental_toggle_found is False


# ------------------------------------------------------------------
# Prompt composition
# ------------------------------------------------------------------

class TestPromptComposition:

    def test_simple_mode_appends_instrumental_wording(self):
        field = FakeLocator(visible=True)
        session = FakeSession({}, {"prompt_input": field})
        SunoWebMusic()._fill_prompt(
            session, {"prompt": "dark cinematic underscore"},
            mode="simple", instrumental=True,
        )
        assert "instrumental" in field.filled.lower()
        assert "no vocals" in field.filled.lower()

    def test_simple_mode_does_not_duplicate_instrumental(self):
        field = FakeLocator(visible=True)
        session = FakeSession({}, {"prompt_input": field})
        SunoWebMusic()._fill_prompt(
            session, {"prompt": "dark underscore, instrumental"},
            mode="simple", instrumental=True,
        )
        assert field.filled.lower().count("instrumental") == 1

    def test_simple_mode_vocals_keeps_prompt_verbatim(self):
        field = FakeLocator(visible=True)
        session = FakeSession({}, {"prompt_input": field})
        SunoWebMusic()._fill_prompt(
            session, {"prompt": "indie pop"}, mode="simple", instrumental=False,
        )
        assert field.filled == "indie pop"

    def test_custom_mode_fills_style_and_title(self):
        style, title = FakeLocator(visible=True), FakeLocator(visible=True)
        session = FakeSession({}, {"style_input": style, "title_input": title})
        SunoWebMusic()._fill_prompt(
            session, {"prompt": "ambient drone", "title": "Debt"},
            mode="custom", instrumental=True,
        )
        assert style.filled == "ambient drone"
        assert title.filled == "Debt"

    def test_custom_mode_skips_lyrics_when_instrumental(self):
        style, lyrics = FakeLocator(visible=True), FakeLocator(visible=True)
        session = FakeSession({}, {"style_input": style, "lyrics_input": lyrics})
        SunoWebMusic()._fill_prompt(
            session, {"prompt": "ambient", "lyrics": "la la la"},
            mode="custom", instrumental=True,
        )
        assert lyrics.filled is None

    def test_custom_mode_writes_lyrics_when_vocals_wanted(self):
        style, lyrics = FakeLocator(visible=True), FakeLocator(visible=True)
        session = FakeSession({}, {"style_input": style, "lyrics_input": lyrics})
        SunoWebMusic()._fill_prompt(
            session, {"prompt": "ambient", "lyrics": "la la la"},
            mode="custom", instrumental=False,
        )
        assert lyrics.filled == "la la la"

    def test_missing_optional_field_is_skipped_silently(self):
        style = FakeLocator(visible=True)
        session = FakeSession({}, {"style_input": style})
        SunoWebMusic()._fill_prompt(
            session, {"prompt": "ambient", "title": "T"},
            mode="custom", instrumental=True,
        )
        assert style.filled == "ambient"

    def test_missing_required_field_raises(self):
        session = FakeSession({}, {})
        with pytest.raises(LookupError):
            SunoWebMusic()._fill_prompt(
                session, {"prompt": "ambient"}, mode="simple", instrumental=True,
            )


# ------------------------------------------------------------------
# Output writing
# ------------------------------------------------------------------

class TestOutputWriting:

    def _item(self, url=CLIP_A):
        return {
            "url": url, "bytes": MP3_BYTES,
            "clip_id": "11111111-2222-3333-4444-555555555555", "format": "mp3",
        }

    def test_single_track_uses_requested_path(self, tmp_path):
        out = tmp_path / "bgm.mp3"
        written = SunoWebMusic()._write_tracks({"output_path": str(out)}, [self._item()])
        assert len(written) == 1
        assert Path(written[0]["path"]).is_file()
        assert Path(written[0]["path"]).name == "bgm.mp3"

    def test_extension_normalized_to_served_format(self, tmp_path):
        out = tmp_path / "bgm.mp3"
        item = self._item()
        item["format"] = "wav"
        written = SunoWebMusic()._write_tracks({"output_path": str(out)}, [item])
        assert Path(written[0]["path"]).suffix == ".wav"

    def test_two_candidates_get_numbered_names(self, tmp_path):
        out = tmp_path / "bgm.mp3"
        written = SunoWebMusic()._write_tracks(
            {"output_path": str(out)}, [self._item(CLIP_A), self._item(CLIP_B)]
        )
        names = sorted(Path(w["path"]).name for w in written)
        assert names == ["bgm.1.mp3", "bgm.2.mp3"]
        assert all(Path(w["path"]).is_file() for w in written)

    def test_creates_missing_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "deeper" / "bgm.mp3"
        written = SunoWebMusic()._write_tracks({"output_path": str(out)}, [self._item()])
        assert Path(written[0]["path"]).is_file()

    def test_bytes_written_verbatim(self, tmp_path):
        out = tmp_path / "bgm.mp3"
        written = SunoWebMusic()._write_tracks({"output_path": str(out)}, [self._item()])
        assert Path(written[0]["path"]).read_bytes() == MP3_BYTES

    def test_probe_reports_file_size_without_ffprobe(self, tmp_path):
        out = tmp_path / "bgm.mp3"
        written = SunoWebMusic()._write_tracks({"output_path": str(out)}, [self._item()])
        assert written[0]["probe"]["file_size_bytes"] == len(MP3_BYTES)


# ------------------------------------------------------------------
# Download filtering
# ------------------------------------------------------------------

class FakeAPIResponse:
    def __init__(self, body=b"", ok=True):
        self._body = body
        self.ok = ok

    def body(self):
        return self._body


class FakeRequest:
    def __init__(self, mapping):
        self._mapping = mapping
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url not in self._mapping:
            raise RuntimeError("network error")
        return self._mapping[url]


class FakePage:
    def __init__(self, mapping):
        self.request = FakeRequest(mapping)


class TestFetchAudio:

    def test_keeps_track_above_size_floor(self):
        page = FakePage({CLIP_A: FakeAPIResponse(MP3_BYTES)})
        got = SunoWebMusic()._fetch_audio(page, [CLIP_A])
        assert len(got) == 1
        assert got[0]["format"] == "mp3"
        assert got[0]["clip_id"] == "11111111-2222-3333-4444-555555555555"

    def test_drops_track_below_size_floor(self):
        """A few-KB response is a UI blip or a placeholder, not a song."""
        page = FakePage({CLIP_A: FakeAPIResponse(b"ID3" + b"\x00" * 100)})
        assert SunoWebMusic()._fetch_audio(page, [CLIP_A]) == []

    def test_skips_failed_response(self):
        page = FakePage({CLIP_A: FakeAPIResponse(MP3_BYTES, ok=False)})
        assert SunoWebMusic()._fetch_audio(page, [CLIP_A]) == []

    def test_network_error_on_one_candidate_does_not_kill_the_other(self):
        page = FakePage({CLIP_B: FakeAPIResponse(MP3_BYTES)})
        got = SunoWebMusic()._fetch_audio(page, [CLIP_A, CLIP_B])
        assert [g["url"] for g in got] == [CLIP_B]

    def test_empty_candidate_list(self):
        assert SunoWebMusic()._fetch_audio(FakePage({}), []) == []


# ------------------------------------------------------------------
# Selector table
# ------------------------------------------------------------------

class TestSelectors:

    def test_suno_selectors_exist(self):
        from tools._browser.selectors import selectors_for
        table = selectors_for("suno")
        for key in (
            "prompt_input", "create_button", "instrumental_toggle",
            "logged_out", "quota_exhausted", "audio_element",
        ):
            assert table.get(key), f"missing selector group: {key}"

    def test_every_group_is_a_nonempty_list_of_strings(self):
        from tools._browser.selectors import selectors_for
        for key, value in selectors_for("suno").items():
            assert isinstance(value, list) and value, key
            assert all(isinstance(item, str) and item for item in value), key

    def test_user_override_replaces_a_group(self, tmp_path, monkeypatch):
        override = tmp_path / "selectors.json"
        override.write_text(
            json.dumps({"suno": {"create_button": ["#my-button"]}}), encoding="utf-8"
        )
        monkeypatch.setenv("OPENMONTAGE_BROWSER_SELECTORS", str(override))
        from tools._browser.selectors import selectors_for
        assert selectors_for("suno")["create_button"] == ["#my-button"]

    def test_quota_selectors_cannot_match_a_healthy_balance(self):
        """Regression: ':text()' is substring matching.

        Suno shows the balance as standing chrome. A quota candidate matching
        it would raise the tool's terminal "do not retry" error on a FULL
        account — the single worst false positive in this tool.
        """
        import re
        from tools._browser.selectors import selectors_for

        healthy_chrome = [
            "2,340 credits remaining",
            "13133 credits remaining",
            "Credits: 500",
            "You have 50 credits",
            "500 credits left this month",
        ]
        literals = []
        for selector in selectors_for("suno")["quota_exhausted"]:
            literals += re.findall(r":text\('([^']+)'\)", selector)
        assert literals, "quota_exhausted has no :text() literals to check"
        for literal in literals:
            for chrome in healthy_chrome:
                assert literal.lower() not in chrome.lower(), (
                    f"quota selector :text('{literal}') matches healthy balance "
                    f"chrome {chrome!r} — it would report a full account as out "
                    f"of credits, terminally"
                )

    def test_quota_selectors_still_catch_real_exhaustion(self):
        """The tightening must not have removed genuine detection."""
        import re
        from tools._browser.selectors import selectors_for

        literals = []
        for selector in selectors_for("suno")["quota_exhausted"]:
            literals += re.findall(r":text\('([^']+)'\)", selector)
        banners = [
            "You are out of credits",
            "Not enough credits to generate",
            "Insufficient credits for this action",
        ]
        for banner in banners:
            assert any(lit.lower() in banner.lower() for lit in literals), (
                f"no quota selector matches a real exhaustion banner: {banner!r}"
            )

    def test_prompt_input_has_no_catch_all(self):
        """A bare 'textarea' would let the login CLI confirm a fake login.

        wait_for_signed_in() treats a prompt_input match as proof of a signed-in
        session, and Suno's logged-out landing page has a describe-your-song box
        as a signup funnel.
        """
        from tools._browser.selectors import selectors_for
        assert "textarea" not in selectors_for("suno")["prompt_input"]
        assert "div[contenteditable='true']" not in selectors_for("suno")["prompt_input"]

    def test_logged_out_covers_log_in_wording(self):
        """Suno's auth is a Clerk modal that says 'Log in', often with no href."""
        from tools._browser.selectors import selectors_for
        joined = " ".join(selectors_for("suno")["logged_out"]).lower()
        assert "log in" in joined
        assert "sign in" in joined

    def test_logged_out_selectors_are_element_scoped(self):
        """Bare ':text()' here would strand a signed-in session on prose."""
        from tools._browser.selectors import selectors_for
        for selector in selectors_for("suno")["logged_out"]:
            assert selector.startswith(("a", "button")), (
                f"logged_out selector {selector!r} is not scoped to a link or "
                f"button — page prose mentioning 'sign up' would falsely read "
                f"as logged out"
            )

    def test_login_url_registered(self):
        from tools._browser.session import PROVIDER_URLS
        assert PROVIDER_URLS["suno"] == "https://suno.com/create"

    def test_login_cli_accepts_suno(self):
        """`login suno` must be a valid choice, not just a URL entry."""
        from tools._browser.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["login", "suno", "--help"])
        assert exc.value.code == 0


# ------------------------------------------------------------------
# Registry discovery
# ------------------------------------------------------------------

class TestRegistryDiscovery:

    def test_discoverable(self):
        from tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        registry.discover()
        assert "suno_web_music" in {t.name for t in registry._tools.values()}

    def test_registered_under_music_generation(self):
        from tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        registry.discover()
        names = {t.name for t in registry.get_by_capability("music_generation")}
        assert "suno_web_music" in names

    def test_distinct_from_the_api_key_suno_tool(self):
        """suno_music (API key) and suno_web_music (subscription) must coexist."""
        from tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        registry.discover()
        by_name = {t.name: t for t in registry._tools.values()}
        assert by_name["suno_music"].provider == "suno"
        assert by_name["suno_web_music"].provider == "suno_web"
        assert by_name["suno_web_music"].runtime == ToolRuntime.BROWSER


# ------------------------------------------------------------------
# Idempotency keys
# ------------------------------------------------------------------

class TestIdempotencyKeys:

    def test_includes_output_affecting_fields(self):
        fields = SunoWebMusic().idempotency_key_fields
        for field in ("prompt", "instrumental", "mode"):
            assert field in fields

    def test_excludes_execution_only_fields(self):
        fields = SunoWebMusic().idempotency_key_fields
        for field in ("output_path", "timeout_seconds", "headless", "download_all"):
            assert field not in fields

    def test_differs_on_instrumental(self):
        tool = SunoWebMusic()
        base = {"prompt": "x"}
        assert tool.idempotency_key({**base, "instrumental": True}) != tool.idempotency_key(
            {**base, "instrumental": False}
        )

    def test_differs_on_prompt(self):
        tool = SunoWebMusic()
        assert tool.idempotency_key({"prompt": "a"}) != tool.idempotency_key({"prompt": "b"})


# ------------------------------------------------------------------
# Schema validation
# ------------------------------------------------------------------

class TestSchemaValidation:

    def test_prompt_max_length_enforced(self):
        import jsonschema
        schema = SunoWebMusic().input_schema
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"prompt": "x" * 1001}, schema)

    def test_mode_enum_enforced(self):
        import jsonschema
        schema = SunoWebMusic().input_schema
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"prompt": "x", "mode": "expert"}, schema)

    def test_valid_inputs_accepted(self):
        import jsonschema
        schema = SunoWebMusic().input_schema
        jsonschema.validate(
            {
                "prompt": "cinematic underscore, instrumental, 80 BPM",
                "instrumental": True,
                "mode": "simple",
                "output_path": "projects/x/assets/music/bgm.mp3",
                "download_all": True,
                "timeout_seconds": 420,
            },
            schema,
        )

    def test_timeout_minimum_enforced(self):
        import jsonschema
        schema = SunoWebMusic().input_schema
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"prompt": "x", "timeout_seconds": 10}, schema)
