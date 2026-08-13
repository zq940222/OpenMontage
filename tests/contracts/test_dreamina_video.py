"""Contract tests for the Jimeng (即梦) membership video provider tool.

These verify the BaseTool contract, input normalization, and the
validate-before-you-spend rules without running the `dreamina` CLI or
consuming any membership credits.

Run: pytest tests/contracts/test_dreamina_video.py -v
"""

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
from tools.video.dreamina_video import DreaminaVideo


@pytest.fixture
def png(tmp_path):
    """A non-empty file standing in for a reference image."""
    def _make(name="ref.png"):
        path = tmp_path / name
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        return str(path)
    return _make


# ------------------------------------------------------------------
# Contract compliance
# ------------------------------------------------------------------

class TestContract:

    def test_inherits_base_tool(self):
        assert issubclass(DreaminaVideo, BaseTool)

    def test_has_required_identity(self):
        tool = DreaminaVideo()
        assert tool.name == "dreamina_video"
        assert tool.version
        assert tool.provider == "dreamina"
        assert tool.capability == "video_generation"
        assert tool.tier == ToolTier.GENERATE
        assert tool.runtime == ToolRuntime.BROWSER
        assert tool.stability == ToolStability.EXPERIMENTAL

    def test_execution_mode_is_async(self):
        assert DreaminaVideo().execution_mode == ExecutionMode.ASYNC

    def test_determinism_is_stochastic(self):
        assert DreaminaVideo().determinism == Determinism.STOCHASTIC

    def test_has_input_schema(self):
        schema = DreaminaVideo().input_schema
        assert schema.get("type") == "object"
        assert "prompt" in schema["properties"]
        assert "operation" in schema["properties"]

    def test_declares_all_five_modes(self):
        tool = DreaminaVideo()
        for mode in (
            "text_to_video", "image_to_video", "frames_to_video",
            "multiframe_to_video", "multimodal_to_video",
        ):
            assert mode in tool.capabilities
            assert mode in tool.input_schema["properties"]["operation"]["enum"]

    def test_does_not_claim_seed_support(self):
        """The CLI exposes no seed — claiming otherwise misleads the scorer."""
        assert DreaminaVideo().supports["seed"] is False

    def test_declares_reference_to_video_for_selector(self):
        """video_selector filters reference briefs on this supports key."""
        assert DreaminaVideo().supports["reference_to_video"] is True

    def test_has_agent_skills(self):
        skills = DreaminaVideo().agent_skills
        assert "dreamina-cli" in skills
        assert "seedance-2-0" in skills

    def test_has_fallbacks(self):
        assert "jimeng_video" in DreaminaVideo().fallback_tools

    def test_install_instructions_mention_login(self):
        text = DreaminaVideo().install_instructions
        assert "dreamina login" in text
        assert "user_credit" in text

    def test_get_info_returns_dict(self):
        info = DreaminaVideo().get_info()
        assert info["name"] == "dreamina_video"
        assert info["runtime"] == "browser"

    def test_cost_is_zero_usd_because_billing_is_credits(self):
        assert DreaminaVideo().estimate_cost({"prompt": "x", "duration": 10}) == 0.0

    def test_estimate_runtime_scales_with_duration(self):
        tool = DreaminaVideo()
        assert tool.estimate_runtime({"duration": 15}) > tool.estimate_runtime({"duration": 4})

    def test_retry_policy_does_not_auto_regenerate(self):
        """Auto-retry would burn membership credits without user consent."""
        assert DreaminaVideo().retry_policy.max_retries == 0

    def test_side_effects_mention_credits(self):
        assert any("credit" in s for s in DreaminaVideo().side_effects)

    def test_has_user_visible_verification(self):
        assert len(DreaminaVideo().user_visible_verification) > 0

    def test_dry_run_returns_dict(self):
        result = DreaminaVideo().dry_run({"prompt": "test"})
        assert result["tool"] == "dreamina_video"


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------

class TestStatus:

    def test_unavailable_without_cli(self, monkeypatch):
        monkeypatch.setattr(DreaminaVideo, "_cli_path", staticmethod(lambda: None))
        assert DreaminaVideo().get_status() == ToolStatus.UNAVAILABLE

    def test_degraded_when_cli_present_but_never_logged_in(self, monkeypatch, tmp_path):
        monkeypatch.setattr(DreaminaVideo, "_cli_path", staticmethod(lambda: "dreamina"))
        monkeypatch.setattr(
            DreaminaVideo, "_state_dir", staticmethod(lambda: tmp_path / "absent")
        )
        assert DreaminaVideo().get_status() == ToolStatus.DEGRADED

    def test_available_when_cli_and_login_state_present(self, monkeypatch, tmp_path):
        state = tmp_path / ".dreamina_cli"
        state.mkdir()
        monkeypatch.setattr(DreaminaVideo, "_cli_path", staticmethod(lambda: "dreamina"))
        monkeypatch.setattr(DreaminaVideo, "_state_dir", staticmethod(lambda: state))
        assert DreaminaVideo().get_status() == ToolStatus.AVAILABLE

    def test_setup_offer_present_when_not_ready(self, monkeypatch):
        monkeypatch.setattr(DreaminaVideo, "_cli_path", staticmethod(lambda: None))
        offer = DreaminaVideo().get_info()["setup_offer"]
        assert offer["command"] == "dreamina login"
        assert offer["kind"] == "one_time_login"

    def test_no_setup_offer_when_ready(self, monkeypatch, tmp_path):
        state = tmp_path / ".dreamina_cli"
        state.mkdir()
        monkeypatch.setattr(DreaminaVideo, "_cli_path", staticmethod(lambda: "dreamina"))
        monkeypatch.setattr(DreaminaVideo, "_state_dir", staticmethod(lambda: state))
        assert "setup_offer" not in DreaminaVideo().get_info()

    def test_execute_without_cli_names_the_fix(self, monkeypatch):
        monkeypatch.setattr(DreaminaVideo, "_cli_path", staticmethod(lambda: None))
        result = DreaminaVideo().execute({"prompt": "test"})
        assert result.success is False
        assert "dreamina login" in result.error


# ------------------------------------------------------------------
# Input normalization (video_selector forwards its own vocabulary)
# ------------------------------------------------------------------

class TestNormalization:

    def test_reference_to_video_maps_to_multimodal(self):
        out = DreaminaVideo._normalize_inputs({"operation": "reference_to_video"})
        assert out["operation"] == "multimodal_to_video"

    def test_reference_image_path_maps_to_image_path(self):
        out = DreaminaVideo._normalize_inputs({
            "operation": "image_to_video",
            "reference_image_path": "a.png",
        })
        assert out["image_path"] == "a.png"

    def test_reference_image_paths_map_to_image_paths_for_multimodal(self):
        out = DreaminaVideo._normalize_inputs({
            "operation": "multimodal_to_video",
            "reference_image_paths": ["a.png", "b.png"],
        })
        assert out["image_paths"] == ["a.png", "b.png"]

    def test_reference_paths_split_into_first_and_last_frame(self):
        out = DreaminaVideo._normalize_inputs({
            "operation": "frames_to_video",
            "reference_image_paths": ["a.png", "b.png"],
        })
        assert out["first_frame_path"] == "a.png"
        assert out["last_frame_path"] == "b.png"

    def test_string_duration_is_coerced(self):
        assert DreaminaVideo._normalize_inputs({"duration": "10"})["duration"] == 10

    def test_unparseable_duration_falls_back_to_default(self):
        assert "duration" not in DreaminaVideo._normalize_inputs({"duration": "auto"})

    def test_resolution_alias(self):
        out = DreaminaVideo._normalize_inputs({"resolution": "1080p"})
        assert out["video_resolution"] == "1080p"

    def test_model_name_alias(self):
        out = DreaminaVideo._normalize_inputs({"model_name": "seedance2.0"})
        assert out["model_version"] == "seedance2.0"

    def test_explicit_keys_win_over_aliases(self):
        out = DreaminaVideo._normalize_inputs({
            "operation": "image_to_video",
            "image_path": "explicit.png",
            "reference_image_path": "alias.png",
            "video_resolution": "720p",
            "resolution": "1080p",
        })
        assert out["image_path"] == "explicit.png"
        assert out["video_resolution"] == "720p"

    def test_normalization_does_not_mutate_caller_input(self):
        original = {"operation": "reference_to_video", "duration": "8"}
        DreaminaVideo._normalize_inputs(original)
        assert original == {"operation": "reference_to_video", "duration": "8"}


# ------------------------------------------------------------------
# Validation — reject before spending credits
# ------------------------------------------------------------------

class TestValidation:

    def _validate(self, inputs):
        tool = DreaminaVideo()
        return tool._validate(tool._normalize_inputs(inputs))

    def test_valid_text_to_video_passes(self):
        assert self._validate({"prompt": "a cat", "duration": 5}) is None

    def test_unknown_operation_rejected(self):
        assert "Unknown operation" in self._validate({"prompt": "x", "operation": "nope"})

    def test_empty_prompt_rejected(self):
        assert "prompt" in self._validate({"prompt": "   "})

    def test_duration_above_model_range_rejected(self):
        error = self._validate({"prompt": "x", "duration": 20})
        assert "out of range" in error

    def test_duration_below_model_range_rejected(self):
        error = self._validate({"prompt": "x", "duration": 3, "model_version": "seedance2.0"})
        assert "out of range" in error

    def test_3_0_model_allows_3_seconds(self, png):
        assert self._validate({
            "prompt": "x", "operation": "image_to_video",
            "image_path": png(), "duration": 3, "model_version": "3.0",
        }) is None

    def test_model_not_valid_for_operation_rejected(self, png):
        error = self._validate({
            "prompt": "x", "operation": "text_to_video", "model_version": "3.0",
        })
        assert "not supported by text_to_video" in error

    def test_1080p_rejected_on_seedance(self, png):
        error = self._validate({
            "prompt": "x", "operation": "image_to_video",
            "image_path": png(), "video_resolution": "1080p",
        })
        assert "not supported" in error

    def test_1080p_allowed_on_3_5pro(self, png):
        assert self._validate({
            "prompt": "x", "operation": "image_to_video", "image_path": png(),
            "video_resolution": "1080p", "model_version": "3.5pro", "duration": 6,
        }) is None

    def test_invalid_aspect_ratio_rejected(self):
        assert "aspect_ratio" in self._validate({"prompt": "x", "aspect_ratio": "5:7"})

    def test_image_to_video_without_image_rejected(self):
        assert "image_path" in self._validate({"prompt": "x", "operation": "image_to_video"})

    def test_url_only_reference_explains_local_file_requirement(self):
        error = self._validate({
            "prompt": "x", "operation": "image_to_video",
            "image_url": "https://example.com/a.png",
        })
        assert "local files" in error

    def test_missing_reference_file_rejected(self, tmp_path):
        error = self._validate({
            "prompt": "x", "operation": "image_to_video",
            "image_path": str(tmp_path / "nope.png"),
        })
        assert "missing" in error

    def test_empty_reference_file_rejected(self, tmp_path):
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        error = self._validate({
            "prompt": "x", "operation": "image_to_video", "image_path": str(empty),
        })
        assert "missing or empty" in error

    def test_frames_to_video_requires_both_frames(self, png):
        error = self._validate({
            "prompt": "x", "operation": "frames_to_video", "first_frame_path": png(),
        })
        assert "last_frame_path" in error

    def test_multiframe_requires_at_least_two_images(self, png):
        error = self._validate({
            "prompt": "x", "operation": "multiframe_to_video", "image_paths": [png()],
        })
        assert "2-20" in error

    def test_multiframe_rejects_model_override(self, png):
        error = self._validate({
            "prompt": "x", "operation": "multiframe_to_video",
            "image_paths": [png("a.png"), png("b.png")],
            "model_version": "seedance2.0",
        })
        assert "does not accept model_version" in error

    def test_multiframe_transition_prompt_count_enforced(self, png):
        error = self._validate({
            "operation": "multiframe_to_video",
            "image_paths": [png("a.png"), png("b.png"), png("c.png")],
            "transition_prompts": ["a to b"],
        })
        assert "2 transition_prompts" in error

    def test_multiframe_segment_duration_bounds_enforced(self, png):
        error = self._validate({
            "prompt": "x", "operation": "multiframe_to_video",
            "image_paths": [png("a.png"), png("b.png")],
            "transition_durations": [12],
        })
        assert "0.5-8" in error

    def test_multiframe_total_duration_floor_enforced(self, png):
        error = self._validate({
            "prompt": "x", "operation": "multiframe_to_video",
            "image_paths": [png("a.png"), png("b.png")],
            "transition_durations": [1],
        })
        assert ">= 2 seconds" in error

    def test_multimodal_requires_a_reference(self):
        error = self._validate({"prompt": "x", "operation": "multimodal_to_video"})
        assert "at least one" in error

    def test_multimodal_image_limit_enforced(self, png):
        error = self._validate({
            "prompt": "x", "operation": "multimodal_to_video",
            "image_paths": [png(f"{i}.png") for i in range(10)],
        })
        assert "at most 9" in error

    def test_multimodal_allows_empty_prompt(self, png):
        assert self._validate({
            "operation": "multimodal_to_video", "image_paths": [png()],
        }) is None


# ------------------------------------------------------------------
# CLI command construction
# ------------------------------------------------------------------

class TestCommandBuilding:

    def _cmd(self, inputs):
        tool = DreaminaVideo()
        return tool._build_command(tool._normalize_inputs(inputs))

    def test_text_to_video_command(self):
        cmd = self._cmd({"prompt": "a cat", "duration": 6, "aspect_ratio": "9:16"})
        assert cmd[:2] == ["dreamina", "text2video"]
        assert "--prompt=a cat" in cmd
        assert "--duration=6" in cmd
        assert "--ratio=9:16" in cmd
        assert "--model_version=seedance2.0fast" in cmd

    def test_submit_never_blocks_on_cli_polling(self):
        """We poll ourselves so progress and timeouts stay under our control."""
        assert "--poll=0" in self._cmd({"prompt": "x"})

    def test_image_to_video_omits_ratio(self):
        """The CLI infers ratio from the image; passing --ratio is an error."""
        cmd = self._cmd({
            "prompt": "push in", "operation": "image_to_video", "image_path": "a.png",
        })
        assert cmd[1] == "image2video"
        assert not any(c.startswith("--ratio") for c in cmd)

    def test_resolution_omitted_when_not_requested(self):
        cmd = self._cmd({"prompt": "x"})
        assert not any(c.startswith("--video_resolution") for c in cmd)

    def test_resolution_included_when_requested(self):
        cmd = self._cmd({"prompt": "x", "video_resolution": "720p"})
        assert "--video_resolution=720p" in cmd

    def test_frames_to_video_command(self):
        cmd = self._cmd({
            "prompt": "seasons change", "operation": "frames_to_video",
            "first_frame_path": "a.png", "last_frame_path": "b.png",
        })
        assert cmd[1] == "frames2video"
        assert "--first=a.png" in cmd
        assert "--last=b.png" in cmd

    def test_multiframe_two_images_uses_shorthand(self):
        cmd = self._cmd({
            "prompt": "turn around", "operation": "multiframe_to_video",
            "image_paths": ["a.png", "b.png"], "duration": 4,
        })
        assert cmd[1] == "multiframe2video"
        assert "--images=a.png,b.png" in cmd
        assert "--prompt=turn around" in cmd
        assert "--duration=4.0" in cmd

    def test_multiframe_three_images_uses_transition_flags(self):
        cmd = self._cmd({
            "operation": "multiframe_to_video",
            "image_paths": ["a.png", "b.png", "c.png"],
            "transition_prompts": ["a to b", "b to c"],
            "transition_durations": [3, 4],
        })
        assert cmd.count("--transition-prompt=a to b") == 1
        assert "--transition-prompt=b to c" in cmd
        assert "--transition-duration=3.0" in cmd
        assert "--transition-duration=4.0" in cmd
        assert not any(c.startswith("--model_version") for c in cmd)

    def test_multimodal_repeats_reference_flags(self):
        cmd = self._cmd({
            "prompt": "@Image1 walks", "operation": "multimodal_to_video",
            "image_paths": ["a.png", "b.png"],
            "video_paths": ["ref.mp4"],
            "audio_paths": ["bgm.mp3"],
            "duration": 8,
        })
        assert cmd[1] == "multimodal2video"
        assert cmd.count("--image=a.png") == 1
        assert "--image=b.png" in cmd
        assert "--video=ref.mp4" in cmd
        assert "--audio=bgm.mp3" in cmd
        assert "--ratio=16:9" in cmd

    def test_reference_to_video_builds_multimodal_command(self):
        cmd = self._cmd({
            "prompt": "x", "operation": "reference_to_video",
            "reference_image_paths": ["a.png"],
        })
        assert cmd[1] == "multimodal2video"

    def test_paths_are_single_argv_elements(self):
        """Spaces and CJK in paths must not need shell quoting."""
        cmd = self._cmd({
            "prompt": "x", "operation": "image_to_video",
            "image_path": "03-design/角色 设定/林晚 front.png",
        })
        assert "--image=03-design/角色 设定/林晚 front.png" in cmd


# ------------------------------------------------------------------
# Output parsing
# ------------------------------------------------------------------

class TestParsing:

    def test_parse_json_reads_plain_object(self):
        assert DreaminaVideo._parse_json('{"submit_id": "abc"}')["submit_id"] == "abc"

    def test_parse_json_tolerates_surrounding_log_lines(self):
        text = 'INFO uploading\n{"submit_id": "abc"}\nDONE\n'
        assert DreaminaVideo._parse_json(text)["submit_id"] == "abc"

    def test_parse_json_returns_none_on_garbage(self):
        assert DreaminaVideo._parse_json("no json here") is None

    def test_parse_json_returns_none_on_empty(self):
        assert DreaminaVideo._parse_json("") is None

    def test_first_video_meta_extracts_dimensions(self):
        payload = {"result_json": {"videos": [
            {"fps": 24, "width": 1280, "height": 720, "duration": 6.042, "path": "x.mp4"},
        ]}}
        meta = DreaminaVideo._first_video_meta(payload)
        assert meta == {"fps": 24, "width": 1280, "height": 720, "duration": 6.042}

    def test_first_video_meta_handles_no_videos(self):
        assert DreaminaVideo._first_video_meta({"result_json": {"videos": []}}) == {}

    def test_compliance_message_is_actionable_and_says_do_not_retry(self):
        message = DreaminaVideo._compliance_message("AigcComplianceConfirmationRequired")
        assert "web UI" in message
        assert "Do NOT auto-retry" in message


# ------------------------------------------------------------------
# Idempotency
# ------------------------------------------------------------------

class TestIdempotencyKeys:

    def test_includes_output_affecting_fields(self):
        fields = DreaminaVideo().idempotency_key_fields
        for field in (
            "operation", "prompt", "image_path", "image_paths",
            "duration", "aspect_ratio", "model_version", "video_resolution",
        ):
            assert field in fields

    def test_excludes_execution_only_fields(self):
        fields = DreaminaVideo().idempotency_key_fields
        for field in ("output_path", "poll_interval_seconds", "timeout_seconds"):
            assert field not in fields

    def test_differs_on_duration(self):
        tool = DreaminaVideo()
        assert tool.idempotency_key({"prompt": "x", "duration": 5}) != tool.idempotency_key(
            {"prompt": "x", "duration": 10}
        )

    def test_differs_on_operation(self):
        tool = DreaminaVideo()
        assert tool.idempotency_key({"prompt": "x"}) != tool.idempotency_key(
            {"prompt": "x", "operation": "multimodal_to_video"}
        )


# ------------------------------------------------------------------
# Registry discovery
# ------------------------------------------------------------------

class TestRegistryDiscovery:

    def test_discoverable(self):
        from tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        registry.discover()
        assert "dreamina_video" in {t.name for t in registry._tools.values()}

    def test_distinct_from_the_api_key_jimeng_tool(self):
        from tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        registry.discover()
        assert registry.get("dreamina_video").provider == "dreamina"
        assert registry.get("jimeng_video").provider == "volcengine"


# ------------------------------------------------------------------
# Schema validation
# ------------------------------------------------------------------

class TestSchemaValidation:

    def test_duration_bounds_declared(self):
        props = DreaminaVideo().input_schema["properties"]["duration"]
        assert props["minimum"] == 3
        assert props["maximum"] == 15

    def test_duration_out_of_bounds_rejected_by_schema(self):
        import jsonschema
        schema = DreaminaVideo().input_schema
        for invalid in (0, 2, 16, 60):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate({"prompt": "x", "duration": invalid}, schema)

    def test_valid_payload_passes_schema(self):
        import jsonschema
        jsonschema.validate(
            {
                "prompt": "x", "operation": "multimodal_to_video",
                "image_paths": ["a.png"], "duration": 15, "aspect_ratio": "9:16",
            },
            DreaminaVideo().input_schema,
        )
