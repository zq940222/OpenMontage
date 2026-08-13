"""Jimeng (即梦 / Dreamina) video generation via the official `dreamina` CLI.

Uses the user's Jimeng web membership (subscription credits) instead of an
API key. The `dreamina` CLI authenticates once via OAuth Device Flow
(`dreamina login`) and stores its session locally; every generation call
here consumes membership credits, not USD.

Flow per mode: build CLI command -> submit (--poll=0, returns submit_id)
-> poll `dreamina query_result --submit_id=<id>` -> download via
`query_result --download_dir=<dir>` (re-download is free) -> move to
output_path -> ffprobe verification.

Supported generation modes (mirroring the CLI's generator commands):
  text_to_video        text2video       (Seedance 2.0 family)
  image_to_video       image2video      (first-frame image; ratio inferred)
  frames_to_video      frames2video     (first+last frame; ratio inferred)
  multiframe_to_video  multiframe2video (2-20 keyframes; ratio inferred)
  multimodal_to_video  multimodal2video (全能参考: images/videos/audio refs)
"""

from __future__ import annotations

import json
import re
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
    ToolCommandError,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


_CLI = "dreamina"
# Login state directory created by `dreamina login` (OAuth Device Flow).
_STATE_DIR_NAME = ".dreamina_cli"

_SEEDANCE_MODELS = {
    "seedance2.0", "seedance2.0fast", "seedance2.0_vip", "seedance2.0fast_vip",
}
_RATIOS = ["1:1", "3:4", "16:9", "4:3", "9:16", "21:9"]

# duration ranges (seconds, inclusive) per model family — from `dreamina <cmd> -h`
_DURATION_RANGES: dict[str, tuple[int, int]] = {
    **{m: (4, 15) for m in _SEEDANCE_MODELS},
    "3.5pro": (4, 12), "3.5_pro": (4, 12),
    "3.0": (3, 10), "3.0fast": (3, 10), "3.0_fast": (3, 10),
    "3.0pro": (3, 10), "3.0_pro": (3, 10),
}

_MODELS_BY_OPERATION: dict[str, set[str]] = {
    "text_to_video": set(_SEEDANCE_MODELS),
    "image_to_video": set(_DURATION_RANGES),
    "frames_to_video": {"3.0", "3.5pro", "3.5_pro"} | _SEEDANCE_MODELS,
    "multimodal_to_video": set(_SEEDANCE_MODELS),
    # multiframe2video accepts no model/resolution overrides
    "multiframe_to_video": set(),
}

_SUBMIT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)

# fail_reason marker for the terminal "authorize this model on the web UI
# first" state. Retrying burns nothing but never succeeds — surface to user.
_COMPLIANCE_MARKER = "AigcComplianceConfirmationRequired"

_FAILED_STATUSES = {"failed", "fail", "error", "cancelled", "canceled", "expired"}


class DreaminaVideo(BaseTool):
    name = "dreamina_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "dreamina"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.ASYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.BROWSER

    dependencies = [f"cmd:{_CLI}"]
    install_instructions = (
        "Install the official Jimeng (即梦) `dreamina` CLI and log in once with "
        "your Jimeng membership account:\n"
        "  1. Put `dreamina` on PATH (即梦 official AIGC CLI).\n"
        "  2. Run `dreamina login` and complete the OAuth device login in your browser.\n"
        "  3. Verify with `dreamina user_credit` (shows your credit balance).\n"
        "No API key needed — generation consumes Jimeng membership credits."
    )
    agent_skills = ["dreamina-cli", "seedance-2-0", "ai-video-gen"]

    capabilities = [
        "text_to_video", "image_to_video", "frames_to_video",
        "multiframe_to_video", "multimodal_to_video",
    ]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "reference_to_video": True,   # video_selector's name for multimodal refs
        "first_last_frame": True,
        "multi_keyframe": True,
        "multimodal_references": True,
        "native_audio": True,   # Seedance 2.0 modes generate synced audio
        "seed": False,          # web/CLI pipeline exposes no seed control
        "subscription_billing": True,
    }
    best_for = [
        "Jimeng (即梦) membership video generation without an API key — Seedance 2.0 family",
        "cinematic multi-beat shots with synced audio (multimodal 全能参考 mode)",
        "character-consistent shots driven by reference images",
        "first/last-frame and multi-keyframe story continuity",
        "Chinese-language prompt understanding",
    ]
    not_good_for = [
        "offline generation",
        "users without a Jimeng account/membership",
        "seed-reproducible generation",
    ]
    fallback_tools = ["jimeng_video", "seedance_video", "kling_video"]

    input_schema = {
        "type": "object",
        "required": [],
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "text_to_video", "image_to_video", "frames_to_video",
                    "multiframe_to_video", "multimodal_to_video",
                    "reference_to_video",
                ],
                "default": "text_to_video",
                "description": (
                    "reference_to_video is video_selector's name for the "
                    "multimodal (全能参考) mode and is treated as such."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Generation prompt. Required for all modes except "
                    "multimodal_to_video (optional there). Supports long "
                    "multi-beat Seedance 2.0 prompts with timecodes and "
                    "@Image1..N reference tags in multimodal mode."
                ),
            },
            "image_path": {
                "type": "string",
                "description": "Local first-frame image for image_to_video.",
            },
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Local reference images. multiframe_to_video: 2-20 ordered "
                    "keyframes. multimodal_to_video: up to 9 references "
                    "(order maps to @Image1..N in the prompt)."
                ),
            },
            "video_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "multimodal_to_video only: up to 3 local reference videos.",
            },
            "audio_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "multimodal_to_video only: up to 3 local audio refs, each 2-15s.",
            },
            "first_frame_path": {
                "type": "string",
                "description": "frames_to_video: local first-frame image.",
            },
            "last_frame_path": {
                "type": "string",
                "description": "frames_to_video: local last-frame image.",
            },
            "transition_prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "multiframe_to_video with 3+ images: one prompt per "
                    "transition segment (N images -> N-1 prompts)."
                ),
            },
            "transition_durations": {
                "type": "array",
                "items": {"type": "number"},
                "description": (
                    "multiframe_to_video with 3+ images: seconds per segment "
                    "(N-1 values, each 0.5-8, total >= 2). Omit for 3s each."
                ),
            },
            "duration": {
                "type": "integer",
                "minimum": 3,
                "maximum": 15,
                "default": 5,
                "description": (
                    "Video duration in seconds. Allowed range depends on model: "
                    "seedance2.0 family 4-15, 3.5pro 4-12, 3.0 family 3-10."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": _RATIOS,
                "default": "16:9",
                "description": (
                    "Only used by text_to_video and multimodal_to_video. Other "
                    "modes infer the ratio from the input image(s)."
                ),
            },
            "model_version": {
                "type": "string",
                "default": "seedance2.0fast",
                "description": (
                    "seedance2.0 | seedance2.0fast | seedance2.0_vip | "
                    "seedance2.0fast_vip (+ 3.0/3.0fast/3.0pro/3.5pro for "
                    "image_to_video, 3.0/3.5pro for frames_to_video). "
                    "multiframe_to_video accepts no model override."
                ),
            },
            "video_resolution": {
                "type": "string",
                "enum": ["720p", "1080p"],
                "description": (
                    "Optional. seedance2.0 family supports 720p only; 1080p "
                    "needs a 3.x model on image/frames modes."
                ),
            },
            "output_path": {"type": "string"},
            "poll_interval_seconds": {
                "type": "number",
                "minimum": 5,
                "default": 30.0,
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 60,
                "default": 1800,
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=0,  # regeneration burns membership credits — agent decides
        backoff_seconds=5.0,
        retryable_errors=[],
    )
    idempotency_key_fields = [
        "operation", "prompt", "image_path", "image_paths", "video_paths",
        "audio_paths", "first_frame_path", "last_frame_path",
        "transition_prompts", "transition_durations", "duration",
        "aspect_ratio", "model_version", "video_resolution",
    ]
    side_effects = [
        "writes video file to output_path",
        "runs the `dreamina` CLI (submit + poll + download)",
        "consumes Jimeng membership credits per generation",
    ]
    user_visible_verification = [
        "Watch generated clip for motion coherence and prompt adherence",
        "Check data.credit_count to see the actual membership credits spent",
    ]

    # ---- Availability ----

    @staticmethod
    def _cli_path() -> Optional[str]:
        return shutil.which(_CLI)

    @staticmethod
    def _state_dir() -> Path:
        return Path.home() / _STATE_DIR_NAME

    def get_status(self) -> ToolStatus:
        if not self._cli_path():
            return ToolStatus.UNAVAILABLE
        # Login-state heuristic: `dreamina login` creates ~/.dreamina_cli.
        # A live-session check would need a network call; execute() verifies
        # the session for real (via `dreamina user_credit`) before submitting.
        if not self._state_dir().is_dir():
            return ToolStatus.DEGRADED
        return ToolStatus.AVAILABLE

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        if self.get_status() != ToolStatus.AVAILABLE:
            info["setup_offer"] = {
                "kind": "one_time_login",
                "fix_complexity": (
                    "1-minute login if `dreamina` is installed; otherwise install the CLI first"
                ),
                "command": "dreamina login",
                "health_check": "dreamina user_credit",
                "what_it_unlocks": [
                    "Seedance 2.0 video generation on your Jimeng (即梦) membership credits",
                    "text/image/first-last-frame/multi-keyframe/multimodal video modes",
                    "no API key required",
                ],
            }
        return info

    # ---- Cost ----

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Billing is Jimeng membership credits, not USD. Observed: a 6s
        # seedance2.0fast 720p clip cost 66 credits (~11 credits/sec).
        # Report 0.0 USD; the credit spend is surfaced in data.credit_count.
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 240.0 + int(inputs.get("duration", 5)) * 30.0

    # ---- Execution ----

    def execute(self, raw_inputs: dict[str, Any]) -> ToolResult:
        if not self._cli_path():
            return ToolResult(
                success=False,
                error=f"`{_CLI}` CLI not found on PATH. " + self.install_instructions,
            )

        inputs = self._normalize_inputs(raw_inputs)
        error = self._validate(inputs)
        if error:
            return ToolResult(success=False, error=error)

        start = time.time()

        # Verify the session is alive BEFORE submitting (a dead OAuth session
        # would otherwise surface as a confusing submit failure). Free call.
        credit_before, auth_error = self._check_login()
        if auth_error:
            return ToolResult(success=False, error=auth_error)

        try:
            cmd = self._build_command(inputs)
            submit_id = self._submit(cmd)
            payload = self._poll(
                submit_id,
                poll_interval=float(inputs.get("poll_interval_seconds", 30.0)),
                timeout_seconds=int(inputs.get("timeout_seconds", 1800)),
            )
            output_path = self._download(submit_id, inputs.get("output_path"))
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Dreamina video generation failed: {exc}",
            )

        from tools.video._shared import probe_output

        operation = inputs.get("operation", "text_to_video")
        result_meta = self._first_video_meta(payload)
        data: dict[str, Any] = {
            "provider": self.provider,
            "route": "dreamina_cli",
            "model": self._model_for(inputs),
            "operation": operation,
            "prompt": inputs.get("prompt", ""),
            "submit_id": submit_id,
            "output": str(output_path),
            "format": "mp4",
            "billing": "jimeng_membership_credits",
            "credit_count": payload.get("credit_count"),
            "credits_before_run": credit_before,
            **result_meta,
            **probe_output(output_path),
        }
        if operation in ("image_to_video", "frames_to_video", "multiframe_to_video"):
            data["ratio_inferred_from_image"] = True

        return ToolResult(
            success=True,
            data=data,
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model=self._model_for(inputs),
        )

    # ---- Input normalization ----

    @staticmethod
    def _normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        """Accept video_selector's vocabulary as well as this tool's own.

        The selector forwards its whole input dict verbatim, so a brief routed
        through it arrives with `reference_image_path`, `resolution`,
        `model_name`, a string `duration`, and operation `reference_to_video`.
        Normalizing here keeps both call styles working.
        """
        out = dict(inputs)

        if out.get("operation") == "reference_to_video":
            out["operation"] = "multimodal_to_video"
        operation = out.get("operation", "text_to_video")

        ref_paths = [
            str(p) for p in (out.get("reference_image_paths") or [])
            if p
        ]
        single_ref = out.get("reference_image_path")
        if single_ref and str(single_ref) not in ref_paths:
            ref_paths.insert(0, str(single_ref))

        if ref_paths:
            if operation == "image_to_video":
                out.setdefault("image_path", ref_paths[0])
            elif operation in ("multimodal_to_video", "multiframe_to_video"):
                if not out.get("image_paths"):
                    out["image_paths"] = ref_paths
            elif operation == "frames_to_video":
                out.setdefault("first_frame_path", ref_paths[0])
                if len(ref_paths) > 1:
                    out.setdefault("last_frame_path", ref_paths[1])

        ref_video = out.get("reference_video_path")
        if ref_video and not out.get("video_paths"):
            out["video_paths"] = [str(ref_video)]

        if out.get("resolution") and not out.get("video_resolution"):
            out["video_resolution"] = out["resolution"]
        if out.get("model_name") and not out.get("model_version"):
            out["model_version"] = out["model_name"]

        duration = out.get("duration")
        if isinstance(duration, str):
            digits = re.match(r"\s*(\d+)", duration)
            if digits:
                out["duration"] = int(digits.group(1))
            else:
                out.pop("duration")
        elif isinstance(duration, float):
            out["duration"] = int(round(duration))

        return out

    # ---- Validation (fail before burning credits) ----

    def _validate(self, inputs: dict[str, Any]) -> Optional[str]:
        operation = inputs.get("operation", "text_to_video")
        if operation not in _MODELS_BY_OPERATION:
            return (
                f"Unknown operation: {operation!r}. "
                f"Valid: {', '.join(sorted(_MODELS_BY_OPERATION))}"
            )

        prompt = (inputs.get("prompt") or "").strip()
        if operation == "multiframe_to_video":
            paths = inputs.get("image_paths") or []
            if not 2 <= len(paths) <= 20:
                return "multiframe_to_video requires 2-20 image_paths."
            missing = self._missing_files(paths)
            if missing:
                return f"multiframe_to_video keyframe(s) missing or empty: {missing}"
            n_transitions = len(paths) - 1
            t_prompts = inputs.get("transition_prompts") or []
            t_durations = inputs.get("transition_durations") or []
            if len(paths) == 2:
                if not prompt and not t_prompts:
                    return "multiframe_to_video with 2 images requires a prompt."
            else:
                if len(t_prompts) != n_transitions:
                    return (
                        f"multiframe_to_video with {len(paths)} images requires "
                        f"{n_transitions} transition_prompts (got {len(t_prompts)})."
                    )
            if t_durations:
                if len(paths) > 2 and len(t_durations) != n_transitions:
                    return (
                        f"transition_durations must have {n_transitions} entries "
                        f"(got {len(t_durations)})."
                    )
                if any(not 0.5 <= float(d) <= 8 for d in t_durations):
                    return "Each transition duration must be within 0.5-8 seconds."
                if sum(float(d) for d in t_durations) < 2:
                    return "Total multiframe duration must be >= 2 seconds."
            if inputs.get("model_version") or inputs.get("video_resolution"):
                return (
                    "multiframe_to_video does not accept model_version or "
                    "video_resolution overrides — omit them."
                )
            return None

        # All remaining modes need a prompt except multimodal (optional there).
        if not prompt and operation != "multimodal_to_video":
            return f"{operation} requires a non-empty prompt."

        model = self._model_for(inputs)
        allowed_models = _MODELS_BY_OPERATION[operation]
        if model not in allowed_models:
            return (
                f"model_version {model!r} is not supported by {operation}. "
                f"Valid: {', '.join(sorted(allowed_models))}"
            )

        lo, hi = _DURATION_RANGES[model]
        duration = int(inputs.get("duration", 5))
        if not lo <= duration <= hi:
            return f"duration {duration}s is out of range for {model}: {lo}-{hi}s."

        resolution = inputs.get("video_resolution")
        if resolution and resolution not in self._allowed_resolutions(model):
            return (
                f"video_resolution {resolution!r} is not supported by {model}. "
                f"Valid: {', '.join(sorted(self._allowed_resolutions(model)))}"
            )

        ratio = inputs.get("aspect_ratio")
        if ratio and ratio not in _RATIOS:
            return f"aspect_ratio {ratio!r} invalid. Valid: {', '.join(_RATIOS)}"

        if operation == "image_to_video":
            image = inputs.get("image_path")
            if not image:
                return (
                    "image_to_video requires image_path (local first-frame image)."
                    + self._url_only_hint(inputs)
                )
            missing = self._missing_files([image])
            if missing:
                return f"image_to_video first frame missing or empty: {missing}"

        elif operation == "frames_to_video":
            first = inputs.get("first_frame_path")
            last = inputs.get("last_frame_path")
            if not first or not last:
                return "frames_to_video requires first_frame_path and last_frame_path."
            missing = self._missing_files([first, last])
            if missing:
                return f"frames_to_video keyframe(s) missing or empty: {missing}"

        elif operation == "multimodal_to_video":
            images = inputs.get("image_paths") or []
            videos = inputs.get("video_paths") or []
            audios = inputs.get("audio_paths") or []
            if not images and not videos:
                return (
                    "multimodal_to_video requires at least one image_paths or "
                    "video_paths entry." + self._url_only_hint(inputs)
                )
            if len(images) > 9:
                return "multimodal_to_video accepts at most 9 image_paths."
            if len(videos) > 3:
                return "multimodal_to_video accepts at most 3 video_paths."
            if len(audios) > 3:
                return "multimodal_to_video accepts at most 3 audio_paths (each 2-15s)."
            missing = self._missing_files([*images, *videos, *audios])
            if missing:
                return f"multimodal_to_video reference file(s) missing or empty: {missing}"

        return None

    @staticmethod
    def _url_only_hint(inputs: dict[str, Any]) -> str:
        """Explain the local-file requirement when only URLs were supplied."""
        url_keys = [
            key for key in (
                "reference_image_url", "reference_image_urls",
                "image_url", "image_urls", "reference_video_url",
            )
            if inputs.get(key)
        ]
        if not url_keys:
            return ""
        return (
            f" You passed {', '.join(url_keys)}, but the dreamina CLI uploads "
            "local files — download the asset first and pass its path."
        )

    @staticmethod
    def _missing_files(paths: list[str]) -> str:
        bad = [
            str(p) for p in paths
            if not Path(p).is_file() or Path(p).stat().st_size == 0
        ]
        return ", ".join(bad)

    @staticmethod
    def _model_for(inputs: dict[str, Any]) -> str:
        if inputs.get("operation", "text_to_video") == "multiframe_to_video":
            return "multiframe2video"  # CLI picks the model internally
        return str(inputs.get("model_version") or "seedance2.0fast")

    @staticmethod
    def _allowed_resolutions(model: str) -> set[str]:
        if model in _SEEDANCE_MODELS:
            return {"720p"}
        if model in ("3.0pro", "3.0_pro"):
            return {"1080p"}
        return {"720p", "1080p"}

    # ---- CLI command building ----

    def _build_command(self, inputs: dict[str, Any]) -> list[str]:
        operation = inputs.get("operation", "text_to_video")
        prompt = (inputs.get("prompt") or "").strip()
        duration = int(inputs.get("duration", 5))
        model = inputs.get("model_version") or "seedance2.0fast"
        resolution = inputs.get("video_resolution")
        ratio = inputs.get("aspect_ratio", "16:9")

        if operation == "text_to_video":
            cmd = [
                _CLI, "text2video",
                f"--prompt={prompt}",
                f"--duration={duration}",
                f"--ratio={ratio}",
                f"--model_version={model}",
            ]
            if resolution:
                cmd.append(f"--video_resolution={resolution}")

        elif operation == "image_to_video":
            cmd = [
                _CLI, "image2video",
                f"--image={inputs['image_path']}",
                f"--prompt={prompt}",
                f"--duration={duration}",
                f"--model_version={model}",
            ]
            if resolution:
                cmd.append(f"--video_resolution={resolution}")

        elif operation == "frames_to_video":
            cmd = [
                _CLI, "frames2video",
                f"--first={inputs['first_frame_path']}",
                f"--last={inputs['last_frame_path']}",
                f"--prompt={prompt}",
                f"--duration={duration}",
                f"--model_version={model}",
            ]
            if resolution:
                cmd.append(f"--video_resolution={resolution}")

        elif operation == "multiframe_to_video":
            paths = [str(p) for p in inputs.get("image_paths") or []]
            cmd = [_CLI, "multiframe2video", "--images=" + ",".join(paths)]
            t_prompts = inputs.get("transition_prompts") or []
            t_durations = inputs.get("transition_durations") or []
            if len(paths) == 2 and not t_prompts:
                cmd.append(f"--prompt={prompt}")
                if t_durations:
                    cmd.append(f"--duration={float(t_durations[0])}")
                elif inputs.get("duration") is not None:
                    cmd.append(f"--duration={float(inputs['duration'])}")
            else:
                for tp in t_prompts:
                    cmd.append(f"--transition-prompt={tp}")
                for td in t_durations:
                    cmd.append(f"--transition-duration={float(td)}")

        else:  # multimodal_to_video
            cmd = [_CLI, "multimodal2video"]
            for p in inputs.get("image_paths") or []:
                cmd.append(f"--image={p}")
            for p in inputs.get("video_paths") or []:
                cmd.append(f"--video={p}")
            for p in inputs.get("audio_paths") or []:
                cmd.append(f"--audio={p}")
            if prompt:
                cmd.append(f"--prompt={prompt}")
            cmd.extend([
                f"--duration={duration}",
                f"--ratio={ratio}",
                f"--model_version={model}",
            ])
            if resolution:
                cmd.append(f"--video_resolution={resolution}")

        cmd.append("--poll=0")
        return cmd

    # ---- CLI interaction ----

    def _check_login(self) -> tuple[Optional[int], Optional[str]]:
        """Verify the OAuth session via `dreamina user_credit` (free call).

        Returns (credit_balance, error). error is None when logged in.
        """
        try:
            proc = self.run_command([_CLI, "user_credit"], timeout=60)
        except Exception as exc:
            return None, (
                "Dreamina session check failed — you are probably not logged "
                f"in (auth issue, not a prompt/tool bug). Fix: run `dreamina login` "
                f"and complete the OAuth device login, then retry. Detail: {exc}"
            )
        payload = self._parse_json(proc.stdout)
        if not payload or "total_credit" not in payload:
            return None, (
                "Could not read Jimeng credit balance — the `dreamina` session "
                "looks logged out. Fix: run `dreamina login`, then retry. "
                f"CLI output: {(proc.stdout or '').strip()[:300]}"
            )
        return int(payload.get("total_credit", 0)), None

    def _submit(self, cmd: list[str]) -> str:
        # Submit also uploads any local reference files first — allow time.
        try:
            proc = self.run_command(cmd, timeout=900)
        except ToolCommandError as exc:
            detail = exc.detail or str(exc)
            if _COMPLIANCE_MARKER in detail:
                raise RuntimeError(self._compliance_message(detail)) from exc
            raise RuntimeError(f"submit failed: {detail[:800]}") from exc

        stdout = proc.stdout or ""
        payload = self._parse_json(stdout)
        submit_id = (payload or {}).get("submit_id")
        if not submit_id:
            match = _SUBMIT_ID_RE.search(stdout)
            submit_id = match.group(0) if match else None
        if not submit_id:
            raise RuntimeError(
                f"submit returned no submit_id. CLI output: {stdout.strip()[:800]}"
            )
        return str(submit_id)

    def _poll(
        self, submit_id: str, *, poll_interval: float, timeout_seconds: int
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        last_status = ""
        while time.time() < deadline:
            time.sleep(min(poll_interval, max(0.0, deadline - time.time())))
            try:
                proc = self.run_command(
                    [_CLI, "query_result", f"--submit_id={submit_id}"], timeout=120
                )
            except ToolCommandError as exc:
                # Transient network/CLI hiccup — keep polling until deadline.
                last_status = f"query_error: {str(exc)[:200]}"
                continue
            payload = self._parse_json(proc.stdout) or {}
            status = str(payload.get("gen_status", "")).lower()
            last_status = status or last_status
            fail_reason = str(payload.get("fail_reason", "") or "")

            if _COMPLIANCE_MARKER in fail_reason:
                raise RuntimeError(self._compliance_message(fail_reason))
            if status == "success":
                return payload
            if status in _FAILED_STATUSES:
                raise RuntimeError(
                    f"generation failed (submit_id={submit_id}, status={status})"
                    + (f": {fail_reason}" if fail_reason else "")
                )
            # in_queue / generating / querying — keep waiting
        raise TimeoutError(
            f"task {submit_id} did not finish within {timeout_seconds}s "
            f"(last status: {last_status or 'unknown'}). It may still complete — "
            f"check later with: dreamina query_result --submit_id={submit_id}"
        )

    def _download(self, submit_id: str, output_path: Optional[str]) -> Path:
        target = Path(output_path) if output_path else None
        download_dir = (target.parent if target else Path(".")).resolve()
        download_dir.mkdir(parents=True, exist_ok=True)

        # Server keeps the finished asset; re-download is free. Retry a couple
        # of times before giving up (transmission failures are the cheap kind).
        last_error: Optional[Exception] = None
        for _ in range(3):
            try:
                proc = self.run_command(
                    [
                        _CLI, "query_result",
                        f"--submit_id={submit_id}",
                        f"--download_dir={download_dir}",
                    ],
                    timeout=600,
                )
                payload = self._parse_json(proc.stdout) or {}
                videos = (payload.get("result_json") or {}).get("videos") or []
                paths = [v.get("path") for v in videos if v.get("path")]
                downloaded = next(
                    (Path(p) for p in paths
                     if Path(p).is_file() and Path(p).stat().st_size > 0),
                    None,
                )
                if downloaded is None:
                    raise RuntimeError(
                        f"download reported no usable video file: {paths or 'none'}"
                    )
                if target is None:
                    return downloaded
                if downloaded.resolve() != target.resolve():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(downloaded), str(target))
                return target
            except Exception as exc:  # noqa: BLE001 — retry then surface
                last_error = exc
                time.sleep(5)
        raise RuntimeError(
            f"download failed after 3 attempts (submit_id={submit_id}; "
            f"re-download later is free): {last_error}"
        )

    # ---- Parsing helpers ----

    @staticmethod
    def _parse_json(text: Optional[str]) -> Optional[dict[str, Any]]:
        """Extract the first JSON object from CLI stdout (tolerates log lines)."""
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start:end + 1])
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _first_video_meta(payload: dict[str, Any]) -> dict[str, Any]:
        videos = (payload.get("result_json") or {}).get("videos") or []
        if not videos:
            return {}
        v = videos[0]
        return {
            k: v[k] for k in ("fps", "width", "height", "duration") if k in v
        }

    @staticmethod
    def _compliance_message(detail: str) -> str:
        return (
            "Jimeng requires a one-time compliance authorization for this model. "
            "Open the Jimeng web UI (jimeng.jianying.com), run one generation "
            "with this model manually to accept the confirmation, then retry. "
            "Do NOT auto-retry — this is a terminal state until authorized. "
            f"Detail: {detail[:300]}"
        )
