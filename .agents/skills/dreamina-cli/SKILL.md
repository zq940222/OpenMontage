---
name: dreamina-cli
description: |
  Generate Seedance 2.0 video on a Jimeng (即梦) membership through the official `dreamina` CLI — no API key, billed in membership credits. Use when: (1) the user has a Jimeng/Dreamina account instead of a Volcengine or fal.ai key, (2) you need the 全能参考 (multimodal) mode with up to 9 image + 3 video + 3 audio references, (3) you need first/last-frame or multi-keyframe continuity between shots, (4) you need native synced audio in one pass, (5) prompts are in Chinese. Backs the `dreamina_video` tool. Covers login, the five generation modes and their parameter matrices, credit budgeting, the download/QC gate, and the retry discipline that keeps a failed shot from burning credits twice.
allowed-tools: Bash, Read, Write
---

# Dreamina CLI (即梦 membership video)

`dreamina` is Jimeng's official AIGC CLI. It authenticates with the user's own
account via OAuth device flow and spends **membership credits**, not USD. This
is the path to use when the user has a Jimeng subscription but no API key.

OpenMontage wraps it as the **`dreamina_video`** tool (`runtime = browser`,
`provider = dreamina`, `capability = video_generation`). Prefer the tool over
raw CLI calls — it validates parameters before submitting, classifies auth and
compliance failures, and handles the poll/download/verify sequence.

Read **`seedance-2-0`** for prompt craft (camera language, multi-beat timecodes,
`@Image1..N` reference tags, quoted dialogue for lip-sync). That skill's
prompting guidance applies verbatim here; this skill covers the delivery path.

## Setup (one time)

```bash
dreamina login            # OAuth device flow; prints a verification URL + code
dreamina user_credit      # confirm: {"total_credit": ..., "vip_level": ...}
```

`dreamina login --headless` prints a `device_code` for a machine without a
browser; finish with `dreamina login checklogin --device_code=<code>`.
Session state lives in `~/.dreamina_cli`. `dreamina relogin` forces a fresh
login when a session goes stale.

**Auth failures are auth failures.** If `user_credit` doesn't return a balance,
stop and tell the user to run `dreamina login` — do not retry generation,
reword prompts, or switch providers unilaterally.

## The five modes

| Tool `operation` | CLI command | Ratio comes from | Use it for |
|---|---|---|---|
| `text_to_video` | `text2video` | `--ratio` | establishing shots, anything with no reference art |
| `image_to_video` | `image2video` | the input image | animating one still (scene concept, product shot) |
| `frames_to_video` | `frames2video` | the first frame | a shot with a specified start **and** end state |
| `multiframe_to_video` | `multiframe2video` | the first image | 2–20 ordered keyframes → one continuous story beat |
| `multimodal_to_video` | `multimodal2video` | `--ratio` | **the flagship** — 全能参考: character/scene/style refs, driving video, audio |

`multimodal_to_video` is the default choice for character-consistent narrative
work. It accepts up to **9 images + 3 videos + 3 audio clips** (audio 2–15 s);
image order maps to `@Image1 … @Image9` in the prompt.

## Parameter matrix (validated by the tool before submitting)

| Model | Modes | Duration | Resolution |
|---|---|---|---|
| `seedance2.0fast` (default) | all except multiframe | 4–15 s | 720p |
| `seedance2.0` | all except multiframe | 4–15 s | 720p |
| `seedance2.0_vip`, `seedance2.0fast_vip` | all except multiframe | 4–15 s | 720p |
| `3.5pro` | image, frames | 4–12 s | 720p / 1080p |
| `3.0`, `3.0fast` | image (frames: 3.0 only) | 3–10 s | 720p / 1080p |
| `3.0pro` | image | 3–10 s | 1080p |
| (none) | multiframe | 0.5–8 s per segment, ≥2 s total | n/a |

Ratios: `1:1 3:4 16:9 4:3 9:16 21:9`. `multiframe2video` accepts **no** model or
resolution override — passing one is an error, not a silent ignore.

Want 1080p on a still? Use `image_to_video` with `3.5pro`/`3.0pro`. The whole
Seedance 2.0 family is 720p-only here.

## Cost — think in credits, not dollars

`estimate_cost()` returns `0.0` USD because there is no per-call charge; the
real budget is the credit balance. Observed rate: **a 6 s `seedance2.0fast`
720p clip cost 66 credits (~11 credits/second)**. `seedance2.0` (non-fast) and
longer durations cost more.

Before a batch: `dreamina user_credit`, multiply seconds × ~11 (fast) as a
floor, and present the estimate to the user at the approval gate. After the
run, report actual spend from each result's `data.credit_count`.

## Execution loop

The tool does this for you; do it in this order if you ever drive the CLI directly:

1. **Verify the session** — `dreamina user_credit` (free). Dead session → stop.
2. **Validate parameters** — duration/resolution/model combination, and that
   every referenced local file exists and is non-empty. An invalid submit still
   costs time and sometimes credits.
3. **Submit** with `--poll=0` and record the `submit_id` immediately.
4. **Poll** `dreamina query_result --submit_id=<id>` every ~30 s. Statuses:
   `in_queue` / `generating` / `querying` → keep waiting; `success` → done;
   anything else → failed.
5. **Download** `dreamina query_result --submit_id=<id> --download_dir=<dir>`.
   Re-downloading a finished task is **free** — the asset stays server-side.
6. **Verify** with ffprobe: non-zero size, parseable container, a video stream,
   duration ≥ requested − 1.5 s, aspect ratio within ±0.02 of the target.

## Retry discipline (this is where credits get wasted)

| Failure | Cost to retry | Rule |
|---|---|---|
| Zero-byte / truncated / unreadable download | free | re-download the same `submit_id`, up to 3 times |
| No video stream, badly short, wrong aspect | credits | regenerate **at most once** per shot |
| Missing audio track | credits | **never regenerate.** Record a warning; add audio in the edit stage |
| Wrong aspect on an image-derived mode | credits | **never regenerate** — the ratio comes from the input image, so it will fail identically. Fix the reference image or the target ratio |
| `AigcComplianceConfirmationRequired` | — | terminal. The user must run this model once in the Jimeng web UI to accept the confirmation |
| Insufficient credits, invalid parameters | — | terminal. Report it; retrying re-fails |

`multiframe2video` output is silent by design — a missing audio track there is
normal, not a defect.

## Gotchas

- **Long submits.** `multimodal2video` uploads every local reference before it
  returns a `submit_id`; a 9-image submit can take minutes. Don't treat slowness
  as failure.
- **Serialize dependent shots.** A shot whose first frame is extracted from the
  previous shot's tail cannot be batched — it waits for that shot to pass QC.
- **Space submissions.** Leave ≥5 s between batch submits to stay clear of rate
  controls.
- **Chinese paths and spaces** in file arguments are fine; the tool passes them
  as a single argv element (no shell quoting needed).
- **No seed.** `supports.seed` is `False`. Reproducibility comes from reusing
  the same reference images, not from a seed.
- **Images too.** The CLI also has `text2image`, `image2image`, `image_upscale`.
  OpenMontage routes stills to `gemini_web_image` (free with the user's Gemini
  subscription); wrap `text2image` only if the user asks for a Jimeng-native
  still or Gemini is unavailable.
