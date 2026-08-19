---
name: suno-web-music
description: Generate music on the user's own Suno subscription through a logged-in browser session (tool suno_web_music) instead of an API key. Covers prompt construction for instrumental beds, the two-candidate workflow, duration handling, credit discipline, and selector repair. Use before calling suno_web_music.
---

# Suno via a logged-in browser session

`suno_web_music` spends the user's **Suno subscription**, not an API key. Suno
has no self-serve public API, so the web app is the only way to reach a
subscription programmatically. Billing is credits, never USD — `estimate_cost()`
returns `0.0` on purpose.

Do not confuse it with `suno_music`, a separate tool that calls the third-party
reseller `sunoapi.org` and needs `SUNO_API_KEY`. Different provider, different
billing, usually unavailable.

## One-time setup

```bash
pip install playwright
python -m playwright install chromium
python -m tools._browser login suno      # sign in once, in the window that opens
python -m tools._browser status           # confirm
```

The login opens a real window, waits for the signed-in create page to render,
and records a marker. OpenMontage never reads, types, or stores credentials.

## Writing the prompt

Suno responds to a **style sentence**, not a story. Name four things:

| Element | Example |
|---|---|
| Genre / form | `cinematic underscore`, `lo-fi beat`, `ambient drone` |
| Mood | `building dread`, `wistful`, `triumphant` |
| Instrumentation | `dark strings and pulsing sub-bass`, `felt piano, tape hiss` |
| Tempo | `80 BPM`, `slow`, `driving` |

```
Cinematic tension underscore, dark strings and pulsing sub-bass,
building dread with a single low piano motif, instrumental, 80 BPM
```

English works better than Chinese. Keep it under ~200 characters — long prompts
dilute rather than refine.

**Always pass `instrumental: true` for anything under narration.** Vocals and a
voiceover fight for the same midrange, and no amount of ducking fixes it. The
tool flips the Instrumental switch *and* adds "instrumental, no vocals" to the
prompt text, because the switch is the selector most likely to have drifted.

Check `data.instrumental_toggle_applied` in the result. If it's `false`, the
switch was not found and only the prompt wording asked for it — listen before
trusting the track.

## Duration: Suno decides, you trim

`supports.exact_duration` is `false`. Suno picks the length (typically 1-4
minutes for a full clip). For a 45-second video:

1. Generate.
2. Read `data.duration_seconds` from the result.
3. Trim to length in the edit/compose stage with ffmpeg, with a fade:

```bash
ffmpeg -y -i bgm.mp3 -t 47 -af "afade=t=out:st=44:d=3" bgm_trimmed.mp3
```

Never ask Suno for "a 45-second track" and expect 45 seconds. Ask for the right
*mood*, then cut. If the piece needs to be longer than the clip, note a loop
point instead of regenerating.

## Two candidates per generation

One generation renders two clips. That is one credit spend for two options.

- `download_all: false` (default) — saves the first clip only.
- `download_all: true` — saves both as `<name>.1.mp3` and `<name>.2.mp3`.

**Prefer `download_all: true` when the music matters.** Auditioning two files
you already paid for costs nothing; regenerating because the first was wrong
costs another credit. Listen to both, keep the better one, delete the other.

## Credit discipline

`retry_policy.max_retries = 0` — deliberately. Regeneration spends credits, so
the tool never retries by itself; the agent decides.

| Failure | Meaning | What to do |
|---|---|---|
| `auth issue` in the error | Session expired or never confirmed | Re-run `python -m tools._browser login suno`. Never retry blind. |
| `quota issue` / `not succeed on retry` | Suno reports no credits | **Terminal.** Stop, tell the user, offer `pixabay_music` (free) or `google_music` (paid, bills GCP). Do not retry. |
| Timeout with no clip | Still rendering, or streaming never started | Raise `timeout_seconds` (default 420). Re-run with `headless: false` to watch. |
| Selector error naming a group | The page changed | Repair the selector — see below. One credit is not spent when submit never happened. |

Music generation takes 1-3 minutes. Budget for it: it is the slowest asset in a
short-video run apart from video generation itself.

## When a selector breaks

The `suno` selector table was written from Suno's documented UI, **not verified
against a live signed-in DOM** — that needs the user's own login. Expect to
repair one or two on the first real run. Failures dump a screenshot and the page
HTML; the error names the directory.

Repair without touching code:

```json
// ~/.openmontage/browser/selectors.json
{"suno": {"create_button": ["#the-real-button"]}}
```

An override replaces that key's list entirely. Groups that matter:
`prompt_input`, `create_button`, `instrumental_toggle`, `logged_out`,
`quota_exhausted`, `audio_element`. If a repair is durable, move it into
`tools/_browser/selectors.py` so the next user inherits it.

Audio capture does **not** go through the download menu. The tool sniffs the
`cdn*.suno.ai/*.mp3` responses the page fetches for playback, then downloads
them through the page context so cookies apply. That is the least drift-prone
surface on the site — prefer fixing selectors over changing this strategy.

## Modes

`mode: "simple"` (default) uses the one Song Description box — fewest selectors,
most robust. Use it unless you need something it cannot express.

`mode: "custom"` fills Styles / Title / Lyrics separately. More control, more
DOM surface that can drift. Worth it when you need a specific title in the
user's Suno library, or actual lyrics (`instrumental: false`).

## One session at a time

One browser profile runs one request at a time — the tool serializes on a
thread lock plus a file lock. Do not try to generate several tracks in parallel;
queue them. Two tracks for one video is usually one too many anyway.

## Example call

```python
from tools.audio.suno_web_music import SunoWebMusic

result = SunoWebMusic().execute({
    "prompt": (
        "Cinematic tension underscore, dark strings and pulsing sub-bass, "
        "sparse low piano pulses that grow heavier, one hard hit then silence, "
        "instrumental, 80 BPM"
    ),
    "instrumental": True,
    "download_all": True,
    "output_path": "projects/<project-id>/assets/music/bgm.mp3",
    "timeout_seconds": 420,
})
```

Always write under `projects/<project-id>/assets/music/` — assets outside the
project workspace are invisible to the Backlot board.
