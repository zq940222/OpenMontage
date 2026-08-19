---
name: gemini-web-image
description: |
  Generate images through the Gemini web app on the user's own subscription (Gemini Pro / Google One AI) instead of an API key, driven by a logged-in Chromium profile. Use when: (1) the user has a Gemini subscription but no GOOGLE_API_KEY, (2) you need character or scene reference sheets that stay consistent across a shoot, (3) you need keyframes that will feed a video model (first/last frame, multi-keyframe modes), (4) you want to edit a generated image conversationally. Backs the `gemini_web_image` tool. Covers the one-time login, prompting for a UI with no ratio or seed control, the mandatory watermark cleanup before an image becomes a video reference, and how to repair broken selectors when the web UI changes.
allowed-tools: Bash, Read, Write
---

# Gemini web image generation (subscription-backed)

The Gemini web app generates images with the user's subscription quota. The
`gemini_web_image` tool (`runtime = browser`, `provider = gemini_web`,
`capability = image_generation`) drives it through a persistent Chromium
profile that holds the login — no API key, no per-image USD cost.

This is a **web session**, not an API. It can log out, hit quota, or change its
DOM at any time. Treat those as auth/availability blockers to surface, not as
prompt problems to work around.

## Setup (one time)

```bash
pip install playwright
python -m playwright install chromium
python -m tools._browser login gemini    # opens a window; sign in by hand
python -m tools._browser status          # confirm the recorded login
```

The login command never reads or types credentials — the user signs in
themselves in the visible browser. Session cookies live in
`~/.openmontage/browser/gemini` (outside the repo, since they are credentials).
`python -m tools._browser logout gemini` forgets the session.

`ffmpeg` on PATH is strongly recommended — see watermark cleanup below.

## Calling it

```python
image_selector.execute({
    "prompt": "...",                       # English works best
    "aspect_ratio": "9:16",                # best-effort: written into the prompt
    "image_paths": ["03-design/characters/lin-front.png"],  # optional references
    "output_path": "projects/<id>/assets/images/scene-03.png",
    "preferred_provider": "gemini_web",
})
```

Always pass an explicit `output_path` under `projects/<project-id>/` — that is
the workspace contract, and it is what the Backlot board reads.

## Prompting for a UI with no controls

The web app exposes no ratio, size, seed, or negative-prompt control. Everything
must live in the prompt text.

- **Write the ratio into the prompt.** The tool appends `Aspect ratio: 16:9.`
  automatically; reinforce it with framing language ("vertical composition,
  full-body subject centered"). It is a request, not a constraint — verify the
  result before using it in a ratio-sensitive video mode.
- **English prompts outperform Chinese** for style fidelity, even on Chinese
  subjects. Write the scene in English; keep proper nouns as needed.
- **No negative prompts.** Say what you want present, not what to avoid.
- **No seed.** Consistency comes from *reference images*, not reproducibility:
  generate the character sheet once, then attach it via `image_paths` for every
  later shot. Reuse appearance wording verbatim across prompts and change only
  the camera angle or action.
- **Write the prompt as one paragraph.** The composer submits on Enter, so the
  tool collapses whitespace before typing. Don't rely on line breaks for
  structure — use sentences and semicolons.
- **Reference sheets** — ask for the sheet explicitly:
  `Character reference sheet, full body, front view, neutral pose, plain
  background` plus every appearance anchor spelled out.
- **Editing** — attach the image via `image_paths` and describe the change
  ("same character, same outfit, now three-quarter view").

## Watermark cleanup (do not skip for video references)

Gemini stamps a small **sparkle glyph in the bottom-right corner** — measured on
a real 1024×572 output: **15×12 px, 31px from the right edge, 42px from the
bottom**. If a stamped image is used as a reference frame, the video model
reproduces the watermark into the footage, where it cannot be fixed.

Removal is on by default (`remove_watermark: true`) and `watermark_mode: "auto"`
picks the best remover installed:

| Mode | Quality | Needs |
|---|---|---|
| `fsr` | **Excellent** — frequency-selective reconstruction continues the surrounding texture through the patch. Verified invisible at 100% on rippled water. ~0.7s | `pip install opencv-contrib-python` |
| `lama` | Excellent — LaMa deep inpainting; strongest on large or structured holes | `pip install simple-lama-inpainting` (pulls torch) |
| `telea` | Fair — diffusion fill; leaves a smooth flat blob on textured areas | `pip install opencv-python` |
| `delogo` | Basic — ffmpeg blur patch; visibly smudged | `ffmpeg` on PATH |
| `crop` | Flawless pixels, but the bottom strip is gone and the ratio changed | `ffmpeg` on PATH |

Two properties make the repair invisible, and both matter more than the
algorithm: the mask **hugs the glyph** (0.11% of the frame, not the 2.2% a
percentage-based box would take — a big box leaves an obvious flat rectangle
whatever inpainter fills it), and the algorithm runs on a **small window** of
surrounding pixels, which is both the texture it reconstructs from and the
reason FSR_BEST takes under a second instead of minutes.

Still do these two things:

1. **Look at the corner of the returned image.** Confirm
   `data.watermark_cleanup.applied` is `true` and read `quality`. If anything
   remains, try `watermark_mode: "lama"`, or `"crop"` when the bottom strip is
   expendable.
2. **If `applied` is `false`**, the image is still watermarked — the `reason`
   names the fix. Do not feed it to a video model until it is cleaned.

`keep_raw: true` keeps the original as `<name>.raw.<ext>` for comparison.
`watermark_box: {x, y, width, height}` overrides the geometry if Gemini ever
moves the mark — check a `keep_raw` original before guessing.

The invisible SynthID watermark is unaffected by any of this and is not
something to try to remove.

## File format

Gemini serves **JPEG**, whatever extension you ask for. The tool re-encodes when
your `output_path` says otherwise, and reports both `format` (what is on disk)
and `source_format` (what Gemini sent). Ask for `.png` and you get real PNG.

## When it breaks

| Symptom | What it means | Fix |
|---|---|---|
| "No confirmed Gemini browser login" | never logged in, or the profile was deleted | `python -m tools._browser login gemini` |
| "session is not logged in" mid-run | cookies expired | same — log in again, then retry |
| "no selector for gemini.X matched" | the web UI's DOM changed | see selector repair below |
| Timeout with no image | prompt refused, quota exhausted, or a layout change | check the debug snapshot before assuming a layout change |
| `BrowserLockTimeout` | another run holds the profile | wait, or delete the stale `openmontage.lock` in the profile dir |

Every failure writes a screenshot and the page HTML to
`~/.openmontage/browser/_debug/gemini/<timestamp>-failure/`. Read those before
theorizing — they usually name the problem outright (a consent dialog, a quota
notice, a redesigned composer).

### Selector repair

Selectors are data, not code, in `tools/_browser/selectors.py`. Override them
without touching the repo by writing `~/.openmontage/browser/selectors.json`:

```json
{"gemini": {"prompt_input": ["div.new-composer[contenteditable='true']"]}}
```

Each key holds an ordered list; the first selector that resolves wins. To find
the right one, run with `headless: false`, or open the dumped `page.html`.

Image extraction deliberately avoids the download button: the tool sniffs image
bytes off the network responses the page itself loads, and falls back to
re-fetching the `img` src inside the page context. That path survives most UI
redesigns, so a selector break usually only affects input, not capture.

## Limits worth stating at proposal time

- **One request at a time.** A Chromium profile cannot be opened twice; parallel
  scene generation serializes here. For a large batch, budget the wall clock or
  use an API provider for the bulk and Gemini for hero frames.
- **Not for unattended runs.** A re-login prompt mid-batch stops the run.
- **Quota is the user's subscription.** There is no per-image price to report —
  say "free with your Gemini subscription" and note the quota risk instead.
