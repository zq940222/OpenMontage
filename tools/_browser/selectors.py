"""Externalized DOM selectors for browser-driven providers.

Web UIs change without notice. Keeping selectors as data means a break is a
one-line edit here — or, with zero code changes, a user-supplied override file:

    ~/.openmontage/browser/selectors.json      (or $OPENMONTAGE_BROWSER_SELECTORS)

    {"gemini": {"prompt_input": ["div.my-new-editor[contenteditable]"]}}

Each value is an ordered list of Playwright selectors tried in turn; the first
one that resolves wins. Overrides replace a key's list entirely.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SELECTORS: dict[str, dict[str, list[str]]] = {
    "gemini": {
        # Where the prompt is typed. Gemini uses a Quill contenteditable.
        "prompt_input": [
            "rich-textarea div[contenteditable='true']",
            "div[contenteditable='true'][role='textbox']",
            "div.ql-editor[contenteditable='true']",
            "textarea[aria-label]",
        ],
        # Submit control (Enter usually suffices; this is the fallback).
        "send_button": [
            "button[aria-label*='Send' i]",
            "button[aria-label*='发送']",
            "button.send-button",
        ],
        # Hidden file input used for reference-image upload.
        "file_input": [
            "input[type='file']",
        ],
        # Any of these means the session is not logged in.
        # The localized entries are not optional garnish: on a zh-CN Gemini the
        # sign-in control is a bare `button` reading 登录 with no href and no
        # English aria-label, so the three ASCII selectors above all miss and
        # is_logged_out() reports a logged-out page as signed in. That false
        # green makes the login CLI's wait_for_signed_in() pass on its first
        # poll (the anonymous landing page also carries a prompt composer),
        # so it records a login that never happened and closes the window
        # before the user can type. Match the text EXACTLY — measured against
        # the live page: `button:has-text('登录')` also matches 退出登录 on a
        # signed-in page (would invert the bug), and `button:text-is('登录')`
        # matches nothing because the label sits in a nested Material span.
        # `button >> text="登录"` is the form that hits the real control and
        # still ignores 退出登录.
        "logged_out": [
            "a[href*='accounts.google.com/ServiceLogin']",
            "a[href*='/signin']",
            "button[aria-label*='Sign in' i]",
            'button >> text="登录"',
            'a >> text="登录"',
        ],
        # Generated images inside the response stream. These must be
        # STRUCTURAL: matching on host (googleusercontent.com) also matches the
        # signed-in user's avatar, and matching on size also matches UI art
        # like gstatic's 512px sparkle logo. Verified against the live DOM:
        # button < div < div < single-image < generated-image < response-element
        "response_image": [
            "generated-image img",
            "single-image img",
            "response-element img",
        ],
        # Present while the model is still producing a response.
        "generating": [
            "button[aria-label*='Stop' i]",
            "button[aria-label*='停止']",
        ],
    },
    # Suno's create page. Two authoring modes: Simple ("Song Description" box +
    # an Instrumental switch) and Custom (separate Styles / Lyrics / Title
    # fields). The tool defaults to Simple because it needs the fewest
    # selectors, and every selector here is a drift risk.
    #
    # NOTE: unlike the gemini table, these were written from Suno's documented
    # UI rather than verified against the live signed-in DOM (that needs the
    # user's own login). Expect to repair one or two on first real run — the
    # tool dumps a screenshot + HTML on failure precisely so that repair is a
    # one-line edit here or in ~/.openmontage/browser/selectors.json.
    "suno": {
        # Simple mode's song-description box. Also the signed-in probe the
        # login CLI waits for, so it must be something that only renders for a
        # signed-in session.
        # Deliberately no bare "textarea" catch-all: the login CLI treats a
        # prompt_input match as proof of a signed-in session, and Suno's
        # logged-out landing page carries a describe-your-song box as a signup
        # funnel. A catch-all here would record a login that never happened.
        #
        # maxlength is what separates the box you can actually type in from the
        # decoys. A signed-in /create page carries four textareas; measured on
        # the live DOM they are lyrics (no maxlength), styles (1000), the real
        # Song Description composer (3000), and a hidden 500-char
        # "Describe the sound you want" box. first_locator picks the first
        # selector with count() > 0 and count() does not care about visibility,
        # so a placeholder match lands on the hidden 500 box and every click
        # against it times out after 120s. Keep the maxlength entry first.
        #
        # Not verified: whether the logged-out signup funnel also uses
        # maxlength=3000. If it does, this entry inherits the same
        # false-signed-in risk the placeholder entries below already carry —
        # it does not add a new one.
        "prompt_input": [
            "textarea[maxlength='3000']",
            "textarea[placeholder*='describe' i]",
            "textarea[placeholder*='description' i]",
            "textarea[placeholder*='song' i]",
            "textarea[data-testid*='prompt' i]",
            "div[contenteditable='true'][role='textbox']",
        ],
        # Custom mode: the genre/instrumentation field.
        "style_input": [
            "textarea[placeholder*='style' i]",
            "input[placeholder*='style' i]",
            "textarea[placeholder*='genre' i]",
            "textarea[data-testid*='style' i]",
        ],
        "title_input": [
            "input[placeholder*='title' i]",
            "input[data-testid*='title' i]",
        ],
        "lyrics_input": [
            "textarea[placeholder*='lyric' i]",
            "textarea[data-testid*='lyric' i]",
        ],
        # Switch that suppresses vocals. Critical for narration-backing BGM:
        # vocals fight the voiceover for the same frequencies.
        "instrumental_toggle": [
            "button[role='switch'][aria-label*='instrumental' i]",
            "[data-testid*='instrumental' i]",
            "button[role='switch']:near(:text('Instrumental'))",
            "label:has-text('Instrumental') button[role='switch']",
            "div:has-text('Instrumental') > button[role='switch']",
        ],
        "custom_mode_toggle": [
            "button:has-text('Custom')",
            "[role='tab']:has-text('Custom')",
            "[data-testid*='custom' i]",
        ],
        "create_button": [
            "button[data-testid*='create' i]",
            "button:has-text('Create')",
            "button:has-text('创作')",
            "button[type='submit']",
        ],
        # Any of these means the session is not logged in.
        "logged_out": [
            "a[href*='/login']",
            "button:has-text('Sign in')",
            "a:has-text('Sign in')",
            "button:has-text('Sign up')",
            "button:has-text('Log in')",
            "a:has-text('Log in')",
            "button:has-text('Continue with Google')",
        ],
        # Present while a clip is still rendering.
        "generating": [
            "[data-testid*='loading' i]",
            ":text('Generating')",
            ":text('Queued')",
        ],
        # Credit exhaustion is terminal, so this group must be RUTHLESSLY
        # specific. Playwright ':text()' is substring matching, and the create
        # page shows the balance as standing chrome ("2,340 credits remaining").
        # A candidate that matches a healthy balance would raise the tool's
        # worst error — "terminal, do not retry" — on a full account. Only
        # phrasings that cannot appear next to a non-zero balance belong here.
        "quota_exhausted": [
            ":text('out of credits')",
            ":text('Out of Credits')",
            ":text('not enough credits')",
            ":text('insufficient credits')",
        ],
        # DOM fallback when no audio response was sniffed off the network.
        "audio_element": [
            "audio[src]",
        ],
    },
}


def selectors_for(provider: str) -> dict[str, list[str]]:
    """Return the selector table for a provider, with user overrides applied."""
    base = {key: list(value) for key, value in SELECTORS.get(provider, {}).items()}
    for key, value in _overrides().get(provider, {}).items():
        if isinstance(value, str):
            base[key] = [value]
        elif isinstance(value, list):
            base[key] = [str(item) for item in value]
    return base


def _overrides() -> dict[str, Any]:
    path = os.environ.get("OPENMONTAGE_BROWSER_SELECTORS")
    override_path = (
        Path(path).expanduser()
        if path
        else Path.home() / ".openmontage" / "browser" / "selectors.json"
    )
    if not override_path.is_file():
        return {}
    try:
        loaded = json.loads(override_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
