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
        "logged_out": [
            "a[href*='accounts.google.com/ServiceLogin']",
            "a[href*='/signin']",
            "button[aria-label*='Sign in' i]",
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
