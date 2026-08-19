"""Shared browser-session infrastructure for account-driven (subscription) tools.

Tools with ``runtime = ToolRuntime.BROWSER`` drive a website the user is
already logged into, spending their subscription quota instead of an API key.
This package owns the persistent Chromium profile, the one-time login CLI, and
the externalized selector table those tools share.

Nothing here imports Playwright at module import time — ``registry.discover()``
walks the whole ``tools`` package, and a machine without Playwright must still
be able to enumerate tools.
"""

from tools._browser.session import (
    BrowserSession,
    LoginState,
    debug_dir_for,
    login_state,
    profile_dir_for,
    selectors_for,
)

__all__ = [
    "BrowserSession",
    "LoginState",
    "debug_dir_for",
    "login_state",
    "profile_dir_for",
    "selectors_for",
]
