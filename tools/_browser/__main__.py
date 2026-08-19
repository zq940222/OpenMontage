"""Entry point for `python -m tools._browser`."""

from __future__ import annotations

import sys

from tools._browser.cli import main

if __name__ == "__main__":
    sys.exit(main())
