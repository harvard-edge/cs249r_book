#!/usr/bin/env python3
"""Compatibility wrapper for the Binder-native bibliography linter.

The implementation lives in ``book/cli/checks/bib_lint.py`` so publishing
checks are owned by Binder. Keep this wrapper for older commands and docs that
still call ``python3 book/tools/bib_lint.py`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parents[1]
if str(BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(BOOK_DIR))

from cli.checks.bib_lint import *  # noqa: F401,F403,E402
from cli.checks.bib_lint import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
