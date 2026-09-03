#!/usr/bin/env python3
"""Standalone CLI for the math-canonical check.

Implementation lives in ``cli.checks.math_canonical``. Prefer::

    ./book/binder check math --scope canonical

This script remains for ad-hoc use and backward-compatible docs links.
"""

from __future__ import annotations

import sys
from pathlib import Path

# book/ on sys.path so `from cli.checks...` resolves like binder does.
_book_dir = Path(__file__).resolve().parent.parent.parent
if str(_book_dir) not in sys.path:
    sys.path.insert(0, str(_book_dir))

from cli.checks.math_canonical import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
