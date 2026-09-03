#!/usr/bin/env python3
"""Compatibility wrapper for the Binder-native LEGO prose literals check."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from book.cli.checks.lego_prose_literals import (  # noqa: E402,F401
    check_file,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
