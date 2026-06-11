#!/usr/bin/env python3
"""Compatibility wrapper for the Binder-native fmt prose contract check."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from book.cli.checks.fmt_prose_contract import (  # noqa: E402,F401
    Violation,
    build_formatter_map,
    build_formatter_records,
    check_file,
    extract_python_cells,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
