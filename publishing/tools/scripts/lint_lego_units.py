#!/usr/bin/env python3
"""Compatibility wrapper for the Binder-native LEGO unit discipline linter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from book.cli.checks.lego_units import (  # noqa: E402,F401
    LintIssue,
    lint_file,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
