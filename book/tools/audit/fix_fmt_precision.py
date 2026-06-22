#!/usr/bin/env python3
"""Backward-compat wrapper — use ``book/tools/audit/fmt/fix_precision.py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "fmt" / "fix_precision.py"
sys.argv[0] = str(_TARGET)
runpy.run_path(str(_TARGET), run_name="__main__")
