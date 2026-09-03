#!/usr/bin/env python3
"""Backward-compat wrapper — use ``book/tools/audit/fmt/audit_prose.py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "fmt" / "audit_prose.py"
sys.argv[0] = str(_TARGET)
runpy.run_path(str(_TARGET), run_name="__main__")
