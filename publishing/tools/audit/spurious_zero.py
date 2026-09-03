"""Backward-compat re-export — prefer ``book.tools.audit.fmt.spurious_zero``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent / "fmt" / "spurious_zero.py"
_spec = importlib.util.spec_from_file_location("_fmt_spurious_zero", _path)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SPURIOUS_ZERO = _mod.SPURIOUS_ZERO
is_spurious_zero_false_positive = _mod.is_spurious_zero_false_positive
find_spurious_zeros = _mod.find_spurious_zeros

__all__ = [
    "SPURIOUS_ZERO",
    "is_spurious_zero_false_positive",
    "find_spurious_zeros",
]
