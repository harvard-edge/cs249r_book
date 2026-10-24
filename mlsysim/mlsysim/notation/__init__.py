"""MLSysIM Notation Package.

Canonical single source of truth for mathematical notations, symbols, and LaTeX macros
across all volumes of MLSysBook.
"""
from __future__ import annotations

from .types import NotationSymbol
from .registry import Notation, Symbols, SYMBOLS, NotationRegistry

__all__ = [
    "NotationSymbol",
    "NotationRegistry",
    "Notation",
    "Symbols",
    "SYMBOLS",
]
