"""Recognize generated trees that source checks must never scan.

A local render leaves build output and caches inside ``books/``: HTML and PDF
output under ``_build/``, Quarto caches under ``.quarto/``, and per-chapter
figure directories named ``<chapter>_files/``. Those trees hold stale copies
of sources and assets, so a check that walks them reports defects that do not
exist in the tracked content.
"""

from __future__ import annotations

from pathlib import Path

GENERATED_DIRS = frozenset(
    {".quarto", "_build", "__pycache__", ".venv", "venv", "node_modules"}
)


def is_generated(path: Path, root: Path) -> bool:
    """True when *path* lies inside a generated tree below *root*."""
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        parts = path.parts[:-1]
    return any(part in GENERATED_DIRS or part.endswith("_files") for part in parts)
