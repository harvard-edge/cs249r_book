"""Dataset registry — dataset profiles.

Leaf reference data (example counts, image dimensions, class counts) lives as
YAML under ``datasets/data/<Dataset>.yaml`` and is loaded + validated against
the ``DatasetProfile`` schema at import (see ``core/loader.py`` and
``.claude/rules/mlsysim.md`` → *Storage format*).
"""
from pathlib import Path

from ..core.loader import load_registry
from .types import DatasetProfile

_DATA = Path(__file__).parent / "data"

Datasets = load_registry(
    _DATA, DatasetProfile, name="Datasets",
    doc="Registry namespace for Datasets.",
)
