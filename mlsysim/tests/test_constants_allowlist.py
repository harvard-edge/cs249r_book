"""CI gate: core/constants.py stays DELETED — units live in core/units.py only.

History: the taxonomy refactor (2026-05) reduced this module to a units-only
re-export; the no-backward-compat sweep (2026-06-06) deleted it outright and
migrated every consumer (package, tests, book LEGO cells, docs, tools) to
``mlsysim.core.units``. This pin prevents the junk drawer — or a compat shim
for it — from quietly coming back: there is exactly one home for measurement
units, and registry/physics values have category homes of their own.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[1] / "mlsysim" / "core"


def test_constants_module_stays_deleted():
    assert not (CORE_DIR / "constants.py").exists(), (
        "core/constants.py was deleted in the 2026-06 no-backward-compat sweep; "
        "units belong in core/units.py and domain values in their registries. "
        "Do not reintroduce the module."
    )
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mlsysim.core.constants")


def test_units_module_carries_the_measurement_surface():
    units = importlib.import_module("mlsysim.core.units")
    # Spot-pin the names every consumer migrated onto, so a units.py refactor
    # cannot silently strand the book's LEGO cells.
    for name in ("ureg", "Q_", "GB", "GiB", "BYTES_FP16", "PRECISION_MAP", "resolve_precision"):
        assert hasattr(units, name), f"core.units lost {name!r}"
