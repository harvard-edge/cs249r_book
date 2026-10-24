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


def test_engine_has_no_csv_references():
    """Tier 1 vs. Tier 2 gate: mlsysim/engine/ must never reference chapter CSV files."""
    engine_dir = CORE_DIR.parent / "engine"
    offenders = []
    for py in sorted(engine_dir.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        if ".csv" in text or "load_series" in text:
            offenders.append(py.name)
    assert not offenders, (
        f"mlsysim/engine/ must not reference chapter CSVs or load_series (Tier 2 data belongs in chapters): {offenders}"
    )


def test_literature_has_only_cross_chapter_registries():
    """Tier 1 vs. Tier 3 gate: Literature owns only cross-chapter scaling/modeling constants."""
    lit_data_dir = CORE_DIR.parent / "literature" / "data"
    allowed_yamls = {
        "batchsize.yaml",
        "benchmarks.yaml",
        "chinchilla.yaml",
        "communication.yaml",
        "training.yaml",
        "training_runs.yaml",
    }
    actual_yamls = {p.name for p in lit_data_dir.glob("*.yaml")}
    assert actual_yamls == allowed_yamls, (
        f"Single-chapter case-study YAMLs belong inlined in their chapter (Tier 3), not in literature/data/: "
        f"extra={sorted(actual_yamls - allowed_yamls)}"
    )
    from mlsysim.literature.registry import Literature
    allowed_attrs = {
        "BatchSize",
        "Benchmarks",
        "Chinchilla",
        "Communication",
        "KaplanScalingLaws",
        "Training",
        "TrainingRuns",
    }
    actual_attrs = {k for k in vars(Literature) if not k.startswith("_")}
    assert actual_attrs == allowed_attrs, (
        f"Literature registry must only expose cross-chapter modeling/scaling classes: "
        f"extra={sorted(actual_attrs - allowed_attrs)}"
    )


def test_provenance_catalog_has_no_orphans():
    """Every Provenance constant in provenance_catalog.py must have at least one consumer in mlsysim."""
    from mlsysim.core.provenance_catalog import Provenance
    pkg_dir = CORE_DIR.parent
    corpus = []
    for p in pkg_dir.rglob("*"):
        if p.suffix in (".py", ".yaml") and p.name != "provenance_catalog.py":
            corpus.append(p.read_text(encoding="utf-8"))
    full_text = "\n".join(corpus)
    defined = {k for k in vars(Provenance) if not k.startswith("_") and k.isupper()}
    orphans = sorted(k for k in defined if k not in full_text)
    assert not orphans, f"Orphaned Provenance constants in provenance_catalog.py: {orphans}"

