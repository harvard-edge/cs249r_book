"""Tests for MLSysIM Mathematical Notation invariants and validity."""
from __future__ import annotations

import re
import pytest

from mlsysim.notation import Notation, Symbols
from mlsysim.notation.types import NotationSymbol


def test_notation_registry_loaded():
    """Verify notation registry loads non-empty collection of validated symbols."""
    assert len(Notation) >= 480
    assert Notation is Symbols


def test_macros_are_strict_pascal_case():
    """All macro names must start with backslash and follow strict PascalCase."""
    for s in Notation:
        assert s.macro.startswith("\\"), f"Macro {s.macro} missing backslash"
        name = s.macro[1:]
        assert re.match(r"^[A-Z][a-zA-Z0-9]*$", name), f"Macro {s.macro} not PascalCase"
        # Ensure lowercase bad patterns do not exist in the exact casing
        assert not re.search(r"(toend|topoint|walkingspeed|ironlaw)", name), f"Bad casing in {s.macro}"


def test_no_duplicate_macros():
    """Every macro name must be globally unique."""
    macros = [s.macro for s in Notation]
    assert len(macros) == len(set(macros)), "Found duplicate macro names in registry"


def test_iron_law_reserved_symbols():
    """Verify Iron Law single letters preserve their canonical series meanings."""
    assert Notation.BatchSize.latex == "B"
    assert Notation.DatasetSize.latex == "D"
    assert Notation.DataVolume.latex == "D_{\\text{vol}}"
    assert Notation.Bandwidth.latex == "\\text{BW}"
    assert Notation.Operations.latex == "O"
    assert Notation.PeakRate.latex == "R_{\\text{peak}}"
    assert Notation.Time.latex == "T"
    assert Notation.Latency.latex == "L_{\\text{lat}}"


def test_all_symbols_render_in_matplotlib():
    """Every symbol's latex string must be renderable in Matplotlib without crashing."""
    mathtext = pytest.importorskip("matplotlib.mathtext")
    _parser = mathtext.MathTextParser("path")
    failures = []
    for s in Notation:
        math_str = f"${s.latex}$"
        try:
            _parser.parse(math_str)
        except Exception as e:
            failures.append((s.macro, s.latex, str(e)))

    assert not failures, f"Matplotlib failed to parse {len(failures)} symbols: {failures[:5]}"
