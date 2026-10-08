"""The notation registry follows its conventions (see notation/data/README.md).

Every entry must satisfy the shape rules (enforced at load) and the naming and
field conventions (``convention_problems``). There is no exemption list: an
entry whose LaTeX keeps a field convention the kind rule would reject says why
in its own ``render_exception`` field.
"""
from __future__ import annotations

from pathlib import Path

from mlsysim.notation import Notation
from mlsysim.notation.types import convention_problems

DATA = Path(__file__).resolve().parents[1] / "mlsysim" / "notation" / "data"


def test_every_entry_follows_conventions():
    failures = {s.macro: convention_problems(s) for s in Notation.list() if convention_problems(s)}
    assert not failures, "\n".join(f"{m}: {'; '.join(p)}" for m, p in sorted(failures.items()))


def test_aliases_point_to_real_symbols():
    for alias, canon in Notation.aliases().items():
        assert Notation.get_by_macro(canon) is not None, f"{alias} -> {canon} (missing)"


def test_every_entry_has_a_domain_file():
    domains = {p.stem for p in DATA.glob("*.yaml")} - {"exempt", "vocabulary"}
    for s in Notation.list():
        assert s.domain in domains, f"{s.macro} has domain {s.domain!r}"


def test_no_two_entries_share_latex_and_meaning():
    seen = {}
    for s in Notation.list():
        key = (s.latex.strip(), " ".join(str(s.definition).lower().split()))
        assert key not in seen, f"{s.macro} duplicates {seen[key]} (same LaTeX and meaning): make one an alias"
        seen[key] = s.macro


def test_unit_dimensions():
    from mlsysim.notation.types import unit_dimension
    assert unit_dimension("bytes/s") == {"information": 1, "time": -1}
    assert unit_dimension("USD/token") == {"currency": 1, "token": -1}
    assert unit_dimension("USD/GB/month") == {"currency": 1, "information": -1, "time": -1}
    assert unit_dimension("V/(rad/s)") == {"mass": 1, "length": 2, "time": -2, "current": -1, "angle": -1}
    assert unit_dimension("kg*m^2") == {"mass": 1, "length": 2}
    assert unit_dimension("dimensionless") == {}
    assert unit_dimension("Bytes") is None
    assert unit_dimension("loss per parameter unit") is None


def test_quantity_words_catch_misuse():
    from mlsysim.notation.types import NotationSymbol, convention_problems
    def probs(macro, kind="scalar", unit="s"):
        return convention_problems(NotationSymbol(macro=macro, latex="x", definition="a test quantity", kind=kind, unit=unit))
    assert probs(r"\QueueWaitTime") == []
    assert probs(r"\QueueWaitTime", unit="bytes")            # a Time in bytes
    assert probs(r"\RequestArrivalRate", unit="requests/s") == []
    assert probs(r"\ErrorRate", unit="dimensionless") == []  # a share per trial
    assert probs(r"\PayloadRate", unit="bytes")               # neither per time nor a share
    assert probs(r"\DeviceCount", kind="scalar", unit="dimensionless")
    assert probs(r"\BaseLearningRate", unit="dimensionless") == []   # idiom
