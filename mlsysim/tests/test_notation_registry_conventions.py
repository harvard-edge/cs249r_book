"""The notation registry follows its conventions (see notation/data/README.md).

Every entry must satisfy the shape rules (enforced at load). Every entry that is
not on the frozen legacy list must also satisfy the naming and field
conventions. The legacy list only shrinks: an entry on it that now passes, or a
name on it that no longer exists, fails the test so the line gets deleted.
"""
from __future__ import annotations

from pathlib import Path

from mlsysim.notation import Notation
from mlsysim.notation.types import convention_problems

DATA = Path(__file__).resolve().parents[1] / "mlsysim" / "notation" / "data"


def _legacy() -> set[str]:
    lines = (DATA / "legacy_names.txt").read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def test_new_entries_follow_conventions():
    legacy = _legacy()
    failures = {
        s.macro: convention_problems(s)
        for s in Notation.list()
        if s.macro not in legacy and convention_problems(s)
    }
    assert not failures, "\n".join(f"{m}: {'; '.join(p)}" for m, p in sorted(failures.items()))


def test_legacy_list_only_shrinks():
    legacy = _legacy()
    gone = sorted(m for m in legacy if Notation.get_by_macro(m) is None)
    fixed = sorted(m for m in legacy if Notation.get_by_macro(m) is not None
                   and not convention_problems(Notation.get_by_macro(m)))
    assert not gone, f"delete these lines from legacy_names.txt (entry removed or renamed): {gone}"
    assert not fixed, f"delete these lines from legacy_names.txt (entry now follows the conventions): {fixed}"


def test_aliases_point_to_real_symbols():
    for alias, canon in Notation.aliases().items():
        assert Notation.get_by_macro(canon) is not None, f"{alias} -> {canon} (missing)"


def test_every_entry_has_a_domain_file():
    domains = {p.stem for p in DATA.glob("*.yaml")} - {"exempt"}
    for s in Notation.list():
        assert s.domain in domains, f"{s.macro} has domain {s.domain!r}"


def test_no_two_entries_share_latex_and_meaning():
    seen = {}
    for s in Notation.list():
        key = (s.latex.strip(), " ".join(str(s.definition).lower().split()))
        assert key not in seen, f"{s.macro} duplicates {seen[key]} (same LaTeX and meaning): make one an alias"
        seen[key] = s.macro
