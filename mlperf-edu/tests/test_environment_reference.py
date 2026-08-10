"""Every environment variable the code reads must be documented.

An undocumented knob is worse than a missing one. A researcher cannot find it,
and a reader of someone else's result cannot tell whether it was set. Several
of these variables change what a workload measures rather than how it runs, so
an undocumented one is a comparability hole rather than a convenience gap.

This test fails when code grows a variable the reference page does not mention.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "site" / "reference" / "environment.qmd"
SOURCE_ROOTS = (ROOT / "src", ROOT / "tools")

VARIABLE = re.compile(r"MLPERF_EDU_[A-Z0-9_]+")


def variables_in_code() -> set[str]:
    found: set[str] = set()
    for root in SOURCE_ROOTS:
        for path in root.rglob("*.py"):
            found.update(VARIABLE.findall(path.read_text(encoding="utf-8")))
    return found


def variables_in_reference() -> set[str]:
    return set(VARIABLE.findall(REFERENCE.read_text(encoding="utf-8")))


def test_reference_page_exists():
    assert REFERENCE.is_file(), "the environment reference page is missing"


def test_every_variable_the_code_reads_is_documented():
    undocumented = sorted(variables_in_code() - variables_in_reference())
    assert not undocumented, (
        "these environment variables are read by the code but absent from "
        f"{REFERENCE.relative_to(ROOT)}: {undocumented}"
    )


def test_reference_does_not_invent_variables():
    """A documented knob that no longer exists sends readers down a dead end."""
    stale = sorted(variables_in_reference() - variables_in_code())
    assert not stale, (
        f"{REFERENCE.relative_to(ROOT)} documents variables no code reads: {stale}"
    )


def test_contract_overrides_carry_their_comparability_warning():
    """The distinction between execution settings and contract overrides is the
    whole point of the page; losing it would make the page actively misleading."""
    text = " ".join(REFERENCE.read_text(encoding="utf-8").split())
    assert "Contract overrides" in text or "Contract Overrides" in text
    assert "research variant" in text
    assert "MLPERF_EDU_MAX_QUALITY_TARGET" in text
    assert "never be score-bearing" in text


def contract_override_variables() -> set[str]:
    """Variables the reference page lists under Contract Overrides."""
    text = REFERENCE.read_text(encoding="utf-8")
    section = text.split("## Contract Overrides", 1)[1].split("## Profile Controls", 1)[0]
    return set(VARIABLE.findall(section))


def test_contract_overrides_are_captured_in_provenance():
    """A knob that changes what is measured must appear in the run's provenance.

    Otherwise two results can differ because of an override that no reviewer
    can see, which is exactly the comparability failure the suite exists to
    prevent.
    """
    # Two capture paths exist by design: the performance fingerprint allowlist
    # and the broader experiment-plan key set. A variable recorded by either is
    # visible to a reviewer.
    captured: set[str] = set()
    for name in ("fingerprint.py", "experiment.py"):
        source = (ROOT / "src" / "mlperf" / name).read_text(encoding="utf-8")
        captured.update(re.findall(r'"(MLPERF_EDU_[A-Z0-9_]+)"', source))
    uncaptured = sorted(contract_override_variables() - captured)
    assert not uncaptured, (
        "these variables change what a workload measures but are not recorded "
        f"in provenance by fingerprint.py: {uncaptured}"
    )
