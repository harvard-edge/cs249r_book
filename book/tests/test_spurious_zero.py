import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/audit/fmt"))

from spurious_zero import find_spurious_zeros  # noqa: E402


def test_unit_bearing_dot_zero_is_not_blanket_suppressed():
    text = "The ridge is 153.0 FLOP/byte and the transfer rate is 0.0 MB/s."
    hits = find_spurious_zeros(text)
    values = [value for value, _ in hits]
    assert "153.0" in values
    assert "0.0" in values


def test_product_versions_remain_false_positives():
    assert find_spurious_zeros("PyTorch 2.0 introduced compiler support.") == []


def test_crisp_dm_version_title_is_false_positive():
    text = "CRISP-DM 1.0: Step-by-Step Data Mining Guide."
    assert find_spurious_zeros(text) == []


def test_standards_version_is_false_positive():
    text = "The MPI Forum's standards history records Version 1.0 in May 1994."
    assert find_spurious_zeros(text) == []


def test_reference_family_identifier_is_false_positive():
    text = "Hardware Requirements for a Device Identifier Composition Engine. Family 2.0, Level 00, Revision 78."
    assert find_spurious_zeros(text) == []
