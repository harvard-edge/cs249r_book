"""Regression tests for the composite-prose semantic scanner.

These guard against the scanner silently rotting into a no-op: each red-flag
pattern must FIRE on a known-bad rendered sentence and stay quiet on good ones.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "book" / "tools" / "audit" / "fmt"))

import audit_prose_semantics as A  # noqa: E402


def codes_for(text: str) -> set[str]:
    found = {code for code, pat in A.CHECKS if pat.search(text)}
    found |= {code for code, _m in A._numeric_semantic_findings(text)}
    return found


def test_unit_duplication_variants_flagged():
    assert "double_unit" in codes_for("about 7.6 PB PB total")
    assert "unit_abbr_plus_word" in codes_for("2.56 MW megawatts used")
    assert "unit_abbr_plus_word" in codes_for("totaling over 7.6 PB petabytes")


def test_percent_and_points_duplication_flagged():
    assert "percent_then_percent" in codes_for("a 26% percentage points drop")
    assert "percent_then_percent" in codes_for("rose 50 percent percent")


def test_multiplier_glyph_word_duplication_flagged():
    assert "times_then_word" in codes_for("a 6× times speedup")
    assert "word_then_times" in codes_for("a 6 times × speedup")
    assert "double_glyph" in codes_for("a 6× × speedup")


def test_multiplier_direction_contradiction_flagged():
    assert "mult_direction" in codes_for("this is 0.5× faster than before")
    assert "mult_direction" in codes_for("roughly 0.3 times larger overall")


def test_currency_called_percent_flagged():
    assert "currency_as_percent" in codes_for("costs $5 percent of budget")


def test_unresolved_or_leaked_markup_flagged():
    assert "unresolved_ref" in codes_for("we spend `{python} X.cost_str`")
    assert "leaked_glyph_cmd" in codes_for(r"a 6 \times speedup in prose")


def test_legitimate_sentences_stay_clean():
    assert codes_for("a clean 5 GB cache at 12% utilization") == set()
    assert codes_for("this is 6× faster than before") == set()
    assert codes_for("about 3× smaller footprint") == set()
    # legitimate display math on a ref-bearing line must not be flagged
    assert codes_for(r"Stall % = \frac{250 - 200}{250} = 20%") == set()
