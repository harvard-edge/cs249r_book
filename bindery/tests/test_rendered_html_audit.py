"""Regression tests for reader-visible HTML defect detection."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "book" / "tools" / "audit"))

from check_rendered_html import CHECKS, visible_text  # noqa: E402


def test_removed_math_does_not_join_neighboring_unit_words():
    raw = "<p>about 6 <span class=\"math\">\\(\\times\\)</span> parameters × tokens</p>"
    text = visible_text(raw)
    doubled_unit = dict(CHECKS)["doubled-unit"]

    assert not doubled_unit.search(text)
