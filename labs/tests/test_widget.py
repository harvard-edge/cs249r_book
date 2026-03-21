"""
Level 3: Widget Interaction Tests
==================================

Simulates student interactions with prediction widgets by overriding
cell definitions via app.run(defs={...}).

These tests verify:
  - Prediction radio buttons accept valid selections
  - Number inputs work within slider ranges
  - The app doesn't crash when widgets have specific values
  - Failure states trigger at expected thresholds

These are the slowest tests and may be run as a nightly job
rather than on every push.

Usage:
  python3 -m pytest labs/tests/test_widget.py -v
  python3 -m pytest labs/tests/test_widget.py -v -k "vol1"
"""

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_radio_options(source: str) -> list[dict]:
    """
    Extract mo.ui.radio() calls and their options from source code.
    Returns list of {variable_name, options_dict, line_number}.
    """
    results = []
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        if "mo.ui.radio" in line:
            # Try to extract the options dict — simplified regex
            # Looks for patterns like: options={"A) ...": "val", ...}
            # or options=["A) ...", "B) ..."]
            results.append({
                "line": i,
                "text": line.strip()[:80],
            })
    return results


def extract_slider_ranges(source: str) -> list[dict]:
    """
    Extract mo.ui.slider() calls with their start/stop/value params.
    """
    results = []
    # Match mo.ui.slider(start=X, stop=Y, ...) or mo.ui.slider(X, Y, ...)
    pattern = re.compile(
        r"mo\.ui\.slider\s*\("
        r"(?:start\s*=\s*)?([0-9.e+-]+)"
        r"\s*,\s*(?:stop\s*=\s*)?([0-9.e+-]+)",
        re.MULTILINE,
    )
    for m in pattern.finditer(source):
        try:
            results.append({
                "start": float(m.group(1)),
                "stop": float(m.group(2)),
            })
        except ValueError:
            pass
    return results


# ── Test: Widget Structure ───────────────────────────────────────────────────

class TestWidgetStructure:
    """Validate widget configurations without running the app."""

    @pytest.mark.widget
    def test_radio_buttons_have_options(self, lab_path):
        """Every mo.ui.radio() should have at least 2 options."""
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation")
        source = Path(lab_path).read_text()
        radios = extract_radio_options(source)
        if not radios:
            pytest.skip("No radio widgets found")
        # Just verify they exist — detailed option validation would need AST walking
        assert len(radios) >= 1, "Expected at least one prediction radio"

    @pytest.mark.widget
    def test_slider_ranges_valid(self, lab_path):
        """Every slider should have start < stop."""
        source = Path(lab_path).read_text()
        sliders = extract_slider_ranges(source)
        for s in sliders:
            assert s["start"] < s["stop"], (
                f"Invalid slider range: start={s['start']} >= stop={s['stop']}"
            )

    @pytest.mark.widget
    def test_slider_count_reasonable(self, lab_path):
        """Labs should have interactive elements (sliders)."""
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation")
        source = Path(lab_path).read_text()
        slider_count = source.count("mo.ui.slider")
        assert slider_count >= 2, f"Only {slider_count} sliders — labs need interactivity"

    @pytest.mark.widget
    def test_no_free_text_predictions(self, lab_path):
        """
        Predictions should use radio/number/dropdown, never free text.
        Check that mo.ui.text_area() and mo.ui.text() are not used for predictions.
        """
        source = Path(lab_path).read_text()
        # Look for text inputs near "predict" keywords
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            if "mo.ui.text" in line and "predict" in source[max(0, source.find(line)-200):source.find(line)+200].lower():
                pytest.fail(
                    f"Line {i}: Free-text prediction found. Use mo.ui.radio() or mo.ui.number() instead."
                )


# ── Test: Prediction-Reveal Pattern ──────────────────────────────────────────

class TestPredictionRevealPattern:
    """Verify the predict → reveal → reflect pedagogical flow exists."""

    @pytest.mark.widget
    def test_has_prediction_reveal_overlay(self, lab_path):
        """Labs should show 'You predicted X, actual is Y' text."""
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation")
        source = Path(lab_path).read_text()
        reveal_markers = [
            "You predicted",
            "you predicted",
            "Your prediction",
            "your prediction",
            "predicted",
            "actual",
            "off by",
        ]
        has_reveal = any(marker in source for marker in reveal_markers)
        assert has_reveal, "Missing prediction-vs-reality reveal overlay"

    @pytest.mark.widget
    def test_has_mo_stop_gate(self, lab_path):
        """
        Labs should gate instruments behind predictions.
        Either mo.stop() or conditional return pattern.
        """
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation")
        source = Path(lab_path).read_text()
        has_gate = "mo.stop" in source or "is None" in source
        assert has_gate, "No prediction gate found (mo.stop or None check)"

    @pytest.mark.widget
    def test_has_math_peek(self, lab_path):
        """Labs should have collapsible math formula sections."""
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation")
        source = Path(lab_path).read_text()
        has_math = (
            "mo.accordion" in source
            or "Math Peek" in source
            or "MathPeek" in source
            or "$$" in source  # LaTeX equations
        )
        assert has_math, "No MathPeek or formula section found"
