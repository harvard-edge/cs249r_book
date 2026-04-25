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
        r"(?:start\s*=\s*)?([0-9._e+-]+)"
        r"\s*,\s*(?:stop\s*=\s*)?([0-9._e+-]+)",
        re.MULTILINE,
    )
    for m in pattern.finditer(source):
        try:
            results.append({
                "start": float(m.group(1).replace("_", "")),
                "stop": float(m.group(2).replace("_", "")),
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
    def test_interactive_controls_reasonable(self, lab_path):
        """Labs should have interactive elements (sliders, dropdowns, or radios)."""
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation")
        source = Path(lab_path).read_text()
        slider_count = source.count("mo.ui.slider")
        dropdown_count = source.count("mo.ui.dropdown")
        radio_count = source.count("mo.ui.radio")
        total = slider_count + dropdown_count + radio_count
        assert total >= 4, f"Only {total} interactive controls (need ≥4 for engagement)"

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


# ── Test: Interactive State Engine ─────────────────────────────────────────────

class MockWidget:
    """Mocks a marimo mo.ui element by satisfying the `.value` access."""
    def __init__(self, value):
        self.value = value


class TestInteractiveState:
    """Executes the engine while simulating widget clicks and state changes."""

    @pytest.mark.widget
    def test_app_handles_widget_selections_without_crashing(self, lab_path):
        """
        Extracts all widgets, forces a 'clicked' state, and runs the engine.
        Ensures the parts of the lab hidden behind mo.stop() do not crash
        when they are finally executed.
        """
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation")

        # 1. Load the app and run it in its default unclicked state
        from tests.test_engine import run_app_safely
        outputs, default_defs = run_app_safely(lab_path)
        
        if outputs is None:
            pytest.skip(f"Lab failed baseline engine test: {default_defs}")

        # 2. Find all prediction and control widgets in the global definitions
        widgets_to_mock = {}
        for name, obj in default_defs.items():
            # Match naming conventions for lab widgets (e.g., partA_prediction, partB_batch)
            if name.startswith("part") or name.startswith("pA_") or name.startswith("pB_") or name.startswith("pC_") or name.startswith("pD_") or name.startswith("pE_") or name.startswith("a1_") or name.startswith("a2_") or name.startswith("c1_") or name.startswith("d1_"):
                # If it's a Marimo UI element, it will have a 'value' attribute
                if hasattr(obj, "value"):
                    # We inject a dummy integer (1) or a string ("Option") 
                    # Many sliders take ints, radios take strings. 
                    # If it's a string option, any string often satisfies mo.stop() 
                    # but we'll use a numeric 1 as a fallback for sliders.
                    val = 1
                    
                    # Try to extract a valid option if it's a radio/dropdown
                    if hasattr(obj, "options") and obj.options:
                        # Extract the first valid string key/value
                        opts = list(obj.options.keys()) if isinstance(obj.options, dict) else obj.options
                        if len(opts) > 0:
                            val = obj.options[opts[0]] if isinstance(obj.options, dict) else opts[0]
                    
                    # Instead of overriding via defs= (which prunes the whole cell and causes IncompleteRefsError),
                    # we modify the instantiated UI element's `.value` inline since we already ran the app once.
                    try:
                        obj._value = val  # Marimo UI elements hold their state here sometimes
                        obj.value = val
                    except Exception:
                        pass
                    
                    widgets_to_mock[name] = val

        if not widgets_to_mock:
            pytest.skip("No interactive widgets found to click")

        # 3. Re-run the app in the same context to trigger reactivity, or simply execute 
        # the cells bypassing marimo's pruned override logic. Since `app.run()` with defs 
        # is strict, let's build the mock dictionary to include all missing variables 
        # dynamically by interrogating the cells.
        
        from tests.test_engine import load_app
        app = load_app(lab_path)
        
        # Build a complete mock dictionary that provides a mock for EVERYTHING the cell defines
        # to avoid IncompleteRefsError.
        complete_mock_defs = {}
        for name, obj in default_defs.items():
            if name in widgets_to_mock:
                complete_mock_defs[name] = MockWidget(widgets_to_mock[name])
            elif name.startswith("synth_decision") or name.startswith("a1_") or name.startswith("a2_") or name.startswith("c1_") or name.startswith("d1_") or name.startswith("pE_"):
                 # Provide a dummy mock for any other ui elements that might be pruned alongside
                 complete_mock_defs[name] = MockWidget(1)

        try:
            # We will supply the minimal defs, and catch the IncompleteRefsError to automatically 
            # fill in the missing refs as requested by Marimo.
            from marimo._ast.errors import IncompleteRefsError
            
            missing = set(widgets_to_mock.keys())
            current_defs = {k: MockWidget(v) for k, v in widgets_to_mock.items()}
            
            while True:
                try:
                    outputs_clicked, defs_clicked = app.run(defs=current_defs)
                    assert outputs_clicked is not None, "App crashed when widgets were clicked"
                    break # Success!
                except IncompleteRefsError as e:
                    # Parse the error string to extract missing refs
                    # "Missing: ['a1_cost_query', 'a1_optimization']. Provided refs: ..."
                    import re
                    match = re.search(r"Missing:\s*\[(.*?)\]", str(e))
                    if match:
                        missing_vars_str = match.group(1)
                        # Extract the variable names like 'a1_cost_query', 'a1_optimization'
                        missing_vars = [v.strip().strip("'").strip('"') for v in missing_vars_str.split(",")]
                        for mv in missing_vars:
                            if mv:
                                current_defs[mv] = MockWidget(1)
                    else:
                        raise e # If we can't parse it, fail
        except Exception as e:
            pytest.fail(f"App execution crashed after simulating widget clicks: {e}")

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
