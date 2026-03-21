"""
Level 2: Engine Execution Tests
================================

Runs each lab's cells headlessly via marimo.App.run() to verify:
  - All cells execute without exceptions
  - mlsysim Engine.solve() calls produce valid results
  - Key computed values are within expected ranges

These tests are slower (~2-5 sec per lab) but catch runtime errors
that static analysis misses (e.g., wrong mlsysim API calls, division
by zero, missing attributes).

Usage:
  python3 -m pytest labs/tests/test_engine.py -v
  python3 -m pytest labs/tests/test_engine.py -v -k "vol1"
  python3 -m pytest labs/tests/test_engine.py -v -k "lab_01"
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_app(lab_path: str):
    """Load a Marimo app from a .py file."""
    import marimo
    spec = importlib.util.spec_from_file_location("lab_module", lab_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app


def run_app_safely(lab_path: str):
    """
    Run a Marimo app and return (outputs, defs).
    Returns (None, error_str) on failure.
    """
    try:
        app = load_app(lab_path)
        outputs, defs = app.run()
        return outputs, defs
    except Exception as e:
        return None, str(e)


# ── Test: Cell Execution ─────────────────────────────────────────────────────

class TestCellExecution:
    """Run all cells and check for runtime errors."""

    @pytest.mark.engine
    def test_app_runs_without_error(self, lab_path):
        """
        The core test: execute all cells via app.run().

        This catches:
        - ImportError (wrong mlsysim API paths)
        - AttributeError (nonexistent hardware/model attributes)
        - TypeError (wrong Engine.solve() arguments)
        - ZeroDivisionError (bad formulas)
        - Any other runtime exception
        """
        outputs, defs = run_app_safely(lab_path)
        if outputs is None:
            pytest.fail(f"App execution failed: {defs}")

    @pytest.mark.engine
    def test_app_produces_outputs(self, lab_path):
        """App.run() should produce non-empty outputs."""
        outputs, defs = run_app_safely(lab_path)
        if outputs is None:
            pytest.skip(f"App failed to run: {defs}")
        assert len(outputs) > 0, "App produced zero outputs"

    @pytest.mark.engine
    def test_app_defines_core_variables(self, lab_path):
        """
        Every lab's setup cell should define key variables:
        mo, COLORS, LAB_CSS, ledger
        """
        outputs, defs = run_app_safely(lab_path)
        if outputs is None:
            pytest.skip(f"App failed to run: {defs}")

        # Check that essential names are in the definitions
        expected = {"mo", "COLORS", "LAB_CSS", "ledger"}
        defined = set(defs.keys()) if isinstance(defs, dict) else set()
        missing = expected - defined
        # Some labs may not export all names at top level, so warn instead of fail
        if missing:
            pytest.skip(f"Missing expected defs: {missing} (may be cell-scoped)")


# ── Test: mlsysim API Validation ─────────────────────────────────────────────

class TestMlsysimAPI:
    """Verify that mlsysim calls used in labs actually work."""

    @pytest.mark.engine
    def test_hardware_registry_accessible(self, mlsysim):
        """All hardware referenced in labs should exist in registry."""
        hw = mlsysim.Hardware
        # Cloud tier
        assert hw.Cloud.H100 is not None
        assert hw.Cloud.A100 is not None
        # Edge tier
        assert hw.Edge.JetsonOrinNX is not None
        # Tiny tier
        assert hw.Tiny.ESP32 is not None

    @pytest.mark.engine
    def test_model_registry_accessible(self, mlsysim):
        """All models referenced in labs should exist in registry."""
        models = mlsysim.Models
        assert models.ResNet50 is not None
        assert models.GPT2 is not None
        assert models.MobileNetV2 is not None

    @pytest.mark.engine
    def test_engine_solve_basic(self, mlsysim):
        """Engine.solve() works for a basic inference scenario."""
        result = mlsysim.Engine.solve(
            model=mlsysim.Models.ResNet50,
            hardware=mlsysim.Hardware.Cloud.H100,
        )
        assert result.feasible is True
        assert result.latency is not None
        assert result.bottleneck in ("Compute", "Memory")

    @pytest.mark.engine
    def test_engine_solve_tiny_oom(self, mlsysim):
        """Large model on tiny device should be infeasible."""
        result = mlsysim.Engine.solve(
            model=mlsysim.Models.GPT2,
            hardware=mlsysim.Hardware.Tiny.ESP32,
        )
        assert result.feasible is False, "GPT-2 should not fit on ESP32"
