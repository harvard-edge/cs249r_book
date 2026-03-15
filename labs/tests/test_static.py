"""
Level 1: Static Analysis Tests
===============================

Fast checks that don't execute any code. Safe for CI on every push.

Tests:
  - AST syntax validity
  - Marimo structure (import marimo, app = marimo.App, @app.cell)
  - WASM bootstrap presence
  - Required imports (mlsysim, plotly, DesignLedger)
  - No @sec- references in student-facing strings
  - No hardcoded Cortex-M7 references (should be ESP32)
  - Cell count sanity (minimum 4 cells per lab)
  - Tab structure (mo.ui.tabs present)
"""

import ast
import re
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_source(lab_path: str) -> str:
    with open(lab_path) as f:
        return f.read()


def parse_tree(lab_path: str) -> ast.Module:
    return ast.parse(read_source(lab_path))


# ── Test: Syntax ─────────────────────────────────────────────────────────────

class TestSyntax:
    """Every lab file must be valid Python."""

    def test_ast_parse(self, lab_path):
        """File parses without SyntaxError."""
        parse_tree(lab_path)

    def test_no_tabs_in_indentation(self, lab_path):
        """No mixed tabs/spaces — Marimo requires consistent indentation."""
        source = read_source(lab_path)
        for i, line in enumerate(source.split("\n"), 1):
            if line and line[0] == "\t":
                pytest.fail(f"Tab indentation at line {i}: {line[:40]}")


# ── Test: Marimo Structure ───────────────────────────────────────────────────

class TestMarimoStructure:
    """Every lab must have valid Marimo notebook structure."""

    def test_imports_marimo(self, lab_path):
        source = read_source(lab_path)
        assert "import marimo" in source, "Missing 'import marimo'"

    def test_has_app(self, lab_path):
        source = read_source(lab_path)
        assert "app = marimo.App" in source, "Missing 'app = marimo.App'"

    def test_has_cells(self, lab_path):
        """At least 4 @app.cell decorators."""
        source = read_source(lab_path)
        cell_count = len(re.findall(r"@app\.cell", source))
        assert cell_count >= 4, f"Only {cell_count} cells (need ≥4)"

    def test_wasm_bootstrap(self, lab_path):
        """WASM bootstrap for browser deployment."""
        source = read_source(lab_path)
        assert 'sys.platform == "emscripten"' in source, "Missing WASM bootstrap"

    def test_has_tabs(self, lab_path):
        """Labs should use mo.ui.tabs() for Part navigation."""
        source = read_source(lab_path)
        # Lab 00 might not have tabs
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 may not use tabs")
        assert "mo.ui.tabs" in source, "Missing mo.ui.tabs() — Parts should be tabs"


# ── Test: Required Imports ───────────────────────────────────────────────────

class TestRequiredImports:
    """Every lab must import the core lab infrastructure."""

    def test_imports_mlsysim(self, lab_path):
        source = read_source(lab_path)
        has_mlsysim = "import mlsysim" in source or "from mlsysim" in source
        assert has_mlsysim, "Missing mlsysim import"

    def test_imports_design_ledger(self, lab_path):
        source = read_source(lab_path)
        assert "DesignLedger" in source, "Missing DesignLedger import"

    def test_imports_style(self, lab_path):
        source = read_source(lab_path)
        assert "COLORS" in source, "Missing COLORS import from style"
        assert "LAB_CSS" in source, "Missing LAB_CSS import from style"

    def test_imports_plotly(self, lab_path):
        source = read_source(lab_path)
        assert "plotly" in source or "go." in source, "Missing Plotly import"


# ── Test: Content Quality ────────────────────────────────────────────────────

class TestContentQuality:
    """Check for common content issues."""

    def test_no_sec_refs_in_strings(self, lab_path):
        """@sec- references don't render in Marimo — should not be in displayed content."""
        source = read_source(lab_path)
        lines = source.split("\n")
        violations = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Check for @sec- in string contexts
            if "@sec-" in line:
                # Heuristic: if the line contains f", f', mo.Html, mo.md, it's displayed
                if any(marker in line for marker in ['f"', "f'", "mo.Html", "mo.md", '"""', "'''"]):
                    violations.append(f"Line {i}: {stripped[:60]}")
        assert not violations, f"@sec- refs in displayed content:\n" + "\n".join(violations)

    def test_no_cortex_m7(self, lab_path):
        """Cortex-M7 not in mlsysim registry — use ESP32."""
        source = read_source(lab_path)
        # Only check non-comment lines
        for line in source.split("\n"):
            if not line.strip().startswith("#") and "Cortex-M7" in line:
                pytest.fail(f"Found 'Cortex-M7' in code — use Hardware.ESP32 instead")

    def test_no_models_generic(self, lab_path):
        """Models.Generic was removed — should not appear."""
        source = read_source(lab_path)
        assert "Models.Generic" not in source, "Models.Generic removed — use specific model or ModelSpec"

    def test_no_hardware_edge_jetson_bare(self, lab_path):
        """Hardware.Edge.Jetson should be Hardware.Edge.JetsonOrinNX."""
        source = read_source(lab_path)
        if re.search(r"Hardware\.Edge\.Jetson(?!OrinNX)\b", source):
            pytest.fail("Use Hardware.Edge.JetsonOrinNX, not Hardware.Edge.Jetson")

    def test_has_prediction_lock(self, lab_path):
        """Every lab (except Lab 00) must have at least one prediction interaction."""
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation — no prediction locks required")
        source = read_source(lab_path)
        has_radio = "mo.ui.radio" in source
        has_number = "mo.ui.number" in source
        has_select = "mo.ui.dropdown" in source or "mo.ui.slider" in source
        assert has_radio or has_number or has_select, (
            "No prediction widget found (mo.ui.radio/number/dropdown)"
        )

    def test_has_failure_state(self, lab_path):
        """Every lab should have at least one failure state visual."""
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation")
        source = read_source(lab_path)
        failure_markers = [
            "kind=\"danger\"",
            'kind="danger"',
            "OOM",
            "SLA",
            "violation",
            "FAIL",
            "failure",
            "#CB202D",  # RedLine color
            "RedLine",
        ]
        has_failure = any(marker in source for marker in failure_markers)
        assert has_failure, "No failure state found — every lab needs at least one"
