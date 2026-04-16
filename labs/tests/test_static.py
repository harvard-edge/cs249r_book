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
  - Wheel version consistency (micropip URL matches pyproject.toml)
  - WASM-incompatible import detection
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 may not use tabs")
        if "mo.ui.tabs" not in source:
            pytest.xfail("Missing mo.ui.tabs() — Parts should be tabs (WIP)")


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
        if "lab_00" in lab_path:
            pytest.skip("Lab 00 is orientation — may not use Plotly")
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
        for line in source.split("\n"):
            if not line.strip().startswith("#") and "Cortex-M7" in line:
                pytest.xfail("Found 'Cortex-M7' in code — use Hardware.ESP32 instead (WIP)")

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


# ── Test: Wheel Version Consistency ─────────────────────────────────────────

class TestWheelConsistency:
    """Ensure the micropip wheel URL in every lab matches the actual mlsysim version."""

    def test_micropip_url_matches_pyproject_version(self, lab_path):
        """The micropip wheel URL must contain the version from mlsysim/pyproject.toml."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        pyproject = REPO_ROOT / "mlsysim" / "pyproject.toml"
        with open(pyproject, "rb") as f:
            version = tomllib.load(f)["project"]["version"]

        source = read_source(lab_path)
        expected_fragment = f"../../wheels/mlsysim-{version}-py3-none-any.whl"
        bad_depth = f"../../../wheels/mlsysim-{version}-py3-none-any.whl"
        assert bad_depth not in source, (
            f"Micropip wheel path has one too many '../' segments (found '{bad_depth}'). "
            f"From labs/volN/ the repo root is two levels up; use '{expected_fragment}'."
        )
        assert expected_fragment in source, (
            f"Wheel version mismatch or not relative. Expected '{expected_fragment}' in micropip URL "
            f"but not found. Update the micropip.install() URL in this lab."
        )

    def test_no_absolute_wheel_url(self, lab_path):
        """Labs must use relative URLs for the wheel, not absolute mlsysbook.ai URLs."""
        source = read_source(lab_path)
        assert "https://mlsysbook.ai/labs/wheels/" not in source, (
            "Found absolute wheel URL. Use relative path '../../wheels/mlsysim-...' instead."
        )


# ── Test: WASM Compatibility ────────────────────────────────────────────────

WASM_BLOCKED_IMPORTS = {
    "scipy", "sklearn", "scikit-learn", "torch", "tensorflow",
    "cv2", "opencv", "psutil", "subprocess", "multiprocessing",
    "tkinter", "PyQt5", "PyQt6",
}


class TestStateImplementation:
    """Ensure state.py uses IndexedDB and not localStorage."""

    def test_no_localstorage_import(self):
        state_py_path = REPO_ROOT / "mlsysim" / "labs" / "state.py"
        with open(state_py_path, "r") as f:
            source = f.read()
        assert "from js import localStorage" not in source, (
            "Found 'from js import localStorage' in state.py. "
            "Use IndexedDB instead, as localStorage is not available in WASM web workers."
        )

class TestWASMCompatibility:
    """Catch imports that will fail in the Pyodide/WASM environment."""

    def test_no_wasm_incompatible_imports(self, lab_path):
        """Labs must not import packages unavailable in Pyodide."""
        source = read_source(lab_path)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in WASM_BLOCKED_IMPORTS:
                        pytest.fail(
                            f"WASM-incompatible import '{alias.name}' at line {node.lineno}. "
                            f"This package is not available in Pyodide."
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in WASM_BLOCKED_IMPORTS:
                    pytest.fail(
                        f"WASM-incompatible import 'from {node.module}' at line {node.lineno}. "
                        f"This package is not available in Pyodide."
                    )


# ── Test: Marimo Dataflow ────────────────────────────────────────────────────

def _has_app_cell_decorator(func):
    """True if the function is decorated with @app.cell or @app.cell(...)."""
    for dec in func.decorator_list:
        if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
            if dec.value.id == "app" and dec.attr == "cell":
                return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
            if (
                isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "app"
                and dec.func.attr == "cell"
            ):
                return True
    return False


def _is_mo_stop_call(stmt):
    """True if stmt is an expression statement calling mo.stop(...)."""
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "mo"
        and func.attr == "stop"
    )


def _is_mo_ui_assign(stmt):
    """If stmt is `<name> = mo.ui.<widget>(...)`, return the name. Else None."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Name) or not isinstance(stmt.value, ast.Call):
        return None
    func = stmt.value.func
    while isinstance(func, ast.Attribute):
        if (
            isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "mo"
            and func.value.attr == "ui"
        ):
            return target.id
        func = func.value if isinstance(func.value, ast.Attribute) else None
        if func is None:
            break
    return None


def _returned_names(func):
    """Names listed in the cell's return statement."""
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            if isinstance(node.value, ast.Tuple):
                return {elt.id for elt in node.value.elts if isinstance(elt, ast.Name)}
            if isinstance(node.value, ast.Name):
                return {node.value.id}
    return set()


class TestMarimoDataflow:
    """Detect the MULTI-widget-in-gated-cell anti-pattern.

    The sequential-unlock idiom used across the labs is:

        @app.cell
        def _(mo, partA_prediction):
            mo.stop(partA_prediction.value is None, mo.md("..."))
            partB_prediction = mo.ui.radio(...)
            return (partB_prediction,)

    This works: gate fires while partA is unanswered, user sees the
    unlock message. Once partA is answered, partB_prediction is defined
    and the next gated cell unblocks. Each gated cell exposes exactly
    ONE next widget. The idiom is used throughout lab_02 through lab_16
    and is verified working by test_engine.py.

    The BUG (originally in lab_01, fixed by #1339) is when a single
    gated cell defines MULTIPLE widgets:

        @app.cell
        def _(mo, partA_prediction):
            mo.stop(partA_prediction.value is None, mo.md("..."))
            partA_instrument = mo.ui.slider(...)  # Part A's own instrument
            partB_prediction = mo.ui.radio(...)    # Next part's prediction
            partB_instrument = mo.ui.slider(...)  # Next part's instrument too
            return (partA_instrument, partB_prediction, partB_instrument)

    When this cell's gate fires, THREE widgets go undefined. Downstream
    cells that depend on any of them also fail. The cascade is what
    breaks the lab visually, not the gate itself.

    This test flags only cells that leak TWO OR MORE widgets to the
    return tuple. Single-widget leaks (the sequential-unlock idiom)
    are allowed.
    """

    # Labs with known multi-widget-leak debt, grandfathered until refactored
    # per-lab (tracked in #1347). As each lab is converted to the
    # one-widget-per-gated-cell pattern (see lab_01 post-#1339 or lab_05_dist_train),
    # remove it from this set. When the set is empty the bug class is closed.
    # Only vol1/lab_00 remains. Its check1/check2/check3 pattern is structurally
    # different from the partX_prediction idiom and the mechanical Pattern C
    # transformation breaks test_engine.py. Needs manual per-cell refactor.
    _KNOWN_MULTI_LEAK_LABS = frozenset({
        "vol1/lab_00_introduction.py",
    })

    def test_no_multi_widget_leak_in_gated_cell(self, lab_path):
        # Grandfather known-debt labs: skip with a pointer to the tracking issue
        # rather than fail. New labs and labs that have been refactored are
        # strictly enforced.
        rel_path = str(Path(lab_path).resolve().relative_to(REPO_ROOT / "labs"))
        if rel_path in self._KNOWN_MULTI_LEAK_LABS:
            pytest.skip(
                f"{rel_path} has known multi-widget-leak debt, tracked in #1347. "
                "remove from _KNOWN_MULTI_LEAK_LABS once refactored to the "
                "one-widget-per-gated-cell pattern."
            )

        tree = parse_tree(lab_path)
        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _has_app_cell_decorator(node):
                continue
            if not any(_is_mo_stop_call(s) for s in node.body):
                continue
            widgets_defined = {}
            for s in node.body:
                name = _is_mo_ui_assign(s)
                if name is not None:
                    widgets_defined[name] = s.lineno
            leaked = widgets_defined.keys() & _returned_names(node)
            # Single-widget leaks are the sequential-unlock idiom and are allowed.
            # Only flag cells that leak two or more widgets through the gate.
            if len(leaked) >= 2:
                violations.append((node.lineno, sorted(leaked)))

        if violations:
            summary = "\n".join(
                f"  cell at line {line}: gated AND leaks {len(names)} widgets {names}"
                for line, names in violations
            )
            pytest.fail(
                "multi-widget leak in gated cell (downstream cascade failure):\n"
                + summary
                + "\n\nfix: split the cell so each gated cell defines at most ONE "
                "new widget. see vol1/lab_01_ml_intro.py (post-#1339) for the "
                "canonical pattern: each gated cell returns only the next "
                "prediction widget."
            )
