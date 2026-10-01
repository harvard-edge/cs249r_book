"""
Stale-export detection and milestone registry provenance.

`tito milestone run` only proves each required symbol is exported; without a
fingerprint it silently runs an OLD package after the student edits a
notebook. These tests pin the fingerprint record written at export time and
the warning `milestone run` prints. All state lives in a temp project, never
the checkout's real .tito/.
"""

import io
import json
import shutil
from pathlib import Path

import pytest
from rich.console import Console

from tito.commands.milestone import MILESTONE_SCRIPTS, _warn_on_stale_exports
from tito.commands.module.workflow import (
    ModuleWorkflowCommand,
    notebook_export_fingerprint,
    record_module_export,
    stale_export_report,
)
from tito.core.config import CLIConfig


TINYTORCH_ROOT = Path(__file__).resolve().parents[2]
MODULE = "02_activations"


def _notebook(cells):
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _code(source, outputs=None, count=None):
    return {"cell_type": "code", "source": source, "metadata": {},
            "outputs": outputs or [], "execution_count": count}


def _write_student_notebook(root: Path, cells) -> Path:
    path = root / "modules" / MODULE / "activations.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_notebook(cells)), encoding="utf-8")
    return path


def _base_cells():
    return [
        _code("#| default_exp core.activations"),
        _code("#| export\nclass ReLU:\n    pass\n"),
        _code("def test_unit_relu():\n    assert True\n"),
        {"cell_type": "markdown", "source": "notes", "metadata": {}},
    ]


def _console():
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=200), buf


# ---------------------------------------------------------------- fingerprint

def test_fingerprint_ignores_outputs_tests_and_markdown(tmp_path):
    nb = _write_student_notebook(tmp_path, _base_cells())
    before = notebook_export_fingerprint(nb)

    cells = _base_cells()
    cells[1] = _code("#| export\nclass ReLU:\n    pass\n",
                     outputs=[{"output_type": "stream", "name": "stdout", "text": "hi"}], count=7)
    cells[2] = _code("def test_unit_relu():\n    assert 1 == 1\n")
    cells[3] = {"cell_type": "markdown", "source": "edited notes", "metadata": {}}
    _write_student_notebook(tmp_path, cells)

    assert notebook_export_fingerprint(nb) == before


def test_fingerprint_changes_when_exported_code_changes(tmp_path):
    nb = _write_student_notebook(tmp_path, _base_cells())
    before = notebook_export_fingerprint(nb)
    cells = _base_cells()
    cells[1] = _code("#| export\nclass ReLU:\n    scale = 2\n")
    _write_student_notebook(tmp_path, cells)
    assert notebook_export_fingerprint(nb) != before


def test_fingerprint_missing_notebook_is_none(tmp_path):
    assert notebook_export_fingerprint(tmp_path / "nope.ipynb") is None


# --------------------------------------------------------------- stale report

def test_no_warning_on_fresh_checkout_without_notebooks(tmp_path):
    """Fresh checkout: no student notebooks, no record -> nothing to report."""
    assert stale_export_report(tmp_path, [1, 2, 3]) == {"stale": [], "unrecorded": []}
    assert not (tmp_path / ".tito").exists()


def test_recorded_then_unchanged_is_clean(tmp_path):
    _write_student_notebook(tmp_path, _base_cells())
    record_module_export(tmp_path, MODULE)
    data = json.loads((tmp_path / ".tito" / "exports.json").read_text())
    assert data["modules"]["02"]["notebook"] == "modules/02_activations/activations.ipynb"
    assert stale_export_report(tmp_path, [1, 2]) == {"stale": [], "unrecorded": []}


def test_edit_after_export_is_stale(tmp_path):
    _write_student_notebook(tmp_path, _base_cells())
    record_module_export(tmp_path, MODULE)
    cells = _base_cells()
    cells[1] = _code("#| export\nclass ReLU:\n    def forward(self, x):\n        return x\n")
    _write_student_notebook(tmp_path, cells)
    assert stale_export_report(tmp_path, [2])["stale"] == ["02"]


def test_notebook_without_record_is_unrecorded_not_stale(tmp_path):
    _write_student_notebook(tmp_path, _base_cells())
    assert stale_export_report(tmp_path, [2]) == {"stale": [], "unrecorded": ["02"]}


def test_corrupt_record_file_degrades_to_unrecorded(tmp_path):
    _write_student_notebook(tmp_path, _base_cells())
    (tmp_path / ".tito").mkdir()
    (tmp_path / ".tito" / "exports.json").write_text("{not json", encoding="utf-8")
    assert stale_export_report(tmp_path, [2]) == {"stale": [], "unrecorded": ["02"]}


# ------------------------------------------------------------ milestone output

def test_milestone_warning_text_for_stale_module(tmp_path):
    _write_student_notebook(tmp_path, _base_cells())
    record_module_export(tmp_path, MODULE)
    cells = _base_cells()
    cells[1] = _code("#| export\nclass ReLU:\n    changed = True\n")
    _write_student_notebook(tmp_path, cells)

    console, buf = _console()
    report = _warn_on_stale_exports(console, tmp_path, [1, 2])
    out = buf.getvalue()
    assert report["stale"] == ["02"]
    assert "Module 02 changed since you last exported" in out
    assert "tito module complete 02" in out


def test_milestone_prints_nothing_when_fresh(tmp_path):
    console, buf = _console()
    _warn_on_stale_exports(console, tmp_path, [1, 2, 3])
    assert buf.getvalue() == ""


# ------------------------------------------------------- export writes record

def test_export_module_records_fingerprint(tmp_path):
    """The real export path (used by `tito module complete`) writes the record."""
    pytest.importorskip("nbdev")
    src_notebook = TINYTORCH_ROOT / "modules" / MODULE / "activations.ipynb"
    if not src_notebook.exists():
        pytest.skip("student notebook for Module 02 not generated in this checkout")
    (tmp_path / "src" / MODULE).mkdir(parents=True)
    shutil.copy(TINYTORCH_ROOT / "src" / MODULE / f"{MODULE}.py", tmp_path / "src" / MODULE)
    (tmp_path / "modules" / MODULE).mkdir(parents=True)
    shutil.copy(src_notebook, tmp_path / "modules" / MODULE / "activations.ipynb")

    cmd = ModuleWorkflowCommand(CLIConfig.from_project_root(tmp_path))
    cmd.console = Console(file=io.StringIO())
    assert cmd.export_module(MODULE) == 0
    assert (tmp_path / "tinytorch" / "core" / "activations.py").is_file()

    record = json.loads((tmp_path / ".tito" / "exports.json").read_text())["modules"]["02"]
    assert record["sha256"] == notebook_export_fingerprint(
        tmp_path / "modules" / MODULE / "activations.ipynb")
    assert stale_export_report(tmp_path, [2]) == {"stale": [], "unrecorded": []}


# ------------------------------------------------------------------- registry

# Measured provenance (cProfile per part, 2026-09) plus each running module's
# declared Prerequisites. A module that neither runs nor is such a
# prerequisite must not gate a milestone.
EXPECTED_PARTS = {
    ("03", 0): [1, 2, 3, 4, 6, 7],
    ("03", 1): [1, 2, 3, 4, 5, 6, 7],
    ("04", 0): [1, 2, 3, 4, 5, 6, 7, 9],
    ("04", 1): [1, 2, 3, 4, 5, 6, 7, 9],
    ("05", 0): [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13],
    ("05", 1): [1, 2, 3, 4, 5, 6, 7, 11, 12, 13],
    ("05", 2): [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13],
    ("06", 0): [1, 2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ("06", 1): [1, 2, 3, 4, 6, 11, 12, 13, 14, 18],
}


@pytest.mark.parametrize("key,expected", sorted(EXPECTED_PARTS.items()))
def test_part_requirements_match_measured_provenance(key, expected):
    milestone_id, index = key
    assert MILESTONE_SCRIPTS[milestone_id]["scripts"][index]["required_modules"] == expected


@pytest.mark.parametrize("milestone_id", ["03", "04", "05", "06"])
def test_top_level_is_union_of_parts(milestone_id):
    milestone = MILESTONE_SCRIPTS[milestone_id]
    union = sorted({m for s in milestone["scripts"] for m in s["required_modules"]})
    assert milestone["required_modules"] == union


def test_milestone_07_keeps_module_17_prerequisites():
    assert MILESTONE_SCRIPTS["07"]["required_modules"] == [1, 6, 9, 14, 17]


def test_cnn_highlight_does_not_claim_every_gradient():
    from tito.commands.milestone import MILESTONE_ACHIEVEMENT_HIGHLIGHTS
    lines = " ".join(MILESTONE_ACHIEVEMENT_HIGHLIGHTS["04"])
    assert "Every gradient" not in lines


def test_gpt_has_no_main_module_fallback():
    source = (TINYTORCH_ROOT / "tinytorch" / "models" / "transformer.py").read_text()
    assert "sys.modules" not in source
    assert "getattr(_main" not in source


@pytest.mark.parametrize("name", ["simd_ops", "mps_ops", "triton_gelu"])
def test_extensions_are_not_labelled_autogenerated(name):
    head = (TINYTORCH_ROOT / "tinytorch" / "extensions" / f"{name}.py").read_text().splitlines()[:6]
    text = "\n".join(head)
    assert "AUTOGENERATED" not in text
    assert "Hand-maintained" in text
