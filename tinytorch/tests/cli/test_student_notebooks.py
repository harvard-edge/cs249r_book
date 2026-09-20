"""
Student notebooks and module completion (#1684, #2117).

`tito module start` must give a learner a notebook with the student-core
solutions cleared, for every module, and `tito module complete` must certify
the learner's notebook, never the reference implementation in src/.
"""

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rich.console import Console

from tito.commands.export_utils import convert_py_to_notebook
from tito.commands.module.test import ModuleTestCommand
from tito.commands.module.workflow import ModuleWorkflowCommand
from tito.core.config import CLIConfig
from tito.core.solutions import (
    TEXT_STUB,
    cells_with_solution_markers,
    clear_solution_regions,
    make_student_notebook,
)


TINYTORCH_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODULES = sorted((TINYTORCH_ROOT / "src").glob("[0-9][0-9]_*/[0-9][0-9]_*.py"))
STUB_COMMENT = "# YOUR CODE HERE"


def _cell(source: str, cell_type: str = "code") -> dict:
    return {"cell_type": cell_type, "metadata": {}, "source": source}


def _quiet_console() -> Console:
    return Console(file=io.StringIO(), width=120)


def _write_notebook(path: Path, code_cells) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [_cell(source) for source in code_cells],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


# ---------------------------------------------------------------------------
# The clearing policy
# ---------------------------------------------------------------------------

class TestStudentNotebookPolicy:
    def test_core_region_becomes_indented_nbgrader_stub(self):
        notebook = {"cells": [_cell(
            "def add_one(x):\n"
            "    ### BEGIN SOLUTION\n"
            "    return x + 1\n"
            "    ### END SOLUTION\n"
        )]}

        assert make_student_notebook(notebook) == []

        source = notebook["cells"][0]["source"]
        assert source == "def add_one(x):\n    # YOUR CODE HERE\n    raise NotImplementedError()\n"
        compile(source, "<cell>", "exec")

    def test_scaffold_region_stays_solved_without_markers(self):
        notebook = {"cells": [_cell(
            "def shape(x):\n"
            '    ### BEGIN SOLUTION role="scaffold"\n'
            "    return x.shape\n"
            "    ### END SOLUTION\n"
        )]}

        assert make_student_notebook(notebook) == []
        assert notebook["cells"][0]["source"] == "def shape(x):\n    return x.shape\n"

    def test_instructor_region_is_removed(self):
        notebook = {"cells": [_cell(
            "x = 1\n"
            "### BEGIN SOLUTION role=instructor\n"
            "secret = 42\n"
            "### END SOLUTION\n"
        )]}

        assert make_student_notebook(notebook) == []
        assert notebook["cells"][0]["source"] == "x = 1\n"

    def test_written_answer_gets_text_stub(self):
        source, errors = clear_solution_regions(
            "Answer:\n### BEGIN SOLUTION\nBecause memory bandwidth.\n### END SOLUTION",
            "markdown",
        )

        assert errors == []
        assert source == f"Answer:\n{TEXT_STUB}"

    def test_unmatched_marker_is_reported(self):
        notebook = {"cells": [_cell("def f():\n    ### BEGIN SOLUTION\n    return 1\n")]}

        errors = make_student_notebook(notebook)

        assert any("without matching END SOLUTION" in error for error in errors)


@pytest.mark.parametrize("source_file", SOURCE_MODULES, ids=[p.parent.name for p in SOURCE_MODULES])
def test_every_module_student_notebook_hides_reference_solutions(source_file):
    """Each of the 20 modules yields a compilable notebook with work left to do."""
    jupytext = pytest.importorskip("jupytext")
    notebook = jupytext.read(source_file)
    stubs_before = sum(cell.source.count(STUB_COMMENT) for cell in notebook.cells)

    assert make_student_notebook(notebook) == []
    assert cells_with_solution_markers(notebook) == []

    stubs_after = sum(cell.source.count(STUB_COMMENT) for cell in notebook.cells)
    assert stubs_after > stubs_before, (
        f"{source_file.parent.name}: the student notebook leaves nothing to implement"
    )
    for index, cell in enumerate(c for c in notebook.cells if c.cell_type == "code"):
        body = "\n".join(
            line for line in cell.source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        compile(body, f"{source_file.name}[code cell {index}]", "exec")


# ---------------------------------------------------------------------------
# Notebook generation on disk
# ---------------------------------------------------------------------------

DEMO_SOURCE = (
    "# %% [markdown]\n"
    "# # Demo\n"
    "\n"
    '# %% nbgrader={"grade": false, "grade_id": "impl", "solution": true}\n'
    "def add_one(x):\n"
    "    ### BEGIN SOLUTION\n"
    "    return x + 1\n"
    "    ### END SOLUTION\n"
)


def _make_demo_module(root: Path, source: str = DEMO_SOURCE) -> Path:
    src_dir = root / "src" / "01_demo"
    src_dir.mkdir(parents=True)
    (src_dir / "01_demo.py").write_text(source, encoding="utf-8")
    return src_dir


requires_jupytext_cli = pytest.mark.skipif(
    shutil.which("jupytext") is None, reason="jupytext CLI not on PATH"
)


@requires_jupytext_cli
def test_student_conversion_writes_notebook_without_solutions(tmp_path):
    src_dir = _make_demo_module(tmp_path)

    ok = convert_py_to_notebook(
        src_dir, tmp_path / ".venv", _quiet_console(), student=True, project_root=tmp_path
    )

    assert ok
    text = (tmp_path / "modules" / "01_demo" / "demo.ipynb").read_text(encoding="utf-8")
    assert "BEGIN SOLUTION" not in text
    assert "return x + 1" not in text
    assert "raise NotImplementedError()" in text


@requires_jupytext_cli
def test_student_conversion_writes_nothing_when_clearing_fails(tmp_path):
    src_dir = _make_demo_module(tmp_path, DEMO_SOURCE.replace("    ### END SOLUTION\n", ""))

    ok = convert_py_to_notebook(
        src_dir, tmp_path / ".venv", _quiet_console(), student=True, project_root=tmp_path
    )

    assert not ok
    assert not (tmp_path / "modules" / "01_demo" / "demo.ipynb").exists()


@requires_jupytext_cli
def test_reference_conversion_keeps_solutions_for_dev_export(tmp_path):
    src_dir = _make_demo_module(tmp_path)

    ok = convert_py_to_notebook(src_dir, tmp_path / ".venv", _quiet_console(), project_root=tmp_path)

    assert ok
    text = (tmp_path / "modules" / "01_demo" / "demo.ipynb").read_text(encoding="utf-8")
    assert "BEGIN SOLUTION" in text
    assert "return x + 1" in text


# ---------------------------------------------------------------------------
# `tito module complete` certifies the student's notebook
# ---------------------------------------------------------------------------

def _workflow(root: Path) -> ModuleWorkflowCommand:
    command = ModuleWorkflowCommand(CLIConfig.from_project_root(root))
    command.console = _quiet_console()
    return command


def test_unit_tests_never_fall_back_to_reference_source(tmp_path, monkeypatch):
    src_file = tmp_path / "src" / "01_tensor" / "01_tensor.py"
    src_file.parent.mkdir(parents=True)
    src_file.write_text('print("✅ reference implementation passes")\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = _workflow(tmp_path)._run_inline_unit_tests("01_tensor", verbose=False)

    assert result["passed"] == 0
    assert result["failed"] == 1


def test_unit_tests_fail_when_notebook_crashes_after_passing_tests(tmp_path, monkeypatch):
    source = tmp_path / "src" / "01_tensor" / "01_tensor.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "#| default_exp core.tensor\n"
        "def test_unit_tensor():\n    pass\n"
        "def test_module():\n    test_unit_tensor()\n",
        encoding="utf-8",
    )
    _write_notebook(
        tmp_path / "modules" / "01_tensor" / "tensor.ipynb",
        [
            "#| default_exp core.tensor\n",
            'def test_unit_tensor():\n    print("✅ Tensor creation works correctly!")\n',
            "def test_module():\n    test_unit_tensor()\n    raise NotImplementedError()\n",
            "test_module()\n",
        ],
    )
    monkeypatch.chdir(tmp_path)

    result = _workflow(tmp_path)._run_inline_unit_tests("01_tensor", verbose=False)

    assert result["passed"] == 1
    assert result["failed"] >= 1
    assert "NotImplementedError" in str(result)


def test_integration_tests_scope_export_gate_and_fail_when_it_trips(tmp_path, monkeypatch):
    test_file = tmp_path / "tests" / "01_tensor" / "test_01_tensor_progressive.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_placeholder():\n    pass\n", encoding="utf-8")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["env"] = kwargs.get("env") or {}
        return subprocess.CompletedProcess(
            cmd, 4, stdout="", stderr="❌ TINYTORCH PACKAGE NOT EXPORTED\n  • Module 01 (Tensor): broken",
        )

    monkeypatch.setattr("tito.commands.module.workflow.subprocess.run", fake_run)
    monkeypatch.chdir(tmp_path)

    result = _workflow(tmp_path)._run_integration_tests("01_tensor", verbose=False)

    assert seen["env"].get("TINYTORCH_EXPORT_CHECK_THROUGH") == "1"
    assert result["failed"] == 1


def test_module_test_command_requires_student_notebook(tmp_path):
    src_file = tmp_path / "src" / "01_tensor" / "01_tensor.py"
    src_file.parent.mkdir(parents=True)
    src_file.write_text('print("✅ reference implementation passes")\n', encoding="utf-8")
    command = ModuleTestCommand(CLIConfig.from_project_root(tmp_path))
    command.console = _quiet_console()

    ok, message = command.run_inline_tests("01_tensor", "01")

    assert not ok
    assert "tito module start 01" in message
