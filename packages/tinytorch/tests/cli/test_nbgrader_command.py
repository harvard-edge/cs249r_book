import argparse
import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from tito.commands.nbgrader import NBGraderCommand
from tito.core.config import CLIConfig


def make_config(project_root: Path) -> CLIConfig:
    return CLIConfig.from_project_root(project_root)


def make_module(project_root: Path, module_name: str = "01_tensor") -> Path:
    short_name = module_name.split("_", 1)[1]
    source_dir = project_root / "src" / module_name
    notebook_dir = project_root / "modules" / module_name
    source_dir.mkdir(parents=True)
    notebook_dir.mkdir(parents=True)
    (source_dir / f"{module_name}.py").write_text("# source of truth\n", encoding="utf-8")

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "nbgrader": {
                        "grade": False,
                        "grade_id": "imports",
                        "solution": True,
                    }
                },
                "outputs": [],
                "source": "# setup\n",
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "nbgrader": {
                        "grade": False,
                        "grade_id": "tensor-class",
                        "solution": True,
                    }
                },
                "outputs": [],
                "source": (
                    "def add_one(x):\n"
                    "    ### BEGIN SOLUTION\n"
                    "    return x + 1\n"
                    "    ### END SOLUTION\n"
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "nbgrader": {
                        "grade": True,
                        "grade_id": "test-basic",
                        "locked": True,
                        "points": 5,
                    }
                },
                "outputs": [],
                "source": "def test_basic():\n    assert True\n",
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = notebook_dir / f"{short_name}.ipynb"
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    return notebook_path


def test_generate_stages_current_notebook_with_normalized_metadata(tmp_path):
    make_module(tmp_path)
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 0
    staged = tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb"
    assert staged.exists()

    notebook = json.loads(staged.read_text(encoding="utf-8"))
    setup_cell = notebook["cells"][0]["metadata"]["nbgrader"]
    solution_cell = notebook["cells"][1]["metadata"]["nbgrader"]
    graded_cell = notebook["cells"][2]["metadata"]["nbgrader"]

    assert setup_cell["schema_version"] == 3
    assert setup_cell["solution"] is False
    assert setup_cell["locked"] is True
    assert solution_cell["schema_version"] == 3
    assert solution_cell["solution"] is True
    assert solution_cell["locked"] is False
    assert graded_cell["schema_version"] == 3
    assert graded_cell["locked"] is True
    assert graded_cell["solution"] is False


def test_generate_resolves_module_suffix(tmp_path):
    make_module(tmp_path)
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="tensor"))

    assert result == 0
    assert (tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb").exists()


def test_generate_tier_option_is_hidden_but_parseable(tmp_path, capsys):
    command = NBGraderCommand(make_config(tmp_path))
    parser = argparse.ArgumentParser()
    command.add_arguments(parser)

    args = parser.parse_args(["generate", "--all", "--tier", "challenge"])
    assert args.tier == "challenge"

    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--help"])
    help_text = capsys.readouterr().out
    assert "--tier" not in help_text


def test_generate_fails_when_notebook_has_no_nbgrader_metadata(tmp_path):
    make_module(tmp_path)
    bad_notebook = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    bad_notebook.write_text(
        json.dumps({
            "cells": [{"cell_type": "code", "metadata": {}, "source": "def test_basic(): pass\n"}],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
        encoding="utf-8",
    )
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 1
    assert not (tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb").exists()


def test_generate_keeps_visible_support_tests_ungraded(tmp_path):
    make_module(tmp_path)
    notebook_path = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"].insert(
        2,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "nbgrader": {
                    "grade": False,
                    "grade_id": "visible-test",
                    "solution": True,
                }
            },
            "outputs": [],
            "source": "def test_visible_example():\n    assert True\n",
        },
    )
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 0
    staged = tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb"
    staged_notebook = json.loads(staged.read_text(encoding="utf-8"))
    support_test = staged_notebook["cells"][2]["metadata"]["nbgrader"]
    assert support_test["grade"] is False
    assert support_test["solution"] is False
    assert support_test["locked"] is True


def test_generate_completes_schema_for_non_solution_cells(tmp_path):
    make_module(tmp_path)
    notebook_path = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"].insert(
        2,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "nbgrader": {
                    "grade": False,
                    "grade_id": "provided-helper",
                    "solution": False,
                }
            },
            "outputs": [],
            "source": "def helper():\n    return 1\n",
        },
    )
    notebook["cells"].insert(
        3,
        {
            "cell_type": "markdown",
            "metadata": {
                "nbgrader": {
                    "grade": False,
                    "grade_id": "editable-reflection",
                    "locked": False,
                }
            },
            "source": "Write your answer here.\n",
        },
    )
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 0
    staged = tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb"
    staged_notebook = json.loads(staged.read_text(encoding="utf-8"))
    provided_helper = staged_notebook["cells"][2]["metadata"]["nbgrader"]
    editable_reflection = staged_notebook["cells"][3]["metadata"]["nbgrader"]
    assert provided_helper["grade"] is False
    assert provided_helper["solution"] is False
    assert provided_helper["locked"] is True
    assert editable_reflection["grade"] is False
    assert editable_reflection["solution"] is False
    assert editable_reflection["locked"] is False


def test_generate_keeps_ungraded_markdown_reflections_editable(tmp_path):
    make_module(tmp_path)
    notebook_path = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"].insert(
        2,
        {
            "cell_type": "markdown",
            "metadata": {
                "nbgrader": {
                    "grade": False,
                    "grade_id": "reflection",
                    "solution": True,
                }
            },
            "source": (
                "Question\n\n"
                "### BEGIN SOLUTION\n"
                "Instructor answer.\n"
                "### END SOLUTION\n"
            ),
        },
    )
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 0
    staged = tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb"
    staged_notebook = json.loads(staged.read_text(encoding="utf-8"))
    reflection = staged_notebook["cells"][2]["metadata"]["nbgrader"]
    assert reflection["grade"] is False
    assert reflection["solution"] is False
    assert reflection["locked"] is False


def test_generate_keeps_scaffold_role_in_student_release(tmp_path):
    make_module(tmp_path)
    notebook_path = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][1]["source"] = (
        "def provided_pattern(x):\n"
        "    ### BEGIN SOLUTION role=scaffold\n"
        "    return x + 2\n"
        "    ### END SOLUTION\n"
    )
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 0
    staged = tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb"
    staged_notebook = json.loads(staged.read_text(encoding="utf-8"))
    source = staged_notebook["cells"][1]["source"]
    solution_cell = staged_notebook["cells"][1]["metadata"]["nbgrader"]
    assert "return x + 2" in source
    assert "BEGIN SOLUTION" not in source
    assert solution_cell["solution"] is False
    assert solution_cell["locked"] is True


def test_generate_challenge_tier_keeps_core_baseline(tmp_path):
    make_module(tmp_path)
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01", tier="challenge"))

    assert result == 0
    staged = tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb"
    staged_notebook = json.loads(staged.read_text(encoding="utf-8"))
    source = staged_notebook["cells"][1]["source"]
    solution_cell = staged_notebook["cells"][1]["metadata"]["nbgrader"]
    assert "return x + 1" in source
    assert "BEGIN SOLUTION" not in source
    assert solution_cell["solution"] is False
    assert solution_cell["locked"] is True


def test_generate_challenge_tier_strips_challenge_role(tmp_path):
    make_module(tmp_path)
    notebook_path = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][1]["source"] = (
        "def optimize(x):\n"
        "    ### BEGIN SOLUTION role=challenge\n"
        "    return x + 3\n"
        "    ### END SOLUTION\n"
    )
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01", tier="challenge"))

    assert result == 0
    staged = tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb"
    staged_notebook = json.loads(staged.read_text(encoding="utf-8"))
    source = staged_notebook["cells"][1]["source"]
    solution_cell = staged_notebook["cells"][1]["metadata"]["nbgrader"]
    assert "return x + 3" in source
    assert "BEGIN SOLUTION role=challenge" in source
    assert solution_cell["solution"] is True
    assert solution_cell["locked"] is False


def test_generate_removes_instructor_only_regions_for_student(tmp_path):
    make_module(tmp_path)
    notebook_path = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][1]["source"] = (
        "def hidden_note(x):\n"
        "    ### BEGIN SOLUTION role=instructor\n"
        "    return 'do not release'\n"
        "    ### END SOLUTION\n"
        "    return x\n"
    )
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 0
    staged = tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb"
    staged_notebook = json.loads(staged.read_text(encoding="utf-8"))
    source = staged_notebook["cells"][1]["source"]
    solution_cell = staged_notebook["cells"][1]["metadata"]["nbgrader"]
    assert "do not release" not in source
    assert "BEGIN SOLUTION" not in source
    assert "return x" in source
    assert solution_cell["solution"] is False


def test_generate_instructor_tier_keeps_instructor_regions(tmp_path):
    make_module(tmp_path)
    notebook_path = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][1]["source"] = (
        "def hidden_note(x):\n"
        "    ### BEGIN SOLUTION role=instructor\n"
        "    return 'reference answer'\n"
        "    ### END SOLUTION\n"
    )
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01", tier="instructor"))

    assert result == 0
    staged = tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb"
    staged_notebook = json.loads(staged.read_text(encoding="utf-8"))
    source = staged_notebook["cells"][1]["source"]
    solution_cell = staged_notebook["cells"][1]["metadata"]["nbgrader"]
    assert "reference answer" in source
    assert "BEGIN SOLUTION" not in source
    assert solution_cell["solution"] is False
    assert solution_cell["locked"] is True


def test_generate_fails_on_unknown_solution_role(tmp_path):
    make_module(tmp_path)
    notebook_path = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][1]["source"] = (
        "def bad_role(x):\n"
        "    ### BEGIN SOLUTION role=surprise\n"
        "    return x\n"
        "    ### END SOLUTION\n"
    )
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 1
    assert not (tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb").exists()


def test_generate_fails_on_mismatched_solution_markers(tmp_path):
    make_module(tmp_path)
    notebook_path = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][1]["source"] = "def broken(x):\n    ### BEGIN SOLUTION\n    return x\n"
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    command = NBGraderCommand(make_config(tmp_path))

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 1
    assert not (tmp_path / "assignments" / "source" / "01_tensor" / "tensor.ipynb").exists()


def test_generate_refreshes_notebook_when_source_is_newer(tmp_path, monkeypatch):
    make_module(tmp_path)
    source = tmp_path / "src" / "01_tensor" / "01_tensor.py"
    notebook = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    os.utime(source, (notebook.stat().st_mtime + 10, notebook.stat().st_mtime + 10))
    command = NBGraderCommand(make_config(tmp_path))
    calls = []

    def fake_run(cmd, *, capture_output=False):
        calls.append(cmd)
        output_path = Path(cmd[-1])
        output_path.write_text(notebook.read_text(encoding="utf-8"), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(command, "_run_external", fake_run)

    result = command._generate(Namespace(all=False, module_range=None, module="01"))

    assert result == 0
    assert calls
    assert calls[0][:3] == ["jupytext", "--to", "ipynb"]


def test_init_creates_default_config_and_runs_nbgrader_db_upgrade(tmp_path, monkeypatch):
    command = NBGraderCommand(make_config(tmp_path))
    calls = []

    def fake_run(cmd, *, capture_output=False):
        calls.append((cmd, capture_output))
        return subprocess.CompletedProcess(cmd, 0, "nbgrader 0.9.5\n", "")

    monkeypatch.setattr(command, "_run_external", fake_run)

    result = command._init()

    assert result == 0
    assert (tmp_path / "assignments" / "source").is_dir()
    config_text = (tmp_path / "nbgrader_config.py").read_text(encoding="utf-8")
    assert 'c.CourseDirectory.root = "assignments"' in config_text
    assert "c.CourseDirectory.db_url" not in config_text
    assert "c.ClearSolutions.enforce_metadata = False" in config_text
    assert calls == [
        (["nbgrader", "--version"], True),
        (["nbgrader", "db", "upgrade"], True),
    ]


def test_release_invokes_nbgrader_generate_assignment_with_course_dir(tmp_path, monkeypatch):
    make_module(tmp_path)
    command = NBGraderCommand(make_config(tmp_path))
    calls = []

    def fake_run(cmd, *, capture_output=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(command, "_run_external", fake_run)

    result = command._release(Namespace(all=False, assignment="01"))

    assert result == 0
    assert calls == [[
        "nbgrader",
        "generate_assignment",
        "01_tensor",
        "--course-dir",
        str(tmp_path / "assignments"),
    ]]


def test_autograde_passes_student_and_force_filters(tmp_path, monkeypatch):
    make_module(tmp_path)
    command = NBGraderCommand(make_config(tmp_path))
    calls = []

    def fake_run(cmd, *, capture_output=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(command, "_run_external", fake_run)

    result = command._autograde(Namespace(all=False, assignment="01_tensor", student="student_a", force=True))

    assert result == 0
    assert calls == [[
        "nbgrader",
        "autograde",
        "01_tensor",
        "--course-dir",
        str(tmp_path / "assignments"),
        "--student",
        "student_a",
        "--force",
    ]]


def test_report_accepts_module_alias_for_assignment_filter(tmp_path, monkeypatch):
    make_module(tmp_path)
    command = NBGraderCommand(make_config(tmp_path))
    parser = argparse.ArgumentParser()
    command.add_arguments(parser)
    args = parser.parse_args(["report", "--module", "01", "--student", "student_a"])
    calls = []

    def fake_run(cmd, *, capture_output=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(command, "_run_external", fake_run)

    result = command._report(args)

    assert result == 0
    # `nbgrader export` has no --course-dir alias (unlike generate_assignment /
    # collect / autograde / feedback); it only honours CourseDirectory.root.
    assert calls == [[
        "nbgrader",
        "export",
        f"--CourseDirectory.root={tmp_path / 'assignments'}",
        "--assignment",
        "01_tensor",
        "--student",
        "student_a",
    ]]
