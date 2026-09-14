"""Reference notebook inventory and process isolation regressions (2026-09-11)."""
import importlib.util
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tinytorch_release_gates", ROOT / "tools/release_check.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.fixture
def curriculum(tmp_path, monkeypatch):
    modules = tmp_path / "modules"
    monkeypatch.setattr(release, "MODULES", modules)
    monkeypatch.setattr(release, "module_files", lambda: [(1, "01_first", None), (2, "02_second", None)])
    return modules


def write_notebook(modules, name, export, checks):
    path = modules / name / f"{name.split('_', 1)[1]}.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(export), nbformat.v4.new_code_cell(checks)])
    nbformat.write(nb, path)
    return path


def test_journey_rejects_empty_inventory(curriculum):
    assert len(release.g_notebook_set()) == 2
    assert release.g_journey() == release.g_notebook_set()


def test_inventory_rejects_extra_notebook(curriculum):
    for name in ["01_first", "02_second"]:
        write_notebook(curriculum, name, "", "")
    write_notebook(curriculum, "03_extra", "", "")
    assert release.g_notebook_set() == ["unexpected notebook: 03_extra/extra.ipynb"]


def test_journey_exports_only_predecessors_and_isolates_interpreters(curriculum):
    write_notebook(curriculum, "01_first", "#| default_exp core.first\n#| export\nvalue = 7",
                   "import builtins, importlib.util\nbuiltins.reference_probe = True\n"
                   "assert importlib.util.find_spec('tinytorch.core.second') is None")
    write_notebook(curriculum, "02_second", "#| default_exp core.second\n#| export\n"
                   "from tinytorch.core.first import value\nanswer = value + 1",
                   "import builtins\nassert not hasattr(builtins, 'reference_probe')\nassert answer == 8")
    assert release.g_journey() == []


def test_journey_surfaces_notebook_assertions(curriculum):
    write_notebook(curriculum, "01_first", "#| default_exp core.first\n#| export\nvalue = 7",
                   "raise AssertionError('broken reference')")
    write_notebook(curriculum, "02_second", "", "")
    errors = release.g_journey()
    assert len(errors) == 1
    assert "01_first" in errors[0] and "broken reference" in errors[0]


def test_reference_check_nonzero_exit_without_output_is_a_failure(monkeypatch):
    import subprocess

    monkeypatch.setattr(release.subprocess, "run",
                        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", ""))
    assert release.g_reference_regressions() == ["reference check exited 1"]
