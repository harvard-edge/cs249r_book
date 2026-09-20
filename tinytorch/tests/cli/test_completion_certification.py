"""Completion must certify the notebook and a freshly generated export."""
import io
from pathlib import Path

import nbformat
import pytest
from rich.console import Console

from tito.commands.module.workflow import ModuleWorkflowCommand
from tito.core.config import CLIConfig


@pytest.fixture
def curriculum(tmp_path, monkeypatch):
    source = tmp_path / 'src/01_demo/01_demo.py'
    source.parent.mkdir(parents=True)
    source.write_text(
        '#| default_exp core.demo\n'
        'def test_unit_demo():\n    assert True\n'
        'def test_module():\n    test_unit_demo()\n'
    )
    notebook = tmp_path / 'modules/01_demo/demo.ipynb'
    notebook.parent.mkdir(parents=True)
    target = tmp_path / 'tinytorch/core/demo.py'
    target.parent.mkdir(parents=True)
    target.write_text('value = "stale reference"\n')
    # nbdev looks for its project settings relative to the working directory.
    (tmp_path / 'pyproject.toml').write_text('[tool.nbdev]\nlib_name="tinytorch"\n')
    monkeypatch.chdir(tmp_path)
    command = ModuleWorkflowCommand(CLIConfig.from_project_root(tmp_path))
    command.console = Console(file=io.StringIO())
    return command, notebook, target


def write_notebook(path, *, target='core.demo', export=True, run=True, tests=True):
    cells = [nbformat.v4.new_code_cell(f'#| default_exp {target}')]
    if export:
        cells.append(nbformat.v4.new_code_cell('#| export\nvalue = "student implementation"'))
    if tests:
        cells.append(nbformat.v4.new_code_cell(
            'def test_unit_demo():\n    assert True\n'
            'def test_module():\n    test_unit_demo()\n'
        ))
    if run:
        cells.append(nbformat.v4.new_code_cell('if __name__ == "__main__":\n    test_module()'))
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def test_empty_notebook_cannot_complete_using_stale_export(curriculum):
    command, notebook, target = curriculum
    nbformat.write(nbformat.v4.new_notebook(), notebook)
    before = target.read_bytes()
    assert command._complete_module_quiet('01', '01_demo', False, False) == 1
    assert target.read_bytes() == before
    assert command.get_progress_data()['completed_modules'] == []


@pytest.mark.parametrize('target_name', ['core.wrong', ''])
def test_wrong_or_missing_notebook_identity_preserves_existing_export(curriculum, target_name):
    command, notebook, target = curriculum
    write_notebook(notebook, target=target_name)
    before = target.read_bytes()
    assert command.export_module('01_demo') == 1
    assert target.read_bytes() == before


def test_missing_locked_tests_is_rejected_even_when_tests_skipped(curriculum):
    command, notebook, target = curriculum
    write_notebook(notebook, tests=False, run=False)
    assert command._complete_module_quiet('01', '01_demo', True, False) == 1
    assert 'missing required tests' in command.console.file.getvalue()


def test_defined_tests_must_actually_execute(curriculum):
    command, notebook, target = curriculum
    write_notebook(notebook, run=False)
    result = command._run_inline_unit_tests('01_demo', False)
    assert result['failed'] > 0
    assert 'Required inline tests did not run' in str(result)


def test_noop_export_cannot_reuse_stale_package(curriculum):
    command, notebook, target = curriculum
    write_notebook(notebook, export=False)
    before = target.read_bytes()
    assert command.export_module('01_demo') == 1
    assert target.read_bytes() == before
    assert 'No fresh' in command.console.file.getvalue()


def test_valid_notebook_executes_tests_and_replaces_export(curriculum):
    command, notebook, target = curriculum
    write_notebook(notebook)
    assert command._run_inline_unit_tests('01_demo', False)['failed'] == 0
    assert command._complete_module_quiet('01', '01_demo', False, False) == 0
    assert command.get_progress_data()['completed_modules'] == ['01']
    assert 'student implementation' in target.read_text()
    assert 'stale reference' not in target.read_text()


def test_partial_export_exception_does_not_replace_existing_package(curriculum, monkeypatch):
    command, notebook, target = curriculum
    write_notebook(notebook)
    before = target.read_bytes()
    def fail_export(notebook_path, lib_path):
        produced = Path(lib_path) / 'core/demo.py'
        produced.parent.mkdir(parents=True)
        produced.write_text('value = "incomplete"\n')
        raise RuntimeError('export interrupted')
    monkeypatch.setattr('nbdev.export.nb_export', fail_export)
    assert command.export_module('01_demo') == 1
    assert target.read_bytes() == before


def test_timed_out_notebook_cannot_complete_or_replace_export(curriculum, monkeypatch):
    import subprocess
    command, notebook, target = curriculum
    write_notebook(notebook)
    before = target.read_bytes()
    def timeout_runner(args, **kwargs):
        assert kwargs["timeout"] == 300
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])
    monkeypatch.setattr('tito.commands.module.workflow.subprocess.run', timeout_runner)
    assert command._complete_module_quiet('01', '01_demo', False, False) == 1
    assert target.read_bytes() == before
    assert command.get_progress_data()['completed_modules'] == []
    result = command._run_inline_unit_tests('01_demo', False)
    assert result['failed'] == 1
    assert 'timed out after 300 seconds' in str(result)
