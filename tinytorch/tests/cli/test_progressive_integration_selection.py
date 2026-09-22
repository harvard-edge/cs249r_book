"""Integration selection must work before future modules have been built."""
from pathlib import Path
import subprocess
import sys

from tito.commands.module.test import ModuleTestCommand
from tito.core.config import CLIConfig

ROOT = Path(__file__).resolve().parents[2]


def test_module12_integration_runs_without_transformers(monkeypatch):
    """Run the actual selected tests with Module 13 imports made unavailable."""
    original_run = subprocess.run
    selected = []

    def run_without_transformers(args, **kwargs):
        selected.extend(args[3:])
        # A fresh interpreter reproduces the student boundary even when this
        # outer pytest process already imported the complete reference package.
        bootstrap = '''
import importlib.abc
import sys
class BeforeTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tinytorch.core.transformers":
            raise ModuleNotFoundError("Module 13 has not been built", name=fullname)
sys.meta_path.insert(0, BeforeTransformers())
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
'''
        return original_run([sys.executable, "-c", bootstrap, *args[3:]], **kwargs)

    monkeypatch.setattr(subprocess, "run", run_without_transformers)
    command = ModuleTestCommand(CLIConfig.from_project_root(ROOT))
    passed, output = command.run_integration_tests("12")
    assert passed, output
    assert any(path.endswith("test_nlp_pipeline_flow.py") for path in selected)
    assert not any(path.endswith("test_transformer_pipeline_flow.py") for path in selected)


def test_module13_adds_transformer_integration(monkeypatch):
    selected = []

    def capture(args, **kwargs):
        selected.extend(args)
        return subprocess.CompletedProcess(args, 0, "passed", "")

    monkeypatch.setattr(subprocess, "run", capture)
    command = ModuleTestCommand(CLIConfig.from_project_root(ROOT))
    passed, _ = command.run_integration_tests("13")
    assert passed
    assert any(path.endswith("test_nlp_pipeline_flow.py") for path in selected)
    assert any(path.endswith("test_transformer_pipeline_flow.py") for path in selected)


def test_inline_command_rejects_empty_notebook(tmp_path):
    import json

    source = tmp_path / "src" / "01_tensor" / "01_tensor.py"
    source.parent.mkdir(parents=True)
    source.write_text("#| default_exp core.tensor\ndef test_unit_tensor():\n    pass\n"
                      "def test_module():\n    test_unit_tensor()\n", encoding="utf-8")
    notebook = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook.parent.mkdir(parents=True)
    notebook.write_text(json.dumps({"cells": []}), encoding="utf-8")
    command = ModuleTestCommand(CLIConfig.from_project_root(tmp_path))
    passed, output = command.run_inline_tests("01_tensor", "01")
    assert not passed
    assert "Cannot certify" in output


def test_inline_command_rejects_tests_that_are_never_called(tmp_path):
    import json

    code = ("#| default_exp core.tensor\ndef test_unit_tensor():\n    pass\n"
            "def test_module():\n    test_unit_tensor()\n")
    source = tmp_path / "src" / "01_tensor" / "01_tensor.py"
    source.parent.mkdir(parents=True)
    source.write_text(code, encoding="utf-8")
    notebook = tmp_path / "modules" / "01_tensor" / "tensor.ipynb"
    notebook.parent.mkdir(parents=True)
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": code}]}), encoding="utf-8")
    command = ModuleTestCommand(CLIConfig.from_project_root(tmp_path))
    passed, output = command.run_inline_tests("01_tensor", "01")
    assert not passed
    assert "Required inline tests did not run" in output
