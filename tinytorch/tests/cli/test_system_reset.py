"""
Tests for SystemResetCommand progress and workspace resetting.
"""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from tito.commands.system.reset import SystemResetCommand
from tito.core.config import CLIConfig


@pytest.fixture
def reset_project(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Modules
    mod_dir = project_root / "modules" / "01_tensor"
    mod_dir.mkdir(parents=True)
    (mod_dir / "tensor.ipynb").write_text("student notebook", encoding="utf-8")

    # Core
    core_dir = project_root / "tinytorch" / "core"
    core_dir.mkdir(parents=True)
    (core_dir / "__init__.py").write_text("# init", encoding="utf-8")
    (core_dir / "tensor.py").write_text("# tensor impl", encoding="utf-8")

    # Pycache
    pycache_dir = core_dir / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "tensor.cpython-312.pyc").write_bytes(b"dummy")

    # .tito progress & milestones
    tito_dir = project_root / ".tito"
    tito_dir.mkdir()
    (tito_dir / "progress.json").write_text(json.dumps({"completed_modules": ["01"]}), encoding="utf-8")
    (tito_dir / "milestones.json").write_text(json.dumps({"completed_milestones": ["01"]}), encoding="utf-8")

    return project_root


class TestSystemReset:
    def test_reset_force_clears_progress_and_core(self, reset_project):
        """Test system reset clears .tito progress and core while keeping core __init__.py."""
        cmd = SystemResetCommand(CLIConfig.from_project_root(reset_project))
        args = Namespace(force=True, keep_progress=False, ci=True)

        exit_code = cmd.run(args)
        assert exit_code == 0

        # modules cleared
        assert not (reset_project / "modules" / "01_tensor").exists()

        # tinytorch/core cleaned
        assert (reset_project / "tinytorch" / "core" / "__init__.py").exists()
        assert not (reset_project / "tinytorch" / "core" / "tensor.py").exists()
        assert not (reset_project / "tinytorch" / "core" / "__pycache__").exists()

        # .tito progress cleared!
        assert not (reset_project / ".tito" / "progress.json").exists()
        assert not (reset_project / ".tito" / "milestones.json").exists()

    def test_reset_keep_progress_preserves_tito_state(self, reset_project):
        """Test --keep-progress preserves .tito/progress.json."""
        cmd = SystemResetCommand(CLIConfig.from_project_root(reset_project))
        args = Namespace(force=True, keep_progress=True, ci=True)

        exit_code = cmd.run(args)
        assert exit_code == 0

        # modules and core cleared
        assert not (reset_project / "modules" / "01_tensor").exists()
        assert not (reset_project / "tinytorch" / "core" / "tensor.py").exists()

        # progress preserved!
        assert (reset_project / ".tito" / "progress.json").exists()
        assert (reset_project / ".tito" / "milestones.json").exists()
