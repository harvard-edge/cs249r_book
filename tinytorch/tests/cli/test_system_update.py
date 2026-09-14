"""
Unit and regression tests for TinyTorch UpdateCommand.
"""

import json
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tito.commands.system.update import UpdateCommand
from tito.core.config import CLIConfig


@pytest.fixture
def temp_project(tmp_path):
    """Create a minimal project structure for testing updates."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Create directories
    (project_root / "src").mkdir()
    (project_root / "tito").mkdir()
    (project_root / "modules").mkdir()
    (project_root / "tinytorch" / "core").mkdir(parents=True)
    (project_root / ".tito").mkdir()

    # Create sample files
    (project_root / "pyproject.toml").write_text('version = "0.1.4"', encoding="utf-8")
    (project_root / "tinytorch" / "__init__.py").write_text('__version__ = "0.1.4"', encoding="utf-8")
    (project_root / "tinytorch" / "core" / "__init__.py").write_text("# core init", encoding="utf-8")
    (project_root / "tinytorch" / "core" / "tensor.py").write_text("# student tensor", encoding="utf-8")
    (project_root / "modules" / "01_tensor.ipynb").write_text("student notebook", encoding="utf-8")
    (project_root / ".tito" / "progress.json").write_text(
        json.dumps({"completed_modules": ["01"]}), encoding="utf-8"
    )

    return project_root


class TestUpdateCommand:
    def test_parse_version_tuple(self):
        """Test semver numeric tuple parsing."""
        assert UpdateCommand._parse_version_tuple("0.1.13") == (0, 1, 13)
        assert UpdateCommand._parse_version_tuple("0.1.2") == (0, 1, 2)
        assert UpdateCommand._parse_version_tuple("0.1.13-rc1") == (0, 1, 13, 1)
        assert UpdateCommand._parse_version_tuple("unknown") == (0,)

    def test_compare_versions(self, temp_project):
        """Test version comparison logic."""
        cmd = UpdateCommand(CLIConfig.from_project_root(temp_project))

        # 0.1.13 > 0.1.2 (standard string comparison would fail this!)
        assert cmd._compare_versions("0.1.2", "0.1.13") == -1
        assert cmd._compare_versions("0.1.13", "0.1.2") == 1
        assert cmd._compare_versions("0.1.13", "0.1.13") == 0
        assert cmd._compare_versions("0.1.9", "0.1.13") == -1

    def test_extract_best_tag_semver_sorted(self, temp_project):
        """Test extracting the semver-highest tag from arbitrary tag list."""
        cmd = UpdateCommand(CLIConfig.from_project_root(temp_project))

        mock_tags = [
            {"name": "vol1-v0.7.0"},
            {"name": "vol2-v0.2.1"},
            {"name": "tinytorch-v0.1.2"},
            {"name": "tinytorch-v0.1.13"},
            {"name": "tinytorch-v0.1.9"},
            {"name": "tinytorch-slides-v0.1.0"},
        ]

        version, tag_name = cmd._extract_best_tag(mock_tags)
        assert version == "0.1.13"
        assert tag_name == "tinytorch-v0.1.13"

    def test_extract_best_tag_empty(self, temp_project):
        """Test behavior when no matching tags exist."""
        cmd = UpdateCommand(CLIConfig.from_project_root(temp_project))
        mock_tags = [{"name": "vol1-v0.7.0"}, {"name": "other-tag"}]
        version, tag_name = cmd._extract_best_tag(mock_tags)
        assert version is None
        assert tag_name is None

    def test_create_backup(self, temp_project):
        """Test backup snapshot creation preserves student work."""
        cmd = UpdateCommand(CLIConfig.from_project_root(temp_project))
        backup_path = cmd._create_backup()

        assert backup_path is not None
        assert backup_path.exists()
        assert (backup_path / "progress.json").exists()
        assert (backup_path / "modules" / "01_tensor.ipynb").exists()
        assert (backup_path / "core" / "tensor.py").exists()

    def test_update_directory_safe_replacement(self, temp_project, tmp_path):
        """Test that directory updating safely swaps contents and handles tito in-place."""
        cmd = UpdateCommand(CLIConfig.from_project_root(temp_project))

        # Test tito directory in-place update
        src_tito = tmp_path / "new_tito"
        src_tito.mkdir()
        (src_tito / "main.py").write_text("# new main", encoding="utf-8")
        dst_tito = temp_project / "tito"

        assert cmd._update_directory(src_tito, dst_tito, "tito") is True
        assert (dst_tito / "main.py").read_text(encoding="utf-8") == "# new main"

        # Test normal directory update with staging
        src_src = tmp_path / "new_src"
        src_src.mkdir()
        (src_src / "01_tensor.py").write_text("# new src tensor", encoding="utf-8")
        dst_src = temp_project / "src"

        assert cmd._update_directory(src_src, dst_src, "src") is True
        assert (dst_src / "01_tensor.py").read_text(encoding="utf-8") == "# new src tensor"

    def test_update_directory_rollback_on_failure(self, temp_project, tmp_path):
        """Test that failing to update a directory rolls back to original state."""
        cmd = UpdateCommand(CLIConfig.from_project_root(temp_project))
        dst_src = temp_project / "src"
        (dst_src / "original.py").write_text("# original", encoding="utf-8")

        src_broken = tmp_path / "broken_src"
        src_broken.mkdir()

        # Mock copytree to raise an exception
        with patch("shutil.copytree", side_effect=OSError("Disk write error")):
            assert cmd._update_directory(src_broken, dst_src, "src") is False

        # Verify original was restored
        assert dst_src.exists()
        assert (dst_src / "original.py").read_text(encoding="utf-8") == "# original"

    def test_update_tinytorch_package_preserves_student_core(self, temp_project, tmp_path):
        """Test update tinytorch package preserves student core implementations."""
        cmd = UpdateCommand(CLIConfig.from_project_root(temp_project))

        src_pkg = tmp_path / "new_tinytorch"
        src_pkg.mkdir()
        (src_pkg / "__init__.py").write_text("__version__ = '0.1.13'", encoding="utf-8")
        (src_pkg / "core").mkdir()
        (src_pkg / "core" / "__init__.py").write_text("# new core init", encoding="utf-8")
        (src_pkg / "core" / "tensor.py").write_text("# UPSTREAM TENSOR", encoding="utf-8")

        dst_pkg = temp_project / "tinytorch"

        assert cmd._update_tinytorch_package(src_pkg, dst_pkg) is True
        # Version and core __init__ updated
        assert (dst_pkg / "__init__.py").read_text(encoding="utf-8") == "__version__ = '0.1.13'"
        assert (dst_pkg / "core" / "__init__.py").read_text(encoding="utf-8") == "# new core init"
        # Student tensor preserved!
        assert (dst_pkg / "core" / "tensor.py").read_text(encoding="utf-8") == "# student tensor"
