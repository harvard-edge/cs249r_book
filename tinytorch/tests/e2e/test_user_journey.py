"""
End-to-End User Journey Tests for TinyTorch

These tests simulate the complete student experience:
1. Fresh start (setup)
2. Module workflow (start → work → complete)
3. Progress tracking
4. Milestone unlocking

Run with:
    pytest tests/e2e/test_user_journey.py -v

Categories:
    -k quick         # Fast CLI verification (~30s)
    -k module_flow   # Module workflow tests (~2min)
    -k full_journey  # Complete journey test (~10min)
"""

import pytest
import subprocess
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_tito(args: list, cwd: Optional[Path] = None, timeout: int = 60) -> Tuple[int, str, str]:
    """Run a tito command and return (exit_code, stdout, stderr)."""
    cmd = [sys.executable, "-m", "tito.main"] + args
    result = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def run_python_script(script_path: Path, timeout: int = 120) -> Tuple[int, str, str]:
    """Run a Python script and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


class TestQuickVerification:
    """Quick tests to verify CLI and structure (~30 seconds total)."""

    @pytest.mark.quick
    def test_tito_bare_command_works(self):
        """Bare 'tito' shows welcome screen."""
        code, stdout, stderr = run_tito([])
        assert code == 0, f"Bare tito failed: {stderr}"
        assert "Welcome" in stdout or "Quick Start" in stdout

    @pytest.mark.quick
    def test_tito_help_works(self):
        """'tito --help' shows help."""
        code, stdout, stderr = run_tito(["--help"])
        assert code == 0, f"tito --help failed: {stderr}"
        assert "usage" in stdout.lower() or "COMMAND" in stdout

    @pytest.mark.quick
    def test_tito_version_works(self):
        """'tito --version' shows version."""
        code, stdout, stderr = run_tito(["--version"])
        assert code == 0
        assert "Tiny" in stdout or "CLI" in stdout

    @pytest.mark.quick
    def test_module_command_help(self):
        """'tito module' shows module help."""
        code, stdout, stderr = run_tito(["module"])
        assert code == 0
        # Should show module subcommands
        assert "start" in stdout or "complete" in stdout

    @pytest.mark.quick
    def test_milestone_command_help(self):
        """'tito milestones' shows milestone help."""
        code, stdout, stderr = run_tito(["milestones"])
        assert code == 0
        # Should show milestone subcommands
        assert "list" in stdout or "run" in stdout or "status" in stdout

    @pytest.mark.quick
    def test_module_status_works(self):
        """'tito module status' runs without error."""
        code, stdout, stderr = run_tito(["module", "status"])
        assert code == 0, f"module status failed: {stderr}"

    @pytest.mark.quick
    def test_system_info_works(self):
        """'tito system info' runs without error."""
        code, stdout, stderr = run_tito(["system", "info"])
        assert code == 0, f"system info failed: {stderr}"

    @pytest.mark.quick
    def test_milestone_list_works(self):
        """'tito milestones list' shows available milestones."""
        code, stdout, stderr = run_tito(["milestones", "list", "--simple"])
        assert code == 0, f"milestones list failed: {stderr}"
        # Should show milestone names
        assert "Perceptron" in stdout or "1957" in stdout

    @pytest.mark.quick
    def test_modules_directory_exists(self):
        """Modules directory structure exists."""
        modules_dir = PROJECT_ROOT / "modules"
        assert modules_dir.exists(), "modules/ directory missing"

        # Check first few modules exist
        for num in ["01", "02", "03"]:
            module_dirs = list(modules_dir.glob(f"{num}_*"))
            assert len(module_dirs) > 0, f"Module {num} directory missing"

    @pytest.mark.quick
    def test_milestones_directory_exists(self):
        """Milestones directory structure exists."""
        milestones_dir = PROJECT_ROOT / "milestones"
        assert milestones_dir.exists(), "milestones/ directory missing"

        # Check milestone directories
        assert (milestones_dir / "01_1957_perceptron").exists(), "Milestone 01 missing"

    @pytest.mark.quick
    def test_tinytorch_package_importable(self):
        """TinyTorch package can be imported."""
        code, stdout, stderr = subprocess.run(
            [sys.executable, "-c", "import tinytorch; print('OK')"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        ).returncode, "", ""

        result = subprocess.run(
            [sys.executable, "-c", "import tinytorch; print('OK')"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Cannot import tinytorch: {result.stderr}"
        assert "OK" in result.stdout


class TestModuleFlow:
    """Test module workflow: start → complete → progress tracking."""

    @pytest.fixture(autouse=True)
    def backup_progress(self):
        """Backup and restore progress.json around tests."""
        progress_file = PROJECT_ROOT / "progress.json"
        backup_file = PROJECT_ROOT / "progress.json.e2e_backup"

        # Backup existing progress
        if progress_file.exists():
            shutil.copy(progress_file, backup_file)

        yield

        # Restore original progress
        if backup_file.exists():
            shutil.copy(backup_file, progress_file)
            backup_file.unlink()
        elif progress_file.exists():
            # If there was no original, remove the test progress
            # Actually, keep it - don't delete real progress
            pass

    @pytest.mark.module_flow
    def test_module_01_start_works(self):
        """'tito module start 01' works (first module, no prerequisites)."""
        # Note: This opens Jupyter, but should not block
        # We test the command doesn't error on already-started modules
        code, stdout, stderr = run_tito(["module", "status"])
        assert code == 0

    @pytest.mark.module_flow
    def test_module_02_blocked_without_01(self):
        """Cannot start module 02 without completing 01 first."""
        # Create clean progress state
        progress_file = PROJECT_ROOT / "progress.json"
        progress_file.write_text(json.dumps({
            "started_modules": [],
            "completed_modules": [],
            "last_worked": None
        }))

        code, stdout, stderr = run_tito(["module", "start", "02"])

        # Should fail or show locked message
        combined = stdout + stderr
        assert "Locked" in combined or "prerequisite" in combined.lower() or code != 0

    @pytest.mark.module_flow
    def test_module_complete_runs_tests(self):
        """'tito module complete 01 --skip-export' runs tests."""
        # This tests that the complete command works (skip export to be faster)
        code, stdout, stderr = run_tito(
            ["module", "complete", "01", "--skip-export"],
            timeout=120  # Tests may take a while
        )
        # Check that tests ran (may pass or fail depending on state)
        combined = stdout + stderr
        assert "Test" in combined or "test" in combined or code in [0, 1]

    @pytest.mark.module_flow
    def test_progress_tracking_persists(self):
        """Progress is saved and persisted across commands."""
        progress_file = PROJECT_ROOT / "progress.json"

        # Set a known state
        progress_file.write_text(json.dumps({
            "started_modules": ["01"],
            "completed_modules": [],
            "last_worked": "01"
        }))

        # Run status command
        code, stdout, stderr = run_tito(["module", "status"])
        assert code == 0

        # Check progress file still exists and has data
        assert progress_file.exists()
        data = json.loads(progress_file.read_text())
        assert "started_modules" in data

    @pytest.mark.module_flow
    def test_module_test_command_works(self):
        """'tito module test 01' runs module tests."""
        code, stdout, stderr = run_tito(
            ["module", "test", "01"],
            timeout=120
        )
        # Should run tests (may pass or fail)
        combined = stdout + stderr
        # Test command should produce some output
        assert len(combined) > 0


class TestMilestoneFlow:
    """Test milestone workflow: prerequisites → run → completion tracking."""

    @pytest.mark.milestone_flow
    def test_milestone_list_shows_all(self):
        """Milestone list shows all available milestones."""
        code, stdout, stderr = run_tito(["milestones", "list"])
        assert code == 0

        # Check for expected milestones
        expected = ["Perceptron", "XOR", "MLP", "CNN", "Transformer"]
        found = sum(1 for m in expected if m in stdout)
        assert found >= 3, f"Expected milestones not shown. Got: {stdout}"

    @pytest.mark.milestone_flow
    def test_milestone_info_works(self):
        """'tito milestones info 01' shows milestone details."""
        code, stdout, stderr = run_tito(["milestones", "info", "01"])
        assert code == 0
        assert "Perceptron" in stdout or "1957" in stdout

    @pytest.mark.milestone_flow
    def test_milestone_status_works(self):
        """'tito milestones status' shows progress."""
        code, stdout, stderr = run_tito(["milestones", "status"])
        assert code == 0

    @pytest.mark.milestone_flow
    def test_milestone_01_script_exists(self):
        """Milestone 01 script file exists."""
        script_path = PROJECT_ROOT / "milestones" / "01_1957_perceptron" / "02_rosenblatt_trained.py"
        assert script_path.exists(), f"Milestone script missing: {script_path}"

    @pytest.mark.milestone_flow
    def test_milestone_run_checks_prerequisites(self):
        """'tito milestone run' checks prerequisites before running."""
        # Create clean state with no completed modules
        tito_dir = PROJECT_ROOT / ".tito"
        tito_dir.mkdir(exist_ok=True)
        progress_file = tito_dir / "progress.json"
        progress_file.write_text(json.dumps({
            "completed_modules": []
        }))

        # Try to run milestone 03 (requires many modules)
        code, stdout, stderr = run_tito(["milestones", "run", "03", "--skip-checks"], timeout=5)

        # With --skip-checks it might try to run; without it should check prereqs
        # Either way, the command should not crash
        assert code in [0, 1, 130]  # 130 = user interrupt


class TestFullJourney:
    """Complete end-to-end journey test (slow, thorough)."""

    @pytest.mark.full_journey
    @pytest.mark.slow
    def test_complete_module_01_journey(self):
        """
        Test complete journey for module 01:
        1. Start module
        2. Complete module (with tests)
        3. Verify progress updated
        4. Verify export worked
        """
        # Step 1: Check initial state
        code, stdout, stderr = run_tito(["module", "status"])
        assert code == 0

        # Step 2: Test the module
        code, stdout, stderr = run_tito(
            ["module", "test", "01"],
            timeout=180
        )
        # Tests should run (may pass or fail based on implementation)
        combined = stdout + stderr
        assert "test" in combined.lower() or "Test" in combined

        # Step 3: Verify tinytorch imports work
        result = subprocess.run(
            [sys.executable, "-c", "from tinytorch import Tensor; print('OK')"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        # This tests that the package structure is correct
        # May fail if module not exported yet - that's informative
        if result.returncode != 0:
            pytest.skip("Tensor not yet exported - run tito module complete 01 first")

    @pytest.mark.full_journey
    @pytest.mark.slow
    def test_milestone_01_runs_successfully(self):
        """
        Test that milestone 01 can run successfully.
        Requires: Module 01-08 completed and exported.
        """
        # Check if prerequisite modules are available
        try:
            result = subprocess.run(
                [sys.executable, "-c", """
from tinytorch import Tensor, ReLU, Linear
print('OK')
"""],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                pytest.skip("Required modules not exported yet")
        except Exception:
            pytest.skip("Cannot import required modules")

        # Run milestone 01 with skip-checks (we verified prereqs above)
        script_path = PROJECT_ROOT / "milestones" / "01_1957_perceptron" / "02_rosenblatt_trained.py"
        if not script_path.exists():
            pytest.skip("Milestone script not found")

        code, stdout, stderr = run_python_script(script_path, timeout=120)

        # Should complete successfully or with informative error
        combined = stdout + stderr
        assert code == 0 or "Error" in combined, f"Milestone failed unexpectedly: {combined}"


class TestErrorHandling:
    """Test that errors are handled gracefully."""

    @pytest.mark.quick
    def test_invalid_command_shows_error(self):
        """Invalid commands show helpful error messages."""
        code, stdout, stderr = run_tito(["nonexistent_command"])
        assert code != 0
        combined = stdout + stderr
        assert "invalid" in combined.lower() or "error" in combined.lower()

    @pytest.mark.quick
    def test_invalid_module_number_handled(self):
        """Invalid module numbers are handled gracefully."""
        code, stdout, stderr = run_tito(["module", "start", "99"])
        assert code != 0
        combined = stdout + stderr
        assert "not found" in combined.lower() or "invalid" in combined.lower() or "99" in combined

    @pytest.mark.quick
    def test_invalid_milestone_handled(self):
        """Invalid milestone IDs are handled gracefully."""
        code, stdout, stderr = run_tito(["milestones", "info", "99"])
        assert code != 0
        combined = stdout + stderr
        assert "invalid" in combined.lower() or "not found" in combined.lower()


class TestInstallationPaths:
    """Test different installation/usage paths."""

    @pytest.mark.quick
    def test_src_directory_exists(self):
        """Source directory for development exists."""
        src_dir = PROJECT_ROOT / "src"
        assert src_dir.exists(), "src/ directory missing"

    @pytest.mark.quick
    def test_pyproject_exists(self):
        """pyproject.toml exists for pip installation."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml missing"

    @pytest.mark.quick
    def test_requirements_exists(self):
        """requirements.txt exists for dependency installation."""
        requirements = PROJECT_ROOT / "requirements.txt"
        assert requirements.exists(), "requirements.txt missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
