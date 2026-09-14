"""
Environment Setup Validation Tests

These tests verify that the TinyTorch environment is correctly configured
and all dependencies work as expected. Run these after `tito setup` to
ensure students can actually use TinyTorch.

Usage:
    pytest tests/environment/test_setup_validation.py -v

    Or via TITO:
    tito system health --verify
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path
import pytest


def _jupyter(*args):
    """Run jupyter from the interpreter running the tests, not from PATH.

    Prefers `python -m jupyter` so it uses the active test interpreter directly
    and avoids rigid or stale shebang headers in venv/bin/ scripts.
    """
    try:
        res = subprocess.run(
            [sys.executable, "-m", "jupyter"] + list(args),
            capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if res.returncode == 0 or "No module named jupyter" not in res.stderr:
            return res
    except Exception:
        pass
    exe = Path(sys.executable).parent / "jupyter"
    cmd = [str(exe)] if exe.exists() else ["jupyter"]
    return subprocess.run(cmd + list(args), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


class TestPythonEnvironment:
    """Verify Python environment is correctly configured."""

    def test_python_version(self):
        """Python version must be 3.10 or higher."""
        assert sys.version_info >= (3, 10), (
            f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}"
        )
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    def test_package_installer_available(self):
        """A package installer must be available: pip or uv.

        This asserted pip specifically until 2026-09-09, which failed on every
        correctly configured machine, because the project's environment is built
        by uv and uv does not install pip into it. What a student needs is an
        installer, not one particular installer.
        """
        pip = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if pip.returncode == 0:
            print(f"✅ pip available: {pip.stdout.strip()}")
            return

        uv = subprocess.run(
            ["uv", "--version"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        assert uv.returncode == 0, (
            "No package installer found. Install one of:\n"
            "  pip:  python -m ensurepip --upgrade\n"
            "  uv:   curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
        print(f"✅ uv available: {uv.stdout.strip()}")


class TestCoreDependencies:
    """Verify core dependencies are installed and working."""

    def test_numpy_import(self):
        """NumPy must be importable."""
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported")

    def test_numpy_operations(self):
        """NumPy must work for basic operations."""
        import numpy as np

        # Create arrays
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        # Basic operations
        c = a + b
        assert np.allclose(c, [5, 7, 9]), "NumPy addition failed"

        # Matrix operations
        m = np.array([[1, 2], [3, 4]])
        result = m @ m.T
        expected = np.array([[5, 11], [11, 25]])
        assert np.allclose(result, expected), "NumPy matmul failed"

        print("✅ NumPy operations work correctly")

    def test_matplotlib_import(self):
        """Matplotlib is optional - warn if not installed."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            print(f"✅ Matplotlib {matplotlib.__version__} imported (optional)")
        except ImportError:
            print("⚠️  Matplotlib not installed (optional dependency)")

    def test_matplotlib_plotting(self):
        """Matplotlib plotting is optional - warn if not installed."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-GUI backend for testing
            import matplotlib.pyplot as plt

            # Create a simple plot
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                fig.savefig(tmp.name)
                assert Path(tmp.name).exists(), "Failed to save plot"

            plt.close(fig)
            print("✅ Matplotlib can create and save plots (optional)")
        except ImportError:
            print("⚠️  Matplotlib not installed - plotting tests skipped (optional dependency)")

    def test_pytest_available(self):
        """pytest must be available for testing."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace'
        )
        assert result.returncode == 0, "pytest not available"
        print(f"✅ pytest available: {result.stdout.strip()}")

    def test_yaml_import(self):
        """PyYAML must be importable."""
        import yaml

        # Test YAML operations
        data = {'key': 'value', 'number': 42}
        yaml_str = yaml.dump(data)
        loaded = yaml.safe_load(yaml_str)
        assert loaded == data, "YAML serialization failed"

        print(f"✅ PyYAML {yaml.__version__} imported and working")

    def test_rich_import(self):
        """Rich must be importable for CLI output."""
        from rich.console import Console
        from rich.panel import Panel

        # Test Rich can create output
        console = Console()
        panel = Panel("Test", title="Test Panel")

        # Render to string to verify it works
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        assert len(output) > 0, "Rich rendering failed"

        print("✅ Rich console library working")


class TestJupyterEnvironment:
    """Verify Jupyter/JupyterLab is correctly configured."""

    def test_jupyter_import(self):
        """Jupyter must be importable."""
        import jupyter
        print("✅ Jupyter installed")

    def test_jupyterlab_import(self):
        """JupyterLab must be importable."""
        import jupyterlab
        print(f"✅ JupyterLab {jupyterlab.__version__} installed")

    def test_jupyter_command_available(self):
        """Jupyter command must be available."""
        result = _jupyter("--version")
        assert result.returncode == 0, "jupyter command not found"
        print(f"✅ jupyter command available:\n{result.stdout.strip()}")

    def test_jupyter_lab_command(self):
        """JupyterLab command must be available."""
        result = _jupyter("lab", "--version")
        assert result.returncode == 0, "jupyter lab command not found"
        print(f"✅ jupyter lab command available: {result.stdout.strip()}")

    def test_jupyter_kernelspec(self):
        """Jupyter kernel must be configured."""
        result = _jupyter("kernelspec", "list")
        assert result.returncode == 0, "Cannot list Jupyter kernels"
        assert "python3" in result.stdout, "Python3 kernel not found"
        print(f"✅ Jupyter kernel configured:\n{result.stdout.strip()}")

    def test_jupytext_available(self):
        """Jupytext must be available for .py ↔ .ipynb conversion."""
        import jupytext
        print(f"✅ Jupytext {jupytext.__version__} available")


class TestTinyTorchPackage:
    """Verify TinyTorch package is correctly installed."""

    def test_tinytorch_import(self):
        """TinyTorch package must be importable."""
        import tinytorch
        print(f"✅ TinyTorch package imported from {tinytorch.__file__}")

    def test_tinytorch_core_import(self):
        """TinyTorch core modules must be importable."""
        from tinytorch import core
        print("✅ TinyTorch core module available")

    def test_tinytorch_version(self):
        """TinyTorch must have version info."""
        import tinytorch
        assert hasattr(tinytorch, '__version__'), "TinyTorch version not defined"
        print(f"✅ TinyTorch version: {tinytorch.__version__}")

    def test_tinytorch_tensor_import(self):
        """Tensor class must be importable."""
        from tinytorch import Tensor
        assert Tensor is not None, "Tensor class not available"
        print("✅ Tensor class available")


class TestProjectStructure:
    """Verify project directory structure is correct."""

    def test_root_directory_exists(self):
        """Project root must exist with expected structure."""
        project_root = Path.cwd()
        assert project_root.exists(), "Project root not found"
        print(f"✅ Project root: {project_root}")

    def test_tinytorch_package_directory(self):
        """tinytorch/ package directory must exist."""
        tinytorch_dir = Path("tinytorch")
        assert tinytorch_dir.exists(), "tinytorch/ directory not found"
        assert tinytorch_dir.is_dir(), "tinytorch/ is not a directory"
        print(f"✅ Package directory: {tinytorch_dir.absolute()}")

    def test_tinytorch_init_file(self):
        """tinytorch/__init__.py must exist."""
        init_file = Path("tinytorch/__init__.py")
        assert init_file.exists(), "tinytorch/__init__.py not found"
        print(f"✅ Package init: {init_file.absolute()}")

    def test_modules_directory(self):
        """modules/ directory must exist for student work."""
        modules_dir = Path("modules")
        assert modules_dir.exists(), "modules/ directory not found"
        assert modules_dir.is_dir(), "modules/ is not a directory"
        print(f"✅ Modules directory: {modules_dir.absolute()}")

    def test_src_directory(self):
        """src/ directory must exist with source modules."""
        src_dir = Path("src")
        assert src_dir.exists(), "src/ directory not found"
        assert src_dir.is_dir(), "src/ is not a directory"

        # Count module directories
        module_dirs = [d for d in src_dir.iterdir() if d.is_dir() and d.name.startswith('0')]
        print(f"✅ Source directory: {src_dir.absolute()} ({len(module_dirs)} modules)")

    def test_tests_directory(self):
        """tests/ directory must exist."""
        tests_dir = Path("tests")
        assert tests_dir.exists(), "tests/ directory not found"
        assert tests_dir.is_dir(), "tests/ is not a directory"
        print(f"✅ Tests directory: {tests_dir.absolute()}")

    def test_tito_cli_exists(self):
        """TITO CLI must be available."""
        # Try to import tito
        try:
            import tito
            print(f"✅ TITO CLI available: {tito.__file__}")
        except ImportError:
            pytest.fail("TITO CLI not importable")


class TestSystemResources:
    """Verify system has adequate resources for TinyTorch development."""

    def test_disk_space_available(self):
        """At least 1GB disk space should be available."""
        import shutil

        stat = shutil.disk_usage(Path.cwd())
        free_gb = stat.free / (1024**3)

        assert free_gb >= 1.0, f"Low disk space: {free_gb:.1f}GB (need at least 1GB)"
        print(f"✅ Disk space: {free_gb:.1f}GB available")

class TestGitConfiguration:
    """Verify Git is configured for version control."""

    def test_git_available(self):
        """Git command must be available."""
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace'
        )
        assert result.returncode == 0, "git command not found"
        print(f"✅ Git available: {result.stdout.strip()}")

class TestStudentProtection:
    """Verify student protection system is configured."""

    def test_src_directory_readable(self):
        """Source directory should be readable."""
        src_dir = Path("src")
        assert src_dir.exists(), "src/ directory not found"

        # Try to read a file
        module_dirs = list(src_dir.glob("0*"))
        if module_dirs:
            test_file = list(module_dirs[0].glob("*.py"))
            if test_file:
                content = test_file[0].read_text(encoding='utf-8')
                assert len(content) > 0, "Cannot read source files"
                print(f"✅ Source files readable: {test_file[0]}")


def run_all_validation_tests():
    """
    Run all validation tests and provide a summary.

    This is called by `tito system health --verify` to ensure
    the environment is correctly configured.
    """
    import pytest

    # Run tests with verbose output
    args = [
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ]

    exit_code = pytest.main(args)

    if exit_code == 0:
        print("\n" + "="*70)
        print("🎉 All validation tests passed!")
        print("✅ TinyTorch environment is correctly configured")
        print("💡 Next: tito module 01")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ Some validation tests failed")
        print("🔧 Please fix the issues above and run: tito system health --verify")
        print("="*70)

    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(run_all_validation_tests())
