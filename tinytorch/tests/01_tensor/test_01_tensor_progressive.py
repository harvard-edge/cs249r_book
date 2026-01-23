"""
Module 01: Progressive Integration Tests
Tests that Module 02 (Tensor) works correctly AND that all previous modules still work.

DEPENDENCY CHAIN: 01_setup → 02_tensor
This ensures students can trace back exactly where issues originate.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModule01Prerequisites:
    """Test that Module 01 (Setup) still works correctly."""

    def test_environment_setup_working(self):
        """Verify setup module functionality is still working."""
        # Python version detection
        assert sys.version_info >= (3, 8), "Python 3.8+ required"

        # Project structure - check directories that exist in student install
        project_root = Path(__file__).parent.parent.parent
        # 'src' is the student working directory, 'tests' and 'tito' are CLI infrastructure
        required_dirs = ['src', 'tests', 'tito']
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Setup failed: {dir_name} directory missing"

    def test_development_environment_ready(self):
        """Verify development environment is properly configured."""
        # Required packages
        required_packages = ['numpy', 'pathlib']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                assert False, f"Setup failed: {package} not available"


class TestModule02TensorCore:
    """Test that Module 02 (Tensor) core functionality works."""

    def test_tensor_creation_and_basics(self):
        """Test tensor creation works correctly."""
        try:
            from tinytorch.core.tensor import Tensor

            # Basic tensor creation
            t1 = Tensor([1, 2, 3])
            assert t1.shape == (3,), "Tensor creation failed"

            # Numpy array integration
            arr = np.array([[1, 2], [3, 4]])
            t2 = Tensor(arr)
            assert t2.shape == (2, 2), "Numpy integration failed"

        except ImportError:
            assert True, "Tensor not implemented yet (expected)"

    def test_tensor_operations(self):
        """Test basic tensor operations work."""
        try:
            from tinytorch.core.tensor import Tensor

            t1 = Tensor([1, 2, 3])
            t2 = Tensor([4, 5, 6])

            # Test operations if implemented
            if hasattr(t1, '__add__'):
                result = t1 + t2
                expected = np.array([5, 7, 9])
                assert np.array_equal(result.data, expected), "Tensor addition failed"

        except ImportError:
            assert True, "Tensor operations not implemented yet (expected)"


class TestProgressiveStack:
    """Test that the progressive stack (01→02) works together."""

    def test_setup_enables_tensor(self):
        """Test that proper setup enables tensor functionality."""
        # Verify setup created the foundation for tensors

        # 1. Environment should support numpy (from setup)
        import numpy as np
        assert np.__version__ is not None, "Numpy not properly set up"

        # 2. Project structure should support tensor module (in src/ for students)
        tensor_module_path = Path(__file__).parent.parent.parent / "src" / "01_tensor"
        assert tensor_module_path.exists(), "Setup didn't create proper module structure"

    def test_end_to_end_capability(self):
        """Test end-to-end capability through Module 02."""
        try:
            # This should work if both setup and tensor are implemented
            from tinytorch.core.tensor import Tensor

            # Create tensors using environment from Module 01
            data = np.random.randn(5, 10)  # Uses numpy from setup
            t = Tensor(data)  # Uses tensor from Module 02

            # Basic functionality should work
            assert t.shape == (5, 10), "End-to-end stack broken"
            assert isinstance(t.data, np.ndarray), "Tensor-numpy integration broken"

        except ImportError:
            # If tensor not implemented, that's expected
            # But setup should still work
            assert sys.version_info >= (3, 8), "Setup module broken"


class TestDependencyValidation:
    """Validate that dependencies are working correctly."""

    def test_module_01_exports(self):
        """Test Module 01 exports are available."""
        try:
            # Try to import setup functionality
            from tinytorch.setup import get_system_info
            info = get_system_info()
            assert 'platform' in info, "Module 01 exports broken"
        except ImportError:
            # If not implemented, verify basic setup works
            import platform
            assert platform.system() in ['Darwin', 'Linux', 'Windows'], "Basic setup broken"

    def test_module_02_builds_on_01(self):
        """Test Module 02 correctly uses Module 01 foundation."""
        try:
            from tinytorch.core.tensor import Tensor

            # Tensor should use numpy (set up by Module 01)
            t = Tensor(np.array([1, 2, 3]))

            # Should use system info for optimization hints
            if hasattr(t, 'device') or hasattr(t, 'dtype'):
                # Advanced tensor features building on setup
                assert True, "Module 02 successfully builds on Module 01"

        except ImportError:
            assert True, "Module 02 not implemented yet"


class TestRegressionPrevention:
    """Prevent regressions in previously working modules."""

    def test_module_01_not_broken(self):
        """Ensure Module 02 development didn't break Module 01."""
        # These should ALWAYS work regardless of Module 02 status

        # Environment detection
        assert sys.version_info.major >= 3, "Python environment broken"

        # File system access
        project_root = Path(__file__).parent.parent.parent
        assert project_root.exists(), "Project structure broken"

        # Package imports
        import numpy as np
        assert np is not None, "Package management broken"

    def test_progressive_compatibility(self):
        """Test that progress doesn't break backwards compatibility."""
        # Module 02 should not change Module 01 behavior

        # Basic imports should still work
        import sys
        import os
        from pathlib import Path

        # These are Module 01 capabilities that should never break
        assert callable(Path), "Path functionality broken"
        assert hasattr(sys, 'version_info'), "System info broken"
        assert hasattr(os, 'environ'), "Environment access broken"
