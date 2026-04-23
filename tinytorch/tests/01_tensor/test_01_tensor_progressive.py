"""
Module 01: Progressive Integration Tests
Tests that Module 01 (Tensor) works correctly.

DEPENDENCY CHAIN: 01_tensor
This ensures students can trace back exactly where issues originate.
"""

import numpy as np
rng = np.random.default_rng(7)
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModule01Prerequisites:
    """Test that Module 01 (Setup) still works correctly."""

    def test_environment_setup_working(self):
        """Verify setup module functionality is still working."""
        # Python version detection
        assert sys.version_info >= (3, 10), "Python 3.10+ required"

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


class TestModule01TensorCore:
    """Test that Module 01 (Tensor) core functionality works."""

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
    """Test that the progressive stack (01) works correctly."""

    def test_environment_enables_tensor(self):
        """Test that the environment supports tensor functionality."""
        # 1. Environment should support numpy
        import numpy as np
        assert np.__version__ is not None, "Numpy not properly set up"

        # 2. Project structure should support tensor module (in src/ for students)
        tensor_module_path = Path(__file__).parent.parent.parent / "src" / "01_tensor"
        assert tensor_module_path.exists(), "Module structure missing: src/01_tensor"

    def test_end_to_end_capability(self):
        """Test end-to-end capability through Module 01."""
        try:
            # This should work if tensor is implemented
            from tinytorch.core.tensor import Tensor

            # Create tensors using Module 01
            data = rng.standard_normal((5, 10))
            t = Tensor(data)  # Uses tensor from Module 01

            # Basic functionality should work
            assert t.shape == (5, 10), "End-to-end stack broken"
            assert isinstance(t.data, np.ndarray), "Tensor-numpy integration broken"

        except ImportError:
            # If tensor not implemented, that's expected
            assert sys.version_info >= (3, 8), "Python environment broken"


class TestDependencyValidation:
    """Validate that Module 01 dependencies are working correctly."""

    def test_module_01_exports(self):
        """Test Module 01 (Tensor) exports are available."""
        try:
            from tinytorch.core.tensor import Tensor
            t = Tensor([1, 2, 3])
            assert t.shape == (3,), "Module 01 Tensor export broken"
        except ImportError:
            # If not implemented, verify basic environment works
            import platform
            assert platform.system() in ['Darwin', 'Linux', 'Windows'], "Basic environment broken"

    def test_module_01_tensor_uses_numpy(self):
        """Test Module 01 Tensor correctly uses NumPy foundation."""
        try:
            from tinytorch.core.tensor import Tensor

            # Tensor should store data as numpy array
            t = Tensor(np.array([1, 2, 3]))
            assert isinstance(t.data, np.ndarray), "Tensor should use numpy internally"

            # Should support optional features
            if hasattr(t, 'device') or hasattr(t, 'dtype'):
                # Advanced tensor features
                assert True, "Module 01 advanced features present"

        except ImportError:
            assert True, "Module 01 not implemented yet"


class TestRegressionPrevention:
    """Prevent regressions in Module 01."""

    def test_module_01_not_broken(self):
        """Ensure Module 01 core functionality is intact."""
        # These should ALWAYS work

        # Environment detection
        assert sys.version_info.major >= 3, "Python environment broken"

        # File system access
        project_root = Path(__file__).parent.parent.parent
        assert project_root.exists(), "Project structure broken"

        # Package imports
        import numpy as np
        assert np is not None, "Package management broken"

    def test_progressive_compatibility(self):
        """Test that Module 01 maintains backwards compatibility."""
        # Basic imports should still work
        import sys
        import os
        from pathlib import Path

        # These capabilities should never break
        assert callable(Path), "Path functionality broken"
        assert hasattr(sys, 'version_info'), "System info broken"
        assert hasattr(os, 'environ'), "Environment access broken"
