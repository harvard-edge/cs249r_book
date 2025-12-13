"""
TinyTorch Test Utilities

Shared utilities for integration tests across all modules.
Provides setup functions and common test helpers.
"""

import sys
import os
from pathlib import Path


def setup_integration_test():
    """
    Set up the environment for integration testing.

    This function ensures:
    1. The TinyTorch package is importable
    2. NumPy random seed is set for reproducibility
    3. Warning filters are set appropriately

    Call this at the top of integration test files before importing TinyTorch.
    """
    import warnings
    import numpy as np

    # Ensure tinytorch is on the path (from project root)
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Set random seed for reproducibility
    np.random.seed(42)

    # Suppress certain warnings during tests
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Set quiet mode for tinytorch imports during tests
    os.environ['TINYTORCH_QUIET'] = '1'


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def get_test_data_path() -> Path:
    """Return the path to test data directory."""
    return get_project_root() / "datasets"


def create_test_tensor(shape, requires_grad=True, seed=None):
    """
    Create a test tensor with random data.

    Args:
        shape: Tuple specifying tensor shape
        requires_grad: Whether tensor should track gradients
        seed: Optional random seed for reproducibility

    Returns:
        Tensor with random data
    """
    import numpy as np
    from tinytorch.core.tensor import Tensor

    if seed is not None:
        np.random.seed(seed)

    data = np.random.randn(*shape).astype(np.float32)
    return Tensor(data, requires_grad=requires_grad)


def assert_tensors_close(t1, t2, rtol=1e-5, atol=1e-8, msg=""):
    """
    Assert that two tensors are element-wise close.

    Args:
        t1: First tensor
        t2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Optional message for assertion error
    """
    import numpy as np

    # Extract data from tensors if needed
    data1 = t1.data if hasattr(t1, 'data') else t1
    data2 = t2.data if hasattr(t2, 'data') else t2

    if not np.allclose(data1, data2, rtol=rtol, atol=atol):
        diff = np.abs(data1 - data2)
        max_diff = np.max(diff)
        raise AssertionError(
            f"Tensors not close (max diff: {max_diff:.6e}). {msg}"
        )


def assert_gradients_exist(tensor, msg=""):
    """Assert that a tensor has computed gradients."""
    if tensor.grad is None:
        raise AssertionError(f"Tensor has no gradients. {msg}")


def skip_if_no_tinytorch():
    """Pytest skip decorator for when tinytorch isn't available."""
    import pytest
    try:
        import tinytorch
        return pytest.mark.skipif(False, reason="TinyTorch available")
    except ImportError:
        return pytest.mark.skip(reason="TinyTorch not installed")
