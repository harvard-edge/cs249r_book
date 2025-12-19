"""
Test gradient flow through all autograd operations.

This test suite validates that all arithmetic operations and activations
properly preserve gradient tracking and enable backpropagation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.activations import GELU
# Import transformer to ensure mean/sqrt monkey-patches are applied
from tinytorch.core import transformer


def test_arithmetic_gradient_flow():
    """Test that arithmetic operations preserve requires_grad and set _grad_fn."""
    print("Testing arithmetic gradient flow...")

    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    y = Tensor(np.array([4.0, 5.0]), requires_grad=True)

    # Test addition
    z_add = x + y
    assert z_add.requires_grad, "Addition should preserve requires_grad"
    assert hasattr(z_add, '_grad_fn'), "Addition should set _grad_fn"

    # Test subtraction
    z_sub = x - y
    assert z_sub.requires_grad, "Subtraction should preserve requires_grad"
    assert hasattr(z_sub, '_grad_fn'), "Subtraction should set _grad_fn"

    # Test multiplication
    z_mul = x * y
    assert z_mul.requires_grad, "Multiplication should preserve requires_grad"
    assert hasattr(z_mul, '_grad_fn'), "Multiplication should set _grad_fn"

    # Test division
    z_div = x / y
    assert z_div.requires_grad, "Division should preserve requires_grad"
    assert hasattr(z_div, '_grad_fn'), "Division should set _grad_fn"

    print("✅ All arithmetic operations preserve gradient tracking")


def test_subtraction_backward():
    """Test that subtraction computes correct gradients."""
    print("Testing subtraction backward pass...")

    a = Tensor(np.array([5.0, 10.0]), requires_grad=True)
    b = Tensor(np.array([2.0, 3.0]), requires_grad=True)

    # Forward: c = a - b
    c = a - b

    # Backward
    loss = c.sum()
    loss.backward()

    # Check gradients: ∂loss/∂a = 1, ∂loss/∂b = -1
    assert a.grad is not None, "Gradient should flow to a"
    assert b.grad is not None, "Gradient should flow to b"
    assert np.allclose(a.grad, np.array([1.0, 1.0])), "Gradient wrt a should be 1"
    assert np.allclose(b.grad, np.array([-1.0, -1.0])), "Gradient wrt b should be -1"

    print("✅ Subtraction backward pass correct")


def test_division_backward():
    """Test that division computes correct gradients."""
    print("Testing division backward pass...")

    a = Tensor(np.array([6.0, 12.0]), requires_grad=True)
    b = Tensor(np.array([2.0, 3.0]), requires_grad=True)

    # Forward: c = a / b
    c = a / b

    # Backward
    loss = c.sum()
    loss.backward()

    # Check gradients: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
    assert a.grad is not None, "Gradient should flow to a"
    assert b.grad is not None, "Gradient should flow to b"
    assert np.allclose(a.grad, 1.0 / b.data), "Gradient wrt a should be 1/b"
    expected_b_grad = -a.data / (b.data ** 2)
    assert np.allclose(b.grad, expected_b_grad), "Gradient wrt b should be -a/b²"

    print("✅ Division backward pass correct")


def test_gelu_gradient_flow():
    """Test that GELU activation preserves gradient flow."""
    print("Testing GELU gradient flow...")

    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    gelu = GELU()

    # Forward
    y = gelu(x)
    assert y.requires_grad, "GELU output should have requires_grad=True"
    assert hasattr(y, '_grad_fn'), "GELU should set _grad_fn"

    # Backward
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Gradient should flow through GELU"
    assert np.abs(x.grad).max() > 1e-10, "GELU gradient should be non-zero"

    print("✅ GELU gradient flow works correctly")


def test_layernorm_operations():
    """Test gradient flow through LayerNorm operations (sqrt, div)."""
    print("Testing LayerNorm operations gradient flow...")

    # Test sqrt (monkey-patched in transformer module)
    x = Tensor(np.array([4.0, 9.0, 16.0]), requires_grad=True)

    # Check if sqrt is available (added in later modules)
    if not hasattr(x, 'sqrt'):
        pytest.skip("sqrt operation not yet implemented (requires transformer module)")

    sqrt_x = x.sqrt()
    assert sqrt_x.requires_grad, "Sqrt should preserve requires_grad"
    loss = sqrt_x.sum()
    loss.backward()
    assert x.grad is not None, "Gradient should flow through sqrt"

    # Test mean (should be available)
    x2 = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)
    if hasattr(x2, 'mean'):
        mean = x2.mean(axis=-1, keepdims=True)
        # Mean uses monkey-patched version in transformer context
        assert mean.requires_grad, "Mean should preserve requires_grad"
        loss2 = mean.sum()
        loss2.backward()
        assert x2.grad is not None, "Gradient should flow through mean"

    print("✅ LayerNorm operations gradient flow works")


def test_reshape_gradient_flow():
    """Test that reshape preserves gradient flow."""
    print("Testing reshape gradient flow...")

    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    y = x.reshape(4)

    assert y.requires_grad, "Reshape should preserve requires_grad"
    assert hasattr(y, '_grad_fn'), "Reshape should set _grad_fn"

    # Backward
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Gradient should flow through reshape"
    assert x.grad.shape == x.shape, "Gradient shape should match input shape"

    print("✅ Reshape gradient flow works correctly")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRADIENT FLOW TEST SUITE")
    print("="*70 + "\n")

    test_arithmetic_gradient_flow()
    test_subtraction_backward()
    test_division_backward()
    test_gelu_gradient_flow()
    test_layernorm_operations()
    test_reshape_gradient_flow()

    print("\n" + "="*70)
    print("✅ ALL GRADIENT FLOW TESTS PASSED")
    print("="*70 + "\n")
