"""
Test gradient flow through all autograd operations.

This test suite validates that all arithmetic operations and activations
properly preserve gradient tracking and enable backpropagation.
"""

import numpy as np
import pytest
import sys
import warnings
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.activations import GELU, Sigmoid
# Try to import transformer for mean/sqrt monkey-patches (Module 13)
# This is optional - tests will skip if not available
try:
    from tinytorch.core import transformer
except ImportError:
    transformer = None  # Transformer module not yet available


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


def test_gelu_backward_large_magnitude_input_does_not_overflow():
    """GELUBackward.apply() computes its own sigmoid term for the derivative.
    The naive 1/(1+exp(-z)) formula overflows np.exp for large |z| (z here is
    1.702*x), so large-magnitude inputs must not raise or warn, and the
    gradient must saturate to the correct limiting values."""
    from tinytorch.core.autograd import GELUBackward

    print("Testing GELU backward large-magnitude stability...")

    for scale in [10.0, 200.0, 1000.0]:
        x = Tensor(np.array([-scale, scale]), requires_grad=True)
        backward_fn = GELUBackward(x)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grad = backward_fn.apply(np.array([1.0, 1.0]))

        assert np.all(np.isfinite(grad[0])), f"Non-finite GELU gradient at scale={scale}: {grad}"

    # At a large enough magnitude, GELU behaves like the identity for x>>0
    # (gradient -> 1) and like a hard zero for x<<0 (gradient -> 0).
    x = Tensor(np.array([-1000.0, 1000.0]), requires_grad=True)
    grad = GELUBackward(x).apply(np.array([1.0, 1.0]))
    assert grad[0][0] == pytest.approx(0.0, abs=1e-6)
    assert grad[0][1] == pytest.approx(1.0, abs=1e-6)

    print("✅ GELU backward stays stable at large magnitudes")


# NOTE: test_layernorm_operations was removed because it tested for Tensor.sqrt()
# which was never implemented. The transformer module uses np.sqrt() directly,
# not a Tensor method. Tests should only test implemented functionality.


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


def test_sigmoid_large_magnitude_input_does_not_overflow():
    """
    The autograd-tracked Sigmoid forward pass (tracked_sigmoid_forward)
    used to reimplement sigmoid with the naive 1/(1+exp(-x)) formula
    instead of delegating to the original, numerically stable
    implementation (which branches on sign to keep every exponent <= 0).
    For large-magnitude inputs, common right after gradients explode
    during training, this raised a spurious RuntimeWarning from np.exp
    overflowing.

    This asserts no such warning fires, and that the result still agrees
    with real PyTorch at extreme magnitudes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        for scale in (10.0, 200.0, 1000.0):
            x = Tensor(np.array([-scale, -1.0, 0.0, 1.0, scale], dtype=np.float32),
                       requires_grad=True)
            result = Sigmoid()(x)
            result.sum().backward()

            assert np.all((result.data >= 0.0) & (result.data <= 1.0))
            assert not np.any(np.isnan(result.data))
            assert x.grad is not None
            assert not np.any(np.isnan(x.grad))

        # At extreme magnitude, sigmoid should have fully saturated.
        x = Tensor(np.array([-1000.0, 1000.0], dtype=np.float32), requires_grad=True)
        result = Sigmoid()(x)
        assert result.data[0] == pytest.approx(0.0, abs=1e-6)
        assert result.data[1] == pytest.approx(1.0, abs=1e-6)

    print("✅ Sigmoid large-magnitude gradient flow works correctly, no overflow warning")


def test_reflected_operator_gradients():
    """
    Gradient-level regression for __radd__/__rsub__/__rmul__/__rtruediv__
    and sum(). These previously computed their result via a duplicate
    formula inside the autograd monkeypatch (tracked_radd/rsub/rmul/rdiv,
    sum_op) instead of calling Module 01's own methods, so a bug
    introduced into Module 01's source would never have been caught by
    any test exercising the built package. They now delegate to the
    original implementations, this locks in both the value and the
    gradient for each.
    """
    # __radd__: 5 + x
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    y = 5 + x
    y.sum().backward()
    assert np.allclose(y.data, [6.0, 7.0])
    assert np.allclose(x.grad, [1.0, 1.0])

    # __rsub__: 10 - x
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = 10 - x
    y.sum().backward()
    assert np.allclose(y.data, [9.0, 8.0, 7.0])
    assert np.allclose(x.grad, [-1.0, -1.0, -1.0])

    # __rmul__: 3 * x
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    y = 3 * x
    y.sum().backward()
    assert np.allclose(y.data, [3.0, 6.0])
    assert np.allclose(x.grad, [3.0, 3.0])

    # __rtruediv__: 20 / x
    x = Tensor(np.array([2.0, 4.0, 5.0]), requires_grad=True)
    y = 20 / x
    y.sum().backward()
    assert np.allclose(y.data, [10.0, 5.0, 4.0])
    assert np.allclose(x.grad, -20.0 / (x.data ** 2))

    # sum(): 2D sum(), no axis
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    y = x.sum()
    y.backward()
    assert np.isclose(y.data, 10.0)
    assert np.allclose(x.grad, np.ones((2, 2)))

    print("✅ Reflected operator and sum() gradients work correctly")


def test_forward_scalar_branch_gradients():
    """
    Gradient-level regression for __add__/__sub__/__truediv__'s scalar
    (non-Tensor other) branch. tracked_add/tracked_sub/tracked_div
    previously reassigned `other = Tensor(other)` before calling
    _original_add/_original_sub/_original_div, so those originals'
    own scalar branch (`else: return Tensor(self.data + other)`) never
    ran, only their Tensor branch did, since `other` always arrived
    pre-converted. tracked_mul already avoided this by keeping a
    separate Tensor-typed copy for gradient bookkeeping while passing
    the original `other` through unchanged, add/sub/div now match that.
    """
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = x + 5
    y.sum().backward()
    assert np.allclose(y.data, [6.0, 7.0, 8.0])
    assert np.allclose(x.grad, [1.0, 1.0, 1.0])

    x = Tensor(np.array([5.0, 10.0]), requires_grad=True)
    y = x - 2
    y.sum().backward()
    assert np.allclose(y.data, [3.0, 8.0])
    assert np.allclose(x.grad, [1.0, 1.0])

    x = Tensor(np.array([4.0, 6.0, 8.0]), requires_grad=True)
    y = x / 2
    y.sum().backward()
    assert np.allclose(y.data, [2.0, 3.0, 4.0])
    assert np.allclose(x.grad, [0.5, 0.5, 0.5])

    print("✅ Forward scalar-branch gradients work correctly")


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
    test_sigmoid_large_magnitude_input_does_not_overflow()
    test_reflected_operator_gradients()
    test_forward_scalar_branch_gradients()

    print("\n" + "="*70)
    print("✅ ALL GRADIENT FLOW TESTS PASSED")
    print("="*70 + "\n")
