"""
Module 06: Gradient Correctness Tests
======================================

Validates that every backward pass computes numerically correct gradients
using finite differences as ground truth.

The core idea: for any function f, the analytical gradient computed by
backward() should match the numerical gradient:

    df/dx = (f(x + e) - f(x - e)) / (2e)

If they disagree, the backward implementation is wrong and training silently
learns in the wrong direction.

Coverage:
- Arithmetic ops: add, sub, mul, div, matmul
- Activations:    ReLU, Sigmoid, Tanh, GELU
- Losses:         MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
- Composed:       multi-layer chains

Tolerance notes (issue #1342):
- Arithmetic ops and simple activations (ReLU, Sigmoid, Tanh) use tight
  tolerances (rtol=1e-3, atol=1e-5) because their derivatives are simple.
- GELU uses looser tolerance (rtol=5e-3) because its derivative chains
  tanh of a cubic through sech-squared, amplifying finite-difference error.
- BCE loss clips predictions, introducing derivative jumps near clip
  boundaries; slightly looser tolerance handles this.
- CrossEntropy with softmax is well-conditioned at the test points chosen.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import order matters: autograd must be enabled AFTER all modules are imported
# so that enable_autograd() can patch their forward methods correctly.
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, GELU
from tinytorch.core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from tinytorch.core.autograd import enable_autograd

enable_autograd()


# -----------------------------------------------
# Finite difference helpers
# -----------------------------------------------

EPS = 1e-5  # smaller epsilon reduces truncation error for smooth functions
RTOL = 1e-3
ATOL = 1e-5


def finite_diff_grad(fn, x_data):
    """
    Numerical gradient via central differences.
    fn: np.ndarray -> scalar float
    """
    x_data = x_data.astype(np.float64)
    grad = np.zeros_like(x_data)
    it = np.nditer(x_data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = float(x_data[idx])

        x_data[idx] = orig + EPS
        f_plus = float(fn(x_data.copy()))

        x_data[idx] = orig - EPS
        f_minus = float(fn(x_data.copy()))

        x_data[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2.0 * EPS)
        it.iternext()
    return grad


def check_grad(fn_tensor, x_data, atol=ATOL, rtol=RTOL):
    """
    Assert analytical gradient from backward() matches finite-difference gradient.
    fn_tensor: Tensor -> scalar Tensor
    """
    x_data = x_data.astype(np.float64)

    # Analytical gradient
    x = Tensor(x_data.copy(), requires_grad=True)
    loss = fn_tensor(x)
    assert loss is not None, "fn_tensor returned None"
    loss.backward(np.ones_like(loss.data))
    assert x.grad is not None, (
        "x.grad is None after backward(). "
        "Check that requires_grad=True is set and the computation graph was built."
    )
    analytical = np.array(x.grad, dtype=np.float64)

    # Numerical gradient
    def scalar_fn(arr):
        t = Tensor(arr.copy())
        out = fn_tensor(t)
        return float(out.data)

    numerical = finite_diff_grad(scalar_fn, x_data.copy())

    np.testing.assert_allclose(
        analytical, numerical,
        rtol=rtol, atol=atol,
        err_msg=(
            "Gradient mismatch.\n"
            "  analytical={}\n"
            "  numerical={}".format(analytical, numerical)
        )
    )


# -----------------------------------------------
# Arithmetic operations
# -----------------------------------------------

class TestArithmeticGradients:
    """Finite-difference checks for arithmetic backward passes."""

    def test_add_backward(self):
        def fn(x):
            c = Tensor(np.array([3.0, 1.0, 2.0]))
            return (x + c).sum()

        check_grad(fn, np.array([1.0, 2.0, 3.0]))

    def test_sub_backward(self):
        def fn(x):
            c = Tensor(np.array([1.0, 1.0, 1.0]))
            return (x - c).sum()

        check_grad(fn, np.array([4.0, 5.0, 6.0]))

    def test_mul_backward(self):
        def fn(x):
            c = Tensor(np.array([2.0, 3.0, 4.0]))
            return (x * c).sum()

        check_grad(fn, np.array([1.0, 2.0, 3.0]))

    def test_div_backward(self):
        def fn(x):
            c = Tensor(np.array([2.0, 4.0, 5.0]))
            return (x / c).sum()

        check_grad(fn, np.array([3.0, 8.0, 10.0]))

    def test_matmul_backward_wrt_left(self):
        W_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        def fn(x):
            W = Tensor(W_data.copy())
            return x.matmul(W).sum()

        check_grad(fn, np.array([[1.0, 2.0, 3.0]]))

    def test_matmul_backward_wrt_right(self):
        A_data = np.array([[1.0, 2.0, 3.0]])

        def fn(x):
            A = Tensor(A_data.copy())
            return A.matmul(x).sum()

        check_grad(fn, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    def test_chained_ops_backward(self):
        def fn(x):
            two = Tensor(np.array([2.0, 2.0, 2.0]))
            one = Tensor(np.array([1.0, 1.0, 1.0]))
            return (x * two + one).sum()

        check_grad(fn, np.array([1.0, -1.0, 3.0]))

    def test_broadcast_add_backward(self):
        bias_data = np.array([1.0, 2.0])

        def fn(x):
            bias = Tensor(bias_data.copy())
            return (x + bias).sum()

        check_grad(fn, np.ones((3, 2)))


# -----------------------------------------------
# Activations
# -----------------------------------------------

class TestActivationGradients:
    """Finite-difference checks for activation backward passes."""

    def test_relu_backward(self):
        relu = ReLU()

        def fn(x):
            return relu(x).sum()

        # Avoid x=0 where ReLU derivative is discontinuous
        check_grad(fn, np.array([1.0, -0.5, 2.0, -1.0, 0.5]))

    def test_sigmoid_backward(self):
        sigmoid = Sigmoid()

        def fn(x):
            return sigmoid(x).sum()

        check_grad(fn, np.array([0.0, 1.0, -1.0, 2.0, -2.0]))

    def test_tanh_backward(self):
        tanh = Tanh()

        def fn(x):
            return tanh(x).sum()

        check_grad(fn, np.array([0.0, 0.5, -0.5, 1.0, -1.0]))

    def test_gelu_backward(self):
        """GELU derivative chains tanh(cubic) through sech^2; needs looser tol."""
        gelu = GELU()

        def fn(x):
            return gelu(x).sum()

        # Use moderate input values where the approximation is well-behaved
        check_grad(fn, np.array([0.0, 1.0, -1.0, 0.5, -0.5]),
                   rtol=5e-3, atol=1e-4)

    def test_relu_zero_boundary(self):
        """ReLU grad at x=0 should be 0."""
        relu = ReLU()
        x = Tensor(np.array([0.0]), requires_grad=True)
        out = relu(x)
        out.backward(np.ones_like(out.data))

        assert x.grad is not None
        assert np.allclose(x.grad, 0.0), (
            "ReLU grad at 0 should be 0, got {}".format(x.grad)
        )


# -----------------------------------------------
# Loss functions
# -----------------------------------------------

class TestLossGradients:
    """Finite-difference checks for loss backward passes."""

    def test_mse_backward(self):
        loss_fn = MSELoss()
        targets_data = np.array([1.0, 2.0, 3.0])

        def fn(x):
            return loss_fn.forward(x, Tensor(targets_data.copy()))

        check_grad(fn, np.array([1.5, 2.5, 2.0]))

    def test_mse_backward_batch(self):
        loss_fn = MSELoss()
        targets_data = np.array([[1.0], [2.0], [3.0]])

        def fn(x):
            return loss_fn.forward(x, Tensor(targets_data.copy()))

        check_grad(fn, np.array([[1.5], [1.8], [3.2]]))

    def test_bce_backward(self):
        """BCE clips predictions; use interior points away from clip boundaries."""
        loss_fn = BinaryCrossEntropyLoss()
        targets_data = np.array([1.0, 0.0, 1.0, 0.0])

        def fn(x):
            # Clamp to avoid log(0) at finite-difference boundaries
            clamped = Tensor(np.clip(x.data, 0.05, 0.95),
                             requires_grad=x.requires_grad)
            return loss_fn.forward(clamped, Tensor(targets_data.copy()))

        # Test points well inside (0.05, 0.95) to avoid clip-boundary artifacts
        check_grad(fn, np.array([0.7, 0.3, 0.8, 0.2]),
                   rtol=2e-3, atol=1e-4)

    def test_crossentropy_backward(self):
        loss_fn = CrossEntropyLoss()
        targets_data = np.array([0, 2])

        def fn(x):
            return loss_fn.forward(x, Tensor(targets_data.copy()))

        check_grad(fn, np.array([[2.0, 1.0, 0.1], [0.5, 1.5, 2.0]]))


# -----------------------------------------------
# Composed graphs
# -----------------------------------------------

class TestComposedGradients:
    """Gradient correctness through multi-operation chains."""

    def test_linear_layer_weight_gradient(self):
        """Weight gradient in y = x @ W is correct."""
        x_data = np.array([[1.0, 2.0]])
        layer = Linear(2, 3)
        original_bias = layer.bias.data.copy()

        def fn(w):
            layer.weight.data = w.data.copy()
            layer.bias.data = original_bias.copy()
            return layer.forward(Tensor(x_data.copy())).sum()

        check_grad(fn, layer.weight.data.copy())

    def test_linear_layer_bias_gradient(self):
        """Bias gradient in y = x @ W + b is correct."""
        x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer = Linear(2, 3)
        original_weight = layer.weight.data.copy()

        def fn(b):
            layer.weight.data = original_weight.copy()
            layer.bias.data = b.data.copy()
            return layer.forward(Tensor(x_data.copy())).sum()

        check_grad(fn, layer.bias.data.copy())

    def test_two_layer_chain(self):
        """Gradient flows correctly through two Linear layers."""
        layer1 = Linear(3, 4)
        layer2 = Linear(4, 2)
        relu = ReLU()

        def fn(x):
            h = relu(layer1.forward(x))
            return layer2.forward(h).sum()

        check_grad(fn, np.array([[1.0, 0.5, -1.0]]),
                   rtol=2e-3, atol=1e-4)

    def test_mse_through_linear(self):
        """End-to-end gradient: input through linear layer through MSE loss."""
        layer = Linear(2, 1)
        loss_fn = MSELoss()
        targets_data = np.array([[1.0]])

        def fn(x):
            pred = layer.forward(x)
            return loss_fn.forward(pred, Tensor(targets_data.copy()))

        check_grad(fn, np.array([[0.5, -0.5]]))

    def test_gradient_accumulates_across_backward_calls(self):
        """Calling backward twice without zero_grad accumulates gradients."""
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        c = Tensor(np.array([1.0, 1.0]))

        loss1 = (x * c).sum()
        loss1.backward(np.ones_like(loss1.data))
        grad_after_first = np.array(x.grad, dtype=np.float64).copy()

        x2 = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        x2.grad = grad_after_first.copy()
        c2 = Tensor(np.array([1.0, 1.0]))
        loss2 = (x2 * c2).sum()
        loss2.backward(np.ones_like(loss2.data))

        np.testing.assert_allclose(
            x2.grad, grad_after_first * 2,
            err_msg="Gradient accumulation should double gradient on second backward"
        )


# -----------------------------------------------
# No-grad context
# -----------------------------------------------

class TestNoGradContext:
    """Verify no_grad() stops gradient tracking."""

    def test_no_grad_disables_tracking(self):
        from tinytorch.core.autograd import no_grad

        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        with no_grad():
            y = x + Tensor(np.array([1.0, 1.0]))

        assert not getattr(y, "requires_grad", False), (
            "Tensor created inside no_grad() should not require gradients"
        )

    def test_no_grad_does_not_affect_outside(self):
        from tinytorch.core.autograd import no_grad

        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        with no_grad():
            pass

        y = x + Tensor(np.array([1.0, 1.0]))
        assert y.requires_grad, (
            "Tensor created outside no_grad() should still require gradients"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
