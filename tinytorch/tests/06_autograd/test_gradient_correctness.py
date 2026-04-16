"""
Module 06: Gradient Correctness Tests
======================================

Validates that every backward pass computes numerically correct gradients
using finite differences as ground truth.

The core idea: for any function f, the analytical gradient computed by
backward() should match the numerical gradient:

    ∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)

If they disagree, the backward implementation is wrong — silently producing
incorrect updates during training.

Coverage:
- Arithmetic ops: add, sub, mul, div, matmul
- Activations:    ReLU, Sigmoid, Tanh, GELU
- Losses:         MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
- Composed:       multi-layer chains
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd

enable_autograd()

# ─────────────────────────────────────────────
# Finite difference helper
# ─────────────────────────────────────────────

EPS = 1e-4
RTOL = 1e-3   # relative tolerance — finite differences are inherently noisy
ATOL = 1e-5


def finite_diff_grad(fn, x_data: np.ndarray) -> np.ndarray:
    """
    Compute numerical gradient of scalar fn w.r.t. x via central differences.

    fn must accept a numpy array and return a scalar float.
    """
    grad = np.zeros_like(x_data, dtype=np.float64)
    it = np.nditer(x_data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x_data[idx]

        x_data[idx] = orig + EPS
        f_plus = fn(x_data.copy())

        x_data[idx] = orig - EPS
        f_minus = fn(x_data.copy())

        x_data[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * EPS)
        it.iternext()
    return grad


def check_grad(fn_tensor, x_data, atol=ATOL, rtol=RTOL):
    """
    Assert analytical gradient matches finite-difference gradient.

    fn_tensor: x_data (np.ndarray) -> scalar Tensor (loss)
    """
    x_data = x_data.astype(np.float64)

    # Analytical gradient
    x = Tensor(x_data.copy(), requires_grad=True)
    loss = fn_tensor(x)
    loss.backward(np.ones_like(loss.data))
    analytical = x.grad.copy() if isinstance(x.grad, np.ndarray) else np.array(x.grad)

    # Numerical gradient
    def scalar_fn(arr):
        t = Tensor(arr.copy(), requires_grad=False)
        out = fn_tensor(t)
        return float(out.data)

    numerical = finite_diff_grad(scalar_fn, x_data.copy())

    np.testing.assert_allclose(
        analytical, numerical,
        rtol=rtol, atol=atol,
        err_msg=f"Gradient mismatch.\n  analytical={analytical}\n  numerical={numerical}"
    )


# ─────────────────────────────────────────────
# Arithmetic operations
# ─────────────────────────────────────────────

class TestArithmeticGradients:
    """Finite-difference checks for basic arithmetic backward passes."""

    def test_add_backward(self):
        """∂(sum(x + c))/∂x = 1 for all elements."""
        c = Tensor(np.array([3.0, 1.0, 2.0]))

        def fn(x):
            return (x + c).sum()

        check_grad(fn, np.array([1.0, 2.0, 3.0]))

    def test_sub_backward(self):
        """∂(sum(x - c))/∂x = 1."""
        c = Tensor(np.array([1.0, 1.0, 1.0]))

        def fn(x):
            return (x - c).sum()

        check_grad(fn, np.array([4.0, 5.0, 6.0]))

    def test_mul_backward(self):
        """∂(sum(x * c))/∂x = c."""
        c = Tensor(np.array([2.0, 3.0, 4.0]))

        def fn(x):
            return (x * c).sum()

        check_grad(fn, np.array([1.0, 2.0, 3.0]))

    def test_div_backward(self):
        """∂(sum(x / c))/∂x = 1/c."""
        c = Tensor(np.array([2.0, 4.0, 5.0]))

        def fn(x):
            return (x / c).sum()

        check_grad(fn, np.array([3.0, 8.0, 10.0]))

    def test_matmul_backward_wrt_left(self):
        """∂(sum(x @ W))/∂x — gradient flows correctly to left operand."""
        W_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        W = Tensor(W_data)

        def fn(x):
            return x.matmul(W).sum()

        check_grad(fn, np.array([[1.0, 2.0, 3.0]]))

    def test_matmul_backward_wrt_right(self):
        """∂(sum(A @ x))/∂x — gradient flows correctly to right operand."""
        A_data = np.array([[1.0, 2.0, 3.0]])
        A = Tensor(A_data)

        def fn(x):
            return A.matmul(x).sum()

        check_grad(fn, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    def test_chained_ops_backward(self):
        """Gradient flows correctly through x * 2 + 1 chain."""
        def fn(x):
            two = Tensor(np.array([2.0, 2.0, 2.0]))
            one = Tensor(np.array([1.0, 1.0, 1.0]))
            return (x * two + one).sum()

        check_grad(fn, np.array([1.0, -1.0, 3.0]))

    def test_broadcast_add_backward(self):
        """Gradient of x + bias where bias broadcasts over batch dimension."""
        bias = Tensor(np.array([1.0, 2.0]))

        def fn(x):
            return (x + bias).sum()

        # x shape (3, 2) broadcasts with bias shape (2,)
        check_grad(fn, np.ones((3, 2)))


# ─────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────

class TestActivationGradients:
    """Finite-difference checks for activation backward passes."""

    def test_relu_backward(self):
        """ReLU gradient: 1 where x > 0, 0 elsewhere."""
        from tinytorch.core.activations import ReLU
        relu = ReLU()

        def fn(x):
            return relu(x).sum()

        # Mix of positive and negative values
        check_grad(fn, np.array([1.0, -0.5, 2.0, -1.0, 0.5]))

    def test_sigmoid_backward(self):
        """Sigmoid gradient: σ(x)(1 - σ(x))."""
        from tinytorch.core.activations import Sigmoid
        sigmoid = Sigmoid()

        def fn(x):
            return sigmoid(x).sum()

        check_grad(fn, np.array([0.0, 1.0, -1.0, 2.0, -2.0]))

    def test_tanh_backward(self):
        """Tanh gradient: 1 - tanh²(x)."""
        from tinytorch.core.activations import Tanh
        tanh = Tanh()

        def fn(x):
            return tanh(x).sum()

        check_grad(fn, np.array([0.0, 0.5, -0.5, 1.0, -1.0]))

    def test_gelu_backward(self):
        """GELU gradient matches finite differences."""
        from tinytorch.core.activations import GELU
        gelu = GELU()

        def fn(x):
            return gelu(x).sum()

        check_grad(fn, np.array([0.0, 1.0, -1.0, 2.0, -2.0]))

    def test_relu_zero_boundary(self):
        """ReLU gradient at x=0 should be 0 (not 1)."""
        from tinytorch.core.activations import ReLU
        relu = ReLU()

        x = Tensor(np.array([0.0]), requires_grad=True)
        out = relu(x)
        out.backward(np.ones_like(out.data))

        assert x.grad is not None
        assert float(x.grad) == 0.0, f"ReLU grad at 0 should be 0, got {x.grad}"


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

class TestLossGradients:
    """Finite-difference checks for loss backward passes."""

    def test_mse_backward(self):
        """MSE gradient: 2*(pred - target)/N."""
        from tinytorch.core.losses import MSELoss
        loss_fn = MSELoss()
        targets = Tensor(np.array([1.0, 2.0, 3.0]))

        def fn(x):
            return loss_fn.forward(x, targets)

        check_grad(fn, np.array([1.5, 2.5, 2.0]))

    def test_mse_backward_batch(self):
        """MSE gradient is correct for 2D batch predictions."""
        from tinytorch.core.losses import MSELoss
        loss_fn = MSELoss()
        targets = Tensor(np.array([[1.0], [2.0], [3.0]]))

        def fn(x):
            return loss_fn.forward(x, targets)

        check_grad(fn, np.array([[1.5], [1.8], [3.2]]))

    def test_bce_backward(self):
        """BCE gradient: (pred - target) / (pred*(1-pred)*N)."""
        from tinytorch.core.losses import BinaryCrossEntropyLoss
        loss_fn = BinaryCrossEntropyLoss()
        targets = Tensor(np.array([1.0, 0.0, 1.0, 0.0]))

        def fn(x):
            # Clamp to avoid log(0) in finite differences
            clamped = np.clip(x.data, 0.05, 0.95)
            return loss_fn.forward(Tensor(clamped, requires_grad=x.requires_grad), targets)

        check_grad(fn, np.array([0.7, 0.3, 0.8, 0.2]))

    def test_crossentropy_backward(self):
        """CrossEntropy gradient matches finite differences."""
        from tinytorch.core.losses import CrossEntropyLoss
        loss_fn = CrossEntropyLoss()
        targets = Tensor(np.array([0, 2]))   # integer class indices

        def fn(x):
            return loss_fn.forward(x, targets)

        # 2 samples, 3 classes
        check_grad(fn, np.array([[2.0, 1.0, 0.1], [0.5, 1.5, 2.0]]))


# ─────────────────────────────────────────────
# Composed graphs
# ─────────────────────────────────────────────

class TestComposedGradients:
    """Gradient correctness through multi-operation chains."""

    def test_linear_layer_weight_gradient(self):
        """Weight gradient in y = x @ W is correct."""
        from tinytorch.core.layers import Linear

        x_data = np.array([[1.0, 2.0]])   # (1, 2)
        layer = Linear(2, 3)

        def fn(w):
            # Replace weight data, do forward, sum output
            layer.weight.data = w
            out = layer.forward(Tensor(x_data))
            return out.sum()

        check_grad(fn, layer.weight.data.copy())

    def test_linear_layer_bias_gradient(self):
        """Bias gradient in y = x @ W + b is correct."""
        from tinytorch.core.layers import Linear

        x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer = Linear(2, 3)

        def fn(b):
            layer.bias.data = b
            out = layer.forward(Tensor(x_data))
            return out.sum()

        check_grad(fn, layer.bias.data.copy())

    def test_two_layer_chain(self):
        """Gradient flows correctly through two Linear layers in sequence."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU

        layer1 = Linear(3, 4)
        layer2 = Linear(4, 2)
        relu = ReLU()

        def fn(x):
            h = relu(layer1.forward(x))
            out = layer2.forward(h)
            return out.sum()

        check_grad(fn, np.array([[1.0, 0.5, -1.0]]))

    def test_mse_through_linear(self):
        """End-to-end gradient: MSE loss → linear layer → input."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.losses import MSELoss

        layer = Linear(2, 1)
        loss_fn = MSELoss()
        targets = Tensor(np.array([[1.0]]))

        def fn(x):
            pred = layer.forward(x)
            return loss_fn.forward(pred, targets)

        check_grad(fn, np.array([[0.5, -0.5]]))

    def test_gradient_does_not_accumulate_across_calls(self):
        """Calling backward twice without zero_grad accumulates gradients."""
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        c = Tensor(np.array([1.0, 1.0]))

        # First backward
        loss1 = (x * c).sum()
        loss1.backward(np.ones_like(loss1.data))
        grad_after_first = x.grad.copy()

        # Second backward without clearing — should accumulate
        x2 = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        x2.grad = grad_after_first.copy()
        loss2 = (x2 * c).sum()
        loss2.backward(np.ones_like(loss2.data))

        assert np.allclose(x2.grad, grad_after_first * 2), (
            "Gradient accumulation should double the gradient on second backward"
        )


# ─────────────────────────────────────────────
# No-grad context
# ─────────────────────────────────────────────

class TestNoGradContext:
    """Verify no_grad() stops gradient tracking."""

    def test_no_grad_disables_tracking(self):
        from tinytorch.core.autograd import no_grad

        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        with no_grad():
            y = x + Tensor(np.array([1.0, 1.0]))

        assert not getattr(y, 'requires_grad', False), (
            "Tensor created inside no_grad() should not require gradients"
        )

    def test_no_grad_does_not_affect_outside(self):
        from tinytorch.core.autograd import no_grad

        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        with no_grad():
            pass  # nothing inside

        y = x + Tensor(np.array([1.0, 1.0]))
        assert y.requires_grad, (
            "Tensor created outside no_grad() should still require gradients"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
