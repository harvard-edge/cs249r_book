"""Loss backward returns a target gradient when targets require one (2026-09-28).

Before this fix MSEFunction.backward and BinaryCrossEntropyFunction.backward
returned None for the target slot, so targets.grad stayed None.
"""
import numpy as np
import pytest

from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import MSELoss, BinaryCrossEntropyLoss
import tinytorch.core.autograd  # noqa: F401  (installs backward rules)


def _mse(p, t):
    return np.mean((p - t) ** 2)


def _bce(p, t, eps=1e-7):
    pc = np.clip(p, eps, 1 - eps)
    return -np.mean(t * np.log(pc) + (1 - t) * np.log(1 - pc))


def _finite_difference(fn, p, t, h=1e-6):
    grad = np.zeros_like(t)
    for i in range(t.size):
        up, down = t.copy(), t.copy()
        up.flat[i] += h
        down.flat[i] -= h
        grad.flat[i] = (fn(p, up) - fn(p, down)) / (2 * h)
    return grad


def test_mse_target_gradient_is_negated_prediction_gradient():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([3.0, 5.0], requires_grad=True)
    MSELoss()(a, b).backward()
    assert b.grad is not None, "target gradient was dropped"
    np.testing.assert_allclose(np.asarray(b.grad), [1.0, 2.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(a.grad), [-1.0, -2.0], rtol=1e-6)
    fd = _finite_difference(_mse, np.array([2.0, 3.0]), np.array([3.0, 5.0]))
    np.testing.assert_allclose(np.asarray(b.grad), fd, rtol=1e-5)


@pytest.mark.parametrize("pred_requires_grad", [True, False])
def test_bce_target_gradient_matches_finite_differences(pred_requires_grad):
    p = np.array([0.7, 0.2, 0.9, 0.5])
    t = np.array([0.6, 0.1, 0.3, 1.0])
    a = Tensor(p, requires_grad=pred_requires_grad)
    b = Tensor(t, requires_grad=True)
    BinaryCrossEntropyLoss()(a, b).backward()
    assert b.grad is not None, "target gradient was dropped"
    # Compare against the float32 values the Tensor actually holds.
    fd = _finite_difference(_bce, a.data.astype(np.float64), b.data.astype(np.float64))
    np.testing.assert_allclose(np.asarray(b.grad), fd, rtol=1e-4, atol=1e-6)
    if not pred_requires_grad:
        assert a.grad is None


def test_targets_without_requires_grad_still_get_no_gradient():
    a = Tensor([0.7, 0.2], requires_grad=True)
    b = Tensor([1.0, 0.0])
    BinaryCrossEntropyLoss()(a, b).backward()
    assert b.grad is None
    a2 = Tensor([2.0, 3.0], requires_grad=True)
    b2 = Tensor([1.0, 2.0])
    MSELoss()(a2, b2).backward()
    assert b2.grad is None
