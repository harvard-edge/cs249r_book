"""
Module 07: Optimizer Update Rules, Checked Against the Formulas
================================================================

Each test drives an optimizer for three steps with a different gradient on
every step, and compares the parameters after each step with the update rule
written out by hand in float64.

WHY THESE TESTS MATTER:
-----------------------
"The weights changed" and "the loss went down" both pass for a wrong update
rule. Dropping Adam's bias correction, or forgetting to carry SGD's velocity
between steps, still trains a small network. Only a value check catches it,
and three steps are needed because the difference shows up from step 2 on
(momentum) or shrinks as beta^t -> 0 (bias correction).
"""

import numpy as np
import pytest

from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import SGD, Adam, AdamW

X0 = np.array([0.5, -1.0, 2.0])
GRADS = [
    np.array([0.1, -0.2, 0.3]),
    np.array([-0.4, 0.1, 0.2]),
    np.array([0.3, 0.5, -0.1]),
]


def _run(optimizer_cls, **kwargs):
    """Apply the three gradients in GRADS and record the parameter after each step."""
    param = Tensor(X0.copy(), requires_grad=True)
    opt = optimizer_cls([param], **kwargs)
    history = []
    for g in GRADS:
        param.grad = g.astype(np.float32)
        opt.step()
        history.append(param.data.astype(np.float64).copy())
    return history


def _assert_matches(history, expected):
    for step, (got, want) in enumerate(zip(history, expected), start=1):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6,
                                   err_msg=f"parameter differs from the formula after step {step}")


class TestSGDUpdateRule:

    def test_plain_sgd(self):
        """x <- x - lr * g"""
        lr = 0.1
        x, expected = X0.copy(), []
        for g in GRADS:
            x = x - lr * g
            expected.append(x.copy())
        _assert_matches(_run(SGD, lr=lr), expected)

    def test_sgd_momentum(self):
        """v <- mu * v + g ;  x <- x - lr * v   (v starts at 0)"""
        lr, mu = 0.1, 0.9
        x, v, expected = X0.copy(), np.zeros_like(X0), []
        for g in GRADS:
            v = mu * v + g
            x = x - lr * v
            expected.append(x.copy())
        _assert_matches(_run(SGD, lr=lr, momentum=mu), expected)

    def test_sgd_momentum_with_weight_decay(self):
        """g <- g + wd * x is applied before the velocity update."""
        lr, mu, wd = 0.1, 0.9, 0.01
        x, v, expected = X0.copy(), np.zeros_like(X0), []
        for g in GRADS:
            g = g + wd * x
            v = mu * v + g
            x = x - lr * v
            expected.append(x.copy())
        _assert_matches(_run(SGD, lr=lr, momentum=mu, weight_decay=wd), expected)


def _adam_reference(lr, b1, b2, eps, wd=0.0, decoupled=False):
    x = X0.copy()
    m = np.zeros_like(X0)
    v = np.zeros_like(X0)
    expected = []
    for t, g in enumerate(GRADS, start=1):
        if wd and not decoupled:
            g = g + wd * x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        if wd and decoupled:
            x = x * (1 - lr * wd)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        expected.append(x.copy())
    return expected


class TestAdamUpdateRule:

    def test_adam_with_bias_correction(self):
        """m, v are EMAs; the step uses m / (1 - b1^t) and v / (1 - b2^t)."""
        lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
        _assert_matches(_run(Adam, lr=lr, betas=(b1, b2), eps=eps),
                        _adam_reference(lr, b1, b2, eps))

    def test_adam_first_step_is_lr_times_sign(self):
        """With bias correction, step 1 moves every coordinate by lr * sign(g)."""
        lr = 0.01
        history = _run(Adam, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        np.testing.assert_allclose(history[0], X0 - lr * np.sign(GRADS[0]), rtol=1e-5, atol=1e-7)

    def test_adam_coupled_weight_decay(self):
        """Adam's weight decay is added to the gradient before the moments."""
        lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.1
        _assert_matches(_run(Adam, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd),
                        _adam_reference(lr, b1, b2, eps, wd=wd))

    def test_adamw_decoupled_weight_decay(self):
        """AdamW shrinks x by (1 - lr * wd) and keeps decay out of the moments."""
        lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.1
        _assert_matches(_run(AdamW, lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd),
                        _adam_reference(lr, b1, b2, eps, wd=wd, decoupled=True))
