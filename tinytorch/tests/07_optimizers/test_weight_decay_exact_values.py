"""
Exact-value tests for weight decay in Adam and AdamW.

Adam and AdamW apply weight decay in deliberately different ways, and the
difference is the entire reason AdamW exists:

- Adam mixes weight decay into the gradient before the moment update:
  grad = grad + weight_decay * param, so decay gets scaled by the
  adaptive learning rate along with the gradient (a known issue with
  plain Adam + L2 regularization).
- AdamW applies weight decay as a separate multiplicative step on the
  parameter after the gradient update: param *= (1 - lr * weight_decay),
  decoupled from the adaptive scaling.

Before this file, weight_decay had no exact-value test anywhere (pytest
or inline) for either optimizer, only "the parameter changed" or "Adam
and AdamW disagree" checks. That's not enough to catch a wrong formula:
hand-mutating AdamW's decoupled decay step from `(1 - lr * weight_decay)`
to `(1 + lr * weight_decay)` (growing weights instead of shrinking them)
passes every existing test, inline included.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import Adam, AdamW
from tinytorch.core.autograd import enable_autograd

enable_autograd()


class TestAdamWeightDecay:
    def test_weight_decay_mixed_into_gradient(self):
        """Adam: grad_used = grad + weight_decay * param, then the usual
        Adam moment update and parameter step run on grad_used."""
        param_data = np.array([1.0, 2.0])
        grad_data = np.array([0.1, 0.2])
        weight_decay = 0.1
        lr = 0.01
        eps = 1e-8

        param = Tensor(param_data.copy(), requires_grad=True)
        optimizer = Adam([param], lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=weight_decay)
        param.grad = Tensor(grad_data.copy())
        optimizer.step()

        effective_grad = grad_data + weight_decay * param_data
        m_hat = effective_grad  # bias-corrected EMA equals the gradient itself at step 1
        v_hat = effective_grad ** 2
        expected = param_data - lr * m_hat / (np.sqrt(v_hat) + eps)

        np.testing.assert_allclose(param.data, expected, rtol=1e-5)


class TestAdamWWeightDecay:
    def test_weight_decay_applied_after_gradient_update_not_mixed_in(self):
        """AdamW: the moment update uses the PURE gradient (no decay mixed
        in), then decay is applied as param *= (1 - lr * weight_decay)
        afterward."""
        param_data = np.array([1.0, 2.0])
        grad_data = np.array([0.1, 0.2])
        weight_decay = 0.1
        lr = 0.01
        eps = 1e-8

        param = Tensor(param_data.copy(), requires_grad=True)
        optimizer = AdamW([param], lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=weight_decay)
        param.grad = Tensor(grad_data.copy())
        optimizer.step()

        # Pure gradient (not grad + weight_decay * param) drives the moment update.
        m_hat = grad_data
        v_hat = grad_data ** 2
        after_gradient_update = param_data - lr * m_hat / (np.sqrt(v_hat) + eps)
        expected = after_gradient_update * (1 - lr * weight_decay)

        np.testing.assert_allclose(param.data, expected, rtol=1e-5)

    def test_zero_weight_decay_is_plain_adam_equivalent(self):
        """With weight_decay=0, AdamW's decoupled step is skipped entirely,
        so it should match Adam step-for-step (both reduce to the same
        undamped adaptive update)."""
        param_data = np.array([1.0, 2.0])
        grad_data = np.array([0.1, 0.2])

        param_adam = Tensor(param_data.copy(), requires_grad=True)
        param_adamw = Tensor(param_data.copy(), requires_grad=True)
        adam = Adam([param_adam], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        adamw = AdamW([param_adamw], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

        param_adam.grad = Tensor(grad_data.copy())
        param_adamw.grad = Tensor(grad_data.copy())
        adam.step()
        adamw.step()

        np.testing.assert_allclose(param_adam.data, param_adamw.data, rtol=1e-10)
