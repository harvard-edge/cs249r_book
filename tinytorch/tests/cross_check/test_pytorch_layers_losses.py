"""
PyTorch cross-check suite: Linear layer and loss functions.

Extends tests/cross_check/test_pytorch_reference.py's coverage (tensor/
autograd ops) to layers and losses, which had never been directly
compared against PyTorch's own implementations. Finite-difference/
mutation testing can only prove internal consistency; only comparison
against an independent reference implementation can catch a bug that's
consistently wrong in both forward and backward.

Requires the optional `cross-check` dependency group; skips cleanly if
torch isn't installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="cross-check tests need the optional 'cross-check' dependency group")

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from tinytorch.core.autograd import enable_autograd

enable_autograd()

rng = np.random.default_rng(0)

ATOL = 1e-5
RTOL = 1e-4


def to_torch(t: Tensor, requires_grad: bool = False) -> "torch.Tensor":
    return torch.tensor(t.data.copy(), dtype=torch.float32, requires_grad=requires_grad)


@pytest.mark.parametrize("draw", range(5))
class TestLinearMatchesPyTorch:
    def test_forward_matches(self, draw):
        in_features, out_features, batch = 5, 3, 4
        x_data = rng.standard_normal((batch, in_features)).astype(np.float32)
        weight_data = rng.standard_normal((in_features, out_features)).astype(np.float32)
        bias_data = rng.standard_normal(out_features).astype(np.float32)

        layer = Linear(in_features, out_features)
        layer.weight.data = weight_data.copy()
        layer.bias.data = bias_data.copy()

        tt_out = layer(Tensor(x_data.copy()))

        torch_linear = torch.nn.Linear(in_features, out_features)
        with torch.no_grad():
            # TinyTorch: (in_features, out_features); PyTorch: (out_features, in_features).
            torch_linear.weight.copy_(torch.tensor(weight_data.T))
            torch_linear.bias.copy_(torch.tensor(bias_data))
        torch_out = torch_linear(torch.tensor(x_data))

        np.testing.assert_allclose(tt_out.data, torch_out.detach().numpy(), atol=ATOL, rtol=RTOL)

    def test_backward_matches(self, draw):
        in_features, out_features, batch = 5, 3, 4
        x_data = rng.standard_normal((batch, in_features)).astype(np.float32)
        weight_data = rng.standard_normal((in_features, out_features)).astype(np.float32)
        bias_data = rng.standard_normal(out_features).astype(np.float32)

        layer = Linear(in_features, out_features)
        layer.weight.data = weight_data.copy()
        layer.bias.data = bias_data.copy()
        layer.weight.requires_grad = True
        layer.bias.requires_grad = True

        x = Tensor(x_data.copy(), requires_grad=True)
        tt_out = layer(x)
        tt_out.sum().backward()

        x_t = torch.tensor(x_data, requires_grad=True)
        torch_linear = torch.nn.Linear(in_features, out_features)
        with torch.no_grad():
            torch_linear.weight.copy_(torch.tensor(weight_data.T))
            torch_linear.bias.copy_(torch.tensor(bias_data))
        torch_out = torch_linear(x_t)
        torch_out.sum().backward()

        np.testing.assert_allclose(x.grad, x_t.grad.numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(layer.weight.grad, torch_linear.weight.grad.numpy().T, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(layer.bias.grad, torch_linear.bias.grad.numpy(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("draw", range(5))
class TestMSELossMatchesPyTorch:
    def test_forward_and_backward_match(self, draw):
        shape = (4, 3)
        pred_data = rng.standard_normal(shape).astype(np.float32)
        target_data = rng.standard_normal(shape).astype(np.float32)

        pred = Tensor(pred_data.copy(), requires_grad=True)
        target = Tensor(target_data.copy())
        tt_loss = MSELoss().forward(pred, target)
        tt_loss.backward()

        pred_t = torch.tensor(pred_data, requires_grad=True)
        target_t = torch.tensor(target_data)
        torch_loss = torch.nn.functional.mse_loss(pred_t, target_t)
        torch_loss.backward()

        np.testing.assert_allclose(tt_loss.data, torch_loss.detach().numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(pred.grad, pred_t.grad.numpy(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("draw", range(5))
class TestCrossEntropyLossMatchesPyTorch:
    def test_forward_and_backward_match(self, draw):
        batch, num_classes = 4, 5
        logits_data = rng.standard_normal((batch, num_classes)).astype(np.float32)
        targets_data = rng.integers(0, num_classes, size=batch)

        logits = Tensor(logits_data.copy(), requires_grad=True)
        targets = Tensor(targets_data.copy())
        tt_loss = CrossEntropyLoss().forward(logits, targets)
        tt_loss.backward()

        logits_t = torch.tensor(logits_data, requires_grad=True)
        targets_t = torch.tensor(targets_data, dtype=torch.long)
        torch_loss = torch.nn.functional.cross_entropy(logits_t, targets_t)
        torch_loss.backward()

        np.testing.assert_allclose(tt_loss.data, torch_loss.detach().numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(logits.grad, logits_t.grad.numpy(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("draw", range(5))
class TestBCELossMatchesPyTorch:
    def test_forward_and_backward_match(self, draw):
        shape = (4,)
        # Interior points, away from the clip boundary (matches the
        # existing finite-difference test's rationale).
        pred_data = rng.uniform(0.2, 0.8, size=shape).astype(np.float32)
        target_data = rng.integers(0, 2, size=shape).astype(np.float32)

        pred = Tensor(pred_data.copy(), requires_grad=True)
        target = Tensor(target_data.copy())
        tt_loss = BinaryCrossEntropyLoss().forward(pred, target)
        tt_loss.backward()

        pred_t = torch.tensor(pred_data, requires_grad=True)
        target_t = torch.tensor(target_data)
        torch_loss = torch.nn.functional.binary_cross_entropy(pred_t, target_t)
        torch_loss.backward()

        np.testing.assert_allclose(tt_loss.data, torch_loss.detach().numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(pred.grad, pred_t.grad.numpy(), atol=1e-3, rtol=1e-3)
