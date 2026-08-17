"""
PyTorch cross-check suite: Conv2d and scaled_dot_product_attention.

Extends tests/cross_check/test_pytorch_reference.py's coverage to
convolutions and attention, which had never been directly compared
against PyTorch's own implementations.

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
from tinytorch.core.spatial import Conv2d
from tinytorch.core.attention import scaled_dot_product_attention
from tinytorch.core.autograd import enable_autograd

enable_autograd()

rng = np.random.default_rng(0)

ATOL = 1e-4
RTOL = 1e-3


@pytest.mark.parametrize("draw", range(5))
class TestConv2dMatchesPyTorch:
    def test_forward_matches(self, draw):
        in_channels, out_channels, kernel_size = 3, 4, 3
        batch, h, w = 2, 8, 8
        x_data = rng.standard_normal((batch, in_channels, h, w)).astype(np.float32)
        weight_data = rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        bias_data = rng.standard_normal(out_channels).astype(np.float32)

        conv = Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        conv.weight.data = weight_data.copy()
        conv.bias.data = bias_data.copy()

        tt_out = conv(Tensor(x_data.copy()))

        torch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        with torch.no_grad():
            torch_conv.weight.copy_(torch.tensor(weight_data))
            torch_conv.bias.copy_(torch.tensor(bias_data))
        torch_out = torch_conv(torch.tensor(x_data))

        np.testing.assert_allclose(tt_out.data, torch_out.detach().numpy(), atol=ATOL, rtol=RTOL)

    def test_backward_matches(self, draw):
        in_channels, out_channels, kernel_size = 3, 4, 3
        batch, h, w = 2, 8, 8
        x_data = rng.standard_normal((batch, in_channels, h, w)).astype(np.float32)
        weight_data = rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        bias_data = rng.standard_normal(out_channels).astype(np.float32)

        conv = Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        conv.weight.data = weight_data.copy()
        conv.bias.data = bias_data.copy()

        x = Tensor(x_data.copy(), requires_grad=True)
        tt_out = conv(x)
        tt_out.sum().backward()

        x_t = torch.tensor(x_data, requires_grad=True)
        torch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        with torch.no_grad():
            torch_conv.weight.copy_(torch.tensor(weight_data))
            torch_conv.bias.copy_(torch.tensor(bias_data))
        torch_out = torch_conv(x_t)
        torch_out.sum().backward()

        np.testing.assert_allclose(x.grad, x_t.grad.numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(conv.weight.grad, torch_conv.weight.grad.numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(conv.bias.grad, torch_conv.bias.grad.numpy(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("draw", range(5))
class TestScaledDotProductAttentionMatchesPyTorch:
    def test_forward_matches(self, draw):
        batch, seq_len, d_model = 2, 4, 8
        q_data = rng.standard_normal((batch, seq_len, d_model)).astype(np.float32)
        k_data = rng.standard_normal((batch, seq_len, d_model)).astype(np.float32)
        v_data = rng.standard_normal((batch, seq_len, d_model)).astype(np.float32)

        Q = Tensor(q_data.copy(), requires_grad=True)
        K = Tensor(k_data.copy(), requires_grad=True)
        V = Tensor(v_data.copy(), requires_grad=True)
        tt_out, tt_weights = scaled_dot_product_attention(Q, K, V)

        q_t = torch.tensor(q_data, requires_grad=True)
        k_t = torch.tensor(k_data, requires_grad=True)
        v_t = torch.tensor(v_data, requires_grad=True)
        scale = 1.0 / (d_model ** 0.5)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
        torch_weights = torch.softmax(scores, dim=-1)
        torch_out = torch.matmul(torch_weights, v_t)

        np.testing.assert_allclose(tt_out.data, torch_out.detach().numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(tt_weights.data, torch_weights.detach().numpy(), atol=ATOL, rtol=RTOL)

    def test_backward_matches(self, draw):
        batch, seq_len, d_model = 2, 4, 8
        q_data = rng.standard_normal((batch, seq_len, d_model)).astype(np.float32)
        k_data = rng.standard_normal((batch, seq_len, d_model)).astype(np.float32)
        v_data = rng.standard_normal((batch, seq_len, d_model)).astype(np.float32)

        Q = Tensor(q_data.copy(), requires_grad=True)
        K = Tensor(k_data.copy(), requires_grad=True)
        V = Tensor(v_data.copy(), requires_grad=True)
        tt_out, _ = scaled_dot_product_attention(Q, K, V)
        tt_out.sum().backward()

        q_t = torch.tensor(q_data, requires_grad=True)
        k_t = torch.tensor(k_data, requires_grad=True)
        v_t = torch.tensor(v_data, requires_grad=True)
        scale = 1.0 / (d_model ** 0.5)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
        torch_weights = torch.softmax(scores, dim=-1)
        torch_out = torch.matmul(torch_weights, v_t)
        torch_out.sum().backward()

        np.testing.assert_allclose(Q.grad, q_t.grad.numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(K.grad, k_t.grad.numpy(), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(V.grad, v_t.grad.numpy(), atol=ATOL, rtol=RTOL)
