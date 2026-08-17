"""
Cross-check tests: TinyTorch vs. real PyTorch
==============================================

Every other test file in this repo verifies TinyTorch is internally
consistent: hand-picked expected values, or (in tests/06_autograd/
test_gradient_correctness.py) finite-difference gradient checks that
confirm backward() agrees with TinyTorch's own forward() function.

Neither of those can catch a bug where TinyTorch's forward pass computes
something subtly different from the operation it's supposed to implement,
if the forward function itself is wrong in a self-consistent way, finite
differences will happily confirm the (wrong) gradient matches the (wrong)
forward pass.

This file closes that gap the same way SQLite's SLT harness cross-checks
query results against independent SQL engines: by comparing TinyTorch's
output directly against real PyTorch on identical inputs, both for forward
values and for backward gradients. See testing_tinytorch.md for the full
rationale.

Requires torch (CPU) as a test-only dependency, it is never imported by
the tinytorch package itself, only by this test file.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

torch = pytest.importorskip("torch", reason="torch is a test-only dependency for cross-checking")

from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, GELU, Softmax
from tinytorch.core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from tinytorch.core.autograd import enable_autograd

enable_autograd()

rng = np.random.default_rng(1234)

FORWARD_RTOL, FORWARD_ATOL = 1e-4, 1e-5
GRAD_RTOL, GRAD_ATOL = 1e-3, 1e-4


def rand_array(shape, low=-3.0, high=3.0):
    return rng.uniform(low, high, size=shape).astype(np.float32)


def to_torch(array, requires_grad=False):
    return torch.tensor(array, dtype=torch.float32, requires_grad=requires_grad)


def assert_forward_matches(tt_result, torch_result, rtol=FORWARD_RTOL, atol=FORWARD_ATOL):
    np.testing.assert_allclose(
        tt_result.data, torch_result.detach().numpy(), rtol=rtol, atol=atol
    )


def assert_grad_matches(tt_tensor, torch_tensor, rtol=GRAD_RTOL, atol=GRAD_ATOL):
    assert tt_tensor.grad is not None, "TinyTorch tensor received no gradient"
    assert torch_tensor.grad is not None, "PyTorch tensor received no gradient"
    np.testing.assert_allclose(
        tt_tensor.grad, torch_tensor.grad.numpy(), rtol=rtol, atol=atol
    )


# -----------------------------------------------------------------------
# Arithmetic ops: forward values
# -----------------------------------------------------------------------

@pytest.fixture(autouse=True, params=range(5))
def reseed_per_draw(request):
    """
    Reseed the module-level rng before every test, parametrized over 5
    seeds. This makes every test in this file run 5 times against 5
    independent random draws, rather than relying on a single fixed draw
    to represent "random inputs". A single draw could get lucky and miss
    a shape- or value-dependent bug that a different draw would catch.
    """
    global rng
    rng = np.random.default_rng(1000 + request.param)


class TestArithmeticForwardMatchesPyTorch:

    def test_add_same_shape(self):
        a_np, b_np = rand_array((4, 5)), rand_array((4, 5))
        result = Tensor(a_np) + Tensor(b_np)
        assert_forward_matches(result, to_torch(a_np) + to_torch(b_np))

    def test_add_broadcasting(self):
        a_np, b_np = rand_array((4, 5)), rand_array((5,))
        result = Tensor(a_np) + Tensor(b_np)
        assert_forward_matches(result, to_torch(a_np) + to_torch(b_np))

    def test_sub_broadcasting(self):
        a_np, b_np = rand_array((3, 4, 5)), rand_array((4, 5))
        result = Tensor(a_np) - Tensor(b_np)
        assert_forward_matches(result, to_torch(a_np) - to_torch(b_np))

    def test_mul_broadcasting(self):
        a_np, b_np = rand_array((4, 5)), rand_array((1, 5))
        result = Tensor(a_np) * Tensor(b_np)
        assert_forward_matches(result, to_torch(a_np) * to_torch(b_np))

    def test_div_broadcasting(self):
        a_np = rand_array((4, 5))
        b_np = rand_array((5,), low=0.5, high=3.0)  # keep denominators away from 0
        result = Tensor(a_np) / Tensor(b_np)
        assert_forward_matches(result, to_torch(a_np) / to_torch(b_np))


# -----------------------------------------------------------------------
# Arithmetic ops: gradients
# -----------------------------------------------------------------------

class TestArithmeticGradientsMatchPyTorch:

    def test_add_broadcasting_gradients(self):
        a_np, b_np = rand_array((4, 5)), rand_array((5,))
        a_tt, b_tt = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
        (a_tt + b_tt).sum().backward()

        a_pt, b_pt = to_torch(a_np, True), to_torch(b_np, True)
        (a_pt + b_pt).sum().backward()

        assert_grad_matches(a_tt, a_pt)
        assert_grad_matches(b_tt, b_pt)

    def test_sub_broadcasting_gradients(self):
        a_np, b_np = rand_array((3, 4, 5)), rand_array((4, 5))
        a_tt, b_tt = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
        (a_tt - b_tt).sum().backward()

        a_pt, b_pt = to_torch(a_np, True), to_torch(b_np, True)
        (a_pt - b_pt).sum().backward()

        assert_grad_matches(a_tt, a_pt)
        assert_grad_matches(b_tt, b_pt)

    def test_mul_broadcasting_gradients(self):
        a_np, b_np = rand_array((4, 5)), rand_array((1, 5))
        a_tt, b_tt = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
        (a_tt * b_tt).sum().backward()

        a_pt, b_pt = to_torch(a_np, True), to_torch(b_np, True)
        (a_pt * b_pt).sum().backward()

        assert_grad_matches(a_tt, a_pt)
        assert_grad_matches(b_tt, b_pt)

    def test_div_broadcasting_gradients(self):
        a_np = rand_array((4, 5))
        b_np = rand_array((5,), low=0.5, high=3.0)
        a_tt, b_tt = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
        (a_tt / b_tt).sum().backward()

        a_pt, b_pt = to_torch(a_np, True), to_torch(b_np, True)
        (a_pt / b_pt).sum().backward()

        assert_grad_matches(a_tt, a_pt)
        assert_grad_matches(b_tt, b_pt)


# -----------------------------------------------------------------------
# Matmul: forward and gradients
# -----------------------------------------------------------------------

class TestMatmulMatchesPyTorch:

    def test_matmul_2d_forward(self):
        a_np, b_np = rand_array((6, 4)), rand_array((4, 3))
        result = Tensor(a_np).matmul(Tensor(b_np))
        assert_forward_matches(result, to_torch(a_np) @ to_torch(b_np))

    def test_matmul_2d_gradients(self):
        a_np, b_np = rand_array((6, 4)), rand_array((4, 3))
        a_tt, b_tt = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
        a_tt.matmul(b_tt).sum().backward()

        a_pt, b_pt = to_torch(a_np, True), to_torch(b_np, True)
        (a_pt @ b_pt).sum().backward()

        assert_grad_matches(a_tt, a_pt)
        assert_grad_matches(b_tt, b_pt)

    def test_matmul_vector_matrix_forward(self):
        a_np, b_np = rand_array((4,)), rand_array((4, 3))
        result = Tensor(a_np).matmul(Tensor(b_np))
        assert_forward_matches(result, to_torch(a_np) @ to_torch(b_np))


# -----------------------------------------------------------------------
# Shape ops: forward and gradients
# -----------------------------------------------------------------------

class TestShapeOpsMatchPyTorch:

    def test_reshape_forward(self):
        a_np = rand_array((3, 4))
        result = Tensor(a_np).reshape(4, 3)
        assert_forward_matches(result, to_torch(a_np).reshape(4, 3))

    def test_reshape_gradients(self):
        a_np = rand_array((3, 4))
        a_tt = Tensor(a_np, requires_grad=True)
        a_tt.reshape(4, 3).sum().backward()

        a_pt = to_torch(a_np, True)
        a_pt.reshape(4, 3).sum().backward()

        assert_grad_matches(a_tt, a_pt)

    def test_transpose_default_forward(self):
        a_np = rand_array((3, 4))
        result = Tensor(a_np).transpose()
        assert_forward_matches(result, to_torch(a_np).t())

    def test_transpose_explicit_dims_forward(self):
        a_np = rand_array((2, 3, 4))
        result = Tensor(a_np).transpose(0, 2)
        assert_forward_matches(result, to_torch(a_np).transpose(0, 2))

    def test_transpose_gradients(self):
        a_np = rand_array((3, 4))
        a_tt = Tensor(a_np, requires_grad=True)
        a_tt.transpose().sum().backward()

        a_pt = to_torch(a_np, True)
        a_pt.t().sum().backward()

        assert_grad_matches(a_tt, a_pt)


# -----------------------------------------------------------------------
# Reductions: sum (differentiable), mean/max (forward-only in TinyTorch)
# -----------------------------------------------------------------------

class TestReductionsMatchPyTorch:

    def test_sum_all_forward(self):
        a_np = rand_array((4, 5))
        result = Tensor(a_np).sum()
        assert_forward_matches(result, to_torch(a_np).sum())

    def test_sum_axis_forward(self):
        a_np = rand_array((4, 5))
        result = Tensor(a_np).sum(axis=0)
        assert_forward_matches(result, to_torch(a_np).sum(dim=0))

    def test_sum_axis_gradients(self):
        a_np = rand_array((4, 5))
        a_tt = Tensor(a_np, requires_grad=True)
        a_tt.sum(axis=0).sum().backward()

        a_pt = to_torch(a_np, True)
        a_pt.sum(dim=0).sum().backward()

        assert_grad_matches(a_tt, a_pt)

    def test_mean_forward(self):
        # mean() is forward-only in TinyTorch (no MeanBackward exists),
        # so only the value is cross-checked here, not gradients.
        a_np = rand_array((4, 5))
        result = Tensor(a_np).mean(axis=1)
        assert_forward_matches(result, to_torch(a_np).mean(dim=1))

    def test_max_forward(self):
        # max() is likewise forward-only in TinyTorch.
        a_np = rand_array((4, 5))
        result = Tensor(a_np).max(axis=1)
        assert_forward_matches(result, to_torch(a_np).max(dim=1).values)


# -----------------------------------------------------------------------
# Activations: forward and gradients
# -----------------------------------------------------------------------

class TestActivationsMatchPyTorch:

    def test_relu_forward_and_gradients(self):
        a_np = rand_array((5, 6))
        a_tt = Tensor(a_np, requires_grad=True)
        result_tt = ReLU()(a_tt)
        result_tt.sum().backward()

        a_pt = to_torch(a_np, True)
        result_pt = torch.relu(a_pt)
        result_pt.sum().backward()

        assert_forward_matches(result_tt, result_pt)
        assert_grad_matches(a_tt, a_pt)

    def test_sigmoid_forward_and_gradients(self):
        a_np = rand_array((5, 6))
        a_tt = Tensor(a_np, requires_grad=True)
        result_tt = Sigmoid()(a_tt)
        result_tt.sum().backward()

        a_pt = to_torch(a_np, True)
        result_pt = torch.sigmoid(a_pt)
        result_pt.sum().backward()

        assert_forward_matches(result_tt, result_pt)
        assert_grad_matches(a_tt, a_pt)

    def test_tanh_forward_and_gradients(self):
        a_np = rand_array((5, 6))
        a_tt = Tensor(a_np, requires_grad=True)
        result_tt = Tanh()(a_tt)
        result_tt.sum().backward()

        a_pt = to_torch(a_np, True)
        result_pt = torch.tanh(a_pt)
        result_pt.sum().backward()

        assert_forward_matches(result_tt, result_pt)
        assert_grad_matches(a_tt, a_pt)

    def test_softmax_forward_and_gradients(self):
        a_np = rand_array((4, 6))
        a_tt = Tensor(a_np, requires_grad=True)
        result_tt = Softmax()(a_tt, dim=-1)
        result_tt.sum().backward()

        a_pt = to_torch(a_np, True)
        result_pt = torch.softmax(a_pt, dim=-1)
        result_pt.sum().backward()

        assert_forward_matches(result_tt, result_pt)
        assert_grad_matches(a_tt, a_pt)

    def test_gelu_forward_and_gradients_against_sigmoid_approximation(self):
        """
        TinyTorch's GELU intentionally uses the cheap sigmoid approximation
        (x * sigmoid(1.702 * x)), documented directly in its docstring, not
        PyTorch's default erf-based GELU or its tanh-approximate variant.
        Comparing against torch.nn.functional.gelu directly would fail here
        by design, not due to a bug, so the reference is built from the
        same sigmoid formula expressed in torch ops instead. This still
        catches a real class of bug (e.g. a wrong constant, a sign error,
        or an incorrect derivative), just not "uses a different GELU
        approximation than PyTorch's default," which is expected.
        """
        a_np = rand_array((5, 6))
        a_tt = Tensor(a_np, requires_grad=True)
        result_tt = GELU()(a_tt)
        result_tt.sum().backward()

        a_pt = to_torch(a_np, True)
        result_pt = a_pt * torch.sigmoid(1.702 * a_pt)
        result_pt.sum().backward()

        assert_forward_matches(result_tt, result_pt)
        assert_grad_matches(a_tt, a_pt)


# -----------------------------------------------------------------------
# Losses: forward and gradients
# -----------------------------------------------------------------------

class TestLossesMatchPyTorch:

    def test_mse_loss_forward_and_gradients(self):
        pred_np = rand_array((8,))
        target_np = rand_array((8,))
        pred_tt = Tensor(pred_np, requires_grad=True)
        target_tt = Tensor(target_np)
        loss_tt = MSELoss()(pred_tt, target_tt)
        loss_tt.backward()

        pred_pt = to_torch(pred_np, True)
        target_pt = to_torch(target_np)
        loss_pt = torch.nn.functional.mse_loss(pred_pt, target_pt)
        loss_pt.backward()

        assert_forward_matches(loss_tt, loss_pt)
        assert_grad_matches(pred_tt, pred_pt)

    def test_cross_entropy_loss_forward_and_gradients(self):
        num_classes = 4
        logits_np = rand_array((6, num_classes))
        targets_np = rng.integers(0, num_classes, size=6)

        logits_tt = Tensor(logits_np, requires_grad=True)
        targets_tt = Tensor(targets_np.astype(np.float32))
        loss_tt = CrossEntropyLoss()(logits_tt, targets_tt)
        loss_tt.backward()

        logits_pt = to_torch(logits_np, True)
        targets_pt = torch.tensor(targets_np, dtype=torch.long)
        loss_pt = torch.nn.functional.cross_entropy(logits_pt, targets_pt)
        loss_pt.backward()

        assert_forward_matches(loss_tt, loss_pt)
        assert_grad_matches(logits_tt, logits_pt)

    def test_binary_cross_entropy_loss_forward_and_gradients(self):
        # Keep predictions away from the 0/1 boundary, both TinyTorch and
        # PyTorch clip near the boundary for numerical stability, but not
        # necessarily with the same epsilon, so values near the edge can
        # legitimately disagree slightly without either being wrong.
        pred_np = rng.uniform(0.15, 0.85, size=(8,)).astype(np.float32)
        target_np = rng.integers(0, 2, size=8).astype(np.float32)

        pred_tt = Tensor(pred_np, requires_grad=True)
        target_tt = Tensor(target_np)
        loss_tt = BinaryCrossEntropyLoss()(pred_tt, target_tt)
        loss_tt.backward()

        pred_pt = to_torch(pred_np, True)
        target_pt = to_torch(target_np)
        loss_pt = torch.nn.functional.binary_cross_entropy(pred_pt, target_pt)
        loss_pt.backward()

        assert_forward_matches(loss_tt, loss_pt)
        assert_grad_matches(pred_tt, pred_pt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
