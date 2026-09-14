"""Graph lifetime and incoming-gradient contracts (regressions, 2026-09-11)."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
import tinytorch.core.autograd


def test_released_intermediate_is_rejected_before_any_gradient_changes():
    x = Tensor([2.0], requires_grad=True)
    h = x * 2
    (h * 3).sum().backward()
    branch = (h * 4).sum()
    with pytest.raises(RuntimeError, match="released"):
        branch.backward()
    np.testing.assert_array_equal(x.grad, [6.0])
    np.testing.assert_array_equal(h.grad, [3.0])
    assert branch.grad is None


def test_retained_intermediate_accumulates_both_branches():
    x = Tensor([2.0], requires_grad=True)
    h = x * 2
    (h * 3).sum().backward(retain_graph=True)
    (h * 4).sum().backward()
    np.testing.assert_array_equal(x.grad, [14.0])
    with pytest.raises(RuntimeError, match="released"):
        h.backward()


@pytest.mark.parametrize("seed", [1.0, [1.0], np.ones((3, 2)), Tensor([[1.0, 1.0]])])
def test_incoming_gradient_must_match_output_shape(seed):
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = x * 2
    with pytest.raises(ValueError, match="shape"):
        y.backward(seed)
    assert x.grad is None and y.grad is None
    # Rejection must leave the graph usable with a valid seed.
    y.backward(Tensor([3.0, 4.0]))
    np.testing.assert_array_equal(x.grad, [6.0, 8.0])


def test_scalar_seed_and_recomputed_graph_remain_supported():
    x = Tensor(2.0, requires_grad=True)
    (x * 2).backward(3.0)
    (x * 4).backward()
    np.testing.assert_array_equal(x.grad, 10.0)
