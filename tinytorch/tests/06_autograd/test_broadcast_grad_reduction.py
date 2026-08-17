"""
Tests for gradient shape reduction after broadcasting.

_reduce_broadcast_grad() has a hand-written "unit test" inside the source
notebook (src/06_autograd/06_autograd.py), but that test is exported into
tinytorch/core/autograd.py as a plain function, not into the tests/
directory, so pytest never collects or runs it. This file gives the helper
actual, CI-executed coverage.

Tensor.backward() also has its own copy of this reduction logic, applied
whenever an external caller passes a gradient shaped for the broadcasted
result rather than the original tensor. Every existing test reaches
backward() through the internal grad_fn.apply() -> parent.backward() chain,
where the operator's own apply() already reduced the gradient first, so
backward()'s copy of the reduction logic is never actually exercised by
the rest of the suite. Calling backward() directly with an oversized
gradient (a legitimate, documented use of the public API) exercises it.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd, _reduce_broadcast_grad

enable_autograd()


class TestReduceBroadcastGradHelper:
    """Direct coverage for the _reduce_broadcast_grad() helper."""

    def test_removes_leading_dimension(self):
        grad = np.ones((32, 128))
        reduced = _reduce_broadcast_grad(grad, (128,))
        assert reduced.shape == (128,)
        assert np.allclose(reduced, np.full(128, 32.0))

    def test_removes_multiple_leading_dimensions(self):
        grad = np.ones((4, 8, 16))
        reduced = _reduce_broadcast_grad(grad, (16,))
        assert reduced.shape == (16,)
        assert np.allclose(reduced, np.full(16, 32.0))

    def test_collapses_singleton_dimension(self):
        grad = np.ones((10, 5))
        reduced = _reduce_broadcast_grad(grad, (10, 1))
        assert reduced.shape == (10, 1)
        assert np.allclose(reduced, np.full((10, 1), 5.0))

    def test_leaves_non_singleton_dimension_untouched(self):
        """Guards the exact boundary of the size-1 check: a dimension that's
        already >1 in the original shape must not be summed away."""
        grad = np.ones((10, 5))
        reduced = _reduce_broadcast_grad(grad, (10, 5))
        assert reduced.shape == (10, 5)
        assert np.allclose(reduced, np.ones((10, 5)))

    def test_matching_shape_is_a_no_op(self):
        """Guards against an off-by-one in the leading-dimension loop:
        equal ndim must not trigger any summation."""
        grad = np.arange(6, dtype=float).reshape(2, 3)
        reduced = _reduce_broadcast_grad(grad, (2, 3))
        assert reduced.shape == (2, 3)
        assert np.array_equal(reduced, grad)

    def test_combined_leading_and_singleton_reduction(self):
        grad = np.ones((4, 10, 5))
        reduced = _reduce_broadcast_grad(grad, (10, 1))
        assert reduced.shape == (10, 1)
        assert np.allclose(reduced, np.full((10, 1), 20.0))


class TestTensorBackwardBroadcastReduction:
    """Coverage for Tensor.backward()'s own copy of the reduction logic,
    triggered by calling backward() directly with a broadcast-shaped
    gradient rather than going through an operator's apply()."""

    def test_backward_removes_leading_dimension(self):
        x = Tensor(np.zeros(4), requires_grad=True)
        x.backward(np.ones((8, 4)))
        assert x.grad.shape == (4,)
        assert np.allclose(x.grad, np.full(4, 8.0))

    def test_backward_collapses_singleton_dimension(self):
        x = Tensor(np.zeros((3, 1)), requires_grad=True)
        x.backward(np.ones((3, 5)))
        assert x.grad.shape == (3, 1)
        assert np.allclose(x.grad, np.full((3, 1), 5.0))

    def test_backward_matching_shape_accumulates_directly(self):
        x = Tensor(np.zeros((2, 3)), requires_grad=True)
        x.backward(np.ones((2, 3)) * 3.0)
        assert x.grad.shape == (2, 3)
        assert np.allclose(x.grad, np.full((2, 3), 3.0))
