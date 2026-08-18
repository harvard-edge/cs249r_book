"""
Module 04: Losses - Direct log_softmax Tests
=============================================

log_softmax is a standalone helper used internally by CrossEntropyLoss,
but had zero direct test coverage: the only test named "log softmax" in
the suite (tests/integration/test_optimizers_integration.py::test_unit_log_softmax)
actually exercises the Softmax activation class, never calls log_softmax at all.

These tests exercise log_softmax directly, covering:
- exact numerical output against a manual log(softmax(x)) reference
- the default dim=-1 parameter
- keepdims correctness on a multi-row 2D input (dim=-1 and dim=0)
- numerical stability at overflow-triggering magnitudes
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import log_softmax


class TestLogSoftmaxExactValue:
    """Verify log_softmax computes exactly log(softmax(x))."""

    def test_matches_manual_log_softmax_1d(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        result = log_softmax(x, dim=-1)

        exp_x = np.exp(x.data - np.max(x.data))
        expected = np.log(exp_x / np.sum(exp_x))

        np.testing.assert_allclose(result.data, expected, atol=1e-6)

    def test_matches_manual_log_softmax_2d_last_axis(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0], [0.1, 0.5, 0.2]]))
        result = log_softmax(x, dim=-1)

        row_max = np.max(x.data, axis=-1, keepdims=True)
        exp_x = np.exp(x.data - row_max)
        expected = np.log(exp_x / np.sum(exp_x, axis=-1, keepdims=True))

        np.testing.assert_allclose(result.data, expected, atol=1e-6)

    def test_output_exponentiates_to_valid_probability_distribution(self):
        """Each row of exp(log_softmax(x)) must sum to 1."""
        x = Tensor(np.array([[5.0, -3.0, 0.2], [1.0, 1.0, 1.0]]))
        result = log_softmax(x, dim=-1)

        probs = np.exp(result.data)
        np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0], atol=1e-6)


class TestLogSoftmaxDefaultDim:
    """The default dim=-1 must actually operate on the last axis."""

    def test_default_dim_matches_explicit_last_axis(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 0.0]]))

        default_result = log_softmax(x)
        explicit_result = log_softmax(x, dim=-1)

        np.testing.assert_allclose(default_result.data, explicit_result.data, atol=1e-9)

    def test_default_dim_differs_from_axis_zero_on_non_square_input(self):
        """A 3x2 input makes dim=-1 (axis 1) and dim=0 produce different results,
        proving the default genuinely selects the last axis rather than some
        other axis by coincidence."""
        x = Tensor(np.array([[1.0, 5.0], [2.0, 1.0], [3.0, 0.0]]))

        default_result = log_softmax(x)
        axis0_result = log_softmax(x, dim=0)

        assert not np.allclose(default_result.data, axis0_result.data)


class TestLogSoftmaxKeepdims:
    """The internal max/sum reductions must use keepdims=True so broadcasting
    against the original tensor shape stays correct (rather than silently
    producing a wrong-but-same-shape result through NumPy's own broadcasting
    rules on a squeezed array)."""

    def test_output_shape_matches_input_shape(self):
        x = Tensor(np.array([[1.0, 2.0, 3.0], [0.1, 0.5, 0.2]]))
        result = log_softmax(x, dim=-1)
        assert result.data.shape == x.data.shape

    def test_dim_zero_reduction_matches_manual_column_softmax(self):
        """Reducing over axis 0 (columns) on a non-square input exercises a
        case where keepdims correctness (broadcasting max/sum back against
        each column) actually changes the numeric result if broken."""
        x = Tensor(np.array([[1.0, 5.0], [2.0, 1.0], [3.0, 0.0]]))
        result = log_softmax(x, dim=0)

        col_max = np.max(x.data, axis=0, keepdims=True)
        exp_x = np.exp(x.data - col_max)
        expected = np.log(exp_x / np.sum(exp_x, axis=0, keepdims=True))

        np.testing.assert_allclose(result.data, expected, atol=1e-6)


class TestLogSoftmaxNumericalStability:
    """log_softmax must not overflow/underflow for large-magnitude logits."""

    def test_large_positive_logits_do_not_overflow_to_inf_or_nan(self):
        x = Tensor(np.array([[1000.0, 1001.0, 1002.0]]))
        result = log_softmax(x, dim=-1)

        assert np.all(np.isfinite(result.data))
        # Exponentiated, it must still sum to a valid probability distribution.
        np.testing.assert_allclose(np.exp(result.data).sum(axis=-1), [1.0], atol=1e-5)

    def test_large_negative_logits_do_not_underflow_to_nan(self):
        x = Tensor(np.array([[-1000.0, -1001.0, -999.0]]))
        result = log_softmax(x, dim=-1)

        assert np.all(np.isfinite(result.data))
        np.testing.assert_allclose(np.exp(result.data).sum(axis=-1), [1.0], atol=1e-5)
