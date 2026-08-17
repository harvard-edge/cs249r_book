"""
Exact-behavior tests for Dropout, found via mutation testing to have
essentially no coverage (only one loose existing test, elsewhere in the
suite, that accepts "output changed OR equals x*2" as sufficient, not
strong enough to catch a wrong scale factor or a wrong drop-probability
comparison).

Covers _should_apply_dropout's boundary conditions, _generate_dropout_mask's
exact scaling, and forward()'s p=1.0 special case, none previously
exercised precisely.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dropout


class TestShouldApplyDropout:
    def test_false_when_not_training(self):
        dropout = Dropout(p=0.5)
        assert dropout._should_apply_dropout(training=False) is False

    def test_false_when_p_is_zero_even_during_training(self):
        dropout = Dropout(p=0.0)
        assert dropout._should_apply_dropout(training=True) is False

    def test_true_when_training_and_p_positive(self):
        dropout = Dropout(p=0.5)
        assert dropout._should_apply_dropout(training=True) is True

    def test_false_when_not_training_even_with_positive_p(self):
        """Both conditions must hold; training=False alone must block
        dropout regardless of p."""
        dropout = Dropout(p=0.9)
        assert dropout._should_apply_dropout(training=False) is False


class TestGenerateDropoutMaskExactScaling:
    def test_kept_elements_scaled_by_exactly_one_over_keep_prob(self):
        """Every nonzero mask element must equal exactly 1/(1-p), not any
        other scale factor."""
        p = 0.25
        dropout = Dropout(p=p)
        mask = dropout._generate_dropout_mask((1000,))

        nonzero = mask.data[mask.data != 0]
        assert len(nonzero) > 0, "Expected at least some kept elements at p=0.25"
        expected_scale = 1.0 / (1.0 - p)
        np.testing.assert_allclose(nonzero, expected_scale, atol=1e-6)

    def test_dropped_elements_are_exactly_zero(self):
        p = 0.5
        dropout = Dropout(p=p)
        mask = dropout._generate_dropout_mask((1000,))

        zeros = mask.data[mask.data == 0]
        assert len(zeros) > 0, "Expected at least some dropped elements at p=0.5"

    def test_drop_rate_is_statistically_close_to_p(self):
        """Over many draws, the fraction of dropped elements should be
        close to p (catches a wrong comparison direction, e.g. keeping
        keep_prob fraction instead of dropping it, or vice versa)."""
        p = 0.7
        dropout = Dropout(p=p)
        mask = dropout._generate_dropout_mask((20000,))

        observed_drop_rate = np.mean(mask.data == 0)
        assert abs(observed_drop_rate - p) < 0.02, (
            f"Observed drop rate {observed_drop_rate} too far from p={p}"
        )

    def test_mask_shape_matches_requested_shape(self):
        dropout = Dropout(p=0.5)
        mask = dropout._generate_dropout_mask((3, 4, 5))
        assert mask.data.shape == (3, 4, 5)


class TestDropoutForward:
    def test_p_zero_is_identity_during_training(self):
        dropout = Dropout(p=0.0)
        x = Tensor(np.ones((10, 10)))

        output = dropout(x, training=True)

        np.testing.assert_array_equal(output.data, x.data)

    def test_p_one_zeros_everything(self):
        dropout = Dropout(p=1.0)
        x = Tensor(np.ones((10, 10)))

        output = dropout(x, training=True)

        np.testing.assert_array_equal(output.data, np.zeros((10, 10)))

    def test_eval_mode_passes_through_unchanged_regardless_of_p(self):
        dropout = Dropout(p=0.9)
        x = Tensor(np.ones((10, 10)))

        output = dropout(x, training=False)

        np.testing.assert_array_equal(output.data, x.data)

    def test_kept_values_during_training_are_exactly_scaled(self):
        """Every nonzero output element, for an all-ones input, must
        equal exactly the scale factor 1/(1-p)."""
        p = 0.3
        dropout = Dropout(p=p)
        x = Tensor(np.ones((1000,)))

        output = dropout(x, training=True)

        nonzero = output.data[output.data != 0]
        expected_scale = 1.0 / (1.0 - p)
        np.testing.assert_allclose(nonzero, expected_scale, atol=1e-6)
