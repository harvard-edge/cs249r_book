"""
Boundary tests for log_softmax with masked (-inf) logits.

log_softmax had zero pytest coverage before this file (only an inline
test existed). Masking invalid classes with -inf before log_softmax is
a real, common pattern (attention padding masks, invalid-action masking
in RL), so both the safe case (some but not all classes masked) and the
degenerate case (every class masked) are worth locking in explicitly.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import log_softmax


class TestLogSoftmaxMasking:
    def test_partial_masking_stays_finite_for_valid_classes(self):
        """Masking some (not all) classes to -inf must not corrupt the
        remaining valid classes' log-probabilities."""
        x = Tensor(np.array([[1.0, -np.inf, 2.0]]))
        result = log_softmax(x, dim=-1)

        assert np.isfinite(result.data[0, 0])
        assert np.isfinite(result.data[0, 2])
        assert result.data[0, 1] == -np.inf

        # exp(log_softmax) must still sum to 1 across the valid classes
        # (the masked class contributes exp(-inf) = 0).
        probs = np.exp(result.data)
        assert np.isclose(probs.sum(), 1.0, atol=1e-6)

    def test_partial_masking_in_batch_does_not_leak_across_rows(self):
        """One row with a masked class must not affect a different row's
        result (per-row max/sum must stay independent). The second row
        holds the larger values on purpose, so a wrong (e.g. global
        instead of per-row) max would leave the first row's own
        (smaller-max) result silently wrong, rather than canceling out
        as it would if both rows shared the same max."""
        first_row = np.array([1.0, -np.inf, 2.0])
        second_row = np.array([0.1, 3.0, 1.5])
        x = Tensor(np.array([first_row, second_row]))
        result = log_softmax(x, dim=-1)

        assert np.all(np.isfinite(result.data[1]))
        finite_mask = np.isfinite(first_row)
        expected_row0_finite = (
            first_row[finite_mask] - np.max(first_row[finite_mask])
            - np.log(np.sum(np.exp(first_row[finite_mask] - np.max(first_row[finite_mask]))))
        )
        np.testing.assert_allclose(result.data[0][finite_mask], expected_row0_finite, atol=1e-5)

    def test_fully_masked_row_matches_pytorch_nan_behavior(self):
        """A row where every class is masked represents an impossible
        distribution. This isn't fixable to something more 'graceful'
        without diverging from the reference implementation: real
        PyTorch's log_softmax on an all -inf row also produces NaN
        (verified directly against torch.log_softmax). This test locks
        in that intentional parity rather than treating it as a bug."""
        x = Tensor(np.array([[-np.inf, -np.inf, -np.inf]]))
        result = log_softmax(x, dim=-1)

        assert np.all(np.isnan(result.data)), (
            "Fully-masked row should match PyTorch's NaN behavior; "
            f"got {result.data}"
        )
