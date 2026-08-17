"""
Exact-value and boundary tests closing three gaps found via mutation
testing in tinytorch.core.activations:

1. GELU had zero pytest coverage at all (only a loose-bound inline test
   that pytest never collects), so its exact formula
   (x * sigmoid(1.702 * x)) was never verified precisely: mutating the
   1.702 constant, or either multiplication to addition, all survived.
2. Softmax's default dim=-1 parameter was never exercised on a
   multi-dimensional, non-symmetric tensor, so mutating the default to
   +1 or -2 survived undetected.
3. Sigmoid's numerically-stable branch selection (x_data >= 0) was
   never tested directly through activations.Sigmoid at a magnitude
   where the naive/unstable branch would actually overflow; the
   autograd-tracked path has this coverage (see the earlier
   tracked_sigmoid_forward fix) but the plain activations.Sigmoid.forward
   path does not.
"""

import numpy as np
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.activations import GELU, Sigmoid, Softmax
from tinytorch.core.tensor import Tensor


class TestGELUExactValues:
    def _expected(self, x):
        return x * (1.0 / (1.0 + np.exp(-1.702 * x)))

    def test_matches_exact_formula(self):
        x_data = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
        gelu = GELU()

        result = gelu(Tensor(x_data.copy()))

        expected = self._expected(x_data)
        np.testing.assert_allclose(result.data, expected, atol=1e-5, rtol=1e-4)

    def test_zero_is_exactly_zero(self):
        gelu = GELU()
        result = gelu(Tensor(np.array([0.0])))
        assert result.data[0] == 0.0


class TestSoftmaxDefaultDim:
    def test_call_default_dim_normalizes_last_axis_of_3d_tensor(self):
        """A 3D, non-symmetric-shape tensor distinguishes dim=-1 from
        dim=+1 or dim=-2: only the correct axis sums to 1 per slice."""
        x_data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        softmax = Softmax()

        result = softmax(Tensor(x_data.copy()))  # dim defaults to -1

        row_sums = result.data.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones((2, 3)), atol=1e-5)
        # Sums along the wrong axis must NOT be 1 (proves dim=-1 was
        # actually used, not some other axis).
        assert not np.allclose(result.data.sum(axis=1), np.ones((2, 4)), atol=1e-5)

    def test_forward_default_dim_normalizes_last_axis_of_3d_tensor(self):
        """Softmax.__call__ has its own dim=-1 default and always passes
        an explicit value through to forward(), so forward()'s own
        dim=-1 default (defined in 02_activations.py) is never exercised
        by going through __call__ (the test above).

        Note this test can't actually reach 02_activations.py's own
        default either: enable_autograd() unconditionally replaces
        Softmax.forward with tracked_softmax_forward (defined
        separately in 06_autograd.py, with its own independently
        hardcoded dim=-1 default), the same monkeypatch-shadowing
        architecture already found elsewhere in this codebase. Both
        defaults currently happen to agree, so there's no live bug, but
        02_activations.py's own default is structurally unreachable
        once autograd is enabled (which happens on every package
        import): if the two were ever changed independently, nothing
        would catch the divergence. This test exercises whichever
        implementation is actually reachable through forward(), which
        is the best available regression guard given that constraint.
        """
        x_data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        softmax = Softmax()

        result = softmax.forward(Tensor(x_data.copy()))  # dim defaults to -1

        row_sums = result.data.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones((2, 3)), atol=1e-5)


class TestSigmoidStableBranchDirect:
    def test_large_positive_input_does_not_overflow(self):
        """Exercises activations.Sigmoid.forward() directly (not via the
        autograd-tracked path), at a magnitude where the unstable branch
        (exp(-x) for very negative x, or exp(x) for very positive x on
        the wrong side) would overflow."""
        sigmoid = Sigmoid()
        x = Tensor(np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0], dtype=np.float32))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = sigmoid(x)

        assert np.all(np.isfinite(result.data))
        assert abs(result.data[0] - 0.0) < 1e-6
        assert abs(result.data[-1] - 1.0) < 1e-6
