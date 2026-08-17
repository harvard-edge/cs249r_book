"""
Numerical edge-case tests for LayerNorm.

LayerNorm divides by sqrt(variance + eps). For a sample where every value
along the normalized dimension is identical (a real scenario: a padding
token's embedding, or a degenerate all-zero row), variance is exactly 0,
so the classic version of this formula (divide by sqrt(variance), no eps)
divides by zero. This module's implementation already guards against it
by adding eps before the sqrt, but that behavior had no test protecting
it: nothing exercises a constant-valued input through forward or backward
to confirm the result stays finite rather than NaN/Inf.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.transformers import LayerNorm

enable_autograd()


class TestLayerNormConstantInput:
    def test_forward_constant_input_produces_no_nan_or_inf(self):
        """A sample with identical values along the normalized axis has
        variance exactly 0. Forward must stay finite, not NaN/Inf."""
        ln = LayerNorm(8)
        x = Tensor(np.full((2, 8), 3.0), requires_grad=True)

        out = ln(x)

        assert np.all(np.isfinite(out.data)), f"LayerNorm produced non-finite output: {out.data}"
        # With variance=0, normalized = (x - mean) / sqrt(eps) = 0, so the
        # output should reduce to just beta (0.0 by default init).
        np.testing.assert_allclose(out.data, np.zeros((2, 8)), atol=1e-3)

    def test_backward_constant_input_produces_finite_gradients(self):
        ln = LayerNorm(8)
        x = Tensor(np.full((2, 8), 3.0), requires_grad=True)

        out = ln(x)
        out.sum().backward()

        assert np.all(np.isfinite(x.grad)), f"LayerNorm backward produced non-finite grad_x: {x.grad}"
        assert np.all(np.isfinite(ln.gamma.grad)), f"Non-finite grad_gamma: {ln.gamma.grad}"
        assert np.all(np.isfinite(ln.beta.grad)), f"Non-finite grad_beta: {ln.beta.grad}"

    def test_forward_all_zero_input_produces_no_nan_or_inf(self):
        """The most common real-world trigger: an all-zero padding embedding."""
        ln = LayerNorm(8)
        x = Tensor(np.zeros((3, 8)), requires_grad=True)

        out = ln(x)

        assert np.all(np.isfinite(out.data))

    def test_mixed_batch_constant_and_varying_rows(self):
        """A constant row next to a normal row in the same batch must not
        let the degenerate row's near-zero variance corrupt the other."""
        ln = LayerNorm(4)
        data = np.array([
            [5.0, 5.0, 5.0, 5.0],
            [1.0, 2.0, 3.0, 4.0],
        ])
        x = Tensor(data, requires_grad=True)

        out = ln(x)

        assert np.all(np.isfinite(out.data))
        np.testing.assert_allclose(out.data[0], np.zeros(4), atol=1e-3)
        # Second row: normalized to zero mean, unit variance (times gamma=1, plus beta=0)
        assert abs(np.mean(out.data[1])) < 1e-5
