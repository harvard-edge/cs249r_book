"""
Numerical edge-case tests for BatchNorm2d.

BatchNorm2d divides by sqrt(var + eps). For a channel whose values are
identical across the entire batch and spatial extent (a constant feature
map, or a degenerate all-zero channel), variance is exactly 0. The eps
guard already exists in the forward pass, but nothing exercised a
constant-valued channel to confirm the result actually stays finite.
BatchNorm2d otherwise had essentially no test coverage at all: it wasn't
referenced by any test in this suite.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.spatial import BatchNorm2d


class TestBatchNorm2dConstantInput:
    def test_forward_constant_channel_produces_no_nan_or_inf(self):
        """A channel with identical values everywhere has variance 0."""
        bn = BatchNorm2d(3)
        x = Tensor(np.full((2, 3, 4, 4), 7.0))

        out = bn(x)

        assert np.all(np.isfinite(out.data)), f"BatchNorm2d produced non-finite output: {out.data}"
        # With variance=0, normalized = (x - mean) / sqrt(eps) = 0, so the
        # output should reduce to just beta (0.0 by default init).
        np.testing.assert_allclose(out.data, np.zeros((2, 3, 4, 4)), atol=1e-2)

    def test_forward_all_zero_input_produces_no_nan_or_inf(self):
        bn = BatchNorm2d(3)
        x = Tensor(np.zeros((2, 3, 4, 4)))

        out = bn(x)

        assert np.all(np.isfinite(out.data))

    def test_mixed_channels_constant_and_varying(self):
        """A constant channel next to a normal one in the same batch must
        not let the degenerate channel's near-zero variance corrupt the
        other channel's normalization."""
        bn = BatchNorm2d(2)
        data = np.zeros((2, 2, 2, 2))
        data[:, 0, :, :] = 5.0  # constant channel
        data[:, 1, :, :] = np.array([[1.0, 2.0], [3.0, 4.0]])  # varying channel

        x = Tensor(data)
        out = bn(x)

        assert np.all(np.isfinite(out.data))
        np.testing.assert_allclose(out.data[:, 0, :, :], np.zeros((2, 2, 2)), atol=1e-2)

    def test_eval_mode_with_default_running_stats_produces_no_nan_or_inf(self):
        """Running variance defaults to 1.0 (not 0), so eval mode isn't at
        risk the same way, but it should still stay finite for any input,
        including one that would be degenerate in training mode."""
        bn = BatchNorm2d(3).eval()
        x = Tensor(np.full((2, 3, 4, 4), 7.0))

        out = bn(x)

        assert np.all(np.isfinite(out.data))
