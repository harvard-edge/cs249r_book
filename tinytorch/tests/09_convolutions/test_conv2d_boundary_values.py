"""
Boundary-value tests for Conv2d/pooling with kernel_size == input spatial size.

This is the "global" convolution/pooling boundary: the kernel exactly
covers the whole input, so the output collapses to a single spatial
position (1x1). Off-by-one errors in the output-size formula
((in + 2*padding - kernel) // stride + 1) are most visible right at this
edge, and no existing test exercised it.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.spatial import Conv2d, MaxPool2d, AvgPool2d
from tinytorch.core.autograd import enable_autograd

enable_autograd()


class TestKernelEqualsInputSize:
    def test_conv2d_kernel_equals_input_produces_1x1_output(self):
        conv = Conv2d(in_channels=3, out_channels=4, kernel_size=5)
        x = Tensor(np.random.randn(2, 3, 5, 5).astype(np.float32), requires_grad=True)

        out = conv(x)

        assert out.shape == (2, 4, 1, 1)
        assert np.all(np.isfinite(out.data))

        out.sum().backward()
        assert np.all(np.isfinite(x.grad))
        assert x.grad.shape == x.shape

    def test_maxpool2d_kernel_equals_input_produces_1x1_output(self):
        pool = MaxPool2d(kernel_size=4)
        x = Tensor(np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4).astype(np.float32))

        out = pool(x)

        assert out.shape == (2, 3, 1, 1)
        # The single output per channel must be that channel's global max.
        expected = x.data.max(axis=(2, 3), keepdims=True)
        np.testing.assert_allclose(out.data, expected)

    def test_avgpool2d_kernel_equals_input_produces_1x1_output(self):
        pool = AvgPool2d(kernel_size=4)
        x = Tensor(np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4).astype(np.float32))

        out = pool(x)

        assert out.shape == (2, 3, 1, 1)
        expected = x.data.mean(axis=(2, 3), keepdims=True)
        np.testing.assert_allclose(out.data, expected)

    def test_conv2d_1x1_input_1x1_kernel(self):
        """The most degenerate case: a single pixel input."""
        conv = Conv2d(in_channels=2, out_channels=3, kernel_size=1)
        x = Tensor(np.random.randn(1, 2, 1, 1).astype(np.float32), requires_grad=True)

        out = conv(x)

        assert out.shape == (1, 3, 1, 1)
        assert np.all(np.isfinite(out.data))
