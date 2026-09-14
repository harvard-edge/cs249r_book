"""Spatial shape errors must not silently discard image channels."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.spatial import Conv2d


@pytest.mark.parametrize('channels', [1, 3])
def test_convolution_rejects_channel_mismatch(channels):
    with pytest.raises(ValueError, match='channels'):
        Conv2d(2, 1, 1)(Tensor(np.ones((1, channels, 2, 2))))


def test_convolution_rejects_kernel_larger_than_input():
    with pytest.raises(ValueError, match='kernel'):
        Conv2d(1, 1, 3)(Tensor(np.ones((1, 1, 2, 2))))
