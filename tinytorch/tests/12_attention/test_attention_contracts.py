"""Invalid masks must never return plausible attention on forbidden keys."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import MultiHeadAttention, scaled_dot_product_attention


@pytest.mark.parametrize('dimensions', [(4, 0), (4, -1), (2, 4), (0, 1), (4, 1.5)])
def test_invalid_head_partition_reports_value_error(dimensions):
    with pytest.raises(ValueError):
        MultiHeadAttention(*dimensions)


@pytest.mark.parametrize('mask', [[[0, 0], [1, 1]], [[1, 0.5], [1, 1]]])
def test_invalid_attention_mask_is_rejected(mask):
    q = Tensor(np.ones((1, 2, 2)))
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, q, q, Tensor(mask))
