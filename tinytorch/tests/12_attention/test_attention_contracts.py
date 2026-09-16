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


@pytest.mark.parametrize('key_values', [[0., 1e5], [-1e5, 0.]])
def test_large_logits_cannot_override_causal_mask(key_values):
    """A forbidden key stays excluded regardless of the finite score scale."""
    q = Tensor([[[1e5], [1e5]]])
    k = Tensor(np.array(key_values).reshape(1, 2, 1))
    v = Tensor([[[1.], [99.]]])
    output, weights = scaled_dot_product_attention(q, k, v, Tensor([[1, 0], [1, 1]]))
    np.testing.assert_array_equal(weights.data[0, 0], [1., 0.])
    assert output.data[0, 0, 0] == 1.
    assert np.isfinite(output.data).all()


def test_broadcast_padding_mask_blocks_value_gradients():
    """Shared padding masks exclude values in every batch and every head."""
    import tinytorch.core.autograd

    q = Tensor(np.ones((2, 3, 2, 1)), requires_grad=True)
    k = Tensor(np.array([0., 1e5]).reshape(1, 1, 2, 1), requires_grad=True)
    v = Tensor(np.array([1., 99.]).reshape(1, 1, 2, 1), requires_grad=True)
    output, weights = scaled_dot_product_attention(q, k, v, Tensor([1, 0]))
    output.sum().backward()
    np.testing.assert_array_equal(weights.data[..., 1], 0.)
    np.testing.assert_array_equal(v.grad.reshape(-1), [12., 0.])
    np.testing.assert_array_equal(q.grad, 0.)
    np.testing.assert_array_equal(k.grad, 0.)


def test_masked_attention_gradients_match_finite_differences():
    """Hard masking preserves the derivative of every allowed score and value."""
    import tinytorch.core.autograd

    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(1, 3, 2)).astype(np.float32) for _ in range(3)]
    mask = Tensor(np.tril(np.ones((3, 3))))
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    scaled_dot_product_attention(*tensors, mask)[0].sum().backward()
    eps = 1e-3
    for array, tensor in zip(arrays, tensors):
        numerical = np.empty_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + eps
            plus = scaled_dot_product_attention(*[Tensor(a) for a in arrays], mask)[0].data.sum()
            array[index] = original - eps
            minus = scaled_dot_product_attention(*[Tensor(a) for a in arrays], mask)[0].data.sum()
            array[index] = original
            numerical[index] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(tensor.grad, numerical, atol=8e-4, rtol=2e-3)


@pytest.mark.parametrize('mask', [0., [0, 0], [[[1, 1], [0, 0]]]])
def test_fully_masked_broadcast_rows_are_rejected(mask):
    q = Tensor(np.ones((2, 3, 2, 2)))
    with pytest.raises(ValueError, match='at least one allowed key'):
        scaled_dot_product_attention(q, q, q, Tensor(mask))
