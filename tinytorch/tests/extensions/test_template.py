import numpy as np
import pytest

from tinytorch.core.tensor import Tensor
from tinytorch.extensions.template import (
    CustomScaledShift,
    custom_scaled_shift,
    CustomResidualBlock,
    custom_accelerated_op,
    has_custom_hardware,
)


def test_custom_scaled_shift_tensor():
    x = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
    y = custom_scaled_shift(x, alpha=3.0, beta=5.0)

    assert isinstance(y, Tensor)
    np.testing.assert_allclose(y.data, [8.0, 11.0, 14.0])

    # Backward pass
    y.backward(Tensor(np.array([1.0, 1.0, 1.0], dtype=np.float32)))
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [3.0, 3.0, 3.0])


def test_custom_scaled_shift_numpy():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = custom_scaled_shift(x, alpha=3.0, beta=5.0)

    assert isinstance(y, np.ndarray)
    np.testing.assert_allclose(y, [8.0, 11.0, 14.0])


def test_custom_residual_block():
    block = CustomResidualBlock(dim=4)
    assert len(block.parameters()) == 2
    assert block.weight.requires_grad is True
    assert block.bias.requires_grad is True

    x = Tensor(np.ones((2, 4), dtype=np.float32))
    out = block(x)
    assert isinstance(out, Tensor)
    assert out.shape == (2, 4)


def test_custom_accelerated_op_fallback():
    assert has_custom_hardware() is False

    # Tensor input
    x_t = Tensor(np.array([-2.0, 0.0, 3.0], dtype=np.float32))
    out_t = custom_accelerated_op(x_t)
    assert isinstance(out_t, Tensor)
    np.testing.assert_allclose(out_t.data, [0.0, 0.0, 3.0])

    # NumPy input
    x_np = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    out_np = custom_accelerated_op(x_np)
    assert isinstance(out_np, np.ndarray)
    np.testing.assert_allclose(out_np, [0.0, 0.0, 3.0])


def test_custom_gradient_transform():
    from tinytorch.extensions.template import CustomGradientTransform
    transform = CustomGradientTransform(clip_value=0.5)
    p = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    p.grad = np.array([-1.2, 0.8], dtype=np.float32)
    transform.transform_gradients([p])
    np.testing.assert_allclose(p.grad, [-0.5, 0.5])



@pytest.mark.parametrize("hardware", [False, True])
def test_custom_accelerated_op_paths_match_numpy_relu(monkeypatch, hardware):
    """Accelerated and fallback paths must compute the same function (ReLU)."""
    import tinytorch.extensions.template as template

    monkeypatch.setattr(template, "has_custom_hardware", lambda: hardware)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5)).astype(np.float32)
    ref = np.maximum(0, x)

    out_np = template.custom_accelerated_op(x)
    assert isinstance(out_np, np.ndarray)
    np.testing.assert_allclose(out_np, ref, atol=1e-6)

    out_t = template.custom_accelerated_op(Tensor(x))
    assert isinstance(out_t, Tensor)
    np.testing.assert_allclose(out_t.data, ref, atol=1e-6)
