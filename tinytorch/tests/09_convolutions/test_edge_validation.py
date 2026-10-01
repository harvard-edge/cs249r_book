"""Edge-case validation for Conv2d and MaxPool2d, run against Module 09's source."""
from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.fixture(scope="module")
def spatial():
    source = Path(__file__).resolve().parents[2] / "src/09_convolutions/09_convolutions.py"
    return runpy.run_path(str(source), run_name="convolutions_edge_audit")


def _reference_conv(x, w, b, stride, padding):
    """Nested-loop cross-correlation with separate height and width strides."""
    sh, sw = stride
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    n, _, h, wd = xp.shape
    oc, _, kh, kw = w.shape
    out = np.zeros((n, oc, (h - kh) // sh + 1, (wd - kw) // sw + 1))
    for i in range(out.shape[2]):
        for j in range(out.shape[3]):
            patch = xp[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            out[:, :, i, j] = np.einsum("nchw,ochw->no", patch, w) + b
    return out


@pytest.mark.parametrize("stride", [2, (2, 2), [2, 2]])
def test_conv2d_int_and_pair_stride_agree(spatial, stride):
    conv = spatial["Conv2d"](1, 1, kernel_size=3, stride=stride)
    assert conv.stride == (2, 2)
    assert conv._compute_output_shape(7, 7) == (3, 3)


def test_conv2d_rectangular_stride_forward_and_backward(spatial):
    Tensor = spatial["Tensor"]
    rng = np.random.default_rng(0)
    conv = spatial["Conv2d"](2, 3, kernel_size=3, stride=(1, 2), padding=1)
    conv.bias.data[:] = rng.standard_normal(3)
    x = Tensor(rng.standard_normal((2, 2, 5, 6)), requires_grad=True)

    out = conv(x)
    expected = _reference_conv(x.data, conv.weight.data, conv.bias.data, (1, 2), 1)
    assert out.shape == (2, 3, 5, 3)
    np.testing.assert_allclose(out.data, expected, rtol=1e-6, atol=1e-6)

    out.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    # d(sum)/d(bias) counts every output position of that channel.
    np.testing.assert_allclose(conv.bias.grad, np.full(3, 2 * 5 * 3))


@pytest.mark.parametrize("kernel,padding", [(2, 2), (3, 2), (2, 5), ((3, 5), 2)])
def test_maxpool_rejects_padding_beyond_half_kernel(spatial, kernel, padding):
    with pytest.raises(ValueError, match="padding"):
        spatial["MaxPool2d"](kernel, padding=padding)


def test_maxpool_rejects_negative_padding(spatial):
    with pytest.raises(ValueError, match="padding"):
        spatial["MaxPool2d"](2, padding=-1)


@pytest.mark.parametrize("kernel,padding", [(2, 1), (3, 1), (4, 2)])
def test_maxpool_valid_padding_has_no_infinite_outputs(spatial, kernel, padding):
    pool = spatial["MaxPool2d"](kernel, stride=1, padding=padding)
    x = spatial["Tensor"](np.arange(25, dtype=float).reshape(1, 1, 5, 5))
    assert np.all(np.isfinite(pool(x).data))
