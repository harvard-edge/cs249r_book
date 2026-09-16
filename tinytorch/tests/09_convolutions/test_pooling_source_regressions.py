"""Exercise the source of truth without requiring a package export."""
from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.fixture(scope="module")
def spatial():
    source = Path(__file__).resolve().parents[2] / "src/09_convolutions/09_convolutions.py"
    return runpy.run_path(str(source))


@pytest.mark.parametrize("name,expected", [
    ("MaxPool2d", [[8, 11], [20, 23]]),
    ("AvgPool2d", [[4, 7], [16, 19]]),
])
def test_rectangular_default_stride(spatial, name, expected):
    x = spatial["Tensor"](np.arange(24).reshape(1, 1, 4, 6))
    output = spatial[name]((2, 3))(x)
    np.testing.assert_array_equal(output.data[0, 0], expected)


@pytest.mark.parametrize("name", ["MaxPool2d", "AvgPool2d"])
@pytest.mark.parametrize("shape,kernel", [((2, 2), 3), ((4, 2), (2, 3)), ((2, 4), (3, 2))])
def test_pooling_rejects_empty_outputs(spatial, name, shape, kernel):
    x = spatial["Tensor"](np.ones((1, 1, *shape)))
    with pytest.raises(ValueError, match="kernel must fit"):
        spatial[name](kernel)(x)


@pytest.mark.parametrize("padding", [0, 1])
def test_negative_infinity_gradients_stay_in_window(spatial, padding):
    x = spatial["Tensor"](np.full((1, 1, 4, 4), -np.inf), requires_grad=True)
    y = spatial["MaxPool2d"](2, padding=padding)(x)
    # Distinct seeds make a gradient routed to the wrong window detectable.
    seeds = np.arange(1, y.data.size + 1, dtype=np.float32).reshape(y.shape)
    y.backward(seeds)
    expected = np.zeros_like(x.data)
    for oh in range(y.shape[2]):
        for ow in range(y.shape[3]):
            row = max(0, oh * 2 - padding)
            col = max(0, ow * 2 - padding)
            expected[0, 0, row, col] += seeds[0, 0, oh, ow]
    np.testing.assert_array_equal(x.grad, expected)


@pytest.mark.parametrize("name", ["MaxPool2d", "AvgPool2d"])
@pytest.mark.parametrize("stride,padding", [(None, 0), ((1, 2), 1), (1, 0)])
def test_rectangular_pooling_numeric_gradients(spatial, name, stride, padding):
    rng = np.random.default_rng(42)
    x = spatial["Tensor"](rng.normal(size=(1, 2, 3, 5)), requires_grad=True)
    pool = spatial[name]((2, 3), stride=stride, padding=padding)
    output = pool(x)
    seeds = rng.normal(size=output.shape).astype(np.float32)
    output.backward(seeds)
    analytic = x.grad.copy()
    numeric = np.zeros_like(x.data)
    step = 0.002
    for index in np.ndindex(x.shape):
        value = x.data[index]
        x.data[index] = value + step
        plus = np.sum(pool(x).data * seeds)
        x.data[index] = value - step
        minus = np.sum(pool(x).data * seeds)
        x.data[index] = value
        numeric[index] = (plus - minus) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, atol=5e-4, rtol=5e-4)


def test_batchnorm_single_image_retains_spatial_variation(spatial):
    x = spatial["Tensor"](np.arange(4).reshape(1, 1, 2, 2))
    output = spatial["BatchNorm2d"](1)(x)
    expected = (x.data - x.data.mean()) / np.sqrt(x.data.var() + 1e-5)
    np.testing.assert_allclose(output.data, expected, atol=1e-6)
    assert np.ptp(output.data) > 1
