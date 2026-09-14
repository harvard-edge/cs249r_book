"""Constant and narrow-range quantization regressions, 2026-09-11."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.perf.quantization import quantize_int8, dequantize_int8


@pytest.mark.parametrize("value", [0.0, 0.25, -0.5, 2.0, -3.0, 500.0, -500.0, 1e-10])
def test_constant_tensor_roundtrip(value):
    x = Tensor(np.full((2, 3), value))
    q, scale, zero = quantize_int8(x)
    restored = dequantize_int8(q, scale, zero)
    assert scale > 0 and -128 <= zero <= 127
    np.testing.assert_allclose(restored.data, x.data, rtol=1e-6, atol=0)


def test_small_nonconstant_range_is_not_collapsed_to_one_constant():
    x = Tensor([-1e-10, 0.0, 1e-10])
    q, scale, zero = quantize_int8(x)
    assert np.unique(q.data).size == 3
    np.testing.assert_allclose(dequantize_int8(q, scale, zero).data, x.data, atol=scale)


def test_calibration_preserves_small_nonzero_activation_range():
    from tinytorch.core.layers import Linear
    from tinytorch.perf.quantization import QuantizedLinear

    layer = QuantizedLinear(Linear(1, 1))
    layer.calibrate([Tensor([[-1e-10], [1e-10]])])
    assert 0 < layer.input_scale < 1e-10
    values = np.array([-1e-10, 0.0, 1e-10])
    codes = np.clip(np.round(values / layer.input_scale + layer.input_zero_point), -128, 127)
    restored = (codes - layer.input_zero_point) * layer.input_scale
    np.testing.assert_allclose(restored, values, atol=layer.input_scale)


@pytest.mark.parametrize('values', [[], [np.nan], [np.inf], [-np.inf, 1.]])
def test_nonfinite_or_empty_values_are_rejected(values):
    with pytest.raises(ValueError):
        quantize_int8(Tensor(values))


def test_each_quantized_array_has_metadata_and_codes_remain_float32():
    from tinytorch.core.layers import Linear
    from tinytorch.perf.quantization import QuantizedLinear
    layer = QuantizedLinear(Linear(4, 3))
    assert layer.memory_usage()['quantized_bytes'] == 15 + 16
    assert sum(p.data.nbytes for p in layer.parameters()) == 15 * 4
    with pytest.raises(ValueError):
        layer.calibrate([])
