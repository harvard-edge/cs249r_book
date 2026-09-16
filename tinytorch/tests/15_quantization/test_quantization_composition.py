"""Source-direct regressions for inference calibration and packed storage reports."""
import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def source():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] /
                             "src/15_quantization/15_quantization.py"))


def test_calibration_disables_dropout_in_nested_containers(source):
    from tinytorch.core.layers import Dropout
    tensor, linear, sequential = (source[name] for name in ("Tensor", "Linear", "Sequential"))
    layer = linear(1, 1)
    layer.weight, layer.bias = tensor([[1.0]]), tensor([0.0])
    model = sequential(sequential(Dropout(1.0)), sequential(layer))
    sample = tensor([[0.2]])
    baseline = model.forward(sample, training=False).data.copy()
    source["quantize_model"](model, [sample])
    assert isinstance(model.layers[1].layers[0], source["QuantizedLinear"])
    np.testing.assert_allclose(model.forward(sample, training=False).data, baseline, atol=1e-6)


def test_nested_storage_analysis_and_wrapper_include_metadata(source):
    linear, sequential = source["Linear"], source["Sequential"]
    original = sequential(sequential(linear(1, 1)))
    model = sequential(sequential(linear(1, 1)))
    report = source["Quantizer"].quantize_model(model)
    source["quantize_model"](model)
    analysis = source["analyze_model_sizes"](original, model)
    assert analysis["original_bytes"] == 8
    assert analysis["quantized_bytes"] == 18
    assert report["quantized_size_mb"] * source["MB_TO_BYTES"] == 18
    assert report["compression_ratio"] == pytest.approx(8 / 18)


@pytest.mark.parametrize("values", [[0.0, 1e-44], [-np.finfo(np.float32).max, np.finfo(np.float32).max]])
def test_finite_float32_extremes_roundtrip_without_invalid_arithmetic(source, values):
    original = source["Tensor"](values)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        codes, scale, zero = source["quantize_int8"](original)
        restored = source["dequantize_int8"](codes, scale, zero)
    assert np.all(np.isfinite(restored.data))
    # Float32 rounding itself contributes one subnormal ULP at the lower limit.
    tolerance = scale / 2 + float(np.nextafter(np.float32(0), np.float32(1)))
    error = np.abs(restored.data.astype(np.float64) - original.data.astype(np.float64))
    assert np.all(error <= tolerance)
    assert restored.data[1] > 0


def test_calibrated_subnormal_inputs_preserve_signal(source):
    tensor = source["Tensor"]
    layer = source["Linear"](1, 1)
    layer.weight, layer.bias = tensor([[1.0]]), tensor([0.0])
    quantized = source["QuantizedLinear"](layer)
    sample = tensor([[0.0], [1e-44]])
    quantized.calibrate([sample])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        result = quantized(sample)
    np.testing.assert_array_equal(result.data, sample.data)
