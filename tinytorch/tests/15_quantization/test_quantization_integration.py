#!/usr/bin/env python3
"""
Integration tests for Module 15: Quantization.

These check the properties the module promises: that INT8 round-trips inside the
quantization grid, that a QuantizedLinear tracks the FP32 layer it replaces, and
that the reported memory saving is real. They are written to fail if quantization
silently becomes a no-op or loses the zero point.
"""

import numpy as np

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.perf.quantization import (
    quantize_int8,
    dequantize_int8,
    QuantizedLinear,
)


def test_int8_roundtrip_stays_inside_the_grid():
    """Dequantizing a quantized tensor must land within half a step of the original."""
    rng = np.random.default_rng(0)
    original = Tensor(rng.standard_normal((64, 32)).astype(np.float32) * 3.0)

    q, scale, zero_point = quantize_int8(original)
    restored = dequantize_int8(q, scale, zero_point)

    assert q.data.shape == original.data.shape, "Quantization changed the tensor shape"
    assert scale > 0, f"Scale must be positive, got {scale}"

    # Every stored code must be representable in INT8.
    assert q.data.min() >= -128 and q.data.max() <= 127, (
        f"Quantized codes escaped INT8 range: [{q.data.min()}, {q.data.max()}]"
    )

    # The defining property: error is bounded by half a quantization step.
    err = np.abs(restored.data - original.data)
    assert err.max() <= scale / 2 + 1e-5, (
        f"Round-trip error {err.max():.6f} exceeds half a step ({scale / 2:.6f})"
    )

    # And it must not be a no-op that simply hands the input back.
    assert not np.array_equal(q.data, original.data), (
        "quantize_int8 returned the input unchanged; nothing was quantized"
    )

    # TinyTorch's Tensor stores float32, so INT8 here is *simulated*: the codes are
    # whole numbers in INT8 range carried in a float array, and the memory saving is
    # accounted analytically rather than realised in the buffer. Assert the invariant
    # that does hold, namely that every code is integral. A quantizer that forgot to
    # round would keep fractional codes and slip past a range check alone.
    assert np.allclose(q.data, np.round(q.data)), (
        "Quantized codes are not whole numbers; the rounding step is missing"
    )


def test_asymmetric_range_uses_the_zero_point():
    """A range that does not straddle zero needs a nonzero zero_point to stay accurate."""
    original = Tensor(np.linspace(5.0, 9.0, 128).astype(np.float32))

    q, scale, zero_point = quantize_int8(original)
    restored = dequantize_int8(q, scale, zero_point)

    err = np.abs(restored.data - original.data).max()
    assert err <= scale / 2 + 1e-5, (
        f"Asymmetric range lost accuracy: max error {err:.6f} vs step/2 {scale / 2:.6f}. "
        "A zero_point pinned to 0 causes exactly this."
    )


def test_quantized_linear_tracks_the_float_layer():
    """QuantizedLinear must preserve shape and stay close to the layer it replaces."""
    rng = np.random.default_rng(1)
    layer = Linear(16, 8)
    x = Tensor(rng.standard_normal((4, 16)).astype(np.float32))

    reference = layer(x)
    qlayer = QuantizedLinear(layer)
    actual = qlayer(x)

    assert actual.data.shape == reference.data.shape, (
        f"QuantizedLinear changed the output shape: {actual.data.shape} vs {reference.data.shape}"
    )

    # Quantization is lossy but must stay in the same neighbourhood.
    scale_of_output = np.abs(reference.data).mean() + 1e-6
    relative = np.abs(actual.data - reference.data).mean() / scale_of_output
    assert relative < 0.10, (
        f"Quantized layer drifted {relative:.1%} from the float layer, expected under 10%"
    )


def test_memory_accounting_reports_a_real_saving():
    """memory_usage() must report INT8 weights as roughly a quarter of FP32."""
    layer = Linear(32, 16)
    qlayer = QuantizedLinear(layer)

    usage = qlayer.memory_usage()
    for key in ("original_bytes", "quantized_bytes", "compression_ratio"):
        assert key in usage, f"memory_usage() is missing the '{key}' key"

    assert usage["quantized_bytes"] < usage["original_bytes"], (
        "Quantized footprint is not smaller than the float footprint"
    )
    assert 3.0 < usage["compression_ratio"] < 5.0, (
        f"Expected roughly 4x compression for FP32->INT8, got {usage['compression_ratio']:.2f}x"
    )


if __name__ == "__main__":
    test_int8_roundtrip_stays_inside_the_grid()
    test_asymmetric_range_uses_the_zero_point()
    test_quantized_linear_tracks_the_float_layer()
    test_memory_accounting_reports_a_real_saving()
    print("✅ Quantization integration tests passed")
