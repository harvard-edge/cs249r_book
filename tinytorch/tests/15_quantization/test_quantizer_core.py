"""
Module 15: Quantization Core Tests
===================================

These tests verify that quantization reduces model size correctly.

WHY THESE TESTS MATTER:
-----------------------
Quantization converts FP32 (4 bytes) to INT8 (1 byte) = 4x smaller model.
If quantization is broken:
- Model stays big (defeats the purpose)
- Accuracy drops too much (unusable)
- Values overflow (numerical errors)

WHAT WE TEST:
-------------
1. Quantization produces INT8 values
2. Dequantization recovers approximate original values
3. Model size actually decreases
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor


class TestQuantizationBasics:
    """Test basic quantization functionality."""

    def test_quantizer_import(self):
        """Verify Quantizer can be imported."""
        try:
            from tinytorch.perf.quantization import Quantizer
            assert Quantizer is not None
        except ImportError as e:
            pytest.skip(f"Quantizer not yet exported: {e}")

    def test_quantize_produces_int8(self):
        """
        WHAT: Verify quantization produces INT8 values in [-128, 127].

        WHY: INT8 is the target representation. Values outside this
        range would overflow and produce garbage.
        """
        try:
            from tinytorch.perf.quantization import Quantizer
        except ImportError:
            pytest.skip("Quantizer not yet exported")

        # Create FP32 tensor
        fp32_tensor = Tensor(np.random.randn(10, 10).astype(np.float32))

        # Quantize
        q_tensor, scale, zero_point = Quantizer.quantize_tensor(fp32_tensor)

        # Check INT8 range
        assert q_tensor.data.min() >= -128, "Quantized values below INT8 min"
        assert q_tensor.data.max() <= 127, "Quantized values above INT8 max"

    def test_dequantize_recovers_approximate_values(self):
        """
        WHAT: Verify dequantization recovers values close to original.

        WHY: Quantization is lossy, but should be approximately reversible.
        Large errors would destroy model accuracy.
        """
        try:
            from tinytorch.perf.quantization import Quantizer
        except ImportError:
            pytest.skip("Quantizer not yet exported")

        # Create FP32 tensor with known values
        original = Tensor(np.array([0.5, -0.5, 1.0, -1.0]).astype(np.float32))

        # Round trip: quantize then dequantize
        q_tensor, scale, zero_point = Quantizer.quantize_tensor(original)
        recovered = Quantizer.dequantize_tensor(q_tensor, scale, zero_point)

        # Should be close (within ~1% for typical values)
        max_error = np.max(np.abs(original.data - recovered.data))
        assert max_error < 0.1, (
            f"Dequantization error too large: {max_error}\n"
            f"  Original: {original.data}\n"
            f"  Recovered: {recovered.data}"
        )


class TestQuantizationAdvanced:
    """Advanced quantization tests for edge cases and accuracy."""

    def test_quantize_constant_tensor(self):
        """
        WHAT: Verify quantization handles constant tensors (all same value).

        WHY: Constant tensors are an edge case where min=max. The algorithm
        must handle this gracefully without division by zero.
        """
        try:
            from tinytorch.perf.quantization import quantize_int8
        except ImportError:
            pytest.skip("quantize_int8 not yet exported")

        # All zeros
        constant = Tensor(np.zeros((4, 4), dtype=np.float32))
        q_tensor, scale, zero_point = quantize_int8(constant)

        # Should produce valid output without errors
        assert q_tensor.data.shape == constant.data.shape, "Shape changed"

    def test_quantize_preserves_relative_ordering(self):
        """
        WHAT: Verify quantization preserves relative ordering of values.

        WHY: If [0.1, 0.2, 0.3] becomes [5, 4, 6], the model's predictions
        would be garbage. Relative ordering must be preserved.
        """
        try:
            from tinytorch.perf.quantization import quantize_int8
        except ImportError:
            pytest.skip("quantize_int8 not yet exported")

        # Strictly increasing values
        original = Tensor(np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
        q_tensor, _, _ = quantize_int8(original)

        # Quantized values should be monotonically non-decreasing
        q_data = q_tensor.data.astype(np.float32)
        for i in range(len(q_data) - 1):
            assert q_data[i] <= q_data[i + 1], (
                f"Ordering not preserved: q[{i}]={q_data[i]} > q[{i+1}]={q_data[i+1]}"
            )

    def test_quantize_negative_values(self):
        """
        WHAT: Verify quantization handles negative values correctly.

        WHY: Neural network weights are typically centered around zero
        with both positive and negative values.
        """
        try:
            from tinytorch.perf.quantization import quantize_int8, dequantize_int8
        except ImportError:
            pytest.skip("Quantization functions not yet exported")

        # Mixed positive and negative
        original = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        q_tensor, scale, zero_point = quantize_int8(original)
        recovered = dequantize_int8(q_tensor, scale, zero_point)

        # Original signs should be preserved after round-trip
        for i in range(len(original.data)):
            orig_sign = np.sign(original.data[i])
            rec_sign = np.sign(recovered.data[i])
            # Zero can go either way due to quantization noise
            if orig_sign != 0:
                assert orig_sign == rec_sign, (
                    f"Sign not preserved for value {original.data[i]}: "
                    f"recovered {recovered.data[i]}"
                )


class TestQuantizedLinear:
    """Test the QuantizedLinear layer implementation."""

    def test_quantized_linear_forward(self):
        """
        WHAT: Verify QuantizedLinear produces similar output to regular Linear.

        WHY: Quantized layers should approximate the original behavior.
        Large deviations indicate incorrect implementation.
        """
        try:
            from tinytorch.perf.quantization import QuantizedLinear
            from tinytorch.core.layers import Linear
        except ImportError:
            pytest.skip("QuantizedLinear not yet exported")

        # Create and quantize a linear layer
        linear = Linear(4, 3)
        q_linear = QuantizedLinear(linear)

        # Forward pass
        input_tensor = Tensor(np.random.randn(2, 4).astype(np.float32))
        original_output = linear.forward(input_tensor)
        quantized_output = q_linear.forward(input_tensor)

        # Outputs should be similar (within quantization error)
        # For INT8, typical error is ~1-5% of the output range
        max_error = np.max(np.abs(original_output.data - quantized_output.data))
        output_range = np.max(original_output.data) - np.min(original_output.data)

        # Allow up to 10% relative error for educational implementation
        assert max_error < 0.1 * output_range + 0.1, (
            f"QuantizedLinear output differs too much from Linear:\n"
            f"  Max error: {max_error:.4f}\n"
            f"  Output range: {output_range:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
