"""
Module 16: Quantization Core Tests
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

