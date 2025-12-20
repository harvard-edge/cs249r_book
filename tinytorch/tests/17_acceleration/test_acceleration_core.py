"""
Module 17: Acceleration Core Tests
===================================

These tests verify optimization techniques for faster inference.

WHY THESE TESTS MATTER:
-----------------------
Acceleration techniques (SIMD, parallel execution, memory layout)
can provide significant speedups. These tests verify:
- Optimizations produce correct results
- Performance actually improves
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor


class TestAccelerationBasics:
    """Test basic acceleration functionality."""

    def test_acceleration_import(self):
        """Verify acceleration module can be imported."""
        try:
            from tinytorch.perf.acceleration import (
                vectorized_matmul,
                fused_gelu,
                tiled_matmul
            )
            assert vectorized_matmul is not None
            assert fused_gelu is not None
            assert tiled_matmul is not None
        except ImportError as e:
            pytest.skip(f"Acceleration functions not yet exported: {e}")

    def test_vectorized_matmul_correctness(self):
        """
        WHAT: Verify vectorized matmul produces correct results.

        WHY: Optimization must not change results. Speed without
        correctness is useless.
        """
        try:
            from tinytorch.perf.acceleration import vectorized_matmul
        except ImportError:
            pytest.skip("vectorized_matmul not yet exported")

        A = Tensor(np.random.randn(32, 64).astype(np.float32))
        B = Tensor(np.random.randn(64, 32).astype(np.float32))

        # Vectorized matmul
        result = vectorized_matmul(A, B)

        # Verify shape
        assert result.shape == (32, 32), f"Wrong shape: {result.shape}"

        # Verify against numpy reference
        reference = np.matmul(A.data, B.data)
        assert np.allclose(result.data, reference, rtol=1e-5), (
            "Vectorized matmul gives different results!"
        )


    def test_fused_gelu_correctness(self):
        """
        WHAT: Verify fused GELU produces correct results.

        WHY: Kernel fusion must preserve numerical accuracy.
        """
        try:
            from tinytorch.perf.acceleration import fused_gelu
        except ImportError:
            pytest.skip("fused_gelu not yet exported")

        # Test basic properties
        x = Tensor(np.array([-2, -1, 0, 1, 2], dtype=np.float32))
        result = fused_gelu(x)

        # GELU(0) should be 0
        assert abs(result.data[2]) < 1e-6, f"GELU(0) should be 0, got {result.data[2]}"

        # GELU should be smooth (no NaN/Inf)
        assert np.all(np.isfinite(result.data)), "GELU should produce finite values"

        # GELU(x) should have positive bias for positive x
        assert result.data[3] > 0.8, f"GELU(1) should be close to 1, got {result.data[3]}"

    def test_tiled_matmul_correctness(self):
        """
        WHAT: Verify tiled matmul produces correct results.

        WHY: Cache-aware tiling must preserve numerical accuracy.
        """
        try:
            from tinytorch.perf.acceleration import tiled_matmul, vectorized_matmul
        except ImportError:
            pytest.skip("tiled_matmul not yet exported")

        A = Tensor(np.random.randn(128, 128).astype(np.float32))
        B = Tensor(np.random.randn(128, 128).astype(np.float32))

        # Tiled matmul
        result_tiled = tiled_matmul(A, B, tile_size=32)

        # Reference (vectorized)
        result_reference = vectorized_matmul(A, B)

        # Should match
        assert np.allclose(result_tiled.data, result_reference.data, rtol=1e-5), (
            "Tiled matmul gives different results!"
        )


class TestMemoryOptimization:
    """Test memory-related optimizations."""

    def test_contiguous_memory_check(self):
        """
        WHAT: Verify we can check if tensor memory is contiguous.

        WHY: Contiguous memory enables SIMD and cache-friendly access.
        Non-contiguous tensors are slower.
        """
        # Create contiguous tensor
        contiguous = Tensor(np.random.randn(10, 10))
        assert contiguous.data.flags['C_CONTIGUOUS'], (
            "Fresh tensor should be contiguous"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
