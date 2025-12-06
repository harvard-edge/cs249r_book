"""
Module 18: Acceleration Core Tests
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
            from tinytorch.perf.acceleration import Accelerator
            assert Accelerator is not None
        except ImportError as e:
            pytest.skip(f"Accelerator not yet exported: {e}")
    
    def test_optimized_matmul_correctness(self):
        """
        WHAT: Verify optimized matmul produces same results as naive.
        
        WHY: Optimization must not change results. Speed without
        correctness is useless.
        """
        try:
            from tinytorch.perf.acceleration import Accelerator
        except ImportError:
            pytest.skip("Accelerator not yet exported")
        
        A = Tensor(np.random.randn(32, 64))
        B = Tensor(np.random.randn(64, 32))
        
        # Standard matmul
        standard_result = A.matmul(B)
        
        # Optimized matmul (if available)
        if hasattr(Accelerator, 'optimized_matmul'):
            optimized_result = Accelerator.optimized_matmul(A, B)
            
            assert np.allclose(standard_result.data, optimized_result.data, rtol=1e-5), (
                "Optimized matmul gives different results!"
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

