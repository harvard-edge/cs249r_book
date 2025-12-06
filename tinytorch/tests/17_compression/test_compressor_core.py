"""
Module 17: Compression Core Tests
===================================

These tests verify that model compression (pruning) works correctly.

WHY THESE TESTS MATTER:
-----------------------
Pruning removes unnecessary weights, making models smaller and faster.
If compression is broken:
- Model doesn't get smaller (no benefit)
- Important weights get removed (accuracy crashes)
- Sparsity calculations are wrong (can't measure compression)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear


class TestCompressionBasics:
    """Test basic compression/pruning functionality."""
    
    def test_compressor_import(self):
        """Verify Compressor can be imported."""
        try:
            from tinytorch.perf.compression import Compressor
            assert Compressor is not None
        except ImportError as e:
            pytest.skip(f"Compressor not yet exported: {e}")
    
    def test_measure_sparsity(self):
        """
        WHAT: Verify sparsity measurement works correctly.
        
        WHY: Sparsity = fraction of zeros. This is how we measure compression.
        50% sparsity means half the weights are zero.
        """
        try:
            from tinytorch.perf.compression import Compressor
        except ImportError:
            pytest.skip("Compressor not yet exported")
        
        # Create a simple model with known sparsity
        class SimpleModel:
            def __init__(self):
                # Half zeros, half ones = 50% sparsity
                self.layer = Linear(4, 4, bias=False)
                self.layer.weight.data = np.array([
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1]
                ], dtype=np.float32)
            
            @property
            def layers(self):
                return [self.layer]
        
        model = SimpleModel()
        sparsity = Compressor.measure_sparsity(model)
        
        # Should be ~50%
        assert 0.4 < sparsity < 0.6, (
            f"Sparsity measurement wrong!\n"
            f"  Expected: ~0.5 (50% zeros)\n"
            f"  Got: {sparsity}"
        )
    
    def test_magnitude_prune_increases_sparsity(self):
        """
        WHAT: Verify pruning increases the number of zeros.
        
        WHY: Pruning should set small weights to zero.
        After pruning, sparsity should increase.
        """
        try:
            from tinytorch.perf.compression import Compressor
        except ImportError:
            pytest.skip("Compressor not yet exported")
        
        # Create model with random weights (low sparsity)
        class SimpleModel:
            def __init__(self):
                self.layer = Linear(10, 10, bias=False)
            
            @property
            def layers(self):
                return [self.layer]
        
        model = SimpleModel()
        initial_sparsity = Compressor.measure_sparsity(model)
        
        # Apply pruning
        Compressor.magnitude_prune(model, sparsity=0.5)
        
        final_sparsity = Compressor.measure_sparsity(model)
        
        assert final_sparsity > initial_sparsity, (
            f"Pruning didn't increase sparsity!\n"
            f"  Before: {initial_sparsity}\n"
            f"  After: {final_sparsity}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

