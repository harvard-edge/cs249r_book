"""
Module 16: Compression Core Tests
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

            def parameters(self):
                return self.layer.parameters()

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

            def parameters(self):
                return self.layer.parameters()

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


class TestCompressionAdvanced:
    """Advanced compression tests for accuracy and edge cases."""

    def test_sparsity_achieves_target(self):
        """
        WHAT: Verify magnitude pruning achieves approximately target sparsity.

        WHY: If we request 80% sparsity, we should get close to 80% zeros.
        Large deviations indicate the pruning algorithm is broken.
        """
        try:
            from tinytorch.perf.compression import measure_sparsity, magnitude_prune
        except ImportError:
            pytest.skip("Compression functions not yet exported")

        # Create model
        class SimpleModel:
            def __init__(self):
                self.layer1 = Linear(100, 50, bias=False)
                self.layer2 = Linear(50, 25, bias=False)

            def parameters(self):
                return self.layer1.parameters() + self.layer2.parameters()

        model = SimpleModel()
        target_sparsity = 0.8  # 80%

        # Apply pruning
        magnitude_prune(model, sparsity=target_sparsity)
        achieved_sparsity = measure_sparsity(model)

        # Should be within 5% of target (sparsity is in percentage)
        assert abs(achieved_sparsity - target_sparsity * 100) < 5, (
            f"Sparsity target not achieved!\n"
            f"  Target: {target_sparsity * 100}%\n"
            f"  Achieved: {achieved_sparsity:.1f}%"
        )

    def test_pruning_preserves_large_weights(self):
        """
        WHAT: Verify that large magnitude weights are preserved during pruning.

        WHY: Magnitude pruning should keep the largest weights. If large
        weights are removed, model accuracy would collapse.
        """
        try:
            from tinytorch.perf.compression import magnitude_prune
        except ImportError:
            pytest.skip("magnitude_prune not yet exported")

        # Create model with one very large weight
        class SimpleModel:
            def __init__(self):
                self.layer = Linear(4, 4, bias=False)
                # Set one weight to be much larger than others
                self.layer.weight.data = np.array([
                    [0.01, 0.02, 0.01, 0.02],
                    [0.01, 100.0, 0.01, 0.02],  # 100.0 is the largest
                    [0.01, 0.02, 0.01, 0.02],
                    [0.01, 0.02, 0.01, 0.02]
                ], dtype=np.float32)

            def parameters(self):
                return self.layer.parameters()

        model = SimpleModel()

        # Prune 90% of weights
        magnitude_prune(model, sparsity=0.9)

        # The largest weight should still be there
        assert model.layer.weight.data[1, 1] == 100.0, (
            "Large weight was incorrectly pruned!\n"
            f"  Expected: 100.0\n"
            f"  Got: {model.layer.weight.data[1, 1]}"
        )

    def test_zero_sparsity_no_change(self):
        """
        WHAT: Verify that 0% sparsity doesn't change the model.

        WHY: This is an edge case - requesting no pruning should leave
        all weights unchanged.
        """
        try:
            from tinytorch.perf.compression import magnitude_prune
        except ImportError:
            pytest.skip("magnitude_prune not yet exported")

        class SimpleModel:
            def __init__(self):
                self.layer = Linear(4, 4, bias=False)

            def parameters(self):
                return self.layer.parameters()

        model = SimpleModel()
        original_weights = model.layer.weight.data.copy()

        # Prune 0% (should be no change)
        magnitude_prune(model, sparsity=0.0)

        assert np.allclose(model.layer.weight.data, original_weights), (
            "0% sparsity changed weights when it shouldn't!"
        )


class TestStructuredPruning:
    """Test structured pruning (removing entire neurons/channels)."""

    def test_structured_prune_import(self):
        """Verify structured_prune function can be imported."""
        try:
            from tinytorch.perf.compression import structured_prune
            assert structured_prune is not None
        except ImportError:
            pytest.skip("structured_prune not yet exported")

    def test_structured_prune_reduces_effective_neurons(self):
        """
        WHAT: Verify structured pruning removes entire rows/columns.

        WHY: Unlike magnitude pruning which creates sparse matrices,
        structured pruning removes whole neurons for actual speedups.
        """
        try:
            from tinytorch.perf.compression import structured_prune
            from tinytorch.core.layers import Sequential
        except ImportError:
            pytest.skip("structured_prune not yet exported")

        # Create model using Sequential (required for structured_prune)
        layer = Linear(10, 10, bias=False)
        model = Sequential(layer)

        # Apply 50% structured pruning
        structured_prune(model, prune_ratio=0.5)

        # Check that some entire columns are now all zeros
        # structured_prune zeros out columns (output channels)
        weights = layer.weight.data
        zero_cols = np.sum(np.all(weights == 0, axis=0))

        # At least some columns should be completely zeroed
        assert zero_cols >= 1, (
            f"Structured pruning didn't zero out entire columns!\n"
            f"  Expected: At least 1 zero column\n"
            f"  Got: {zero_cols} zero columns"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
