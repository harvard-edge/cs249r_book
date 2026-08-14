"""
Module 14: Profiler Core Tests
===============================

These tests verify that the profiling tools work correctly.

WHY THESE TESTS MATTER:
-----------------------
Profiling is essential for ML systems engineering. Without it:
- You can't find bottlenecks
- You can't measure improvement
- Optimization is guesswork

WHAT WE TEST:
-------------
1. Profiler can measure execution time
2. Profiler can count parameters
3. Profiler can analyze weight distributions

CONNECTION TO OTHER MODULES:
----------------------------
- Works with any model (Modules 03, 09, 13)
- Enables optimization decisions (Modules 15-18)
- Essential for benchmarking (Module 19)
"""

import pytest
import numpy as np
rng = np.random.default_rng(7)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.perf.profiling import Profiler


class TestProfilerBasics:
    """Test basic profiler functionality."""

    def test_profiler_import(self):
        """
        WHAT: Verify profiler module can be imported.

        WHY: Basic sanity check that the module exists and exports correctly.
        """
        assert Profiler is not None

    def test_profiler_can_instantiate(self):
        """
        WHAT: Verify Profiler class can be created.

        WHY: The profiler must be instantiable to use.
        """
        profiler = Profiler()
        assert profiler is not None

    def test_profiler_can_count_parameters(self):
        """
        WHAT: Verify profiler can count model parameters.

        WHY: Parameter count is a fundamental metric:
        - Memory usage scales with parameters
        - Larger models need more compute
        - This is the first thing you check about a model
        """
        # Create a simple model
        class SimpleModel:
            def __init__(self):
                self.layer = Linear(10, 5)
            def parameters(self):
                return self.layer.parameters()

        model = SimpleModel()
        profiler = Profiler()

        # Count parameters
        param_count = profiler.count_parameters(model)

        # Linear(10, 5) has: 10*5 weights + 5 bias = 55 parameters
        expected = 10 * 5 + 5
        assert param_count == expected, (
            f"Parameter count wrong!\n"
            f"  Expected: {expected} (10*5 weights + 5 bias)\n"
            f"  Got: {param_count}"
        )


class TestCountFlopsDispatch:
    """Test count_flops dispatch logic for non-Linear/Conv2d models."""

    def test_count_flops_routes_unnamed_layers_holder_to_sequential(self):
        """
        WHAT: A mock model with a `.layers` attribute but a class name other
        than 'Sequential' is still routed to the sequential-flops handling.

        WHY: count_flops dispatches on `model_name == 'Sequential' or
        hasattr(model, 'layers')`, so any object exposing `.layers` should
        be treated as a container of sub-layers, not fall through to the
        generic "1 FLOP per element" branch.
        """
        class LayerStack:
            def __init__(self, layers):
                self.layers = layers

        model = Linear(10, 5)
        mock = LayerStack([model])

        profiler = Profiler()
        input_shape = (1, 10)

        dispatched = profiler.count_flops(mock, input_shape)
        direct = profiler._count_sequential_flops(mock, input_shape)

        assert dispatched == direct, (
            "count_flops should route objects with a .layers attribute to "
            "_count_sequential_flops even when the class isn't named 'Sequential'"
        )


class TestLatencyMeasurement:
    """Test timing and latency measurement."""

    def test_measure_latency_returns_positive(self):
        """
        WHAT: Verify latency measurement returns positive time.

        WHY: Execution time must be positive and non-zero.
        """
        class SimpleModel:
            def __init__(self):
                self.weight = Tensor(rng.standard_normal((10, 10)))
            def forward(self, x):
                return x.matmul(self.weight)

        model = SimpleModel()
        x = Tensor(rng.standard_normal((1, 10)))
        profiler = Profiler()

        latency = profiler.measure_latency(model, x, warmup=1, iterations=3)

        assert latency > 0, (
            f"Latency should be positive, got {latency}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
