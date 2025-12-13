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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear


class TestProfilerBasics:
    """Test basic profiler functionality."""

    def test_profiler_import(self):
        """
        WHAT: Verify profiler module can be imported.

        WHY: Basic sanity check that the module exists and exports correctly.
        """
        try:
            from tinytorch.perf.profiling import Profiler
            assert Profiler is not None
        except ImportError as e:
            pytest.skip(f"Profiler not yet exported: {e}")

    def test_profiler_can_instantiate(self):
        """
        WHAT: Verify Profiler class can be created.

        WHY: The profiler must be instantiable to use.
        """
        try:
            from tinytorch.perf.profiling import Profiler
            profiler = Profiler()
            assert profiler is not None
        except ImportError:
            pytest.skip("Profiler not yet exported")

    def test_profiler_can_count_parameters(self):
        """
        WHAT: Verify profiler can count model parameters.

        WHY: Parameter count is a fundamental metric:
        - Memory usage scales with parameters
        - Larger models need more compute
        - This is the first thing you check about a model
        """
        try:
            from tinytorch.perf.profiling import Profiler
        except ImportError:
            pytest.skip("Profiler not yet exported")

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


class TestLatencyMeasurement:
    """Test timing and latency measurement."""

    def test_measure_latency_returns_positive(self):
        """
        WHAT: Verify latency measurement returns positive time.

        WHY: Execution time must be positive and non-zero.
        """
        try:
            from tinytorch.perf.profiling import Profiler
        except ImportError:
            pytest.skip("Profiler not yet exported")

        class SimpleModel:
            def __init__(self):
                self.weight = Tensor(np.random.randn(10, 10))
            def forward(self, x):
                return x.matmul(self.weight)

        model = SimpleModel()
        x = Tensor(np.random.randn(1, 10))
        profiler = Profiler()

        latency = profiler.measure_latency(model, x, warmup=1, iterations=3)

        assert latency > 0, (
            f"Latency should be positive, got {latency}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
