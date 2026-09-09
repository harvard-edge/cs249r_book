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


def test_throughput_scales_with_batch_size():
    """Reported GFLOP/s must account for the batch the latency was measured over.

    count_flops() is per sample by design, while measure_latency() times the whole
    batch. Dividing one by the other without multiplying by the batch size reports
    a throughput that *falls* as the batch grows, and pins the bottleneck label to
    'memory' for every model. This test pins the two conventions together.
    """
    import numpy as np

    from tinytorch.core.layers import Linear
    from tinytorch.core.tensor import Tensor
    from tinytorch.perf.profiling import Profiler

    rng = np.random.default_rng(0)
    profiler = Profiler()
    model = Linear(256, 128)

    profiles = {}
    for batch in (1, 64):
        x = Tensor(rng.standard_normal((batch, 256)).astype(np.float32))
        profiles[batch] = profiler.profile_forward_pass(model, x)

    # The per-sample figure must not depend on the batch...
    assert profiles[1]["flops"] == profiles[64]["flops"], (
        "count_flops is documented as per-sample but changed with the batch size"
    )
    # ...while the figure throughput is computed from must scale with it.
    assert profiles[64]["batch_flops"] == 64 * profiles[64]["flops"], (
        f"batch_flops {profiles[64]['batch_flops']} is not 64x the per-sample "
        f"{profiles[64]['flops']}"
    )

    # A 64x larger batch does 64x the arithmetic in well under 64x the time, so
    # throughput must rise. Without the batch factor it would fall by ~64x.
    assert profiles[64]["gflops_per_second"] > profiles[1]["gflops_per_second"], (
        "Throughput did not improve with a 64x larger batch: "
        f"{profiles[1]['gflops_per_second']:.4f} -> {profiles[64]['gflops_per_second']:.4f} "
        "GFLOP/s. The batch factor is probably missing from the derived metrics."
    )
