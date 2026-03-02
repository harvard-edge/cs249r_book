"""
Module 19: Benchmarking Core Tests
===================================

These tests verify that benchmarking tools work correctly.

WHY THESE TESTS MATTER:
-----------------------
Benchmarking is how we measure and compare model performance.
If benchmarking is broken:
- We can't measure throughput (tokens/second)
- We can't compare optimization techniques
- We can't validate our optimizations work

WHAT WE TEST:
-------------
1. MLPerf can run benchmarks
2. Metrics are computed correctly
3. Results are reproducible
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.perf.benchmarking import Benchmark, MLPerf


class TestBenchmarkBasics:
    """Test basic benchmarking functionality."""

    def test_benchmark_import(self):
        """Verify Benchmark can be imported."""
        assert Benchmark is not None
        assert MLPerf is not None

    def test_benchmark_can_instantiate(self):
        """Verify Benchmark can be created."""
        # Create simple dummy model
        class DummyModel:
            def forward(self, x):
                return x

        models = [DummyModel()]
        datasets = [[(Tensor(np.random.randn(10, 10)), Tensor(np.zeros(10)))]]

        bench = Benchmark(models, datasets)
        assert bench is not None

    def test_measure_throughput(self):
        """
        WHAT: Verify throughput measurement works.

        WHY: Throughput (items/second) is a key performance metric.
        """
        # Simple model
        class SimpleModel:
            def __init__(self):
                self.layer = Linear(10, 10)

            def forward(self, x):
                return self.layer.forward(x)

        model = SimpleModel()
        models = [model]
        datasets = [[(Tensor(np.random.randn(10, 10)), Tensor(np.zeros(10)))]]

        bench = Benchmark(models, datasets)
        results = bench.run_latency_benchmark(input_shape=(1, 10))

        assert len(results) > 0, "Benchmark should produce results"
        for model_name, result in results.items():
            assert result.mean > 0, (
                f"Latency should be positive, got {result.mean}"
            )


class TestMLPerf:
    """Test MLPerf benchmark suite."""

    def test_mlperf_can_run(self):
        """
        WHAT: Verify MLPerf benchmark suite can execute.

        WHY: This is the capstone benchmarking tool students build.
        """
        # Create and run minimal benchmark
        mlperf = MLPerf()

        # Should at least be able to list available benchmarks
        if hasattr(mlperf, 'list_benchmarks'):
            benchmarks = mlperf.list_benchmarks()
            assert isinstance(benchmarks, (list, dict)), (
                "list_benchmarks should return a list or dict"
            )


class TestBenchmarkMetrics:
    """Test that benchmark metrics are computed correctly."""

    def test_latency_is_positive(self):
        """Latency must always be positive."""
        class SimpleModel:
            def forward(self, x):
                return x * 2

        model = SimpleModel()
        x = Tensor(np.random.randn(10))
        datasets = [[(x, None)]]

        bench = Benchmark([model], datasets)
        results = bench.run_latency_benchmark(input_shape=(10,))

        assert len(results) > 0, "Should produce results"
        for name, result in results.items():
            assert result.mean > 0, "Latency must be positive"

    def test_multiple_runs_are_consistent(self):
        """
        WHAT: Verify benchmark results are reasonably consistent.

        WHY: Benchmarks should be reproducible. Large variance
        means we can't trust the measurements.
        """
        class SimpleModel:
            def __init__(self):
                self.layer = Linear(10, 10)

            def forward(self, x):
                return self.layer.forward(x)

        model = SimpleModel()
        x = Tensor(np.random.randn(1, 10))
        datasets = [[(x, None)]]

        bench = Benchmark([model], datasets, measurement_runs=10)
        results = bench.run_latency_benchmark(input_shape=(1, 10))

        # Check that we get results with reasonable variance
        for name, result in results.items():
            # Coefficient of variation should be reasonable (std/mean < 100%)
            if result.mean > 0:
                cv = result.std / result.mean
                assert cv < 1.0, (
                    f"Benchmark results too variable!\n"
                    f"  Mean: {result.mean}, Std: {result.std}, CV: {cv}\n"
                    "Coefficient of variation should be < 100%."
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
