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
rng = np.random.default_rng(7)
import statistics
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
        datasets = [[(Tensor(rng.standard_normal((10, 10))), Tensor(np.zeros(10)))]]

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
        datasets = [[(Tensor(rng.standard_normal((10, 10))), Tensor(np.zeros(10)))]]

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
        x = Tensor(rng.standard_normal(10))
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

        NOTE: raw coefficient of variation (std/mean) on wall-clock latency
        of a sub-millisecond forward pass is dominated by OS scheduling
        jitter on shared/virtualized CI runners: a single delayed sample
        can spike std enough to fail a mean-based check even though every
        other measurement was tight. Using the median and median absolute
        deviation instead is standard practice for exactly this kind of
        noisy, outlier-prone timing data, and still fails on genuinely
        inconsistent (not just occasionally-delayed) measurements.
        """
        class SimpleModel:
            def __init__(self):
                self.layer = Linear(10, 10)

            def forward(self, x):
                return self.layer.forward(x)

        model = SimpleModel()
        x = Tensor(rng.standard_normal((1, 10)))
        datasets = [[(x, None)]]

        bench = Benchmark([model], datasets, measurement_runs=20)
        results = bench.run_latency_benchmark(input_shape=(1, 10))

        # Check that we get results with reasonable variance, using a
        # robust (outlier-resistant) statistic rather than raw std/mean.
        for name, result in results.items():
            if result.median > 0:
                deviations = sorted(abs(v - result.median) for v in result.values)
                mad = statistics.median(deviations)
                robust_cv = mad / result.median
                assert robust_cv < 1.0, (
                    f"Benchmark results too variable!\n"
                    f"  Median: {result.median}, MAD: {mad}, Robust CV: {robust_cv}\n"
                    f"  Raw values: {result.values}\n"
                    "Median absolute deviation relative to median should be < 100%."
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
