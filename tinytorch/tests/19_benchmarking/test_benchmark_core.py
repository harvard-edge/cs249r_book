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
1. TinyMLPerf can run benchmarks
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


class TestBenchmarkBasics:
    """Test basic benchmarking functionality."""
    
    def test_benchmark_import(self):
        """Verify Benchmark can be imported."""
        try:
            from tinytorch.bench import Benchmark, TinyMLPerf
            assert Benchmark is not None
            assert TinyMLPerf is not None
        except ImportError as e:
            pytest.skip(f"Benchmark not yet exported: {e}")
    
    def test_benchmark_can_instantiate(self):
        """Verify Benchmark can be created."""
        try:
            from tinytorch.bench import Benchmark

            # Create simple dummy model
            class DummyModel:
                def forward(self, x):
                    return x

            models = [DummyModel()]
            datasets = [[(Tensor(np.random.randn(10, 10)), Tensor(np.zeros(10)))]]

            bench = Benchmark(models, datasets)
            assert bench is not None
        except ImportError:
            pytest.skip("Benchmark not yet exported")
    
    def test_measure_throughput(self):
        """
        WHAT: Verify throughput measurement works.

        WHY: Throughput (items/second) is a key performance metric.
        """
        try:
            from tinytorch.bench import Benchmark
        except ImportError:
            pytest.skip("Benchmark not yet exported")

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


class TestTinyMLPerf:
    """Test TinyMLPerf benchmark suite."""
    
    def test_tiny_mlperf_can_run(self):
        """
        WHAT: Verify TinyMLPerf benchmark suite can execute.
        
        WHY: This is the capstone benchmarking tool students build.
        """
        try:
            from tinytorch.bench import TinyMLPerf
        except ImportError:
            pytest.skip("TinyMLPerf not yet exported")
        
        # Create and run minimal benchmark
        mlperf = TinyMLPerf()
        
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
        try:
            from tinytorch.bench import Benchmark
        except ImportError:
            pytest.skip("Benchmark not yet exported")
        
        class SimpleModel:
            def forward(self, x):
                return x * 2
        
        model = SimpleModel()
        x = Tensor(np.random.randn(10))
        
        bench = Benchmark()
        latency = bench.measure_latency(model, x)
        
        assert latency > 0, "Latency must be positive"
    
    def test_multiple_runs_are_consistent(self):
        """
        WHAT: Verify benchmark results are reasonably consistent.
        
        WHY: Benchmarks should be reproducible. Large variance
        means we can't trust the measurements.
        """
        try:
            from tinytorch.bench import Benchmark
        except ImportError:
            pytest.skip("Benchmark not yet exported")
        
        class SimpleModel:
            def __init__(self):
                self.layer = Linear(10, 10)
            
            def forward(self, x):
                return self.layer.forward(x)
        
        model = SimpleModel()
        x = Tensor(np.random.randn(1, 10))
        
        bench = Benchmark()
        
        # Run 3 times
        latencies = [
            bench.measure_latency(model, x, iterations=10)
            for _ in range(3)
        ]
        
        # Check variance is reasonable (within 2x of each other)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        assert max_latency < min_latency * 3, (
            f"Benchmark results too variable!\n"
            f"  Latencies: {latencies}\n"
            "Results should be within 3x of each other."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

