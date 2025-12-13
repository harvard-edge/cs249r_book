#!/usr/bin/env python3
"""
Scientific Performance Testing Framework for TinyTorch
====================================================

This framework provides rigorous, scientific performance measurement
with proper statistical analysis and confidence intervals.

Key Features:
- Statistical timing with warmup and multiple runs
- Memory profiling with peak usage tracking
- Confidence intervals and significance testing
- Controlled environment for reliable measurements
"""

import numpy as np
import time
import gc
import tracemalloc
from typing import Dict, List, Tuple, Callable, Any, Optional
import statistics


class PerformanceTimer:
    """Statistical timing with proper warmup and confidence intervals."""

    def __init__(self, warmup_runs: int = 3, timing_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs

    def measure(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Measure function performance with statistical rigor."""
        # Force garbage collection before measurement
        gc.collect()

        # Warmup runs (not timed)
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)

        # Actual timing runs
        times = []
        for _ in range(self.timing_runs):
            gc.collect()  # Clean state for each run

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        # Statistical analysis
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)

        # 95% confidence interval
        if len(times) > 1:
            confidence_95 = 1.96 * std_time / (len(times) ** 0.5)
        else:
            confidence_95 = 0.0

        return {
            'mean': mean_time,
            'std': std_time,
            'median': median_time,
            'min': min_time,
            'max': max_time,
            'runs': len(times),
            'confidence_95': confidence_95,
            'coefficient_of_variation': std_time / mean_time if mean_time > 0 else 0.0,
            'result': result  # Store last result for validation
        }


class MemoryProfiler:
    """Memory usage profiling with peak usage tracking."""

    def measure(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Measure memory usage during function execution."""
        tracemalloc.start()

        # Baseline memory
        baseline_mem = tracemalloc.get_traced_memory()[0]

        # Execute function
        result = func(*args, **kwargs)

        # Peak memory during execution
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            'baseline_bytes': baseline_mem,
            'peak_bytes': peak_mem,
            'current_bytes': current_mem,
            'allocated_bytes': peak_mem - baseline_mem,
            'baseline_mb': baseline_mem / 1024 / 1024,
            'peak_mb': peak_mem / 1024 / 1024,
            'allocated_mb': (peak_mem - baseline_mem) / 1024 / 1024,
            'result': result
        }


class AccuracyTester:
    """Test accuracy preservation during optimizations."""

    @staticmethod
    def compare_outputs(original: Any, optimized: Any, tolerance: float = 1e-6) -> Dict[str, float]:
        """Compare two outputs for numerical equivalence."""
        if hasattr(original, 'data'):
            original = original.data
        if hasattr(optimized, 'data'):
            optimized = optimized.data

        # Convert to numpy arrays
        orig_array = np.array(original)
        opt_array = np.array(optimized)

        # Check shapes match
        if orig_array.shape != opt_array.shape:
            return {
                'shapes_match': False,
                'max_diff': float('inf'),
                'mean_diff': float('inf'),
                'accuracy_preserved': False
            }

        # Calculate differences
        diff = np.abs(orig_array - opt_array)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Relative accuracy
        if np.max(np.abs(orig_array)) > 0:
            relative_error = max_diff / np.max(np.abs(orig_array))
        else:
            relative_error = max_diff

        accuracy_preserved = max_diff < tolerance

        return {
            'shapes_match': True,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'relative_error': float(relative_error),
            'accuracy_preserved': accuracy_preserved,
            'tolerance': tolerance
        }


class PerformanceTester:
    """Main performance testing framework combining timing, memory, and accuracy."""

    def __init__(self, warmup_runs: int = 3, timing_runs: int = 10):
        self.timer = PerformanceTimer(warmup_runs, timing_runs)
        self.memory = MemoryProfiler()
        self.accuracy = AccuracyTester()

    def compare_performance(self,
                          baseline_func: Callable,
                          optimized_func: Callable,
                          args: Tuple = (),
                          kwargs: Dict = None,
                          test_name: str = "Performance Test") -> Dict[str, Any]:
        """Compare baseline vs optimized implementations comprehensively."""
        if kwargs is None:
            kwargs = {}

        print(f"\n🧪 {test_name}")
        print("=" * 50)

        # Test baseline performance
        print("  Testing baseline implementation...")
        baseline_timing = self.timer.measure(baseline_func, *args, **kwargs)
        baseline_memory = self.memory.measure(baseline_func, *args, **kwargs)

        # Test optimized performance
        print("  Testing optimized implementation...")
        optimized_timing = self.timer.measure(optimized_func, *args, **kwargs)
        optimized_memory = self.memory.measure(optimized_func, *args, **kwargs)

        # Compare accuracy
        accuracy_comparison = self.accuracy.compare_outputs(
            baseline_timing['result'],
            optimized_timing['result']
        )

        # Calculate speedup
        speedup = baseline_timing['mean'] / optimized_timing['mean']
        memory_ratio = optimized_memory['peak_mb'] / baseline_memory['peak_mb']

        # Statistical significance of speedup
        baseline_ci = baseline_timing['confidence_95']
        optimized_ci = optimized_timing['confidence_95']
        speedup_significant = (baseline_timing['mean'] - baseline_ci) > (optimized_timing['mean'] + optimized_ci)

        results = {
            'test_name': test_name,
            'baseline': {
                'timing': baseline_timing,
                'memory': baseline_memory
            },
            'optimized': {
                'timing': optimized_timing,
                'memory': optimized_memory
            },
            'comparison': {
                'speedup': speedup,
                'memory_ratio': memory_ratio,
                'accuracy': accuracy_comparison,
                'speedup_significant': speedup_significant
            }
        }

        # Print results
        self._print_results(results)

        return results

    def _print_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        baseline = results['baseline']
        optimized = results['optimized']
        comparison = results['comparison']

        print(f"\n  📊 Results:")
        print(f"    Baseline:   {baseline['timing']['mean']*1000:.3f} ± {baseline['timing']['confidence_95']*1000:.3f} ms")
        print(f"    Optimized:  {optimized['timing']['mean']*1000:.3f} ± {optimized['timing']['confidence_95']*1000:.3f} ms")
        print(f"    Speedup:    {comparison['speedup']:.2f}× {'✅ significant' if comparison['speedup_significant'] else '⚠️ not significant'}")

        print(f"\n    Memory Usage:")
        print(f"    Baseline:   {baseline['memory']['peak_mb']:.2f} MB")
        print(f"    Optimized:  {optimized['memory']['peak_mb']:.2f} MB")
        print(f"    Ratio:      {comparison['memory_ratio']:.2f}× {'(less memory)' if comparison['memory_ratio'] < 1 else '(more memory)'}")

        print(f"\n    Accuracy:")
        if comparison['accuracy']['shapes_match']:
            print(f"    Max diff:   {comparison['accuracy']['max_diff']:.2e}")
            print(f"    Accuracy:   {'✅ preserved' if comparison['accuracy']['accuracy_preserved'] else '❌ lost'}")
        else:
            print(f"    Shapes:     ❌ don't match")

        # Overall assessment
        overall_success = (
            comparison['speedup'] > 1.1 and  # At least 10% speedup
            comparison['speedup_significant'] and  # Statistically significant
            comparison['accuracy']['accuracy_preserved']  # Accuracy preserved
        )

        print(f"\n  🎯 Overall: {'✅ OPTIMIZATION SUCCESSFUL' if overall_success else '⚠️ NEEDS IMPROVEMENT'}")


def create_test_data(size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Create standard test data for benchmarks."""
    np.random.seed(42)  # Reproducible results
    X = np.random.randn(size, size).astype(np.float32)
    y = np.random.randn(size, size).astype(np.float32)
    return X, y


if __name__ == "__main__":
    # Demo of the framework
    print("🧪 TinyTorch Performance Testing Framework")
    print("=========================================")

    # Example: Compare naive vs numpy matrix multiplication
    def naive_matmul(a, b):
        """Naive O(n³) matrix multiplication."""
        n, m = a.shape[0], b.shape[1]
        k = a.shape[1]
        result = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                for idx in range(k):
                    result[i, j] += a[i, idx] * b[idx, j]
        return result

    def optimized_matmul(a, b):
        """NumPy optimized matrix multiplication."""
        return np.dot(a, b)

    # Test with small matrices for speed
    test_size = 100
    A, B = create_test_data(test_size)

    tester = PerformanceTester(warmup_runs=2, timing_runs=5)
    results = tester.compare_performance(
        naive_matmul, optimized_matmul,
        args=(A, B),
        test_name="Matrix Multiplication: Naive vs NumPy"
    )

    print(f"\nFramework demonstrates real {results['comparison']['speedup']:.1f}× speedup!")
