"""
Performance Tests for Module 16: Hardware Acceleration

Tests whether the acceleration techniques actually provide measurable speedups
over baseline implementations.

Key questions:
- Does blocked matrix multiplication actually improve cache performance?
- How much faster is NumPy compared to naive loops?
- Does the smart backend system work correctly?
- Are the claimed 10-100× speedups realistic?
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add the performance framework to path
sys.path.append(str(Path(__file__).parent))
from performance_test_framework import PerformanceTestSuite, PerformanceComparator, WorkloadGenerator

# Add module path
sys.path.append(str(Path(__file__).parent.parent.parent / 'modules' / '16_acceleration'))

try:
    from acceleration_dev import (
        matmul_naive, matmul_blocked, matmul_numpy,
        OptimizedBackend, matmul
    )
    ACCELERATION_AVAILABLE = True
except ImportError:
    print("❌ Module 16 acceleration tools not available")
    ACCELERATION_AVAILABLE = False

class Module16PerformanceTests:
    """Test suite for Module 16 acceleration techniques."""

    def __init__(self):
        self.suite = PerformanceTestSuite()
        self.comparator = PerformanceComparator()
        self.workloads = WorkloadGenerator()

    def test_naive_vs_blocked_matmul(self):
        """Test whether blocked matrix multiplication improves over naive loops."""
        if not ACCELERATION_AVAILABLE:
            return "Acceleration module not available"

        print("🔄 Testing naive vs blocked matrix multiplication")

        # Use small matrices for naive implementation (it's very slow)
        size = 64  # Small enough that naive doesn't take forever
        A, B = self.workloads.matrix_multiply_workload(size)

        # Wrapper functions for testing
        def naive_implementation():
            return matmul_naive(A, B)

        def blocked_implementation():
            return matmul_blocked(A, B, block_size=32)

        # First verify results are the same
        try:
            naive_result = naive_implementation()
            blocked_result = blocked_implementation()
            numpy_result = A @ B

            # Check correctness
            naive_correct = np.allclose(naive_result, numpy_result, rtol=1e-3, atol=1e-3)
            blocked_correct = np.allclose(blocked_result, numpy_result, rtol=1e-3, atol=1e-3)

            if not naive_correct:
                return "Naive implementation produces incorrect results"
            if not blocked_correct:
                return "Blocked implementation produces incorrect results"

        except Exception as e:
            return f"Implementation error: {e}"

        # Performance comparison
        comparison = self.comparator.compare_implementations(
            naive_implementation,
            blocked_implementation,
            baseline_name="naive_matmul",
            optimized_name="blocked_matmul"
        )

        # Blocked should be faster than naive (cache-friendly access)
        speedup_achieved = comparison.speedup > 1.2  # At least 20% improvement

        result = {
            'correctness_naive': naive_correct,
            'correctness_blocked': blocked_correct,
            'speedup': comparison.speedup,
            'speedup_achieved': speedup_achieved,
            'naive_time_ms': comparison.baseline.mean_time_ms,
            'blocked_time_ms': comparison.optimized.mean_time_ms,
            'matrix_size': size
        }

        if speedup_achieved:
            print(f"✅ Blocked matmul speedup achieved: {comparison.speedup:.2f}×")
        else:
            print(f"❌ Blocked matmul speedup insufficient: {comparison.speedup:.2f}×")

        return comparison

    def test_blocked_vs_numpy_matmul(self):
        """Test blocked implementation against NumPy (production baseline)."""
        if not ACCELERATION_AVAILABLE:
            return "Acceleration module not available"

        print("🚀 Testing blocked vs NumPy matrix multiplication")

        # Use medium size matrices
        size = 256
        A, B = self.workloads.matrix_multiply_workload(size)

        def blocked_implementation():
            return matmul_blocked(A, B, block_size=64)

        def numpy_implementation():
            return matmul_numpy(A, B)

        # Verify correctness
        try:
            blocked_result = blocked_implementation()
            numpy_result = numpy_implementation()

            results_match = np.allclose(blocked_result, numpy_result, rtol=1e-3, atol=1e-3)
            if not results_match:
                return "Blocked and NumPy implementations produce different results"

        except Exception as e:
            return f"Implementation error: {e}"

        # Performance comparison
        comparison = self.comparator.compare_implementations(
            blocked_implementation,
            numpy_implementation,
            baseline_name="blocked_matmul",
            optimized_name="numpy_matmul"
        )

        # NumPy should be significantly faster than blocked
        numpy_advantage = comparison.speedup > 2.0  # NumPy should be 2×+ faster

        result = {
            'correctness': results_match,
            'numpy_speedup': comparison.speedup,
            'numpy_advantage': numpy_advantage,
            'blocked_time_ms': comparison.baseline.mean_time_ms,
            'numpy_time_ms': comparison.optimized.mean_time_ms,
            'matrix_size': size
        }

        if numpy_advantage:
            print(f"✅ NumPy dominance confirmed: {comparison.speedup:.2f}× faster than blocked")
        else:
            print(f"⚠️  NumPy advantage lower than expected: {comparison.speedup:.2f}×")

        return comparison

    def test_naive_vs_numpy_full_spectrum(self):
        """Test the full optimization spectrum: naive → blocked → NumPy."""
        if not ACCELERATION_AVAILABLE:
            return "Acceleration module not available"

        print("📊 Testing full optimization spectrum")

        # Use very small matrix for naive (it's extremely slow)
        size = 32
        A, B = self.workloads.matrix_multiply_workload(size)

        def naive_impl():
            return matmul_naive(A, B)

        def numpy_impl():
            return matmul_numpy(A, B)

        # Test naive vs NumPy to see full improvement
        comparison = self.comparator.compare_implementations(
            naive_impl,
            numpy_impl,
            baseline_name="naive_loops",
            optimized_name="numpy_optimized"
        )

        # Should see dramatic improvement (10×+ claimed in module)
        dramatic_improvement = comparison.speedup > 5.0

        result = {
            'full_spectrum_speedup': comparison.speedup,
            'dramatic_improvement': dramatic_improvement,
            'naive_time_ms': comparison.baseline.mean_time_ms,
            'numpy_time_ms': comparison.optimized.mean_time_ms,
            'matrix_size': size
        }

        if dramatic_improvement:
            print(f"🎉 Dramatic optimization achieved: {comparison.speedup:.1f}× improvement!")
        else:
            print(f"⚠️  Full optimization less dramatic: {comparison.speedup:.1f}× improvement")

        return comparison

    def test_backend_system(self):
        """Test the smart backend dispatch system."""
        if not ACCELERATION_AVAILABLE:
            return "Acceleration module not available"

        print("🧠 Testing smart backend system")

        size = 128
        A, B = self.workloads.matrix_multiply_workload(size)

        # Test backend function
        def backend_matmul():
            return matmul(A, B)

        def direct_numpy():
            return matmul_numpy(A, B)

        # Verify results match
        try:
            backend_result = backend_matmul()
            numpy_result = direct_numpy()

            results_match = np.allclose(backend_result, numpy_result, rtol=1e-5, atol=1e-5)
            if not results_match:
                return "Backend system produces different results than NumPy"

        except Exception as e:
            return f"Backend system error: {e}"

        # Performance should be equivalent (backend uses NumPy)
        comparison = self.comparator.compare_implementations(
            backend_matmul,
            direct_numpy,
            baseline_name="backend_matmul",
            optimized_name="direct_numpy"
        )

        # Backend should have minimal overhead (< 20%)
        low_overhead = comparison.speedup < 1.2 and comparison.speedup > 0.8

        result = {
            'correctness': results_match,
            'overhead_factor': comparison.speedup,
            'low_overhead': low_overhead,
            'backend_time_ms': comparison.baseline.mean_time_ms,
            'numpy_time_ms': comparison.optimized.mean_time_ms
        }

        if low_overhead:
            print(f"✅ Backend overhead acceptable: {comparison.speedup:.2f}× factor")
        else:
            print(f"❌ Backend overhead too high: {comparison.speedup:.2f}× factor")

        return result

    def test_scaling_behavior(self):
        """Test how optimizations scale with matrix size."""
        if not ACCELERATION_AVAILABLE:
            return "Acceleration module not available"

        print("📈 Testing optimization scaling behavior")

        sizes = [64, 128, 256]  # Keep reasonable for testing
        results = {}

        for size in sizes:
            print(f"  Testing size {size}×{size}")
            A, B = self.workloads.matrix_multiply_workload(size)

            # Compare blocked vs NumPy at this size
            def blocked_impl():
                return matmul_blocked(A, B, block_size=min(64, size//2))

            def numpy_impl():
                return matmul_numpy(A, B)

            # Quick timing comparison (fewer runs for speed)
            timer = self.comparator.timer
            timer.measurement_runs = 10

            comparison = self.comparator.compare_implementations(
                blocked_impl, numpy_impl,
                baseline_name=f"blocked_{size}",
                optimized_name=f"numpy_{size}"
            )

            results[size] = {
                'speedup': comparison.speedup,
                'blocked_time_ms': comparison.baseline.mean_time_ms,
                'numpy_time_ms': comparison.optimized.mean_time_ms
            }

        # Analyze scaling trends
        speedups = [results[size]['speedup'] for size in sizes]
        speedup_increases = all(speedups[i] <= speedups[i+1] for i in range(len(speedups)-1))

        scaling_result = {
            'size_results': results,
            'speedup_increases_with_size': speedup_increases,
            'speedups': speedups,
            'sizes': sizes
        }

        print(f"Speedup scaling: {' → '.join(f'{s:.1f}×' for s in speedups)}")

        if speedup_increases:
            print("✅ NumPy advantage increases with size (expected)")
        else:
            print("⚠️  Inconsistent scaling behavior")

        return scaling_result

    def test_cache_blocking_effectiveness(self):
        """Test whether blocking actually improves cache performance."""
        if not ACCELERATION_AVAILABLE:
            return "Acceleration module not available"

        print("💾 Testing cache blocking effectiveness")

        # Test different block sizes
        size = 128
        A, B = self.workloads.matrix_multiply_workload(size)

        block_sizes = [16, 32, 64, 128]
        block_results = {}

        for block_size in block_sizes:
            def blocked_impl():
                return matmul_blocked(A, B, block_size=block_size)

            timer = self.comparator.timer
            timer.measurement_runs = 10

            result = timer.measure_function(blocked_impl, name=f"block_{block_size}")
            block_results[block_size] = result.mean_time_ms

        # Find optimal block size (should be around 32-64 for typical L1 cache)
        optimal_block_size = min(block_results.keys(), key=lambda k: block_results[k])
        performance_variation = max(block_results.values()) / min(block_results.values())

        cache_result = {
            'block_sizes': list(block_sizes),
            'timings_ms': list(block_results.values()),
            'optimal_block_size': optimal_block_size,
            'performance_variation': performance_variation,
            'cache_blocking_effective': performance_variation > 1.2
        }

        print(f"Block size performance: {dict(block_results)}")
        print(f"Optimal block size: {optimal_block_size}")

        if cache_result['cache_blocking_effective']:
            print(f"✅ Cache blocking shows {performance_variation:.1f}× variation")
        else:
            print(f"❌ Cache blocking shows minimal impact: {performance_variation:.1f}× variation")

        return cache_result

    def test_ml_model_acceleration(self):
        """Test acceleration on realistic ML model operations."""
        if not ACCELERATION_AVAILABLE:
            return "Acceleration module not available"

        print("🤖 Testing acceleration on ML model operations")

        # Simulate MLP forward pass
        batch_size = 32
        input_dim = 256
        hidden_dim = 128
        output_dim = 64

        # Create model data
        x = np.random.randn(batch_size, input_dim).astype(np.float32)
        W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32)
        W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32)

        def naive_mlp():
            # Use naive matmul for "educational" version (very small for speed)
            x_small = x[:4, :32]  # Much smaller for naive
            W1_small = W1[:32, :16]
            W2_small = W2[:16, :8]

            h1 = matmul_naive(x_small, W1_small)
            h1_relu = np.maximum(0, h1)
            output = matmul_naive(h1_relu, W2_small)
            return output

        def optimized_mlp():
            h1 = matmul(x, W1)
            h1_relu = np.maximum(0, h1)
            output = matmul(h1_relu, W2)
            return output

        try:
            # Time both implementations
            timer = self.comparator.timer
            timer.measurement_runs = 5  # Fewer runs since naive is slow

            naive_result = timer.measure_function(naive_mlp, name="naive_mlp")
            optimized_result = timer.measure_function(optimized_mlp, name="optimized_mlp")

            # Compare (note: different sizes, so this is qualitative)
            ml_acceleration = {
                'naive_time_ms': naive_result.mean_time_ms,
                'optimized_time_ms': optimized_result.mean_time_ms,
                'operations_comparison': "Different sizes - qualitative comparison",
                'naive_much_slower': naive_result.mean_time_ms > optimized_result.mean_time_ms
            }

            if ml_acceleration['naive_much_slower']:
                print("✅ ML acceleration effective - optimized version much faster")
            else:
                print("❌ ML acceleration test inconclusive")

            return ml_acceleration

        except Exception as e:
            return f"ML acceleration test error: {e}"

def run_module_16_performance_tests():
    """Run all performance tests for Module 16."""
    print("🧪 TESTING MODULE 16: HARDWARE ACCELERATION")
    print("=" * 60)
    print("Verifying that acceleration techniques provide real speedups")

    if not ACCELERATION_AVAILABLE:
        print("❌ Cannot test Module 16 - acceleration tools not available")
        return

    test_suite = Module16PerformanceTests()

    tests = {
        'naive_vs_blocked': test_suite.test_naive_vs_blocked_matmul,
        'blocked_vs_numpy': test_suite.test_blocked_vs_numpy_matmul,
        'full_spectrum': test_suite.test_naive_vs_numpy_full_spectrum,
        'backend_system': test_suite.test_backend_system,
        'scaling_behavior': test_suite.test_scaling_behavior,
        'cache_blocking': test_suite.test_cache_blocking_effectiveness,
        'ml_model_acceleration': test_suite.test_ml_model_acceleration
    }

    results = test_suite.suite.run_module_tests('module_16_acceleration', tests)

    # Summary
    print(f"\n📊 MODULE 16 TEST SUMMARY")
    print("=" * 40)

    speedup_tests = []
    correctness_tests = []

    for test_name, result in results.items():
        if hasattr(result, 'speedup'):  # ComparisonResult
            speedup_tests.append((test_name, result.speedup, result.is_significant))
            print(f"⚡ {test_name}: {result.speedup:.2f}× speedup {'✅' if result.is_significant else '❌'}")
        elif isinstance(result, dict):
            # Check for various success criteria
            success = False
            if 'speedup_achieved' in result:
                success = result['speedup_achieved']
            elif 'dramatic_improvement' in result:
                success = result['dramatic_improvement']
            elif 'low_overhead' in result:
                success = result['low_overhead']
            elif 'cache_blocking_effective' in result:
                success = result['cache_blocking_effective']

            correctness_tests.append((test_name, success))
            print(f"🔧 {test_name}: {'✅ PASS' if success else '❌ FAIL'}")
        else:
            print(f"❌ {test_name}: ERROR - {result}")

    # Overall assessment
    significant_speedups = sum(1 for _, speedup, significant in speedup_tests if significant and speedup > 1.5)
    successful_tests = sum(1 for _, success in correctness_tests if success)

    total_meaningful_tests = len(speedup_tests) + len(correctness_tests)
    total_successes = significant_speedups + successful_tests

    success_rate = total_successes / total_meaningful_tests if total_meaningful_tests > 0 else 0

    print(f"\nSUCCESS RATE: {success_rate:.1%} ({total_successes}/{total_meaningful_tests})")
    print(f"Significant speedups: {significant_speedups}/{len(speedup_tests)}")
    print(f"System tests passed: {successful_tests}/{len(correctness_tests)}")

    if success_rate >= 0.7:
        print("🎉 Module 16 acceleration techniques are working well!")
    else:
        print("⚠️  Module 16 acceleration techniques need improvement")

    return results

if __name__ == "__main__":
    run_module_16_performance_tests()
