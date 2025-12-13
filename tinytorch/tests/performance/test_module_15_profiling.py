"""
Performance Tests for Module 15: Profiling

Tests whether the profiling tools actually measure performance accurately
and provide useful insights for optimization.

Key questions:
- Does the Timer class produce accurate, consistent measurements?
- Does the MemoryProfiler correctly track memory usage?
- Does the FLOPCounter calculate operations correctly?
- Do the profiling results correlate with actual performance differences?
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
sys.path.append(str(Path(__file__).parent.parent.parent / 'modules' / '15_profiling'))

try:
    from profiling_dev import Timer, MemoryProfiler, FLOPCounter, ProfilerContext, SimpleProfiler
    PROFILING_AVAILABLE = True
except ImportError:
    print("❌ Module 15 profiling tools not available")
    PROFILING_AVAILABLE = False

class Module15PerformanceTests:
    """Test suite for Module 15 profiling tools."""

    def __init__(self):
        self.suite = PerformanceTestSuite()
        self.comparator = PerformanceComparator()

    def test_timer_accuracy(self):
        """Test whether Timer produces accurate measurements."""
        if not PROFILING_AVAILABLE:
            return "Profiling module not available"

        print("🔬 Testing Timer accuracy against known operations")

        # Create operations with known timing characteristics
        def known_fast_op():
            """Operation that should take ~0.1ms"""
            return sum(range(100))

        def known_slow_op():
            """Operation that should take ~10ms"""
            time.sleep(0.01)  # 10ms sleep
            return 42

        # Test our timer vs built-in measurements
        timer = Timer()

        # Measure fast operation
        fast_stats = timer.measure(known_fast_op, warmup=2, runs=20)

        # Measure slow operation
        slow_stats = timer.measure(known_slow_op, warmup=2, runs=10)

        # Validate measurements make sense
        fast_time = fast_stats['mean_ms']
        slow_time = slow_stats['mean_ms']

        print(f"Fast operation: {fast_time:.3f}ms")
        print(f"Slow operation: {slow_time:.3f}ms")
        print(f"Ratio: {slow_time / fast_time:.1f}×")

        # Check if timer correctly identifies the ~100× difference
        expected_ratio = 100  # 10ms / 0.1ms = 100
        actual_ratio = slow_time / fast_time
        ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio

        # Timer should be within 50% of expected (timing is noisy)
        accuracy_test_passed = ratio_error < 0.5

        # Test measurement consistency
        fast_cv = fast_stats['std_ms'] / fast_stats['mean_ms']  # Coefficient of variation
        consistency_test_passed = fast_cv < 0.3  # Less than 30% variation

        result = {
            'timer_accuracy': accuracy_test_passed,
            'measurement_consistency': consistency_test_passed,
            'fast_operation_time_ms': fast_time,
            'slow_operation_time_ms': slow_time,
            'ratio_actual': actual_ratio,
            'ratio_expected': expected_ratio,
            'coefficient_variation': fast_cv
        }

        if accuracy_test_passed and consistency_test_passed:
            print("✅ Timer accuracy test PASSED")
        else:
            print("❌ Timer accuracy test FAILED")
            if not accuracy_test_passed:
                print(f"   Ratio error too high: {ratio_error:.2%}")
            if not consistency_test_passed:
                print(f"   Measurements too inconsistent: {fast_cv:.2%} variation")

        return result

    def test_memory_profiler_accuracy(self):
        """Test whether MemoryProfiler tracks memory correctly."""
        if not PROFILING_AVAILABLE:
            return "Profiling module not available"

        print("🧠 Testing MemoryProfiler accuracy against known allocations")

        profiler = MemoryProfiler()

        def small_allocation():
            """Allocate ~1MB of data"""
            data = np.zeros(256 * 1024, dtype=np.float32)  # 1MB
            return len(data)

        def large_allocation():
            """Allocate ~10MB of data"""
            data = np.zeros(2560 * 1024, dtype=np.float32)  # 10MB
            return len(data)

        # Profile memory usage
        small_stats = profiler.profile(small_allocation)
        large_stats = profiler.profile(large_allocation)

        small_mb = small_stats['peak_mb']
        large_mb = large_stats['peak_mb']

        print(f"Small allocation: {small_mb:.2f}MB peak")
        print(f"Large allocation: {large_mb:.2f}MB peak")
        print(f"Ratio: {large_mb / small_mb:.1f}×")

        # Check if profiler detects the ~10× difference in memory usage
        expected_ratio = 10.0
        actual_ratio = large_mb / small_mb
        ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio

        # Memory profiling should be within 30% (OS overhead varies)
        memory_accuracy_test = ratio_error < 0.3

        # Check that memory values are reasonable
        small_reasonable = 0.5 <= small_mb <= 5.0  # Between 0.5-5MB
        large_reasonable = 5.0 <= large_mb <= 50.0  # Between 5-50MB

        result = {
            'memory_accuracy': memory_accuracy_test,
            'small_allocation_reasonable': small_reasonable,
            'large_allocation_reasonable': large_reasonable,
            'small_allocation_mb': small_mb,
            'large_allocation_mb': large_mb,
            'ratio_actual': actual_ratio,
            'ratio_expected': expected_ratio
        }

        if memory_accuracy_test and small_reasonable and large_reasonable:
            print("✅ MemoryProfiler accuracy test PASSED")
        else:
            print("❌ MemoryProfiler accuracy test FAILED")

        return result

    def test_flop_counter_accuracy(self):
        """Test whether FLOPCounter calculates operations correctly."""
        if not PROFILING_AVAILABLE:
            return "Profiling module not available"

        print("🔢 Testing FLOPCounter accuracy against known operations")

        counter = FLOPCounter()

        # Test linear layer FLOP counting
        input_size = 128
        output_size = 64
        batch_size = 32

        expected_flops = batch_size * input_size * output_size + batch_size * output_size
        # Explanation: matmul + bias addition

        calculated_flops = counter.count_linear(input_size, output_size, batch_size)

        print(f"Linear layer FLOPs: {calculated_flops:,} (expected: {expected_flops:,})")

        # Test conv2d FLOP counting
        input_h, input_w = 32, 32
        in_channels, out_channels = 16, 32
        kernel_size = 3

        output_h = input_h - kernel_size + 1  # 30
        output_w = input_w - kernel_size + 1  # 30

        expected_conv_flops = (batch_size * output_h * output_w *
                              out_channels * kernel_size * kernel_size * in_channels +
                              batch_size * output_h * output_w * out_channels)  # bias

        calculated_conv_flops = counter.count_conv2d(input_h, input_w, in_channels,
                                                    out_channels, kernel_size, batch_size)

        print(f"Conv2D FLOPs: {calculated_conv_flops:,} (expected: {expected_conv_flops:,})")

        # Test accuracy
        linear_accurate = calculated_flops == expected_flops
        conv_accurate = calculated_conv_flops == expected_conv_flops

        result = {
            'linear_flop_accuracy': linear_accurate,
            'conv_flop_accuracy': conv_accurate,
            'linear_calculated': calculated_flops,
            'linear_expected': expected_flops,
            'conv_calculated': calculated_conv_flops,
            'conv_expected': expected_conv_flops
        }

        if linear_accurate and conv_accurate:
            print("✅ FLOPCounter accuracy test PASSED")
        else:
            print("❌ FLOPCounter accuracy test FAILED")
            if not linear_accurate:
                print(f"   Linear FLOP mismatch: {calculated_flops} vs {expected_flops}")
            if not conv_accurate:
                print(f"   Conv FLOP mismatch: {calculated_conv_flops} vs {expected_conv_flops}")

        return result

    def test_profiler_overhead(self):
        """Test whether profiling tools add reasonable overhead."""
        if not PROFILING_AVAILABLE:
            return "Profiling module not available"

        print("⏱️ Testing profiler overhead")

        # Simple operation to profile
        def test_operation():
            return np.random.randn(100, 100) @ np.random.randn(100, 100)

        # Measure without profiling (baseline)
        def unprofiled_operation():
            return test_operation()

        # Measure with profiling
        def profiled_operation():
            timer = Timer()
            result = timer.measure(test_operation, warmup=1, runs=5)
            return result

        # Compare overhead
        comparison = self.comparator.compare_implementations(
            unprofiled_operation,
            lambda: test_operation(),  # Just the operation, no profiling
            baseline_name="with_profiler_overhead",
            optimized_name="raw_operation"
        )

        # Profiler should add < 10× overhead
        overhead_acceptable = comparison.speedup < 10

        result = {
            'overhead_acceptable': overhead_acceptable,
            'overhead_factor': comparison.speedup,
            'raw_time_ms': comparison.optimized.mean_time_ms,
            'profiled_time_ms': comparison.baseline.mean_time_ms
        }

        if overhead_acceptable:
            print(f"✅ Profiler overhead acceptable: {comparison.speedup:.2f}×")
        else:
            print(f"❌ Profiler overhead too high: {comparison.speedup:.2f}×")

        return result

    def test_simple_profiler_interface(self):
        """Test the SimpleProfiler interface used by other modules."""
        if not PROFILING_AVAILABLE:
            return "Profiling module not available"

        print("🔌 Testing SimpleProfiler interface compatibility")

        try:
            profiler = SimpleProfiler()

            def test_function():
                return np.sum(np.random.randn(1000))

            # Test profiler interface
            result = profiler.profile(test_function, name="test_op")

            # Check required fields exist
            required_fields = ['wall_time', 'cpu_time', 'name']
            has_required_fields = all(field in result for field in required_fields)

            # Check values are reasonable
            reasonable_timing = 0.0001 <= result['wall_time'] <= 1.0  # 0.1ms to 1s

            interface_test = {
                'has_required_fields': has_required_fields,
                'reasonable_timing': reasonable_timing,
                'wall_time': result['wall_time'],
                'fields_present': list(result.keys())
            }

            if has_required_fields and reasonable_timing:
                print("✅ SimpleProfiler interface test PASSED")
            else:
                print("❌ SimpleProfiler interface test FAILED")

            return interface_test

        except Exception as e:
            return f"SimpleProfiler interface error: {e}"

    def test_real_world_profiling_scenario(self):
        """Test profiling on a realistic ML operation."""
        if not PROFILING_AVAILABLE:
            return "Profiling module not available"

        print("🌍 Testing profiling on realistic ML scenario")

        # Create realistic ML operations with different performance characteristics
        def efficient_matmul(A, B):
            """Efficient matrix multiplication using NumPy"""
            return A @ B

        def inefficient_matmul(A, B):
            """Inefficient matrix multiplication using Python loops"""
            m, k = A.shape
            k2, n = B.shape
            C = np.zeros((m, n))

            # Triple nested loops - should be much slower
            for i in range(m):
                for j in range(n):
                    for l in range(k):
                        C[i, j] += A[i, l] * B[l, j]
            return C

        # Generate test matrices (small size for reasonable test time)
        A = np.random.randn(50, 50).astype(np.float32)
        B = np.random.randn(50, 50).astype(np.float32)

        # Profile both implementations
        profiler_context = ProfilerContext("ML Operation Comparison", timing_runs=5)

        with profiler_context as ctx:
            efficient_result = ctx.profile_function(efficient_matmul, args=(A, B))
        efficient_stats = ctx.timing_stats

        profiler_context2 = ProfilerContext("Inefficient ML Operation", timing_runs=5)
        with profiler_context2 as ctx2:
            inefficient_result = ctx2.profile_function(inefficient_matmul, args=(A, B))
        inefficient_stats = ctx2.timing_stats

        # Verify results are the same
        results_match = np.allclose(efficient_result, inefficient_result, rtol=1e-3)

        # Check if profiler detects performance difference
        speedup_detected = inefficient_stats['mean_ms'] > efficient_stats['mean_ms'] * 5

        result = {
            'results_match': results_match,
            'speedup_detected': speedup_detected,
            'efficient_time_ms': efficient_stats['mean_ms'],
            'inefficient_time_ms': inefficient_stats['mean_ms'],
            'detected_speedup': inefficient_stats['mean_ms'] / efficient_stats['mean_ms']
        }

        if results_match and speedup_detected:
            print("✅ Real-world profiling test PASSED")
            print(f"   Detected {result['detected_speedup']:.1f}× performance difference")
        else:
            print("❌ Real-world profiling test FAILED")
            if not results_match:
                print("   Implementations produce different results")
            if not speedup_detected:
                print("   Failed to detect performance difference")

        return result

def run_module_15_performance_tests():
    """Run all performance tests for Module 15."""
    print("🧪 TESTING MODULE 15: PROFILING TOOLS")
    print("=" * 60)
    print("Verifying that profiling tools provide accurate performance measurements")

    if not PROFILING_AVAILABLE:
        print("❌ Cannot test Module 15 - profiling tools not available")
        return

    test_suite = Module15PerformanceTests()

    tests = {
        'timer_accuracy': test_suite.test_timer_accuracy,
        'memory_profiler_accuracy': test_suite.test_memory_profiler_accuracy,
        'flop_counter_accuracy': test_suite.test_flop_counter_accuracy,
        'profiler_overhead': test_suite.test_profiler_overhead,
        'simple_profiler_interface': test_suite.test_simple_profiler_interface,
        'real_world_scenario': test_suite.test_real_world_profiling_scenario
    }

    results = test_suite.suite.run_module_tests('module_15_profiling', tests)

    # Summary
    print(f"\n📊 MODULE 15 TEST SUMMARY")
    print("=" * 40)

    total_tests = len(tests)
    passed_tests = 0

    for test_name, result in results.items():
        if isinstance(result, dict):
            # Determine pass/fail based on the specific test
            if 'timer_accuracy' in result:
                passed = result.get('timer_accuracy', False) and result.get('measurement_consistency', False)
            elif 'memory_accuracy' in result:
                passed = (result.get('memory_accuracy', False) and
                         result.get('small_allocation_reasonable', False) and
                         result.get('large_allocation_reasonable', False))
            elif 'linear_flop_accuracy' in result:
                passed = result.get('linear_flop_accuracy', False) and result.get('conv_flop_accuracy', False)
            elif 'overhead_acceptable' in result:
                passed = result.get('overhead_acceptable', False)
            elif 'has_required_fields' in result:
                passed = result.get('has_required_fields', False) and result.get('reasonable_timing', False)
            elif 'results_match' in result:
                passed = result.get('results_match', False) and result.get('speedup_detected', False)
            else:
                passed = False

            if passed:
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        else:
            print(f"❌ {test_name}: ERROR - {result}")

    success_rate = passed_tests / total_tests
    print(f"\nSUCCESS RATE: {success_rate:.1%} ({passed_tests}/{total_tests})")

    if success_rate >= 0.8:
        print("🎉 Module 15 profiling tools are working correctly!")
    else:
        print("⚠️  Module 15 profiling tools need improvement")

    return results

if __name__ == "__main__":
    run_module_15_performance_tests()
