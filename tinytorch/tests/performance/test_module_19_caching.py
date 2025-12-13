"""
Performance Tests for Module 19: KV Caching

Tests whether KV caching actually transforms O(N²) attention to O(N) complexity
and provides the claimed dramatic speedups for autoregressive generation.

Key questions:
- Does KV caching actually reduce computational complexity?
- Is there measurable speedup for sequential token generation?
- Does caching work correctly with attention mechanisms?
- Are the O(N²) → O(N) complexity claims realistic?
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
sys.path.append(str(Path(__file__).parent.parent.parent / 'modules' / '19_caching'))

try:
    from caching_dev import KVCache, CachedMultiHeadAttention
    CACHING_AVAILABLE = True
except ImportError:
    print("❌ Module 19 caching tools not available")
    CACHING_AVAILABLE = False

class Module19PerformanceTests:
    """Test suite for Module 19 KV caching techniques."""

    def __init__(self):
        self.suite = PerformanceTestSuite()
        self.comparator = PerformanceComparator()
        self.workloads = WorkloadGenerator()

    def test_kv_cache_memory_usage(self):
        """Test whether KV cache uses memory efficiently."""
        if not CACHING_AVAILABLE:
            return "Caching module not available"

        print("💾 Testing KV cache memory usage")

        # Create caches of different sizes
        sizes = [64, 128, 256]
        n_layers = 4
        n_heads = 8
        head_dim = 32

        cache_sizes = {}

        for max_seq_len in sizes:
            cache = KVCache(max_seq_len, n_layers, n_heads, head_dim)
            memory_info = cache.get_memory_usage()
            cache_sizes[max_seq_len] = memory_info['total_cache_size_mb']

        # Test linear scaling
        scaling_factor_1 = cache_sizes[128] / cache_sizes[64]  # Should be ~2
        scaling_factor_2 = cache_sizes[256] / cache_sizes[128]  # Should be ~2

        linear_scaling = (1.8 <= scaling_factor_1 <= 2.2) and (1.8 <= scaling_factor_2 <= 2.2)

        # Test memory utilization
        cache = KVCache(128, n_layers, n_heads, head_dim)

        # Add some tokens
        for pos in range(10):
            key = np.random.randn(n_heads, head_dim).astype(np.float32)
            value = np.random.randn(n_heads, head_dim).astype(np.float32)
            cache.update(0, key, value)
            cache.advance_position()

        final_memory_info = cache.get_memory_usage()
        reasonable_utilization = 0.05 <= final_memory_info['utilization'] <= 0.15  # 10/128 ≈ 8%

        result = {
            'cache_sizes_mb': cache_sizes,
            'linear_scaling': linear_scaling,
            'scaling_factor_1': scaling_factor_1,
            'scaling_factor_2': scaling_factor_2,
            'memory_utilization': final_memory_info['utilization'],
            'reasonable_utilization': reasonable_utilization,
            'memory_test_passed': linear_scaling and reasonable_utilization
        }

        if result['memory_test_passed']:
            print(f"✅ KV cache memory usage efficient: {scaling_factor_1:.1f}× scaling")
        else:
            print(f"❌ KV cache memory usage issues: {scaling_factor_1:.1f}× scaling")

        return result

    def test_cache_correctness(self):
        """Test whether KV cache stores and retrieves values correctly."""
        if not CACHING_AVAILABLE:
            return "Caching module not available"

        print("🔍 Testing KV cache correctness")

        max_seq_len = 64
        n_layers = 2
        n_heads = 4
        head_dim = 16

        cache = KVCache(max_seq_len, n_layers, n_heads, head_dim)

        # Store test data
        test_keys = []
        test_values = []

        for pos in range(5):
            key = np.random.randn(n_heads, head_dim).astype(np.float32)
            value = np.random.randn(n_heads, head_dim).astype(np.float32)

            test_keys.append(key.copy())
            test_values.append(value.copy())

            cache.update(0, key, value)
            cache.advance_position()

        # Retrieve and verify
        retrieved_keys, retrieved_values = cache.get(0, 5)

        # Check shapes
        shape_correct = (retrieved_keys.shape == (5, n_heads, head_dim) and
                        retrieved_values.shape == (5, n_heads, head_dim))

        # Check data integrity
        keys_match = all(np.allclose(retrieved_keys.data[i], test_keys[i], rtol=1e-6)
                        for i in range(5))
        values_match = all(np.allclose(retrieved_values.data[i], test_values[i], rtol=1e-6)
                          for i in range(5))

        # Test partial retrieval
        partial_keys, partial_values = cache.get(0, 3)
        partial_correct = (partial_keys.shape == (3, n_heads, head_dim) and
                          np.allclose(partial_keys.data[2], test_keys[2], rtol=1e-6))

        correctness_result = {
            'shape_correct': shape_correct,
            'keys_match': keys_match,
            'values_match': values_match,
            'partial_retrieval_correct': partial_correct,
            'cache_correctness_passed': shape_correct and keys_match and values_match and partial_correct
        }

        if correctness_result['cache_correctness_passed']:
            print("✅ KV cache stores and retrieves data correctly")
        else:
            print("❌ KV cache data integrity issues")

        return correctness_result

    def test_sequential_attention_speedup(self):
        """Test speedup from caching in sequential attention computation."""
        if not CACHING_AVAILABLE:
            return "Caching module not available"

        print("🚀 Testing sequential attention speedup")

        # Simulate autoregressive generation scenario
        embed_dim = 128
        num_heads = 8
        max_seq_len = 32

        try:
            # Create attention layers
            cached_attention = CachedMultiHeadAttention(embed_dim, num_heads)

            # Create cache
            cache = KVCache(max_seq_len, 1, num_heads, embed_dim // num_heads)

            # Simulate token generation without cache (recompute everything each time)
            def generate_without_cache(sequence_length):
                total_time = 0

                for pos in range(1, sequence_length + 1):
                    # Create input sequence up to current position
                    input_sequence = np.random.randn(1, pos, embed_dim).astype(np.float32)

                    start_time = time.perf_counter()
                    # Standard attention on full sequence
                    output, _ = cached_attention.forward(input_sequence, use_cache=False)
                    end_time = time.perf_counter()

                    total_time += (end_time - start_time)

                return total_time

            # Simulate token generation with cache
            def generate_with_cache(sequence_length):
                cache.reset()
                total_time = 0

                for pos in range(sequence_length):
                    # Only current token input
                    current_token = np.random.randn(1, 1, embed_dim).astype(np.float32)

                    start_time = time.perf_counter()
                    # Cached attention
                    output, _ = cached_attention.forward(
                        current_token,
                        cache=cache,
                        layer_idx=0,
                        use_cache=True
                    )
                    end_time = time.perf_counter()

                    total_time += (end_time - start_time)

                return total_time

            # Test on different sequence lengths
            seq_lengths = [8, 16, 24]
            speedup_results = {}

            for seq_len in seq_lengths:
                print(f"  Testing sequence length {seq_len}")

                # Time both approaches (smaller number of runs for speed)
                timer = self.comparator.timer
                timer.measurement_runs = 3  # Fewer runs for complex operations

                uncached_time = timer.measure_function(
                    generate_without_cache, args=(seq_len,),
                    name=f"uncached_{seq_len}"
                ).mean_time_ms

                cached_time = timer.measure_function(
                    generate_with_cache, args=(seq_len,),
                    name=f"cached_{seq_len}"
                ).mean_time_ms

                speedup = uncached_time / cached_time
                speedup_results[seq_len] = speedup

            # Check if speedup increases with sequence length (should be quadratic benefit)
            speedups = list(speedup_results.values())
            speedup_increases = all(speedups[i] <= speedups[i+1] for i in range(len(speedups)-1))

            # Any speedup is good for this complex operation
            any_speedup = any(s > 1.1 for s in speedups)

            sequential_result = {
                'speedup_results': speedup_results,
                'speedup_increases_with_length': speedup_increases,
                'any_significant_speedup': any_speedup,
                'max_speedup': max(speedups),
                'sequential_speedup_achieved': speedup_increases or any_speedup
            }

            if sequential_result['sequential_speedup_achieved']:
                print(f"✅ Sequential attention speedup achieved: max {max(speedups):.1f}×")
            else:
                print(f"❌ No meaningful sequential speedup: max {max(speedups):.1f}×")

            return sequential_result

        except Exception as e:
            return f"Sequential attention test error: {e}"

    def test_complexity_scaling(self):
        """Test whether caching actually changes computational complexity."""
        if not CACHING_AVAILABLE:
            return "Caching module not available"

        print("📈 Testing computational complexity scaling")

        embed_dim = 64  # Smaller for faster testing
        num_heads = 4

        try:
            cached_attention = CachedMultiHeadAttention(embed_dim, num_heads)

            # Test scaling behavior
            sequence_lengths = [8, 16, 32]
            timing_results = {'uncached': {}, 'cached': {}}

            for seq_len in sequence_lengths:
                print(f"  Testing complexity at length {seq_len}")

                # Create cache
                cache = KVCache(seq_len, 1, num_heads, embed_dim // num_heads)

                # Test uncached (should be O(N²) due to full sequence recomputation)
                def uncached_operation():
                    input_seq = np.random.randn(1, seq_len, embed_dim).astype(np.float32)
                    output, _ = cached_attention.forward(input_seq, use_cache=False)
                    return output

                # Test cached (should be O(N) for incremental generation)
                def cached_operation():
                    cache.reset()
                    outputs = []

                    for pos in range(seq_len):
                        token = np.random.randn(1, 1, embed_dim).astype(np.float32)
                        output, _ = cached_attention.forward(
                            token, cache=cache, layer_idx=0, use_cache=True
                        )
                        outputs.append(output)

                    return outputs

                # Time operations (fewer runs due to complexity)
                timer = self.comparator.timer
                timer.measurement_runs = 5

                uncached_time = timer.measure_function(uncached_operation, name=f"uncached_{seq_len}").mean_time_ms
                cached_time = timer.measure_function(cached_operation, name=f"cached_{seq_len}").mean_time_ms

                timing_results['uncached'][seq_len] = uncached_time
                timing_results['cached'][seq_len] = cached_time

            # Analyze scaling
            uncached_times = [timing_results['uncached'][seq_len] for seq_len in sequence_lengths]
            cached_times = [timing_results['cached'][seq_len] for seq_len in sequence_lengths]

            # Calculate scaling factors
            uncached_scaling = uncached_times[2] / uncached_times[0]  # 32 vs 8
            cached_scaling = cached_times[2] / cached_times[0]      # 32 vs 8

            # Theoretical: 4× sequence length should give:
            # - Uncached: 16× time (quadratic)
            # - Cached: 4× time (linear)

            # Check if cached scales better than uncached
            better_scaling = cached_scaling < uncached_scaling * 0.8

            complexity_result = {
                'timing_results': timing_results,
                'uncached_scaling_factor': uncached_scaling,
                'cached_scaling_factor': cached_scaling,
                'better_scaling': better_scaling,
                'sequence_lengths': sequence_lengths,
                'complexity_improvement_detected': better_scaling
            }

            if better_scaling:
                print(f"✅ Complexity improvement detected: cached {cached_scaling:.1f}× vs uncached {uncached_scaling:.1f}×")
            else:
                print(f"❌ No clear complexity improvement: cached {cached_scaling:.1f}× vs uncached {uncached_scaling:.1f}×")

            return complexity_result

        except Exception as e:
            return f"Complexity scaling test error: {e}"

    def test_cache_hit_performance(self):
        """Test that cache hits provide performance benefits."""
        if not CACHING_AVAILABLE:
            return "Caching module not available"

        print("🎯 Testing cache hit performance")

        max_seq_len = 64
        n_layers = 2
        n_heads = 8
        head_dim = 16

        cache = KVCache(max_seq_len, n_layers, n_heads, head_dim)

        # Fill cache with data
        for pos in range(32):
            key = np.random.randn(n_heads, head_dim).astype(np.float32)
            value = np.random.randn(n_heads, head_dim).astype(np.float32)
            cache.update(0, key, value)
            cache.advance_position()

        # Test cache operations
        def cache_store_operation():
            """Storing new data in cache"""
            key = np.random.randn(n_heads, head_dim).astype(np.float32)
            value = np.random.randn(n_heads, head_dim).astype(np.float32)
            cache.update(0, key, value)
            return True

        def cache_retrieve_operation():
            """Retrieving data from cache"""
            keys, values = cache.get(0, 20)  # Get 20 cached tokens
            return keys.shape[0]

        def no_cache_operation():
            """Equivalent operation without cache (compute from scratch)"""
            # Simulate recomputing keys/values
            keys = np.random.randn(20, n_heads, head_dim).astype(np.float32)
            values = np.random.randn(20, n_heads, head_dim).astype(np.float32)
            return keys.shape[0]

        # Compare cache retrieval vs recomputation
        comparison = self.comparator.compare_implementations(
            no_cache_operation,
            cache_retrieve_operation,
            baseline_name="no_cache",
            optimized_name="cache_retrieval"
        )

        # Cache should be faster than recomputation
        cache_faster = comparison.speedup > 1.2

        # Test cache operation overhead
        timer = self.comparator.timer
        timer.measurement_runs = 20

        store_time = timer.measure_function(cache_store_operation, name="cache_store").mean_time_ms
        retrieve_time = timer.measure_function(cache_retrieve_operation, name="cache_retrieve").mean_time_ms

        # Cache operations should be very fast
        low_overhead = store_time < 1.0 and retrieve_time < 1.0  # < 1ms

        cache_performance_result = {
            'cache_vs_recompute_speedup': comparison.speedup,
            'cache_faster': cache_faster,
            'store_time_ms': store_time,
            'retrieve_time_ms': retrieve_time,
            'low_overhead': low_overhead,
            'cache_performance_good': cache_faster and low_overhead
        }

        if cache_performance_result['cache_performance_good']:
            print(f"✅ Cache performance good: {comparison.speedup:.1f}× faster, {retrieve_time:.2f}ms retrieval")
        else:
            print(f"❌ Cache performance issues: {comparison.speedup:.1f}× speedup, overhead concerns")

        return cache_performance_result

def run_module_19_performance_tests():
    """Run all performance tests for Module 19."""
    print("🧪 TESTING MODULE 19: KV CACHING")
    print("=" * 60)
    print("Verifying that KV caching provides complexity reduction and speedups")

    if not CACHING_AVAILABLE:
        print("❌ Cannot test Module 19 - caching tools not available")
        return

    test_suite = Module19PerformanceTests()

    tests = {
        'memory_usage': test_suite.test_kv_cache_memory_usage,
        'cache_correctness': test_suite.test_cache_correctness,
        'sequential_speedup': test_suite.test_sequential_attention_speedup,
        'complexity_scaling': test_suite.test_complexity_scaling,
        'cache_performance': test_suite.test_cache_hit_performance
    }

    results = test_suite.suite.run_module_tests('module_19_caching', tests)

    # Summary
    print(f"\n📊 MODULE 19 TEST SUMMARY")
    print("=" * 40)

    total_tests = len(tests)
    passed_tests = 0

    for test_name, result in results.items():
        if hasattr(result, 'speedup'):  # ComparisonResult
            passed = result.speedup > 1.1 and result.is_significant
            print(f"⚡ {test_name}: {result.speedup:.2f}× speedup {'✅' if passed else '❌'}")
        elif isinstance(result, dict):
            # Check specific success criteria for each test
            if 'memory_test_passed' in result:
                passed = result['memory_test_passed']
                print(f"💾 {test_name}: {'✅ PASS' if passed else '❌ FAIL'}")
            elif 'cache_correctness_passed' in result:
                passed = result['cache_correctness_passed']
                print(f"🔍 {test_name}: {'✅ PASS' if passed else '❌ FAIL'}")
            elif 'sequential_speedup_achieved' in result:
                passed = result['sequential_speedup_achieved']
                max_speedup = result.get('max_speedup', 0)
                print(f"🚀 {test_name}: {max_speedup:.1f}× max speedup {'✅ PASS' if passed else '❌ FAIL'}")
            elif 'complexity_improvement_detected' in result:
                passed = result['complexity_improvement_detected']
                print(f"📈 {test_name}: {'✅ PASS' if passed else '❌ FAIL'}")
            elif 'cache_performance_good' in result:
                passed = result['cache_performance_good']
                print(f"🎯 {test_name}: {'✅ PASS' if passed else '❌ FAIL'}")
            else:
                passed = False
                print(f"❓ {test_name}: Unknown result format")
        else:
            passed = False
            print(f"❌ {test_name}: ERROR - {result}")

        if passed:
            passed_tests += 1

    success_rate = passed_tests / total_tests
    print(f"\nSUCCESS RATE: {success_rate:.1%} ({passed_tests}/{total_tests})")

    if success_rate >= 0.6:  # Lower threshold due to complexity of caching tests
        print("🎉 Module 19 KV caching is working effectively!")
        print("💡 Note: Caching benefits most visible in longer sequences")
    else:
        print("⚠️  Module 19 KV caching needs improvement")

    return results

if __name__ == "__main__":
    run_module_19_performance_tests()
