"""
Performance Tests for Module 20: Benchmarking

Tests whether the benchmarking suite actually provides meaningful performance
measurements and can drive optimization competitions.

Key questions:
- Does TinyMLPerf provide fair, reproducible benchmarks?
- Can the benchmarking system detect real performance differences?
- Do the competition metrics correlate with actual improvements?
- Is the benchmarking framework scientifically sound?
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
sys.path.append(str(Path(__file__).parent.parent.parent / 'modules' / '20_benchmarking'))

try:
    from benchmarking_dev import TinyMLPerf
    BENCHMARKING_AVAILABLE = True
except ImportError:
    print("❌ Module 20 benchmarking tools not available")
    BENCHMARKING_AVAILABLE = False

class Module20PerformanceTests:
    """Test suite for Module 20 benchmarking system."""

    def __init__(self):
        self.suite = PerformanceTestSuite()
        self.comparator = PerformanceComparator()
        self.workloads = WorkloadGenerator()

    def test_benchmark_suite_loading(self):
        """Test whether TinyMLPerf benchmark suite loads correctly."""
        if not BENCHMARKING_AVAILABLE:
            return "Benchmarking module not available"

        print("📋 Testing TinyMLPerf benchmark suite loading")

        try:
            # Initialize benchmark suite
            tinyperf = TinyMLPerf(profiler_warmup_runs=2, profiler_timing_runs=3)

            # Test available events
            events = tinyperf.get_available_events()
            expected_events = {'mlp_sprint', 'cnn_marathon', 'transformer_decathlon'}
            has_all_events = expected_events.issubset(set(events.keys()))

            # Test loading each benchmark
            load_results = {}
            for event_name in expected_events:
                try:
                    model, dataset = tinyperf.load_benchmark(event_name)

                    # Test model inference
                    inputs = dataset['inputs']
                    outputs = model.predict(inputs)

                    # Verify output shape
                    batch_size = inputs.shape[0]
                    output_shape_correct = outputs.shape[0] == batch_size

                    load_results[event_name] = {
                        'loaded': True,
                        'inference_works': True,
                        'output_shape_correct': output_shape_correct,
                        'input_shape': inputs.shape,
                        'output_shape': outputs.shape
                    }

                except Exception as e:
                    load_results[event_name] = {'loaded': False, 'error': str(e)}

            all_benchmarks_work = all(
                result.get('loaded', False) and
                result.get('inference_works', False) and
                result.get('output_shape_correct', False)
                for result in load_results.values()
            )

            loading_result = {
                'has_all_events': has_all_events,
                'load_results': load_results,
                'all_benchmarks_work': all_benchmarks_work,
                'events_available': list(events.keys()),
                'suite_loading_successful': has_all_events and all_benchmarks_work
            }

            if loading_result['suite_loading_successful']:
                print("✅ TinyMLPerf benchmark suite loaded successfully")
                print(f"   Events: {', '.join(events.keys())}")
            else:
                print("❌ TinyMLPerf benchmark suite loading issues")

            return loading_result

        except Exception as e:
            return f"Benchmark suite loading error: {e}"

    def test_benchmark_reproducibility(self):
        """Test whether benchmarks produce reproducible results."""
        if not BENCHMARKING_AVAILABLE:
            return "Benchmarking module not available"

        print("🔄 Testing benchmark reproducibility")

        try:
            tinyperf = TinyMLPerf(profiler_warmup_runs=2, profiler_timing_runs=5)
            model, dataset = tinyperf.load_benchmark('mlp_sprint')

            inputs = dataset['inputs']

            # Run inference multiple times
            results = []
            for run in range(5):
                outputs = model.predict(inputs)
                results.append(outputs.copy())

            # Check if all results are identical (they should be with deterministic model)
            all_identical = all(np.allclose(results[0], result, rtol=1e-10, atol=1e-10)
                               for result in results[1:])

            # Check output consistency across multiple instantiations
            tinyperf2 = TinyMLPerf(profiler_warmup_runs=2, profiler_timing_runs=5)
            model2, dataset2 = tinyperf2.load_benchmark('mlp_sprint')

            # Same inputs should produce same outputs (models initialized the same way)
            outputs1 = model.predict(inputs)
            outputs2 = model2.predict(inputs)

            cross_instance_identical = np.allclose(outputs1, outputs2, rtol=1e-10, atol=1e-10)

            reproducibility_result = {
                'multiple_runs_identical': all_identical,
                'cross_instance_identical': cross_instance_identical,
                'reproducible': all_identical and cross_instance_identical
            }

            if reproducibility_result['reproducible']:
                print("✅ Benchmarks produce reproducible results")
            else:
                print("❌ Benchmark reproducibility issues")
                if not all_identical:
                    print("   Multiple runs produce different results")
                if not cross_instance_identical:
                    print("   Different instances produce different results")

            return reproducibility_result

        except Exception as e:
            return f"Reproducibility test error: {e}"

    def test_performance_detection(self):
        """Test whether benchmarks can detect performance differences."""
        if not BENCHMARKING_AVAILABLE:
            return "Benchmarking module not available"

        print("🔍 Testing performance difference detection")

        try:
            tinyperf = TinyMLPerf(profiler_warmup_runs=2, profiler_timing_runs=10)
            model, dataset = tinyperf.load_benchmark('mlp_sprint')

            inputs = dataset['inputs']

            # Create fast and slow versions of the same operation
            def fast_inference():
                """Standard model inference"""
                return model.predict(inputs)

            def slow_inference():
                """Artificially slowed model inference"""
                result = model.predict(inputs)
                # Add artificial delay
                time.sleep(0.001)  # 1ms delay
                return result

            # Compare performance
            comparison = self.comparator.compare_implementations(
                slow_inference,
                fast_inference,
                baseline_name="slow_model",
                optimized_name="fast_model"
            )

            # Should detect the artificial slowdown
            detects_difference = comparison.speedup > 1.5  # Should see significant speedup
            results_identical = np.allclose(
                slow_inference(), fast_inference(), rtol=1e-10, atol=1e-10
            )

            detection_result = {
                'speedup_detected': comparison.speedup,
                'detects_performance_difference': detects_difference,
                'results_remain_identical': results_identical,
                'detection_working': detects_difference and results_identical
            }

            if detection_result['detection_working']:
                print(f"✅ Performance difference detected: {comparison.speedup:.1f}× speedup")
            else:
                print(f"❌ Failed to detect performance difference: {comparison.speedup:.1f}× speedup")

            return detection_result

        except Exception as e:
            return f"Performance detection test error: {e}"

    def test_cross_event_fairness(self):
        """Test whether different benchmark events provide fair comparisons."""
        if not BENCHMARKING_AVAILABLE:
            return "Benchmarking module not available"

        print("⚖️ Testing cross-event benchmark fairness")

        try:
            tinyperf = TinyMLPerf(profiler_warmup_runs=1, profiler_timing_runs=3)

            # Test all events
            events = ['mlp_sprint', 'cnn_marathon', 'transformer_decathlon']
            event_metrics = {}

            for event in events:
                try:
                    model, dataset = tinyperf.load_benchmark(event)
                    inputs = dataset['inputs']

                    # Time inference
                    timer = self.comparator.timer
                    timer.measurement_runs = 5

                    result = timer.measure_function(
                        lambda: model.predict(inputs),
                        name=f"{event}_inference"
                    )

                    event_metrics[event] = {
                        'mean_time_ms': result.mean_time_ms,
                        'std_time_ms': result.std_time_ms,
                        'batch_size': inputs.shape[0],
                        'input_size': np.prod(inputs.shape[1:]),
                        'time_per_sample_ms': result.mean_time_ms / inputs.shape[0],
                        'measurement_stable': result.std_time_ms / result.mean_time_ms < 0.2  # CV < 20%
                    }

                except Exception as e:
                    event_metrics[event] = {'error': str(e)}

            # Check measurement stability across events
            all_stable = all(
                metrics.get('measurement_stable', False)
                for metrics in event_metrics.values()
                if 'error' not in metrics
            )

            # Check reasonable timing ranges (different events should have different characteristics)
            timing_ranges_reasonable = len(set(
                int(metrics['mean_time_ms'] // 10) * 10  # Round to nearest 10ms
                for metrics in event_metrics.values()
                if 'error' not in metrics
            )) >= 2  # At least 2 different timing buckets

            fairness_result = {
                'event_metrics': event_metrics,
                'all_measurements_stable': all_stable,
                'timing_ranges_reasonable': timing_ranges_reasonable,
                'fairness_good': all_stable and timing_ranges_reasonable
            }

            if fairness_result['fairness_good']:
                print("✅ Cross-event benchmarks provide fair comparisons")
                for event, metrics in event_metrics.items():
                    if 'error' not in metrics:
                        print(f"   {event}: {metrics['mean_time_ms']:.1f}ms ± {metrics['std_time_ms']:.1f}ms")
            else:
                print("❌ Cross-event benchmark fairness issues")

            return fairness_result

        except Exception as e:
            return f"Cross-event fairness test error: {e}"

    def test_scaling_measurement(self):
        """Test whether benchmarks measure scaling behavior correctly."""
        if not BENCHMARKING_AVAILABLE:
            return "Benchmarking module not available"

        print("📈 Testing benchmark scaling measurement")

        try:
            tinyperf = TinyMLPerf(profiler_warmup_runs=1, profiler_timing_runs=3)
            model, dataset = tinyperf.load_benchmark('mlp_sprint')

            # Test different batch sizes
            base_inputs = dataset['inputs']
            batch_sizes = [25, 50, 100]  # Different batch sizes

            scaling_results = {}

            for batch_size in batch_sizes:
                if batch_size <= base_inputs.shape[0]:
                    test_inputs = base_inputs[:batch_size]
                else:
                    # Repeat inputs to get larger batch
                    repeats = (batch_size // base_inputs.shape[0]) + 1
                    repeated_inputs = np.tile(base_inputs, (repeats, 1))[:batch_size]
                    test_inputs = repeated_inputs

                # Time inference at this batch size
                timer = self.comparator.timer
                timer.measurement_runs = 5

                result = timer.measure_function(
                    lambda inputs=test_inputs: model.predict(inputs),
                    name=f"batch_{batch_size}"
                )

                scaling_results[batch_size] = {
                    'total_time_ms': result.mean_time_ms,
                    'time_per_sample_ms': result.mean_time_ms / batch_size,
                    'throughput_samples_per_sec': 1000 * batch_size / result.mean_time_ms
                }

            # Analyze scaling behavior
            times_per_sample = [scaling_results[bs]['time_per_sample_ms'] for bs in batch_sizes]
            throughputs = [scaling_results[bs]['throughput_samples_per_sec'] for bs in batch_sizes]

            # Throughput should generally increase with batch size (more efficient)
            throughput_scaling_reasonable = throughputs[-1] >= throughputs[0] * 0.8

            # Per-sample time should decrease or stay similar (batch efficiency)
            per_sample_scaling_reasonable = times_per_sample[-1] <= times_per_sample[0] * 1.2

            scaling_measurement_result = {
                'scaling_results': scaling_results,
                'times_per_sample_ms': times_per_sample,
                'throughputs_samples_per_sec': throughputs,
                'throughput_scaling_reasonable': throughput_scaling_reasonable,
                'per_sample_scaling_reasonable': per_sample_scaling_reasonable,
                'scaling_measurement_good': throughput_scaling_reasonable and per_sample_scaling_reasonable
            }

            if scaling_measurement_result['scaling_measurement_good']:
                print("✅ Benchmark scaling measurement working correctly")
                print(f"   Throughput: {throughputs[0]:.0f} → {throughputs[-1]:.0f} samples/sec")
            else:
                print("❌ Benchmark scaling measurement issues")

            return scaling_measurement_result

        except Exception as e:
            return f"Scaling measurement test error: {e}"

    def test_competition_scoring(self):
        """Test whether the competition scoring system works fairly."""
        if not BENCHMARKING_AVAILABLE:
            return "Benchmarking module not available"

        print("🏆 Testing competition scoring system")

        try:
            tinyperf = TinyMLPerf(profiler_warmup_runs=1, profiler_timing_runs=5)

            # Simulate different optimization submissions
            model, dataset = tinyperf.load_benchmark('mlp_sprint')
            inputs = dataset['inputs']

            # Create different "optimization" versions
            def baseline_submission():
                """Baseline unoptimized version"""
                return model.predict(inputs)

            def fast_submission():
                """Optimized version (simulated)"""
                result = model.predict(inputs)
                # Simulate faster execution (no added delay)
                return result

            def slow_submission():
                """Poorly optimized version"""
                result = model.predict(inputs)
                # Add delay to simulate poor optimization
                time.sleep(0.0005)  # 0.5ms delay
                return result

            # Score each submission
            timer = self.comparator.timer
            timer.measurement_runs = 5

            baseline_time = timer.measure_function(baseline_submission, name="baseline").mean_time_ms
            fast_time = timer.measure_function(fast_submission, name="fast").mean_time_ms
            slow_time = timer.measure_function(slow_submission, name="slow").mean_time_ms

            # Calculate relative scores (speedup relative to baseline)
            fast_score = baseline_time / fast_time
            slow_score = baseline_time / slow_time
            baseline_score = 1.0

            # Verify scoring makes sense
            scores_ordered_correctly = fast_score >= baseline_score >= slow_score
            meaningful_score_differences = (fast_score - slow_score) > 0.2

            scoring_result = {
                'baseline_score': baseline_score,
                'fast_score': fast_score,
                'slow_score': slow_score,
                'scores_ordered_correctly': scores_ordered_correctly,
                'meaningful_differences': meaningful_score_differences,
                'competition_scoring_working': scores_ordered_correctly and meaningful_score_differences
            }

            if scoring_result['competition_scoring_working']:
                print(f"✅ Competition scoring working: Fast {fast_score:.2f}, Base {baseline_score:.2f}, Slow {slow_score:.2f}")
            else:
                print(f"❌ Competition scoring issues: Fast {fast_score:.2f}, Base {baseline_score:.2f}, Slow {slow_score:.2f}")

            return scoring_result

        except Exception as e:
            return f"Competition scoring test error: {e}"

def run_module_20_performance_tests():
    """Run all performance tests for Module 20."""
    print("🧪 TESTING MODULE 20: BENCHMARKING SYSTEM")
    print("=" * 60)
    print("Verifying that the benchmarking suite provides fair, meaningful measurements")

    if not BENCHMARKING_AVAILABLE:
        print("❌ Cannot test Module 20 - benchmarking tools not available")
        return

    test_suite = Module20PerformanceTests()

    tests = {
        'suite_loading': test_suite.test_benchmark_suite_loading,
        'reproducibility': test_suite.test_benchmark_reproducibility,
        'performance_detection': test_suite.test_performance_detection,
        'cross_event_fairness': test_suite.test_cross_event_fairness,
        'scaling_measurement': test_suite.test_scaling_measurement,
        'competition_scoring': test_suite.test_competition_scoring
    }

    results = test_suite.suite.run_module_tests('module_20_benchmarking', tests)

    # Summary
    print(f"\n📊 MODULE 20 TEST SUMMARY")
    print("=" * 40)

    total_tests = len(tests)
    passed_tests = 0

    for test_name, result in results.items():
        if hasattr(result, 'speedup'):  # ComparisonResult
            passed = result.speedup > 1.1 and result.is_significant
            print(f"⚡ {test_name}: {result.speedup:.2f}× speedup {'✅' if passed else '❌'}")
        elif isinstance(result, dict):
            # Check specific success criteria for each test
            if 'suite_loading_successful' in result:
                passed = result['suite_loading_successful']
                print(f"📋 {test_name}: {'✅ PASS' if passed else '❌ FAIL'}")
            elif 'reproducible' in result:
                passed = result['reproducible']
                print(f"🔄 {test_name}: {'✅ PASS' if passed else '❌ FAIL'}")
            elif 'detection_working' in result:
                passed = result['detection_working']
                speedup = result.get('speedup_detected', 0)
                print(f"🔍 {test_name}: {speedup:.1f}× detected {'✅ PASS' if passed else '❌ FAIL'}")
            elif 'fairness_good' in result:
                passed = result['fairness_good']
                print(f"⚖️ {test_name}: {'✅ PASS' if passed else '❌ FAIL'}")
            elif 'scaling_measurement_good' in result:
                passed = result['scaling_measurement_good']
                print(f"📈 {test_name}: {'✅ PASS' if passed else '❌ FAIL'}")
            elif 'competition_scoring_working' in result:
                passed = result['competition_scoring_working']
                print(f"🏆 {test_name}: {'✅ PASS' if passed else '❌ FAIL'}")
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

    if success_rate >= 0.8:
        print("🎉 Module 20 benchmarking system is working well!")
        print("🏆 Ready for optimization competitions!")
    else:
        print("⚠️  Module 20 benchmarking system needs improvement")

    return results

if __name__ == "__main__":
    run_module_20_performance_tests()
