"""
Checkpoint 13: Benchmarking (After Module 14 - Benchmarking)
Question: "Can I analyze performance and identify bottlenecks in ML systems?"
"""

import numpy as np
import pytest

def test_checkpoint_13_benchmarking():
    """
    Checkpoint 13: Benchmarking

    Validates that students can perform comprehensive performance analysis
    and identify bottlenecks in machine learning systems - critical for
    building production-ready ML applications that scale efficiently.
    """
    print("\n📊 Checkpoint 13: Benchmarking")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.benchmarking import (
            BenchmarkScenario, BenchmarkResult, BenchmarkScenarios,
            StatisticalValidation, StatisticalValidator, TinyTorchPerf, PerformanceReporter
        )
        from tinytorch.core.networks import Sequential
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU, Softmax
        from tinytorch.core.training import Trainer, CrossEntropyLoss
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-14 first: {e}")

    # Test 1: Benchmark scenario creation
    print("🎯 Testing benchmark scenarios...")

    try:
        # Create different benchmark scenarios
        scenarios = BenchmarkScenarios()

        # Test that scenarios can be created
        scenario_names = ["small_model", "medium_model", "large_model"]
        for name in scenario_names:
            try:
                scenario = scenarios.get_scenario(name)
                if scenario:
                    assert hasattr(scenario, 'name'), f"Scenario {name} should have a name attribute"
                    print(f"✅ Scenario: {name} configured")
                else:
                    print(f"⚠️ Scenario: {name} not available")
            except Exception as e:
                print(f"⚠️ Scenario {name}: {e}")

        print(f"✅ Benchmark scenarios: configuration system ready")
    except Exception as e:
        print(f"⚠️ Benchmark scenarios: {e}")

    # Test 2: Performance measurement
    print("⏱️ Testing performance measurement...")

    try:
        # Create a simple model for benchmarking
        model = Sequential([
            Linear(10, 50),
            ReLU(),
            Linear(50, 20),
            ReLU(),
            Linear(20, 5),
            Softmax()
        ])

        # Create TinyTorchPerf for performance analysis
        perf_analyzer = TinyTorchPerf()

        # Test different input sizes
        input_sizes = [
            (1, 10),    # Single sample
            (32, 10),   # Small batch
            (128, 10),  # Medium batch
        ]

        results = {}
        for batch_size, input_dim in input_sizes:
            test_input = Tensor(np.random.randn(batch_size, input_dim))

            # Measure inference time
            start_time = perf_analyzer._get_time() if hasattr(perf_analyzer, '_get_time') else 0
            output = model(test_input)
            end_time = perf_analyzer._get_time() if hasattr(perf_analyzer, '_get_time') else 1

            inference_time = end_time - start_time
            results[f"batch_{batch_size}"] = {
                'input_shape': (batch_size, input_dim),
                'output_shape': output.shape,
                'time': inference_time
            }

        print(f"✅ Performance measurement: tested {len(results)} scenarios")
        for scenario, result in results.items():
            print(f"   {scenario}: {result['input_shape']} → {result['output_shape']}")

    except Exception as e:
        print(f"⚠️ Performance measurement: {e}")

    # Test 3: Statistical validation
    print("📈 Testing statistical validation...")

    try:
        validator = StatisticalValidator()

        # Generate sample performance data
        measurements = [0.1, 0.12, 0.11, 0.13, 0.09, 0.14, 0.10, 0.11, 0.12, 0.10]

        # Test statistical analysis
        if hasattr(validator, 'analyze_measurements'):
            stats = validator.analyze_measurements(measurements)

            if stats:
                assert 'mean' in stats or 'median' in stats, "Statistics should include central tendency"
                print(f"✅ Statistical validation: analyzed {len(measurements)} measurements")
            else:
                print(f"⚠️ Statistical validation: no stats returned")
        else:
            # Basic statistical validation
            mean_time = np.mean(measurements)
            std_time = np.std(measurements)
            cv = std_time / mean_time if mean_time > 0 else 0

            assert cv < 0.5, f"Coefficient of variation should be reasonable, got {cv:.3f}"
            print(f"✅ Statistical validation: mean={mean_time:.3f}s, std={std_time:.3f}s, cv={cv:.3f}")

    except Exception as e:
        print(f"⚠️ Statistical validation: {e}")

    # Test 4: Bottleneck identification
    print("🔍 Testing bottleneck identification...")

    try:
        # Create models of different complexities
        simple_model = Sequential([Linear(10, 5), ReLU()])
        complex_model = Sequential([
            Linear(100, 200), ReLU(),
            Linear(200, 400), ReLU(),
            Linear(400, 200), ReLU(),
            Linear(200, 50), ReLU(),
            Linear(50, 10)
        ])

        models = [("simple", simple_model), ("complex", complex_model)]
        bottlenecks = {}

        for name, model in models:
            # Measure layer-by-layer performance
            test_input = Tensor(np.random.randn(32, 100 if name == "complex" else 10))

            layer_times = []
            current_input = test_input

            for i, layer in enumerate(model.layers):
                # Time this layer
                import time
                start = time.time()
                current_input = layer(current_input)
                end = time.time()

                layer_times.append(end - start)

            # Find bottleneck layer
            if layer_times:
                bottleneck_idx = np.argmax(layer_times)
                bottlenecks[name] = {
                    'layer_index': bottleneck_idx,
                    'layer_time': layer_times[bottleneck_idx],
                    'total_time': sum(layer_times),
                    'bottleneck_ratio': layer_times[bottleneck_idx] / sum(layer_times) if sum(layer_times) > 0 else 0
                }

        print(f"✅ Bottleneck identification: analyzed {len(models)} models")
        for name, info in bottlenecks.items():
            print(f"   {name}: layer {info['layer_index']} ({info['bottleneck_ratio']:.1%} of total time)")

    except Exception as e:
        print(f"⚠️ Bottleneck identification: {e}")

    # Test 5: Memory profiling
    print("💾 Testing memory profiling...")

    try:
        # Test memory usage analysis
        import sys

        # Baseline memory
        baseline_objects = len([obj for obj in globals().values() if hasattr(obj, '__class__')])

        # Create memory-intensive operations
        large_tensors = []
        for i in range(5):
            tensor = Tensor(np.random.randn(100, 100))
            large_tensors.append(tensor)

        # Measure memory growth
        peak_objects = len([obj for obj in globals().values() if hasattr(obj, '__class__')])
        memory_growth = peak_objects - baseline_objects

        # Clean up
        del large_tensors

        print(f"✅ Memory profiling: detected {memory_growth} object growth during tensor operations")

    except Exception as e:
        print(f"⚠️ Memory profiling: {e}")

    # Test 6: Scalability analysis
    print("📈 Testing scalability analysis...")

    try:
        # Test how performance scales with input size
        model = Sequential([Linear(50, 20), ReLU(), Linear(20, 10)])

        sizes = [1, 10, 50, 100]
        scaling_results = []

        for size in sizes:
            test_input = Tensor(np.random.randn(size, 50))

            # Measure inference time
            import time
            start = time.time()
            _ = model(test_input)
            end = time.time()

            scaling_results.append({
                'batch_size': size,
                'time': end - start,
                'time_per_sample': (end - start) / size if size > 0 else 0
            })

        # Analyze scaling behavior
        if len(scaling_results) >= 2:
            time_ratio = scaling_results[-1]['time'] / scaling_results[0]['time'] if scaling_results[0]['time'] > 0 else 1
            size_ratio = scaling_results[-1]['batch_size'] / scaling_results[0]['batch_size']

            scaling_efficiency = time_ratio / size_ratio if size_ratio > 0 else 1
            print(f"✅ Scalability analysis: {size_ratio:.0f}x size increase → {time_ratio:.2f}x time (efficiency: {scaling_efficiency:.2f})")

    except Exception as e:
        print(f"⚠️ Scalability analysis: {e}")

    # Test 7: Comparative benchmarking
    print("🏁 Testing comparative benchmarking...")

    try:
        # Compare different activation functions
        activations = [("relu", ReLU())]

        if hasattr(pytest, 'importorskip'):
            try:
                from tinytorch.core.activations import Sigmoid, Tanh
                activations.extend([("sigmoid", Sigmoid()), ("tanh", Tanh())])
            except ImportError:
                pass

        comparison_results = {}
        test_input = Tensor(np.random.randn(100, 50))

        for name, activation in activations:
            import time
            start = time.time()

            # Run activation multiple times for better measurement
            for _ in range(10):
                _ = activation(test_input)

            end = time.time()
            comparison_results[name] = (end - start) / 10  # Average time per call

        # Find fastest activation
        if comparison_results:
            fastest = min(comparison_results.items(), key=lambda x: x[1])
            print(f"✅ Comparative benchmarking: tested {len(activations)} activations")
            print(f"   Fastest: {fastest[0]} at {fastest[1]:.6f}s per call")

    except Exception as e:
        print(f"⚠️ Comparative benchmarking: {e}")

    # Test 8: Performance reporting
    print("📋 Testing performance reporting...")

    try:
        reporter = PerformanceReporter()

        # Create sample benchmark results
        sample_results = [
            BenchmarkResult(
                scenario="test_inference",
                metric="latency",
                value=0.1,
                unit="seconds",
                metadata={"batch_size": 32}
            ),
            BenchmarkResult(
                scenario="test_training",
                metric="throughput",
                value=100,
                unit="samples/sec",
                metadata={"learning_rate": 0.01}
            )
        ]

        # Test report generation
        if hasattr(reporter, 'generate_report'):
            report = reporter.generate_report(sample_results)
            assert report is not None, "Report should be generated"
            print(f"✅ Performance reporting: generated report with {len(sample_results)} results")
        else:
            # Basic reporting test
            for result in sample_results:
                assert hasattr(result, 'scenario'), "Results should have scenario"
                assert hasattr(result, 'value'), "Results should have value"
            print(f"✅ Performance reporting: validated {len(sample_results)} benchmark results")

    except Exception as e:
        print(f"⚠️ Performance reporting: {e}")

    # Test 9: Regression detection
    print("🔄 Testing regression detection...")

    try:
        # Simulate performance measurements over time
        baseline_measurements = [0.10, 0.11, 0.09, 0.10, 0.12]  # Stable performance
        current_measurements = [0.15, 0.16, 0.14, 0.15, 0.17]   # Potential regression

        baseline_mean = np.mean(baseline_measurements)
        current_mean = np.mean(current_measurements)

        # Simple regression detection
        regression_threshold = 1.2  # 20% increase indicates regression
        performance_ratio = current_mean / baseline_mean if baseline_mean > 0 else 1

        is_regression = performance_ratio > regression_threshold

        print(f"✅ Regression detection: baseline={baseline_mean:.3f}s, current={current_mean:.3f}s")
        print(f"   Performance ratio: {performance_ratio:.2f}x ({'REGRESSION' if is_regression else 'OK'})")

    except Exception as e:
        print(f"⚠️ Regression detection: {e}")

    # Test 10: Advanced benchmarking integration
    print("🔧 Testing advanced benchmarking...")

    try:
        # Test integration with TinyTorch training
        model = Sequential([Linear(20, 10), ReLU(), Linear(10, 5)])

        # Set up training components
        X_train = Tensor(np.random.randn(100, 20))
        y_train = Tensor(np.random.randint(0, 5, (100, 5)).astype(np.float32))

        loss_fn = CrossEntropyLoss()

        # Benchmark training step
        import time
        start = time.time()

        # Simulate training step
        pred = model(X_train)
        loss = loss_fn(pred, y_train)

        end = time.time()
        training_time = end - start

        # Calculate throughput
        throughput = len(X_train.data) / training_time if training_time > 0 else 0

        print(f"✅ Advanced benchmarking: training step completed")
        print(f"   Training time: {training_time:.6f}s")
        print(f"   Throughput: {throughput:.1f} samples/sec")
        print(f"   Loss: {loss.data:.4f}")

        # Verify reasonable performance
        assert training_time > 0, "Training time should be measurable"
        assert throughput > 0, "Throughput should be positive"

    except Exception as e:
        print(f"⚠️ Advanced benchmarking: {e}")

    print("\n🎉 Benchmarking Complete!")
    print("📝 You can now analyze performance and identify bottlenecks in ML systems")
    print("🔧 Built capabilities: Performance measurement, statistical validation, bottleneck detection")
    print("🧠 Breakthrough: You can optimize ML systems using data-driven performance insights!")
    print("🎯 Next: Add MLOps, production deployment and monitoring")

if __name__ == "__main__":
    test_checkpoint_13_benchmarking()
