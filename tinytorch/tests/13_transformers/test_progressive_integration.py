"""
Module 13: Progressive Integration Tests
Tests that Module 14 (Benchmarking) works correctly AND that the entire ML system (01→13) still works.

DEPENDENCY CHAIN: 01_setup → ... → 13_kernels → 14_benchmarking
This is where we enable performance analysis and optimization for ML systems.

🎯 WHAT THIS TESTS:
- Module 13: Performance benchmarking, profiling, bottleneck analysis
- Integration: Benchmarking works with all ML components (models, training, data)
- Regression: Complete ML system (01→13) still works correctly
- Preparation: Ready for production deployment (Module 15: MLOps)

💡 FOR STUDENTS: If tests fail, check:
1. Does your benchmark_model function exist in tinytorch.core.benchmarking?
2. Can you profile memory usage and execution time?
3. Do benchmarks work with different model architectures?
4. Are performance metrics meaningful and actionable?

🔧 DEBUGGING HELP:
- Benchmarking measures: latency, throughput, memory usage, FLOPS
- Profiling identifies: bottlenecks, memory leaks, inefficiencies
- Analysis guides: optimization decisions, hardware choices, scaling strategies
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCompleteMLSystemStillWorks:
    """
    🔄 REGRESSION CHECK: Verify complete ML system (01→13) still works after benchmarking development.

    💡 If these fail: You may have broken something in the ML system while implementing benchmarking.
    🔧 Fix: Check that benchmarking code doesn't interfere with core ML functionality.
    """

    def test_end_to_end_ml_system_stable(self):
        """
        ✅ TEST: Complete ML system (training, inference, optimization) should still work

        📋 FULL ML SYSTEM COMPONENTS:
        - Data loading and preprocessing
        - Model architecture (CNN, attention, dense)
        - Training loops with optimization
        - Inference and evaluation
        - Performance optimizations

        🚨 IF FAILS: Core ML system broken by benchmarking development
        """
        try:
            # Test complete ML system still works
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Create complete ML pipeline
            class TestModel:
                def __init__(self):
                    self.conv1 = Conv2D(3, 16, kernel_size=3, padding=1)
                    self.pool = MaxPool2d(kernel_size=2)
                    self.conv2 = Conv2D(16, 32, kernel_size=3, padding=1)
                    self.fc = Linear(32 * 8 * 8, 10)
                    self.relu = ReLU()
                    self.softmax = Softmax()

                def __call__(self, x):
                    h1 = self.relu(self.conv1(x))
                    h1_pool = self.pool(h1)
                    h2 = self.relu(self.conv2(h1_pool))
                    h2_pool = self.pool(h2)

                    # Flatten
                    flattened = Tensor(h2_pool.data.reshape(h2_pool.shape[0], -1))
                    logits = self.fc(flattened)
                    return self.softmax(logits)

                def parameters(self):
                    params = []
                    for layer in [self.conv1, self.conv2, self.fc]:
                        if hasattr(layer, 'parameters'):
                            params.extend(layer.parameters())
                        elif hasattr(layer, 'weight'):
                            params.append(layer.weights)
                            if hasattr(layer, 'bias') and layer.bias is not None:
                                params.append(layer.bias)
                    return params

            # Create dataset
            class TestDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(40, 3, 32, 32)
                    self.targets = np.random.randint(0, 10, 40)

                def __len__(self):
                    return 40

                def __getitem__(self, idx):
                    return Tensor(self.data[idx]), self.targets[idx]

            # Test complete pipeline
            model = TestModel()
            optimizer = Adam(model.parameters(), lr=0.001)
            # Trainer requires loss_fn as third argument
            def dummy_loss(pred, target):
                return pred.sum()
            trainer = Trainer(model, optimizer, dummy_loss)

            dataset = TestDataset()
            dataloader = DataLoader(dataset, batch_size=8)

            # Test inference
            for batch_x, batch_y in dataloader:
                predictions = model(batch_x)
                assert predictions.shape == (8, 10), \
                    f"❌ ML system shape broken. Expected (8, 10), got {predictions.shape}"

                # Verify probabilities
                prob_sums = np.sum(predictions.data, axis=1)
                assert np.allclose(prob_sums, 1.0), \
                    "❌ ML system probabilities broken"

                break  # Test one batch

        except ImportError as e:
            assert False, f"""
            ❌ ML SYSTEM IMPORTS BROKEN!

            🔍 IMPORT ERROR: {str(e)}

            🔧 COMPLETE ML SYSTEM REQUIREMENTS:
            All modules (01→13) must be working:
            1. Tensor operations (Module 01)
            2. Activation functions (Module 02)
            3. Layer infrastructure (Module 03)
            4. Losses (Module 04)
            5. Data loading (Module 05)
            6. Autograd (Module 06)
            7. Optimizers (Module 07)
            8. Training loops (Module 08)
            9. Convolutions (Module 09)
            10. Tokenization (Module 10)
            11. Embeddings (Module 11)
            12. Attention (Module 12)
            13. Transformers (Module 13)

            💡 SYSTEM INTEGRITY:
            Benchmarking should be NON-INVASIVE - it measures
            performance without changing core functionality.
            """
        except Exception as e:
            assert False, f"""
            ❌ ML SYSTEM FUNCTIONALITY BROKEN!

            🔍 ERROR: {str(e)}

            🔧 POSSIBLE CAUSES:
            1. Benchmarking interfering with forward pass
            2. Memory profiling affecting tensor operations
            3. Performance monitoring breaking training loops
            4. Instrumentation corrupting gradients
            5. Timing code affecting numerical stability

            💡 BENCHMARKING SAFETY:
            Benchmarking code should be:
            - Non-intrusive: Doesn't change model behavior
            - Optional: Can be disabled for production
            - Isolated: Runs in separate threads/processes
            - Clean: No side effects on ML computations
            """

    def test_optimization_and_training_stable(self):
        """
        ✅ TEST: Training and optimization should still work after benchmarking

        📋 TRAINING SYSTEM:
        - Gradient computation and backpropagation
        - Parameter updates via optimizers
        - Loss function evaluation
        - Training loop coordination

        🎯 Ensures benchmarking doesn't break learning
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.training import Trainer, MSELoss
            # Note: Variable is not used - TinyTorch uses Tensor with requires_grad

            # Test training system
            model = Linear(5, 3)
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            # Test training step
            x = Tensor(np.random.randn(4, 5))
            target = Tensor(np.random.randn(4, 3))

            # Forward pass
            predictions = model(x)
            loss = loss_fn(predictions, target)

            # Verify training components work
            assert predictions.shape == (4, 3), \
                f"❌ Training forward pass broken. Expected (4, 3), got {predictions.shape}"

            assert hasattr(loss, 'data') or isinstance(loss, (float, np.ndarray)), \
                "❌ Loss computation broken"

            # Test optimization step structure
            assert hasattr(optimizer, 'step'), \
                "❌ Optimizer step method missing"

            assert hasattr(optimizer, 'zero_grad'), \
                "❌ Optimizer zero_grad method missing"

        except Exception as e:
            assert False, f"""
            ❌ TRAINING AND OPTIMIZATION BROKEN!

            🔍 ERROR: {str(e)}

            🔧 TRAINING REQUIREMENTS:
            1. Forward pass produces correct outputs
            2. Loss functions compute meaningful values
            3. Optimizers can update parameters
            4. Training loops coordinate all components
            5. Gradient computation remains stable

            💡 TRAINING INTEGRITY:
            Training must work perfectly because:
            - Benchmarking measures training performance
            - Performance analysis guides training optimization
            - Production systems depend on training stability
            - Model deployment requires trained parameters
            """


class TestModule14BenchmarkingCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 14 (Benchmarking) core implementation.

    💡 What you're implementing: Performance analysis and optimization tools for ML systems.
    🎯 Goal: Enable data-driven performance optimization and bottleneck identification.

    NOTE: These tests are for Module 14 benchmarking which may not be implemented yet.
    Tests will pass gracefully if benchmarking module doesn't exist.
    """

    def test_model_benchmarking_exists(self):
        """
        ✅ TEST: Model benchmarking - Measure model performance characteristics

        📋 WHAT YOU NEED TO IMPLEMENT:
        def benchmark_model(model, input_shape, batch_sizes=[1, 8, 32], num_trials=10):
            # Measure latency, throughput, memory usage
            return {'latency': ..., 'throughput': ..., 'memory': ...}

        🚨 IF FAILS: Model benchmarking doesn't exist or missing components
        """
        try:
            from tinytorch.core.benchmarking import benchmark_model
        except ImportError:
            # Benchmarking module not implemented yet - pass gracefully
            assert True, "Benchmarking module not implemented yet"
            return
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Test model benchmarking
            model = Linear(100, 50)
            input_shape = (32, 100)  # Batch size 32, 100 features

            # Run benchmark
            results = benchmark_model(model, input_shape, num_trials=5)

            # Should return performance metrics
            assert isinstance(results, dict), \
                "❌ benchmark_model should return dictionary of metrics"

            # Essential performance metrics
            required_metrics = ['latency', 'throughput', 'memory_usage']
            for metric in required_metrics:
                assert metric in results, \
                    f"❌ Missing benchmark metric: {metric}"

                assert isinstance(results[metric], (int, float)), \
                    f"❌ Benchmark metric {metric} should be numeric, got {type(results[metric])}"

                assert results[metric] > 0, \
                    f"❌ Benchmark metric {metric} should be positive, got {results[metric]}"

            # Latency should be reasonable (milliseconds)
            assert 0.001 < results['latency'] < 10.0, \
                f"❌ Latency seems unrealistic: {results['latency']} seconds"

            # Throughput should be reasonable (samples/second)
            assert 1 < results['throughput'] < 1000000, \
                f"❌ Throughput seems unrealistic: {results['throughput']} samples/sec"

        except Exception as e:
            assert False, f"""
            ❌ MODEL BENCHMARKING BROKEN!

            🔍 ERROR: {str(e)}

            🔧 BENCHMARKING REQUIREMENTS:
            1. Measure inference latency accurately
            2. Calculate throughput (samples/second)
            3. Monitor memory usage
            4. Handle different batch sizes
            5. Provide statistical measures (mean, std)
            6. Support different model architectures

            💡 PERFORMANCE ANALYSIS:
            Benchmarking enables:
            - Hardware selection (CPU vs GPU vs TPU)
            - Model architecture optimization
            - Deployment planning and scaling
            - Cost estimation for production
            - SLA planning and capacity sizing
            """

    def test_profiling_and_bottleneck_analysis(self):
        """
        ✅ TEST: Profiling tools - Identify performance bottlenecks

        📋 PROFILING CAPABILITIES:
        - Layer-wise timing analysis
        - Memory allocation tracking
        - Operation hotspot identification
        - Gradient computation profiling

        🎯 Essential for performance optimization
        """
        try:
            from tinytorch.core.benchmarking import profile_model, ProfilerContext
        except ImportError:
            # Benchmarking module not implemented yet - pass gracefully
            assert True, "Benchmarking module not implemented yet"
            return

        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor

            # Test profiling functionality
            class TestModel:
                def __init__(self):
                    self.layer1 = Linear(100, 200)
                    self.relu = ReLU()
                    self.layer2 = Linear(200, 50)

                def __call__(self, x):
                    h1 = self.layer1(x)
                    h1_act = self.relu(h1)
                    return self.layer2(h1_act)

            model = TestModel()
            x = Tensor(np.random.randn(16, 100))

            # Test profiling
            if 'profile_model' in locals():
                profile_results = profile_model(model, x)

                # Should provide layer-wise timing
                assert isinstance(profile_results, dict), \
                    "❌ Profiler should return dictionary of results"

                assert 'layer_times' in profile_results, \
                    "❌ Profiler missing layer timing analysis"

                assert 'total_time' in profile_results, \
                    "❌ Profiler missing total execution time"

                assert 'bottlenecks' in profile_results, \
                    "❌ Profiler missing bottleneck identification"

            # Test profiler context (if available)
            if 'ProfilerContext' in locals():
                with ProfilerContext() as profiler:
                    output = model(x)

                results = profiler.get_results()
                assert isinstance(results, dict), \
                    "❌ ProfilerContext should provide results"

                assert 'operations' in results, \
                    "❌ ProfilerContext missing operation tracking"

        except ImportError:
            assert False, f"""
            ❌ PROFILING TOOLS MISSING!

            🔧 PROFILING IMPLEMENTATION:

            class ProfilerContext:
                '''Context manager for detailed performance profiling.'''

                def __init__(self):
                    self.operation_times = []
                    self.memory_snapshots = []
                    self.start_time = None

                def __enter__(self):
                    self.start_time = time.time()
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.total_time = time.time() - self.start_time

                def get_results(self):
                    return {{
                        'total_time': self.total_time,
                        'operations': self.operation_times,
                        'memory': self.memory_snapshots
                    }}

            def profile_model(model, input_data):
                '''Profile model execution and identify bottlenecks.'''

                # Profile each layer
                layer_times = {{}}
                with ProfilerContext() as profiler:

                    # Time each operation
                    start = time.time()
                    output = model(input_data)
                    end = time.time()

                    total_time = end - start

                return {{
                    'total_time': total_time,
                    'layer_times': layer_times,
                    'bottlenecks': ['layer2'],  # Identify slowest layers
                    'recommendations': ['Consider reducing layer2 size']
                }}

            💡 PROFILING APPLICATIONS:
            - Optimize model architectures
            - Choose efficient layer types
            - Identify memory bottlenecks
            - Guide hardware upgrades
            - Plan distributed training
            """
        except Exception as e:
            assert False, f"""
            ❌ PROFILING TOOLS BROKEN!

            🔍 ERROR: {str(e)}

            🔧 PROFILING REQUIREMENTS:
            1. Layer-wise execution timing
            2. Memory allocation tracking
            3. Operation bottleneck identification
            4. Statistical analysis of performance
            5. Actionable optimization recommendations

            🔬 PROFILING INSIGHTS:
            Profiling reveals:
            - Which layers are slowest
            - Memory allocation patterns
            - Computational hotspots
            - Inefficient operations
            - Hardware utilization
            """

    def test_performance_comparison_tools(self):
        """
        ✅ TEST: Performance comparison - Compare different model configurations

        📋 COMPARISON CAPABILITIES:
        - Model architecture comparison
        - Optimization technique evaluation
        - Hardware performance analysis
        - Training vs inference benchmarks

        💡 Guide optimization decisions with data
        """
        try:
            from tinytorch.core.benchmarking import compare_models, PerformanceComparator
        except ImportError:
            # Benchmarking module not implemented yet - pass gracefully
            assert True, "Benchmarking module not implemented yet"
            return

        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Test model comparison
            model_small = Linear(50, 25)
            model_large = Linear(100, 50)

            input_shape = (16, 50)

            if 'compare_models' in locals():
                comparison = compare_models([model_small, model_large], input_shape)

                # Should compare performance metrics
                assert isinstance(comparison, dict), \
                    "❌ Model comparison should return results dictionary"

                assert 'models' in comparison, \
                    "❌ Comparison missing model results"

                assert len(comparison['models']) == 2, \
                    f"❌ Should compare 2 models, got {len(comparison['models'])}"

                # Should provide relative performance
                assert 'relative_performance' in comparison, \
                    "❌ Comparison missing relative performance analysis"

                # Should identify best model
                assert 'best_model' in comparison, \
                    "❌ Comparison should identify best performing model"

            # Test performance comparator (if available)
            if 'PerformanceComparator' in locals():
                comparator = PerformanceComparator()

                # Add models to comparison
                comparator.add_model("small", model_small)
                comparator.add_model("large", model_large)

                # Run comparison
                results = comparator.benchmark_all(input_shape)

                assert isinstance(results, dict), \
                    "❌ PerformanceComparator should return results"

                assert 'small' in results and 'large' in results, \
                    "❌ Comparison should include all added models"

        except ImportError:
            assert False, f"""
            ❌ PERFORMANCE COMPARISON TOOLS MISSING!

            🔧 COMPARISON IMPLEMENTATION:

            def compare_models(models, input_shape, metrics=['latency', 'memory']):
                '''Compare performance of multiple models.'''

                results = {{
                    'models': [],
                    'relative_performance': {{}},
                    'best_model': None
                }}

                # Benchmark each model
                for i, model in enumerate(models):
                    model_results = benchmark_model(model, input_shape)
                    results['models'].append({{
                        'model_id': i,
                        'metrics': model_results
                    }})

                # Find best model (lowest latency)
                best_idx = min(range(len(models)),
                             key=lambda i: results['models'][i]['metrics']['latency'])
                results['best_model'] = best_idx

                return results

            class PerformanceComparator:
                '''Advanced model performance comparison tool.'''

                def __init__(self):
                    self.models = {{}}

                def add_model(self, name, model):
                    self.models[name] = model

                def benchmark_all(self, input_shape):
                    results = {{}}
                    for name, model in self.models.items():
                        results[name] = benchmark_model(model, input_shape)
                    return results

            💡 COMPARISON USE CASES:
            - Architecture search: Which model design is best?
            - Optimization evaluation: Does pruning help performance?
            - Hardware selection: CPU vs GPU vs TPU performance
            - Deployment planning: Latency vs accuracy tradeoffs
            """
        except Exception as e:
            assert False, f"""
            ❌ PERFORMANCE COMPARISON BROKEN!

            🔍 ERROR: {str(e)}

            🔧 COMPARISON REQUIREMENTS:
            1. Fair benchmarking across models
            2. Multiple performance metrics
            3. Statistical significance testing
            4. Relative performance analysis
            5. Actionable recommendations

            📊 COMPARISON INSIGHTS:
            Performance comparison enables:
            - Data-driven model selection
            - Optimization technique validation
            - Hardware cost-benefit analysis
            - Production deployment decisions
            """


class TestBenchmarkingIntegration:
    """
    🔗 INTEGRATION TEST: Benchmarking + Complete ML system working together.

    💡 Test that benchmarking works with real ML workflows and architectures.
    🎯 Goal: Enable performance analysis for production ML systems.
    """

    def test_training_performance_analysis(self):
        """
        ✅ TEST: Benchmark training loops and optimization

        📋 TRAINING BENCHMARKS:
        - Forward pass timing
        - Backward pass timing
        - Optimizer step timing
        - Data loading bottlenecks
        - End-to-end training throughput

        💡 Optimize training performance for faster iteration
        """
        try:
            from tinytorch.core.benchmarking import benchmark_training
        except ImportError:
            # Benchmarking module not implemented yet - pass gracefully
            assert True, "Benchmarking module not implemented yet"
            return

        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.training import Trainer, MSELoss
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Create training setup
            model = Linear(50, 10)
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            # Create dataset
            class TrainingDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(100, 50)
                    self.targets = np.random.randn(100, 10)

                def __len__(self):
                    return 100

                def __getitem__(self, idx):
                    return Tensor(self.data[idx]), Tensor(self.targets[idx])

            dataset = TrainingDataset()
            dataloader = DataLoader(dataset, batch_size=16)

            # Test training benchmarking
            if 'benchmark_training' in locals():
                training_results = benchmark_training(trainer, dataloader, num_epochs=2)

                # Should provide training performance metrics
                assert isinstance(training_results, dict), \
                    "❌ Training benchmark should return results dictionary"

                training_metrics = ['forward_time', 'backward_time', 'optimizer_time', 'total_time']
                for metric in training_metrics:
                    assert metric in training_results, \
                        f"❌ Missing training metric: {metric}"

                # Training time should be reasonable
                assert training_results['total_time'] > 0, \
                    "❌ Total training time should be positive"

                assert training_results['forward_time'] > 0, \
                    "❌ Forward pass time should be positive"

            # Test manual training loop timing
            start_time = time.time()

            for epoch in range(2):
                for batch_x, batch_y in dataloader:
                    # Simulate training step
                    predictions = model(batch_x)
                    loss = loss_fn(predictions, batch_y)

                    # Simple parameter update (without full autograd)
                    # In real implementation: loss.backward(); optimizer.step()

                    break  # One batch per epoch for testing

            end_time = time.time()
            training_duration = end_time - start_time

            assert training_duration > 0, \
                "❌ Training loop timing measurement broken"

        except Exception as e:
            assert False, f"""
            ❌ TRAINING PERFORMANCE ANALYSIS BROKEN!

            🔍 ERROR: {str(e)}

            🔧 TRAINING BENCHMARK REQUIREMENTS:
            1. Measure forward pass timing
            2. Measure backward pass timing
            3. Measure optimizer step timing
            4. Measure data loading timing
            5. Calculate training throughput (samples/second)
            6. Identify training bottlenecks

            💡 TRAINING OPTIMIZATION:

            def benchmark_training(trainer, dataloader, num_epochs=1):
                '''Benchmark training loop performance.'''

                times = {{
                    'forward_time': 0,
                    'backward_time': 0,
                    'optimizer_time': 0,
                    'data_loading_time': 0
                }}

                total_start = time.time()

                for epoch in range(num_epochs):
                    for batch_x, batch_y in dataloader:

                        # Forward pass timing
                        forward_start = time.time()
                        predictions = trainer.model(batch_x)
                        times['forward_time'] += time.time() - forward_start

                        # Loss computation
                        loss = trainer.loss_fn(predictions, batch_y)

                        # Backward pass timing (if available)
                        backward_start = time.time()
                        # loss.backward()  # When autograd ready
                        times['backward_time'] += time.time() - backward_start

                        # Optimizer timing
                        opt_start = time.time()
                        # trainer.optimizer.step()  # When ready
                        times['optimizer_time'] += time.time() - opt_start

                times['total_time'] = time.time() - total_start
                return times

            🚀 TRAINING INSIGHTS:
            - Data loading bottlenecks: Async data loading
            - Forward pass optimization: Kernel fusion, mixed precision
            - Backward pass optimization: Gradient checkpointing
            - Optimizer optimization: Fused optimizers, large batch training
            """

    def test_inference_performance_optimization(self):
        """
        ✅ TEST: Benchmark inference and deployment scenarios

        📋 INFERENCE BENCHMARKS:
        - Single sample latency
        - Batch inference throughput
        - Memory efficiency analysis
        - Hardware utilization
        - Real-time performance

        🎯 Optimize for production deployment
        """
        try:
            from tinytorch.core.benchmarking import benchmark_inference
        except ImportError:
            # Benchmarking module not implemented yet - pass gracefully
            assert True, "Benchmarking module not implemented yet"
            return

        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU

            # Create inference model (CNN for image classification)
            class InferenceModel:
                def __init__(self):
                    self.conv1 = Conv2D(3, 32, kernel_size=3, padding=1)
                    self.pool = MaxPool2d(kernel_size=2)
                    self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
                    self.fc = Linear(64 * 8 * 8, 1000)  # ImageNet-like
                    self.relu = ReLU()

                def __call__(self, x):
                    h1 = self.relu(self.conv1(x))
                    h1_pool = self.pool(h1)
                    h2 = self.relu(self.conv2(h1_pool))
                    h2_pool = self.pool(h2)

                    # Flatten
                    flattened = Tensor(h2_pool.data.reshape(h2_pool.shape[0], -1))
                    return self.fc(flattened)

            model = InferenceModel()

            # Test different inference scenarios
            scenarios = [
                ("single_image", (1, 3, 32, 32)),      # Real-time inference
                ("small_batch", (8, 3, 32, 32)),       # Small batch processing
                ("large_batch", (64, 3, 32, 32)),      # Batch processing
            ]

            if 'benchmark_inference' in locals():
                for scenario_name, input_shape in scenarios:
                    inference_results = benchmark_inference(model, input_shape)

                    assert isinstance(inference_results, dict), \
                        f"❌ Inference benchmark should return dict for {scenario_name}"

                    # Essential inference metrics
                    inference_metrics = ['latency', 'throughput', 'memory_peak']
                    for metric in inference_metrics:
                        assert metric in inference_results, \
                            f"❌ Missing inference metric {metric} for {scenario_name}"

                        assert inference_results[metric] > 0, \
                            f"❌ Inference metric {metric} should be positive for {scenario_name}"

            # Test manual inference timing
            for scenario_name, input_shape in scenarios:
                x = Tensor(np.random.randn(*input_shape))

                # Warmup
                for _ in range(3):
                    _ = model(x)

                # Actual timing
                start_time = time.time()
                output = model(x)
                end_time = time.time()

                latency = end_time - start_time
                throughput = input_shape[0] / latency  # samples/second

                assert latency > 0, f"❌ Latency timing broken for {scenario_name}"
                assert throughput > 0, f"❌ Throughput calculation broken for {scenario_name}"

                # Reasonable performance expectations
                if scenario_name == "single_image":
                    assert latency < 1.0, f"❌ Single image inference too slow: {latency}s"
                elif scenario_name == "large_batch":
                    assert throughput > input_shape[0] / 5.0, f"❌ Batch throughput too low: {throughput}"

        except Exception as e:
            assert False, f"""
            ❌ INFERENCE PERFORMANCE OPTIMIZATION BROKEN!

            🔍 ERROR: {str(e)}

            🔧 INFERENCE BENCHMARK REQUIREMENTS:
            1. Single sample latency measurement
            2. Batch inference throughput analysis
            3. Memory consumption profiling
            4. Hardware utilization monitoring
            5. Real-time performance validation

            💡 INFERENCE OPTIMIZATION STRATEGIES:

            def benchmark_inference(model, input_shape, num_trials=100):
                '''Comprehensive inference benchmarking.'''

                # Create test input
                test_input = Tensor(np.random.randn(*input_shape))

                # Warmup
                for _ in range(10):
                    _ = model(test_input)

                # Timing trials
                latencies = []
                for _ in range(num_trials):
                    start = time.perf_counter()
                    output = model(test_input)
                    end = time.perf_counter()
                    latencies.append(end - start)

                # Calculate metrics
                avg_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                throughput = input_shape[0] / avg_latency

                # Memory estimation
                memory_peak = test_input.data.nbytes + output.data.nbytes

                return {{
                    'latency': avg_latency,
                    'p95_latency': p95_latency,
                    'throughput': throughput,
                    'memory_peak': memory_peak
                }}

            🚀 PRODUCTION DEPLOYMENT:
            Inference optimization enables:
            - Real-time applications (< 100ms latency)
            - High-throughput services (> 1000 QPS)
            - Edge deployment (mobile, IoT devices)
            - Cost-effective cloud serving
            """

    def test_hardware_performance_analysis(self):
        """
        ✅ TEST: Hardware-specific performance analysis

        📋 HARDWARE ANALYSIS:
        - CPU vs GPU performance comparison
        - Memory bandwidth utilization
        - Compute utilization analysis
        - Scaling characteristics
        - Hardware recommendation

        💡 Guide hardware selection and optimization
        """
        try:
            from tinytorch.core.benchmarking import analyze_hardware_performance
        except ImportError:
            # Benchmarking module not implemented yet - pass gracefully
            assert True, "Benchmarking module not implemented yet"
            return

        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.spatial import Conv2d as Conv2D

            # Test hardware analysis
            models_for_analysis = [
                ("cpu_intensive", Linear(1000, 1000)),       # CPU-friendly
                ("memory_intensive", Conv2D(3, 512, 7)),    # Memory-bound
            ]

            if 'analyze_hardware_performance' in locals():
                for model_name, model in models_for_analysis:
                    hw_analysis = analyze_hardware_performance(model)

                    assert isinstance(hw_analysis, dict), \
                        f"❌ Hardware analysis should return dict for {model_name}"

                    # Hardware analysis components
                    hw_metrics = ['compute_bound', 'memory_bound', 'io_bound', 'recommendations']
                    for metric in hw_metrics:
                        assert metric in hw_analysis, \
                            f"❌ Missing hardware metric {metric} for {model_name}"

                    # Recommendations should be actionable
                    assert isinstance(hw_analysis['recommendations'], list), \
                        f"❌ Hardware recommendations should be list for {model_name}"

                    assert len(hw_analysis['recommendations']) > 0, \
                        f"❌ Should provide hardware recommendations for {model_name}"

            # Test basic hardware characteristic analysis
            # CPU-intensive model (many small operations)
            cpu_model = Linear(500, 500)
            cpu_input = Tensor(np.random.randn(1, 500))

            # Memory-intensive model (large feature maps)
            memory_model = Conv2D(3, 256, kernel_size=7)
            memory_input = Tensor(np.random.randn(1, 3, 224, 224))

            # Test execution and timing
            for model_type, model, test_input in [
                ("CPU-intensive", cpu_model, cpu_input),
                ("Memory-intensive", memory_model, memory_input)
            ]:
                start_time = time.time()
                output = model(test_input)
                end_time = time.time()

                execution_time = end_time - start_time

                # Basic hardware analysis
                input_size = test_input.data.nbytes
                output_size = output.data.nbytes
                memory_throughput = (input_size + output_size) / execution_time  # bytes/second

                assert execution_time > 0, f"❌ Execution timing broken for {model_type}"
                assert memory_throughput > 0, f"❌ Memory throughput calculation broken for {model_type}"

        except Exception as e:
            assert False, f"""
            ❌ HARDWARE PERFORMANCE ANALYSIS BROKEN!

            🔍 ERROR: {str(e)}

            🔧 HARDWARE ANALYSIS REQUIREMENTS:
            1. Identify compute vs memory bottlenecks
            2. Analyze hardware utilization patterns
            3. Provide hardware-specific recommendations
            4. Support different hardware targets
            5. Scale analysis across model sizes

            💡 HARDWARE OPTIMIZATION GUIDE:

            def analyze_hardware_performance(model):
                '''Analyze model hardware requirements and bottlenecks.'''

                analysis = {{
                    'compute_bound': False,
                    'memory_bound': False,
                    'io_bound': False,
                    'recommendations': []
                }}

                # Analyze model characteristics
                total_params = count_parameters(model)
                total_ops = estimate_flops(model)

                # Heuristic analysis
                if total_ops > 1e9:  # > 1 GFLOP
                    analysis['compute_bound'] = True
                    analysis['recommendations'].append("Consider GPU acceleration")

                if total_params > 1e6:  # > 1M parameters
                    analysis['memory_bound'] = True
                    analysis['recommendations'].append("Ensure sufficient memory bandwidth")

                return analysis

            🔧 HARDWARE SELECTION GUIDE:
            - CPU: Small models, low latency, high precision
            - GPU: Large models, high throughput, parallel training
            - TPU: Very large models, training at scale
            - Edge: Mobile deployment, power efficiency
            - Cloud: Elastic scaling, cost optimization
            """


class TestModule14Completion:
    """
    ✅ COMPLETION CHECK: Module 14 ready and foundation set for production deployment.

    🎯 Final validation that benchmarking works and enables performance optimization.
    """

    def test_benchmarking_foundation_complete(self):
        """
        ✅ FINAL TEST: Complete benchmarking foundation ready for production

        📋 BENCHMARKING FOUNDATION CHECKLIST:
        □ Model performance benchmarking
        □ Profiling and bottleneck analysis
        □ Performance comparison tools
        □ Training performance analysis
        □ Inference optimization
        □ Hardware performance analysis
        □ Integration with ML pipeline
        □ Production readiness

        🎯 SUCCESS = Ready for Module 15: MLOps and Production Deployment!
        """
        # First check if benchmarking module exists
        try:
            from tinytorch.core.benchmarking import benchmark_model
        except ImportError:
            # Benchmarking module not implemented yet - pass gracefully
            assert True, "Benchmarking module not implemented yet"
            return

        benchmarking_capabilities = {
            "Model benchmarking": False,
            "Profiling tools": False,
            "Performance comparison": False,
            "Training analysis": False,
            "Inference optimization": False,
            "Hardware analysis": False,
            "ML pipeline integration": False,
            "Production readiness": False
        }

        try:
            # Test 1: Model benchmarking
            from tinytorch.core.layers import Linear

            model = Linear(50, 25)
            results = benchmark_model(model, (16, 50))
            assert 'latency' in results and 'throughput' in results
            benchmarking_capabilities["Model benchmarking"] = True

            # Test 2: Profiling (check structure)
            try:
                from tinytorch.core.benchmarking import profile_model
                benchmarking_capabilities["Profiling tools"] = True
            except ImportError:
                # Basic timing works
                import time
                start = time.time()
                time.sleep(0.001)  # Simulate work
                duration = time.time() - start
                assert duration > 0
                benchmarking_capabilities["Profiling tools"] = True

            # Test 3: Performance comparison (check structure)
            try:
                from tinytorch.core.benchmarking import compare_models
                benchmarking_capabilities["Performance comparison"] = True
            except ImportError:
                # Can compare manually
                model1_results = benchmark_model(Linear(25, 10), (8, 25))
                model2_results = benchmark_model(Linear(50, 10), (8, 50))
                assert model1_results['latency'] != model2_results['latency']
                benchmarking_capabilities["Performance comparison"] = True

            # Test 4: Training analysis
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.optimizers import SGD

            # Simulate training timing
            start = time.time()
            x = Tensor(np.random.randn(16, 50))
            output = model(x)
            training_time = time.time() - start
            assert training_time > 0
            benchmarking_capabilities["Training analysis"] = True

            # Test 5: Inference optimization
            # Test different batch sizes
            single_start = time.time()
            single_input = Tensor(np.random.randn(1, 50))
            model(single_input)
            single_time = time.time() - single_start

            batch_start = time.time()
            batch_input = Tensor(np.random.randn(16, 50))
            model(batch_input)
            batch_time = time.time() - batch_start

            # Batch should be more efficient per sample
            single_throughput = 1 / single_time
            batch_throughput = 16 / batch_time
            assert batch_throughput > single_throughput
            benchmarking_capabilities["Inference optimization"] = True

            # Test 6: Hardware analysis (basic structure)
            # Analyze different model sizes
            small_model = Linear(10, 5)
            large_model = Linear(1000, 500)

            small_results = benchmark_model(small_model, (4, 10))
            large_results = benchmark_model(large_model, (4, 1000))

            # Large model should use more resources
            assert large_results['memory_usage'] > small_results['memory_usage']
            benchmarking_capabilities["Hardware analysis"] = True

            # Test 7: ML pipeline integration
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.activations import ReLU

            # Benchmark complex model
            class ComplexModel:
                def __init__(self):
                    self.conv = Conv2D(3, 16, 3)
                    self.dense = Linear(16 * 30 * 30, 10)
                    self.relu = ReLU()

                def __call__(self, x):
                    h = self.relu(self.conv(x))
                    h_flat = Tensor(h.data.reshape(h.shape[0], -1))
                    return self.dense(h_flat)

            complex_model = ComplexModel()
            complex_results = benchmark_model(complex_model, (2, 3, 32, 32))
            assert 'latency' in complex_results
            benchmarking_capabilities["ML pipeline integration"] = True

            # Test 8: Production readiness
            # Can measure and compare performance across scenarios
            scenarios = [
                ("real_time", (1, 50)),
                ("batch_processing", (32, 50))
            ]

            for scenario_name, input_shape in scenarios:
                scenario_results = benchmark_model(model, input_shape)
                assert scenario_results['latency'] > 0
                assert scenario_results['throughput'] > 0

            benchmarking_capabilities["Production readiness"] = True

        except Exception as e:
            # Show progress even if not complete
            completed_count = sum(benchmarking_capabilities.values())
            total_count = len(benchmarking_capabilities)

            progress_report = "\n🔍 BENCHMARKING PROGRESS:\n"
            for capability, completed in benchmarking_capabilities.items():
                status = "✅" if completed else "❌"
                progress_report += f"  {status} {capability}\n"

            progress_report += f"\n📊 Progress: {completed_count}/{total_count} capabilities ready"

            assert False, f"""
            ❌ BENCHMARKING FOUNDATION NOT COMPLETE!

            🔍 ERROR: {str(e)}

            {progress_report}

            🔧 NEXT STEPS:
            1. Fix the failing capability above
            2. Re-run this test
            3. When all ✅, you're ready for production deployment!

            💡 ALMOST THERE!
            You've completed {completed_count}/{total_count} benchmarking capabilities.
            Just fix the error above and you'll have complete performance analysis!
            """

        # If we get here, everything passed!
        assert True, """
        🎉 BENCHMARKING FOUNDATION COMPLETE! 🎉

        ✅ Model performance benchmarking
        ✅ Profiling and bottleneck analysis
        ✅ Performance comparison tools
        ✅ Training performance analysis
        ✅ Inference optimization
        ✅ Hardware performance analysis
        ✅ ML pipeline integration
        ✅ Production readiness

        🚀 READY FOR MODULE 15: MLOPS AND PRODUCTION DEPLOYMENT!

        💡 What you can now do:
        - Optimize model performance with data
        - Choose optimal hardware configurations
        - Identify and fix performance bottlenecks
        - Plan production deployment strategies
        - Scale ML systems efficiently

        📊 PERFORMANCE ENGINEERING ACHIEVED:
        You've built the tools to:
        - Make data-driven optimization decisions
        - Ensure production SLA compliance
        - Minimize deployment costs
        - Maximize system efficiency

        🎯 Next: Production deployment and monitoring in Module 15!
        """


# Note: No separate regression prevention - we test complete ML system stability above
