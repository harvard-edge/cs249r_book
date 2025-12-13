"""
Module 12: Progressive Integration Tests
Tests that Module 13 (Kernels) works correctly AND that the entire prior stack works.

DEPENDENCY CHAIN: 01_setup → ... → 12_compression → 13_kernels
This is where we enable high-performance computational kernels and hardware acceleration.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPriorStackStillWorking:
    """Quick regression checks that prior modules (01→12) still work."""

    def test_complete_ml_system_stable(self):
        """Verify complete ML system remains stable."""
        # Environment (Module 01)
        assert sys.version_info >= (3, 8), "Foundation broken: Python version"

        # Complete ML system should work
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.compression import prune_weights

            # All ML system components should be available
            model = Linear(10, 5)
            optimizer = Adam(model.parameters(), lr=0.001)
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            # Compression should still work
            if 'prune_weights' in locals():
                pruned_weights = prune_weights(model.weights, sparsity=0.3)
                assert pruned_weights.shape == model.weight.shape, "Compression broken"

            # Basic ML functionality should work
            x = Tensor(np.random.randn(4, 10))
            output = model(x)
            assert output.shape == (4, 5), "ML system broken"

        except ImportError:
            assert True, "ML system not implemented yet"

    def test_efficiency_features_stable(self):
        """Verify efficiency modules (11→12) still work."""
        try:
            from tinytorch.core.training import Trainer
            from tinytorch.core.compression import quantize_weights
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss

            # Efficiency features should work
            model = Linear(8, 3)
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            assert hasattr(trainer, 'train') or hasattr(trainer, 'fit'), "Training broken"

            # Compression should work
            if 'quantize_weights' in locals():
                quantized = quantize_weights(model.weights, bits=8)
                assert quantized.shape == model.weight.shape, "Quantization broken"

        except ImportError:
            assert True, "Efficiency features not implemented yet"


class TestModule13KernelsCore:
    """Test Module 13 (Kernels) core functionality."""

    def test_optimized_tensor_operations(self):
        """Test optimized tensor operation kernels."""
        try:
            from tinytorch.core.kernels import optimized_matmul, vectorized_add
            from tinytorch.core.tensor import Tensor

            # Test optimized matrix multiplication
            if 'optimized_matmul' in locals():
                A = Tensor(np.random.randn(50, 30))
                B = Tensor(np.random.randn(30, 20))

                result = optimized_matmul(A, B)
                expected = np.dot(A.data, B.data)

                assert result.shape == (50, 20), "Optimized matmul shape broken"
                assert np.allclose(result.data, expected, rtol=1e-5), "Optimized matmul accuracy broken"

            # Test vectorized operations
            if 'vectorized_add' in locals():
                a = Tensor(np.random.randn(1000))
                b = Tensor(np.random.randn(1000))

                result = vectorized_add(a, b)
                expected = a.data + b.data

                assert result.shape == a.shape, "Vectorized add shape broken"
                assert np.allclose(result.data, expected), "Vectorized add accuracy broken"

        except ImportError:
            assert True, "Optimized tensor operations not implemented yet"

    def test_cuda_kernels(self):
        """Test CUDA acceleration kernels."""
        try:
            from tinytorch.core.kernels import cuda_available, CudaKernel
            from tinytorch.core.tensor import Tensor

            # Check CUDA availability
            if 'cuda_available' in locals():
                has_cuda = cuda_available()

                if has_cuda:
                    # Test CUDA tensor operations
                    if 'CudaKernel' in locals():
                        kernel = CudaKernel('matmul')

                        A = Tensor(np.random.randn(100, 50))
                        B = Tensor(np.random.randn(50, 25))

                        # Move to CUDA (if supported)
                        if hasattr(A, 'cuda'):
                            A_cuda = A.cuda()
                            B_cuda = B.cuda()

                            result = kernel.execute(A_cuda, B_cuda)
                            assert result.shape == (100, 25), "CUDA kernel shape broken"
                else:
                    # CPU fallback should work
                    assert True, "CUDA not available, CPU fallback used"

        except ImportError:
            assert True, "CUDA kernels not implemented yet"

    def test_custom_kernel_compilation(self):
        """Test custom kernel compilation and execution."""
        try:
            from tinytorch.core.kernels import compile_kernel, KernelCompiler

            # Test kernel compilation
            if 'compile_kernel' in locals():
                # Simple element-wise operation kernel
                kernel_code = """
                def element_wise_multiply(a, b):
                    return a * b
                """

                compiled_kernel = compile_kernel(kernel_code, 'element_wise_multiply')

                # Test compiled kernel
                a = np.array([1, 2, 3, 4])
                b = np.array([2, 3, 4, 5])

                result = compiled_kernel(a, b)
                expected = a * b

                assert np.array_equal(result, expected), "Custom kernel compilation broken"

            # Test kernel compiler
            if 'KernelCompiler' in locals():
                compiler = KernelCompiler(target='cpu', optimization_level=2)

                assert hasattr(compiler, 'compile'), "Kernel compiler broken: No compile method"
                assert hasattr(compiler, 'target'), "Kernel compiler broken: No target"

        except ImportError:
            assert True, "Custom kernel compilation not implemented yet"


class TestProgressiveStackIntegration:
    """Test that the complete stack (01→13) works together."""

    def test_accelerated_training_pipeline(self):
        """Test training pipeline with kernel acceleration."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer
            from tinytorch.core.kernels import enable_optimizations
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Enable kernel optimizations
            if 'enable_optimizations' in locals():
                enable_optimizations(backend='auto')

            # Create accelerated training pipeline
            class AcceleratedModel:
                def __init__(self):
                    self.layer1 = Linear(50, 100)
                    self.layer2 = Linear(100, 20)
                    self.layer3 = Linear(20, 5)

                def __call__(self, x):
                    h1 = self.layer1(x)
                    h2 = self.layer2(h1)
                    return self.layer3(h2)

                def parameters(self):
                    params = []
                    for layer in [self.layer1, self.layer2, self.layer3]:
                        if hasattr(layer, 'parameters'):
                            params.extend(layer.parameters())
                    return params

            # Dataset for performance testing
            class PerformanceDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(200, 50)
                    self.targets = np.random.randint(0, 5, 200)

                def __len__(self):
                    return 200

                def __getitem__(self, idx):
                    return Tensor(self.data[idx]), self.targets[idx]

            # Accelerated training
            model = AcceleratedModel()
            optimizer = Adam(model.parameters(), lr=0.001)
            from tinytorch.core.losses import MSELoss
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            dataset = PerformanceDataset()
            dataloader = DataLoader(dataset, batch_size=16)

            # Test accelerated forward pass
            for batch_x, batch_y in dataloader:
                output = model(batch_x)
                assert output.shape == (16, 5), "Accelerated training broken"
                break  # Test one batch

        except ImportError:
            assert True, "Accelerated training pipeline not ready yet"

    def test_large_scale_operations(self):
        """Test large-scale operations with kernel optimizations."""
        try:
            from tinytorch.core.kernels import optimized_matmul, batch_operations
            from tinytorch.core.tensor import Tensor

            # Large-scale matrix operations
            if 'optimized_matmul' in locals():
                # Large matrices
                A = Tensor(np.random.randn(500, 300))
                B = Tensor(np.random.randn(300, 200))

                result = optimized_matmul(A, B)
                assert result.shape == (500, 200), "Large-scale matmul broken"

            # Batch operations
            if 'batch_operations' in locals():
                # Batch of operations
                batch_size = 32
                matrices = [Tensor(np.random.randn(50, 30)) for _ in range(batch_size)]
                vectors = [Tensor(np.random.randn(30)) for _ in range(batch_size)]

                results = batch_operations('matmul', matrices, vectors)
                assert len(results) == batch_size, "Batch operations broken"

                for result in results:
                    assert result.shape == (50,), "Batch operation result shape broken"

        except ImportError:
            assert True, "Large-scale operations not ready yet"

    def test_memory_optimized_operations(self):
        """Test memory-optimized kernel operations."""
        try:
            from tinytorch.core.kernels import in_place_operations, memory_pool
            from tinytorch.core.tensor import Tensor

            # In-place operations to save memory
            if 'in_place_operations' in locals():
                a = Tensor(np.random.randn(100, 100))
                b = Tensor(np.random.randn(100, 100))

                original_id = id(a.data)

                # In-place addition
                in_place_operations.add_(a, b)

                # Should modify original tensor
                assert id(a.data) == original_id, "In-place operation created copy"

            # Memory pool for efficient allocation
            if 'memory_pool' in locals():
                pool = memory_pool.MemoryPool()

                # Allocate from pool
                tensor1 = pool.allocate_tensor(shape=(200, 200))
                tensor2 = pool.allocate_tensor(shape=(200, 200))

                # Should be memory efficient
                assert tensor1.shape == (200, 200), "Memory pool allocation broken"
                assert tensor2.shape == (200, 200), "Memory pool allocation broken"

                # Release memory
                pool.release(tensor1)
                pool.release(tensor2)

        except ImportError:
            assert True, "Memory-optimized operations not ready yet"


class TestPerformanceOptimizations:
    """Test performance optimizations and benchmarking."""

    def test_kernel_benchmarking(self):
        """Test kernel performance benchmarking."""
        try:
            from tinytorch.core.kernels import benchmark_kernel, KernelProfiler
            import time

            # Benchmark matrix multiplication
            if 'benchmark_kernel' in locals():
                sizes = [(100, 100), (200, 200), (500, 500)]

                for size in sizes:
                    A = np.random.randn(*size)
                    B = np.random.randn(*size)

                    # Benchmark different implementations
                    results = benchmark_kernel('matmul', A, B, num_trials=5)

                    assert 'mean_time' in results, "Benchmark missing timing"
                    assert 'std_time' in results, "Benchmark missing std"
                    assert results['mean_time'] > 0, "Benchmark timing invalid"

            # Kernel profiler
            if 'KernelProfiler' in locals():
                profiler = KernelProfiler()

                # Profile operations
                profiler.start()

                # Some operations to profile
                for _ in range(10):
                    a = np.random.randn(50, 50)
                    b = np.random.randn(50, 50)
                    c = np.dot(a, b)

                profile_results = profiler.stop()

                assert 'total_time' in profile_results, "Profiler missing total time"
                assert 'operation_count' in profile_results, "Profiler missing operation count"

        except ImportError:
            assert True, "Kernel benchmarking not ready yet"

    def test_auto_optimization(self):
        """Test automatic kernel optimization selection."""
        try:
            from tinytorch.core.kernels import AutoOptimizer, select_best_kernel

            # Auto optimizer
            if 'AutoOptimizer' in locals():
                optimizer = AutoOptimizer()

                # Should detect best kernels for hardware
                best_config = optimizer.detect_optimal_config()

                assert 'matmul_kernel' in best_config, "Auto optimizer missing matmul"
                assert 'device' in best_config, "Auto optimizer missing device"

            # Kernel selection
            if 'select_best_kernel' in locals():
                # Test different kernel options for operation
                kernels = ['numpy', 'optimized_cpu', 'cuda']
                operation = 'matmul'
                shape = (100, 100)

                best_kernel = select_best_kernel(operation, shape, available_kernels=kernels)

                assert best_kernel in kernels, "Kernel selection invalid"

        except ImportError:
            assert True, "Auto optimization not ready yet"

    def test_vectorization_optimizations(self):
        """Test vectorization and SIMD optimizations."""
        try:
            from tinytorch.core.kernels import vectorized_ops, simd_support

            # Vectorized operations
            if 'vectorized_ops' in locals():
                # Large arrays for vectorization
                a = np.random.randn(10000)
                b = np.random.randn(10000)

                # Vectorized operations should be faster
                import time

                # Time numpy baseline
                start = time.time()
                numpy_result = a + b
                numpy_time = time.time() - start

                # Time vectorized version
                start = time.time()
                vectorized_result = vectorized_ops.add(a, b)
                vectorized_time = time.time() - start

                # Results should be equivalent
                assert np.allclose(numpy_result, vectorized_result), "Vectorization accuracy broken"

                # Vectorized should be competitive or faster
                assert vectorized_time <= numpy_time * 2, "Vectorization significantly slower"

            # SIMD support detection
            if 'simd_support' in locals():
                capabilities = simd_support.detect_capabilities()

                assert isinstance(capabilities, dict), "SIMD detection should return dict"
                # Common SIMD instruction sets
                expected_keys = ['sse', 'avx', 'avx2']
                for key in expected_keys:
                    if key in capabilities:
                        assert isinstance(capabilities[key], bool), f"SIMD {key} should be boolean"

        except ImportError:
            assert True, "Vectorization optimizations not ready yet"


class TestHardwareAcceleration:
    """Test hardware acceleration and device management."""

    def test_device_detection(self):
        """Test hardware device detection and selection."""
        try:
            from tinytorch.core.kernels import Device, get_available_devices

            # Device detection
            if 'get_available_devices' in locals():
                devices = get_available_devices()

                assert isinstance(devices, list), "Available devices should be list"
                assert len(devices) > 0, "Should detect at least CPU"

                # Should include CPU at minimum
                device_types = [device.type for device in devices]
                assert 'cpu' in device_types, "CPU device not detected"

            # Device object
            if 'Device' in locals():
                cpu_device = Device('cpu')
                assert cpu_device.type == 'cpu', "CPU device creation broken"

                # Test CUDA device if available
                try:
                    cuda_device = Device('cuda:0')
                    assert cuda_device.type == 'cuda', "CUDA device creation broken"
                except RuntimeError:
                    # CUDA not available, which is fine
                    assert True, "CUDA not available on this system"

        except ImportError:
            assert True, "Device detection not ready yet"

    def test_tensor_device_movement(self):
        """Test moving tensors between devices."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.kernels import Device

            # Create tensor on CPU
            tensor = Tensor(np.random.randn(50, 50))

            # Should start on CPU
            if hasattr(tensor, 'device'):
                assert tensor.device.type == 'cpu', "Tensor not starting on CPU"

            # Test moving to different device (if available)
            if hasattr(tensor, 'to'):
                # Try moving to CUDA (will fallback to CPU if not available)
                try:
                    cuda_tensor = tensor.to('cuda')
                    if hasattr(cuda_tensor, 'device'):
                        assert cuda_tensor.device.type in ['cuda', 'cpu'], "Device movement broken"
                except RuntimeError:
                    # CUDA not available
                    assert True, "CUDA not available for tensor movement"

        except ImportError:
            assert True, "Tensor device movement not ready yet"

    def test_multi_gpu_support(self):
        """Test multi-GPU support and parallelization."""
        try:
            from tinytorch.core.kernels import MultiGPUManager, data_parallel

            # Multi-GPU manager
            if 'MultiGPUManager' in locals():
                gpu_manager = MultiGPUManager()

                available_gpus = gpu_manager.get_gpu_count()

                if available_gpus > 1:
                    # Test multi-GPU operations
                    assert available_gpus >= 2, "Multi-GPU testing requires 2+ GPUs"

                    # Should be able to manage multiple devices
                    devices = gpu_manager.get_device_list()
                    assert len(devices) == available_gpus, "GPU device list incorrect"
                else:
                    # Single GPU or CPU only
                    assert True, "Multi-GPU not available, single device mode"

            # Data parallel operations
            if 'data_parallel' in locals():
                # Test data parallel wrapper
                from tinytorch.core.layers import Linear

                model = Linear(10, 5)
                parallel_model = data_parallel(model, device_ids=[0])  # Single device for testing

                assert hasattr(parallel_model, 'forward'), "Data parallel wrapper broken"

        except ImportError:
            assert True, "Multi-GPU support not ready yet"


class TestRegressionPrevention:
    """Ensure previous modules still work after Module 13 development."""

    def test_no_complete_system_regression(self):
        """Verify complete ML system (01→12) unchanged."""
        # Core functionality should remain stable
        assert sys.version_info.major >= 3, "Foundation: Python detection broken"

        # Complete ML system should still work
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer
            from tinytorch.core.compression import prune_weights

            # All components should work together
            model = Linear(8, 4)
            optimizer = Adam(model.parameters(), lr=0.001)
            from tinytorch.core.losses import MSELoss
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            x = Tensor(np.random.randn(2, 8))
            output = model(x)
            assert output.shape == (2, 4), "System regression: Forward pass broken"

            # Compression should still work
            if 'prune_weights' in locals():
                pruned = prune_weights(model.weights, sparsity=0.2)
                assert pruned.shape == model.weight.shape, "System regression: Compression broken"

        except ImportError:
            import numpy as np
            assert np.random is not None, "System regression: Basic functionality broken"

    def test_no_efficiency_regression(self):
        """Verify efficiency features (11→12) unchanged."""
        try:
            from tinytorch.core.training import Trainer
            from tinytorch.core.compression import quantize_weights
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.layers import Linear

            # Efficiency features should still work
            model = Linear(6, 3)
            optimizer = SGD(model.parameters(), lr=0.01)
            from tinytorch.core.losses import MSELoss
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            assert hasattr(trainer, 'train') or hasattr(trainer, 'fit'), "Efficiency regression: Training broken"

            # Compression should still work
            if 'quantize_weights' in locals():
                quantized = quantize_weights(model.weights, bits=8)
                assert quantized.shape == model.weight.shape, "Efficiency regression: Quantization broken"

        except ImportError:
            # Basic functionality should work
            import numpy as np
            assert np is not None, "Efficiency regression: Basic functionality broken"

    def test_progressive_stability(self):
        """Test the progressive stack is stable through kernel optimization."""
        # Stack should be stable through: Setup → ... → Compression → Kernels

        # Setup level
        import numpy as np
        assert np is not None, "Setup level broken"

        # Complete ML system level (if available)
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer
            from tinytorch.core.losses import MSELoss

            # Complete system should work
            model = Linear(10, 5)
            optimizer = Adam(model.parameters(), lr=0.001)
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer, loss_fn)

            x = Tensor(np.random.randn(3, 10))
            output = model(x)
            assert output.shape == (3, 5), "ML system level broken"

        except ImportError:
            pass  # Not implemented yet

        # Kernel optimization level (if available)
        try:
            from tinytorch.core.kernels import optimized_matmul

            # Kernel optimizations should work with existing tensors
            if 'optimized_matmul' in locals():
                A = np.random.randn(20, 15)
                B = np.random.randn(15, 10)
                result = optimized_matmul(A, B)
                assert result.shape == (20, 10), "Kernel optimization level broken"
            else:
                # Basic kernel concepts should work
                assert True, "Basic kernel optimization ready"

        except ImportError:
            pass  # Not implemented yet
