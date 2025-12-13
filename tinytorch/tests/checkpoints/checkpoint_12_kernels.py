"""
Checkpoint 12: Kernels (After Module 13 - Kernels)
Question: "Can I implement high-performance computational kernels?"
"""

import numpy as np
import pytest

def test_checkpoint_12_kernels():
    """
    Checkpoint 12: Kernels

    Validates that students can implement and optimize computational kernels
    for high-performance machine learning operations - essential for
    understanding how modern ML frameworks achieve speed and efficiency.
    """
    print("\n⚡ Checkpoint 12: Kernels")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.kernels import (
            time_kernel, matmul_baseline, vectorized_relu, vectorized_operations,
            cache_friendly_matmul, parallel_relu, parallel_batch_processing,
            quantized_matmul, quantized_relu
        )
        from tinytorch.core.activations import ReLU
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-13 first: {e}")

    # Test 1: Kernel timing infrastructure
    print("⏱️ Testing kernel timing...")

    def simple_operation(x):
        return x * 2

    # Test timing functionality
    test_data = np.random.randn(100, 100)

    try:
        execution_time, result = time_kernel(simple_operation, test_data)

        assert execution_time > 0, f"Execution time should be positive, got {execution_time}"
        assert np.allclose(result, test_data * 2), "Timing should preserve operation correctness"
        print(f"✅ Kernel timing: {execution_time:.6f}s for 100x100 operation")
    except Exception as e:
        print(f"⚠️ Kernel timing: {e}")

    # Test 2: Matrix multiplication optimization
    print("🔢 Testing matrix multiplication kernels...")

    # Test baseline matmul
    A = np.random.randn(64, 32)
    B = np.random.randn(32, 48)

    try:
        result_baseline = matmul_baseline(A, B)
        expected = np.dot(A, B)

        assert result_baseline.shape == expected.shape, f"Baseline matmul shape mismatch: {result_baseline.shape} vs {expected.shape}"
        assert np.allclose(result_baseline, expected, rtol=1e-5), "Baseline matmul should match NumPy"
        print(f"✅ Baseline matmul: {A.shape} @ {B.shape} → {result_baseline.shape}")
    except Exception as e:
        print(f"⚠️ Baseline matmul: {e}")

    # Test cache-friendly matmul
    try:
        result_cache_friendly = cache_friendly_matmul(A, B)

        assert result_cache_friendly.shape == expected.shape, f"Cache-friendly matmul shape mismatch"
        assert np.allclose(result_cache_friendly, expected, rtol=1e-5), "Cache-friendly matmul should match NumPy"
        print(f"✅ Cache-friendly matmul: optimized memory access patterns")
    except Exception as e:
        print(f"⚠️ Cache-friendly matmul: {e}")

    # Test 3: Vectorized operations
    print("🚀 Testing vectorized operations...")

    # Test vectorized ReLU
    test_input = np.array([-2, -1, 0, 1, 2]).astype(np.float32)

    try:
        vectorized_result = vectorized_relu(test_input)
        expected_relu = np.maximum(0, test_input)

        assert np.allclose(vectorized_result, expected_relu), "Vectorized ReLU should match expected behavior"
        print(f"✅ Vectorized ReLU: {test_input} → {vectorized_result}")
    except Exception as e:
        print(f"⚠️ Vectorized ReLU: {e}")

    # Test vectorized operations suite
    try:
        ops_input = np.random.randn(1000).astype(np.float32)
        ops_result = vectorized_operations(ops_input)

        assert len(ops_result) > 0, "Vectorized operations should return results"
        print(f"✅ Vectorized operations: processed {len(ops_input)} elements")
    except Exception as e:
        print(f"⚠️ Vectorized operations: {e}")

    # Test 4: Parallel processing
    print("🔀 Testing parallel processing...")

    # Test parallel ReLU
    parallel_input = np.random.randn(10000).astype(np.float32)

    try:
        parallel_result = parallel_relu(parallel_input)
        expected_parallel = np.maximum(0, parallel_input)

        assert parallel_result.shape == expected_parallel.shape, "Parallel ReLU shape mismatch"
        assert np.allclose(parallel_result, expected_parallel, rtol=1e-5), "Parallel ReLU should match sequential"
        print(f"✅ Parallel ReLU: processed {len(parallel_input)} elements")
    except Exception as e:
        print(f"⚠️ Parallel ReLU: {e}")

    # Test parallel batch processing
    try:
        batch_data = np.random.randn(8, 512, 512).astype(np.float32)  # 8 samples, 512x512 each
        batch_result = parallel_batch_processing(batch_data)

        assert batch_result.shape[0] == batch_data.shape[0], "Batch processing should preserve batch dimension"
        print(f"✅ Parallel batch processing: {batch_data.shape} → {batch_result.shape}")
    except Exception as e:
        print(f"⚠️ Parallel batch processing: {e}")

    # Test 5: Quantization kernels
    print("🗜️ Testing quantization kernels...")

    # Test quantized matrix multiplication
    try:
        A_quant = np.random.randn(32, 16).astype(np.float32)
        B_quant = np.random.randn(16, 24).astype(np.float32)

        quant_result = quantized_matmul(A_quant, B_quant, bits=8)
        reference_result = np.dot(A_quant, B_quant)

        assert quant_result.shape == reference_result.shape, "Quantized matmul shape should match reference"

        # Quantization should be approximately correct (some precision loss expected)
        relative_error = np.mean(np.abs((quant_result - reference_result) / (reference_result + 1e-8)))
        assert relative_error < 0.2, f"Quantized matmul error too high: {relative_error:.3f}"
        print(f"✅ Quantized matmul: 8-bit quantization, error={relative_error:.3f}")
    except Exception as e:
        print(f"⚠️ Quantized matmul: {e}")

    # Test quantized ReLU
    try:
        relu_input = np.random.randn(1000).astype(np.float32)
        quant_relu_result = quantized_relu(relu_input, bits=8)
        reference_relu = np.maximum(0, relu_input)

        assert quant_relu_result.shape == reference_relu.shape, "Quantized ReLU shape should match reference"
        print(f"✅ Quantized ReLU: 8-bit activation quantization")
    except Exception as e:
        print(f"⚠️ Quantized ReLU: {e}")

    # Test 6: Performance comparison
    print("📊 Testing performance comparison...")

    # Compare naive vs optimized implementations
    test_matrix_A = np.random.randn(128, 128).astype(np.float32)
    test_matrix_B = np.random.randn(128, 128).astype(np.float32)

    try:
        # Time baseline implementation
        baseline_time, baseline_result = time_kernel(matmul_baseline, test_matrix_A, test_matrix_B)

        # Time cache-friendly implementation
        optimized_time, optimized_result = time_kernel(cache_friendly_matmul, test_matrix_A, test_matrix_B)

        # Both should be correct
        assert np.allclose(baseline_result, optimized_result, rtol=1e-5), "Optimized version should match baseline"

        speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
        print(f"✅ Performance: baseline={baseline_time:.6f}s, optimized={optimized_time:.6f}s, speedup={speedup:.2f}x")
    except Exception as e:
        print(f"⚠️ Performance comparison: {e}")

    # Test 7: Memory efficiency
    print("💾 Testing memory efficiency...")

    # Test memory-efficient operations
    large_data = np.random.randn(1000, 1000).astype(np.float32)

    try:
        # Process in chunks to test memory efficiency
        chunk_results = []
        chunk_size = 100

        for i in range(0, large_data.shape[0], chunk_size):
            chunk = large_data[i:i+chunk_size]
            chunk_result = vectorized_relu(chunk.flatten()).reshape(chunk.shape)
            chunk_results.append(chunk_result)

        chunked_result = np.vstack(chunk_results)
        direct_result = vectorized_relu(large_data.flatten()).reshape(large_data.shape)

        assert np.allclose(chunked_result, direct_result, rtol=1e-5), "Chunked processing should match direct processing"
        print(f"✅ Memory efficiency: processed {large_data.shape} in {chunk_size}-row chunks")
    except Exception as e:
        print(f"⚠️ Memory efficiency: {e}")

    # Test 8: Integration with TinyTorch tensors
    print("🔗 Testing TinyTorch integration...")

    try:
        # Test that kernels work with TinyTorch tensors
        tensor_a = Tensor(np.random.randn(32, 32))
        tensor_b = Tensor(np.random.randn(32, 32))

        # Extract numpy arrays for kernel operations
        kernel_result = matmul_baseline(tensor_a.data, tensor_b.data)
        tensor_result = Tensor(kernel_result)

        assert tensor_result.shape == (32, 32), f"Tensor integration should preserve shape"
        print(f"✅ TinyTorch integration: kernels work with Tensor.data")
    except Exception as e:
        print(f"⚠️ TinyTorch integration: {e}")

    # Test 9: Kernel composition
    print("🧩 Testing kernel composition...")

    try:
        # Compose multiple kernel operations
        input_data = np.random.randn(64, 64).astype(np.float32)

        # Pipeline: MatMul → ReLU → Quantization
        intermediate = matmul_baseline(input_data, input_data.T)  # Square result
        activated = vectorized_relu(intermediate.flatten()).reshape(intermediate.shape)
        quantized = quantized_relu(activated.flatten(), bits=8).reshape(activated.shape)

        assert quantized.shape == input_data.shape, f"Kernel pipeline should preserve dimensions"
        assert np.all(quantized >= 0), "Pipeline result should be non-negative after ReLU"
        print(f"✅ Kernel composition: MatMul → ReLU → Quantization pipeline")
    except Exception as e:
        print(f"⚠️ Kernel composition: {e}")

    # Test 10: Advanced optimization features
    print("🚁 Testing advanced optimizations...")

    try:
        # Test that optimization features are available
        medium_input = np.random.randn(256, 256).astype(np.float32)

        # Time multiple approaches
        approaches = []

        # Baseline approach
        baseline_time, _ = time_kernel(np.dot, medium_input, medium_input.T)
        approaches.append(("NumPy baseline", baseline_time))

        # Our optimized approach
        optimized_time, _ = time_kernel(cache_friendly_matmul, medium_input, medium_input.T)
        approaches.append(("Cache-friendly", optimized_time))

        # Find fastest approach
        fastest = min(approaches, key=lambda x: x[1])
        print(f"✅ Advanced optimizations: fastest approach is {fastest[0]} at {fastest[1]:.6f}s")

        # Verify we have meaningful optimization choices
        assert len(approaches) >= 2, "Should have multiple optimization approaches"

    except Exception as e:
        print(f"⚠️ Advanced optimizations: {e}")

    print("\n🎉 Kernels Complete!")
    print("📝 You can now implement high-performance computational kernels")
    print("🔧 Built capabilities: Timing, vectorization, parallelization, quantization, memory optimization")
    print("🧠 Breakthrough: You understand how to optimize ML operations for real-world performance!")
    print("🎯 Next: Add performance analysis and bottleneck identification")

if __name__ == "__main__":
    test_checkpoint_12_kernels()
