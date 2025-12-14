#!/usr/bin/env python3
"""
Integration Tests for TinyTorch Layers Module

This file contains the integration tests that were removed from Module 03
to keep the module focused on unit testing only. These tests demonstrate
how layers work together with other modules and complete system behaviors.
"""

import sys
import os
import numpy as np
import pytest

# Try to import from the package first, fall back to dev files
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Linear, Sequential, Flatten
    from tinytorch.nn import Module
except ImportError:
    pytest.skip("TinyTorch package not properly installed", allow_module_level=True)


def test_complete_neural_networks():
    """Integration test: Complete neural networks using all implemented components."""
    print("🔥 Complete Neural Network Integration Demo")
    print("=" * 50)

    print("\n1. MLP for Classification (MNIST-style):")
    # Multi-layer perceptron for image classification
    mlp = Sequential([
        Flatten(),              # Flatten input images
        Linear(784, 256),       # First hidden layer
        Linear(256, 128),       # Second hidden layer
        Linear(128, 10)         # Output layer (10 classes)
    ])

    # Test with batch of "images"
    batch_images = Tensor(np.random.randn(32, 28, 28))  # 32 MNIST-like images
    mlp_output = mlp(batch_images)
    print(f"   Input: {batch_images.shape} (batch of 28x28 images)")
    print(f"   Output: {mlp_output.shape} (class logits for 32 images)")
    print(f"   Parameters: {len(mlp.parameters())} tensors")

    # Validate shapes
    assert batch_images.shape == (32, 28, 28), "Input batch shape incorrect"
    assert mlp_output.shape == (32, 10), "MLP output shape incorrect"
    print("   ✅ MLP integration test passed")

    print("\n2. CNN-style Architecture (with Flatten):")
    # Simulate CNN -> Flatten -> Dense pattern
    cnn_style = Sequential([
        # Simulate Conv2D output with random "features"
        Flatten(),              # Flatten spatial features
        Linear(512, 256),       # Dense layer after convolution
        Linear(256, 10)         # Classification head
    ])

    # Test with simulated conv output
    conv_features = Tensor(np.random.randn(16, 8, 8, 8))  # Simulated (B,C,H,W)
    cnn_output = cnn_style(conv_features)
    print(f"   Input: {conv_features.shape} (simulated conv features)")
    print(f"   Output: {cnn_output.shape} (class predictions)")

    # Validate shapes
    assert conv_features.shape == (16, 8, 8, 8), "Conv features shape incorrect"
    assert cnn_output.shape == (16, 10), "CNN-style output shape incorrect"
    print("   ✅ CNN-style integration test passed")

    print("\n3. Deep Network with Many Layers:")
    # Demonstrate deep composition
    deep_net = Sequential()
    layer_sizes = [100, 80, 60, 40, 20, 10]

    for i in range(len(layer_sizes) - 1):
        deep_net.add(Linear(layer_sizes[i], layer_sizes[i+1]))
        print(f"   Added layer: {layer_sizes[i]} -> {layer_sizes[i+1]}")

    # Test deep network
    deep_input = Tensor(np.random.randn(8, 100))
    deep_output = deep_net(deep_input)
    print(f"   Deep network: {deep_input.shape} -> {deep_output.shape}")
    print(f"   Total parameters: {len(deep_net.parameters())} tensors")

    # Validate shapes
    assert deep_input.shape == (8, 100), "Deep network input shape incorrect"
    assert deep_output.shape == (8, 10), "Deep network output shape incorrect"
    print("   ✅ Deep network integration test passed")

    print("\n4. Parameter Management Across Networks:")
    networks = {'MLP': mlp, 'CNN-style': cnn_style, 'Deep': deep_net}

    for name, net in networks.items():
        params = net.parameters()
        total_params = sum(p.data.size for p in params)
        memory_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
        print(f"   {name}: {len(params)} param tensors, {total_params:,} total params, {memory_mb:.2f} MB")

    print("\n🎉 ALL INTEGRATION TESTS PASSED!")
    print("   • Module system enables automatic parameter collection")
    print("   • Linear layers handle matrix transformations")
    print("   • Sequential composes layers into complete architectures")
    print("   • Flatten connects different layer types")
    print("   • Everything integrates for production-ready neural networks!")


def test_cross_module_compatibility():
    """Test that layers work correctly with tensor operations."""
    print("\n🔬 Cross-Module Compatibility Testing")
    print("=" * 40)

    # Test 1: Layers work with different tensor creation methods
    layer = Linear(5, 3)

    # From numpy array
    numpy_input = Tensor(np.random.randn(2, 5))
    numpy_output = layer(numpy_input)
    assert numpy_output.shape == (2, 3), "Numpy tensor compatibility failed"
    print("   ✅ Numpy array input compatibility")

    # From list
    list_input = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    list_output = layer(list_input)
    assert list_output.shape == (2, 3), "List tensor compatibility failed"
    print("   ✅ List input compatibility")

    # Test 2: Sequential networks with mixed operations
    complex_net = Sequential([
        Linear(10, 8),
        Flatten(),  # Should be no-op for 2D tensors
        Linear(8, 5)
    ])

    test_input = Tensor(np.random.randn(3, 10))
    complex_output = complex_net(test_input)
    assert complex_output.shape == (3, 5), "Complex network compatibility failed"
    print("   ✅ Mixed operations compatibility")

    print("\n✅ All cross-module compatibility tests passed!")


def run_performance_benchmarks():
    """Run performance benchmarks for integrated systems."""
    print("\n📊 Integration Performance Benchmarks")
    print("=" * 40)

    import time

    # Benchmark: Large MLP forward pass
    large_mlp = Sequential([
        Linear(1000, 500),
        Linear(500, 250),
        Linear(250, 100),
        Linear(100, 10)
    ])

    large_batch = Tensor(np.random.randn(1000, 1000))  # 1000 samples, 1000 features

    # Warm up
    _ = large_mlp(large_batch)

    # Benchmark
    start_time = time.time()
    for _ in range(10):
        output = large_mlp(large_batch)
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    samples_per_sec = 1000 / avg_time

    print(f"   Large MLP (1000→500→250→100→10):")
    print(f"   Average time: {avg_time:.4f} seconds")
    print(f"   Throughput: {samples_per_sec:.0f} samples/second")
    print(f"   Output shape: {output.shape}")

    # Memory usage estimate
    total_params = sum(p.data.size for p in large_mlp.parameters())
    param_memory_mb = total_params * 4 / (1024 * 1024)
    activation_memory_mb = (large_batch.data.size + output.data.size) * 4 / (1024 * 1024)

    print(f"   Parameter memory: {param_memory_mb:.2f} MB")
    print(f"   Activation memory: {activation_memory_mb:.2f} MB")
    print(f"   Total estimated memory: {param_memory_mb + activation_memory_mb:.2f} MB")

    print("\n✅ Performance benchmarks completed!")


if __name__ == "__main__":
    print("🚀 TINYTORCH LAYERS INTEGRATION TESTS")
    print("=" * 50)
    print("Testing how layers work together with other modules...")

    try:
        # Run all integration tests
        test_complete_neural_networks()
        test_cross_module_compatibility()
        run_performance_benchmarks()

        print("\n" + "=" * 50)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Layers module integrates perfectly with the TinyTorch system!")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
