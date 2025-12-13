#!/usr/bin/env python3
"""
Simple Integration Test - Core Functionality
============================================

This test validates basic functionality of modules 1-4 without complex learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.layers import Linear

def test_basic_tensor_operations():
    """Test basic tensor operations."""
    print("🧪 Testing Basic Tensor Operations...")

    # Test creation and basic properties
    t1 = Tensor([1, 2, 3])
    assert t1.shape == (3,), f"Expected shape (3,), got {t1.shape}"

    t2 = Tensor([[1, 2], [3, 4]])
    assert t2.shape == (2, 2), f"Expected shape (2, 2), got {t2.shape}"

    print("  ✓ Tensor creation and shapes work")

    # Test basic arithmetic
    t3 = Tensor([1, 2, 3])
    t4 = Tensor([4, 5, 6])

    # Test addition
    t5 = t3 + t4
    expected = np.array([5, 7, 9])
    np.testing.assert_array_equal(t5.data, expected)
    print("  ✓ Tensor addition works")

    # Test scalar operations
    t6 = t3 * 2
    expected = np.array([2, 4, 6])
    np.testing.assert_array_equal(t6.data, expected)
    print("  ✓ Tensor scalar multiplication works")

    print("✅ Basic tensor operations working!")
    return True

def test_activation_functions():
    """Test activation functions."""
    print("🔥 Testing Activation Functions...")

    # Test ReLU
    relu = ReLU()
    test_data = Tensor([[-2, -1, 0, 1, 2]])
    relu_out = relu(test_data)
    expected = np.array([[0, 0, 0, 1, 2]])
    np.testing.assert_array_equal(relu_out.data, expected)
    print("  ✓ ReLU activation works")

    # Test Sigmoid
    sigmoid = Sigmoid()
    sig_in = Tensor([[0.0]])
    sig_out = sigmoid(sig_in)
    assert abs(sig_out.data[0, 0] - 0.5) < 0.01, "Sigmoid(0) should be ~0.5"
    print("  ✓ Sigmoid activation works")

    print("✅ Activation functions working!")
    return True

def test_dense_layer_basic():
    """Test basic dense layer functionality."""
    print("🏗️  Testing Dense Layer...")

    # Create a simple dense layer
    dense = Linear(3, 2)  # 3 inputs, 2 outputs

    # Test with simple input
    x = Tensor([[1, 0, 1]])  # batch_size=1, input_size=3
    output = dense(x)

    print(f"  ✓ Dense layer forward pass successful")
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Weights shape: {linear.weight.shape}")
    print(f"    Bias shape: {linear.bias.shape}")

    # Check output shape is correct
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"

    # Test with batch input
    x_batch = Tensor([[1, 0, 1], [0, 1, 0]])  # batch_size=2
    output_batch = dense(x_batch)
    assert output_batch.shape == (2, 2), f"Expected batch output shape (2, 2), got {output_batch.shape}"

    print("✅ Dense layer working!")
    return True

def test_simple_forward_pass():
    """Test a simple 2-layer forward pass."""
    print("🚀 Testing Simple Forward Pass...")

    # Create simple 2-layer network manually
    layer1 = Linear(2, 3)  # 2 -> 3
    layer2 = Linear(3, 1)  # 3 -> 1
    relu = ReLU()
    sigmoid = Sigmoid()

    # Simple forward pass
    x = Tensor([[1, 0]])  # Single sample

    # Layer 1
    h1 = layer1(x)
    print(f"  ✓ Layer 1 output shape: {h1.shape}")

    # ReLU
    h1_activated = relu(h1)
    print(f"  ✓ ReLU output shape: {h1_activated.shape}")

    # Layer 2
    h2 = layer2(h1_activated)
    print(f"  ✓ Layer 2 output shape: {h2.shape}")

    # Final activation
    output = sigmoid(h2)
    print(f"  ✓ Final output shape: {output.shape}")
    print(f"  ✓ Final output value: {output.data[0, 0]}")

    # Verify output is in sigmoid range
    assert 0 <= output.data[0, 0] <= 1, "Sigmoid output should be in [0, 1]"

    print("✅ Simple forward pass working!")
    return True

def run_simple_integration_test():
    """Run simple integration tests."""
    print("=" * 60)
    print("🔥 SIMPLE INTEGRATION TEST - Core Modules")
    print("=" * 60)
    print()

    success = True
    tests = [
        test_basic_tensor_operations,
        test_activation_functions,
        test_dense_layer_basic,
        test_simple_forward_pass
    ]

    for test in tests:
        try:
            if not test():
                success = False
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            success = False
            print()

    if success:
        print("🎉 SIMPLE INTEGRATION TEST PASSED!")
        print("✅ Core modules are working correctly")
    else:
        print("❌ SIMPLE INTEGRATION TEST FAILED!")
        print("Check module implementations")

    print("=" * 60)
    return success

if __name__ == "__main__":
    run_simple_integration_test()
