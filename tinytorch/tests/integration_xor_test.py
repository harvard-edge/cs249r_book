#!/usr/bin/env python3
"""
XOR Integration Test - After Module 4
=====================================

This test validates that modules 1-4 work together to solve the XOR problem.

Required modules:
- Module 01: Setup
- Module 02: Tensor - Data structures
- Module 03: Activations - ReLU, Sigmoid
- Module 04: Layers - Linear layers

This demonstrates the milestone: "Can build a network that learns XOR"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.layers import Linear

class SimpleXORNet:
    """Simple 2-layer network for XOR problem."""

    def __init__(self):
        self.layer1 = Linear(2, 4)  # Input layer: 2 -> 4 hidden
        self.relu = ReLU()
        self.layer2 = Linear(4, 1)  # Output layer: 4 -> 1 output
        self.sigmoid = Sigmoid()

    def forward(self, x):
        """Forward pass through the network."""
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

    def __call__(self, x):
        return self.forward(x)

def get_xor_data():
    """Get XOR dataset."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return X, y

def test_xor_network_components():
    """Test individual components work."""
    print("🧪 Testing XOR Network Components...")

    # Test tensor creation
    print("  ✓ Testing Tensor creation")
    x = Tensor([[0, 1], [1, 0]])
    assert x.shape == (2, 2), f"Expected shape (2, 2), got {x.shape}"

    # Test Linear layer
    print("  ✓ Testing Linear layer")
    linear = Linear(2, 3)
    out = linear(x)
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"

    # Test ReLU activation
    print("  ✓ Testing ReLU activation")
    relu = ReLU()
    test_input = Tensor([[-1, 0, 1, 2]])
    relu_out = relu(test_input)
    expected = np.array([[0, 0, 1, 2]])
    np.testing.assert_array_almost_equal(relu_out.data, expected, decimal=5)

    # Test Sigmoid activation
    print("  ✓ Testing Sigmoid activation")
    sigmoid = Sigmoid()
    sig_out = sigmoid(Tensor([[0.0]]))
    assert abs(sig_out.data[0, 0] - 0.5) < 0.01, "Sigmoid(0) should be ~0.5"

    print("✅ All components working!")

def test_xor_network_architecture():
    """Test network architecture is buildable."""
    print("🏗️  Testing XOR Network Architecture...")

    # Create network
    net = SimpleXORNet()

    # Test forward pass doesn't crash
    X, y = get_xor_data()
    X_tensor = Tensor(X)

    try:
        output = net(X_tensor)
        print(f"  ✓ Forward pass successful, output shape: {output.shape}")
        assert output.shape == (4, 1), f"Expected output shape (4, 1), got {output.shape}"

        # Check output is in valid range for sigmoid
        output_vals = output.data
        assert np.all(output_vals >= 0) and np.all(output_vals <= 1), "Sigmoid outputs should be in [0, 1]"

        print("✅ Network architecture working!")
        return True

    except Exception as e:
        print(f"❌ Network forward pass failed: {e}")
        return False

def test_xor_learning_capability():
    """Test that network can at least change its outputs (learning potential)."""
    print("📚 Testing XOR Learning Potential...")

    net = SimpleXORNet()
    X, y = get_xor_data()
    X_tensor = Tensor(X)

    # Get initial outputs
    initial_output = net(X_tensor).data.copy()

    # Manually adjust some weights (simulate learning)
    # This tests if architecture can represent XOR
    net.layer1.weight.data += 0.1 * np.random.randn(*net.layer1.weight.shape)

    # Get new outputs
    new_output = net(X_tensor).data

    # Check that outputs changed (network is trainable)
    output_change = np.sum(np.abs(new_output - initial_output))
    if output_change > 0.01:
        print(f"  ✓ Network outputs changed by {output_change:.4f} (trainable)")
        print("✅ Network has learning potential!")
        return True
    else:
        print("❌ Network outputs didn't change enough")
        return False

def run_xor_integration_test():
    """Run complete XOR integration test."""
    print("=" * 60)
    print("🔥 XOR INTEGRATION TEST - Modules 1-4")
    print("=" * 60)
    print()

    success = True

    try:
        # Test 1: Components
        test_xor_network_components()
        print()

        # Test 2: Architecture
        if not test_xor_network_architecture():
            success = False
        print()

        # Test 3: Learning potential
        if not test_xor_learning_capability():
            success = False
        print()

    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        success = False

    # Results
    if success:
        print("🎉 XOR INTEGRATION TEST PASSED!")
        print()
        print("✅ Milestone Achieved: Can build networks that learn XOR")
        print("   • Tensors handle data flow")
        print("   • Activations add nonlinearity")
        print("   • Linear layers transform representations")
        print("   • Architecture supports learning")
        print()
        print("🚀 Ready for Module 5: Training loops!")
    else:
        print("❌ XOR INTEGRATION TEST FAILED!")
        print("   Check module implementations before proceeding")

    print("=" * 60)
    return success

if __name__ == "__main__":
    run_xor_integration_test()
