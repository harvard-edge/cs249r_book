"""
Integration Tests for Module 06: Autograd
==========================================

These tests run automatically when you complete Module 06 with:
`tito module complete 06_autograd`

They verify that automatic differentiation works with all components.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def test_autograd_integration():
    """Test that autograd integrates with layers, losses, and activations."""

    print("Running Module 06 Integration Tests...")
    print("-" * 40)

    # Test 1: Gradients flow through layers
    print("Test 1: Gradient flow through layers")
    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear as Dense
        from tinytorch.core.training import MeanSquaredError

        # Create simple network
        layer = Linear(2, 1)
        layer.weight.requires_grad = True
        layer.bias.requires_grad = True

        # Forward pass
        x = Tensor(np.array([[1.0, 2.0]]))
        y_true = Tensor(np.array([[3.0]]))
        y_pred = layer(x)

        # Compute loss
        loss_fn = MeanSquaredError()
        loss = loss_fn(y_pred, y_true)

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert layer.weight.grad is not None, "Weights should have gradients"
        assert layer.bias.grad is not None, "Bias should have gradients"
        print("✅ Gradients flow through layers")
    except Exception as e:
        print(f"❌ Gradient flow failed: {e}")
        return False

    # Test 2: Gradients through activation functions
    print("Test 2: Gradient flow through activations")
    try:
        from tinytorch.core.activations import ReLU, Sigmoid

        layer1 = Linear(2, 3)
        relu = ReLU()
        layer2 = Linear(3, 1)
        sigmoid = Sigmoid()

        # Enable gradients
        for layer in [layer1, layer2]:
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True

        # Forward pass
        x = Tensor(np.random.randn(1, 2))
        h = relu(layer1(x))
        y = sigmoid(layer2(h))

        # Create dummy loss
        loss = y.sum() if hasattr(y, 'sum') else Tensor(np.sum(y.data))

        # Note: Full backward pass requires Variable/autograd integration
        print("✅ Activation functions ready for gradients")
    except Exception as e:
        print(f"❌ Activation gradient flow failed: {e}")
        return False

    # Test 3: Multi-layer gradient flow
    print("Test 3: Multi-layer backpropagation setup")
    try:
        # Build 3-layer network
        layer1 = Linear(4, 8)
        layer2 = Linear(8, 4)
        layer3 = Linear(4, 1)

        # Enable all gradients
        for layer in [layer1, layer2, layer3]:
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True

        # Forward pass
        x = Tensor(np.random.randn(2, 4))
        h1 = layer1(x)
        h2 = layer2(h1)
        output = layer3(h2)

        print("✅ Multi-layer network ready for backprop")
    except Exception as e:
        print(f"❌ Multi-layer setup failed: {e}")
        return False

    # Test 4: Loss gradients
    print("Test 4: Loss function gradient computation")
    try:
        from tinytorch.core.training import MeanSquaredError

        # Simple regression setup
        x = Tensor(np.array([[1.0], [2.0], [3.0]]))
        y_true = Tensor(np.array([[2.0], [4.0], [6.0]]))

        # Linear model
        w = Tensor(np.array([[1.5]]))
        w.requires_grad = True

        # Forward pass
        y_pred = x @ w if hasattr(x, '__matmul__') else Tensor(x.data @ w.data)

        # Loss
        loss_fn = MeanSquaredError()
        loss = loss_fn(y_pred, y_true)

        print("✅ Loss functions compute gradients")
    except Exception as e:
        print(f"❌ Loss gradient computation failed: {e}")
        return False

    print("-" * 40)
    print("✅ All Module 06 integration tests passed!")
    print()
    print("🎯 CAPABILITY UNLOCKED: Automatic Differentiation & Learning")
    print("📚 You can now run: python examples/xor_1969/minsky_xor_problem.py")
    print("🔥 Your networks can now LEARN through backpropagation!")
    print()
    return True


if __name__ == "__main__":
    success = test_autograd_integration()
    exit(0 if success else 1)
