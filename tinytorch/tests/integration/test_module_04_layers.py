"""
Integration Tests for Module 04: Layers
========================================

These tests run automatically when you complete Module 04 with:
`tito module complete 04_layers`

They verify that layers work correctly with other completed modules.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def test_layers_integration():
    """Test that layers integrate with tensors and activations."""

    print("Running Module 04 Integration Tests...")
    print("-" * 40)

    # Test 1: Layers work with Tensors
    print("Test 1: Layer + Tensor integration")
    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear as Dense

        layer = Linear(3, 2)
        x = Tensor(np.random.randn(5, 3))
        output = layer(x)

        assert output.shape == (5, 2), f"Expected shape (5, 2), got {output.shape}"
        print("✅ Layers work with Tensors")
    except Exception as e:
        print(f"❌ Layer-Tensor integration failed: {e}")
        return False

    # Test 2: Layers work with Activations
    print("Test 2: Layer + Activation integration")
    try:
        from tinytorch.core.activations import ReLU, Sigmoid

        layer1 = Linear(4, 8)
        relu = ReLU()
        layer2 = Linear(8, 4)
        sigmoid = Sigmoid()

        x = Tensor(np.random.randn(2, 4))
        h = relu(layer1(x))
        y = sigmoid(layer2(h))

        assert y.shape == (2, 4), f"Expected shape (2, 4), got {y.shape}"
        print("✅ Layers work with Activations")
    except Exception as e:
        print(f"❌ Layer-Activation integration failed: {e}")
        return False

    # Test 3: Multi-layer stacking
    print("Test 3: Multi-layer network construction")
    try:
        layers = [
            Linear(10, 20),
            ReLU(),
            Linear(20, 15),
            ReLU(),
            Linear(15, 5)
        ]

        x = Tensor(np.random.randn(3, 10))
        for layer in layers:
            x = layer(x)

        assert x.shape == (3, 5), f"Expected final shape (3, 5), got {x.shape}"
        print("✅ Multi-layer networks work")
    except Exception as e:
        print(f"❌ Multi-layer stacking failed: {e}")
        return False

    # Test 4: Parameter access
    print("Test 4: Parameter management")
    try:
        layer = Linear(5, 3)

        assert hasattr(layer, 'weight'), "Layer missing weights"
        assert hasattr(layer, 'bias'), "Layer missing bias"
        assert layer.weight.shape == (5, 3), f"Wrong weight shape: {layer.weight.shape}"
        assert layer.bias.shape == (3,), f"Wrong bias shape: {layer.bias.shape}"

        total_params = layer.weight.size + layer.bias.size
        assert total_params == 18, f"Expected 18 parameters, got {total_params}"
        print("✅ Parameter management works")
    except Exception as e:
        print(f"❌ Parameter management failed: {e}")
        return False

    print("-" * 40)
    print("✅ All Module 04 integration tests passed!")
    print()
    print("🎯 CAPABILITY UNLOCKED: Network Architecture Construction")
    print("📚 You can now run: python examples/perceptron_1957/rosenblatt_perceptron.py")
    print()
    return True


if __name__ == "__main__":
    success = test_layers_integration()
    exit(0 if success else 1)
