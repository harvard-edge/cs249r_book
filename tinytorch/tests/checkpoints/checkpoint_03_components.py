"""
Checkpoint 3: Components (After Module 4 - Layers)
Question: "Can I build the fundamental building blocks of neural networks?"
"""

import numpy as np
import pytest

def test_checkpoint_03_components():
    """
    Checkpoint 3: Components

    Validates that students can create learnable layers with parameters - the
    fundamental building blocks that can be trained to transform data and learn
    patterns from examples.
    """
    print("\n⚙️ Checkpoint 3: Components")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-4 first: {e}")

    # Test 1: Dense layer creation with parameters
    print("🔧 Testing Dense layer creation...")
    layer = Linear(input_size=10, output_size=5)

    assert hasattr(layer, 'weight'), "Dense layer should have weights"
    assert hasattr(layer, 'bias'), "Dense layer should have bias"
    assert layer.weight.shape == (10, 5), f"Weights shape should be (10, 5), got {layer.weight.shape}"
    assert layer.bias.shape == (5,), f"Bias shape should be (5,), got {layer.bias.shape}"
    print(f"✅ Dense layer created: {layer.weight.shape} weights, {layer.bias.shape} bias")

    # Test 2: Forward pass through layer
    print("➡️ Testing forward pass...")
    input_data = Tensor(np.random.randn(1, 10))  # Single sample
    output = layer(input_data)

    assert output.shape == (1, 5), f"Output shape should be (1, 5), got {output.shape}"
    print(f"✅ Forward pass: {input_data.shape} → {output.shape}")

    # Test 3: Batch processing through layer
    print("📦 Testing batch processing...")
    batch_input = Tensor(np.random.randn(3, 10))  # 3 samples
    batch_output = layer(batch_input)

    assert batch_output.shape == (3, 5), f"Batch output shape should be (3, 5), got {batch_output.shape}"
    print(f"✅ Batch processing: {batch_input.shape} → {batch_output.shape}")

    # Test 4: Parameter learning capability
    print("📚 Testing parameter access for learning...")
    original_weights = layer.weight.data.copy()
    original_bias = layer.bias.data.copy()

    # Simulate parameter update (what optimizers will do)
    layer.weight.data += 0.1
    layer.bias.data += 0.01

    assert not np.array_equal(layer.weight.data, original_weights), "Weights should be modifiable for learning"
    assert not np.array_equal(layer.bias.data, original_bias), "Bias should be modifiable for learning"
    print(f"✅ Parameters are learnable: weights and bias can be updated")

    # Test 5: Integration with activation functions
    print("🔗 Testing layer + activation integration...")
    relu = ReLU()

    # Create a small network: input → Dense → ReLU
    test_input = Tensor([[-1, 0, 1, 2, -2, 3, -3, 4, -4, 5]])  # 1 sample, 10 features
    linear_output = layer(test_input)
    activated_output = relu(linear_output)

    assert activated_output.shape == linear_output.shape, "Activation should preserve shape"
    assert np.all(activated_output.data >= 0), "ReLU should produce non-negative outputs"
    print(f"✅ Layer + Activation: {test_input.shape} → Dense → ReLU → {activated_output.shape}")

    # Test 6: Multiple layer types
    print("🏗️ Testing different layer configurations...")
    small_layer = Linear(5, 3)
    large_layer = Linear(100, 50)

    small_test = Tensor(np.random.randn(2, 5))
    large_test = Tensor(np.random.randn(1, 100))

    small_output = small_layer(small_test)
    large_output = large_layer(large_test)

    assert small_output.shape == (2, 3), f"Small layer output should be (2, 3), got {small_output.shape}"
    assert large_output.shape == (1, 50), f"Large layer output should be (1, 50), got {large_output.shape}"
    print(f"✅ Flexible architectures: small{small_output.shape}, large{large_output.shape}")

    # Test 7: Parameter count calculation
    print("📊 Testing parameter counting...")
    param_count = layer.weight.data.size + layer.bias.data.size
    expected_count = 10 * 5 + 5  # weights + bias = 55

    assert param_count == expected_count, f"Parameter count should be {expected_count}, got {param_count}"
    print(f"✅ Parameter counting: {param_count} learnable parameters")

    print("\n🎉 Components Complete!")
    print("📝 You can now build the fundamental building blocks of neural networks")
    print("🔧 Built capabilities: Dense layers, learnable parameters, forward pass, batch processing")
    print("🧠 Breakthrough: You have the basic components that can learn from data!")
    print("🎯 Next: Compose components into complete networks")

if __name__ == "__main__":
    test_checkpoint_03_components()
