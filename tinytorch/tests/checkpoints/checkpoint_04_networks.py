"""
Checkpoint 4: Networks (After Module 5 - Dense)
Question: "Can I build complete multi-layer neural networks?"
"""

import numpy as np
import pytest

def test_checkpoint_04_networks():
    """
    Checkpoint 4: Networks

    Validates that students can combine layers into complete multi-layer
    perceptrons - the first step toward building real neural networks that
    can solve complex problems.
    """
    print("\n🔗 Checkpoint 4: Networks")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU, Sigmoid
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-5 first: {e}")

    # Test 1: Simple 2-layer network
    print("🏗️ Testing 2-layer network construction...")
    input_layer = Linear(input_size=4, output_size=8)
    output_layer = Linear(input_size=8, output_size=3)
    activation = ReLU()

    # Test network architecture
    sample_input = Tensor(np.random.randn(2, 4))  # 2 samples, 4 features

    # Forward pass through network
    hidden = input_layer(sample_input)
    hidden_activated = activation(hidden)
    output = output_layer(hidden_activated)

    assert output.shape == (2, 3), f"Network output should be (2, 3), got {output.shape}"
    print(f"✅ 2-layer network: {sample_input.shape} → {hidden.shape} → {output.shape}")

    # Test 2: Deep network (3+ layers)
    print("🏢 Testing deep network construction...")
    layer1 = Linear(10, 16)
    layer2 = Linear(16, 8)
    layer3 = Linear(8, 4)
    layer4 = Linear(4, 1)
    relu = ReLU()
    sigmoid = Sigmoid()

    # Build a classifier network
    x = Tensor(np.random.randn(1, 10))

    # Deep forward pass
    h1 = relu(layer1(x))
    h2 = relu(layer2(h1))
    h3 = relu(layer3(h2))
    prediction = sigmoid(layer4(h3))

    assert prediction.shape == (1, 1), f"Prediction shape should be (1, 1), got {prediction.shape}"
    assert 0 <= prediction.data[0, 0] <= 1, "Sigmoid output should be between 0 and 1"
    print(f"✅ Deep network: {x.shape} → 16 → 8 → 4 → {prediction.shape}")

    # Test 3: Network with different architectures
    print("🔧 Testing flexible architectures...")

    # Wide network
    wide_net = [
        Linear(5, 50),
        ReLU(),
        Linear(50, 50),
        ReLU(),
        Linear(50, 10)
    ]

    # Narrow network
    narrow_net = [
        Linear(20, 10),
        ReLU(),
        Linear(10, 5),
        ReLU(),
        Linear(5, 2)
    ]

    # Test both architectures
    wide_input = Tensor(np.random.randn(1, 5))
    narrow_input = Tensor(np.random.randn(1, 20))

    # Wide network forward pass
    wide_x = wide_input
    for layer in wide_net:
        wide_x = layer(wide_x)

    # Narrow network forward pass
    narrow_x = narrow_input
    for layer in narrow_net:
        narrow_x = layer(narrow_x)

    assert wide_x.shape == (1, 10), f"Wide network output should be (1, 10), got {wide_x.shape}"
    assert narrow_x.shape == (1, 2), f"Narrow network output should be (1, 2), got {narrow_x.shape}"
    print(f"✅ Flexible architectures: wide{wide_x.shape}, narrow{narrow_x.shape}")

    # Test 4: Parameter counting across network
    print("📊 Testing network parameter counting...")
    total_params = (
        input_layer.weight.data.size + input_layer.bias.data.size +
        output_layer.weight.data.size + output_layer.bias.data.size
    )

    expected_params = (4*8 + 8) + (8*3 + 3)  # (weights + bias) for each layer
    assert total_params == expected_params, f"Total parameters should be {expected_params}, got {total_params}"
    print(f"✅ Network parameters: {total_params} learnable parameters")

    # Test 5: Batch processing through network
    print("📦 Testing batch processing...")
    batch_input = Tensor(np.random.randn(5, 4))  # 5 samples

    batch_hidden = input_layer(batch_input)
    batch_hidden_activated = activation(batch_hidden)
    batch_output = output_layer(batch_hidden_activated)

    assert batch_output.shape == (5, 3), f"Batch output should be (5, 3), got {batch_output.shape}"
    print(f"✅ Batch processing: {batch_input.shape} → network → {batch_output.shape}")

    # Test 6: Universal approximation demonstration
    print("🎯 Testing nonlinear function approximation...")

    # Create a simple nonlinear function to approximate: f(x) = x^2
    def target_function(x):
        return x * x

    # Generate training data
    x_data = np.linspace(-2, 2, 10).reshape(-1, 1)
    y_target = target_function(x_data)

    # Simple approximator network
    approx_net = [
        Linear(1, 5),
        ReLU(),
        Linear(5, 5),
        ReLU(),
        Linear(5, 1)
    ]

    # Test that network can process the data
    x_tensor = Tensor(x_data)
    net_output = x_tensor
    for layer in approx_net:
        net_output = layer(net_output)

    assert net_output.shape == y_target.shape, f"Approximator output shape mismatch: {net_output.shape} vs {y_target.shape}"
    print(f"✅ Function approximation setup: {x_tensor.shape} → network → {net_output.shape}")

    print("\n🎉 Networks Complete!")
    print("📝 You can now build complete multi-layer neural networks")
    print("🔧 Built capabilities: Multi-layer perceptrons, deep networks, flexible architectures")
    print("🧠 Breakthrough: You have complete networks that can learn complex patterns!")
    print("🎯 Next: Add automatic differentiation for learning")

if __name__ == "__main__":
    test_checkpoint_04_networks()
