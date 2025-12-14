"""
Checkpoint 8: Differentiation (After Module 9 - Autograd)
Question: "Can I automatically compute gradients for learning?"
"""

import numpy as np
import pytest

def test_checkpoint_08_differentiation():
    """
    Checkpoint 8: Differentiation

    Validates that students can automatically compute gradients through
    computational graphs - the foundation that makes neural network learning
    possible and practical.
    """
    print("\n∇ Checkpoint 8: Differentiation")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU, Sigmoid
        from tinytorch.core.losses import MeanSquaredError
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-9 first: {e}")

    # Test 1: Basic gradient computation
    print("📐 Testing basic gradient computation...")

    # Create tensor that requires gradients
    x = Tensor([[2.0, 3.0]], requires_grad=True)

    # Simple computation: y = x^2 + 2x + 1
    y = x * x + 2 * x + 1

    # Compute gradients
    y.backward()

    # Check that gradients were computed
    assert hasattr(x, 'grad'), "Tensor should have gradient after backward()"
    assert x.grad is not None, "Gradient should not be None"

    # Expected gradient: dy/dx = 2x + 2 = [6, 8] for x = [2, 3]
    expected_grad = np.array([[6.0, 8.0]])
    assert np.allclose(x.grad.data, expected_grad, atol=1e-5), f"Expected gradient {expected_grad}, got {x.grad.data}"
    print(f"✅ Basic gradients: y = x² + 2x + 1 → dy/dx = {x.grad.data}")

    # Test 2: Neural network gradient computation
    print("🧠 Testing neural network gradients...")

    # Create simple network
    layer = Linear(input_size=2, output_size=1)
    activation = Sigmoid()
    loss_fn = MeanSquaredError()

    # Set network to require gradients
    layer.weight.requires_grad = True
    layer.bias.requires_grad = True

    # Forward pass
    input_data = Tensor([[1.0, 2.0]], requires_grad=True)
    target = Tensor([[0.5]])

    hidden = layer(input_data)
    output = activation(hidden)
    loss = loss_fn(output, target)

    # Backward pass
    loss.backward()

    # Check that all parameters have gradients
    assert layer.weight.grad is not None, "Weights should have gradients"
    assert layer.bias.grad is not None, "Bias should have gradients"
    assert input_data.grad is not None, "Input should have gradients"

    print(f"✅ Network gradients: weights{layer.weight.grad.shape}, bias{layer.bias.grad.shape}, input{input_data.grad.shape}")

    # Test 3: Chain rule verification
    print("🔗 Testing chain rule...")

    # Multi-layer computation to test chain rule
    x = Tensor([[1.0]], requires_grad=True)

    # z = (x * 2)^2 = 4x^2, dz/dx = 8x = 8 for x=1
    intermediate = x * 2  # u = 2x, du/dx = 2
    z = intermediate * intermediate  # z = u^2, dz/du = 2u = 4x

    z.backward()

    expected_chain_grad = 8.0  # dz/dx = dz/du * du/dx = 4x * 2 = 8x = 8
    assert np.allclose(x.grad.data, expected_chain_grad, atol=1e-5), f"Chain rule: expected {expected_chain_grad}, got {x.grad.data}"
    print(f"✅ Chain rule: z = (2x)² → dz/dx = {x.grad.data[0, 0]}")

    # Test 4: Multi-layer network gradients
    print("🏗️ Testing multi-layer network gradients...")

    # Build deeper network
    layer1 = Linear(3, 5)
    layer2 = Linear(5, 2)
    layer3 = Linear(2, 1)
    relu = ReLU()

    # Enable gradients for all parameters
    for layer in [layer1, layer2, layer3]:
        layer.weight.requires_grad = True
        layer.bias.requires_grad = True

    # Forward and backward pass
    batch_input = Tensor(np.random.randn(2, 3), requires_grad=True)
    batch_target = Tensor(np.random.randn(2, 1))

    h1 = relu(layer1(batch_input))
    h2 = relu(layer2(h1))
    prediction = layer3(h2)

    batch_loss = loss_fn(prediction, batch_target)
    batch_loss.backward()

    # Verify all layers have gradients
    gradient_shapes = []
    for i, layer in enumerate([layer1, layer2, layer3], 1):
        assert layer.weight.grad is not None, f"Layer {i} weights should have gradients"
        assert layer.bias.grad is not None, f"Layer {i} bias should have gradients"
        gradient_shapes.append(f"L{i}_w{layer.weight.grad.shape}")

    print(f"✅ Multi-layer gradients: {', '.join(gradient_shapes)}")

    # Test 5: Gradient accumulation
    print("📈 Testing gradient accumulation...")

    # Create parameter for accumulation test
    param = Tensor([[1.0, 2.0]], requires_grad=True)

    # First computation
    loss1 = (param * 2).sum()
    loss1.backward()
    first_grad = param.grad.data.copy()

    # Second computation (without zeroing gradients)
    loss2 = (param * 3).sum()
    loss2.backward()
    accumulated_grad = param.grad.data

    # Gradients should accumulate: grad = 2 + 3 = 5 for each element
    expected_accumulated = first_grad + np.array([[3.0, 3.0]])
    assert np.allclose(accumulated_grad, expected_accumulated), f"Gradients should accumulate: {accumulated_grad} vs {expected_accumulated}"
    print(f"✅ Gradient accumulation: {first_grad} + [3, 3] = {accumulated_grad}")

    # Test 6: Gradient zeroing
    print("🔄 Testing gradient zeroing...")

    # Zero gradients and recompute
    if hasattr(param, 'zero_grad'):
        param.zero_grad()
    else:
        param.grad = None

    loss3 = (param * 4).sum()
    loss3.backward()
    zeroed_grad = param.grad.data

    expected_fresh = np.array([[4.0, 4.0]])
    assert np.allclose(zeroed_grad, expected_fresh), f"Zeroed gradients should be fresh: {zeroed_grad} vs {expected_fresh}"
    print(f"✅ Gradient zeroing: fresh computation → {zeroed_grad}")

    # Test 7: Computational graph complexity
    print("🕸️ Testing complex computational graph...")

    # Complex computation with multiple paths
    a = Tensor([[2.0]], requires_grad=True)
    b = Tensor([[3.0]], requires_grad=True)

    # Multiple paths: c = a*b + a^2 + b^2
    path1 = a * b      # ab, da = b, db = a
    path2 = a * a      # a^2, da = 2a
    path3 = b * b      # b^2, db = 2b

    c = path1 + path2 + path3
    c.backward()

    # Expected gradients:
    # dc/da = b + 2a = 3 + 4 = 7
    # dc/db = a + 2b = 2 + 6 = 8
    expected_a_grad = 7.0
    expected_b_grad = 8.0

    assert np.allclose(a.grad.data, expected_a_grad), f"Complex graph grad_a: expected {expected_a_grad}, got {a.grad.data}"
    assert np.allclose(b.grad.data, expected_b_grad), f"Complex graph grad_b: expected {expected_b_grad}, got {b.grad.data}"
    print(f"✅ Complex graph: c = ab + a² + b² → da = {a.grad.data[0,0]}, db = {b.grad.data[0,0]}")

    # Test 8: Memory efficiency
    print("💾 Testing gradient computation efficiency...")

    # Test that intermediate computations don't leak memory
    large_param = Tensor(np.random.randn(100, 100), requires_grad=True)

    # Multiple forward-backward cycles
    for i in range(3):
        output = (large_param * (i + 1)).sum()
        output.backward()

        # Check gradient exists and has correct shape
        assert large_param.grad is not None, f"Gradient should exist in cycle {i}"
        assert large_param.grad.shape == large_param.shape, f"Gradient shape should match parameter shape"

        # Zero gradients for next iteration
        if hasattr(large_param, 'zero_grad'):
            large_param.zero_grad()
        else:
            large_param.grad = None

    print(f"✅ Memory efficiency: multiple cycles on {large_param.shape} tensor")

    print("\n🎉 Differentiation Complete!")
    print("📝 You can now automatically compute gradients for learning")
    print("🔧 Built capabilities: Autograd, chain rule, gradient accumulation, complex graphs")
    print("🧠 Breakthrough: You have the foundation for all neural network learning!")
    print("🎯 Next: Build optimizers to update parameters using gradients")

if __name__ == "__main__":
    test_checkpoint_08_differentiation()
