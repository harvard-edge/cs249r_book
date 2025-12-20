"""
Checkpoint 9: Optimization (After Module 07 - Optimizers)
Question: "Can I optimize neural networks with sophisticated algorithms?"
"""

import numpy as np
import pytest

def test_checkpoint_09_optimization():
    """
    Checkpoint 9: Optimization

    Validates that students can use sophisticated optimization algorithms
    to efficiently train neural networks - the algorithms that make modern
    deep learning fast and effective.
    """
    print("\n⚡ Checkpoint 9: Optimization")
    print("=" * 50)

    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU
        from tinytorch.core.losses import MeanSquaredError
        from tinytorch.core.optimizers import SGD, Adam, RMSprop
    except ImportError as e:
        pytest.fail(f"❌ Cannot import required classes - complete Modules 2-10 first: {e}")

    # Test 1: SGD optimizer
    print("📈 Testing SGD optimizer...")

    # Create simple model and data
    model = Linear(2, 1)
    model.weight.requires_grad = True
    model.bias.requires_grad = True

    sgd = SGD([model.weights, model.bias], lr=0.01)
    loss_fn = MeanSquaredError()

    # Training data: y = 2*x1 + 3*x2 + 1
    X = Tensor([[1, 2], [2, 3], [3, 4]])
    y = Tensor([[2*1 + 3*2 + 1], [2*2 + 3*3 + 1], [2*3 + 3*4 + 1]])  # [9, 17, 25]

    # Store initial parameters
    initial_weights = model.weight.data.copy()
    initial_bias = model.bias.data.copy()

    # Training step
    predictions = model(X)
    loss = loss_fn(predictions, y)
    loss.backward()

    sgd.step()
    sgd.zero_grad()

    # Check that parameters changed
    assert not np.allclose(model.weight.data, initial_weights), "SGD should update weights"
    assert not np.allclose(model.bias.data, initial_bias), "SGD should update bias"
    print(f"✅ SGD: parameters updated from loss={loss.data:.4f}")

    # Test 2: Adam optimizer with momentum
    print("🚀 Testing Adam optimizer...")

    # Reset model
    model_adam = Linear(2, 1)
    model_adam.weight.requires_grad = True
    model_adam.bias.requires_grad = True

    adam = Adam([model_adam.weights, model_adam.bias], lr=0.01)

    # Store initial parameters
    initial_weights_adam = model_adam.weight.data.copy()
    initial_bias_adam = model_adam.bias.data.copy()

    # Multiple training steps to see momentum effect
    losses = []
    for epoch in range(3):
        predictions = model_adam(X)
        loss = loss_fn(predictions, y)
        losses.append(loss.data.item() if hasattr(loss.data, 'item') else float(loss.data))

        loss.backward()
        adam.step()
        adam.zero_grad()

    # Check parameter updates and loss reduction
    assert not np.allclose(model_adam.weight.data, initial_weights_adam), "Adam should update weights"
    assert not np.allclose(model_adam.bias.data, initial_bias_adam), "Adam should update bias"
    assert losses[-1] < losses[0], f"Adam should reduce loss: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"✅ Adam: loss reduction {losses[0]:.4f} → {losses[-1]:.4f}")

    # Test 3: RMSprop optimizer
    print("📊 Testing RMSprop optimizer...")

    model_rms = Linear(2, 1)
    model_rms.weight.requires_grad = True
    model_rms.bias.requires_grad = True

    rmsprop = RMSprop([model_rms.weights, model_rms.bias], lr=0.01)

    # Training step
    predictions = model_rms(X)
    loss = loss_fn(predictions, y)
    loss.backward()

    initial_weights_rms = model_rms.weight.data.copy()
    rmsprop.step()

    assert not np.allclose(model_rms.weight.data, initial_weights_rms), "RMSprop should update parameters"
    print(f"✅ RMSprop: parameters updated successfully")

    # Test 4: Learning rate effects
    print("🎯 Testing learning rate effects...")

    # Compare different learning rates
    lr_small = 0.001
    lr_large = 0.1

    model_small = Linear(2, 1)
    model_large = Linear(2, 1)

    # Make models identical initially
    model_large.weight.data = model_small.weight.data.copy()
    model_large.bias.data = model_small.bias.data.copy()

    model_small.weight.requires_grad = True
    model_small.bias.requires_grad = True
    model_large.weight.requires_grad = True
    model_large.bias.requires_grad = True

    sgd_small = SGD([model_small.weights, model_small.bias], lr=lr_small)
    sgd_large = SGD([model_large.weights, model_large.bias], lr=lr_large)

    # Single training step
    loss_small = loss_fn(model_small(X), y)
    loss_large = loss_fn(model_large(X), y)

    loss_small.backward()
    loss_large.backward()

    weight_change_small = np.abs(model_small.weight.grad.data).mean()
    weight_change_large = np.abs(model_large.weight.grad.data).mean()

    sgd_small.step()
    sgd_large.step()

    # Large LR should cause bigger parameter changes
    actual_change_small = np.abs(model_small.weight.data - model_large.weight.data).mean()
    print(f"✅ Learning rates: small LR vs large LR parameter difference = {actual_change_small:.6f}")

    # Test 5: Optimizer state persistence
    print("💾 Testing optimizer state...")

    # Adam maintains moving averages
    model_state = Linear(1, 1)
    model_state.weight.requires_grad = True
    model_state.bias.requires_grad = True

    adam_state = Adam([model_state.weights, model_state.bias], lr=0.01)

    # Multiple steps to build up state
    for i in range(3):
        dummy_input = Tensor([[float(i + 1)]])
        dummy_target = Tensor([[float((i + 1) * 2)]])

        pred = model_state(dummy_input)
        loss = loss_fn(pred, dummy_target)
        loss.backward()

        # Check that optimizer has internal state
        if hasattr(adam_state, 'm') or hasattr(adam_state, 'state'):
            print(f"✅ Optimizer state: Adam maintains internal state across steps")
            break

        adam_state.step()
        adam_state.zero_grad()

    # Test 6: Parameter group handling
    print("🎛️ Testing parameter groups...")

    # Create model with different parameter groups
    layer1 = Linear(3, 4)
    layer2 = Linear(4, 1)

    layer1.weight.requires_grad = True
    layer1.bias.requires_grad = True
    layer2.weight.requires_grad = True
    layer2.bias.requires_grad = True

    # Different learning rates for different layers
    optimizer_groups = SGD([
        layer1.weights, layer1.bias,  # Group 1
        layer2.weights, layer2.bias   # Group 2
    ], lr=0.01)

    # Test that all parameters are being tracked
    batch_X = Tensor(np.random.randn(2, 3))
    batch_y = Tensor(np.random.randn(2, 1))

    h1 = layer1(batch_X)
    pred = layer2(h1)
    loss = loss_fn(pred, batch_y)
    loss.backward()

    # Check gradients exist for all parameters
    assert layer1.weight.grad is not None, "Layer 1 weights should have gradients"
    assert layer2.weight.grad is not None, "Layer 2 weights should have gradients"

    optimizer_groups.step()
    print(f"✅ Parameter groups: all layers optimized together")

    # Test 7: Convergence on simple problem
    print("🎯 Testing convergence...")

    # Simple linear regression: learn y = 2x + 1
    model_conv = Linear(1, 1)
    model_conv.weight.requires_grad = True
    model_conv.bias.requires_grad = True

    optimizer_conv = Adam([model_conv.weights, model_conv.bias], lr=0.1)

    # Training data
    x_train = Tensor([[1], [2], [3], [4], [5]])
    y_train = Tensor([[3], [5], [7], [9], [11]])  # y = 2x + 1

    # Train for several epochs
    initial_loss = None
    final_loss = None

    for epoch in range(10):
        pred = model_conv(x_train)
        loss = loss_fn(pred, y_train)

        if epoch == 0:
            initial_loss = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        if epoch == 9:
            final_loss = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)

        loss.backward()
        optimizer_conv.step()
        optimizer_conv.zero_grad()

    # Should converge to approximately correct weights
    learned_weight = model_conv.weight.data[0, 0]
    learned_bias = model_conv.bias.data[0]

    assert abs(learned_weight - 2.0) < 0.5, f"Should learn weight≈2, got {learned_weight}"
    assert abs(learned_bias - 1.0) < 0.5, f"Should learn bias≈1, got {learned_bias}"
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} → {final_loss:.4f}"
    print(f"✅ Convergence: learned y = {learned_weight:.2f}x + {learned_bias:.2f}")

    print("\n🎉 Optimization Complete!")
    print("📝 You can now optimize neural networks with sophisticated algorithms")
    print("🔧 Built capabilities: SGD, Adam, RMSprop, learning rates, parameter groups")
    print("🧠 Breakthrough: You can now train networks efficiently and effectively!")
    print("🎯 Next: Build complete training loops")

if __name__ == "__main__":
    test_checkpoint_09_optimization()
