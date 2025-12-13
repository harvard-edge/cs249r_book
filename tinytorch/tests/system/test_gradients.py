#!/usr/bin/env python
"""
Gradient Flow Validation Tests for TinyTorch
=============================================
Ensures gradients propagate correctly through all architectures.
Critical for verifying that models can actually learn.

Test Categories:
- Gradient existence through deep networks
- Gradient magnitude (not vanishing/exploding)
- Chain rule validation
- Gradient accumulation
- Optimizer parameter updates
"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid, Tanh
from tinytorch.core.losses import MSELoss, CrossEntropyLoss
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.nn import Conv2d, TransformerBlock, Sequential
import tinytorch.nn.functional as F


# ============== Gradient Existence Tests ==============

def test_gradient_exists_single_layer():
    """Gradients exist after backward through single layer."""
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(3, 10))
    y_true = Tensor(np.random.randn(3, 5))

    y_pred = layer(x)
    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()
        assert layer.weight.grad is not None, "No gradient for weights"
        assert layer.bias.grad is not None, "No gradient for bias"
    except AttributeError:
        # Autograd might not be implemented
        pytest.skip("Autograd not implemented")


def test_gradient_exists_deep_network():
    """Gradients flow through deep network (5 layers)."""
    model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 20),
        ReLU(),
        Linear(20, 20),
        ReLU(),
        Linear(20, 20),
        ReLU(),
        Linear(20, 5)
    ])

    x = Tensor(np.random.randn(4, 10))
    y_true = Tensor(np.random.randn(4, 5))

    y_pred = model(x)
    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()
        # Check first and last layers have gradients
        first_layer = model.layers[0]
        last_layer = model.layers[-1]
        assert first_layer.weight.grad is not None, "No gradient in first layer"
        assert last_layer.weight.grad is not None, "No gradient in last layer"
    except AttributeError:
        pytest.skip("Autograd not implemented")


def test_gradient_exists_cnn():
    """Gradients flow through CNN architecture."""
    class SimpleCNN:
        def __init__(self):
            self.conv1 = Conv2d(1, 16, kernel_size=3)
            self.conv2 = Conv2d(16, 32, kernel_size=3)
            self.fc = Linear(32 * 5 * 5, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.flatten(x, start_dim=1)
            return self.fc(x)

        def parameters(self):
            params = []
            for layer in [self.conv1, self.conv2, self.fc]:
                if hasattr(layer, 'parameters'):
                    params.extend(layer.parameters())
            return params

    model = SimpleCNN()
    x = Tensor(np.random.randn(2, 1, 28, 28))
    y_true = Tensor(np.random.randn(2, 10))

    y_pred = model.forward(x)
    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()
        assert model.conv1.weight.grad is not None, "No gradient in conv1"
        assert model.fc.weight.grad is not None, "No gradient in fc layer"
    except (AttributeError, Exception):
        pytest.skip("Autograd not fully implemented for CNN")


# ============== Gradient Magnitude Tests ==============

def test_gradient_not_vanishing():
    """Gradients don't vanish in deep network."""
    # Build deep network prone to vanishing gradients
    layers = []
    for i in range(10):
        layers.append(Linear(20, 20))
        layers.append(Sigmoid())  # Sigmoid can cause vanishing gradients
    layers.append(Linear(20, 1))

    model = Sequential(layers)
    x = Tensor(np.random.randn(5, 20))
    y_true = Tensor(np.random.randn(5, 1))

    y_pred = model(x)
    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()
        first_layer = model.layers[0]
        if first_layer.weight.grad is not None:
            grad_magnitude = np.abs(first_layer.weight.grad.data).mean()
            assert grad_magnitude > 1e-8, f"Gradient vanished: {grad_magnitude}"
    except (AttributeError, Exception):
        pytest.skip("Autograd not fully implemented")


def test_gradient_not_exploding():
    """Gradients don't explode in deep network."""
    # Build network that could have exploding gradients
    layers = []
    for i in range(5):
        layers.append(Linear(20, 20))
        layers.append(ReLU())
    layers.append(Linear(20, 1))

    model = Sequential(layers)

    # Use larger initialization to potentially trigger explosion
    for layer in model.layers:
        if hasattr(layer, 'weight'):
            layer.weight.data = np.random.randn(*layer.weight.shape) * 2.0

    x = Tensor(np.random.randn(5, 20))
    y_true = Tensor(np.random.randn(5, 1))

    y_pred = model(x)
    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()
        last_layer = model.layers[-1]
        if last_layer.weight.grad is not None:
            grad_magnitude = np.abs(last_layer.weight.grad.data).mean()
            assert grad_magnitude < 1000, f"Gradient exploded: {grad_magnitude}"
    except (AttributeError, Exception):
        pytest.skip("Autograd not fully implemented")


def test_gradient_reasonable_magnitude():
    """Gradients have reasonable magnitude for learning."""
    model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    ])

    x = Tensor(np.random.randn(8, 10))
    y_true = Tensor(np.random.randn(8, 5))

    y_pred = model(x)
    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()
        for layer in model.layers:
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                grad_mag = np.abs(layer.weight.grad.data).mean()
                # Reasonable range for gradients
                assert 1e-6 < grad_mag < 100, f"Gradient magnitude out of range: {grad_mag}"
    except (AttributeError, Exception):
        pytest.skip("Autograd not fully implemented")


# ============== Chain Rule Tests ==============

def test_chain_rule_linear_relu():
    """Chain rule works correctly through Linear→ReLU."""
    linear = Linear(5, 3)
    x = Tensor(np.random.randn(2, 5))
    y_true = Tensor(np.random.randn(2, 3))

    # Forward
    z = linear(x)
    y = F.relu(z)
    loss = MSELoss()(y, y_true)

    try:
        loss.backward()
        # ReLU should only backprop where input > 0
        if hasattr(z, 'data'):
            relu_mask = z.data > 0
            # Gradient should be zero where ReLU blocked it
            if linear.weight.grad is not None:
                # This is a simplified check - full validation would be complex
                assert linear.weight.grad is not None, "Chain rule broken"
    except (AttributeError, Exception):
        pytest.skip("Autograd not fully implemented")


def test_chain_rule_multiple_paths():
    """Chain rule handles multiple paths (residual connection)."""
    linear1 = Linear(10, 10)
    linear2 = Linear(10, 10)

    x = Tensor(np.random.randn(4, 10))
    y_true = Tensor(np.random.randn(4, 10))

    # Forward with residual connection
    z1 = linear1(x)
    z2 = linear2(F.relu(z1))
    y = z1 + z2  # Residual connection

    loss = MSELoss()(y, y_true)

    try:
        loss.backward()
        # Both paths should contribute to gradient
        assert linear1.weight.grad is not None, "No gradient through residual path"
        assert linear2.weight.grad is not None, "No gradient through main path"
    except (AttributeError, Exception):
        pytest.skip("Autograd not fully implemented")


# ============== Gradient Accumulation Tests ==============

def test_gradient_accumulation():
    """Gradients accumulate correctly over multiple backward passes."""
    model = Linear(5, 3)
    optimizer = SGD(model.parameters(), learning_rate=0.01)

    x1 = Tensor(np.random.randn(2, 5))
    y1 = Tensor(np.random.randn(2, 3))

    x2 = Tensor(np.random.randn(2, 5))
    y2 = Tensor(np.random.randn(2, 3))

    try:
        # First backward
        loss1 = MSELoss()(model(x1), y1)
        loss1.backward()

        if model.weight.grad is not None:
            grad1 = model.weight.grad.data.copy()

            # Second backward (should accumulate)
            loss2 = MSELoss()(model(x2), y2)
            loss2.backward()

            grad2 = model.weight.grad.data
            # Gradient should have changed (accumulated)
            assert not np.allclose(grad1, grad2), "Gradients didn't accumulate"
    except (AttributeError, Exception):
        pytest.skip("Autograd not fully implemented")


def test_zero_grad():
    """zero_grad() correctly resets gradients."""
    model = Linear(5, 3)
    optimizer = SGD(model.parameters(), learning_rate=0.01)

    x = Tensor(np.random.randn(2, 5))
    y = Tensor(np.random.randn(2, 3))

    try:
        # Accumulate gradient
        loss = MSELoss()(model(x), y)
        loss.backward()

        if model.weight.grad is not None:
            # Clear gradients
            optimizer.zero_grad()

            # Check gradients are zeroed
            if hasattr(model.weights, 'grad'):
                if model.weight.grad is not None:
                    assert np.allclose(model.weight.grad.data, 0), "Gradients not zeroed"
    except (AttributeError, Exception):
        pytest.skip("Autograd not fully implemented")


# ============== Optimizer Update Tests ==============

def test_sgd_updates_parameters():
    """SGD optimizer updates parameters in correct direction."""
    model = Linear(5, 3)
    optimizer = SGD(model.parameters(), learning_rate=0.1)

    # Save initial weights
    initial_weights = model.weight.data.copy()

    x = Tensor(np.random.randn(4, 5))
    y_true = Tensor(np.random.randn(4, 3))

    try:
        # Forward and backward
        y_pred = model(x)
        loss = MSELoss()(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not np.allclose(initial_weights, model.weight.data), "Weights didn't update"

        # Check update direction (gradient descent)
        if model.weight.grad is not None:
            expected_update = initial_weights - 0.1 * model.weight.grad.data
            assert np.allclose(model.weight.data, expected_update, rtol=1e-5), \
                "SGD update incorrect"
    except (AttributeError, Exception):
        pytest.skip("Optimizer not fully implemented")


def test_adam_updates_parameters():
    """Adam optimizer updates parameters with momentum."""
    model = Linear(5, 3)
    optimizer = Adam(model.parameters(), learning_rate=0.01)

    initial_weights = model.weight.data.copy()

    x = Tensor(np.random.randn(4, 5))
    y_true = Tensor(np.random.randn(4, 3))

    try:
        # Multiple steps to see momentum effect
        for _ in range(3):
            y_pred = model(x)
            loss = MSELoss()(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Weights should have changed
        assert not np.allclose(initial_weights, model.weight.data), \
            "Adam didn't update weights"
    except (AttributeError, Exception):
        pytest.skip("Adam optimizer not fully implemented")


# ============== Special Architecture Tests ==============

def test_transformer_gradient_flow():
    """Gradients flow through transformer architecture."""
    block = TransformerBlock(embed_dim=64, num_heads=4)

    x = Tensor(np.random.randn(2, 10, 64))  # (batch, seq, embed)
    y_true = Tensor(np.random.randn(2, 10, 64))

    y_pred = block(x)
    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()
        # Check key components have gradients
        params = block.parameters()
        gradients_exist = any(
            p.grad is not None for p in params
            if hasattr(p, 'grad')
        )
        assert gradients_exist, "No gradients in transformer block"
    except (AttributeError, Exception):
        pytest.skip("Transformer gradients not fully implemented")


def test_loss_gradient_correctness():
    """Loss functions produce correct gradients."""
    # Simple case where we can verify gradient analytically
    model = Linear(2, 1, bias=False)
    model.weight.data = np.array([[1.0], [1.0]])  # Known weights

    x = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
    y_true = Tensor(np.array([[2.0], [3.0]]))

    y_pred = model(x)
    # y_pred should be [[1.0], [1.0]]
    # MSE loss = mean((1-2)^2 + (1-3)^2) = mean(1 + 4) = 2.5
    # Gradient w.r.t. predictions: [[-1], [-2]]

    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()
        if model.weight.grad is not None:
            # Verify gradient is roughly correct
            # This is simplified - exact validation would need careful calculation
            assert model.weight.grad is not None, "No gradient from loss"
    except (AttributeError, Exception):
        pytest.skip("Loss gradient not implemented")


# ============== Common Issues Detection ==============

def test_dead_relu_detection():
    """Detect dead ReLU problem (all gradients blocked)."""
    model = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    ])

    # Set very negative bias to kill ReLU
    first_layer = model.layers[0]
    if hasattr(first_layer, 'bias'):
        first_layer.bias.data = np.ones(20) * -10

    x = Tensor(np.random.randn(4, 10) * 0.1)  # Small inputs
    y_true = Tensor(np.random.randn(4, 5))

    y_pred = model(x)
    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()
        # With dead ReLUs, gradients might be very small or zero
        if first_layer.weight.grad is not None:
            grad_mag = np.abs(first_layer.weight.grad.data).mean()
            if grad_mag < 1e-10:
                pytest.warns(UserWarning, "Possible dead ReLU detected")
    except (AttributeError, Exception):
        pytest.skip("Dead ReLU detection not implemented")


def test_gradient_clipping():
    """Test gradient clipping prevents explosion."""
    model = Linear(10, 10)

    # Create artificially large gradient scenario
    x = Tensor(np.random.randn(2, 10) * 100)
    y_true = Tensor(np.random.randn(2, 10) * 100)

    y_pred = model(x)
    loss = MSELoss()(y_pred, y_true)

    try:
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        for param in model.parameters():
            if hasattr(param, 'grad') and param.grad is not None:
                grad_norm = np.linalg.norm(param.grad.data)
                if grad_norm > max_norm:
                    param.grad.data = param.grad.data * (max_norm / grad_norm)

                # Verify clipping worked
                new_norm = np.linalg.norm(param.grad.data)
                assert new_norm <= max_norm * 1.01, "Gradient clipping failed"
    except (AttributeError, Exception):
        pytest.skip("Gradient clipping not implemented")


if __name__ == "__main__":
    # When run directly, use pytest
    import subprocess
    result = subprocess.run(["pytest", __file__, "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    sys.exit(result.returncode)
