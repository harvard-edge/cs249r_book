"""
Integration tests for TinyTorch optimizers with other modules.

Tests that optimizers correctly integrate with:
- Module 01: Tensor operations
- Module 02: Activation functions
- Module 03: Layers (Linear, Sequential)
- Module 06: Autograd (Tensor with gradients)
- Module 04: Losses (MSE, CrossEntropy)
"""

import sys
import os
import numpy as np
import pytest

# Import from tinytorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid, Softmax, Tanh
from tinytorch.core.layers import Linear, Layer, Dropout
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from tinytorch.core.optimizers import SGD, Adam, AdamW

# Enable autograd
enable_autograd()


def test_sgd_with_linear_layer():
    """Test SGD optimizer with Linear layer and autograd."""
    print("🔬 Integration Test: SGD + Linear Layer + Autograd")

    # Create a simple linear layer
    layer = Linear(3, 2)

    # Create optimizer with layer parameters
    parameters = layer.parameters()
    sgd = SGD(parameters, lr=0.1)

    # Forward pass
    x = Tensor(np.random.randn(1, 3), requires_grad=False)
    y = layer(x)

    # Create a simple loss (sum of outputs)
    loss = y.sum()

    # Backward pass
    loss.backward()

    # Check that gradients exist
    for param in parameters:
        assert param.grad is not None, "Parameter should have gradient after backward"

    # Store original values
    original_values = [param.data.copy() for param in parameters]

    # Optimizer step
    sgd.step()

    # Check parameters were updated
    for orig, param in zip(original_values, parameters):
        assert not np.allclose(orig, param.data), "Parameters should change after optimizer step"

    print("✅ SGD integrates correctly with Linear layers and autograd!")


def test_adam_with_multi_layer_network():
    """Test Adam optimizer with multi-layer network."""
    print("🔬 Integration Test: Adam + Multi-Layer Network")

    # Build a small network (layers manually)
    layer1 = Linear(4, 8)
    relu1 = ReLU()
    layer2 = Linear(8, 4)
    relu2 = ReLU()
    layer3 = Linear(4, 2)

    # Collect all parameters
    params = layer1.parameters() + layer2.parameters() + layer3.parameters()

    # Create Adam optimizer
    adam = Adam(params, lr=0.01)

    # Training loop simulation
    for step in range(3):
        # Forward pass
        x = Tensor(np.random.randn(2, 4), requires_grad=True)
        h1 = relu1(layer1(x))
        h2 = relu2(layer2(h1))
        output = layer3(h2)

        # Simple loss - MSE
        target = Tensor(np.ones((2, 2)))
        diff = output - target
        loss = (diff * diff).sum()

        # Backward pass
        adam.zero_grad()
        loss.backward()

        # Update
        adam.step()

    print("✅ Adam works with multi-layer networks!")


def test_optimizer_with_mse_loss():
    """Test optimizer integration with MSE loss function."""
    print("🔬 Integration Test: Optimizer + MSE Loss")

    layer = Linear(3, 1)
    optimizer = SGD(layer.parameters(), lr=0.01)
    loss_fn = MSELoss()

    # Forward pass
    x = Tensor(np.random.randn(4, 3), requires_grad=True)
    target = Tensor(np.random.randn(4, 1))
    output = layer(x)
    loss = loss_fn(output, target)

    # Backward and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("✅ Optimizer integrates with MSE loss!")


def test_optimizer_with_activations():
    """Test optimizer with activated layers."""
    print("🔬 Integration Test: Optimizer + Activations")

    # Network with various activations
    layer1 = Linear(5, 10)
    relu = ReLU()
    layer2 = Linear(10, 5)
    sigmoid = Sigmoid()

    params = layer1.parameters() + layer2.parameters()
    optimizer = Adam(params, lr=0.001)

    x = Tensor(np.random.randn(3, 5), requires_grad=True)
    h = relu(layer1(x))
    output = sigmoid(layer2(h))

    # Check sigmoid output range
    assert np.all(output.data >= 0) and np.all(output.data <= 1), \
        "Sigmoid should output in [0, 1]"

    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("✅ Optimizer works with activation functions!")


def test_learning_rate_scheduler():
    """Test learning rate scheduler with optimizer."""
    print("🔬 Integration Test: LR Scheduler + Optimizer")

    # Simple parameter
    param = Tensor(np.array([1.0]), requires_grad=True)
    optimizer = SGD([param], lr=0.1)

    # Manually test different learning rates
    initial_lr = optimizer.lr

    # Simulate training with learning rate decay
    for epoch in range(5):
        param.grad = Tensor(np.array([1.0]))
        optimizer.step()
        # Decay learning rate
        optimizer.lr = initial_lr * (0.9 ** epoch)

    assert optimizer.lr < initial_lr, "Learning rate should have decayed"
    print("✅ LR scheduler works with optimizer!")


def test_optimizer_memory_consistency():
    """Test that optimizer maintains consistent memory references."""
    print("🔬 Integration Test: Optimizer Memory Consistency")

    layer = Linear(3, 2)
    params = layer.parameters()
    optimizer = Adam(params, lr=0.01)

    # Store original references
    param_ids = [id(p) for p in params]

    # Do optimization steps
    for _ in range(3):
        x = Tensor(np.random.randn(1, 3))
        output = layer(x)
        loss = output.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check references are same
    new_param_ids = [id(p) for p in layer.parameters()]
    assert param_ids == new_param_ids, "Parameter references should be stable"

    print("✅ Optimizer maintains memory consistency!")


# ============================================================================
# Unit tests for individual components (originally loaded from modules)
# ============================================================================

def test_unit_tensor_creation():
    """Test basic tensor creation."""
    print("🧪 Unit Test: Tensor Creation...")
    t = Tensor(np.array([1, 2, 3]))
    assert t.shape == (3,)
    assert np.array_equal(t.data, np.array([1, 2, 3]))
    print("✅ Tensor creation works!")


def test_unit_shape_manipulation():
    """Test tensor reshape operations."""
    print("🧪 Unit Test: Shape Manipulation...")
    t = Tensor(np.arange(6))

    # Valid reshape
    reshaped = t.reshape(2, 3)
    assert reshaped.shape == (2, 3)

    # Invalid reshape should raise
    try:
        t.reshape(2, 2)  # 6 elements can't fit in 2×2=4
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Total elements must match" in str(e)

    print("✅ Shape manipulation works!")


def test_unit_relu_activation():
    """Test ReLU activation."""
    print("🧪 Unit Test: ReLU Activation...")
    relu = ReLU()
    x = Tensor(np.array([-1, 0, 1, 2]))
    output = relu(x)
    expected = np.array([0, 0, 1, 2])
    assert np.array_equal(output.data, expected)
    print("✅ ReLU activation works!")


def test_unit_sigmoid():
    """Test Sigmoid activation."""
    print("🧪 Unit Test: Sigmoid Activation...")
    sigmoid = Sigmoid()
    x = Tensor(np.array([0.0]))
    output = sigmoid(x)
    assert np.isclose(output.data[0], 0.5, atol=1e-6)
    print("✅ Sigmoid activation works!")


def test_unit_linear_layer():
    """Test Linear layer forward pass."""
    print("🧪 Unit Test: Linear Layer...")
    layer = Linear(3, 2)
    x = Tensor(np.random.randn(1, 3))
    output = layer(x)
    assert output.shape == (1, 2)
    print("✅ Linear layer works!")


def test_edge_cases_linear():
    """Test edge cases for Linear layer."""
    print("🧪 Edge Cases: Linear Layer...")

    # Single sample
    layer = Linear(2, 3)
    x = Tensor(np.array([[1.0, 2.0]]))
    output = layer(x)
    assert output.shape == (1, 3)

    # Batch
    x_batch = Tensor(np.random.randn(5, 2))
    output_batch = layer(x_batch)
    assert output_batch.shape == (5, 3)

    print("✅ Linear layer edge cases pass!")


def test_gradient_preparation_linear():
    """Test that Linear layer gradients are prepared correctly."""
    print("🧪 Unit Test: Linear Gradient Preparation...")
    layer = Linear(2, 2)

    x = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    # Weight should have gradient
    assert layer.weight.grad is not None
    print("✅ Linear gradient preparation works!")


def test_unit_dropout_layer():
    """Test Dropout layer behavior."""
    print("🧪 Unit Test: Dropout Layer...")
    dropout = Dropout(p=0.5)
    x = Tensor(np.ones((10, 10)))

    # During training (default), some values should be zeroed/scaled
    output = dropout(x)

    # Create new input for eval
    x_eval = Tensor(np.ones((10, 10)))

    # During eval, all values should pass through
    if hasattr(dropout, 'eval'):
        dropout.eval()
        output_eval = dropout(x_eval)
        assert np.allclose(output_eval.data, x_eval.data)
    else:
        # If no eval mode, just check dropout changes values
        assert not np.array_equal(output.data, x.data) or np.all(output.data == x.data * 2)

    print("✅ Dropout layer works!")


def test_unit_function_classes():
    """Test activation function classes."""
    print("🧪 Unit Test: Function Classes...")

    activations = [ReLU(), Sigmoid(), Tanh()]
    x = Tensor(np.array([-1.0, 0.0, 1.0]))

    for act in activations:
        output = act(x)
        assert output.shape == x.shape

    print("✅ Function classes work!")


def test_unit_tensor_autograd():
    """Test tensor autograd integration."""
    print("🧪 Unit Test: Tensor Autograd...")

    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = x * 2
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    print("✅ Tensor autograd works!")


def test_unit_log_softmax():
    """Test log softmax computation."""
    print("🧪 Unit Test: Log Softmax...")

    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    softmax = Softmax()
    output = softmax(x)

    # Output should sum to ~1
    assert np.isclose(output.data.sum(), 1.0, atol=1e-5)
    print("✅ Log Softmax works!")


def test_unit_mse_loss():
    """Test MSE loss computation."""
    print("🧪 Unit Test: MSE Loss...")

    pred = Tensor(np.array([[1.0, 2.0]]))
    target = Tensor(np.array([[1.0, 3.0]]))

    loss_fn = MSELoss()
    loss = loss_fn(pred, target)

    # MSE should be 0.5 (average of [0, 1])
    assert np.isclose(loss.data, 0.5, atol=1e-5)
    print("✅ MSE Loss works!")


def test_unit_cross_entropy_loss():
    """Test Cross Entropy loss computation."""
    print("🧪 Unit Test: Cross Entropy Loss...")

    pred = Tensor(np.array([[0.1, 0.9]]))  # Logits
    target = Tensor(np.array([1]))  # Class index

    loss_fn = CrossEntropyLoss()
    loss = loss_fn(pred, target)

    # Loss should be positive
    assert loss.data > 0
    print("✅ Cross Entropy Loss works!")


def test_unit_binary_cross_entropy_loss():
    """Test Binary Cross Entropy loss computation."""
    print("🧪 Unit Test: BCE Loss...")

    pred = Tensor(np.array([[0.8]]))  # Probability
    target = Tensor(np.array([[1.0]]))  # Label

    loss_fn = BinaryCrossEntropyLoss()
    loss = loss_fn(pred, target)

    # Loss should be positive
    assert loss.data > 0
    print("✅ BCE Loss works!")


def test_unit_optimizer_base():
    """Test base optimizer functionality."""
    print("🧪 Unit Test: Optimizer Base...")

    param = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    optimizer = SGD([param], lr=0.1)

    # Set gradient
    param.grad = Tensor(np.array([1.0, 1.0]))

    # Step
    optimizer.step()

    # Values should decrease (gradient descent)
    assert param.data[0] < 1.0
    assert param.data[1] < 2.0

    print("✅ Optimizer base works!")


def test_unit_sgd_optimizer():
    """Test SGD optimizer with momentum."""
    print("🧪 Unit Test: SGD Optimizer...")

    param = Tensor(np.array([1.0]), requires_grad=True)
    sgd = SGD([param], lr=0.1, momentum=0.9)

    for _ in range(5):
        param.grad = Tensor(np.array([1.0]))
        sgd.step()

    # With momentum, parameter should have moved significantly
    assert param.data[0] < 0

    print("✅ SGD Optimizer works!")


def test_unit_adam_optimizer():
    """Test Adam optimizer."""
    print("🧪 Unit Test: Adam Optimizer...")

    param = Tensor(np.array([1.0]), requires_grad=True)
    adam = Adam([param], lr=0.1)

    for _ in range(5):
        param.grad = Tensor(np.array([1.0]))
        adam.step()

    # Adam should have moved the parameter
    assert param.data[0] < 1.0

    print("✅ Adam Optimizer works!")


def test_unit_adamw_optimizer():
    """Test AdamW optimizer (Adam with weight decay)."""
    print("🧪 Unit Test: AdamW Optimizer...")

    param = Tensor(np.array([1.0]), requires_grad=True)
    adamw = AdamW([param], lr=0.1, weight_decay=0.01)

    for _ in range(5):
        param.grad = Tensor(np.array([0.0]))  # Zero gradient
        adamw.step()

    # With weight decay, parameter should decrease even with zero gradient
    assert param.data[0] < 1.0

    print("✅ AdamW Optimizer works!")
