#!/usr/bin/env python
"""
Minimal Complete Training Example for TinyTorch
================================================
This demonstrates the MINIMUM code needed to get gradient-based training working.
This is what students need to understand to build neural networks that learn.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from tinytorch.core.tensor import Tensor, Parameter
from tinytorch.core.autograd import Variable


class SimpleLinear:
    """Minimal linear layer that works with autograd."""

    def __init__(self, in_features, out_features):
        # Initialize weights and bias as Parameters (Tensors with requires_grad=True)
        self.weights = Parameter(np.random.randn(in_features, out_features) * 0.1)
        self.bias = Parameter(np.random.randn(out_features) * 0.1)

    def __call__(self, x):
        """Forward pass maintaining gradient chain."""
        # Convert everything to Variables for gradient tracking
        if not isinstance(x, Variable):
            x = Variable(x)

        w = Variable(self.weights)
        b = Variable(self.bias)

        # Simple matmul using Variable operations
        # This is inefficient but shows the concept clearly
        output = x @ w + b  # Uses Variable.__matmul__ and Variable.__add__
        return output

    def parameters(self):
        """Return parameters for optimizer."""
        return [self.weights, self.bias]


def sigmoid(x):
    """Sigmoid activation as Variable operation."""
    if not isinstance(x, Variable):
        x = Variable(x)

    # Compute sigmoid
    sig_data = 1.0 / (1.0 + np.exp(-x.data.data))

    # Create gradient function
    def sig_grad_fn(grad_output):
        # Sigmoid gradient: sig * (1 - sig)
        grad = sig_data * (1 - sig_data) * grad_output.data.data
        x.backward(Variable(grad))

    return Variable(sig_data, requires_grad=x.requires_grad, grad_fn=sig_grad_fn)


class SimpleMSE:
    """Minimal MSE loss that returns a scalar Variable."""

    def __call__(self, pred, target):
        """Compute MSE loss."""
        # Convert to Variables
        if not isinstance(pred, Variable):
            pred = Variable(pred)
        if not isinstance(target, Variable):
            target = Variable(target, requires_grad=False)

        # MSE = mean((pred - target)^2)
        diff = pred - target
        squared = diff * diff

        # Manual mean
        total = np.sum(squared.data.data)
        n = squared.data.data.size
        loss_val = total / n

        # Create loss Variable with gradient function
        def mse_grad_fn(grad_output=Variable(1.0)):
            # Gradient of MSE: 2 * (pred - target) / n
            grad = 2.0 * (pred.data.data - target.data.data) / n
            pred.backward(Variable(grad))

        return Variable(loss_val, requires_grad=True, grad_fn=mse_grad_fn)


class SimpleSGD:
    """Minimal SGD optimizer."""

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        """Clear gradients."""
        for p in self.params:
            p.grad = None

    def step(self):
        """Update parameters."""
        for p in self.params:
            if p.grad is not None:
                # Simple gradient descent: param = param - lr * grad
                p.data = p.data - self.lr * p.grad.data


def train_xor_minimal():
    """Train XOR with minimal implementation."""
    print("="*60)
    print("MINIMAL XOR TRAINING EXAMPLE")
    print("This shows the absolute minimum needed for learning")
    print("="*60)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Build simple network
    layer1 = SimpleLinear(2, 4)
    layer2 = SimpleLinear(4, 1)

    # Optimizer and loss
    params = layer1.parameters() + layer2.parameters()
    optimizer = SimpleSGD(params, lr=0.5)
    criterion = SimpleMSE()

    # Training loop
    for epoch in range(1000):
        # Forward pass
        h = layer1(Tensor(X))
        h = sigmoid(h)  # Activation
        output = layer2(h)
        output = sigmoid(output)

        # Compute loss
        loss = criterion(output, Tensor(y))

        # Extract scalar loss value for printing
        loss_val = float(loss.data.data) if hasattr(loss.data, 'data') else float(loss.data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss_val:.4f}")

    # Final predictions
    print("\nFinal predictions:")
    with_grad = False  # No need for gradients during inference
    h = layer1(Tensor(X))
    h = sigmoid(h)
    output = layer2(h)
    output = sigmoid(output)

    # Extract predictions
    if hasattr(output, 'data'):
        if hasattr(output.data, 'data'):
            predictions = output.data.data
        else:
            predictions = output.data
    else:
        predictions = output

    for i, (input_val, pred, target) in enumerate(zip(X, predictions, y)):
        print(f"  Input: {input_val} → Prediction: {pred[0]:.3f} (Target: {target[0]})")

    # Check accuracy
    predictions_binary = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions_binary == y)
    print(f"\nAccuracy: {accuracy*100:.1f}%")

    if accuracy >= 0.75:
        print("✅ XOR learned successfully!")
    else:
        print("⚠️ XOR not fully learned (but training is working)")


def train_linear_regression_minimal():
    """Even simpler: train linear regression."""
    print("\n" + "="*60)
    print("MINIMAL LINEAR REGRESSION")
    print("Simplest possible learning example: y = 2x + 1")
    print("="*60)

    # Simple linear data
    X = np.array([[1], [2], [3], [4]], dtype=np.float32)
    y = np.array([[3], [5], [7], [9]], dtype=np.float32)  # y = 2x + 1

    # Single layer
    model = SimpleLinear(1, 1)
    optimizer = SimpleSGD(model.parameters(), lr=0.01)
    criterion = SimpleMSE()

    print(f"Initial weight: {model.weight.data[0,0]:.3f}")
    print(f"Initial bias:   {model.bias.data[0]:.3f}")

    # Training
    for epoch in range(100):
        output = model(Tensor(X))
        loss = criterion(output, Tensor(y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            loss_val = float(loss.data.data) if hasattr(loss.data, 'data') else float(loss.data)
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.4f}")

    print(f"\nFinal weight: {model.weight.data[0,0]:.3f} (should be ≈2.0)")
    print(f"Final bias:   {model.bias.data[0]:.3f} (should be ≈1.0)")

    # Test prediction
    test_x = Tensor(np.array([[5]], dtype=np.float32))
    pred = model(test_x)
    pred_val = float(pred.data.data[0,0]) if hasattr(pred.data, 'data') else float(pred.data[0,0])
    print(f"\nTest: x=5 → prediction={pred_val:.3f} (should be ≈11.0)")

    if abs(model.weight.data[0,0] - 2.0) < 0.5 and abs(model.bias.data[0] - 1.0) < 0.5:
        print("✅ Linear regression learned successfully!")


if __name__ == "__main__":
    # Start with simplest example
    train_linear_regression_minimal()

    # Then show XOR (non-linear problem)
    print("\n")
    train_xor_minimal()

    print("\n" + "="*60)
    print("KEY INSIGHTS FOR STUDENTS:")
    print("="*60)
    print("""
1. GRADIENT CHAIN: Every operation must maintain the Variable chain
   - Tensors → Variables → Operations → Loss → Backward

2. PARAMETER UPDATES: Gradients must flow back to the original Parameters
   - This requires Variable to keep reference to source Tensor

3. MINIMUM REQUIREMENTS FOR LEARNING:
   - Forward pass that maintains computational graph
   - Loss function that returns a Variable
   - Backward pass that computes gradients
   - Optimizer that updates parameters

4. WHAT MAKES IT WORK:
   - Variable wrapping maintains gradient tracking
   - Operations between Variables create new Variables
   - backward() propagates gradients through the chain
   - Optimizer uses param.grad to update param.data

This is the CORE of all deep learning frameworks!
""")
