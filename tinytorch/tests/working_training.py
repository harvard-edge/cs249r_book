#!/usr/bin/env python
"""
Working Training Example - Proper Solution
===========================================
This shows how to make training work with the current architecture.
The key: ensure Variables maintain connection to Parameters.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from tinytorch.core.tensor import Tensor, Parameter
from tinytorch.core.autograd import Variable


class WorkingLinear:
    """Linear layer that properly maintains gradient connections."""

    def __init__(self, in_features, out_features):
        # Parameters with requires_grad=True
        self.weights = Parameter(np.random.randn(in_features, out_features) * 0.1)
        self.bias = Parameter(np.random.randn(out_features) * 0.1)

        # Keep Variable versions that maintain connection
        self._weight_var = Variable(self.weights)
        self._bias_var = Variable(self.bias)

    def forward(self, x):
        """Forward pass maintaining gradient chain."""
        # Ensure input is Variable
        if not isinstance(x, Variable):
            x = Variable(x, requires_grad=False)

        # Use Variable versions of parameters
        # These maintain connection via _source_tensor
        output = x @ self._weight_var + self._bias_var
        return output

    def parameters(self):
        """Return original parameters for optimizer."""
        return [self.weights, self.bias]

    def __call__(self, x):
        return self.forward(x)


def sigmoid_variable(x):
    """Sigmoid that works with Variables."""
    if not isinstance(x, Variable):
        x = Variable(x)

    # Forward
    sig_data = 1.0 / (1.0 + np.exp(-x.data.data))

    # Backward
    def grad_fn(grad_output):
        grad = sig_data * (1 - sig_data) * grad_output.data.data
        x.backward(Variable(grad))

    return Variable(sig_data, requires_grad=x.requires_grad, grad_fn=grad_fn)


def relu_variable(x):
    """ReLU that works with Variables."""
    if not isinstance(x, Variable):
        x = Variable(x)

    # Forward
    relu_data = np.maximum(0, x.data.data)

    # Backward
    def grad_fn(grad_output):
        grad = (x.data.data > 0) * grad_output.data.data
        x.backward(Variable(grad))

    return Variable(relu_data, requires_grad=x.requires_grad, grad_fn=grad_fn)


class WorkingMSE:
    """MSE loss that properly computes gradients."""

    def __call__(self, pred, target):
        # Convert to Variables
        if not isinstance(pred, Variable):
            pred = Variable(pred)
        if not isinstance(target, Variable):
            target = Variable(target, requires_grad=False)

        # Forward: MSE = mean((pred - target)^2)
        diff = pred - target
        squared = diff * diff

        # Manual mean
        n = squared.data.data.size
        loss_val = np.mean(squared.data.data)

        # Backward
        def grad_fn(grad_output=Variable(1.0)):
            # Gradient: 2 * (pred - target) / n
            grad = 2.0 * (pred.data.data - target.data.data) / n
            pred.backward(Variable(grad))

        return Variable(loss_val, requires_grad=True, grad_fn=grad_fn)


class WorkingSGD:
    """SGD optimizer that updates parameters."""

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data = p.data - self.lr * p.grad.data


def train_xor_working():
    """Train XOR with working implementation."""
    print("="*60)
    print("WORKING XOR TRAINING")
    print("="*60)

    # Data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Network
    layer1 = WorkingLinear(2, 8)
    layer2 = WorkingLinear(8, 1)

    # Training setup
    params = layer1.parameters() + layer2.parameters()
    optimizer = WorkingSGD(params, lr=0.5)
    criterion = WorkingMSE()

    # Training loop
    losses = []
    for epoch in range(1000):
        # Forward
        h = layer1(Tensor(X))
        h = relu_variable(h)
        output = layer2(h)
        output = sigmoid_variable(output)

        # Loss
        loss = criterion(output, Tensor(y))
        loss_val = float(loss.data.data)
        losses.append(loss_val)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients (first epoch only)
        if epoch == 0:
            print("Gradient check:")
            for i, p in enumerate(params):
                if p.grad is not None:
                    grad_norm = np.linalg.norm(p.grad.data)
                    print(f"  Param {i}: gradient norm = {grad_norm:.4f}")
                else:
                    print(f"  Param {i}: NO GRADIENT!")

        # Update
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss_val:.4f}")

    # Results
    print("\nFinal predictions:")
    h = layer1(Tensor(X))
    h = relu_variable(h)
    output = layer2(h)
    output = sigmoid_variable(output)

    predictions = output.data.data
    for x_val, pred, target in zip(X, predictions, y):
        print(f"  {x_val} → {pred[0]:.3f} (target: {target[0]})")

    # Accuracy
    binary_preds = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_preds == y)
    print(f"\nAccuracy: {accuracy*100:.0f}%")

    if accuracy == 1.0:
        print("✅ XOR learned perfectly!")
    elif accuracy >= 0.75:
        print("✅ XOR learned well!")
    else:
        print("⚠️ XOR partially learned")


def train_linear_regression_working():
    """Train linear regression with working implementation."""
    print("\n" + "="*60)
    print("WORKING LINEAR REGRESSION")
    print("="*60)

    # Data: y = 2x + 1
    X = np.array([[1], [2], [3], [4]], dtype=np.float32)
    y = np.array([[3], [5], [7], [9]], dtype=np.float32)

    # Model
    model = WorkingLinear(1, 1)
    print(f"Initial: weight={model.weight.data[0,0]:.3f}, bias={model.bias.data[0]:.3f}")

    optimizer = WorkingSGD(model.parameters(), lr=0.01)
    criterion = WorkingMSE()

    # Training
    for epoch in range(200):
        output = model(Tensor(X))
        loss = criterion(output, Tensor(y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            loss_val = float(loss.data.data)
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.4f}")

    print(f"Final: weight={model.weight.data[0,0]:.3f}, bias={model.bias.data[0]:.3f}")
    print(f"Target: weight=2.000, bias=1.000")

    # Check
    w_err = abs(model.weight.data[0,0] - 2.0)
    b_err = abs(model.bias.data[0] - 1.0)

    if w_err < 0.1 and b_err < 0.1:
        print("✅ Linear regression learned perfectly!")


if __name__ == "__main__":
    # Test simple case first
    train_linear_regression_working()

    # Test XOR
    print()
    train_xor_working()

    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("""
The working solution shows that we need:

1. Variables that maintain connection to source Parameters (_source_tensor)
2. Operations between Variables that create new Variables with grad_fn
3. Backward pass that propagates gradients back to original Parameters

The current TinyTorch architecture CAN work, but layers need to:
- Keep Variable versions of parameters that maintain connections
- Use these Variables in forward passes
- Return Variables, not Tensors

This is why PyTorch unified Tensor and Variable - to avoid this complexity!
""")
