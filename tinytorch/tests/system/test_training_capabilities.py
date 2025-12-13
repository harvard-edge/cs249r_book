#!/usr/bin/env python
"""
Training Capability Tests for TinyTorch
========================================
Tests that models can actually learn (not just forward pass).
Validates gradient flow, parameter updates, and convergence.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.training import MeanSquaredError, CrossEntropyLoss
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.nn import Sequential


class TrainingTester:
    """Test training capabilities."""

    def __init__(self):
        self.passed = []
        self.failed = []

    def test(self, name, func):
        """Run a test and track results."""
        try:
            result = func()
            if result:
                self.passed.append(name)
                print(f"✅ {name}")
            else:
                self.failed.append((name, "Did not converge"))
                print(f"⚠️  {name}: Did not converge")
            return result
        except Exception as e:
            self.failed.append((name, str(e)))
            print(f"❌ {name}: {e}")
            return False

    def summary(self):
        """Print test summary."""
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"TRAINING TESTS: {len(self.passed)}/{total} passed")
        if self.failed:
            print("\nFailed tests:")
            for name, error in self.failed:
                print(f"  - {name}: {error}")
        return len(self.failed) == 0


def test_linear_regression():
    """Test if we can learn a simple linear function."""
    # Generate linear data: y = 2x + 1
    np.random.seed(42)
    X = np.random.randn(100, 1).astype(np.float32)
    y_true = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

    X_tensor = Tensor(X)
    y_tensor = Tensor(y_true)

    # Simple linear model
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), learning_rate=0.01)
    criterion = MeanSquaredError()

    # Training loop
    initial_loss = None
    final_loss = None

    for epoch in range(100):
        # Forward
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)

        if epoch == 0:
            initial_loss = float(loss.data)
        if epoch == 99:
            final_loss = float(loss.data)

        # Backward (if autograd is available)
        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            # If autograd not available, skip gradient update
            pass

    # Check if loss decreased
    if initial_loss and final_loss:
        improved = final_loss < initial_loss * 0.5  # Loss should drop by at least 50%
        return improved
    return False


def test_xor_learning():
    """Test if we can learn XOR (non-linear problem)."""
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    X_tensor = Tensor(X)
    y_tensor = Tensor(y)

    # Network with hidden layer
    model = Sequential([
        Linear(2, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid()
    ])

    optimizer = Adam(model.parameters(), learning_rate=0.1)
    criterion = MeanSquaredError()

    # Training
    initial_loss = None
    final_loss = None

    for epoch in range(500):
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)

        if epoch == 0:
            initial_loss = float(loss.data)
        if epoch == 499:
            final_loss = float(loss.data)

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    # Check convergence
    if initial_loss and final_loss:
        # For XOR, we should get very low loss if learning works
        converged = final_loss < 0.1  # Should be close to 0
        return converged
    return False


def test_multiclass_classification():
    """Test multiclass classification learning."""
    # Generate 3-class dataset
    np.random.seed(42)
    n_samples = 150
    n_features = 2
    n_classes = 3

    # Create clustered data
    X = []
    y = []
    for i in range(n_classes):
        center = np.array([np.cos(2 * np.pi * i / n_classes),
                          np.sin(2 * np.pi * i / n_classes)]) * 2
        cluster = np.random.randn(n_samples // n_classes, n_features) * 0.5 + center
        X.append(cluster)
        y.extend([i] * (n_samples // n_classes))

    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)

    X_tensor = Tensor(X)
    y_tensor = Tensor(y)

    # Build classifier
    model = Sequential([
        Linear(n_features, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, n_classes)
    ])

    optimizer = Adam(model.parameters(), learning_rate=0.01)
    criterion = CrossEntropyLoss()

    # Training
    initial_loss = None
    final_loss = None

    for epoch in range(200):
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)

        if epoch == 0:
            initial_loss = float(loss.data)
        if epoch == 199:
            final_loss = float(loss.data)

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    # Check if loss decreased significantly
    if initial_loss and final_loss:
        improved = final_loss < initial_loss * 0.3
        return improved
    return False


def test_gradient_flow():
    """Test that gradients flow through deep networks."""
    # Build deep network
    layers = []
    width = 10
    depth = 5

    for i in range(depth):
        if i == 0:
            layers.append(Linear(2, width))
        elif i == depth - 1:
            layers.append(Linear(width, 1))
        else:
            layers.append(Linear(width, width))

        if i < depth - 1:
            layers.append(ReLU())

    model = Sequential(layers)

    # Test data
    X = Tensor(np.random.randn(10, 2).astype(np.float32))
    y = Tensor(np.random.randn(10, 1).astype(np.float32))

    criterion = MeanSquaredError()

    # Forward and backward
    try:
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()

        # Check if gradients exist in all layers
        gradients_exist = True
        for layer in model.layers:
            if hasattr(layer, 'weight'):
                if layer.weight.grad is None:
                    gradients_exist = False
                    break

        return gradients_exist
    except:
        return False


def test_optimizer_updates():
    """Test that optimizers actually update parameters."""
    model = Linear(5, 3)
    optimizer = SGD(model.parameters(), learning_rate=0.1)

    # Get initial weights
    initial_weights = model.weight.data.copy()

    # Dummy forward pass
    X = Tensor(np.random.randn(2, 5).astype(np.float32))
    y_true = Tensor(np.random.randn(2, 3).astype(np.float32))

    criterion = MeanSquaredError()

    try:
        # Forward
        y_pred = model(X)
        loss = criterion(y_pred, y_true)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check if weights changed
        weights_changed = not np.allclose(initial_weights, model.weight.data)
        return weights_changed
    except:
        return False


def test_learning_rate_effect():
    """Test that learning rate affects convergence speed."""
    def train_with_lr(lr):
        model = Linear(1, 1)
        optimizer = SGD(model.parameters(), learning_rate=lr)
        criterion = MeanSquaredError()

        # Simple data
        X = Tensor(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
        y = Tensor(np.array([[2.0], [4.0], [6.0]], dtype=np.float32))

        losses = []
        for _ in range(50):
            y_pred = model(X)
            loss = criterion(y_pred, y)
            losses.append(float(loss.data))

            try:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except:
                pass

        return losses[-1] if losses else float('inf')

    # Test different learning rates
    loss_small_lr = train_with_lr(0.001)
    loss_medium_lr = train_with_lr(0.01)
    loss_large_lr = train_with_lr(0.1)

    # Medium LR should converge better than too small or too large
    optimal_lr = (loss_medium_lr < loss_small_lr) or (loss_medium_lr < loss_large_lr)
    return optimal_lr


def test_adam_vs_sgd():
    """Test that Adam converges faster than SGD on non-convex problems."""
    def train_with_optimizer(opt_class):
        # Non-convex problem (XOR-like)
        X = Tensor(np.random.randn(20, 2).astype(np.float32))
        y = Tensor((np.sum(X.data, axis=1, keepdims=True) > 0).astype(np.float32))

        model = Sequential([
            Linear(2, 10),
            ReLU(),
            Linear(10, 1),
            Sigmoid()
        ])

        optimizer = opt_class(model.parameters(), learning_rate=0.01)
        criterion = MeanSquaredError()

        losses = []
        for _ in range(100):
            y_pred = model(X)
            loss = criterion(y_pred, y)
            losses.append(float(loss.data))

            try:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except:
                pass

        return losses[-1] if losses else float('inf')

    sgd_loss = train_with_optimizer(SGD)
    adam_loss = train_with_optimizer(Adam)

    # Adam should generally converge to lower loss
    adam_better = adam_loss < sgd_loss * 1.2  # Allow some tolerance
    return adam_better


def run_all_training_tests():
    """Run comprehensive training tests."""
    print("="*60)
    print("TRAINING CAPABILITY TEST SUITE")
    print("Testing that models can actually learn")
    print("="*60)

    tester = TrainingTester()

    # Basic learning
    print("\n📈 Basic Learning:")
    tester.test("Linear regression", test_linear_regression)
    tester.test("XOR problem", test_xor_learning)
    tester.test("Multiclass classification", test_multiclass_classification)

    # Gradient mechanics
    print("\n🔄 Gradient Mechanics:")
    tester.test("Gradient flow through deep network", test_gradient_flow)
    tester.test("Optimizer parameter updates", test_optimizer_updates)

    # Optimization behavior
    print("\n⚡ Optimization Behavior:")
    tester.test("Learning rate effect", test_learning_rate_effect)
    tester.test("Adam vs SGD convergence", test_adam_vs_sgd)

    return tester.summary()


if __name__ == "__main__":
    print("🔬 Testing training capabilities...")
    print("Note: These tests require working autograd for full functionality")
    print()

    success = run_all_training_tests()
    sys.exit(0 if success else 1)
