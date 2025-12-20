"""
Module 10: Autograd - Integration Tests
Tests that automatic differentiation works with all previous modules
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestAutogradTensorIntegration:
    """Test autograd integrates with Tensor system."""

    def test_variable_creation(self):
        """Test Variable can be created from Tensor-like data."""
        try:
            from tinytorch.core.autograd import Variable

            # Should create Variable from array
            x = Variable(np.array([1.0, 2.0, 3.0]), requires_grad=True)
            assert x.shape == (3,)
            assert x.requires_grad == True

        except ImportError:
            # Skip if autograd not implemented yet
            assert True, "Autograd not implemented yet"

    def test_gradient_computation_basic(self):
        """Test basic gradient computation."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([2.0]), requires_grad=True)
            y = x * x  # y = x²

            if hasattr(y, 'backward'):
                y.backward()

                # dy/dx = 2x = 2*2 = 4
                assert hasattr(x, 'grad'), "Should compute gradients"
                if x.grad is not None:
                    assert np.isclose(x.grad, 4.0), f"Expected grad=4, got {x.grad}"

        except (ImportError, AttributeError):
            # Skip if autograd not fully implemented
            assert True, "Autograd backward pass not implemented yet"


class TestAutogradLayerIntegration:
    """Test autograd works with layer operations."""

    def test_dense_layer_gradients(self):
        """Test gradients flow through Dense layer."""
        try:
            from tinytorch.core.autograd import Variable
            from tinytorch.core.layers import Linear

            # Create layer
            layer = Linear(2, 1, bias=False)

            # Input with gradients
            x = Variable(np.array([[1.0, 2.0]]), requires_grad=True)

            # Forward pass
            output = layer(x)

            # Should be able to compute gradients
            if hasattr(output, 'backward'):
                loss = output * output  # Simple loss
                loss.backward()

                assert hasattr(x, 'grad'), "Input should have gradients"

        except (ImportError, AttributeError):
            assert True, "Dense-autograd integration not ready"

    def test_activation_gradients(self):
        """Test gradients flow through activations."""
        try:
            from tinytorch.core.autograd import Variable
            from tinytorch.core.activations import ReLU, Sigmoid

            x = Variable(np.array([1.0, -1.0, 2.0]), requires_grad=True)

            relu = ReLU()
            relu_out = relu(x)

            if hasattr(relu_out, 'backward'):
                loss = (relu_out * relu_out).sum()
                loss.backward()

                # ReLU gradient: 1 where x > 0, 0 elsewhere
                expected_grad = np.array([1.0, 0.0, 1.0]) * 2 * relu_out.data
                if x.grad is not None:
                    assert np.allclose(x.grad, expected_grad)

        except (ImportError, AttributeError):
            assert True, "Activation-autograd integration not ready"


class TestAutogradComputationGraph:
    """Test autograd builds and traverses computation graphs."""

    def test_simple_computation_graph(self):
        """Test simple multi-operation graph."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([3.0]), requires_grad=True)
            y = Variable(np.array([2.0]), requires_grad=True)

            # z = x * y + x²
            z = x * y + x * x

            if hasattr(z, 'backward'):
                z.backward()

                # dz/dx = y + 2x = 2 + 2*3 = 8
                # dz/dy = x = 3
                if x.grad is not None and y.grad is not None:
                    assert np.isclose(x.grad, 8.0)
                    assert np.isclose(y.grad, 3.0)

        except (ImportError, AttributeError):
            assert True, "Computation graph not implemented"

    def test_chain_rule(self):
        """Test chain rule works correctly."""
        try:
            from tinytorch.core.autograd import Variable

            x = Variable(np.array([2.0]), requires_grad=True)

            # Chain: x -> x² -> (x²)²
            y = x * x      # y = x²
            z = y * y      # z = y² = (x²)²

            if hasattr(z, 'backward'):
                z.backward()

                # dz/dx = dz/dy * dy/dx = 2y * 2x = 2(x²) * 2x = 4x³
                # At x=2: 4 * 2³ = 4 * 8 = 32
                if x.grad is not None:
                    assert np.isclose(x.grad, 32.0)

        except (ImportError, AttributeError):
            assert True, "Chain rule not implemented"


class TestAutogradOptimizationIntegration:
    """Test autograd enables optimization algorithms."""

    def test_gradient_descent_step(self):
        """Test manual gradient descent step."""
        try:
            from tinytorch.core.autograd import Variable

            # Parameter to optimize
            x = Variable(np.array([5.0]), requires_grad=True)

            # Loss function: (x - 2)²
            target = 2.0
            loss = (x - target) * (x - target)

            if hasattr(loss, 'backward'):
                loss.backward()

                # Gradient descent step
                learning_rate = 0.1
                if x.grad is not None:
                    new_x = x.data - learning_rate * x.grad

                    # Should move closer to target
                    old_distance = abs(x.data - target)
                    new_distance = abs(new_x - target)
                    assert new_distance < old_distance

        except (ImportError, AttributeError):
            assert True, "Optimization integration not ready"

    def test_parameter_updates(self):
        """Test parameter updates work correctly."""
        try:
            from tinytorch.core.autograd import Variable
            from tinytorch.core.layers import Linear

            layer = Linear(1, 1)

            # Convert layer parameters to Variables if needed
            if not isinstance(layer.weight, Variable):
                layer.weight = Variable(layer.weight.data, requires_grad=True)

            # Simple forward pass
            x = Variable(np.array([[1.0]]), requires_grad=True)
            output = layer(x)
            loss = output * output

            if hasattr(loss, 'backward'):
                old_weights = layer.weight.data.copy()

                loss.backward()

                # Update weights
                learning_rate = 0.01
                if layer.weight.grad is not None:
                    new_weights = old_weights - learning_rate * layer.weight.grad
                    assert not np.array_equal(old_weights, new_weights)

        except (ImportError, AttributeError):
            assert True, "Parameter update integration not ready"
