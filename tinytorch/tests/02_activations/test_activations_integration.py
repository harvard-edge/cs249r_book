"""
Module 02: Activations - Integration Tests
Tests that activations work with Tensor and enable non-linear networks
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestActivationTensorIntegration:
    """Test activations work seamlessly with Tensor."""

    def test_relu_with_tensor(self):
        """Test ReLU activation with Tensor inputs."""
        from tinytorch.core.activations import ReLU
        from tinytorch.core.tensor import Tensor

        relu = ReLU()
        x = Tensor(np.array([-2, -1, 0, 1, 2]))
        output = relu(x)

        assert isinstance(output, Tensor)
        assert np.array_equal(output.data, [0, 0, 0, 1, 2])
        assert output.shape == x.shape

    def test_sigmoid_with_tensor(self):
        """Test Sigmoid activation with Tensor inputs."""
        from tinytorch.core.activations import Sigmoid
        from tinytorch.core.tensor import Tensor

        sigmoid = Sigmoid()
        x = Tensor(np.array([0, 1, -1, 2]))
        output = sigmoid(x)

        assert isinstance(output, Tensor)
        assert output.shape == x.shape
        assert np.all(output.data >= 0) and np.all(output.data <= 1)
        assert np.isclose(output.data[0], 0.5, atol=1e-6)  # sigmoid(0) = 0.5

    def test_tanh_with_tensor(self):
        """Test Tanh activation with Tensor inputs."""
        from tinytorch.core.activations import Tanh
        from tinytorch.core.tensor import Tensor

        tanh = Tanh()
        x = Tensor(np.array([0, 1, -1]))
        output = tanh(x)

        assert isinstance(output, Tensor)
        assert output.shape == x.shape
        assert np.all(output.data >= -1) and np.all(output.data <= 1)
        assert np.isclose(output.data[0], 0, atol=1e-6)  # tanh(0) = 0


class TestActivationNetworkIntegration:
    """Test activations enable non-linear neural networks."""

    def test_xor_requires_nonlinearity(self):
        """Test that XOR problem demonstrates need for activations."""
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.activations import ReLU

        # XOR inputs and targets
        X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        Y_target = np.array([0, 1, 1, 0])

        # Simple linear transformation (no activation)
        W = Tensor(np.array([[1, -1], [-1, 1]]))  # Arbitrary weights
        linear_output = Tensor(X.data @ W.data)

        # With ReLU activation
        relu = ReLU()
        nonlinear_output = relu(linear_output)

        # Outputs should be different (activation adds non-linearity)
        assert not np.array_equal(linear_output.data, nonlinear_output.data)
        assert nonlinear_output.shape == linear_output.shape

    def test_activation_chaining(self):
        """Test chaining multiple activations."""
        from tinytorch.core.activations import ReLU, Sigmoid
        from tinytorch.core.tensor import Tensor

        x = Tensor(np.random.randn(10, 5))
        relu = ReLU()
        sigmoid = Sigmoid()

        # Chain: input -> ReLU -> Sigmoid
        h1 = relu(x)
        output = sigmoid(h1)

        assert output.shape == x.shape
        assert np.all(output.data >= 0) and np.all(output.data <= 1)

    def test_activation_with_negative_inputs(self):
        """Test activations handle negative inputs correctly."""
        from tinytorch.core.activations import ReLU, Sigmoid, Tanh
        from tinytorch.core.tensor import Tensor

        x = Tensor(np.array([-5, -1, 0, 1, 5]))

        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()

        # Test each activation with negative inputs
        relu_out = relu(x)
        sig_out = sigmoid(x)
        tanh_out = tanh(x)

        # ReLU zeros out negatives
        assert np.array_equal(relu_out.data, [0, 0, 0, 1, 5])

        # Sigmoid maps all to (0,1)
        assert np.all(sig_out.data > 0) and np.all(sig_out.data < 1)

        # Tanh maps all to (-1,1)
        assert np.all(tanh_out.data > -1) and np.all(tanh_out.data < 1)


class TestActivationDerivatives:
    """Test activation derivatives for future gradient computation."""

    def test_relu_derivative(self):
        """Test ReLU derivative behavior."""
        from tinytorch.core.activations import ReLU
        from tinytorch.core.tensor import Tensor

        relu = ReLU()
        x = Tensor(np.array([-1, 0, 1]))
        output = relu(x)

        # ReLU derivative is 0 for x < 0, undefined at 0, 1 for x > 0
        # For implementation, we usually use: derivative = (output > 0)
        derivative_mask = output.data > 0
        expected_derivative = np.array([False, False, True])

        assert np.array_equal(derivative_mask, expected_derivative)

    def test_sigmoid_derivative_property(self):
        """Test sigmoid has the derivative property: σ'(x) = σ(x)(1-σ(x))."""
        from tinytorch.core.activations import Sigmoid
        from tinytorch.core.tensor import Tensor

        sigmoid = Sigmoid()
        x = Tensor(np.array([0, 1, -1]))
        output = sigmoid(x)

        # Sigmoid derivative: σ(x) * (1 - σ(x))
        derivative = output.data * (1 - output.data)

        # At x=0, σ(0)=0.5, so derivative should be 0.5*0.5=0.25
        assert np.isclose(derivative[0], 0.25, atol=1e-6)
