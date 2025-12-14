"""
Tests for Module 04: Linear/Networks
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLinearExports:
    """Test that Linear layer is properly exported."""

    def test_dense_import(self):
        """Test Linear can be imported from correct location."""
        from tinytorch.core.layers import Linear
        assert Linear is not None

    def test_dense_creation(self):
        """Test Linear layer can be created."""
        from tinytorch.core.layers import Linear
        layer = Linear(10, 5)
        assert layer.weight.shape == (10, 5)


class TestLinearForward:
    """Test Linear layer forward pass."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.tensor import Tensor

        layer = Linear(10, 5)
        x = Tensor(np.random.randn(32, 10))
        output = layer(x)

        assert output.shape == (32, 5)

    def test_forward_with_bias(self):
        """Test forward pass with bias."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.tensor import Tensor

        layer = Linear(10, 5, bias=True)
        x = Tensor(np.zeros((1, 10)))
        output = layer(x)

        # With zero input, output should equal bias
        assert np.allclose(output.data, layer.bias.data)

    def test_forward_without_bias(self):
        """Test forward pass without bias."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.tensor import Tensor

        layer = Linear(10, 5, bias=False)
        x = Tensor(np.zeros((1, 10)))
        output = layer(x)

        # With zero input and no bias, output should be zero
        assert np.allclose(output.data, 0)


class TestLinearIntegration:
    """Test Linear layer integration with other modules."""

    def test_dense_with_tensor(self):
        """Test Linear works with Tensor (Module 02)."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.tensor import Tensor

        layer = Linear(10, 5)

        # Weights and bias should be Tensors
        assert isinstance(layer.weight, Tensor)
        if layer.bias is not None:
            assert isinstance(layer.bias, Tensor)

    def test_dense_with_activations(self):
        """Test Linear works with activations (Module 03)."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU, Sigmoid
        from tinytorch.core.tensor import Tensor

        layer = Linear(10, 5)
        relu = ReLU()
        sigmoid = Sigmoid()

        x = Tensor(np.random.randn(16, 10))
        h = layer(x)
        h_relu = relu(h)
        h_sigmoid = sigmoid(h)

        assert h_relu.shape == h.shape
        assert h_sigmoid.shape == h.shape
        assert np.all(h_sigmoid.data >= 0) and np.all(h_sigmoid.data <= 1)

    def test_dense_chain(self):
        """Test chaining multiple Linear layers."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.tensor import Tensor

        layer1 = Linear(784, 128)
        layer2 = Linear(128, 64)
        layer3 = Linear(64, 10)

        x = Tensor(np.random.randn(32, 784))
        h1 = layer1(x)
        h2 = layer2(h1)
        output = layer3(h2)

        assert output.shape == (32, 10)
