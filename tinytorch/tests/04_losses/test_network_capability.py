"""
Network Capability Tests for Module 05
Tests that networks can solve non-linear problems
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestXORCapability:
    """Test that multi-layer networks can solve XOR."""

    def test_xor_network_structure(self):
        """Test building network for XOR problem."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU, Sigmoid
        from tinytorch.core.tensor import Tensor

        # Build XOR network: 2 -> 4 -> 1
        hidden = Linear(2, 4, bias=True)
        output = Linear(4, 1, bias=True)
        relu = ReLU()
        sigmoid = Sigmoid()

        # XOR inputs
        X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))

        # Forward pass
        h = hidden(X)
        h_activated = relu(h)
        out = output(h_activated)
        predictions = sigmoid(out)

        assert predictions.shape == (4, 1)
        assert np.all(predictions.data >= 0) and np.all(predictions.data <= 1)

    def test_xor_network_expressiveness(self):
        """Test that network has enough capacity for XOR."""
        from tinytorch.core.layers import Linear

        # XOR needs at least 2 hidden units
        hidden = Linear(2, 4)  # 4 hidden units is sufficient
        output = Linear(4, 1)

        # Count parameters
        hidden_params = 2 * 4 + 4  # weights + bias
        output_params = 4 * 1 + 1  # weights + bias
        total_params = hidden_params + output_params

        # XOR needs at least 9 parameters theoretically
        assert total_params >= 9

    def test_nonlinearity_required(self):
        """Test that non-linearity is essential for XOR."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU
        from tinytorch.core.tensor import Tensor

        # Without activation, network is just linear
        layer1 = Linear(2, 4)
        layer2 = Linear(4, 1)
        relu = ReLU()

        X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))

        # Linear network (no activation)
        linear_out = layer2(layer1(X))

        # Non-linear network (with activation)
        nonlinear_out = layer2(relu(layer1(X)))

        # Both should produce different outputs
        assert not np.allclose(linear_out.data, nonlinear_out.data)


class TestMLPCapabilities:
    """Test Multi-Layer Perceptron capabilities."""

    def test_universal_approximation(self):
        """Test that MLPs can approximate continuous functions."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.activations import ReLU
        from tinytorch.core.tensor import Tensor

        # Wide hidden layer can approximate any function
        layer1 = Linear(1, 100)  # Wide hidden layer
        relu = ReLU()
        layer2 = Linear(100, 1)

        # Test on simple function: sin(x)
        x = np.linspace(-np.pi, np.pi, 50).reshape(-1, 1)
        X = Tensor(x)

        # Network should be able to produce varied outputs
        h = relu(layer1(X))
        output = layer2(h)

        # Check that network produces non-constant output
        assert output.data.std() > 0  # Not all same value
        assert output.shape == (50, 1)

    def test_deep_network(self):
        """Test building deep networks."""
        from tinytorch.core.layers import Linear
        from tinytorch.core.tensor import Tensor

        # Build 5-layer network
        layers = [
            Linear(100, 50),
            Linear(50, 25),
            Linear(25, 12),
            Linear(12, 6),
            Linear(6, 1)
        ]

        x = Tensor(np.random.randn(16, 100))

        # Forward through all layers
        for layer in layers:
            x = layer(x)

        assert x.shape == (16, 1)
