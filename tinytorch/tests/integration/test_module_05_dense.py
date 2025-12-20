"""
Integration Tests for Module 05: DataLoader
These tests verify that the module exports correctly and works as expected.
Run with pytest for detailed reporting.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDenseModuleExports:
    """Test that all expected exports are available."""

    def test_dense_class_exports(self):
        """Test Dense class exports from layers module."""
        from tinytorch.core.layers import Linear as Dense
        assert Dense is not None, "Dense class should be exported"

    def test_dense_is_callable(self):
        """Test Dense can be instantiated."""
        from tinytorch.core.layers import Linear as Dense
        layer = Dense(10, 5)
        assert layer is not None, "Should create Dense layer instance"
        assert hasattr(layer, 'forward'), "Dense should have forward method"


class TestDenseLayerFunctionality:
    """Test that Dense layer works correctly."""

    def test_dense_forward_pass(self):
        """Test Dense layer forward pass produces correct shape."""
        from tinytorch.core.layers import Linear as Dense
        from tinytorch.core.tensor import Tensor

        # Create layer
        layer = Dense(10, 5)

        # Create input
        batch_size = 32
        x = Tensor(np.random.randn(batch_size, 10))

        # Forward pass
        output = layer.forward(x)

        # Check output
        assert output.shape == (32, 5), f"Expected shape (32, 5), got {output.shape}"
        assert isinstance(output, Tensor), "Output should be a Tensor"

    def test_dense_with_bias(self):
        """Test Dense layer with bias enabled."""
        from tinytorch.core.layers import Linear as Dense
        from tinytorch.core.tensor import Tensor

        layer = Dense(10, 5, bias=True)
        assert hasattr(layer, 'bias'), "Layer should have bias"
        assert layer.bias is not None, "Bias should be initialized"

        x = Tensor(np.random.randn(1, 10))
        output = layer(x)
        assert output.shape == (1, 5), "Output shape should be correct with bias"

    def test_dense_without_bias(self):
        """Test Dense layer with bias disabled."""
        from tinytorch.core.layers import Linear as Dense
        from tinytorch.core.tensor import Tensor

        layer = Dense(10, 5, bias=False)
        assert layer.bias is None, "Bias should be None when disabled"

        x = Tensor(np.random.randn(1, 10))
        output = layer(x)
        assert output.shape == (1, 5), "Output shape should be correct without bias"

    def test_dense_callable_interface(self):
        """Test Dense layer can be called directly."""
        from tinytorch.core.layers import Linear as Dense
        from tinytorch.core.tensor import Tensor

        layer = Dense(10, 5)
        x = Tensor(np.random.randn(4, 10))

        # Test both forward() and __call__()
        output1 = layer.forward(x)
        output2 = layer(x)

        assert output1.shape == output2.shape, "forward() and __call__() should produce same shape"


class TestNetworkComposition:
    """Test building multi-layer networks."""

    def test_sequential_exists(self):
        """Test if Sequential is available for network composition."""
        try:
            from tinytorch.nn import Sequential
            assert Sequential is not None
        except ImportError:
            # Sequential might be in a different location
            try:
                from tinytorch.core.networks import Sequential
                assert Sequential is not None
            except ImportError:
                pytest.skip("Sequential not yet implemented")

    def test_multi_layer_network(self):
        """Test building a multi-layer network."""
        from tinytorch.core.layers import Linear as Dense
        from tinytorch.core.activations import ReLU, Sigmoid
        from tinytorch.core.tensor import Tensor

        # Build network manually (without Sequential)
        layer1 = Dense(784, 128)
        relu = ReLU()
        layer2 = Dense(128, 10)
        sigmoid = Sigmoid()

        # Test forward pass through all layers
        x = Tensor(np.random.randn(32, 784))

        h1 = layer1(x)
        assert h1.shape == (32, 128), "First layer output shape incorrect"

        h1_activated = relu(h1)
        assert h1_activated.shape == (32, 128), "ReLU should preserve shape"

        output = layer2(h1_activated)
        assert output.shape == (32, 10), "Second layer output shape incorrect"

        final = sigmoid(output)
        assert final.shape == (32, 10), "Sigmoid should preserve shape"
        assert np.all(final.data >= 0) and np.all(final.data <= 1), "Sigmoid output should be in [0,1]"


class TestXORCapability:
    """Test that the network can represent XOR problem."""

    def test_xor_network_structure(self):
        """Test building XOR network structure."""
        from tinytorch.core.layers import Linear as Dense
        from tinytorch.core.activations import ReLU, Sigmoid
        from tinytorch.core.tensor import Tensor

        # XOR network: 2 -> 4 -> 1
        hidden = Dense(2, 4)
        relu = ReLU()
        output = Dense(4, 1)
        sigmoid = Sigmoid()

        # XOR inputs
        X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

        # Forward pass
        h = hidden(X)
        h_activated = relu(h)
        out = output(h_activated)
        predictions = sigmoid(out)

        assert predictions.shape == (4, 1), "XOR network should produce 4 predictions"
        assert np.all(predictions.data >= 0) and np.all(predictions.data <= 1), "Predictions should be probabilities"


def run_integration_tests():
    """Run all integration tests and return summary."""
    # This would be called by tito module complete
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', __file__, '-v', '--tb=short'],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("\n❌ Some integration tests failed. Please fix the issues above.")
    else:
        print("\n✅ All integration tests passed! Module 05 is working correctly.")
        print("🎯 Capability Unlocked: Neural Networks That Learn")
        print("\nTry the demo: python capabilities/05_neural_networks/demonstrate.py")
