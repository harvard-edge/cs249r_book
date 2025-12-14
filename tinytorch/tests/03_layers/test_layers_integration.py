"""
Module 03: Layers - Integration Tests
Tests that Layer base class enables building neural network components
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLayerFoundation:
    """Test Layer base class provides foundation for all layers."""

    def test_layer_base_import(self):
        """Test Layer base class can be imported."""
        from tinytorch.core.layers import Layer
        assert Layer is not None

    def test_layer_interface(self):
        """Test Layer has required interface."""
        from tinytorch.core.layers import Layer

        layer = Layer()

        # Should have forward method
        assert hasattr(layer, 'forward'), "Layer should have forward method"

        # Should be callable (implement __call__)
        assert callable(layer), "Layer should be callable"

    def test_layer_inheritance(self):
        """Test Layer can be inherited from."""
        from tinytorch.core.layers import Layer
        from tinytorch.core.tensor import Tensor

        class TestLayer(Layer):
            def forward(self, x):
                return x  # Identity layer

        layer = TestLayer()
        x = Tensor(np.array([1, 2, 3]))
        output = layer(x)

        assert isinstance(output, Tensor)
        assert np.array_equal(output.data, x.data)


class TestLayerTensorIntegration:
    """Test layers work seamlessly with Tensor inputs/outputs."""

    def test_layer_accepts_tensor_input(self):
        """Test layers accept Tensor inputs."""
        from tinytorch.core.layers import Layer
        from tinytorch.core.tensor import Tensor

        class IdentityLayer(Layer):
            def forward(self, x):
                assert isinstance(x, Tensor), "Layer should receive Tensor"
                return x

        layer = IdentityLayer()
        x = Tensor(np.random.randn(5, 10))
        output = layer(x)

        assert isinstance(output, Tensor)

    def test_layer_produces_tensor_output(self):
        """Test layers produce Tensor outputs."""
        from tinytorch.core.layers import Layer
        from tinytorch.core.tensor import Tensor

        class ScaleLayer(Layer):
            def __init__(self, scale=2.0):
                self.scale = scale

            def forward(self, x):
                return Tensor(x.data * self.scale)

        layer = ScaleLayer(3.0)
        x = Tensor(np.array([1, 2, 3]))
        output = layer(x)

        assert isinstance(output, Tensor)
        assert np.array_equal(output.data, [3, 6, 9])

    def test_layer_batch_processing(self):
        """Test layers handle batch dimensions correctly."""
        from tinytorch.core.layers import Layer
        from tinytorch.core.tensor import Tensor

        class AddBiasLayer(Layer):
            def __init__(self, bias_value=1.0):
                self.bias = bias_value

            def forward(self, x):
                return Tensor(x.data + self.bias)

        layer = AddBiasLayer(5.0)

        # Test with different batch sizes
        for batch_size in [1, 16, 32]:
            x = Tensor(np.zeros((batch_size, 10)))
            output = layer(x)

            assert output.shape == (batch_size, 10)
            assert np.all(output.data == 5.0)


class TestLayerChaining:
    """Test layers can be chained together."""

    def test_layer_output_as_input(self):
        """Test output of one layer can be input to another."""
        from tinytorch.core.layers import Layer
        from tinytorch.core.tensor import Tensor

        class MultiplyLayer(Layer):
            def __init__(self, factor):
                self.factor = factor

            def forward(self, x):
                return Tensor(x.data * self.factor)

        layer1 = MultiplyLayer(2)
        layer2 = MultiplyLayer(3)

        x = Tensor(np.array([1, 2, 3]))
        h = layer1(x)  # [2, 4, 6]
        output = layer2(h)  # [6, 12, 18]

        assert np.array_equal(output.data, [6, 12, 18])

    def test_sequential_layer_processing(self):
        """Test sequential processing through multiple layers."""
        from tinytorch.core.layers import Layer
        from tinytorch.core.tensor import Tensor

        class MultiplyLayer(Layer):
            def __init__(self, factor):
                self.factor = factor

            def forward(self, x):
                return Tensor(x.data * self.factor)

        class ReshapeLayer(Layer):
            def __init__(self, new_shape):
                self.new_shape = new_shape

            def forward(self, x):
                return Tensor(x.data.reshape(self.new_shape))

        # Chain: flatten -> scale -> reshape
        flatten = ReshapeLayer((4,))
        scale = MultiplyLayer(2)
        reshape = ReshapeLayer((2, 2))

        x = Tensor(np.array([[1, 2], [3, 4]]))

        h1 = flatten(x)    # Shape: (4,)
        h2 = scale(h1)     # Values doubled
        output = reshape(h2)  # Shape: (2, 2)

        assert output.shape == (2, 2)
        assert np.array_equal(output.data, [[2, 4], [6, 8]])


class TestLayerParameterManagement:
    """Test layers can manage parameters (weights, biases)."""

    def test_layer_with_parameters(self):
        """Test layer can store and use parameters."""
        from tinytorch.core.layers import Layer
        from tinytorch.core.tensor import Tensor

        class ParameterizedLayer(Layer):
            def __init__(self, input_size, output_size):
                self.weight = Tensor(np.random.randn(input_size, output_size))
                self.bias = Tensor(np.zeros(output_size))

            def forward(self, x):
                output = x.data @ self.weight.data + self.bias.data
                return Tensor(output)

        layer = ParameterizedLayer(10, 5)
        x = Tensor(np.random.randn(3, 10))
        output = layer(x)

        assert output.shape == (3, 5)
        assert hasattr(layer, 'weight')
        assert hasattr(layer, 'bias')
        assert isinstance(layer.weight, Tensor)
        assert isinstance(layer.bias, Tensor)

    def test_layer_parameter_shapes(self):
        """Test layer parameters have correct shapes."""
        from tinytorch.core.layers import Layer
        from tinytorch.core.tensor import Tensor

        class LinearLayer(Layer):
            def __init__(self, in_features, out_features):
                self.in_features = in_features
                self.out_features = out_features
                self.weight = Tensor(np.random.randn(in_features, out_features))
                self.bias = Tensor(np.zeros(out_features))

            def forward(self, x):
                return Tensor(x.data @ self.weight.data + self.bias.data)

        layer = LinearLayer(128, 64)

        assert layer.weight.shape == (128, 64)
        assert layer.bias.shape == (64,)

        # Test with input
        x = Tensor(np.random.randn(16, 128))
        output = layer(x)
        assert output.shape == (16, 64)
