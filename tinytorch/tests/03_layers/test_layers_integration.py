"""
Module 03: Layers - Integration Tests
======================================

Tests that Layer components (Linear, Dropout, Sequential) compose cleanly
with each other and integrate seamlessly with prior modules (Tensor, Activations).
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Layer, Linear, Dropout, Sequential
from tinytorch.core.activations import ReLU, Sigmoid, Tanh


class TestLayerActivationIntegration:
    """Test integration between Linear layers and Activation functions."""

    def test_dense_with_activations(self):
        """Test Linear works with nonlinear activations (ReLU, Sigmoid)."""
        layer = Linear(10, 5)
        relu = ReLU()
        sigmoid = Sigmoid()

        rng = np.random.default_rng(42)
        x = Tensor(rng.standard_normal((16, 10)))
        h = layer(x)
        h_relu = relu(h)
        h_sigmoid = sigmoid(h)

        assert h_relu.shape == h.shape
        assert h_sigmoid.shape == h.shape
        assert np.all(h_sigmoid.data >= 0.0) and np.all(h_sigmoid.data <= 1.0)
        assert np.all(h_relu.data >= 0.0)

    def test_dense_chain(self):
        """Test chaining multiple Linear layers directly."""
        layer1 = Linear(32, 16)
        layer2 = Linear(16, 8)
        layer3 = Linear(8, 2)

        rng = np.random.default_rng(42)
        x = Tensor(rng.standard_normal((4, 32)))
        h1 = layer1(x)
        h2 = layer2(h1)
        output = layer3(h2)

        assert output.shape == (4, 2)


class TestDenseModuleIntegration:
    """Test multi-layer composition with exact arithmetic and parameter ownership."""

    def test_dense_module_integration(self):
        """A hand-calculated two-layer network checks values and parameter ownership."""
        hidden = Linear(2, 2)
        output = Linear(2, 1)
        hidden.weight.data[:] = [[1, -1], [2, 1]]
        hidden.bias.data[:] = [0, -1]
        output.weight.data[:] = [[2], [-1]]
        output.bias.data[:] = [0.5]
        model = Sequential(hidden, ReLU(), output)

        result = model(Tensor([[1, 2], [-2, 1]]))
        # Hidden preactivations are [5, 0] and [0, 2]; ReLU preserves both.
        np.testing.assert_allclose(result.data, [[10.5], [-1.5]])
        assert model.parameters() == [hidden.weight, hidden.bias, output.weight, output.bias]

    def test_sequential_collects_shared_parameters_once(self):
        """Reusing a layer must not ask the optimizer to update its weights twice."""
        shared = Linear(2, 2)
        model = Sequential(shared, ReLU(), Sequential(shared))
        assert model.parameters() == [shared.weight, shared.bias]


class TestLayersDenseNetworkInterface:
    """Test interface compatibility between individual Layers and Sequential networks."""

    def test_dense_layer_to_sequential_network(self):
        """Test that Dense layers can be integrated into Sequential networks."""
        layer1 = Linear(4, 8)
        layer2 = Linear(8, 3)

        network = Sequential([layer1, ReLU(), layer2])
        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((2, 4)))
        result = network(x)

        assert isinstance(result, Tensor)
        assert result.shape == (2, 3)

    def test_dense_layer_compatibility_with_sequential(self):
        """Test that Dense layers are compatible with Sequential construction."""
        individual_layer = Linear(6, 10)
        sequential_network = Sequential([Linear(6, 10), ReLU(), Linear(10, 3)])

        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((1, 6)))

        layer_output = individual_layer(x)
        sequential_output = sequential_network(x)

        assert isinstance(layer_output, Tensor)
        assert isinstance(sequential_output, Tensor)
        assert layer_output.shape == (1, 10)
        assert sequential_output.shape == (1, 3)

    def test_layer_output_as_network_input(self):
        """Test that Dense layer output can be used as network input."""
        preprocessor = Linear(5, 8)
        network = Sequential([
            Linear(8, 12),
            ReLU(),
            Linear(12, 4),
        ])

        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((3, 5)))
        preprocessed = preprocessor(x)
        final_output = network(preprocessed)

        assert isinstance(preprocessed, Tensor)
        assert isinstance(final_output, Tensor)
        assert final_output.shape == (3, 4)

    def test_network_layer_composition(self):
        """Test that networks can be composed with individual layers."""
        base_network = Sequential([Linear(4, 6), ReLU(), Linear(6, 8)])
        final_layer = Linear(8, 2)

        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((2, 4)))

        network_output = base_network(x)
        final_output = final_layer(network_output)

        assert isinstance(network_output, Tensor)
        assert isinstance(final_output, Tensor)
        assert network_output.shape == (2, 8)
        assert final_output.shape == (2, 2)


class TestLayerNetworkDataFlow:
    """Test data flow compatibility between layers and networks."""

    def test_shape_preservation_across_layer_network_boundary(self):
        """Test shape preservation when crossing layer-network boundaries."""
        shape_configs = [
            (1, 4, 8, 2),
            (5, 6, 10, 3),
            (10, 8, 16, 4),
        ]

        rng = np.random.default_rng(7)
        for batch_size, input_size, hidden_size, output_size in shape_configs:
            layer = Linear(input_size, hidden_size)
            network = Sequential([
                Linear(hidden_size, hidden_size),
                ReLU(),
                Linear(hidden_size, output_size),
            ])

            x = Tensor(rng.standard_normal((batch_size, input_size)))
            layer_out = layer(x)
            network_out = network(layer_out)

            assert layer_out.shape == (batch_size, hidden_size)
            assert network_out.shape == (batch_size, output_size)

    def test_dtype_normalization_across_layer_network_boundary(self):
        """Test that TinyTorch normalizes data to float32 consistently."""
        layer = Linear(4, 6)
        network = Sequential([Linear(6, 8), ReLU(), Linear(8, 2)])

        rng = np.random.default_rng(7)
        x_f32 = Tensor(rng.standard_normal((2, 4)).astype(np.float32))
        layer_out_f32 = layer(x_f32)
        network_out_f32 = network(layer_out_f32)

        assert layer_out_f32.dtype == np.float32
        assert network_out_f32.dtype == np.float32

        x_f64 = Tensor(rng.standard_normal((2, 4)).astype(np.float64))
        layer_out_f64 = layer(x_f64)
        network_out_f64 = network(layer_out_f64)

        assert layer_out_f64.dtype == np.float32
        assert network_out_f64.dtype == np.float32

    def test_error_handling_at_layer_network_boundary(self):
        """Test error handling when layer-network interfaces are incompatible."""
        layer = Linear(4, 6)
        mismatched_network = Sequential([Linear(8, 2)])

        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((1, 4)))
        layer_output = layer(x)

        with pytest.raises((ValueError, AssertionError, TypeError)):
            mismatched_network(layer_output)


class TestLayerNetworkSystemIntegration:
    """Test system-level integration scenarios with layers and networks."""

    def test_multi_stage_processing_pipeline(self):
        """Test multi-stage processing using layers and networks."""
        preprocessor = Linear(8, 12)
        feature_extractor = Sequential([
            Linear(12, 16),
            ReLU(),
            Linear(16, 10),
        ])
        classifier = Linear(10, 3)

        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((4, 8)))

        preprocessed = preprocessor(x)
        features = feature_extractor(preprocessed)
        predictions = classifier(features)

        assert isinstance(preprocessed, Tensor)
        assert isinstance(features, Tensor)
        assert isinstance(predictions, Tensor)
        assert predictions.shape == (4, 3)

    def test_parallel_layer_processing(self):
        """Test parallel processing with multiple layers feeding into network."""
        branch1 = Linear(6, 4)
        branch2 = Linear(6, 4)
        branch3 = Linear(6, 4)

        fusion_network = Sequential([
            Linear(12, 8),
            ReLU(),
            Linear(8, 2),
        ])

        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((2, 6)))

        out1 = branch1(x)
        out2 = branch2(x)
        out3 = branch3(x)

        fused_data = np.concatenate([out1.data, out2.data, out3.data], axis=1)
        fused_tensor = Tensor(fused_data)
        final_output = fusion_network(fused_tensor)

        assert out1.shape == (2, 4)
        assert out2.shape == (2, 4)
        assert out3.shape == (2, 4)
        assert fused_tensor.shape == (2, 12)
        assert final_output.shape == (2, 2)

    def test_layer_network_modularity(self):
        """Test that layers and networks can be replaced modularly."""
        input_processors = [Linear(5, 8), Linear(5, 8)]
        core_networks = [
            Sequential([Linear(8, 10), ReLU(), Linear(10, 6)]),
            Sequential([Linear(8, 6)]),
        ]
        output_processors = [Linear(6, 3), Linear(6, 3)]

        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((1, 5)))

        for input_proc in input_processors:
            for core_net in core_networks:
                for output_proc in output_processors:
                    intermediate1 = input_proc(x)
                    intermediate2 = core_net(intermediate1)
                    final = output_proc(intermediate2)

                    assert isinstance(final, Tensor)
                    assert final.shape == (1, 3)


class TestLayerNetworkInterfaceStandards:
    """Test that layers and networks follow consistent interface standards."""

    def test_consistent_call_interface(self):
        """Test that layers and networks have consistent callable interface."""
        components = [
            Linear(4, 6),
            Sequential([Linear(4, 6)]),
            Sequential([Linear(4, 8), ReLU(), Linear(8, 6)]),
        ]

        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((1, 4)))

        for component in components:
            result = component(x)
            assert isinstance(result, Tensor)
            assert result.shape[0] == 1
            assert result.shape[1] == 6

    def test_component_property_consistency(self):
        """Test that layers and networks have consistent properties."""
        layer = Linear(3, 5)
        network = Sequential([Linear(3, 5)])
        multi_layer = Sequential([Linear(3, 4), ReLU(), Linear(4, 5)])

        rng = np.random.default_rng(7)
        x = Tensor(rng.standard_normal((2, 3)))

        for component in [layer, network, multi_layer]:
            result = component(x)
            assert hasattr(result, 'shape')
            assert hasattr(result, 'data')
            assert hasattr(result, 'dtype')
            assert result.shape == (2, 5)


def test_full_dropout_keeps_zero_gradient_connection():
    """Module 06 must reach the input even when every activation is dropped."""
    from tinytorch.core.layers import Dropout

    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    output = Dropout(1.0)(x)
    assert output.requires_grad
    output.sum().backward()
    np.testing.assert_array_equal(output.data, np.zeros((2, 2)))
    np.testing.assert_array_equal(x.grad, np.zeros((2, 2)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
