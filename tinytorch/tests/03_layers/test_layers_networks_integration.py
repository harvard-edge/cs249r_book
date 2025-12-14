"""
Integration Tests - Layers and Dense Networks

Tests cross-module interfaces and compatibility between individual Layers and Dense Network modules.
Focuses on integration, not re-testing individual module functionality.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU, Sigmoid, Tanh


class TestLayersDenseNetworkInterface:
    """Test interface compatibility between individual Layers and Dense Networks."""

    def test_dense_layer_to_sequential_network(self):
        """Test that Dense layers can be integrated into Sequential networks."""
        # Create individual dense layers
        layer1 = Linear(4, 8)
        layer2 = Linear(8, 3)

        # Test integration into Sequential
        network = Sequential([layer1, ReLU(), layer2])

        # Test interface compatibility
        x = Tensor(np.random.randn(2, 4))
        result = network(x)

        # Verify integration works
        assert isinstance(result, Tensor), "Sequential should work with Dense layers"
        assert result.shape == (2, 3), "Sequential should process through all layers"

    def test_dense_layer_compatibility_with_sequential(self):
        """Test that Dense layers are compatible with Sequential construction."""
        # Test that Sequential uses same interface as individual Dense layers
        individual_layer = Linear(6, 10)
        sequential_network = Sequential([Linear(6, 10), ReLU(), Linear(10, 3)])

        # Test same input works with both
        x = Tensor(np.random.randn(1, 6))

        # Individual layer output
        layer_output = individual_layer(x)

        # Sequential output (should accept same input)
        sequential_output = sequential_network(x)

        # Verify interface compatibility
        assert isinstance(layer_output, Tensor), "Dense layer should return Tensor"
        assert isinstance(sequential_output, Tensor), "Sequential should return Tensor"
        assert layer_output.shape == (1, 10), "Dense layer should have expected output shape"
        assert sequential_output.shape == (1, 3), "Sequential should have expected output shape"

    def test_layer_output_as_network_input(self):
        """Test that Dense layer output can be used as network input."""
        # Create preprocessing layer
        preprocessor = Linear(5, 8)

        # Create network that processes preprocessor output
        network = Sequential([
            Linear(8, 12),
            ReLU(),
            Linear(12, 4)
        ])

        # Test pipeline: input → layer → network
        x = Tensor(np.random.randn(3, 5))
        preprocessed = preprocessor(x)
        final_output = network(preprocessed)

        # Verify pipeline interface
        assert isinstance(preprocessed, Tensor), "Layer should produce Tensor for network"
        assert isinstance(final_output, Tensor), "Network should accept layer output"
        assert final_output.shape == (3, 4), "Pipeline should work end-to-end"

    def test_network_layer_composition(self):
        """Test that networks can be composed with individual layers."""
        # Create base network
        base_network = Sequential([Linear(4, 6), ReLU(), Linear(6, 8)])

        # Add additional processing layer
        final_layer = Linear(8, 2)

        # Test composition
        x = Tensor(np.random.randn(2, 4))

        # Pipeline: input → network → layer
        network_output = base_network(x)
        final_output = final_layer(network_output)

        # Verify composition interface
        assert isinstance(network_output, Tensor), "Network should produce Tensor for layer"
        assert isinstance(final_output, Tensor), "Layer should accept network output"
        assert network_output.shape == (2, 8), "Network output should have expected shape"
        assert final_output.shape == (2, 2), "Layer should process network output correctly"


class TestLayerNetworkDataFlow:
    """Test data flow compatibility between layers and networks."""

    def test_shape_preservation_across_layer_network_boundary(self):
        """Test shape preservation when crossing layer-network boundaries."""
        shape_configs = [
            (1, 4, 8, 2),    # Single sample
            (5, 6, 10, 3),   # Small batch
            (10, 8, 16, 4),  # Larger batch
        ]

        for batch_size, input_size, hidden_size, output_size in shape_configs:
            # Create layer and network
            layer = Linear(input_size, hidden_size)
            network = Sequential([
                Linear(hidden_size, hidden_size),
            ReLU(),
                Linear(hidden_size, output_size)
            ])

            # Test data flow
            x = Tensor(np.random.randn(batch_size, input_size))
            layer_out = layer(x)
            network_out = network(layer_out)

            # Verify shape flow
            assert layer_out.shape == (batch_size, hidden_size), f"Layer should output correct shape for config {shape_configs}"
            assert network_out.shape == (batch_size, output_size), f"Network should output correct shape for config {shape_configs}"

    def test_dtype_normalization_across_layer_network_boundary(self):
        """Test that TinyTorch normalizes all data to float32 consistently."""
        # TinyTorch uses float32 for all operations (standard for deep learning)
        layer = Linear(4, 6)
        network = Sequential([Linear(6, 8), ReLU(), Linear(8, 2)])

        # Test with float32 input
        x_f32 = Tensor(np.random.randn(2, 4).astype(np.float32))
        layer_out_f32 = layer(x_f32)
        network_out_f32 = network(layer_out_f32)

        assert layer_out_f32.dtype == np.float32, "Layer output should be float32"
        assert network_out_f32.dtype == np.float32, "Network output should be float32"

        # Test with float64 input - should normalize to float32
        x_f64 = Tensor(np.random.randn(2, 4).astype(np.float64))
        layer_out_f64 = layer(x_f64)
        network_out_f64 = network(layer_out_f64)

        # TinyTorch normalizes to float32 for consistency and efficiency
        assert layer_out_f64.dtype == np.float32, "Layer normalizes to float32"
        assert network_out_f64.dtype == np.float32, "Network normalizes to float32"

    def test_error_handling_at_layer_network_boundary(self):
        """Test error handling when layer-network interfaces are incompatible."""
        # Create mismatched layer and network
        layer = Linear(4, 6)
        mismatched_network = Sequential([Linear(8, 2)])  # Expects 8, gets 6

        x = Tensor(np.random.randn(1, 4))
        layer_output = layer(x)  # Shape (1, 6)

        # Should fail gracefully with dimension mismatch
        try:
            result = mismatched_network(layer_output)  # Expects (1, 8)
            assert False, "Should have failed with dimension mismatch"
        except (ValueError, AssertionError, TypeError) as e:
            # Expected behavior
            assert isinstance(e, (ValueError, AssertionError, TypeError)), "Should fail gracefully with dimension mismatch"


class TestLayerNetworkSystemIntegration:
    """Test system-level integration scenarios with layers and networks."""

    def test_multi_stage_processing_pipeline(self):
        """Test multi-stage processing using layers and networks."""
        # Stage 1: Preprocessing layer
        preprocessor = Linear(8, 12)

        # Stage 2: Feature extraction network
        feature_extractor = Sequential([
            Linear(12, 16),
            ReLU(),
            Linear(16, 10)
        ])

        # Stage 3: Classification layer
        classifier = Linear(10, 3)

        # Test complete pipeline
        x = Tensor(np.random.randn(4, 8))

        preprocessed = preprocessor(x)
        features = feature_extractor(preprocessed)
        predictions = classifier(features)

        # Verify multi-stage integration
        assert isinstance(preprocessed, Tensor), "Preprocessor should output Tensor"
        assert isinstance(features, Tensor), "Feature extractor should output Tensor"
        assert isinstance(predictions, Tensor), "Classifier should output Tensor"
        assert predictions.shape == (4, 3), "Pipeline should produce expected final shape"

    def test_parallel_layer_processing(self):
        """Test parallel processing with multiple layers feeding into network."""
        # Create parallel processing layers
        branch1 = Linear(6, 4)
        branch2 = Linear(6, 4)
        branch3 = Linear(6, 4)

        # Fusion network
        fusion_network = Sequential([
            Linear(12, 8),  # 4+4+4=12 from parallel branches
            ReLU(),
            Linear(8, 2)
        ])

        # Test parallel processing
        x = Tensor(np.random.randn(2, 6))

        # Process in parallel
        out1 = branch1(x)
        out2 = branch2(x)
        out3 = branch3(x)

        # Manually concatenate (simulating fusion)
        # In a real implementation, this would be handled by a concatenation layer
        fused_data = np.concatenate([out1.data, out2.data, out3.data], axis=1)
        fused_tensor = Tensor(fused_data)

        # Final processing
        final_output = fusion_network(fused_tensor)

        # Verify parallel processing integration
        assert out1.shape == (2, 4), "Branch 1 should output correct shape"
        assert out2.shape == (2, 4), "Branch 2 should output correct shape"
        assert out3.shape == (2, 4), "Branch 3 should output correct shape"
        assert fused_tensor.shape == (2, 12), "Fusion should combine all branches"
        assert final_output.shape == (2, 2), "Final network should process fused input"

    def test_layer_network_modularity(self):
        """Test that layers and networks can be replaced modularly."""
        # Create modular components
        input_processors = [
            Linear(5, 8),
            Linear(5, 8),  # Different instance
        ]

        core_networks = [
            Sequential([Linear(8, 10), ReLU(), Linear(10, 6)]),
            Sequential([Linear(8, 6)]),  # Different architecture
        ]

        output_processors = [
            Linear(6, 3),
            Linear(6, 3),  # Different instance
        ]

        # Test all combinations work
        x = Tensor(np.random.randn(1, 5))

        for input_proc in input_processors:
            for core_net in core_networks:
                for output_proc in output_processors:
                    # Test modular pipeline
                    intermediate1 = input_proc(x)
                    intermediate2 = core_net(intermediate1)
                    final = output_proc(intermediate2)

                    # Verify modularity
                    assert isinstance(final, Tensor), "Modular combination should work"
                    assert final.shape == (1, 3), "Modular combination should produce expected output"


class TestLayerNetworkInterfaceStandards:
    """Test that layers and networks follow consistent interface standards."""

    def test_consistent_call_interface(self):
        """Test that layers and networks have consistent callable interface."""
        # Create different components
        components = [
            Linear(4, 6),
            Sequential([Linear(4, 6)]),
            Sequential([Linear(4, 8), ReLU(), Linear(8, 6)]),
            Sequential([Linear(4, 8), ReLU(), Linear(8, 6)])
        ]

        x = Tensor(np.random.randn(1, 4))

        # Test all components have consistent interface
        for component in components:
            # Should be callable with same signature
            result = component(x)

            # Verify consistent interface
            assert isinstance(result, Tensor), f"{type(component).__name__} should return Tensor"
            assert result.shape[0] == 1, f"{type(component).__name__} should preserve batch dimension"
            assert result.shape[1] == 6, f"{type(component).__name__} should produce expected output size"

    def test_component_property_consistency(self):
        """Test that layers and networks have consistent properties."""
        # Create components
        layer = Linear(3, 5)
        network = Sequential([Linear(3, 5)])
        multi_layer = Sequential([Linear(3, 4), ReLU(), Linear(4, 5)])

        # Test that all components can be used interchangeably
        x = Tensor(np.random.randn(2, 3))

        results = []
        for component in [layer, network, multi_layer]:
            result = component(x)
            results.append(result)

            # Verify consistent interface properties
            assert hasattr(result, 'shape'), f"{type(component).__name__} result should have shape"
            assert hasattr(result, 'data'), f"{type(component).__name__} result should have data"
            assert hasattr(result, 'dtype'), f"{type(component).__name__} result should have dtype"

        # All should produce same output shape
        expected_shape = (2, 5)
        for i, result in enumerate(results):
            assert result.shape == expected_shape, f"Component {i} should produce consistent shape"


if __name__ == "__main__":
    pytest.main([__file__])
