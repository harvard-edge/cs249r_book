"""
Integration Tests - Tensor and Activations

Tests cross-module interfaces and compatibility between Tensor and Activation modules.
Focuses on integration, not re-testing individual module functionality.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax


class TestTensorActivationInterface:
    """Test interface compatibility between Tensor and Activation modules."""

    def test_activation_accepts_tensor_input(self):
        """Test that activation functions accept Tensor objects as input."""
        activations = [ReLU(), Sigmoid(), Tanh()]

        # Test with different tensor shapes
        test_tensors = [
            Tensor([1.0, -1.0, 0.0]),                    # 1D tensor
            Tensor([[1.0, -1.0], [0.0, 2.0]]),          # 2D tensor
            Tensor([[[1.0], [-1.0]], [[0.0], [2.0]]]),  # 3D tensor
        ]

        for activation in activations:
            for tensor in test_tensors:
                # Test interface: activation should accept Tensor
                result = activation(tensor)

                # Verify interface compatibility
                assert isinstance(result, Tensor), f"{type(activation).__name__} should return Tensor"
                assert result.shape == tensor.shape, f"{type(activation).__name__} should preserve shape"
                assert result.dtype == tensor.dtype, f"{type(activation).__name__} should preserve dtype"

    def test_softmax_tensor_interface(self):
        """Test Softmax interface with Tensor objects."""
        softmax = Softmax()

        # Test with different tensor configurations
        test_cases = [
            Tensor([[1.0, 2.0, 3.0]]),           # Single batch
            Tensor([[1.0, 2.0], [3.0, 4.0]]),   # Multiple samples
        ]

        for tensor in test_cases:
            result = softmax(tensor)

            # Verify interface compatibility
            assert isinstance(result, Tensor), "Softmax should return Tensor"
            assert result.shape == tensor.shape, "Softmax should preserve shape"
            assert result.dtype in [np.float32, np.float64], "Softmax should return float type"

    def test_activation_output_tensor_compatibility(self):
        """Test that activation outputs are compatible with further Tensor operations."""
        relu = ReLU()

        # Create input tensor
        x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

        # Apply activation
        activated = relu(x)

        # Test that output can be used in Tensor operations
        doubled = activated * Tensor([[2.0, 2.0, 2.0, 2.0, 2.0]])
        summed = activated + Tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

        # Verify interface compatibility
        assert isinstance(doubled, Tensor), "Activation output should work with Tensor multiplication"
        assert isinstance(summed, Tensor), "Activation output should work with Tensor addition"
        assert doubled.shape == activated.shape, "Tensor operations should preserve shape"
        assert summed.shape == activated.shape, "Tensor operations should preserve shape"

    def test_chained_activations_with_tensors(self):
        """Test chaining multiple activations with Tensor interface."""
        # Create activation chain
        relu = ReLU()
        sigmoid = Sigmoid()

        # Test chaining
        x = Tensor([[1.0, -1.0, 2.0, -2.0]])

        # Chain: input → ReLU → Sigmoid
        relu_output = relu(x)
        sigmoid_output = sigmoid(relu_output)

        # Verify chaining works
        assert isinstance(relu_output, Tensor), "First activation should return Tensor"
        assert isinstance(sigmoid_output, Tensor), "Second activation should accept first activation output"
        assert sigmoid_output.shape == x.shape, "Chained activations should preserve shape"


class TestTensorActivationDataTypes:
    """Test data type compatibility between Tensor and Activation modules."""

    def test_float32_tensor_activation_compatibility(self):
        """Test activations with float32 Tensor inputs."""
        activations = [ReLU(), Sigmoid(), Tanh()]

        x_f32 = Tensor(np.array([1.0, -1.0, 0.0]).astype(np.float32))

        for activation in activations:
            result = activation(x_f32)

            # Verify dtype preservation
            assert result.dtype == np.float32, f"{type(activation).__name__} should preserve float32"
            assert isinstance(result, Tensor), f"{type(activation).__name__} should return Tensor"

    def test_float64_tensor_activation_compatibility(self):
        """Test activations with float64 Tensor inputs."""
        activations = [ReLU(), Sigmoid(), Tanh()]

        x_f64 = Tensor(np.array([1.0, -1.0, 0.0]).astype(np.float64))

        for activation in activations:
            result = activation(x_f64)

            # TinyTorch uses float32 for efficiency - verify it works regardless of input dtype
            assert result.dtype == np.float32, f"{type(activation).__name__} should output float32"
            assert isinstance(result, Tensor), f"{type(activation).__name__} should return Tensor"

    def test_integer_tensor_activation_compatibility(self):
        """Test activations with integer Tensor inputs."""
        relu = ReLU()  # ReLU should work with integers

        x_int = Tensor([1, -1, 0, 2])  # Integer tensor
        result = relu(x_int)

        # Verify interface handles integer input
        assert isinstance(result, Tensor), "ReLU should return Tensor for integer input"
        assert result.shape == x_int.shape, "ReLU should preserve shape for integer input"


class TestActivationTensorSystemIntegration:
    """Test system-level integration scenarios with Tensor and Activation."""

    def test_tensor_activation_tensor_roundtrip(self):
        """Test Tensor → Activation → Tensor operations roundtrip."""
        activations = [ReLU(), Sigmoid(), Tanh()]

        original_tensor = Tensor([[1.0, -1.0, 0.5, -0.5]])

        for activation in activations:
            # Apply activation
            activated = activation(original_tensor)

            # Use in further tensor operations
            scaled = activated * Tensor([[2.0, 2.0, 2.0, 2.0]])
            final = scaled + Tensor([[1.0, 1.0, 1.0, 1.0]])

            # Verify complete workflow
            assert isinstance(final, Tensor), f"{type(activation).__name__} workflow should produce Tensor"
            assert final.shape == original_tensor.shape, f"{type(activation).__name__} workflow should preserve shape"

    def test_multiple_tensor_activation_operations(self):
        """Test multiple activation operations in sequence."""
        # Create multiple tensors
        tensors = [
            Tensor([[1.0, 2.0]]),
            Tensor([[-1.0, -2.0]]),
            Tensor([[0.0, 3.0]]),
        ]

        relu = ReLU()

        # Apply activation to all tensors
        results = []
        for tensor in tensors:
            result = relu(tensor)
            results.append(result)

        # Verify all operations work
        for i, result in enumerate(results):
            assert isinstance(result, Tensor), f"Activation {i} should return Tensor"
            assert result.shape == tensors[i].shape, f"Activation {i} should preserve shape"

        # Test combining results
        combined = results[0] + results[1] + results[2]
        assert isinstance(combined, Tensor), "Combined activation results should create Tensor"

    def test_activation_error_handling_with_tensors(self):
        """Test activation error handling with edge case Tensors."""
        relu = ReLU()

        # Test with empty tensor
        try:
            empty_tensor = Tensor(np.array([]))
            result = relu(empty_tensor)
            assert isinstance(result, Tensor), "Should handle empty tensor gracefully"
            assert result.shape == empty_tensor.shape, "Should preserve empty shape"
        except (ValueError, TypeError) as e:
            # Expected behavior - should fail gracefully
            assert isinstance(e, (ValueError, TypeError)), "Should fail gracefully with empty tensor"

        # Test with single element
        single_tensor = Tensor([5.0])
        result = relu(single_tensor)
        assert isinstance(result, Tensor), "Should handle single element tensor"
        assert result.shape == (1,), "Should preserve single element shape"


class TestActivationInterfaceCompatibility:
    """Test activation function interface compatibility requirements."""

    def test_activation_preserves_tensor_properties(self):
        """Test that activations preserve essential Tensor properties."""
        activations = [ReLU(), Sigmoid(), Tanh()]

        # Test tensor with specific properties
        original = Tensor([[1.0, -1.0], [2.0, -2.0]])

        for activation in activations:
            result = activation(original)

            # Verify property preservation
            assert hasattr(result, 'shape'), f"{type(activation).__name__} result should have shape property"
            assert hasattr(result, 'data'), f"{type(activation).__name__} result should have data property"
            assert hasattr(result, 'dtype'), f"{type(activation).__name__} result should have dtype property"

            # Verify properties are accessible
            assert result.shape == original.shape, f"{type(activation).__name__} should preserve shape property"
            assert isinstance(result.data, np.ndarray), f"{type(activation).__name__} should have numpy data"

    def test_softmax_special_interface_requirements(self):
        """Test Softmax special interface requirements."""
        softmax = Softmax()

        # Softmax needs 2D input for proper operation
        x_2d = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = softmax(x_2d)

        # Verify softmax interface requirements
        assert isinstance(result, Tensor), "Softmax should return Tensor"
        assert result.shape == x_2d.shape, "Softmax should preserve 2D shape"
        assert len(result.shape) == 2, "Softmax should maintain 2D structure"

    def test_activation_batch_compatibility(self):
        """Test activation compatibility with batch processing."""
        relu = ReLU()

        # Simulate batch processing
        batch_size = 3
        feature_size = 4

        batch_tensor = Tensor(np.random.randn(batch_size, feature_size))
        batch_result = relu(batch_tensor)

        # Verify batch compatibility
        assert isinstance(batch_result, Tensor), "Activation should handle batch Tensors"
        assert batch_result.shape == (batch_size, feature_size), "Activation should preserve batch dimensions"

        # Test processing individual batch items
        for i in range(batch_size):
            item_tensor = Tensor(batch_tensor.data[i:i+1])
            item_result = relu(item_tensor)
            assert isinstance(item_result, Tensor), f"Activation should handle batch item {i}"


if __name__ == "__main__":
    pytest.main([__file__])
