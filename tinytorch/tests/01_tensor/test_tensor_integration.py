"""
Module 01: Tensor - Integration Tests
Tests that Tensor works as foundation for all other modules
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTensorFoundation:
    """Test Tensor as foundation for the framework."""

    def test_tensor_import(self):
        """Test Tensor can be imported from package."""
        from tinytorch.core.tensor import Tensor
        assert Tensor is not None

    def test_tensor_creation(self):
        """Test various ways to create tensors."""
        from tinytorch.core.tensor import Tensor

        # From list
        t1 = Tensor([1, 2, 3])
        assert t1.shape == (3,)

        # From numpy array
        t2 = Tensor(np.array([[1, 2], [3, 4]]))
        assert t2.shape == (2, 2)

        # From other tensor data
        t3 = Tensor(t2.data)
        assert t3.shape == t2.shape

    def test_tensor_properties(self):
        """Test tensor properties work correctly."""
        from tinytorch.core.tensor import Tensor

        data = np.random.randn(3, 4, 5)
        t = Tensor(data)

        assert t.shape == (3, 4, 5)
        # TinyTorch uses float32 for efficiency
        assert t.dtype == np.float32
        assert np.allclose(t.data, data)

    def test_tensor_for_neural_networks(self):
        """Test tensor supports operations needed by neural networks."""
        from tinytorch.core.tensor import Tensor

        # Weights and inputs
        weights = Tensor(np.random.randn(10, 20))
        inputs = Tensor(np.random.randn(32, 10))

        # Should support matrix multiplication (key for Dense layers)
        # This tests if the tensor data can be used with numpy operations
        output_data = inputs.data @ weights.data
        output = Tensor(output_data)

        assert output.shape == (32, 20)


class TestTensorMemoryManagement:
    """Test tensor memory usage and copying behavior."""

    def test_tensor_memory_sharing(self):
        """Test tensor memory behavior."""
        from tinytorch.core.tensor import Tensor

        original_data = np.array([1, 2, 3])
        t = Tensor(original_data)

        # Tensor should maintain reference to data
        assert np.shares_memory(t.data, original_data) or np.array_equal(t.data, original_data)

    def test_tensor_copy_semantics(self):
        """Test tensor copying doesn't break."""
        from tinytorch.core.tensor import Tensor

        t1 = Tensor([1, 2, 3])
        t2 = Tensor(t1.data.copy())

        # Should be different tensors with same values
        assert np.array_equal(t1.data, t2.data)
        assert not np.shares_memory(t1.data, t2.data)


class TestTensorIntegrationReadiness:
    """Test tensor is ready for integration with other modules."""

    def test_ready_for_activations(self):
        """Test tensor works with activation-like operations."""
        from tinytorch.core.tensor import Tensor

        t = Tensor(np.array([-1, 0, 1, 2]))

        # Should support element-wise operations (for ReLU, Sigmoid, etc.)
        relu_result = Tensor(np.maximum(0, t.data))
        assert relu_result.shape == t.shape
        assert np.array_equal(relu_result.data, [0, 0, 1, 2])

    def test_ready_for_layers(self):
        """Test tensor works with layer-like operations."""
        from tinytorch.core.tensor import Tensor

        # Batch of inputs
        x = Tensor(np.random.randn(32, 784))  # MNIST-like
        weights = Tensor(np.random.randn(784, 128))
        bias = Tensor(np.random.randn(128))

        # Dense layer operation: x @ W + b
        output_data = x.data @ weights.data + bias.data
        output = Tensor(output_data)

        assert output.shape == (32, 128)

    def test_ready_for_spatial_operations(self):
        """Test tensor works with spatial/CNN operations."""
        from tinytorch.core.tensor import Tensor

        # Image tensor (batch, height, width, channels)
        image = Tensor(np.random.randn(8, 32, 32, 3))

        # Should support reshaping for spatial operations
        flattened = Tensor(image.data.reshape(8, -1))
        assert flattened.shape == (8, 32*32*3)

        # Should support slicing for convolution-like operations
        patch = Tensor(image.data[:, :3, :3, :])  # 3x3 patch
        assert patch.shape == (8, 3, 3, 3)

    def test_ready_for_autograd(self):
        """Test tensor is ready for gradient computation."""
        from tinytorch.core.tensor import Tensor

        # Should be able to create tensors that could track gradients
        t = Tensor(np.array([1.0, 2.0, 3.0]))

        # Should support operations that will need gradients
        squared = Tensor(t.data ** 2)
        assert squared.shape == t.shape
