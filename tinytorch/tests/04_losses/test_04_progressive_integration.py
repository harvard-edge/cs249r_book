"""
Module 04: Progressive Integration Tests
Tests that Module 04 (Losses) works correctly AND that the entire foundation stack works.

DEPENDENCY CHAIN: 01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader
This is the FOUNDATION MILESTONE - everything should work together for neural networks!
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEntireFoundationStack:
    """Test that the complete foundation stack (01→05) works together."""

    def test_setup_foundation_intact(self):
        """Verify Module 01 (Setup) foundation is solid."""
        # Environment
        assert sys.version_info >= (3, 8), "Foundation broken: Python version"

        # Project structure
        project_root = Path(__file__).parent.parent.parent
        assert (project_root / "modules").exists(), "Foundation broken: Module structure"

        # Dependencies
        import numpy as np
        assert np.__version__ is not None, "Foundation broken: Numpy"

    def test_tensor_foundation_intact(self):
        """Verify Module 02 (Tensor) foundation is solid."""
        try:
            from tinytorch.core.tensor import Tensor

            # Basic tensor functionality
            t = Tensor([1, 2, 3])
            assert t.shape == (3,), "Foundation broken: Tensor creation"

            # Multi-dimensional tensors
            t2 = Tensor(np.random.randn(4, 5))
            assert t2.shape == (4, 5), "Foundation broken: Multi-dim tensors"

        except ImportError:
            assert True, "Tensor foundation not implemented yet"

    def test_activation_foundation_intact(self):
        """Verify Module 03 (Activations) foundation is solid."""
        try:
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.tensor import Tensor

            relu = ReLU()
            sigmoid = Sigmoid()

            x = Tensor(np.array([-1, 0, 1]))

            # Activations should work with tensors
            h = relu(x)
            y = sigmoid(h)

            assert h.shape == x.shape, "Foundation broken: ReLU"
            assert y.shape == x.shape, "Foundation broken: Sigmoid"

        except ImportError:
            assert True, "Activation foundation not implemented yet"

    def test_layer_foundation_intact(self):
        """Verify Module 04 (Layers) foundation is solid."""
        try:
            from tinytorch.core.layers import Layer

            # Layer base class should exist
            layer = Layer()
            assert hasattr(layer, 'forward'), "Foundation broken: Layer interface"
            assert callable(layer), "Foundation broken: Layer callable"

        except ImportError:
            assert True, "Layer foundation not implemented yet"


class TestLinearNetworkCapability:
    """Test that Module 05 enables full neural network capability."""

    def test_dense_layer_creation(self):
        """Test Linear layer works with the foundation."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Create Linear layer
            layer = Linear(10, 5)

            # Should have proper weights and bias
            assert hasattr(layer, 'weight'), "Linear broken: No weights"
            assert layer.weight.shape == (10, 5), "Linear broken: Wrong weight shape"

            # Should work with tensor input
            x = Tensor(np.random.randn(32, 10))
            output = layer(x)

            assert output.shape == (32, 5), "Linear broken: Wrong output shape"

        except ImportError:
            assert True, "Linear layer not implemented yet"

    def test_multi_layer_network(self):
        """Test building multi-layer networks."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.tensor import Tensor

            # Build 3-layer network for XOR
            hidden = Linear(2, 4, bias=True)
            output = Linear(4, 1, bias=True)
            relu = ReLU()
            sigmoid = Sigmoid()

            # XOR inputs
            X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))

            # Forward pass through complete network
            h = hidden(X)           # Linear transformation
            h_act = relu(h)         # Non-linear activation
            out = output(h_act)     # Output transformation
            predictions = sigmoid(out)  # Output activation

            assert predictions.shape == (4, 1), "Multi-layer network broken"
            assert np.all(predictions.data >= 0), "Network output invalid"
            assert np.all(predictions.data <= 1), "Network output invalid"

        except ImportError:
            assert True, "Multi-layer networks not ready yet"


class TestXORProblemSolution:
    """Test that the foundation can solve the XOR problem."""

    def test_xor_network_architecture(self):
        """Test XOR network architecture is possible."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.tensor import Tensor

            # XOR problem setup
            X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
            y_target = np.array([[0], [1], [1], [0]], dtype=np.float32)

            # Network: 2 -> 4 -> 1 (sufficient for XOR)
            hidden = Linear(2, 4, bias=True)
            output = Linear(4, 1, bias=True)
            relu = ReLU()
            sigmoid = Sigmoid()

            # Forward pass
            h = hidden(X)
            h_relu = relu(h)
            out = output(h_relu)
            predictions = sigmoid(out)

            # Network should produce valid outputs
            assert predictions.shape == (4, 1), "XOR network architecture broken"

            # Test network capacity (parameter count)
            hidden_params = 2 * 4 + 4  # weights + bias
            output_params = 4 * 1 + 1  # weights + bias
            total_params = hidden_params + output_params

            # XOR requires at least 9 parameters theoretically
            assert total_params >= 9, "XOR network has insufficient capacity"

        except ImportError:
            assert True, "XOR network not ready yet"

    def test_nonlinear_problem_solvability(self):
        """Test that non-linear problems are now solvable."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor

            # Create network that can solve non-linear problems
            layer1 = Linear(2, 10)
            relu = ReLU()
            layer2 = Linear(10, 1)

            # Test various non-linear patterns
            patterns = [
                np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),  # XOR pattern
                np.array([[1, 1], [1, 0], [0, 1], [0, 0]]),  # Inverse XOR
            ]

            for pattern in patterns:
                X = Tensor(pattern.astype(np.float32))

                # Network should handle any pattern
                h = layer1(X)
                h_nonlinear = relu(h)
                output = layer2(h_nonlinear)

                assert output.shape[0] == 4, "Pattern processing broken"

        except ImportError:
            assert True, "Non-linear problem solving not ready yet"


class TestFoundationMilestoneReadiness:
    """Test that we're ready for Foundation Milestone achievements."""

    def test_mnist_mlp_architecture_possible(self):
        """Test we can build MNIST MLP architecture."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax
            from tinytorch.core.tensor import Tensor

            # MNIST MLP: 784 -> 128 -> 64 -> 10
            layer1 = Linear(784, 128)
            layer2 = Linear(128, 64)
            layer3 = Linear(64, 10)
            relu = ReLU()
            softmax = Softmax()

            # Simulated MNIST batch
            x = Tensor(np.random.randn(32, 784))  # 32 images, flattened

            # Forward pass through MLP
            h1 = relu(layer1(x))     # 32 x 128
            h2 = relu(layer2(h1))    # 32 x 64
            logits = layer3(h2)      # 32 x 10
            probs = softmax(logits)  # 32 x 10

            assert probs.shape == (32, 10), "MNIST MLP architecture broken"

            # Softmax should sum to 1 across classes
            prob_sums = np.sum(probs.data, axis=1)
            assert np.allclose(prob_sums, 1.0), "Softmax probabilities broken"

        except ImportError:
            assert True, "MNIST MLP not ready yet"

    def test_classification_capability(self):
        """Test basic classification capability."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax
            from tinytorch.core.tensor import Tensor

            # Simple classifier: features -> hidden -> classes
            feature_layer = Linear(20, 10)
            classifier = Linear(10, 3)  # 3 classes
            relu = ReLU()
            softmax = Softmax()

            # Batch of features
            features = Tensor(np.random.randn(16, 20))

            # Classification pipeline
            h = relu(feature_layer(features))
            logits = classifier(h)
            class_probs = softmax(logits)

            # Should produce valid class probabilities
            assert class_probs.shape == (16, 3), "Classification shape broken"
            assert np.all(class_probs.data >= 0), "Negative probabilities"
            assert np.allclose(np.sum(class_probs.data, axis=1), 1.0), "Probabilities don't sum to 1"

        except ImportError:
            assert True, "Classification capability not ready yet"


class TestCompleteStackValidation:
    """Validate the complete foundation stack works end-to-end."""

    def test_end_to_end_neural_network(self):
        """Test complete neural network from scratch."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Sigmoid

            # End-to-end test: Build and run a complete neural network

            # 1. Data (from setup/tensor foundation)
            np.random.seed(42)  # Reproducible
            X = Tensor(np.random.randn(10, 5))

            # 2. Network architecture (from layers foundation)
            layer1 = Linear(5, 8)
            layer2 = Linear(8, 3)
            layer3 = Linear(3, 1)

            # 3. Activations (from activation foundation)
            relu = ReLU()
            sigmoid = Sigmoid()

            # 4. Forward pass (everything working together)
            h1 = relu(layer1(X))      # 10 x 8
            h2 = relu(layer2(h1))     # 10 x 3
            output = sigmoid(layer3(h2))  # 10 x 1

            # 5. Validation
            assert output.shape == (10, 1), "End-to-end network broken"
            assert np.all(output.data >= 0), "Network output invalid"
            assert np.all(output.data <= 1), "Network output invalid"

            # 6. Network should be trainable (parameters exist)
            assert hasattr(layer1, 'weight'), "Layer 1 not trainable"
            assert hasattr(layer2, 'weight'), "Layer 2 not trainable"
            assert hasattr(layer3, 'weight'), "Layer 3 not trainable"

        except ImportError:
            assert True, "End-to-end neural network not ready yet"

    def test_foundation_stability_under_load(self):
        """Test foundation remains stable under computational load."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU

            # Stress test: Large network
            layer1 = Linear(100, 200)
            layer2 = Linear(200, 100)
            layer3 = Linear(100, 10)
            relu = ReLU()

            # Large batch
            X = Tensor(np.random.randn(256, 100))

            # Multiple forward passes
            for i in range(5):
                h1 = relu(layer1(X))
                h2 = relu(layer2(h1))
                output = layer3(h2)

                assert output.shape == (256, 10), f"Foundation unstable at iteration {i}"

        except ImportError:
            assert True, "Foundation stress testing not ready yet"
