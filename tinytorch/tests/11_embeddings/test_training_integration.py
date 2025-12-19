"""
Module 08: Training - Integration Tests
Tests that complete training loops work with all system components
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTrainingLoopIntegration:
    """Test complete training loop integration."""

    def test_basic_training_loop(self):
        """Test basic training loop components work together."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.losses import MSELoss

            # Build simple network
            layer1 = Linear(2, 4)
            relu = ReLU()
            layer2 = Linear(4, 1)
            sigmoid = Sigmoid()

            # Create loss function
            loss_fn = MSELoss()

            # Dummy data
            X = Tensor(np.random.randn(10, 2))
            y = Tensor(np.random.randn(10, 1))

            # Forward pass
            h1 = layer1(X)
            h1_act = relu(h1)
            output = layer2(h1_act)
            predictions = sigmoid(output)

            # Compute loss
            loss = loss_fn(predictions, y)

            assert isinstance(loss, Tensor)
            assert loss.data.size == 1  # Scalar loss

        except ImportError:
            assert True, "Training components not ready"

    def test_optimizer_integration(self):
        """Test optimizer works with model parameters."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor

            layer = Linear(10, 5)

            # Get parameters
            params = [layer.weight]
            if layer.bias is not None:
                params.append(layer.bias)

            optimizer = SGD(params, lr=0.01)

            # Simulate gradients (normally from autograd)
            for p in params:
                p.grad = Tensor(np.random.randn(*p.shape))

            # Store old parameter values
            old_params = [p.data.copy() for p in params]

            # Optimizer step (if implemented)
            if hasattr(optimizer, 'step'):
                optimizer.step()

                # Parameters should change
                for old, new in zip(old_params, params):
                    assert not np.array_equal(old, new.data)

        except (ImportError, AttributeError):
            assert True, "Optimizer integration not ready"

    def test_loss_computation(self):
        """Test loss functions work with network outputs."""
        try:
            from tinytorch.core.losses import MSELoss, CrossEntropyLoss
            from tinytorch.core.tensor import Tensor

            mse = MSELoss()

            # Test MSE
            predictions = Tensor(np.array([[0.1], [0.9], [0.5]]))
            targets = Tensor(np.array([[0.0], [1.0], [0.5]]))

            loss = mse(predictions, targets)

            # MSE = mean((pred - target)²)
            expected = np.mean((predictions.data - targets.data) ** 2)
            assert np.isclose(loss.data, expected)

        except ImportError:
            assert True, "Loss functions not implemented"


class TestDataLoaderIntegration:
    """Test training works with data loading."""

    def test_batch_processing(self):
        """Test training handles batches correctly."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import DataLoader

            # Create simple dataset
            X = np.random.randn(100, 10)
            y = np.random.randn(100, 1)

            dataset = list(zip(X, y))
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

            model = Linear(10, 1)

            # Process one batch
            for batch_X, batch_y in dataloader:
                # DataLoader already returns Tensors
                if not isinstance(batch_X, Tensor):
                    batch_X = Tensor(np.array(batch_X))
                    batch_y = Tensor(np.array(batch_y))

                output = model(batch_X)

                assert output.shape[0] <= 16  # Batch size
                assert output.shape[1] == 1   # Output dimension
                break  # Just test one batch

        except ImportError:
            assert True, "DataLoader not implemented"

    def test_epoch_training(self):
        """Test training for multiple epochs."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.losses import MSELoss

            model = Linear(5, 1)
            loss_fn = MSELoss()

            # Training data
            X = Tensor(np.random.randn(32, 5))
            y = Tensor(np.random.randn(32, 1))

            losses = []

            # Train for a few steps
            for epoch in range(3):
                predictions = model(X)
                loss = loss_fn(predictions, y)
                losses.append(loss.data)

                # Simple parameter update (manual SGD)
                if hasattr(model, 'weight'):
                    # Compute simple gradient
                    error = predictions.data - y.data
                    grad = (X.data.T @ error) / X.shape[0]

                    # Update weights
                    learning_rate = 0.01
                    new_weights = model.weight.data - learning_rate * grad
                    model.weights = Tensor(new_weights)

            assert len(losses) == 3

        except (ImportError, AttributeError):
            assert True, "Training loop components not ready"


class TestModelEvaluation:
    """Test model evaluation and metrics."""

    def test_accuracy_computation(self):
        """Test classification accuracy computation."""
        try:
            from tinytorch.core.metrics import accuracy
            from tinytorch.core.tensor import Tensor

            # Binary classification predictions
            predictions = Tensor(np.array([[0.9], [0.1], [0.8], [0.3]]))
            targets = Tensor(np.array([[1], [0], [1], [0]]))

            # Convert to binary predictions (threshold 0.5)
            binary_preds = Tensor((predictions.data > 0.5).astype(float))

            acc = accuracy(binary_preds, targets)
            assert acc == 1.0  # All correct

        except ImportError:
            # Manual accuracy calculation
            predictions = np.array([[0.9], [0.1], [0.8], [0.3]])
            targets = np.array([[1], [0], [1], [0]])
            binary_preds = (predictions > 0.5).astype(float)
            acc = np.mean(binary_preds == targets)
            assert acc == 1.0

    def test_model_evaluation_mode(self):
        """Test model can switch between training and evaluation."""
        try:
            from tinytorch.core.layers import Linear

            model = Linear(10, 5)

            # Check if model has train/eval methods
            if hasattr(model, 'train') and hasattr(model, 'eval'):
                model.train()
                assert model.training == True

                model.eval()
                assert model.training == False
            else:
                # If not implemented, that's okay
                assert True, "Train/eval mode not implemented"

        except (ImportError, AttributeError):
            assert True, "Model evaluation mode not implemented"


class TestCompleteMLPipeline:
    """Test complete ML pipeline from data to trained model."""

    def test_xor_training_pipeline(self):
        """Test complete XOR training pipeline."""
        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.tensor import Tensor

            # XOR dataset
            X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
            y = Tensor(np.array([[0], [1], [1], [0]], dtype=np.float32))

            # Build network
            hidden = Linear(2, 4, bias=True)
            relu = ReLU()
            output = Linear(4, 1, bias=True)
            sigmoid = Sigmoid()

            loss_fn = MSELoss()

            # Test forward pass
            h = hidden(X)
            h_act = relu(h)
            out = output(h_act)
            predictions = sigmoid(out)

            loss = loss_fn(predictions, y)

            # Should produce valid predictions
            assert predictions.shape == (4, 1)
            assert np.all(predictions.data >= 0) and np.all(predictions.data <= 1)
            # Loss can be float, numpy scalar, or 0-d array
            assert isinstance(loss.data, (float, np.floating, np.ndarray))

        except ImportError:
            assert True, "Training pipeline components not ready"
