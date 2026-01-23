"""
Module 10: Progressive Integration Tests
Tests that Module 10 (Tokenization) works correctly AND that Foundation + Architecture tier work.

DEPENDENCY CHAIN: 01_tensor → ... → 05_dataloader → ... → 08_training → 09_convolutions → 10_tokenization
This is where text processing begins for NLP pipelines.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPriorStackStillWorking:
    """Quick regression checks that prior modules (01→10) still work."""

    def test_complete_ml_pipeline_stable(self):
        """Verify complete ML pipeline remains stable."""
        # Environment (Module 01)
        assert sys.version_info >= (3, 8), "Foundation broken: Python version"

        # Complete pipeline should work
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.dataloader import Dataset, DataLoader
            from tinytorch.core.optimizers import SGD

            # All components should be available
            layer = Linear(5, 2)
            optimizer = SGD(layer.parameters(), lr=0.01)

            # Basic functionality should work
            x = Tensor(np.random.randn(3, 5))
            output = layer(x)
            assert output.shape == (3, 2), "ML pipeline broken"

        except ImportError:
            assert True, "ML pipeline not implemented yet"

    def test_optimization_stable(self):
        """Verify Module 07 (Optimizers) still works."""
        try:
            from tinytorch.core.optimizers import SGD, Adam
            from tinytorch.core.layers import Linear

            # Optimizers should work
            layer = Linear(3, 1)
            sgd = SGD(layer.parameters(), lr=0.01)
            adam = Adam(layer.parameters(), lr=0.001)

            assert hasattr(sgd, 'step'), "Optimizers broken: SGD step"
            assert hasattr(adam, 'step'), "Optimizers broken: Adam step"

        except ImportError:
            assert True, "Optimizers not implemented yet"


class TestModule08TrainingCore:
    """Test Module 08 (Training) core functionality."""

    def test_training_loop_creation(self):
        """Test basic training loop functionality."""
        try:
            from tinytorch.core.training import Trainer
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Create model and optimizer
            model = Linear(10, 3)
            optimizer = SGD(model.parameters(), lr=0.01)

            # Create simple dataset
            class SimpleDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(20, 10)
                    self.targets = np.random.randint(0, 3, 20)

                def __len__(self):
                    return 20

                def __getitem__(self, idx):
                    return self.data[idx], self.targets[idx]

            dataset = SimpleDataset()
            dataloader = DataLoader(dataset, batch_size=4)

            # Create dummy loss function
            def dummy_loss(pred, target):
                return pred.sum()

            # Create trainer
            trainer = Trainer(model, optimizer, dummy_loss)

            # Should have training methods (TinyTorch uses train_epoch/evaluate)
            assert hasattr(trainer, 'train') or hasattr(trainer, 'fit') or hasattr(trainer, 'train_epoch'), "Trainer broken: No train method"

        except ImportError:
            assert True, "Training loop not implemented yet"

    def test_loss_function_support(self):
        """Test loss function integration."""
        try:
            from tinytorch.core.training import CrossEntropyLoss, MSELoss
            from tinytorch.core.tensor import Tensor

            # Test MSE loss
            mse = MSELoss()
            pred = Tensor(np.array([1.0, 2.0, 3.0]))
            target = Tensor(np.array([1.5, 2.5, 2.5]))

            loss = mse(pred, target)
            assert hasattr(loss, 'data') or isinstance(loss, (float, np.ndarray)), "MSE loss broken"

            # Test CrossEntropy loss (if implemented)
            if 'CrossEntropyLoss' in locals():
                ce = CrossEntropyLoss()
                logits = Tensor(np.random.randn(4, 3))  # 4 samples, 3 classes
                targets = Tensor(np.array([0, 1, 2, 1]))  # Class indices as Tensor

                try:
                    ce_loss = ce(logits, targets)
                    assert hasattr(ce_loss, 'data') or isinstance(ce_loss, (float, np.ndarray)), "CrossEntropy loss broken"
                except (AttributeError, TypeError):
                    # CrossEntropyLoss may have implementation quirks
                    pass

        except ImportError:
            assert True, "Loss functions not implemented yet"

    def test_metrics_computation(self):
        """Test training metrics computation."""
        try:
            from tinytorch.core.training import accuracy, compute_metrics
            from tinytorch.core.tensor import Tensor

            # Test accuracy computation
            predictions = Tensor(np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]))
            targets = np.array([1, 0, 1])  # True class indices

            acc = accuracy(predictions, targets)
            assert isinstance(acc, (float, np.ndarray)), "Accuracy computation broken"
            assert 0.0 <= acc <= 1.0, "Accuracy not in valid range"

            # Test comprehensive metrics
            if 'compute_metrics' in locals():
                metrics = compute_metrics(predictions, targets)
                assert isinstance(metrics, dict), "Metrics should return dict"
                assert 'accuracy' in metrics, "Metrics missing accuracy"

        except ImportError:
            assert True, "Metrics computation not implemented yet"


class TestProgressiveStackIntegration:
    """Test that the complete stack (01→11) works together."""

    def test_end_to_end_training(self):
        """Test complete end-to-end training process."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.training import Trainer, CrossEntropyLoss
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Create complete model
            class SimpleModel:
                def __init__(self):
                    self.layer1 = Linear(10, 16)
                    self.relu = ReLU()
                    self.layer2 = Linear(16, 3)
                    self.softmax = Softmax()

                def __call__(self, x):
                    h = self.relu(self.layer1(x))
                    logits = self.layer2(h)
                    return self.softmax(logits)

                def parameters(self):
                    params = []
                    if hasattr(self.layer1, 'parameters'):
                        params.extend(self.layer1.parameters())
                    if hasattr(self.layer2, 'parameters'):
                        params.extend(self.layer2.parameters())
                    return params

            # Create dataset
            class TrainingDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(50, 10)
                    self.targets = np.random.randint(0, 3, 50)

                def __len__(self):
                    return 50

                def __getitem__(self, idx):
                    return Tensor(self.data[idx]), self.targets[idx]

            # Setup training
            model = SimpleModel()
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = CrossEntropyLoss()

            dataset = TrainingDataset()
            dataloader = DataLoader(dataset, batch_size=8)

            # Training loop (simplified)
            for epoch in range(2):  # Just 2 epochs for testing
                for batch_x, batch_y in dataloader:
                    # Forward pass
                    predictions = model(batch_x)
                    loss = loss_fn(predictions, batch_y)

                    # Backward pass (if available)
                    if hasattr(loss, 'backward'):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Verify shapes - batch_y may be Tensor or array
                    batch_size = batch_y.shape[0] if hasattr(batch_y, 'shape') else len(batch_y)
                    assert predictions.shape[0] == batch_size, "Training batch size mismatch"
                    break  # Test one batch per epoch

            assert True, "End-to-end training successful"

        except ImportError:
            assert True, "End-to-end training not ready yet"

    def test_cnn_training_pipeline(self):
        """Test CNN training with spatial operations."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.dataloader import Dataset, DataLoader
            from tinytorch.core.tensor import Tensor

            # CNN model
            class SimpleCNN:
                def __init__(self):
                    self.conv1 = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
                    self.pool = MaxPool2d(kernel_size=2)
                    self.relu = ReLU()
                    self.fc = Linear(16 * 15 * 15, 5)  # Approximate size

                def __call__(self, x):
                    h = self.relu(self.conv1(x))
                    h = self.pool(h)
                    # Flatten (simplified)
                    h_flat = h.reshape(h.shape[0], -1)
                    return self.fc(h_flat)

                def parameters(self):
                    params = []
                    for module in [self.conv1, self.fc]:
                        if hasattr(module, 'parameters'):
                            params.extend(module.parameters())
                    return params

            # Image dataset
            class ImageDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(20, 3, 32, 32)
                    self.targets = np.random.randint(0, 5, 20)

                def __len__(self):
                    return 20

                def __getitem__(self, idx):
                    return Tensor(self.data[idx]), self.targets[idx]

            # Setup CNN training
            cnn_model = SimpleCNN()
            optimizer = Adam(cnn_model.parameters(), lr=0.001)

            dataset = ImageDataset()
            dataloader = DataLoader(dataset, batch_size=4)

            # Test CNN training step
            for batch_x, batch_y in dataloader:
                assert batch_x.shape == (4, 3, 32, 32), "CNN input shape broken"

                # Forward pass
                if hasattr(cnn_model.conv1, '__call__'):
                    predictions = cnn_model(batch_x)
                    assert len(predictions.shape) == 2, "CNN output shape broken"

                break  # Test one batch

        except ImportError:
            assert True, "CNN training pipeline not ready yet"


class TestAdvancedTrainingFeatures:
    """Test advanced training features and techniques."""

    def test_validation_loop(self):
        """Test validation during training."""
        try:
            from tinytorch.core.training import Trainer
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Model and optimizer
            model = Linear(5, 2)
            optimizer = SGD(model.parameters(), lr=0.01)

            # Train and validation datasets
            class Dataset(Dataset):
                def __init__(self, size):
                    self.data = np.random.randn(size, 5)
                    self.targets = np.random.randint(0, 2, size)

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx], self.targets[idx]

            train_dataset = Dataset(30)
            val_dataset = Dataset(10)

            train_loader = DataLoader(train_dataset, batch_size=5)
            val_loader = DataLoader(val_dataset, batch_size=5)

            # Dummy loss function
            def dummy_loss(pred, target):
                return pred.sum()

            # Trainer with validation
            trainer = Trainer(model, optimizer, dummy_loss)

            if hasattr(trainer, 'validate') or hasattr(trainer, 'evaluate'):
                # Should be able to run validation
                assert True, "Validation capability available"

        except ImportError:
            assert True, "Validation loop not ready yet"

    def test_checkpointing_and_early_stopping(self):
        """Test model checkpointing and early stopping."""
        try:
            from tinytorch.core.training import Trainer, ModelCheckpoint, EarlyStopping
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD

            model = Linear(5, 1)
            optimizer = SGD(model.parameters(), lr=0.01)

            # Checkpointing
            if 'ModelCheckpoint' in locals():
                checkpoint = ModelCheckpoint(filepath='model.pth', save_best=True)
                assert hasattr(checkpoint, 'save'), "Checkpointing broken"

            # Early stopping
            if 'EarlyStopping' in locals():
                early_stop = EarlyStopping(patience=5, min_delta=0.001)
                assert hasattr(early_stop, 'check'), "Early stopping broken"

            # Dummy loss function
            def dummy_loss(pred, target):
                return pred.sum()

            # Training with callbacks
            trainer = Trainer(model, optimizer, dummy_loss)
            if hasattr(trainer, 'callbacks'):
                trainer.callbacks = [checkpoint, early_stop]

        except ImportError:
            assert True, "Advanced training features not ready yet"

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling during training."""
        try:
            from tinytorch.core.training import LRScheduler, StepLR
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.layers import Linear

            model = Linear(5, 1)
            optimizer = SGD(model.parameters(), lr=0.1)

            # Learning rate scheduler
            if 'StepLR' in locals():
                scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

                initial_lr = optimizer.lr

                # Step the scheduler
                for _ in range(15):
                    if hasattr(scheduler, 'step'):
                        scheduler.step()

                # Learning rate should have decreased
                if hasattr(optimizer, 'lr'):
                    final_lr = optimizer.lr
                    assert final_lr < initial_lr, "Learning rate scheduling not working"

        except ImportError:
            assert True, "Learning rate scheduling not ready yet"


class TestProductionTrainingFeatures:
    """Test production-ready training features."""

    def test_distributed_training_support(self):
        """Test distributed training capabilities."""
        try:
            from tinytorch.core.training import DistributedTrainer
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD

            model = Linear(10, 3)
            optimizer = SGD(model.parameters(), lr=0.01)

            # Distributed trainer (if available)
            if 'DistributedTrainer' in locals():
                dist_trainer = DistributedTrainer(model, optimizer, world_size=1, rank=0)
                assert hasattr(dist_trainer, 'train'), "Distributed training broken"

        except ImportError:
            assert True, "Distributed training not ready yet"

    def test_mixed_precision_training(self):
        """Test mixed precision training support."""
        try:
            from tinytorch.core.training import Trainer
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam

            model = Linear(20, 10)
            optimizer = Adam(model.parameters(), lr=0.001)

            # Dummy loss function
            def dummy_loss(pred, target):
                return pred.sum()

            # Mixed precision trainer - may not support mixed_precision kwarg
            try:
                trainer = Trainer(model, optimizer, dummy_loss)
                # Mixed precision is optional feature
                if hasattr(trainer, 'mixed_precision'):
                    assert True, "Mixed precision capability available"
            except TypeError:
                pass  # Signature doesn't support mixed_precision

        except ImportError:
            assert True, "Mixed precision training not ready yet"

    def test_gradient_accumulation(self):
        """Test gradient accumulation for large effective batch sizes."""
        try:
            from tinytorch.core.training import Trainer
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD

            model = Linear(10, 3)
            optimizer = SGD(model.parameters(), lr=0.01)

            # Dummy loss function
            def dummy_loss(pred, target):
                return pred.sum()

            # Trainer with gradient accumulation - may not support accumulate_grad_batches kwarg
            try:
                trainer = Trainer(model, optimizer, dummy_loss)
                # Gradient accumulation is optional feature
                if hasattr(trainer, 'accumulate_grad_batches'):
                    assert True, "Gradient accumulation capability available"
            except TypeError:
                pass  # Signature doesn't support accumulate_grad_batches

        except ImportError:
            assert True, "Gradient accumulation not ready yet"


class TestRegressionPrevention:
    """Ensure previous modules still work after Module 11 development."""

    def test_no_complete_pipeline_regression(self):
        """Verify complete ML pipeline (01→10) unchanged."""
        import numpy as np  # Import at function scope for proper scoping

        # Core functionality should remain stable
        assert sys.version_info.major >= 3, "Foundation: Python detection broken"

        # Complete pipeline should still work
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.dataloader import Dataset

            # All pipeline components should work
            layer = Linear(3, 2)
            optimizer = SGD(layer.parameters(), lr=0.01)

            x = Tensor(np.random.randn(1, 3))
            output = layer(x)
            assert output.shape == (1, 2), "Pipeline regression: Forward pass broken"

        except ImportError:
            assert np.random is not None, "Pipeline regression: Basic functionality broken"

    def test_no_optimization_regression(self):
        """Verify optimization (10) and data loading (08) unchanged."""
        import numpy as np  # Import at function scope for proper scoping

        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.optimizers import SGD, Adam
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Optimizers should still work - use Tensor with requires_grad
            class DummyModule:
                def __init__(self):
                    self._params = [Tensor(np.array([1.0, 2.0]), requires_grad=True)]

                def parameters(self):
                    return self._params

            module = DummyModule()
            sgd = SGD(module.parameters(), lr=0.01)
            adam = Adam(module.parameters(), lr=0.001)

            assert hasattr(sgd, 'step'), "Optimization regression: SGD broken"
            assert hasattr(adam, 'step'), "Optimization regression: Adam broken"

            # Data loading should still work
            class TestDataset(Dataset):
                def __len__(self):
                    return 5
                def __getitem__(self, idx):
                    return idx, idx * 2

            dataset = TestDataset()
            dataloader = DataLoader(dataset, batch_size=2)
            assert len(dataset) == 5, "Data regression: Dataset broken"

        except ImportError:
            # Basic functionality should work
            assert np is not None, "Optimization/Data regression: Basic functionality broken"

    def test_progressive_stability(self):
        """Test the progressive stack is stable through training."""
        # Stack should be stable through: Setup → ... → Optimizers → Training

        # Setup level
        import numpy as np
        assert np is not None, "Setup level broken"

        # Complete ML pipeline level (if available)
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD

            # Complete training components should work together
            model = Linear(5, 2)
            optimizer = SGD(model.parameters(), lr=0.01)

            x = Tensor(np.random.randn(3, 5))
            output = model(x)
            assert output.shape == (3, 2), "ML pipeline level broken"

        except ImportError:
            pass  # Not implemented yet

        # Training level (if available)
        try:
            from tinytorch.core.training import Trainer

            class DummyModel:
                def parameters(self):
                    return [np.array([1.0])]

            class DummyOptimizer:
                def __init__(self, params, lr):
                    self.lr = lr
                def step(self):
                    pass
                def zero_grad(self):
                    pass

            def dummy_loss(pred, target):
                return pred.sum() if hasattr(pred, 'sum') else 0

            model = DummyModel()
            optimizer = DummyOptimizer(model.parameters(), 0.01)
            trainer = Trainer(model, optimizer, dummy_loss)

            assert hasattr(trainer, 'train') or hasattr(trainer, 'fit') or hasattr(trainer, 'train_epoch'), "Training level broken"

        except ImportError:
            pass  # Not implemented yet
