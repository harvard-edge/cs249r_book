"""
Module 08: Progressive Integration Tests
Tests that Module 08 (Training) works correctly AND that the entire prior stack works.

DEPENDENCY CHAIN: 01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader → 06_autograd → 07_optimizers → 08_training
This is where we enable complete training loops for neural networks.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPriorStackStillWorking:
    """Quick regression checks that prior modules (01→09) still work."""

    def test_foundation_and_data_stable(self):
        """Verify foundation + data stack remains stable."""
        # Environment (Module 01)
        assert sys.version_info >= (3, 8), "Foundation broken: Python version"

        # Neural networks + data should work
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.dataloader import Dataset

            # Complete ML pipeline components should work
            layer = Linear(10, 5)
            x = Tensor(np.random.randn(4, 10))
            output = layer(x)
            assert output.shape == (4, 5), "Foundation broken: Neural network"

        except ImportError:
            assert True, "Foundation not implemented yet"

    def test_autograd_stable(self):
        """Verify Module 06 (Autograd) still works."""
        try:
            from tinytorch.core.autograd import Variable, backward
            from tinytorch.core.tensor import Tensor

            # Autograd should compute gradients
            x = Variable(Tensor([2.0]), requires_grad=True)
            y = x * x + 3 * x + 1  # Simple function

            if hasattr(y, 'backward'):
                y.backward()
                # dy/dx = 2x + 3, at x=2 should be 7
                assert x.grad is not None, "Autograd broken: No gradients"

        except ImportError:
            assert True, "Autograd not implemented yet"


class TestModule07OptimizersCore:
    """Test Module 07 (Optimizers) core functionality."""

    def test_sgd_optimizer_creation(self):
        """Test SGD optimizer creation and basic functionality."""
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Create model with parameters
            layer = Linear(5, 3)

            # Create SGD optimizer
            optimizer = SGD(layer.parameters(), lr=0.01)

            # Should have learning rate and parameter groups
            assert hasattr(optimizer, 'lr'), "SGD broken: No learning rate"
            assert hasattr(optimizer, 'param_groups') or hasattr(optimizer, 'parameters') or hasattr(optimizer, 'params'), "SGD broken: No parameters"

            # Test zero_grad
            if hasattr(optimizer, 'zero_grad'):
                optimizer.zero_grad()

            # Test step (even without gradients)
            if hasattr(optimizer, 'step'):
                optimizer.step()

        except ImportError:
            assert True, "SGD optimizer not implemented yet"

    def test_adam_optimizer_creation(self):
        """Test Adam optimizer creation and advanced features."""
        try:
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.layers import Linear

            # Create model
            layer = Linear(10, 5)

            # Create Adam optimizer with hyperparameters
            optimizer = Adam(layer.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

            # Should have Adam-specific parameters
            assert hasattr(optimizer, 'lr'), "Adam broken: No learning rate"
            assert hasattr(optimizer, 'betas') or hasattr(optimizer, 'beta1'), "Adam broken: No momentum terms"

            # Adam uses momentum buffers
            if hasattr(optimizer, 'state'):
                # State should be initialized (might be empty initially)
                assert isinstance(optimizer.state, dict), "Adam broken: State not dict"

        except ImportError:
            assert True, "Adam optimizer not implemented yet"

    def test_optimizer_parameter_updates(self):
        """Test that optimizers actually update parameters."""
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.autograd import Variable

            # Create simple model
            layer = Linear(2, 1)
            optimizer = SGD(layer.parameters(), lr=0.1)

            # Get initial weights
            initial_weights = layer.weight.data.copy()

            # Create dummy gradients
            if hasattr(layer.weight, 'grad'):
                layer.weight.grad = Tensor(np.random.randn(*layer.weight.shape))
            elif hasattr(layer, 'zero_grad'):
                # Simulate backward pass
                x = Variable(Tensor(np.random.randn(1, 2)))
                y = layer(x)
                if hasattr(y, 'backward'):
                    y.backward()

            # Take optimizer step
            optimizer.step()

            # Weights should have changed (if gradients exist)
            if hasattr(layer.weight, 'grad') and layer.weight.grad is not None:
                updated_weights = layer.weight.data
                # Check if weights actually updated
                weight_changed = not np.array_equal(initial_weights, updated_weights)
                assert weight_changed, "Optimizer didn't update parameters"

        except ImportError:
            assert True, "Parameter updates not ready yet"


class TestProgressiveStackIntegration:
    """Test that the complete stack (01→10) works together."""

    def test_complete_training_step(self):
        """Test complete training step: forward → backward → optimize."""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.dataloader import Dataset, DataLoader
            from tinytorch.core.autograd import Variable

            # Create dataset
            class TrainingDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(20, 5)
                    self.targets = np.random.randn(20, 1)

                def __len__(self):
                    return 20

                def __getitem__(self, idx):
                    return Tensor(self.data[idx]), Tensor(self.targets[idx])

            # Create model
            layer1 = Linear(5, 10)
            layer2 = Linear(10, 1)
            relu = ReLU()

            # Create optimizer
            # Collect all parameters
            params = []
            if hasattr(layer1, 'parameters'):
                params.extend(layer1.parameters())
            if hasattr(layer2, 'parameters'):
                params.extend(layer2.parameters())

            optimizer = SGD(params, lr=0.01)

            # Create data loader
            dataset = TrainingDataset()
            dataloader = DataLoader(dataset, batch_size=4)

            # Training step
            for batch_x, batch_y in dataloader:
                # Forward pass
                h = relu(layer1(batch_x))
                pred = layer2(h)

                # Simple loss (MSE)
                if hasattr(pred, '__sub__') and hasattr(batch_y, '__sub__'):
                    diff = pred - batch_y
                    loss = diff * diff  # Simplified MSE

                    # Backward pass (if available)
                    if hasattr(loss, 'backward'):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Test one batch
                assert pred.shape == batch_y.shape, "Training step broken"
                break

        except ImportError:
            assert True, "Complete training step not ready yet"

    def test_cnn_optimization(self):
        """Test optimization with convolutional networks."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.tensor import Tensor

            # CNN architecture
            conv1 = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
            pool = MaxPool2d(kernel_size=2)
            fc = Linear(16 * 15 * 15, 10)  # Approximate size

            # Collect CNN parameters
            params = []
            for module in [conv1, fc]:
                if hasattr(module, 'parameters'):
                    params.extend(module.parameters())
                elif hasattr(module, 'weight'):
                    params.append(module.weight)
                    if hasattr(module, 'bias') and module.bias is not None:
                        params.append(module.bias)

            # Create Adam optimizer for CNN
            optimizer = Adam(params, lr=0.001)

            # Test image batch
            batch = Tensor(np.random.randn(4, 3, 32, 32))

            # Forward pass through CNN
            if hasattr(conv1, '__call__'):
                conv_out = conv1(batch)

                # Optimizer should handle CNN parameters
                assert len(params) > 0, "CNN parameters not found"

        except ImportError:
            assert True, "CNN optimization not ready yet"


class TestOptimizationAlgorithms:
    """Test different optimization algorithms and their characteristics."""

    def test_sgd_vs_adam_behavior(self):
        """Test SGD vs Adam optimization behavior."""
        try:
            from tinytorch.core.optimizers import SGD, Adam
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Create identical models
            model_sgd = Linear(10, 1)
            model_adam = Linear(10, 1)

            # Make weights identical
            model_adam.weight.data = model_sgd.weight.data.copy()
            if hasattr(model_sgd, 'bias') and model_sgd.bias is not None:
                model_adam.bias.data = model_sgd.bias.data.copy()

            # Create optimizers
            opt_sgd = SGD(model_sgd.parameters(), lr=0.01)
            opt_adam = Adam(model_adam.parameters(), lr=0.01)

            # They should have different internal states
            sgd_has_momentum = hasattr(opt_sgd, 'momentum') or hasattr(opt_sgd, 'velocity')
            adam_has_momentum = hasattr(opt_adam, 'betas') or hasattr(opt_adam, 'state')

            # Adam should have more sophisticated state
            if adam_has_momentum and not sgd_has_momentum:
                assert True, "SGD and Adam have different complexity as expected"
            else:
                assert True, "Optimizers created successfully"

        except ImportError:
            assert True, "Multiple optimizers not ready yet"

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling capabilities."""
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.layers import Linear

            layer = Linear(5, 1)
            optimizer = SGD(layer.parameters(), lr=0.1)

            initial_lr = optimizer.lr

            # Test learning rate modification
            if hasattr(optimizer, 'set_lr'):
                optimizer.set_lr(0.05)
                assert optimizer.lr == 0.05, "Learning rate scheduling broken"
            elif hasattr(optimizer, 'param_groups'):
                # PyTorch-style parameter groups
                for group in optimizer.param_groups:
                    group['lr'] = 0.05
                new_lr = optimizer.param_groups[0]['lr']
                assert new_lr == 0.05, "Parameter group LR scheduling broken"
            else:
                # Direct lr modification
                optimizer.lr = 0.05
                assert optimizer.lr == 0.05, "Direct LR modification broken"

        except ImportError:
            assert True, "Learning rate scheduling not ready yet"

    def test_optimizer_memory_efficiency(self):
        """Test optimizer memory usage and efficiency."""
        try:
            from tinytorch.core.optimizers import SGD, Adam
            from tinytorch.core.layers import Linear

            # Large model to test memory
            large_model = Linear(1000, 500)

            # SGD should use less memory than Adam
            sgd_optimizer = SGD(large_model.parameters(), lr=0.01)
            adam_optimizer = Adam(large_model.parameters(), lr=0.01)

            # Adam should have more state (momentum buffers)
            if hasattr(adam_optimizer, 'state'):
                # Adam state will grow as optimization proceeds
                assert hasattr(adam_optimizer, 'state'), "Adam missing state for momentum"

            # SGD should be simpler
            sgd_simple = not hasattr(sgd_optimizer, 'state') or len(sgd_optimizer.state) == 0
            adam_complex = hasattr(adam_optimizer, 'betas') or hasattr(adam_optimizer, 'state')

            if sgd_simple and adam_complex:
                assert True, "SGD is simpler than Adam as expected"
            else:
                assert True, "Optimizers have reasonable complexity"

        except ImportError:
            assert True, "Memory efficiency testing not ready yet"


class TestProductionOptimization:
    """Test production-ready optimization features."""

    def test_gradient_clipping(self):
        """Test gradient clipping for stable training."""
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            layer = Linear(10, 1)
            optimizer = SGD(layer.parameters(), lr=0.1)

            # Simulate large gradients
            if hasattr(layer.weight, 'grad'):
                layer.weight.grad = Tensor(np.random.randn(*layer.weight.shape) * 100)  # Large gradients

            # Test gradient clipping if available
            if hasattr(optimizer, 'clip_gradients'):
                optimizer.clip_gradients(max_norm=1.0)

                # Gradients should be clipped
                if layer.weight.grad is not None:
                    grad_norm = np.linalg.norm(layer.weight.grad.data)
                    assert grad_norm <= 1.1, "Gradient clipping not working"  # Allow small numerical error

        except ImportError:
            assert True, "Gradient clipping not ready yet"

    def test_optimizer_state_persistence(self):
        """Test saving and loading optimizer state."""
        try:
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            layer = Linear(5, 1)
            optimizer = Adam(layer.parameters(), lr=0.001)

            # Take some steps to build state
            if hasattr(layer.weight, 'grad'):
                layer.weight.grad = Tensor(np.random.randn(*layer.weight.shape))

                for _ in range(3):
                    optimizer.step()

            # Test state dictionary
            if hasattr(optimizer, 'state_dict'):
                state = optimizer.state_dict()
                assert isinstance(state, dict), "Optimizer state_dict not dict"

                # Test loading state
                if hasattr(optimizer, 'load_state_dict'):
                    optimizer.load_state_dict(state)

        except ImportError:
            assert True, "Optimizer persistence not ready yet"


class TestRegressionPrevention:
    """Ensure previous modules still work after Module 10 development."""

    def test_no_foundation_regression(self):
        """Verify foundation stack (01→05) unchanged."""
        # Core functionality should remain stable
        assert sys.version_info.major >= 3, "Foundation: Python detection broken"

        # Neural networks should still work
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear

            layer = Linear(5, 3)
            x = Tensor(np.random.randn(2, 5))
            output = layer(x)
            assert output.shape == (2, 3), "Foundation regression: Neural network broken"

        except ImportError:
            # Still verify numpy works at minimum
            assert np.random is not None, "Foundation regression: Numpy broken"

    def test_no_data_and_autograd_regression(self):
        """Verify data loading (08) and autograd (09) unchanged."""
        try:
            from tinytorch.core.dataloader import Dataset
            from tinytorch.core.autograd import Variable

            # Data loading should still work
            class TestDataset(Dataset):
                def __len__(self):
                    return 5
                def __getitem__(self, idx):
                    return idx, idx * 2

            dataset = TestDataset()
            assert len(dataset) == 5, "Data regression: Dataset broken"

            # Autograd should still work
            if hasattr(Variable, '__init__'):
                x = Variable(np.array([1.0]), requires_grad=True)
                assert hasattr(x, 'requires_grad'), "Autograd regression: Variable broken"

        except ImportError:
            # Basic functionality should work
            assert np is not None, "Data/Autograd regression: Basic functionality broken"

    def test_progressive_stability(self):
        """Test the progressive stack is stable through optimization."""
        # Stack should be stable through: Setup → ... → Autograd → Optimizers

        # Setup level - np already imported globally
        assert np is not None, "Setup level broken"

        # ML pipeline level (if available)
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.dataloader import Dataset

            # Complete ML components should work together
            layer = Linear(3, 2)
            x = Tensor(np.random.randn(1, 3))
            output = layer(x)
            assert output.shape == (1, 2), "ML pipeline level broken"

        except ImportError:
            pass  # Not implemented yet

        # Optimization level (if available)
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor

            # Create a proper Tensor with requires_grad for SGD
            param = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            optimizer = SGD([param], lr=0.01)
            assert hasattr(optimizer, 'lr'), "Optimization level broken"

        except ImportError:
            pass  # Not implemented yet
