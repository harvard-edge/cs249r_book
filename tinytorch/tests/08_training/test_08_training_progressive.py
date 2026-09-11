"""
Module 08: Progressive Integration Tests
Tests that Module 08 (Training) works correctly AND that prior modules (01→07) still work.

DEPENDENCY CHAIN: 01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader → 06_autograd → 07_optimizers → 08_training

⚠️ IMPORTANT: This test ONLY uses modules 01-08.
   Future modules (09_convolutions, 12_attention, etc.) are NOT tested here.

🎯 WHAT THIS TESTS:
- Module 08: Training loops, Trainer class, training infrastructure
- Integration: Training works with all prior modules (01-07)
- Regression: All previous modules still work correctly
"""

import numpy as np
rng = np.random.default_rng(7)
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTrainingCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 08 (Training) core implementation.
    """

    def test_trainer_class_exists(self):
        """
        ✅ TEST: Trainer class exists and is importable
        """
        try:
            from tinytorch.core.training import Trainer
            
            assert Trainer is not None, "Trainer class not found"
            
        except ImportError:
            raise

    def test_trainer_initialization(self):
        """
        ✅ TEST: Trainer can be initialized with model, optimizer, loss
        """
        try:
            from tinytorch.core.training import Trainer
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.losses import MSELoss
            
            # Create components
            model = Linear(10, 2)
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = MSELoss()
            
            # Create trainer
            trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
            
            assert hasattr(trainer, 'model'), "Trainer missing model"
            assert hasattr(trainer, 'optimizer'), "Trainer missing optimizer"
            
        except ImportError:
            raise
        except TypeError:
            raise

    def test_training_step(self):
        """
        ✅ TEST: Trainer can run a single training step
        """
        try:
            from tinytorch.core.training import Trainer
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.losses import MSELoss
            
            model = Linear(5, 2)
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = MSELoss()
            
            trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
            
            # Create batch
            batch_x = Tensor(rng.standard_normal((4, 5)))
            batch_y = Tensor(rng.standard_normal((4, 2)))
            
            # The public unit of training is an epoch; one batch is one step.
            before = model.weight.data.copy()
            loss = trainer.train_epoch([(batch_x, batch_y)])
            assert np.isfinite(loss)
            assert trainer.step == 1
            assert not np.array_equal(model.weight.data, before)

        except ImportError:
            raise
        except TypeError:
            raise

    def test_training_epoch(self):
        """
        ✅ TEST: Trainer can run a full training epoch
        """
        try:
            from tinytorch.core.training import Trainer
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            # Create model and training components
            model = Linear(5, 2)
            optimizer = SGD(model.parameters(), lr=0.01)
            loss_fn = MSELoss()
            
            # Create dataloader
            data = Tensor(rng.standard_normal((20, 5)))
            targets = Tensor(rng.standard_normal((20, 2)))
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=4)
            
            # Create trainer
            trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
            
            avg_loss = trainer.train_epoch(dataloader)
            assert np.isfinite(avg_loss)
            assert trainer.step == len(dataloader)

        except ImportError:
            raise
        except TypeError:
            raise


class TestManualTrainingLoop:
    """
    🔗 INTEGRATION: Test manual training loop using modules 01-08.
    
    Even without a Trainer class, we can train using prior modules.
    """

    def test_complete_training_loop(self):
        """
        ✅ TEST: Complete manual training loop
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            # Create model
            layer1 = Linear(5, 10)
            layer2 = Linear(10, 2)
            relu = ReLU()
            loss_fn = MSELoss()
            
            # Collect parameters
            params = layer1.parameters() + layer2.parameters()
            optimizer = SGD(params, lr=0.1)
            
            # Create data
            data = Tensor(rng.standard_normal((40, 5)))
            targets = Tensor(np.zeros((40, 2)))  # Simple target
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=8)
            
            # Training loop
            losses = []
            for epoch in range(3):
                epoch_loss = 0.0
                batch_count = 0
                
                for batch_x, batch_y in dataloader:
                    # Forward pass
                    h = relu(layer1(batch_x))
                    pred = layer2(h)
                    loss = loss_fn(pred, batch_y)
                    
                    # Backward pass
                    if hasattr(loss, 'backward'):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += float(loss.data)
                    batch_count += 1
                
                losses.append(epoch_loss / batch_count)
            
            # Loss should generally decrease (or at least change)
            assert len(losses) == 3, "Training loop didn't complete"
            
        except ImportError as e:
            raise

    def test_learning_verification(self):
        """
        ✅ TEST: Verify actual learning (loss decreases)
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD
            
            # Simple problem: learn to output zeros
            rng = np.random.default_rng(7)
            
            layer = Linear(4, 2)
            loss_fn = MSELoss()
            optimizer = SGD(layer.parameters(), lr=0.1)
            
            # Fixed data
            x = Tensor(rng.standard_normal((8, 4)))
            target = Tensor(np.zeros((8, 2)))
            
            # Train for several steps
            losses = []
            for _ in range(20):
                pred = layer(x)
                loss = loss_fn(pred, target)
                losses.append(float(loss.data))
                
                if hasattr(loss, 'backward'):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            assert losses[-1] < losses[0], "Training must reduce loss on this fixed problem"

        except ImportError as e:
            raise


class TestTrainingUtilities:
    """
    Test training utilities and helpers.
    """

    def test_loss_tracking(self):
        """
        ✅ TEST: Can track losses during training
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            
            layer = Linear(3, 1)
            loss_fn = MSELoss()
            
            # Track losses
            losses = []
            for _ in range(5):
                x = Tensor(rng.standard_normal((2, 3)))
                y = Tensor(rng.standard_normal((2, 1)))
                
                pred = layer(x)
                loss = loss_fn(pred, y)
                losses.append(float(loss.data))
            
            assert len(losses) == 5, "Loss tracking failed"
            assert all(isinstance(l, float) for l in losses), "Losses not floats"
            
        except ImportError as e:
            raise

    def test_batch_processing(self):
        """
        ✅ TEST: Efficient batch processing
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            # Create model
            layer = Linear(10, 5)
            
            # Create batched data
            data = Tensor(rng.standard_normal((100, 10)))
            targets = Tensor(rng.standard_normal((100, 5)))
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=16)
            
            # Process all batches
            outputs = []
            for batch_x, batch_y in dataloader:
                out = layer(batch_x)
                outputs.append(out.shape[0])
            
            # Should have processed all data
            assert sum(outputs) == 100, "Batch processing incomplete"
            
        except ImportError as e:
            raise


class TestRegressionPrevention:
    """
    🔄 REGRESSION: Verify all previous modules (01-07) still work correctly.
    """

    def test_tensor_still_works(self):
        """✅ Module 01 regression check"""
        try:
            from tinytorch.core.tensor import Tensor
            
            a = Tensor([1.0, 2.0])
            b = Tensor([3.0, 4.0])
            c = a + b
            
            assert np.allclose(c.data, [4.0, 6.0]), "Tensor broken"
            
        except Exception as e:
            assert False, f"Module 01 regression: {e}"

    def test_activations_still_work(self):
        """✅ Module 02 regression check"""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU
            
            relu = ReLU()
            x = Tensor([-1.0, 0.0, 1.0])
            y = relu(x)
            
            assert y.data[0] == 0.0, "ReLU broken"
            
        except Exception as e:
            assert False, f"Module 02 regression: {e}"

    def test_layers_still_work(self):
        """✅ Module 03 regression check"""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            
            layer = Linear(4, 2)
            x = Tensor(rng.standard_normal((2, 4)))
            y = layer(x)
            
            assert y.shape == (2, 2), "Linear broken"
            
        except Exception as e:
            assert False, f"Module 03 regression: {e}"

    def test_losses_still_work(self):
        """✅ Module 04 regression check"""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.losses import MSELoss
            
            loss_fn = MSELoss()
            pred = Tensor([[1.0, 2.0]])
            target = Tensor([[1.5, 2.5]])
            loss = loss_fn(pred, target)
            
            assert loss.data.size == 1, "MSELoss broken"
            
        except Exception as e:
            assert False, f"Module 04 regression: {e}"

    def test_dataloader_still_works(self):
        """✅ Module 05 regression check"""
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            data = Tensor(rng.standard_normal((10, 3)))
            targets = Tensor(np.arange(10).astype(float))
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=2)
            
            batches = list(dataloader)
            assert len(batches) == 5, "DataLoader broken"
            
        except Exception as e:
            assert False, f"Module 05 regression: {e}"

    def test_autograd_still_works(self):
        """✅ Module 06 regression check"""
        try:
            from tinytorch.core.tensor import Tensor
            
            x = Tensor([2.0], requires_grad=True)
            
            assert hasattr(x, 'requires_grad'), "Autograd broken"
            assert hasattr(x, 'grad'), "Autograd grad attribute broken"
            
        except TypeError:
            raise
        except Exception as e:
            assert False, f"Module 06 regression: {e}"

    def test_optimizers_still_work(self):
        """✅ Module 07 regression check"""
        try:
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.tensor import Tensor
            
            param = Tensor([1.0, 2.0], requires_grad=True)
            optimizer = SGD([param], lr=0.1)
            
            assert hasattr(optimizer, 'step'), "SGD broken"
            assert hasattr(optimizer, 'zero_grad'), "SGD broken"
            
        except TypeError:
            # Might fail if requires_grad not supported
            from tinytorch.core.layers import Linear
            layer = Linear(2, 1)
            optimizer = SGD(layer.parameters(), lr=0.1)
            assert hasattr(optimizer, 'step'), "SGD broken"
        except Exception as e:
            assert False, f"Module 07 regression: {e}"


class TestModule08Completion:
    """
    ✅ COMPLETION CHECK: Module 08 ready for next module.
    """

    def test_training_foundation_complete(self):
        """
        ✅ FINAL TEST: Training infrastructure ready
        
        🎯 SUCCESS = Ready for Module 09: Convolutions!
        """
        capabilities = {
            "Manual training loop": False,
            "Loss computation": False,
            "Optimizer step": False,
            "Batch iteration": False,
        }
        
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            # Test 1: Loss computation
            layer = Linear(4, 2)
            loss_fn = MSELoss()
            x = Tensor(rng.standard_normal((2, 4)))
            y = Tensor(rng.standard_normal((2, 2)))
            pred = layer(x)
            loss = loss_fn(pred, y)
            if loss.data.size == 1:
                capabilities["Loss computation"] = True
            
            # Test 2: Optimizer step
            optimizer = SGD(layer.parameters(), lr=0.1)
            if hasattr(optimizer, 'step') and hasattr(optimizer, 'zero_grad'):
                capabilities["Optimizer step"] = True
            
            # Test 3: Batch iteration
            data = Tensor(rng.standard_normal((10, 4)))
            targets = Tensor(rng.standard_normal((10, 2)))
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=2)
            if sum(1 for _ in dataloader) == 5:
                capabilities["Batch iteration"] = True
            
            # Test 4: Manual training loop. A failure here is a real failure --
            # swallowing it would let "training is broken" score 3/4 and pass.
            for batch_x, batch_y in dataloader:
                pred = layer(batch_x)
                loss = loss_fn(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            capabilities["Manual training loop"] = True

            missing = [k for k, v in capabilities.items() if not v]
            assert not missing, f"Training capabilities missing: {missing}"
            
        except ImportError as e:
            assert False, f"Module 08 import failed: {e}"
