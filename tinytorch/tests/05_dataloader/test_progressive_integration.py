"""
Module 05: Progressive Integration Tests
Tests that Module 05 (DataLoader) works correctly AND that Foundation tier (01→04) still works.

DEPENDENCY CHAIN: 01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader

🎯 WHAT THIS TESTS:
- Module 05: Dataset abstraction, batching, shuffling, data pipelines
- Integration: DataLoader works with Foundation tier modules (01-04)
- Regression: All previous modules still work correctly

⚠️ IMPORTANT: This test ONLY uses modules 01-05.
   Future modules (06_autograd, 09_convolutions, 12_attention, etc.) are NOT tested here.

💡 FOR STUDENTS: If tests fail, check:
1. Does your Dataset base class exist?
2. Does TensorDataset work correctly?
3. Does DataLoader iterate and batch properly?
4. Does shuffling work deterministically with a seed?
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDataLoaderCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 05 (DataLoader) core implementation.
    
    Tests the DataLoader infrastructure using ONLY modules 01-05.
    """

    def test_dataset_abstraction(self):
        """
        ✅ TEST: Dataset base class exists and is abstract
        """
        try:
            from tinytorch.core.dataloader import Dataset
            
            # Dataset should be importable
            assert Dataset is not None, "Dataset class not found"
            
            # Dataset should be abstract (can't instantiate directly)
            try:
                ds = Dataset()
                # If we get here, check if it raises NotImplementedError on methods
                try:
                    len(ds)
                    assert False, "Dataset.__len__ should be abstract"
                except (NotImplementedError, TypeError):
                    pass
            except TypeError:
                # Good - can't instantiate abstract class
                pass
                
        except ImportError as e:
            assert False, f"Dataset import failed: {e}"

    def test_tensor_dataset(self):
        """
        ✅ TEST: TensorDataset wraps tensors correctly
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import TensorDataset
            
            # Create test data
            data = Tensor(np.random.randn(100, 10))
            targets = Tensor(np.random.randint(0, 5, 100).astype(float))
            
            # Create TensorDataset
            dataset = TensorDataset(data, targets)
            
            # Test length
            assert len(dataset) == 100, f"Expected length 100, got {len(dataset)}"
            
            # Test indexing
            x, y = dataset[0]
            assert x.shape == (10,), f"Expected shape (10,), got {x.shape}"
            
        except ImportError as e:
            assert False, f"TensorDataset import failed: {e}"

    def test_dataloader_iteration(self):
        """
        ✅ TEST: DataLoader iterates and batches correctly
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            # Create test data
            data = Tensor(np.random.randn(20, 5))
            targets = Tensor(np.arange(20).astype(float))
            
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=4)
            
            # Test iteration
            batch_count = 0
            for batch_x, batch_y in dataloader:
                batch_count += 1
                assert batch_x.shape[0] <= 4, f"Batch size exceeded: {batch_x.shape[0]}"
                assert batch_x.shape[1] == 5, f"Feature dimension wrong: {batch_x.shape[1]}"
            
            # Should have 5 batches (20 / 4)
            assert batch_count == 5, f"Expected 5 batches, got {batch_count}"
            
        except ImportError as e:
            assert False, f"DataLoader import failed: {e}"

    def test_dataloader_shuffle(self):
        """
        ✅ TEST: DataLoader shuffling works
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            # Create ordered data
            data = Tensor(np.arange(20).reshape(20, 1).astype(float))
            targets = Tensor(np.arange(20).astype(float))
            
            dataset = TensorDataset(data, targets)
            
            # Create shuffled dataloader
            dl = DataLoader(dataset, batch_size=4, shuffle=True)
            
            # Collect all batches
            all_values = []
            for batch_x, batch_y in dl:
                all_values.extend(batch_x.data.flatten().tolist())
            
            # Should have all 20 values
            assert len(all_values) == 20, f"Missing values: got {len(all_values)}"
            
            # All original values should be present (even if shuffled)
            expected = set(range(20))
            actual = set(int(v) for v in all_values)
            assert expected == actual, "Shuffling lost some values"
                
        except ImportError as e:
            assert False, f"DataLoader shuffle test failed: {e}"


class TestDataLoaderWithLayers:
    """
    🔗 INTEGRATION: DataLoader + Layers (Modules 01-05)
    
    Tests that DataLoader works with the neural network layers from modules 01-04.
    """

    def test_dataloader_with_linear_layer(self):
        """
        ✅ TEST: DataLoader feeds data to Linear layer correctly
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            # Create test data
            data = Tensor(np.random.randn(20, 10))
            targets = Tensor(np.random.randn(20, 3))
            
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=4)
            
            # Create model
            layer = Linear(10, 3)
            
            # Test forward pass with batches from dataloader
            for batch_x, batch_y in dataloader:
                output = layer(batch_x)
                assert output.shape == (batch_x.shape[0], 3), \
                    f"Linear output shape wrong: {output.shape}"
                break  # Test one batch
                
        except ImportError as e:
            assert False, f"DataLoader + Linear integration failed: {e}"

    def test_dataloader_with_activations(self):
        """
        ✅ TEST: DataLoader + Linear + Activation pipeline
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            # Create test data
            data = Tensor(np.random.randn(20, 10))
            targets = Tensor(np.random.randn(20, 5))
            
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=4)
            
            # Create simple model
            layer1 = Linear(10, 8)
            relu = ReLU()
            layer2 = Linear(8, 5)
            softmax = Softmax()
            
            # Test pipeline
            for batch_x, batch_y in dataloader:
                h = relu(layer1(batch_x))
                output = softmax(layer2(h))
                
                assert output.shape == (batch_x.shape[0], 5), \
                    f"Pipeline output shape wrong: {output.shape}"
                
                # Verify softmax output sums to 1
                sums = np.sum(output.data, axis=1)
                assert np.allclose(sums, 1.0), \
                    f"Softmax outputs don't sum to 1: {sums}"
                break
                
        except ImportError as e:
            assert False, f"DataLoader + Activation pipeline failed: {e}"

    def test_dataloader_with_loss(self):
        """
        ✅ TEST: Complete forward pass: DataLoader → Model → Loss
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU
            from tinytorch.core.losses import MSELoss
            from tinytorch.core.dataloader import TensorDataset, DataLoader
            
            # Create test data
            data = Tensor(np.random.randn(20, 10))
            targets = Tensor(np.random.randn(20, 3))
            
            dataset = TensorDataset(data, targets)
            dataloader = DataLoader(dataset, batch_size=4)
            
            # Create model and loss
            layer = Linear(10, 3)
            relu = ReLU()
            loss_fn = MSELoss()
            
            # Test complete forward pass
            total_loss = 0.0
            batch_count = 0
            
            for batch_x, batch_y in dataloader:
                # Forward pass
                output = relu(layer(batch_x))
                
                # Compute loss
                loss = loss_fn(output, batch_y)
                
                assert loss.data.shape == () or loss.data.shape == (1,), \
                    f"Loss should be scalar, got shape {loss.data.shape}"
                
                total_loss += float(loss.data)
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            assert avg_loss > 0, "Loss should be positive for random data"
            
        except ImportError as e:
            assert False, f"DataLoader + Loss pipeline failed: {e}"


class TestRegressionPrevention:
    """
    🔄 REGRESSION: Verify all previous modules (01-04) still work correctly.
    """

    def test_tensor_operations_still_work(self):
        """
        ✅ TEST: Module 01 (Tensor) operations still work
        """
        try:
            from tinytorch.core.tensor import Tensor
            
            # Basic operations
            a = Tensor([1.0, 2.0, 3.0])
            b = Tensor([4.0, 5.0, 6.0])
            
            # Arithmetic
            c = a + b
            assert np.allclose(c.data, [5.0, 7.0, 9.0]), "Tensor addition broken"
            
            d = a * b
            assert np.allclose(d.data, [4.0, 10.0, 18.0]), "Tensor multiplication broken"
            
            # Matrix operations
            m1 = Tensor([[1, 2], [3, 4]])
            m2 = Tensor([[5, 6], [7, 8]])
            m3 = m1 @ m2
            assert m3.shape == (2, 2), "Matrix multiplication broken"
            
        except Exception as e:
            assert False, f"Module 01 regression: {e}"

    def test_activations_still_work(self):
        """
        ✅ TEST: Module 02 (Activations) still work
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.activations import ReLU, Sigmoid, Softmax
            
            x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]))
            
            # ReLU
            relu = ReLU()
            r = relu(x)
            assert np.allclose(r.data, [0.0, 0.0, 1.0, 2.0]), "ReLU broken"
            
            # Sigmoid
            sigmoid = Sigmoid()
            s = sigmoid(x)
            assert s.data[2] > 0.5, "Sigmoid broken"
            
            # Softmax
            softmax = Softmax()
            sm = softmax(x)
            assert np.allclose(np.sum(sm.data), 1.0), "Softmax broken"
            
        except Exception as e:
            assert False, f"Module 02 regression: {e}"

    def test_layers_still_work(self):
        """
        ✅ TEST: Module 03 (Layers) still work
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            
            layer = Linear(10, 5)
            x = Tensor(np.random.randn(4, 10))
            
            output = layer(x)
            assert output.shape == (4, 5), f"Linear layer broken: {output.shape}"
            
        except Exception as e:
            assert False, f"Module 03 regression: {e}"

    def test_losses_still_work(self):
        """
        ✅ TEST: Module 04 (Losses) still work
        """
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.losses import MSELoss, CrossEntropyLoss
            
            pred = Tensor([[0.1, 0.9], [0.8, 0.2]])
            target = Tensor([[0.0, 1.0], [1.0, 0.0]])
            
            # MSE Loss
            mse = MSELoss()
            loss = mse(pred, target)
            assert loss.data.size == 1, "MSE loss should be scalar"
            
            # Cross Entropy Loss
            ce = CrossEntropyLoss()
            logits = Tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
            labels = Tensor([2, 0])
            ce_loss = ce(logits, labels)
            assert ce_loss.data.size == 1, "CrossEntropy loss should be scalar"
            
        except Exception as e:
            assert False, f"Module 04 regression: {e}"


class TestModule05Completion:
    """
    ✅ COMPLETION CHECK: Module 05 ready for next module.
    """

    def test_dataloader_foundation_complete(self):
        """
        ✅ FINAL TEST: DataLoader foundation ready for training infrastructure
        
        🎯 SUCCESS = Ready for Module 06: Autograd!
        """
        capabilities = {
            "Dataset abstraction": False,
            "TensorDataset works": False,
            "DataLoader iteration": False,
            "Batching works": False,
            "Shuffling works": False,
            "Layer integration": False,
            "Loss integration": False,
        }
        
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.dataloader import Dataset, TensorDataset, DataLoader
            from tinytorch.core.layers import Linear
            from tinytorch.core.losses import MSELoss
            
            # Test 1: Dataset abstraction
            assert Dataset is not None
            capabilities["Dataset abstraction"] = True
            
            # Test 2: TensorDataset
            data = Tensor(np.random.randn(20, 5))
            targets = Tensor(np.random.randn(20, 2))
            dataset = TensorDataset(data, targets)
            assert len(dataset) == 20
            capabilities["TensorDataset works"] = True
            
            # Test 3: DataLoader iteration
            dataloader = DataLoader(dataset, batch_size=4)
            batch_count = sum(1 for _ in dataloader)
            assert batch_count == 5
            capabilities["DataLoader iteration"] = True
            
            # Test 4: Batching
            for batch_x, batch_y in dataloader:
                assert batch_x.shape[0] <= 4
                break
            capabilities["Batching works"] = True
            
            # Test 5: Shuffling
            dl_shuffled = DataLoader(dataset, batch_size=4, shuffle=True)
            _ = list(dl_shuffled)
            capabilities["Shuffling works"] = True
            
            # Test 6: Layer integration
            layer = Linear(5, 2)
            for batch_x, batch_y in dataloader:
                output = layer(batch_x)
                assert output.shape == (batch_x.shape[0], 2)
                break
            capabilities["Layer integration"] = True
            
            # Test 7: Loss integration
            loss_fn = MSELoss()
            for batch_x, batch_y in dataloader:
                output = layer(batch_x)
                loss = loss_fn(output, batch_y)
                assert loss.data.size == 1
                break
            capabilities["Loss integration"] = True
            
            # All passed!
            assert all(capabilities.values()), \
                f"Not all capabilities ready: {capabilities}"
                
        except Exception as e:
            completed = sum(capabilities.values())
            total = len(capabilities)
            
            progress = "\n".join(
                f"  {'✅' if v else '❌'} {k}" 
                for k, v in capabilities.items()
            )
            
            assert False, f"""
            ❌ MODULE 05 NOT COMPLETE!
            
            Error: {e}
            
            Progress ({completed}/{total}):
            {progress}
            """
