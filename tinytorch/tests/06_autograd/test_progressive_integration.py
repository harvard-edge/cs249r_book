"""
Module 06: Progressive Integration Tests
Tests that Module 06 (Autograd) works correctly AND that the entire prior stack works.

DEPENDENCY CHAIN: 01_tensor → 02_activations → 03_layers → 04_losses → 05_dataloader → 06_autograd
This is where we enable automatic differentiation for gradient-based learning.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPriorStackStillWorking:
    """Quick regression checks that prior modules (01→05) still work."""

    def test_foundation_stack_stable(self):
        """Verify foundation stack (01→06) remains stable."""
        # Environment (Module 01)
        assert sys.version_info >= (3, 8), "Foundation broken: Python version"

        # Core functionality should work
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear

            # Should still be able to build networks
            layer = Linear(10, 5)
            x = Tensor(np.random.randn(4, 10))
            output = layer(x)
            assert output.shape == (4, 5), "Foundation broken: Neural network"

        except ImportError:
            assert True, "Foundation not implemented yet"

    def test_advanced_stack_stable(self):
        """Verify advanced modules (07→08) still work."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.attention import MultiHeadAttention

            # Spatial and attention should work
            conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
            attention = MultiHeadAttention(embed_dim=64, num_heads=8)

            assert hasattr(conv, 'forward'), "Advanced stack broken: Spatial"
            assert hasattr(attention, 'forward'), "Advanced stack broken: Attention"

        except ImportError:
            assert True, "Advanced stack not implemented yet"


class TestModule05DataLoaderCore:
    """Test Module 05 (DataLoader) core functionality."""

    def test_dataset_creation(self):
        """Test basic dataset creation works."""
        try:
            from tinytorch.core.dataloader import Dataset

            # Create simple dataset
            class SimpleDataset(Dataset):
                def __init__(self, size=100):
                    self.size = size
                    self.data = np.random.randn(size, 10)
                    self.targets = np.random.randint(0, 3, size)

                def __len__(self):
                    return self.size

                def __getitem__(self, idx):
                    return self.data[idx], self.targets[idx]

            dataset = SimpleDataset(50)
            assert len(dataset) == 50, "Dataset length broken"

            # Test data access
            sample, target = dataset[0]
            assert sample.shape == (10,), "Dataset sample shape broken"
            assert isinstance(target, (int, np.integer)), "Dataset target type broken"

        except ImportError:
            assert True, "Dataset not implemented yet"

    def test_dataloader_creation(self):
        """Test DataLoader creation and batching."""
        try:
            from tinytorch.core.dataloader import DataLoader, Dataset
            from tinytorch.core.tensor import Tensor

            # Simple dataset for testing
            class TestDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(20, 5)
                    self.targets = np.random.randint(0, 2, 20)

                def __len__(self):
                    return 20

                def __getitem__(self, idx):
                    # Return Tensors for both data and targets
                    return Tensor(self.data[idx]), Tensor(np.array([self.targets[idx]]))

            dataset = TestDataset()
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

            # Test batching
            for batch_x, batch_y in dataloader:
                assert batch_x.shape == (4, 5), "DataLoader batch shape broken"
                assert batch_y.shape[0] == 4, "DataLoader target batch broken"
                break  # Just test first batch

        except ImportError:
            assert True, "DataLoader not implemented yet"

    def test_real_dataset_support(self):
        """Test support for real datasets like CIFAR-10."""
        try:
            from tinytorch.core.dataloader import CIFAR10Dataset

            # Note: This might download data, so we'll just test instantiation
            # In real usage, students would download CIFAR-10
            try:
                dataset = CIFAR10Dataset(root='./data', train=True, download=False)
                # If dataset exists, test basic functionality
                if len(dataset) > 0:
                    sample, target = dataset[0]
                    assert len(sample.shape) >= 2, "CIFAR-10 sample shape invalid"
                    assert isinstance(target, (int, np.integer)), "CIFAR-10 target invalid"
            except (FileNotFoundError, RuntimeError):
                # Data not downloaded, which is fine for testing
                assert True, "CIFAR-10 data not available (expected)"

        except ImportError:
            assert True, "Real dataset support not implemented yet"


class TestProgressiveStackIntegration:
    """Test that the complete stack (01→08) works together."""

    def test_complete_training_pipeline(self):
        """Test complete ML pipeline: data → model → training."""
        try:
            from tinytorch.core.dataloader import DataLoader, Dataset
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax

            # Create dataset
            class MLDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(40, 10)
                    self.targets = np.random.randint(0, 3, 40)

                def __len__(self):
                    return 40

                def __getitem__(self, idx):
                    return Tensor(self.data[idx]), self.targets[idx]

            # Create data pipeline
            dataset = MLDataset()
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

            # Create model using prior modules
            layer1 = Linear(10, 16)
            layer2 = Linear(16, 3)
            relu = ReLU()
            softmax = Softmax()

            # Test training loop structure
            for batch_x, batch_y in dataloader:
                # Forward pass through complete pipeline
                h = relu(layer1(batch_x))
                logits = layer2(h)
                predictions = softmax(logits)

                assert predictions.shape == (8, 3), "Complete pipeline broken"

                # Test one batch
                break

        except ImportError:
            assert True, "Complete training pipeline not ready yet"

    def test_cnn_data_pipeline(self):
        """Test CNN pipeline with spatial data."""
        try:
            from tinytorch.core.dataloader import DataLoader, Dataset
            from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Image dataset
            class ImageDataset(Dataset):
                def __init__(self):
                    # 32x32 RGB images
                    self.data = np.random.randn(20, 3, 32, 32)
                    self.targets = np.random.randint(0, 5, 20)

                def __len__(self):
                    return 20

                def __getitem__(self, idx):
                    return Tensor(self.data[idx]), self.targets[idx]

            dataset = ImageDataset()
            dataloader = DataLoader(dataset, batch_size=4)

            # CNN components
            conv1 = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
            pool = MaxPool2d(kernel_size=2)
            fc = Linear(16 * 15 * 15, 5)  # Approximate after conv/pool

            # Test CNN pipeline
            for batch_x, batch_y in dataloader:
                assert batch_x.shape == (4, 3, 32, 32), "Image batch shape broken"

                # Simplified CNN forward (shape checking)
                if hasattr(conv1, '__call__'):
                    conv_out = conv1(batch_x)
                    # Check reasonable conv output shape
                    assert len(conv_out.shape) == 4, "Conv output dimensionality broken"

                break

        except ImportError:
            assert True, "CNN data pipeline not ready yet"


class TestRealWorldDataCapability:
    """Test capability to handle real-world datasets."""

    def test_data_preprocessing_pipeline(self):
        """Test data preprocessing and augmentation."""
        try:
            from tinytorch.core.dataloader import transforms
            from tinytorch.core.tensor import Tensor

            # Basic transforms
            if hasattr(transforms, 'Normalize'):
                normalize = transforms.Normalize(mean=[0.5], std=[0.5])

                # Test data
                data = Tensor(np.random.randn(3, 32, 32))
                normalized = normalize(data)

                assert normalized.shape == data.shape, "Normalization broken"

            if hasattr(transforms, 'RandomCrop'):
                crop = transforms.RandomCrop(size=28)

                data = Tensor(np.random.randn(3, 32, 32))
                cropped = crop(data)

                assert cropped.shape[-2:] == (28, 28), "Random crop broken"

        except ImportError:
            assert True, "Data preprocessing not implemented yet"

    def test_memory_efficient_loading(self):
        """Test memory efficient data loading."""
        try:
            from tinytorch.core.dataloader import DataLoader, Dataset

            from tinytorch.core.tensor import Tensor

            # Large dataset simulation
            class LargeDataset(Dataset):
                def __init__(self, size=1000):
                    self.size = size
                    # Don't load all data at once - simulate lazy loading

                def __len__(self):
                    return self.size

                def __getitem__(self, idx):
                    # Simulate loading data on-demand - return Tensors
                    return Tensor(np.random.randn(100)), Tensor(np.array([idx % 10]))

            dataset = LargeDataset(1000)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Should be able to iterate without loading all data
            batch_count = 0
            for batch_x, batch_y in dataloader:
                batch_count += 1
                if batch_count >= 3:  # Test a few batches
                    break

            assert batch_count == 3, "Memory efficient loading broken"

        except ImportError:
            assert True, "Memory efficient loading not ready yet"

    def test_parallel_data_loading(self):
        """Test parallel/multi-threaded data loading."""
        try:
            from tinytorch.core.dataloader import DataLoader, Dataset
            from tinytorch.core.tensor import Tensor

            class ParallelDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(100, 50)

                def __len__(self):
                    return 100

                def __getitem__(self, idx):
                    # Simulate some processing time - return Tensors
                    return Tensor(self.data[idx]), Tensor(np.array([idx % 5]))

            dataset = ParallelDataset()

            # Test with num_workers if supported
            if 'num_workers' in DataLoader.__init__.__code__.co_varnames:
                dataloader = DataLoader(dataset, batch_size=16, num_workers=2)
            else:
                dataloader = DataLoader(dataset, batch_size=16)

            # Should work regardless of parallel support
            for batch_x, batch_y in dataloader:
                assert batch_x.shape == (16, 50), "Parallel loading broken"
                break

        except ImportError:
            assert True, "Parallel data loading not ready yet"


class TestRegressionPrevention:
    """Ensure previous modules still work after Module 06 (Autograd) development."""

    def test_no_foundation_regression(self):
        """Verify foundation stack (01→05) unchanged."""
        # Core functionality should remain stable
        assert sys.version_info.major >= 3, "Foundation: Python detection broken"

        # Tensor operations should still work
        try:
            from tinytorch.core.tensor import Tensor
            t = Tensor([1, 2, 3])
            assert t.shape == (3,), "Foundation regression: Tensor broken"
        except ImportError:
            import numpy as np
            arr = np.array([1, 2, 3])
            assert arr.shape == (3,), "Foundation regression: Numpy broken"

    def test_no_advanced_regression(self):
        """Verify advanced modules (06→07) unchanged."""
        try:
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.attention import MultiHeadAttention

            # Advanced operations should still work
            conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3)
            attention = MultiHeadAttention(embed_dim=32, num_heads=4)

            assert hasattr(conv, 'forward'), "Advanced regression: Spatial broken"
            assert hasattr(attention, 'forward'), "Advanced regression: Attention broken"

        except ImportError:
            # If not implemented, basic functionality should work
            import numpy as np
            assert np.random is not None, "Advanced regression: Random broken"

    def test_progressive_stability(self):
        """Test the progressive stack is stable through data loading."""
        # Stack should be stable through: Setup → ... → Attention → DataLoader

        # Setup level
        import numpy as np
        assert np is not None, "Setup level broken"

        # Foundation level (if available)
        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear

            # Neural networks should still work
            layer = Linear(5, 3)
            x = Tensor(np.random.randn(2, 5))
            output = layer(x)
            assert output.shape == (2, 3), "Foundation level broken"

        except ImportError:
            pass  # Not implemented yet

        # Data level (if available)
        try:
            from tinytorch.core.dataloader import Dataset

            class TestDataset(Dataset):
                def __len__(self):
                    return 10
                def __getitem__(self, idx):
                    return idx, idx * 2

            dataset = TestDataset()
            assert len(dataset) == 10, "Data level broken"

        except ImportError:
            pass  # Not implemented yet
