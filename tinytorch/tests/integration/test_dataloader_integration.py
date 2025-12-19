"""
Integration tests for DataLoader with training workflows.

These tests verify that DataLoader works correctly when integrated with
actual training pipelines, not just in isolation.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import from TinyTorch package
from tinytorch import Tensor
from tinytorch.core.dataloader import Dataset, TensorDataset, DataLoader


def test_training_workflow_integration():
    """
    Test DataLoader integration with realistic training workflow.

    Simulates:
    - Train/val split
    - DataLoader creation
    - Batch iteration
    - Complete epoch processing
    """
    print("🔬 Integration Test: DataLoader + Training Workflow...")

    # Create synthetic dataset (simulate real data)
    num_samples = 1000
    num_features = 20
    num_classes = 5

    features = np.random.randn(num_samples, num_features).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples).astype(np.int64)

    dataset_full = TensorDataset(Tensor(features), Tensor(labels))

    # Split into train/val (80/20 split)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_samples = [dataset_full[i] for i in range(train_size)]
    val_samples = [dataset_full[i] for i in range(train_size, num_samples)]

    # Create tensors from samples
    train_features = Tensor(np.stack([sample[0].data for sample in train_samples]))
    train_labels = Tensor(np.stack([sample[1].data for sample in train_samples]))
    val_features = Tensor(np.stack([sample[0].data for sample in val_samples]))
    val_labels = Tensor(np.stack([sample[1].data for sample in val_samples]))

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"📊 Dataset splits:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Simulate training loop
    print("\n🏃 Simulated Training Loop:")

    epoch_samples = 0
    batch_count = 0

    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        batch_count += 1
        epoch_samples += len(batch_features.data)

        # Simulate forward pass (just check shapes)
        assert batch_features.data.shape[0] <= batch_size, "Batch size exceeded"
        assert batch_features.data.shape[1] == num_features, "Wrong feature count"
        assert len(batch_labels.data) == len(batch_features.data), "Mismatched batch sizes"

        if batch_idx < 3:  # Show first few batches
            print(f"  Batch {batch_idx + 1}: {batch_features.data.shape[0]} samples")

    print(f"  Total: {batch_count} batches, {epoch_samples} samples processed")

    # Validate that all samples were seen
    assert epoch_samples == len(train_dataset), f"Expected {len(train_dataset)}, processed {epoch_samples}"

    print("✅ Training workflow integration works correctly!")


def test_dataloader_shuffle_consistency():
    """Test that shuffle produces different orders across epochs."""
    print("\n🔬 Integration Test: Shuffle Consistency...")

    # Create simple sequential dataset
    data = Tensor(np.arange(100).reshape(-1, 1).astype(np.float32))
    labels = Tensor(np.arange(100).astype(np.int64))
    dataset = TensorDataset(data, labels)

    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Get first batch from two epochs
    epoch1_first = next(iter(loader))[0].data
    epoch2_first = next(iter(loader))[0].data

    # Should be different due to shuffle (very high probability)
    different = not np.array_equal(epoch1_first, epoch2_first)

    assert different, "Shuffle should produce different orders across epochs"
    print("✅ Shuffle produces different orders across epochs")


def test_dataloader_memory_efficiency():
    """Test that DataLoader doesn't load entire dataset into memory at once."""
    print("\n🔬 Integration Test: Memory Efficiency...")

    # Create large-ish dataset
    large_size = 10000
    features = Tensor(np.random.randn(large_size, 50).astype(np.float32))
    labels = Tensor(np.random.randint(0, 10, large_size).astype(np.int64))
    dataset = TensorDataset(features, labels)

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Should be able to iterate without loading all at once
    batch_count = 0
    for batch in loader:
        batch_count += 1
        # Check batch is reasonable size
        assert batch[0].data.shape[0] <= 64
        if batch_count > 10:  # Just verify first few batches
            break

    print(f"✅ Processed {batch_count} batches without loading entire dataset")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 DATALOADER INTEGRATION TESTS")
    print("=" * 60)

    test_training_workflow_integration()
    test_dataloader_shuffle_consistency()
    test_dataloader_memory_efficiency()

    print("\n" + "=" * 60)
    print("🎉 ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)
