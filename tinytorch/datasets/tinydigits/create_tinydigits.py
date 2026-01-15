#!/usr/bin/env python3
"""
Create TinyDigits Dataset
=========================

Extracts a balanced, curated subset from sklearn's digits dataset (8x8 grayscale).
This creates a TinyTorch-branded educational dataset optimized for fast iteration.

Following Karpathy's "~1000 samples" philosophy for educational datasets.

Target sizes:
- Training: 1000 samples (100 per digit class 0-9)
- Test: 200 samples (20 per digit class 0-9)
"""

import numpy as np
import pickle
from pathlib import Path

def create_tinydigits():
    """Create TinyDigits train/test split from sklearn digits dataset."""

    # Load directly from sklearn
    from sklearn.datasets import load_digits
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0  # Normalize to [0, 1]
    labels = digits.target  # (1797,)

    print(f"📊 Source dataset: {images.shape[0]} samples")
    print(f"   Shape: {images.shape}, dtype: {images.dtype}")
    print(f"   Range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"   ✓ Normalized to [0, 1]")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create balanced splits
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    # For each digit class (0-9)
    for digit in range(10):
        # Get all samples of this digit
        digit_indices = np.where(labels == digit)[0]
        digit_count = len(digit_indices)

        # Shuffle indices
        np.random.shuffle(digit_indices)

        # Split: 100 for training, 20 for test (Karpathy's ~1000 samples philosophy)
        train_count = 100
        test_count = 20

        # Training: First 100 samples
        train_images.append(images[digit_indices[:train_count]])
        train_labels.extend([digit] * train_count)

        # Test: Next 20 samples
        test_images.append(images[digit_indices[train_count:train_count+test_count]])
        test_labels.extend([digit] * test_count)

        print(f"   Digit {digit}: {train_count} train, {test_count} test (from {digit_count} total)")

    # Stack into arrays
    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_images = np.vstack(test_images)
    test_labels = np.array(test_labels, dtype=np.int64)

    # Shuffle both sets
    train_shuffle = np.random.permutation(len(train_images))
    train_images = train_images[train_shuffle]
    train_labels = train_labels[train_shuffle]

    test_shuffle = np.random.permutation(len(test_images))
    test_images = test_images[test_shuffle]
    test_labels = test_labels[test_shuffle]

    print(f"\n✅ Created TinyDigits:")
    print(f"   Training: {train_images.shape} images, {train_labels.shape} labels")
    print(f"   Test: {test_images.shape} images, {test_labels.shape} labels")

    # Save as pickle files
    output_dir = Path(__file__).parent

    train_data = {'images': train_images, 'labels': train_labels}
    with open(output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    print(f"\n💾 Saved: train.pkl")

    test_data = {'images': test_images, 'labels': test_labels}
    with open(output_dir / 'test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    print(f"💾 Saved: test.pkl")

    # Calculate file sizes
    train_size = (output_dir / 'train.pkl').stat().st_size / 1024
    test_size = (output_dir / 'test.pkl').stat().st_size / 1024
    total_size = train_size + test_size

    print(f"\n📦 File sizes:")
    print(f"   train.pkl: {train_size:.1f} KB")
    print(f"   test.pkl: {test_size:.1f} KB")
    print(f"   Total: {total_size:.1f} KB")

    print(f"\n🎯 TinyDigits created successfully!")
    print(f"   Perfect for TinyTorch on RasPi0 - only {total_size:.1f} KB!")

if __name__ == "__main__":
    create_tinydigits()
