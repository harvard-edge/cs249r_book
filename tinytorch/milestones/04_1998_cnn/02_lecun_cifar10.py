#!/usr/bin/env python3
"""
CNN Revolution (1998) - Part 2: CIFAR-10
========================================

Scale YOUR spatial modules to natural images using YOUR DataLoader!
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import os
import numpy as np
import argparse
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# =============================================================================
# YOUR TINYTORCH MODULES IN ACTION
# =============================================================================
#
# This milestone showcases modules YOU built. Here's what powers this CNN:
#
# What You Built          How It's Used Here                Systems Impact
# -------------------------------------------------------------------------------
# Module 01: Tensor       Every computation flows through   Automatic gradient
#                         YOUR Tensor with autodiff         tracking enables
#                                                           backpropagation
#
# Module 02: Activations  ReLU after each conv/linear       Non-linearity lets
#                         layer enables feature learning    CNN learn complex
#                                                           patterns
#
# Module 03: Layers       Linear layers for final           Classification head
#                         classification (2304->256->10)    maps features to
#                                                           class probabilities
#
# Module 04: Losses       CrossEntropy computes how         Gradient signal that
#                         wrong predictions are             drives learning
#
# Module 05: Autograd     .backward() computes all          Chain rule through
#                         gradients automatically           conv, pool, linear
#
# Module 06: Optimizers   Adam updates 600K+ parameters     Adaptive learning
#                         with momentum and scaling         rates per parameter
#
# Module 07: Training     Training loop orchestrates        Batch processing,
#                         forward/backward/update cycle     loss tracking
#
# Module 08: DataLoader   Batches 50,000 images into        Memory: 0.2MB/batch
#                         digestible chunks with shuffle    vs 150MB all at once
#
# Module 09: Spatial      Conv2d extracts features,         Local connectivity +
#                         MaxPool2d reduces dimensions,     weight sharing =
#                         BatchNorm stabilizes training     100x fewer params
# =============================================================================

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.spatial import Conv2d, MaxPool2D, BatchNorm2d
from tinytorch.core.optimizers import Adam
from tinytorch.core.dataloader import DataLoader, Dataset
from tinytorch.data.loader import RandomHorizontalFlip, RandomCrop, Compose

from data_manager import DatasetManager


# =============================================================================
# DATASET CLASS - Your Module 08 Enables This Pattern
# =============================================================================
#
# The Dataset abstraction YOU built (Module 08) defines a contract:
#   __len__() -> how many samples
#   __getitem__(idx) -> return (data, label) for index idx
#
# This simple interface lets YOUR DataLoader handle any data source uniformly.
# Here we wrap CIFAR-10's 50,000 images to follow YOUR Dataset protocol.

class CIFARDataset(Dataset):
    """CIFAR-10 wrapped in YOUR Dataset interface."""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform  # YOUR augmentation transforms (Module 08)

    def __getitem__(self, idx):
        """YOUR DataLoader calls this for each sample."""
        img = self.data[idx]

        if self.transform is not None:
            img = self.transform(img)
            if isinstance(img, Tensor):
                img = img.data

        return Tensor(img), Tensor([self.labels[idx]])

    def __len__(self):
        return len(self.data)

    def get_num_classes(self):
        return 10


# Data augmentation using YOUR transforms (Module 08)
train_transforms = Compose([
    RandomHorizontalFlip(p=0.5),   # Cars look similar flipped
    RandomCrop(32, padding=4),     # Simulates translation
])


# =============================================================================
# CNN MODEL - Your Spatial Modules Power Feature Extraction
# =============================================================================
#
# CNNs exploit spatial structure that MLPs ignore:
#   - LOCAL CONNECTIVITY: 3x3 kernel sees only nearby pixels
#   - WEIGHT SHARING: Same filter detects patterns anywhere
#   - HIERARCHICAL FEATURES: Edges -> Shapes -> Objects
#
# Architecture: Conv -> BatchNorm -> ReLU -> Pool (modern pattern)
#   Input: 32x32x3 RGB image
#   Conv1: Detect edges, colors (3->32 channels)
#   Pool1: Reduce to 15x15 (translation invariance)
#   Conv2: Combine into shapes (32->64 channels)
#   Pool2: Reduce to 6x6
#   Linear: Classify (2304->256->10)

def flatten(x):
    """Bridge between spatial (conv) and dense (linear) layers."""
    batch_size = x.data.shape[0]
    return Tensor(x.data.reshape(batch_size, -1))


class CIFARCNN:
    """CNN for CIFAR-10 using YOUR TinyTorch spatial modules."""

    def __init__(self):
        print("Building CNN with YOUR TinyTorch modules...")

        # YOUR Conv2d (Module 09): Local feature extraction
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.bn1 = BatchNorm2d(32)  # YOUR BatchNorm: Stabilizes training
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.bn2 = BatchNorm2d(64)

        # YOUR MaxPool2d (Module 09): Spatial downsampling
        self.pool = MaxPool2D(pool_size=(2, 2))

        # YOUR ReLU (Module 02): Non-linearity
        self.relu = ReLU()

        # YOUR Linear (Module 03): Classification head
        # After conv/pool: 64 channels * 6 * 6 spatial = 2304 features
        self.fc1 = Linear(64 * 6 * 6, 256)
        self.fc2 = Linear(256, 10)

        self._training = True

        # Parameter count
        conv1_params = 3 * 3 * 3 * 32 + 32
        bn1_params = 32 * 2
        conv2_params = 3 * 3 * 32 * 64 + 64
        bn2_params = 64 * 2
        fc1_params = 64 * 6 * 6 * 256 + 256
        fc2_params = 256 * 10 + 10
        self.total_params = conv1_params + bn1_params + conv2_params + bn2_params + fc1_params + fc2_params

        print(f"   Conv layers: 3->32->64 channels")
        print(f"   Dense layers: 2304->256->10")
        print(f"   Total parameters: {self.total_params:,}")

    def train(self):
        """Training mode: BatchNorm uses batch statistics."""
        self._training = True
        self.bn1.train()
        self.bn2.train()
        return self

    def eval(self):
        """Eval mode: BatchNorm uses running statistics."""
        self._training = False
        self.bn1.eval()
        self.bn2.eval()
        return self

    def forward(self, x):
        """Forward pass through YOUR CNN."""
        # Block 1: Conv -> BatchNorm -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2: Same pattern, higher-level features
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Classification: Flatten -> Linear -> ReLU -> Linear
        x = flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        """All trainable parameters for YOUR optimizer."""
        return [
            self.conv1.weight, self.conv1.bias,
            self.bn1.gamma, self.bn1.beta,
            self.conv2.weight, self.conv2.bias,
            self.bn2.gamma, self.bn2.beta,
            self.fc1.weights, self.fc1.bias,
            self.fc2.weights, self.fc2.bias
        ]


# =============================================================================
# TRAINING LOOP - Your Modules Working Together
# =============================================================================

def train_cifar_cnn(model, train_loader, epochs=3, learning_rate=0.001):
    """Train CNN using YOUR complete TinyTorch system."""
    print(f"\nTraining CIFAR-10 CNN...")
    print(f"   Dataset: {len(train_loader.dataset)} images")
    print(f"   Batch size: {train_loader.batch_size}")

    model.train()  # BatchNorm uses batch statistics

    # YOUR Adam optimizer (Module 06)
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)

    for epoch in range(epochs):
        print(f"\n   Epoch {epoch+1}/{epochs}:")
        epoch_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        # YOUR DataLoader (Module 08) yields batches
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            if batch_idx >= 100:  # Demo mode
                break

            # Forward: YOUR CNN extracts features
            outputs = model(batch_data)

            # Loss computation (manual cross-entropy)
            batch_size = len(batch_labels.data)
            num_classes = 10
            targets_one_hot = np.zeros((batch_size, num_classes))
            for i in range(batch_size):
                targets_one_hot[i, int(batch_labels.data[i])] = 1.0

            outputs_np = np.array(outputs.data.data if hasattr(outputs.data, 'data') else outputs.data)
            exp_outputs = np.exp(outputs_np - np.max(outputs_np, axis=1, keepdims=True))
            softmax_outputs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

            eps = 1e-8
            loss_value = -np.mean(np.sum(targets_one_hot * np.log(softmax_outputs + eps), axis=1))
            loss = Tensor([loss_value])

            # Backward: YOUR autograd (Module 05) computes gradients
            optimizer.zero_grad()
            loss.backward()

            # Update: YOUR optimizer (Module 06) adjusts parameters
            optimizer.step()

            # Track accuracy
            predictions = np.argmax(outputs_np, axis=1)
            correct += np.sum(predictions == batch_labels.data.flatten())
            total += len(batch_labels.data)

            epoch_loss += loss_value
            batch_count += 1

            if (batch_idx + 1) % 20 == 0:
                acc = 100 * correct / total
                print(f"      Batch {batch_idx+1}: Loss={loss_value:.4f}, Acc={acc:.1f}%")

        epoch_acc = 100 * correct / total
        avg_loss = epoch_loss / max(1, batch_count)
        print(f"   Epoch Complete: Loss={avg_loss:.4f}, Accuracy={epoch_acc:.1f}%")

    return model


def test_cifar_cnn(model, test_loader, class_names):
    """Test YOUR CNN on CIFAR-10 test set."""
    print("\nTesting CNN...")

    model.eval()  # BatchNorm uses running statistics

    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
        if batch_idx >= 20:  # Demo mode
            break

        outputs = model(batch_data)

        outputs_np = np.array(outputs.data.data if hasattr(outputs.data, 'data') else outputs.data)
        predictions = np.argmax(outputs_np, axis=1)
        batch_y = batch_labels.data.flatten()
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)

        for j in range(len(batch_y)):
            label = int(batch_y[j])
            class_total[label] += 1
            if predictions[j] == label:
                class_correct[label] += 1

    accuracy = 100 * correct / total
    return accuracy, class_correct, class_total


# =============================================================================
# VISUALIZATIONS - Teaching Aids (Not Core Pipeline)
# =============================================================================
#
# These functions help you understand what's happening inside YOUR modules.
# They're not part of the ML pipeline - just educational displays.

def visualize_dataloader(train_size, test_size, batch_size):
    """Show how YOUR DataLoader processes data."""
    batches_per_epoch = (train_size + batch_size - 1) // batch_size
    last_batch_size = train_size % batch_size or batch_size

    print("\n" + "=" * 70)
    print("YOUR DataLoader in Action (Module 08)")
    print("=" * 70)
    print()
    print("  Dataset Configuration:")
    print("  +" + "-" * 20 + "+" + "-" * 44 + "+")
    print(f"  | {'Training Images':<18} | {train_size:<42} |")
    print(f"  | {'Test Images':<18} | {test_size:<42} |")
    print(f"  | {'Batch Size':<18} | {batch_size:<42} |")
    print(f"  | {'Batches/Epoch':<18} | {batches_per_epoch:<42} |")
    print(f"  | {'Shuffle':<18} | {'Enabled (training only)':<42} |")
    print("  +" + "-" * 20 + "+" + "-" * 44 + "+")
    print()
    print("  How YOUR DataLoader Processes Data:")
    print()
    print(f"  Raw Dataset ({train_size:,} images):")
    print("  +-----------------------------------------------+")
    print("  | img_0 | img_1 | img_2 | ... | img_49999       |")
    print("  +-----------------------------------------------+")
    print("              |")
    print("              v YOUR DataLoader shuffles indices")
    print("              |")
    print("  Shuffled Order: [23451, 8923, 45021, 102, ...]")
    print("              |")
    print("              v YOUR DataLoader creates batches")
    print("              |")
    print(f"  +--------------+  +--------------+       +--------------+")
    print(f"  | Batch 1      |  | Batch 2      |  ...  | Batch {batches_per_epoch:<5} |")
    print(f"  | {batch_size} images   |  | {batch_size} images   |       | {last_batch_size:<2} images   |")
    print(f"  | shuffled!    |  | shuffled!    |       | (remainder)  |")
    print(f"  +--------------+  +--------------+       +--------------+")
    print("              |")
    print("              v Fed to YOUR CNN one batch at a time")
    print("              |")
    print(f"  Memory: Only {batch_size} images loaded at once (not {train_size:,}!)")
    print()
    print("  Why This Matters:")
    mem_all = train_size * 32 * 32 * 3 / (1024 * 1024)
    mem_batch = batch_size * 32 * 32 * 3 / (1024 * 1024)
    print(f"    Without batching: {train_size:,} x 32x32x3 = {mem_all:.1f} MB")
    print(f"    With YOUR DataLoader: {batch_size} x 32x32x3 = {mem_batch:.2f} MB per batch")
    print(f"    Shuffling prevents model from memorizing order")
    print("=" * 70)


def visualize_cnn_architecture():
    """Show CNN architecture and feature hierarchy."""
    print("\n" + "=" * 70)
    print("YOUR CNN Architecture")
    print("=" * 70)
    print("""
    Input (32x32x3)
         |
         v
    +------------------+
    | Conv2d 3->32     |  YOUR Module 09: 3x3 kernels detect edges
    | BatchNorm2d      |  Normalize activations for stable training
    | ReLU             |  YOUR Module 02: Non-linearity
    | MaxPool2d 2x2    |  YOUR Module 09: Reduce to 15x15
    +------------------+
         |
         v
    +------------------+
    | Conv2d 32->64    |  Higher-level features (shapes, textures)
    | BatchNorm2d      |
    | ReLU             |
    | MaxPool2d 2x2    |  Reduce to 6x6
    +------------------+
         |
         v
    +------------------+
    | Flatten          |  64*6*6 = 2304 features
    | Linear 2304->256 |  YOUR Module 03
    | ReLU             |
    | Linear 256->10   |  10 class probabilities
    +------------------+
         |
         v
    Output: [plane, car, bird, cat, deer, dog, frog, horse, ship, truck]

    Why CNNs Excel:
    - Local connectivity: 3x3 kernel sees nearby pixels (not all 1024)
    - Weight sharing: Same filter works everywhere in image
    - Hierarchical: Edges -> Shapes -> Objects
    """)
    print("=" * 70)


def show_results(accuracy, class_correct, class_total, class_names, train_time, dataset_size, epochs):
    """Display final results."""
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\n   Overall Test Accuracy: {accuracy:.1f}%")

    print("\n   Per-Class Performance:")
    print("   " + "-" * 50)
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            bar = "=" * int(class_acc / 5) + " " * (20 - int(class_acc / 5))
            print(f"   {name:10} [{bar}] {class_acc:5.1f}%")
    print("   " + "-" * 50)

    print(f"\n   Training time: {train_time:.1f}s")
    print(f"   Images/sec: {dataset_size * epochs / train_time:.0f}")

    print("\n" + "=" * 70)
    print("What YOU Accomplished")
    print("=" * 70)
    print("""
    YOUR Conv2d (Module 09) extracted spatial features from natural images
    YOUR MaxPool2d (Module 09) reduced dimensions while preserving information
    YOUR DataLoader (Module 08) efficiently batched 50,000 images
    YOUR Adam (Module 06) optimized 600K+ parameters
    YOUR complete ML system achieved {:.1f}% on CIFAR-10!
    """.format(accuracy))

    if accuracy >= 65:
        print("   EXCELLENT! YOUR CNN mastered natural image recognition!")
    elif accuracy >= 50:
        print("   Good progress! YOUR CNN is learning visual features.")
    else:
        print("   YOUR CNN is still learning... (normal for demo mode)")

    print("\n   Next: Continue to Transformers (Milestone 05) after Module 13")
    print("=" * 70)


# =============================================================================
# MAIN - Orchestrating Your Modules
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN with YOUR TinyTorch')
    parser.add_argument('--test-only', action='store_true', help='Test architecture only')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--quick-test', action='store_true', help='Use small subset')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    args = parser.parse_args()

    print("=" * 70)
    print("CNN Revolution (1998) - Part 2: CIFAR-10")
    print("=" * 70)
    print("\nProve YOUR spatial modules work on natural images!")
    print("This milestone showcases YOUR DataLoader handling 50,000 images.")

    class_names = ['plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # --- Step 1: Load Data ---
    print("\nLoading CIFAR-10 dataset...")
    data_manager = DatasetManager()
    (train_data, train_labels), (test_data, test_labels) = data_manager.get_cifar10()
    print(f"Loaded {len(train_data)} training, {len(test_data)} test images")

    if args.quick_test:
        train_data = train_data[:1000]
        train_labels = train_labels[:1000]
        test_data = test_data[:500]
        test_labels = test_labels[:500]
        print("(Using subset for quick testing)")

    # --- Step 2: Create DataLoaders ---
    # YOUR Dataset wraps the raw arrays
    train_dataset = CIFARDataset(train_data, train_labels, transform=train_transforms)
    test_dataset = CIFARDataset(test_data, test_labels, transform=None)

    # YOUR DataLoader handles batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    if not args.no_viz:
        visualize_dataloader(len(train_dataset), len(test_dataset), args.batch_size)
        visualize_cnn_architecture()

    # --- Step 3: Build Model ---
    model = CIFARCNN()

    if args.test_only:
        print("\nARCHITECTURE TEST MODE")
        test_data_mini = np.random.randn(2, 3, 32, 32).astype(np.float32)
        test_labels_mini = np.array([0, 1], dtype=np.int64)
        mini_dataset = CIFARDataset(test_data_mini, test_labels_mini)
        mini_loader = DataLoader(mini_dataset, batch_size=1, shuffle=False)
        for batch_data, batch_labels in mini_loader:
            test_output = model(batch_data)
            print(f"Forward pass successful! Output shape: {test_output.data.shape}")
            break
        return

    # --- Step 4: Train ---
    start_time = time.time()
    model = train_cifar_cnn(model, train_loader, epochs=args.epochs)
    train_time = time.time() - start_time

    # --- Step 5: Test ---
    accuracy, class_correct, class_total = test_cifar_cnn(model, test_loader, class_names)

    # --- Step 6: Results ---
    show_results(accuracy, class_correct, class_total, class_names,
                 train_time, len(train_dataset), args.epochs)


if __name__ == "__main__":
    main()
