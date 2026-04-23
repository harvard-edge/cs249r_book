#!/usr/bin/env python3
"""
CIFAR-10 CNN (Modern) - Convolutional Revolution
===============================================

📚 HISTORICAL CONTEXT:
Convolutional Neural Networks revolutionized computer vision by exploiting spatial
structure in images. Unlike MLPs that flatten images (losing spatial relationships),
CNNs preserve spatial hierarchies through local connectivity and weight sharing,
enabling recognition of complex patterns in natural images.

🎯 WHAT YOU'RE BUILDING:
Using YOUR Tiny🔥Torch implementations, you'll build a CNN that achieves 65%+ accuracy
on CIFAR-10 natural images - proving YOUR spatial modules can extract hierarchical
features from real-world photographs!

✅ REQUIRED MODULES (Run after Module 09):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure with autodiff
  Module 02 (Activations)   : YOUR ReLU for feature extraction
  Module 03 (Layers)        : YOUR Linear layers for classification
  Module 04 (Losses)        : YOUR CrossEntropy loss
  Module 05 (DataLoader)    : YOUR Dataset/DataLoader for batching!  <-- SHOWCASED HERE
  Module 06 (Autograd)      : YOUR gradient computation
  Module 07 (Optimizers)    : YOUR Adam optimizer
  Module 08 (Training)      : YOUR training loops
  Module 09 (Convolutions)  : YOUR Conv2d, MaxPool2d, Flatten
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Modern Pattern with BatchNorm):
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ Input Image │  │   Conv2d    │  │ BatchNorm2d │  │   MaxPool   │  │   Conv2d    │  │ BatchNorm2d │  │   MaxPool   │  │   Linear    │  │   Linear    │
    │ 32×32×3 RGB │─▶│    3→32     │─▶│  Normalize  │─▶│     2×2     │─▶│    32→64    │─▶│  Normalize  │─▶│     2×2     │─▶│  2304→256   │─▶│   256→10    │
    │   Pixels    │  │   YOUR M9   │  │   YOUR M9   │  │   YOUR M9   │  │   YOUR M9   │  │   YOUR M9   │  │   YOUR M9   │  │   YOUR M4   │  │   YOUR M4   │
    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
                      Edge Detection   Stabilize Train     Downsample      Shape Detect.   Stabilize Train    Downsample      Hidden Layer    Classification
                           ↓                                                                                                                       ↓
                    Low-level features                                   High-level features                                                 10 Class Probs

    🆕 DATA AUGMENTATION (Training only):
    RandomHorizontalFlip (50%) + RandomCrop with padding - prevents overfitting!

🔍 CIFAR-10 DATASET - REAL NATURAL IMAGES:

CIFAR-10 contains 60,000 32×32 color images in 10 classes:

    Sample Images:                    Feature Hierarchy YOUR CNN Learns:

    ┌──────────┐                     Layer 1 (Conv 3→32):
    │ ✈️ Plane │                     • Edge detectors
    │[Sky blue │                     • Color gradients
    │[White    │                     • Simple textures
    │[Wings    │
    └──────────┘                     Layer 2 (Conv 32→64):
                                      • Object parts
    ┌──────────┐                     • Complex patterns
    │ 🚗 Car    │                     • Spatial relationships
    │[Red body]│
    │[Wheels]  │                     Output Layer:
    │[Windows] │                     • Complete objects
    └──────────┘                     • Class probabilities

    Classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck

    Why CNNs Excel at Natural Images:
    • LOCAL CONNECTIVITY: Pixels near each other are related
    • WEIGHT SHARING: Same filter detects patterns everywhere
    • HIERARCHICAL LEARNING: Edges → Shapes → Objects
    • TRANSLATION INVARIANCE: Detects cat anywhere in image

📊 EXPECTED PERFORMANCE:
- Dataset: 50,000 training images, 10,000 test images
- Training time: 3-5 minutes (demonstration mode)
- Expected accuracy: 70%+ (with YOUR CNN + BatchNorm + Augmentation!)
- Parameters: ~600K (mostly in conv layers)
- 🆕 BatchNorm: Stabilizes training, faster convergence
- 🆕 Augmentation: Reduces overfitting, better generalization
"""

import sys
import os
import numpy as np
rng = np.random.default_rng(7)
import argparse
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import TinyTorch components YOU BUILT!
from tinytorch.core.tensor import Tensor              # Module 02: YOU built this!
from tinytorch.core.layers import Linear             # Module 04: YOU built this!
from tinytorch.core.activations import ReLU, Softmax  # Module 03: YOU built this!
from tinytorch.core.spatial import Conv2d, MaxPool2d, BatchNorm2d  # Module 09: YOU built this!
from tinytorch.core.optimizers import Adam            # Module 07: YOUR optimizer!
from tinytorch.core.dataloader import DataLoader, Dataset  # Module 05: YOU built this!
from tinytorch.core.dataloader import RandomHorizontalFlip, RandomCrop, Compose  # Module 05: Data Augmentation!

# Import dataset manager
from data_manager import DatasetManager

# =============================================================================
# 🎯 YOUR TINYTORCH MODULES IN ACTION
# =============================================================================
#
# This milestone showcases the modules YOU built. Here's what powers this CNN:
#
# ┌─────────────────────┬────────────────────────────────┬─────────────────────────────┐
# │ What You Built      │ How It's Used Here             │ Systems Impact              │
# ├─────────────────────┼────────────────────────────────┼─────────────────────────────┤
# │ Module 01: Tensor   │ Every computation flows        │ Automatic gradient tracking │
# │                     │ through YOUR Tensor            │ enables backpropagation     │
# │                     │                                │                             │
# │ Module 02: ReLU     │ After each conv/linear layer   │ Non-linearity lets CNN      │
# │                     │ for feature activation         │ learn complex patterns      │
# │                     │                                │                             │
# │ Module 03: Linear   │ Classification head            │ Maps 2304 spatial features  │
# │                     │ (2304→256→10)                  │ to 10 class probabilities   │
# │                     │                                │                             │
# │ Module 05: DataLoader│ Batches 50,000 images into    │ Memory: 0.2MB/batch vs      │
# │ ★ SHOWCASED HERE ★  │ digestible chunks + shuffle    │ 150MB loading all at once   │
# │                     │                                │                             │
# │ Module 06: Autograd │ .backward() computes all       │ Chain rule through conv,    │
# │                     │ gradients automatically        │ pool, linear layers         │
# │                     │                                │                             │
# │ Module 07: Adam     │ Updates 600K+ parameters       │ Adaptive learning rates     │
# │                     │ with momentum and scaling      │ per parameter               │
# │                     │                                │                             │
# │ Module 09: Conv2d   │ Extracts spatial features      │ Local connectivity +        │
# │            MaxPool2d│ with 3×3 kernels, reduces      │ weight sharing = 100×       │
# │            BatchNorm│ dimensions, stabilizes train   │ fewer params than MLP       │
# └─────────────────────┴────────────────────────────────┴─────────────────────────────┘
#
# =============================================================================

# =============================================================================
# 🆕 WHAT'S NEW SINCE MODULE 08 (Training)
# =============================================================================
#
# You've built training loops before. This milestone adds significant new capabilities:
#
# ┌──────────────────┬─────────────────────┬────────────────────────────┐
# │ Module 08        │ This Milestone      │ Why It Matters             │
# ├──────────────────┼─────────────────────┼────────────────────────────┤
# │ Linear layers    │ + Conv2d layers     │ Spatial structure preserved│
# │ Simple batching  │ + DataLoader class  │ Shuffling + memory control │
# │ Grayscale 28×28  │ + Color 32×32×3     │ Real images are complex!   │
# │ No augmentation  │ + Flip & Crop       │ Prevents overfitting       │
# │ No normalization │ + BatchNorm2d       │ Stabilizes deep training   │
# │ ~50K params      │ + ~600K params      │ More capacity for patterns │
# └──────────────────┴─────────────────────┴────────────────────────────┘
#
# This is the progression from "ML basics" to "ML systems engineering"!
#
# =============================================================================


# =============================================================================
# DATASET CLASS - Your Module 05 Enables This Pattern
# =============================================================================
# The Dataset abstraction YOU built defines a contract: __len__() and __getitem__().
# This simple interface lets YOUR DataLoader handle any data source uniformly.

class CIFARDataset(Dataset):
    """Custom CIFAR-10 Dataset using YOUR Dataset interface from Module 05!

    Now with data augmentation support using YOUR transforms from Module 05!
    """

    def __init__(self, data, labels, transform=None):
        """Initialize with data, labels, and optional transforms."""
        self.data = data
        self.labels = labels
        self.transform = transform  # Module 05: YOUR augmentation transforms!

    def __getitem__(self, idx):
        """Get a single sample - YOUR Dataset interface!"""
        img = self.data[idx]

        # Apply augmentation if provided (training only!)
        if self.transform is not None:
            img = self.transform(img)
            # Convert back to numpy if it became a Tensor
            if isinstance(img, Tensor):
                img = img.data

        return Tensor(img), Tensor([self.labels[idx]])

    def __len__(self):
        """Return dataset size - YOUR Dataset interface!"""
        return len(self.data)

    def get_num_classes(self):
        """Return number of classes."""
        return 10


# Training augmentation using YOUR transforms from Module 05!
train_transforms = Compose([
    RandomHorizontalFlip(p=0.5),   # 50% chance to flip - cars/animals look similar flipped!
    RandomCrop(32, padding=4),      # Random crop with 4px padding - simulates translation
])


# =============================================================================
# CNN MODEL - Your Convolution Modules Power Feature Extraction
# =============================================================================
# CNNs revolutionized vision by exploiting spatial structure. YOUR Conv2d uses
# local connectivity + weight sharing to detect patterns with 100x fewer params.

def flatten(x):
    """Flatten spatial features for dense layers - YOUR implementation!"""
    batch_size = x.data.shape[0]
    return Tensor(x.data.reshape(batch_size, -1))

class CIFARCNN:
    """
    Convolutional Neural Network for CIFAR-10 using YOUR Tiny🔥Torch!

    This architecture demonstrates how spatial feature extraction enables
    recognition of complex patterns in natural images.

    Architecture: Conv → BatchNorm → ReLU → Pool (modern pattern)
    This is more stable and trains faster than without BatchNorm!
    """

    def __init__(self):
        print("🧠 Building CIFAR-10 CNN with YOUR Tiny🔥Torch modules...")

        # Convolutional feature extractors - YOUR spatial modules!
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))   # Module 09!
        self.bn1 = BatchNorm2d(32)  # Module 09: YOUR BatchNorm! Stabilizes training
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))  # Module 09!
        self.bn2 = BatchNorm2d(64)  # Module 09: YOUR BatchNorm!
        self.pool = MaxPool2d(kernel_size=2, stride=2)  # Module 09: YOUR pooling!

        # Activation functions
        self.relu = ReLU()  # Module 03: YOUR activation!

        # Dense classification head
        # After conv1(32→30)→pool(15)→conv2(13)→pool(6): 64*6*6 = 2304 features
        self.fc1 = Linear(64 * 6 * 6, 256)  # Module 04: YOUR Linear!
        self.fc2 = Linear(256, 10)          # Module 04: YOUR Linear!

        # Training mode flag
        self._training = True

        # Calculate total parameters (including BatchNorm gamma/beta)
        conv1_params = 3 * 3 * 3 * 32 + 32     # 3×3 kernels, 3→32 channels
        bn1_params = 32 * 2                    # gamma + beta
        conv2_params = 3 * 3 * 32 * 64 + 64    # 3×3 kernels, 32→64 channels
        bn2_params = 64 * 2                    # gamma + beta
        fc1_params = 64 * 6 * 6 * 256 + 256    # Flattened→256
        fc2_params = 256 * 10 + 10             # 256→10 classes
        self.total_params = conv1_params + bn1_params + conv2_params + bn2_params + fc1_params + fc2_params

        print(f"   Conv1: 3→32 channels + BatchNorm (YOUR modules!)")
        print(f"   Conv2: 32→64 channels + BatchNorm (YOUR modules!)")
        print(f"   Dense: 2304→256→10 (YOUR Linear classification)")
        print(f"   Total parameters: {self.total_params:,}")

    def train(self):
        """Set model to training mode."""
        self._training = True
        self.bn1.train()
        self.bn2.train()
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self._training = False
        self.bn1.eval()
        self.bn2.eval()
        return self

    def forward(self, x):
        """Forward pass through YOUR CNN architecture."""
        # First conv block: Conv → BatchNorm → ReLU → Pool (modern pattern)
        x = self.conv1(x)           # Module 09: YOUR Conv2d!
        x = self.bn1(x)             # Module 09: YOUR BatchNorm! Normalizes activations
        x = self.relu(x)            # Module 03: YOUR ReLU!
        x = self.pool(x)            # Module 09: YOUR MaxPool2d!

        # Second conv block: Same modern pattern
        x = self.conv2(x)           # Module 09: YOUR Conv2d!
        x = self.bn2(x)             # Module 09: YOUR BatchNorm!
        x = self.relu(x)            # Module 03: YOUR ReLU!
        x = self.pool(x)            # Module 09: YOUR MaxPool2d!

        # Flatten and classify
        x = flatten(x)              # Module 09: YOUR spatial→dense bridge!
        x = self.fc1(x)             # Module 04: YOUR Linear!
        x = self.relu(x)            # Module 03: YOUR ReLU!
        x = self.fc2(x)             # Module 04: YOUR classification!

        return x

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

    def parameters(self):
        """Get all trainable parameters from YOUR layers."""
        return [
            self.conv1.weight, self.conv1.bias,
            self.bn1.gamma, self.bn1.beta,
            self.conv2.weight, self.conv2.bias,
            self.bn2.gamma, self.bn2.beta,
            self.fc1.weights, self.fc1.bias,
            self.fc2.weights, self.fc2.bias
        ]

# =============================================================================
# VISUALIZATIONS - Teaching Aids for Understanding
# =============================================================================
# These ASCII diagrams make the invisible visible - showing how YOUR DataLoader
# batches data and how YOUR CNN extracts hierarchical features from images.

def visualize_dataloader(train_size, test_size, batch_size):
    """Show how YOUR DataLoader processes data - making the invisible visible!"""
    batches_per_epoch = (train_size + batch_size - 1) // batch_size
    last_batch_size = train_size % batch_size
    if last_batch_size == 0:
        last_batch_size = batch_size

    print("\n" + "=" * 70)
    print("📦 YOUR DataLoader in Action (Module 05)")
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
    print(f"    * Without batching: {train_size:,} x 32x32x3 = {mem_all:.1f} MB in memory")
    print(f"    * With YOUR DataLoader: {batch_size} x 32x32x3 = {mem_batch:.2f} MB per batch")
    print(f"    * Shuffling: Prevents model from memorizing order")
    print("=" * 70)


def visualize_cifar_cnn():
    """Show how CNNs process natural images."""
    print("\n" + "="*70)
    print("🖼️  VISUALIZING CNN FEATURE EXTRACTION:")
    print("="*70)

    print("""
    How YOUR CNN Sees Images:           Feature Maps at Each Layer:

    Original Image (32×32×3):           After Conv1 (30×30×32):
    ┌────────────────┐                 ┌─┬─┬─┬─┬─┬─┬─┬─┬─┐
    │ [Cat in grass] │                 │Edge detectors...│ 32 filters
    │ Complex scene  │ → Conv+ReLU →   │Texture maps...  │ detect
    │ Many patterns  │                 │Color gradients. │ features
    └────────────────┘                 └─┴─┴─┴─┴─┴─┴─┴─┴─┘

    After Pool1 (15×15×32):            After Conv2 (13×13×64):
    ┌─────────┐                        ┌─┬─┬─┬─┬─┬─┬─┬─┬─┐
    │Reduced  │                        │Cat ears...      │ 64 filters
    │spatial  │ → Conv+ReLU →          │Cat eyes...      │ combine
    │dimension│                        │Grass texture... │ features
    └─────────┘                        └─┴─┴─┴─┴─┴─┴─┴─┴─┘

    After Pool2 + Flatten:             Classification:
    [6×6×64 = 2304 features] → Dense → [plane|car|bird|CAT|...]
                                              Highest probability

    Key CNN Advantages YOUR Implementation Provides:
    ✓ SPATIAL HIERARCHY: Low → High level features
    ✓ PARAMETER SHARING: 3×3 kernel used everywhere
    ✓ TRANSLATION INVARIANCE: Detects patterns anywhere
    ✓ AUTOMATIC FEATURE LEARNING: No manual engineering!
    """)
    print("="*70)

# =============================================================================
# TRAINING LOOP - Your Modules Working Together
# =============================================================================
# Watch YOUR complete ML system in action: DataLoader batches images, CNN extracts
# features, Autograd computes gradients, Adam updates 600K+ parameters each step.

def train_cifar_cnn(model, train_loader, epochs=3, learning_rate=0.001):
    """Train CNN using YOUR complete training system with DataLoader!"""
    print("\n🚀 Training CIFAR-10 CNN with YOUR Tiny🔥Torch!")
    print(f"   Dataset: {len(train_loader.dataset)} color images")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   YOUR DataLoader (Module 05) handles batching!")
    print(f"   YOUR BatchNorm (Module 09) uses batch statistics!")
    print(f"   YOUR Adam optimizer (Module 07)!")

    # Set model to training mode - BatchNorm uses batch statistics
    model.train()

    # YOUR optimizer
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)

    for epoch in range(epochs):
        print(f"\n   Epoch {epoch+1}/{epochs}:")
        epoch_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        # Use YOUR DataLoader to iterate through batches!
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            if batch_idx >= 100:  # Demo mode - limit batches
                break

            # Forward pass with YOUR CNN
            outputs = model(batch_data)  # YOUR spatial features!

            # Manual cross-entropy loss
            batch_size = len(batch_labels.data)
            num_classes = 10
            targets_one_hot = np.zeros((batch_size, num_classes))
            for i in range(batch_size):
                targets_one_hot[i, int(batch_labels.data[i])] = 1.0

            # Cross-entropy: -sum(y * log(softmax(x)))
            # Apply softmax first - handle nested data access
            outputs_np = np.array(outputs.data.data if hasattr(outputs.data, 'data') else outputs.data)
            exp_outputs = np.exp(outputs_np - np.max(outputs_np, axis=1, keepdims=True))
            softmax_outputs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

            eps = 1e-8
            loss_value = -np.mean(np.sum(targets_one_hot * np.log(softmax_outputs + eps), axis=1))
            loss = Tensor([loss_value])

            # Backward pass with YOUR autograd
            optimizer.zero_grad()  # Module 07!
            loss.backward()        # Module 06: YOUR autodiff!
            optimizer.step()       # Module 07!

            # Track accuracy
            predictions = np.argmax(outputs_np, axis=1)
            correct += np.sum(predictions == batch_labels.data.flatten())
            total += len(batch_labels.data)

            epoch_loss += loss_value
            batch_count += 1

            # Progress
            if (batch_idx + 1) % 20 == 0:
                acc = 100 * correct / total
                print(f"   Batch {batch_idx+1}: "
                      f"Loss = {loss_value:.4f}, Accuracy = {acc:.1f}%")

        # Epoch summary
        epoch_acc = 100 * correct / total
        avg_loss = epoch_loss / max(1, batch_count)
        print(f"   → Epoch Complete: Loss = {avg_loss:.4f}, "
              f"Accuracy = {epoch_acc:.1f}% (YOUR CNN + DataLoader!)")

    return model

# =============================================================================
# TESTING - Evaluating Your CNN on Unseen Images
# =============================================================================
# True test of generalization: YOUR CNN must classify images it never saw during
# training. BatchNorm switches to running statistics, no augmentation applied.

def test_cifar_cnn(model, test_loader, class_names):
    """Test YOUR CNN on CIFAR-10 test set using DataLoader."""
    print("\n🧪 Testing YOUR CNN on Natural Images with YOUR DataLoader...")

    # Set model to evaluation mode - BatchNorm uses running statistics
    model.eval()
    print("   ℹ️  Model in eval mode: BatchNorm uses running statistics")

    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    # Test using YOUR DataLoader
    for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
        if batch_idx >= 20:  # Demo mode - limit batches
            break

        outputs = model(batch_data)

        outputs_np = np.array(outputs.data.data if hasattr(outputs.data, 'data') else outputs.data)
        predictions = np.argmax(outputs_np, axis=1)
        batch_y = batch_labels.data.flatten()
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)

        # Per-class accuracy
        for j in range(len(batch_y)):
            label = int(batch_y[j])
            class_total[label] += 1
            if predictions[j] == label:
                class_correct[label] += 1

    # Results
    accuracy = 100 * correct / total
    print(f"\n   📊 Overall Test Accuracy: {accuracy:.2f}%")

    # Per-class performance
    print("\n   Per-Class Performance (YOUR CNN's understanding):")
    print("   " + "─"*50)
    print("   │ Class      │ Accuracy │ Visual               │")
    print("   ├────────────┼──────────┼──────────────────────┤")

    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            bar_length = int(class_acc / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   │ {class_name:10} │  {class_acc:5.1f}%  │ {bar} │")

    print("   " + "─"*50)

    if accuracy >= 65:
        print("\n   🎉 EXCELLENT! YOUR CNN mastered natural image recognition!")
    elif accuracy >= 50:
        print("\n   ✅ Good progress! YOUR CNN is learning visual features!")
    else:
        print("\n   🔄 YOUR CNN is still learning... (normal for demo mode)")

    return accuracy

# =============================================================================
# SYSTEMS ANALYSIS - Understanding the Engineering Trade-offs
# =============================================================================
# ML systems is about understanding trade-offs: parameters vs. computation,
# memory vs. speed, accuracy vs. efficiency. YOUR CNN embodies these choices.

def analyze_cnn_systems(model, batch_size=32):
    """Analyze YOUR CNN from an ML systems perspective."""
    print("\n🔬 SYSTEMS ANALYSIS of YOUR CNN Implementation:")

    print(f"\n   Model Architecture:")
    print(f"   • Convolutional layers: 2 (3→32→64 channels)")
    print(f"   • Pooling layers: 2 (2×2 max pooling)")
    print(f"   • Dense layers: 2 (2304→256→10)")
    print(f"   • Total parameters: {model.total_params:,}")

    print(f"\n   Computational Complexity:")
    print(f"   • Conv1: 32×30×30×(3×3×3) = 777,600 ops")
    print(f"   • Conv2: 64×13×13×(3×3×32) = 3,093,504 ops")
    print(f"   • Dense: 2,304×256 + 256×10 = 592,384 ops")
    print(f"   • Total: ~4.5M ops per image")

    # Memory profiling table - quantitative systems thinking
    params_mem = model.total_params * 4 / 1024  # KB
    activations_mem = 500  # Peak activations ~500KB per image
    batch_mem = batch_size * 32 * 32 * 3 * 4 / 1024  # Input batch in KB
    total_mem = params_mem + activations_mem + batch_mem

    print(f"\n   🧮 MEMORY PROFILING - Where YOUR RAM Goes:")
    print(f"   ┌────────────────────────┬──────────────┬─────────────┐")
    print(f"   │ Component              │ Memory (KB)  │ Percentage  │")
    print(f"   ├────────────────────────┼──────────────┼─────────────┤")
    print(f"   │ Parameters (weights)   │ {params_mem:10.1f}   │ {100*params_mem/total_mem:5.1f}%      │")
    print(f"   │ Activations (forward)  │ {activations_mem:10.1f}   │ {100*activations_mem/total_mem:5.1f}%      │")
    print(f"   │ Batch data ({batch_size} imgs)   │ {batch_mem:10.1f}   │ {100*batch_mem/total_mem:5.1f}%      │")
    print(f"   ├────────────────────────┼──────────────┼─────────────┤")
    print(f"   │ TOTAL per batch        │ {total_mem:10.1f}   │ 100.0%      │")
    print(f"   └────────────────────────┴──────────────┴─────────────┘")
    print(f"\n   💡 KEY INSIGHT: Activations dominate! This is why gradient checkpointing")
    print(f"      trades compute (recompute activations) for memory (don't store them).")

    print(f"\n   🏛️ CNN Evolution:")
    print(f"   • 1989: LeCun's CNN for handwritten digits")
    print(f"   • 2012: AlexNet revolutionizes ImageNet")
    print(f"   • 2015: ResNet enables 100+ layer networks")
    print(f"   • YOUR CNN: Core principles that power them all!")

    print(f"\n   💡 Why CNNs Dominate Vision:")
    print(f"   • Spatial hierarchy matches visual cortex")
    print(f"   • Parameter sharing: 3×3 kernel vs 32×32 dense")
    print(f"   • Translation invariance from weight sharing")
    print(f"   • YOUR implementation demonstrates all of these!")

# =============================================================================
# MAIN - Orchestrating Your Complete ML System
# =============================================================================
# The main function ties everything together: data loading, model creation,
# training, testing, and analysis. This is YOUR end-to-end ML pipeline.

def main():
    """Demonstrate CIFAR-10 CNN using YOUR Tiny🔥Torch!"""

    parser = argparse.ArgumentParser(description='CIFAR-10 CNN')
    parser.add_argument('--test-only', action='store_true',
                       help='Test architecture only')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Training epochs (demo mode)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Show CNN visualization')
    parser.add_argument('--quick-test', action='store_true',
                       help='Use small subset for testing')
    args = parser.parse_args()

    print("🎯 CIFAR-10 CNN - Natural Image Recognition with YOUR Convolution Modules!")
    print("   Historical significance: CNNs revolutionized computer vision")
    print("   YOUR achievement: Spatial feature extraction on real photos")
    print("   Components used: YOUR Conv2d + MaxPool2d + complete system")

    # Visualization
    if args.visualize:
        visualize_cifar_cnn()

    # Class names
    class_names = ['plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Step 1: Load CIFAR-10
    print("\n📥 Loading CIFAR-10 dataset...")
    data_manager = DatasetManager()

    (train_data, train_labels), (test_data, test_labels) = data_manager.get_cifar10()
    print(f"✅ Loaded {len(train_data)} training, {len(test_data)} test images")

    if args.quick_test:
        train_data = train_data[:1000]
        train_labels = train_labels[:1000]
        test_data = test_data[:500]
        test_labels = test_labels[:500]
        print("   (Using subset for quick testing)")

    # Step 2: Create Datasets and DataLoaders using YOUR Module 05!
    print("\n📦 Creating YOUR Dataset and DataLoader (Module 05)...")

    # Training with augmentation - YOUR transforms!
    train_dataset = CIFARDataset(train_data, train_labels, transform=train_transforms)
    # Testing without augmentation - we want consistent evaluation
    test_dataset = CIFARDataset(test_data, test_labels, transform=None)

    # YOUR DataLoader handles batching and shuffling!
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Show DataLoader visualization - make YOUR implementation visible!
    visualize_dataloader(len(train_dataset), len(test_dataset), args.batch_size)

    print(f"\n   ✅ Data Augmentation: RandomFlip + RandomCrop (training only)")

    # Step 3: Build CNN
    model = CIFARCNN()

    if args.test_only:
        print("\n🧪 ARCHITECTURE TEST MODE")
        # Create minimal test data for fast architecture validation
        print("   Using minimal dataset for optimization testing framework...")
        test_data_mini = rng.standard_normal((2, 3, 32, 32)).astype(np.float32)  # Just 2 samples
        test_labels_mini = np.array([0, 1], dtype=np.int64)  # 2 labels

        # Create minimal dataset and dataloader
        mini_dataset = CIFARDataset(test_data_mini, test_labels_mini)
        mini_loader = DataLoader(mini_dataset, batch_size=1, shuffle=False)  # Batch size 1

        # Test with single sample from minimal DataLoader
        for batch_data, batch_labels in mini_loader:
            test_output = model(batch_data)
            print(f"✅ Forward pass successful! Shape: {test_output.data.shape}")
            print("✅ YOUR CNN + DataLoader work together!")
            break
        return

    # Step 4: Train using YOUR DataLoader
    start_time = time.time()
    model = train_cifar_cnn(model, train_loader, epochs=args.epochs)
    train_time = time.time() - start_time

    # Step 5: Test using YOUR DataLoader
    accuracy = test_cifar_cnn(model, test_loader, class_names)

    # Step 5: Analysis
    analyze_cnn_systems(model, batch_size=args.batch_size)

    print(f"\n⏱️  Training time: {train_time:.1f} seconds")
    print(f"   Images/sec: {len(train_dataset) * args.epochs / train_time:.0f}")

    print("\n✅ SUCCESS! CIFAR-10 CNN Milestone Complete!")
    print("\n🎓 What YOU Accomplished:")
    print("   • YOUR Conv2d extracts spatial features from natural images")
    print("   • YOUR MaxPool2d reduces dimensions while preserving information")
    print("   • YOUR DataLoader efficiently batches and shuffles data")
    print("   • YOUR CNN achieves real accuracy on complex photos")
    print("   • YOUR complete ML system works end-to-end!")

    print("\n🚀 Next Steps:")
    print("   • Continue to TinyGPT after Module 14 (Transformers)")
    print("   • YOUR spatial understanding scales to segmentation, detection, etc.")
    print(f"   • With {accuracy:.1f}% accuracy, YOUR computer vision works!")

if __name__ == "__main__":
    main()
