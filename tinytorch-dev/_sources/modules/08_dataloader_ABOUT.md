---
title: "DataLoader - Data Pipeline Engineering"
description: "Build production-grade data loading infrastructure for efficient ML training"
difficulty: "⭐⭐⭐"
time_estimate: "4-5 hours"
prerequisites: ["Tensor", "Layers", "Training"]
next_steps: ["Spatial (CNNs)"]
learning_objectives:
  - "Design memory-efficient dataset abstractions for scalable training"
  - "Implement batching and shuffling for mini-batch gradient descent"
  - "Master the Python iterator protocol for streaming data pipelines"
  - "Understand PyTorch's DataLoader architecture and design patterns"
  - "Analyze trade-offs between batch size, memory usage, and throughput"
---

# 08. DataLoader

**ARCHITECTURE TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 4-5 hours

## Overview

This module implements the data loading infrastructure that powers neural network training at scale. You'll build the Dataset/DataLoader abstraction pattern used by PyTorch, TensorFlow, and every major ML framework—implementing batching, shuffling, and memory-efficient iteration from first principles. This is where data engineering meets systems thinking.

## Learning Objectives

By the end of this module, you will be able to:

- **Design Dataset Abstractions**: Implement the protocol-based interface (`__getitem__`, `__len__`) that separates data storage from data access
- **Build Efficient DataLoaders**: Create batching and shuffling mechanisms that stream data without loading entire datasets into memory
- **Master Iterator Patterns**: Understand how Python's `for` loops work under the hood and implement custom iterators
- **Optimize Data Pipelines**: Analyze throughput bottlenecks and balance batch size against memory constraints
- **Apply to Real Datasets**: Use your DataLoader with actual image datasets like MNIST and CIFAR-10 in milestone projects

## Build → Use → Optimize

This module follows TinyTorch's **Build → Use → Optimize** framework:

1. **Build**: Implement Dataset abstraction, TensorDataset for in-memory data, and DataLoader with batching/shuffling
2. **Use**: Load synthetic datasets, create train/validation splits, and integrate with training loops
3. **Optimize**: Profile throughput, analyze memory scaling, and measure shuffle overhead

## Implementation Guide

### Dataset Abstraction

The foundation of all data loading—a protocol-based interface for accessing samples:

```python
from abc import ABC, abstractmethod

class Dataset(ABC):
    """
    Abstract base class defining the dataset interface.

    All datasets must implement:
    - __len__(): Return total number of samples
    - __getitem__(idx): Return sample at given index

    This enables Pythonic usage:
        len(dataset)       # How many samples?
        dataset[42]        # Get sample 42
        for x in dataset   # Iterate over all samples
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """Return sample at given index."""
        pass
```

**Why This Design:**
- **Protocol-based**: Uses Python's `__len__` and `__getitem__` for natural syntax
- **Framework-agnostic**: Same pattern used by PyTorch, TensorFlow, JAX
- **Separation of concerns**: Decouples *what data exists* from *how to load it*
- **Enables optimization**: Makes caching, prefetching, and parallel loading possible

### TensorDataset Implementation

When your data fits in memory, TensorDataset provides efficient access:

```python
class TensorDataset(Dataset):
    """
    Dataset for in-memory tensors.

    Wraps multiple tensors with aligned first dimension:
        features: (N, feature_dim)
        labels: (N,)

    Returns tuple of tensors for each sample:
        dataset[i] → (features[i], labels[i])
    """

    def __init__(self, *tensors):
        """Store tensors, validate first dimension alignment."""
        assert len(tensors) > 0
        first_size = len(tensors[0].data)
        for tensor in tensors:
            assert len(tensor.data) == first_size
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0].data)

    def __getitem__(self, idx: int):
        return tuple(Tensor(t.data[idx]) for t in self.tensors)
```

**Key Features:**
- **Memory locality**: All data pre-loaded for fast access
- **Vectorized operations**: No conversion overhead during training
- **Flexible**: Handles any number of aligned tensors (features, labels, metadata)

### DataLoader with Batching and Shuffling

The core engine that transforms samples into training-ready batches:

```python
class DataLoader:
    """
    Efficient batch loader with shuffling support.

    Transforms:
        Individual samples → Batched tensors

    Features:
    - Automatic batching with configurable batch_size
    - Optional shuffling for training randomization
    - Memory-efficient iteration (one batch at a time)
    - Handles uneven final batch automatically
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Yield batches of data.

        Algorithm:
        1. Generate indices [0, 1, ..., N-1]
        2. Shuffle indices if requested
        3. Group into chunks of batch_size
        4. Load samples and collate into batch tensors
        5. Yield each batch
        """
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self._collate_batch(batch)

    def _collate_batch(self, batch):
        """Stack individual samples into batch tensors."""
        num_tensors = len(batch[0])
        batched_tensors = []

        for tensor_idx in range(num_tensors):
            tensor_list = [sample[tensor_idx].data for sample in batch]
            batched_data = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(batched_data))

        return tuple(batched_tensors)
```

**The Batching Transformation:**

```
Individual Samples (from Dataset):
  dataset[0] → (features: [1, 2, 3], label: 0)
  dataset[1] → (features: [4, 5, 6], label: 1)
  dataset[2] → (features: [7, 8, 9], label: 0)

DataLoader Batching (batch_size=2):
  Batch 1:
    features: [[1, 2, 3],    ← Shape: (2, 3)
               [4, 5, 6]]
    labels: [0, 1]           ← Shape: (2,)

  Batch 2:
    features: [[7, 8, 9]]    ← Shape: (1, 3) [last batch]
    labels: [0]              ← Shape: (1,)
```

## Getting Started

### Prerequisites

Ensure you understand the foundations:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules
tito test tensor
tito test layers
tito test training
```

**Required Knowledge:**
- Tensor operations and NumPy arrays (Module 01)
- Neural network basics (Modules 03-04)
- Training loop structure (Module 07)
- Python protocols (`__getitem__`, `__len__`, `__iter__`)

### Development Workflow

1. **Open the development file**: `modules/08_dataloader/dataloader.py`
2. **Implement Dataset abstraction**: Define abstract base class with `__len__` and `__getitem__`
3. **Build TensorDataset**: Create concrete implementation for tensor-based data
4. **Create DataLoader**: Implement batching, shuffling, and iterator protocol
5. **Test integration**: Verify with training workflow simulation
6. **Export and verify**: `tito module complete 08 && tito test dataloader`

## Testing

### Comprehensive Test Suite

Run the full test suite to verify DataLoader functionality:

```bash
# TinyTorch CLI (recommended)
tito test dataloader

# Direct pytest execution
python -m pytest tests/ -k dataloader -v
```

### Test Coverage Areas

- ✅ **Dataset Interface**: Abstract base class enforcement, protocol implementation
- ✅ **TensorDataset**: Tensor alignment validation, indexing correctness
- ✅ **DataLoader Batching**: Batch shape consistency, handling uneven final batch
- ✅ **Shuffling**: Randomization correctness, deterministic seeding
- ✅ **Training Integration**: Complete workflow with train/validation splits

### Inline Testing & Validation

The module includes comprehensive unit tests:

```python
# Run inline tests during development
python modules/08_dataloader/dataloader.py

# Expected output:
🔬 Unit Test: Dataset Abstract Base Class...
✅ Dataset is properly abstract
✅ Dataset interface works correctly!

🔬 Unit Test: TensorDataset...
✅ TensorDataset works correctly!

🔬 Unit Test: DataLoader...
✅ DataLoader works correctly!

🔬 Unit Test: DataLoader Deterministic Shuffling...
✅ Deterministic shuffling works correctly!

🔬 Integration Test: Training Workflow...
✅ Training integration works correctly!
```

### Manual Testing Examples

```python
from tinytorch.core.tensor import Tensor
from tinytorch.core.dataloader import TensorDataset, DataLoader

# Create synthetic dataset
features = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
labels = Tensor([0, 1, 0, 1])
dataset = TensorDataset(features, labels)

# Create DataLoader with batching
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through batches
for batch_features, batch_labels in loader:
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    # Output: (2, 2) and (2,)
```

## Systems Thinking Questions

### Real-World Applications

- **Image Classification**: How would you design a DataLoader for ImageNet (1.2M images, 150GB)? What if the dataset doesn't fit in RAM?
- **Language Modeling**: LLM training streams billions of tokens—how does batch size affect memory and throughput for variable-length sequences?
- **Autonomous Vehicles**: Tesla trains on terabytes of sensor data—how would you handle multi-modal data (camera + LIDAR + GPS) in a DataLoader?
- **Medical Imaging**: 3D CT scans are too large for GPU memory—what batching strategy would you use for patch extraction?

### Performance Characteristics

- **Memory Scaling**: Why does doubling batch size double memory usage? What memory components scale with batch size (activations, gradients, optimizer states)?
- **Throughput Bottleneck**: Your GPU can process 1000 images/sec but disk reads at 100 images/sec—where's the bottleneck? How would you diagnose this?
- **Shuffle Overhead**: Does shuffling slow down training? Measure the overhead and explain when it becomes significant.
- **Batch Size Trade-off**: What's the optimal batch size for training ResNet-50 on a 16GB GPU? How would you find it systematically?

### Data Pipeline Theory

- **Iterator Protocol**: How does Python's `for` loop work under the hood? What methods must an object implement to be iterable?
- **Memory Efficiency**: Why can DataLoader handle datasets larger than RAM? What design pattern enables this?
- **Collation Strategy**: Why do we stack individual samples into batch tensors? What happens if we don't?
- **Shuffling Impact**: How does shuffling affect gradient estimates and convergence? What happens if you forget to shuffle training data?

## Ready to Build?

You're about to implement the data loading infrastructure that powers modern AI systems. Understanding how to build efficient, scalable data pipelines is critical for production ML engineering—this isn't just plumbing, it's a first-class systems problem with dedicated engineering teams at major AI labs.

Every production training system depends on robust data loaders. Your implementation will follow the exact patterns used by PyTorch's `torch.utils.data.DataLoader` and TensorFlow's `tf.data.Dataset`—the same code running at Meta, Tesla, OpenAI, and every major ML organization.

Open `/Users/VJ/GitHub/TinyTorch/modules/08_dataloader/dataloader.py` and start building. Take your time with each component, run the inline tests frequently, and think deeply about the memory and throughput trade-offs you're making.

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/08_dataloader/dataloader_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/08_dataloader/dataloader_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/08_dataloader/dataloader.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

**After completing this module**, you'll apply your DataLoader to real datasets in the milestone projects:
- **Milestone 03**: Train MLP on MNIST handwritten digits (28×28 images)
- **Milestone 04**: Train CNN on CIFAR-10 natural images (32×32×3 images)

These milestones include download utilities and preprocessing for production datasets.
```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/07_training.html" title="previous page">← Previous Module: Training</a>
<a class="right-next" href="../chapters/09_spatial.html" title="next page">Next Module: Spatial (CNNs) →</a>
</div>
