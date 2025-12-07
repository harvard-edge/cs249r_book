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

## Common Pitfalls

### Forgetting to Shuffle Training Data

**Problem**: Not shuffling training data causes the model to learn spurious patterns from data ordering, leading to poor generalization and potential overfitting to batch structure.

**Solution**: Always set `shuffle=True` for training data, `shuffle=False` for validation/test:

```python
# ❌ Wrong - no shuffling for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
# Model may learn data ordering patterns instead of true features!

# ✅ Correct - shuffle training, don't shuffle validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Mismatched Tensor Dimensions in TensorDataset

**Problem**: Creating TensorDataset with tensors that have different first dimensions causes indexing errors when loading batches.

**Solution**: Ensure all tensors have aligned first dimension (number of samples):

```python
# ❌ Wrong - mismatched dimensions
features = Tensor(np.random.randn(100, 10))  # 100 samples
labels = Tensor(np.random.randn(50))         # 50 samples (mismatch!)
dataset = TensorDataset(features, labels)    # Error when accessing!

# ✅ Correct - aligned dimensions
features = Tensor(np.random.randn(100, 10))  # 100 samples
labels = Tensor(np.random.randn(100))        # 100 samples (matches!)
dataset = TensorDataset(features, labels)    # Works correctly
```

### Incorrect Batch Size Leading to Single-Batch Training

**Problem**: Setting batch_size larger than dataset length creates single batch with all data, defeating the purpose of mini-batch training.

**Solution**: Choose batch_size appropriate for dataset size and memory constraints:

```python
# ❌ Wrong - batch size exceeds dataset
dataset = TensorDataset(features, labels)  # 100 samples
loader = DataLoader(dataset, batch_size=500, shuffle=True)
# Creates 1 batch with all 100 samples - no mini-batch training!

# ✅ Correct - reasonable batch size
loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Creates ~3 batches (32, 32, 36 samples) - proper mini-batch training
```

### Not Handling Uneven Final Batch

**Problem**: Assuming all batches have identical size causes errors when the final batch contains fewer samples than batch_size.

**Solution**: Design model and training code to handle variable batch sizes:

```python
# ❌ Wrong - assumes fixed batch size
for batch_features, batch_labels in loader:
    # Assumes batch_features.shape[0] == batch_size always
    outputs = model(batch_features)
    assert outputs.shape[0] == 32  # Fails on last batch!

# ✅ Correct - handles variable batch sizes
for batch_features, batch_labels in loader:
    batch_size = batch_features.shape[0]  # Get actual batch size
    outputs = model(batch_features)
    assert outputs.shape[0] == batch_size  # Works for all batches
```

### Modifying Dataset During Iteration

**Problem**: Changing dataset contents while DataLoader is iterating causes inconsistent batches or crashes.

**Solution**: Avoid modifying dataset during training iteration:

```python
# ❌ Wrong - modifying dataset during iteration
for batch in train_loader:
    # Training code...
    train_dataset.tensors[0].data += 1  # Modifying dataset mid-iteration!

# ✅ Correct - modify dataset between epochs
for epoch in range(epochs):
    for batch in train_loader:
        # Training code (don't modify dataset)
        pass
    # Safe to modify dataset here (between epochs)
    if epoch % 10 == 0:
        train_dataset = augment_dataset(train_dataset)
```

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

## Production Context

### Your Implementation vs. Production Frameworks

Understanding what you're building vs. what production frameworks provide:

| Feature | Your DataLoader (Module 08) | PyTorch DataLoader | TensorFlow tf.data |
|---------|---------------------------|-------------------|-------------------|
| **Backend** | NumPy (CPU-only) | Python/C++ (CPU) | C++/XLA |
| **Dataset Protocol** | `__len__`, `__getitem__` | ✅ Same protocol | ✅ `tf.data.Dataset` |
| **Batching** | Manual np.stack | ✅ Automatic collate_fn | ✅ batch() method |
| **Shuffling** | Random index permutation | ✅ Same approach | ✅ shuffle(buffer_size) |
| **Multi-Processing** | ❌ Single thread | ✅ num_workers parameter | ✅ parallel interleave |
| **Prefetching** | ❌ No async loading | ✅ pin_memory, prefetch | ✅ prefetch() method |
| **Data Augmentation** | ❌ Manual | ✅ torchvision.transforms | ✅ tf.image.* functions |
| **Memory Mapping** | ❌ Load all to RAM | ✅ Memory-mapped files | ✅ TFRecord format |
| **Distributed Loading** | ❌ Single device | ✅ DistributedSampler | ✅ distribute strategy |

**Educational Focus**: Your implementation demonstrates core batching and iteration mechanics. Production frameworks add parallel data loading, prefetching, and memory-efficient file formats while maintaining the same Dataset/DataLoader abstraction.

### Side-by-Side Code Comparison

**Your implementation:**
```python
from tinytorch.core.tensor import Tensor
from tinytorch.core.dataloader import TensorDataset, DataLoader

# Create dataset from tensors
features = Tensor(np.random.randn(1000, 784))  # 1000 samples, 784 features
labels = Tensor(np.random.randint(0, 10, 1000))  # 10 classes

dataset = TensorDataset(features, labels)

# Create DataLoader with batching and shuffling
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch_features, batch_labels in train_loader:
    # Training code
    outputs = model(batch_features)
    loss = criterion(outputs, batch_labels)
    loss.backward()
    optimizer.step()
```

**Equivalent PyTorch:**
```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Create dataset from tensors (same API!)
features = torch.randn(1000, 784)
labels = torch.randint(0, 10, (1000,))

dataset = TensorDataset(features, labels)

# Create DataLoader with multi-processing and prefetching
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,    # Parallel data loading
    pin_memory=True   # Faster GPU transfer
)

# Iterate through batches (same API!)
for batch_features, batch_labels in train_loader:
    # Training code
    outputs = model(batch_features)
    loss = criterion(outputs, batch_labels)
    loss.backward()
    optimizer.step()
```

**Key Differences:**
1. **Parallel Loading**: PyTorch spawns worker processes (`num_workers=4`) to load and preprocess data in parallel with training, hiding data loading latency.
2. **Pin Memory**: `pin_memory=True` allocates batches in pinned (non-pageable) memory for faster CPU→GPU transfer.
3. **Collation**: PyTorch's `collate_fn` parameter allows custom batch assembly logic (e.g., padding variable-length sequences).
4. **Samplers**: PyTorch separates shuffling logic into Sampler classes (RandomSampler, SequentialSampler, DistributedSampler) for flexibility.

### Real-World Production Usage

**ImageNet Training**: Standard workflow loads 1.2M 224×224 RGB images (150GB total) using DataLoader with `num_workers=8`. Each worker loads and preprocesses images (decode JPEG, resize, augment) in parallel with GPU training. Prefetching ensures GPU never waits for data. Typical setup: batch_size=256, 8 workers, ~10GB RAM for prefetch buffers.

**Google BERT Pre-training**: Uses TensorFlow's `tf.data.Dataset` to stream billions of tokens from TFRecord files. Data pipeline: read TFRecord → parse examples → shuffle (buffer_size=10K) → batch (batch_size=256) → prefetch (buffer_size=2). Distributed training shards data across TPU pods using `distribute.Strategy` ensuring each TPU sees different data.

**Tesla Autopilot**: Streams terabytes of driving videos from SSD arrays. DataLoader handles multi-modal data: 8 camera streams + LIDAR + GPS + CAN bus. Custom collate_fn synchronizes timestamps across modalities. Uses memory-mapped files to avoid loading entire dataset into RAM. Batch size limited by GPU memory (typically 8-16 clips per GPU).

**Hugging Face Datasets**: Library built on Apache Arrow provides efficient DataLoader for NLP tasks. Supports datasets larger than RAM through memory mapping. Typical workflow: load dataset from Hub → map tokenization → DataLoader for batching. Handles variable-length sequences with dynamic padding collate_fn.

**Medical Imaging (PathAI)**: Whole-slide pathology images (10GB+ per slide) processed via patch extraction. DataLoader yields 256×256 patches sampled from slide. Uses worker processes to decode and extract patches in parallel. Batch size=128 patches, 16 workers to saturate GPU. Careful memory management prevents OOM on large slides.

### Performance Characteristics at Scale

**Data Loading Bottleneck Analysis**: For ResNet-50 ImageNet training on single V100 GPU:
- **Without parallel loading** (num_workers=0): GPU utilization 60% (GPU waits for data 40% of time)
- **With num_workers=4**: GPU utilization 95% (data loading parallelized)
- **With num_workers=8**: GPU utilization 98% (diminishing returns, CPU-bound)
- **Optimal workers**: 4-8 for most systems (balance I/O parallelism vs CPU overhead)

**Memory Scaling with Batch Size**: For ImageNet training (224×224 RGB images):
- **Batch size 32**: ~600MB GPU memory (32 × 224 × 224 × 3 × 4 bytes + activations)
- **Batch size 256**: ~4.8GB GPU memory (8× batch size → 8× activation memory)
- **Batch size 512**: OOM on 16GB GPU (activations + model + gradients exceed capacity)
- **Rule**: Activation memory scales linearly with batch size, often dominates GPU memory

**Shuffle Overhead Measurement**: For 1M sample dataset with batch_size=256:
- **No shuffle**: 0.5 seconds to create indices
- **Shuffle=True**: 1.2 seconds (0.7s shuffle overhead via random.shuffle)
- **Overhead per epoch**: ~0.07% of total training time (negligible)
- **Production practice**: Always shuffle training data despite small overhead

**Prefetching Impact**: For BERT pre-training with tf.data pipeline:
- **Without prefetch**: 180ms per step (110ms training + 70ms data loading serialized)
- **With prefetch(2)**: 115ms per step (110ms training, data loading parallelized)
- **Speedup**: 1.56× throughput improvement by hiding data loading latency
- **Buffer size**: Typically 1-4 batches (balance latency hiding vs memory usage)

**Distributed Data Loading**: For 8-GPU ImageNet training:
- Each GPU gets 1/8 of dataset via `DistributedSampler` (no data duplication)
- Worker processes: 8 GPUs × 4 workers = 32 parallel data loaders
- Data loading scales linearly with GPUs (8× training throughput, 8× data pipeline throughput)
- Critical: Ensure each GPU sees different data (avoid redundant computation)

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
