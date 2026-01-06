# Module 05: DataLoader

:::{admonition} Module Info
:class: note

**FOUNDATION TIER** | Difficulty: ●●○○ | Time: 3-5 hours | Prerequisites: 01-04

**Prerequisites:** You should be comfortable with tensors, activations, layers, and losses from Modules 01-04. This module introduces data loading infrastructure that will be used by autograd, optimizers, and training loops in the following modules.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F05_dataloader%2F05_dataloader.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/05_dataloader/05_dataloader.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/05_dataloader.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Training a neural network on 50,000 images presents an immediate systems challenge: you cannot load all data into memory simultaneously, and even if you could, processing one sample at a time wastes GPU parallelism. The DataLoader solves this by transforming raw datasets into batches that feed efficiently into training loops.

In this module, you'll build the data pipeline infrastructure that sits between storage and computation. Your implementation will provide a clean abstraction that handles batching, shuffling, and memory-efficient iteration, working identically whether processing 1,000 samples or 1 million. By the end, you'll understand why data loading is often the hidden bottleneck in training pipelines.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** the Dataset abstraction and TensorDataset for in-memory data storage
- **Build** a DataLoader with intelligent batching, shuffling, and memory-efficient iteration
- **Master** the Python iterator protocol for streaming data without loading entire datasets
- **Analyze** throughput bottlenecks and memory scaling characteristics with different batch sizes
- **Connect** your implementation to PyTorch data loading patterns used in production ML systems
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Data Pipeline
flowchart LR
    subgraph "Your Data Pipeline"
        A["Dataset<br/>__len__, __getitem__"]
        B["TensorDataset<br/>In-memory storage"]
        C["DataLoader<br/>Batching + Shuffling"]
        D["Iterator<br/>Yields batches"]
    end

    A --> B --> C --> D
    D --> E["Training Loop<br/>for batch in loader"]

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
    style E fill:#e2d5f1
```

**Implementation roadmap:**

| Step | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `Dataset` abstract base class | Universal data access interface |
| 2 | `TensorDataset(Dataset)` | Tensor-based in-memory storage |
| 3 | `DataLoader.__init__()` | Store dataset, batch size, shuffle flag |
| 4 | `DataLoader.__iter__()` | Index shuffling and batch grouping |
| 5 | `DataLoader._collate_batch()` | Stack samples into batch tensors |

**The pattern you'll enable:**
```python
# Transform individual samples into training-ready batches
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_features, batch_labels in loader:
    # batch_features: (32, feature_dim) - ready for model.forward()
    predictions = model(batch_features)
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Multi-process data loading (PyTorch uses `num_workers` for parallel loading)
- Automatic dataset downloads (you'll use pre-downloaded data or write custom loaders)
- Prefetching mechanisms (loading next batch while GPU processes current batch)
- Custom collation functions for variable-length sequences (that's for NLP modules)

**You are building the batching foundation.** Parallel loading optimizations come later.

## API Reference

This section provides a quick reference for the data loading classes you'll build. Use it while implementing to verify signatures and expected behavior.

### Dataset (Abstract Base Class)

```python
class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int

    @abstractmethod
    def __getitem__(self, idx: int)
```

The Dataset interface enforces two requirements on all subclasses:

| Method | Returns | Description |
|--------|---------|-------------|
| `__len__()` | `int` | Total number of samples in dataset |
| `__getitem__(idx)` | Sample | Retrieve sample at index `idx` (0-indexed) |

### TensorDataset

```python
TensorDataset(*tensors)
```

Wraps one or more tensors into a dataset where samples are tuples of aligned tensor slices.

**Constructor Arguments:**
- `*tensors`: Variable number of Tensor objects, all with same first dimension

**Behavior:**
- All tensors must have identical length in dimension 0 (sample dimension)
- Returns tuple `(tensor1[idx], tensor2[idx], ...)` for each sample

### DataLoader

```python
DataLoader(dataset, batch_size, shuffle=False)
```

Wraps a dataset to provide batched iteration with optional shuffling.

**Constructor Arguments:**
- `dataset`: Dataset instance to load from
- `batch_size`: Number of samples per batch
- `shuffle`: Whether to randomize sample order each iteration

**Core Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `__len__()` | `int` | Number of batches (ceiling of samples divided by batch_size) |
| `__iter__()` | `Iterator` | Returns generator yielding batched tensors |
| `_collate_batch(batch)` | `Tuple[Tensor, ...]` | Stacks list of samples into batch tensors |

### Data Augmentation Transforms

```python
RandomHorizontalFlip(p=0.5)
RandomCrop(size, padding=4)
Compose(transforms)
```

Transform classes for data augmentation during training. Applied to individual samples before batching.

**RandomHorizontalFlip:**
- `p`: Probability of flipping (0.0 to 1.0)
- Flips images horizontally along width axis with given probability

**RandomCrop:**
- `size`: Target crop size (int for square, tuple for (H, W))
- `padding`: Pixels to pad on each side before cropping
- Standard augmentation for CIFAR-10: pads to 40×40, crops back to 32×32

**Compose:**
- `transforms`: List of transform callables to apply sequentially
- Chains multiple transforms into a pipeline

## Core Concepts

This section explains the fundamental ideas behind efficient data loading. Understanding these concepts is essential for building and debugging ML training pipelines.

### Dataset Abstraction

The Dataset abstraction separates how data is stored from how it's accessed. This separation enables the same DataLoader code to work with data stored in files, databases, memory, or even generated on-demand.

The interface is deliberately minimal: `__len__()` returns the count and `__getitem__(idx)` retrieves a specific sample. A dataset backed by 50,000 JPEG files implements the same interface as a dataset with 50,000 tensors in RAM. The DataLoader doesn't care about implementation details.

Here's the complete abstract base class from your implementation:

```python
class Dataset(ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """Return the sample at the given index."""
        pass
```

The `@abstractmethod` decorator forces any subclass to implement these methods. Attempting `Dataset()` raises `TypeError` because the abstract methods haven't been implemented. This pattern ensures every dataset provides the minimum interface that DataLoader requires.

The systems insight: by defining a minimal interface, you enable composition. A caching layer can wrap any Dataset, a subset can slice any Dataset, and a concatenation can merge multiple Datasets, all without knowing the underlying storage mechanism.

### Batching Mechanics

Batching transforms individual samples into the stacked tensors that GPUs process efficiently. When you call `dataset[0]`, you might get `(features: (784,), label: scalar)` for an MNIST digit. When you call `next(iter(dataloader))`, you get `(features: (32, 784), labels: (32,))`. The DataLoader collected 32 individual samples and stacked them along a new batch dimension.

Here's how collation happens in your implementation:

```python
def _collate_batch(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
    """Collate individual samples into batch tensors."""
    if len(batch) == 0:
        return ()

    # Determine number of tensors per sample
    num_tensors = len(batch[0])

    # Group tensors by position
    batched_tensors = []
    for tensor_idx in range(num_tensors):
        # Extract all tensors at this position
        tensor_list = [sample[tensor_idx].data for sample in batch]

        # Stack into batch tensor
        batched_data = np.stack(tensor_list, axis=0)
        batched_tensors.append(Tensor(batched_data))

    return tuple(batched_tensors)
```

The algorithm: for each position in the sample tuple (features, labels, etc.), collect all samples' values at that position, then stack them using `np.stack()` along axis 0. The result is a batch tensor where the first dimension is batch size.

Consider the memory transformation. Five individual samples might each be a `(784,)` tensor consuming 3 KB. After collation, you have a single `(5, 784)` tensor consuming 15 KB. The data is identical, but the layout is now batch-friendly: all 5 samples are contiguous in memory, enabling efficient vectorized operations.

### Shuffling and Randomization

Shuffling prevents the model from learning the order of training data rather than actual patterns. Without shuffling, a model sees identical batch combinations every epoch, creating correlations between gradient updates.

The naive implementation would load all samples, shuffle the data array, then iterate. But this requires memory proportional to dataset size. Your implementation is smarter: it shuffles indices, not data.

Here's the shuffling logic from your `__iter__` method:

```python
def __iter__(self) -> Iterator:
    """Return iterator over batches."""
    # Create list of indices
    indices = list(range(len(self.dataset)))

    # Shuffle if requested
    if self.shuffle:
        random.shuffle(indices)

    # Yield batches
    for i in range(0, len(indices), self.batch_size):
        batch_indices = indices[i:i + self.batch_size]
        batch = [self.dataset[idx] for idx in batch_indices]

        # Collate batch
        yield self._collate_batch(batch)
```

The key insight: `random.shuffle(indices)` randomizes a list of integers, not actual data. For 50,000 samples, this shuffles 50,000 integers (400 KB) instead of 50,000 images (potentially gigabytes). The actual data stays in place; only the access order changes.

Each epoch generates a fresh shuffle, so the same samples appear in different batches. If sample 42 and sample 1337 were in the same batch in epoch 1, they're likely in different batches in epoch 2. This decorrelation is essential for generalization.

The memory cost of shuffling is `8 bytes × dataset_size`. For 1 million samples, that's 8 MB, negligible compared to the actual data. The time cost is O(n) for generating and shuffling indices, which happens once per epoch, not per batch.

### Iterator Protocol and Generator Pattern

Python's iterator protocol enables `for batch in dataloader` syntax. When Python encounters this loop, it first calls `dataloader.__iter__()` to get an iterator object. Your `__iter__` method is a generator function (contains `yield`), so Python automatically creates a generator that produces values lazily.

Here's the complete implementation showing the generator pattern:

```python
def __iter__(self) -> Iterator:
    """Return iterator over batches."""
    # Create list of indices
    indices = list(range(len(self.dataset)))

    # Shuffle if requested
    if self.shuffle:
        random.shuffle(indices)

    # Yield batches - this is a generator function
    for i in range(0, len(indices), self.batch_size):
        batch_indices = indices[i:i + self.batch_size]
        batch = [self.dataset[idx] for idx in batch_indices]

        # Collate batch
        yield self._collate_batch(batch)
```

Each time the loop needs the next batch, Python calls `next()` on the generator, which executes `__iter__` until the next `yield` statement. The generator pauses at yield, returns the batch, then resumes when next() is called again. This is a generator function, not a regular function that returns an iterator object.

This lazy evaluation is crucial for memory efficiency. At any moment, only the current batch exists in memory. The previous batch has been freed (assuming the training code doesn't hold references), and future batches haven't been created yet.

Consider iterating through 1,000 batches of 32 images each. If you pre-generated all batches, you'd need memory for 32,000 images simultaneously. With the generator protocol, you only need memory for 32 images at a time, a 1,000× reduction.

The generator also enables infinite datasets. If your dataset generates samples on-demand (synthetic data), the generator can yield batches forever without running out. The training loop controls when to stop, not the dataset.

### Memory-Efficient Loading

The combination of Dataset abstraction and DataLoader iteration creates a memory-efficient pipeline regardless of dataset size.

For in-memory datasets like TensorDataset, all data is pre-loaded, but DataLoader still provides memory benefits by controlling how much data is active at once. Your training loop processes one batch, computes gradients, updates weights, then discards that batch before loading the next. Peak memory is `batch_size × sample_size`, not `dataset_size × sample_size`.

For disk-backed datasets, the benefits are dramatic. Consider an ImageDataset that loads JPEGs on-demand:

```python
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths  # Just file paths (tiny memory)
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image only when requested
        image = load_jpeg(self.image_paths[idx])
        return Tensor(image), Tensor(self.labels[idx])
```

When DataLoader calls `dataset[idx]`, the image is loaded from disk at that moment, not at dataset creation time. After the batch is processed, the image memory is freed. A 100 GB dataset can be trained on a machine with 8 GB RAM because only one batch worth of images exists in memory at a time.

This is why Dataset separates length from access. The dataset knows it has 50,000 images without loading them. Only when `__getitem__` is called does actual loading happen. DataLoader orchestrates these calls to load exactly the data needed for the current batch.

## Common Errors

These are the most frequent mistakes encountered when implementing and using data loaders.

### Mismatched Tensor Dimensions

**Error**: `ValueError: All tensors must have same size in first dimension`

This happens when you try to create a TensorDataset with tensors that have different numbers of samples:

```python
features = Tensor(np.random.randn(100, 10))  # 100 samples
labels = Tensor(np.random.randn(90))         # 90 labels - MISMATCH!
dataset = TensorDataset(features, labels)    # Raises ValueError
```

The first dimension is the sample dimension. If features has 100 samples but labels has 90, TensorDataset cannot pair them correctly.

**Fix**: Ensure all tensors have identical first dimension before constructing TensorDataset.

### Forgetting to Shuffle Training Data

**Symptom**: Model converges slowly or gets stuck at suboptimal accuracy

Without shuffling, the model sees identical batch combinations every epoch. If your dataset is sorted by class (all cats, then all dogs), early batches are all cats and later batches are all dogs. The model oscillates between cat features and dog features rather than learning a unified representation.

```python
# Wrong - no shuffling means same batches every epoch
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Correct - shuffle for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# But don't shuffle validation - you want consistent evaluation
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

**Fix**: Always shuffle training data, never shuffle validation or test data.

### Assuming Fixed Batch Size

**Symptom**: Index errors or shape mismatches on last batch

If your dataset has 100 samples and batch_size=32, you get batches of size [32, 32, 32, 4]. The last batch is smaller because 100 is not divisible by 32. Code that assumes every batch has exactly 32 samples will fail on the last batch.

```python
def train_step(batch):
    features, labels = batch
    # Wrong - assumes batch_size=32
    assert features.shape[0] == 32  # Fails on last batch!

    # Correct - get actual batch size
    batch_size = features.shape[0]
```

**Fix**: Always derive batch size from tensor shape, never hardcode it.

### Index Out of Bounds

**Error**: `IndexError: Index 100 out of range for dataset of size 100`

This happens when trying to access an index that doesn't exist. Remember that Python uses 0-indexing: valid indices for a dataset of size 100 are 0 through 99, not 1 through 100.

**Fix**: Ensure index range is `0 <= idx < len(dataset)`.

## Production Context

### Your Implementation vs. PyTorch

Your DataLoader and PyTorch's `torch.utils.data.DataLoader` share the same conceptual design and interface. The differences are in advanced features and performance optimizations.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Interface** | Dataset + DataLoader | Identical pattern |
| **Batching** | Sequential in main process | Parallel with `num_workers` |
| **Shuffling** | Index-based, O(n) | Same algorithm |
| **Collation** | `np.stack()` in Python | Custom collate functions supported |
| **Prefetching** | None | Loads next batch during compute |
| **Memory** | One batch at a time | Configurable buffer with workers |

### Code Comparison

The following comparison shows identical usage patterns between TinyTorch and PyTorch. Notice how the APIs mirror each other exactly.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.core.dataloader import TensorDataset, DataLoader

# Create dataset
features = Tensor(X_train)
labels = Tensor(y_train)
dataset = TensorDataset(features, labels)

# Create loader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Training loop
for epoch in range(num_epochs):
    for batch_features, batch_labels in train_loader:
        predictions = model(batch_features)
        loss = loss_fn(predictions, batch_labels)
        loss.backward()
        optimizer.step()
```
````

````{tab-item} ⚡ PyTorch
```python
from torch.utils.data import TensorDataset, DataLoader

# Create dataset
features = torch.tensor(X_train)
labels = torch.tensor(y_train)
dataset = TensorDataset(features, labels)

# Create loader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4  # Parallel loading
)

# Training loop
for epoch in range(num_epochs):
    for batch_features, batch_labels in train_loader:
        predictions = model(batch_features)
        loss = loss_fn(predictions, batch_labels)
        loss.backward()
        optimizer.step()
```
````
`````

Walking through the differences:

- **Lines 1-6 (Dataset Creation)**: Identical. Both frameworks use TensorDataset to wrap tensors with the same interface.
- **Lines 8-12 (DataLoader Creation)**: PyTorch adds `num_workers` for parallel data loading. With `num_workers=4`, four processes load batches in parallel, overlapping data loading with GPU computation. Your implementation is single-process.
- **Lines 14-20 (Training Loop)**: Completely identical. The iterator protocol means both frameworks use the same `for batch in loader` syntax.

```{tip} What's Identical

The Dataset abstraction, DataLoader interface, and batching semantics are identical. When you understand TinyTorch's data pipeline, you understand PyTorch's data pipeline. The only difference is PyTorch adds parallel loading to hide I/O latency.
```

### Why DataLoaders Matter at Scale

To appreciate why data loading infrastructure matters, consider the scale of production training:

- **ImageNet training**: 1.2 million images at 224×224×3 pixels = **600 GB** of uncompressed data
- **Batch memory**: batch_size=256 with 150 KB per image = **38 MB** per batch
- **I/O throughput**: Loading from SSD at 500 MB/s = **76 ms per batch** just for disk reads

Without proper batching and prefetching, data loading would dominate training time. A forward and backward pass might take 50 ms, but loading the data takes 76 ms. The GPU sits idle 60% of the time waiting for data.

Production solutions:

- **Prefetching**: Load batch N+1 while GPU processes batch N (PyTorch's `num_workers`)
- **Data caching**: Keep decoded images in RAM across epochs (eliminates JPEG decode overhead)
- **Faster formats**: Use LMDB or TFRecords instead of individual files (reduces filesystem overhead)

Your DataLoader provides the interface that enables these optimizations. Add `num_workers`, swap TensorDataset for a disk-backed dataset, and the training loop code stays identical.

## Check Your Understanding

Test your understanding with these systems thinking questions. Focus on quantitative analysis and performance trade-offs.

**Q1: Memory Calculation**

You're training on CIFAR-10 with 50,000 RGB images (32×32×3 pixels, float32). What's the memory usage for batch_size=128?

```{admonition} Answer
:class: dropdown

Each image: 32 × 32 × 3 × 4 bytes = 12,288 bytes ≈ 12 KB

Batch of 128 images: 128 × 12 KB = **1,536 KB ≈ 1.5 MB**

This is the minimum memory just for the input batch. Add activations, gradients, and model parameters, and peak memory might be 50-100× higher. But the **batch size directly controls the baseline memory consumption**.
```

**Q2: Throughput Analysis**

Your training reports these timings per batch:
- Data loading: 45ms
- Forward pass: 30ms
- Backward pass: 35ms
- Optimizer step: 10ms

Total: 120ms per batch. Where's the bottleneck? How much faster could training be if you eliminated data loading overhead?

```{admonition} Answer
:class: dropdown

Data loading takes 45ms out of 120ms = **37.5% of total time**.

If data loading were instant (via prefetching or caching), total time would be 30+35+10 = **75ms per batch**.

Speedup: 120ms → 75ms = **1.6× faster training** just by fixing data loading!

This shows why production systems use prefetching with `num_workers`: while the GPU computes batch N, the CPU loads batch N+1. Data loading and computation overlap, hiding the I/O latency.
```

**Q3: Shuffle Memory Overhead**

You're training on a dataset with 10 million samples. How much extra memory does `shuffle=True` require compared to `shuffle=False`?

```{admonition} Answer
:class: dropdown

Shuffling requires storing the index array: 10,000,000 indices × 8 bytes = **80 MB**

This is the complete overhead. The actual data isn't copied or moved, only the index array is shuffled.

For comparison, if each sample is 10 KB, the full dataset is 100 GB. Shuffling adds 80 MB to randomize access to 100 GB of data, **0.08% overhead**. This is why index-based shuffling scales to massive datasets.
```

**Q4: Batch Size Trade-offs**

You're deciding between batch_size=32 and batch_size=256 for ImageNet training:

- batch_size=32: 14 hours training, 76.1% accuracy
- batch_size=256: 6 hours training, 75.8% accuracy

Which would you choose for a research experiment where accuracy is critical? Which for a production job where you train 100 models per day?

```{admonition} Answer
:class: dropdown

**Research (accuracy critical):** batch_size=32

- 14 hours is acceptable for research (run overnight)
- 76.1% vs 75.8% = 0.3% accuracy gain might be significant for publication
- Smaller batches often generalize better (noisier gradients act as regularization)

**Production (throughput critical):** batch_size=256

- 6 hours vs 14 hours = **2.3× faster**, enabling 100 models to train in reasonable time
- 0.3% accuracy difference is negligible for many production applications
- Can try learning rate adjustments to recover accuracy while keeping speed

**Systems insight**: Batch size creates a three-way trade-off between training speed, memory usage, and model quality. The "right" answer depends on your bottleneck: time, memory, or accuracy.
```

**Q5: Collation Cost**

Your DataLoader collates batches using `np.stack()`. For batch_size=128 with samples of shape (3, 224, 224), how much data is copied during collation?

```{admonition} Answer
:class: dropdown

Each sample: 3 × 224 × 224 × 4 bytes = 602,112 bytes ≈ 588 KB

Batch of 128 samples: 128 × 588 KB = **75,264 KB ≈ 73.5 MB**

`np.stack()` allocates a new array of this size and copies all 128 samples into contiguous memory. On a modern CPU with 20 GB/s memory bandwidth, this copy takes approximately **3.7 milliseconds**.

This is why larger batch sizes can have higher absolute collation costs (more data to copy), but the per-sample overhead decreases because you're copying 128 samples in one operation instead of processing 128 tiny batches separately.
```

## Further Reading

For students who want to understand the academic foundations and engineering decisions behind data loading systems:

### Seminal Papers

- **ImageNet Classification with Deep Convolutional Neural Networks** - Krizhevsky et al. (2012). The AlexNet paper that popularized large-scale image training and highlighted data augmentation as essential for generalization. [NeurIPS](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

- **Accurate, Large Minibatch SGD** - Goyal et al. (2017). Facebook AI Research paper exploring how to scale batch size to 8192 while maintaining accuracy, revealing the relationship between batch size, learning rate, and convergence. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)

- **Mixed Precision Training** - Micikevicius et al. (2018). NVIDIA paper showing how batch size interacts with numerical precision for memory and speed trade-offs. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

### Additional Resources

- **Engineering Blog**: "PyTorch DataLoader Internals" - Detailed explanation of multi-process loading and prefetching strategies
- **Documentation**: [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) - See how production frameworks extend the patterns you've built

## What's Next

```{seealso} Coming Up: Module 06 - Autograd

Implement automatic differentiation that computes gradients through computation graphs. Your DataLoader will feed batches to models, and autograd will enable learning from those batches.
```

**Preview - How Your DataLoader Gets Used in Future Modules:**

| Module | What It Does | Your DataLoader In Action |
|--------|--------------|--------------------------|
| **06: Autograd** | Automatic differentiation | Tensors from DataLoader flow through computation graphs |
| **08: Training** | Complete training loops | `for batch in loader:` orchestrates the full training process |
| **09: Convolutions** | Convolutional layers for images | `for images, labels in loader:` feed batches to CNNs |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/05_dataloader/05_dataloader.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/05_dataloader/05_dataloader.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
