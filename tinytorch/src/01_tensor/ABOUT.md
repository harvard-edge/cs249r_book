---
title: "Tensor"
description: "Build the fundamental N-dimensional array data structure that powers all machine learning"
difficulty: "⭐"
time_estimate: "4-6 hours"
prerequisites: []
next_steps: ["02_activations"]
learning_objectives:
  - "Understand tensors as N-dimensional arrays and their memory/performance implications in ML systems"
  - "Implement a complete Tensor class with arithmetic, shape operations, and efficient data handling"
  - "Master broadcasting rules and understand how they enable efficient computations without data copying"
  - "Recognize how tensor operations form the foundation of PyTorch/TensorFlow architecture"
  - "Analyze computational complexity, memory usage, and view-vs-copy trade-offs in tensor operations"
---

# 01. Tensor

**FOUNDATION TIER** | Difficulty: ⭐ (1/4) | Time: 4-6 hours

## Overview

The Tensor class is the foundational data structure of machine learning - every neural network, from simple linear models to GPT and Stable Diffusion, operates on tensors. You'll build N-dimensional arrays from scratch with arithmetic operations, broadcasting, and shape manipulation. This module gives you deep insight into how PyTorch and TensorFlow work under the hood, understanding the memory and performance implications that matter in production ML systems.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand memory and performance implications**: Recognize how tensor operations dominate compute time and memory usage in ML systems - a single matrix multiplication can consume 90% of forward pass time in production frameworks like PyTorch
- **Implement core tensor functionality**: Build a complete Tensor class with arithmetic (`+`, `-`, `*`, `/`), matrix multiplication, shape manipulation (`reshape`, `transpose`), and reductions (`sum`, `mean`, `max`) with proper error handling and validation
- **Master broadcasting semantics**: Understand NumPy broadcasting rules that enable efficient computations across different tensor shapes without data copying - critical for batch processing and efficient neural network operations
- **Connect to production frameworks**: See how your implementation mirrors PyTorch's `torch.Tensor` and TensorFlow's `tf.Tensor` design patterns, understanding the architectural decisions that power real ML systems
- **Analyze performance trade-offs**: Understand computational complexity (O(n³) for matrix multiplication), memory usage patterns (contiguous vs. strided), and when to copy data vs. create views for optimization

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement the Tensor class from scratch using NumPy as the underlying array library - creating `__init__`, operator overloading (`__add__`, `__mul__`, etc.), shape manipulation methods, and reduction operations
2. **Use**: Apply your Tensor to real problems like matrix multiplication for neural network layers, data normalization with broadcasting, and statistical computations across various shapes and dimensions
3. **Reflect**: Understand systems-level implications - why tensor operations dominate training time, how memory layout (row-major vs. column-major) affects cache performance, and how broadcasting eliminates redundant data copying

## What You'll Build

By completing this module, you'll create a production-ready Tensor class with:

**Core Data Structure:**
- N-dimensional array wrapper around NumPy with clean API
- Properties for shape, size, dtype, and data access
- Dormant gradient tracking attributes (activated in Module 05)

**Arithmetic Operations:**
- Element-wise operations: `+`, `-`, `*`, `/`, `**`
- Full broadcasting support for Tensor-Tensor and Tensor-scalar operations
- Automatic shape alignment following NumPy broadcasting rules

**Matrix Operations:**
- `matmul()` for matrix multiplication with shape validation
- Support for matrix-matrix, matrix-vector multiplication
- Clear error messages for dimension mismatches

**Shape Manipulation:**
- `reshape()` with -1 inference for automatic dimension calculation
- `transpose()` for dimension swapping
- View vs. copy semantics understanding

**Reduction Operations:**
- `sum()`, `mean()`, `max()`, `min()` with axis parameter
- Global reductions (entire tensor) and axis-specific reductions
- `keepdims` support for maintaining dimensionality

**Real-World Usage Pattern:**
Your Tensor enables the fundamental neural network forward pass: `output = x.matmul(W) + b` - exactly how PyTorch and TensorFlow work internally.

## Core Concepts

### Tensors as Multidimensional Arrays

A tensor is a generalization of scalars (0D), vectors (1D), and matrices (2D) to N dimensions:

- **Scalar**: `Tensor(5.0)` - shape `()`
- **Vector**: `Tensor([1, 2, 3])` - shape `(3,)`
- **Matrix**: `Tensor([[1, 2], [3, 4]])` - shape `(2, 2)`
- **3D Tensor**: Image batch `(batch, height, width)` - shape `(32, 224, 224)`
- **4D Tensor**: CNN features `(batch, channels, height, width)` - shape `(32, 3, 224, 224)`

**Why tensors matter**: They provide a unified interface for all ML data - images, text embeddings, audio spectrograms, and model parameters are all tensors with different shapes.

### Broadcasting: Efficient Shape Alignment

Broadcasting automatically expands smaller tensors to match larger ones without copying data:

```python
# Matrix (2,2) + Vector (2,) → broadcasts to (2,2)
matrix = Tensor([[1, 2], [3, 4]])
vector = Tensor([10, 20])
result = matrix + vector  # [[11, 22], [13, 24]]
```

**Broadcasting rules** (NumPy-compatible):
1. Align shapes from right to left
2. Dimensions are compatible if they're equal or one is 1
3. Missing dimensions are treated as size 1

**Why broadcasting matters**: Eliminates redundant data copying. Adding a bias vector to 1000 feature maps broadcasts once instead of copying the vector 1000 times - saving memory and enabling vectorization.

### Views vs. Copies: Memory Efficiency

Some operations return **views** (sharing memory) vs. **copies** (duplicating data):

- **Views** (O(1)): `reshape()`, `transpose()` when possible - no data movement
- **Copies** (O(n)): Arithmetic operations, explicit `.copy()` - duplicate storage

**Why this matters**: A view of a 1GB tensor is free (just metadata). A copy allocates another 1GB. Understanding view semantics prevents memory blowup in production systems.

### Computational Complexity

Different operations have vastly different costs:

- **Element-wise** (`+`, `-`, `*`): O(n) - linear in tensor size
- **Reductions** (`sum`, `mean`): O(n) - must visit every element
- **Matrix multiply** (`matmul`): O(n³) for square matrices - dominates training time

**Why this matters**: In a neural network forward pass, matrix multiplications consume 90%+ of compute time. Optimizing matmul is critical - hence specialized hardware (GPUs, TPUs) and libraries (cuBLAS, MKL).

## Architecture Overview

### Tensor Class Design

```
┌─────────────────────────────────────────┐
│         Tensor Class                    │
├─────────────────────────────────────────┤
│  Properties:                            │
│  - data: np.ndarray (underlying storage)│
│  - shape: tuple (dimensions)            │
│  - size: int (total elements)           │
│  - dtype: np.dtype (data type)          │
│  - requires_grad: bool (autograd flag)  │
│  - grad: Tensor (gradient - Module 05)  │
├─────────────────────────────────────────┤
│  Operator Overloading:                  │
│  - __add__, __sub__, __mul__, __truediv__│
│  - __pow__ (exponentiation)             │
│  - Returns new Tensor instances         │
├─────────────────────────────────────────┤
│  Methods:                               │
│  - matmul(other): Matrix multiplication │
│  - reshape(*shape): Shape manipulation  │
│  - transpose(): Dimension swap          │
│  - sum/mean/max/min(axis): Reductions   │
└─────────────────────────────────────────┘
```

### Data Flow Architecture

```
Python Interface (your code)
         ↓
    Tensor Class
         ↓
   NumPy Backend (vectorized operations)
         ↓
  C/Fortran Libraries (BLAS, LAPACK)
         ↓
    Hardware (CPU SIMD, cache)
```

**Your implementation**: Python wrapper → NumPy
**PyTorch/TensorFlow**: Python wrapper → C++ engine → GPU kernels

The architecture is identical in concept - you're learning the same design patterns used in production, just with NumPy instead of custom CUDA kernels.

### Module Integration

```
Module 01: Tensor (THIS MODULE)
    ↓ provides foundation
Module 02: Activations (ReLU, Sigmoid operate on Tensors)
    ↓ uses tensors
Module 03: Layers (Linear, Conv2d store weights as Tensors)
    ↓ uses tensors
Module 05: Autograd (adds .grad attribute to Tensors)
    ↓ enhances tensors
Module 06: Optimizers (updates Tensor parameters)
```

Your Tensor is the universal foundation - every subsequent module builds on what you create here.

## Prerequisites

This is the first module - no prerequisites! Verify your environment is ready:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Check system health
tito system health
```

All checks should pass (Python 3.8+, NumPy, pytest installed) before starting.

## Getting Started

### Development Workflow

1. **Open the development notebook**: `modules/01_tensor/tensor_dev.ipynb` in Jupyter or your preferred editor
2. **Implement Tensor.__init__**: Create constructor that converts data to NumPy array, stores shape/size/dtype, initializes gradient attributes
3. **Build arithmetic operations**: Implement `__add__`, `__sub__`, `__mul__`, `__truediv__` with broadcasting support for both Tensor-Tensor and Tensor-scalar operations
4. **Add matrix multiplication**: Implement `matmul()` with shape validation and clear error messages for dimension mismatches
5. **Create shape manipulation**: Implement `reshape()` (with -1 support) and `transpose()` for dimension swapping
6. **Implement reductions**: Build `sum()`, `mean()`, `max()` with axis parameter and keepdims support
7. **Export and verify**: Run `tito export 01` to export to package, then `tito test 01` to validate all tests pass

## Implementation Guide

### Tensor Class Foundation

Your Tensor class wraps NumPy arrays and provides ML-specific functionality:

```python
from tinytorch.core.tensor import Tensor

# Create tensors from Python lists or NumPy arrays
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
y = Tensor([[0.5, 1.5], [2.5, 3.5]])

# Properties provide clean API access
print(x.shape)    # (2, 2)
print(x.size)     # 4
print(x.dtype)    # float32
```

**Implementation details**: You'll implement `__init__` to convert input data to NumPy arrays, store shape/size/dtype as properties, and initialize dormant gradient attributes (`requires_grad`, `grad`) that activate in Module 05.

### Arithmetic Operations

Implement operator overloading for element-wise operations with broadcasting:

```python
# Element-wise operations via operator overloading
z = x + y         # Addition: [[1.5, 3.5], [5.5, 7.5]]
w = x * y         # Element-wise multiplication
p = x ** 2        # Exponentiation
s = x - y         # Subtraction
d = x / y         # Division

# Broadcasting: scalar operations automatically expand
scaled = x * 2    # [[2.0, 4.0], [6.0, 8.0]]
shifted = x + 10  # [[11.0, 12.0], [13.0, 14.0]]

# Broadcasting: vector + matrix
matrix = Tensor([[1, 2], [3, 4]])
vector = Tensor([10, 20])
result = matrix + vector  # [[11, 22], [13, 24]]
```

**Systems insight**: These operations vectorize automatically via NumPy, achieving ~100x speedup over Python loops. This is why all ML frameworks use tensors - the performance difference between `for i in range(n): result[i] = a[i] + b[i]` and `result = a + b` is dramatic at scale.

### Matrix Multiplication

Matrix multiplication is the heart of neural networks - every layer performs it:

```python
# Matrix multiplication (the @ operator)
a = Tensor([[1, 2], [3, 4]])  # 2×2
b = Tensor([[5, 6], [7, 8]])  # 2×2
c = a.matmul(b)               # 2×2 result: [[19, 22], [43, 50]]

# Neural network forward pass pattern: y = xW + b
x = Tensor([[1, 2, 3], [4, 5, 6]])     # Input: (batch=2, features=3)
W = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Weights: (3, 2)
b = Tensor([0.1, 0.2])                 # Bias: (2,)
output = x.matmul(W) + b               # (2, 2)
```

**Computational complexity**: For matrices `(M,K) @ (K,N)`, the cost is `O(M×K×N)` floating-point operations. A 1000×1000 matrix multiplication requires 2 billion FLOPs - this dominates training time in production systems.

### Shape Manipulation

Neural networks constantly reshape tensors to match layer requirements:

```python
# Reshape: change interpretation of same data (O(1) operation)
tensor = Tensor([1, 2, 3, 4, 5, 6])
reshaped = tensor.reshape(2, 3)  # [[1, 2, 3], [4, 5, 6]]
flat = reshaped.reshape(-1)      # [1, 2, 3, 4, 5, 6]

# Transpose: swap dimensions (data rearrangement)
matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
transposed = matrix.transpose()          # (3, 2): [[1, 4], [2, 5], [3, 6]]

# CNN data flow example
images = Tensor(np.random.rand(32, 3, 224, 224))  # (batch, channels, H, W)
features = images.reshape(32, -1)                 # (batch, 3*224*224) - flatten for MLP
```

**Memory consideration**: `reshape` often returns *views* (no data copying) when possible - an O(1) operation. `transpose` may require data rearrangement depending on memory layout. Understanding views vs. copies is crucial: views share memory (efficient), copies duplicate data (expensive for large tensors).

### Reduction Operations

Aggregation operations collapse dimensions for statistics and loss computation:

```python
# Reduce along different axes
total = x.sum()             # Scalar: sum all elements
col_sums = x.sum(axis=0)    # Sum columns: [4, 6]
row_sums = x.sum(axis=1)    # Sum rows: [3, 7]

# Statistical reductions
means = x.mean(axis=0)      # Column-wise mean
minimums = x.min(axis=1)    # Row-wise minimum
maximums = x.max()          # Global maximum

# Batch loss averaging (common pattern)
losses = Tensor([0.5, 0.3, 0.8, 0.2])  # Per-sample losses
avg_loss = losses.mean()                # 0.45 - batch average
```

**Production pattern**: Every loss function uses reductions. Cross-entropy loss computes per-sample losses then averages: `loss = -log(predictions[correct_class]).mean()`. Understanding axis semantics prevents bugs in multi-dimensional operations.

## Testing

### Comprehensive Test Suite

Run the full test suite to verify tensor functionality:

```bash
# TinyTorch CLI (recommended - runs all 01_tensor tests)
tito test 01

# Direct pytest execution (more verbose output)
python -m pytest tests/01_tensor/ -v

# Run specific test class
python -m pytest tests/01_tensor/test_tensor_core.py::TestTensorCreation -v
```

Expected output: All tests pass with green checkmarks showing your Tensor implementation works correctly.

### Test Coverage Areas

Your implementation is validated across these dimensions:

- **Initialization** (`test_tensor_from_list`, `test_tensor_from_numpy`, `test_tensor_shapes`): Creating tensors from Python lists, NumPy arrays, and nested structures with correct shape/dtype handling
- **Arithmetic Operations** (`test_tensor_addition`, `test_tensor_multiplication`): Element-wise addition, subtraction, multiplication, division with both Tensor-Tensor and Tensor-scalar combinations
- **Broadcasting** (`test_scalar_broadcasting`, `test_vector_broadcasting`): Automatic shape alignment for different tensor shapes, scalar expansion, matrix-vector broadcasting
- **Matrix Multiplication** (`test_matrix_multiplication`): Matrix-matrix, matrix-vector multiplication with shape validation and error handling for incompatible dimensions
- **Shape Manipulation** (`test_tensor_reshape`, `test_tensor_transpose`, `test_tensor_flatten`): Reshape with -1 inference, transpose with dimension swapping, validation for incompatible sizes
- **Reductions** (`test_sum`, `test_mean`, `test_max`): Aggregation along various axes (None, 0, 1, multiple), keepdims behavior, global vs. axis-specific reduction
- **Memory Management** (`test_tensor_data_access`, `test_tensor_copy_semantics`, `test_tensor_memory_efficiency`): Data access patterns, copy vs. view semantics, memory usage validation

### Inline Testing & Validation

The development notebook includes comprehensive inline tests with immediate feedback:

```python
# Example inline test output
🧪 Unit Test: Tensor Creation...
✅ Tensor created from list
✅ Shape property correct: (2, 2)
✅ Size property correct: 4
✅ dtype is float32
📈 Progress: Tensor initialization ✓

🧪 Unit Test: Arithmetic Operations...
✅ Addition: [[6, 8], [10, 12]]
✅ Multiplication works element-wise
✅ Broadcasting: scalar + tensor
✅ Broadcasting: matrix + vector
📈 Progress: Arithmetic operations ✓

🧪 Unit Test: Matrix Multiplication...
✅ 2×2 @ 2×2 = [[19, 22], [43, 50]]
✅ Shape validation catches 2×2 @ 3×1 error
✅ Error message shows: "2 ≠ 3"
📈 Progress: Matrix operations ✓
```

### Manual Testing Examples

Validate your implementation interactively:

```python
from tinytorch.core.tensor import Tensor
import numpy as np

# Test basic operations
x = Tensor([[1, 2], [3, 4]])
y = Tensor([[5, 6], [7, 8]])

assert x.shape == (2, 2)
assert (x + y).data.tolist() == [[6, 8], [10, 12]]
assert x.sum().data == 10
print("✓ Basic operations working")

# Test broadcasting
small = Tensor([1, 2])
result = x + small
assert result.data.tolist() == [[2, 4], [4, 6]]
print("✓ Broadcasting functional")

# Test reductions
col_means = x.mean(axis=0)
assert np.allclose(col_means.data, [2.0, 3.0])
print("✓ Reductions working")

# Test neural network pattern: y = xW + b
batch = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
weights = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # (3, 2)
bias = Tensor([0.1, 0.2])
output = batch.matmul(weights) + bias
assert output.shape == (2, 2)
print("✓ Neural network forward pass pattern works!")
```

## Production Context

### Your Implementation vs. Production Frameworks

Understanding what you're building vs. what production frameworks provide:

| Feature | Your Tensor (Module 01) | PyTorch torch.Tensor | TensorFlow tf.Tensor |
|---------|------------------------|---------------------|---------------------|
| **Backend** | NumPy (CPU-only) | C++/CUDA (CPU/GPU/TPU) | C++/CUDA/XLA |
| **Dtype Support** | float32 (primary) | float16/32/64, int8/16/32/64, bool, complex | Same + bfloat16 |
| **Operations** | Arithmetic, matmul, reshape, transpose, reductions | 1000+ operations | 1000+ operations |
| **Broadcasting** | ✅ Full NumPy rules | ✅ Same rules | ✅ Same rules |
| **Autograd** | Dormant (activates Module 05) | ✅ Full computation graph | ✅ GradientTape |
| **GPU Support** | ❌ CPU-only | ✅ CUDA, Metal, ROCm | ✅ CUDA, TPU |
| **Memory Pooling** | ❌ Python GC | ✅ Caching allocator | ✅ Memory pools |
| **JIT Compilation** | ❌ Interpreted | ✅ TorchScript, torch.compile | ✅ XLA, TF Graph |
| **Distributed** | ❌ Single process | ✅ DDP, FSDP | ✅ tf.distribute |

**Educational focus**: Your implementation prioritizes clarity and understanding over performance. The core concepts (broadcasting, shape manipulation, reductions) are identical - you're learning the same patterns used in production, just with simpler infrastructure.

**Line count**: Your implementation is ~1927 lines in the notebook (including tests and documentation). PyTorch's tensor implementation spans 50,000+ lines across multiple C++ files - your simplified version captures the essential concepts.

### Side-by-Side Code Comparison

**Your implementation:**
```python
from tinytorch.core.tensor import Tensor

# Create tensors
x = Tensor([[1, 2], [3, 4]])
w = Tensor([[0.5, 0.6], [0.7, 0.8]])

# Forward pass
output = x.matmul(w)  # (2,2) @ (2,2) → (2,2)
loss = output.mean()  # Scalar loss
```

**Equivalent PyTorch (production):**
```python
import torch

# Create tensors (GPU-enabled)
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).cuda()
w = torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=torch.float32).cuda()

# Forward pass (automatic gradient tracking)
output = x @ w        # Uses cuBLAS for GPU acceleration
loss = output.mean()  # Builds computation graph for backprop
loss.backward()       # Automatic differentiation
```

**Key differences:**
1. **GPU Support**: PyTorch tensors can move to GPU (`.cuda()`) for 10-100x speedup via parallel processing
2. **Autograd**: PyTorch automatically tracks operations and computes gradients - you'll build this in Module 05
3. **Memory Pooling**: PyTorch reuses GPU memory via caching allocator - avoids expensive malloc/free calls
4. **Optimized Kernels**: PyTorch uses cuBLAS/cuDNN (GPU) and Intel MKL (CPU) - hand-tuned assembly for max performance

### Real-World Production Usage

**Meta (Facebook AI)**: PyTorch was developed at Meta and powers their recommendation systems, computer vision models, and LLaMA language models. Their production infrastructure processes billions of tensor operations per second.

**Tesla**: Uses PyTorch tensors for Autopilot neural networks. Each camera frame (6-9 cameras) is converted to tensors, processed through vision models (millions of parameters stored as tensors), and outputs driving decisions in real-time at 36 FPS.

**OpenAI**: GPT-4 training involved tensors with billions of parameters distributed across thousands of GPUs. Each training step performs matrix multiplications on tensors larger than single GPU memory.

**Google**: TensorFlow powers Google Search, Translate, Photos, and Assistant. Google's TPUs (Tensor Processing Units) are custom hardware designed specifically for accelerating tensor operations.

### Performance Characteristics at Scale

**Memory usage**: GPT-3 scale models (175B parameters) require ~350GB memory just for weights stored as float16 tensors (175B × 2 bytes). Mixed precision training (float16/float32) reduces memory by 2x while maintaining accuracy.

**Computational bottlenecks**: In production training, tensor operations consume 95%+ of runtime. A single linear layer's matrix multiplication might take 100ms of a 110ms forward pass - optimizing tensor operations is critical.

**Cache efficiency**: Modern CPUs have ~32KB L1 cache, ~256KB L2, ~8MB L3. Accessing memory in tensor-friendly patterns (contiguous, row-major) can be 10-100x faster than cache-unfriendly patterns (strided, column-major).

### Package Integration

After export, your Tensor implementation becomes the foundation of TinyTorch:

**Package Export**: Code exports to `tinytorch.core.tensor`

```python
# When students install tinytorch, they import YOUR work:
from tinytorch.core.tensor import Tensor  # Your implementation!

# Future modules build on YOUR tensor:
from tinytorch.core.activations import ReLU  # Module 02 - operates on your Tensors
from tinytorch.core.layers import Linear     # Module 03 - uses your Tensor for weights
from tinytorch.core.autograd import backward # Module 05 - adds gradients to your Tensor
from tinytorch.core.optimizers import SGD    # Module 06 - updates your Tensor parameters
```

**Package structure:**
```
tinytorch/
├── core/
│   ├── tensor.py          ← YOUR implementation exports here
│   ├── activations.py     ← Module 02 builds on your Tensor
│   ├── layers.py          ← Module 03 builds on your Tensor
│   ├── losses.py          ← Module 04 builds on your Tensor
│   ├── autograd.py        ← Module 05 adds gradients to your Tensor
│   ├── optimizers.py      ← Module 06 updates your Tensor weights
│   └── ...
```

Your Tensor class is the universal foundation - every subsequent module depends on what you build here.

### How Your Implementation Maps to PyTorch

**What you just built:**
```python
# Your TinyTorch Tensor implementation
from tinytorch.core.tensor import Tensor

# Create a tensor
x = Tensor([[1, 2], [3, 4]])

# Core operations you implemented
y = x + 2              # Broadcasting
z = x.matmul(other)    # Matrix multiplication
mean = x.mean(axis=0)  # Reductions
reshaped = x.reshape(-1)  # Shape manipulation
```

**How PyTorch does it:**
```python
# PyTorch equivalent
import torch

# Create a tensor
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Same operations, identical semantics
y = x + 2              # Broadcasting (same rules)
z = x @ other          # Matrix multiplication (@ operator)
mean = x.mean(dim=0)   # Reductions (dim instead of axis)
reshaped = x.reshape(-1)  # Shape manipulation (same API)
```

**Key Insight**: Your implementation uses the **same mathematical operations and design patterns** that PyTorch uses internally. The `@` operator is syntactic sugar for matrix multiplication—the actual computation is identical. Broadcasting rules, shape semantics, and reduction operations all follow the same NumPy conventions.

**What's the SAME?**
- Tensor abstraction and API design
- Broadcasting rules and memory layout principles
- Shape manipulation semantics (`reshape`, `transpose`)
- Reduction operation behavior (`sum`, `mean`, `max`)
- Conceptual architecture: data + operations + metadata

**What's different in production PyTorch?**
- **Backend**: C++/CUDA for 10-100× speed vs. NumPy
- **GPU support**: `.cuda()` moves tensors to GPU for parallel processing
- **Autograd integration**: `requires_grad=True` enables automatic differentiation (you'll build this in Module 05)
- **Memory optimization**: Caching allocator reuses GPU memory, avoiding expensive malloc/free

**Why this matters**: When you debug PyTorch code, you'll understand what's happening under tensor operations because you implemented them yourself. Shape mismatch errors, broadcasting bugs, memory issues—you know exactly how they work internally, not just how to call the API.

**Production usage example**:
```python
# PyTorch production code (after TinyTorch)
import torch.nn as nn

class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)  # Uses torch.Tensor internally

    def forward(self, x):
        return self.linear(x)  # Matrix multiply + bias (same as your Tensor.matmul)
```

After building your own Tensor class, you understand that `nn.Linear(in_features, out_features)` is essentially creating weight and bias tensors, then performing `x @ weights + bias` with your same broadcasting and matmul operations—just optimized in C++/CUDA.

## Common Pitfalls

### Shape Mismatch Errors

**Problem**: Matrix multiplication fails with cryptic errors like "shapes (2,3) and (2,2) not aligned"

**Solution**: Always verify inner dimensions match: `(M,K) @ (K,N)` requires K to be equal. Add shape validation with clear error messages:
```python
if a.shape[1] != b.shape[0]:
    raise ValueError(f"Cannot multiply ({a.shape[0]},{a.shape[1]}) @ ({b.shape[0]},{b.shape[1]}): {a.shape[1]} ≠ {b.shape[0]}")
```

### Broadcasting Confusion

**Problem**: Expected `(2,3) + (3,)` to broadcast but got error

**Solution**: Broadcasting aligns shapes *from the right*. `(2,3) + (3,)` works (broadcasts to `(2,3)`), but `(2,3) + (2,)` fails. Add dimension with reshape if needed: `tensor.reshape(2,1)` to make `(2,1)` broadcastable with `(2,3)`.

### View vs Copy Confusion

**Problem**: Modified a reshaped tensor and original changed unexpectedly

**Solution**: `reshape()` returns a *view* when possible - they share memory. Changes to the view affect the original. Use `.copy()` if you need independent data:
```python
view = tensor.reshape(2, 3)      # Shares memory
copy = tensor.reshape(2, 3).copy()  # Independent storage
```

### Axis Parameter Mistakes

**Problem**: `sum(axis=1)` on `(batch, features)` returned wrong shape

**Solution**: Axis semantics: `axis=0` reduces over first dimension (batch), `axis=1` reduces over second (features). For `(32, 128)` tensor, `sum(axis=0)` gives `(128,)`, `sum(axis=1)` gives `(32,)`. Visualize which dimension you're collapsing.

### Dtype Issues

**Problem**: Lost precision after operations, or got integer division instead of float

**Solution**: NumPy defaults to preserving dtype. Integer tensors do integer division (`5 / 2 = 2`). Always create tensors with float dtype explicitly: `Tensor([[1, 2]], dtype=np.float32)` or convert: `tensor.astype(np.float32)`.

### Memory Leaks with Large Tensors

**Problem**: Memory usage grows unbounded during training loop

**Solution**: Clear intermediate results in loops. Don't accumulate tensors in lists unnecessarily. Use in-place operations when safe. Example:
```python
# Bad: accumulates memory
losses = []
for batch in data:
    loss = model(batch)
    losses.append(loss)  # Keeps all tensors in memory

# Good: extract values
losses = []
for batch in data:
    loss = model(batch)
    losses.append(loss.data.item())  # Store scalar, release tensor
```

## Systems Thinking Questions

### Real-World Applications

- **Deep Learning Training**: All neural network layers operate on tensors - Linear layers perform matrix multiplication, Conv2d applies tensor convolutions, Attention mechanisms compute tensor dot products. How would doubling model size affect memory and compute requirements?
- **Computer Vision**: Images are 3D tensors (height × width × channels), and every transformation (resize, crop, normalize) is a tensor operation. What's the memory footprint of a batch of 32 images at 224×224 resolution with 3 color channels in float32?
- **Natural Language Processing**: Text embeddings are 2D tensors (sequence_length × embedding_dim), and Transformer models manipulate these through attention. For BERT with 512 sequence length and 768 hidden dimension, how many elements per sample?
- **Scientific Computing**: Tensors represent multidimensional data in climate models, molecular simulations, physics engines. What makes tensors more efficient than nested Python lists for these applications?

### Mathematical Foundations

- **Linear Algebra**: Tensors generalize matrices to arbitrary dimensions. How does broadcasting relate to outer products? When is `(M,K) @ (K,N)` more efficient than `(K,M).T @ (K,N)`?
- **Numerical Stability**: Operations like softmax require careful implementation to avoid overflow/underflow. Why does `exp(x - max(x))` prevent overflow in softmax computation?
- **Broadcasting Semantics**: NumPy's broadcasting rules enable elegant code but require understanding shape compatibility. Can you predict the output shape of `(32, 1, 10) + (1, 5, 10)`?
- **Computational Complexity**: Matrix multiplication is O(n³) while element-wise operations are O(n). For large models, which dominates training time and why?

### Performance Characteristics

- **Memory Contiguity**: Contiguous memory enables SIMD vectorization and cache efficiency. How much can non-contiguous tensors slow down operations (10x? 100x?)?
- **View vs Copy**: Views are O(1) with shared memory, copies are O(n) with duplicated storage. When might a view cause unexpected behavior (e.g., in-place operations)?
- **Operation Fusion**: Frameworks optimize `(a + b) * c` by fusing operations to reduce memory reads. How many memory passes does unfused require vs. fused?
- **Batch Processing**: Processing 32 images at once is much faster than 32 sequential passes. Why? (Hint: GPU parallelism, cache reuse, reduced Python overhead)

## What's Next

After mastering tensors, you're ready to build the computational layers of neural networks:

**Module 02: Activations** - Implement ReLU, Sigmoid, Tanh, and Softmax activation functions that introduce non-linearity. You'll operate on your Tensor class and understand why activation functions are essential for learning complex patterns.

**Module 03: Layers** - Build Linear (fully-connected) and convolutional layers using tensor operations. See how weight matrices and bias vectors (stored as Tensors) transform inputs through matrix multiplication and broadcasting.

**Module 05: Autograd** - Add automatic differentiation to your Tensor class, enabling gradient computation for training. Your tensors will track operations and compute gradients automatically - the magic behind `loss.backward()`.

**Preview of tensor usage ahead:**
- Activations: `output = ReLU()(input_tensor)` - element-wise operations on tensors
- Layers: `output = Linear(in_features=128, out_features=64)(input_tensor)` - matmul with weight tensors
- Loss: `loss = MSELoss()(predictions, targets)` - tensor reductions for error measurement
- Training: `optimizer.step()` updates parameter tensors using gradients

Every module builds on your Tensor foundation - understanding tensors deeply means understanding how neural networks actually compute.

## Ready to Build?

You're about to implement the foundation of all machine learning systems! The Tensor class you'll build is the universal data structure that powers everything from simple neural networks to GPT, Stable Diffusion, and AlphaFold.

This is where mathematical abstraction meets practical implementation. You'll see how N-dimensional arrays enable elegant representations of complex data, how operator overloading makes tensor math feel natural like `z = x + y`, and how careful memory management (views vs. copies) enables working with massive models. Every decision you make - from how to handle broadcasting to when to validate shapes - reflects trade-offs that production ML engineers face daily.

Take your time with this module. Understand each operation deeply. Test your implementations thoroughly. The Tensor foundation you build here will support every subsequent module - if you understand tensors from first principles, you'll understand how neural networks actually work, not just how to use them.

Every neural network you've ever used - ResNet, BERT, GPT, Stable Diffusion - is fundamentally built on tensor operations. Understanding tensors means understanding the computational substrate of modern AI.

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/01_tensor/tensor_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/01_tensor/tensor_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/01_tensor/tensor_dev.ipynb
:class-header: bg-light

Browse the Jupyter notebook and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

```

---

<div class="prev-next-area">
<a class="right-next" href="../02_activations/ABOUT.html" title="next page">Next Module →</a>
</div>
