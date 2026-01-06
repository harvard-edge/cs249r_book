# Module 01: Tensor

:::{admonition} Module Info
:class: note

**FOUNDATION TIER** | Difficulty: ●○○○ | Time: 4-6 hours | Prerequisites: None

**Prerequisites: None** means exactly that. This module assumes:
- Basic Python (lists, classes, methods)
- Basic math (matrix multiplication from linear algebra)
- No machine learning background required

If you can multiply two matrices by hand and write a Python class, you're ready.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F01_tensor%2F01_tensor.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/01_tensor/01_tensor.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/01_tensor.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

The Tensor class is the foundational data structure of machine learning. Every neural network, from recognizing handwritten digits to translating languages, operates on tensors. These networks process millions of numbers per second, and tensors are the data structure that makes this possible. In this module, you'll build N-dimensional arrays from scratch, gaining deep insight into how PyTorch works under the hood.

By the end, your tensor will support arithmetic, broadcasting, matrix multiplication, and shape manipulation - exactly like `torch.Tensor`.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** a complete Tensor class with arithmetic, matrix multiplication, shape manipulation, and reductions
- **Master** broadcasting semantics that enable efficient computation without data copying
- **Understand** computational complexity (O(n³) for matmul) and memory trade-offs (views vs copies)
- **Connect** your implementation to production PyTorch patterns and design decisions
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Tensor Class
flowchart LR
    subgraph "Your Tensor Class"
        A["Properties<br/>shape, size, dtype"]
        B["Arithmetic<br/>+, -, *, /"]
        C["Matrix Ops<br/>matmul()"]
        D["Shape Ops<br/>reshape, transpose"]
        E["Reductions<br/>sum, mean, max"]
    end

    A --> B --> C --> D --> E

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
    style E fill:#e2d5f1
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `__init__`, `shape`, `size`, `dtype` | Tensor as NumPy wrapper |
| 2 | `__add__`, `__sub__`, `__mul__`, `__truediv__` | Operator overloading + broadcasting |
| 3 | `matmul()` | Matrix multiplication with shape validation |
| 4 | `reshape()`, `transpose()` | Shape manipulation, views vs copies |
| 5 | `sum()`, `mean()`, `max()` | Reductions along axes |

**The pattern you'll enable:**
```python
# Computing predictions from data
output = x.matmul(W) + b  # Matrix multiplication + bias (used in every neural network)
```

### What You're NOT Building

To keep this module focused, you will **not** implement:

- GPU support (NumPy runs on CPU only)
- Automatic differentiation
- Hundreds of tensor operations (PyTorch has 2000+, you'll build ~15 core ones)
- Memory optimization tricks (PyTorch uses lazy evaluation, memory pools, etc.)

**You are building the conceptual foundation.**

## API Reference

This section provides a quick reference for the Tensor class you'll build. Think of it as your cheat sheet while implementing and debugging. Each method is documented with its signature and expected behavior.

### Constructor

```python
Tensor(data)
```
- `data`: list, numpy array, or scalar

### Properties

Your Tensor wraps a NumPy array and exposes several properties that describe its structure. These properties are read-only and computed from the underlying data.

| Property | Type | Description |
|----------|------|-------------|
| `data` | `np.ndarray` | Underlying NumPy array |
| `shape` | `tuple` | Dimensions, e.g., `(2, 3)` |
| `size` | `int` | Total number of elements |
| `dtype` | `np.dtype` | Data type (float32) |

### Arithmetic Operations

Python lets you override operators like `+` and `*` by implementing special methods. When you write `x + y`, Python calls `x.__add__(y)`. Your implementations should handle both Tensor-Tensor operations and Tensor-scalar operations, letting NumPy's broadcasting do the heavy lifting.

| Operation | Method | Example |
|-----------|--------|---------|
| Addition | `__add__` | `x + y` or `x + 2` |
| Subtraction | `__sub__` | `x - y` |
| Multiplication | `__mul__` | `x * y` |
| Division | `__truediv__` | `x / y` |

### Matrix & Shape Operations

These methods transform tensors without changing their data (for views) or perform mathematical operations that produce new data (for matmul).

| Method | Signature | Description |
|--------|-----------|-------------|
| `matmul` | `matmul(other) -> Tensor` | Matrix multiplication |
| `reshape` | `reshape(*shape) -> Tensor` | Change shape (-1 to infer) |
| `transpose` | `transpose(dim0=None, dim1=None) -> Tensor` | Swap dimensions (defaults to last two) |

### Reductions

Reduction operations collapse one or more dimensions by aggregating values. The `axis` parameter controls which dimension gets collapsed. If `axis=None`, all dimensions collapse to a single scalar.

| Method | Signature | Description |
|--------|-----------|-------------|
| `sum` | `sum(axis=None, keepdims=False) -> Tensor` | Sum elements |
| `mean` | `mean(axis=None, keepdims=False) -> Tensor` | Average elements |
| `max` | `max(axis=None, keepdims=False) -> Tensor` | Maximum element |

## Core Concepts

This section covers the fundamental ideas you need to understand tensors deeply. These concepts apply to every ML framework, not just TinyTorch, so mastering them here will serve you throughout your career.

### Tensor Dimensionality

Tensors generalize the familiar concepts you already know. A scalar is just a single number, like a temperature reading of 72.5 degrees. Stack scalars into a list and you get a vector, like a series of temperature measurements throughout the day. Arrange vectors into rows and you get a matrix, like a spreadsheet where each row is a different day's measurements. Keep stacking and you reach 3D and 4D tensors that can represent video frames or collections of images.

The beauty of the Tensor abstraction is that your single class handles all of these cases. The same code that adds two scalars can add two 4D tensors, thanks to broadcasting.

| Rank | Name | Shape | Concrete Example |
|------|------|-------|------------------|
| 0D | Scalar | `()` | Temperature reading: `72.5` |
| 1D | Vector | `(768,)` | Audio sample: 768 measurements |
| 2D | Matrix | `(128, 768)` | Spreadsheet: 128 rows × 768 columns |
| 3D | 3D Tensor | `(32, 224, 224)` | Video frames: 32 grayscale images |
| 4D | 4D Tensor | `(32, 3, 224, 224)` | Video frames: 32 color (RGB) images |

### Broadcasting

When you add a vector to a matrix, the shapes don't match. Should this fail? In most programming contexts, yes. But many computations need to apply the same operation across rows or columns. For example, if you want to adjust all values in a spreadsheet by adding a different offset to each column, you need to add a vector to a matrix. NumPy and PyTorch implement broadcasting to handle this: automatically expanding smaller tensors to match larger ones without actually copying data.

Consider adding a bias vector `[10, 20, 30]` to every row of a matrix. Without broadcasting, you'd need to manually tile the vector into a matrix first, wasting memory. With broadcasting, the operation just works, and the framework handles alignment internally.

Here's how your `__add__` implementation handles this elegantly:

```python
def __add__(self, other):
    """Add two tensors element-wise with broadcasting support."""
    if isinstance(other, Tensor):
        return Tensor(self.data + other.data)  # NumPy handles broadcasting!
    else:
        return Tensor(self.data + other)  # Scalar broadcast
```

The elegance is that NumPy's broadcasting rules apply automatically when you write `self.data + other.data`. NumPy aligns the shapes from right to left and expands dimensions where needed, all without copying data.

```{mermaid}
:align: center
:caption: Broadcasting Example
flowchart LR
    subgraph "Broadcasting Example"
        M["Matrix (2,3)<br/>[[1,2,3], [4,5,6]]"]
        V["Vector (3,)<br/>[10,20,30]"]
        R["Result (2,3)<br/>[[11,22,33], [14,25,36]]"]
    end

    M --> |"+"| R
    V --> |"expands"| R

    style M fill:#e1f5ff
    style V fill:#fff3cd
    style R fill:#d4edda
```

The rules are simpler than they look. Compare shapes from right to left. At each position, dimensions are compatible if they're equal or if one of them is 1. Missing dimensions on the left are treated as 1. If any position fails this check, broadcasting fails.

| Shape A | Shape B | Result | Valid? |
|---------|---------|--------|--------|
| `(3, 4)` | `(4,)` | `(3, 4)` | ✓ |
| `(3, 4)` | `(3, 1)` | `(3, 4)` | ✓ |
| `(3, 4)` | `(3,)` | Error | ✗ (3 ≠ 4) |
| `(2, 3, 4)` | `(3, 4)` | `(2, 3, 4)` | ✓ |

The memory savings are dramatic. Adding a `(768,)` vector to a `(32, 512, 768)` tensor would require copying the vector 32×512 times without broadcasting, allocating 50 MB of redundant data (12.5 million float32 numbers). With broadcasting, you store just the original 3 KB vector.

### Views vs. Copies

When you reshape a tensor, does it allocate new memory or just create a different view of the same data? The answer has huge implications for both performance and correctness.

A view shares memory with its source. Reshaping a 1 GB tensor is instant because you're just changing the metadata that describes how to interpret the bytes, not copying the bytes themselves. But this creates an important gotcha: modifying a view modifies the original.

```python
x = Tensor([1, 2, 3, 4])
y = x.reshape(2, 2)  # y is a VIEW of x
y.data[0, 0] = 99    # This also changes x!
```

Arithmetic operations like addition always create copies because they compute new values. This is safer but uses more memory. Production code carefully manages views to avoid both memory blowup (too many copies) and silent bugs (unexpected mutations through views).

| Operation | Type | Memory | Time |
|-----------|------|--------|------|
| `reshape()` | View* | Shared | O(1) |
| `transpose()` | View* | Shared | O(1) |
| `+ - * /` | Copy | New allocation | O(n) |

*When data is contiguous in memory

### Matrix Multiplication

Matrix multiplication is the computational workhorse of neural networks. Every linear layer, every attention head, every embedding lookup involves matmul. Understanding its mechanics and cost is essential.

The operation is simple in concept: for each output element, compute a dot product of a row from the first matrix with a column from the second. But this simplicity hides cubic complexity. Multiplying two n×n matrices requires n³ multiplications and n³ additions.

Here's how the educational implementation in your module works:

```python
def matmul(self, other):
    """Matrix multiplication of two tensors."""
    if not isinstance(other, Tensor):
        raise TypeError(f"Expected Tensor for matrix multiplication, got {type(other)}")

    # Shape validation: inner dimensions must match
    if len(self.shape) >= 2 and len(other.shape) >= 2:
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(
                f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}. "
                f"Inner dimensions must match: {self.shape[-1]} ≠ {other.shape[-2]}"
            )

    a = self.data
    b = other.data

    # Handle 2D matrices with explicit loops (educational)
    if len(a.shape) == 2 and len(b.shape) == 2:
        M, K = a.shape
        K2, N = b.shape
        result_data = np.zeros((M, N), dtype=a.dtype)

        # Each output element is a dot product
        for i in range(M):
            for j in range(N):
                result_data[i, j] = np.dot(a[i, :], b[:, j])
    else:
        # For batched operations, use np.matmul
        result_data = np.matmul(a, b)

    return Tensor(result_data)
```

The explicit loops in the 2D case are intentionally slower than `np.matmul` because they reveal exactly what matrix multiplication does: each output element requires K operations, and there are M×N outputs, giving O(M×K×N) total operations. For square matrices, this is O(n³).

### Shape Manipulation

Shape manipulation operations change how data is interpreted without changing the values themselves. Understanding when data is copied versus viewed is crucial for both correctness and performance.

The `reshape` method reinterprets the same data with different dimensions:

```python
def reshape(self, *shape):
    """Reshape tensor to new dimensions."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        new_shape = tuple(shape[0])
    else:
        new_shape = shape

    # Handle -1 for automatic dimension inference
    if -1 in new_shape:
        known_size = 1
        unknown_idx = new_shape.index(-1)
        for i, dim in enumerate(new_shape):
            if i != unknown_idx:
                known_size *= dim
        unknown_dim = self.size // known_size
        new_shape = list(new_shape)
        new_shape[unknown_idx] = unknown_dim
        new_shape = tuple(new_shape)

    # Validate total elements match
    if np.prod(new_shape) != self.size:
        raise ValueError(
            f"Total elements must match: {self.size} ≠ {int(np.prod(new_shape))}"
        )

    reshaped_data = np.reshape(self.data, new_shape)
    return Tensor(reshaped_data, requires_grad=self.requires_grad)
```

The `-1` syntax is particularly useful: it tells NumPy to infer one dimension automatically. When flattening a batch of images, `x.reshape(batch_size, -1)` lets NumPy calculate the feature dimension.

### Computational Complexity

Not all tensor operations are equal. Element-wise operations like addition visit each element once: O(n) time where n is the total number of elements. Reductions like sum also visit each element once. But matrix multiplication is fundamentally different.

Multiplying two n×n matrices requires n³ operations: for each of the n² output elements, you compute a dot product of n values. This cubic scaling is why a 2000×2000 matmul takes 8x longer than a 1000×1000 matmul, not 4x. In neural networks, matrix multiplications consume over 90% of compute time. This is precisely why GPUs exist for ML: a modern GPU has thousands of cores that can compute thousands of dot products simultaneously, turning an 800ms CPU operation into an 8ms GPU operation.

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Element-wise (`+`, `-`, `*`) | O(n) | Linear in tensor size |
| Reductions (`sum`, `mean`) | O(n) | Must visit every element |
| Matrix multiply (`matmul`) | O(n³) | Dominates training time |

### Axis Semantics

The `axis` parameter in reductions specifies which dimension to collapse. Think of it as "sum along this axis" or "average out this dimension." The result has one fewer dimension than the input.

For a 2D tensor with shape `(rows, columns)`, summing along axis 0 collapses the rows, giving you column totals. Summing along axis 1 collapses the columns, giving you row totals. Summing with `axis=None` collapses everything to a single scalar.

Your reduction implementations simply pass the axis to NumPy:

```python
def sum(self, axis=None, keepdims=False):
    """Sum tensor along specified axis."""
    result = np.sum(self.data, axis=axis, keepdims=keepdims)
    return Tensor(result)
```

The `keepdims=True` option preserves the reduced dimension as size 1, which is useful for broadcasting the result back.

```
For shape (rows, columns) = (32, 768):

sum(axis=0) → collapse rows    → shape (768,)  - column totals
sum(axis=1) → collapse columns → shape (32,)   - row totals
sum(axis=None) → collapse all  → scalar

Visual:
[[1, 2, 3],      sum(axis=0)     sum(axis=1)
 [4, 5, 6]]  →   [5, 7, 9]   or  [6, 15]
                 (down cols)     (across rows)
```

## Architecture

Your Tensor sits at the top of a stack that reaches down to hardware. When you call `x + y`, Python calls your `__add__` method, which delegates to NumPy, which calls optimized BLAS libraries written in C and Fortran, which use CPU SIMD instructions that process multiple numbers in a single clock cycle.

```{mermaid}
:align: center
:caption: Your Code
flowchart TB
    subgraph "Your Code"
        A["Python Interface<br/>x = Tensor([[1,2],[3,4]])"]
    end

    subgraph "TinyTorch"
        B["Tensor Class<br/>shape, data, operations"]
    end

    subgraph "Backend"
        C["NumPy<br/>Vectorized operations"]
        D["BLAS/LAPACK<br/>C/Fortran libraries"]
    end

    subgraph "Hardware"
        E["CPU SIMD<br/>Cache optimization"]
    end

    A --> B --> C --> D --> E

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#d4edda
    style E fill:#f8d7da
```

This is the same architecture used by PyTorch and TensorFlow, just with different backends. PyTorch replaces NumPy with a C++ engine and BLAS with CUDA kernels running on GPUs. But the Python interface and the abstractions are identical. When you understand TinyTorch's Tensor, you understand PyTorch's Tensor.

### Module Integration

Your Tensor is the foundation for everything that follows in TinyTorch. Every subsequent module builds on the work you do here.

```
Module 01: Tensor (THIS MODULE)
    ↓ provides foundation
    All other modules build on this
```

## Common Errors & Debugging

These are the errors you'll encounter most often when working with tensors. Understanding why they happen will save you hours of debugging, both in this module and throughout your ML career.

### Shape Mismatch in matmul

**Error**: `ValueError: shapes (2,3) and (2,2) not aligned`

Matrix multiplication requires the inner dimensions to match. If you're multiplying `(M, K)` by `(K, N)`, both K values must be equal. The error above happens when trying to multiply a (2,3) matrix by a (2,2) matrix: 3 ≠ 2.

**Fix**: Check your shapes. The rule is `a.shape[-1]` must equal `b.shape[-2]`.

### Broadcasting Failures

**Error**: `ValueError: operands could not be broadcast together`

Broadcasting fails when shapes can't be aligned according to the rules. Remember: compare right to left, and dimensions must be equal or one must be 1.

**Examples**:
- `(2,3) + (3,)` ✓ works - 3 matches 3, and the missing dimension becomes 1
- `(2,3) + (2,)` ✗ fails - comparing right to left: 3 ≠ 2

**Fix**: Reshape to make dimensions compatible: `vector.reshape(-1, 1)` or `vector.reshape(1, -1)`

### Reshape Size Mismatch

**Error**: `ValueError: cannot reshape array of size X into shape Y`

Reshape only rearranges elements; it can't create or destroy them. If you have 12 elements, you can reshape to (3, 4) or (2, 6) or (2, 2, 3), but not to (5, 5).

**Fix**: Ensure `np.prod(old_shape) == np.prod(new_shape)`

### Missing Attributes

**Error**: `AttributeError: 'Tensor' has no attribute 'shape'`

Your `__init__` method needs to set all required attributes. If you forget to set `self.shape`, any code that accesses `tensor.shape` will fail.

**Fix**: Add `self.shape = self.data.shape` in your constructor

### Type Errors in Arithmetic

**Error**: `TypeError: unsupported operand type(s) for +: 'Tensor' and 'int'`

Your arithmetic methods need to handle both Tensor and scalar operands. When someone writes `x + 2`, your `__add__` receives the integer 2, not a Tensor.

**Fix**: Check for scalars: `if isinstance(other, (int, float)): ...`

## Production Context

### Your Implementation vs. PyTorch

Your TinyTorch Tensor and PyTorch's `torch.Tensor` share the same conceptual design. The differences are in implementation: PyTorch uses a C++ backend for speed, supports GPUs for massive parallelism, and implements thousands of specialized operations. But the Python API, broadcasting rules, and shape semantics are identical.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Backend** | NumPy (Python) | C++/CUDA |
| **Speed** | 1x (baseline) | 10-100x faster |
| **GPU** | ✗ CPU only | ✓ CUDA, Metal, ROCm |
| **Operations** | ~15 core ops | 2000+ operations |

### Code Comparison

The following comparison shows equivalent operations in TinyTorch and PyTorch. Notice how closely the APIs mirror each other. This is intentional: by learning TinyTorch's patterns, you're simultaneously learning PyTorch's patterns.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.core.tensor import Tensor

x = Tensor([[1, 2], [3, 4]])
y = x + 2
z = x.matmul(w)
loss = z.mean()
```
````

````{tab-item} ⚡ PyTorch
```python
import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = x + 2
z = x @ w
loss = z.mean()
```
````
`````

Let's walk through each line to understand the comparison:

- **Line 1 (Import)**: Both frameworks use a simple import. TinyTorch exposes `Tensor` from `core.tensor`; PyTorch uses `torch.tensor()` as a factory function.
- **Line 3 (Creation)**: TinyTorch infers dtype from input; PyTorch requires explicit `dtype=torch.float32` for floating-point operations. This explicitness matters for performance tuning in production.
- **Line 4 (Broadcasting)**: Both handle `x + 2` identically, broadcasting the scalar across all elements. Same semantics, same result.
- **Line 5 (Matrix multiply)**: TinyTorch uses `.matmul()` method; PyTorch supports both `.matmul()` and the `@` operator. The operation is identical.
- **Line 6 (Reduction)**: Both use `.mean()` to reduce the tensor to a scalar. Reductions like this are fundamental to computing loss values.

```{tip} What's Identical

Broadcasting rules, shape semantics, and API design patterns. When you debug PyTorch shape errors, you'll understand exactly what's happening because you built the same abstractions.
```

### Why Tensors Matter at Scale

To appreciate why tensor operations matter, consider the scale of modern ML systems:

- **Large language models**: 175 billion numbers stored as tensors = **350 GB** (like storing 70,000 full-resolution photos)
- **Image processing**: A batch of 128 images = **77 MB** of tensor data
- **Self-driving cars**: Process tensor operations at **36 FPS** across multiple cameras (each frame = millions of operations in 28 milliseconds)

A single matrix multiplication can consume **90% of computation time** in neural networks. Understanding tensor operations isn't just academic; it's essential for building and debugging real ML systems.

## Check Your Understanding

Test yourself with these systems thinking questions. They're designed to build intuition for the performance characteristics you'll encounter in production ML.

**Q1: Memory Calculation**

A batch of 32 RGB images (224×224 pixels) stored as float32. How much memory?

```{admonition} Answer
:class: dropdown

32 × 3 × 224 × 224 × 4 = **77,070,336 bytes ≈ 77 MB**

This is why batch size matters - double the batch, double the memory!
```

**Q2: Broadcasting Savings**

Adding a vector `(768,)` to a 3D tensor `(32, 512, 768)`. How much memory does broadcasting save?

```{admonition} Answer
:class: dropdown

Without broadcasting: 32 × 512 × 768 × 4 = **50.3 MB**

With broadcasting: 768 × 4 = **3 KB**

Savings: **~50 MB per operation** - this adds up across hundreds of operations in a neural network!
```

**Q3: Matmul Scaling**

If a 1000×1000 matmul takes 100ms, how long will 2000×2000 take?

```{admonition} Answer
:class: dropdown

Matmul is O(n³). Doubling n → 2³ = **8x longer** → ~800ms

This is why matrix size matters so much for transformer scaling!
```

**Q4: Shape Prediction**

What's the output shape of `(32, 1, 768) + (512, 768)`?

```{admonition} Answer
:class: dropdown

Broadcasting aligns right-to-left:
- `(32,   1, 768)`
- `(    512, 768)`

Result: **(32, 512, 768)**

The 1 broadcasts to 512, and 32 is prepended.
```

**Q5: Views vs Copies**

You reshape a 1GB tensor, then modify one element in the reshaped version. What happens to the original tensor? What if you had used `x + 0` instead of reshape?

```{admonition} Answer
:class: dropdown

**Reshape (view)**: The original tensor IS modified. Reshape creates a view that shares memory with the original. Changing `y.data[0,0] = 99` also changes `x.data[0]`.

**Addition (copy)**: The original tensor is NOT modified. `x + 0` creates a new tensor with freshly allocated memory. The values are identical but stored in different locations.

This distinction matters enormously for:
- **Memory**: Views use 0 extra bytes; copies use n extra bytes
- **Performance**: Views are O(1); copies are O(n)
- **Correctness**: Unexpected mutations through views are a common source of bugs
```

## Further Reading

For students who want to understand the academic foundations and mathematical underpinnings of tensor operations:

### Seminal Papers

- **NumPy: Array Programming** - Harris et al. (2020). The definitive reference for NumPy, which underlies your Tensor implementation. Explains broadcasting, views, and the design philosophy. [Nature](https://doi.org/10.1038/s41586-020-2649-2)

- **BLAS (Basic Linear Algebra Subprograms)** - Lawson et al. (1979). The foundation of all high-performance matrix operations. Your `np.matmul` ultimately calls BLAS routines optimized over 40+ years. Understanding BLAS levels (1, 2, 3) explains why matmul is special. [ACM TOMS](https://doi.org/10.1145/355841.355847)

- **Automatic Differentiation in ML** - Baydin et al. (2018). Survey of automatic differentiation techniques. [JMLR](https://www.jmlr.org/papers/v18/17-468.html)

### Additional Resources

- **Textbook**: "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter 2 covers linear algebra foundations including tensor operations
- **Documentation**: [PyTorch Tensor Tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) - See how production frameworks implement similar concepts
