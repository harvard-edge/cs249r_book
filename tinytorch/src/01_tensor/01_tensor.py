# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Module 01: Tensor Foundation - Building Blocks of ML

Welcome to Module 01! You're about to build the foundational Tensor class that powers all machine learning operations.

## 🔗 Prerequisites & Progress
**You've Built**: Nothing - this is our foundation!
**You'll Build**: A complete Tensor class with arithmetic, matrix operations, and shape manipulation
**You'll Enable**: Foundation for activations, layers, and all future neural network components

**Connection Map**:
```
NumPy Arrays → Tensor → Activations (Module 02)
(raw data)   (ML ops)  (intelligence)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement a complete Tensor class with fundamental operations
2. Understand tensors as the universal data structure in ML
3. Master broadcasting, matrix multiplication, and shape manipulation
4. Test tensor operations with immediate validation

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/01_tensor/tensor_dev.py
**Building Side:** Code exports to tinytorch.core.tensor

```python
# Final package structure:
# Other modules will import and use this Tensor
```

**Why this matters:**
- **Learning:** Complete tensor system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.Tensor with all core operations together
- **Consistency:** All tensor operations and data manipulation in core.tensor
- **Integration:** Foundation that every other module will build upon
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp core.tensor
#| export

import numpy as np

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: NONE - This is the foundation module

**External Dependencies**:
- `numpy` (for array operations and numerical computing)

**TinyTorch Dependencies**: NONE

**Important**: This module has NO TinyTorch dependencies.
Other modules will import FROM this module.

**Dependency Flow**:
```
Module 01 (Tensor) → All Other Modules
     ↓
  Foundation for entire TinyTorch system
```

Students completing this module will have built the foundation
that every other TinyTorch component depends on.
"""

# %% [markdown]
"""
## 💡 Introduction: What is a Tensor?

A tensor is a multi-dimensional array that serves as the fundamental data structure in machine learning. Think of it as a universal container that can hold data in different dimensions:

```
Tensor Dimensions:
┌─────────────┐
│ 0D: Scalar  │  5.0          (just a number)
│ 1D: Vector  │  [1, 2, 3]    (list of numbers)
│ 2D: Matrix  │  [[1, 2]      (grid of numbers)
│             │   [3, 4]]
│ 3D: Cube    │  [[[...       (stack of matrices)
└─────────────┘
```

In machine learning, tensors flow through operations like water through pipes:

```
Neural Network Data Flow:
Input Tensor → Layer 1 → Activation → Layer 2 → ... → Output Tensor
   [batch,     [batch,     [batch,     [batch,          [batch,
    features]   hidden]     hidden]     hidden2]         classes]
```

Every neural network, from simple linear regression to modern transformers, processes tensors. Understanding tensors means understanding the foundation of all ML computations.

### Why Tensors Matter in ML Systems

In production ML systems, tensors carry more than just data - they carry the computational graph, memory layout information, and execution context:

```
Real ML Pipeline:
Raw Data → Preprocessing → Tensor Creation → Model Forward Pass → Loss Computation
   ↓           ↓              ↓               ↓                    ↓
 Files     NumPy Arrays    Tensors        GPU Tensors         Scalar Loss
```

**Key Insight**: Tensors bridge the gap between mathematical concepts and efficient computation on modern hardware.
"""

# %% [markdown]
"""
## 📐 Foundations: Mathematical Background

### Core Operations We'll Implement

Our Tensor class will support all fundamental operations that neural networks need:

```
Operation Types:
┌─────────────────┬─────────────────┬─────────────────┐
│ Element-wise    │ Matrix Ops      │ Shape Ops       │
├─────────────────┼─────────────────┼─────────────────┤
│ + Addition      │ @ Matrix Mult   │ .reshape()      │
│ - Subtraction   │ .transpose()    │ .sum()          │
│ * Multiplication│                 │ .mean()         │
│ / Division      │                 │ .max()          │
└─────────────────┴─────────────────┴─────────────────┘
```

### Broadcasting: Making Tensors Work Together

Broadcasting automatically aligns tensors of different shapes for operations:

```
Broadcasting Examples:
┌─────────────────────────────────────────────────────────┐
│ Scalar + Vector:                                        │
│    5    + [1, 2, 3] → [5, 5, 5] + [1, 2, 3] = [6, 7, 8] │
│                                                         │
│ Matrix + Vector (row-wise):                             │
│ [[1, 2]]   [10]   [[1, 2]]   [[10, 10]]   [[11, 12]]    │
│ [[3, 4]] + [10] = [[3, 4]] + [[10, 10]] = [[13, 14]]    │
└─────────────────────────────────────────────────────────┘
```

**Memory Layout**: NumPy uses row-major (C-style) storage where elements are stored row by row in memory for cache efficiency:

```
Memory Layout (2×3 matrix):
Matrix:     Memory:
[[1, 2, 3]  [1][2][3][4][5][6]
 [4, 5, 6]]  ↑  Row 1   ↑  Row 2

Cache Behavior:
Sequential Access: Fast (uses cache lines efficiently)
  Row access: [1][2][3] → cache hit, hit, hit
Random Access: Slow (cache misses)
  Column access: [1][4] → cache hit, miss
```

This memory layout affects performance in real ML workloads - algorithms that access data sequentially run faster than those that access randomly.
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Tensor Foundation

Let's build our Tensor class step by step, testing each component as we go.

### Tensor Class Architecture

```
Tensor Class Structure:
┌─────────────────────────────────┐
│ Core Attributes:                │
│ • data: np.array (the numbers)  │
│ • shape: tuple (dimensions)     │
│ • size: int (total elements)    │
│ • dtype: type (float32)         │
├─────────────────────────────────┤
│ Arithmetic Operations:          │
│ • __add__, __sub__, __mul__     │
│ • __truediv__, matmul()         │
├─────────────────────────────────┤
│ Shape Operations:               │
│ • reshape(), transpose()        │
│ • sum(), mean(), max()          │
│ • __getitem__ (indexing)        │
├─────────────────────────────────┤
│ Utility Methods:                │
│ • __repr__(), __str__()         │
│ • numpy(), memory_footprint()   │
└─────────────────────────────────┘
```

This clean design focuses on what tensors fundamentally do: store and manipulate numerical data efficiently.
"""

# %% [markdown]
"""
### Tensor Creation and Initialization

Before we implement operations, let's understand how tensors store data and manage their attributes. This initialization is the foundation that everything else builds upon.

```
Tensor Initialization Process:
Input Data → Validation → NumPy Array → Tensor Wrapper → Ready for Operations
   [1,2,3] →    types   →  np.array   →    shape=(3,)  →     + - * / @ ...
     ↓             ↓          ↓             ↓
  List/Array    Type Check   Memory      Attributes Set
               (optional)    Allocation

Memory Allocation Example:
Input: [[1, 2, 3], [4, 5, 6]]
         ↓
NumPy allocates: [1][2][3][4][5][6] in contiguous memory
         ↓
Tensor wraps with: shape=(2,3), size=6, dtype=int64
```

**Key Design Principle**: Our Tensor is a wrapper around NumPy arrays that adds ML-specific functionality. We leverage NumPy's battle-tested memory management and computation kernels while adding the gradient tracking and operation chaining needed for deep learning.

**Why This Approach?**
- **Performance**: NumPy's C implementations are highly optimized
- **Compatibility**: Easy integration with scientific Python ecosystem
- **Memory Efficiency**: No unnecessary data copying
- **Future-Proof**: Easy transition to GPU tensors in advanced modules
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-class", "solution": true}
#| export
class Tensor:
    """Educational tensor - the foundation of machine learning computation.

    This class provides the core data structure for all ML operations:
    - data: The actual numerical values (NumPy array)
    - shape: Dimensions of the tensor
    - size: Total number of elements
    - dtype: Data type (float32)

    All arithmetic, matrix, and shape operations are built on this foundation.
    """

    def __init__(self, data):
        """Create a new tensor from data.

        TODO: Initialize a Tensor by wrapping data in a NumPy array and setting attributes.

        APPROACH:
        1. Convert data to NumPy array with dtype=float32
        2. Store the array as self.data
        3. Set self.shape from the array's shape
        4. Set self.size from the array's size
        5. Set self.dtype from the array's dtype

        EXAMPLE:
        >>> t = Tensor([1, 2, 3])
        >>> print(t.shape)
        (3,)
        >>> print(t.size)
        3

        HINT: Use np.array(data, dtype=np.float32) to convert data to NumPy array
        """
        ### BEGIN SOLUTION
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        ### END SOLUTION

    def __repr__(self):
        """String representation of tensor for debugging."""
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        """Human-readable string representation."""
        return f"Tensor({self.data})"

    def numpy(self):
        """Return the underlying NumPy array."""
        return self.data

    def memory_footprint(self):
        """Calculate exact memory usage in bytes.

        Systems Concept: Understanding memory footprint is fundamental to ML systems.
        Before running any operation, engineers should know how much memory it requires.

        Returns:
            int: Memory usage in bytes (e.g., 1000x1000 float32 = 4MB)
        """
        return self.data.nbytes

    def __add__(self, other):
        """Add two tensors element-wise with broadcasting support.

        TODO: Implement element-wise addition that works with both Tensors and scalars.

        APPROACH:
        1. Check if other is a Tensor (use isinstance)
        2. If Tensor: add self.data + other.data
        3. If scalar: add self.data + other (broadcasting)
        4. Wrap result in new Tensor

        EXAMPLE:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> c = a + b
        >>> print(c.data)
        [5. 7. 9.]

        HINT: NumPy's + operator handles broadcasting automatically
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)
        ### END SOLUTION

    def __sub__(self, other):
        """Subtract two tensors element-wise.

        TODO: Implement element-wise subtraction (same pattern as __add__).

        APPROACH:
        1. Check if other is a Tensor
        2. If Tensor: subtract self.data - other.data
        3. If scalar: subtract self.data - other
        4. Return new Tensor with result

        EXAMPLE:
        >>> a = Tensor([5, 7, 9])
        >>> b = Tensor([1, 2, 3])
        >>> c = a - b
        >>> print(c.data)
        [4. 5. 6.]

        HINT: Follow the same pattern as __add__ but with subtraction
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)
        ### END SOLUTION

    def __mul__(self, other):
        """Multiply two tensors element-wise (NOT matrix multiplication).

        TODO: Implement element-wise multiplication (same pattern as __add__).

        APPROACH:
        1. Check if other is a Tensor
        2. If Tensor: multiply self.data * other.data
        3. If scalar: multiply self.data * other
        4. Return new Tensor with result

        EXAMPLE:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> c = a * b
        >>> print(c.data)
        [ 4. 10. 18.]

        HINT: Element-wise multiplication is *, not matrix multiplication (@)
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)
        ### END SOLUTION

    def __truediv__(self, other):
        """Divide two tensors element-wise.

        TODO: Implement element-wise division (same pattern as __add__).

        APPROACH:
        1. Check if other is a Tensor
        2. If Tensor: divide self.data / other.data
        3. If scalar: divide self.data / other
        4. Return new Tensor with result

        EXAMPLE:
        >>> a = Tensor([4, 6, 8])
        >>> b = Tensor([2, 2, 2])
        >>> c = a / b
        >>> print(c.data)
        [2. 3. 4.]

        HINT: Division creates float results automatically due to float32 dtype
        """
        ### BEGIN SOLUTION
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)
        ### END SOLUTION

    def matmul(self, other):
        """Matrix multiplication of two tensors.

        TODO: Implement matrix multiplication with shape validation.

        APPROACH:
        1. Validate other is a Tensor (raise TypeError if not)
        2. Check for scalar cases (0D tensors) - use element-wise multiply
        3. For 2D+ matrices: validate inner dimensions match (shape[-1] == shape[-2])
        4. For 2D matrices: use explicit nested loops (educational)
        5. For batched (3D+): use np.matmul for correctness
        6. Return result wrapped in Tensor

        EXAMPLE:
        >>> a = Tensor([[1, 2], [3, 4]])  # 2×2
        >>> b = Tensor([[5, 6], [7, 8]])  # 2×2
        >>> c = a.matmul(b)
        >>> print(c.data)
        [[19. 22.]
         [43. 50.]]

        HINTS:
        - Inner dimensions must match: (M, K) @ (K, N) = (M, N)
        - For 2D case: use np.dot(a[i, :], b[:, j]) for each output element
        - Raise ValueError with clear message if shapes incompatible
        """
        ### BEGIN SOLUTION
        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor for matrix multiplication, got {type(other)}")
        if self.shape == () or other.shape == ():
            return Tensor(self.data * other.data)
        if len(self.shape) == 0 or len(other.shape) == 0:
            return Tensor(self.data * other.data)
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}. "
                    f"Inner dimensions must match: {self.shape[-1]} ≠ {other.shape[-2]}"
                )

        # Educational implementation: explicit loops to show what matrix multiplication does
        # This is intentionally slower than np.matmul to demonstrate the value of vectorization
        # In Module 17 (Acceleration), students will learn to use optimized BLAS operations

        a = self.data
        b = other.data

        # Handle 2D matrices with explicit loops (educational)
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape
            result_data = np.zeros((M, N), dtype=a.dtype)

            # Explicit nested loops - students can see exactly what's happening!
            # Each output element is a dot product of a row from A and a column from B
            for i in range(M):
                for j in range(N):
                    # Dot product of row i from A with column j from B
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        else:
            # For batched operations (3D+), use np.matmul for correctness
            # Students will understand this once they grasp the 2D case
            result_data = np.matmul(a, b)

        return Tensor(result_data)
        ### END SOLUTION

    def __matmul__(self, other):
        """Enable @ operator for matrix multiplication."""
        return self.matmul(other)

    def __getitem__(self, key):
        """Enable indexing and slicing operations on Tensors.

        TODO: Implement indexing and slicing that returns a new Tensor.

        APPROACH:
        1. Use NumPy indexing: self.data[key]
        2. If result is not an ndarray, wrap in np.array
        3. Return result wrapped in new Tensor

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> row = t[0]  # First row
        >>> print(row.data)
        [1. 2. 3.]
        >>> element = t[0, 1]  # Single element
        >>> print(element.data)
        2.0

        HINT: NumPy's indexing already handles all complex cases (slicing, fancy indexing)
        """
        ### BEGIN SOLUTION
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)
        ### END SOLUTION

    def reshape(self, *shape):
        """Reshape tensor to new dimensions.

        TODO: Reshape tensor while preserving total element count.

        APPROACH:
        1. Handle both reshape(2, 3) and reshape((2, 3)) calling styles
        2. If -1 in shape, infer that dimension from total size
        3. Validate total elements match: np.prod(new_shape) == self.size
        4. Use np.reshape to create new view
        5. Return result wrapped in new Tensor

        EXAMPLE:
        >>> t = Tensor([1, 2, 3, 4, 5, 6])
        >>> reshaped = t.reshape(2, 3)
        >>> print(reshaped.data)
        [[1. 2. 3.]
         [4. 5. 6.]]
        >>> auto = t.reshape(2, -1)  # Infers -1 as 3
        >>> print(auto.shape)
        (2, 3)

        HINTS:
        - Use isinstance(shape[0], (tuple, list)) to detect tuple input
        - For -1: unknown_dim = self.size // known_size
        - Raise ValueError if total elements don't match
        """
        ### BEGIN SOLUTION
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Can only specify one unknown dimension with -1")
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)
        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Total elements must match: {self.size} ≠ {target_size}"
            )
        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data)
        ### END SOLUTION

    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions.

        TODO: Swap tensor dimensions (default: swap last two dimensions).

        APPROACH:
        1. If no dims specified: swap last two dimensions (most common case)
        2. For 1D tensors: return copy (no transpose needed)
        3. If both dims specified: swap those specific dimensions
        4. Use np.transpose with axes list to perform the swap
        5. Return result wrapped in new Tensor

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3
        >>> transposed = t.transpose()
        >>> print(transposed.data)
        [[1. 4.]
         [2. 5.]
         [3. 6.]]  # 3×2

        HINTS:
        - Create axes list: [0, 1, 2, ...] then swap positions
        - For default: axes[-2], axes[-1] = axes[-1], axes[-2]
        - Use np.transpose(self.data, axes)
        """
        ### BEGIN SOLUTION
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)
        ### END SOLUTION

    def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis.

        TODO: Sum all elements or along specific axes.

        APPROACH:
        1. Use np.sum with axis and keepdims parameters
        2. axis=None sums all elements (scalar result)
        3. axis=N sums along dimension N
        4. keepdims=True preserves original number of dimensions
        5. Return result wrapped in Tensor

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> total = t.sum()
        >>> print(total.data)
        21.0
        >>> col_sum = t.sum(axis=0)
        >>> print(col_sum.data)
        [5. 7. 9.]

        HINT: np.sum(data, axis=axis, keepdims=keepdims) does all the work
        """
        ### BEGIN SOLUTION
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
        ### END SOLUTION

    def mean(self, axis=None, keepdims=False):
        """Compute mean of tensor along specified axis.

        TODO: Calculate average of elements along axis (same pattern as sum).

        APPROACH:
        1. Use np.mean with axis and keepdims parameters
        2. axis=None averages all elements
        3. axis=N averages along dimension N
        4. Return result wrapped in Tensor

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> avg = t.mean()
        >>> print(avg.data)
        3.5
        >>> col_mean = t.mean(axis=0)
        >>> print(col_mean.data)
        [2.5 3.5 4.5]

        HINT: Follow the same pattern as sum() but with np.mean
        """
        ### BEGIN SOLUTION
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
        ### END SOLUTION

    def max(self, axis=None, keepdims=False):
        """Find maximum values along specified axis.

        TODO: Find maximum element(s) along axis (same pattern as sum).

        APPROACH:
        1. Use np.max with axis and keepdims parameters
        2. axis=None finds maximum of all elements
        3. axis=N finds maximum along dimension N
        4. Return result wrapped in Tensor

        EXAMPLE:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> maximum = t.max()
        >>> print(maximum.data)
        6.0
        >>> row_max = t.max(axis=1)
        >>> print(row_max.data)
        [3. 6.]

        HINT: Follow the same pattern as sum() and mean() but with np.max
        """
        ### BEGIN SOLUTION
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Tensor Creation

This test validates our Tensor constructor works correctly with various data types and properly initializes all attributes.

**What we're testing**: Basic tensor creation and attribute setting
**Why it matters**: Foundation for all other operations - if creation fails, nothing works
**Expected**: Tensor wraps data correctly with proper attributes and consistent dtype
"""

# %% nbgrader={"grade": true, "grade_id": "test-tensor-creation", "locked": true, "points": 10}
def test_unit_tensor_creation():
    """🧪 Test Tensor creation with various data types."""
    print("🧪 Unit Test: Tensor Creation...")

    # Test scalar creation
    scalar = Tensor(5.0)
    assert scalar.data == 5.0
    assert scalar.shape == ()
    assert scalar.size == 1
    assert scalar.dtype == np.float32

    # Test vector creation
    vector = Tensor([1, 2, 3])
    assert np.array_equal(vector.data, np.array([1, 2, 3], dtype=np.float32))
    assert vector.shape == (3,)
    assert vector.size == 3

    # Test matrix creation
    matrix = Tensor([[1, 2], [3, 4]])
    assert np.array_equal(matrix.data, np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert matrix.shape == (2, 2)
    assert matrix.size == 4

    # Test 3D tensor creation
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor_3d.shape == (2, 2, 2)
    assert tensor_3d.size == 8

    print("✅ Tensor creation works correctly!")

if __name__ == "__main__":
    test_unit_tensor_creation()

# %% [markdown]
"""
## 🏗️ Element-wise Arithmetic Operations

Element-wise operations are the workhorses of neural network computation. They apply the same operation to corresponding elements in tensors, often with broadcasting to handle different shapes elegantly.

### Why Element-wise Operations Matter

In neural networks, element-wise operations appear everywhere:
- **Activation functions**: Apply ReLU, sigmoid to every element
- **Batch normalization**: Subtract mean, divide by std per element
- **Loss computation**: Compare predictions vs. targets element-wise
- **Gradient updates**: Add scaled gradients to parameters element-wise

### Element-wise Addition: The Foundation

Addition is the simplest and most fundamental operation. Understanding it deeply helps with all others.

```
Element-wise Addition Visual:
[1, 2, 3] + [4, 5, 6] = [1+4, 2+5, 3+6] = [5, 7, 9]

Matrix Addition:
[[1, 2]]   [[5, 6]]   [[1+5, 2+6]]   [[6, 8]]
[[3, 4]] + [[7, 8]] = [[3+7, 4+8]] = [[10, 12]]

Broadcasting Addition (Matrix + Vector):
[[1, 2]]   [10]   [[1, 2]]   [[10, 10]]   [[11, 12]]
[[3, 4]] + [20] = [[3, 4]] + [[20, 20]] = [[23, 24]]
     ↑      ↑           ↑         ↑            ↑
  (2,2)   (2,1)      (2,2)    broadcast    result

Broadcasting Rules:
1. Start from rightmost dimension
2. Dimensions must be equal OR one must be 1 OR one must be missing
3. Missing dimensions are assumed to be 1
```

**Key Insight**: Broadcasting makes tensors of different shapes compatible by automatically expanding dimensions. This is crucial for batch processing where you often add a single bias vector to an entire batch of data.

**Memory Efficiency**: Broadcasting doesn't actually create expanded copies in memory - NumPy computes results on-the-fly, saving memory.
"""

# %% [markdown]
"""
### Subtraction, Multiplication, and Division

These operations follow the same pattern as addition, working element-wise with broadcasting support. Each serves specific purposes in neural networks:

```
Element-wise Operations in Neural Networks:

┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Subtraction     │ Multiplication  │ Division        │ Use Cases       │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ [6,8] - [1,2]   │ [2,3] * [4,5]   │ [8,9] / [2,3]   │ • Gradient      │
│ = [5,6]         │ = [8,15]        │ = [4.0, 3.0]    │   computation   │
│                 │                 │                 │ • Normalization │
│ Center data:    │ Gate values:    │ Scale features: │ • Loss functions│
│ x - mean        │ x * mask        │ x / std         │ • Attention     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

Broadcasting with Scalars (very common in ML):
[1, 2, 3] * 2     = [2, 4, 6]      (scale all values)
[1, 2, 3] - 1     = [0, 1, 2]      (shift all values)
[2, 4, 6] / 2     = [1, 2, 3]      (normalize all values)

Real ML Example - Batch Normalization:
batch_data = [[1, 2], [3, 4], [5, 6]]  # Shape: (3, 2)
mean = [3, 4]                           # Shape: (2,)
std = [2, 2]                            # Shape: (2,)

# Normalize: (x - mean) / std
normalized = (batch_data - mean) / std
# Broadcasting: (3,2) - (2,) = (3,2), then (3,2) / (2,) = (3,2)
```

**Performance Note**: Element-wise operations are highly optimized in NumPy and run efficiently on modern CPUs with vectorization (SIMD instructions).
"""


# %% [markdown]
"""
### 🧪 Unit Test: Arithmetic Operations

This test validates our arithmetic operations work correctly with both tensor-tensor and tensor-scalar operations, including broadcasting behavior.

**What we're testing**: Addition, subtraction, multiplication, division with broadcasting
**Why it matters**: Foundation for neural network forward passes, batch processing, normalization
**Expected**: Operations work with both tensors and scalars, proper broadcasting alignment
"""

# %% nbgrader={"grade": true, "grade_id": "test-arithmetic", "locked": true, "points": 15}
def test_unit_arithmetic_operations():
    """🧪 Test arithmetic operations with broadcasting."""
    print("🧪 Unit Test: Arithmetic Operations...")

    # Test tensor + tensor
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    result = a + b
    assert np.array_equal(result.data, np.array([5, 7, 9], dtype=np.float32))

    # Test tensor + scalar (very common in ML)
    result = a + 10
    assert np.array_equal(result.data, np.array([11, 12, 13], dtype=np.float32))

    # Test broadcasting with different shapes (matrix + vector)
    matrix = Tensor([[1, 2], [3, 4]])
    vector = Tensor([10, 20])
    result = matrix + vector
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test subtraction (data centering)
    result = b - a
    assert np.array_equal(result.data, np.array([3, 3, 3], dtype=np.float32))

    # Test multiplication (scaling)
    result = a * 2
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))

    # Test division (normalization)
    result = b / 2
    assert np.array_equal(result.data, np.array([2.0, 2.5, 3.0], dtype=np.float32))

    # Test chaining operations (common in ML pipelines)
    normalized = (a - 2) / 2  # Center and scale
    expected = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    assert np.allclose(normalized.data, expected)

    print("✅ Arithmetic operations work correctly!")

if __name__ == "__main__":
    test_unit_arithmetic_operations()

# %% [markdown]
"""
## 🏗️ Matrix Multiplication: The Heart of Neural Networks

Matrix multiplication is fundamentally different from element-wise multiplication. It's the operation that gives neural networks their power to transform and combine information across features.

### Why Matrix Multiplication is Central to ML

Every neural network layer essentially performs matrix multiplication:

```
Linear Layer (the building block of neural networks):
Input Features × Weight Matrix = Output Features
    (N, D_in)   ×    (D_in, D_out)  =    (N, D_out)

Real Example - Image Classification:
Flattened Image × Hidden Weights = Hidden Features
  (32, 784)     ×    (784, 256)   =   (32, 256)
     ↑                   ↑              ↑
  32 images         784→256 transform  32 feature vectors
```

### Matrix Multiplication Visualization

```
Matrix Multiplication Process:
    A (2×3)      B (3×2)         C (2×2)
   ┌       ┐    ┌     ┐       ┌         ┐
   │ 1 2 3 │    │ 7 8 │       │ 1×7+2×9+3×1 │   ┌      ┐
   │       │ ×  │ 9 1 │  =    │             │ = │ 28 13│
   │ 4 5 6 │    │ 1 2 │       │ 4×7+5×9+6×1 │   │ 79 37│
   └       ┘    └     ┘       └             ┘   └      ┘

Computation Breakdown:
C[0,0] = A[0,:] · B[:,0] = [1,2,3] · [7,9,1] = 1×7 + 2×9 + 3×1 = 28
C[0,1] = A[0,:] · B[:,1] = [1,2,3] · [8,1,2] = 1×8 + 2×1 + 3×2 = 13
C[1,0] = A[1,:] · B[:,0] = [4,5,6] · [7,9,1] = 4×7 + 5×9 + 6×1 = 79
C[1,1] = A[1,:] · B[:,1] = [4,5,6] · [8,1,2] = 4×8 + 5×1 + 6×2 = 37

Key Rule: Inner dimensions must match!
A(m,n) @ B(n,p) = C(m,p)
     ↑     ↑
   these must be equal
```

### Computational Complexity and Performance

```
Computational Cost:
For C = A @ B where A is (M×K), B is (K×N):
- Multiplications: M × N × K
- Additions: M × N × (K-1) ≈ M × N × K
- Total FLOPs: ≈ 2 × M × N × K

Example: (1000×1000) @ (1000×1000)
- FLOPs: 2 × 1000³ = 2 billion operations
- On 1 GHz CPU: ~2 seconds if no optimization
- With optimized BLAS: ~0.1 seconds (20× speedup!)

Memory Access Pattern:
A: M×K (row-wise access)  ✓ Good cache locality
B: K×N (column-wise)      ✗ Poor cache locality
C: M×N (row-wise write)   ✓ Good cache locality

This is why optimized libraries like OpenBLAS, Intel MKL use:
- Blocking algorithms (process in cache-sized chunks)
- Vectorization (SIMD instructions)
- Parallelization (multiple cores)
```

### Neural Network Context

```
Multi-layer Neural Network:
Input (batch=32, features=784)
  ↓ W1: (784, 256)
Hidden1 (batch=32, features=256)
  ↓ W2: (256, 128)
Hidden2 (batch=32, features=128)
  ↓ W3: (128, 10)
Output (batch=32, classes=10)

Each arrow represents a matrix multiplication:
- Forward pass: 3 matrix multiplications
- Backward pass: 3 more matrix multiplications (with transposes)
- Total: 6 matrix mults per forward+backward pass

For training batch: 32 × (784×256 + 256×128 + 128×10) FLOPs
= 32 × (200,704 + 32,768 + 1,280) = 32 × 234,752 = 7.5M FLOPs per batch
```

This is why GPU acceleration matters - modern GPUs can perform thousands of these operations in parallel!
"""


# %% [markdown]
"""
### 🧪 Unit Test: Matrix Multiplication

This test validates matrix multiplication works correctly with proper shape checking and error handling.

**What we're testing**: Matrix multiplication with shape validation and edge cases
**Why it matters**: Core operation in neural networks (linear layers, attention mechanisms)
**Expected**: Correct results for valid shapes, clear error messages for invalid shapes
"""

# %% nbgrader={"grade": true, "grade_id": "test-matmul", "locked": true, "points": 15}
def test_unit_matrix_multiplication():
    """🧪 Test matrix multiplication operations."""
    print("🧪 Unit Test: Matrix Multiplication...")

    # Test 2×2 matrix multiplication (basic case)
    a = Tensor([[1, 2], [3, 4]])  # 2×2
    b = Tensor([[5, 6], [7, 8]])  # 2×2
    result = a.matmul(b)
    # Expected: [[1×5+2×7, 1×6+2×8], [3×5+4×7, 3×6+4×8]] = [[19, 22], [43, 50]]
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test rectangular matrices (common in neural networks)
    c = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3 (like batch_size=2, features=3)
    d = Tensor([[7, 8], [9, 10], [11, 12]])  # 3×2 (like features=3, outputs=2)
    result = c.matmul(d)
    # Expected: [[1×7+2×9+3×11, 1×8+2×10+3×12], [4×7+5×9+6×11, 4×8+5×10+6×12]]
    expected = np.array([[58, 64], [139, 154]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test matrix-vector multiplication (common in forward pass)
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3
    vector = Tensor([1, 2, 3])  # 3×1 (conceptually)
    result = matrix.matmul(vector)
    # Expected: [1×1+2×2+3×3, 4×1+5×2+6×3] = [14, 32]
    expected = np.array([14, 32], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test shape validation - should raise clear error
    try:
        incompatible_a = Tensor([[1, 2]])     # 1×2
        incompatible_b = Tensor([[1], [2], [3]])  # 3×1
        incompatible_a.matmul(incompatible_b)  # 1×2 @ 3×1 should fail (2 ≠ 3)
        assert False, "Should have raised ValueError for incompatible shapes"
    except ValueError as e:
        assert "Inner dimensions must match" in str(e)
        assert "2 ≠ 3" in str(e)  # Should show specific dimensions

    print("✅ Matrix multiplication works correctly!")

if __name__ == "__main__":
    test_unit_matrix_multiplication()

# %% [markdown]
"""
## 🏗️ Shape Manipulation: Reshape and Transpose

Neural networks constantly change tensor shapes to match layer requirements. Understanding these operations is crucial for data flow through networks.

### Why Shape Manipulation Matters

Real neural networks require constant shape changes:

```
CNN Data Flow Example:
Input Image: (32, 3, 224, 224)     # batch, channels, height, width
     ↓ Convolutional layers
Feature Maps: (32, 512, 7, 7)      # batch, features, spatial
     ↓ Global Average Pool
Pooled: (32, 512, 1, 1)            # batch, features, 1, 1
     ↓ Flatten for classifier
Flattened: (32, 512)               # batch, features
     ↓ Linear classifier
Output: (32, 1000)                 # batch, classes

Each ↓ involves reshape or view operations!
```

### Reshape: Changing Interpretation of the Same Data

```
Reshaping (changing dimensions without changing data):
Original: [1, 2, 3, 4, 5, 6]  (shape: (6,))
         ↓ reshape(2, 3)
Result:  [[1, 2, 3],          (shape: (2, 3))
          [4, 5, 6]]

Memory Layout (unchanged):
Before: [1][2][3][4][5][6]
After:  [1][2][3][4][5][6]  ← Same memory, different interpretation

Key Insight: Reshape is O(1) operation - no data copying!
Just changes how we interpret the memory layout.

Common ML Reshapes:
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Flatten for MLP     │ Unflatten for CNN   │ Batch Dimension     │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ (N,H,W,C) → (N,H×W×C) │ (N,D) → (N,H,W,C)   │ (H,W) → (1,H,W)     │
│ Images to vectors   │ Vectors to images   │ Add batch dimension │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

### Transpose: Swapping Dimensions

```
Transposing (swapping dimensions - data rearrangement):
Original: [[1, 2, 3],    (shape: (2, 3))
           [4, 5, 6]]
         ↓ transpose()
Result:  [[1, 4],        (shape: (3, 2))
          [2, 5],
          [3, 6]]

Memory Layout (rearranged):
Before: [1][2][3][4][5][6]
After:  [1][4][2][5][3][6]  ← Data actually moves in memory

Key Insight: Transpose involves data movement - more expensive than reshape.

Neural Network Usage:
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Weight Matrices     │ Attention Mechanism │ Gradient Computation│
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Forward: X @ W      │ Q @ K^T attention   │ ∂L/∂W = X^T @ ∂L/∂Y │
│ Backward: X @ W^T   │ scores              │                     │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

### Performance Implications

```
Operation Performance (for 1000×1000 matrix):
┌─────────────────┬──────────────┬─────────────────┬─────────────────┐
│ Operation       │ Time         │ Memory Access   │ Cache Behavior  │
├─────────────────┼──────────────┼─────────────────┼─────────────────┤
│ reshape()       │ ~0.001 ms    │ No data copy    │ No cache impact │
│ transpose()     │ ~10 ms       │ Full data copy  │ Poor locality   │
│ view() (future) │ ~0.001 ms    │ No data copy    │ No cache impact │
└─────────────────┴──────────────┴─────────────────┴─────────────────┘

Why transpose() is slower:
- Must rearrange data in memory
- Poor cache locality (accessing columns)
- Can't be parallelized easily
```

This is why frameworks like PyTorch often use "lazy" transpose operations that defer the actual data movement until necessary.
"""


# %% [markdown]
"""
### 🧪 Unit Test: Shape Manipulation

This test validates reshape and transpose operations work correctly with validation and edge cases.

**What we're testing**: Reshape and transpose operations with proper error handling
**Why it matters**: Essential for data flow in neural networks, CNN/RNN architectures
**Expected**: Correct shape changes, proper error handling for invalid operations
"""

# %% nbgrader={"grade": true, "grade_id": "test-shape-ops", "locked": true, "points": 15}
def test_unit_shape_manipulation():
    """🧪 Test reshape and transpose operations."""
    print("🧪 Unit Test: Shape Manipulation...")

    # Test basic reshape (flatten → matrix)
    tensor = Tensor([1, 2, 3, 4, 5, 6])  # Shape: (6,)
    reshaped = tensor.reshape(2, 3)      # Shape: (2, 3)
    assert reshaped.shape == (2, 3)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert np.array_equal(reshaped.data, expected)

    # Test reshape with tuple (alternative calling style)
    reshaped2 = tensor.reshape((3, 2))   # Shape: (3, 2)
    assert reshaped2.shape == (3, 2)
    expected2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    assert np.array_equal(reshaped2.data, expected2)

    # Test reshape with -1 (automatic dimension inference)
    auto_reshaped = tensor.reshape(2, -1)  # Should infer -1 as 3
    assert auto_reshaped.shape == (2, 3)

    # Test reshape validation - should raise error for incompatible sizes
    try:
        tensor.reshape(2, 2)  # 6 elements can't fit in 2×2=4
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Total elements must match" in str(e)
        assert "6 ≠ 4" in str(e)

    # Test matrix transpose (most common case)
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    transposed = matrix.transpose()          # (3, 2)
    assert transposed.shape == (3, 2)
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    assert np.array_equal(transposed.data, expected)

    # Test 1D transpose (should be identity)
    vector = Tensor([1, 2, 3])
    vector_t = vector.transpose()
    assert np.array_equal(vector.data, vector_t.data)

    # Test specific dimension transpose
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    swapped = tensor_3d.transpose(0, 2)  # Swap first and last dimensions
    assert swapped.shape == (2, 2, 2)  # Same shape but data rearranged

    # Test neural network reshape pattern (flatten for MLP)
    batch_images = Tensor(np.random.rand(2, 3, 4))  # (batch=2, height=3, width=4)
    flattened = batch_images.reshape(2, -1)  # (batch=2, features=12)
    assert flattened.shape == (2, 12)

    print("✅ Shape manipulation works correctly!")

if __name__ == "__main__":
    test_unit_shape_manipulation()

# %% [markdown]
"""
## 🏗️ Reduction Operations: Aggregating Information

Reduction operations collapse dimensions by aggregating data, which is essential for computing statistics, losses, and preparing data for different layers.

### Why Reductions are Crucial in ML

Reduction operations appear throughout neural networks:

```
Common ML Reduction Patterns:

┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Loss Computation    │ Batch Normalization │ Global Pooling      │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Per-sample losses → │ Batch statistics →  │ Feature maps →      │
│ Single batch loss   │ Normalization       │ Single features     │
│                     │                     │                     │
│ losses.mean()       │ batch.mean(axis=0)  │ fmaps.mean(axis=(2,3))│
│ (N,) → scalar       │ (N,D) → (D,)        │ (N,C,H,W) → (N,C)   │
└─────────────────────┴─────────────────────┴─────────────────────┘

Real Examples:
• Cross-entropy loss: -log(predictions).mean()  [average over batch]
• Batch norm: (x - x.mean()) / x.std()          [normalize each feature]
• Global avg pool: features.mean(dim=(2,3))     [spatial → scalar per channel]
```

### Understanding Axis Operations

```
Visual Axis Understanding:
Matrix:     [[1, 2, 3],      All reductions operate on this data
             [4, 5, 6]]      Shape: (2, 3)

        axis=0 (↓)
       ┌─────────┐
axis=1 │ 1  2  3 │ →  axis=1 reduces across columns (→)
   (→) │ 4  5  6 │ →  Result shape: (2,) [one value per row]
       └─────────┘
         ↓ ↓ ↓
      axis=0 reduces down rows (↓)
      Result shape: (3,) [one value per column]

Reduction Results:
├─ .sum() → 21                    (sum all: 1+2+3+4+5+6)
├─ .sum(axis=0) → [5, 7, 9]       (sum columns: [1+4, 2+5, 3+6])
├─ .sum(axis=1) → [6, 15]         (sum rows: [1+2+3, 4+5+6])
├─ .mean() → 3.5                  (average all: 21/6)
├─ .mean(axis=0) → [2.5, 3.5, 4.5] (average columns)
└─ .max() → 6                     (maximum element)

3D Tensor Example (batch, height, width):
data.shape = (2, 3, 4)  # 2 samples, 3×4 images
│
├─ .sum(axis=0) → (3, 4)    # Sum across batch dimension
├─ .sum(axis=1) → (2, 4)    # Sum across height dimension
├─ .sum(axis=2) → (2, 3)    # Sum across width dimension
└─ .sum(axis=(1,2)) → (2,)  # Sum across both spatial dims (global pool)
```

### Memory and Performance Considerations

```
Reduction Performance:
┌─────────────────┬──────────────┬─────────────────┬─────────────────┐
│ Operation       │ Time Complex │ Memory Access   │ Cache Behavior  │
├─────────────────┼──────────────┼─────────────────┼─────────────────┤
│ .sum()          │ O(N)         │ Sequential read │ Excellent       │
│ .sum(axis=0)    │ O(N)         │ Column access   │ Poor (strided)  │
│ .sum(axis=1)    │ O(N)         │ Row access      │ Excellent       │
│ .mean()         │ O(N)         │ Sequential read │ Excellent       │
│ .max()          │ O(N)         │ Sequential read │ Excellent       │
└─────────────────┴──────────────┴─────────────────┴─────────────────┘

Why axis=0 is slower:
- Accesses elements with large strides
- Poor cache locality (jumping rows)
- Less vectorization-friendly

Optimization strategies:
- Prefer axis=-1 operations when possible
- Use keepdims=True to maintain shape for broadcasting
- Consider reshaping before reduction for better cache behavior
```
"""


# %% [markdown]
"""
### 🧪 Unit Test: Reduction Operations

This test validates reduction operations work correctly with axis control and maintain proper shapes.

**What we're testing**: Sum, mean, max operations with axis parameter and keepdims
**Why it matters**: Essential for loss computation, batch processing, and pooling operations
**Expected**: Correct reduction along specified axes with proper shape handling
"""

# %% nbgrader={"grade": true, "grade_id": "test-reductions", "locked": true, "points": 10}
def test_unit_reduction_operations():
    """🧪 Test reduction operations."""
    print("🧪 Unit Test: Reduction Operations...")

    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)

    # Test sum all elements (common for loss computation)
    total = matrix.sum()
    assert total.data == 21.0  # 1+2+3+4+5+6
    assert total.shape == ()   # Scalar result

    # Test sum along axis 0 (columns) - batch dimension reduction
    col_sum = matrix.sum(axis=0)
    expected_col = np.array([5, 7, 9], dtype=np.float32)  # [1+4, 2+5, 3+6]
    assert np.array_equal(col_sum.data, expected_col)
    assert col_sum.shape == (3,)

    # Test sum along axis 1 (rows) - feature dimension reduction
    row_sum = matrix.sum(axis=1)
    expected_row = np.array([6, 15], dtype=np.float32)  # [1+2+3, 4+5+6]
    assert np.array_equal(row_sum.data, expected_row)
    assert row_sum.shape == (2,)

    # Test mean (average loss computation)
    avg = matrix.mean()
    assert np.isclose(avg.data, 3.5)  # 21/6
    assert avg.shape == ()

    # Test mean along axis (batch normalization pattern)
    col_mean = matrix.mean(axis=0)
    expected_mean = np.array([2.5, 3.5, 4.5], dtype=np.float32)  # [5/2, 7/2, 9/2]
    assert np.allclose(col_mean.data, expected_mean)

    # Test max (finding best predictions)
    maximum = matrix.max()
    assert maximum.data == 6.0
    assert maximum.shape == ()

    # Test max along axis (argmax-like operation)
    row_max = matrix.max(axis=1)
    expected_max = np.array([3, 6], dtype=np.float32)  # [max(1,2,3), max(4,5,6)]
    assert np.array_equal(row_max.data, expected_max)

    # Test keepdims (important for broadcasting)
    sum_keepdims = matrix.sum(axis=1, keepdims=True)
    assert sum_keepdims.shape == (2, 1)  # Maintains 2D shape
    expected_keepdims = np.array([[6], [15]], dtype=np.float32)
    assert np.array_equal(sum_keepdims.data, expected_keepdims)

    # Test 3D reduction (simulating global average pooling)
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    spatial_mean = tensor_3d.mean(axis=(1, 2))  # Average across spatial dimensions
    assert spatial_mean.shape == (2,)  # One value per batch item

    print("✅ Reduction operations work correctly!")

if __name__ == "__main__":
    test_unit_reduction_operations()

# %% [markdown]
"""
## 📊 Systems Analysis: Memory Layout and Performance

Let's understand ONE key systems concept: **memory layout and cache behavior**.

This single analysis reveals why certain operations are fast while others are slow, and why framework designers make specific architectural choices.
"""

# %%
def analyze_memory_layout():
    """📊 Demonstrate cache effects with row vs column access patterns."""
    print("📊 Analyzing Memory Access Patterns...")
    print("=" * 60)

    # Create a moderately-sized matrix (large enough to show cache effects)
    size = 2000
    matrix = Tensor(np.random.rand(size, size))

    import time

    print(f"\nTesting with {size}×{size} matrix ({matrix.size * BYTES_PER_FLOAT32 / MB_TO_BYTES:.1f} MB)")
    print("-" * 60)

    # Test 1: Row-wise access (cache-friendly)
    # Memory layout: [row0][row1][row2]... stored contiguously
    print("\n🔬 Test 1: Row-wise Access (Cache-Friendly)")
    start = time.time()
    row_sums = []
    for i in range(size):
        row_sum = matrix.data[i, :].sum()  # Access entire row sequentially
        row_sums.append(row_sum)
    row_time = time.time() - start
    print(f"   Time: {row_time*1000:.1f}ms")
    print(f"   Access pattern: Sequential (follows memory layout)")

    # Test 2: Column-wise access (cache-unfriendly)
    # Must jump between rows, poor spatial locality
    print("\n🔬 Test 2: Column-wise Access (Cache-Unfriendly)")
    start = time.time()
    col_sums = []
    for j in range(size):
        col_sum = matrix.data[:, j].sum()  # Access entire column with large strides
        col_sums.append(col_sum)
    col_time = time.time() - start
    print(f"   Time: {col_time*1000:.1f}ms")
    print(f"   Access pattern: Strided (jumps {size * BYTES_PER_FLOAT32} bytes per element)")

    # Calculate slowdown
    slowdown = col_time / row_time
    print("\n" + "=" * 60)
    print(f"📊 PERFORMANCE IMPACT:")
    print(f"   Slowdown factor: {slowdown:.2f}× ({col_time/row_time:.1f}× slower)")
    print(f"   Cache misses cause {(slowdown-1)*100:.0f}% performance loss")

    # Educational insights
    print("\n💡 KEY INSIGHTS:")
    print(f"   1. Memory layout matters: Row-major (C-style) storage is sequential")
    print(f"   2. Cache lines are ~64 bytes: Row access loads nearby elements \"for free\"")
    print(f"   3. Column access misses cache: Must reload from DRAM every time")
    print(f"   4. This is O(n) algorithm but {slowdown:.1f}× different wall-clock time!")

    print("\n🚀 REAL-WORLD IMPLICATIONS:")
    print(f"   • CNNs use NCHW format (channels sequential) for cache efficiency")
    print(f"   • Matrix multiplication optimized with blocking (tile into cache-sized chunks)")
    print(f"   • Transpose is expensive ({slowdown:.1f}×) because it changes memory layout")
    print(f"   • This is why GPU frameworks obsess over memory coalescing")

    print("\n" + "=" * 60)

# Run the systems analysis
if __name__ == "__main__":
    analyze_memory_layout()


# %% [markdown]
"""
## 🔧 Integration: Bringing It Together

Let's test how our Tensor operations work together in realistic scenarios that mirror neural network computations. This integration demonstrates that our individual operations combine correctly for complex ML workflows.

### Neural Network Layer Simulation

The fundamental building block of neural networks is the linear transformation: **y = xW + b**

```
Linear Layer Forward Pass: y = xW + b

Input Features → Weight Matrix → Matrix Multiply → Add Bias → Output Features
  (batch, in)   (in, out)        (batch, out)     (batch, out)   (batch, out)

Step-by-Step Breakdown:
1. Input:   X shape (batch_size, input_features)
2. Weight:  W shape (input_features, output_features)
3. Matmul:  XW shape (batch_size, output_features)
4. Bias:    b shape (output_features,)
5. Result:  XW + b shape (batch_size, output_features)

Example Flow:
Input: [[1, 2, 3],    Weight: [[0.1, 0.2],    Bias: [0.1, 0.2]
        [4, 5, 6]]            [0.3, 0.4],
       (2, 3)                 [0.5, 0.6]]
                             (3, 2)

Step 1: Matrix Multiply
[[1, 2, 3]] @ [[0.1, 0.2]] = [[1×0.1+2×0.3+3×0.5, 1×0.2+2×0.4+3×0.6]]
[[4, 5, 6]]   [[0.3, 0.4]]   [[4×0.1+5×0.3+6×0.5, 4×0.2+5×0.4+6×0.6]]
              [[0.5, 0.6]]
                           = [[1.6, 2.6],
                              [4.9, 6.8]]

Step 2: Add Bias (Broadcasting)
[[1.6, 2.6]] + [0.1, 0.2] = [[1.7, 2.8],
 [4.9, 6.8]]                 [5.0, 7.0]]

This is the foundation of every neural network layer!
```

### Why This Integration Matters

This simulation shows how our basic operations combine to create the computational building blocks of neural networks:

- **Matrix Multiplication**: Transforms input features into new feature space
- **Broadcasting Addition**: Applies learned biases efficiently across batches
- **Shape Handling**: Ensures data flows correctly through layers
- **Memory Management**: Creates new tensors without corrupting inputs

Every layer in a neural network - from simple MLPs to complex transformers - uses this same pattern.
"""


# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_tensor_creation()
    test_unit_arithmetic_operations()
    test_unit_matrix_multiplication()
    test_unit_shape_manipulation()
    test_unit_reduction_operations()

    print("\nRunning integration scenarios...")

    # Test realistic neural network computation
    print("🧪 Integration Test: Two-Layer Neural Network...")

    # Create input data (2 samples, 3 features)
    x = Tensor([[1, 2, 3], [4, 5, 6]])

    # First layer: 3 inputs → 4 hidden units
    W1 = Tensor([[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.6, 0.7, 0.8],
                 [0.9, 1.0, 1.1, 1.2]])
    b1 = Tensor([0.1, 0.2, 0.3, 0.4])

    # Forward pass: hidden = xW1 + b1
    hidden = x.matmul(W1) + b1
    assert hidden.shape == (2, 4), f"Expected (2, 4), got {hidden.shape}"

    # Second layer: 4 hidden → 2 outputs
    W2 = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    b2 = Tensor([0.1, 0.2])

    # Output layer: output = hiddenW2 + b2
    output = hidden.matmul(W2) + b2
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    # Verify data flows correctly (no NaN, reasonable values)
    assert not np.isnan(output.data).any(), "Output contains NaN values"
    assert np.isfinite(output.data).all(), "Output contains infinite values"

    print("✅ Two-layer neural network computation works!")

    # Test complex shape manipulations
    print("🧪 Integration Test: Complex Shape Operations...")
    data = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # Reshape to 3D tensor (simulating batch processing)
    tensor_3d = data.reshape(2, 2, 3)  # (batch=2, height=2, width=3)
    assert tensor_3d.shape == (2, 2, 3)

    # Global average pooling simulation
    pooled = tensor_3d.mean(axis=(1, 2))  # Average across spatial dimensions
    assert pooled.shape == (2,), f"Expected (2,), got {pooled.shape}"

    # Flatten for MLP
    flattened = tensor_3d.reshape(2, -1)  # (batch, features)
    assert flattened.shape == (2, 6)

    # Transpose for different operations
    transposed = tensor_3d.transpose()  # Should transpose last two dims
    assert transposed.shape == (2, 3, 2)

    print("✅ Complex shape operations work!")

    # Test broadcasting edge cases
    print("🧪 Integration Test: Broadcasting Edge Cases...")

    # Scalar broadcasting
    scalar = Tensor(5.0)
    vector = Tensor([1, 2, 3])
    result = scalar + vector  # Should broadcast scalar to vector shape
    expected = np.array([6, 7, 8], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Matrix + vector broadcasting
    matrix = Tensor([[1, 2], [3, 4]])
    vec = Tensor([10, 20])
    result = matrix + vec
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    print("✅ Broadcasting edge cases work!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 01_tensor")

# Run comprehensive module test
if __name__ == "__main__":
    test_module()


# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of tensor operations and their systems implications:

### 1. Memory Layout and Cache Performance
**Question**: How does row-major vs column-major storage affect cache performance in tensor operations?

**Consider**:
- What happens when you access matrix elements sequentially vs. with large strides?
- Why did our analysis show column-wise access being ~2-3× slower than row-wise?
- How would this affect the design of a convolutional neural network's memory layout?

**Real-world context**: PyTorch uses NCHW (batch, channels, height, width) format specifically because accessing channels sequentially has better cache locality than NHWC format.

---

### 2. Batch Processing and Scaling
**Question**: If you double the batch size in a neural network, what happens to memory usage? What about computation time?

**Consider**:
- A linear layer with input (batch, features): y = xW + b
- Memory for: input tensor, weight matrix, output tensor, intermediate results
- How does matrix multiplication time scale with batch size?

**Think about**:
- If (32, 784) @ (784, 256) takes 10ms, how long does (64, 784) @ (784, 256) take?
- Does doubling batch size double memory usage? Why or why not?
- What are the trade-offs between large and small batch sizes?

---

### 3. Data Type Precision and Memory
**Question**: What's the memory difference between float64 and float32 for a (1000, 1000) tensor? When would you choose each?

**Calculate**:
- float64: 8 bytes per element
- float32: 4 bytes per element
- Total elements in (1000, 1000): ___________
- Memory difference: ___________

**Trade-offs to consider**:
- Training accuracy vs. memory consumption
- GPU memory limits (often 8-16GB for consumer GPUs)
- Numerical stability in gradient computation
- Inference speed on mobile devices

---

### 4. Production Scale: Memory Requirements
**Question**: A GPT-3-scale model has 175 billion parameters. How much RAM is needed just to store the weights in float32? What about with an optimizer like Adam?

**Calculate**:
- Parameters: 175 × 10^9
- Bytes per float32: 4
- Weight memory: ___________GB

**Additional memory for Adam optimizer**:
- Adam stores: parameters, gradients, first moment (m), second moment (v)
- Total multiplier: 4× the parameter count
- Total with Adam: ___________GB

**Real-world implications**:
- Why do we need 8× A100 GPUs (40GB each) for training?
- What is mixed-precision training (float16/bfloat16)?
- How does gradient checkpointing help?

---

### 5. Hardware Awareness: GPU Efficiency
**Question**: Why do GPUs strongly prefer operations on large tensors over many small ones?

**Consider these scenarios**:
- **Scenario A**: 1000 separate (10, 10) matrix multiplications
- **Scenario B**: 1 batched (1000, 10, 10) matrix multiplication

**Think about**:
- GPU kernel launch overhead (~5-10 microseconds per launch)
- Thread parallelism utilization (GPUs have 1000s of cores)
- Memory transfer costs (CPU→GPU has ~10GB/s bandwidth, GPU memory has ~900GB/s)
- When is the GPU actually doing computation vs. waiting?

**Design principle**: Batch operations together to amortize overhead and maximize parallelism.

---

### Bonus Challenge: Optimization Analysis

**Scenario**: You're implementing a custom activation function that will be applied to every element in a tensor. You have two implementation choices:

**Option A**: Python loop over each element
```python
def custom_activation(tensor):
    result = np.empty_like(tensor.data)
    for i in range(tensor.data.size):
        result.flat[i] = complex_math_function(tensor.data.flat[i])
    return Tensor(result)
```

**Option B**: NumPy vectorized operation
```python
def custom_activation(tensor):
    return Tensor(complex_math_function(tensor.data))
```

**Questions**:
1. For a (1000, 1000) tensor, estimate the speedup of Option B vs Option A
2. Why is vectorization faster even though both are O(n) operations?
3. What if the tensor is tiny (10, 10) - does the answer change?
4. How would this change if we move to GPU computation?

**Key insight**: Algorithmic complexity (Big-O) doesn't tell the whole performance story. Constant factors from vectorization, cache behavior, and parallelism dominate in practice.
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Your Tensor Works Like NumPy

**What you built:** A complete Tensor class with arithmetic operations and matrix multiplication.

**Why it matters:** Your Tensor is the foundation of everything to come. Every neural network
operation—from simple addition to complex attention mechanisms—will use this class. The fact
that it works exactly like NumPy means you've built something production-ready.

Your Tensor is ready for machine learning operations.
Every operation you just implemented will be called millions of times during training!
"""

# %%
def demo_tensor():
    """🎯 See your Tensor work just like NumPy."""
    print("🎯 AHA MOMENT: Your Tensor Works Like NumPy")
    print("=" * 45)

    # Create tensors
    a = Tensor(np.array([1, 2, 3]))
    b = Tensor(np.array([4, 5, 6]))

    # Tensor operations
    tensor_sum = a + b
    tensor_prod = a * b

    # NumPy equivalents
    np_sum = np.array([1, 2, 3]) + np.array([4, 5, 6])
    np_prod = np.array([1, 2, 3]) * np.array([4, 5, 6])

    print(f"Tensor a + b: {tensor_sum.data}")
    print(f"NumPy  a + b: {np_sum}")
    print(f"Match: {np.allclose(tensor_sum.data, np_sum)}")

    print(f"\nTensor a * b: {tensor_prod.data}")
    print(f"NumPy  a * b: {np_prod}")
    print(f"Match: {np.allclose(tensor_prod.data, np_prod)}")

    print("\n✨ Your Tensor is NumPy-compatible—ready for ML!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_tensor()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Tensor Foundation

Congratulations! You've built the foundational Tensor class that powers all machine learning operations!

### Key Accomplishments
- **Built a complete Tensor class** with arithmetic operations, matrix multiplication, and shape manipulation
- **Implemented broadcasting semantics** that match NumPy for automatic shape alignment
- **Created reduction operations** (sum, mean, max) for loss computation and pooling
- **Added comprehensive ASCII diagrams** showing tensor operations visually
- **All tests pass ✅** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory scaling**: Matrix operations create new tensors (3× memory during computation)
- **Broadcasting efficiency**: NumPy's automatic shape alignment vs. explicit operations
- **Cache behavior**: Row-wise access is faster than column-wise due to memory layout
- **Shape validation trade-offs**: Clear errors vs. performance in tight loops

Export with: `tito module complete 01_tensor`
"""
