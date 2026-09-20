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
r"""
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

**Learning Side:** You work in `modules/01_tensor/tensor.ipynb`
**Building Side:** Code exports to tinytorch.core.tensor

```python
from tinytorch.core.tensor import Tensor   # every later module starts here
```

**Why this matters:**
- **Learning:** Complete tensor system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.Tensor with all core operations together
- **Consistency:** All tensor operations and data manipulation in core.tensor
- **Integration:** Foundation that every other module will build upon
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

**Prerequisites**: NONE - This is the foundation module

**External Dependencies**:
- `numpy` (for array operations and numerical computing)

**TinyTorch Dependencies**: NONE

This module has NO TinyTorch dependencies.
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

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp core.tensor
#| export

import numpy as np
rng = np.random.default_rng(7)

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
r"""
## 💡 Introduction: What is a Tensor?

A tensor is a multi-dimensional array that serves as the fundamental data structure in machine learning. Think of it as a universal container that can hold data in different dimensions:

| Dimension | Concept | Coordinate Shape | Concrete Value Example | ML Systems Role |
| :--- | :--- | :--- | :--- | :--- |
| **0D** | Scalar | `()` | `5.0` | Loss value, learning rate, temperature |
| **1D** | Vector | `(D,)` | `[1.0, 2.0, 3.0]` | Bias parameters, token sequence |
| **2D** | Matrix | `(N, D)` | `[[1, 2], [3, 4]]` | Dense layer weights, batched embeddings |
| **3D** | 3-Tensor | `(B, S, D)` | `(Batch, SeqLen, D_model)` | Batched token sequences in Transformers |
| **4D** | 4-Tensor | `(B, C, H, W)` | `(Batch, Channels, Height, Width)` | Batched feature maps in Convolutional Networks |

In computation, tensors flow through operations like transformations along a processing pipeline:

$$\underbrace{\mathbf{X}}_{\text{Raw Input } (N, D_0)} \xrightarrow{\mathbf{W}_1} \underbrace{\mathbf{H}_1}_{\text{Features } (N, D_1)} \xrightarrow{\mathbf{W}_2} \underbrace{\mathbf{H}_2}_{\text{Features } (N, D_2)} \xrightarrow{\mathbf{W}_3} \underbrace{\hat{\mathbf{Y}}}_{\text{Predictions } (N, C)}$$

From simple statistics to large-scale scientific computing, tensors are the universal data container. Understanding tensors means understanding the foundation of numerical computation.

### Why Tensors Matter in ML Systems

In production ML systems, tensors carry more than just data — they carry operation history, memory layout information, and execution context:

$$\text{Disk / Storage} \xrightarrow{\text{I/O Ingestion}} \text{NumPy Buffer} \xrightarrow{\text{Class Wrap}} \text{TinyTorch Tensor} \xrightarrow{\text{Kernel Execution}} \text{Engine Output}$$

**Key Insight**: Tensors bridge the gap between mathematical concepts and efficient computation on modern hardware.
"""

# %% [markdown]
r"""
## 📐 Foundations: Mathematical Background

### Core Operations We'll Implement

Our Tensor class will support all fundamental operations that neural networks need:

| Category | Operations | Syntax | Return Shape / Behavior |
| :--- | :--- | :--- | :--- |
| **Element-wise** | Addition, Subtraction, Multiplication, Division | `a + b`, `a - b`, `a * b`, `a / b` | Broadcasts shapes; preserves rank |
| **Linear Algebra** | Matrix Multiplication, Transpose | `a @ b`, `a.matmul(b)`, `a.transpose()` | Inner dimension contract: $(M, K) \times (K, N) \to (M, N)$ |
| **Shape & View** | Reshape, View, Slicing, Masked Fill | `a.reshape(...)`, `a.view(...)`, `a[i]`, `a.masked_fill(...)` | Modifies coordinate strides or view geometry |
| **Reductions** | Sum, Mean, Maximum | `a.sum(axis=...)`, `a.mean(...)`, `a.max(...)` | Collapses specified axis $(N, D) \xrightarrow{\text{axis}=0} (D,)$ |

### Broadcasting: Making Tensors Work Together

Broadcasting automatically aligns tensors of different shapes for operations:

$$\begin{aligned}
\mathbf{\text{Scalar } + \text{ Vector:}} \quad & 5 + \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \xrightarrow{\text{broadcast}} \begin{bmatrix} 5 & 5 & 5 \end{bmatrix} + \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} = \begin{bmatrix} 6 & 7 & 8 \end{bmatrix} \\
\mathbf{\text{Matrix } + \text{ Row Vector:}} \quad & \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 10 & 20 \end{bmatrix} \xrightarrow{\text{broadcast}} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 10 & 20 \\ 10 & 20 \end{bmatrix} = \begin{bmatrix} 11 & 22 \\ 13 & 24 \end{bmatrix} \\
\mathbf{\text{Matrix } + \text{ Column Vector:}} \quad & \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 10 \\ 20 \end{bmatrix} \xrightarrow{\text{broadcast}} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 10 & 10 \\ 20 & 20 \end{bmatrix} = \begin{bmatrix} 11 & 12 \\ 23 & 24 \end{bmatrix}
\end{aligned}$$

**Memory Layout**: NumPy uses row-major (C-style) storage where elements are stored row by row in contiguous linear memory for cache efficiency:

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3} \implies \text{Linear Storage: } \underbrace{\begin{array}{|c|c|c|} \hline 1 & 2 & 3 \\ \hline \end{array}}_{\text{Row 0: offsets } 0, 4, 8 \text{ B}} \quad \underbrace{\begin{array}{|c|c|c|} \hline 4 & 5 & 6 \\ \hline \end{array}}_{\text{Row 1: offsets } 12, 16, 20 \text{ B}}$$

| Traversal Pattern | Memory Footprint | Hardware Cache Behavior |
| :--- | :--- | :--- |
| **Sequential (Row-wise)** | `[1, 2, 3]` $\to$ continuous 64-byte cache line | 🟢 **Cache Hit**: Full line utilized |
| **Strided (Column-wise)** | `[1, 4]` $\to$ jumps memory stride per element | 🔴 **Cache Miss**: Potential bandwidth thrashing |

The 2×3 example above is far too small to miss: all six float32 values fit in a single 64-byte cache line. The pattern matters once a row is longer than a cache line, which is the case for every matrix you will use for real work. Algorithms that access data sequentially run faster than those that stride through memory; the Systems Analysis section measures this on a 2000×2000 matrix.
"""

# %% [markdown]
r"""
## 🏗️ Implementation: Building Tensor Foundation

Let's build our Tensor class step by step, testing each component as we go.

### Tensor Class Architecture

![Tensor Class Structure](tensor_class_structure.svg)

This clean design focuses on what tensors fundamentally do: store numerical data and route every operation through one shared mechanism, `Function.apply`.
"""

# %% [markdown]
r"""
### Operations as Objects

Every Tensor method below hands its work to an operation class instead of computing the result itself. `a + b` calls `Tensor.__add__`, which calls `Add.apply(a, b)`. `apply` unwraps the Tensors to NumPy arrays, runs the operation's `forward`, and wraps the result in a new Tensor.

This is how PyTorch is built (`torch.autograd.Function`), and the reason is Module 06, which will add automatic differentiation: `apply` is the one place that has to remember which operation produced which Tensor. Nothing in the Tensor class changes. In this module you write the `forward` half of every operation, in the cells that follow each section's explanation; `backward` raises until Module 06, where you write the other half of the same classes. The Tensor class refers to these operations by name, so run this cell before running any test below.
"""

# %% nbgrader={"grade": false, "grade_id": "function-base", "solution": false}
#| export
def _as_tensor(x):
    """Wrap a scalar or array as a Tensor; pass a Tensor through unchanged."""
    return x if isinstance(x, Tensor) else Tensor(x)


class Function:
    """
    Base class for every operation the framework can differentiate.

    An operation has two halves. forward() computes the result from NumPy
    arrays and is written in this module. backward() turns the gradient of the
    output into gradients for the inputs and is written in Module 06. apply()
    is the one method they share: it unwraps the input Tensors, runs forward(),
    and wraps the result. Module 06 teaches apply() to also remember the
    operation so that gradients can flow back through it.

    **Example Usage:**
    ```python
    class Double(Function):
        def forward(self, a):
            return a * 2

    y = Double.apply(x)   # x.data doubled, wrapped in a new Tensor
    ```
    """

    def __init__(self, *inputs, **params):
        self.inputs = inputs                 # the Tensors this operation consumed
        for name, value in params.items():   # axis, shape, key, ... for the operation
            setattr(self, name, value)

    def forward(self, *arrays):
        """Compute the result array from the input arrays. Each operation implements this."""
        raise NotImplementedError(f"{type(self).__name__}.forward is not implemented")

    def backward(self, grad_output):
        """Return one gradient array per input. Module 06 implements this."""
        raise NotImplementedError(f"Module 06 implements {type(self).__name__}.backward")

    @classmethod
    def apply(cls, *inputs, **params):
        """Run the operation on Tensors and return a new Tensor."""
        node = cls(*inputs, **params)
        arrays = [t.data for t in inputs]
        return Tensor(node.forward(*arrays))

# %% [markdown]
r"""
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
Tensor wraps with: shape=(2,3), size=6, dtype=float32
```

**Key Design Principle**: Our Tensor is a wrapper around NumPy arrays that adds ML-specific functionality. We leverage NumPy's battle-tested memory management and computation kernels while adding the operation chaining needed for machine learning.

**Why This Approach?**
- **Performance**: NumPy's C implementations are highly optimized
- **Compatibility**: Easy integration with scientific Python ecosystem
- **Memory Discipline**: `Function.apply` wraps each result in a fresh Tensor with independent storage. Operations leave their inputs unchanged; direct writes through `.data` or `.numpy()` remain the caller's responsibility
- **Familiar Surface**: The method names match PyTorch's, so what you learn here transfers

The complete class stays together so its public interface is visible in one
place. Follow the path from `__init__` to an arithmetic method such as `__add__`,
then to `Function.apply`. The operation classes after the Tensor definition
provide the numerical work, and their adjacent tests check each operation.

Three shape rules explain the methods in the class before we inspect them.
`reshape` changes the grouping of elements without changing their count:
six values can become a `(2, 3)` matrix, but not a `(2, 4)` matrix. `transpose`
swaps axes: a `(2, 3)` matrix becomes `(3, 2)`, with entry `(i, j)` moving to
`(j, i)`. Matrix multiplication contracts the shared dimension:
`(2, 3) @ (3, 4)` produces `(2, 4)`, whereas `(2, 3) @ (2, 4)` is invalid.
`_validate_matmul_shapes` checks that contract before NumPy performs the work.
The later operation sections extend these examples to batched inputs and test
the rules individually. Utility methods such as `__repr__` help inspect results;
they can be read after the numerical path is clear.
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-class", "solution": true}
#| export
class Tensor:
    """Educational tensor - the foundation of machine learning computation.

    This class provides the core data structure for all ML operations:
    - data: The actual numerical values (NumPy array)
    - shape: Dimensions of the tensor
    - ndim: Number of dimensions (0=scalar, 1=vector, 2=matrix, ...)
    - size: Total number of elements
    - dtype: Data type (float32)

    All arithmetic, matrix, and shape operations are built on this foundation.
    """

    def __init__(self, data, requires_grad=False):
        """Create a new tensor from data.

        TODO: Initialize a Tensor by wrapping data in a NumPy array and setting attributes.

        APPROACH:
        1. If data is a list of Tensors, stack their arrays along a new first axis
           (the same convenience as torch.stack)
        2. Convert data to NumPy array with dtype=float32 and store it as self.data
        3. Set self.shape from the array's shape
        4. Set self.size from the array's size
        5. Set self.dtype from the array's dtype

        EXAMPLE:
        >>> t = Tensor([1, 2, 3])
        >>> print(t.shape)
        (3,)
        >>> print(t.size)
        3
        >>> Tensor([t, t]).shape
        (2, 3)

        HINT: Use np.array(data, dtype=np.float32) to convert data to NumPy array
        """
        ### BEGIN SOLUTION
        if isinstance(data, Tensor):
            data = data.data
        elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            data = np.stack([t.data for t in data])
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        ### END SOLUTION
        self.requires_grad = requires_grad   # Module 06 (autograd) will use this
        self.grad = None                     # Module 06 (autograd) will use this
        self._grad_fn = None                 # Module 06 (autograd) will use this
        self._graph_released = False         # Module 06 (autograd) will use this

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

    @property
    def ndim(self):
        """Number of tensor dimensions (0=scalar, 1=vector, 2=matrix, ...)."""
        return len(self.shape)

    def numel(self):
        """Return total number of elements (PyTorch-compatible)."""
        return self.size

    def contiguous(self):
        """Return a contiguous copy of the tensor data (PyTorch-compatible)."""
        return Copy.apply(self)

    def view(self, *shape):
        """Reshape alias with independent storage; unlike PyTorch's view(), this copies."""
        return self.reshape(*shape)

    def masked_fill(self, mask, value):
        """Fill positions where mask is True with value, matching PyTorch's masked_fill.

        Nothing in this module needs it yet. Module 12 will use it to blank out
        positions a model must not look at before normalizing scores.

        Args:
            mask:  A Tensor or numpy array of booleans, same shape as self (or broadcastable).
            value: Scalar fill value (Module 12 will pass float('-inf')).

        Returns:
            New Tensor with masked positions replaced by value.
        """
        mask_array = mask.data.astype(bool) if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
        return MaskedFill.apply(self, mask=mask_array, value=value)

    def __add__(self, other):
        """Add two tensors element-wise with broadcasting (the Add operation below)."""
        return Add.apply(self, _as_tensor(other))

    def __radd__(self, other):
        """Support natural scalar arithmetic: scalar + tensor."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract two tensors element-wise (the Sub operation below)."""
        return Sub.apply(self, _as_tensor(other))

    def __rsub__(self, other):
        """Support natural scalar arithmetic: scalar - tensor."""
        return Sub.apply(_as_tensor(other), self)

    def __mul__(self, other):
        """Multiply two tensors element-wise, NOT matrix multiplication (the Mul operation below)."""
        return Mul.apply(self, _as_tensor(other))

    def __rmul__(self, other):
        """Support natural scalar arithmetic: scalar * tensor."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide two tensors element-wise (the Div operation below)."""
        return Div.apply(self, _as_tensor(other))

    def __rtruediv__(self, other):
        """Support natural scalar arithmetic: scalar / tensor."""
        return Div.apply(_as_tensor(other), self)

    def _validate_matmul_shapes(self, other):
        """Validate that two tensors are compatible for matrix multiplication.

        This helper checks three conditions before any computation begins:
        1. The other operand must be a Tensor (not a plain number or array)
        2. Neither operand can be a 0D scalar (scalars use * instead)
        3. The inner dimensions must align. Matrix multiplication contracts the
           LAST axis of self against the ROWS axis of other. For a matrix that
           rows axis is shape[-2]; a 1D vector has only one axis, so its rows
           axis is shape[0]. Every case follows from that one rule:
               (M, K) @ (K, N)  ->  (M, N)      matrix @ matrix
               (M, K) @ (K,)    ->  (M,)        matrix @ vector
               (K,)   @ (K, N)  ->  (N,)        vector @ matrix
               (K,)   @ (K,)    ->  ()          vector @ vector (dot product)
           and (2, 3) @ (2,) is a mismatch because 3 != 2.

        TODO: Implement the three validation checks for matrix multiplication.

        APPROACH:
        1. Check isinstance(other, Tensor) - raise TypeError if not
        2. Check both tensors are at least 1D - raise ValueError if 0D
        3. inner_self = self.shape[-1]
           inner_other = other.shape[-2] if other has 2+ dims, else other.shape[0]
           raise ValueError if they differ (put both numbers in the message)

        EXAMPLE:
        >>> a = Tensor([[1, 2], [3, 4]])  # 2x2
        >>> b = Tensor([[5, 6], [7, 8]])  # 2x2
        >>> a._validate_matmul_shapes(b)  # No error - shapes are compatible
        >>> c = Tensor([[1, 2, 3]])        # 1x3
        >>> d = Tensor([[1], [2]])         # 2x1
        >>> c._validate_matmul_shapes(d)   # ValueError - 3 != 2
        >>> m = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
        >>> m._validate_matmul_shapes(Tensor([1, 2, 3]))  # No error - 3 == 3
        >>> m._validate_matmul_shapes(Tensor([1, 2]))     # ValueError - 3 != 2

        HINT: Use len(tensor.shape) to check dimensionality and tensor.shape[-1]
        to access the last dimension.
        """
        ### BEGIN SOLUTION role="scaffold"
        if not isinstance(other, Tensor):
            raise TypeError(
                f"Matrix multiplication requires Tensor, got {type(other).__name__}\n"
                f"  ❌ Cannot perform: Tensor @ {type(other).__name__}\n"
                f"  💡 Matrix multiplication (@) only works between two Tensors\n"
                f"  🔧 Wrap the {type(other).__name__} first: tensor @ Tensor(other)"
            )
        if len(self.shape) == 0 or len(other.shape) == 0:
            raise ValueError(
                f"Matrix multiplication requires at least 1D tensors\n"
                f"  ❌ Got shapes: {self.shape} @ {other.shape}\n"
                f"  💡 Scalars (0D tensors) cannot be matrix-multiplied; use * for element-wise\n"
                f"  🔧 Use tensor * scalar instead"
            )
        inner_self = self.shape[-1]
        inner_other = other.shape[-2] if len(other.shape) >= 2 else other.shape[0]
        if inner_self != inner_other:
            if len(other.shape) >= 2:
                fix = f"other.transpose() to get shape {other.shape[::-1]}, or reshape self"
            else:
                fix = f"a vector of length {inner_self}, or transpose self"
            raise ValueError(
                f"Matrix multiplication shape mismatch: {self.shape} @ {other.shape}\n"
                f"  ❌ Inner dimensions don't match: {inner_self} vs {inner_other}\n"
                f"  💡 For A @ B, A's last dim must equal B's rows (shape[-2] for a matrix, shape[0] for a vector)\n"
                f"  🔧 Try: {fix}"
            )
        ### END SOLUTION

    def matmul(self, other):
        """Matrix multiplication of two tensors.

        Validates shapes via _validate_matmul_shapes, then hands the arrays to
        the MatMul operation, where the explicit-loop multiply lives.
        """
        self._validate_matmul_shapes(other)
        return MatMul.apply(self, other)

    def __matmul__(self, other):
        """Enable @ operator for matrix multiplication."""
        return self.matmul(other)

    def __getitem__(self, key):
        """Indexing and slicing, t[key]. Delegates to the Slice operation."""
        return Slice.apply(self, key=key)

    def reshape(self, *shape):
        """Reshape tensor to new dimensions.

        A reshape keeps every element and changes only how they are grouped, so
        the new shape must hold exactly self.size elements. One dimension may
        be given as -1, meaning "whatever is left": with 6 elements,
        reshape(2, -1) is reshape(2, 3) and reshape(-1, 3) is reshape(2, 3).
        Any other negative size, or a zero, has no meaning and is rejected.

        TODO: Reshape tensor while preserving total element count.

        APPROACH:
        1. Handle both reshape(2, 3) and reshape((2, 3)) calling styles
        2. Reject any dimension that is 0 or below -1 (only -1 is special)
        3. If -1 in shape, infer that dimension from total size:
           known_size = product of the other dimensions;
           the -1 becomes self.size // known_size (must divide evenly; only one -1 allowed)
        4. Validate total elements match: np.prod(new_shape) == self.size
        5. Hand the validated shape to the operation: return Reshape.apply(self, shape=new_shape)

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
        ### BEGIN SOLUTION role="scaffold"
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        bad = [d for d in new_shape if d == 0 or d < -1]
        if bad:
            raise ValueError(
                f"Cannot reshape {self.shape} to {new_shape}\n"
                f"  ❌ Invalid dimension size {bad[0]}: sizes must be positive, or -1 to infer\n"
                f"  💡 -1 is the only special value; it stands for 'whatever is left'\n"
                f"  🔧 Replace {bad[0]} with a positive size or -1"
            )
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError(
                    f"Cannot reshape {self.shape} with multiple unknown dimensions\n"
                    f"  ❌ Found {new_shape.count(-1)} dimensions set to -1 in {new_shape}\n"
                    f"  💡 Only one dimension can be inferred; others must be specified\n"
                    f"  🔧 Replace all but one -1 with explicit sizes (total elements: {self.size})"
                )
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            if self.size % known_size != 0:
                raise ValueError(
                    f"Cannot infer -1 dimension: {self.size} elements is not "
                    f"divisible by the known dimensions product {known_size}\n"
                    f"  ❌ {self.size} % {known_size} = {self.size % known_size}\n"
                    f"  💡 The -1 dimension must be a whole number"
                )
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)
        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Cannot reshape {self.shape} to {new_shape}\n"
                f"  ❌ Element count mismatch: {self.size} elements vs {target_size} elements\n"
                f"  💡 Reshape preserves data, so total elements must stay the same\n"
                f"  🔧 Use -1 to infer a dimension: reshape(-1, {new_shape[-1] if len(new_shape) > 0 else 1}) lets NumPy calculate"
            )
        return Reshape.apply(self, shape=new_shape)
        ### END SOLUTION

    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions.

        Transposing swaps two axes: a (2, 3) matrix becomes (3, 2), with
        element [i, j] moving to [j, i]. Permute asks NumPy to reorder axes
        using a list such as (1, 0). NumPy can represent this as a view, but
        Function.apply wraps it in a new Tensor with its own copied storage.
        Your job here is to build the axes list.

        TODO: Swap tensor dimensions (default: swap last two dimensions).

        APPROACH:
        1. If no dims specified: swap last two dimensions (most common case)
        2. For 1D tensors: return Copy.apply(self) (no transpose needed)
        3. If both dims specified: swap those specific dimensions
        4. Hand the axes list to the operation: return Permute.apply(self, axes=tuple(axes))

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
        - The Permute operation calls np.transpose(a, axes); here you only build the axes list
        """
        ### BEGIN SOLUTION role="scaffold"
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Copy.apply(self)
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
        else:
            if dim0 is None or dim1 is None:
                provided = f"dim0={dim0}" if dim1 is None else f"dim1={dim1}"
                missing = "dim1" if dim1 is None else "dim0"
                raise ValueError(
                    f"Transpose requires both dimensions to be specified\n"
                    f"  ❌ Got {provided}, but {missing} is None\n"
                    f"  💡 Either provide both dims or neither (default swaps last two)\n"
                    f"  🔧 Use transpose({dim0 if dim0 is not None else 0}, {dim1 if dim1 is not None else 1}) or just transpose()"
                )
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Permute.apply(self, axes=tuple(axes))
        ### END SOLUTION

    def permute(self, *axes):
        """Permute tensor dimensions according to axes. Delegates to Permute.apply."""
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = axes[0]
        return Permute.apply(self, axes=tuple(axes))

    def sum(self, axis=None, keepdims=False):
        """Sum all elements or along an axis. Delegates to the Sum operation."""
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        """Average all elements or along an axis. Delegates to the Mean operation."""
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """Maximum over all elements or along an axis. Delegates to the Max operation."""
        return Max.apply(self, axis=axis, keepdims=keepdims)

    def backward(self, gradient=None, retain_graph=False):
        """Propagate gradients to every tensor this one was computed from. Module 06 implements this."""
        raise NotImplementedError("Module 06 (autograd) implements Tensor.backward")

    def zero_grad(self):
        """Forget the accumulated gradient. Module 06 implements this."""
        raise NotImplementedError("Module 06 (autograd) implements Tensor.zero_grad")

# %% [markdown]
r"""
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

    # Test PyTorch-compatible utility methods
    assert scalar.ndim == 0, "Scalar should be 0-dimensional"
    assert vector.ndim == 1, "Vector should be 1-dimensional"
    assert matrix.ndim == 2, "Matrix should be 2-dimensional"
    assert tensor_3d.ndim == 3, "3D tensor should be 3-dimensional"

    assert scalar.numel() == 1, "Scalar has 1 element"
    assert vector.numel() == 3, "Vector has 3 elements"
    assert matrix.numel() == 4, "2x2 matrix has 4 elements"

    # A list of Tensors stacks along a new first axis (APPROACH step 1)
    stacked = Tensor([vector, vector])
    assert stacked.shape == (2, 3), f"Stacking two (3,) Tensors should give (2, 3), got {stacked.shape}"
    assert np.array_equal(stacked.data[1], vector.data)
    assert stacked.dtype == np.float32

    print("✅ Tensor creation works correctly!")

if __name__ == "__main__":
    test_unit_tensor_creation()

# %% [markdown]
r"""
## 🏗️ Element-wise Arithmetic Operations

Element-wise operations are the workhorses of neural network computation. They apply the same operation to corresponding elements in tensors, often with broadcasting to handle different shapes elegantly.

### Why Element-wise Operations Matter

Element-wise operations are fundamental to numerical computing:
- **Scaling**: Multiply every element by a constant (e.g., unit conversion)
- **Thresholding**: Set values below zero to zero (clamp negatives)
- **Normalization**: Subtract the mean from every element to center data
- **Comparison**: Compute difference between two arrays element-by-element

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
r"""
### Subtraction, Multiplication, and Division

These operations follow the same pattern as addition, working element-wise with broadcasting support. Each serves specific purposes in data processing:

| Operation | Mathematical Form | Concrete Example | Machine Learning Use Case |
| :--- | :--- | :--- | :--- |
| **Subtraction** | $\mathbf{x} - \mathbf{y}$ | $[6, 8] - [1, 2] = [5, 6]$ | Data centering: $\mathbf{x} - \boldsymbol{\mu}$ |
| **Multiplication** | $\mathbf{x} \odot \mathbf{y}$ | $[2, 3] \times [4, 5] = [8, 15]$ | Gating & attention masks: $\mathbf{x} \odot \mathbf{m}$ |
| **Division** | $\mathbf{x} \oslash \mathbf{y}$ | $[8, 9] / [2, 3] = [4.0, 3.0]$ | Variance normalization: $(\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$ |

**Broadcasting with Scalars:**
$$\begin{aligned}
\text{Scale: } & \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \times 2 = \begin{bmatrix} 2 & 4 & 6 \end{bmatrix} \\
\text{Shift: } & \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} - 1 = \begin{bmatrix} 0 & 1 & 2 \end{bmatrix} \\
\text{Normalize: } & \begin{bmatrix} 2 & 4 & 6 \end{bmatrix} / 2 = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}
\end{aligned}$$

**Feature Standardization in ML:**
Given batch data $\mathbf{X} \in \mathbb{R}^{3 \times 2}$, feature mean $\boldsymbol{\mu} \in \mathbb{R}^2$, and std $\boldsymbol{\sigma} \in \mathbb{R}^2$:
$$\mathbf{X}_{\text{norm}} = \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}} \implies (3, 2) - (2,) \xrightarrow{\text{broadcast}} (3, 2) \oslash (2,) \to (3, 2)$$

**Performance Note**: Element-wise operations are vectorized through CPU SIMD instructions (AVX-512/NEON), processing 4–16 float32 elements per instruction cycle.

**⚠️ Broadcasting Pitfall**: Broadcasting is powerful but dangerous. When shapes are accidentally mismatched, broadcasting silently fills missing dimensions instead of raising an error:

| Tensor | Shape | Meaning |
| :--- | :--- | :--- |
| `predictions` | `(32, 4)` | 32 batch samples, 4 class scores each |
| `targets` | `(4,)` | Intended 4 classes, but missing batch dimension $(32, 1)$! |
| **Silent Broadcast** | `(4,)` $\to$ `(32, 4)` | Target is replicated across all 32 samples with no warning |

This is a frequent source of silent bugs in ML code. Always verify tensor shapes before element-wise operations like loss computation.

This is the #1 source of silent bugs in ML code. Always verify shapes match
before element-wise operations like loss computation.
"""


# %% [markdown]
r"""
### Implement: Add, Sub, Mul, Div

Write `forward` for each operation. The inputs are NumPy arrays; return an array. `Tensor` already wrapped scalars and will wrap your result.
"""

# %% nbgrader={"grade": false, "grade_id": "ops-arithmetic", "solution": true}
#| export
class Add(Function):
    """Element-wise addition c = a + b, with NumPy broadcasting."""

    def forward(self, a, b):
        """
        Add two arrays element-wise with broadcasting support.

        TODO: Return the element-wise sum of a and b.

        APPROACH:
        1. Both inputs arrive as NumPy arrays (Tensor.__add__ already wrapped any scalar in a Tensor)
        2. NumPy's + handles broadcasting automatically

        EXAMPLE:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> c = a + b          # Tensor.__add__ -> Add.apply(a, b) -> this forward
        >>> print(c.data)
        [5. 7. 9.]
        """
        ### BEGIN SOLUTION
        return a + b
        ### END SOLUTION


class Sub(Function):
    """Element-wise subtraction c = a - b, with NumPy broadcasting."""

    def forward(self, a, b):
        """
        Subtract two arrays element-wise.

        TODO: Return a - b.

        HINT: NumPy's - operator handles broadcasting automatically
        """
        ### BEGIN SOLUTION role="scaffold"
        return a - b
        ### END SOLUTION


class Mul(Function):
    """Element-wise multiplication c = a * b (NOT matrix multiplication), with NumPy broadcasting."""

    def forward(self, a, b):
        """
        Multiply two arrays element-wise.

        TODO: Return a * b.

        EXAMPLE:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> print((a * b).data)
        [ 4. 10. 18.]
        """
        ### BEGIN SOLUTION role="scaffold"
        return a * b
        ### END SOLUTION


class Div(Function):
    """Element-wise division c = a / b, with NumPy broadcasting."""

    def forward(self, a, b):
        """
        Divide two arrays element-wise.

        TODO: Return a / b.

        HINT: Do not guard against zero. float32 division by zero gives inf, which is the honest answer.
        """
        ### BEGIN SOLUTION role="scaffold"
        return a / b
        ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: Arithmetic Operations

This test validates our arithmetic operations work correctly with both tensor-tensor and tensor-scalar operations, including broadcasting behavior. Scalar arithmetic should feel natural whether the scalar appears before or after the tensor.

**What we're testing**: Addition, subtraction, multiplication, division with broadcasting
**Why it matters**: Foundation for batch processing, data normalization, and feature scaling
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

    # Scalar on the left should behave naturally too
    result = 10 + a
    assert np.array_equal(result.data, np.array([11, 12, 13], dtype=np.float32))

    # Test broadcasting with different shapes (matrix + vector)
    matrix = Tensor([[1, 2], [3, 4]])
    vector = Tensor([10, 20])
    result = matrix + vector
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # ⚠️ Broadcasting pitfall: predictions (batch, features) minus a targets
    # vector (features,) does NOT raise. It broadcasts, and the single target
    # row is subtracted from every sample. Watch the silent broadcast happen:
    predictions = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])  # 4 samples, 3 features
    targets_bad = Tensor([1, 2, 3])                                          # missing batch dim
    diff = predictions - targets_bad
    assert diff.shape == (4, 3), f"Broadcast should give (4, 3), got {diff.shape}"
    assert np.array_equal(diff.data[0], np.zeros(3, dtype=np.float32))     # row 0 happens to match
    assert np.array_equal(diff.data[3], np.array([9, 9, 9], dtype=np.float32))  # row 3 was never meant to see [1,2,3]
    # No error, no warning, and rows 1 to 3 are compared against the wrong target.
    # Check shapes yourself before a loss computation; NumPy will not.

    # Test subtraction (data centering)
    result = b - a
    assert np.array_equal(result.data, np.array([3, 3, 3], dtype=np.float32))

    # Test multiplication (scaling)
    result = a * 2
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))

    result = 2 * a
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))

    # Test division (normalization)
    result = b / 2
    assert np.array_equal(result.data, np.array([2.0, 2.5, 3.0], dtype=np.float32))

    result = 12 / b
    assert np.allclose(result.data, np.array([3.0, 12.0 / 5.0, 2.0], dtype=np.float32))

    # Order matters for subtraction, but scalar-left arithmetic should still work
    result = 10 - a
    assert np.array_equal(result.data, np.array([9, 8, 7], dtype=np.float32))

    # Test chaining operations (common in ML pipelines)
    normalized = (a - 2) / 2  # Center and scale
    expected = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    assert np.allclose(normalized.data, expected)

    print("✅ Arithmetic operations work correctly!")

if __name__ == "__main__":
    test_unit_arithmetic_operations()

# %% [markdown]
r"""
## 🏗️ Matrix Multiplication: The Core Computational Operation

Matrix multiplication is fundamentally different from element-wise multiplication. It's the operation that powers linear transformations — combining information across features to produce new representations.

### Why Matrix Multiplication Matters

Many scientific and data-processing tasks rely on matrix multiplication:

$$\mathbf{X} \in \mathbb{R}^{N \times D_{\text{in}}}, \quad \mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}} \implies \mathbf{Y} = \mathbf{X}\mathbf{W} \in \mathbb{R}^{N \times D_{\text{out}}}$$

For example, in feature projection (e.g. MNIST dimensionality reduction):
$$\underbrace{\mathbf{X}}_{(32, 784)} \times \underbrace{\mathbf{W}}_{(784, 256)} = \underbrace{\mathbf{Y}}_{(32, 256)}$$

### Matrix Multiplication Visualization

$$\underbrace{\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}}_{A \in \mathbb{R}^{2 \times 3}} \times \underbrace{\begin{bmatrix} 7 & 8 \\ 9 & 1 \\ 1 & 2 \end{bmatrix}}_{B \in \mathbb{R}^{3 \times 2}} = \begin{bmatrix} 1\cdot 7 + 2\cdot 9 + 3\cdot 1 & 1\cdot 8 + 2\cdot 1 + 3\cdot 2 \\ 4\cdot 7 + 5\cdot 9 + 6\cdot 1 & 4\cdot 8 + 5\cdot 1 + 6\cdot 2 \end{bmatrix} = \underbrace{\begin{bmatrix} 28 & 16 \\ 79 & 49 \end{bmatrix}}_{C \in \mathbb{R}^{2 \times 2}}$$

**Computation Breakdown:**
$$\begin{aligned}
C_{0,0} &= \mathbf{A}[0, :] \cdot \mathbf{B}[:, 0] = [1, 2, 3] \cdot [7, 9, 1] = 1\cdot 7 + 2\cdot 9 + 3\cdot 1 = 7 + 18 + 3 = 28 \\
C_{0,1} &= \mathbf{A}[0, :] \cdot \mathbf{B}[:, 1] = [1, 2, 3] \cdot [8, 1, 2] = 1\cdot 8 + 2\cdot 1 + 3\cdot 2 = 8 + 2 + 6 = 16 \\
C_{1,0} &= \mathbf{A}[1, :] \cdot \mathbf{B}[:, 0] = [4, 5, 6] \cdot [7, 9, 1] = 4\cdot 7 + 5\cdot 9 + 6\cdot 1 = 28 + 45 + 6 = 79 \\
C_{1,1} &= \mathbf{A}[1, :] \cdot \mathbf{B}[:, 1] = [4, 5, 6] \cdot [8, 1, 2] = 4\cdot 8 + 5\cdot 1 + 6\cdot 2 = 32 + 5 + 12 = 49
\end{aligned}$$

**Key Rule: Inner dimensions must match!**
$$\mathbf{A}_{(M \times K)} @ \mathbf{B}_{(K \times N)} = \mathbf{C}_{(M \times N)} \quad \text{where the shared dimension } K \text{ contracts}$$

### Computational Complexity and Performance

For $\mathbf{C} = \mathbf{A} @ \mathbf{B}$ where $\mathbf{A} \in \mathbb{R}^{M \times K}$ and $\mathbf{B} \in \mathbb{R}^{K \times N}$:
* **Multiplications**: $M \times N \times K$
* **Additions**: $M \times N \times (K - 1) \approx M \times N \times K$
* **Total FLOPs**: $\approx 2MNK$ operations

| Operand | Shape | Traversal Pattern | Memory Locality & Hardware Implications |
| :--- | :--- | :--- | :--- |
| **Matrix $A$** | $(M, K)$ | Row-by-row (`A[i, :]`) | 🟢 **Sequential**: Cache-line streaming friendly |
| **Matrix $B$** | $(K, N)$ | Column-by-column (`B[:, j]`) | 🔴 **Strided**: Cache line thrashing unless tiled/transposed |
| **Matrix $C$** | $(M, N)$ | Row-by-row accumulation | 🟢 **Sequential**: Spatial write locality |

This is why optimized BLAS libraries (OpenBLAS, Apple Accelerate, Intel MKL) employ cache blocking, register tiling, and SIMD vectorization.

### Chained Matrix Multiplications

$$\mathbf{X}_{(100 \times 50)} \xrightarrow{\mathbf{W}_1 \in \mathbb{R}^{50 \times 20}} \mathbf{H}_{(100 \times 20)} \xrightarrow{\mathbf{W}_2 \in \mathbb{R}^{20 \times 5}} \mathbf{Y}_{(100 \times 5)}$$

FLOPs budget for $100$ batch samples across both projections:
$$\text{Total Compute} = 100 \times (2 \times 50 \times 20 + 2 \times 20 \times 5) = 100 \times (2{,}000 + 200) = 220{,}000 \text{ FLOPs}$$

This is why hardware acceleration matters: specialized matrix engines execute thousands of these multiply-accumulate operations in parallel!
"""

# %% [markdown]
r"""
### Shape Validation for Matrix Multiplication

Before performing any computation, matrix multiplication must verify that the
two operands are compatible. There are three things that can go wrong, and each
one deserves a distinct, educational error message.

| Check Order | Invariant Verified | Exception Raised | Educational Error Explanation |
| :--- | :--- | :--- | :--- |
| **1. Type Check** | `isinstance(other, Tensor)` | `TypeError` | `f"matmul requires Tensor, got {type(other).__name__}"` |
| **2. Scalar Rejection** | `self.ndim > 0` and `other.ndim > 0` | `ValueError` | `f"matmul does not support 0D scalars, use * for scalar multiplication"` |
| **3. Inner Dimension** | `self.shape[-1] == other.shape[0]` | `ValueError` | `f"matmul shape mismatch: ({self.shape[-1]}) != ({other.shape[0]})"` |

Separating validation from computation keeps each function focused on a single
concept: `_validate_matmul_shapes` teaches input checking, while
`MatMul.forward` teaches the algorithm itself.
"""

# %% [markdown]
r"""
### 🧪 Unit Test: Validate Matmul Shapes

**What we're testing**: All three shape-mismatch categories are caught and named, for matrices and vectors alike
**Why it matters**: A shape error caught at the boundary names the problem; one
that slips through surfaces as a NumPy error deep inside a matmul, pages away
from the line that caused it
**Expected**: Valid shapes pass silently; each invalid case raises with its reason
"""

# %% nbgrader={"grade": true, "grade_id": "tensor-validate-matmul", "locked": true, "points": 5}
def test_unit_validate_matmul_shapes():
    """🧪 Test matmul shape validation catches all three error categories."""
    print("🧪 Unit Test: Validate Matmul Shapes...")

    # Valid shapes should pass without error
    a = Tensor([[1, 2], [3, 4]])  # 2x2
    b = Tensor([[5, 6], [7, 8]])  # 2x2
    a._validate_matmul_shapes(b)  # No exception

    # Valid rectangular shapes
    c = Tensor([[1, 2, 3]])       # 1x3
    d = Tensor([[1], [2], [3]])   # 3x1
    c._validate_matmul_shapes(d)  # No exception (inner dim 3 matches)

    # Check 1: TypeError when other is not a Tensor
    try:
        a._validate_matmul_shapes([[1, 2], [3, 4]])
        assert False, "Should have raised TypeError for non-Tensor"
    except TypeError as e:
        assert "requires Tensor" in str(e)
        assert "list" in str(e)

    # Check 2: ValueError when either operand is a 0D scalar
    try:
        scalar = Tensor(5.0)
        scalar._validate_matmul_shapes(a)
        assert False, "Should have raised ValueError for 0D tensor"
    except ValueError as e:
        assert "at least 1D" in str(e)

    # Check 3: ValueError when inner dimensions don't match
    try:
        incompatible_a = Tensor([[1, 2]])         # 1x2
        incompatible_b = Tensor([[1], [2], [3]])   # 3x1
        incompatible_a._validate_matmul_shapes(incompatible_b)
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        assert "Inner dimensions don't match" in str(e)
        assert "2 vs 3" in str(e)

    # Check 3, vector case: a (2,3) matrix times a length-3 vector is fine,
    # a length-2 vector is the same mismatch and must be caught HERE, not
    # deep inside np.matmul
    m = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    m._validate_matmul_shapes(Tensor([1, 2, 3]))  # No exception (3 == 3)
    try:
        m._validate_matmul_shapes(Tensor([1, 2]))  # (2,3) @ (2,)
        assert False, "Should have raised ValueError for matrix-vector mismatch"
    except ValueError as e:
        assert "Inner dimensions don't match" in str(e)
        assert "3 vs 2" in str(e)

    print("✅ Matmul shape validation works correctly!")

if __name__ == "__main__":
    test_unit_validate_matmul_shapes()

# %% [markdown]
r"""
### Implement: MatMul

Write `forward` for the matrix product. The inputs are NumPy arrays whose shapes `Tensor.matmul` already validated; return an array. The explicit-loop guidance lives in the docstring below.
"""

# %% nbgrader={"grade": false, "grade_id": "ops-matmul", "solution": true}
#| export
class MatMul(Function):
    """Matrix multiplication c = a @ b. Shapes were validated by Tensor.matmul."""

    def forward(self, a, b):
        """
        Multiply two matrices.

        For 2D matrices, uses explicit nested loops so you can see exactly how
        each output element is a dot product of a row and a column. For anything
        other than 2D @ 2D (a vector on either side, or batched 3D+ inputs),
        delegates to np.matmul.

        TODO: Compute the matrix product using explicit loops for 2D @ 2D and
        np.matmul for every other case.

        APPROACH:
        1. For 2D @ 2D: use explicit nested loops with np.dot per element
        2. For every other case (1D operands, batched 3D+): use np.matmul
        3. Return the result array

        EXAMPLE:
        >>> a = Tensor([[1, 2], [3, 4]])  # 2x2
        >>> b = Tensor([[5, 6], [7, 8]])  # 2x2
        >>> print((a @ b).data)
        [[19. 22.]
         [43. 50.]]

        HINTS:
        - Inner dimensions must match: (M, K) @ (K, N) = (M, N)
        - For 2D case: use np.dot(a[i, :], b[:, j]) for each output element
        """
        ### BEGIN SOLUTION
        # Educational implementation: explicit loops to show what matrix multiplication does
        # This is intentionally slower than np.matmul to demonstrate the value of vectorization

        # Handle 2D matrices with explicit loops (educational)
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            _, N = b.shape   # b's row count is K; Tensor.matmul already checked it
            result_data = np.zeros((M, N), dtype=a.dtype)

            # Explicit nested loops - students can see exactly what's happening!
            # Each output element is a dot product of a row from A and a column from B
            for i in range(M):
                for j in range(N):
                    # Dot product of row i from A with column j from B
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        else:
            # Anything other than 2D @ 2D (a 1D operand, or batched 3D+ inputs)
            # goes to np.matmul. The mechanism is the same dot product per element.
            result_data = np.matmul(a, b)

        return result_data
        ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: Matrix Multiplication

Now that validation is handled by `_validate_matmul_shapes`, this test focuses
on the computational correctness of `matmul` itself. We verify square matrices,
rectangular matrices, and matrix-vector products all produce the expected
numerical results.

**What we're testing**: Matrix multiplication computation for various shape combinations
**Why it matters**: Core operation in linear algebra and data transformations
**Expected**: Correct numerical results matching hand-calculated dot products
"""

# %% nbgrader={"grade": true, "grade_id": "tensor-matmul", "locked": true, "points": 15}
def test_unit_matrix_multiplication():
    """🧪 Test matrix multiplication operations."""
    print("🧪 Unit Test: Matrix Multiplication...")

    # Test 2x2 matrix multiplication (basic case)
    a = Tensor([[1, 2], [3, 4]])  # 2x2
    b = Tensor([[5, 6], [7, 8]])  # 2x2
    result = a.matmul(b)
    # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test rectangular matrices (common in data transformations)
    c = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3 (like samples=2, features=3)
    d = Tensor([[7, 8], [9, 10], [11, 12]])  # 3x2 (like features=3, outputs=2)
    result = c.matmul(d)
    # Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    expected = np.array([[58, 64], [139, 154]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test matrix-vector multiplication (common in linear transforms)
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    vector = Tensor([1, 2, 3])  # 1D vector
    result = matrix.matmul(vector)
    # Expected: [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    expected = np.array([14, 32], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test @ operator sugar
    result_at = a @ b
    assert np.array_equal(result_at.data, np.array([[19, 22], [43, 50]], dtype=np.float32))

    print("✅ Matrix multiplication works correctly!")

if __name__ == "__main__":
    test_unit_matrix_multiplication()

# %% [markdown]
r"""
## 🏗️ Shape Manipulation: Reshape and Transpose

Data processing pipelines constantly change tensor shapes to match computation requirements. Understanding these operations is crucial for efficient data flow.

### Why Shape Manipulation Matters

Many computations require constant shape changes across the network pipeline:

$$\underbrace{\mathbf{X}_{(32, 3, 224, 224)}}_{\text{Batch of RGB Images}} \xrightarrow{\text{Convolutions}} \underbrace{\mathbf{F}_{(32, 512, 7, 7)}}_{\text{Feature Maps}} \xrightarrow{\text{Global Pool}} \underbrace{\mathbf{H}_{(32, 512)}}_{\text{Vector Embeddings}} \xrightarrow{\text{Linear Projection}} \underbrace{\hat{\mathbf{Y}}_{(32, 10)}}_{\text{Class Logits}}$$

### Reshape: Changing Interpretation of the Same Data

Reshaping alters coordinate dimensions without moving or reordering elements:

$$\begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 \end{bmatrix}_{(6,)} \xrightarrow{\text{reshape}(2, 3)} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}_{(2, 3)}$$

**Underlying Linear Memory Buffer:**
$$\begin{array}{|c|c|c|c|c|c|} \hline 1 & 2 & 3 & 4 & 5 & 6 \\ \hline \end{array} \quad \text{(Elements remain at identical byte offsets)}$$

**Key Insight**: NumPy can reshape this contiguous array in $O(1)$ by returning a view (other layouts may require a copy). Our Tensor wraps the result with `np.array()`, so TinyTorch's reshape is $O(N)$. The layout reasoning is unchanged; the copy is the price of every Tensor owning its buffer outright.

| Reshape Pattern | Shape Transformation | Deep Learning Purpose |
| :--- | :--- | :--- |
| **Flatten Spatial Grid** | $(N, C, H, W) \to (N, C \cdot H \cdot W)$ | Transition from Conv2D feature map to Linear classifier |
| **Unflatten Sequence** | $(N, D) \to (N, H, W, C)$ | Latent representation to image decoder grid |
| **Inject Batch Axis** | $(H, W) \to (1, 1, H, W)$ | Single-sample inference into batched model pipeline |

### Transpose: Swapping Dimensions

Transposing reinterprets axes by inverting coordinate strides:

$$\mathbf{X} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}_{(2, 3)} \implies \mathbf{X}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}_{(3, 2)}$$

**Strides Reinterpretation:**
$$\begin{aligned}
\text{Original Strides } (3, 1): \quad & \text{row step} = 3 \text{ elements}, \quad \text{col step} = 1 \text{ element} \\
\text{Transposed Strides } (1, 3): \quad & \text{row step} = 1 \text{ element}, \quad \text{col step} = 3 \text{ elements}
\end{aligned}$$

The bytes never move; only the coordinate strides do. Walking a row of the transposed view jumps through memory across stride boundaries:

| Mathematical Formulation | Operation | Systems & ML Purpose |
| :--- | :--- | :--- |
| **$\mathbf{X}^T \mathbf{X}$** | Gram / Covariance Matrix | Computes inter-feature correlations |
| **$\mathbf{A}^T \mathbf{b}$** | Least Squares Projection | Projects target vector onto column space of $\mathbf{A}$ |
| **$\mathbf{Q} \mathbf{K}^T$** | Attention Matrix Transpose | Aligns sequence positions for dot-product attention |

### Performance Implications

| Operation | NumPy Underlying Behavior | TinyTorch Tensor Implementation | Memory & Cache Implications |
| :--- | :--- | :--- | :--- |
| **`reshape()`** | $O(1)$ zero-copy view when C-contiguous | $O(N)$ copies $N$ float32 values | Independent buffer ownership prevents subtle mutation leaks |
| **`transpose()`** | $O(1)$ view with inverted coordinate strides | $O(N)$ copies with non-contiguous strides | Strided access on transposed data jumps cache lines |

Measure the operation and its consumer separately. A cheap view can lead to a more expensive subsequent computation; a copy can cost time now and improve a later access pattern. The result depends on layout, kernel, and hardware.

"""


# %% [markdown]
r"""
### Implement: Slice, Reshape, Permute, Copy, MaskedFill

Write `forward` for each operation. The input is a NumPy array; return an array. The Tensor methods pass the extra information (`key`, `shape`, `axes`, `mask` and `value`) as keyword parameters to `apply`, and `Function.__init__` stores each one on the node, so it is available as `self.key`, `self.shape`, and so on.
"""

# %% nbgrader={"grade": false, "grade_id": "ops-shape", "solution": true}
#| export
class Slice(Function):
    """Indexing and slicing x[key]. Parameter: key (anything NumPy indexing accepts)."""

    def forward(self, a):
        """
        Index or slice the array.

        TODO: Return a[self.key].

        APPROACH:
        1. self.key is whatever the caller wrote inside the brackets (an int, a slice, a tuple, ...)
        2. NumPy indexing already understands every one of those, so hand it the key
        """
        ### BEGIN SOLUTION role="scaffold"
        return a[self.key]
        ### END SOLUTION


class Reshape(Function):
    """Same elements, new shape. Parameter: shape (already validated by Tensor.reshape)."""

    def forward(self, a):
        """
        Reshape the array.

        TODO: Return np.reshape(a, self.shape).
        """
        ### BEGIN SOLUTION role="scaffold"
        return np.reshape(a, self.shape)
        ### END SOLUTION


class Permute(Function):
    """Reorder axes. Parameter: axes, the new order of the old axes (Tensor.transpose computes it)."""

    def forward(self, a):
        """
        Permute the axes of the array.

        TODO: Return np.transpose(a, self.axes).
        """
        ### BEGIN SOLUTION role="scaffold"
        return np.transpose(a, self.axes)
        ### END SOLUTION


class Copy(Function):
    """A contiguous copy of the array (used by contiguous() and 1D transpose)."""

    def forward(self, a):
        """
        Copy the array into contiguous memory.

        TODO: Return np.ascontiguousarray(a), preserving the original shape.

        HINT: NumPy promotes a scalar to shape (1,); reshape back to a.shape.
        """
        ### BEGIN SOLUTION role="scaffold"
        return np.ascontiguousarray(a).reshape(a.shape)
        ### END SOLUTION


class MaskedFill(Function):
    """Replace masked positions with a value. Parameters: mask (bool array), value."""

    def forward(self, a):
        """
        Fill the positions where self.mask is True with self.value.

        TODO: Copy the array, broadcast the mask to its shape, and fill masked positions.

        HINT: Copy first. Boolean indexing needs a full-size mask; use
        np.broadcast_to(self.mask, a.shape) to expand a shared attention mask.
        """
        ### BEGIN SOLUTION role="scaffold"
        result = a.copy()
        result[np.broadcast_to(self.mask, a.shape)] = self.value
        return result
        ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: Shape Manipulation

This test validates reshape and transpose operations work correctly with validation and edge cases.

**What we're testing**: Reshape and transpose operations with proper error handling
**Why it matters**: Essential for data manipulation and multi-dimensional array processing
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
        assert "Element count mismatch" in str(e)
        assert "6 elements vs 4 elements" in str(e)

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

    # Non-symmetric shape so a wrong axis swap produces the wrong shape and data.
    tensor_3d = Tensor(np.arange(24).reshape(2, 3, 4))  # (2, 3, 4)
    swapped = tensor_3d.transpose(0, 2)  # (4, 3, 2)
    assert swapped.shape == (4, 3, 2), (
        f"transpose(0, 2) on (2,3,4) should give (4,3,2), got {swapped.shape}"
    )
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert swapped.data[k, j, i] == tensor_3d.data[i, j, k], (
                    f"Data mismatch at [{k},{j},{i}]: expected {tensor_3d.data[i,j,k]}, "
                    f"got {swapped.data[k,j,i]}"
                )

    scalar = Tensor(3.0)
    assert scalar.contiguous().shape == (), "Contiguous must preserve scalar rank"
    assert scalar.transpose().shape == (), "Scalar transpose must preserve rank"

    # Test contiguous returns a copy with same data
    contig = matrix.contiguous()
    assert np.array_equal(contig.data, matrix.data)
    assert contig.data is not matrix.data, "contiguous() should return a copy"

    # Test common reshape pattern (flatten multi-dimensional data)
    batch_images = Tensor(rng.random((2, 3, 4)))  # (batch=2, height=3, width=4)
    flattened = batch_images.reshape(2, -1)  # (batch=2, features=12)
    assert flattened.shape == (2, 12)

    # A shared feature mask broadcasts across rows (later used by attention).
    masked = matrix.masked_fill(np.array([False, True, False]), -1)
    assert np.array_equal(masked.data, [[1, -1, 3], [4, -1, 6]])
    assert np.array_equal(matrix.data, [[1, 2, 3], [4, 5, 6]])
    assert not np.shares_memory(matrix.data, transposed.data), "Transpose owns copied storage"

    print("✅ Shape manipulation works correctly!")

if __name__ == "__main__":
    test_unit_shape_manipulation()

# %% [markdown]
r"""
## 🏗️ Reduction Operations: Aggregating Information

Reduction operations collapse dimensions by aggregating data, which is essential for computing statistics and preparing data for further processing.

### Why Reductions are Crucial in ML

Reduction operations appear throughout neural networks:

| Reduction Type | Syntax | Dimensional Shift | Machine Learning Use Case |
| :--- | :--- | :--- | :--- |
| **Column Statistics** | `data.mean(axis=0)` | $(N, D) \to (D,)$ | Batch normalization: feature mean & variance |
| **Row Aggregation** | `data.mean(axis=1)` | $(N, D) \to (N,)$ | Sample summary: token energy, sample norm |
| **Spatial Averaging** | `img.mean(axis=(1, 2))` | $(N, H, W) \to (N,)$ | Global average pooling across image channels |

### Understanding Axis Operations

Given matrix $\mathbf{X} \in \mathbb{R}^{2 \times 3}$:

$$\mathbf{X} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$$

* **Global Reduction (All Elements):**
  $$\text{sum}(\mathbf{X}) = 21, \quad \text{mean}(\mathbf{X}) = 3.5, \quad \max(\mathbf{X}) = 6$$
* **$\text{axis}=0$ (Collapse Rows $\downarrow$):** Reduces along height, producing one value per column:
  $$\text{sum}(\mathbf{X}, \text{axis}=0) = \begin{bmatrix} 1+4 & 2+5 & 3+6 \end{bmatrix} = \begin{bmatrix} 5 & 7 & 9 \end{bmatrix} \in \mathbb{R}^3$$
  $$\text{mean}(\mathbf{X}, \text{axis}=0) = \begin{bmatrix} 2.5 & 3.5 & 4.5 \end{bmatrix}$$
* **$\text{axis}=1$ (Collapse Columns $\to$):** Reduces along width, producing one value per row:
  $$\text{sum}(\mathbf{X}, \text{axis}=1) = \begin{bmatrix} 1+2+3 \\ 4+5+6 \end{bmatrix} = \begin{bmatrix} 6 \\ 15 \end{bmatrix} \to [6, 15] \in \mathbb{R}^2$$
  $$\text{mean}(\mathbf{X}, \text{axis}=1) = [2.0, 5.0]$$

**Multi-axis Reductions on 3D Tensor $(B, H, W) = (2, 3, 4)$:**
$$\begin{aligned}
\text{sum}(\text{axis}=0) &\implies (3, 4) \quad \text{(Batch sum)} \\
\text{sum}(\text{axis}=1) &\implies (2, 4) \quad \text{(Height sum)} \\
\text{sum}(\text{axis}=2) &\implies (2, 3) \quad \text{(Width sum)} \\
\text{sum}(\text{axis}=(1, 2)) &\implies (2,) \quad \text{(Global spatial pooling)}
\end{aligned}$$

### Memory and Performance Considerations

| Operation | Complexity | Access Strategy | Hardware Cache Locality |
| :--- | :--- | :--- | :--- |
| **`.sum()`** | $O(N)$ | Sequential read | 🟢 **Optimal**: Linear memory streaming |
| **`.sum(axis=0)`** | $O(N)$ | Sweeps rows in sequence | 🟢 **High**: Accumulates into row buffer, vectorizes cleanly |
| **`.sum(axis=1)`** | $O(N)$ | Horizontal row reduce | 🟡 **Fair**: Requires horizontal SIMD reduction |
| **`.mean()` / `.max()`** | $O(N)$ | Sequential read | 🟢 **Optimal**: Single streaming pass |

**Why `axis=0` is usually FASTER than `axis=1` (measured ~1.7× speedup on a 4000×4000 array):**
- NumPy reduces over axis 0 by sweeping memory sequentially and accumulating into an output row, which vectorizes cleanly with SIMD instructions.
- `axis=1` is a horizontal reduction within each row, which requires horizontal SIMD lane shuffling.
- The intuition that "column access must be strided and slow" applies to element-at-a-time Python loops, not to NumPy's compiled sequential sweep.
"""


# %% [markdown]
r"""
### Implement: Sum, Mean, Max

Write `forward` for each operation. The inputs are NumPy arrays; return an array. `Tensor` already wrapped scalars and will wrap your result.
"""

# %% nbgrader={"grade": false, "grade_id": "ops-reductions", "solution": true}
#| export
class Sum(Function):
    """Sum over all elements or along an axis. Parameters: axis, keepdims."""
    axis = None
    keepdims = False

    def forward(self, a):
        """
        Sum the array along self.axis.

        TODO: Return np.sum(a, axis=self.axis, keepdims=self.keepdims).

        APPROACH:
        1. self.axis and self.keepdims were set by Tensor.sum (or default to the class attributes)
        2. np.sum does the reduction; pass both parameters straight through

        HINT: axis=None (the default) sums every element.
        """
        ### BEGIN SOLUTION role="scaffold"
        return np.sum(a, axis=self.axis, keepdims=self.keepdims)
        ### END SOLUTION


class Mean(Function):
    """Average over all elements or along an axis. Parameters: axis, keepdims."""
    axis = None
    keepdims = False

    def forward(self, a):
        """
        Average the array along self.axis.

        TODO: Return np.mean(a, axis=self.axis, keepdims=self.keepdims).
        """
        ### BEGIN SOLUTION role="scaffold"
        return np.mean(a, axis=self.axis, keepdims=self.keepdims)
        ### END SOLUTION


class Max(Function):
    """Largest element over all elements or along an axis. Parameters: axis, keepdims."""
    axis = None
    keepdims = False

    def forward(self, a):
        """
        Take the maximum of the array along self.axis.

        TODO: Return np.max(a, axis=self.axis, keepdims=self.keepdims).
        """
        ### BEGIN SOLUTION role="scaffold"
        return np.max(a, axis=self.axis, keepdims=self.keepdims)
        ### END SOLUTION

# %% [markdown]
r"""
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

    # Test mean along axis (per-column statistics)
    col_mean = matrix.mean(axis=0)
    expected_mean = np.array([2.5, 3.5, 4.5], dtype=np.float32)  # [5/2, 7/2, 9/2]
    assert np.allclose(col_mean.data, expected_mean)

    # Test max (finding largest value)
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

    # Test 3D reduction (averaging across spatial dimensions)
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    spatial_mean = tensor_3d.mean(axis=(1, 2))  # Average across spatial dimensions
    assert spatial_mean.shape == (2,)  # One value per batch item

    print("✅ Reduction operations work correctly!")

if __name__ == "__main__":
    test_unit_reduction_operations()

# %% [markdown]
r"""
## 🔧 Integration: Bringing It Together

Let's test how our Tensor operations work together in realistic scenarios. This integration demonstrates that our individual operations combine correctly for complex workflows.

### Linear Transformation Simulation

A common pattern in machine learning and scientific computing is the affine transformation:

$$\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

$$\underbrace{\mathbf{X}}_{(N, D_{\text{in}})} \xrightarrow{\text{Matrix Multiply with } \mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}} \underbrace{\mathbf{X}\mathbf{W}}_{(N, D_{\text{out}})} \xrightarrow{\text{Broadcast Add with } \mathbf{b} \in \mathbb{R}^{D_{\text{out}}}} \underbrace{\mathbf{Y}}_{(N, D_{\text{out}})}$$

**Concrete Numerical Trace:**
Given input $\mathbf{X} \in \mathbb{R}^{2 \times 3}$, weight $\mathbf{W} \in \mathbb{R}^{3 \times 2}$, and bias $\mathbf{b} \in \mathbb{R}^2$:

$$\mathbf{X} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad \mathbf{W} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 0.1 & 0.2 \end{bmatrix}$$

* **Step 1: Inner-Dimension Contraction ($\mathbf{X}\mathbf{W}$)**
  $$\mathbf{X}\mathbf{W} = \begin{bmatrix} 1(0.1)+2(0.3)+3(0.5) & 1(0.2)+2(0.4)+3(0.6) \\ 4(0.1)+5(0.3)+6(0.5) & 4(0.2)+5(0.4)+6(0.6) \end{bmatrix} = \begin{bmatrix} 2.2 & 2.8 \\ 4.9 & 6.4 \end{bmatrix}$$

* **Step 2: Offset Injection via Broadcasting ($+ \mathbf{b}$)**
  $$\mathbf{Y} = \begin{bmatrix} 2.2 & 2.8 \\ 4.9 & 6.4 \end{bmatrix} + \begin{bmatrix} 0.1 & 0.2 \end{bmatrix} = \begin{bmatrix} 2.3 & 3.0 \\ 5.0 & 6.6 \end{bmatrix}$$

This affine transformation pattern is the foundational linear layer underlying MLPs, CNNs, and Transformer projections!

### Why This Integration Matters

This simulation shows how our basic operations combine to create powerful computational building blocks:

- **Matrix Multiplication**: Transforms input features into a new feature space
- **Broadcasting Addition**: Applies offsets efficiently across batches of data
- **Shape Handling**: Ensures data flows correctly through transformation stages
- **Memory Management**: Creates new tensors without corrupting inputs

You'll see this affine transformation pattern used extensively as we build more complex systems in later modules.
"""


# %% [markdown]
r"""
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
    matrix = Tensor(rng.random((size, size)))

    import time

    print(f"\nTesting with {size}×{size} matrix ({matrix.size * BYTES_PER_FLOAT32 / MB_TO_BYTES:.1f} MB)")
    print("-" * 60)

    # Test 1: Row-wise access (cache-friendly)
    # Memory layout: [row0][row1][row2]... stored contiguously
    print("\nTest 1: Row-wise Access (Cache-Friendly)")
    # perf_counter, not time(): the wall clock ticks in ~15 ms steps on
    # Windows, so a fast loop can measure 0 and break the ratio below.
    start = time.perf_counter()
    row_sums = []
    for i in range(size):
        row_sum = matrix.data[i, :].sum()  # Access entire row sequentially
        row_sums.append(row_sum)
    row_time = time.perf_counter() - start
    print(f"   Time: {row_time*1000:.1f}ms")
    print("   Access pattern: Sequential (follows memory layout)")

    # Test 2: Column-wise access (cache-unfriendly)
    # Must jump between rows, poor spatial locality
    print("\nTest 2: Column-wise Access (Cache-Unfriendly)")
    start = time.perf_counter()
    col_sums = []
    for j in range(size):
        col_sum = matrix.data[:, j].sum()  # Access entire column with large strides
        col_sums.append(col_sum)
    col_time = time.perf_counter() - start
    print(f"   Time: {col_time*1000:.1f}ms")
    print(f"   Access pattern: Strided (jumps {size * BYTES_PER_FLOAT32} bytes per element)")

    # Calculate slowdown
    slowdown = col_time / max(row_time, 1e-9)
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE IMPACT:")
    print(f"   Slowdown factor: {slowdown:.2f}× ({slowdown:.1f}× slower)")
    print("   This timing ratio includes loop and reduction overhead; it does not count cache misses")

    # Educational insights
    print("\n💡 KEY INSIGHTS:")
    print("   1. Memory layout matters: Row-major (C-style) storage is sequential")
    print("   2. Cache lines are ~64 bytes: Row access loads nearby elements \"for free\"")
    print("   3. Strided column access can use cache lines less efficiently")
    print(f"   4. This is O(n) algorithm but {slowdown:.1f}× different wall-clock time!")

    print("\n🚀 REAL-WORLD IMPLICATIONS:")
    print("   • Image processing libraries use specific memory formats for cache efficiency")
    print("   • Matrix multiplication optimized with blocking (tile into cache-sized chunks)")
    print(f"   • This run's column/row time ratio is {slowdown:.1f}×; other kernels may behave differently")
    print("   • Hardware-optimized libraries leverage memory layout for better performance")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_memory_layout()


# %% [markdown]
r"""
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
    test_unit_validate_matmul_shapes()
    test_unit_matrix_multiplication()
    test_unit_shape_manipulation()
    test_unit_reduction_operations()

    print("\nRunning integration scenarios...")

    # Test realistic multi-stage computation
    print("🧪 Integration Test: Two-Stage Linear Transformation...")

    # Create input data (2 samples, 3 features)
    x = Tensor([[1, 2, 3], [4, 5, 6]])

    # First stage: 3 inputs → 4 intermediate values
    W1 = Tensor([[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.6, 0.7, 0.8],
                 [0.9, 1.0, 1.1, 1.2]])
    b1 = Tensor([0.1, 0.2, 0.3, 0.4])

    # Forward pass: hidden = xW1 + b1
    hidden = x.matmul(W1) + b1
    assert hidden.shape == (2, 4), f"Expected (2, 4), got {hidden.shape}"

    # Second stage: 4 intermediate → 2 outputs
    W2 = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    b2 = Tensor([0.1, 0.2])

    # Output stage: output = hiddenW2 + b2
    output = hidden.matmul(W2) + b2
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    # Verify data flows correctly (no NaN, reasonable values)
    assert not np.isnan(output.data).any(), "Output contains NaN values"
    assert np.isfinite(output.data).all(), "Output contains infinite values"

    print("✅ Two-stage linear transformation works!")

    # Test complex shape manipulations
    print("🧪 Integration Test: Complex Shape Operations...")
    data = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # Reshape to 3D tensor (simulating batch processing)
    tensor_3d = data.reshape(2, 2, 3)  # (batch=2, height=2, width=3)
    assert tensor_3d.shape == (2, 2, 3)

    # Spatial averaging (collapse height and width)
    pooled = tensor_3d.mean(axis=(1, 2))  # Average across spatial dimensions
    assert pooled.shape == (2,), f"Expected (2,), got {pooled.shape}"

    # Flatten to 2D
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
    print("Run: tito module complete 01")


# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

Now that you've built a complete Tensor class, let's think about its systems-level implications.
Understanding memory layout, scaling behavior, and computational costs helps you make informed
decisions when building production ML systems.

### Question 1: Memory Layout and Cache Performance

How does row-major vs column-major storage affect cache performance in tensor operations?

**Consider**:
- What happens when you access matrix elements sequentially vs. with large strides?
- Why did our analysis show column-wise access being slower than row-wise?
- How would this affect the design of an image processing pipeline's memory layout?

**Key Insight**: Libraries choose specific memory formats because accessing certain dimensions
sequentially has better cache locality. You'll see this principle applied throughout later modules.

### Question 2: Batch Processing and Scaling

If you double the number of samples in a batch, what happens to memory usage? What about
computation time?

**Consider**:
- An affine transformation with input (batch, features): y = xW + b
- Memory for: input tensor, weight matrix, output tensor
- If (32, 784) @ (784, 256) takes 10ms, how long does (64, 784) @ (784, 256) take?

**Key Insight**: Input/output memory scales linearly with batch size, but weight memory stays constant.
Computation time also scales linearly for matrix multiplication.

### Question 3: Data Type Precision and Memory

What's the memory difference between float64 and float32 for a (1000, 1000) tensor?

**Calculate**:
- float64: 8 bytes per element, float32: 4 bytes per element
- Total elements: 1,000,000
- Memory: float64 = 8MB, float32 = 4MB (2x difference)

**Key Insight**: Production systems often use float16 or bfloat16 for 2x memory savings over float32 (2 bytes vs 4),
trading precision for capacity. GPU memory limits (8-16GB) make this critical.

### Question 4: Production Scale Memory

A large-scale model has 175 billion parameters. How much RAM is needed just to store the weights?

**Calculate**:
- Parameters: 175 x 10^9
- Bytes per float32: 4
- Weight memory: 700 GB

**Key Insight**: This is why large-scale systems require significant hardware resources.
You'll explore what "additional training state" means in later modules.

### Question 5: Hardware Awareness

Why do parallel processors strongly prefer operations on large tensors over many small ones?

**Compare**:
- Scenario A: 1000 separate (10, 10) matrix multiplications
- Scenario B: 1 batched (1000, 10, 10) matrix multiplication

**Key Insight**: Computation launch overhead (~5-10 microseconds per launch) dominates for small operations.
Batching amortizes this overhead and maximizes parallelism across processing units.
"""

# %% [markdown]
r"""
## ⭐ Aha Moment: Your Tensor Works Like NumPy

**What you built:** A complete Tensor class with arithmetic operations and matrix multiplication.

**Why it matters:** Your Tensor is the foundation of everything to come. Every ML
operation — from simple addition to complex multi-step computations — will use this class. The fact
that it works exactly like NumPy means you've built something production-ready.

Your Tensor is ready for machine learning operations.
Every operation you just implemented will be used extensively as we build the full framework!
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
r"""
## 🚀 MODULE SUMMARY: Tensor Foundation

Congratulations! You've built the foundational Tensor class that powers all machine learning operations!

### Key Accomplishments
- Built a complete Tensor class with arithmetic operations, matrix multiplication, and shape manipulation
- Implemented broadcasting semantics that match NumPy for automatic shape alignment
- Created reduction operations (sum, mean, max) for aggregating data across dimensions
- Discovered cache performance implications through memory layout analysis
- All tests pass (validated by `test_module()`)

### Systems Insights Discovered
- Memory layout matters: Row-wise access is faster than column-wise due to cache locality
- Broadcasting efficiency: NumPy handles shape alignment without explicit data copying
- Matrix multiplication is the computational foundation of linear transformations
- Shape validation provides clear error messages at minimal performance cost

### Ready for Next Steps
Your Tensor implementation enables all future ML operations.
Export with: `tito module complete 01`

**Next**: Module 02 will add Activations that introduce nonlinearity to your tensors!
"""
