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
# Module 17: Acceleration - Hardware-Aware Optimization

Welcome to Module 17! Compare vectorized operations, blocked multiplication, and GELU allocation patterns, connect them to production kernel fusion, and turn Module 09's convolution loops into a single matrix multiply.

## 🔗 Prerequisites & Progress
**You've Built**: Complete neural network foundation with tensors (01), layers (03), autograd (06), training (08), and CNNs (09)
**You'll Build**: Vectorized inference operations, tiled multiplication, an allocation comparison, and an im2col convolution you can train through
**You'll Enable**: Hardware-efficient execution for production deployment

**Connection Map**:
```
Layers (03) → Training (08) → CNNs (09) → Acceleration (17)
(building blocks) (learning)   (spatial)  (speed up)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement vectorized operations for maximum throughput
2. Compare intermediate allocation costs and explain what true kernel fusion changes
3. Understand the relationship between compute and memory bandwidth
4. Lower a convolution to one matrix multiply with im2col, write its backward pass with col2im, and measure its speed and memory cost
5. Analyze acceleration trade-offs in production systems

Let's optimize for speed!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/17_acceleration/acceleration.ipynb`
**Building Side:** Code exports to `tinytorch.perf.acceleration`

```python
# How to use this module:
from tinytorch.perf.acceleration import vectorized_matmul, fused_gelu, im2col_conv2d
```

**Why this matters:**
- **Learning:** Complete acceleration system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.cuda and torch.backends with optimization components
- **Consistency:** All acceleration operations and optimization components in perf.acceleration
- **Integration:** Works seamlessly with neural network layers for complete performance optimization
"""

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Module 01 (Tensor); Module 09 (Convolutions), whose Conv2d is the reference the im2col convolution must match; Module 14 (Profiling) for the measurement habits

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for performance measurement)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor and Function from Module 01; Module 06 makes Function record gradients)
- `tinytorch.core.spatial` (Conv2d from Module 09, used by the tests as the reference)
- `tinytorch.perf.profiling` (Profiler from Module 14)

**Dependency Flow**:
```
Module 01 (Tensor) → Module 09 (Conv2d) → Module 14 (Profiling) → Module 17 (Acceleration)
     ↓                     ↓                     ↓                       ↓
  Foundation         Loop reference       Measurement Tools     Performance Optimization
```

Students completing this module will have built acceleration techniques
that work with the complete TinyTorch performance optimization stack.
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp perf.acceleration
#| export

import numpy as np
rng = np.random.default_rng(7)
import time
from typing import Any

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor, Function

# Constants for performance measurement
DEFAULT_WARMUP_ITERATIONS = 2  # Default warmup iterations for timing
DEFAULT_TIMING_ITERATIONS = 5  # Default timing iterations for measurement
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes

# %% [markdown]
"""
## 💡 Introduction: The Performance Challenge

Before we learn acceleration techniques, let's understand the performance gap.
Neural networks often underutilize hardware due to:
- Sequential operations (no parallelism)
- Poor memory access patterns (cache misses)
- Missing SIMD (Single Instruction, Multiple Data) opportunities
- Separate operations (memory bandwidth waste)

We'll measure how vectorization and allocation choices affect execution, then examine kernel fusion as a production extension.

### The Two Enemies of Performance

Modern neural networks face two fundamental bottlenecks that limit their speed:

**1. Compute Bound Operations:**
```
CPU/GPU Cores: [====BUSY====] [====BUSY====] [====BUSY====]
Memory Bus:    [---idle---] [---idle---] [---idle---]

When: Matrix multiplication, convolutions
Solution: Vectorization, better algorithms
```

**2. Memory Bound Operations:**
```
CPU/GPU Cores: [--idle--] [--idle--] [--idle--]
Memory Bus:    [========SATURATED========]

When: Element-wise operations, small tensors
Solution: Kernel fusion, memory layout optimization
```

### The Roofline Model: Your Performance Compass

Every processor has fundamental limits:

```
Performance
(GFLOP/s)
    │                    ┌────────────────────────  Peak compute (the roof)
    │                  ╱ │
    │                ╱   │  Compute bound: more FLOPs
    │              ╱     │  per byte no longer buys speed
    │            ╱       │
    │          ╱         │
    │        ╱  Memory   │
    │      ╱    bound    │
    │    ╱   (slope =    │
    │  ╱     bandwidth)  │
    └────────────────────┴────────────────────────  Arithmetic Intensity
     Low               Ridge                 High   (FLOPs/Byte)
```

**Key Insight**: Understand where your operations live on this graph to optimize effectively.

### Why This Module Matters

Performance questions you can investigate:
- How much Python-loop overhead does vectorization remove?
- How much do retained intermediate copies cost?
- When does explicit tiling help or hurt compared with one BLAS call?
- What does it cost in memory to run a convolution as one matrix multiply?
"""

# %% [markdown]
"""
## 📐 Foundations: Vectorization, From Loops to Lightning

### The SIMD Revolution

Modern processors can execute **Single Instruction, Multiple Data** operations:

```
Traditional Loop (Scalar):               SIMD Vectorized:
for i in range(4):        ┌─────┐      ┌─────┬─────┬─────┬─────┐
    c[i] = a[i] + b[i]    │ ALU │  →   │ALU 0│ALU 1│ALU 2│ALU 3│
                          └─────┘      └─────┴─────┴─────┴─────┘
                          1 element     4 elements per cycle
                          per cycle
```

### Memory Access Patterns: The Hidden Performance Killer

```
Sequential Access (FAST):
Memory: [A][B][C][D][E][F][G][H]
Access:  ↓  ↓  ↓  ↓  → Cache friendly

Strided Access (SLOWER):
Memory: [A][ ][B][ ][C][ ][D][ ]
Access:  ↓     ↓     ↓     ↓   → Cache misses

Random Access (SLOWEST):
Memory: [A][B][C][D][E][F][G][H]
Access:  ↓     ↑  ↓     ↑       → Cache chaos
```

### Matrix Multiplication: The King of Vectorization

Matrix multiplication is **perfectly suited** for vectorization:

```
Matrix A (M×K) × Matrix B (K×N) = Matrix C (M×N)

Computation Pattern:
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ a₁₁ a₁₂ a₁₃ a₁₄ │ × │ b₁₁ b₁₂ b₁₃ b₁₄ │ = │ c₁₁ c₁₂ c₁₃ c₁₄ │
│ a₂₁ a₂₂ a₂₃ a₂₄ │   │ b₂₁ b₂₂ b₂₃ b₂₄ │   │ c₂₁ c₂₂ c₂₃ c₂₄ │
│ a₃₁ a₃₂ a₃₃ a₃₄ │   │ b₃₁ b₃₂ b₃₃ b₃₄ │   │ c₃₁ c₃₂ c₃₃ c₃₄ │
│ a₄₁ a₄₂ a₄₃ a₄₄ │   │ b₄₁ b₄₂ b₄₃ b₄₄ │   │ c₄₁ c₄₂ c₄₃ c₄₄ │
└─────────────────┘   └─────────────────┘   └─────────────────┘

For c₁₁: Row₁ · Column₁ = a₁₁×b₁₁ + a₁₂×b₂₁ + a₁₃×b₃₁ + a₁₄×b₄₁
                                    ↑
                              VECTORIZABLE!
```

**Why vectorization wins:**
- **High arithmetic intensity**: 2N³ FLOPs for only 3N² elements of data
- **Predictable memory access**: Sequential row/column reads
- **Parallelizable**: Independent dot products
- **Cache-friendly**: Data reuse in inner loops
"""

# %% nbgrader={"grade": false, "grade_id": "vectorized-matmul", "solution": true}
#| export

def vectorized_matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    High-performance matrix multiplication using vectorized operations.

    This implementation leverages optimized BLAS libraries that use:
    - SIMD instructions for parallel computation
    - Cache-blocking for memory efficiency
    - Multi-threading for CPU parallelization

    TODO: Implement vectorized inference matrix multiplication

    APPROACH:
    1. Validate shapes are compatible for matrix multiplication
    2. Use NumPy's optimized dot product (calls BLAS GEMM)
    3. Return result wrapped in Tensor

    Args:
        a: First tensor for multiplication (M×K or batch×M×K)
        b: Second tensor for multiplication (K×N or batch×K×N)

    Returns:
        Result tensor of shape (M×N or batch×M×N)

    EXAMPLE:
    Matrix multiplication visualization:
    >>> a = Tensor([[1, 2], [3, 4]])  # 2×2
    >>> b = Tensor([[5, 6], [7, 8]])  # 2×2
    >>> result = vectorized_matmul(a, b)
    >>> print(result.data)
    [[19. 22.]    # [1×5+2×7, 1×6+2×8] = [19, 22]
     [43. 50.]]   # [3×5+4×7, 3×6+4×8] = [43, 50]

    PERFORMANCE CHARACTERISTICS:
    - Time Complexity: O(N³) but highly optimized
    - Space Complexity: O(N²) for result
    - Arithmetic Intensity: 2N³ FLOPs / (3N² × 4) bytes = N/6 (good for large N)

    HINTS:
    - Check a.shape[-1] == b.shape[-2] for inner dimension match
    - Use np.matmul() for batch support and optimization
    - Trust BLAS to handle the vectorization magic
    """
    ### BEGIN SOLUTION role="scaffold"
    # Input validation for matrix multiplication
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError(
            f"Matrix multiplication requires 2D+ tensors\n"
            f"  ❌ Got shapes {a.shape} and {b.shape} ({len(a.shape)}D and {len(b.shape)}D tensors)\n"
            f"  💡 Matrix multiplication computes dot products between rows and columns, which requires at least 2D tensors\n"
            f"  🔧 Add dimensions with reshape: a.reshape(1, {a.shape[-1] if len(a.shape) >= 1 else 'n'}) for a row vector"
        )

    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Matrix multiplication shape mismatch: {a.shape} @ {b.shape}\n"
            f"  ❌ Inner dimensions don't match: a.shape[-1]={a.shape[-1]} vs b.shape[-2]={b.shape[-2]}\n"
            f"  💡 For A @ B, each row of A (length {a.shape[-1]}) must match each column of B (length {b.shape[-2]})\n"
            f"  🔧 Try: b.reshape({a.shape[-1]}, -1) or a.reshape(-1, {b.shape[-2]})"
        )

    # Use NumPy's highly optimized matrix multiplication
    # This calls BLAS GEMM (General Matrix Multiply), which uses:
    # - SIMD vectorization for parallel arithmetic
    # - Cache blocking for memory efficiency
    # - Multi-threading on multicore systems
    result_data = np.matmul(a.data, b.data)

    return Tensor(result_data)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Vectorized Matrix Multiplication

This test validates that replacing explicit loops with a single vectorized call
produces identical results.

**What we're testing**: Correctness of batched matmul and its shape validation
**Why it matters**: Vectorization is only a win if the answer is unchanged -- a
faster wrong answer is worthless
**Expected**: Matches hand-computed products, rejects mismatched inner dimensions
"""

# %% nbgrader={"grade": true, "grade_id": "test-vectorized-matmul", "locked": true, "points": 10}
def test_unit_vectorized_matmul():
    """🧪 Test vectorized matrix multiplication implementation."""
    print("🧪 Unit Test: Vectorized Matrix Multiplication...")

    # Test basic 2D multiplication
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    result = vectorized_matmul(a, b)

    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(result.data, expected), f"Basic matmul failed: expected {expected}, got {result.data}"

    # Test batch multiplication (3D tensors)
    batch_size, m, k, n = 2, 3, 4, 5
    a_batch = Tensor(rng.standard_normal((batch_size, m, k)))
    b_batch = Tensor(rng.standard_normal((batch_size, k, n)))
    result_batch = vectorized_matmul(a_batch, b_batch)

    assert result_batch.shape == (batch_size, m, n), f"Wrong batch shape: {result_batch.shape}"

    # Test broadcasting (different batch dimensions)
    a_single = Tensor(rng.standard_normal((m, k)))
    b_batch = Tensor(rng.standard_normal((batch_size, k, n)))
    result_broadcast = vectorized_matmul(a_single, b_batch)

    assert result_broadcast.shape == (batch_size, m, n), f"Broadcasting failed: {result_broadcast.shape}"

    # Test error cases
    try:
        vectorized_matmul(Tensor([1, 2, 3]), Tensor([4, 5]))  # 1D tensors
        assert False, "Should reject 1D tensors"
    except ValueError as e:
        assert "2D+" in str(e)

    try:
        vectorized_matmul(Tensor([[1, 2]]), Tensor([[1], [2], [3]]))  # Shape mismatch
        assert False, "Should reject incompatible shapes"
    except ValueError as e:
        assert "shape mismatch" in str(e).lower()

    print("✅ vectorized_matmul works correctly!")

if __name__ == "__main__":
    test_unit_vectorized_matmul()

# %% [markdown]
"""
## 🏗️ Implementation: Kernel Fusion

### The Memory Bandwidth Crisis

Consider this computation: `y = gelu(x * weight + bias)`.
Assume `x`, `weight`, and `bias` are each 4 GB arrays. Count one read of
each input and one write of each output per operation, ignoring cache reuse.
Treat GELU itself as one compiled operation in both paths.

**Naive Implementation (Memory Intensive):**
```
Step 1: temp1 = x * weight     → Read 8GB, Write 4GB
Step 2: temp2 = temp1 + bias   → Read 8GB, Write 4GB
Step 3: y = gelu(temp2)        → Read 4GB, Write 4GB
                                 Total: 32GB memory traffic!
```

**Fused Implementation (Memory Efficient):**
```
Single Step: y = gelu(x * weight + bias)  → Read 12GB, Write 4GB
                                            Total: 16GB memory traffic!
                                            50% less modeled memory traffic!
```

### Understanding GELU: The Smooth Activation

GELU (Gaussian Error Linear Unit) is used in transformers because it's **smooth** (differentiable everywhere):

```
Activation Functions Compared:

ReLU: max(0, x)                 GELU: x·Φ(x)                    Sigmoid: 1/(1+e⁻ˣ)
     │            ╱                  │            ╱                1 ┤         ╭─────────
     │           ╱                   │           ╱                  │        ╱
     │          ╱                    │          ╱              0.5 ┤       ╱
     │         ╱                     │         ╱                    │      ╱
   0 ┼────────┼──────────── x      0 ┼──╮     ┼──────────── x     0 ┼─────╯─┼──────────── x
              x=0                       ╰───╱ x=0                          x=0
Kink at 0: the gradient jumps    Smooth everywhere, with a small    Smooth, but flat at both
from 0 to 1                      dip below 0 just left of x=0       ends (gradient saturates)
```

**GELU Formula**: `GELU(x) = x * Φ(x)` where Φ is the standard normal CDF

**Fast Approximation**: `GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

### Kernel Fusion Strategy

This diagram describes a compiled fused kernel. Our NumPy comparison below
removes intermediate Tensor wrappers, but does not implement this single traversal.

```
Unfused Operations:                    Fused Operation:
┌─────────────────┐                   ┌────────────────────┐
│ x³, ×c, + x     │ → temp1..3        │                    │
└─────────────────┘                   │                    │
┌─────────────────┐                   │                    │
│ ×√(2/π), tanh   │ → temp4..5        │   All operations   │
└─────────────────┘                   │   combined in      │
┌─────────────────┐                   │   single kernel    │
│ 1 + …, x × …    │ → temp6..7        │                    │
└─────────────────┘                   │                    │
┌─────────────────┐                   │                    │
│ 0.5 × …         │ → result          │                    │
└─────────────────┘                   └────────────────────┘

8 arrays written (7 temporaries)       1 array written
```
"""

# %% nbgrader={"grade": false, "grade_id": "fused-gelu", "solution": true}
#| export

def fused_gelu(x: Tensor) -> Tensor:
    """
    Compact GELU expression that avoids retaining intermediate Tensor copies.

    GELU combines the benefits of ReLU and sigmoid:
    - Smooth everywhere (ReLU is continuous but not differentiable at 0)
    - Non-saturating for positive values (unlike sigmoid)
    - Probabilistic interpretation: x * P(X ≤ x) where X ~ N(0,1)

    Mathematical Definition:
    GELU(x) = x * Φ(x) where Φ(x) is the standard normal CDF

    Fast Approximation (used here):
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    TODO: Compute GELU without retaining intermediate Tensor objects

    APPROACH:
    1. Compute all intermediate values in a single expression
    2. Avoid retaining intermediate Tensor wrappers (NumPy still allocates arrays)
    3. Let NumPy's broadcasting handle vectorization

    Args:
        x: Input tensor to apply GELU activation

    Returns:
        GELU-activated tensor (same shape as input)

    EXAMPLE:
    >>> x = Tensor([-2, -1, 0, 1, 2])
    >>> result = fused_gelu(x)
    >>> print(result.data)
    [-0.04540231 -0.15880801  0.          0.84119199  1.95459769]
    # Notice: smooth transition through 0, positive bias

    MEMORY EFFICIENCY:
    - Unfused: 7 temporary arrays × input_size × 4 bytes, each kept alive as a Tensor
    - Compact: one NumPy expression, which still creates temporary arrays
    - A compiled fused kernel can read the input once and write the output once

    HINTS:
    - Use np.sqrt(2.0 / np.pi) for the constant
    - NumPy evaluates each array operation separately; it does not fuse this expression
    - These raw NumPy helpers are inference examples and do not record autograd
    """
    ### BEGIN SOLUTION
    # Mathematical constant for GELU approximation
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    # Fused GELU computation - all operations in single expression
    # By computing the full expression in a single line, we avoid creating intermediate
    # Tensor objects. Note: NumPy still allocates temporary arrays internally —
    # real kernel fusion requires compiled frameworks like XLA or torch.compile.
    result_data = 0.5 * x.data * (
        1.0 + np.tanh(sqrt_2_over_pi * (x.data + 0.044715 * x.data**3))
    )

    return Tensor(result_data)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Fused GELU

This test validates the fused GELU activation against the mathematical
properties GELU must satisfy.

**What we're testing**: GELU(0) = 0, increasing nonnegative outputs, and the negative dip
**Why it matters**: Reducing retained Tensor copies must preserve the activation;
our NumPy expression still executes multiple array operations
**Expected**: Exact zero at the origin, a nonmonotonic negative region, correct tails
"""

# %% nbgrader={"grade": true, "grade_id": "test-fused-gelu", "locked": true, "points": 10}
def test_unit_fused_gelu():
    """🧪 Test 🔬 Test fused GELU activation implementation."""
    print("🧪 Unit Test: Fused GELU...")

    # Test basic properties
    x = Tensor([-3, -1, 0, 1, 3])
    result = fused_gelu(x)

    # GELU(0) = 0 (exact property)
    assert abs(result.data[2]) < 1e-6, f"GELU(0) should be 0, got {result.data[2]}"

    # GELU increases for nonnegative inputs, but dips on the negative side.
    assert result.data[4] > result.data[3] > result.data[2], \
        "GELU should increase for nonnegative inputs"
    assert result.data[0] > result.data[1] < result.data[2], \
        "GELU should retain its nonmonotonic negative dip"

    # GELU has positive bias (unlike ReLU)
    assert result.data[3] > 0.8, "GELU(1) should be close to 1"
    assert result.data[1] > -0.2, "GELU(-1) should be slightly negative"

    # Test numerical stability with extreme values
    x_extreme = Tensor([-10, -5, 0, 5, 10])
    result_extreme = fused_gelu(x_extreme)

    assert not np.any(np.isnan(result_extreme.data)), "No NaN values allowed"
    assert not np.any(np.isinf(result_extreme.data)), "No infinite values allowed"

    # Test large tensor processing
    x_large = Tensor(rng.standard_normal((1000, 1000)).astype(np.float32))
    result_large = fused_gelu(x_large)

    assert result_large.shape == x_large.shape, "Shape preservation failed"
    assert result_large.data.dtype == np.float32, "Data type preservation failed"

    # Test that positive inputs are mostly preserved (GELU ≈ x for large positive x)
    x_positive = Tensor([5.0])
    result_positive = fused_gelu(x_positive)
    assert result_positive.data[0] > 4.9, "Large positive values should be nearly preserved"

    print("✅ fused_gelu works correctly!")

if __name__ == "__main__":
    test_unit_fused_gelu()

# %% [markdown]
"""
### 🧪 Unit Test: Fusion Performance

Compare the compact GELU expression with a version that retains each intermediate Tensor.

**What we're testing**: Both implementations produce equivalent values
**Why it matters**: Optimizations must preserve the answer before their timing matters
**Expected**: Identical outputs; any timing difference depends on the machine
"""

# %% nbgrader={"grade": false, "grade_id": "unfused-gelu", "solution": true}
#| export
def unfused_gelu(x: Tensor) -> Tensor:
    """
    Deliberately unfused GELU implementation for performance comparison.

    This version creates multiple intermediate tensors to simulate
    the memory bandwidth overhead of unfused operations.

    TODO: Implement GELU with explicit intermediate steps

    APPROACH:
    1. Break computation into individual steps
    2. Create temporary Tensor objects for each step
    3. This simulates real memory allocation overhead

    Args:
        x: Input tensor

    Returns:
        GELU-activated tensor (same shape as input)

    EXAMPLE:
    >>> x = Tensor([0.5, 1.0, -0.5])
    >>> result = unfused_gelu(x)
    >>> print(result.shape)
    (3,)  # Same as input

    PERFORMANCE IMPACT:
    - Creates 7 temporary arrays
    - Each array allocation/deallocation has overhead
    - More memory bandwidth usage
    - Potential cache misses between operations

    HINTS:
    - Create each step as: temp = Tensor(operation)
    - This forces memory allocation for educational comparison
    """
    ### BEGIN SOLUTION role="scaffold"
    # Unfused version - creates many intermediate arrays
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    # Each operation creates a temporary array (simulating kernel launches)
    temp1 = Tensor(x.data**3)  # x³
    temp2 = Tensor(0.044715 * temp1.data)  # 0.044715 * x³
    temp3 = Tensor(x.data + temp2.data)  # x + 0.044715 * x³
    temp4 = Tensor(sqrt_2_over_pi * temp3.data)  # √(2/π) * (...)
    temp5 = Tensor(np.tanh(temp4.data))  # tanh(...)
    temp6 = Tensor(1.0 + temp5.data)  # 1 + tanh(...)
    temp7 = Tensor(x.data * temp6.data)  # x * (1 + tanh(...))
    result = Tensor(0.5 * temp7.data)  # 0.5 * x * (...)

    return result
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Kernel Fusion Performance Impact

This test compares the compact NumPy expression with retained Tensor intermediates.

**What we're testing**: Timed comparison of fused vs unfused GELU after warmup
**Why it matters**: Fewer Tensor copies may reduce allocation overhead. Timing
is machine-dependent; NumPy does not compile this expression into one kernel
**Expected**: Numerically equivalent results; timing is reported, never graded
"""

# %% nbgrader={"grade": true, "grade_id": "test-fusion-speedup", "locked": true, "points": 10}
def test_unit_fusion_speedup():
    """🧪 Measure the performance impact of kernel fusion."""
    print("🧪 Unit Test: Kernel Fusion Performance Impact...")

    # Create moderately large tensor for meaningful timing
    size = 2000
    x = Tensor(rng.standard_normal((size, size)).astype(np.float32))
    warmup_iterations = DEFAULT_WARMUP_ITERATIONS
    timing_iterations = DEFAULT_TIMING_ITERATIONS

    # Warmup both implementations
    for _ in range(warmup_iterations):
        _ = unfused_gelu(x)
        _ = fused_gelu(x)

    # Time unfused version
    start = time.perf_counter()
    for _ in range(timing_iterations):
        result_unfused = unfused_gelu(x)
    unfused_time = time.perf_counter() - start

    # Time fused version
    start = time.perf_counter()
    for _ in range(timing_iterations):
        result_fused = fused_gelu(x)
    fused_time = time.perf_counter() - start

    # Verify numerical correctness
    assert np.allclose(result_unfused.data, result_fused.data, atol=1e-6), \
        "Fused and unfused implementations must be numerically equivalent"

    # Calculate performance metrics
    speedup = unfused_time / fused_time if fused_time > 0 else 1.0
    unfused_per_elem = (unfused_time / timing_iterations) / (size * size) * 1e9  # ns per element
    fused_per_elem = (fused_time / timing_iterations) / (size * size) * 1e9

    print(f"📊 Kernel Fusion Performance Analysis:")
    print(f"   Tensor size: {size}×{size} = {size*size:,} elements")
    print(f"   Unfused time: {unfused_time/timing_iterations*1000:.2f} ms")
    print(f"   Fused time:   {fused_time/timing_iterations*1000:.2f} ms")
    print(f"   Speedup: {speedup:.2f}× faster")
    print(f"   Per-element: {unfused_per_elem:.1f} ns → {fused_per_elem:.1f} ns")

    # Timing does not measure memory traffic. A one-read/one-write model applies
    # to a compiled fused kernel, not to this sequence of NumPy operations.
    print("   NumPy still creates temporary arrays; bandwidth is not measured here.")

    # Interpret results
    if speedup > 1.5:
        print("🚀 Excellent! Fewer retained Tensor intermediates providing significant speedup")
    elif speedup > 1.1:
        print("✅ Good! Fewer retained Tensor intermediates providing measurable benefit")
    else:
        print("⚠️  Limited speedup - may be compute-bound or small tensor size")

    print("✅ Fusion performance analysis completed!")

if __name__ == "__main__":
    test_unit_fusion_speedup()

# %% [markdown]
"""
## 🏗️ Cache-Aware Matrix Multiplication

For large matrices that don't fit in cache, we need **tiling** (also called blocking).
This breaks the computation into cache-sized chunks for better performance.

### Why Cache Awareness Matters

Modern processors have a memory hierarchy:
```
L1 Cache:   32-64 KB   (fastest, 1-4 cycles)
L2 Cache:   256 KB-1MB (fast, 10-20 cycles)
L3 Cache:   8-32 MB    (moderate, 40-75 cycles)
Main RAM:   8-64 GB    (slow, 100-300 cycles)
```

When matrices are larger than cache, we get **cache misses** that slow us down dramatically.
Tiling keeps working set in cache for maximum reuse.

### Sizing a Tile

Computing one output tile touches three blocks at once: a tile of A, a tile of
B, and the tile of C being accumulated. So the working set is roughly
`3 x tile_size^2 x 4 bytes` for float32. Solving for a 32 KB L1 cache:

```
3 x t^2 x 4 <= 32,768   ->   t <= 52
```

which is why 32 and 64 are the tile sizes you see in real kernels.

### What You Are and Are Not Building

You are writing the **loop order** -- the three tile loops that decide which
blocks are in flight together. The multiply inside each block is still a NumPy
call, because writing scalar loops in Python would be thousands of times slower
and would teach nothing about cache behavior.

Be honest about the benchmark: your tiled version will be **slower** than
`vectorized_matmul`, because NumPy already hands the whole matrix to a BLAS
kernel that does this same blocking in tuned C with prefetching and register
tiling. The point is to see the mechanism BLAS is using, not to beat it.
"""

# %% nbgrader={"grade": false, "grade_id": "tiled-matmul", "solution": true}
#| export

def tiled_matmul(a: Tensor, b: Tensor, tile_size: int = 64) -> Tensor:
    """
    Cache-aware matrix multiplication using tiling (also called blocking).

    Splits the output into tile_size x tile_size blocks and computes each block
    from matching strips of A and B, so the working set stays small enough to
    live in cache while it is being reused.

    TODO: Implement blocked matrix multiplication.

    APPROACH:
    1. Validate that both inputs are 2D and that their inner dimensions agree
    2. Allocate the output C as an (M, N) array of zeros
    3. Loop over tiles of i (rows of C), then tiles of j (columns of C)
    4. For each output tile, loop over tiles of k and ACCUMULATE the block
       products into that tile: C[i, j] += A[i, k] @ B[k, j]
    5. Wrap the finished array in a Tensor

    Args:
        a: First matrix (M x K)
        b: Second matrix (K x N)
        tile_size: Block edge length; the working set is three tile_size x
            tile_size blocks (default: 64)

    Returns:
        Result matrix (M x N)

    EXAMPLE:
    >>> a = Tensor([[1, 2], [3, 4]])
    >>> b = Tensor([[5, 6], [7, 8]])
    >>> print(tiled_matmul(a, b, tile_size=1).data)
    [[19. 22.]
     [43. 50.]]

    PERFORMANCE CHARACTERISTICS:
    - Same FLOP count as the naive order; only the memory access pattern changes
    - Three blocks of tile_size^2 floats must fit in cache together, so a good
      tile_size satisfies 3 * tile_size^2 * 4 bytes < L1/L2 size
    - This exposes block reuse, but Python tile loops may be slower than one
      BLAS call, whose implementation already tiles internally

    HINTS:
    - Use min(start + tile_size, limit) so the last tile can be a partial one
    - Accumulate with += into a slice of C; each output tile is touched once
      per k-tile
    - The innermost block product is still NumPy's `@`. The lesson here is the
      LOOP ORDER, not scalar arithmetic -- Python-level scalar loops would be
      thousands of times slower and teach nothing about cache behavior
    """
    ### BEGIN SOLUTION
    # Input validation
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError(
            f"Tiled matrix multiplication requires 2D tensors\n"
            f"  ❌ Got shapes {a.shape} and {b.shape} ({len(a.shape)}D and {len(b.shape)}D tensors)\n"
            f"  💡 Tiling partitions a matrix into 2D blocks, so there must be exactly two axes\n"
            f"  🔧 Add dimensions with reshape: tensor.reshape(1, -1) for a row vector or tensor.reshape(-1, 1) for a column"
        )

    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Tiled matrix multiplication shape mismatch: {a.shape} @ {b.shape}\n"
            f"  ❌ Inner dimensions don't match: a.shape[-1]={a.shape[-1]} vs b.shape[-2]={b.shape[-2]}\n"
            f"  💡 Each tile of A's columns must align with tiles of B's rows for block multiplication\n"
            f"  🔧 Reshape to align: b.reshape({a.shape[-1]}, -1) or transpose if dimensions are swapped"
        )

    if not isinstance(tile_size, (int, np.integer)) or tile_size < 1:
        raise ValueError(
            f"Tile size must be at least 1, got {tile_size}\n"
            f"  💡 The tile is the block edge length, so it has to be a positive number of rows/columns"
        )

    A, B = a.data, b.data
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), dtype=A.dtype)

    # Three loops over TILES. The inner block product is one NumPy matmul on
    # data small enough to stay in cache for the whole (i, j) tile.
    for i0 in range(0, M, tile_size):
        i1 = min(i0 + tile_size, M)
        for j0 in range(0, N, tile_size):
            j1 = min(j0 + tile_size, N)
            for k0 in range(0, K, tile_size):
                k1 = min(k0 + tile_size, K)
                C[i0:i1, j0:j1] += A[i0:i1, k0:k1] @ B[k0:k1, j0:j1]

    return Tensor(C)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Tiled Matrix Multiplication

This test validates that blocking the loops preserves the result.

**What we're testing**: Tiled output matches the vectorized reference across
several tile sizes, plus shape validation
**Why it matters**: Tiling reorders the accumulation, and float addition is not
associative, so "close enough" has to be defined rather than assumed
**Expected**: Agreement within float32 reassociation tolerance at every tile size
"""

# %% nbgrader={"grade": true, "grade_id": "test-tiled-matmul", "locked": true, "points": 10}
def test_unit_tiled_matmul():
    """🧪 Test cache-aware tiled matrix multiplication."""
    print("🧪 Unit Test: Tiled Matrix Multiplication...")

    # Test correctness against vectorized version
    a = Tensor(rng.standard_normal((128, 128)).astype(np.float32))
    b = Tensor(rng.standard_normal((128, 128)).astype(np.float32))

    result_tiled = tiled_matmul(a, b, tile_size=32)
    result_reference = vectorized_matmul(a, b)

    assert np.allclose(result_tiled.data, result_reference.data, atol=1e-5), \
        "Tiled and vectorized results should match"

    # Test different tile sizes
    for tile_size in [16, 32, 64]:
        result = tiled_matmul(a, b, tile_size=tile_size)
        assert result.shape == (128, 128), f"Wrong shape for tile_size={tile_size}"

    # Test shape validation
    try:
        wrong_a = Tensor(rng.standard_normal((128, 64)).astype(np.float32))
        wrong_b = Tensor(rng.standard_normal((128, 64)).astype(np.float32))
        tiled_matmul(wrong_a, wrong_b)
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        assert "shape mismatch" in str(e).lower()

    print("✅ tiled_matmul works correctly!")

if __name__ == "__main__":
    test_unit_tiled_matmul()

# %% [markdown]
"""
## 🏗️ Convolution as Matrix Multiplication: im2col

Module 09's `Conv2d` computes every output pixel as a dot product written out in
Python: seven nested loops over batch, output channel, output row, output column,
kernel row, kernel column, and input channel. That was the right way to learn
what a convolution does. It is also why the CNN milestones are slow.

### Why the Loops Are Slow

Count the work for one layer of the CIFAR-10 milestone: a batch of 4 images,
32 input channels, 64 output channels, a 32×32 feature map, and a 3×3 kernel.

```
multiply-adds = batch × out_ch × out_h × out_w × in_ch × k_h × k_w
              = 4 × 64 × 32 × 32 × 32 × 3 × 3
              = 75,497,472
```

In `_convolve_loops`, each of those 75 million multiply-adds is a separate
Python operation. The arithmetic is not the problem; the interpreter is. The
same 75 million multiply-adds inside one BLAS call finish in milliseconds.

### The im2col Trick

Look at a single output pixel. It is the dot product of two lists of
`in_ch × k_h × k_w` numbers: the input patch under the filter, flattened, and
the filter itself, flattened. So the whole layer is a matrix multiply in
disguise:

```
Patch matrix (one row per output position)      Filter matrix (one column per filter)
┌──────────────────────────────┐                ┌──────────────┐
│ patch at (n=0, oh=0, ow=0)   │                │ f₀  f₁ … f₆₃ │
│ patch at (n=0, oh=0, ow=1)   │       @        │  ↓   ↓    ↓  │   =   output rows
│ ...                          │                │ (in_ch·k·k   │       (one per position,
│ patch at (n=3, oh=31, ow=31) │                │  rows each)  │        one column per filter)
└──────────────────────────────┘                └──────────────┘
  (N·H_out·W_out) × (in_ch·k·k)                  (in_ch·k·k) × out_ch
```

Building the patch matrix is called **im2col** ("image to columns"; we store the
patches as rows, the transposed convention, so the product needs no transpose of
the big matrix). After one matmul, reshaping the `(N·H_out·W_out) × out_ch`
result gives back the `(N, out_ch, H_out, W_out)` feature map.

A 1-channel 3×3 input with a 2×2 kernel and stride 1 has four output positions,
so four patches:

```
Input          Patches (rows of the patch matrix)
┌───┬───┬───┐  position (0,0): [1, 2, 4, 5]
│ 1 │ 2 │ 3 │  position (0,1): [2, 3, 5, 6]
├───┼───┼───┤  position (1,0): [4, 5, 7, 8]
│ 4 │ 5 │ 6 │  position (1,1): [5, 6, 8, 9]
├───┼───┼───┤
│ 7 │ 8 │ 9 │  Pixel 5 appears in all four rows.
└───┴───┴───┘
```

### The Memory Price

That repetition is the cost. Every interior pixel is copied once for each
kernel position that covers it, up to `k_h × k_w` times. For the CIFAR layer
above, the input is `4 × 32 × 32 × 32 × 4 bytes = 524 KB`, and the patch matrix
is `(4 × 32 × 32) × (32 × 3 × 3) × 4 bytes = 4.7 MB`, nine times larger. im2col
trades memory for speed. On a laptop that is an easy trade; on a microcontroller
with 256 KB of RAM it may not fit at all.

### Building the Patch Matrix Without Looping Over Pixels

The obvious way to build the patch matrix loops over every output position and
copies one patch at a time, which is a Python loop over `N × H_out × W_out`
positions and gives back much of the speed. Turn it around and loop over the
**kernel offsets** instead. For kernel element `(i, j)`, the input values it
touches at every output position form one strided slice of the padded input:

```
padded[:, :, i : i + stride*H_out : stride, j : j + stride*W_out : stride]
        ↑  ↑                                                   shape (N, C, H_out, W_out)
      all images, all channels, all output positions at once
```

A 3×3 kernel needs nine slice copies, whatever the image size. The only care
needed is the column order: `weight.reshape(out_ch, -1)` flattens each filter in
`(channel, kernel row, kernel column)` order, so each patch must be flattened in
the same order or the dot products pair the wrong numbers.

### What You Are and Are Not Building

Like the other helpers in this module, `im2col_conv2d` is an inference path: it
returns a new Tensor without recording autograd, so it computes the same forward
pass as Module 09's `Conv2d` but cannot train one. The next section adds the
reverse step, **col2im**, and turns the convolution into a differentiable
`Function` that can.
"""

# %% nbgrader={"grade": false, "grade_id": "im2col", "solution": true}
#| export

def im2col(x: Tensor, kernel_size: int, stride: int = 1, padding: int = 0) -> Tensor:
    """
    Unroll every convolution patch of a batch of images into one row of a matrix.

    Row r of the result is the input patch for output position r, with positions
    ordered (image, output row, output column). Each row is flattened in
    (channel, kernel row, kernel column) order, the same order in which
    weight.reshape(out_channels, -1) flattens a Conv2d filter.

    TODO: Build the patch matrix by looping over kernel offsets, not output pixels

    APPROACH:
    1. Validate that x is 4D (batch, channels, height, width) and that the kernel
       fits inside the padded input
    2. Zero-pad the two spatial dimensions, exactly as Conv2d._apply_padding does
    3. Compute out_h = (H + 2*padding - kernel_size) // stride + 1, and out_w likewise
    4. Allocate cols with shape (N, C, kernel_size, kernel_size, out_h, out_w)
    5. For each kernel offset (i, j), copy the strided slice
       padded[:, :, i : i + stride*out_h : stride, j : j + stride*out_w : stride]
       into cols[:, :, i, j, :, :]
    6. Move the position axes to the front with transpose(0, 4, 5, 1, 2, 3) and
       reshape to (N * out_h * out_w, C * kernel_size * kernel_size)

    Args:
        x: Input images, shape (N, C, H, W)
        kernel_size: Edge length of the square kernel
        stride: Step between neighboring patches (default: 1)
        padding: Zeros added on each side of both spatial dimensions (default: 0)

    Returns:
        Patch matrix of shape (N * out_h * out_w, C * kernel_size * kernel_size)

    EXAMPLE:
    >>> x = Tensor(np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3))
    >>> print(im2col(x, kernel_size=2).data)
    [[1. 2. 4. 5.]
     [2. 3. 5. 6.]
     [4. 5. 7. 8.]
     [5. 6. 8. 9.]]

    MEMORY CHARACTERISTICS:
    - The patch matrix holds N * out_h * out_w * C * kernel_size^2 values
    - With stride 1 and "same" padding that is about kernel_size^2 times the input
    - The loop runs kernel_size^2 times, independent of the image size

    HINTS:
    - np.pad with ((0, 0), (0, 0), (padding, padding), (padding, padding)) pads
      only height and width
    - A slice with a step, a[start:stop:step], selects every stride-th position
    - Check your column order with the 3x3 example above before moving on
    """
    ### BEGIN SOLUTION
    if len(x.shape) != 4:
        raise ValueError(
            f"im2col requires a 4D tensor (batch, channels, height, width)\n"
            f"  ❌ Got shape {x.shape} ({len(x.shape)}D)\n"
            f"  💡 Convolution slides over the last two axes of every channel of every image, so all four axes must be present\n"
            f"  🔧 Add a batch axis for a single image: x.reshape(1, *x.shape)"
        )

    N, C, H, W = x.shape
    if kernel_size < 1 or stride < 1 or padding < 0:
        raise ValueError(
            f"Invalid im2col parameters: kernel_size={kernel_size}, stride={stride}, padding={padding}\n"
            f"  💡 kernel_size and stride must be at least 1, and padding cannot be negative"
        )
    if kernel_size > H + 2 * padding or kernel_size > W + 2 * padding:
        raise ValueError(
            f"Kernel does not fit the padded input\n"
            f"  ❌ kernel_size={kernel_size} but the padded input is {H + 2 * padding}×{W + 2 * padding}\n"
            f"  💡 Every patch must lie inside the padded image, so the kernel can be no larger than it\n"
            f"  🔧 Use a smaller kernel or more padding"
        )

    padded = np.pad(x.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                    mode='constant', constant_values=0)
    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1

    # One strided slice per kernel offset: slice (i, j) holds, for every output
    # position at once, the input value that kernel element (i, j) multiplies.
    cols = np.empty((N, C, kernel_size, kernel_size, out_h, out_w), dtype=padded.dtype)
    for i in range(kernel_size):
        for j in range(kernel_size):
            cols[:, :, i, j, :, :] = padded[:, :,
                                            i:i + stride * out_h:stride,
                                            j:j + stride * out_w:stride]

    # Positions first (n, oh, ow), then the patch in (c, ki, kj) order to match
    # weight.reshape(out_channels, -1).
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, C * kernel_size * kernel_size)
    return Tensor(cols)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: im2col

This test validates the patch matrix against patches written out by hand.

**What we're testing**: Row contents and order, column order, output shape with
stride and padding, and rejection of non-image input
**Why it matters**: A patch matrix with its columns in the wrong order still
multiplies without error; it just pairs the wrong pixels with the wrong weights
**Expected**: The four hand-written patches of the 3×3 example, and shapes that
follow the convolution output formula
"""

# %% nbgrader={"grade": true, "grade_id": "test-im2col", "locked": true, "points": 10}
def test_unit_im2col():
    """🧪 Test the im2col patch matrix."""
    print("🧪 Unit Test: im2col...")

    # The worked example: one 3×3 image, 2×2 kernel, stride 1, no padding
    x = Tensor(np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3))
    cols = im2col(x, kernel_size=2)
    expected = np.array([[1, 2, 4, 5],
                         [2, 3, 5, 6],
                         [4, 5, 7, 8],
                         [5, 6, 8, 9]], dtype=np.float32)
    assert cols.shape == (4, 4), f"Expected a 4×4 patch matrix, got {cols.shape}"
    assert np.array_equal(cols.data, expected), f"Patches are wrong:\n{cols.data}"

    # Column order must be (channel, kernel row, kernel column): with two channels,
    # the first four columns come from channel 0 and the next four from channel 1
    two_channel = Tensor(np.stack([np.arange(9), 100 + np.arange(9)]).reshape(1, 2, 3, 3).astype(np.float32))
    first_patch = im2col(two_channel, kernel_size=2).data[0]
    assert np.array_equal(first_patch, [0, 1, 3, 4, 100, 101, 103, 104]), \
        f"Columns must be ordered (channel, kernel row, kernel column), got {first_patch}"

    # Shape follows the convolution output formula with stride and padding
    batch = Tensor(rng.standard_normal((2, 3, 8, 8)).astype(np.float32))
    padded_cols = im2col(batch, kernel_size=3, stride=1, padding=1)
    assert padded_cols.shape == (2 * 8 * 8, 3 * 3 * 3), f"Wrong shape with padding: {padded_cols.shape}"
    strided_cols = im2col(batch, kernel_size=3, stride=2, padding=1)
    assert strided_cols.shape == (2 * 4 * 4, 3 * 3 * 3), f"Wrong shape with stride 2: {strided_cols.shape}"

    # Padding supplies zeros: the first patch of a padded image has a zero border
    corner = im2col(Tensor(np.ones((1, 1, 3, 3), dtype=np.float32)), kernel_size=3, padding=1).data[0]
    assert np.array_equal(corner, [0, 0, 0, 0, 1, 1, 0, 1, 1]), f"Padding is wrong: {corner}"

    # Non-image input is rejected
    try:
        im2col(Tensor(np.ones((3, 8, 8))), kernel_size=3)
        assert False, "Should reject a 3D tensor"
    except ValueError as e:
        assert "4D" in str(e)

    print("✅ im2col works correctly!")

if __name__ == "__main__":
    test_unit_im2col()

# %% [markdown]
"""
### Convolution Through One Matrix Multiply

With the patch matrix built, the convolution itself is three steps: flatten the
filters into a matrix, multiply, and put the result back into image layout.

```
cols      = im2col(x, k, stride, padding)          (N·H_out·W_out, C·k·k)
w_matrix  = weight.reshape(out_ch, C·k·k).T        (C·k·k, out_ch)
out       = cols @ w_matrix (+ bias)               (N·H_out·W_out, out_ch)
feature map = out.reshape(N, H_out, W_out, out_ch)
                 .transpose(0, 3, 1, 2)            (N, out_ch, H_out, W_out)
```

The multiply is your `vectorized_matmul`, so all of the speed comes from the
vectorized matrix multiply you already wrote. The final transpose matters: the
matmul produces one row per position with the channels last, and Conv2d's
output puts the channels second.
"""

# %% nbgrader={"grade": false, "grade_id": "im2col-conv2d", "solution": true}
#| export

def im2col_conv2d(x: Tensor, weight: Tensor, bias: Tensor = None,
                  stride: int = 1, padding: int = 0) -> Tensor:
    """
    2D convolution computed as one matrix multiply over an im2col patch matrix.

    Computes the same forward pass as Module 09's Conv2d with the same weight,
    bias, stride, and padding, but replaces its seven nested loops with one
    vectorized_matmul. Inference only: the result does not record autograd.

    TODO: Compute the convolution with im2col and vectorized_matmul

    APPROACH:
    1. Validate that weight is 4D (out_ch, in_ch, k, k) with a square kernel and
       that in_ch matches x's channel count
    2. Build the patch matrix with im2col(x, k, stride, padding)
    3. Reshape the weight to (out_ch, in_ch * k * k) and transpose it
    4. Multiply with vectorized_matmul; add the bias to every row if one is given
    5. Reshape to (N, out_h, out_w, out_ch) and transpose to (N, out_ch, out_h, out_w)

    Args:
        x: Input images, shape (N, C, H, W)
        weight: Filters, shape (out_ch, C, k, k), as in Conv2d.weight
        bias: Optional per-filter bias, shape (out_ch,), as in Conv2d.bias
        stride: Step between neighboring patches (default: 1)
        padding: Zeros added on each side of both spatial dimensions (default: 0)

    Returns:
        Feature map of shape (N, out_ch, out_h, out_w)

    EXAMPLE:
    >>> conv = Conv2d(3, 16, kernel_size=3, padding=1)
    >>> x = Tensor(np.random.randn(2, 3, 8, 8))
    >>> fast = im2col_conv2d(x, conv.weight, conv.bias, padding=1)
    >>> print(fast.shape)
    (2, 16, 8, 8)
    >>> np.allclose(fast.data, conv(x).data)
    True

    PERFORMANCE CHARACTERISTICS:
    - Same multiply-add count as the loop version
    - All of them run inside one BLAS call instead of the Python interpreter
    - Extra memory: the patch matrix, about k^2 times the input

    HINTS:
    - N, H_out, and W_out can be recovered from x.shape and the patch count, or
      recomputed with the output-size formula
    - Adding a (out_ch,) bias to an (rows, out_ch) array broadcasts across rows
    - Compare against Conv2d on a small input before timing anything
    """
    ### BEGIN SOLUTION
    if len(weight.shape) != 4 or weight.shape[2] != weight.shape[3]:
        raise ValueError(
            f"im2col_conv2d requires a 4D weight with a square kernel\n"
            f"  ❌ Got weight shape {weight.shape}\n"
            f"  💡 Conv2d stores its filters as (out_channels, in_channels, k, k)\n"
            f"  🔧 Pass conv.weight from a Conv2d layer"
        )
    if len(x.shape) != 4 or x.shape[1] != weight.shape[1]:
        raise ValueError(
            f"Input channels do not match the filters\n"
            f"  ❌ Input shape {x.shape}, weight shape {weight.shape}\n"
            f"  💡 Each filter has one slice per input channel, so x.shape[1] must equal weight.shape[1]\n"
            f"  🔧 Check that the input is (N, C, H, W) with C = {weight.shape[1]}"
        )

    N, _, H, W = x.shape
    out_ch, _, k, _ = weight.shape
    out_h = (H + 2 * padding - k) // stride + 1
    out_w = (W + 2 * padding - k) // stride + 1

    cols = im2col(x, kernel_size=k, stride=stride, padding=padding)
    w_matrix = Tensor(weight.data.reshape(out_ch, -1).T)

    out = vectorized_matmul(cols, w_matrix).data          # (N*out_h*out_w, out_ch)
    if bias is not None:
        out = out + bias.data

    return Tensor(out.reshape(N, out_h, out_w, out_ch).transpose(0, 3, 1, 2))
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: im2col Convolution

This test validates the matrix-multiply convolution against Module 09's loops.

**What we're testing**: Agreement with `Conv2d` for padding, stride, and no-bias
cases, output layout, and channel validation
**Why it matters**: An optimization is only correct if it computes the same
function as the reference; Module 09's `Conv2d` is that reference
**Expected**: Outputs match `Conv2d` within float tolerance in every case
"""

# %% nbgrader={"grade": true, "grade_id": "test-im2col-conv2d", "locked": true, "points": 10}
def test_unit_im2col_conv2d():
    """🧪 Test im2col convolution against Module 09's Conv2d."""
    print("🧪 Unit Test: im2col Convolution...")

    from tinytorch.core.spatial import Conv2d

    x = Tensor(rng.standard_normal((2, 3, 8, 8)).astype(np.float32))

    # Same padding, stride 1, with bias
    conv = Conv2d(3, 4, kernel_size=3, padding=1)
    conv.bias.data[:] = rng.standard_normal(4)
    fast = im2col_conv2d(x, conv.weight, conv.bias, stride=1, padding=1)
    reference = conv(x)
    assert fast.shape == reference.shape == (2, 4, 8, 8), f"Wrong output shape: {fast.shape}"
    assert np.allclose(fast.data, reference.data, atol=1e-5), \
        f"Differs from Conv2d by up to {np.abs(fast.data - reference.data).max():.2e}"

    # Stride 2, no padding, no bias
    conv_strided = Conv2d(3, 5, kernel_size=3, stride=2, bias=False)
    fast_strided = im2col_conv2d(x, conv_strided.weight, None, stride=2, padding=0)
    reference_strided = conv_strided(x)
    assert fast_strided.shape == reference_strided.shape == (2, 5, 3, 3), \
        f"Wrong strided shape: {fast_strided.shape}"
    assert np.allclose(fast_strided.data, reference_strided.data, atol=1e-5), \
        "Strided im2col convolution differs from Conv2d"

    # Channel layout: the output channel axis is second, as in Conv2d
    one_filter = Tensor(np.zeros((2, 3, 3, 3)))
    one_filter.data[1, 0, 1, 1] = 1.0  # filter 1 copies channel 0 unchanged
    copied = im2col_conv2d(x, one_filter, padding=1)
    assert np.allclose(copied.data[:, 1], x.data[:, 0]), "Output channels are in the wrong axis"
    assert np.allclose(copied.data[:, 0], 0.0), "Filter 0 is all zeros, so its channel must be zero"

    # Mismatched channels are rejected
    try:
        im2col_conv2d(Tensor(np.ones((1, 2, 8, 8))), conv.weight, padding=1)
        assert False, "Should reject input with the wrong number of channels"
    except ValueError as e:
        assert "channels" in str(e).lower()

    print("✅ im2col_conv2d works correctly!")

if __name__ == "__main__":
    test_unit_im2col_conv2d()

# %% [markdown]
"""
## 🏗️ Training Through im2col: col2im and the Backward Pass

`im2col_conv2d` computes the forward pass, but it wraps a NumPy result in a new
Tensor, so autograd cannot see through it. A model that uses it can predict but
not learn. Training needs the backward pass, and im2col makes most of it easy.

### Three Gradients From One Matrix Multiply

Write the forward pass in matrix form. With `cols` the patch matrix
`(R, C·k·k)` where `R = N·H_out·W_out`, and `W` the filters flattened and
transposed to `(C·k·k, out_ch)`:

```
out_rows = cols @ W + b                          (R, out_ch)
```

Let `G` be the gradient of the loss with respect to `out_rows`, the incoming
`grad_output` moved into the same `(R, out_ch)` layout. The chain rule for a
matrix multiply, which you used in Module 06, gives:

```
grad_W    = cols.T @ G          (C·k·k, out_ch)   one matmul
grad_b    = G.sum(axis=0)       (out_ch,)         one reduction
grad_cols = G @ W.T             (R, C·k·k)        one matmul
```

Two of the three gradients are matrix multiplies the same size as the forward
pass, so they get the same speedup. The third is not yet the gradient we need:
it is the gradient with respect to the **patch matrix**, and the layer's input
is the image.

### col2im: Putting Patch Gradients Back Into the Image

Every pixel was copied into several rows of `cols`, once for each kernel
position that covered it. A pixel's gradient is therefore the **sum** of the
gradients of all its copies. col2im is im2col run backwards with the copy turned
into an addition:

```
im2col (forward)                          col2im (backward)
cols[.., i, j, ..] = padded[slice(i, j)]   padded_grad[slice(i, j)] += cols_grad[.., i, j, ..]
```

It loops over the same `k × k` kernel offsets with the same strided slices. The
`+=` is the whole point: with `=` each pixel would keep only the gradient from
the last patch that touched it. After the loop, cropping off the padding gives
the input gradient, because padded zeros are constants and have no gradient.

On the 3×3 example with a 2×2 kernel, sending a gradient of 1 from every patch
entry back through col2im counts how many patches covered each pixel:

```
┌───┬───┬───┐
│ 1 │ 2 │ 1 │   corners: one patch
├───┼───┼───┤   edges:   two patches
│ 2 │ 4 │ 2 │   center:  all four
├───┼───┼───┤
│ 1 │ 2 │ 1 │
└───┴───┴───┘
```

### The Memory the Backward Pass Keeps

`grad_W = cols.T @ G` needs the patch matrix from the forward pass. Keeping it
alive between forward and backward holds the `k × k` times larger buffer for
every convolution in the model until its backward pass runs. The alternative is
to throw `cols` away and rebuild it from the saved input during backward, which
trades memory for a second im2col. This implementation keeps it.
"""

# %% nbgrader={"grade": false, "grade_id": "col2im", "solution": true}
#| export

def col2im(cols: Tensor, x_shape: tuple, kernel_size: int, stride: int = 1, padding: int = 0) -> Tensor:
    """
    Add every row of a patch-gradient matrix back into the image it came from.

    The reverse of im2col: where im2col copied each input value into every
    patch that covers it, col2im sums the gradients of those copies into one
    gradient per input value. Overlapping patches therefore accumulate.

    TODO: Scatter-add the patch gradients back into image layout

    APPROACH:
    1. Recover N, C, H, W from x_shape and compute out_h and out_w with the
       same formula as im2col
    2. Undo im2col's final reshape and transpose: reshape cols to
       (N, out_h, out_w, C, k, k), then transpose(0, 3, 4, 5, 1, 2) to
       (N, C, k, k, out_h, out_w)
    3. Allocate a zero array for the padded image, (N, C, H + 2p, W + 2p)
    4. For each kernel offset (i, j), ADD cols6[:, :, i, j, :, :] into the
       strided slice padded[:, :, i : i + stride*out_h : stride,
       j : j + stride*out_w : stride]
    5. Crop the padding off and wrap the result in a Tensor

    Args:
        cols: Patch gradients, shape (N * out_h * out_w, C * k * k)
        x_shape: Shape of the original input, (N, C, H, W)
        kernel_size: Edge length of the square kernel
        stride: Step between neighboring patches (default: 1)
        padding: Zeros that im2col added on each side (default: 0)

    Returns:
        Image-shaped gradient, shape x_shape

    EXAMPLE:
    >>> ones = Tensor(np.ones((4, 4), dtype=np.float32))
    >>> print(col2im(ones, (1, 1, 3, 3), kernel_size=2).data[0, 0])
    [[1. 2. 1.]
     [2. 4. 2.]
     [1. 2. 1.]]

    HINTS:
    - Use += in the loop; = keeps only the last patch's contribution
    - Within one offset (i, j) the slice touches each pixel at most once, so
      the in-place add over the slice is safe
    - If padding is 0, the crop is the whole array
    """
    ### BEGIN SOLUTION
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1
    expected = (N * out_h * out_w, C * kernel_size * kernel_size)
    if tuple(cols.shape) != expected:
        raise ValueError(
            f"Patch matrix does not match the image it should come from\n"
            f"  ❌ Got cols shape {cols.shape}, expected {expected} for x_shape={x_shape}, "
            f"kernel_size={kernel_size}, stride={stride}, padding={padding}\n"
            f"  💡 col2im must use the same x_shape, kernel_size, stride, and padding as the im2col it reverses\n"
            f"  🔧 Pass the parameters the forward pass used"
        )

    # Undo im2col's layout: rows back to (n, oh, ow), columns back to (c, i, j)
    cols6 = cols.data.reshape(N, out_h, out_w, C, kernel_size, kernel_size).transpose(0, 3, 4, 5, 1, 2)

    padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols6.dtype)
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Accumulate: a pixel covered by several patches collects all their gradients
            padded[:, :,
                   i:i + stride * out_h:stride,
                   j:j + stride * out_w:stride] += cols6[:, :, i, j, :, :]

    return Tensor(padded[:, :, padding:padding + H, padding:padding + W])
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: col2im

This test validates col2im as the exact reverse of im2col.

**What we're testing**: Patch-count accumulation on the 3×3 example, the
adjoint identity `<im2col(x), c> = <x, col2im(c)>`, and padding removal
**Why it matters**: A col2im that copies instead of adds produces gradients of
the right shape and the wrong size, and training silently goes wrong
**Expected**: The 1-2-1 / 2-4-2 coverage counts, equal inner products, and an
output with the input's shape
"""

# %% nbgrader={"grade": true, "grade_id": "test-col2im", "locked": true, "points": 10}
def test_unit_col2im():
    """🧪 Test col2im as the reverse of im2col."""
    print("🧪 Unit Test: col2im...")

    # Coverage counts: a gradient of 1 from every patch entry
    counts = col2im(Tensor(np.ones((4, 4), dtype=np.float32)), (1, 1, 3, 3), kernel_size=2)
    expected = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    assert counts.shape == (1, 1, 3, 3), f"Wrong output shape: {counts.shape}"
    assert np.array_equal(counts.data[0, 0], expected), \
        f"Overlapping patches must add up, got:\n{counts.data[0, 0]}"

    # Adjoint identity: col2im is the transpose of im2col, so for any x and c,
    # sum(im2col(x) * c) == sum(x * col2im(c)). Checked with stride and padding.
    for stride, padding in ((1, 0), (1, 1), (2, 1)):
        x = Tensor(rng.standard_normal((2, 3, 7, 7)))
        cols = im2col(x, kernel_size=3, stride=stride, padding=padding)
        c = Tensor(rng.standard_normal(cols.shape))
        back = col2im(c, x.shape, kernel_size=3, stride=stride, padding=padding)
        assert back.shape == x.shape, f"col2im must return the input shape, got {back.shape}"
        lhs = float(np.sum(cols.data * c.data))
        rhs = float(np.sum(x.data * back.data))
        assert np.isclose(lhs, rhs, rtol=1e-4), \
            f"col2im is not the reverse of im2col at stride={stride}, padding={padding}: {lhs:.4f} vs {rhs:.4f}"

    # Mismatched parameters are rejected
    try:
        col2im(Tensor(np.ones((4, 4))), (1, 1, 3, 3), kernel_size=3)
        assert False, "Should reject a patch matrix that does not match the parameters"
    except ValueError as e:
        assert "does not match" in str(e)

    print("✅ col2im works correctly!")

if __name__ == "__main__":
    test_unit_col2im()

# %% [markdown]
"""
### A Differentiable im2col Convolution

With col2im in hand, the convolution becomes a `Function` in the same shape as
Module 09's `Conv2dFunction`: `forward` receives NumPy arrays and returns the
output array, `backward` receives the output gradient and returns one gradient
per input. `Function.apply` records the operation, so `loss.backward()` reaches
it like any other.

```
forward(x, weight, bias)                         backward(grad_output)
────────────────────────                         ─────────────────────
cols = im2col(x)          ── saved ──►           G = grad_output as (R, out_ch)
W = weight → (C·k·k, out_ch)  ── saved ──►       grad_W    = cols.T @ G  → weight shape
out = cols @ W + bias                            grad_b    = G.sum(axis=0)
→ (N, out_ch, H_out, W_out)                      grad_x    = col2im(G @ W.T)
```

`stride` and `padding` arrive as keyword parameters of `apply`, which the base
class stores as attributes, exactly as `layer` does for `Conv2dFunction`.
"""

# %% nbgrader={"grade": false, "grade_id": "im2col-conv2d-function", "solution": true}
#| export

class Im2colConv2dFunction(Function):
    """
    2D convolution as a differentiable operation: im2col + one matmul forward,
    two matmuls + col2im backward.

    Usage:
        out = Im2colConv2dFunction.apply(x, weight, bias, stride=1, padding=1)
        out = Im2colConv2dFunction.apply(x, weight, stride=1, padding=1)   # no bias
    """

    def forward(self, x, weight, bias=None):
        """
        Convolve with one matrix multiply and save what backward needs.

        TODO: Compute the convolution and save the patch matrix and flattened filters

        APPROACH:
        1. Read out_ch and k from weight.shape; compute out_h and out_w
        2. Build the patch matrix: im2col(Tensor(x), k, self.stride, self.padding).data
        3. Flatten the filters: weight.reshape(out_ch, -1).T
        4. Save both on self (self.cols, self.w_matrix) with the output shape
        5. Multiply, add the bias if given, and return the array in
           (N, out_ch, out_h, out_w) layout

        EXAMPLE:
        >>> conv = Conv2d(3, 8, kernel_size=3, padding=1)
        >>> out = Im2colConv2dFunction.apply(x, conv.weight, conv.bias, stride=1, padding=1)
        >>> out.shape
        (2, 8, 16, 16)

        HINT: The arrays arriving here are NumPy arrays, not Tensors.
        """
        ### BEGIN SOLUTION
        N, _, H, W = x.shape
        out_ch, _, k, _ = weight.shape
        out_h = (H + 2 * self.padding - k) // self.stride + 1
        out_w = (W + 2 * self.padding - k) // self.stride + 1

        self.cols = im2col(Tensor(x), k, self.stride, self.padding).data
        self.w_matrix = weight.reshape(out_ch, -1).T
        self.out_shape = (N, out_h, out_w, out_ch)

        out = self.cols @ self.w_matrix
        if bias is not None:
            out = out + bias
        return out.reshape(self.out_shape).transpose(0, 3, 1, 2)
        ### END SOLUTION

    def backward(self, grad_output):
        """
        Turn the output gradient into gradients for the input, weight, and bias.

        TODO: Compute grad_x, grad_weight, and (if there is a bias) grad_bias

        APPROACH:
        1. Move grad_output from (N, out_ch, out_h, out_w) to (R, out_ch):
           transpose(0, 2, 3, 1), then reshape(-1, out_ch)
        2. grad_W = self.cols.T @ G, then transpose and reshape to weight's shape
        3. grad_cols = G @ self.w_matrix.T, then col2im it back to x's shape
        4. grad_b = G.sum(axis=0)
        5. Return one gradient per input: (grad_x, grad_weight) or
           (grad_x, grad_weight, grad_b)

        EXAMPLE:
        >>> out = Im2colConv2dFunction.apply(x, conv.weight, conv.bias, stride=1, padding=1)
        >>> out.sum().backward()
        >>> conv.weight.grad.shape
        (8, 3, 3, 3)

        HINTS:
        - self.inputs holds the input Tensors: (x, weight) or (x, weight, bias)
        - The transpose in step 1 undoes the one at the end of forward
        """
        ### BEGIN SOLUTION
        x, weight = self.inputs[0], self.inputs[1]
        out_ch, _, k, _ = weight.shape

        G = np.asarray(grad_output).transpose(0, 2, 3, 1).reshape(-1, out_ch)

        grad_weight = (self.cols.T @ G).T.reshape(weight.shape)
        grad_cols = G @ self.w_matrix.T
        grad_x = col2im(Tensor(grad_cols), x.shape, k, self.stride, self.padding).data

        if len(self.inputs) > 2:
            return grad_x, grad_weight, G.sum(axis=0)
        return grad_x, grad_weight
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: im2col Convolution Gradients

This test validates the im2col backward pass against Module 09's loops.

**What we're testing**: Output, input gradient, weight gradient, and bias
gradient all match `Conv2d` on the same weights, with and without stride
**Why it matters**: A convolution that computes the right output with the wrong
gradients trains a different model; only a gradient check catches it
**Expected**: Every gradient agrees with Conv2d's backward within float tolerance
"""

# %% nbgrader={"grade": true, "grade_id": "test-im2col-conv2d-function", "locked": true, "points": 15}
def test_unit_im2col_conv2d_function():
    """🧪 Test im2col convolution gradients against Module 09's Conv2d."""
    print("🧪 Unit Test: im2col Convolution Gradients...")

    from tinytorch.core.spatial import Conv2d

    def grads_of(output, tensors):
        # A fixed random projection makes every output element matter differently
        projection = Tensor(np.random.default_rng(11).standard_normal(output.shape))
        (output * projection).sum().backward()
        return [np.asarray(getattr(t.grad, "data", t.grad)) for t in tensors]

    for stride, padding in ((1, 1), (2, 0)):
        conv = Conv2d(3, 4, kernel_size=3, stride=stride, padding=padding)
        conv.bias.data[:] = rng.standard_normal(4)
        x_data = rng.standard_normal((2, 3, 7, 7)).astype(np.float32)

        # Reference: Module 09's loops
        x_ref = Tensor(x_data, requires_grad=True)
        out_ref = conv(x_ref)
        ref = grads_of(out_ref, [x_ref, conv.weight, conv.bias])

        # im2col path on copies of the same parameters
        x = Tensor(x_data, requires_grad=True)
        w = Tensor(conv.weight.data.copy(), requires_grad=True)
        b = Tensor(conv.bias.data.copy(), requires_grad=True)
        out = Im2colConv2dFunction.apply(x, w, b, stride=stride, padding=padding)
        assert out._grad_fn is not None, \
            "Output has no _grad_fn: backward() would never reach this convolution"
        assert np.allclose(out.data, out_ref.data, atol=1e-5), \
            f"Forward pass differs from Conv2d at stride={stride}, padding={padding}"
        got = grads_of(out, [x, w, b])

        for name, g, r in zip(("input", "weight", "bias"), got, ref):
            assert g.shape == r.shape, f"{name} gradient shape {g.shape}, expected {r.shape}"
            assert np.allclose(g, r, atol=1e-4), \
                f"{name} gradient differs from Conv2d by up to {np.abs(g - r).max():.2e} " \
                f"at stride={stride}, padding={padding}"

    # Without a bias there are exactly two gradients
    x = Tensor(rng.standard_normal((1, 2, 5, 5)), requires_grad=True)
    w = Tensor(rng.standard_normal((3, 2, 3, 3)), requires_grad=True)
    out = Im2colConv2dFunction.apply(x, w, stride=1, padding=0)
    out.sum().backward()
    assert x.grad is not None and w.grad is not None, "Gradients must reach input and weight without a bias"

    print("✅ Im2colConv2dFunction works correctly!")

if __name__ == "__main__":
    test_unit_im2col_conv2d_function()

# %% [markdown]
"""
## 🔧 Integration: Measuring Acceleration Gains with Profiler

Now let's use the **Profiler** tool you built in Module 14 to measure the actual performance improvements from vectorization. This demonstrates the full workflow: build profiling tools (M14), apply optimizations (M15-M17), measure gains.

This is how professional ML engineers work: profile → optimize → measure → repeat.
"""

# %% nbgrader={"grade": false, "grade_id": "demo-profiler-acceleration", "solution": false}
# Import Profiler from Module 14 (Module 17 comes after Module 14)
from tinytorch.perf.profiling import Profiler

def explore_acceleration_with_profiler():
    """📊 Demonstrate acceleration gains using Profiler from Module 14."""

    print("📊 Measuring Acceleration Gains with Profiler")
    print("=" * 70)

    profiler = Profiler()

    # Create two simple models: one slow (loop-based), one fast (vectorized)
    class SlowLinear:
        """Linear layer using explicit loops (slow)."""
        def __init__(self, in_features, out_features):
            self.weight = Tensor(rng.standard_normal((in_features, out_features)).astype(np.float32) * 0.01)

        def forward(self, x):
            # Explicit loop implementation (for demonstration)
            batch_size = x.shape[0]
            out_features = self.weight.shape[1]
            result = np.zeros((batch_size, out_features), dtype=np.float32)

            for i in range(batch_size):
                for j in range(out_features):
                    for k in range(x.shape[1]):
                        result[i, j] += x.data[i, k] * self.weight.data[k, j]

            return Tensor(result)

    class FastLinear:
        """Linear layer using vectorized matmul (fast)."""
        def __init__(self, in_features, out_features):
            self.weight = Tensor(rng.standard_normal((in_features, out_features)).astype(np.float32) * 0.01)

        def forward(self, x):
            # Vectorized implementation
            return vectorized_matmul(x, self.weight)

    in_features, out_features = 128, 64
    batch_size = 32

    # Create models
    slow_model = SlowLinear(in_features, out_features)
    fast_model = FastLinear(in_features, out_features)
    fast_model.weight.data[:] = slow_model.weight.data

    # Create input
    input_tensor = Tensor(rng.standard_normal((batch_size, in_features)).astype(np.float32))

    np.testing.assert_allclose(fast_model.forward(input_tensor).data,
                               slow_model.forward(input_tensor).data, rtol=1e-4, atol=1e-6)
    print("\n🐢 BEFORE: Loop-based implementation")
    print("-" * 70)

    # Both models do exactly the same arithmetic: one multiply and one add per
    # (batch, in, out) triple. We count it here rather than calling
    # profiler.count_flops, because that dispatches on the class name and these
    # local classes are neither 'Linear' nor 'Sequential'. It would silently fall
    # through to prod(input_shape) and report 4,096 instead of 524,288.
    total_flops = 2 * batch_size * in_features * out_features

    # Measure slow model
    slow_latency = profiler.measure_latency(slow_model, input_tensor, warmup=3, iterations=10)

    print(f"   Latency: {slow_latency:.2f} ms")
    print(f"   FLOPs: {total_flops:,}")
    print(f"   Throughput: {total_flops / (slow_latency / 1000) / 1e9:.2f} GFLOP/s")

    print("\n🚀 AFTER: Vectorized implementation")
    print("-" * 70)

    # Measure fast model. Same FLOP count: the arithmetic is identical, only the
    # execution differs. That is the whole point of the comparison.
    fast_latency = profiler.measure_latency(fast_model, input_tensor, warmup=3, iterations=10)

    print(f"   Latency: {fast_latency:.2f} ms")
    print(f"   FLOPs: {total_flops:,}")
    print(f"   Throughput: {total_flops / (fast_latency / 1000) / 1e9:.2f} GFLOP/s")

    print("\n📈 ACCELERATION GAINS")
    print("=" * 70)
    speedup = slow_latency / fast_latency
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Time saved: {slow_latency - fast_latency:.2f} ms per inference")
    print(f"   Throughput improvement: {speedup:.1f}x more inferences/second")

    print("\n💡 Key Insight:")
    print(f"   Vectorization with numpy.matmul leverages optimized BLAS libraries")
    print(f"   that use SIMD instructions and cache-friendly memory access patterns.")
    print(f"   This is why {speedup:.0f}x speedups are possible with the same FLOPs!")
    print("\n✅ This is the power of acceleration: same math, different execution!")

if __name__ == "__main__":
    explore_acceleration_with_profiler()

# %% [markdown]
"""
## 📊 Systems Analysis: Performance Scaling Patterns

Let's analyze how our acceleration techniques perform across different scenarios and understand their scaling characteristics.
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-vectorization", "solution": false}
def analyze_vectorization_scaling():
    """📊 Analyze vectorization performance across different tensor sizes."""
    print("📊 Analyzing vectorization scaling behavior...")
    print("AI and GB/s use idealized byte counts, not measured memory traffic.")

    # Test sizes spanning different cache regimes
    sizes = [64, 128, 256, 512, 1024, 2048]

    print("\n🔍 Vectorization Scaling Analysis:")
    print("┌─────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("│  Size   │ Time (ms)   │ GFLOPS      │ Bandwidth   │ Arith. Int. │")
    print("│         │             │             │ (GB/s)      │ (FLOP/byte) │")
    print("├─────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

    for size in sizes:
        # Create test matrices
        a = Tensor(rng.standard_normal((size, size)).astype(np.float32))
        b = Tensor(rng.standard_normal((size, size)).astype(np.float32))

        # Warm up
        for _ in range(2):
            _ = vectorized_matmul(a, b)

        # Time vectorized implementation
        iterations = max(1, 100 // (size // 64))  # Fewer iterations for larger sizes
        start = time.perf_counter()
        for _ in range(iterations):
            result = vectorized_matmul(a, b)
        elapsed = (time.perf_counter() - start) / iterations

        # Calculate performance metrics
        flops = 2 * size**3  # 2N³ FLOPs for matrix multiplication
        gflops = flops / (elapsed * 1e9)

        bytes_accessed = 3 * size * size * 4  # 3 matrices × size² × 4 bytes
        bandwidth = bytes_accessed / (elapsed * 1e9)

        # Arithmetic intensity grows with N (2N³ FLOPs over 12N² bytes = N/6):
        # bigger matmuls do more work per byte moved, so they become compute-bound
        intensity = flops / bytes_accessed

        print(f"│ {size:6d}  │ {elapsed*1000:9.2f}   │ {gflops:9.1f}   │ {bandwidth:9.1f}   │ {intensity:9.1f}   │")

    print("└─────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

    print(f"\n💡 Vectorization insights:")
    print(f"   • Small matrices: Limited by overhead and cache effects")
    print(f"   • Medium matrices: Sweet spot for cache reuse")
    print(f"   • Large matrices: Cache reuse and compute throughput both matter")
    print(f"   • BLAS libraries automatically optimize for each size regime")
    print("🚀 Vectorization effectiveness depends on problem size and hardware")

if __name__ == "__main__":
    analyze_vectorization_scaling()

# %% nbgrader={"grade": false, "grade_id": "analyze-arithmetic-intensity", "solution": false}
def analyze_arithmetic_intensity():
    """📊 Demonstrate the roofline model with different operations."""
    print("📊 Analyzing arithmetic intensity patterns...")
    print("AI and GB/s below use idealized byte counts, not measured memory traffic.")
    print("GELU assumes a compiled fused kernel; our NumPy version allocates temporaries.")

    size = 1024
    iterations = 10

    operations = []

    # Create test data
    x = Tensor(rng.standard_normal((size, size)).astype(np.float32))
    y = Tensor(rng.standard_normal((size, size)).astype(np.float32))

    print("\n🎯 Arithmetic Intensity Analysis:")
    print("┌─────────────────────┬─────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Operation           │ AI      │ Time (ms)   │ GFLOPS      │ GB/s        │")
    print("│                     │(FLOPs/B)│             │             │             │")
    print("├─────────────────────┼─────────┼─────────────┼─────────────┼─────────────┤")

    # 1. Element-wise addition (very low arithmetic intensity)
    start = time.perf_counter()
    for _ in range(iterations):
        _ = Tensor(x.data + y.data)
    add_time = (time.perf_counter() - start) / iterations

    add_flops = size * size  # One addition per element
    add_bytes = 3 * size * size * 4  # Read x, read y, write result
    add_ai = add_flops / add_bytes
    add_gflops = add_flops / (add_time * 1e9)
    add_bandwidth = add_bytes / (add_time * 1e9)

    print(f"│ Element-wise Add    │ {add_ai:7.3f} │ {add_time*1000:9.2f}   │ {add_gflops:9.1f}   │ {add_bandwidth:9.1f}   │")

    # 2. Element-wise multiply (still low, but slightly higher)
    start = time.perf_counter()
    for _ in range(iterations):
        _ = Tensor(x.data * y.data)
    mul_time = (time.perf_counter() - start) / iterations

    mul_flops = size * size
    mul_bytes = 3 * size * size * 4
    mul_ai = mul_flops / mul_bytes
    mul_gflops = mul_flops / (mul_time * 1e9)
    mul_bandwidth = mul_bytes / (mul_time * 1e9)

    print(f"│ Element-wise Mult   │ {mul_ai:7.3f} │ {mul_time*1000:9.2f}   │ {mul_gflops:9.1f}   │ {mul_bandwidth:9.1f}   │")

    # 3. GELU (medium arithmetic intensity)
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fused_gelu(x)
    gelu_time = (time.perf_counter() - start) / iterations

    gelu_flops = size * size * 8  # Approximate: x³, add, mul, tanh, etc.
    gelu_bytes = 2 * size * size * 4  # Read x, write result
    gelu_ai = gelu_flops / gelu_bytes
    gelu_gflops = gelu_flops / (gelu_time * 1e9)
    gelu_bandwidth = gelu_bytes / (gelu_time * 1e9)

    print(f"│ Fused GELU          │ {gelu_ai:7.3f} │ {gelu_time*1000:9.2f}   │ {gelu_gflops:9.1f}   │ {gelu_bandwidth:9.1f}   │")

    # 4. Matrix multiplication (high arithmetic intensity)
    start = time.perf_counter()
    for _ in range(iterations):
        _ = vectorized_matmul(x, y)
    matmul_time = (time.perf_counter() - start) / iterations

    matmul_flops = 2 * size**3  # 2N³ FLOPs
    matmul_bytes = 3 * size * size * 4  # 3 matrices
    matmul_ai = matmul_flops / matmul_bytes
    matmul_gflops = matmul_flops / (matmul_time * 1e9)
    matmul_bandwidth = matmul_bytes / (matmul_time * 1e9)

    print(f"│ Matrix Multiply     │ {matmul_ai:7.3f} │ {matmul_time*1000:9.2f}   │ {matmul_gflops:9.1f}   │ {matmul_bandwidth:9.1f}   │")

    print("└─────────────────────┴─────────┴─────────────┴─────────────┴─────────────┘")

    print(f"\n💡 Roofline Model Insights:")
    print("   📊 Ridge point = peak FLOP/s / memory bandwidth in bytes/s")
    print("   📊 Below the ridge: the roofline is limited by memory bandwidth")
    print("   📊 Above the ridge: the roofline is limited by compute throughput")
    print("   📊 The ridge depends on hardware, precision, and the memory level")
    print("   Example: 10 TFLOP/s / 100 GB/s gives a ridge of 100 FLOPs/byte")
    print("   At 20 FLOPs/byte, that example is memory-bandwidth limited")
    print("   Actual performance can fall below either roof due to other overheads")
    print(f"   🎯 Matrix multiplication ({matmul_ai:.1f} AI) is ideal for GPUs/TPUs")
    print(f"   ⚡ Element-wise ops ({add_ai:.3f} AI) need memory optimization")
    print("🚀 Design algorithms with high arithmetic intensity for performance")

if __name__ == "__main__":
    analyze_arithmetic_intensity()

# %% [markdown]
"""
### Memory Efficiency Analysis

Understanding memory allocation patterns is crucial for perf.
Let's measure how different implementations use memory.
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-memory", "solution": false}
def analyze_memory_efficiency():
    """📊 Analyze memory allocation patterns for different operations."""
    print("📊 Analyzing memory efficiency patterns...")

    import tracemalloc

    sizes = [100, 500, 1000]

    print("\n🔍 Memory Allocation Analysis:")
    print("┌─────────┬──────────────┬──────────────┬──────────────┐")
    print("│  Size   │ Vectorized   │ Unfused GELU │ Fused GELU   │")
    print("│         │ Matmul (MB)  │ (MB)         │ (MB)         │")
    print("├─────────┼──────────────┼──────────────┼──────────────┤")

    for size in sizes:
        x = Tensor(rng.standard_normal((size, size)).astype(np.float32))
        y = Tensor(rng.standard_normal((size, size)).astype(np.float32))

        # Measure vectorized matmul
        tracemalloc.start()
        _ = vectorized_matmul(x, y)
        _, matmul_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure unfused GELU
        tracemalloc.start()
        _ = unfused_gelu(x)
        _, unfused_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure fused GELU
        tracemalloc.start()
        _ = fused_gelu(x)
        _, fused_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"│ {size:6d}  │ {matmul_peak/1e6:10.2f}   │ {unfused_peak/1e6:10.2f}   │ {fused_peak/1e6:10.2f}   │")

    print("└─────────┴──────────────┴──────────────┴──────────────┘")

    print("\n💡 Key insights:")
    print("   • Inputs were allocated before tracing and are excluded from these peaks")
    print("   • Unfused GELU retains intermediate Tensor copies")
    print("   • Compact GELU still allocates NumPy temporaries")
    print("   • Allocation peaks are not memory bandwidth measurements")
    print("🚀 Memory efficiency critical for large batch sizes and limited GPU memory")

if __name__ == "__main__":
    analyze_memory_efficiency()

# %% [markdown]
"""
### Convolution Lowering Analysis

im2col makes two claims: the matrix multiply is much faster than Module 09's
loops, and the patch matrix costs about `k × k` times the input's memory. Measure
both on layers shaped like the CNN milestones, kept small enough that the loop
version finishes in a few seconds, first for the forward pass alone and then for
a full training step through `Im2colConv2dFunction`.
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-im2col", "solution": false}
def analyze_im2col_tradeoff():
    """📊 Measure the speed and memory trade-off of im2col convolution."""
    print("📊 Analyzing im2col convolution against Module 09's loops...")

    from tinytorch.core.spatial import Conv2d

    # (batch, in_ch, out_ch, size): a first layer on RGB input, then a deeper one
    layers = [(2, 3, 16, 16), (2, 16, 32, 16)]
    kernel_size, padding = 3, 1

    print("\n🔍 Loops vs im2col + one matmul (3×3 kernel, same padding):")
    print("┌──────────────────────┬────────────┬────────────┬──────────┬──────────────┐")
    print("│ Layer                │ Loops (ms) │ im2col (ms)│ Speedup  │ Patch memory │")
    print("├──────────────────────┼────────────┼────────────┼──────────┼──────────────┤")

    for batch, in_ch, out_ch, size in layers:
        conv = Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        x = Tensor(rng.standard_normal((batch, in_ch, size, size)).astype(np.float32))

        # The loop version is slow, so it runs once; im2col is timed over several runs
        start = time.perf_counter()
        reference = conv(x)
        loop_time = time.perf_counter() - start

        _ = im2col_conv2d(x, conv.weight, conv.bias, padding=padding)  # warmup
        start = time.perf_counter()
        for _ in range(DEFAULT_TIMING_ITERATIONS):
            fast = im2col_conv2d(x, conv.weight, conv.bias, padding=padding)
        im2col_time = (time.perf_counter() - start) / DEFAULT_TIMING_ITERATIONS

        assert np.allclose(fast.data, reference.data, atol=1e-5), "im2col must match the loops"

        patch_bytes = im2col(x, kernel_size, padding=padding).data.size * BYTES_PER_FLOAT32
        input_bytes = x.data.size * BYTES_PER_FLOAT32
        label = f"{batch}×{in_ch}→{out_ch} @ {size}×{size}"
        print(f"│ {label:20s} │ {loop_time*1000:10.1f} │ {im2col_time*1000:10.3f} │ "
              f"{loop_time/im2col_time:7.0f}× │ {patch_bytes/input_bytes:5.1f}× input │")

    print("└──────────────────────┴────────────┴────────────┴──────────┴──────────────┘")

    # Training needs forward AND backward. Time one full step on each path.
    print("\n🔍 One training step (forward + backward):")
    print("┌──────────────────────┬────────────┬────────────┬──────────┐")
    print("│ Layer                │ Loops (ms) │ im2col (ms)│ Speedup  │")
    print("├──────────────────────┼────────────┼────────────┼──────────┤")
    for batch, in_ch, out_ch, size in layers:
        conv = Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        x_data = rng.standard_normal((batch, in_ch, size, size)).astype(np.float32)

        start = time.perf_counter()
        conv(Tensor(x_data, requires_grad=True)).sum().backward()
        loop_step = time.perf_counter() - start

        w = Tensor(conv.weight.data.copy(), requires_grad=True)
        b = Tensor(conv.bias.data.copy(), requires_grad=True)
        start = time.perf_counter()
        for _ in range(DEFAULT_TIMING_ITERATIONS):
            Im2colConv2dFunction.apply(Tensor(x_data, requires_grad=True), w, b,
                                       stride=1, padding=padding).sum().backward()
        im2col_step = (time.perf_counter() - start) / DEFAULT_TIMING_ITERATIONS

        label = f"{batch}×{in_ch}→{out_ch} @ {size}×{size}"
        print(f"│ {label:20s} │ {loop_step*1000:10.1f} │ {im2col_step*1000:10.3f} │ {loop_step/im2col_step:7.0f}× │")
    print("└──────────────────────┴────────────┴────────────┴──────────┘")

    print("\n💡 Key insights:")
    print("   • Both versions do the same multiply-adds; the loops run them one at a time in Python")
    print("   • The patch matrix repeats each interior pixel up to k×k = 9 times")
    print("   • Speedups depend on the machine; the memory ratio depends only on the shapes")
    print("🚀 im2col trades memory for one large, fast matrix multiply")

if __name__ == "__main__":
    analyze_im2col_tradeoff()

# %% [markdown]
"""
### Optimization Insights: Production Acceleration Strategy

Understanding when and how to apply different acceleration techniques in real-world scenarios.
"""

# %% nbgrader={"grade": false, "grade_id": "acceleration-decision-framework", "solution": false}
def analyze_acceleration_decision_framework():
    """📊 Decision framework for choosing acceleration techniques."""
    print("📊 Acceleration Technique Decision Framework...")

    # Define workload characteristics
    workloads = [
        ("Research Training", {
            "memory_pressure": "medium",
            "latency_sensitive": False,
            "stability_critical": False,
            "development_speed": "high",
            "hardware_variety": "high"
        }),
        ("Production Training", {
            "memory_pressure": "high",
            "latency_sensitive": False,
            "stability_critical": True,
            "development_speed": "medium",
            "hardware_variety": "low"
        }),
        ("Real-time Inference", {
            "memory_pressure": "medium",
            "latency_sensitive": True,
            "stability_critical": True,
            "development_speed": "low",
            "hardware_variety": "medium"
        }),
        ("Edge Deployment", {
            "memory_pressure": "very_high",
            "latency_sensitive": True,
            "stability_critical": True,
            "development_speed": "low",
            "hardware_variety": "very_high"
        }),
        ("Batch Inference", {
            "memory_pressure": "low",
            "latency_sensitive": False,
            "stability_critical": True,
            "development_speed": "medium",
            "hardware_variety": "low"
        })
    ]

    # Define technique characteristics
    techniques = {
        "Vectorization": {
            "implementation_cost": "low",
            "memory_benefit": "none",
            "latency_benefit": "high",
            "stability_risk": "none",
            "hardware_dependency": "low"
        },
        "Kernel Fusion": {
            "implementation_cost": "medium",
            "memory_benefit": "medium",
            "latency_benefit": "medium",
            "stability_risk": "low",
            "hardware_dependency": "medium"
        },
        "Graph Optimization": {
            "implementation_cost": "very_high",
            "memory_benefit": "medium",
            "latency_benefit": "very_high",
            "stability_risk": "low",
            "hardware_dependency": "very_high"
        }
    }

    print("\n🎯 Acceleration Technique Recommendations:")
    print("┌─────────────────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Workload            │ Vectorize   │ Fuse Kernels│ Graph Opt   │")
    print("├─────────────────────┼─────────────┼─────────────┼─────────────┤")

    for workload_name, workload_chars in workloads:
        recommendations = []

        for technique_name in ["Vectorization", "Kernel Fusion", "Graph Optimization"]:
            tech_chars = techniques[technique_name]
            score = 0

            # Benefit vs requirement matching
            if workload_chars["memory_pressure"] in ["high", "very_high"]:
                if tech_chars["memory_benefit"] in ["medium", "high"]:
                    score += 2

            if workload_chars["latency_sensitive"]:
                if tech_chars["latency_benefit"] in ["medium", "high", "very_high"]:
                    score += 2

            # Risk vs tolerance matching
            if workload_chars["stability_critical"]:
                if tech_chars["stability_risk"] in ["none", "low"]:
                    score += 1
                elif tech_chars["stability_risk"] == "medium":
                    score -= 1

            # Implementation cost vs development speed
            if workload_chars["development_speed"] == "high":
                if tech_chars["implementation_cost"] in ["low", "medium"]:
                    score += 1
                elif tech_chars["implementation_cost"] in ["high", "very_high"]:
                    score -= 1

            # Hardware dependency vs variety
            if workload_chars["hardware_variety"] in ["high", "very_high"]:
                if tech_chars["hardware_dependency"] in ["low", "medium"]:
                    score += 1
                elif tech_chars["hardware_dependency"] in ["high", "very_high"]:
                    score -= 2

            # Convert score to recommendation
            if score >= 3:
                rec = "✅ High"
            elif score >= 1:
                rec = "⚡ Medium"
            elif score >= 0:
                rec = "⚠️  Low"
            else:
                rec = "❌ Skip"

            recommendations.append(rec)

        rec_line = " │ ".join(f"{rec:10s}" for rec in recommendations)
        print(f"│ {workload_name:19s} │ {rec_line} │")

    print("└─────────────────────┴─────────────┴─────────────┴─────────────┘")

    # Implementation priority framework
    print(f"\n🛠️  Implementation Priority Framework:")
    print(f"   📊 Phase 1 (Always): Vectorization")
    print(f"      • Low risk, high reward")
    print(f"      • Works on any hardware")
    print(f"      • Foundation for other optimizations")
    print(f"   ")
    print(f"   📊 Phase 2 (Memory constrained): Kernel Fusion")
    print(f"      • Targets memory-bound operations")
    print(f"      • Moderate complexity")
    print(f"      • Significant wins on element-wise ops")
    print(f"   ")
    print(f"   📊 Phase 3 (Scale): Mixed Precision and Batching")
    print(f"      • Essential for large model training")
    print(f"      • Requires careful validation")
    print(f"      • Hardware-dependent benefits")
    print(f"   ")
    print(f"   📊 Phase 4 (Production): Graph Optimization")
    print(f"      • Maximum performance extraction")
    print(f"      • High implementation cost")
    print(f"      • Deployment-specific tuning")

    print(f"\n💡 Key Decision Factors:")
    print(f"   🎯 Start simple: Vectorization first, always")
    print(f"   📈 Scale up: Add complexity only when needed")
    print(f"   ⚡ Measure impact: Profile before and after each optimization")
    print(f"   🔄 Iterate: Optimization is an ongoing process, not one-time")
    print("🚀 Systematic acceleration beats random optimization")

if __name__ == "__main__":
    analyze_acceleration_decision_framework()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that all acceleration components work together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-module", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire acceleration module functionality.

    This final test ensures:
    - All acceleration techniques work correctly
    - Performance improvements are measurable
    - Components integrate seamlessly
    - Module is ready for production use
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_vectorized_matmul()
    test_unit_fused_gelu()
    test_unit_fusion_speedup()
    test_unit_tiled_matmul()
    test_unit_im2col()
    test_unit_im2col_conv2d()
    test_unit_col2im()
    test_unit_im2col_conv2d_function()

    print("\nRunning integration scenarios...")

    # Test realistic acceleration pipeline
    print("🧪 Integration Test: Complete acceleration pipeline...")

    # Create realistic model scenario
    batch_size, seq_len, hidden_dim = 16, 64, 256
    print(f"   Model config: batch={batch_size}, seq_len={seq_len}, hidden={hidden_dim}")

    # Test data
    x = Tensor(rng.standard_normal((batch_size, seq_len, hidden_dim)).astype(np.float32))
    weight = Tensor(rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32))
    print(f"   Input tensor: {x.shape}, Weight tensor: {weight.shape}")

    # Test complete pipeline: reshape → matmul → activation
    print("   Testing vectorized operations...")

    # Reshape for matrix multiplication (flatten batch and sequence)
    x_reshaped = Tensor(x.data.reshape(-1, hidden_dim))
    assert x_reshaped.shape == (batch_size * seq_len, hidden_dim)

    # Vectorized matrix multiplication
    linear_output = vectorized_matmul(x_reshaped, weight)
    assert linear_output.shape == (batch_size * seq_len, hidden_dim)
    print(f"   ✅ Matrix multiplication: {x_reshaped.shape} @ {weight.shape} → {linear_output.shape}")

    # Fused activation
    activated = fused_gelu(linear_output)
    assert activated.shape == linear_output.shape
    print(f"   ✅ Fused GELU activation: {linear_output.shape} → {activated.shape}")

    # Reshape back to original structure
    final_output = Tensor(activated.data.reshape(batch_size, seq_len, hidden_dim))
    assert final_output.shape == x.shape
    print(f"   ✅ Output reshape: {activated.shape} → {final_output.shape}")
    class AcceleratedMLP:
        def __init__(self, hidden_dim):
            self.hidden_dim = hidden_dim
            self.weight1 = Tensor(rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32))
            self.weight2 = Tensor(rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32))

        def __call__(self, x):
            # Simulate transformer block: linear → activation → linear
            batch_size, seq_len, hidden_dim = x.shape
            x_flat = Tensor(x.data.reshape(-1, hidden_dim))

            # First linear layer
            h1 = vectorized_matmul(x_flat, self.weight1)
            h1_activated = fused_gelu(h1)

            # Second linear layer
            h2 = vectorized_matmul(h1_activated, self.weight2)

            # Reshape back
            output = Tensor(h2.data.reshape(batch_size, seq_len, hidden_dim))
            return output

        def parameters(self):
            return [self.weight1, self.weight2]

    # Initialize model and test forward pass
    model = AcceleratedMLP(hidden_dim)
    print(f"   Model parameters: {len(model.parameters())}")

    # Test model forward pass with accelerated operations
    print("   Testing model forward pass with accelerated operations...")
    output = model(x)
    assert output.shape == x.shape
    print(f"   ✅ Model forward pass: {x.shape} → {output.shape}")

    # Verify accelerated operations provide correct results
    print("   Validating numerical correctness...")
    # Check output is finite and has reasonable values
    assert np.all(np.isfinite(output.data)), "Model output contains NaN or Inf"
    output_mean = np.mean(np.abs(output.data))
    # Random initialization can produce larger values - verify reasonable range
    assert output_mean < 1000.0, f"Output values unreasonably large: {output_mean}"
    print(f"   ✅ Numerical validation passed (mean magnitude: {output_mean:.4f})")

    print("   Testing performance characteristics...")

    # Verify acceleration provides measurable benefits
    test_sizes = [128, 256]
    for size in test_sizes:
        test_x = Tensor(rng.standard_normal((size, size)).astype(np.float32))
        test_y = Tensor(rng.standard_normal((size, size)).astype(np.float32))

        # Time operations and verify reasonable performance
        start = time.perf_counter()
        _ = vectorized_matmul(test_x, test_y)
        matmul_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = fused_gelu(test_x)
        gelu_time = time.perf_counter() - start

        # Verify operations complete in reasonable time
        # Wall-clock thresholds would grade the machine, not the implementation.

        print(f"   ✅ Size {size}: matmul={matmul_time*1000:.1f}ms, gelu={gelu_time*1000:.1f}ms")

    print("✅ End-to-end acceleration pipeline works!")

    # Convolution lowered to a matmul feeds the same accelerated pipeline
    print("🧪 Integration Test: im2col convolution in a CNN forward pass...")
    from tinytorch.core.spatial import Conv2d

    images = Tensor(rng.standard_normal((4, 3, 8, 8)).astype(np.float32))
    conv = Conv2d(3, 8, kernel_size=3, padding=1)
    features = im2col_conv2d(images, conv.weight, conv.bias, padding=1)
    assert np.allclose(features.data, conv(images).data, atol=1e-5), \
        "im2col convolution must match Conv2d inside a pipeline"
    print(f"   ✅ Convolution: {images.shape} → {features.shape} (matches Conv2d)")

    # Flatten the feature map, then classify with the vectorized ops from above
    flat = Tensor(features.data.reshape(images.shape[0], -1))
    classifier = Tensor(rng.standard_normal((flat.shape[1], 10)).astype(np.float32) * 0.01)
    logits = fused_gelu(vectorized_matmul(flat, classifier))
    assert logits.shape == (4, 10), f"Wrong classifier output shape: {logits.shape}"
    assert np.all(np.isfinite(logits.data)), "CNN pipeline produced NaN or Inf"
    print(f"   ✅ Classifier head: {flat.shape} → {logits.shape}")
    print("✅ im2col convolution composes with the rest of the pipeline!")

    # One training step: the im2col Function must update weights exactly as Conv2d does
    print("🧪 Integration Test: one SGD step through im2col matches Conv2d...")
    learning_rate = 0.1
    x_data = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
    target = rng.standard_normal((2, 4, 6, 6)).astype(np.float32)

    reference_conv = Conv2d(3, 4, kernel_size=3, padding=1)
    w = Tensor(reference_conv.weight.data.copy(), requires_grad=True)
    b = Tensor(reference_conv.bias.data.copy(), requires_grad=True)

    def squared_error(out):
        diff = out - Tensor(target)
        return (diff * diff).sum()

    squared_error(reference_conv(Tensor(x_data))).backward()
    squared_error(Im2colConv2dFunction.apply(Tensor(x_data), w, b, stride=1, padding=1)).backward()

    grad_of = lambda t: np.asarray(getattr(t.grad, "data", t.grad))
    updated_reference = reference_conv.weight.data - learning_rate * grad_of(reference_conv.weight)
    updated_im2col = w.data - learning_rate * grad_of(w)
    assert np.allclose(updated_im2col, updated_reference, atol=1e-3), \
        "One SGD step through im2col must move the weights exactly as Conv2d does"
    print(f"   ✅ Weight update matches Conv2d (max difference {np.abs(updated_im2col - updated_reference).max():.1e})")
    print("✅ im2col convolution trains like Conv2d!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 17")

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of acceleration techniques and their systems implications:

### Question 1: Arithmetic Intensity Analysis
You implemented vectorized matrix multiplication and fused GELU.
- Matrix multiplication (1024×1024): Performs ~2.1 billion FLOPs, reads ~12 MB data
- Arithmetic intensity: _____ FLOPs/byte
- Compared to element-wise addition (0.083 FLOPs/byte): _____× higher intensity
- Why does this make matrix multiplication ideal for GPUs? _____

---

### Question 2: Kernel Fusion Memory Benefits
Your fused_gelu writes the computation as one NumPy expression.
- Why does that still allocate intermediate arrays? _____
- What extra copies does unfused_gelu retain in Tensor wrappers? _____
- A compiled fused kernel could read and write each element once. What would
  need to change to achieve that here? _____
- Why is reduced memory traffic useful for transformer inference? _____

---

### Question 3: Production Optimization Strategy
Based on your decision framework analysis:
For edge deployment (memory critical, stability required, hardware diverse):
- Priority 1 technique: _____ (low risk, universal)
- Priority 2 technique: _____ (memory benefits)
- Skip technique: _____ (why: _____)
- What's the primary constraint: memory, compute, or power? _____

---

### Question 4: What Training Through im2col Costs
Your Im2colConv2dFunction saves the patch matrix in forward and uses it in backward.
- For the 4×32×32×32 layer with a 3×3 kernel and same padding, how many bytes does
  that saved patch matrix hold? How many for the input it came from? _____
- A 20-layer CNN keeps one such buffer per convolution until backward reaches
  it. Roughly how does that compare with storing only the layer inputs? _____
- The alternative is to rebuild `cols` from the saved input during backward.
  What does that cost in time, and when would you choose it? _____
- col2im is the only step of the backward pass that is not a matrix multiply.
  Why can't it be one? _____
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Vectorization and Fusion Speed Things Up

**What you built:** Vectorized operations, blocked matrix multiplication, a
GELU comparison that exposes the cost of retaining intermediate Tensor objects,
and a convolution that runs as one matrix multiply.

**Why it matters:** The same mathematical expression can allocate and copy different
amounts of data. The compact GELU expression avoids those Tensor wrappers, but NumPy
still executes several array operations. True kernel fusion is the production next
step: a compiler combines the operations into one traversal. Measure your local
speedup rather than assuming a fixed multiplier.
"""

# %%
def demo_acceleration():
    """🎯 See fused operations produce correct results."""
    print("🎯 AHA MOMENT: Fused Operations Match Reference")
    print("=" * 45)

    # Use concrete small values for clear demonstration
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Compute GELU using fused implementation
    result_fused = fused_gelu(x)

    # Compute reference using NumPy directly
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    result_reference = 0.5 * x.data * (
        1.0 + np.tanh(sqrt_2_over_pi * (x.data + 0.044715 * x.data**3))
    )

    # Display inputs and outputs
    print(f"Input: {x.data}")
    print(f"GELU output: {result_fused.data}")
    print(f"Reference:   {result_reference}")

    # Validate results match
    match = np.allclose(result_fused.data, result_reference)
    print(f"\nResults match: {match}")

    print("\n✨ Same math, optimized execution!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_acceleration()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Acceleration

Congratulations! You've mastered the fundamental techniques for accelerating neural networks!

### Key Accomplishments
- Built **vectorized operations** using optimized BLAS and measured their timing
- Compared **GELU implementations** with different intermediate Tensor allocation costs; actual kernel fusion remains a production bridge
- Created **cache-aware tiling** for efficient large matrix operations
- Lowered **convolution to one matrix multiply** with im2col and checked it against Module 09's Conv2d
- Wrote its **backward pass** with col2im, so a convolution can train through two matmuls and a scatter-add instead of seven nested loops
- Analyzed **arithmetic intensity patterns** and their impact on the roofline model
- Measured **memory efficiency** across different operation types
- Developed **production decision framework** for systematic optimization
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Discovered
- **Roofline Model**: Operations with high arithmetic intensity (FLOPs/byte) scale better
- **Memory Bandwidth**: Often the limiting factor for modern accelerators
- **Cache Awareness**: Tiling keeps working sets in cache for better performance
- **Lowering**: im2col turns a convolution into a GEMM by copying each pixel up to k×k times; the same multiply-adds run thousands of times faster once they leave the Python interpreter
- **Kernel Fusion**: A compiled fused kernel can eliminate intermediate arrays; the savings depend on the expression
- **Optimization Strategy**: Start simple (vectorization), add complexity as needed

In production, these techniques enable:
- **Training larger models** within memory constraints
- **Faster iteration cycles** during research and development
- **Better hardware utilization** across different deployment targets
- **Cost reduction** through improved efficiency

### Ready for Next Steps
Your acceleration implementations provide the foundation for advanced optimization modules.
The performance analysis skills transfer directly to production optimization workflows.

Export with: `tito module complete 17`

**Next**: Module 18 will add memoization techniques including KV caching for efficient transformer inference!
"""
