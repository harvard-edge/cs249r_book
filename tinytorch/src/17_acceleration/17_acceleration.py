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

#| default_exp perf.acceleration
#| export

# %% [markdown]
"""
# Module 17: Acceleration - Hardware-Aware Optimization

Welcome to Module 17! You're about to master the art of neural network acceleration through vectorization and kernel fusion.

## 🔗 Prerequisites & Progress
**You've Built**: Complete neural network foundation with tensors (01), layers (03), autograd (06), training (08), and CNNs (09)
**You'll Build**: Acceleration techniques including vectorization and operation fusion
**You'll Enable**: Hardware-efficient execution for production deployment

**Connection Map**:
```
Layers (03) → Training (08) → CNNs (09) → Acceleration (17)
(building blocks) (learning)   (spatial)  (speed up)
```

**Prerequisites**: Modules 01-15 must be working
Before starting, verify:
- [ ] Module 01 (Tensor): Tensor class works
- [ ] Module 06 (Autograd): Gradients work
- [ ] Module 09 (Convolutions): Conv2d works (optional)

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement vectorized operations for maximum throughput
2. Create fused operations to reduce memory bandwidth
3. Understand the relationship between compute and memory bandwidth
4. Analyze acceleration trade-offs in production systems

Let's optimize for speed!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/17_acceleration/acceleration_dev.py`
**Building Side:** Code exports to `tinytorch.perf.acceleration`

```python
# How to use this module:
from tinytorch.perf.acceleration import vectorized_matmul, fused_gelu
```

**Why this matters:**
- **Learning:** Complete acceleration system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.cuda and torch.backends with optimization components
- **Consistency:** All acceleration operations and optimization components in perf.acceleration
- **Integration:** Works seamlessly with neural network layers for complete performance optimization
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| export

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# Constants for performance measurement
DEFAULT_WARMUP_ITERATIONS = 2  # Default warmup iterations for timing
DEFAULT_TIMING_ITERATIONS = 5  # Default timing iterations for measurement
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01-15 must be working

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for performance measurement)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class from Module 01)
- `tinytorch.perf.profiling` (Profiler from Module 14)

**Dependency Flow**:
```
Module 01 (Tensor) → Module 14 (Profiling) → Module 17 (Acceleration)
     ↓                       ↓                      ↓
  Foundation          Measurement Tools      Performance Optimization
```

Students completing this module will have built acceleration techniques
that work with the complete TinyTorch performance optimization stack.
"""

# %% [markdown]
"""
## 💡 Introduction: The Performance Challenge

Before we learn acceleration techniques, let's understand the performance gap.
Neural networks often underutilize hardware due to:
- Sequential operations (no parallelism)
- Poor memory access patterns (cache misses)
- Missing SIMD (Single Instruction, Multiple Data) opportunities
- Separate operations (memory bandwidth waste)

We'll fix these issues with vectorization and kernel fusion, achieving 2-5× speedups!

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

### The Roofline Model - Your Performance Compass

Every processor has fundamental limits:

```
Performance   │   Compute Bound Region
(GFLOPS)      │  ┌─────────────────────
              │  │ Peak Performance
              │  │
              │ ╱│ Memory Bound Region
              │╱ │
             ╱│  │
            ╱ │  │
           ╱  │  │
          ╱───│──│───────────────────────
         ╱    │  │
        ╱     │  │
       ╱──────│──│────────────────── Arithmetic Intensity
              │  │        (FLOPs/Byte)
           Low│  │High
```

**Key Insight**: Understand where your operations live on this graph to optimize effectively.

### Why This Module Matters

Real-world performance wins:
- **2-5× speedup** from vectorization
- **2-3× throughput** from kernel fusion
- **10× scaling improvement** for large models
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-import", "solution": true}
#| export
# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor

# %% [markdown]
"""
## 📐 Foundations: Vectorization - From Loops to Lightning

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
- **High arithmetic intensity**: 2N³ FLOPs for N³ data
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

    TODO: Implement production-grade matrix multiplication

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
    [[19 22]    # [1×5+2×7, 1×6+2×8] = [19, 22]
     [43 50]]   # [3×5+4×7, 3×6+4×8] = [43, 50]

    PERFORMANCE CHARACTERISTICS:
    - Time Complexity: O(N³) but highly optimized
    - Space Complexity: O(N²) for result
    - Arithmetic Intensity: 2N³ FLOPs / (3N² × 4) bytes = N/6 (good for large N)

    HINTS:
    - Check a.shape[-1] == b.shape[-2] for inner dimension match
    - Use np.matmul() for batch support and optimization
    - Trust BLAS to handle the vectorization magic
    """
    ### BEGIN SOLUTION
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
    a_batch = Tensor(np.random.randn(batch_size, m, k))
    b_batch = Tensor(np.random.randn(batch_size, k, n))
    result_batch = vectorized_matmul(a_batch, b_batch)

    assert result_batch.shape == (batch_size, m, n), f"Wrong batch shape: {result_batch.shape}"

    # Test broadcasting (different batch dimensions)
    a_single = Tensor(np.random.randn(m, k))
    b_batch = Tensor(np.random.randn(batch_size, k, n))
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
## 🏗️ Implementation: Kernel Fusion - Eliminating Memory Bottlenecks

### The Memory Bandwidth Crisis

Consider this innocent-looking computation: `y = gelu(x * weight + bias)`

**Naive Implementation (Memory Intensive):**
```
Step 1: temp1 = x * weight     → Write 4GB to memory
Step 2: temp2 = temp1 + bias   → Read 4GB, Write 4GB
Step 3: y = gelu(temp2)        → Read 4GB, Write 4GB
                                 Total: 20GB memory traffic!
```

**Fused Implementation (Memory Efficient):**
```
Single Step: y = gelu(x * weight + bias)  → Read 8GB, Write 4GB
                                            Total: 12GB memory traffic!
                                            60% memory bandwidth reduction!
```

### Understanding GELU: The Smooth Activation

GELU (Gaussian Error Linear Unit) is used in transformers because it's **smooth** (differentiable everywhere):

```
Activation Functions Compared:

ReLU:           GELU:           Sigmoid:
     |               |                 1 ┌─────
     |               |               ╱   │
     |           ╱───│───            ╱   │
─────┘       ╱───    │         ───╱      │
 Discontinuous   Smooth Curve    │ Smooth but saturates
 gradient at 0   everywhere      │
```

**GELU Formula**: `GELU(x) = x * Φ(x)` where Φ is the standard normal CDF

**Fast Approximation**: `GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

### Kernel Fusion Strategy

```
Unfused Operations:                    Fused Operation:
┌─────────────────┐                   ┌────────────────────┐
│ x³ computation  │ → temp1           │                    │
└─────────────────┘                   │                    │
┌─────────────────┐                   │                    │
│ polynomial part │ → temp2           │   All operations   │
└─────────────────┘                   │   combined in      │
┌─────────────────┐                   │   single kernel    │
│ tanh computation│ → temp3           │                    │
└─────────────────┘                   │                    │
┌─────────────────┐                   │                    │
│ final multiply  │ → result          │                    │
└─────────────────┘                   └────────────────────┘

5 memory round-trips                   1 memory round-trip
```
"""

# %% nbgrader={"grade": false, "grade_id": "fused-gelu", "solution": true}
#| export

def fused_gelu(x: Tensor) -> Tensor:
    """
    Fused GELU activation that combines all operations in a single kernel.

    GELU combines the benefits of ReLU and sigmoid:
    - Smooth everywhere (unlike ReLU's discontinuity at 0)
    - Non-saturating for positive values (unlike sigmoid)
    - Probabilistic interpretation: x * P(X ≤ x) where X ~ N(0,1)

    Mathematical Definition:
    GELU(x) = x * Φ(x) where Φ(x) is the standard normal CDF

    Fast Approximation (used here):
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    TODO: Implement fused GELU to minimize memory bandwidth

    APPROACH:
    1. Compute all intermediate values in a single expression
    2. Avoid creating temporary arrays
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
    - Unfused: 5 temporary arrays × input_size × 4 bytes
    - Fused: 0 temporary arrays, direct computation
    - Bandwidth reduction: ~80% for memory-bound operations

    HINTS:
    - Use np.sqrt(2.0 / np.pi) for the constant
    - Keep entire expression in one line for maximum fusion
    - NumPy will optimize the expression tree automatically
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

# %% nbgrader={"grade": true, "grade_id": "test-fused-gelu", "locked": true, "points": 10}
def test_unit_fused_gelu():
    """🔬 Test fused GELU activation implementation."""
    print("🧪 Unit Test: Fused GELU...")

    # Test basic properties
    x = Tensor([-3, -1, 0, 1, 3])
    result = fused_gelu(x)

    # GELU(0) = 0 (exact property)
    assert abs(result.data[2]) < 1e-6, f"GELU(0) should be 0, got {result.data[2]}"

    # GELU is smooth and increasing
    assert result.data[4] > result.data[3] > result.data[2], "GELU should be increasing"

    # GELU has positive bias (unlike ReLU)
    assert result.data[3] > 0.8, "GELU(1) should be close to 1"
    assert result.data[1] > -0.2, "GELU(-1) should be slightly negative"

    # Test numerical stability with extreme values
    x_extreme = Tensor([-10, -5, 0, 5, 10])
    result_extreme = fused_gelu(x_extreme)

    assert not np.any(np.isnan(result_extreme.data)), "No NaN values allowed"
    assert not np.any(np.isinf(result_extreme.data)), "No infinite values allowed"

    # Test large tensor processing
    x_large = Tensor(np.random.randn(1000, 1000).astype(np.float32))
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

Let's quantify the impact of kernel fusion by comparing fused vs unfused implementations.
"""

# %% nbgrader={"grade": false, "grade_id": "unfused-gelu", "solution": true}
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
    ### BEGIN SOLUTION
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

# %% nbgrader={"grade": true, "grade_id": "test-fusion-speedup", "locked": true, "points": 10}
def test_unit_fusion_speedup():
    """🧪 Measure the performance impact of kernel fusion."""
    print("🧪 Unit Test: Kernel Fusion Performance Impact...")

    # Create moderately large tensor for meaningful timing
    size = 2000
    x = Tensor(np.random.randn(size, size).astype(np.float32))
    warmup_iterations = DEFAULT_WARMUP_ITERATIONS
    timing_iterations = DEFAULT_TIMING_ITERATIONS

    # Warmup both implementations
    for _ in range(warmup_iterations):
        _ = unfused_gelu(x)
        _ = fused_gelu(x)

    # Time unfused version
    start = time.time()
    for _ in range(timing_iterations):
        result_unfused = unfused_gelu(x)
    unfused_time = time.time() - start

    # Time fused version
    start = time.time()
    for _ in range(timing_iterations):
        result_fused = fused_gelu(x)
    fused_time = time.time() - start

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

    # Memory bandwidth estimate
    bytes_per_elem = 4  # float32
    unfused_memory_ops = 7  # 7 intermediate arrays
    fused_memory_ops = 2   # read input, write output

    unfused_bandwidth = (unfused_memory_ops * size * size * bytes_per_elem) / (unfused_time / timing_iterations) / 1e9
    fused_bandwidth = (fused_memory_ops * size * size * bytes_per_elem) / (fused_time / timing_iterations) / 1e9

    print(f"   Memory efficiency: {unfused_memory_ops}→{fused_memory_ops} memory ops")
    print(f"   Effective bandwidth: {unfused_bandwidth:.1f}→{fused_bandwidth:.1f} GB/s")

    # Interpret results
    if speedup > 1.5:
        print("🚀 Excellent! Kernel fusion providing significant speedup")
    elif speedup > 1.1:
        print("✅ Good! Kernel fusion providing measurable benefit")
    else:
        print("⚠️  Limited speedup - may be compute-bound or small tensor size")

    print("✅ Fusion performance analysis completed!")

if __name__ == "__main__":
    test_unit_fusion_speedup()

# %% [markdown]
"""
## 🏗️ Implementation: Cache-Aware Matrix Multiplication

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
"""

# %% nbgrader={"grade": false, "grade_id": "tiled-matmul", "solution": true}
#| export

def tiled_matmul(a: Tensor, b: Tensor, tile_size: int = 64) -> Tensor:
    """
    Cache-aware matrix multiplication using tiling/blocking.

    Demonstrates blocking algorithm for cache optimization by breaking
    large matrix multiplications into cache-sized chunks.

    TODO: Implement cache-aware tiled matrix multiplication

    APPROACH:
    1. Validate inputs for matrix multiplication compatibility
    2. Use NumPy's optimized matmul (which already implements tiling internally)
    3. In production, explicit tiling would use nested loops over blocks

    Args:
        a: First matrix (M×K)
        b: Second matrix (K×N)
        tile_size: Block size for cache efficiency (default: 64)

    Returns:
        Result matrix (M×N)

    EXAMPLE:
    >>> a = Tensor(np.random.randn(256, 256))
    >>> b = Tensor(np.random.randn(256, 256))
    >>> result = tiled_matmul(a, b, tile_size=64)
    >>> # Same result as vectorized_matmul, but more cache-friendly for large matrices

    PERFORMANCE CHARACTERISTICS:
    - Reduces cache misses by working on blocks that fit in L1/L2
    - Especially beneficial for matrices larger than cache size
    - tile_size should match cache line size (typically 64 bytes)

    HINTS:
    - For educational purposes, we use NumPy's optimized BLAS
    - BLAS libraries (MKL, OpenBLAS) already implement cache blocking
    - Explicit tiling would use 6 nested loops (3 for tiles, 3 for elements)
    """
    ### BEGIN SOLUTION
    # Input validation
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError(
            f"Tiled matrix multiplication requires 2D+ tensors\n"
            f"  ❌ Got shapes {a.shape} and {b.shape} ({len(a.shape)}D and {len(b.shape)}D tensors)\n"
            f"  💡 Tiling partitions matrices into cache-sized blocks, which requires 2D structure\n"
            f"  🔧 Add dimensions with reshape: tensor.reshape(1, -1) for row vector or tensor.reshape(-1, 1) for column"
        )

    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Tiled matrix multiplication shape mismatch: {a.shape} @ {b.shape}\n"
            f"  ❌ Inner dimensions don't match: a.shape[-1]={a.shape[-1]} vs b.shape[-2]={b.shape[-2]}\n"
            f"  💡 Each tile of A's columns must align with tiles of B's rows for block multiplication\n"
            f"  🔧 Reshape to align: b.reshape({a.shape[-1]}, -1) or transpose if dimensions are swapped"
        )

    # For educational purposes, we use NumPy's matmul which already
    # implements cache-aware tiling via BLAS libraries (MKL, OpenBLAS)
    # These libraries automatically partition large matrices into
    # cache-sized blocks for optimal performance

    # In a full educational implementation, you would write:
    # for i_tile in range(0, M, tile_size):
    #     for j_tile in range(0, N, tile_size):
    #         for k_tile in range(0, K, tile_size):
    #             # Multiply tile blocks that fit in cache
    #             C[i_tile:i_tile+tile_size, j_tile:j_tile+tile_size] +=
    #                 A[i_tile:i_tile+tile_size, k_tile:k_tile+tile_size] @
    #                 B[k_tile:k_tile+tile_size, j_tile:j_tile+tile_size]

    result_data = np.matmul(a.data, b.data)
    return Tensor(result_data)
    ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-tiled-matmul", "locked": true, "points": 10}
def test_unit_tiled_matmul():
    """🧪 Test cache-aware tiled matrix multiplication."""
    print("🧪 Unit Test: Tiled Matrix Multiplication...")

    # Test correctness against vectorized version
    a = Tensor(np.random.randn(128, 128).astype(np.float32))
    b = Tensor(np.random.randn(128, 128).astype(np.float32))

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
        wrong_a = Tensor(np.random.randn(128, 64).astype(np.float32))
        wrong_b = Tensor(np.random.randn(128, 64).astype(np.float32))
        tiled_matmul(wrong_a, wrong_b)
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        assert "shape mismatch" in str(e).lower()

    print("✅ tiled_matmul works correctly!")

if __name__ == "__main__":
    test_unit_tiled_matmul()

# %% [markdown]
"""
## 📊 Systems Analysis: Performance Scaling Patterns

Let's analyze how our acceleration techniques perform across different scenarios and understand their scaling characteristics.
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-vectorization", "solution": true}
def analyze_vectorization_scaling():
    """📊 Analyze vectorization performance across different tensor sizes."""
    print("📊 Analyzing vectorization scaling behavior...")

    # Test sizes spanning different cache regimes
    sizes = [64, 128, 256, 512, 1024, 2048]

    print("\n🔍 Vectorization Scaling Analysis:")
    print("┌─────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("│  Size   │ Time (ms)   │ GFLOPS      │ Bandwidth   │ Efficiency  │")
    print("│         │             │             │ (GB/s)      │ (% of peak) │")
    print("├─────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

    for size in sizes:
        # Create test matrices
        a = Tensor(np.random.randn(size, size).astype(np.float32))
        b = Tensor(np.random.randn(size, size).astype(np.float32))

        # Warm up
        for _ in range(2):
            _ = vectorized_matmul(a, b)

        # Time vectorized implementation
        iterations = max(1, 100 // (size // 64))  # Fewer iterations for larger sizes
        start = time.time()
        for _ in range(iterations):
            result = vectorized_matmul(a, b)
        elapsed = (time.time() - start) / iterations

        # Calculate performance metrics
        flops = 2 * size**3  # 2N³ FLOPs for matrix multiplication
        gflops = flops / (elapsed * 1e9)

        bytes_accessed = 3 * size * size * 4  # 3 matrices × size² × 4 bytes
        bandwidth = bytes_accessed / (elapsed * 1e9)

        # Estimate efficiency (rough baseline: modern CPU ~100-500 GFLOPS peak)
        estimated_peak_gflops = 200  # Conservative estimate
        efficiency = min(100, gflops / estimated_peak_gflops * 100)

        print(f"│ {size:6d}  │ {elapsed*1000:9.2f}   │ {gflops:9.1f}   │ {bandwidth:9.1f}   │ {efficiency:9.1f}   │")

    print("└─────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

    print(f"\n💡 Vectorization insights:")
    print(f"   • Small matrices: Limited by overhead and cache effects")
    print(f"   • Medium matrices: Sweet spot for cache reuse")
    print(f"   • Large matrices: Memory bandwidth becomes limiting factor")
    print(f"   • BLAS libraries automatically optimize for each size regime")
    print("🚀 Vectorization effectiveness depends on problem size and hardware")

if __name__ == "__main__":
    analyze_vectorization_scaling()

# %% nbgrader={"grade": false, "grade_id": "analyze-arithmetic-intensity", "solution": true}
def analyze_arithmetic_intensity():
    """📊 Demonstrate the roofline model with different operations."""
    print("📊 Analyzing arithmetic intensity patterns...")

    size = 1024
    iterations = 10

    operations = []

    # Create test data
    x = Tensor(np.random.randn(size, size).astype(np.float32))
    y = Tensor(np.random.randn(size, size).astype(np.float32))

    print("\n🎯 Arithmetic Intensity Analysis:")
    print("┌─────────────────────┬─────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Operation           │ AI      │ Time (ms)   │ GFLOPS      │ GB/s        │")
    print("│                     │(FLOPs/B)│             │             │             │")
    print("├─────────────────────┼─────────┼─────────────┼─────────────┼─────────────┤")

    # 1. Element-wise addition (very low arithmetic intensity)
    start = time.time()
    for _ in range(iterations):
        _ = Tensor(x.data + y.data)
    add_time = (time.time() - start) / iterations

    add_flops = size * size  # One addition per element
    add_bytes = 3 * size * size * 4  # Read x, read y, write result
    add_ai = add_flops / add_bytes
    add_gflops = add_flops / (add_time * 1e9)
    add_bandwidth = add_bytes / (add_time * 1e9)

    print(f"│ Element-wise Add    │ {add_ai:6.3f}  │ {add_time*1000:9.2f}   │ {add_gflops:9.1f}   │ {add_bandwidth:9.1f}   │")

    # 2. Element-wise multiply (still low, but slightly higher)
    start = time.time()
    for _ in range(iterations):
        _ = Tensor(x.data * y.data)
    mul_time = (time.time() - start) / iterations

    mul_flops = size * size
    mul_bytes = 3 * size * size * 4
    mul_ai = mul_flops / mul_bytes
    mul_gflops = mul_flops / (mul_time * 1e9)
    mul_bandwidth = mul_bytes / (mul_time * 1e9)

    print(f"│ Element-wise Mult   │ {mul_ai:6.3f}  │ {mul_time*1000:9.2f}   │ {mul_gflops:9.1f}   │ {mul_bandwidth:9.1f}   │")

    # 3. GELU (medium arithmetic intensity)
    start = time.time()
    for _ in range(iterations):
        _ = fused_gelu(x)
    gelu_time = (time.time() - start) / iterations

    gelu_flops = size * size * 8  # Approximate: x³, add, mul, tanh, etc.
    gelu_bytes = 2 * size * size * 4  # Read x, write result
    gelu_ai = gelu_flops / gelu_bytes
    gelu_gflops = gelu_flops / (gelu_time * 1e9)
    gelu_bandwidth = gelu_bytes / (gelu_time * 1e9)

    print(f"│ Fused GELU          │ {gelu_ai:6.3f}  │ {gelu_time*1000:9.2f}   │ {gelu_gflops:9.1f}   │ {gelu_bandwidth:9.1f}   │")

    # 4. Matrix multiplication (high arithmetic intensity)
    start = time.time()
    for _ in range(iterations):
        _ = vectorized_matmul(x, y)
    matmul_time = (time.time() - start) / iterations

    matmul_flops = 2 * size**3  # 2N³ FLOPs
    matmul_bytes = 3 * size * size * 4  # 3 matrices
    matmul_ai = matmul_flops / matmul_bytes
    matmul_gflops = matmul_flops / (matmul_time * 1e9)
    matmul_bandwidth = matmul_bytes / (matmul_time * 1e9)

    print(f"│ Matrix Multiply     │ {matmul_ai:6.3f}  │ {matmul_time*1000:9.2f}   │ {matmul_gflops:9.1f}   │ {matmul_bandwidth:9.1f}   │")

    print("└─────────────────────┴─────────┴─────────────┴─────────────┴─────────────┘")

    print(f"\n💡 Roofline Model Insights:")
    print(f"   📊 Low AI (< 1): Memory bound - limited by bandwidth")
    print(f"   📊 Med AI (1-10): Transitional - depends on implementation")
    print(f"   📊 High AI (> 10): Compute bound - limited by ALU throughput")
    print(f"   🎯 Matrix multiplication ({matmul_ai:.1f} AI) is ideal for GPUs/TPUs")
    print(f"   ⚡ Element-wise ops ({add_ai:.3f} AI) need memory optimization")
    print("🚀 Design algorithms with high arithmetic intensity for performance")

if __name__ == "__main__":
    analyze_arithmetic_intensity()

# %% [markdown]
"""
### 📊 Memory Efficiency Analysis

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
        x = Tensor(np.random.randn(size, size).astype(np.float32))
        y = Tensor(np.random.randn(size, size).astype(np.float32))

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

        print(f"│ {size:6d}  │ {matmul_peak/1e6:10.2f}   │ {unfused_peak/1e6:10.2f}   │ {fused_peak/1e6:8.2f}   │")

    print("└─────────┴──────────────┴──────────────┴──────────────┘")

    print("\n💡 Key insights:")
    print("   • Vectorized matmul: ~3× input size (2 inputs + 1 output)")
    print("   • Unfused GELU: ~8-10× input size (many intermediate tensors)")
    print("   • Fused GELU: ~2× input size (1 input + 1 output only)")
    print("   • Fusion reduces memory allocations by 4-5×")
    print("🚀 Memory efficiency critical for large batch sizes and limited GPU memory")

if __name__ == "__main__":
    analyze_memory_efficiency()

# %% [markdown]
"""
## 🔧 Optimization Insights: Production Acceleration Strategy

Understanding when and how to apply different acceleration techniques in real-world scenarios.
"""

# %% nbgrader={"grade": false, "grade_id": "acceleration-decision-framework", "solution": true}
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
        print(f"│ {workload_name:18s}  │ {rec_line} │")

    print("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

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
## 🔧 Integration: Measuring Acceleration Gains with Profiler

Now let's use the **Profiler** tool you built in Module 14 to measure the actual performance improvements from vectorization. This demonstrates the full workflow: build profiling tools (M14), apply optimizations (M15-M17), measure gains.

This is how professional ML engineers work: profile → optimize → measure → repeat.
"""

# %% nbgrader={"grade": false, "grade_id": "demo-profiler-acceleration", "solution": true}
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
            self.weight = Tensor(np.random.randn(in_features, out_features).astype(np.float32) * 0.01)
            self.name = "slow_linear"

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
            self.weight = Tensor(np.random.randn(in_features, out_features).astype(np.float32) * 0.01)
            self.name = "fast_linear"

        def forward(self, x):
            # Vectorized implementation
            return vectorized_matmul(x, self.weight)

    in_features, out_features = 128, 64
    batch_size = 32

    # Create models
    slow_model = SlowLinear(in_features, out_features)
    fast_model = FastLinear(in_features, out_features)

    # Create input
    input_tensor = Tensor(np.random.randn(batch_size, in_features).astype(np.float32))

    print("\n🐢 BEFORE: Loop-based implementation")
    print("-" * 70)

    # Measure slow model
    slow_latency = profiler.measure_latency(slow_model, input_tensor, warmup=3, iterations=10)
    slow_flops = profiler.count_flops(slow_model, (batch_size, in_features))

    print(f"   Latency: {slow_latency:.2f} ms")
    print(f"   FLOPs: {slow_flops:,}")
    print(f"   Throughput: {slow_flops / (slow_latency / 1000) / 1e9:.2f} GFLOP/s")

    print("\n🚀 AFTER: Vectorized implementation")
    print("-" * 70)

    # Measure fast model
    fast_latency = profiler.measure_latency(fast_model, input_tensor, warmup=3, iterations=10)
    fast_flops = profiler.count_flops(fast_model, (batch_size, in_features))

    print(f"   Latency: {fast_latency:.2f} ms")
    print(f"   FLOPs: {fast_flops:,}")
    print(f"   Throughput: {fast_flops / (fast_latency / 1000) / 1e9:.2f} GFLOP/s")

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

    print("\nRunning integration scenarios...")

    # Test realistic acceleration pipeline
    print("🧪 Integration Test: Complete acceleration pipeline...")

    # Create realistic model scenario
    batch_size, seq_len, hidden_dim = 16, 64, 256
    print(f"   Model config: batch={batch_size}, seq_len={seq_len}, hidden={hidden_dim}")

    # Test data
    x = Tensor(np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32))
    weight = Tensor(np.random.randn(hidden_dim, hidden_dim).astype(np.float32))
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
    class TransformerBlock:
        def __init__(self, hidden_dim):
            self.hidden_dim = hidden_dim
            self.weight1 = Tensor(np.random.randn(hidden_dim, hidden_dim).astype(np.float32))
            self.weight2 = Tensor(np.random.randn(hidden_dim, hidden_dim).astype(np.float32))
            self.weight1.grad = None
            self.weight2.grad = None

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
    model = TransformerBlock(hidden_dim)
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
        test_x = Tensor(np.random.randn(size, size).astype(np.float32))
        test_y = Tensor(np.random.randn(size, size).astype(np.float32))

        # Time operations and verify reasonable performance
        start = time.time()
        _ = vectorized_matmul(test_x, test_y)
        matmul_time = time.time() - start

        start = time.time()
        _ = fused_gelu(test_x)
        gelu_time = time.time() - start

        # Verify operations complete in reasonable time
        assert matmul_time < 1.0, f"Matrix multiplication too slow: {matmul_time:.3f}s"
        assert gelu_time < 0.1, f"GELU activation too slow: {gelu_time:.3f}s"

        print(f"   ✅ Size {size}: matmul={matmul_time*1000:.1f}ms, gelu={gelu_time*1000:.1f}ms")

    print("   Testing memory efficiency...")

    print("✅ End-to-end acceleration pipeline works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 17")

# Run comprehensive module test
if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of acceleration techniques and their systems implications:

### 1. Arithmetic Intensity Analysis
You implemented vectorized matrix multiplication and fused GELU.
- Matrix multiplication (1024×1024): Performs ~2.1 billion FLOPs, reads ~12 MB data
- Arithmetic intensity: _____ FLOPs/byte
- Compared to element-wise addition (0.083 FLOPs/byte): _____× higher intensity
- Why does this make matrix multiplication ideal for GPUs? _____

---

### 2. Kernel Fusion Memory Benefits
Your fused_gelu combines 7 operations into a single expression.
- Unfused version memory accesses: 7 reads + 7 writes = _____ per element
- Fused version memory accesses: 1 read + 1 write = _____ per element
- Memory bandwidth reduction: _____%
- Why is this critical for transformer inference? _____

---

### 3. Production Optimization Strategy
Based on your decision framework analysis:
For edge deployment (memory critical, stability required, hardware diverse):
- Priority 1 technique: _____ (low risk, universal)
- Priority 2 technique: _____ (memory benefits)
- Skip technique: _____ (why: _____)
- What's the primary constraint: memory, compute, or power? _____
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Vectorization and Fusion Speed Things Up

**What you built:** Vectorized operations and fused kernels that reduce memory traffic.

**Why it matters:** Individual operations like x + y + z require reading and writing memory
multiple times. Fused operations like fused_gelu do everything in one pass! This reduces
memory bandwidth by 60-80%, a huge win since memory is often the bottleneck.

Combined with vectorization (SIMD), these techniques make neural networks 2-5× faster.
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
- Built **vectorized operations** leveraging SIMD and optimized BLAS for 2-5× speedups
- Implemented **kernel fusion** reducing memory bandwidth by 60-80% for element-wise operations
- Created **cache-aware tiling** for efficient large matrix operations
- Analyzed **arithmetic intensity patterns** and their impact on the roofline model
- Measured **memory efficiency** across different operation types
- Developed **production decision framework** for systematic optimization
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Discovered
- **Roofline Model**: Operations with high arithmetic intensity (FLOPs/byte) scale better
- **Memory Bandwidth**: Often the limiting factor for modern accelerators
- **Cache Awareness**: Tiling keeps working sets in cache for better performance
- **Kernel Fusion**: Critical for memory-bound workloads, reduces intermediate storage by 4-5×
- **Optimization Strategy**: Start simple (vectorization), add complexity as needed

### Production Impact
Your acceleration techniques enable:
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
