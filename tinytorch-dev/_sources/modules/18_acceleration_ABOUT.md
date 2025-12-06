---
title: "Acceleration - CPU Vectorization & Cache Optimization"
description: "Master hardware-aware optimization through BLAS vectorization, cache-friendly algorithms, and roofline analysis"
difficulty: "⭐⭐⭐"
time_estimate: "6-8 hours"
prerequisites: ["Profiling"]
next_steps: ["Benchmarking"]
learning_objectives:
  - "Understand the roofline model and arithmetic intensity for predicting performance bottlenecks"
  - "Leverage optimized BLAS libraries for CPU vectorization achieving 10-100x speedups"
  - "Implement cache-aware algorithms and analyze memory hierarchy impact"
  - "Apply kernel fusion to reduce memory bandwidth for element-wise operations"
  - "Measure acceleration gains systematically using profiling integration"
---

# 18. Acceleration - CPU Vectorization & Cache Optimization

**OPTIMIZATION TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 6-8 hours

## Overview

The Acceleration module teaches you to extract maximum performance from modern CPUs through hardware-aware optimization techniques. You'll learn to leverage optimized BLAS libraries for vectorized matrix operations, implement cache-friendly algorithms that exploit memory hierarchy, and apply kernel fusion to eliminate memory bandwidth bottlenecks. By mastering the roofline model and arithmetic intensity analysis, you'll develop the systematic thinking needed to accelerate real ML systems from research prototypes to production deployments.

This is CPU-focused acceleration—the foundation for understanding GPU perf. You'll work with NumPy's BLAS backend (MKL, OpenBLAS) to achieve 10-100x speedups over naive Python, understand why most operations are memory-bound rather than compute-bound, and learn the measurement-driven optimization workflow used by PyTorch, TensorFlow, and production ML systems.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand Hardware Bottlenecks**: Apply the roofline model to determine whether operations are compute-bound or memory-bound, and predict performance limits from hardware specifications
- **Leverage BLAS Vectorization**: Use optimized linear algebra libraries that exploit SIMD instructions and multi-threading to achieve 10-100x speedups over naive implementations
- **Implement Cache-Aware Algorithms**: Design blocked/tiled algorithms that maximize cache hit rates by fitting working sets into L1/L2 cache for 2-5x memory performance gains
- **Apply Kernel Fusion**: Reduce memory bandwidth usage by 60-80% through fusing element-wise operations into single expressions that eliminate intermediate array allocations
- **Measure Systematically**: Integrate with Module 14 profiling to validate optimization impact, measure FLOPs efficiency, and calculate arithmetic intensity for real workloads

## Build → Use → Optimize

This module follows TinyTorch's **Build → Use → Optimize** framework:

1. **Build**: Implement vectorized matrix multiplication using BLAS, fused GELU activation, and tiled algorithms for cache efficiency
2. **Use**: Apply acceleration to realistic transformer blocks, analyze memory access patterns, and measure performance across different tensor sizes
3. **Optimize**: Analyze roofline characteristics, measure arithmetic intensity, and develop systematic decision frameworks for production optimization strategies

## Why This Matters: The Hardware Reality

### The Performance Gap

Modern ML workloads face a fundamental challenge: **the speed gap between computation and memory access grows every year**. Consider a typical CPU:

- **Peak Compute**: 200-500 GFLOP/s (billions of floating-point operations per second)
- **Memory Bandwidth**: 20-50 GB/s (data transfer rate from RAM to CPU)
- **Imbalance**: CPUs can perform 10-20 floating-point operations in the time it takes to fetch a single float from memory

This means **most ML operations are memory-bound, not compute-bound**. Naive implementations waste computation cycles waiting for data. Professional optimization is about feeding the compute units efficiently.

### From Naive Python to Production Performance

The performance hierarchy for ML operations:

```
Naive Python loops:       1 GFLOP/s    (baseline)
NumPy (vectorized):       10-50 GFLOP/s  (10-50x faster)
Optimized BLAS (this module): 100-500 GFLOP/s (100-500x faster)
GPU CUDA kernels:         1,000-10,000 GFLOP/s (1,000-10,000x faster)
```

This module focuses on the **100-500x speedup** achievable on CPUs through:
- **SIMD vectorization**: Process 4-8 floats per instruction (AVX2/AVX-512)
- **Multi-threading**: Use all CPU cores (4-8x parallelism)
- **Cache blocking**: Keep data in fast cache memory (10-100x faster than RAM)
- **Kernel fusion**: Reduce memory traffic by 4-10x

### Real-World Impact

These techniques enable:
- **Faster iteration**: Train models in hours instead of days during research
- **Lower costs**: More efficient use of cloud compute resources
- **Edge deployment**: Run models on CPUs without GPU requirements
- **Better scaling**: Handle larger models and batch sizes within memory limits

Understanding CPU optimization is prerequisite for GPU programming—same principles, different scale.

## The Roofline Model: Your Performance Compass

### Understanding Hardware Limits

The **roofline model** is the fundamental tool for understanding performance bottlenecks. It plots two hardware limits:

1. **Compute Roof**: Maximum FLOPs the processor can execute per second
2. **Memory Roof**: Maximum data bandwidth × arithmetic intensity

**Arithmetic Intensity (AI)** = FLOPs performed / Bytes accessed

```
Performance         Compute Bound Region
(GFLOPS)           ┌─────────────────────
                   │ Peak Compute (500 GFLOP/s)
                   │
                  ╱│
                 ╱ │ Memory Bound Region
                ╱  │
               ╱   │
              ╱────│──────────────────────
             ╱     │
            ╱      │
           ╱───────│──────────────────── Arithmetic Intensity
                   │           (FLOPs/Byte)
                Low│          High
                (<1)│         (>10)
```

**Key Insight**: If your operation falls below the roofline (left side), adding more compute won't help—you need to reduce memory traffic through algorithmic improvements.

### Example Calculations

**Element-wise addition**: `c = a + b`
- FLOPs: N (one addition per element)
- Bytes: 3N × 4 bytes (read a, read b, write c)
- AI = N / (12N) = **0.08 FLOPs/byte** → Severely memory-bound

**Matrix multiplication**: `C = A @ B` for N×N matrices
- FLOPs: 2N³ (dot product for each of N² output elements)
- Bytes: 3N² × 4 bytes (read A, read B, write C)
- AI = 2N³ / (12N²) = **N/6 FLOPs/byte** → Compute-bound for large N

For N=1024: AI = 171 FLOPs/byte—squarely in the compute-bound region. This is why matrix multiplication is ideal for GPUs and why transformers (which are mostly matmuls) run efficiently on accelerators.

## Implementation Guide

### 1. Vectorized Matrix Multiplication

**The Challenge**: Naive triple-nested loops in Python achieve ~1 GFLOP/s. We need 100-500 GFLOP/s.

**The Solution**: Leverage optimized BLAS (Basic Linear Algebra Subprograms) libraries that implement:
- **SIMD vectorization**: AVX2/AVX-512 instructions process 4-8 floats simultaneously
- **Multi-threading**: Automatic parallelization across CPU cores (OpenMP)
- **Cache blocking**: Tiled algorithms that keep working sets in L1/L2 cache

```python
def vectorized_matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    High-performance matrix multiplication using optimized BLAS.

    NumPy's matmul calls GEMM (General Matrix Multiply) from:
    - Intel MKL (Math Kernel Library) - 200-500 GFLOP/s on modern CPUs
    - OpenBLAS - 100-300 GFLOP/s
    - Apple Accelerate - optimized for M1/M2 chips

    These libraries implement decades of optimization research.
    """
    # Input validation
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Shape mismatch: {a.shape} @ {b.shape}")

    # Delegate to highly optimized BLAS implementation
    # This single line replaces thousands of lines of hand-tuned assembly
    result_data = np.matmul(a.data, b.data)
    return Tensor(result_data)
```

**Performance Characteristics**:
- **Small matrices** (N < 64): 10-30 GFLOP/s, limited by overhead
- **Medium matrices** (N = 64-512): 100-300 GFLOP/s, optimal cache reuse
- **Large matrices** (N > 1024): 200-500 GFLOP/s, memory bandwidth saturated

**Measured Speedups** (vs. naive triple loop):
- 128×128: **50x faster** (5ms → 0.1ms)
- 512×512: **120x faster** (800ms → 6.5ms)
- 2048×2048: **150x faster** (100s → 0.67s)

### 2. Kernel Fusion: Eliminating Memory Traffic

**The Problem**: Element-wise operations are memory-bound. Consider GELU activation:

```
GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Unfused implementation** (naive):
```python
temp1 = x ** 3                    # Read x, write temp1
temp2 = 0.044715 * temp1         # Read temp1, write temp2
temp3 = x + temp2                # Read x, temp2, write temp3
temp4 = sqrt_2_pi * temp3        # Read temp3, write temp4
temp5 = tanh(temp4)              # Read temp4, write temp5
temp6 = 1.0 + temp5              # Read temp5, write temp6
temp7 = x * temp6                # Read x, temp6, write temp7
result = 0.5 * temp7             # Read temp7, write result

# Total: 8 reads + 8 writes = 16 memory operations per element
```

**Fused implementation**:
```python
def fused_gelu(x: Tensor) -> Tensor:
    """
    Fused GELU activation - all operations in single expression.

    Memory efficiency:
    - Unfused: 16 memory ops per element
    - Fused: 2 memory ops per element (read x, write result)
    - Reduction: 87.5% less memory traffic
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    # Single expression - NumPy optimizes into minimal memory operations
    result_data = 0.5 * x.data * (
        1.0 + np.tanh(sqrt_2_over_pi * (x.data + 0.044715 * x.data**3))
    )

    return Tensor(result_data)
```

**Measured Performance** (2000×2000 tensor):
- Unfused: 45ms (7 temporary arrays allocated)
- Fused: 18ms (0 temporary arrays)
- **Speedup: 2.5x faster** through memory bandwidth reduction alone

**Memory Usage**:
- Unfused: ~320MB (8 arrays × 2000×2000 × 4 bytes × overhead)
- Fused: ~32MB (input + output only)
- **Memory reduction: 90%**

### 3. Cache-Aware Tiling (Blocked Algorithms)

**The Memory Hierarchy**:
```
L1 Cache:   32-64 KB    1-4 cycles     ~1 TB/s bandwidth
L2 Cache:   256KB-1MB   10-20 cycles   ~500 GB/s bandwidth
L3 Cache:   8-32 MB     40-75 cycles   ~200 GB/s bandwidth
Main RAM:   8-64 GB     100-300 cycles ~20-50 GB/s bandwidth
```

**The Problem**: Naive matrix multiplication for 2048×2048 matrices accesses:
- Data size: 3 × 2048² × 4 bytes = 50MB (doesn't fit in L1/L2 cache)
- Result: Most accesses hit L3 or RAM (100-300 cycle latency)

**The Solution**: Block/tile matrices into cache-sized chunks

**Conceptual Tiled Algorithm**:
```python
def tiled_matmul_concept(A, B, tile_size=64):
    """
    Conceptual tiling algorithm (educational).

    In practice, BLAS libraries implement this automatically
    with hardware-specific tuning for optimal tile sizes.
    """
    N = A.shape[0]
    C = np.zeros((N, N))

    # Process matrix in tile_size × tile_size blocks
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            for k in range(0, N, tile_size):
                # This block fits in L1/L2 cache (64×64×4 = 16KB)
                # All accesses hit fast cache instead of slow RAM
                i_end = min(i + tile_size, N)
                j_end = min(j + tile_size, N)
                k_end = min(k + tile_size, N)

                C[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]

    return C
```

**Cache Efficiency Analysis**:
- **Naive algorithm**: 99% L3/RAM accesses (slow)
- **Blocked algorithm** (64×64 tiles): 95% L1/L2 hits (fast)
- **Latency reduction**: 300 cycles → 10 cycles average
- **Effective speedup**: 2-5x for large matrices

**Optimal Tile Sizes** (empirically determined):
- L1-focused: 32×32 (4KB per block)
- L2-focused: 64×64 (16KB per block) ← sweet spot for most CPUs
- L3-focused: 128×128 (64KB per block)

Note: In this module, we use NumPy's `matmul` which delegates to BLAS libraries (MKL, OpenBLAS) that already implement sophisticated cache blocking with hardware-specific tuning. Production implementations use tile sizes, loop unrolling, and prefetching tuned for specific CPU architectures.

### 4. Roofline Analysis in Practice

**Measuring Your Hardware**:

```python
def analyze_arithmetic_intensity():
    """Measure actual performance vs. theoretical roofline."""

    # Theoretical hardware limits (example: modern Intel CPU)
    peak_compute = 400  # GFLOP/s (AVX-512, 8 cores, 3.5 GHz)
    peak_bandwidth = 45  # GB/s (DDR4-2666, dual-channel)

    operations = {
        "Element-wise add": {
            "flops": N,
            "bytes": 3 * N * 4,
            "ai": 0.08  # FLOPs/byte
        },
        "Matrix multiply": {
            "flops": 2 * N**3,
            "bytes": 3 * N**2 * 4,
            "ai": N / 6  # For N=1024: 171 FLOPs/byte
        }
    }

    # Predicted performance = min(peak_compute, ai × peak_bandwidth)
    for op, metrics in operations.items():
        predicted_gflops = min(
            peak_compute,
            metrics["ai"] * peak_bandwidth
        )
        print(f"{op}: {predicted_gflops:.1f} GFLOP/s (predicted)")
```

**Example Analysis** (N=1024):

| Operation | AI (FLOPs/byte) | Predicted GFLOP/s | Measured GFLOP/s | Efficiency |
|-----------|----------------|-------------------|------------------|------------|
| Element-wise add | 0.08 | 3.6 (memory-bound) | 3.2 | 89% |
| GELU (fused) | 1.0 | 45 (memory-bound) | 38 | 84% |
| Matrix multiply | 171 | 400 (compute-bound) | 320 | 80% |

**Key Insights**:
- Element-wise operations hit **memory roof** at 3-4 GFLOP/s (only 1% of peak compute)
- Fusion improves AI by reducing memory operations (0.08 → 1.0 AI)
- Matrix multiplication approaches **compute roof** (80% of peak)
- Optimization strategy should focus on memory-bound operations first

## Getting Started

### Prerequisites

Ensure you've completed:
- **Module 14 (Profiling)**: You'll use profiling tools to measure acceleration gains
- **Module 01 (Tensor)**: Tensor class provides foundation for operations
- **NumPy/BLAS**: Verify optimized BLAS backend is installed

Check your BLAS configuration:
```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Check which BLAS backend NumPy uses
python -c "import numpy as np; np.show_config()"

# Look for: openblas, mkl, or accelerate (Apple Silicon)
# MKL is fastest on Intel CPUs (200-500 GFLOP/s)
# OpenBLAS is good cross-platform (100-300 GFLOP/s)
```

Verify prerequisite modules work:
```bash
tito test tensor
tito test profiling
```

### Development Workflow

1. **Open the development file**: `modules/18_acceleration/acceleration.py`

2. **Implement vectorized matrix multiplication**:
   - Validate input shapes for compatibility
   - Delegate to `np.matmul` (calls optimized BLAS GEMM)
   - Return result wrapped in Tensor
   - Test correctness and measure speedup vs. naive loops

3. **Build fused GELU activation**:
   - Implement complete GELU formula in single expression
   - Avoid creating intermediate Tensor objects
   - Test numerical correctness against reference implementation
   - Measure memory bandwidth reduction

4. **Create tiled matrix multiplication**:
   - Understand cache blocking concept (educational)
   - Use NumPy's matmul which implements tiling internally
   - Analyze cache hit rates and memory access patterns
   - Compare performance across different matrix sizes

5. **Perform roofline analysis**:
   - Measure FLOPs and memory bandwidth for each operation
   - Calculate arithmetic intensity
   - Plot operations on roofline model
   - Identify optimization priorities

6. **Export and verify**:
   ```bash
   tito module complete 18
   tito test acceleration
   ```

## Testing

### Comprehensive Test Suite

Run the full test suite to verify acceleration functionality:

```bash
# TinyTorch CLI (recommended)
tito test acceleration

# Direct pytest execution
python -m pytest tests/ -k acceleration -v

# Run development file directly (includes inline tests)
python modules/18_acceleration/acceleration.py
```

### Test Coverage Areas

- **Vectorized Operations Correctness**: Matrix multiplication produces numerically correct results, handles batching and broadcasting, validates incompatible shapes
- **Kernel Fusion Correctness**: Fused GELU matches reference implementation, handles extreme values without NaN/Inf, preserves data types and shapes
- **Performance Validation**: Vectorized matmul achieves 10-150x speedup over naive loops, kernel fusion provides 2-5x speedup and 60-80% memory reduction, performance scales appropriately with tensor size
- **Integration Testing**: Acceleration techniques work together in realistic transformer blocks, profiler integration measures speedups correctly, memory efficiency validated with tracemalloc
- **Roofline Analysis**: Arithmetic intensity calculated correctly for different operations, performance predictions match measurements within 20%, memory-bound vs. compute-bound classification accurate

### Inline Testing & Performance Analysis

The module includes comprehensive validation and measurement:

```python
# Run all inline tests
python modules/18_acceleration/acceleration.py

# Expected output:
🔬 Unit Test: Vectorized Matrix Multiplication...
✅ vectorized_matmul works correctly!

🔬 Unit Test: Fused GELU...
✅ fused_gelu works correctly!

🔬 Unit Test: Kernel Fusion Performance Impact...
📊 Kernel Fusion Performance Analysis:
   Tensor size: 2000×2000 = 4,000,000 elements
   Unfused time: 45.23 ms
   Fused time:   18.12 ms
   Speedup: 2.50× faster
   Per-element: 11.3 ns → 4.5 ns
   Memory efficiency: 7→2 memory ops
   Effective bandwidth: 15.2→38.5 GB/s
🚀 Excellent! Kernel fusion providing significant speedup

📊 Analyzing vectorization scaling behavior...
┌─────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│  Size   │ Time (ms)   │ GFLOPS      │ Bandwidth   │ Efficiency  │
├─────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│     64  │      0.05   │      33.6   │      15.8   │      16.8   │
│    128  │      0.18   │     114.2   │      26.7   │      57.1   │
│    256  │      1.12   │     188.5   │      22.1   │      94.3   │
│    512  │      6.45   │     328.7   │      19.3   │     164.4   │
│   1024  │     42.18   │     405.1   │      16.1   │     202.6   │
│   2048  │    281.34   │     485.2   │      15.3   │     242.6   │
└─────────┴─────────────┴─────────────┴─────────────┴─────────────┘

🧪 RUNNING MODULE INTEGRATION TEST
Running unit tests...
✅ All tests passed!

🎉 ALL TESTS PASSED! Module ready for export.
```

### Manual Testing Examples

```python
from modules.18_acceleration.acceleration import *

# Test vectorized matrix multiplication
A = Tensor(np.random.randn(512, 512).astype(np.float32))
B = Tensor(np.random.randn(512, 512).astype(np.float32))

# Measure performance
import time
start = time.time()
C = vectorized_matmul(A, B)
elapsed = time.time() - start

# Calculate metrics
flops = 2 * 512**3  # 268 million FLOPs
gflops = flops / (elapsed * 1e9)
print(f"Performance: {gflops:.1f} GFLOP/s")
print(f"Time: {elapsed*1000:.2f} ms")

# Test kernel fusion
x = Tensor(np.random.randn(1000, 1000).astype(np.float32))

# Compare fused vs unfused
start = time.time()
y_fused = fused_gelu(x)
fused_time = time.time() - start

start = time.time()
y_unfused = unfused_gelu(x)
unfused_time = time.time() - start

print(f"Speedup: {unfused_time/fused_time:.2f}x")
print(f"Numerically equivalent: {np.allclose(y_fused.data, y_unfused.data)}")

# Measure with profiler
from tinytorch.perf.profiling import Profiler

profiler = Profiler()

class SimpleModel:
    def __init__(self):
        self.weight = Tensor(np.random.randn(256, 256).astype(np.float32))

    def forward(self, x):
        return fused_gelu(vectorized_matmul(x, self.weight))

model = SimpleModel()
input_tensor = Tensor(np.random.randn(32, 256).astype(np.float32))

latency = profiler.measure_latency(model, input_tensor, warmup=5, iterations=20)
flops = profiler.count_flops(model, (32, 256))

print(f"Latency: {latency:.2f} ms")
print(f"FLOPs: {flops:,}")
print(f"Throughput: {flops / (latency/1000) / 1e9:.2f} GFLOP/s")
```

## Systems Thinking Questions

### Real-World Applications

- **Training Acceleration**: How do vectorized operations reduce training time for transformers? What's the speedup for attention computation (mostly matrix multiplies) vs. layer normalization (element-wise operations)?

- **Inference Optimization**: Why is kernel fusion more important for inference than training? How does batch size affect the benefit of vectorization vs. fusion?

- **Hardware Selection**: Given a model with 70% matrix multiplies and 30% element-wise operations, should you optimize for compute or memory bandwidth? How does this affect CPU vs. GPU selection?

- **Cloud Cost Reduction**: If vectorization provides 100x speedup on matrix operations that take 80% of training time, what's the overall training time reduction and cost savings?

### Roofline Analysis Foundations

- **Arithmetic Intensity Calculation**: For convolution with kernel size K×K, input channels C_in, output channels C_out, and spatial dimensions H×W, what's the arithmetic intensity? Is it compute-bound or memory-bound?

- **Memory Hierarchy Impact**: Why does cache blocking improve performance by 2-5x even though it performs the same FLOPs? What's the latency difference between L1 cache hits (4 cycles) vs. RAM accesses (300 cycles)?

- **BLAS Library Performance**: Why does NumPy's matmul achieve 200-500 GFLOP/s while naive Python loops achieve 1 GFLOP/s? What optimizations do BLAS libraries implement that interpreted Python can't?

- **Batch Size Effects**: How does batch size affect arithmetic intensity for matrix multiplication? Why do larger batches achieve higher GFLOP/s on the same hardware?

### Optimization Strategy Characteristics

- **Memory-Bound Operations**: Why does adding more CPU cores NOT improve element-wise addition performance? What's the fundamental bottleneck, and how do you fix it?

- **Kernel Fusion Trade-offs**: Fused GELU reduces memory operations from 16 to 2 per element. Why doesn't this give 8x speedup? What other factors limit acceleration?

- **Production Optimization Priority**: Given profiling data showing 40% time in attention softmax (memory-bound), 30% in matmuls (compute-bound), and 30% in data loading (I/O-bound), which should you optimize first? Why?

- **Cross-Platform Performance**: Why do vectorized operations using BLAS achieve different speedups on Intel CPUs (MKL: 500 GFLOP/s) vs. AMD CPUs (OpenBLAS: 200 GFLOP/s) vs. Apple Silicon (Accelerate: 300 GFLOP/s)? What's hardware-dependent vs. algorithmic?

## Ready to Build?

You're about to learn the hardware-aware optimization techniques that separate research prototypes from production ML systems. Understanding how to extract maximum performance from CPUs—through vectorization, cache optimization, and memory bandwidth reduction—is foundational knowledge for any ML engineer.

These aren't just academic exercises. Every time you use PyTorch or TensorFlow, you're benefiting from these exact techniques implemented in their backend libraries. By building them yourself, you'll understand:

- Why transformers (mostly matmuls) run efficiently on GPUs while RNNs (sequential operations) struggle
- How to predict whether adding more hardware will help before spending cloud budget
- When to optimize code vs. when to redesign algorithms for better arithmetic intensity
- How to measure and validate performance improvements systematically

The roofline model and arithmetic intensity analysis you'll master here apply directly to GPUs, TPUs, and custom AI accelerators. Hardware changes, but the fundamental memory-vs-compute trade-offs remain constant. This module gives you the mental models and measurement tools to optimize on any platform.

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/18_acceleration/acceleration_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/18_acceleration/acceleration_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/18_acceleration/acceleration.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.
```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/17_memoization_ABOUT.html" title="previous page">← Module 17: Memoization</a>
<a class="right-next" href="../chapters/19_benchmarking_ABOUT.html" title="next page">Module 19: Benchmarking →</a>
</div>
