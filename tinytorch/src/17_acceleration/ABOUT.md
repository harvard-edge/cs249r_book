# Module 17: Acceleration

:::{admonition} Module Info
:class: note

**OPTIMIZATION TIER** | Difficulty: ●●●○ | Time: 5-7 hours | Prerequisites: 01-14

**Prerequisites: Modules 01-14** means you need:
- Tensor operations (Module 01) for understanding data structures
- Neural network layers (Module 03) for knowing what to accelerate
- Training loops (Module 08) for understanding the performance context
- Profiling tools (Module 14) for measuring acceleration gains

If you can multiply matrices and understand why matrix multiplication is expensive, you're ready.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F17_acceleration%2F17_acceleration.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/17_acceleration/17_acceleration.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/17_acceleration.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Neural network training and inference spend 90% of their time on matrix operations. A single forward pass through a transformer model can involve billions of floating-point operations, and training requires thousands of these passes. The difference between a model that trains in hours versus days, or that serves predictions in milliseconds versus seconds, comes down to how efficiently these operations execute on hardware.

This module teaches you hardware-aware optimization through vectorization and kernel fusion. You'll learn to leverage SIMD instructions, optimize memory access patterns, and eliminate unnecessary memory traffic. By the end, you'll understand why a naive matrix multiplication can be 100x slower than an optimized one, and how to achieve 2-5x speedups in your own code.

Acceleration isn't about clever algorithms. It's about understanding how processors work and writing code that exploits their design.

## Why Acceleration Before Memoization?

The Optimization tier divides into **model-level** (15-16) and **runtime** (17-18) optimizations:

- **Model-level** (Quantization, Compression): Change the model itself
- **Runtime** (Acceleration, Memoization): Change how execution happens

Within runtime optimizations, **Acceleration comes first** because:
1. **General before specific**: Vectorization and kernel fusion apply to ANY numerical computation—matrix multiplication, convolutions, attention, everything
2. **Memoization (KV-cache)** is domain-specific: it only applies to transformer autoregressive generation
3. Once you understand general speedup techniques, you can appreciate the specialized optimization that makes LLM inference economically viable

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** vectorized matrix multiplication using optimized BLAS libraries for maximum throughput
- **Master** kernel fusion techniques that eliminate memory bandwidth bottlenecks by combining operations
- **Understand** the roofline model and arithmetic intensity to predict performance bottlenecks
- **Analyze** production acceleration strategies for different deployment scenarios (edge, cloud, GPU)
```

## What You'll Build

```{mermaid}
:align: center
:caption: Acceleration Techniques
flowchart LR
    subgraph "Acceleration Techniques"
        A["Vectorized Matmul<br/>BLAS optimization"]
        B["Fused GELU<br/>Kernel fusion"]
        C["Tiled Matmul<br/>Cache awareness"]
        D["Performance Analysis<br/>Roofline model"]
    end

    A --> B --> C --> D

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `vectorized_matmul()` | SIMD and BLAS optimization |
| 2 | `fused_gelu()` | Memory bandwidth reduction |
| 3 | `unfused_gelu()` | Comparison baseline |
| 4 | `tiled_matmul()` | Cache-aware computation |
| 5 | Performance analysis | Roofline and arithmetic intensity |

**The pattern you'll enable:**
```python
# Fast matrix operations using BLAS
output = vectorized_matmul(x, weights)  # 10-100x faster than naive loops

# Memory-efficient activations
activated = fused_gelu(output)  # 60% less memory bandwidth
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- GPU kernels (that requires CUDA programming, covered in production frameworks)
- Custom CPU assembly (BLAS libraries already provide this)
- Automatic kernel fusion (compilers like XLA do this automatically)
- Multi-threading control (NumPy handles this via OpenBLAS/MKL)

**You are building the understanding.** Hardware-specific implementations come later.

## API Reference

This section provides a quick reference for the acceleration functions you'll build. These functions demonstrate optimization techniques that apply to any ML framework.

### Vectorized Operations

```python
vectorized_matmul(a: Tensor, b: Tensor) -> Tensor
```

High-performance matrix multiplication using optimized BLAS libraries that leverage SIMD instructions and cache blocking.

### Kernel Fusion

| Function | Signature | Description |
|----------|-----------|-------------|
| `fused_gelu` | `fused_gelu(x: Tensor) -> Tensor` | GELU activation with all operations in single kernel |
| `unfused_gelu` | `unfused_gelu(x: Tensor) -> Tensor` | Baseline implementation for comparison |

### Cache-Aware Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `tiled_matmul` | `tiled_matmul(a: Tensor, b: Tensor, tile_size: int) -> Tensor` | Cache-optimized matrix multiplication |

## Core Concepts

This section covers the fundamental acceleration techniques that apply to any hardware platform. Understanding these concepts will help you optimize neural networks whether you're targeting CPUs, GPUs, or specialized accelerators.

### Vectorization with NumPy

Modern processors can execute the same operation on multiple data elements simultaneously through SIMD (Single Instruction, Multiple Data) instructions. A traditional loop processes one element per clock cycle, but SIMD can process 4, 8, or even 16 elements in the same time.

Consider a simple element-wise addition. A naive Python loop visits each element sequentially:

```python
# Slow: one element per iteration
for i in range(len(x)):
    result[i] = x[i] + y[i]
```

NumPy's vectorized operations automatically use SIMD when you write `x + y`. The processor loads multiple elements into special vector registers and adds them in parallel. This is why vectorized NumPy code can be 10-100x faster than explicit loops.

Here's how vectorized matrix multiplication works in your implementation:

```python
def vectorized_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication using optimized BLAS libraries."""
    # Validate shapes - inner dimensions must match
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Matrix multiplication shape mismatch: {a.shape} @ {b.shape}. "
            f"Inner dimensions must match: a.shape[-1]={a.shape[-1]} != b.shape[-2]={b.shape[-2]}"
        )

    # NumPy calls BLAS GEMM which uses:
    # - SIMD vectorization for parallel arithmetic
    # - Cache blocking for memory efficiency
    # - Multi-threading on multi-core systems
    result_data = np.matmul(a.data, b.data)

    return Tensor(result_data)
```

The magic happens inside `np.matmul`. NumPy delegates to BLAS (Basic Linear Algebra Subprograms) libraries like OpenBLAS or Intel MKL. These libraries have been optimized over decades to exploit every hardware feature: SIMD instructions, cache hierarchies, and multiple cores. The same Python code that takes 800ms with naive loops completes in 8ms with BLAS.

### BLAS and LAPACK

BLAS provides three levels of operations, each with different performance characteristics:

- **Level 1**: Vector operations (AXPY: y = αx + y). These are memory-bound with low arithmetic intensity.
- **Level 2**: Matrix-vector operations (GEMV: y = αAx + βy). Better arithmetic intensity but still memory-limited.
- **Level 3**: Matrix-matrix operations (GEMM: C = αAB + βC). High arithmetic intensity, compute-bound.

Matrix multiplication (GEMM) dominates neural network training because every linear layer, every attention mechanism, and every convolution ultimately reduces to matrix multiplication. GEMM performs 2N³ floating-point operations while reading only 3N² elements from memory. For a 1024×1024 matrix, that's 2.1 billion operations on just 12 MB of data - an arithmetic intensity of 170 FLOPs/byte. This high ratio of computation to memory access makes GEMM perfect for hardware acceleration.

### Memory Layout Optimization

When a processor needs data from main memory, it doesn't fetch individual bytes. It fetches entire cache lines (typically 64 bytes). If your data is laid out sequentially in memory, you get spatial locality: one cache line brings in many useful values. If your data is scattered randomly, every access causes a cache miss and a 100-cycle stall.

Matrix multiplication has interesting memory access patterns. Computing one output element requires reading an entire row from the first matrix and an entire column from the second matrix. Rows are stored sequentially in memory (good), but columns are strided by the matrix width (potentially bad). This is why cache-aware tiling helps:

```python
# Cache-aware tiling breaks large matrices into blocks
# Each block fits in cache for maximum reuse
for i_tile in range(0, M, tile_size):
    for j_tile in range(0, N, tile_size):
        for k_tile in range(0, K, tile_size):
            # Multiply tile blocks that fit in L1/L2 cache
            C[i_tile:i_tile+tile_size, j_tile:j_tile+tile_size] +=
                A[i_tile:i_tile+tile_size, k_tile:k_tile+tile_size] @
                B[k_tile:k_tile+tile_size, j_tile:j_tile+tile_size]
```

Your `tiled_matmul` implementation demonstrates this concept, though in practice NumPy's BLAS backend already implements optimal tiling:

```python
def tiled_matmul(a: Tensor, b: Tensor, tile_size: int = 64) -> Tensor:
    """Cache-aware matrix multiplication using tiling."""
    # Validate shapes
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Shape mismatch: {a.shape} @ {b.shape}")

    # BLAS libraries automatically implement cache-aware tiling
    # tile_size would control block size in explicit implementation
    result_data = np.matmul(a.data, b.data)
    return Tensor(result_data)
```

### Kernel Fusion

Element-wise operations like GELU activation are memory-bound: they spend more time loading and storing data than computing results. Consider the GELU formula:

```
GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

A naive implementation creates seven intermediate arrays:

```python
def unfused_gelu(x: Tensor) -> Tensor:
    """Unfused GELU - creates many temporary arrays."""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    temp1 = Tensor(x.data**3)                        # x³
    temp2 = Tensor(0.044715 * temp1.data)           # 0.044715 * x³
    temp3 = Tensor(x.data + temp2.data)             # x + 0.044715 * x³
    temp4 = Tensor(sqrt_2_over_pi * temp3.data)     # √(2/π) * (...)
    temp5 = Tensor(np.tanh(temp4.data))             # tanh(...)
    temp6 = Tensor(1.0 + temp5.data)                # 1 + tanh(...)
    temp7 = Tensor(x.data * temp6.data)             # x * (1 + tanh(...))
    result = Tensor(0.5 * temp7.data)               # 0.5 * x * (...)

    return result
```

Each temporary array allocation writes to memory, and each subsequent operation reads from memory. For a 4 million element tensor, this unfused version performs 28 million memory operations (7 reads + 7 writes per element). Memory bandwidth on a typical CPU is around 50 GB/s, so moving 112 MB takes 2.24 milliseconds - just for memory traffic, before any computation.

Kernel fusion combines all operations into a single expression:

```python
def fused_gelu(x: Tensor) -> Tensor:
    """Fused GELU - all operations in single kernel."""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    # Single expression - no intermediate arrays
    result_data = 0.5 * x.data * (
        1.0 + np.tanh(sqrt_2_over_pi * (x.data + 0.044715 * x.data**3))
    )

    return Tensor(result_data)
```

Now there are only two memory operations: read the input, write the output. For the same 4 million element tensor, that's just 32 MB of memory traffic, completing in 0.64 milliseconds. The fused version is 3.5x faster purely from memory bandwidth reduction, even though both versions perform the same arithmetic.

### Parallel Processing

Modern CPUs have multiple cores that can execute operations simultaneously. BLAS libraries automatically spawn threads to parallelize matrix multiplication across cores. A 4-core system can theoretically achieve 4x speedup on compute-bound operations.

However, parallel processing has overhead. Creating threads, synchronizing results, and merging data takes time. For small matrices, this overhead exceeds the benefit. BLAS libraries use heuristics to decide when to parallelize: large matrices get multiple threads, small matrices run on a single core.

This is why you see sublinear speedups in practice. A 4-core system might achieve 3x speedup rather than 4x, due to:
- Thread creation and destruction overhead
- Cache coherence traffic between cores
- Memory bandwidth saturation (all cores sharing the same memory bus)
- Load imbalance (some threads finish before others)

### Hardware Acceleration

This module uses NumPy and BLAS for CPU acceleration. Production frameworks go further with specialized hardware:

**GPUs** have thousands of simple cores optimized for data parallelism. A matrix multiplication that takes 100ms on a CPU can complete in 1ms on a GPU - a 100x speedup. But GPUs require explicit data transfer between CPU and GPU memory, and this transfer can dominate small operations.

**TPUs** (Tensor Processing Units) are Google's custom accelerators with systolic array architectures designed specifically for matrix multiplication. A TPU can sustain 100+ TFLOPS on matrix operations.

The acceleration techniques you implement in this module - vectorization, fusion, and cache awareness - apply to all these platforms. The specific implementations differ, but the principles remain constant.

### Arithmetic Intensity and the Roofline Model

Not all operations are created equal. The roofline model helps predict whether an operation will be limited by memory bandwidth or computational throughput. Arithmetic intensity is the ratio of floating-point operations to bytes transferred:

```
Arithmetic Intensity (AI) = FLOPs / Bytes
```

For element-wise addition of two N-element arrays:
- FLOPs: N (one addition per element)
- Bytes: 3N × 4 = 12N (read A, read B, write C, each 4 bytes for float32)
- AI = N / 12N = 0.083 FLOPs/byte

For matrix multiplication of N×N matrices:
- FLOPs: 2N³ (N³ multiplications + N³ additions)
- Bytes: 3N² × 4 = 12N² (read A, read B, write C)
- AI = 2N³ / 12N² = N/6 FLOPs/byte

For a 1024×1024 matrix: AI = 170 FLOPs/byte. Matrix multiplication performs 2000x more computation per byte transferred than element-wise addition. This is why GPUs excel at matrix operations but struggle with element-wise ops.

| Operation | Arithmetic Intensity | Bottleneck | Optimization Strategy |
|-----------|---------------------|------------|----------------------|
| Element-wise add | ~0.08 FLOPs/byte | Memory bandwidth | Kernel fusion |
| Element-wise multiply | ~0.08 FLOPs/byte | Memory bandwidth | Kernel fusion |
| GELU activation | ~1.0 FLOPs/byte | Memory bandwidth | Kernel fusion |
| Matrix multiply (1024×1024) | ~170 FLOPs/byte | Compute throughput | Vectorization, tiling |

The roofline model plots achievable performance against arithmetic intensity. Your hardware has a peak memory bandwidth (horizontal line) and peak computational throughput (diagonal line). The minimum of these two lines is your performance ceiling.

## Common Errors

These are the errors you'll encounter when optimizing neural networks. Understanding them will save you from subtle performance bugs.

### Shape Mismatches in Vectorized Code

**Error**: `ValueError: shapes (128, 256) and (128, 512) not aligned`

Matrix multiplication requires inner dimensions to match. For A @ B, `A.shape[-1]` must equal `B.shape[-2]`. This error occurs when you try to multiply incompatible shapes.

**Fix**: Always validate shapes before matrix operations:

```python
assert a.shape[-1] == b.shape[-2], f"Shape mismatch: {a.shape} @ {b.shape}"
```

### Memory Bandwidth Bottlenecks

**Symptom**: GPU shows 20% utilization but code is still slow

This indicates a memory-bound operation. The GPU cores are idle, waiting for data from memory. Element-wise operations often hit this bottleneck.

**Fix**: Use kernel fusion to reduce memory traffic. Combine multiple element-wise operations into a single fused kernel.

### Cache Thrashing

**Symptom**: Performance degrades dramatically for matrices larger than 1024×1024

When your working set exceeds cache size, the CPU spends most of its time loading data from main memory rather than computing.

**Fix**: Use tiling/blocking to keep working sets in cache. Break large matrices into smaller tiles that fit in L2 or L3 cache.

### False Dependencies

**Symptom**: Parallel code runs slower than sequential code

Creating temporary arrays in a loop can prevent parallelization because each iteration depends on the previous one's memory allocation.

**Fix**: Preallocate output arrays and reuse them:

```python
# Bad: creates new array each iteration
for i in range(1000):
    result = x + y

# Good: reuses same output array
result = np.zeros_like(x)
for i in range(1000):
    np.add(x, y, out=result)
```

## Production Context

### Your Implementation vs. PyTorch

Your acceleration techniques demonstrate the same principles PyTorch uses internally. The difference is scale: PyTorch supports GPUs, automatic kernel fusion through compilers, and thousands of optimized operations.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Vectorization** | NumPy BLAS | CUDA/cuBLAS for GPU |
| **Kernel Fusion** | Manual fusion | Automatic via TorchScript/JIT |
| **Backend** | CPU only | CPU, CUDA, Metal, ROCm |
| **Multi-threading** | Automatic (OpenBLAS) | Configurable thread pools |
| **Operations** | ~5 accelerated ops | 2000+ optimized ops |

### Code Comparison

The following comparison shows how acceleration appears in TinyTorch versus PyTorch. The API patterns are similar, but PyTorch adds GPU support and automatic optimization.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.perf.acceleration import vectorized_matmul, fused_gelu

# CPU-based acceleration
x = Tensor(np.random.randn(128, 512))
w = Tensor(np.random.randn(512, 256))

# Vectorized matrix multiplication
h = vectorized_matmul(x, w)

# Fused activation
output = fused_gelu(h)
```
````

````{tab-item} ⚡ PyTorch
```python
import torch

# GPU acceleration with same concepts
x = torch.randn(128, 512, device='cuda')
w = torch.randn(512, 256, device='cuda')

# Vectorized (cuBLAS on GPU)
h = x @ w

# Fused via JIT compilation
@torch.jit.script
def fused_gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.797885 * (x + 0.044715 * x**3)))

output = fused_gelu(h)
```
````
`````

Let's walk through the key differences:

- **Line 1 (Import)**: TinyTorch provides explicit acceleration functions; PyTorch integrates acceleration into the core tensor operations.
- **Line 4-5 (Device)**: TinyTorch runs on CPU via NumPy; PyTorch supports `device='cuda'` for GPU acceleration.
- **Line 8 (Matrix multiply)**: Both use optimized BLAS, but PyTorch uses cuBLAS on GPU for 10-100x additional speedup.
- **Line 11-13 (Fusion)**: TinyTorch requires manual fusion; PyTorch's JIT compiler can automatically fuse operations.
- **Performance**: For this example, TinyTorch might take 5ms on CPU; PyTorch takes 0.05ms on GPU - a 100x speedup.

```{tip} What's Identical

The acceleration principles: vectorization reduces instruction count, fusion reduces memory traffic, and hardware awareness guides optimization choices. These concepts apply everywhere.
```

### Why Acceleration Matters at Scale

Real-world systems demonstrate the impact of acceleration:

- **GPT-3 training**: 175 billion parameters × 300 billion tokens = **10²³ FLOPs**. Without GPU acceleration, this would take centuries. With optimized TPUs, it takes weeks.
- **Real-time inference**: Serving 1000 requests/second requires **sub-millisecond latency** per request. Every 2x speedup doubles your throughput.
- **Cost efficiency**: Cloud GPU time costs $2-10/hour. A 2x speedup saves **$1000-5000 per week** for a production model.

Small percentage improvements at this scale translate to millions in savings and fundamentally new capabilities.

## Check Your Understanding

Test your understanding of acceleration techniques with these quantitative questions.

**Q1: Arithmetic Intensity**

Matrix multiplication of two 1024×1024 float32 matrices performs 2,147,483,648 FLOPs. It reads 8 MB (matrix A) + 8 MB (matrix B) = 16 MB and writes 8 MB (matrix C) = 24 MB total. What is the arithmetic intensity?

```{admonition} Answer
:class: dropdown

Arithmetic Intensity = 2,147,483,648 FLOPs / 24,000,000 bytes = **~89 FLOPs/byte**

This high arithmetic intensity (compared to ~0.08 for element-wise ops) is why matrix multiplication is ideal for GPUs and why it dominates neural network training time.
```

**Q2: Memory Bandwidth Savings**

Your fused GELU processes a tensor with 1,000,000 elements (4 MB as float32). The unfused version creates 7 intermediate arrays. How much memory bandwidth does fusion save?

```{admonition} Answer
:class: dropdown

**Unfused**: 7 reads + 7 writes + 1 input read + 1 output write = 16 memory operations × 4 MB = **64 MB**

**Fused**: 1 input read + 1 output write = 2 memory operations × 4 MB = **8 MB**

**Savings**: 64 - 8 = **56 MB saved (87.5% reduction)**

For typical CPUs with ~50 GB/s bandwidth, this saves ~1 millisecond per GELU call. In a transformer with 96 GELU activations per forward pass, that's 96ms saved - enough to improve throughput by 10-20%.
```

**Q3: Cache Tiling**

A CPU has 256 KB L2 cache. You're multiplying two 2048×2048 float32 matrices (16 MB each). What tile size keeps the working set in L2 cache?

```{admonition} Answer
:class: dropdown

For tiled multiplication, we need 3 tiles in cache simultaneously:
- Tile from matrix A: tile_size × tile_size × 4 bytes
- Tile from matrix B: tile_size × tile_size × 4 bytes
- Output tile: tile_size × tile_size × 4 bytes

Total: 3 × tile_size² × 4 bytes ≤ 256 KB

Solving: tile_size² ≤ 256,000 / 12 = 21,333

**tile_size ≈ 146**

In practice, use powers of 2: **128 works well** (3 × 128² × 4 = 196 KB, leaving room for other data).
```

**Q4: BLAS Performance**

Your vectorized matmul completes a 1024×1024 multiplication in 10ms. The operation requires 2.15 billion FLOPs. What is your achieved performance in GFLOPS?

```{admonition} Answer
:class: dropdown

GFLOPS = 2,150,000,000 FLOPs / (0.01 seconds × 1,000,000,000) = **215 GFLOPS**

For reference:
- Modern CPU peak: 500-1000 GFLOPS (AVX-512)
- Your efficiency: 215/500 = **43% of peak** (typical for real code)
- GPU equivalent: ~50 TFLOPS (230x faster than single CPU core)
```

**Q5: Speedup from Fusion**

Unfused GELU takes 8ms on a 2000×2000 tensor. Fused GELU takes 2.5ms. What percentage of the unfused time was memory overhead?

```{admonition} Answer
:class: dropdown

Speedup = 8ms / 2.5ms = **3.2x faster**

Assuming both versions do the same computation, the difference is memory bandwidth:
- Memory overhead = (8 - 2.5) / 8 = **68.75%**

Nearly **70% of the unfused version's time** was spent waiting for memory! This is typical for element-wise operations with low arithmetic intensity.
```

## Further Reading

For students who want to understand the academic foundations and implementation details of neural network acceleration:

### Seminal Papers

- **Roofline Model** - Williams et al. (2009). The foundational framework for understanding performance bottlenecks based on arithmetic intensity. Essential for diagnosing whether your code is compute-bound or memory-bound. [IEEE](https://doi.org/10.1145/1498765.1498785)

- **BLAS: The Basic Linear Algebra Subprograms** - Lawson et al. (1979). The specification that defines standard matrix operations. Every ML framework ultimately calls BLAS for performance-critical operations. [ACM TOMS](https://doi.org/10.1145/355841.355847)

- **Optimizing Matrix Multiplication** - Goto & Geijn (2008). Detailed explanation of cache blocking, register tiling, and microkernel design for high-performance GEMM. This is how BLAS libraries achieve near-peak performance. [ACM TOMS](https://doi.org/10.1145/1356052.1356053)

- **TVM: An Automated End-to-End Optimizing Compiler** - Chen et al. (2018). Demonstrates automatic optimization including kernel fusion and memory planning for deep learning. Shows how compilers can automatically apply the techniques you learned manually. [OSDI](https://www.usenix.org/conference/osdi18/presentation/chen)

### Additional Resources

- **Tutorial**: "What Every Programmer Should Know About Memory" by Ulrich Drepper - Deep dive into cache hierarchies and their performance implications
- **Documentation**: [Intel MKL Developer Reference](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top.html) - See how production BLAS libraries implement vectorization and threading

## What's Next

```{seealso} Coming Up: Module 19 - Benchmarking

Learn to measure and compare performance systematically. You'll build benchmarking tools that isolate hardware effects, statistical analysis for reliable measurements, and comparison frameworks for evaluating optimization techniques.
```

**Preview - How Acceleration Gets Used in Future Modules:**

| Module | What It Does | Your Acceleration In Action |
|--------|--------------|---------------------------|
| **19: Benchmarking** | Systematic performance measurement | `benchmark(vectorized_matmul, sizes=[128, 256, 512])` |
| **20: Capstone** | Complete optimized model | Acceleration throughout model pipeline |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/17_acceleration/17_acceleration.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/17_acceleration/17_acceleration.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```

```{warning} Performance Note

Acceleration techniques depend on hardware. Results will vary between CPUs. Use Module 14's profiler to measure your specific hardware's characteristics.
```
