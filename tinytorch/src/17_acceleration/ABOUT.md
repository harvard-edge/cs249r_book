---
file_format: mystnb
kernelspec:
  name: python3
---

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
````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/17_acceleration.mp3" type="audio/mpeg">
</audio>
```

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F17_acceleration%2Facceleration.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/17_acceleration/17_acceleration.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #14b8a6; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

````

```{raw} html
<style>
.slide-viewer-container {
  margin: 0.5rem 0 1.5rem 0;
  background: #0f172a;
  border-radius: 1rem;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.slide-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 1rem;
  background: rgba(255,255,255,0.03);
}
.slide-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #94a3b8;
  font-weight: 500;
  font-size: 0.85rem;
}
.slide-subtitle {
  color: #64748b;
  font-weight: 400;
  font-size: 0.75rem;
}
.slide-toolbar {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}
.slide-toolbar button {
  background: transparent;
  border: none;
  color: #64748b;
  width: 32px;
  height: 32px;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 1.1rem;
  transition: all 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
}
.slide-toolbar button:hover {
  background: rgba(249, 115, 22, 0.15);
  color: #f97316;
}
.slide-nav-group {
  display: flex;
  align-items: center;
}
.slide-page-info {
  color: #64748b;
  font-size: 0.75rem;
  padding: 0 0.5rem;
  font-weight: 500;
}
.slide-zoom-group {
  display: flex;
  align-items: center;
  margin-left: 0.25rem;
  padding-left: 0.5rem;
  border-left: 1px solid rgba(255,255,255,0.1);
}
.slide-canvas-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0.5rem 1rem 1rem 1rem;
  min-height: 380px;
  background: #0f172a;
}
.slide-canvas {
  max-width: 100%;
  max-height: 350px;
  height: auto;
  border-radius: 0.5rem;
  box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.slide-progress-wrapper {
  padding: 0 1rem 0.5rem 1rem;
}
.slide-progress-bar {
  height: 3px;
  background: rgba(255,255,255,0.08);
  border-radius: 1.5px;
  overflow: hidden;
  cursor: pointer;
}
.slide-progress-fill {
  height: 100%;
  background: #f97316;
  border-radius: 1.5px;
  transition: width 0.2s ease;
}
.slide-loading {
  color: #f97316;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.slide-loading::before {
  content: '';
  width: 18px;
  height: 18px;
  border: 2px solid rgba(249, 115, 22, 0.2);
  border-top-color: #f97316;
  border-radius: 50%;
  animation: slide-spin 0.8s linear infinite;
}
@keyframes slide-spin {
  to { transform: rotate(360deg); }
}
.slide-footer {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.6rem 1rem;
  background: rgba(255,255,255,0.02);
  border-top: 1px solid rgba(255,255,255,0.05);
}
.slide-footer a {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  background: #f97316;
  color: white;
  padding: 0.4rem 0.9rem;
  border-radius: 2rem;
  text-decoration: none;
  font-weight: 500;
  font-size: 0.75rem;
  transition: all 0.15s;
}
.slide-footer a:hover {
  background: #ea580c;
  color: white;
}
.slide-footer a.secondary {
  background: transparent;
  color: #94a3b8;
  border: 1px solid rgba(255,255,255,0.15);
}
.slide-footer a.secondary:hover {
  background: rgba(255,255,255,0.05);
  color: #f8fafc;
}
@media (max-width: 600px) {
  .slide-header { flex-direction: column; gap: 0.5rem; padding: 0.5rem 0.75rem; }
  .slide-toolbar button { width: 28px; height: 28px; }
  .slide-canvas-wrapper { min-height: 260px; padding: 0.5rem; }
  .slide-canvas { max-height: 220px; }
}
</style>

<div class="slide-viewer-container" id="slide-viewer-17_acceleration">
  <div class="slide-header">
    <div class="slide-title">
      <span>🔥</span>
      <span>Slide Deck</span>
      <span class="slide-subtitle">· AI-generated</span>
    </div>
    <div class="slide-toolbar">
      <div class="slide-nav-group">
        <button onclick="slideNav('17_acceleration', -1)" title="Previous">‹</button>
        <span class="slide-page-info"><span id="slide-num-17_acceleration">1</span> / <span id="slide-count-17_acceleration">-</span></span>
        <button onclick="slideNav('17_acceleration', 1)" title="Next">›</button>
      </div>
      <div class="slide-zoom-group">
        <button onclick="slideZoom('17_acceleration', -0.25)" title="Zoom out">−</button>
        <button onclick="slideZoom('17_acceleration', 0.25)" title="Zoom in">+</button>
      </div>
    </div>
  </div>
  <div class="slide-canvas-wrapper">
    <div id="slide-loading-17_acceleration" class="slide-loading">Loading slides...</div>
    <canvas id="slide-canvas-17_acceleration" class="slide-canvas" style="display:none;"></canvas>
  </div>
  <div class="slide-progress-wrapper">
    <div class="slide-progress-bar" onclick="slideProgress('17_acceleration', event)">
      <div class="slide-progress-fill" id="slide-progress-17_acceleration" style="width: 0%;"></div>
    </div>
  </div>
  <div class="slide-footer">
    <a href="../_static/slides/17_acceleration.pdf" download>⬇ Download</a>
    <a href="#" onclick="slideFullscreen('17_acceleration'); return false;" class="secondary">⛶ Fullscreen</a>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<script>
(function() {
  if (window.slideViewersInitialized) return;
  window.slideViewersInitialized = true;

  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

  window.slideViewers = {};

  window.initSlideViewer = function(id, pdfUrl) {
    const viewer = { pdf: null, page: 1, scale: 1.3, rendering: false, pending: null };
    window.slideViewers[id] = viewer;

    const canvas = document.getElementById('slide-canvas-' + id);
    const ctx = canvas.getContext('2d');

    function render(num) {
      viewer.rendering = true;
      viewer.pdf.getPage(num).then(function(page) {
        const viewport = page.getViewport({scale: viewer.scale});
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        page.render({canvasContext: ctx, viewport: viewport}).promise.then(function() {
          viewer.rendering = false;
          if (viewer.pending !== null) { render(viewer.pending); viewer.pending = null; }
        });
      });
      document.getElementById('slide-num-' + id).textContent = num;
      document.getElementById('slide-progress-' + id).style.width = (num / viewer.pdf.numPages * 100) + '%';
    }

    function queue(num) { if (viewer.rendering) viewer.pending = num; else render(num); }

    pdfjsLib.getDocument(pdfUrl).promise.then(function(pdf) {
      viewer.pdf = pdf;
      document.getElementById('slide-count-' + id).textContent = pdf.numPages;
      document.getElementById('slide-loading-' + id).style.display = 'none';
      canvas.style.display = 'block';
      render(1);
    }).catch(function() {
      document.getElementById('slide-loading-' + id).innerHTML = 'Unable to load. <a href="' + pdfUrl + '" style="color:#f97316;">Download PDF</a>';
    });

    viewer.queue = queue;
  };

  window.slideNav = function(id, dir) {
    const v = window.slideViewers[id];
    if (!v || !v.pdf) return;
    const newPage = v.page + dir;
    if (newPage >= 1 && newPage <= v.pdf.numPages) { v.page = newPage; v.queue(newPage); }
  };

  window.slideZoom = function(id, delta) {
    const v = window.slideViewers[id];
    if (!v) return;
    v.scale = Math.max(0.5, Math.min(3, v.scale + delta));
    v.queue(v.page);
  };

  window.slideProgress = function(id, event) {
    const v = window.slideViewers[id];
    if (!v || !v.pdf) return;
    const bar = event.currentTarget;
    const pct = (event.clientX - bar.getBoundingClientRect().left) / bar.offsetWidth;
    const newPage = Math.max(1, Math.min(v.pdf.numPages, Math.ceil(pct * v.pdf.numPages)));
    if (newPage !== v.page) { v.page = newPage; v.queue(newPage); }
  };

  window.slideFullscreen = function(id) {
    const el = document.getElementById('slide-viewer-' + id);
    if (el.requestFullscreen) el.requestFullscreen();
    else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
  };
})();

initSlideViewer('17_acceleration', '../_static/slides/17_acceleration.pdf');
</script>
```
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

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# GEMM for N=1024: FLOPs = 2 * N^3, memory = 3 * N^2 elements * 4 bytes (float32)
N_blas = 1024
blas_flops = 2 * N_blas**3
blas_elements = 3 * N_blas**2
blas_bytes = blas_elements * 4
blas_data_mb = blas_bytes / 1024**2
blas_ai = blas_flops / blas_bytes

glue("blas_flops_billions", f"{blas_flops / 1e9:.1f}")
glue("blas_data_mb", f"{blas_data_mb:.0f}")
glue("blas_ai", f"{blas_ai:.0f}")
```

BLAS provides three levels of operations, each with different performance characteristics:

- **Level 1**: Vector operations (AXPY: y = αx + y). These are memory-bound with low arithmetic intensity.
- **Level 2**: Matrix-vector operations (GEMV: y = αAx + βy). Better arithmetic intensity but still memory-limited.
- **Level 3**: Matrix-matrix operations (GEMM: C = αAB + βC). High arithmetic intensity, compute-bound.

Matrix multiplication (GEMM) dominates neural network training because every linear layer, every attention mechanism, and every convolution ultimately reduces to matrix multiplication. GEMM performs 2N³ floating-point operations while reading only 3N² elements from memory. For a 1024×1024 matrix, that's {glue:text}`blas_flops_billions` billion operations on just {glue:text}`blas_data_mb` MB of data - an arithmetic intensity of {glue:text}`blas_ai` FLOPs/byte. This high ratio of computation to memory access makes GEMM perfect for hardware acceleration.

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

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Kernel fusion memory traffic analysis
# 4 million element tensor, float32 (4 bytes each)
num_elements = 4_000_000
bytes_per_element = 4
tensor_bytes = num_elements * bytes_per_element
tensor_mb = tensor_bytes / 1024**2

# Unfused: 7 intermediate arrays created and consumed
# Each intermediate array is written then read = 2 memory ops per intermediate
# Plus 1 input read + 1 output write = 7*2 + 2 = 16 memory operations
# But the original text counts: "7 reads + 7 writes per element" = 14 per element
# Total memory ops = 14 per element => 14 * tensor_size_in_bytes
unfused_mem_ops = 14
unfused_traffic_bytes = unfused_mem_ops * tensor_bytes
unfused_traffic_mb = unfused_traffic_bytes / 1024**2

# Fused: 1 read + 1 write = 2 memory operations
fused_mem_ops = 2
fused_traffic_bytes = fused_mem_ops * tensor_bytes
fused_traffic_mb = fused_traffic_bytes / 1024**2

# Bandwidth calculation: 50 GB/s
bandwidth_gb_s = 50
bandwidth_bytes_s = bandwidth_gb_s * 1e9
unfused_time_ms = (unfused_traffic_bytes / bandwidth_bytes_s) * 1000
fused_time_ms = (fused_traffic_bytes / bandwidth_bytes_s) * 1000
fusion_speedup = unfused_time_ms / fused_time_ms

glue("fusion_tensor_elements", f"{num_elements:,}")
glue("fusion_tensor_mb", f"{tensor_mb:.0f}")
glue("fusion_unfused_mem_ops", f"{num_elements * unfused_mem_ops / 1e6:.0f}")
glue("fusion_unfused_traffic_mb", f"{unfused_traffic_mb:.0f}")
glue("fusion_unfused_time_ms", f"{unfused_time_ms:.2f}")
glue("fusion_fused_traffic_mb", f"{fused_traffic_mb:.0f}")
glue("fusion_fused_time_ms", f"{fused_time_ms:.2f}")
glue("fusion_speedup", f"{fusion_speedup:.1f}")
```

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

Each temporary array allocation writes to memory, and each subsequent operation reads from memory. For a {glue:text}`fusion_tensor_elements` element tensor, this unfused version performs {glue:text}`fusion_unfused_mem_ops` million memory operations (7 reads + 7 writes per element). Memory bandwidth on a typical CPU is around 50 GB/s, so moving {glue:text}`fusion_unfused_traffic_mb` MB takes {glue:text}`fusion_unfused_time_ms` milliseconds - just for memory traffic, before any computation.

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

Now there are only two memory operations: read the input, write the output. For the same {glue:text}`fusion_tensor_elements` element tensor, that's just {glue:text}`fusion_fused_traffic_mb` MB of memory traffic, completing in {glue:text}`fusion_fused_time_ms` milliseconds. The fused version is {glue:text}`fusion_speedup`x faster purely from memory bandwidth reduction, even though both versions perform the same arithmetic.

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

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Element-wise addition: AI = N / (3N * 4) = 1/12
ai_elemwise = 1 / 12

# Matrix multiplication: AI = 2N^3 / (3N^2 * 4) = N/6
N_roof = 1024
ai_matmul_formula = "N/6"
ai_matmul_1024 = N_roof / 6
ai_ratio = ai_matmul_1024 / ai_elemwise

glue("roof_ai_elemwise", f"{ai_elemwise:.3f}")
glue("roof_ai_matmul_1024", f"{ai_matmul_1024:.0f}")
glue("roof_ai_ratio", f"{ai_ratio:.0f}")
```

Not all operations are created equal. The roofline model helps predict whether an operation will be limited by memory bandwidth or computational throughput. Arithmetic intensity is the ratio of floating-point operations to bytes transferred:

```
Arithmetic Intensity (AI) = FLOPs / Bytes
```

For element-wise addition of two N-element arrays:
- FLOPs: N (one addition per element)
- Bytes: 3N × 4 = 12N (read A, read B, write C, each 4 bytes for float32)
- AI = N / 12N = {glue:text}`roof_ai_elemwise` FLOPs/byte

For matrix multiplication of N×N matrices:
- FLOPs: 2N³ (N³ multiplications + N³ additions)
- Bytes: 3N² × 4 = 12N² (read A, read B, write C)
- AI = 2N³ / 12N² = N/6 FLOPs/byte

For a 1024×1024 matrix: AI = {glue:text}`roof_ai_matmul_1024` FLOPs/byte. Matrix multiplication performs {glue:text}`roof_ai_ratio`x more computation per byte transferred than element-wise addition. This is why GPUs excel at matrix operations but struggle with element-wise ops.

| Operation | Arithmetic Intensity | Bottleneck | Optimization Strategy |
|-----------|---------------------|------------|----------------------|
| Element-wise add | ~0.08 FLOPs/byte | Memory bandwidth | Kernel fusion |
| Element-wise multiply | ~0.08 FLOPs/byte | Memory bandwidth | Kernel fusion |
| GELU activation | ~1.0 FLOPs/byte | Memory bandwidth | Kernel fusion |
| Matrix multiply (1024×1024) | ~{glue:text}`roof_ai_matmul_1024` FLOPs/byte | Compute throughput | Vectorization, tiling |

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

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q1: Arithmetic Intensity for 1024x1024 float32 matmul
N_q1 = 1024
q1_flops = 2 * N_q1**3
q1_matrix_bytes = N_q1 * N_q1 * 4  # float32 = 4 bytes
q1_matrix_mb = q1_matrix_bytes / 1024**2
q1_total_bytes = 3 * q1_matrix_bytes  # read A + read B + write C
q1_total_mb = 3 * q1_matrix_mb
q1_ai = q1_flops / q1_total_bytes

glue("q1_flops", f"{q1_flops:,}")
glue("q1_matrix_mb", f"{q1_matrix_mb:.0f}")
glue("q1_total_mb", f"{q1_total_mb:.0f}")
glue("q1_total_bytes", f"{q1_total_bytes:,}")
glue("q1_ai", f"{q1_ai:.0f}")
```

**Q1: Arithmetic Intensity**

Matrix multiplication of two 1024×1024 float32 matrices performs {glue:text}`q1_flops` FLOPs. It reads {glue:text}`q1_matrix_mb` MB (matrix A) + {glue:text}`q1_matrix_mb` MB (matrix B) = {glue:text}`q1_total_mb` MB and writes {glue:text}`q1_matrix_mb` MB (matrix C) = {glue:text}`q1_total_mb` MB total. What is the arithmetic intensity?

```{admonition} Answer
:class: dropdown

Arithmetic Intensity = {glue:text}`q1_flops` FLOPs / {glue:text}`q1_total_bytes` bytes = **~{glue:text}`q1_ai` FLOPs/byte**

This high arithmetic intensity (compared to ~0.08 for element-wise ops) is why matrix multiplication is ideal for GPUs and why it dominates neural network training time.
```

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q2: Memory bandwidth savings from kernel fusion
q2_elements = 1_000_000
q2_tensor_bytes = q2_elements * 4  # float32
q2_tensor_mb = q2_tensor_bytes / 1024**2

# Unfused: 7 intermediate arrays => 7 reads + 7 writes + 1 input read + 1 output write = 16 ops
q2_unfused_ops = 16
q2_unfused_mb = q2_unfused_ops * q2_tensor_mb

# Fused: 1 input read + 1 output write = 2 ops
q2_fused_ops = 2
q2_fused_mb = q2_fused_ops * q2_tensor_mb

q2_savings_mb = q2_unfused_mb - q2_fused_mb
q2_savings_pct = (q2_savings_mb / q2_unfused_mb) * 100

glue("q2_elements", f"{q2_elements:,}")
glue("q2_tensor_mb", f"{q2_tensor_mb:.0f}")
glue("q2_unfused_mb", f"{q2_unfused_mb:.0f}")
glue("q2_fused_mb", f"{q2_fused_mb:.0f}")
glue("q2_savings_mb", f"{q2_savings_mb:.0f}")
glue("q2_savings_pct", f"{q2_savings_pct:.1f}")
```

**Q2: Memory Bandwidth Savings**

Your fused GELU processes a tensor with {glue:text}`q2_elements` elements ({glue:text}`q2_tensor_mb` MB as float32). The unfused version creates 7 intermediate arrays. How much memory bandwidth does fusion save?

```{admonition} Answer
:class: dropdown

**Unfused**: 7 reads + 7 writes + 1 input read + 1 output write = 16 memory operations × {glue:text}`q2_tensor_mb` MB = **{glue:text}`q2_unfused_mb` MB**

**Fused**: 1 input read + 1 output write = 2 memory operations × {glue:text}`q2_tensor_mb` MB = **{glue:text}`q2_fused_mb` MB**

**Savings**: {glue:text}`q2_unfused_mb` - {glue:text}`q2_fused_mb` = **{glue:text}`q2_savings_mb` MB saved ({glue:text}`q2_savings_pct`% reduction)**

For typical CPUs with ~50 GB/s bandwidth, this saves ~1 millisecond per GELU call. In a transformer with 96 GELU activations per forward pass, that's 96ms saved - enough to improve throughput by 10-20%.
```

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue
import math

# Q3: Cache tiling for 2048x2048 float32 with 256 KB L2
q3_cache_kb = 256
q3_cache_bytes = q3_cache_kb * 1024  # binary: 262,144 bytes
q3_matrix_dim = 2048
q3_matrix_bytes = q3_matrix_dim * q3_matrix_dim * 4
q3_matrix_mb = q3_matrix_bytes / 1024**2

# 3 tiles must fit in cache: 3 * tile^2 * 4 <= cache_bytes
q3_max_tile_sq = q3_cache_bytes / 12  # 3 tiles * 4 bytes
q3_max_tile = math.isqrt(int(q3_max_tile_sq))  # integer sqrt (floor)
# Practical power-of-2 tile size
q3_practical_tile = 128
q3_practical_bytes = 3 * q3_practical_tile**2 * 4
q3_practical_kb = q3_practical_bytes / 1024

glue("q3_cache_kb", f"{q3_cache_kb}")
glue("q3_matrix_dim", f"{q3_matrix_dim}")
glue("q3_matrix_mb", f"{q3_matrix_mb:.0f}")
glue("q3_cache_bytes", f"{q3_cache_bytes:,}")
glue("q3_max_tile_sq", f"{q3_max_tile_sq:,.0f}")
glue("q3_max_tile", f"{q3_max_tile}")
glue("q3_practical_tile", f"{q3_practical_tile}")
glue("q3_practical_kb", f"{q3_practical_kb:.0f}")
```

**Q3: Cache Tiling**

A CPU has {glue:text}`q3_cache_kb` KB L2 cache. You're multiplying two {glue:text}`q3_matrix_dim`×{glue:text}`q3_matrix_dim` float32 matrices ({glue:text}`q3_matrix_mb` MB each). What tile size keeps the working set in L2 cache?

```{admonition} Answer
:class: dropdown

For tiled multiplication, we need 3 tiles in cache simultaneously:
- Tile from matrix A: tile_size × tile_size × 4 bytes
- Tile from matrix B: tile_size × tile_size × 4 bytes
- Output tile: tile_size × tile_size × 4 bytes

Total: 3 × tile_size² × 4 bytes ≤ {glue:text}`q3_cache_kb` KB

Solving: tile_size² ≤ {glue:text}`q3_cache_bytes` / 12 = {glue:text}`q3_max_tile_sq`

**tile_size ≈ {glue:text}`q3_max_tile`**

In practice, use powers of 2: **{glue:text}`q3_practical_tile` works well** (3 × {glue:text}`q3_practical_tile`² × 4 = {glue:text}`q3_practical_kb` KB, leaving room for other data).
```

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q4: BLAS performance calculation
q4_flops = 2.15e9
q4_time_s = 0.01  # 10ms
q4_gflops = q4_flops / (q4_time_s * 1e9)
q4_peak_gflops = 500
q4_efficiency_pct = (q4_gflops / q4_peak_gflops) * 100

glue("q4_gflops", f"{q4_gflops:.0f}")
glue("q4_efficiency_pct", f"{q4_efficiency_pct:.0f}")
```

**Q4: BLAS Performance**

Your vectorized matmul completes a 1024×1024 multiplication in 10ms. The operation requires 2.15 billion FLOPs. What is your achieved performance in GFLOPS?

```{admonition} Answer
:class: dropdown

GFLOPS = 2,150,000,000 FLOPs / (0.01 seconds × 1,000,000,000) = **{glue:text}`q4_gflops` GFLOPS**

For reference:
- Modern CPU peak: 500-1000 GFLOPS (AVX-512)
- Your efficiency: {glue:text}`q4_gflops`/500 = **{glue:text}`q4_efficiency_pct`% of peak** (typical for real code)
- GPU equivalent: ~50 TFLOPS (230x faster than single CPU core)
```

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q5: Speedup from fusion
q5_unfused_ms = 8.0
q5_fused_ms = 2.5
q5_speedup = q5_unfused_ms / q5_fused_ms
q5_mem_overhead_pct = ((q5_unfused_ms - q5_fused_ms) / q5_unfused_ms) * 100

glue("q5_speedup", f"{q5_speedup:.1f}")
glue("q5_mem_overhead_pct", f"{q5_mem_overhead_pct:.2f}")
glue("q5_mem_overhead_approx", f"{round(q5_mem_overhead_pct)}")
```

**Q5: Speedup from Fusion**

Unfused GELU takes 8ms on a 2000×2000 tensor. Fused GELU takes 2.5ms. What percentage of the unfused time was memory overhead?

```{admonition} Answer
:class: dropdown

Speedup = 8ms / 2.5ms = **{glue:text}`q5_speedup`x faster**

Assuming both versions do the same computation, the difference is memory bandwidth:
- Memory overhead = (8 - 2.5) / 8 = **{glue:text}`q5_mem_overhead_pct`%**

Nearly **{glue:text}`q5_mem_overhead_approx`% of the unfused version's time** was spent waiting for memory! This is typical for element-wise operations with low arithmetic intensity.
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

```{seealso} Coming Up: Module 18 - Memoization

Implement caching and memoization strategies to eliminate redundant computations. You'll build KV-caching for transformer generation, cache repeated forward passes, and store attention patterns for dramatic speedups in production inference.
```

**Preview - How Acceleration Gets Used in Future Modules:**

| Module | What It Does | Your Acceleration In Action |
|--------|--------------|---------------------------|
| **18: Memoization** | Cache repeated computations | Fused kernels + KV cache minimize memory traffic |
| **19: Benchmarking** | Systematic performance measurement | `benchmark(vectorized_matmul, sizes=[128, 256, 512])` |
| **20: Capstone** | Complete optimized model | Acceleration throughout model pipeline |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/17_acceleration/acceleration.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/17_acceleration/17_acceleration.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```

```{warning} Performance Note

Acceleration techniques depend on hardware. Results will vary between CPUs. Use Module 14's profiler to measure your specific hardware's characteristics.
```
