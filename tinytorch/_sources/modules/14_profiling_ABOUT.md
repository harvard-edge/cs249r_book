# Module 14: Profiling

:::{admonition} Module Info
:class: note

**OPTIMIZATION TIER** | Difficulty: ●●○○ | Time: 3-5 hours | Prerequisites: 01-13

**Prerequisites: Modules 01-13** means you should have:
- Built the complete ML stack (Modules 01-08)
- Implemented CNN architectures (Module 09) or Transformers (Modules 10-13)
- Models to profile and optimize

**Why these prerequisites**: You'll profile models built in Modules 1-13. Understanding the implementations helps you interpret profiling results (e.g., why attention is memory-bound).
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F14_profiling%2F14_profiling.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/14_profiling/14_profiling.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/14_profiling.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Profiling is the foundation of performance optimization. Before making a model faster or smaller, you need to measure where time and memory go. In this module, you'll build professional profiling tools that measure parameters, FLOPs, memory usage, and latency with statistical rigor.

Every optimization decision starts with measurement. Is your model memory-bound or compute-bound? Which layers consume the most resources? How does batch size affect throughput? Your profiler will answer these questions with data, not guesses, enabling the targeted optimizations in later modules.

By the end, you'll have built the same measurement infrastructure used by production ML teams to make data-driven optimization decisions.

## The Optimization Tier Flow

Profiling (Module 14) is the gateway to the Optimization tier, which follows **Measure → Transform → Validate**:

```
Profiling (14) → Model-Level (15-16) → Runtime (17-18) → Benchmarking (19)
     ↓                  ↓                    ↓                  ↓
 "What's slow?"   "Shrink the model"   "Speed up execution"  "Did it work?"
```

**Model-Level Optimizations (15-16)**: Change the model itself
- Quantization: FP32 → INT8 for 4× compression
- Compression: Prune unnecessary weights

**Runtime Optimizations (17-18)**: Change how execution happens
- Acceleration: Vectorization, kernel fusion (general-purpose)
- Memoization: KV-cache for transformers (domain-specific)

You can't optimize what you can't measure. That's why profiling comes first.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** a comprehensive Profiler class that measures parameters, FLOPs, memory, and latency
- **Analyze** performance characteristics to identify compute-bound vs memory-bound workloads
- **Master** statistical measurement techniques with warmup runs and outlier handling
- **Connect** profiling insights to optimization opportunities in quantization, compression, and caching
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Profiler Class
flowchart LR
    subgraph "Your Profiler Class"
        A["Parameter Counter<br/>count_parameters()"]
        B["FLOP Counter<br/>count_flops()"]
        C["Memory Tracker<br/>measure_memory()"]
        D["Latency Timer<br/>measure_latency()"]
        E["Analysis Tools<br/>profile_forward_pass()"]
    end

    A --> E
    B --> E
    C --> E
    D --> E

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
    style E fill:#e2d5f1
```

**Implementation roadmap:**

| Step | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `count_parameters()` | Model size and memory footprint |
| 2 | `count_flops()` | Computational cost estimation |
| 3 | `measure_memory()` | Activation and gradient memory tracking |
| 4 | `measure_latency()` | Statistical timing with warmup |
| 5 | `profile_forward_pass()` | Comprehensive performance analysis |
| 6 | `profile_backward_pass()` | Training cost estimation |

**The pattern you'll enable:**
```python
# Comprehensive model analysis for optimization decisions
profiler = Profiler()
profile = profiler.profile_forward_pass(model, input_data)
print(f"Bottleneck: {profile['bottleneck']}")  # "memory" or "compute"
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- GPU profiling (we measure CPU performance with NumPy)
- Distributed profiling (that's for multi-GPU setups)
- CUDA kernel profilers (PyTorch uses `torch.profiler` for GPU analysis)
- Layer-by-layer visualization dashboards (TensorBoard provides this)

**You are building the measurement foundation.** Visualization and GPU profiling come with production frameworks.

## API Reference

This section provides a quick reference for the Profiler class you'll build. Use it while implementing and debugging.

### Constructor

```python
Profiler()
```
Initializes profiler with measurement tracking structures.

### Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `count_parameters` | `count_parameters(model) -> int` | Count total trainable parameters |
| `count_flops` | `count_flops(model, input_shape) -> int` | Count FLOPs per sample (batch-size independent) |
| `measure_memory` | `measure_memory(model, input_shape) -> Dict` | Measure memory usage components |
| `measure_latency` | `measure_latency(model, input_tensor, warmup, iterations) -> float` | Measure inference latency in milliseconds |

### Analysis Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `profile_layer` | `profile_layer(layer, input_shape) -> Dict` | Comprehensive single-layer profile |
| `profile_forward_pass` | `profile_forward_pass(model, input_tensor) -> Dict` | Complete forward pass analysis |
| `profile_backward_pass` | `profile_backward_pass(model, input_tensor) -> Dict` | Training iteration analysis |

## Core Concepts

This section covers the fundamental ideas you need to understand profiling deeply. Measurement is the foundation of optimization, and understanding what you're measuring matters as much as how you measure it.

### Why Profile First

Optimization without measurement is guessing. You might spend days optimizing the wrong bottleneck, achieving minimal speedup while the real problem goes untouched. Profiling reveals ground truth: where time and memory actually go, not where you think they go.

Consider a transformer model running slowly. Is it the attention mechanism? The feed-forward layers? Matrix multiplications? Memory transfers? Without profiling, you're guessing. With profiling, you know. If 80% of time is in attention and it's memory-bound, you know exactly what to optimize and how.

The profiling workflow follows a systematic process. You measure first to establish a baseline. Then you analyze the measurements to identify bottlenecks. Next you optimize the critical path, not every operation. Finally you measure again to verify improvement. This cycle repeats until you hit performance targets.

Your profiler implements the measurement and analysis steps, providing the data needed for optimization decisions in later modules.

### Timing Operations

Accurate timing is harder than it looks. Systems have variance, warmup effects, and measurement overhead. Your `measure_latency` method handles these challenges with statistical rigor:

```python
def measure_latency(self, model, input_tensor, warmup: int = 10, iterations: int = 100) -> float:
    """Measure model inference latency with statistical rigor."""
    # Warmup runs to stabilize performance
    for _ in range(warmup):
        _ = model.forward(input_tensor)

    # Measurement runs
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = model.forward(input_tensor)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Calculate statistics - use median for robustness
    times = np.array(times)
    median_latency = np.median(times)

    return float(median_latency)
```

The warmup phase is critical. The first few runs are slower due to cold CPU caches, Python interpreter warmup, and NumPy initialization. Running 10+ warmup iterations ensures the system reaches steady state before measurement begins.

Using median instead of mean makes the measurement robust against outliers. If the operating system interrupts your process once during measurement, that outlier won't skew the result. The median captures typical performance, not worst-case spikes.

### Memory Profiling

Memory profiling reveals three distinct components: parameter memory (model weights), activation memory (forward pass intermediate values), and gradient memory (backward pass derivatives). Each has different characteristics and optimization strategies.

Here's how your profiler tracks memory usage:

```python
def measure_memory(self, model, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """Measure memory usage during forward pass."""
    # Start memory tracking
    tracemalloc.start()

    # Calculate parameter memory
    param_count = self.count_parameters(model)
    parameter_memory_bytes = param_count * BYTES_PER_FLOAT32
    parameter_memory_mb = parameter_memory_bytes / MB_TO_BYTES

    # Create input and measure activation memory
    dummy_input = Tensor(np.random.randn(*input_shape))
    input_memory_bytes = dummy_input.data.nbytes

    # Estimate activation memory (simplified)
    activation_memory_bytes = input_memory_bytes * 2  # Rough estimate
    activation_memory_mb = activation_memory_bytes / MB_TO_BYTES

    # Run forward pass to measure peak memory usage
    _ = model.forward(dummy_input)

    # Get peak memory
    _current_memory, peak_memory = tracemalloc.get_traced_memory()
    peak_memory_mb = (peak_memory - _baseline_memory) / MB_TO_BYTES

    tracemalloc.stop()

    # Calculate efficiency metrics
    useful_memory = parameter_memory_mb + activation_memory_mb
    memory_efficiency = useful_memory / max(peak_memory_mb, 0.001)  # Avoid division by zero

    return {
        'parameter_memory_mb': parameter_memory_mb,
        'activation_memory_mb': activation_memory_mb,
        'peak_memory_mb': max(peak_memory_mb, useful_memory),
        'memory_efficiency': min(memory_efficiency, 1.0)
    }
```

Parameter memory is persistent and constant regardless of batch size. A model with 125 million parameters uses 500 MB (125M × 4 bytes per float32) whether you're processing one sample or a thousand.

Activation memory scales with batch size. Doubling the batch doubles activation memory. This is why large batch training requires more GPU memory than inference.

Gradient memory matches parameter memory exactly. Every parameter needs a gradient during training, adding another 500 MB for a 125M parameter model.

### Bottleneck Identification

The most important profiling insight is whether your workload is compute-bound or memory-bound. This determines which optimizations will help.

Compute-bound workloads are limited by arithmetic throughput. The CPU or GPU can't compute fast enough to keep up with available memory bandwidth. Optimizations focus on better algorithms, vectorization, and reducing FLOPs.

Memory-bound workloads are limited by data movement. The hardware can compute faster than it can load data from memory. Optimizations focus on reducing memory transfers, improving cache locality, and data layout.

Your profiler identifies bottlenecks by comparing computational intensity to memory bandwidth. If you're achieving low GFLOP/s despite high theoretical compute capability, you're memory-bound. If you're achieving high GFLOP/s and high computational efficiency, you're compute-bound.

### Profiling Tools

Your implementation uses Python's built-in profiling tools: `time.perf_counter()` for high-precision timing and `tracemalloc` for memory tracking. These provide the foundation for accurate measurement.

`time.perf_counter()` uses the system's highest-resolution timer, typically nanosecond precision. It measures wall-clock time, which includes all system effects (cache misses, context switches) that affect real-world performance.

`tracemalloc` tracks Python memory allocations with byte-level precision. It records both current and peak memory usage, letting you identify memory spikes during execution.

Production profilers add GPU support (CUDA events, NVTX markers), distributed tracing (for multi-GPU setups), and kernel-level analysis. But the core concepts remain the same: measure, analyze, identify bottlenecks, optimize.

## Production Context

### Your Implementation vs. PyTorch

Your TinyTorch Profiler and PyTorch's profiling tools share the same conceptual foundation. The differences are in implementation detail: PyTorch adds GPU support, kernel-level profiling, and distributed tracing. But the core metrics (parameters, FLOPs, memory, latency) are identical.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Parameter counting** | Direct tensor size | `model.parameters()` |
| **FLOP counting** | Per-layer formulas | FlopCountAnalysis (fvcore) |
| **Memory tracking** | tracemalloc | torch.profiler, CUDA events |
| **Latency measurement** | time.perf_counter() | torch.profiler, NVTX |
| **GPU profiling** | ✗ CPU only | ✓ CUDA kernels, memory |
| **Distributed** | ✗ Single process | ✓ Multi-GPU, NCCL |

### Code Comparison

The following comparison shows equivalent profiling operations in TinyTorch and PyTorch. Notice how the concepts transfer directly, even though PyTorch provides more sophisticated tooling.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.perf.profiling import Profiler

# Create profiler
profiler = Profiler()

# Profile model
params = profiler.count_parameters(model)
flops = profiler.count_flops(model, input_shape)
memory = profiler.measure_memory(model, input_shape)
latency = profiler.measure_latency(model, input_tensor)

# Comprehensive analysis
profile = profiler.profile_forward_pass(model, input_tensor)
print(f"Bottleneck: {profile['bottleneck']}")
print(f"GFLOP/s: {profile['gflops_per_second']:.2f}")
```
````

````{tab-item} ⚡ PyTorch
```python
import torch
from torch.profiler import profile, ProfilerActivity

# Count parameters
params = sum(p.numel() for p in model.parameters())

# Profile with PyTorch profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(input_tensor)

# Analyze results
print(prof.key_averages().table(sort_by="cpu_time_total"))

# FLOPs (requires fvcore)
from fvcore.nn import FlopCountAnalysis
flops = FlopCountAnalysis(model, input_tensor)
print(f"FLOPs: {flops.total()}")
```
````
`````

Let's walk through the comparison:

- **Parameter counting**: Both frameworks count total trainable parameters. TinyTorch uses `count_parameters()`, PyTorch uses `sum(p.numel() for p in model.parameters())`.
- **FLOP counting**: TinyTorch implements per-layer formulas. PyTorch uses the `fvcore` library's `FlopCountAnalysis` for more sophisticated analysis.
- **Memory tracking**: TinyTorch uses `tracemalloc`. PyTorch profiler tracks CUDA memory events for GPU memory analysis.
- **Latency measurement**: TinyTorch uses `time.perf_counter()` with warmup. PyTorch profiler uses CUDA events for precise GPU timing.
- **Analysis output**: Both provide bottleneck identification and throughput metrics. PyTorch adds kernel-level detail and distributed profiling.

```{tip} What's Identical

The profiling workflow: measure parameters, FLOPs, memory, and latency to identify bottlenecks. Production frameworks add GPU support and more sophisticated analysis, but the core measurement principles you're learning here transfer directly.
```

### Why Profiling Matters at Scale

To appreciate profiling's importance, consider production ML systems:

- **GPT-3 (175B parameters)**: 700 GB model size at FP32. Profiling reveals which layers to quantize for deployment.
- **BERT training**: 80% of time in self-attention. Profiling identifies FlashAttention as the optimization to implement.
- **Image classification**: Batch size 256 uses 12 GB GPU memory. Profiling shows 10 GB is activations, suggesting gradient checkpointing.

A single profiling session can reveal optimization opportunities worth 10× speedups or 4× memory reduction. Understanding profiling isn't just academic; it's essential for deploying real ML systems.

## Check Your Understanding

Test yourself with these systems thinking questions about profiling and performance measurement.

**Q1: Parameter Memory Calculation**

A transformer model has 12 layers, each with a feed-forward network containing two Linear layers: Linear(768, 3072) and Linear(3072, 768). How much memory do the feed-forward network parameters consume across all layers?

```{admonition} Answer
:class: dropdown

Each feed-forward network:
- First layer: (768 × 3072) + 3072 = 2,362,368 parameters
- Second layer: (3072 × 768) + 768 = 2,360,064 parameters
- Total per layer: 4,722,432 parameters

Across 12 layers: 12 × 4,722,432 = 56,669,184 parameters

Memory: 56,669,184 × 4 bytes = 226,676,736 bytes ≈ **227 MB**

This is just the feed-forward networks. Attention adds more parameters.
```

**Q2: FLOP Counting and Computational Cost**

A Linear(512, 512) layer processes a batch of 64 samples. Your profiler's `count_flops()` method returns FLOPs per sample (batch-size independent). How many FLOPs are required for one sample? For the whole batch, if each sample is processed independently?

```{admonition} Answer
:class: dropdown

Per-sample FLOPs (what `count_flops()` returns): 512 × 512 × 2 = **524,288 FLOPs**

Note: The `count_flops()` method is batch-size independent. It returns per-sample FLOPs whether you pass input_shape=(1, 512) or (64, 512).

If processing a batch of 64 samples: 64 × 524,288 = 33,554,432 total FLOPs

Minimum latency at 50 GFLOP/s: 33,554,432 FLOPs ÷ 50 GFLOP/s = **0.67 ms** for the full batch

This assumes perfect computational efficiency (100%). Real latency is higher due to memory bandwidth and overhead.
```

**Q3: Memory Bottleneck Analysis**

A model achieves 5 GFLOP/s on hardware with 100 GFLOP/s peak compute. The memory bandwidth is 50 GB/s. Is this workload compute-bound or memory-bound?

```{admonition} Answer
:class: dropdown

Computational efficiency: 5 GFLOP/s ÷ 100 GFLOP/s = **5% efficiency**

This extremely low efficiency suggests the workload is **memory-bound**. The hardware can compute 100 GFLOP/s but only achieves 5 GFLOP/s because it spends most of the time waiting for data transfers.

Optimization strategy: Focus on reducing memory transfers, improving cache locality, and data layout optimization. Improving the algorithm's FLOPs won't help because compute isn't the bottleneck.
```

**Q4: Training Memory Estimation**

A model has 125M parameters (500 MB). You're training with Adam optimizer. What's the total memory requirement during training, including gradients and optimizer state?

```{admonition} Answer
:class: dropdown

- Parameters: 500 MB
- Gradients: 500 MB (same as parameters)
- Adam momentum: 500 MB (first moment estimates)
- Adam velocity: 500 MB (second moment estimates)

Total: 500 + 500 + 500 + 500 = **2,000 MB (2 GB)**

This is just model state. Activations add more memory that scales with batch size. A typical training run might use 4-8 GB total including activations.
```

**Q5: Latency Measurement Statistics**

You measure latency 100 times and get: median = 10.5 ms, mean = 12.3 ms, min = 10.1 ms, max = 45.2 ms. Which statistic should you report and why?

```{admonition} Answer
:class: dropdown

Report the **median (10.5 ms)** as the typical latency.

The mean (12.3 ms) is skewed by the outlier (45.2 ms), likely caused by OS interruption or garbage collection. The median is robust to outliers and represents typical performance.

For production SLA planning, you might also report p95 or p99 latency (95th or 99th percentile) to capture worst-case behavior without being skewed by extreme outliers.
```

## Further Reading

For students who want to understand the academic foundations and professional practices of ML profiling:

### Seminal Papers

- **Roofline: An Insightful Visual Performance Model** - Williams et al. (2009). Introduces the roofline model for understanding compute vs memory bottlenecks. Essential framework for performance analysis. [ACM CACM](https://doi.org/10.1145/1498765.1498785)

- **PyTorch Profiler: Performance Analysis Tool** - Ansel et al. (2024). Describes PyTorch's production profiling infrastructure. Shows how profiling scales to distributed GPU systems. [arXiv](https://arxiv.org/abs/2404.05033)

- **MLPerf Inference Benchmark** - Reddi et al. (2020). Industry-standard benchmarking methodology for ML systems. Defines rigorous profiling protocols. [arXiv](https://arxiv.org/abs/1911.02549)

### Additional Resources

- **Tool**: [PyTorch Profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) - Production profiling with GPU support
- **Tool**: [TensorFlow Profiler](https://www.tensorflow.org/guide/profiler) - Alternative framework's profiling approach
- **Book**: "Computer Architecture: A Quantitative Approach" - Hennessy & Patterson - Chapter 4 covers memory hierarchy and performance measurement

## What's Next

```{seealso} Coming Up: Module 15 - Quantization

Implement quantization to reduce model size and accelerate inference. You'll use profiling insights to identify which layers benefit most from reduced precision, achieving 4× memory reduction with minimal accuracy loss.
```

**Preview - How Your Profiler Gets Used in Future Modules:**

| Module | What It Does | Your Profiler In Action |
|--------|--------------|------------------------|
| **15: Quantization** | Reduce precision to INT8 | `profile_layer()` identifies quantization candidates |
| **16: Compression** | Prune and compress weights | `count_parameters()` measures compression ratio |
| **17: Acceleration** | Vectorize computations | `measure_latency()` validates speedup |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/14_profiling/14_profiling.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/14_profiling/14_profiling.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
