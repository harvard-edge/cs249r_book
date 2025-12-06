---
title: "Profiling - Performance Measurement for ML Systems"
description: "Build profilers that measure parameters, FLOPs, memory, and latency to guide optimization decisions"
difficulty: "⭐⭐⭐"
time_estimate: "5-6 hours"
prerequisites: ["Modules 01-13 - Complete ML implementation stack"]
next_steps: ["Module 15 - Quantization"]
learning_objectives:
  - "Implement parameter counting to predict model memory requirements"
  - "Build FLOP counters to measure computational complexity across architectures"
  - "Create memory profilers that track allocations and identify usage patterns"
  - "Design timing profilers with statistical rigor to measure latency accurately"
  - "Apply profiling data to identify bottlenecks and prioritize optimizations"
---

# 14. Profiling - Performance Measurement for ML Systems

**OPTIMIZATION TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 5-6 hours

## Overview

Build profiling tools that measure where compute and memory go in ML systems. This module implements parameter counters, FLOP analyzers, memory trackers, and timing profilers with statistical rigor. You'll profile real models to identify bottlenecks—memory-bound vs compute-bound, attention vs feedforward, batch size effects—and use data to guide optimization decisions.

**Optimization Tier Focus**: Modules 1-13 taught you to build ML systems. Modules 14-20 teach you to measure and optimize them. Profiling is the foundation—you can't optimize what you don't measure.

## Why This Matters

### Production Context: Profiling Drives Optimization Economics

Every major ML organization profiles extensively:

- **Google TPU teams** profile every kernel to achieve 40-50% MFU (Model FLOPs Utilization), translating to millions in compute savings
- **OpenAI** profiles GPT training runs to identify gradient checkpointing opportunities, reducing memory by 10× with minimal speed cost
- **Meta** profiles PyTorch inference serving billions of requests daily, using data to guide operator fusion and quantization decisions
- **NVIDIA** uses Nsight profiler to optimize cuDNN kernels, achieving near-theoretical-peak performance on tensor cores

**The Economics**: A 10% optimization on a $10M training run saves $1M. But only if you measure first—guessing wastes engineering time on non-bottlenecks.

### Historical Evolution: From Ad-Hoc Timing to Systematic Measurement

Profiling evolved with ML scale:

- **Pre-2012 (Small models)**: Ad-hoc timing with `time.time()`, no systematic methodology
- **2012-2017 (Deep learning era)**: NVIDIA profiler, TensorBoard timing; focus on GPU utilization
- **2018+ (Production scale)**: Comprehensive profiling (compute, memory, I/O, network); optimization becomes economically critical
- **2020+ (Modern systems)**: Automated profiling guides ML compilers; tools like PyTorch Profiler integrate with training workflows

### What You'll Actually Build

Let's be precise about what you implement in this module:

**You WILL build**:
- Parameter counter: Walks model structure, sums weight and bias elements
- FLOP counter: Calculates theoretical operations for Linear, Conv2d based on dimensions
- Memory profiler: Uses Python's tracemalloc to track allocations during forward/backward
- Timing profiler: Uses time.perf_counter() with warmup runs and statistical analysis (median latency)

**You will NOT build** (these are production tools requiring kernel instrumentation):
- GPU profiler (requires CUDA kernel hooks)
- PyTorch Profiler integration (requires autograd instrumentation)
- Operator-level timeline traces (requires framework integration)

**Why this scope matters**: You'll understand profiling fundamentals that transfer to production tools. The techniques you implement (parameter counting formulas, FLOP calculations, statistical timing) are exactly what PyTorch Profiler and TensorBoard use internally. You're building the same measurement primitives, just without kernel-level instrumentation.

## Learning Objectives

By the end of this module, you will be able to:

- **Count parameters accurately**: Predict model size and memory footprint by counting weights and biases across different layer types
- **Measure computational cost**: Implement FLOP counters that calculate theoretical compute for matrix multiplications, convolutions, and attention operations
- **Track memory usage**: Build memory profilers using tracemalloc to measure parameter, activation, and gradient memory during forward and backward passes
- **Profile latency rigorously**: Create timing profilers with warmup runs, multiple iterations, and statistical analysis (median, confidence intervals)
- **Identify performance bottlenecks**: Analyze profiling data to distinguish memory-bound from compute-bound operations and prioritize optimization efforts

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement Profiler class with parameter counting, FLOP calculation, memory tracking, and latency measurement using time.perf_counter() and tracemalloc
2. **Use**: Profile complete models to measure characteristics, compare MLP vs attention operations, analyze batch size impact on throughput, and benchmark different architectures
3. **Reflect**: Where does compute time actually go in transformers? When is your system memory-bound vs compute-bound? How do measurement choices affect optimization decisions?

## Implementation Guide

### Core Component: Profiler Class

The Profiler class provides comprehensive performance analysis:

```python
class Profiler:
    """Professional-grade ML model profiler.

    Measures parameters, FLOPs, memory, and latency with statistical rigor.
    Used for bottleneck identification and optimization guidance.
    """

    def __init__(self):
        self.measurements = {}
        self.operation_counts = defaultdict(int)

    def count_parameters(self, model) -> int:
        """Count total trainable parameters.

        Returns:
            Total parameter count (e.g., 125M for GPT-2 Small)
        """
        total = 0
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                total += param.data.size  # Count elements
        return total

    def count_flops(self, model, input_shape: Tuple) -> int:
        """Count FLOPs (Floating Point Operations) for forward pass.

        Linear layer: 2 × M × K × N (matmul is M×K @ K×N)
        Conv2d: 2 × output_h × output_w × kernel_h × kernel_w × in_ch × out_ch

        Returns:
            Total FLOPs for one forward pass (hardware-independent)
        """
        # Implementation calculates based on layer type and dimensions

    def measure_memory(self, model, input_shape: Tuple) -> Dict:
        """Measure memory usage during forward pass.

        Uses tracemalloc to track:
        - Parameter memory (weights, biases)
        - Activation memory (intermediate tensors)
        - Peak memory (maximum allocation)

        Returns:
            Dict with memory breakdown in MB
        """
        tracemalloc.start()
        # Run forward pass, measure peak allocation

    def measure_latency(self, model, input_tensor,
                       warmup: int = 10, iterations: int = 100) -> float:
        """Measure inference latency with statistical rigor.

        Protocol:
        1. Warmup runs (cache warming, JIT compilation)
        2. Multiple measurements (statistical significance)
        3. Median calculation (robust to outliers)

        Returns:
            Median latency in milliseconds
        """
        # Warmup runs (discard results)
        for _ in range(warmup):
            _ = model.forward(input_tensor)

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()  # High-precision timer
            _ = model.forward(input_tensor)
            times.append((time.perf_counter() - start) * 1000)  # Convert to ms

        return np.median(times)  # Median is robust to outliers
```

### Parameter Counting: Memory Footprint Analysis

Parameter counting predicts model size and memory requirements:

```python
# Linear layer example
layer = Linear(768, 3072)  # GPT-2 feedforward dimension

# Manual calculation:
weight_params = 768 × 3072 = 2,359,296
bias_params = 3072
total_params = 2,362,368

# Memory at FP32 (4 bytes per parameter):
memory_bytes = 2,362,368 × 4 = 9,449,472 bytes = 9.01 MB

# Profiler implementation:
profiler = Profiler()
count = profiler.count_parameters(layer)
assert count == 2_362_368

# Why this matters:
# GPT-2 Small: 124M params → 496 MB
# GPT-2 XL: 1.5B params → 6.0 GB
# Knowing parameter count predicts deployment hardware requirements
```

**Parameter Counting Strategy**:
- Linear layers: `(input_features × output_features) + output_features`
- Conv2d layers: `(kernel_h × kernel_w × in_channels × out_channels) + out_channels`
- Embeddings: `vocab_size × embedding_dim`
- Attention: Count Q/K/V projection weights separately

### FLOP Counting: Computational Cost Analysis

FLOPs measure compute independently of hardware:

```python
# Matrix multiplication FLOP calculation
# C = A @ B where A is (M, K) and B is (K, N)

def count_matmul_flops(M, K, N):
    """Each output element C[i,j] requires K multiply-adds.

    Total outputs: M × N
    FLOPs per output: 2 × K (multiply + add)
    Total FLOPs: 2 × M × K × N
    """
    return 2 * M * K * N

# Example: GPT-2 feedforward forward pass
batch_size = 32
seq_len = 512
d_model = 768
d_ff = 3072

# First linear: (batch × seq, d_model) @ (d_model, d_ff)
flops_1 = count_matmul_flops(batch_size * seq_len, d_model, d_ff)
# = 2 × 16384 × 768 × 3072 = 77,309,411,328 FLOPs

# Second linear: (batch × seq, d_ff) @ (d_ff, d_model)
flops_2 = count_matmul_flops(batch_size * seq_len, d_ff, d_model)
# = 2 × 16384 × 3072 × 768 = 77,309,411,328 FLOPs

total_flops = flops_1 + flops_2  # ~154 GFLOPs for one feedforward layer

# Hardware context:
# NVIDIA A100: 312 TFLOPS (FP16) → theoretical time = 154 / 312000 = 0.5 ms
# Actual time will be higher due to memory bandwidth and kernel overhead
```

**FLOP Formulas Reference**:
```python
# Linear layer
flops = 2 × batch_size × seq_len × input_features × output_features

# Conv2d
flops = 2 × batch × output_h × output_w × kernel_h × kernel_w × in_ch × out_ch

# Multi-head attention (simplified)
# QKV projections: 3 × linear projections
# Attention scores: batch × heads × seq × seq × d_k
# Attention weighting: batch × heads × seq × seq × d_k
# Output projection: 1 × linear projection
flops = (4 × batch × seq × d_model × d_model) +
        (4 × batch × heads × seq × seq × d_k)
```

### Memory Profiling: Understanding Allocation Patterns

Memory profiling reveals where RAM goes during training:

```python
class MemoryProfiler:
    """Track memory allocations and identify usage patterns."""

    def __init__(self):
        self.snapshots = []

    def snapshot(self, label: str):
        """Take memory snapshot at execution point."""
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()

        self.snapshots.append({
            'label': label,
            'rss': mem_info.rss / 1024**2,  # Resident Set Size (MB)
            'timestamp': time.time()
        })

    def report(self):
        """Generate memory usage report."""
        print("Memory Timeline:")
        for i, snap in enumerate(self.snapshots):
            delta = ""
            if i > 0:
                delta_val = snap['rss'] - self.snapshots[i-1]['rss']
                delta = f" ({delta_val:+.2f} MB)"
            print(f"  {snap['label']:30s}: {snap['rss']:8.2f} MB{delta}")

# Example: Profile transformer forward pass
mem = MemoryProfiler()
mem.snapshot("baseline")

# Forward pass
output = model.forward(input_tensor)
mem.snapshot("after_forward")

# Backward pass
loss = criterion(output, target)
loss.backward()
mem.snapshot("after_backward")

# Update weights
optimizer.step()
mem.snapshot("after_optimizer")

mem.report()

# Output interpretation:
# baseline                     : 1024.00 MB
# after_forward               : 1124.00 MB (+100.00 MB)  ← Activation memory
# after_backward              : 1624.00 MB (+500.00 MB)  ← Gradient memory
# after_optimizer             : 2124.00 MB (+500.00 MB)  ← Adam state (momentum + velocity)
#
# Total training memory = 2.1× forward memory (for Adam optimizer)
```

**Memory Components Breakdown**:
```
Training Memory = Parameters + Activations + Gradients + Optimizer State

Example for GPT-2 Small (124M parameters):
Parameters:    496 MB  (124M × 4 bytes)
Activations:   200 MB  (depends on batch size and sequence length)
Gradients:     496 MB  (same as parameters)
Adam state:    992 MB  (momentum + velocity = 2× parameters)
─────────────────────────────────────
Total:        2184 MB  (4.4× parameter memory!)

Optimization strategies by component:
- Parameters: Quantization (reduce precision)
- Activations: Gradient checkpointing (recompute instead of store)
- Gradients: Mixed precision (FP16 gradients)
- Optimizer: SGD instead of Adam (0× vs 2× parameter memory)
```

### Latency Measurement: Statistical Timing Methodology

Accurate latency measurement requires handling variance:

```python
def measure_latency_correctly(model, input_tensor):
    """Production-quality latency measurement."""

    # Step 1: Warmup runs (stabilize system state)
    # - JIT compilation happens on first runs
    # - CPU/GPU caches warm up
    # - Operating system scheduling stabilizes
    warmup_runs = 10
    for _ in range(warmup_runs):
        _ = model.forward(input_tensor)

    # Step 2: Multiple measurements (statistical significance)
    times = []
    measurement_runs = 100

    for _ in range(measurement_runs):
        start = time.perf_counter()  # Nanosecond precision
        _ = model.forward(input_tensor)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to milliseconds

    # Step 3: Statistical analysis
    times = np.array(times)

    results = {
        'mean': np.mean(times),
        'median': np.median(times),      # Robust to outliers
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'p50': np.percentile(times, 50),  # Median
        'p95': np.percentile(times, 95),  # 95th percentile
        'p99': np.percentile(times, 99)   # 99th percentile (tail latency)
    }

    return results

# Example output:
# {
#   'mean': 5.234,
#   'median': 5.180,    ← Use this for reporting (robust)
#   'std': 0.456,
#   'min': 4.890,
#   'max': 8.120,       ← Outlier (OS scheduling event)
#   'p50': 5.180,
#   'p95': 5.890,
#   'p99': 6.340        ← Important for user-facing latency
# }

# Why median, not mean?
# Mean is sensitive to outliers (8.120 ms max skews average)
# Median represents typical performance
# For user-facing systems, report p95 or p99 (worst-case experience)
```

**Measurement Pitfalls and Solutions**:
```python
# ❌ WRONG: Single measurement
start = time.time()  # Low precision
output = model(input)
latency = time.time() - start  # Affected by system noise

# ✅ CORRECT: Statistical measurement
profiler = Profiler()
latency = profiler.measure_latency(model, input, warmup=10, iterations=100)
# Returns median of 100 measurements after 10 warmup runs

# ❌ WRONG: Measuring cold start
latency = time_function_once(model.forward, input)  # Includes JIT compilation

# ✅ CORRECT: Warmup runs
for _ in range(10):
    model.forward(input)  # Discard these results
latency = measure_with_statistics(model.forward, input)  # Now measure

# ❌ WRONG: Using mean with outliers
times = [5.1, 5.2, 5.0, 5.3, 50.0]  # 50ms outlier from OS scheduling
mean = np.mean(times)  # = 14.12 ms (misleading!)

# ✅ CORRECT: Using median
median = np.median(times)  # = 5.2 ms (representative)
```

## Getting Started

### Prerequisites

Ensure you understand the foundations from previous modules:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules (all modules 1-13)
tito test tensor
tito test activations
tito test transformer
```

**Why these prerequisites**: You'll profile models built in Modules 1-13. Understanding the implementations helps you interpret profiling results (e.g., why attention is memory-bound).

### Development Workflow

1. **Open the development file**: `modules/14_profiling/profiling_dev.ipynb` or `.py`
2. **Implement parameter counting**: Walk model structure, sum parameter elements
3. **Build FLOP counter**: Calculate operations based on layer types and dimensions
4. **Create memory profiler**: Use tracemalloc to track allocations during forward/backward
5. **Add timing profiler**: Implement warmup runs, multiple measurements, statistical analysis
6. **Implement advanced profiling**: Build `profile_forward_pass()` and `profile_backward_pass()` combining all metrics
7. **Export and verify**: `tito module complete 14 && tito test profiling`

**Development tips**:
```python
# Test parameter counting manually first
layer = Linear(128, 64)
expected_params = (128 * 64) + 64  # weight + bias = 8256
actual_params = profiler.count_parameters(layer)
assert actual_params == expected_params

# Verify FLOP calculations with small examples
flops = profiler.count_flops(layer, (1, 128))
expected_flops = 2 * 128 * 64  # matmul FLOPs = 16384
assert flops == expected_flops

# Check memory profiler returns expected keys
mem = profiler.measure_memory(layer, (32, 128))
assert 'parameter_memory_mb' in mem
assert 'activation_memory_mb' in mem
assert 'peak_memory_mb' in mem

# Validate latency measurement stability
latencies = [profiler.measure_latency(layer, input_tensor) for _ in range(3)]
std_dev = np.std(latencies)
assert std_dev < np.mean(latencies) * 0.2  # Coefficient of variation < 20%
```

## Testing

### Comprehensive Test Suite

Run the full test suite to verify profiling functionality:

```bash
# TinyTorch CLI (recommended)
tito test profiling

# Direct pytest execution
python -m pytest tests/ -k profiling -v
```

### Test Coverage Areas

- ✅ **Parameter counting accuracy**: Verifies correct counts for Linear, Conv2d, models with/without parameters
- ✅ **FLOP calculation correctness**: Validates formulas for different layer types (Linear, Conv2d, attention)
- ✅ **Memory measurement reliability**: Checks tracemalloc integration, memory component tracking
- ✅ **Latency measurement consistency**: Tests statistical timing with warmup runs and multiple iterations
- ✅ **Advanced profiling completeness**: Validates forward/backward profiling returns all required metrics

### Inline Testing & Validation

The module includes comprehensive unit tests:

```python
# Parameter counting validation
🔬 Unit Test: Parameter Counting...
✅ Simple model: 55 parameters (10×5 weight + 5 bias)
✅ No parameter model: 0 parameters
✅ Direct tensor: 0 parameters
✅ Parameter counting works correctly!

# FLOP counting validation
🔬 Unit Test: FLOP Counting...
✅ Tensor operation: 32 FLOPs
✅ Linear layer: 16384 FLOPs (128 × 64 × 2)
✅ Batch independence: 16384 FLOPs (same for batch 1 and 32)
✅ FLOP counting works correctly!

# Memory measurement validation
🔬 Unit Test: Memory Measurement...
✅ Basic measurement: 0.153 MB peak
✅ Scaling: Small 0.002 MB → Large 0.020 MB
✅ Efficiency: 0.524 (0-1 range)
✅ Memory measurement works correctly!

# Latency measurement validation
🔬 Unit Test: Latency Measurement...
✅ Basic latency: 0.008 ms
✅ Consistency: 0.010 ± 0.002 ms
✅ Scaling: Small 0.006 ms, Large 0.012 ms
✅ Latency measurement works correctly!
```

### Manual Testing Examples

```python
from profiling_dev import Profiler, quick_profile
from tinytorch.nn.layers import Linear
from tinytorch.core.tensor import Tensor

# Example 1: Profile a simple layer
layer = Linear(256, 128)
input_tensor = Tensor(np.random.randn(32, 256))

profiler = Profiler()
profile = profiler.profile_forward_pass(layer, input_tensor)

print(f"Parameters: {profile['parameters']:,}")
print(f"FLOPs: {profile['flops']:,}")
print(f"Latency: {profile['latency_ms']:.2f} ms")
print(f"Memory: {profile['peak_memory_mb']:.2f} MB")
print(f"Bottleneck: {profile['bottleneck']}")
# Output:
# Parameters: 32,896
# FLOPs: 2,097,152
# Latency: 0.15 ms
# Memory: 2.10 MB
# Bottleneck: memory

# Example 2: Compare architectures
mlp = Linear(512, 512)
attention = MultiHeadAttention(d_model=512, num_heads=8)

mlp_profile = profiler.profile_forward_pass(mlp, mlp_input)
attention_profile = profiler.profile_forward_pass(attention, attention_input)

print(f"MLP GFLOP/s: {mlp_profile['gflops_per_second']:.2f}")
print(f"Attention GFLOP/s: {attention_profile['gflops_per_second']:.2f}")
# Output reveals which operation is more efficient

# Example 3: Analyze training memory
training_profile = profiler.profile_backward_pass(model, input_tensor)

print(f"Forward memory: {training_profile['forward_memory_mb']:.1f} MB")
print(f"Gradient memory: {training_profile['gradient_memory_mb']:.1f} MB")
print(f"Total training memory: {training_profile['total_memory_mb']:.1f} MB")

for opt_name, opt_memory in training_profile['optimizer_memory_estimates'].items():
    total_with_opt = training_profile['total_memory_mb'] + opt_memory
    print(f"{opt_name.upper()}: {total_with_opt:.1f} MB total")
# Output:
# Forward memory: 2.1 MB
# Gradient memory: 2.0 MB
# Total training memory: 4.1 MB
# SGD: 4.1 MB total
# ADAM: 8.1 MB total (2× extra for momentum + velocity)
```

## Systems Thinking Questions

### Real-World Applications

- **Google TPU Optimization**: Profile every kernel to achieve 40-50% MFU (Model FLOPs Utilization). Google improved T5 training from 35% to 48% MFU through profiling-guided optimization, saving millions in compute costs at scale across thousands of TPUs. How would you use profiling to identify and fix utilization bottlenecks?

- **OpenAI GPT Training**: Profile forward and backward passes separately to measure memory usage across parameters, activations, gradients, and optimizer state. OpenAI identified activation memory as the bottleneck and implemented gradient checkpointing, reducing memory by 10× with only 20% compute overhead while achieving 50%+ MFU. What trade-offs exist between recomputation time and storage memory?

- **Meta PyTorch Inference**: Profile operator-by-operator timelines to measure kernel launch overhead and identify operator fusion opportunities. Meta reduced inference latency by 2-3× through operator fusion and optimized p99 latency for billions of daily requests serving Facebook/Instagram recommendations. Why optimize for latency percentiles rather than average?

- **NVIDIA cuDNN Development**: Use Nsight profiler to analyze warp occupancy, register pressure, and memory bandwidth utilization to achieve 90%+ of theoretical peak performance. NVIDIA's profiling data guides both kernel optimization and next-generation hardware design (H100 architecture). How do you distinguish compute-bound from memory-bound kernels?

### Profiling Foundations

- **Amdahl's Law and ROI**: If attention takes 70% of time and you achieve 2× speedup on attention only, overall speedup is just 1.53× (not 2×) because unoptimized portions limit gains. Why does this mean optimization is iterative—requiring re-profiling after each change to identify new bottlenecks?

- **Memory Bandwidth Bottlenecks**: An elementwise ReLU operation on 1B elements achieves only 112 GFLOPs/s despite 100 TFLOPS peak compute (0.11% utilization) because it's memory-bound (8.89 ms to move 8 GB data vs 0.01 ms to compute). What optimization strategies help memory-bound operations vs compute-bound operations?

- **Statistical Timing Methodology**: Single measurements include system noise (OS scheduling, thermal throttling, cache effects). Proper profiling uses warmup runs (JIT compilation, cache warming), multiple measurements (100+ iterations), and reports median (robust to outliers) plus p95/p99 percentiles (tail latency). Why does mean latency hide outliers that affect user experience?

- **Profiling Overhead Trade-offs**: Instrumentation profiling (15% overhead) provides precise per-operation timing but distorts fast operations, while sampling profiling (2% overhead) enables always-on production monitoring but may miss operations <1 ms. When should you choose instrumentation vs sampling profilers?

### Performance Characteristics

- **Batch Size Scaling**: Throughput doesn't scale linearly with batch size due to fixed overhead (kernel launch amortizes), memory bandwidth saturation (transfers dominate at large batches), and memory constraints (OOM limits maximum batch size). For a system showing 200→667→914→985 samples/s at batch sizes 1→8→32→64, what's the optimal batch size for throughput vs efficiency vs latency?

- **GPU vs CPU Crossover**: Small matrices (128×128) run faster on CPU despite GPU's 1000× more cores because GPU overhead (1 ms kernel launch) dominates compute time. Large matrices (4096×4096) achieve 267× GPU speedup because overhead amortizes and parallelism saturates GPU cores. What's the crossover point and why does PyTorch automatically dispatch based on operation size?

- **Parameter vs Activation Memory**: Training memory = Parameters + Activations + Gradients + Optimizer State. For GPT-2 Small (124M params = 496 MB), total training memory is 2.18 GB (4.4× parameter memory) due to activations (200 MB), gradients (496 MB), and Adam state (992 MB = 2× parameters). Which component should you optimize for different memory constraints?

- **FLOPs vs Latency**: Theoretical FLOPs predict compute cost hardware-independently, but actual latency depends on memory bandwidth and kernel efficiency. A GPT-2 feedforward layer requires 154 GFLOPs, suggesting 0.5 ms on A100 (312 TFLOPS), but actual time is higher due to memory overhead. Why is profiling real hardware essential despite theoretical calculations?

## Ready to Build?

You're about to implement the profiling tools that enable all subsequent optimization work. These techniques transform research models into production systems by revealing exactly where time and memory go.

**What you'll achieve**:
- Understand where compute time actually goes in ML models (measure, don't guess)
- Distinguish memory-bound from compute-bound operations (guides optimization strategy)
- Make data-driven optimization decisions using Amdahl's Law (maximize ROI on engineering time)
- Build the measurement foundation for Modules 15-20 (optimization techniques)

**The profiling mindset**:
> "Measure twice, optimize once. Profile before every optimization decision. Without measurement, you're flying blind."
> — Every production ML engineer

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/14_profiling/profiling_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/14_profiling/profiling_dev.ipynb
:class-header: bg-light

Use Google Colab for cloud compute power and easy sharing.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/14_profiling/profiling_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.
```

```bash
cd modules/14_profiling
tito module start 14
python profiling_dev.py  # Inline tests as you build
```

---

<div class="prev-next-area">
<a class="left-prev" href="../13_transformers/ABOUT.html" title="previous page">← Module 13: Transformers</a>
<a class="right-next" href="../15_quantization/ABOUT.html" title="next page">Module 15: Quantization →</a>
</div>
