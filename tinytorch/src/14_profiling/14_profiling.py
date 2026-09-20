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
# Module 14: Profiling - Measuring What Matters in ML Systems

Welcome to Module 14! You'll build professional profiling tools to measure model performance, identify hardware bottlenecks, and uncover optimization opportunities.

## 🔗 Prerequisites & Progress

**You've Built**: Complete ML stack from tensors to autoregressive transformers (`Tensor`, `Linear`, `Conv2d`, `MultiHeadAttention`, `TransformerBlock`, `GPT`).
**You'll Build**: Comprehensive profiling system for parameters, FLOPs, memory allocations, and latency (a `Profiler` class whose methods are `count_parameters`, `count_flops`, `measure_memory`, and `measure_latency`, plus the free functions `arithmetic_intensity`, `quick_profile`, and `analyze_weight_distribution`).
**You'll Enable**: Data-driven optimization decisions across quantization (`15_quantization`), compression (`16_compression`), acceleration (`17_acceleration`), memory caching (`18_memoization`), and the benchmark harness that consumes these measurements directly (`19_benchmarking`).

<div align="center">
  <img src="profiling_blueprint.svg" alt="TinyTorch Architecture Blueprint: Module 14 Profiling" width="380px">
</div>

### Architectural Roadmap

| Tier | Subsystem | Primitives & Capabilities | Status |
| :--- | :--- | :--- | :--- |
| **Modules 01–08** | Foundation Tier | `Tensor`, `Function`, `Linear`, `GELU`, `SGD`, `Adam`, `Trainer` | Completed |
| **Modules 09–13** | Architecture Tier | `Conv2d`, `BPETokenizer`, `EmbeddingLayer`, `MultiHeadAttention`, `GPT` | Completed |
| **Module 14** | **Profiling & Diagnostics** | `Profiler`, `count_flops`, `measure_memory`, `measure_latency` | **Active Subsystem** |
| **Modules 15–20** | Optimization & Capstone | `QuantizedLinear`, `Compressor`, `im2col_conv2d`, `KVCache`, `BenchmarkSuite` | Downstream Consumers |

## 🎯 Learning Objectives

By the end of this module, you will:

1. **Implement a Unified Profiler Engine**: Construct a multi-pass measurement harness tracking model weights, runtime activations, execution latency, and peak memory allocations.
2. **Derive Precise FLOP and Parameter Formulations**: Distinguish persistent memory footprint ($W \times 4\text{ B}$) from computational work ($2 \cdot M \cdot N \cdot K$), uncovering why convolutions and attention diverge from linear layers.
3. **Isolate Hardware Bottlenecks with the Roofline Model**: Calculate arithmetic intensity ($I = \text{FLOPs} / \text{Bytes}$) to classify workloads into memory-bandwidth bound versus compute-bound regimes.
4. **Budget Multi-Stage Training Lifecycles**: Estimate the roughly $4\times$ memory expansion across forward activations, backward gradients, and first/second moment optimizer state from a parameter count alone, before a training run exists to measure.

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/14_profiling/profiling.ipynb`
**Building Side:** Code exports to `tinytorch.perf.profiling`

```python
# Final package structure:
from tinytorch.perf.profiling import Profiler, quick_profile, analyze_weight_distribution
```

Every module up to this one exported into `tinytorch.core`, because every module up to
this one added a piece of the framework. This one exports to `tinytorch.perf` instead,
and the change of namespace is the point. Nothing here is a framework component. A
profiler does not participate in a forward pass, holds no parameters, and appears in no
computational graph. It is an instrument you point at the framework from outside, which
is why it lives beside the framework rather than inside it, and why Modules 15 through
20 can import it without the framework importing anything of theirs.

This is also where the course turns. Modules 01 to 13 built the thing. Modules 14 to 20
measure it and then change it on the evidence, and the measuring has to come first.

<div align="center">
  <img src="prof_margin_source.svg" alt="Source Code Mapping: Module 14 Profiling" width="260px">
</div>

## 📋 Module Dependencies

| Dependency | Origin | Imported Symbols | Architectural Purpose in Profiling |
| :--- | :--- | :--- | :--- |
| `tinytorch.core.tensor` | Module 01 | `Tensor` | Multi-dimensional array container and gradient storage tracker |
| `tinytorch.core.layers` | Module 03 | `Linear` | Dense projection layers evaluated for FLOPs and parameter memory |
| `tinytorch.core.spatial` | Module 09 | `Conv2d` | Sliding window spatial convolutions with high compute reuse |
| `numpy` | External | `np`, `default_rng` | High-performance array generation and statistical reductions |
| `tracemalloc` | Standard Lib | `tracemalloc` | Python memory allocation snapshotting and peak heap tracking |
| `time` | Standard Lib | `perf_counter` | High-precision nanosecond monotonic timestamping for latency benchmarking |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp perf.profiling
#| export

import time
import tracemalloc
from typing import Any, Dict, Tuple

import numpy as np
rng = np.random.default_rng(7)

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.spatial import Conv2d

# Constants for memory and performance measurement
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
r"""
## 💡 Introduction: Why Profiling Matters in ML Systems

Imagine you're an engineer investigating an ML systems regression. Your model runs unacceptably slowly, exhausts device memory, or inflates cloud inference bills. Without profiling, you are flying blind, guessing whether to optimize matrix multiplication kernels, reduce activation precision, or shard model weights. With profiling, you have empirical ground truth.

<div align="center">
  <img src="profiling_engineering_workflow.svg" alt="Systems Profiling and Optimization Workflow" width="680px">
</div>

### The Profiling and Optimization Engineering Loop

| Phase | Primary Objective | Concrete Metric / Telemetry | Downstream Systems Action |
| :--- | :--- | :--- | :--- |
| **1. Measure** | Establish empirical telemetry | Latency ($T_{\text{median}}$), Peak VRAM ($M_{\text{peak}}$), FLOPs | Detect resource bottlenecks |
| **2. Analyze** | Compute arithmetic intensity | $I = \text{FLOPs} / \text{Bytes}$, Roofline position | Classify compute vs memory-bound regimes |
| **3. Optimize** | Eliminate hardware constraint | INT8 quantization, KV cache, operator fusion | Restructure memory/compute datapath |
| **4. Validate** | Verify speedup and fidelity | Speedup ratio $S = T_{\text{base}} / T_{\text{opt}}$, numerical parity | Confirm production readiness |

Without profiling, engineers routinely waste months optimizing code that accounts for less than 2% of runtime. As Amdahl's Law dictates, system speedup is strictly bounded by the fraction of execution time an optimization addresses:

$$S_{\text{overall}} = \frac{1}{(1 - f) + \frac{f}{s}}$$

Where $f$ is the execution fraction of the targeted component and $s$ is its isolated speedup. Profiling identifies the high-$f$ bottlenecks before a single line of optimization code is written.
"""

# %% [markdown]
r"""
## 📐 Foundations: Performance Measurement Principles

Before building the profiler class, let us establish the five pillars of machine learning systems telemetry. Four of them are quantities you measure, namely parameters, compute (FLOPs), memory residency, and execution latency. The fifth, arithmetic intensity, is the ratio that turns two of the others into a verdict about which hardware ceiling you are under, and every optimization module after this one argues from it.

### 1. Parameter Counting: The Static Memory Footprint

Parameters determine persistent storage requirements, weight transfer latency from high-bandwidth memory (HBM/DRAM) to on-chip SRAM, and minimum GPU VRAM capacity.

$$\text{Memory}_{\text{FP32}} = P \times 4\text{ bytes} = \frac{P \times 4}{1024^2}\text{ MB}$$

| Layer Type | Learnable Weights | Learnable Biases | Total Parameters ($P$) | Memory at FP32 ($4\text{ B/elem}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear** | $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ | $b \in \mathbb{R}^{d_{\text{out}}}$ | $d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}}$ | $(d_{\text{in}} + 1) \cdot d_{\text{out}} \times 4\text{ B}$ |
| **Conv2d** | $K \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w}$ | $b \in \mathbb{R}^{C_{\text{out}}}$ | $C_{\text{out}} \cdot (C_{\text{in}} \cdot k_h \cdot k_w + 1)$ | $C_{\text{out}} \cdot (C_{\text{in}} k_h k_w + 1) \times 4\text{ B}$ |
| **LayerNorm** | $\gamma \in \mathbb{R}^{d_{\text{embed}}}$ | $\beta \in \mathbb{R}^{d_{\text{embed}}}$ | $2 \cdot d_{\text{embed}}$ | $8 \cdot d_{\text{embed}}\text{ B}$ |
| **Embedding** | $E \in \mathbb{R}^{V \times d_{\text{embed}}}$ | None | $V \cdot d_{\text{embed}}$ | $4 \cdot V \cdot d_{\text{embed}}\text{ B}$ |

### 2. FLOP Counting: Theoretical Arithmetic Work

A floating-point operation (FLOP) measures arithmetic work independent of hardware implementation. In deep learning primitives, each multiply-accumulate (MAC) counts as **2 FLOPs** (one multiplication and one addition):

$$\text{FLOP}_{\text{GEMM}}(M, K, N) = 2 \cdot M \cdot K \cdot N$$

$$\text{FLOP}_{\text{Conv2d}} = 2 \cdot B \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot (C_{\text{in}} \cdot k_h \cdot k_w) \cdot C_{\text{out}}$$

| Operation Type | Input / Weight Shapes | FLOP Count (Per Sample) | Arithmetic Intensity ($I = \text{FLOPs} / \text{Bytes}$) | Hardware Regime |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Projection** | $(d_{\text{in}}) \times (d_{\text{in}}, d_{\text{out}})$ | $2 \cdot d_{\text{in}} \cdot d_{\text{out}}$ | $0.5\text{ FLOP/B}$ (Batch=1) | Memory-bandwidth bound |
| **Batched Linear** | $(B, d_{\text{in}}) \times (d_{\text{in}}, d_{\text{out}})$ | $2 \cdot B \cdot d_{\text{in}} \cdot d_{\text{out}}$ | $0.5 \cdot B\text{ FLOP/B}$ (Batch=B) | Compute-bound at high $B$ |
| **Conv2d** | $(C_{\text{in}}, H, W) \ast (C_{\text{out}}, C_{\text{in}}, k_h, k_w)$ | $2 \cdot H_{\text{out}} W_{\text{out}} C_{\text{in}} C_{\text{out}} k_h k_w$ | $\approx 0.5 \cdot H_{\text{out}} W_{\text{out}}\text{ FLOP/B}$ | Compute-bound (kernel reuse) |
| **Pointwise (GELU/Add)**| $(B, S, D)$ | $1 \text{ to } 8 \text{ FLOPs/elem}$ | $0.125 \text{ to } 1\text{ FLOP/B}$ | Strictly memory-bound |

Every entry in that column is one division, and it is worth doing once by hand.
A Linear layer at $B=1$ performs $2 d_{\text{in}} d_{\text{out}}$ FLOPs while moving
the whole weight matrix, $4 d_{\text{in}} d_{\text{out}}$ bytes at FP32, so
$I = 2/4 = 0.5\text{ FLOP/B}$ no matter how wide the layer is. Batching to $B$ samples
multiplies the FLOPs by $B$ and leaves the weight traffic unchanged, giving $I = 0.5B$.
At $B=32$ that is $16\text{ FLOP/B}$, and Question 1 walks the same arithmetic.
Convolution reuses each weight across the output map, so $I$ grows with the output
area. A pointwise kernel reads one element and writes one, $8\text{ B}$ of traffic for
$1$ to $8$ FLOPs, which is the lowest intensity in the table and the reason fusion
exists.

### 3. Arithmetic Intensity: The Roofline and Its Ridge Point

Intensity only becomes a verdict when you compare it against a machine. That
comparison is the roofline, and it is the single most reused idea in the rest of this
course, so it belongs here rather than at the end.

<div align="center">
  <img src="roofline_model_hardware_limits.svg" alt="Illustrative Hardware Limits and the Roofline Model" width="680px">
</div>

The attainable floating-point performance $P$ (in $\text{GFLOP/s}$) on any physical processor is strictly bounded by two fundamental hardware ceilings:

$$P \le \min\left(P_{\text{peak}},\, I \times \text{BW}_{\text{mem}}\right)$$

Where:
- $P_{\text{peak}}$ is the peak arithmetic compute throughput of the device (ALUs / Tensor Cores).
- $\text{BW}_{\text{mem}}$ is the sustained memory bandwidth between device DRAM/HBM and on-chip caches/SRAM.
- $I = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$ is the **arithmetic intensity** of the operation, the quantity the column above computes.

The hardware **ridge point** is defined as the operational intensity where the memory ceiling intersects peak compute:

$$I_{\text{ridge}} = \frac{P_{\text{peak}}}{\text{BW}_{\text{mem}}}$$

| Operational Regime | Condition | Limiting Hardware Subsystem | Systems Remedy |
| :--- | :--- | :--- | :--- |
| **Memory-Bandwidth Bound** | $I < I_{\text{ridge}}$ | DRAM / HBM transfer bus | Weight quantization, operator fusion, KV caching |
| **Compute Bound** | $I \ge I_{\text{ridge}}$ | ALU / Tensor Core matrix engines | Algorithmic transforms, FP16 Tensor Cores |

An A100 at $19.5\text{ TFLOP/s}$ FP32 over $2,039\text{ GB/s}$ of HBM2e has
$I_{\text{ridge}} = 9.6\text{ FLOP/B}$. Our batched Linear at $B=32$ sits at
$16\text{ FLOP/B}$, above that ridge, which is why batching is the first thing anyone
reaches for. The same layer at $B=1$ sits at $0.5$, a factor of 19 below the ridge,
and no kernel rewrite can help it, because the weights have to travel either way.

### 4. Memory Profiling: The Training Memory Lifecycle

Deep learning training memory divides into four distinct pools that undergo dynamic lifecycle transitions during forward and backward passes.

<div align="center">
  <img src="training_memory_lifecycle.svg" alt="Deep Learning Training Memory Lifecycle" width="680px">
</div>

| Memory Pool | Lifecycle Invariant | Scaling Dimensions | Budget Formula (FP32) | Example (125M GPT) |
| :--- | :--- | :--- | :--- | :--- |
| **Parameters ($W$)** | Persistent in device VRAM | Model dimension $P$ | $P \times 4\text{ B}$ | $500\text{ MB}$ ($1.0\times$) |
| **Activations ($A$)** | Cached in forward; freed in backward | Batch $\times$ Sequence $\times$ Layers | $B \cdot S \cdot L \cdot d_{\text{embed}} \cdot c_{\text{act}} \times 4\text{ B}$ | $\approx 200\text{ MB}$ ($0.4\times$) |
| **Gradients ($\nabla_W L$)** | Allocated during backward pass | 1:1 match with parameters $P$ | $P \times 4\text{ B}$ | $500\text{ MB}$ ($1.0\times$) |
| **Optimizer State (Adam)**| Persistent in device VRAM | First moment $m$ + second moment $v$ | $2 \times P \times 4\text{ B}$ | $1,000\text{ MB}$ ($2.0\times$) |
| **Total Training Budget** | Peak concurrent residency | $16\text{ B/param}$ plus activations, so never below $4\times$ the weight bytes | $\underbrace{4P}_{W} + A + \underbrace{4P}_{\nabla W} + \underbrace{8P}_{m, v} = 16P + A$ | **$2,200\text{ MB}$ ($4.4\times$)** |

Throughout this table $P$ is the parameter **count**, never a byte figure, so every
term carries its own $4\text{ B}$ factor. Four pools at FP32 come to $16\text{ B}$
per parameter (weights, gradients, and Adam's two moments), and activations are the
one term that batch and sequence length control rather than the model.

### 5. Latency Measurement: Statistical Rigor

Measuring execution latency on modern multi-core CPUs and GPUs is subject to background operating system interrupts, CPU frequency governors (thermal throttling), dynamic cache warmups, and garbage collection pauses. Professional profiling demands strict statistical isolation:

| Protocol Stage | Iteration Budget | System State | Treatment of Output |
| :--- | :--- | :--- | :--- |
| **Warmup Passes** | $3 \text{ to } 10$ iterations | Cold caches, BLAS thread-pool spin-up, first-touch page faults | Excluded from measurements to eliminate cold-start noise |
| **Timed Benchmark** | $10 \text{ to } 100+$ iterations | Steady-state thermal and memory cache profile | Report the median ($Q_2$) rather than the mean |
| **Memory Cleanup** | Between runs | Explicit garbage collection (`gc.collect()`) | Prevents heap fragmentation contamination |

The first two rows describe what `measure_latency` does below. The third is what a
production harness adds and ours does not, along with the interquartile range that
would report the spread beside the median. Both are noted here so you know the shape of
the gap, and Module 19 will close it.

Warmup also hides something. The timed loop below re-runs the same input tensor, so
after the first pass it is resident in cache and every number the module reports is a
best case. Real serving traffic arrives cold, one request at a time, from memory the
processor has not touched.
"""

# %% [markdown]
r"""
## 🏗️ Implementation: Building the Profiler Class

Let us now implement the complete `Profiler` class. We structure the profiling harness into four core telemetry engines and two multi-pass synthesis pipelines.

<div align="center">
  <img src="profiler_architecture_pipeline.svg" alt="Profiler Architecture and Diagnostic Pipeline" width="680px">
</div>

### Profiler Architectural Specification

| Component Method | Telemetry Category | Measurement Mechanism | Systems Output |
| :--- | :--- | :--- | :--- |
| `count_parameters()` | Static Footprint | Recursive weight and bias buffer enumeration | Learnable parameter count and persistent VRAM budget |
| `count_flops()` | Arithmetic Complexity | Theoretical multiply-accumulate formulations | Hardware-agnostic compute work ($2 \cdot M \cdot K \cdot N$) |
| `measure_memory()` | Memory Dynamics | Python `tracemalloc` peak heap allocation snapshots | Activation memory, peak memory, and allocation efficiency |
| `measure_latency()` | Temporal Latency | High-precision monotonic timestamps (`perf_counter`) | Warmup-filtered median latency in milliseconds |
| `profile_forward_pass()`| Runtime Synthesis | Fused latency, FLOP, and activation analysis | Throughput ($\text{samp/s}$), GFLOP/s, bandwidth, bottleneck |
| `profile_backward_pass()`| Training Synthesis | 2x FLOP scaling, gradient tracking, optimizer estimation | Total training latency, gradient memory, and Adam states |
"""

# %% [markdown]
r"""
### Layer Parameters: The Atom of Model Size

Parameter count is the first number anyone quotes about a model, and it is the
one every memory estimate starts from. A layer's parameters are whatever arrays
it learns: the weight matrix, plus a bias vector when it has one. Nothing else
counts, because activations are recomputed each forward pass and belong to a different
budget.

The reason this is a separate function rather than a line inside the traversal
is that "does this thing have parameters" is a question about an object, not
about a model. Keeping it small lets `Profiler.count_parameters`, the model-wide
traversal later in this module, stay a plain traversal.
"""

# %% nbgrader={"grade": false, "grade_id": "count-layer-parameters", "solution": true}
#| export
def _count_layer_parameters(layer) -> int:
    """
    Count the learnable parameters in a single layer.

    ```
    Parameters = weight.size + bias.size (when a bias exists)
    ```

    TODO: Sum the sizes of the layer's learnable arrays.

    APPROACH:
    1. Start a running total at zero
    2. If the layer has a weight, add weight.data.size
    3. If it also has a non-None bias, add bias.data.size
    4. Return the total

    EXAMPLE:
    >>> layer = MockLinear(128, 64)   # weight only, no bias
    >>> _count_layer_parameters(layer)
    8192

    HINTS:
    - Use hasattr(layer, 'weight'), since not every layer has parameters
    - A bias attribute can exist and still be None; check both
    - .data.size gives the element count, which is what we want here

    Args:
        layer: Any layer object, with or without parameters

    Returns:
        int: Number of learnable parameters (0 for parameterless layers)
    """
    ### BEGIN SOLUTION role="scaffold"
    params = 0
    if hasattr(layer, 'weight') and layer.weight is not None:
        params += layer.weight.data.size
    if hasattr(layer, 'bias') and layer.bias is not None:
        params += layer.bias.data.size
    return params
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _count_layer_parameters

This test validates the helper that counts parameters from a single layer's weight and bias.

**What we're testing**: Single-layer parameter counting from weight/bias attributes
**Why it matters**: This is the atomic unit of parameter counting that count_parameters delegates to
**Expected**: Correct weight + bias element counts
"""

# %% nbgrader={"grade": true, "grade_id": "test-count-layer-parameters", "locked": true, "points": 3}
def test_unit_count_layer_parameters():
    """🧪 Test _count_layer_parameters helper."""
    print("🧪 Unit Test: _count_layer_parameters...")

    # Test 1: Layer with weight and bias
    class LayerWithBias:
        def __init__(self):
            self.weight = Tensor(rng.standard_normal((10, 5)))
            self.bias = Tensor(rng.standard_normal(5))

    layer = LayerWithBias()
    count = _count_layer_parameters(layer)
    assert count == 55, f"Expected 55 (10*5 + 5), got {count}"
    print(f"✅ Layer with bias: {count} parameters")

    # Test 2: Layer with weight only (no bias)
    class LayerNoBias:
        def __init__(self):
            self.weight = Tensor(rng.standard_normal((8, 4)))

    layer_no_bias = LayerNoBias()
    count = _count_layer_parameters(layer_no_bias)
    assert count == 32, f"Expected 32 (8*4), got {count}"
    print(f"✅ Layer without bias: {count} parameters")

    # Test 3: Object without weight attribute
    class NoWeight:
        pass

    count = _count_layer_parameters(NoWeight())
    assert count == 0, f"Expected 0, got {count}"
    print("✅ No weight attribute: 0 parameters")

    print("✅ _count_layer_parameters works correctly!")

if __name__ == "__main__":
    test_unit_count_layer_parameters()

# %% [markdown]
r"""
### Convolution FLOPs: Where Parameters and Compute Diverge

A convolution costs far more than its parameter count suggests, and the gap is
the whole point. A Linear layer uses each weight once. A convolution slides the
same small kernel across every output position, so each weight is reused
`out_H x out_W` times, and the FLOP count multiplies by that same factor.

That is why a 3x3 conv with a few thousand parameters can dominate a network's
arithmetic while a Linear layer with a million parameters barely registers.
Parameters measure what you store; FLOPs measure what you compute. Convolution
is where the two diverge most sharply, and noticing that divergence is the point
of counting them separately.
"""

# %% nbgrader={"grade": false, "grade_id": "count-conv-flops", "solution": true}
#| export
def _count_conv_flops(model, input_shape: Tuple[int, ...]) -> int:
    """
    Count FLOPs for a Conv2d layer forward pass.

    ```
    Conv2d FLOP Formula:
    FLOPs = out_H x out_W x kernel_H x kernel_W x in_C x out_C x 2
              |       |        |          |         |       |      |
          Output spatial    Kernel spatial     Channel dims   Mul+Add
    ```

    TODO: Compute the forward FLOP count for a Conv2d layer.

    APPROACH:
    1. Bail out with 0 if the layer lacks conv attributes
    2. Normalize kernel_size, which may be an int or a pair
    3. Derive the output spatial dims: (in + 2*pad - kernel) // stride + 1
    4. Multiply output area, kernel area, both channel counts, and 2

    EXAMPLE:
    >>> conv = MockConv2d(in_channels=3, out_channels=16, kernel_size=3)
    >>> _count_conv_flops(conv, (1, 3, 32, 32))
    777600

    HINTS:
    - The output-shape formula is the same one Module 09 derived
    - Use hasattr for stride and padding; they may not be set
    - Every kernel position does one multiply and one add, hence the 2

    Args:
        model: A Conv2d layer with kernel_size, in_channels, out_channels
        input_shape: Input tensor shape (batch, channels, height, width)

    Returns:
        int: FLOP count for one forward pass
    """
    ### BEGIN SOLUTION role="scaffold"
    if not (hasattr(model, 'kernel_size') and hasattr(model, 'in_channels') and hasattr(model, 'out_channels')):
        return 0

    in_channels = model.in_channels
    out_channels = model.out_channels
    kernel_h = model.kernel_size if isinstance(model.kernel_size, int) else model.kernel_size[0]
    kernel_w = model.kernel_size if isinstance(model.kernel_size, int) else model.kernel_size[1]

    input_h, input_w = input_shape[-2], input_shape[-1]
    stride = model.stride if hasattr(model, 'stride') else 1
    stride_h = stride if isinstance(stride, int) else stride[0]
    stride_w = stride if isinstance(stride, int) else stride[1]
    padding = model.padding if hasattr(model, 'padding') else 0
    pad_h = padding if isinstance(padding, int) else padding[0]
    pad_w = padding if isinstance(padding, int) else padding[1]
    output_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
    output_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1

    return output_h * output_w * kernel_h * kernel_w * in_channels * out_channels * 2
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _count_conv_flops

This test validates the helper that computes FLOPs for a Conv2d layer.

**What we're testing**: Conv2d FLOP formula: out_H x out_W x k^2 x in_C x out_C x 2
**Why it matters**: Convolutions are the most compute-intensive operations in vision models
**Expected**: Correct FLOPs accounting for kernel size and channel dimensions
"""

# %% nbgrader={"grade": true, "grade_id": "test-count-conv-flops", "locked": true, "points": 3}
def test_unit_count_conv_flops():
    """🧪 Test _count_conv_flops helper."""
    print("🧪 Unit Test: _count_conv_flops...")

    # Create mock Conv2d layer
    class MockConv:
        def __init__(self, in_c, out_c, k, s=1, p=0):
            self.in_channels = in_c
            self.out_channels = out_c
            self.kernel_size = k
            self.stride = s
            self.padding = p
            self.__class__.__name__ = 'Conv2d'

    # Test 1: Simple 3x3 conv, stride 1
    conv = MockConv(3, 16, 3, 1)
    flops = _count_conv_flops(conv, (1, 3, 32, 32))
    expected = 30 * 30 * 3 * 3 * 3 * 16 * 2
    assert flops == expected, f"Expected {expected}, got {flops}"
    print(f"✅ Conv2d(3, 16, 3): {flops} FLOPs")

    # Test 2: Stride 2 halves output spatial dims
    conv_s2 = MockConv(3, 64, 7, 2)
    flops_s2 = _count_conv_flops(conv_s2, (1, 3, 224, 224))
    out_h = (224 + 2 * 0 - 7) // 2 + 1
    out_w = (224 + 2 * 0 - 7) // 2 + 1
    expected_s2 = out_h * out_w * 7 * 7 * 3 * 64 * 2
    assert flops_s2 == expected_s2, f"Expected {expected_s2}, got {flops_s2}"
    print(f"✅ Conv2d(3, 64, 7, stride=2): {flops_s2} FLOPs")

    # Test 3: Padding size 3 for each side
    conv_p3 = MockConv(3, 10, 3, 1, 3)
    flops_p3 = _count_conv_flops(conv_p3, (1, 3, 28, 28))
    out_h_p3 = (28 + 2 * 3 - 3) // 1 + 1
    out_w_p3 = (28 + 2 * 3 - 3) // 1 + 1
    expected_p3 = out_h_p3 * out_w_p3 * 3 * 3 * 3 * 10 * 2
    assert flops_p3 == expected_p3, f"Expected {expected_p3}, got {flops_p3}"
    print(f"✅ Conv2d(3, 10, 3, stride=1, padding=3): {flops_p3} FLOPs")

    # Test 4: Missing attributes returns 0
    class Incomplete:
        pass

    assert _count_conv_flops(Incomplete(), (1, 3, 32, 32)) == 0
    print("✅ Missing attributes returns 0")

    print("✅ _count_conv_flops works correctly!")

if __name__ == "__main__":
    test_unit_count_conv_flops()

# %% [markdown]
r"""
### Linear FLOPs: The Cost of One Matrix Multiply

A Linear layer is a single matrix multiply, so its arithmetic cost is fixed by
two numbers: how wide the input is and how wide the output is. Producing one
output element means multiplying `in_features` values by their weights and
summing them, which is `in_features` multiplies plus `in_features` adds. Across
all `out_features` outputs that gives `in x out x 2` floating point operations.

Two consequences are worth carrying forward. The count is per-sample, so
batching changes total work but never this number. And because cost grows with
the PRODUCT of the two widths, doubling a layer's width quadruples its FLOPs,
which is why a handful of wide layers usually dominate a network's FLOP budget.
"""

# %% nbgrader={"grade": false, "grade_id": "count-linear-flops", "solution": true}
#| export
def _count_linear_flops(model, input_shape: Tuple[int, ...]) -> int:
    """
    Count FLOPs for a Linear layer forward pass.

    ```
    Linear FLOP Formula:
    FLOPs = in_features x out_features x 2
                 |              |            |
          Input dimension  Output dimension  Multiply + Add
    ```

    TODO: Compute the per-sample forward FLOP count for a Linear layer.

    APPROACH:
    1. Read in_features from the last axis of input_shape
    2. Read out_features from model.weight.shape[1]
    3. Multiply by positions per sample (e.g., sequence length), excluding batch
    4. Return positions * in_features * out_features * 2

    EXAMPLE:
    >>> layer = MockLinear(128, 64)
    >>> _count_linear_flops(layer, (32, 128))
    16384

    HINTS:
    - Use input_shape[-1] so the function works for any batch dimension
    - Guard the missing-weight case with hasattr(model, 'weight')
    - The factor of 2 is the multiply and the add, not the batch

    Args:
        model: A Linear layer with a .weight attribute
        input_shape: Input tensor shape (batch, in_features)

    Returns:
        int: FLOP count for one forward pass (batch-independent)
    """
    ### BEGIN SOLUTION
    in_features = input_shape[-1]
    out_features = model.weight.shape[1] if hasattr(model, 'weight') else 1
    positions = int(np.prod(input_shape[1:-1])) if len(input_shape) > 2 else 1
    return positions * in_features * out_features * 2
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _count_linear_flops

This test validates the helper that computes FLOPs for a single Linear layer.

**What we're testing**: Linear layer FLOP formula: in_features x out_features x 2
**Why it matters**: Linear layers dominate FLOP counts in most ML models
**Expected**: Exact FLOP count matching the formula
"""

# %% nbgrader={"grade": true, "grade_id": "test-count-linear-flops", "locked": true, "points": 3}
def test_unit_count_linear_flops():
    """🧪 Test _count_linear_flops helper."""
    print("🧪 Unit Test: _count_linear_flops...")

    # Create mock Linear layer
    class MockLinear:
        def __init__(self, in_f, out_f):
            self.weight = Tensor(rng.standard_normal((in_f, out_f)))
            self.__class__.__name__ = 'Linear'

    # Test 1: Known dimensions
    layer = MockLinear(128, 64)
    flops = _count_linear_flops(layer, (1, 128))
    assert flops == 128 * 64 * 2, f"Expected {128*64*2}, got {flops}"
    print(f"✅ Linear(128, 64): {flops} FLOPs")

    # Test 2: Square layer
    layer_sq = MockLinear(256, 256)
    flops_sq = _count_linear_flops(layer_sq, (1, 256))
    assert flops_sq == 256 * 256 * 2, f"Expected {256*256*2}, got {flops_sq}"
    print(f"✅ Linear(256, 256): {flops_sq} FLOPs")

    # Test 3: Batch independence (uses last dim only)
    flops_b1 = _count_linear_flops(layer, (1, 128))
    flops_b32 = _count_linear_flops(layer, (32, 128))
    assert flops_b1 == flops_b32, "FLOPs should be batch-independent"
    print("✅ Batch-independent FLOPs confirmed")

    print("✅ _count_linear_flops works correctly!")

if __name__ == "__main__":
    test_unit_count_linear_flops()

# %% [markdown]
r"""
### Arithmetic Intensity: Placing a Workload on the Roofline

📐 defined arithmetic intensity as FLOPs performed per byte moved, and the ridge point
as the intensity where a machine's memory ceiling meets its compute ceiling. Those two
numbers are all you need to say which ceiling a workload is under, so they are worth
writing down as code rather than carrying in your head.

The function below is three lines and it is the most reused thing in this module.
Quantization, pruning, fusion, and KV caching are all arguments about one of its two
inputs, and every one of them is judged by which side of the ridge the result lands on.
Note what it does not do. It takes the byte count as given, because nothing in a NumPy
framework can observe real DRAM traffic. You supply the bytes an ideal kernel would
move, and the answer is an upper bound on intensity.
"""

# %% nbgrader={"grade": false, "grade_id": "arithmetic-intensity", "solution": true}
#| export
def arithmetic_intensity(flops: int, bytes_moved: float,
                         ridge_point: float) -> Dict[str, Any]:
    """
    Compute arithmetic intensity and place it against a machine's ridge point.

    ```
    I          = flops / bytes_moved                 (FLOP per byte)
    I_ridge    = peak_compute / peak_bandwidth       (supplied by the caller)
    regime     = 'memory' if I < I_ridge else 'compute'
    ```

    TODO: Divide work by traffic, then compare against the ridge point.

    APPROACH:
    1. Divide flops by bytes_moved, guarding a zero denominator
    2. Compare the result against ridge_point
    3. Return the intensity, the ridge point, and the regime label

    EXAMPLE:
    >>> # A Linear(1000, 500) at batch 1 moves its whole weight matrix for 1 MFLOP
    >>> r = arithmetic_intensity(flops=1_000_000, bytes_moved=2_000_000, ridge_point=9.56)
    >>> r['intensity'], r['regime']
    (0.5, 'memory')

    HINTS:
    - Use max(bytes_moved, 1e-9) so a zero byte count cannot raise
    - The boundary case belongs to compute, matching 📐's $I \\ge I_{ridge}$
    - Return the ridge point back to the caller so a printed table can show both

    Args:
        flops: Floating point operations the kernel performs
        bytes_moved: Bytes the kernel must move to perform them
        ridge_point: The machine's peak FLOP/s divided by its peak bytes/s

    Returns:
        dict with intensity, ridge_point, and a regime label of 'memory' or 'compute'
    """
    ### BEGIN SOLUTION
    intensity = flops / max(bytes_moved, 1e-9)
    return {
        'intensity': intensity,
        'ridge_point': ridge_point,
        'regime': 'compute' if intensity >= ridge_point else 'memory'
    }
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: arithmetic_intensity

This test validates the roofline placement the rest of the curriculum argues from.

**What we're testing**: $I = \text{FLOPs}/\text{Bytes}$ and its comparison against a ridge point
**Why it matters**: Every optimization in Modules 15 to 19 claims to move a workload across this line
**Expected**: $0.5\text{ FLOP/B}$ at batch 1 and $16\text{ FLOP/B}$ at batch 32, straddling the A100 ridge
"""

# %% nbgrader={"grade": true, "grade_id": "test-arithmetic-intensity", "locked": true, "points": 3}
def test_unit_arithmetic_intensity():
    """🧪 Test arithmetic_intensity helper."""
    print("🧪 Unit Test: arithmetic_intensity...")

    # An A100 at 19.5 TFLOP/s FP32 over 2,039 GB/s of HBM2e.
    a100_ridge = 19.5e12 / 2039e9

    # Test 1: Linear(1000, 500) at batch 1. The weight matrix is 2,000,000 bytes
    # at FP32 and the layer performs 2 * 1000 * 500 = 1,000,000 FLOPs.
    single = arithmetic_intensity(flops=1_000_000, bytes_moved=2_000_000,
                                 ridge_point=a100_ridge)
    assert abs(single['intensity'] - 0.5) < 1e-9, f"Expected 0.5, got {single['intensity']}"
    assert single['regime'] == 'memory', "0.5 FLOP/B is far below the A100 ridge"
    print(f"✅ Batch 1: {single['intensity']:.2f} FLOP/B -> {single['regime']}-bound")

    # Test 2: The same layer at batch 32. FLOPs scale, weight traffic does not.
    batched = arithmetic_intensity(flops=32_000_000, bytes_moved=2_000_000,
                                  ridge_point=a100_ridge)
    assert abs(batched['intensity'] - 16.0) < 1e-9, f"Expected 16.0, got {batched['intensity']}"
    assert batched['regime'] == 'compute', "16 FLOP/B is above the A100 ridge of 9.6"
    print(f"✅ Batch 32: {batched['intensity']:.2f} FLOP/B -> {batched['regime']}-bound")

    # Test 3: Sitting exactly on the ridge counts as compute-bound
    on_ridge = arithmetic_intensity(flops=96, bytes_moved=10.0, ridge_point=9.6)
    assert on_ridge['regime'] == 'compute', "I == I_ridge belongs to the compute side"
    print("✅ Exactly on the ridge: compute-bound")

    # Test 4: Zero bytes cannot raise
    guarded = arithmetic_intensity(flops=0, bytes_moved=0.0, ridge_point=9.6)
    assert guarded['regime'] == 'memory', "Zero work is not compute-bound"
    print("✅ Zero-byte safety handled")

    print("✅ arithmetic_intensity works correctly!")

if __name__ == "__main__":
    test_unit_arithmetic_intensity()

# %% [markdown]
r"""
### Bottleneck Classification: Compute-Bound or Memory-Bound

Every optimization decision starts with one question. Is this workload waiting on
arithmetic, or waiting on data? The two answers point in opposite directions. A
compute-bound layer gets faster from lower precision or a better kernel. A
memory-bound layer ignores both and responds only to moving fewer bytes, whether by
fusion, caching, or quantizing the weights that have to travel.

The classifier below is deliberately crude. It compares achieved memory
bandwidth against achieved compute throughput and calls a lopsided ratio
memory-bound. That is a screening heuristic, not the real analysis. The rigorous
version is the `arithmetic_intensity` you just wrote, read against the hardware's
ridge point. What we need here is a fast first read that needs no byte count and is
honest about being a screen.
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-bottleneck", "solution": true}
#| export
def _analyze_bottleneck(gflops_per_second: float,
                       memory_bandwidth_mbs: float) -> Dict[str, Any]:
    """
    Illustrate a heuristic memory/compute classification.

    This is not a hardware diagnosis: allocation footprint is not transferred
    bytes, and this classifier has no measured machine bandwidth or compute peak.

    ```
    Bottleneck Decision:
    If bandwidth >> GFLOP/s x 100 -> Memory-bound (data movement dominates)
    Otherwise                     -> Compute-bound (arithmetic dominates)
    ```

    TODO: Classify the workload and report the result.

    APPROACH:
    1. Compare memory_bandwidth_mbs against gflops_per_second * 100
    2. If bandwidth is larger, the workload is memory-bound
    3. Return a dict with is_memory_bound, is_compute_bound, and a label

    EXAMPLE:
    >>> _analyze_bottleneck(gflops_per_second=1.0, memory_bandwidth_mbs=10000.0)
    {'is_memory_bound': True, 'is_compute_bound': False, 'bottleneck': 'memory'}

    HINTS:
    - The two boolean flags are mutually exclusive, so derive one from the other
    - The 100 is a rule-of-thumb scale factor, not a physical constant
    - Return the label as a plain string so callers can print it directly

    Args:
        gflops_per_second: Achieved compute throughput
        memory_bandwidth_mbs: Achieved memory bandwidth in MB/s

    Returns:
        dict with is_memory_bound, is_compute_bound, and a bottleneck label
    """
    ### BEGIN SOLUTION
    is_memory_bound = memory_bandwidth_mbs > gflops_per_second * 100
    return {
        'is_memory_bound': is_memory_bound,
        'is_compute_bound': not is_memory_bound,
        'bottleneck': 'memory' if is_memory_bound else 'compute'
    }
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _analyze_bottleneck

This test validates the helper that identifies memory-bound vs compute-bound workloads.

**What we're testing**: Bottleneck classification based on bandwidth/compute ratio
**Why it matters**: Knowing the bottleneck determines the right optimization strategy
**Expected**: Correct classification of memory-bound and compute-bound workloads
"""

# %% nbgrader={"grade": true, "grade_id": "test-analyze-bottleneck", "locked": true, "points": 3}
def test_unit_analyze_bottleneck():
    """🧪 Test _analyze_bottleneck helper."""
    print("🧪 Unit Test: _analyze_bottleneck...")

    # Test 1: Memory-bound (high bandwidth relative to compute)
    result = _analyze_bottleneck(gflops_per_second=1.0, memory_bandwidth_mbs=10000.0)
    assert result['is_memory_bound'] is True, "High bandwidth should be memory-bound"
    assert result['bottleneck'] == 'memory'
    print("✅ High bandwidth -> memory-bound")

    # Test 2: Compute-bound (low bandwidth relative to compute)
    result = _analyze_bottleneck(gflops_per_second=50.0, memory_bandwidth_mbs=100.0)
    assert result['is_compute_bound'] is True, "Low bandwidth should be compute-bound"
    assert result['bottleneck'] == 'compute'
    print("✅ Low bandwidth -> compute-bound")

    # Test 3: Mutually exclusive flags
    result = _analyze_bottleneck(gflops_per_second=10.0, memory_bandwidth_mbs=500.0)
    assert result['is_memory_bound'] != result['is_compute_bound'], \
        "Memory-bound and compute-bound should be mutually exclusive"
    print(f"✅ Mutually exclusive: bottleneck = {result['bottleneck']}")

    print("✅ _analyze_bottleneck works correctly!")

if __name__ == "__main__":
    test_unit_analyze_bottleneck()

# %% [markdown]
r"""
### Memory Efficiency: Useful Bytes vs. Peak Bytes

Peak memory is almost never the memory you asked for. Allocators round up,
intermediate buffers outlive their use, fragmentation strands free blocks that
are individually too small to reuse. The ratio of useful bytes to peak bytes is
the cheapest available signal for how much of that overhead you are carrying.

A low ratio does not tell you which cause is responsible, and it is not a
verdict on the model. It tells you where to look next, and whether the answer
to an out-of-memory error is a smaller model or a better allocation pattern.
"""

# %% nbgrader={"grade": false, "grade_id": "calculate-memory-efficiency", "solution": true}
#| export
def _calculate_memory_efficiency(useful_memory_mb: float, peak_memory_mb: float) -> float:
    """
    Compute the fraction of peak memory that was actually useful.

    ```
    efficiency = useful_memory / peak_memory, clamped to at most 1.0
    ```

    TODO: Return the useful-to-peak memory ratio.

    APPROACH:
    1. Divide useful by peak, guarding against a zero denominator
    2. Clamp the result to 1.0 so rounding cannot report over 100 percent

    EXAMPLE:
    >>> _calculate_memory_efficiency(useful_memory_mb=80.0, peak_memory_mb=100.0)
    0.8

    HINTS:
    - Use max(peak, 0.001) rather than an if-statement for the zero guard
    - min(ratio, 1.0) is the clamp; efficiency above 1.0 is a measurement error

    Args:
        useful_memory_mb: Memory attributable to parameters and activations
        peak_memory_mb: Highest memory actually held at once

    Returns:
        float: Efficiency in [0.0, 1.0]
    """
    ### BEGIN SOLUTION role="scaffold"
    ratio = useful_memory_mb / max(peak_memory_mb, 0.001)
    return min(ratio, 1.0)
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _calculate_memory_efficiency

This test validates the helper that computes useful-to-total memory ratio.

**What we're testing**: Efficiency = useful_memory / peak_memory, clamped to [0, 1]
**Why it matters**: Low efficiency means memory fragmentation or allocator overhead
**Expected**: Values between 0 and 1, with division-by-zero safety
"""

# %% nbgrader={"grade": true, "grade_id": "test-calculate-memory-efficiency", "locked": true, "points": 3}
def test_unit_calculate_memory_efficiency():
    """🧪 Test _calculate_memory_efficiency helper."""
    print("🧪 Unit Test: _calculate_memory_efficiency...")

    # Test 1: Perfect efficiency
    eff = _calculate_memory_efficiency(10.0, 10.0)
    assert abs(eff - 1.0) < 0.01, f"Expected 1.0, got {eff}"
    print(f"✅ Perfect efficiency: {eff}")

    # Test 2: Half efficiency
    eff_half = _calculate_memory_efficiency(5.0, 10.0)
    assert abs(eff_half - 0.5) < 0.01, f"Expected 0.5, got {eff_half}"
    print(f"✅ Half efficiency: {eff_half}")

    # Test 3: Clamped at 1.0 (useful > peak shouldn't exceed 1.0)
    eff_clamped = _calculate_memory_efficiency(20.0, 10.0)
    assert eff_clamped <= 1.0, f"Efficiency should be clamped to 1.0, got {eff_clamped}"
    print(f"✅ Clamped efficiency: {eff_clamped}")

    # Test 4: Division by zero safety
    eff_zero = _calculate_memory_efficiency(5.0, 0.0)
    assert eff_zero <= 1.0, f"Should handle zero peak safely, got {eff_zero}"
    print("✅ Zero-peak safety handled")

    print("✅ _calculate_memory_efficiency works correctly!")

if __name__ == "__main__":
    test_unit_calculate_memory_efficiency()

# %% [markdown]
r"""
### Derived Metrics: Turning Counts into Rates

Raw measurements are not yet insight. A FLOP count and a latency are two
unrelated numbers until you divide them, at which point they become throughput
and can be compared against hardware that has a known ceiling. The same is true
of memory over time, which becomes bandwidth.

One value below deserves suspicion. The theoretical peak is hard-coded at 100
GFLOP/s. That stands in for a real hardware number the profiler has no way to
query from pure NumPy, so the efficiency figure it produces is a relative
indicator, not a hardware utilization percentage. Treat a rising number as
progress and ignore its absolute value.
"""

# %% nbgrader={"grade": false, "grade_id": "compute-derived-metrics", "solution": true}
#| export
def _compute_derived_metrics(flops: int, latency_ms: float,
                             peak_memory_mb: float) -> Dict[str, float]:
    """
    Turn raw counts and timings into comparable rates.

    The bandwidth field is a footprint/time proxy, not measured memory traffic.
    Neither it nor the assumed compute peak establishes a hardware bottleneck.

    ```
    GFLOP/s   = (flops / 1e9) / seconds
    MB/s      = peak_memory_mb / seconds
    efficiency = GFLOP/s / theoretical_peak
    ```

    TODO: Convert measurements into throughput, bandwidth, and efficiency.

    APPROACH:
    1. Convert latency from milliseconds to seconds
    2. Divide GFLOPs by seconds for compute throughput
    3. Divide peak memory by seconds for effective bandwidth
    4. Divide throughput by the assumed peak, clamped to 1.0

    EXAMPLE:
    >>> m = _compute_derived_metrics(flops=1_000_000, latency_ms=1.0, peak_memory_mb=10.0)
    >>> round(m['gflops_per_second'], 3)
    1.0

    HINTS:
    - Guard every division with max(seconds, 1e-6); a zero latency is possible
    - theoretical_peak_gflops is a placeholder constant, not a measured value
    - Return a dict so callers can name what they read

    Args:
        flops: Total floating point operations
        latency_ms: Measured wall-clock latency in milliseconds
        peak_memory_mb: Peak memory held during the measurement

    Returns:
        dict with gflops_per_second, memory_bandwidth_mbs, computational_efficiency
    """
    ### BEGIN SOLUTION role="scaffold"
    latency_seconds = latency_ms / 1000.0
    gflops_per_second = (flops / 1e9) / max(latency_seconds, 1e-6)
    memory_bandwidth = peak_memory_mb / max(latency_seconds, 1e-6)
    theoretical_peak_gflops = 100.0
    computational_efficiency = min(gflops_per_second / theoretical_peak_gflops, 1.0)

    return {
        'gflops_per_second': gflops_per_second,
        'memory_bandwidth_mbs': memory_bandwidth,
        'computational_efficiency': computational_efficiency
    }
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _compute_derived_metrics

This test validates the helper that converts raw FLOPs and latency into throughput metrics.

**What we're testing**: GFLOP/s, memory bandwidth, and computational efficiency calculations
**Why it matters**: These derived metrics determine whether a workload is memory-bound or compute-bound
**Expected**: Correct throughput calculations from known FLOP counts and latencies
"""

# %% nbgrader={"grade": true, "grade_id": "test-compute-derived-metrics", "locked": true, "points": 3}
def test_unit_compute_derived_metrics():
    """🧪 Test _compute_derived_metrics helper."""
    print("🧪 Unit Test: _compute_derived_metrics...")

    # Test 1: Known values -> known throughput
    # 1e9 FLOPs in 1000ms (1 second) = 1.0 GFLOP/s
    metrics = _compute_derived_metrics(
        flops=1_000_000_000, latency_ms=1000.0, peak_memory_mb=100.0
    )
    assert abs(metrics['gflops_per_second'] - 1.0) < 0.01, \
        f"Expected 1.0 GFLOP/s, got {metrics['gflops_per_second']}"
    print(f"✅ 1B FLOPs / 1s = {metrics['gflops_per_second']:.1f} GFLOP/s")

    # Test 2: Memory bandwidth calculation
    # 100 MB in 1 second = 100 MB/s
    assert abs(metrics['memory_bandwidth_mbs'] - 100.0) < 0.1, \
        f"Expected 100 MB/s, got {metrics['memory_bandwidth_mbs']}"
    print(f"✅ Memory bandwidth: {metrics['memory_bandwidth_mbs']:.1f} MB/s")

    # Test 3: Efficiency bounded by [0, 1]
    assert 0 <= metrics['computational_efficiency'] <= 1.0, \
        f"Efficiency out of bounds: {metrics['computational_efficiency']}"
    print(f"✅ Efficiency: {metrics['computational_efficiency']:.3f}")

    print("✅ _compute_derived_metrics works correctly!")

if __name__ == "__main__":
    test_unit_compute_derived_metrics()

# %% [markdown]
r"""
### Backward Pass Cost: Why Training Is 3x Inference

Training costs roughly three times what inference costs, and the split is worth
knowing precisely, one unit forward and two units backward. The backward pass is
twice the forward because it computes two gradients at every layer, one with
respect to the inputs, so the chain rule can continue downstream, and one with
respect to the weights, so the optimizer has something to apply.

This 2x is a rule of thumb, not a measurement. It holds well for the dense
matrix multiplies that dominate the networks in this course, and it is exactly
the kind of estimate you want before committing to a training run rather than
after.
"""

# %% nbgrader={"grade": false, "grade_id": "estimate-backward-costs", "solution": true}
#| export
def _estimate_backward_costs(forward_flops: int,
                             forward_latency_ms: float) -> Dict[str, float]:
    """
    Estimate backward-pass cost from forward-pass measurements.

    ```
    backward ~= 2 x forward
      (one gradient w.r.t. inputs, one w.r.t. weights)
    ```

    TODO: Apply the 2x rule to both FLOPs and latency.

    APPROACH:
    1. Multiply forward FLOPs by 2
    2. Multiply forward latency by 2
    3. Return both under descriptive keys

    EXAMPLE:
    >>> _estimate_backward_costs(forward_flops=1000, forward_latency_ms=5.0)
    {'backward_flops': 2000, 'backward_latency_ms': 10.0}

    HINTS:
    - The factor is 2, not 3 (the 3x figure is forward PLUS backward)
    - This is an estimate; a real measurement would time an actual backward call

    Args:
        forward_flops: FLOPs measured for the forward pass
        forward_latency_ms: Latency measured for the forward pass

    Returns:
        dict with backward_flops and backward_latency_ms
    """
    ### BEGIN SOLUTION role="scaffold"
    return {
        'backward_flops': forward_flops * 2,
        'backward_latency_ms': forward_latency_ms * 2
    }
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _estimate_backward_costs

This test validates the helper that estimates backward pass FLOPs and latency from forward measurements.

**What we're testing**: Backward costs = 2x forward costs (standard ML heuristic)
**Why it matters**: Training cost = forward + backward; backward is typically 2x forward
**Expected**: Backward FLOPs and latency are exactly 2x the forward values
"""

# %% nbgrader={"grade": true, "grade_id": "test-estimate-backward-costs", "locked": true, "points": 3}
def test_unit_estimate_backward_costs():
    """🧪 Test _estimate_backward_costs helper."""
    print("🧪 Unit Test: _estimate_backward_costs...")

    # Test 1: Known forward values -> 2x backward
    costs = _estimate_backward_costs(forward_flops=1000, forward_latency_ms=5.0)
    assert costs['backward_flops'] == 2000, f"Expected 2000, got {costs['backward_flops']}"
    assert costs['backward_latency_ms'] == 10.0, f"Expected 10.0, got {costs['backward_latency_ms']}"
    print(f"✅ 1000 forward FLOPs -> {costs['backward_flops']} backward FLOPs")

    # Test 2: Zero forward -> zero backward
    costs_zero = _estimate_backward_costs(forward_flops=0, forward_latency_ms=0.0)
    assert costs_zero['backward_flops'] == 0
    assert costs_zero['backward_latency_ms'] == 0.0
    print("✅ Zero forward -> zero backward")

    print("✅ _estimate_backward_costs works correctly!")

if __name__ == "__main__":
    test_unit_estimate_backward_costs()

# %% [markdown]
r"""
### Optimizer Memory: The Hidden Cost of Adam

Optimizer state is the memory cost people forget. SGD keeps nothing between
steps, so it adds nothing. Adam keeps two running averages per parameter, the
first and second moments built in Module 07, which doubles the gradient-sized
footprint before a single activation is stored.

For a large model that difference decides whether training fits at all. It is
also why the choice of optimizer is a systems decision and not only a
convergence one. Switching from Adam to SGD can buy back more memory than any
batch-size reduction you were considering.
"""

# %% nbgrader={"grade": false, "grade_id": "estimate-optimizer-memory", "solution": true}
#| export
def _estimate_optimizer_memory(gradient_memory_mb: float) -> Dict[str, float]:
    """
    Estimate per-optimizer state memory.

    ```
    SGD   : no persistent state             -> 0
    Adam  : first + second moment           -> 2 x gradient memory
    AdamW : same state as Adam              -> 2 x gradient memory
    ```

    TODO: Report the extra memory each optimizer holds between steps.

    APPROACH:
    1. SGD stores no state, so its cost is 0
    2. Adam stores two moment buffers, each the size of the gradients
    3. AdamW stores the same state as Adam; only its decay differs

    EXAMPLE:
    >>> _estimate_optimizer_memory(gradient_memory_mb=100.0)
    {'sgd': 0, 'adam': 200.0, 'adamw': 200.0}

    HINTS:
    - This is state held BETWEEN steps, not transient working memory
    - AdamW differs from Adam in its weight decay, not its memory
    - Momentum-SGD would be 1x, but plain SGD is 0

    Args:
        gradient_memory_mb: Memory occupied by one full set of gradients

    Returns:
        dict mapping optimizer name to its extra memory in MB
    """
    ### BEGIN SOLUTION role="scaffold"
    return {
        'sgd': 0,
        'adam': gradient_memory_mb * 2,
        'adamw': gradient_memory_mb * 2,
    }
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _estimate_optimizer_memory

This test validates the helper that estimates memory requirements for different optimizers.

**What we're testing**: Per-optimizer memory multipliers (SGD: 0x, Adam: 2x gradient memory)
**Why it matters**: Adam uses 2x extra memory vs SGD; this affects hardware requirements
**Expected**: SGD = 0 extra, Adam = 2x gradient memory, AdamW = 2x gradient memory
"""

# %% nbgrader={"grade": true, "grade_id": "test-estimate-optimizer-memory", "locked": true, "points": 3}
def test_unit_estimate_optimizer_memory():
    """🧪 Test _estimate_optimizer_memory helper."""
    print("🧪 Unit Test: _estimate_optimizer_memory...")

    # Test with 100 MB gradient memory
    estimates = _estimate_optimizer_memory(gradient_memory_mb=100.0)

    assert estimates['sgd'] == 0, f"SGD should need 0 extra, got {estimates['sgd']}"
    assert estimates['adam'] == 200.0, f"Adam should need 200 MB, got {estimates['adam']}"
    assert estimates['adamw'] == 200.0, f"AdamW should need 200 MB, got {estimates['adamw']}"
    print(f"✅ SGD: {estimates['sgd']} MB, Adam: {estimates['adam']} MB, AdamW: {estimates['adamw']} MB")

    # Test with zero gradients
    estimates_zero = _estimate_optimizer_memory(gradient_memory_mb=0.0)
    assert estimates_zero['adam'] == 0.0, "Zero gradients -> zero optimizer memory"
    print("✅ Zero gradient memory handled correctly")

    print("✅ _estimate_optimizer_memory works correctly!")

if __name__ == "__main__":
    test_unit_estimate_optimizer_memory()

# %% [markdown]
r"""
### Profiler: The Object That Ties the Measurements Together

Every helper above answers one narrow question: how many parameters, how many
FLOPs, how much memory, how long. The Profiler is the object that runs them
against a real model and returns one report.

| Profiler Probe | Analytical Domain | Measurement Mechanism |
| :--- | :--- | :--- |
| **`count_parameters()`** | Static Model Topology | Iterates tensor weights and biases to sum total element count |
| **`count_flops()`** | Computational Work | Arithmetic operations ($2 \times M \times K \times N$ for GEMMs) |
| **`estimate_memory()`** | Memory Footprint | Parameter bytes, activation tensors, gradients, and optimizer buffers |
| **`measure_latency()`** | Temporal Wall-Clock | Monotonic interval timer across warm and active forward executions |

These four diagnostic probes are orchestrated by `Profiler.profile_forward_pass(model, input)` into a unified report dictionary.

Read the class through one concrete call: `profile_forward_pass(model,
input_tensor)`. It uses the supplied input for warmup runs and timed forward
passes, then combines their summary with size estimates. Follow `count_parameters` separately
to see a quantity computed from tensor sizes, then `count_flops` to see an
operation-count estimate. These are different kinds of evidence; a FLOP count
does not measure elapsed time. The memory helpers add estimates of activations,
gradients, and optimizer state once that distinction is clear.

The complete class is kept together to show how these measurements share state.
Its individual methods are exercised in the parameter, FLOP, memory, and
latency sections below. Module 19 will add comparisons across repeated runs
and candidate models.
"""

# %% nbgrader={"grade": false, "grade_id": "profiler_class", "solution": true}
#| export
class Profiler:
    """
    ML model profiler for parameters, FLOPs, memory, and latency.

    Measures what pure NumPy can see, and is explicit about the rest. Parameter
    and FLOP counts are exact. Latency is a median over repeated warm runs.
    Memory is an allocation footprint from tracemalloc, not device bytes moved,
    and the compute peak it divides by is a placeholder constant rather than a
    measured machine. Read the derived rates as relative indicators.

    Every profile_forward_pass report is also stored in .measurements, keyed by the
    model's class name, so one profiler reused across candidate models keeps a
    record you can compare after the sweep.
    """

    def __init__(self):
        """
        Initialize profiler with measurement state.

        TODO: Set up the record every profiling call writes into

        APPROACH:
        1. Create an empty measurements dictionary keyed by model class name
        2. That is the whole constructor; profile_forward_pass fills it in

        EXAMPLE:
        >>> profiler = Profiler()
        >>> profiler.measurements
        {}

        HINTS:
        - One instance can profile several models, so keep the record on the instance
        - Reuse one Profiler across a comparison and .measurements holds every report
        """
        ### BEGIN SOLUTION role="scaffold"
        self.measurements = {}
        ### END SOLUTION

    def __enter__(self):
        """
        Start timing, so that this times the block it wraps:

            with Profiler() as p:
                model.forward(x)
            print(p.elapsed)   # milliseconds
        """
        self._context_start = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Stop timing and store the elapsed time in milliseconds on .elapsed."""
        self.elapsed = (time.perf_counter() - self._context_start) * 1000

    def count_parameters(self, model) -> int:
        """
        Count total trainable parameters in a model.

        TODO: Implement parameter counting for any model with parameters() method

        APPROACH:
        1. Get all parameters from model.parameters() if available
        2. For single layers, use _count_layer_parameters() helper
        3. Sum total element count across all parameter tensors

        EXAMPLE:
        >>> linear = Linear(128, 64)  # 128*64 + 64 = 8256 parameters
        >>> profiler = Profiler()
        >>> count = profiler.count_parameters(linear)
        >>> print(count)
        8256

        HINTS:
        - Use _count_layer_parameters() for single layers
        - Use parameter.data.size for tensor element count
        - Handle models with and without parameters() method
        """
        ### BEGIN SOLUTION role="scaffold"
        if hasattr(model, 'parameters'):
            return sum(p.data.size for p in model.parameters())
        if hasattr(model, 'weight'):
            return _count_layer_parameters(model)
        return 0
        ### END SOLUTION



    def _count_sequential_flops(self, model, input_shape: Tuple[int, ...]) -> int:
        """
        Count FLOPs for a Sequential model by summing per-layer FLOPs.

        ```
        Sequential FLOP Accumulation:
        Layer 1 FLOPs + Layer 2 FLOPs + ... + Layer N FLOPs = Total FLOPs
             ↓               ↓                    ↓
          Shape propagated through each layer
        ```

        Args:
            model: A model with .layers attribute (list of layers)
            input_shape: Input tensor shape for the first layer

        Returns:
            int: Total FLOP count across all layers
        """
        ### BEGIN SOLUTION role="scaffold"
        total_flops = 0
        current_shape = input_shape
        for layer in model.layers:
            total_flops += self.count_flops(layer, current_shape)
            if layer.__class__.__name__ in ('Conv2d', 'MaxPool2d', 'AvgPool2d'):
                kernel = layer.kernel_size
                kh, kw = (kernel, kernel) if isinstance(kernel, int) else kernel
                stride = layer.stride
                sh, sw = (stride, stride) if isinstance(stride, int) else stride
                padding = layer.padding
                ph, pw = (padding, padding) if isinstance(padding, int) else padding
                h, w = current_shape[-2:]
                channels = getattr(layer, 'out_channels', current_shape[1])
                current_shape = (current_shape[0], channels,
                                 (h + 2 * ph - kh) // sh + 1,
                                 (w + 2 * pw - kw) // sw + 1)
            elif hasattr(layer, 'weight') and layer.weight.ndim == 2:
                current_shape = current_shape[:-1] + (layer.weight.shape[1],)
        return total_flops
        ### END SOLUTION

    def count_flops(self, model, input_shape: Tuple[int, ...]) -> int:
        """
        Count per-sample FLOPs for one forward pass.

        Rules cover Linear, Conv2d, flat Sequential chains, and GPT. Pooling
        propagates its output shape but uses the same one-operation-per-input
        estimate as other fallback layers. Custom shape-changing layers need
        their own counting rule.

        TODO: Implement FLOP counting by dispatching to per-layer-type helpers

        APPROACH:
        1. Identify model type by class name
        2. Dispatch to _count_linear_flops, _count_conv_flops, or self._count_sequential_flops
        3. A GPT-style model (it has .blocks and .embed_dim) goes to self._count_transformer_flops
        4. Fall back to 1 FLOP per element for activations

        EXAMPLE:
        >>> linear = Linear(128, 64)
        >>> profiler = Profiler()
        >>> flops = profiler.count_flops(linear, (1, 128))
        >>> print(flops)  # 128 * 64 * 2 = 16384
        16384

        HINT: Use model.__class__.__name__ to identify layer type
        """
        ### BEGIN SOLUTION role="scaffold"
        model_name = model.__class__.__name__

        if model_name == 'Linear':
            return _count_linear_flops(model, input_shape)
        elif model_name == 'Conv2d':
            return _count_conv_flops(model, input_shape)
        elif model_name == 'Sequential' or hasattr(model, 'layers'):
            return self._count_sequential_flops(model, input_shape)
        elif hasattr(model, 'blocks') and hasattr(model, 'embed_dim'):
            return self._count_transformer_flops(model, input_shape)
        else:
            sample_shape = input_shape[1:] if len(input_shape) > 1 else input_shape
            return int(np.prod(sample_shape))
        ### END SOLUTION

    def _count_transformer_flops(self, model, input_shape: Tuple[int, ...]) -> int:
        """
        Count FLOPs for one sequence through a GPT-style model (Module 13).

        input_shape is (batch, seq_len) of token ids. Like the other counters this
        is per sample: every Linear layer costs 2 * in * out per token, so a
        sequence of seq_len tokens pays seq_len times that, and each block adds the
        two attention products Q K^T and weights V, 4 * seq_len^2 * embed_dim.
        Embeddings, LayerNorm, GELU, and softmax are a few operations per element
        and are left out.
        """
        seq_len = input_shape[-1]
        embed_dim = model.embed_dim
        flops = 0
        for block in model.blocks:
            attn, mlp = block.attention, block.mlp
            for layer in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj, mlp.linear1, mlp.linear2):
                flops += seq_len * _count_linear_flops(layer, (1, layer.in_features))
            flops += 4 * seq_len * seq_len * embed_dim
        flops += seq_len * _count_linear_flops(model.lm_head, (1, embed_dim))
        return flops

    def _calculate_parameter_memory(self, model) -> float:
        """
        Calculate memory used by model parameters in megabytes.

        ```
        Parameter Memory Formula:
        Memory (MB) = parameter_count × 4 bytes / (1024 × 1024)
                           ↑              ↑
                     From count_parameters  FP32 size
        ```

        Args:
            model: Model to analyze

        Returns:
            float: Parameter memory in megabytes
        """
        ### BEGIN SOLUTION role="scaffold"
        param_count = self.count_parameters(model)
        return (param_count * BYTES_PER_FLOAT32) / MB_TO_BYTES
        ### END SOLUTION

    def _dummy_input(self, model, input_shape: Tuple[int, ...]) -> Tensor:
        """
        Build an input the model can consume, for timing and memory runs.

        A token model (anything with a vocab_size, like Module 13's GPT) takes
        integer ids; every other layer in TinyTorch takes float activations.
        """
        if hasattr(model, 'vocab_size'):
            return Tensor(rng.integers(0, model.vocab_size, size=input_shape))
        return Tensor(rng.standard_normal(input_shape))


    def measure_memory(self, model, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """
        Estimate the live footprint and trace allocations during a forward pass.

        An existing caller-owned tracing session is preserved; its earlier peak
        can make this an upper bound. The activation estimate is not graph liveness.

        TODO: Implement memory tracking using tracemalloc and helper methods

        APPROACH:
        1. Use _calculate_parameter_memory() for parameter bytes
        2. Use tracemalloc to track peak allocation during forward pass
        3. Use _calculate_memory_efficiency() for efficiency ratio

        EXAMPLE:
        >>> linear = Linear(1024, 512)
        >>> profiler = Profiler()
        >>> memory = profiler.measure_memory(linear, (32, 1024))
        >>> print(f"Parameters: {memory['parameter_memory_mb']:.1f} MB")
        Parameters: 2.0 MB

        HINT: tracemalloc.start() / get_traced_memory() / stop() lifecycle
        """
        ### BEGIN SOLUTION role="scaffold"
        # Own the tracing session only if the caller has not already started it.
        owns_trace = not tracemalloc.is_tracing()
        if owns_trace:
            tracemalloc.start()
        try:
            baseline_memory = tracemalloc.get_traced_memory()[0]
            parameter_memory_mb = self._calculate_parameter_memory(model)
            dummy_input = self._dummy_input(model, input_shape)
            output = model.forward(dummy_input)
            # Count the actual input and output buffers, not two input-sized
            # buffers. A view or identity output does not allocate another one.
            activation_bytes = dummy_input.data.nbytes
            if not np.shares_memory(dummy_input.data, output.data):
                activation_bytes += output.data.nbytes
            activation_memory_mb = activation_bytes / MB_TO_BYTES
            _, peak_memory = tracemalloc.get_traced_memory()
            # Parameters were already live at baseline. Add them to the NEW
            # allocations, even when they were tracked by a caller's session:
            # baseline subtraction removes those existing bytes first.
            peak_memory_mb = parameter_memory_mb + max(0, peak_memory - baseline_memory) / MB_TO_BYTES
        finally:
            if owns_trace:
                tracemalloc.stop()

        useful_memory = parameter_memory_mb + activation_memory_mb
        # Keep a lower bound from known live buffers even if tracing misses one.
        peak_memory_mb = max(peak_memory_mb, useful_memory)
        return {
            'parameter_memory_mb': parameter_memory_mb,
            'activation_memory_mb': activation_memory_mb,
            'peak_memory_mb': peak_memory_mb,
            'memory_efficiency': _calculate_memory_efficiency(useful_memory, peak_memory_mb)
        }
        ### END SOLUTION

    def measure_latency(self, model, input_tensor, warmup: int = 10, iterations: int = 100) -> float:
        """
        Measure model inference latency with statistical rigor.

        Every iteration reuses the SAME input tensor, so after warmup it is
        cache-resident and the median is a warm best case. Serving traffic arrives
        cold and one request at a time, and will be slower than this reports.

        TODO: Implement accurate latency measurement

        APPROACH:
        1. Run warmup iterations to stabilize performance
        2. Measure multiple iterations for statistical accuracy
        3. Calculate median latency to handle outliers
        4. Return latency in milliseconds

        PARAMETERS:
        - warmup: Number of warmup runs (default 10)
        - iterations: Number of measurement runs (default 100)

        EXAMPLE:
        >>> linear = Linear(128, 64)
        >>> input_tensor = Tensor(rng.standard_normal((1, 128)))
        >>> profiler = Profiler()
        >>> latency = profiler.measure_latency(linear, input_tensor)
        >>> print(f"Latency: {latency:.2f} ms")
        Latency: 0.03 ms      # machine-dependent, yours will differ

        HINTS:
        - Use time.perf_counter() for high precision
        - Use median instead of mean for robustness against outliers
        """
        ### BEGIN SOLUTION role="scaffold"
        if iterations < 1 or warmup < 0:
            raise ValueError("iterations must be positive and warmup nonnegative")
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
        ### END SOLUTION

    def profile_layer(self, layer, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Profile a single layer comprehensively.

        TODO: Implement layer-wise profiling

        APPROACH:
        1. Count parameters for this layer
        2. Count FLOPs for this layer
        3. Measure memory usage
        4. Measure latency
        5. Return comprehensive layer profile

        EXAMPLE:
        >>> linear = Linear(256, 128)
        >>> profiler = Profiler()
        >>> profile = profiler.profile_layer(linear, (32, 256))
        >>> print(f"Layer uses {profile['parameters']} parameters")
        Layer uses 32896 parameters

        HINTS:
        - Use existing profiler methods (count_parameters, count_flops, etc.)
        - Create dummy input for latency measurement
        - Include layer type information in profile
        """
        ### BEGIN SOLUTION role="scaffold"
        # Create dummy input for latency measurement
        dummy_input = self._dummy_input(layer, input_shape)

        # Gather all measurements
        params = self.count_parameters(layer)
        flops = self.count_flops(layer, input_shape)
        memory = self.measure_memory(layer, input_shape)
        latency = self.measure_latency(layer, dummy_input, warmup=3, iterations=10)

        # Compute derived metrics
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        gflops_per_second = (flops * batch_size / 1e9) / max(latency / 1000, 1e-6)

        return {
            'layer_type': layer.__class__.__name__,
            'parameters': params,
            'flops': flops,
            'latency_ms': latency,
            'gflops_per_second': gflops_per_second,
            **memory
        }
        ### END SOLUTION



    def profile_forward_pass(self, model, input_tensor) -> Dict[str, Any]:
        """
        Comprehensive profiling of a model's forward pass.

        TODO: Gather measurements, then use _compute_derived_metrics and analyze_bottleneck

        APPROACH:
        1. Gather raw measurements (parameters, FLOPs, memory, latency)
        2. Use _compute_derived_metrics() for throughput and efficiency
        3. Use _analyze_bottleneck() for bottleneck identification

        EXAMPLE:
        >>> model = Linear(256, 128)
        >>> input_data = Tensor(rng.standard_normal((32, 256)))
        >>> profiler = Profiler()
        >>> profile = profiler.profile_forward_pass(model, input_data)
        >>> print(f"Throughput: {profile['gflops_per_second']:.2f} GFLOP/s")
        Throughput: 0.04 GFLOP/s   # machine-dependent, yours will differ

        HINT: Compose helper outputs with ** unpacking into return dict
        """
        ### BEGIN SOLUTION role="scaffold"
        param_count = self.count_parameters(model)
        flops = self.count_flops(model, input_tensor.shape)
        memory_stats = self.measure_memory(model, input_tensor.shape)
        latency_ms = self.measure_latency(model, input_tensor, warmup=5, iterations=20)

        # count_flops is per sample; measure_latency times the whole batch. Dividing
        # one by the other without this factor understates throughput by the batch
        # size, which is why every GFLOP/s and bottleneck label used to look
        # memory-bound no matter what the model did.
        batch_size = input_tensor.shape[0] if len(input_tensor.shape) > 1 else 1
        batch_flops = flops * batch_size

        derived = _compute_derived_metrics(batch_flops, latency_ms, memory_stats['peak_memory_mb'])
        bottleneck = _analyze_bottleneck(derived['gflops_per_second'],
                                              derived['memory_bandwidth_mbs'])

        report = {
            # 'flops' stays per sample, matching count_flops. 'batch_flops' is the
            # figure the throughput below was computed from.
            'parameters': param_count, 'flops': flops, 'batch_flops': batch_flops,
            'latency_ms': latency_ms,
            **memory_stats, **derived, **bottleneck
        }
        # Keep the report on the instance so one profiler reused across several
        # models leaves a record the caller can compare afterwards.
        self.measurements[model.__class__.__name__] = report
        return report
        ### END SOLUTION



    def profile_backward_pass(self, model, input_tensor) -> Dict[str, Any]:
        """
        Profile both forward and backward passes for training analysis.

        TODO: Use _estimate_backward_costs and _estimate_optimizer_memory helpers

        APPROACH:
        1. Profile forward pass with profile_forward_pass()
        2. Use _estimate_backward_costs() for backward FLOPs and latency
        3. Use _estimate_optimizer_memory() for optimizer memory estimates
        4. Combine into total training iteration metrics

        EXAMPLE:
        >>> model = Linear(128, 64)
        >>> input_data = Tensor(rng.standard_normal((16, 128)))
        >>> profiler = Profiler()
        >>> profile = profiler.profile_backward_pass(model, input_data)
        >>> print(f"Training iteration: {profile['total_latency_ms']:.2f} ms")
        Training iteration: 1.13 ms  # machine-dependent, yours will differ

        HINT: Gradient memory equals parameter memory (one gradient per parameter)
        """
        ### BEGIN SOLUTION role="scaffold"
        fwd = self.profile_forward_pass(model, input_tensor)
        bwd = _estimate_backward_costs(fwd['flops'], fwd['latency_ms'])

        gradient_memory_mb = fwd['parameter_memory_mb']
        total_flops = fwd['flops'] + bwd['backward_flops']
        total_latency_ms = fwd['latency_ms'] + bwd['backward_latency_ms']
        total_memory_mb = fwd['parameter_memory_mb'] + fwd['activation_memory_mb'] + gradient_memory_mb

        return {
            'forward_flops': fwd['flops'],
            'forward_latency_ms': fwd['latency_ms'],
            'forward_memory_mb': fwd['peak_memory_mb'],
            **bwd,
            'gradient_memory_mb': gradient_memory_mb,
            'total_flops': total_flops,
            'total_latency_ms': total_latency_ms,
            'total_memory_mb': total_memory_mb,
            'total_gflops_per_second': (total_flops * (input_tensor.shape[0] if input_tensor.ndim > 1 else 1) / 1e9) / max(total_latency_ms / 1000.0, 1e-6),
            'optimizer_memory_estimates': _estimate_optimizer_memory(gradient_memory_mb),
            'memory_efficiency': fwd['memory_efficiency'],
            'bottleneck': fwd['bottleneck']
        }
        ### END SOLUTION

# %% [markdown]
r"""
## 🏗️ Helper Functions: Quick Profiling Utilities

These helper functions provide simplified interfaces for common profiling tasks. They make it easy to quickly profile models and analyze characteristics without manually calling multiple profiler methods.

### Why Helper Functions Matter

In production ML engineering, you often need quick insights without setting up full profiling workflows. These utilities provide:
- **Quick profiling**: One-line model analysis with formatted output
- **Weight analysis**: Understanding parameter distributions for compression
- **Student-friendly output**: Clear, formatted results for learning

These functions wrap our core Profiler class with convenience interfaces used in real ML workflows for rapid iteration and debugging.
"""

# %% nbgrader={"grade": false, "grade_id": "helper_quick_profile", "solution": false}
#| export
def quick_profile(model, input_tensor, profiler=None):
    """
    Quick profiling function for immediate insights.

    Provides a simplified interface for profiling that displays key metrics
    in a student-friendly format.

    Args:
        model: Model to profile
        input_tensor: Input data for profiling
        profiler: Optional Profiler instance (creates new one if None)

    Returns:
        dict: Profile results with key metrics

    Example:
        >>> model = Linear(128, 64)
        >>> input_data = Tensor(rng.standard_normal((16, 128)))
        >>> results = quick_profile(model, input_data)
        >>> # Displays formatted output automatically
    """
    if profiler is None:
        profiler = Profiler()

    profile = profiler.profile_forward_pass(model, input_tensor)

    # Display formatted results
    print("🧪 Quick Profile Results:")
    print(f"   Parameters: {profile['parameters']:,}")
    print(f"   FLOPs: {profile['flops']:,}")
    print(f"   Latency: {profile['latency_ms']:.2f} ms")
    print(f"   Memory: {profile['peak_memory_mb']:.2f} MB")
    print(f"   Heuristic bottleneck (not hardware measured): {profile['bottleneck']}")
    print(f"   Efficiency: {profile['computational_efficiency']*100:.1f}%")

    return profile

# %% nbgrader={"grade": false, "grade_id": "helper_weight_distribution", "solution": false}
#| export
def analyze_weight_distribution(model, percentiles=[10, 25, 50, 75, 90]):
    """
    Analyze weight distribution across layers.

    Helps understand how weights are distributed across layers.
    Useful for identifying patterns in parameter magnitudes.

    Args:
        model: Model to analyze
        percentiles: List of percentiles to compute

    Returns:
        dict: Weight distribution statistics

    Example:
        >>> model = Linear(512, 512)
        >>> stats = analyze_weight_distribution(model)
        >>> print(f"Weights < 0.01: {stats['below_threshold_001']:.1f}%")
    """
    # Collect all weights
    weights = []
    if hasattr(model, 'parameters'):
        for param in model.parameters():
            weights.extend(param.data.flatten().tolist())
    elif hasattr(model, 'weight'):
        weights.extend(model.weight.data.flatten().tolist())
    else:
        return {'error': 'No weights found'}

    if not weights:
        return {'error': 'No weights found'}

    weights = np.array(weights)
    abs_weights = np.abs(weights)

    # Calculate statistics
    stats = {
        'total_weights': len(weights),
        'mean': float(np.mean(abs_weights)),
        'std': float(np.std(abs_weights)),
        'min': float(np.min(abs_weights)),
        'max': float(np.max(abs_weights)),
    }

    # Percentile analysis
    for p in percentiles:
        stats[f'percentile_{p}'] = float(np.percentile(abs_weights, p))

    # Threshold analysis (useful for pruning)
    for threshold in [0.001, 0.01, 0.1]:
        below = np.sum(abs_weights < threshold) / len(weights) * 100
        stats[f'below_threshold_{str(threshold).replace(".", "")}'] = below

    return stats

# %% [markdown]
r"""
### 🧪 Unit Test: Helper Functions

This test validates our helper utilities work correctly and provide useful output.

**What we're testing**: Quick profiling and weight distribution analysis
**Why it matters**: These utilities are used daily in production ML workflows
**Expected**: Correct profiles with formatted output
"""

# %% nbgrader={"grade": true, "grade_id": "test-helper-functions", "locked": true, "points": 5}
def test_unit_helper_functions():
    """🧪 Test helper function implementations."""
    print("🧪 Unit Test: Helper Functions...")

    # Test 1: Quick profile function
    test_model = Linear(16, 8)
    test_input = Tensor(rng.standard_normal((8, 16)))
    profile = quick_profile(test_model, test_input, profiler=Profiler())

    # Validate profile contains expected keys
    assert 'parameters' in profile, "Quick profile should include parameters"
    assert 'flops' in profile, "Quick profile should include FLOPs"
    assert 'latency_ms' in profile, "Quick profile should include latency"
    print("✅ Quick profile provides comprehensive metrics")

    # Test 2: Weight distribution analysis
    class SimpleModel:
        def __init__(self):
            self.weight = Tensor(rng.standard_normal((10, 5)) * 0.1)  # Small weights

    model = SimpleModel()
    stats = analyze_weight_distribution(model)

    # Validate statistics structure
    assert 'total_weights' in stats, "Should count total weights"
    assert 'mean' in stats, "Should compute mean"
    assert 'std' in stats, "Should compute standard deviation"
    assert stats['total_weights'] == 50, f"Expected 50 weights, got {stats['total_weights']}"
    print(f"✅ Weight distribution analysis: {stats['total_weights']} weights analyzed")

    # Test 3: Weight distribution with no weights
    class NoWeightModel:
        pass

    no_weight_model = NoWeightModel()
    stats = analyze_weight_distribution(no_weight_model)
    assert 'error' in stats, "Should handle models without weights"
    print("✅ Handles models without weights gracefully")

    print("✅ Helper functions work correctly!")

if __name__ == "__main__":
    test_unit_helper_functions()

# %% [markdown]
r"""
## 🏗️ Parameter Counting: Model Size Analysis

Parameter counting is the foundation of model profiling. Every parameter contributes directly to persistent memory footprint, initialization overhead, checkpoint storage size, and weight transfer latency across the PCIe/NVLink bus.

### Hardware Footprint Across Model Scales

$$\text{Memory}_{\text{FP32}} = P \times 4\text{ bytes}, \quad \text{Memory}_{\text{FP16}} = P \times 2\text{ bytes}, \quad \text{Memory}_{\text{INT8}} = P \times 1\text{ byte}$$

| Architecture Scale | Parameter Count ($P$) | FP32 Footprint ($4\text{ B}$) | FP16 / BF16 Footprint ($2\text{ B}$) | INT8 Footprint ($1\text{ B}$) | Target Serving Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-2 Small** | $124\text{M}$ | $496\text{ MB}$ | $248\text{ MB}$ | $124\text{ MB}$ | Edge device / CPU socket |
| **GPT-2 Medium** | $350\text{M}$ | $1.40\text{ GB}$ | $700\text{ MB}$ | $350\text{ MB}$ | Single consumer GPU ($4\text{ GB}$) |
| **GPT-2 Large** | $774\text{M}$ | $3.10\text{ GB}$ | $1.55\text{ GB}$ | $774\text{ MB}$ | Single consumer GPU ($8\text{ GB}$) |
| **GPT-2 XL** | $1,558\text{M}$ | $6.23\text{ GB}$ | $3.12\text{ GB}$ | $1.56\text{ GB}$ | Single workstation GPU ($16\text{ GB}$) |
"""

# %% [markdown]
r"""
### 🧪 Unit Test: Parameter Counting

This test validates our parameter counting works correctly for different model types.

**What we're testing**: Parameter counting accuracy for various architectures
**Why it matters**: Accurate parameter counts predict memory usage and model complexity
**Expected**: Correct counts for known model configurations
"""

# %% nbgrader={"grade": true, "grade_id": "test-parameter-counting", "locked": true, "points": 10}
def test_unit_parameter_counting():
    """🧪 Test parameter counting implementation."""
    print("🧪 Unit Test: Parameter Counting...")

    profiler = Profiler()

    # Test 1: Simple model with known parameters
    class SimpleModel:
        def __init__(self):
            self.weight = Tensor(rng.standard_normal((10, 5)))
            self.bias = Tensor(rng.standard_normal(5))

        def parameters(self):
            return [self.weight, self.bias]

    simple_model = SimpleModel()
    param_count = profiler.count_parameters(simple_model)
    expected_count = 10 * 5 + 5  # weight + bias
    assert param_count == expected_count, f"Expected {expected_count} parameters, got {param_count}"
    print(f"✅ Simple model: {param_count} parameters")

    # Test 2: Model without parameters
    class NoParamModel:
        def __init__(self):
            pass

    no_param_model = NoParamModel()
    param_count = profiler.count_parameters(no_param_model)
    assert param_count == 0, f"Expected 0 parameters, got {param_count}"
    print(f"✅ No parameter model: {param_count} parameters")

    # Test 3: Direct tensor (no parameters)
    test_tensor = Tensor(rng.standard_normal((2, 3)))
    param_count = profiler.count_parameters(test_tensor)
    assert param_count == 0, f"Expected 0 parameters for tensor, got {param_count}"
    print(f"✅ Direct tensor: {param_count} parameters")

    print("✅ Parameter counting works correctly!")

if __name__ == "__main__":
    test_unit_parameter_counting()

# %% [markdown]
r"""
## 🏗️ FLOP Counting: Computational Cost Estimation

FLOPs (Floating Point Operations) quantify theoretical arithmetic work independent of execution hardware, operating system, or compiler optimizations. Comparing theoretical FLOPs against measured execution time reveals whether an operation achieves high computational efficiency or stalls waiting on memory bandwidth.

### Mathematical FLOP Formulations

$$\text{FLOP}_{\text{Linear}} = 2 \cdot B \cdot d_{\text{in}} \cdot d_{\text{out}} + \underbrace{B \cdot d_{\text{out}}}_{\text{bias, dropped below}}$$

The bias term is written once for completeness and then ignored, by `_count_linear_flops`, by 📐, and by the table below. For the first row it is $32 \times 3,072 = 98,304$ FLOPs against $150,994,944$, or $0.065\%$, and it stays under a tenth of a percent for any layer worth profiling. Dropping a term you have bounded is different from forgetting it, and knowing which terms you may drop is most of what FLOP accounting is.

$$\text{FLOP}_{\text{Conv2d}} = 2 \cdot B \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot (C_{\text{in}} \cdot k_h \cdot k_w) \cdot C_{\text{out}}$$

| Workload Configuration | Mathematical Dimension | Total Arithmetic FLOPs | Per-Sample FLOPs | Arithmetic Reuse |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Layer** | $B=32, d_{\text{in}}=768, d_{\text{out}}=3072$ | $150,994,944\text{ FLOPs}$ | $4,718,592\text{ FLOPs}$ | $32\text{ MACs/param}$ |
| **Conv2d Layer** | $B=1, C=3 \to 64, 224 \to 112, k=7$ | $236,027,904\text{ FLOPs}$ | $236,027,904\text{ FLOPs}$ | $12,544\text{ MACs/param}$ |
| **GELU Non-linearity**| $B=32, S=128, D=768$ | $3,145,728\text{ FLOPs}$ | $98,304\text{ FLOPs}$ | $1\text{ FLOP/elem}$ (memory bound) |

### Algorithmic FLOP Counting Strategy

- **Dense Matrix Products**: $2 \cdot M \cdot K \cdot N$ floating point operations (fused multiply-accumulate).
- **Spatial Convolutions**: $2 \times$ output spatial elements $\times$ kernel footprint $\times$ input/output channels.
- **Pointwise Non-linearities**: Evaluated at 1 to 8 FLOPs per tensor element.
"""

# %% [markdown]
r"""
### 🧪 Unit Test: _count_sequential_flops

This test validates the helper that sums FLOPs across layers in a sequential model.

**What we're testing**: Accumulation of per-layer FLOPs with shape propagation
**Why it matters**: Real models are sequences of layers; total FLOPs = sum of per-layer FLOPs
**Expected**: Sum of individual layer FLOPs with correct shape propagation
"""

# %% nbgrader={"grade": true, "grade_id": "test-count-sequential-flops", "locked": true, "points": 3}
def test_unit_count_sequential_flops():
    """🧪 Test _count_sequential_flops helper."""
    print("🧪 Unit Test: _count_sequential_flops...")

    profiler = Profiler()

    # Create mock sequential model with two Linear layers
    class MockLinear:
        def __init__(self, in_f, out_f):
            self.weight = Tensor(rng.standard_normal((in_f, out_f)))
            self.__class__.__name__ = 'Linear'

    class MockSequential:
        def __init__(self, *layer_list):
            self.layers = list(layer_list)

    model = MockSequential(MockLinear(128, 64), MockLinear(64, 10))
    total_flops = profiler._count_sequential_flops(model, (1, 128))

    expected = (128 * 64 * 2) + (64 * 10 * 2)
    assert total_flops == expected, f"Expected {expected}, got {total_flops}"
    print(f"✅ Sequential(128->64->10): {total_flops} FLOPs")

    # Single layer sequential
    model_single = MockSequential(MockLinear(32, 16))
    flops_single = profiler._count_sequential_flops(model_single, (1, 32))
    assert flops_single == 32 * 16 * 2, f"Expected {32*16*2}, got {flops_single}"
    print(f"✅ Single-layer sequential: {flops_single} FLOPs")

    print("✅ _count_sequential_flops works correctly!")

if __name__ == "__main__":
    test_unit_count_sequential_flops()

# %% [markdown]
r"""
### 🧪 Unit Test: FLOP Counting

This test validates our FLOP counting for different operations and architectures.

**What we're testing**: FLOP calculation accuracy for various layer types
**Why it matters**: FLOPs predict computational cost and energy usage
**Expected**: Correct FLOP counts for known operation types
"""

# %% nbgrader={"grade": true, "grade_id": "test-flop-counting", "locked": true, "points": 10}
def test_unit_flop_counting():
    """🧪 Test FLOP counting implementation."""
    print("🧪 Unit Test: FLOP Counting...")

    profiler = Profiler()

    # Test 1: Simple tensor operations
    test_tensor = Tensor(rng.standard_normal((4, 8)))
    flops = profiler.count_flops(test_tensor, (4, 8))
    expected_flops = 8  # per sample, excluding the batch axis  # 1 FLOP per element for generic operation
    assert flops == expected_flops, f"Expected {expected_flops} FLOPs, got {flops}"
    print(f"✅ Tensor operation: {flops} FLOPs")

    # Test 2: Simulated Linear layer
    class MockLinear:
        def __init__(self, in_features, out_features):
            self.weight = Tensor(rng.standard_normal((in_features, out_features)))
            self.__class__.__name__ = 'Linear'

    mock_linear = MockLinear(128, 64)
    flops = profiler.count_flops(mock_linear, (1, 128))
    expected_flops = 128 * 64 * 2  # matmul FLOPs
    assert flops == expected_flops, f"Expected {expected_flops} FLOPs, got {flops}"
    print(f"✅ Linear layer: {flops} FLOPs")

    # Test 3: Batch size independence
    flops_batch1 = profiler.count_flops(mock_linear, (1, 128))
    flops_batch32 = profiler.count_flops(mock_linear, (32, 128))
    assert flops_batch1 == flops_batch32, "FLOPs should be independent of batch size"
    print(f"✅ Batch independence: {flops_batch1} FLOPs (same for batch 1 and 32)")

    print("✅ FLOP counting works correctly!")

if __name__ == "__main__":
    test_unit_flop_counting()

# %% [markdown]
r"""
## 🏗️ Memory Profiling: Understanding Memory Usage Patterns

Memory profiling reveals how much RAM and VRAM your model consumes during inference and training. In production deep learning, memory limits dictate allowable batch sizes, maximum context windows, and target serving hardware.

### Memory Allocation Pools & Scaling Dynamics

| Allocation Component | Lifecycle Duration | Dimension Dependency | Scaling Complexity | Systems Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **Parameters ($W$)** | Static / Persistent | Model parameter count $P$ | $O(1)$ w.r.t batch/sequence | Weight quantization (INT8/FP4), sharding (ZeRO) |
| **Activations ($A$)** | Stored forward, freed backward | Batch $B \times \text{Seq } S \times \text{Dim } d$ | $O(B \cdot S)$ (Linear) / $O(B \cdot S^2)$ (Attention) | Activation checkpointing, FlashAttention |
| **Gradients ($\nabla_W L$)** | Backward pass accumulation | Matches parameter count $P$ | $O(1)$ w.r.t batch/sequence | Gradient accumulation, mixed precision (FP16) |
| **Optimizer States** | Persistent across steps | Multiplier of parameter count | $2 \times P$ (Adam) vs $0 \times P$ (SGD) | 8-bit Adam (bitsandbytes), Adafactor |

### Memory Measurement Strategy

We use Python's standard `tracemalloc` to track peak allocation deltas during model execution. This yields byte-precise telemetry of memory consumption patterns across arbitrary tensor graphs.
"""

# %% [markdown]
r"""
### 🧪 Unit Test: _calculate_parameter_memory

This test validates the helper that converts parameter count to memory in MB.

**What we're testing**: Parameter count to megabytes conversion using FP32 (4 bytes each)
**Why it matters**: Memory budgets determine which hardware can run your model
**Expected**: Exact byte-level accuracy for known parameter counts
"""

# %% nbgrader={"grade": true, "grade_id": "test-calculate-parameter-memory", "locked": true, "points": 3}
def test_unit_calculate_parameter_memory():
    """🧪 Test _calculate_parameter_memory helper."""
    print("🧪 Unit Test: _calculate_parameter_memory...")

    profiler = Profiler()

    # Test 1: Known parameter count -> known memory
    class KnownModel:
        def __init__(self):
            # 1024 * 1024 = 1,048,576 parameters = exactly 4 MB at FP32
            self.weight = Tensor(rng.standard_normal((1024, 1024)))

    model = KnownModel()
    memory_mb = profiler._calculate_parameter_memory(model)
    expected_mb = (1024 * 1024 * 4) / (1024 * 1024)  # 4.0 MB
    assert abs(memory_mb - expected_mb) < 0.01, f"Expected {expected_mb} MB, got {memory_mb}"
    print(f"✅ 1M params = {memory_mb:.1f} MB")

    # Test 2: Zero parameter model
    class EmptyModel:
        pass

    empty_mb = profiler._calculate_parameter_memory(EmptyModel())
    assert empty_mb == 0.0, f"Expected 0.0 MB, got {empty_mb}"
    print("✅ Empty model = 0.0 MB")

    print("✅ _calculate_parameter_memory works correctly!")

if __name__ == "__main__":
    test_unit_calculate_parameter_memory()

# %% [markdown]
r"""
### 🧪 Unit Test: Memory Measurement

This test validates our memory tracking works correctly and provides useful metrics.

**What we're testing**: Memory usage measurement and calculation accuracy
**Why it matters**: Memory constraints often limit model deployment
**Expected**: Reasonable memory measurements with proper components
"""

# %% nbgrader={"grade": true, "grade_id": "test-memory-measurement", "locked": true, "points": 10}
def test_unit_memory_measurement():
    """🧪 Test memory measurement implementation."""
    print("🧪 Unit Test: Memory Measurement...")

    profiler = Profiler()

    # Test 1: Basic memory measurement
    test_tensor = Tensor(rng.standard_normal((10, 20)))
    test_model = Linear(20, 10)
    memory_stats = profiler.measure_memory(test_model, (10, 20))

    # Validate dictionary structure
    required_keys = ['parameter_memory_mb', 'activation_memory_mb', 'peak_memory_mb', 'memory_efficiency']
    for key in required_keys:
        assert key in memory_stats, f"Missing key: {key}"

    # Validate non-negative values
    for key in required_keys:
        assert memory_stats[key] >= 0, f"{key} should be non-negative, got {memory_stats[key]}"

    print(f"✅ Basic measurement: {memory_stats['peak_memory_mb']:.3f} MB peak")

    # Test 2: Memory scaling with size
    small_model = Linear(5, 5)
    large_model = Linear(50, 50)

    small_memory = profiler.measure_memory(small_model, (5, 5))
    large_memory = profiler.measure_memory(large_model, (50, 50))

    # Larger tensor should use more activation memory
    assert large_memory['activation_memory_mb'] >= small_memory['activation_memory_mb'], \
        "Larger tensor should use more activation memory"

    print(f"✅ Scaling: Small {small_memory['activation_memory_mb']:.3f} MB → Large {large_memory['activation_memory_mb']:.3f} MB")

    # Test 3: Efficiency bounds
    assert 0 <= memory_stats['memory_efficiency'] <= 1.0, \
        f"Memory efficiency should be between 0 and 1, got {memory_stats['memory_efficiency']}"

    print(f"✅ Efficiency: {memory_stats['memory_efficiency']:.3f} (0-1 range)")

    print("✅ Memory measurement works correctly!")

if __name__ == "__main__":
    test_unit_memory_measurement()

# %% [markdown]
r"""
## 🏗️ Latency Measurement: Accurate Performance Timing

Latency measurement is the most challenging facet of profiling because execution time is perturbed by CPU thread scheduling, hardware cache misses, memory bus contention, and thermal frequency governor transitions. Statistical rigor is essential to extract reproducible metrics.

### Systemic Latency Variance & Experimental Safeguards

| Variance Source | Physical / OS Mechanism | Impact on Raw Latency | Experimental Safeguard |
| :--- | :--- | :--- | :--- |
| **Cold Start Costs** | Cold L1/L2/L3 lines, BLAS thread-pool spin-up, first-touch page faults | First runs take $5\times \text{ to } 20\times$ longer | Execute 3–10 warmup passes before recording timestamps |
| **OS Thread Scheduling**| Kernel interrupts, context switches, daemons | Sporadic high-latency spikes in single runs | Report sample median ($Q_2$) rather than arithmetic mean |
| **Thermal Throttling** | Dynamic frequency scaling (DVFS) under sustained load | Late runs slow down as junction temperature rises | Benchmark in short, burst-controlled sample batches |
| **Garbage Collection** | Python runtime stop-the-world heap sweeps | Unpredictable multi-millisecond halts | Synchronize explicit `gc.collect()` passes outside timing blocks |

### Measurement Protocol

Our `measure_latency` implements the first three stages below. The fourth is in the
table above because it belongs in any serious harness, not because this one does it.
`measure_latency` never collects, and TinyTorch does not even import `gc`.
1. **Warmup Passes**: Execute un-timed forward iterations to warm hardware caches and trigger any lazy initialization.
2. **Repeated Measurements**: Collect multiple steady-state iterations to form an empirical timing distribution.
3. **Median Reduction**: Compute the median ($50^{\text{th}}$ percentile) to discard asymmetrical OS context-switching outliers.
4. **Memory Hygiene (not implemented here)**: A production harness collects garbage between sweeps and reports the interquartile range beside the median. Module 19 will build that harness.
"""

# %% [markdown]
r"""
### 🧪 Unit Test: Latency Measurement

This test validates our latency measurement provides consistent and reasonable results.

**What we're testing**: Timing accuracy and statistical robustness
**Why it matters**: Latency determines real-world deployment feasibility
**Expected**: Consistent timing measurements with proper statistical handling
"""

# %% nbgrader={"grade": true, "grade_id": "test-latency-measurement", "locked": true, "points": 10}
def test_unit_latency_measurement():
    """🧪 Test latency measurement implementation."""
    print("🧪 Unit Test: Latency Measurement...")

    profiler = Profiler()

    # Test 1: Basic latency measurement
    test_model = Linear(8, 4)
    test_input = Tensor(rng.standard_normal((4, 8)))
    latency = profiler.measure_latency(test_model, test_input, warmup=2, iterations=5)

    assert latency >= 0, f"Latency should be non-negative, got {latency}"
    assert latency < 1000, f"Latency seems too high for simple operation: {latency} ms"
    print(f"✅ Basic latency: {latency:.3f} ms")

    # Test 2: Measurement consistency
    latencies = []
    for _ in range(3):
        lat = profiler.measure_latency(test_model, test_input, warmup=1, iterations=3)
        latencies.append(lat)

    # Measurements should be in reasonable range
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    assert np.all(np.isfinite(latencies)), "Timing samples must be finite"
    print(f"✅ Consistency: {avg_latency:.3f} ± {std_latency:.3f} ms")

    # Test 3: Size scaling
    small_model = Linear(2, 2)
    large_model = Linear(20, 20)
    small_input = Tensor(rng.standard_normal((2, 2)))
    large_input = Tensor(rng.standard_normal((20, 20)))

    small_latency = profiler.measure_latency(small_model, small_input, warmup=1, iterations=3)
    large_latency = profiler.measure_latency(large_model, large_input, warmup=1, iterations=3)

    # Larger operations might take longer (though not guaranteed for simple operations)
    print(f"✅ Scaling: Small {small_latency:.3f} ms, Large {large_latency:.3f} ms")

    print("✅ Latency measurement works correctly!")

if __name__ == "__main__":
    test_unit_latency_measurement()

# %% [markdown]
r"""
## 🔧 Integration: Advanced Profiling Functions

Now let's validate our higher-level profiling functions that combine core measurements into comprehensive analysis tools.

### Advanced Profiling Architecture

```
Core Profiler Methods → Advanced Analysis Functions → Optimization Insights
        ↓                         ↓                         ↓
count_parameters()      profile_forward_pass()      "Memory-bound workload"
count_flops()          profile_backward_pass()      "Optimize data movement"
measure_memory()       profile_layer()              "Focus on bandwidth"
measure_latency()      quick_profile()              "Use quantization"
```

### Forward Pass Profiling: Complete Performance Picture

A forward pass profile combines all our measurements to understand model behavior comprehensively. This is essential for optimization decisions.
"""

# %% [markdown]
r"""
### Backward Pass Profiling: Training Analysis

Training deep neural networks requires executing both forward inference and reverse-mode automatic differentiation. The backward pass requires approximately **$2\times$ the compute** of the forward pass (computing input activations gradients $\nabla_X L$ and weight parameter gradients $\nabla_W L$) and introduces massive memory residency requirements for gradients and optimizer states.

### Training Memory Expansion Dynamics

$$\text{Compute}_{\text{backward}} \approx 2 \times \text{Compute}_{\text{forward}}$$

$$\text{Memory}_{\text{training}} = M_{\text{params}} + M_{\text{activations}} + M_{\text{gradients}} + M_{\text{optimizer}}$$

| Training Phase | Resident Memory Pools | Analytical Formula (FP32) | Example Footprint ($125\text{M}$ GPT) |
| :--- | :--- | :--- | :--- |
| **Forward Pass** | Model Weights ($W$) + Saved Activations ($A$) | $P \times 4\text{ B} + B \cdot S \cdot d \cdot c_{\text{act}} \times 4\text{ B}$ | $500\text{ MB} + 200\text{ MB} = 700\text{ MB}$ |
| **Backward Pass** | Weights ($W$) + Activations ($A$) + Gradients ($\nabla_W L$) | $P \times 4\text{ B} + A + P \times 4\text{ B}$ | $500\text{ MB} + 200\text{ MB} + 500\text{ MB} = 1,200\text{ MB}$ |
| **Adam Optimizer Step** | Weights ($W$) + Gradients ($\nabla_W L$) + First & Second Moments ($m, v$) | $P \times 4\text{ B} + P \times 4\text{ B} + 2 \times P \times 4\text{ B}$ | $500\text{ MB} + 500\text{ MB} + 1,000\text{ MB} = 2,000\text{ MB}$ |
| **Peak Resident Footprint**| All concurrent buffers before gradient zeroing | $16P + A$, with $P$ the parameter count | **$2,200\text{ MB}$ ($4.4\times$ model weight size)** |
"""

# %% [markdown]
r"""
### 🧪 Unit Test: Advanced Profiling Functions

This test validates our advanced profiling functions provide comprehensive analysis.

**What we're testing**: Forward and backward pass profiling completeness
**Why it matters**: Training optimization requires understanding both passes
**Expected**: Complete profiles with all required metrics and relationships
"""

# %% nbgrader={"grade": true, "grade_id": "test-advanced-profiling", "locked": true, "points": 15}
def test_unit_advanced_profiling():
    """🧪 Test advanced profiling functions."""
    print("🧪 Unit Test: Advanced Profiling Functions...")

    # Create profiler and test model
    profiler = Profiler()
    test_model = Linear(8, 4)
    test_input = Tensor(rng.standard_normal((4, 8)))

    # Test forward pass profiling
    forward_profile = profiler.profile_forward_pass(test_model, test_input)

    # Validate forward profile structure
    required_forward_keys = [
        'parameters', 'flops', 'latency_ms', 'gflops_per_second',
        'memory_bandwidth_mbs', 'bottleneck'
    ]

    for key in required_forward_keys:
        assert key in forward_profile, f"Missing key: {key}"

    assert forward_profile['parameters'] >= 0
    assert forward_profile['flops'] >= 0
    assert forward_profile['latency_ms'] >= 0
    assert forward_profile['gflops_per_second'] >= 0

    print(f"✅ Forward profiling: {forward_profile['gflops_per_second']:.2f} GFLOP/s")

    # Test backward pass profiling
    backward_profile = profiler.profile_backward_pass(test_model, test_input)

    # Validate backward profile structure
    required_backward_keys = [
        'forward_flops', 'backward_flops', 'total_flops',
        'total_latency_ms', 'total_memory_mb', 'optimizer_memory_estimates'
    ]

    for key in required_backward_keys:
        assert key in backward_profile, f"Missing key: {key}"

    # Validate relationships
    assert backward_profile['total_flops'] >= backward_profile['forward_flops']
    assert backward_profile['total_latency_ms'] >= backward_profile['forward_latency_ms']
    assert 'sgd' in backward_profile['optimizer_memory_estimates']
    assert 'adam' in backward_profile['optimizer_memory_estimates']

    # Check backward pass estimates are reasonable
    assert backward_profile['backward_flops'] >= backward_profile['forward_flops'], \
        "Backward pass should have at least as many FLOPs as forward"
    assert backward_profile['gradient_memory_mb'] >= 0, \
        "Gradient memory should be non-negative"

    print(f"✅ Backward profiling: {backward_profile['total_latency_ms']:.2f} ms total")
    print(f"✅ Memory breakdown: {backward_profile['total_memory_mb']:.2f} MB training")
    print("✅ Advanced profiling functions work correctly!")

if __name__ == "__main__":
    test_unit_advanced_profiling()

# %% [markdown]
r"""
## 📊 Systems Analysis: Understanding Performance Characteristics

Model profiling reveals the empirical performance characteristics of deep learning architectures across dimensions of model scale, batch size, and arithmetic intensity.

The roofline, the ridge point, and the two regimes were defined in 📐 Foundations,
before any of this module's code. What follows is the measurement side of that
picture. We sweep model size and batch size, time real forward passes, and read the
numbers against those ceilings. Watch for the places where the measurement refuses to
behave like the model predicts, because those are the places the profiler is telling
you something about this implementation rather than about the hardware.

<div align="center">
  <img src="weight_streaming_vs_reuse.svg" alt="Weight Streaming vs Cache Reuse in Autoregressive Decode" width="680px">
</div>
"""

# %% nbgrader={"grade": false, "grade_id": "performance_analysis", "solution": false}
def analyze_model_scaling():
    """📊 Analyze how model performance scales with size."""
    print("📊 Analyzing Model Scaling Characteristics...")

    profiler = Profiler()
    results = []

    # Test different model sizes
    sizes = [64, 128, 256, 512]

    print("\nModel Scaling Analysis:")
    print("Size\tParams\t\tFLOPs\t\tLatency(ms)\tMemory(MB)\tGFLOP/s")
    print("-" * 80)

    for size in sizes:
        # Create models of different sizes for comparison
        test_model = Linear(size, size)
        input_shape = (32, size)  # Batch of 32
        dummy_input = Tensor(rng.standard_normal(input_shape))

        # Use the profiler's own counters (count_flops is per sample; scale by batch)
        linear_params = profiler.count_parameters(test_model)
        linear_flops = profiler.count_flops(test_model, input_shape) * input_shape[0]

        # Measure actual performance
        latency = profiler.measure_latency(test_model, dummy_input, warmup=3, iterations=10)
        memory = profiler.measure_memory(test_model, input_shape)

        gflops_per_second = (linear_flops / 1e9) / (latency / 1000)

        results.append({
            'size': size,
            'parameters': linear_params,
            'flops': linear_flops,
            'latency_ms': latency,
            'memory_mb': memory['peak_memory_mb'],
            'gflops_per_second': gflops_per_second
        })

        print(f"{size}\t{linear_params:,}\t\t{linear_flops:,}\t\t"
              f"{latency:.2f}\t\t{memory['peak_memory_mb']:.2f}\t\t"
              f"{gflops_per_second:.2f}")

    # Analysis insights
    print("\n💡 Scaling Analysis Insights:")

    # Memory scaling
    memory_growth = results[-1]['memory_mb'] / max(results[0]['memory_mb'], 0.001)
    print(f"Memory grows {memory_growth:.1f}× from {sizes[0]} to {sizes[-1]} size")

    # Compute scaling
    compute_growth = results[-1]['gflops_per_second'] / max(results[0]['gflops_per_second'], 0.001)
    print(f"Compute efficiency changes {compute_growth:.1f}× with size")

    # Performance characteristics
    avg_efficiency = np.mean([r['gflops_per_second'] for r in results])
    print(f"Average throughput across these four sizes: {avg_efficiency:.2f} GFLOP/s")
    print("🚀 A couple of GFLOP/s from a NumPy Linear is interpreter and framework")
    print("   dispatch cost, not DRAM bandwidth. Calling it memory-bound would need")
    print("   measured bytes moved, which this profiler never sees.")

def analyze_batch_size_effects():
    """📊 Analyze how batch size affects performance and efficiency."""
    print("\n📊 Analyzing Batch Size Effects...")

    profiler = Profiler()
    batch_sizes = [1, 8, 32, 128]
    feature_size = 256

    print("\nBatch Size Effects Analysis:")
    print("Batch\tLatency(ms)\tThroughput(samples/s)\tMemory(MB)\tSamples/s per MB")
    print("-" * 85)

    throughputs = []
    for batch_size in batch_sizes:
        test_model = Linear(feature_size, feature_size)
        input_shape = (batch_size, feature_size)
        dummy_input = Tensor(rng.standard_normal(input_shape))

        # Measure performance
        latency = profiler.measure_latency(test_model, dummy_input, warmup=3, iterations=10)
        memory = profiler.measure_memory(test_model, input_shape)

        # Calculate throughput
        samples_per_second = (batch_size * 1000) / latency  # samples/second
        throughputs.append(samples_per_second)

        # Samples per second per MB of peak memory. Not an efficiency in any
        # hardware sense, just throughput divided by footprint.
        samples_per_mb = samples_per_second / max(memory['peak_memory_mb'], 0.001)

        print(f"{batch_size}\t{latency:.2f}\t\t{samples_per_second:.0f}\t\t\t"
              f"{memory['peak_memory_mb']:.2f}\t\t{samples_per_mb:.1f}")

    print("\n💡 Batch Size Insights:")
    print(f"Throughput went {throughputs[0]:.0f} -> {throughputs[-1]:.0f} samples/s from "
          f"batch {batch_sizes[0]} to {batch_sizes[-1]}, a factor of "
          f"{throughputs[-1] / max(throughputs[0], 1e-9):.2f}.")
    print("Batching cannot buy throughput here, whatever this run happened to show.")
    print("TinyTorch's 2D matmul is an explicit Python loop over output elements, so it")
    print("runs one iteration per sample. Per-sample cost is constant, total cost is")
    print("exactly linear in batch, leaving nothing to amortize. A vectorized kernel")
    print("would instead show the factor climb as one weight load serves many samples,")
    print("which is what Module 17 adds. Peak memory grows with batch either way, so the")
    print("last column falls no matter which way the throughput wanders.")

if __name__ == "__main__":
    analyze_model_scaling()
    analyze_batch_size_effects()

# %% [markdown]
r"""
### Optimization Insights: Production Performance Patterns

Profiling results guide targeted systems interventions. Different machine learning operations inhabit fundamentally different regimes on the roofline curve, demanding distinct optimization strategies.

### Operational Characteristics & Optimization Taxonomy

| Operation Class | Dominant Workload | Arithmetic Intensity ($I$) | Primary Hardware Bound | High-Leverage Optimization Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Matrix Multiplications (GEMM)** | Linear projections, MLP up/down | High ($I \gg I_{\text{ridge}}$ at large $B$) | Compute (ALU / Tensor Core) | Optimized BLAS kernels, Tensor Core MMA, loop tiling |
| **Spatial Convolutions** | Conv2d feature extraction | High ($I \gg I_{\text{ridge}}$) | Compute (ALU) | Im2col GEMM transforms, Winograd minimal filtering |
| **Self-Attention ($QK^\top, SV$)** | Dynamic relational routing | Variable ($I \propto \text{Seq Len}$) | Memory bandwidth (SRAM transfers) | FlashAttention (online softmax tiling in SRAM) |
| **Pointwise Elements** | GELU, ReLU, residual additions | Very low ($I \le 1\text{ FLOP/B}$) | Memory bandwidth | Kernel fusion (fused pointwise passes) |
| **Reductions & Normalizations** | LayerNorm, Softmax, sum pooling | Very low ($I \le 1\text{ FLOP/B}$) | Memory bandwidth | Fused two-pass reduction kernels, Welford's algorithm |

### Systematic Optimization Strategy

1. **Profile First**: Establish baseline latency, peak memory, and GFLOP/s to identify the high-overhead stages.
2. **Compute-Bound Operations**: Address with precision reduction (FP32 $\to$ FP16/BF16/FP8) and hardware-specialized Tensor Core instructions.
3. **Memory-Bound Operations**: Address with data movement minimization, activation recomputation, KV cache reuse, and kernel fusion.
4. **Iterative Verification**: Re-profile after each code change to confirm speedup on the empirical roofline curve.
"""

# %% nbgrader={"grade": false, "grade_id": "optimization_insights", "solution": false}
def benchmark_operation_efficiency():
    """📊 Compare efficiency of different operations for optimization guidance."""
    print("📊 Benchmarking Operation Efficiency...")
    print("Efficiency classes below are illustrative hypotheses, not measured diagnoses.")

    profiler = Profiler()
    operations = []

    # Test different operation types
    size = 256
    input_tensor = Tensor(rng.standard_normal((32, size)))

    # Elementwise operations (memory-bound)
    # Create a simple model wrapper for elementwise operations
    class ElementwiseModel:
        def forward(self, x):
            return x + x  # Simple elementwise operation

    elementwise_model = ElementwiseModel()
    elementwise_latency = profiler.measure_latency(elementwise_model, input_tensor, iterations=20)
    elementwise_flops = size * 32  # One operation per element

    operations.append({
        'operation': 'Elementwise',
        'latency_ms': elementwise_latency,
        'flops': elementwise_flops,
        'gflops_per_second': (elementwise_flops / 1e9) / (elementwise_latency / 1000),
        'efficiency_class': 'memory-bound',
        'optimization_focus': 'data_locality'
    })

    # Matrix operations (compute-bound)
    matrix_model = Linear(size, size)
    matrix_latency = profiler.measure_latency(matrix_model, input_tensor, iterations=10)
    matrix_flops = profiler.count_flops(matrix_model, input_tensor.shape) * input_tensor.shape[0]

    operations.append({
        'operation': 'Matrix Multiply',
        'latency_ms': matrix_latency,
        'flops': matrix_flops,
        'gflops_per_second': (matrix_flops / 1e9) / (matrix_latency / 1000),
        'efficiency_class': 'compute-bound',
        'optimization_focus': 'algorithms'
    })

    # Reduction operations (memory-bound)
    class ReductionModel:
        def forward(self, x):
            return x.sum()  # Sum reduction operation

    reduction_model = ReductionModel()
    reduction_latency = profiler.measure_latency(reduction_model, input_tensor, iterations=20)
    reduction_flops = size * 32  # Sum reduction

    operations.append({
        'operation': 'Reduction',
        'latency_ms': reduction_latency,
        'flops': reduction_flops,
        'gflops_per_second': (reduction_flops / 1e9) / (reduction_latency / 1000),
        'efficiency_class': 'memory-bound',
        'optimization_focus': 'parallelization'
    })

    print("\nOperation Efficiency Comparison:")
    print("Operation\t\tLatency(ms)\tGFLOP/s\t\tExpected Class (a priori)\tOptimization Focus")
    print("-" * 95)

    for op in operations:
        print(f"{op['operation']:<15}\t{op['latency_ms']:.3f}\t\t"
              f"{op['gflops_per_second']:.2f}\t\t{op['efficiency_class']:<15}\t{op['optimization_focus']}")

    print("\n💡 Operation Optimization Insights:")

    # Find most and least efficient
    best_op = max(operations, key=lambda x: x['gflops_per_second'])
    worst_op = min(operations, key=lambda x: x['gflops_per_second'])

    print(f"Most efficient: {best_op['operation']} ({best_op['gflops_per_second']:.2f} GFLOP/s)")
    print(f"Least efficient: {worst_op['operation']} ({worst_op['gflops_per_second']:.2f} GFLOP/s)")
    print("Note the measurement inverts the a priori labels. The GEMM is labeled")
    print("compute-bound and still comes last, because TinyTorch's matmul runs in")
    print("Python, so it never reaches the regime the label describes.")

    # Count operation types. This counts the LABELS above, which were written by
    # hand before anything ran, so the priority below is a hypothesis about the
    # hardware and not a reading of the three timings.
    memory_bound_ops = [op for op in operations if op['efficiency_class'] == 'memory-bound']
    compute_bound_ops = [op for op in operations if op['efficiency_class'] == 'compute-bound']

    print(f"\n🚀 Optimization Priority (from the a priori labels, not the timings):")
    if len(memory_bound_ops) > len(compute_bound_ops):
        print("Focus on memory optimization: data locality, bandwidth, caching")
    else:
        print("Focus on compute optimization: better algorithms, vectorization")

def analyze_profiling_overhead():
    """📊 Measure the overhead of profiling itself."""
    print("\n📊 Analyzing Profiling Overhead...")

    # Test with and without profiling
    test_tensor = Tensor(rng.standard_normal((100, 100)))
    iterations = 50

    class TestModel:
        def forward(self, x):
            return x + 1.0

    test_model = TestModel()

    # Without profiling - the same forward call, timed as one block
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = test_model.forward(test_tensor)
    end_time = time.perf_counter()
    baseline_ms = (end_time - start_time) * 1000

    # With profiling - the same call through measure_latency. warmup=0 matters:
    # a warmup pass is a second forward, so warmup=1 would time 2x the work and
    # report the extra forward as instrumentation cost.
    profiler = Profiler()
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = profiler.measure_latency(test_model, test_tensor, warmup=0, iterations=1)
    end_time = time.perf_counter()
    profiled_ms = (end_time - start_time) * 1000

    overhead_factor = profiled_ms / max(baseline_ms, 0.001)

    print(f"\nProfiling Overhead Analysis:")
    print(f"Baseline execution: {baseline_ms:.2f} ms")
    print(f"With profiling: {profiled_ms:.2f} ms")
    print(f"Profiling overhead: {overhead_factor:.1f}× slower")

    print(f"\n💡 Profiling Overhead Insights:")
    if overhead_factor < 2:
        print("Low overhead - suitable for frequent profiling")
    elif overhead_factor < 10:
        print("Moderate overhead - use for development and debugging")
    else:
        print("High overhead - use sparingly in production")

if __name__ == "__main__":
    benchmark_operation_efficiency()
    analyze_profiling_overhead()

# %% [markdown]
r"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire profiling module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests (helpers first, then composition functions)
    print("Running unit tests (helpers first, then compositions)...")
    test_unit_count_layer_parameters()
    test_unit_count_linear_flops()
    test_unit_count_conv_flops()
    test_unit_count_sequential_flops()
    test_unit_calculate_parameter_memory()
    test_unit_calculate_memory_efficiency()
    test_unit_compute_derived_metrics()
    test_unit_arithmetic_intensity()
    test_unit_analyze_bottleneck()
    test_unit_estimate_backward_costs()
    test_unit_estimate_optimizer_memory()

    print("\nRunning composition unit tests...")
    test_unit_helper_functions()
    test_unit_parameter_counting()
    test_unit_flop_counting()
    test_unit_memory_measurement()
    test_unit_latency_measurement()
    test_unit_advanced_profiling()

    print("\nRunning integration scenarios...")

    # Test realistic usage patterns
    print("🧪 Integration Test: Complete Profiling Workflow...")

    # Create profiler
    profiler = Profiler()

    # Create test model and data
    test_model = Linear(16, 32)
    test_input = Tensor(rng.standard_normal((8, 16)))

    # Run complete profiling workflow
    print("1. Measuring model characteristics...")
    params = profiler.count_parameters(test_model)
    flops = profiler.count_flops(test_model, test_input.shape)
    memory = profiler.measure_memory(test_model, test_input.shape)
    latency = profiler.measure_latency(test_model, test_input, warmup=2, iterations=5)

    print(f"   Parameters: {params}")
    print(f"   FLOPs: {flops}")
    print(f"   Memory: {memory['peak_memory_mb']:.2f} MB")
    print(f"   Latency: {latency:.2f} ms")

    # Test advanced profiling
    print("2. Running advanced profiling...")
    forward_profile = profiler.profile_forward_pass(test_model, test_input)
    backward_profile = profiler.profile_backward_pass(test_model, test_input)

    assert 'gflops_per_second' in forward_profile
    assert 'total_latency_ms' in backward_profile
    print(f"   Forward GFLOP/s: {forward_profile['gflops_per_second']:.2f}")
    print(f"   Training latency: {backward_profile['total_latency_ms']:.2f} ms")

    # Test bottleneck analysis
    print("3. Analyzing performance bottlenecks...")
    bottleneck = forward_profile['bottleneck']
    efficiency = forward_profile['computational_efficiency']
    print(f"   Bottleneck: {bottleneck}")
    print(f"   Compute efficiency: {efficiency:.3f}")

    # Validate end-to-end workflow
    assert params >= 0, "Parameter count should be non-negative"
    assert flops >= 0, "FLOP count should be non-negative"
    assert memory['peak_memory_mb'] >= 0, "Memory usage should be non-negative"
    assert latency >= 0, "Latency should be non-negative"
    assert forward_profile['gflops_per_second'] >= 0, "GFLOP/s should be non-negative"
    assert backward_profile['total_latency_ms'] >= 0, "Total latency should be non-negative"
    assert bottleneck in ['memory', 'compute'], "Bottleneck should be memory or compute"
    assert 0 <= efficiency <= 1, "Efficiency should be between 0 and 1"

    print("✅ End-to-end profiling workflow works!")

    # Test production-like scenario
    print("4. Testing production profiling scenario...")

    # Simulate larger model analysis
    large_model = Linear(512, 256)
    large_input = Tensor(rng.standard_normal((32, 512)))  # Larger model input
    large_profile = profiler.profile_forward_pass(large_model, large_input)

    # Verify profile contains optimization insights
    assert 'bottleneck' in large_profile, "Profile should identify bottlenecks"
    assert 'memory_bandwidth_mbs' in large_profile, "Profile should measure memory bandwidth"

    print(f"   Large model analysis: {large_profile['bottleneck']} bottleneck")
    print(f"   Memory bandwidth: {large_profile['memory_bandwidth_mbs']:.1f} MB/s")

    print("✅ Production profiling scenario works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 14")

# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

### Question 1: FLOP Analysis & Hardware Projection

**Scenario**: You profile a `Linear(1000, 500)` projection layer with input dimension $d_{\text{in}} = 1,000$ and output dimension $d_{\text{out}} = 500$, with weights $W \in \mathbb{R}^{1000 \times 500}$ and bias $b \in \mathbb{R}^{500}$.

#### 1. Forward Pass Arithmetic Work (Single Sample $B=1$):
- **Matrix Multiplication**: Each output feature requires an inner dot product of length $d_{\text{in}}$: $1,000$ multiplications and $999$ additions $\approx 2 \cdot d_{\text{in}} = 2,000$ floating-point operations. Across all $500$ output features:
  $$\text{FLOP}_{\text{matmul}} = 2 \cdot d_{\text{in}} \cdot d_{\text{out}} = 2 \times 1,000 \times 500 = 1,000,000\text{ FLOPs } (1.00\text{ MFLOP})$$
- **Bias Addition**: Adding $b \in \mathbb{R}^{500}$ requires $500$ additions:
  $$\text{FLOP}_{\text{total}} = 1,000,000 + 500 = 1,000,500\text{ FLOPs } (1.0005\text{ MFLOP})$$

#### 2. Scaling to Batch Size $B=32$:
- **Total Work**: $\text{FLOP}_{\text{batch}} = 32 \times 1,000,500 = 32,016,000\text{ FLOPs } (32.016\text{ MFLOPs})$.
- **Per-Sample FLOP Invariant**: Per-sample FLOPs remain exactly constant at $1,000,500\text{ FLOPs/sample}$.
- **Systems Consequence**: Although per-sample compute is invariant, **arithmetic reuse increases $32\times$**. At $B=1$, the $500,000$ parameter weights ($2.0\text{ MB}$ at FP32) must be streamed from DRAM for only $1.0\text{ MFLOP}$ ($I \approx 0.5\text{ FLOP/B}$). At $B=32$, the weights are loaded once into high-speed L1/SRAM and reused across 32 tokens ($I \approx 16\text{ FLOP/B}$), shifting the kernel toward the compute-bound roofline regime.

#### 3. Cross-Hardware Latency Projection:
Theoretical lower-bound execution latency is predicted by dividing FLOP count by device peak arithmetic throughput:

$$T_{\text{compute\_bound}} = \frac{\text{FLOPs}}{P_{\text{peak}}}$$

| Hardware Device | Peak FP32 Throughput ($P_{\text{peak}}$) | Memory Bandwidth ($\text{BW}$) | Ridge Point ($I_{\text{ridge}}$) | Theoretical Compute Latency | Hardware Regime at $I = 16$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intel Xeon CPU** | $1.5\text{ TFLOP/s}$ | $100\text{ GB/s}$ | $15.0\text{ FLOP/B}$ | $21.3\ \mu\text{s}$ | Compute-bound (only just) |
| **NVIDIA A100 GPU** | $19.5\text{ TFLOP/s}$ (FP32) | $2,039\text{ GB/s}$ (HBM2e) | $9.6\text{ FLOP/B}$ | $1.64\ \mu\text{s}$ | Compute-bound |
| **NVIDIA H100 SXM GPU** | $67.0\text{ TFLOP/s}$ (FP32) | $3,350\text{ GB/s}$ (HBM3) | $20.0\text{ FLOP/B}$ | $0.48\ \mu\text{s}$ | Memory-bound |

Read the last two columns together, because they disagree on purpose. The H100 has the
shortest compute time and is the one machine here that cannot reach it. Its FP32 ALUs
outran its memory system, pushing the ridge to $20\text{ FLOP/B}$, so a workload at
$I = 16$ starves. The A100, slower on paper, has a ridge of $9.6$ and runs the same
kernel compute-bound. A faster device does not move a workload toward compute-bound; it
moves the ridge point to the right and can strand a kernel that used to be fine. The
Xeon clears its own ridge by a hair, which is a reminder that $I$ is a property of the
kernel and the regime is a property of the pairing. (The PCIe H100 is rated at
$51\text{ TFLOP/s}$ FP32, giving a ridge of $15.2$, which would flip this row.)

---

### Question 2: Memory Scaling & Precision Economics

**Scenario**: A 125M parameter transformer ($P = 125 \times 10^6$) trained at batch size $B=16$.

#### 1. Minimum Gradient Memory:
In reverse-mode automatic differentiation, every learnable weight and bias requires an accumulator tensor $\nabla_W L$ of identical dimensions:

$$\text{Memory}_{\text{grad}} = P \times 4\text{ bytes} = 125 \times 10^6 \times 4\text{ B} = 500,000,000\text{ B} \approx 500\text{ MB}$$

#### 2. Adam Optimizer State & Total Training Budget:
The standard Adam/AdamW optimizer maintains two full-precision float32 state tensors per parameter:
1. First moment vector $m_t$ (exponential moving average of gradients): $P \times 4\text{ B} = 500\text{ MB}$.
2. Second moment vector $v_t$ (exponential moving average of squared gradients): $P \times 4\text{ B} = 500\text{ MB}$.

$$\text{Memory}_{\text{Adam}} = 2 \times P \times 4\text{ B} = 1,000\text{ MB} = 1.0\text{ GB}$$

Assuming activation memory $A \approx 200\text{ MB}$ at $B=16$:

$$\text{Total Training VRAM} = W (500\text{ MB}) + A (200\text{ MB}) + \nabla W (500\text{ MB}) + \text{Adam } (1,000\text{ MB}) = \mathbf{2,200\text{ MB}} \quad (4.4\times \text{ Model Size})$$

#### 3. Mixed Precision (FP16 / BF16 AMP) Impact (a preview, not a review):
Everything in this part is new. Automatic Mixed Precision, FP32 master weights, gradient
underflow, and Tensor Cores appear nowhere earlier in the course, and Module 15 will build
the precision machinery that makes them make sense. What carries over from this
module is only the budget arithmetic, so read the list as the same four pools priced at
two different byte widths. Automatic Mixed Precision keeps the compute copies of the
weights in 16 bits while holding one 32-bit copy, the master weights, for the optimizer
to update, because repeatedly adding tiny updates to a 16-bit number loses them.

In Automatic Mixed Precision (AMP):
- Forward/backward model weights: $P \times 2\text{ B} = 250\text{ MB}$ (50% reduction).
- Backward gradients: $P \times 2\text{ B} = 250\text{ MB}$ (50% reduction).
- Activation caches: $A \times 0.5 \approx 100\text{ MB}$ (50% reduction).
- Master weights (FP32 for numerical stability against underflow): $P \times 4\text{ B} = 500\text{ MB}$.
- Adam optimizer states (FP32 moments): $2 \times P \times 4\text{ B} = 1,000\text{ MB}$.
- **Net Static Footprint**: $250 + 250 + 100 + 500 + 1,000 = \mathbf{2,100\text{ MB}}$.
- **Key Takeaway**: While static weight memory savings are modest in standard AMP due to FP32 master weights and Adam states, **activation memory drops by 50%**, enabling $2\times$ larger batch sizes or $2\times$ longer context windows, while Tensor Cores deliver up to $4\times$ throughput speedup.

---

### Question 3: Performance Bottlenecks & The Roofline Bound

**Scenario**: A model kernel achieves $10\text{ GFLOP/s}$ on hardware rated at $100\text{ GFLOP/s}$ peak. Doubling batch size ($B \to 2B$) yields zero change in achieved GFLOP/s.

#### 1. Computational Efficiency:
$$\eta = \frac{P_{\text{achieved}}}{P_{\text{peak}}} = \frac{10\text{ GFLOP/s}}{100\text{ GFLOP/s}} = 10.0\%$$
The hardware compute units are active for only 10% of total elapsed execution time; the system is severely bottlenecked.

#### 2. Root Cause Diagnosis:
Because doubling batch size does **not** improve arithmetic throughput, the workload is **not** simply under-batched, and it is not short of parallel work to fill the device. Note what this does not rule out. Its arithmetic intensity may still be far below the ridge point, and step 2 below raises it; batching is only one of the ways to do that, and here it is the one that has already failed. The root cause is:
- **Memory Bandwidth Saturation**: The memory bus between DRAM and processor is already saturated at 100% capacity ($\text{BW}_{\text{achieved}} \approx \text{BW}_{\text{peak}}$). Increasing batch size simply scales compute and memory transfers proportionally, locking throughput at the memory bandwidth ceiling.
- **Alternative Contributor (Small Sizes)**: High per-call runtime overhead (Python interpreter dispatch latency or asynchronous kernel launch overhead) dominating wall-clock time.

#### 3. Profiling-Guided Optimization Strategy:
1. **Refuse to write micro-kernel GEMM optimizations**: Optimizing matrix multiply algorithms will yield zero speedup because the ALU is already idle waiting on data.
2. **Apply Quantization**: Convert FP32 weights to INT8. Cutting data bus transfers by $4\times$ at unchanged FLOPs multiplies arithmetic intensity by $4\times$, which in a bandwidth-saturated regime is an immediate theoretical $4\times$ throughput improvement. Batching raises $I$ by adding work; quantization raises it by removing bytes, and only the second one is still available here.
3. **Fuse Pointwise Kernels**: Combine adjacent operations (e.g. Bias + GELU + LayerNorm) to execute in on-chip SRAM registers, eliminating redundant intermediate VRAM write/read round-trips.

---

### Question 4: Profiling Trade-offs & Production Economics

**Scenario**: `analyze_profiling_overhead` printed a measured overhead factor on your machine. Call it $k$. On the run that produced the output in this notebook $k$ was a little under $3\times$, almost all of it from two `perf_counter` calls and a `np.median` wrapped around a forward pass that takes a couple of microseconds. That measurement exposes a targeted optimization delivering a 50% runtime reduction ($2\times$ speedup).

#### 1. Justification in Development:
Profiling overhead is an **offline capital investment**. You pay $k$ once, across the few hundred iterations of a diagnostic sweep. The resulting 50% runtime reduction is harvested across **billions of production requests** continuously, so the break-even point arrives in the first seconds of serving. Note also where $k$ came from. The instrumented work here is a single elementwise add, so the fixed per-call cost of timing dominates it. Profile something substantial and the same absolute overhead becomes a rounding error, which is why $k$ is a property of what you measure and not of the profiler.

#### 2. Production Profiling Policy (a preview of production practice):
Nothing in Modules 01 to 14 builds any of the machinery named here, so read this part as a map of where the subject goes rather than as something to derive. Invasive synchronous profilers (using Python's `tracemalloc`, full execution hooks, or monotonic timer calls on every layer) must **never run inline on production user traffic**:
- They disable GPU kernel concurrency, serializing asynchronous streams.
- They incur multi-millisecond CPU scheduling penalties that degrade the slowest 1% of requests, the tail a serving contract is usually written against.
- **Production Solution**: Employ asynchronous out-of-band statistical sampling, sampling hardware counters or tracing roughly one request in ten thousand, for well under $0.1\%$ overhead. Module 19 will build the measurement harness that this discipline sits on top of.

#### 3. The Cost of Profiling vs. The Cost of Misdirected Optimization:
As formalized by **Amdahl's Law**:

$$S_{\text{overall}} = \frac{1}{(1 - f) + \frac{f}{s}}$$

If an engineering team spends three months optimizing an unprofiled routine that accounts for only $f = 5\%$ of total execution time, even an infinite speedup ($s \to \infty$) improves total system throughput by a negligible **$5.2\%$**. Profiling identifies the true $80\%$ bottlenecks, ensuring that engineering hours directly translate to order-of-magnitude systems acceleration.
"""

# %% [markdown]
r"""
## ⭐ Aha Moment: Know Your Model

**What you built:** A complete profiler that measures parameters, FLOPs, memory, and latency.

**Why it matters:** You can't optimize what you can't measure! Before making a model faster or smaller, you need to know where the time and memory go. Your profiler reveals these secrets, telling you exactly what your model costs in compute and memory.

Profiling data guides optimization decisions. Quantization, compression, and acceleration all start with measurement.
"""

# %%
def demo_profiling():
    """🎯 See your profiler reveal model secrets."""
    print("🎯 AHA MOMENT: Know Your Model")
    print("=" * 45)

    # Create a simple model
    layer = Linear(784, 128)

    # Profile it
    profiler = Profiler()
    params = profiler.count_parameters(layer)
    flops = profiler.count_flops(layer, input_shape=(1, 784))

    print(f"Model: Linear(784 → 128)")
    print(f"\nParameters: {params:,}")
    print(f"  = 784 × 128 weights + 128 biases")

    print(f"\nFLOPs: {flops:,}")
    print(f"  = 784 × 128 × 2 (multiply-add per output)")

    print(f"\nMemory: {params * BYTES_PER_FLOAT32 / KB_TO_BYTES:.1f} KB (at FP32)")

    print("\n✨ Profiling reveals optimization opportunities!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_profiling()

# %% [markdown]
r"""
## 🚀 MODULE SUMMARY: Profiling

Congratulations! You've built a comprehensive profiling system for ML performance analysis!

### Key Accomplishments
- **Built complete Profiler class** with parameter, FLOP, memory, and latency measurement
- **Implemented advanced profiling functions** for forward and backward pass analysis
- **Discovered performance characteristics** through scaling and efficiency analysis
- **Built a profiler that is explicit about what it cannot see**: exact counts, honest estimates, and named placeholders instead of invented precision
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **FLOPs vs Reality**: Theoretical operations don't always predict actual performance
- **Memory Bottlenecks**: Many ML operations are limited by memory bandwidth, not compute
- **Batch Size Effects**: Bigger batches raise arithmetic intensity only when the kernel can exploit weight reuse. Our own sweep bought no throughput at all from batch 1 to 128, because a Python-level matmul makes total cost exactly linear in batch
- **Profiling Overhead**: Measurement tools have costs but enable data-driven optimization

### Ready for Next Steps
Your profiling implementation provides the measurement foundation for all optimization work.
You can't optimize what you can't measure, and you can now measure parameters, FLOPs,
allocation footprint, and latency, while knowing exactly which numbers are counts, which
are estimates, and which stand in for hardware this profiler cannot reach.

Export with: `tito module complete 14`

**Next**: Module 15 will add quantization, the first optimization your profiler will let you measure honestly!
"""
