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
# Module 15: Quantization - Reduced Precision for System Efficiency

Welcome to Module 15! You will build an INT8 post-training quantization system, implement affine scaling with zero-point offsets, construct a `QuantizedLinear` layer, and model the $4\times$ memory reduction achieved by moving from FP32 parameters to packed 8-bit integer storage.

## 🔗 Prerequisites & Progress

**You've Built**: Complete ML pipeline with model profiling and roofline bottleneck analysis (`14_profiling`).
**You'll Build**: An INT8 quantization engine (`quantize_int8`, `dequantize_int8`, `QuantizedLinear`, `quantize_model`) with empirical activation calibration.
**You'll Enable**: Modeling the $4\times$ weight-storage compression and memory bandwidth reduction of INT8 serving, unlocking deployment on memory-constrained edge hardware.

<div align="center">
  <img src="quantization_blueprint.svg" alt="TinyTorch Architecture Blueprint: Module 15 Quantization" width="380px">
</div>

### Architectural Roadmap

| Tier | Subsystem | Primitives & Capabilities | Status |
| :--- | :--- | :--- | :--- |
| **Modules 01–08** | Foundation Tier | `Tensor`, `Function`, `Linear`, `GELU`, `SGD`, `Adam`, `Trainer` | Completed |
| **Modules 09–13** | Architecture Tier | `Conv2d`, `BPETokenizer`, `EmbeddingLayer`, `MultiHeadAttention`, `GPT` | Completed |
| **Module 14** | Systems Diagnostics | `Profiler`, `count_flops`, `measure_memory`, `measure_latency` | Completed |
| **Module 15** | **Reduced Precision** | `quantize_int8`, `dequantize_int8`, `QuantizedLinear`, `quantize_model` | **Active Subsystem** |
| **Modules 16–20** | Advanced Optimization | `Pruner`, `TritonKernels`, `KVCache`, `BenchmarkingSuite`, `TinyGPT` | Downstream Consumers |

## 🎯 Learning Objectives

By the end of this module, you will:

1. **Implement Min-Max Affine Quantization**: Derive scale $s = \frac{\beta - \alpha}{255}$ and zero point $z = \text{round}(-\alpha / s) - 128$ to map continuous $\mathbb{R}$ distributions onto the discrete INT8 grid $[-128, 127]$.
2. **Build a QuantizedLinear Layer**: Construct a drop-in linear replacement that stores INT8 weight codes, maintains quantization metadata, and simulates quantized forward inference.
3. **Execute Runtime Activation Calibration**: Collect intermediate activation tensors using representative forward passes with `training=False`, finding optimal dynamic ranges without retraining.
4. **Distinguish Modeled INT8 Memory from NumPy Storage**: Understand that TinyTorch simulates quantization arithmetic while keeping codes in NumPy/Tensor storage, modeling real-world $4\times$ packed byte savings ($W \times 1\text{ B}$ vs $W \times 4\text{ B}$).
5. **Measure Layer-Wise Quantization Sensitivity**: Quantify relative error $\|X - \hat{X}\|_F / \|X\|_F$ across layers to guide mixed-precision decisions.

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/15_quantization/quantization.ipynb`
**Building Side:** Code exports to `tinytorch.perf.quantization`

```python
# Final package structure:
from tinytorch.perf.quantization import quantize_int8, dequantize_int8, QuantizedLinear, quantize_model
```

<div align="center">
  <img src="quant_margin_source.svg" alt="Source Code Mapping: Module 15 Quantization" width="260px">
</div>

## 📋 Module Dependencies

| Dependency | Origin | Imported Symbols | Architectural Purpose in Quantization |
| :--- | :--- | :--- | :--- |
| `tinytorch.core.tensor` | Module 01 | `Tensor` | Multi-dimensional array container holding weights and activations |
| `tinytorch.core.layers` | Module 03 | `Linear`, `Sequential` | Base linear transformation layers replaced during quantization |
| `tinytorch.core.activations` | Module 02 | `ReLU` | Non-linear activations traversed during forward calibration |
| `tinytorch.perf.profiling` | Module 14 | `Profiler` | Diagnostic profiler used to benchmark memory and parameter savings |
| `numpy` | External | `np`, `default_rng` | Vectorized numerical operations and uniform array generation |
| `inspect` | Standard Lib | `signature` | Reflection utility ensuring inference mode (`training=False`) during calibration |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp perf.quantization
#| export
import inspect
import numpy as np
rng = np.random.default_rng(7)
from typing import Tuple, Dict, List, Optional, Any

# Import dependencies from other modules
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU

# Constants for INT8 quantization
INT8_MIN_VALUE = -128
INT8_MAX_VALUE = 127
INT8_RANGE = 256  # Number of possible INT8 values (from -128 to 127 inclusive)
EPSILON = 1e-8  # Stabilize relative-error measurements near zero variance

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
BYTES_PER_INT8 = 1  # INT8 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
"""
## 💡 Introduction: Why Quantization Matters

Before we learn quantization, let's profile a model to see how much memory
FP32 weights actually consume. This will show us why reduced precision matters.
"""

# %%
def explore_motivation_profiling():
    """Profile model memory usage to discover the quantization problem."""
    from tinytorch.perf.profiling import Profiler

    profiler = Profiler()

    # Illustrative budgets for a model's weights on two device classes. They
    # are round numbers chosen to make the table's verdicts concrete, not
    # measurements of any particular phone or microcontroller.
    MOBILE_BUDGET_MB = 100
    EDGE_BUDGET_MB = 10

    # Create models of increasing size
    print("Profiling Memory Usage (FP32 Precision):\n")
    print(f"   Budgets: mobile {MOBILE_BUDGET_MB} MB, edge {EDGE_BUDGET_MB} MB\n")
    print("   Parameters   |  FP32 Memory  |  Device Fit?")
    print("   -------------|---------------|---------------")

    model_configs = [
        (256, 256, "Tiny"),
        (1024, 1024, "Small"),
        (2048, 2048, "Medium"),
        (4096, 4096, "Large"),
        (4096, 8192, "XL"),
    ]

    largest_mb = 0.0
    over_mobile = 0
    over_edge = 0
    for in_feat, out_feat, name in model_configs:
        model = Linear(in_feat, out_feat)

        # Count parameters with the profiler, then convert to FP32 bytes
        params = profiler.count_parameters(model)
        memory_fp32_mb = params * BYTES_PER_FLOAT32 / MB_TO_BYTES
        largest_mb = max(largest_mb, memory_fp32_mb)

        # Check if it fits on different devices
        fits_mobile = memory_fp32_mb < MOBILE_BUDGET_MB
        fits_edge = memory_fp32_mb < EDGE_BUDGET_MB
        over_mobile += 0 if fits_mobile else 1
        over_edge += 0 if fits_edge else 1

        print(f"   {params:>10,}  |  {memory_fp32_mb:7.1f} MB  |  "
              f"Mobile:{'Y' if fits_mobile else 'N'} Edge:{'Y' if fits_edge else 'N'}")

    n = len(model_configs)
    int8_largest_mb = largest_mb * BYTES_PER_INT8 / BYTES_PER_FLOAT32
    int8_verdict = "fits" if int8_largest_mb < MOBILE_BUDGET_MB else "still exceeds"

    print("\nKey Observations:")
    print("   Every parameter uses 4 bytes (32 bits) in FP32")
    print(f"   {over_mobile} of {n} layers exceed the mobile budget; {over_edge} of {n} exceed the edge budget")
    print("   Memory grows linearly with parameter count")

    print("\nThe Problem:")
    print("   Do we really need 32-bit precision for inference?")
    print("   FP32: Can represent 2^32 = 4.3 billion unique values")
    print("   Trained weights cluster in a narrow range around zero, so most of that range is never used")

    print("\nThe Solution:")
    print("   Quantize to INT8 (8-bit integers):")
    print(f"   FP32 -> INT8: 32 bits -> 8 bits ({BYTES_PER_FLOAT32 // BYTES_PER_INT8}x compression)")
    print(f"   Memory: {largest_mb:.0f} MB -> {int8_largest_mb:.0f} MB (the largest layer {int8_verdict} the mobile budget)")
    print("   Accuracy: rounding to 256 levels costs something; this module measures how much, per layer\n")

if __name__ == "__main__":
    explore_motivation_profiling()

# %% [markdown]
r"""
### The Memory Wall Problem

Modern deep neural networks face a fundamental hardware reality: **compute capacity has outpaced memory bandwidth and memory capacity by orders of magnitude**. While accelerators can perform trillions of arithmetic operations per second (teraFLOPs), streaming weights from high-bandwidth memory (HBM) or system DRAM to on-chip arithmetic logic units (ALUs) creates a severe latency and energy bottleneck.

### The Precision Paradox

Standard neural network training takes place in 32-bit single-precision floating point (`float32`), defined by IEEE 754:

$$\text{FP32 Value} = (-1)^s \times 2^{e - 127} \times (1 + m), \quad s \in \{0, 1\}, \, e \in [0, 255], \, m \in [0, 1)$$

$$\underbrace{1 \text{ sign bit}}_{s} \quad + \quad \underbrace{8 \text{ exponent bits}}_{e} \quad + \quad \underbrace{23 \text{ mantissa bits}}_{m} \quad = \quad 32 \text{ bits } (4 \text{ bytes per parameter})$$

While full 32-bit dynamic range ($\approx 10^{-38}$ to $10^{38}$) is critical for gradient accumulation during backpropagation, **trained inference weights cluster tightly within narrow, bounded intervals** (typically $[-1.0, 1.0]$ or $[-3.0, 3.0]$). Maintaining $4.3 \times 10^9$ discrete representable numbers per parameter during inference represents massive over-provisioning of memory bandwidth and capacity.

### The Growing Memory Crisis

When moving modern architectures to mobile devices, embedded systems, or edge robotics, memory footprint dictates deployment feasibility:

| Model Architecture | Parameters | FP32 Footprint (4B) | INT8 Modeled Footprint (1B) | Mobile RAM Budget (4–8 GB) | Edge Deployment Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT-Base** | 110M | 440 MB | 110 MB | 4–8 GB | Fits comfortably |
| **GPT-2 (1.5B)** | 1.5B | 6.0 GB | 1.5 GB | 4–8 GB | Feasible in INT8 only |
| **LLaMA-7B** | 7.0B | 28.0 GB | 7.0 GB | 4–8 GB | Fits single-device INT8 |
| **GPT-3 (175B)** | 175B | 700.0 GB | 175.0 GB | Multi-GPU Node | Requires cluster parallelism |

### The Quantization Solution

Quantization maps continuous 32-bit floating point parameters to discrete 8-bit integer coordinates:

$$\text{FP32} \xrightarrow{\text{Quantization}} \text{INT8} \quad \implies \quad 4 \text{ bytes} \to 1 \text{ byte } (4\times \text{ memory compression})$$

| Precision Format | Bits / Parameter | Dynamic Range | Relative Memory Footprint | Hardware Acceleration Primitives |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 bits (4 B) | $\approx \pm 3.4 \times 10^{38}$ | 100% (Baseline) | Standard IEEE 754 FPU |
| **FP16 / BF16** | 16 bits (2 B) | $\approx \pm 6.5 \times 10^{4} \text{ / } \pm 3.4 \times 10^{38}$ | 50% ($2\times$ compression) | Half-precision Tensor Cores |
| **INT8** | 8 bits (1 B) | $[-128, 127]$ (256 discrete levels) | 25% ($4\times$ compression) | Intel VNNI, ARM NEON, NVIDIA DP4A / Tensor Cores |
| **INT4** | 4 bits (0.5 B) | $[-8, 7]$ (16 discrete levels) | 12.5% ($8\times$ compression) | Sub-byte Packed Integer ALU |

### Real-World Systems Impact

1. **Memory Footprint**: Reduces parameter storage by $4\times$, allowing multi-gigabyte models to reside in resource-constrained on-device memory.
2. **Bandwidth Savings**: Cuts memory bus traffic by $75\%$, directly lowering memory latency and power consumption during autoregressive decoding.
3. **Hardware Throughput**: Supported architectures execute SIMD integer dot products (e.g., AVX-512 VNNI `VPDPBUSD`) at $2\times$ to $4\times$ the throughput of floating-point operations.
4. **Energy Efficiency**: Accessing off-chip DRAM consumes up to $100\times$ more energy than executing an arithmetic instruction; reducing transferred bytes yields proportional battery life gains.
"""

# %% [markdown]
r"""
## 📐 Foundations: The Mathematics of Compression

### Understanding the Core Challenge

Quantization projects continuous high-precision real numbers onto a discrete lattice of $2^b$ integer values (for $b=8$, exactly 256 discrete levels). The primary systems challenge is minimizing quantization distortion (measured by mean-squared error or Signal-to-Quantization-Noise Ratio) while preserving the zero-value boundary without rounding bias.

<div align="center">
  <div align="center">
  <img src="affine_quantization_grid.svg" alt="Affine Quantization Grid" width="680px">
</div>
</div>

### The Affine (Asymmetric) Quantization Formulation

Affine quantization applies a linear scaling and translation operator that maps an arbitrary real interval $[x_{\min}, x_{\max}]$ onto the signed 8-bit integer coordinate space $[q_{\min}, q_{\max}] = [-128, 127]$:

$$q = \text{clamp}\left( \left\lfloor \frac{x}{s} \right\rceil + z, \, -128, \, 127 \right)$$

where:
- $\lfloor \cdot \rceil$ denotes the round-to-nearest-integer operation.
- $s \in \mathbb{R}^+$ is the **scale factor**, representing the continuous step size corresponding to one integer quantum.
- $z \in \mathbb{Z}$ is the **zero point**, representing the integer coordinate that maps exactly to continuous zero ($x = 0.0$).

The inverse transformation (Dequantization) maps discrete integer codes back to continuous real approximations:

$$\hat{x} = (q - z) \times s$$

### Derivation of Scale and Zero Point

Given a target continuous dynamic range $[x_{\min}, x_{\max}]$ (nudged to ensure $0.0 \in [x_{\min}, x_{\max}]$ so that real zero has an exact representation):

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}} = \frac{x_{\max} - x_{\min}}{127 - (-128)} = \frac{x_{\max} - x_{\min}}{255}$$

To guarantee that $x_{\min}$ aligns with the lowest integer coordinate $q_{\min} = -128$:

$$-128 = \frac{x_{\min}}{s} + z \implies z = \left\lfloor -128 - \frac{x_{\min}}{s} \right\rceil$$

<div align="center">
  <div align="center">
  <img src="zero_point_centering.svg" alt="Zero Point Centering" width="680px">
</div>
</div>

### Symmetric vs. Asymmetric Quantization

| Architectural Dimension | Asymmetric (Affine) Quantization | Symmetric Quantization |
| :--- | :--- | :--- |
| **Quantization Mapping** | $q = \text{clamp}(\lfloor x/s \rceil + z, -128, 127)$ | $q = \text{clamp}(\lfloor x/s \rceil, -127, 127)$ |
| **Zero Point Alignment** | $z \in [-128, 127]$ (Dynamic integer offset) | $z = 0$ (Implicitly fixed at integer zero) |
| **Scale Derivation** | $s = (x_{\max} - x_{\min}) / 255$ | $s = \max(\|x_{\min}\|, \|x_{\max}\|) / 127$ |
| **Zero Representation** | Real $0.0$ maps exactly to $z$ without precision penalty | Real $0.0$ maps symmetrically to $0$ |
| **GEMM Arithmetic Overhead** | Cross-terms $\sum x_i z_w$ require compensation logic | Cross-terms cancel out entirely ($\mathcal{O}(1)$ bias update) |
| **Typical Target Domain** | Activations post-ReLU ($\ge 0$, highly skewed) | Weights & activations centered at zero (GELU, LayerNorm) |

### Quantization Error and Signal-to-Noise Ratio (SQNR)

When discretizing a continuous tensor, rounding introduces quantization noise $\epsilon = x - \hat{x}$:

$$\epsilon = x - (q - z) \times s, \quad \text{bounded by} \quad |\epsilon| \le \frac{s}{2} = \frac{x_{\max} - x_{\min}}{510}$$

Assuming a uniform distribution of rounding noise over the quantization interval $[-\frac{s}{2}, \frac{s}{2}]$, the expected noise power is:

$$\sigma_q^2 = \mathbb{E}[\epsilon^2] = \frac{1}{s} \int_{-s/2}^{s/2} \epsilon^2 d\epsilon = \frac{s^2}{12}$$

For a $b$-bit uniform quantizer covering the full signal dynamic range, the theoretical Signal-to-Quantization-Noise Ratio is:

$$\text{SQNR} \approx 6.02 \times b + 1.76 \text{ dB}$$

$$\text{At } b=8 \text{ (INT8)}: \quad \text{SQNR} \approx 6.02 \times 8 + 1.76 = 49.92 \text{ dB}$$

An SQNR of $\approx 50\text{ dB}$ provides sufficient dynamic fidelity for deep neural inference, explaining why 8-bit integers preserve over $99\%$ of FP32 baseline task accuracy across transformers and convolutional networks.
"""

# %% [markdown]
r"""
## 🏗️ Implementation: Building the Quantization Engine

### Our Implementation Strategy

We construct the quantization engine across four modular abstraction boundaries, matching the design of production inference toolchains (such as PyTorch's `torch.ao.quantization` and ONNX Runtime):

<div align="center">
  <div align="center">
  <img src="quantization_pipeline_stages.svg" alt="Quantization Pipeline Stages" width="680px">
</div>
</div>

| Abstraction Tier | Target Primitives | Engineering Functionality | Precision Transformation |
| :--- | :--- | :--- | :--- |
| **Tier 1: Parameters** | `scale`, `zero_point` | Bounded dynamic range extraction and grid resolution | Continuous $[x_{\min}, x_{\max}] \to (s \in \mathbb{R}^+, z \in \mathbb{Z})$ |
| **Tier 2: Tensor Primitives** | `quantize_int8`, `dequantize_int8` | Discrete coordinate projection & reconstructed float recovery | $\text{FP32 Tensor} \longleftrightarrow \text{INT8 Codes}$ |
| **Tier 3: Layer Abstraction** | `QuantizedLinear` | Weight quantization, metadata storage, and runtime dequantization | $\text{Linear}(\text{FP32}) \to \text{QuantizedLinear}(\text{INT8})$ |
| **Tier 4: Graph Integration** | `quantize_model` | Recursive sequential graph walk, calibration, & layer substitution | Complete FP32 Pipeline $\to$ Quantized Inference Graph |

### Quantization Pipeline Primitives

- `quantize_int8()`: Maps continuous FP32 tensors into signed 8-bit integer coordinates $[-128, 127]$.
- `dequantize_int8()`: Reconstructs FP32 floating-point values from integer coordinates and quantization metadata.
- `QuantizedLinear`: Drop-in replacement for `Linear`, storing weights in INT8 format with modeled $4\times$ memory reduction.
- `_collect_layer_inputs()` & `_quantize_single_layer()`: Runtime activation profiling and calibration primitives.
- `quantize_model()`: Transforms entire neural architectures into memory-efficient quantized models.
"""

# %% [markdown]
r"""
## 🏗️ INT8 Quantization: The Foundation

The foundational primitive `quantize_int8` discretizes continuous tensors into 8-bit signed integer codes while preserving zero alignment:

| Execution Phase | Operational Logic | Formal Definition | Numerical Trace Example |
| :--- | :--- | :--- | :--- |
| **1. Dynamic Range Bound** | Evaluate clamped tensor extrema including zero | $[x_{\min}, x_{\max}] = [\min(X \cup \{0\}), \max(X \cup \{0\})]$ | $[-1.5, 2.8] \implies x_{\min}=-1.5, x_{\max}=2.8$ |
| **2. Grid Scaling** | Compute continuous quantum width | $s = (x_{\max} - x_{\min}) / 255$ | $s = (2.8 - (-1.5)) / 255 = 0.016863$ |
| **3. Zero Point Offset** | Align real $0.0$ to exact integer coordinate | $z = \text{round}(-128 - x_{\min} / s)$ | $z = \text{round}(-128 - (-1.5 / 0.016863)) = -39$ |
| **4. Projection & Clamp** | Project continuous values onto integer grid | $q = \text{clamp}(\text{round}(x / s + z), -128, 127)$ | $x=2.8 \implies \text{round}(2.8/0.016863 - 39) = 127$ |

**Key Systems Challenges Solved:**
- **Exact Zero Preservation**: Ensures padding zeros and sparse activations map identically to $z$ without rounding distortion.
- **Dynamic Scale Resolution**: Adapts the quantization lattice to the empirical range of each individual tensor.
- **Asymmetric Offsetting**: Allocates all 256 discrete levels across the observed domain rather than wasting bins on unused regions.
"""

# %% nbgrader={"grade": false, "grade_id": "quantize_int8", "solution": true}
#| export
def quantize_int8(tensor: Tensor) -> Tuple[Tensor, float, int]:
    """
    Quantize FP32 tensor to INT8 using asymmetric (min-max) quantization.

    TODO: Implement INT8 quantization with scale and zero_point calculation

    APPROACH:
    1. Find min/max values in tensor data
    2. Handle the constant tensor (min_val == max_val) as a special case; see HINTS
    3. Nudge the range to include zero: min_val = min(min_val, 0), max_val = max(max_val, 0)
    4. Calculate scale: (max_val - min_val) / 255 (INT8 range: -128 to 127)
    5. Calculate zero_point: offset that maps min_val to INT8_MIN (-128)
       Formula: zero_point = round(INT8_MIN - (min_val / scale))
    6. Apply quantization formula round(value / scale + zero_point),
       clamp to the INT8 range [-128, 127], and cast to int8

    Args:
        tensor: Input FP32 tensor to quantize

    Returns:
        q_tensor: Quantized INT8 tensor
        scale: Scaling factor (float)
        zero_point: Zero point offset (int)

    NOTE: the returned codes are computed as int8, but TinyTorch's Tensor
    stores every array as float32, so q_tensor.data holds INT8-range values
    in a float32 array (the codes print as -128.0 and 127.0). This is
    *simulated* quantization: the rounding is exactly what INT8 hardware
    does, but the array still occupies four bytes per element. The memory
    saving is accounted for analytically in QuantizedLinear.memory_usage()
    and _measure_layer_bytes(), not read off the dtype.

    EXAMPLE:
    >>> tensor = Tensor([[-1.0, 0.0, 2.0], [0.5, 1.5, -0.5]])
    >>> q_tensor, scale, zero_point = quantize_int8(tensor)
    >>> print(f"Scale: {scale:.4f}, Zero point: {zero_point}")
    Scale: 0.0118, Zero point: -43

    HINTS:
    - Use np.round() for quantization
    - Clamp with np.clip(values, -128, 127)
    - Constant tensor (every element equals c): there is no range to map, so
      encode every element as code 0 and pick zero_point and scale so that
      (0 - zero_point) * scale == c. For nonzero c, use scale = |c| and
      zero_point = -1 (c > 0) or +1 (c < 0). For c = 0, use scale = 1
      and zero_point = 0. Fractional constants need a fractional scale too.
    """
    ### BEGIN SOLUTION
    data = tensor.data
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Quantization requires nonempty finite values")

    # Step 1: Find dynamic range
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    # Step 2: Handle edge case (constant tensor).
    # All elements have the same value c, so there is no range to map. We encode
    # every element as q=0 and choose zero_point/scale so dequantization recovers
    # the constant via (0 - zero_point) * scale = c.
    if max_val == min_val:
        c = min_val
        if c == 0:
            scale = 1.0
            zero_point = 0
        else:
            # Encode the magnitude in the scale, preserving both fractional
            # and large constants with a zero point that fits in one byte.
            zero_point = -1 if c > 0 else 1
            scale = abs(c)
        quantized_data = np.zeros_like(data, dtype=np.int8)
        return Tensor(quantized_data), scale, zero_point

    # Step 3: Nudge the range to include zero before computing scale.
    # If min_val and max_val share a sign (every post-ReLU activation, for
    # instance), the zero_point implied by the raw range falls outside
    # [-128, 127]. Clamping it there destroys the affine mapping and silently
    # corrupts every value.
    # Widening the range so it straddles zero is what PyTorch and TFLite do,
    # and it costs at most one quantization level of precision.
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)

    # Step 4: Calculate scale
    # Map [min_val, max_val] to [INT8_MIN_VALUE, INT8_MAX_VALUE] (INT8 range)
    scale = (max_val - min_val) / (INT8_RANGE - 1)

    # Step 5: Calculate zero_point (the code that min_val maps to is -128)
    zero_point = int(np.round(INT8_MIN_VALUE - min_val / scale))

    # zero_point is now guaranteed inside the INT8 range by construction;
    # clamp defensively rather than as a correctness crutch.
    zero_point = int(np.clip(zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE))

    # Step 6: Apply quantization formula q = (x / scale) + zero_point, clamp to
    # the INT8 range, and cast to int8. Tensor() will widen the int8 array back
    # to float32 (see the NOTE in the docstring), so the codes are exact but the
    # array is not one byte per element.
    quantized_data = np.round(data.astype(np.float64) / scale + zero_point)
    quantized_data = np.clip(quantized_data, INT8_MIN_VALUE, INT8_MAX_VALUE).astype(np.int8)

    return Tensor(quantized_data), scale, zero_point
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: INT8 Quantization

This test validates our INT8 quantization function works correctly with various data types and edge cases.

**What we're testing**: Basic quantization and dequantization roundtrip
**Why it matters**: Foundation for all memory reduction - if quantization fails, nothing works
**Expected**: Quantized values in INT8 range with acceptable reconstruction error
"""

# %% nbgrader={"grade": true, "grade_id": "test-quantize-int8", "locked": true, "points": 5}
def test_unit_quantize_int8():
    """🧪 Test INT8 quantization implementation."""
    print("🧪 Unit Test: INT8 Quantization...")

    # Test basic quantization
    tensor = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    q_tensor, scale, zero_point = quantize_int8(tensor)

    # Verify quantized values are in INT8 range
    assert np.all(q_tensor.data >= INT8_MIN_VALUE)
    assert np.all(q_tensor.data <= INT8_MAX_VALUE)
    assert isinstance(scale, float)
    assert isinstance(zero_point, int)

    # Test dequantization preserves approximate values
    dequantized = (q_tensor.data - zero_point) * scale
    error = np.mean(np.abs(tensor.data - dequantized))
    # Round-to-nearest bounds the error at scale/2 -- one half of a quantization
    # step. Asserting against that real bound rather than a loose constant is
    # what catches a broken scale or a clamped zero_point; a tolerance of 0.25
    # here (25x the true bound) would let both through unnoticed.
    max_error = scale / 2
    assert error <= max_error * 1.01, (
        f"Quantization error {error:.6f} exceeds the INT8 bound scale/2 = {max_error:.6f}. "
        f"A round-trip should never lose more than half a quantization step."
    )

    # Test edge case: constant tensor -- dequantize must recover the original value,
    # not zero. A zero_point of 0 gives (0 - 0) * 1.0 = 0.0 for every element
    # regardless of what the constant was, so the zero_point must carry it.
    constant_tensor = Tensor([[2.0, 2.0], [2.0, 2.0]])
    q_const, scale_const, zp_const = quantize_int8(constant_tensor)
    assert scale_const > 0
    restored_const = (q_const.data.astype(np.float32) - zp_const) * scale_const
    assert np.allclose(restored_const, 2.0), (
        f"Constant tensor dequantized to {restored_const} instead of 2.0. "
        "zero_point must encode the constant value, not default to 0."
    )

    # Negative constant
    neg_tensor = Tensor([[-3.0, -3.0]])
    q_neg, scale_neg, zp_neg = quantize_int8(neg_tensor)
    restored_neg = (q_neg.data.astype(np.float32) - zp_neg) * scale_neg
    assert np.allclose(restored_neg, -3.0, atol=0.01), (
        f"Negative constant tensor dequantized to {restored_neg} instead of -3.0."
    )

    # Large constant outside the INT8 range. With scale=1.0 the zero_point would
    # be clamped to the INT8 range and silently corrupt the value, so the
    # constant must still be recovered exactly (the |c| > 127 case in the HINTS).
    large_tensor = Tensor([[500.0, 500.0]])
    q_large, scale_large, zp_large = quantize_int8(large_tensor)
    restored_large = (q_large.data.astype(np.float32) - zp_large) * scale_large
    assert np.allclose(restored_large, 500.0), (
        f"Large constant tensor dequantized to {restored_large} instead of 500.0. "
        "zero_point must not be clamped for |c| > 127."
    )

    # A fixed scale of 1 would erase fractional constants such as 0.25.
    for value in (0.0, 0.25, -0.5):
        q, scale, zero_point = quantize_int8(Tensor([value, value]))
        restored = (q.data - zero_point) * scale
        assert np.allclose(restored, value), f"Constant {value} reconstructed as {restored}"

    print("✅ INT8 quantization works correctly!")

if __name__ == "__main__":
    test_unit_quantize_int8()

# %% [markdown]
r"""
## 🏗️ INT8 Dequantization: Restoring Precision

Dequantization evaluates the inverse affine transformation, projecting discrete 8-bit integer coordinates back onto the continuous floating-point manifold:

$$\hat{x} = (q - z) \times s$$

| Stage | Input Representation | Transformation Step | Output Representation | Numerical Trace Example |
| :--- | :--- | :--- | :--- | :--- |
| **Quantized Code** | $q \in [-128, 127]$ (INT8) | Raw quantized coordinate | Integer tensor | $q = [-128, -27, 127]$ |
| **Zero Point Offset** | $z \in [-128, 127]$ | Center subtraction $(q - z)$ | Zero-centered integer | $q - z = [-89, 12, 166]$ (for $z = -39$) |
| **Scale Expansion** | $s \in \mathbb{R}^+$ | Continuous scaling $\times s$ | Reconstructed FP32 | $\hat{x} = [-1.5008, 0.2024, 2.7993]$ |
| **Error Bound** | $|\hat{x} - x|$ | Pointwise absolute error | Bounded by $\le s/2$ | $\epsilon = [0.0008, 0.0024, 0.0007] \le 0.0084$ |

**Systems Significance of Dequantization:**
- **Inference Emulation**: In systems without native INT8 GEMM accelerators, weights are stored packed in INT8 (saving $4\times$ memory and disk bandwidth) and dequantized on-the-fly into vector registers before executing FP32 BLAS routines.
- **Hardware GEMM Fusion**: In production INT8 hardware (e.g. NVIDIA Tensor Cores or Intel VNNI), the integer dot product $\sum q_{w} q_{x}$ is computed directly in INT32 accumulators, and the dequantization scale $s_w s_x$ is applied once at the end of the tile accumulation.
"""

# %% nbgrader={"grade": false, "grade_id": "dequantize_int8", "solution": true}
#| export
def dequantize_int8(q_tensor: Tensor, scale: float, zero_point: int) -> Tensor:
    """
    Dequantize INT8 tensor back to FP32.

    TODO: Implement dequantization using the inverse formula

    APPROACH:
    1. Apply inverse quantization: (quantized_value - zero_point) * scale
    2. Return as new FP32 Tensor

    Args:
        q_tensor: Quantized INT8 tensor
        scale: Scaling factor from quantization
        zero_point: Zero point offset from quantization

    Returns:
        Reconstructed FP32 tensor

    EXAMPLE:
    >>> q_tensor = Tensor([[-100, 0, 50]])  # INT8 values
    >>> scale, zero_point = 0.02, -25
    >>> fp32_tensor = dequantize_int8(q_tensor, scale, zero_point)
    >>> print(fp32_tensor.data)
    [[-1.5, 0.5, 1.5]]  # Reconstructed FP32 values

    HINT:
    - Formula: dequantized = (quantized - zero_point) * scale
    """
    ### BEGIN SOLUTION
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"Scale must be positive and finite, got {scale}")
    if not (-128 <= zero_point <= 127):
        raise ValueError(f"Zero point must be in [-128, 127], got {zero_point}")

    # Apply inverse quantization formula
    # This is the correct inverse of: quantized = (value / scale) + zero_point
    # Therefore: value = (quantized - zero_point) * scale
    # Compute on a wider grid: a valid float32 endpoint can round just beyond
    # float32's range, while a subnormal scale may not fit in float32 at all.
    dequantized_data = (q_tensor.data.astype(np.float64) - zero_point) * scale
    fp32_limit = np.finfo(np.float32).max
    return Tensor(np.clip(dequantized_data, -fp32_limit, fp32_limit))
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: INT8 Dequantization

This test validates our dequantization function correctly restores FP32 values from INT8.

**What we're testing**: Roundtrip quantize -> dequantize preserves values
**Why it matters**: Neural networks need FP32 values for computation
**Expected**: Small reconstruction error after roundtrip
"""

# %% nbgrader={"grade": true, "grade_id": "test-dequantize-int8", "locked": true, "points": 5}
def test_unit_dequantize_int8():
    """🧪 Test INT8 dequantization implementation."""
    print("🧪 Unit Test: INT8 Dequantization...")

    # Test round-trip: quantize → dequantize
    original = Tensor([[-1.5, 0.0, 3.2], [1.1, -0.8, 2.7]])
    q_tensor, scale, zero_point = quantize_int8(original)
    restored = dequantize_int8(q_tensor, scale, zero_point)

    # Verify round-trip error is small
    error = np.mean(np.abs(original.data - restored.data))
    assert error < 0.1, f"Round-trip error too high: {error}"

    # Verify output is float32
    assert restored.data.dtype == np.float32

    print("✅ INT8 dequantization works correctly!")

if __name__ == "__main__":
    test_unit_dequantize_int8()

# %% [markdown]
r"""
## 🏗️ QuantizedLinear: The Heart of Efficient Networks

### Architectural Comparison: FP32 vs Quantized Linear

| Architectural Dimension | Baseline `Linear` | `QuantizedLinear` (Simulated) | Production INT8 GEMM |
| :--- | :--- | :--- | :--- |
| **Weight Storage** | FP32 (4 bytes / weight) | INT8 codes in FP32 (1 byte modeled) | Packed INT8 (`int8_t`, 1 byte physical) |
| **Weight Memory Footprint** | $N \times M \times 4$ bytes | $N \times M \times 1 + 8$ bytes modeled | $N \times M \times 1 + 8$ bytes physical |
| **Execution Path** | Direct FP32 GEMM: $Y = X W^T + b$ | Dequantize $\hat{W} \to$ FP32 GEMM | Integer Tensor Core GEMM $\to$ Rescale |
| **Memory Bandwidth Demand** | $100\%$ (baseline) | $25\%$ weight streaming bandwidth | $25\%$ weight & activation bandwidth |
| **Dynamic Calibration** | Not required | Optional input calibration | Required for activation scales |

### Quantized Forward Pass Mathematics

In `QuantizedLinear`, weights and biases are stored as discrete INT8 coordinates with accompanying scale and zero-point metadata:

$$\hat{W} = (Q_W - z_W) \times s_W, \quad \hat{b} = (Q_b - z_b) \times s_b$$

The forward pass reconstructs the weights on-the-fly and executes high-precision matrix multiplication:

$$Y = X \hat{W}^T + \hat{b} = X \left( (Q_W - z_W) s_W \right)^T + (Q_b - z_b) s_b$$

### Activation Calibration Pipeline

| Calibration Stage | Operations | Systems Rationale |
| :--- | :--- | :--- |
| **1. Sample Profiling** | Forward $N$ calibration batches through preceding layers | Capture empirical activation statistics without updating weights |
| **2. Distribution Bound** | Evaluate $[\min(X_{\text{calib}}), \max(X_{\text{calib}})]$ across collected inputs | Prevent clipping dynamic range on normal inference inputs |
| **3. Parameter Derivation** | Compute $s_{\text{in}} = (x_{\max} - x_{\min}) / 255$ and $z_{\text{in}}$ | Establish optimal input quantization lattice |
| **4. Quantization Ready** | Store $s_{\text{in}}, z_{\text{in}}$ on layer instance | Enables full input discretization when moving to INT8 execution |

### Lifecycle of a Quantized Layer

| Phase | When Executed | Operations Performed | Computational Cost |
| :--- | :--- | :--- | :--- |
| **Initialization** | Quantization time (Offline) | Quantize $W \to Q_W$, $b \to Q_b$; store $s_W, z_W, s_b, z_b$ | One-time overhead ($\mathcal{O}(NM)$) |
| **Calibration** | Pre-deployment (Offline) | Pass unlabelled validation samples, compute input scale $s_X, z_X$ | One-time forward pass ($\mathcal{O}(B \cdot NM)$) |
| **Inference Forward** | Runtime (Per query) | Dequantize $\hat{W}$, evaluate matrix multiply $Y = X \hat{W}^T + \hat{b}$ | Amortized $\mathcal{O}(NM)$ GEMM |

**Memory Layout Note**:
In TinyTorch's pedagogical environment, integer codes are simulated inside NumPy `float32` arrays to preserve broad framework compatibility. The reported $4\times$ memory reduction models the packed byte representation ($1$ byte per parameter) achieved when deployed with C++ runtime engines (like TensorRT or ONNX Runtime).
"""

# %% nbgrader={"grade": false, "grade_id": "quantized_linear", "solution": true}
#| export
class QuantizedLinear:
    """Linear layer with INT8-range codes stored and computed in float32."""

    def __init__(self, linear_layer: Linear):
        """
        Create quantized version of existing linear layer.

        TODO: Quantize weights and bias, store quantization parameters

        APPROACH:
        1. Quantize weights using quantize_int8
        2. Quantize bias if it exists
        3. Store original layer reference for forward pass
        4. Store quantization parameters for dequantization

        EXAMPLE:
        >>> original_layer = Linear(128, 64)
        >>> original_layer.weight = Tensor(rng.standard_normal((128, 64)) * 0.1)
        >>> original_layer.bias = Tensor(rng.standard_normal(64) * 0.01)
        >>> quantized_layer = QuantizedLinear(original_layer)
        >>> print(quantized_layer.q_weight.data.min(), quantized_layer.q_weight.data.max())
        -128.0 127.0

        NOTE: q_weight holds INT8-RANGE values but its dtype is float32, because
        TinyTorch's Tensor stores everything as float32. This is *simulated*
        quantization: it reproduces the accuracy effects exactly, but not the
        memory saving. Real INT8 inference stores these values in an int8 array,
        which is where the actual 4x reduction comes from. We compute the saving
        analytically in memory_usage() and _measure_layer_bytes() rather than reading it off dtype.

        HINTS:
        - Use quantize_int8() to convert weight and bias tensors
        - Store all quantization parameters (scale, zero_point) for later dequantization
        - Initialize input_scale and input_zero_point to None (set during calibration)
        """
        ### BEGIN SOLUTION role="scaffold"
        self.original_layer = linear_layer

        # Quantize weights
        self.q_weight, self.weight_scale, self.weight_zero_point = quantize_int8(linear_layer.weight)

        # Quantize bias if it exists
        if linear_layer.bias is not None:
            self.q_bias, self.bias_scale, self.bias_zero_point = quantize_int8(linear_layer.bias)
        else:
            self.q_bias = None
            self.bias_scale = None
            self.bias_zero_point = None

        # Store input quantization parameters (set during calibration)
        self.input_scale = None
        self.input_zero_point = None
        ### END SOLUTION

    def calibrate(self, sample_inputs: List[Tensor]):
        """
        Calibrate input quantization parameters using sample data.

        TODO: Calculate optimal input quantization parameters

        APPROACH:
        1. Collect statistics from sample inputs
        2. Calculate optimal scale and zero_point for inputs
        3. Store for use in forward pass

        EXAMPLE:
        >>> layer = QuantizedLinear(Linear(64, 32))
        >>> sample_data = [Tensor(rng.standard_normal((1, 64))) for _ in range(10)]
        >>> layer.calibrate(sample_data)
        >>> print(layer.input_scale is not None)
        True

        HINTS:
        - Flatten all sample inputs and find global min/max values
        - Use the same scale/zero_point formula as quantize_int8()
        - Handle edge case where all inputs have the same value (constant tensor)
        """
        ### BEGIN SOLUTION role="core"
        # Collect all input values
        all_values = []
        for inp in sample_inputs:
            all_values.extend(inp.data.flatten())

        all_values = np.array(all_values)
        if all_values.size == 0 or not np.all(np.isfinite(all_values)):
            raise ValueError("Calibration requires nonempty finite samples")

        # Calculate input quantization parameters, widening the range to
        # straddle zero exactly as quantize_int8 does (post-ReLU inputs are
        # all non-negative, and a zero_point outside the INT8 range would
        # otherwise be clamped and corrupt every value)
        min_val = min(float(np.min(all_values)), 0.0)
        max_val = max(float(np.max(all_values)), 0.0)

        if max_val == min_val:
            self.input_scale = 1.0
            self.input_zero_point = 0
        else:
            self.input_scale = (max_val - min_val) / (INT8_RANGE - 1)
            self.input_zero_point = int(np.round(INT8_MIN_VALUE - min_val / self.input_scale))
            self.input_zero_point = int(np.clip(self.input_zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE))
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with quantized computation.

        TODO: Implement quantized forward pass

        APPROACH:
        1. Quantize input (if calibrated)
        2. Dequantize weights and input for computation (educational approach)
        3. Perform matrix multiplication
        4. Return FP32 result

        EXAMPLE:
        >>> layer = QuantizedLinear(Linear(4, 3))
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
        >>> output = layer.forward(x)
        >>> print(output.shape)
        (1, 3)

        HINTS:
        - If calibrate() has set input_scale, round the input onto its grid and
          back first (clip to the INT8 range), so saturation is visible
        - Use dequantize_int8() to restore weights to FP32 before computation
        - Use x.matmul() for matrix multiplication
        - Add bias after matmul if it exists (dequantize bias first)

        NOTE: Production quantization uses INT8 GEMM libraries for speed
        """
        ### BEGIN SOLUTION role="core"
        # For educational purposes, we dequantize and compute in FP32
        # Production systems use specialized INT8 GEMM operations

        # Round the input onto the calibrated grid and back, so the activations
        # carry the same rounding (and saturation) they would on INT8 hardware
        if self.input_scale is not None:
            if not np.all(np.isfinite(x.data)):
                raise ValueError("QuantizedLinear forward requires finite input values")
            q_x = np.clip(np.round(x.data.astype(np.float64) / self.input_scale + self.input_zero_point),
                          INT8_MIN_VALUE, INT8_MAX_VALUE)
            x = dequantize_int8(Tensor(q_x), self.input_scale, self.input_zero_point)

        # Dequantize weights
        weight_fp32 = dequantize_int8(self.q_weight, self.weight_scale, self.weight_zero_point)

        # Perform computation (same as original layer)
        result = x.matmul(weight_fp32)

        # Add bias if it exists
        if self.q_bias is not None:
            bias_fp32 = dequantize_int8(self.q_bias, self.bias_scale, self.bias_zero_point)
            result = Tensor(result.data + bias_fp32.data)

        return result
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the quantized linear layer to be called like a function."""
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """Return quantized parameters."""
        params = [self.q_weight]
        if self.q_bias is not None:
            params.append(self.q_bias)
        return params

    def memory_usage(self) -> Dict[str, float]:
        """Model packed INT8 bytes, including metadata; not actual NumPy storage."""
        ### BEGIN SOLUTION role="core"
        # Original FP32 usage
        original_weight_bytes = self.original_layer.weight.data.size * BYTES_PER_FLOAT32
        original_bias_bytes = 0
        if self.original_layer.bias is not None:
            original_bias_bytes = self.original_layer.bias.data.size * BYTES_PER_FLOAT32

        # Quantized INT8 usage
        quantized_weight_bytes = self.q_weight.data.size * BYTES_PER_INT8
        quantized_bias_bytes = 0
        if self.q_bias is not None:
            quantized_bias_bytes = self.q_bias.data.size * BYTES_PER_INT8

        # Overhead for the quantization parameters (a scale and a zero point):
        # a few bytes, negligible next to the arrays
        # Each quantized array needs its own float32 scale and integer zero point.
        overhead_bytes = BYTES_PER_FLOAT32 * 2 * (1 + int(self.q_bias is not None))

        quantized_total = quantized_weight_bytes + quantized_bias_bytes + overhead_bytes
        original_total = original_weight_bytes + original_bias_bytes

        return {
            'original_bytes': original_total,
            'quantized_bytes': quantized_total,
            'compression_ratio': original_total / quantized_total if quantized_total > 0 else 1.0
        }
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: QuantizedLinear

This test validates our QuantizedLinear layer works correctly and achieves memory savings.

**What we're testing**: Quantized layer forward pass and compression ratio
**Why it matters**: This is the core component that replaces Linear layers in models
**Expected**: Forward pass produces similar output with ~4x compression
"""

# %% nbgrader={"grade": true, "grade_id": "test-quantized-linear", "locked": true, "points": 5}
def test_unit_quantized_linear():
    """🧪 Test QuantizedLinear implementation."""
    print("🧪 Unit Test: QuantizedLinear...")

    # Create original linear layer
    original = Linear(4, 3)
    original.weight = Tensor(rng.standard_normal((4, 3)) * 0.5)  # Smaller range for testing
    original.bias = Tensor(rng.standard_normal(3) * 0.1)

    # Create quantized version
    quantized = QuantizedLinear(original)

    # Test forward pass
    x = Tensor(rng.standard_normal((2, 4)) * 0.5)

    # Original forward pass
    original_output = original.forward(x)

    # Quantized forward pass
    quantized_output = quantized.forward(x)

    # Compare outputs (should be close but not identical due to quantization)
    error = np.mean(np.abs(original_output.data - quantized_output.data))
    assert error < 0.1, f"Quantization error too high: {error}"

    # Test memory usage
    memory_info = quantized.memory_usage()
    print(f"  Compression ratio: {memory_info['compression_ratio']:.2f}×")
    print(f"  Original bytes: {memory_info['original_bytes']}")
    print(f"  Quantized bytes: {memory_info['quantized_bytes']}")

    # Tiny layers expose metadata overhead: 15 codes plus two scale/zero-point pairs.
    assert memory_info['quantized_bytes'] == 15 + 16
    assert np.isclose(memory_info['compression_ratio'], 60 / 31)

    print(f"  Memory reduction: {memory_info['compression_ratio']:.1f}x")
    print("✅ QuantizedLinear works correctly!")

if __name__ == "__main__":
    test_unit_quantized_linear()

# %% [markdown]
"""
## 🏗️ Model Quantization: Scaling to Full Networks

### The Model Quantization Challenge

Quantizing individual tensors is useful, but real applications need to quantize entire neural networks with multiple layers, activations, and complex data flows. The key is replacing standard layers (like Linear) with their quantized equivalents (QuantizedLinear) while keeping activation functions unchanged since they have no parameters.

### Smart Layer Selection

Not all layers benefit equally from quantization. Linear and convolutional layers with many parameters see the largest benefits, while activation functions (which have no parameters) cannot be quantized. Some layers like input/output projections may be sensitive to quantization and should be kept in higher precision for critical applications.

### Calibration Data Flow

Calibration runs sample data through the model layer-by-layer, collecting activation statistics at each layer. These statistics (min/max values, distributions) determine optimal quantization parameters for each layer, ensuring minimal accuracy loss during quantization.

### Memory Impact

Packed INT8 codes use one quarter of FP32 code storage. Scale and zero-point metadata reduce the overall saving, particularly for small arrays. TinyTorch models these bytes rather than allocating packed INT8 arrays.

Now let's implement the functions that make this transformation possible!
"""

# %% [markdown]
r"""
### From One Layer to a Whole Model

Quantizing individual layers is useful, but production pipelines quantize entire deep architectures end-to-end. We structure full-model quantization into two sequential stages:

1. **Collect layer inputs (`_collect_layer_inputs`)**: Forward calibration samples through layers $0 \dots i-1$ with `training=False` to capture empirical activation statistics at layer $i$.
2. **Quantize single layer (`_quantize_single_layer`)**: Instantiate `QuantizedLinear` with INT8 weights and calibrate input scales.

The composition function `quantize_model()` automates this transformation across the container graph:

| Model Layer Index | Original FP32 Layer Type | Quantized Model Replacement | Parameter Precision | Memory Compression |
| :--- | :--- | :--- | :--- | :--- |
| `layers[0]` | `Linear(784, 128)` | `QuantizedLinear(784, 128)` | INT8 weights + metadata | $\approx 4\times$ reduction |
| `layers[1]` | `ReLU()` | `ReLU()` (Unchanged) | Stateless (0 params) | Identical (0 B) |
| `layers[2]` | `Linear(128, 64)` | `QuantizedLinear(128, 64)` | INT8 weights + metadata | $\approx 4\times$ reduction |
| `layers[3]` | `ReLU()` | `ReLU()` (Unchanged) | Stateless (0 params) | Identical (0 B) |
| `layers[4]` | `Linear(64, 10)` | `QuantizedLinear(64, 10)` | INT8 weights + metadata | $\approx 4\times$ reduction |
"""

# %% [markdown]
r"""
## 🏗️ Collecting Layer Inputs: Calibration Data Flow

Before calibrating a layer's input scale, we must extract the exact distribution of activations arriving at its inputs during evaluation. `_collect_layer_inputs` executes a forward pass through preceding layers $0 \dots i-1$ for up to `max_samples`:

| Pipeline Stage | Component | Execution Details | Dataflow Shape |
| :--- | :--- | :--- | :--- |
| **Input Batch** | Calibration Dataset | Unlabelled domain samples $\{x^{(1)}, \dots, x^{(N)}\}$ | $(B, d_{\text{in}})$ |
| **Prefix Forward** | Subgraph `layers[0:i]` | Forward pass with `training=False` (disabling dropout) | $(B, d_{\text{intermediate}})$ |
| **Activation Tap** | Layer $i$ Input Buffer | Collect intermediate tensors without gradient tracking | Stored activations for calibration |
"""

# %% nbgrader={"grade": false, "grade_id": "collect_layer_inputs", "solution": true}
#| export
def _collect_layer_inputs(model, layer_index: int, calibration_data: List[Tensor], max_samples: int = 10) -> List[Tensor]:
    """
    Forward calibration data through preceding layers to collect inputs for a specific layer.

    TODO: Forward each calibration sample through layers 0..layer_index-1

    APPROACH:
    1. Take up to max_samples from calibration_data for efficiency
    2. For each sample, forward through all layers before layer_index in inference mode
    3. Collect the resulting activations as the input distribution for this layer

    Args:
        model: Model with .layers attribute (Sequential pattern)
        layer_index: Index of the layer we want inputs for
        calibration_data: List of sample input tensors
        max_samples: Maximum number of samples to process (default 10)

    Returns:
        List of Tensor activations arriving at layer_index

    EXAMPLE:
    >>> model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    >>> samples = [Tensor(rng.standard_normal((1, 4))) for _ in range(5)]
    >>> inputs_at_layer2 = _collect_layer_inputs(model, 2, samples)
    >>> print(len(inputs_at_layer2))  # 5 activation tensors
    5

    HINT:
    - Pass through each preceding layer; if forward accepts training, set it to False
    """
    ### BEGIN SOLUTION role="scaffold"
    sample_inputs = []
    for data in calibration_data[:max_samples]:
        x = data
        for j in range(layer_index):
            layer = model.layers[j]
            # Calibration describes inference, so disable Dropout (also inside
            # nested Sequential containers) using the Module 03 mode flag.
            if 'training' in inspect.signature(layer.forward).parameters:
                x = layer.forward(x, training=False)
            else:
                x = layer.forward(x)
        sample_inputs.append(x)
    return sample_inputs
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Collect Layer Inputs

This test validates that we correctly forward calibration data through preceding layers.

**What we're testing**: Intermediate activation collection for calibration
**Why it matters**: Accurate calibration requires knowing the true input distribution at each layer
**Expected**: Correct number of samples with correct shape after forwarding through preceding layers
"""

# %% nbgrader={"grade": true, "grade_id": "test-collect-layer-inputs", "locked": true, "points": 3}
def test_unit_collect_layer_inputs():
    """🧪 Test collecting intermediate activations for calibration."""
    print("🧪 Unit Test: Collect Layer Inputs...")

    # Create a simple model
    layer1 = Linear(4, 8)
    layer1.weight = Tensor(rng.standard_normal((4, 8)) * 0.5)
    layer1.bias = Tensor(rng.standard_normal(8) * 0.1)
    activation = ReLU()
    layer2 = Linear(8, 3)
    layer2.weight = Tensor(rng.standard_normal((8, 3)) * 0.5)
    layer2.bias = Tensor(rng.standard_normal(3) * 0.1)
    model = Sequential(layer1, activation, layer2)

    samples = [Tensor(rng.standard_normal((1, 4))) for _ in range(5)]

    # Collect inputs for layer at index 0 (no preceding layers)
    inputs_at_0 = _collect_layer_inputs(model, 0, samples)
    assert len(inputs_at_0) == 5
    assert inputs_at_0[0].shape == (1, 4), "Layer 0 inputs should match original shape"

    # Collect inputs for layer at index 2 (after Linear + ReLU)
    inputs_at_2 = _collect_layer_inputs(model, 2, samples)
    assert len(inputs_at_2) == 5
    assert inputs_at_2[0].shape == (1, 8), f"Layer 2 inputs should be (1, 8), got {inputs_at_2[0].shape}"

    # Verify max_samples limiting
    inputs_limited = _collect_layer_inputs(model, 2, samples, max_samples=2)
    assert len(inputs_limited) == 2, "Should respect max_samples"

    print("✅ Collect layer inputs works correctly!")

if __name__ == "__main__":
    test_unit_collect_layer_inputs()

# %% [markdown]
r"""
## 🏗️ Quantizing a Single Layer: The Replacement Step

This helper wraps an isolated `Linear` layer into a `QuantizedLinear` instance, quantizing its weights and biases to INT8 and optionally executing input activation calibration:

| Component | Unquantized `Linear` Layer | Target `QuantizedLinear` Layer | Memory Footprint Impact |
| :--- | :--- | :--- | :--- |
| **Weights ($W$)** | `float32` ($4$ bytes / parameter) | `int8` ($1$ byte / parameter) | $75\%$ reduction ($4\times$ smaller) |
| **Biases ($b$)** | `float32` ($4$ bytes / parameter) | `int8` ($1$ byte / parameter) | $75\%$ reduction ($4\times$ smaller) |
| **Quantization Scales** | None | $s_W, s_b, s_X \in \mathbb{R}^+$ (`float32`) | $+12$ bytes metadata |
| **Zero Points** | None | $z_W, z_b, z_X \in \mathbb{Z}$ (`int32`) | $+12$ bytes metadata |
| **Forward Interface** | `forward(X)` | `forward(X)` (Identical signature) | Drop-in replacement |
"""

# %% nbgrader={"grade": false, "grade_id": "quantize_single_layer", "solution": true}
#| export
def _quantize_single_layer(layer: Linear, calibration_inputs: Optional[List[Tensor]] = None) -> QuantizedLinear:
    """
    Quantize a single Linear layer and optionally calibrate it.

    TODO: Create a QuantizedLinear from a Linear layer, then calibrate if inputs provided

    APPROACH:
    1. Wrap the Linear layer in a QuantizedLinear (quantizes weights/bias)
    2. If calibration_inputs provided, call calibrate() on the quantized layer

    Args:
        layer: Linear layer to quantize
        calibration_inputs: Optional list of activation tensors for calibration

    Returns:
        QuantizedLinear: The quantized replacement layer

    EXAMPLE:
    >>> original = Linear(8, 3)
    >>> original.weight = Tensor(rng.standard_normal((8, 3)) * 0.5)
    >>> quantized = _quantize_single_layer(original)
    >>> print(quantized.q_weight.data.min(), quantized.q_weight.data.max())
    -128.0 127.0

    NOTE: values are INT8-range but stored as float32 (see QuantizedLinear) --
    simulated quantization, so the memory saving is computed, not measured.

    HINT:
    - QuantizedLinear(layer) handles weight/bias quantization
    - quantized_layer.calibrate(inputs) sets input quantization parameters
    """
    ### BEGIN SOLUTION role="scaffold"
    quantized_layer = QuantizedLinear(layer)

    if calibration_inputs is not None:
        quantized_layer.calibrate(calibration_inputs)

    return quantized_layer
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Quantize Single Layer

This test validates that we correctly quantize one Linear layer with optional calibration.

**What we're testing**: Single-layer quantization and calibration
**Why it matters**: This is the atomic building block for full model quantization
**Expected**: INT8 weights, optional calibration parameters set
"""

# %% nbgrader={"grade": true, "grade_id": "test-quantize-single-layer", "locked": true, "points": 3}
def test_unit_quantize_single_layer():
    """🧪 Test single layer quantization with and without calibration."""
    print("🧪 Unit Test: Quantize Single Layer...")

    # Create a linear layer
    layer = Linear(4, 3)
    layer.weight = Tensor(rng.standard_normal((4, 3)) * 0.5)
    layer.bias = Tensor(rng.standard_normal(3) * 0.1)

    # Quantize without calibration
    q_layer = _quantize_single_layer(layer)
    assert isinstance(q_layer, QuantizedLinear)
    assert q_layer.q_weight is not None, "Quantized weights should exist"
    assert q_layer.input_scale is None, "Without calibration, input_scale should be None"

    # Quantize with calibration
    cal_inputs = [Tensor(rng.standard_normal((1, 4))) for _ in range(5)]
    q_layer_cal = _quantize_single_layer(layer, calibration_inputs=cal_inputs)
    assert isinstance(q_layer_cal, QuantizedLinear)
    assert q_layer_cal.input_scale is not None, "With calibration, input_scale should be set"

    # Verify forward pass works
    x = Tensor(rng.standard_normal((2, 4)))
    output = q_layer.forward(x)
    assert output.shape == (2, 3), f"Output shape should be (2, 3), got {output.shape}"

    print("✅ Quantize single layer works correctly!")

if __name__ == "__main__":
    test_unit_quantize_single_layer()

# %% [markdown]
r"""
## 🏗️ Model Quantization: The Composition Function

Now we compose the helpers into the full model quantization pipeline. For each module in the sequential container, `quantize_model` evaluates its architectural type, gathers empirical input distributions, and performs in-place replacement:

| Layer Classification | Evaluation Condition | Applied Transformation | Resulting State in `model.layers[i]` |
| :--- | :--- | :--- | :--- |
| **Parametric Linear Layer** | `isinstance(layer, Linear)` | 1. `_collect_layer_inputs(model, i, data)`<br>2. `_quantize_single_layer(layer, inputs)` | In-place replaced with `QuantizedLinear` |
| **Nested Subgraph** | `hasattr(layer, 'layers')` | Recursive invocation: `quantize_model(child, inputs)` | Nested Linear layers quantized |
| **Stateless Activation** | `ReLU`, `Dropout`, etc. | No-op (preserved without modification) | Retained as-is (0 parameters) |
"""

# %% nbgrader={"grade": false, "grade_id": "quantize_model", "solution": true}
#| export
def quantize_model(model, calibration_data: Optional[List[Tensor]] = None) -> None:
    """
    Quantize all Linear layers in a model in-place, including nested containers.

    TODO: Replace all Linear layers with QuantizedLinear versions

    APPROACH:
    1. Validate model has .layers attribute (Sequential pattern)
    2. Iterate through layers, find Linear layers
    3. For each Linear layer, collect calibration inputs (if data provided)
    4. Replace Linear layers; recurse into nested containers with their input samples

    Args:
        model: Model to quantize (with .layers or similar structure)
        calibration_data: Optional list of sample inputs for calibration

    Returns:
        None (modifies model in-place)

    EXAMPLE:
    >>> layer1 = Linear(10, 5)
    >>> activation = ReLU()
    >>> layer2 = Linear(5, 2)
    >>> model = Sequential(layer1, activation, layer2)
    >>> quantize_model(model)
    >>> # Now model uses quantized layers

    HINT:
    - Use _collect_layer_inputs() to get calibration activations
    - Use _quantize_single_layer() to create the replacement
    """
    ### BEGIN SOLUTION role="scaffold"
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Linear) or hasattr(layer, 'layers'):
                # Collect calibration inputs if data provided
                cal_inputs = None
                if calibration_data is not None:
                    cal_inputs = _collect_layer_inputs(model, i, calibration_data)

                # Replace with quantized version
                if isinstance(layer, Linear):
                    model.layers[i] = _quantize_single_layer(layer, cal_inputs)
                else:
                    quantize_model(layer, cal_inputs)

    elif isinstance(model, Linear):
        raise ValueError(
            f"Cannot quantize single Linear layer in-place\n"
            f"  ❌ quantize_model() modifies models in-place, but a single layer has no container to modify\n"
            f"  💡 In-place modification requires a container (like Sequential) that holds layer references\n"
            f"  🔧 Use QuantizedLinear directly: quantized_layer = QuantizedLinear(your_linear_layer)"
        )

    else:
        raise ValueError(
            f"Unsupported model type for quantization: {type(model).__name__}\n"
            f"  ❌ quantize_model() expects a model with .layers attribute (like Sequential)\n"
            f"  💡 The function iterates through model.layers to find and replace Linear layers\n"
            f"  🔧 Wrap your layers in Sequential: model = Sequential(layer1, activation, layer2)"
        )
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Model Quantization

This test validates our model quantization function correctly replaces Linear layers with QuantizedLinear.

**What we're testing**: Full model quantization and layer replacement
**Why it matters**: Real applications need to quantize entire neural networks
**Expected**: Linear layers replaced, ReLU unchanged, output shape preserved
"""

# %% nbgrader={"grade": true, "grade_id": "test-quantize-model", "locked": true, "points": 5}
def test_unit_quantize_model():
    """🧪 Test model quantization implementation."""
    print("🧪 Unit Test: Model Quantization...")

    # Create test model using explicit layer composition (TinyTorch pattern)
    layer1 = Linear(4, 8)
    activation = ReLU()
    layer2 = Linear(8, 3)

    # Initialize weights
    layer1.weight = Tensor(rng.standard_normal((4, 8)) * 0.5)
    layer1.bias = Tensor(rng.standard_normal(8) * 0.1)
    layer2.weight = Tensor(rng.standard_normal((8, 3)) * 0.5)
    layer2.bias = Tensor(rng.standard_normal(3) * 0.1)

    # Use Sequential from tinytorch.core.layers
    model = Sequential(layer1, activation, layer2)

    # Test original model
    x = Tensor(rng.standard_normal((2, 4)))
    original_output = model.forward(x)

    # Create calibration data. Calibration fixes each layer's input range, and
    # an input outside that range saturates, so the set must cover the inputs
    # the layer will see; the batch under test is included so this check
    # measures rounding error, not saturation (the calibration exercise in the
    # book shows what saturation does).
    calibration_data = [x] + [Tensor(rng.standard_normal((1, 4))) for _ in range(5)]

    # Quantize model
    quantize_model(model, calibration_data)

    # Verify layers were replaced
    assert isinstance(model.layers[0], QuantizedLinear)
    assert isinstance(model.layers[1], ReLU)  # Should remain unchanged
    assert isinstance(model.layers[2], QuantizedLinear)

    # Test quantized model
    quantized_output = model.forward(x)

    # Compare outputs
    error = np.mean(np.abs(original_output.data - quantized_output.data))
    print(f"  Model quantization error: {error:.4f}")
    assert error < 0.2, f"Model quantization error too high: {error}"

    print("✅ Model quantization works correctly!")

if __name__ == "__main__":
    test_unit_quantize_model()

# %% [markdown]
r"""
## 🏗️ Model Size Comparison: Measuring the Impact

To quantify memory savings between unquantized FP32 and quantized architectures, we conduct hierarchical byte accounting:

1. **Per-Layer Profiling (`_measure_layer_bytes`)**: Quantify parameters and footprint for each layer type (FP32 vs INT8).
2. **Model-Level Aggregation (`analyze_model_sizes`)**: Sum parameter counts and memory bytes across all constituent modules.

| Layer Type | Storage Format | Parameter Measurement Formula | Modeled Byte Footprint |
| :--- | :--- | :--- | :--- |
| **FP32 `Linear`** | IEEE 754 float32 | $(d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}})$ | $\text{params} \times 4 \text{ bytes}$ |
| **`QuantizedLinear`** | INT8 codes + metadata | $(d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}})$ | $\text{params} \times 1 \text{ byte} + \text{metadata overhead}$ |
| **Non-Parametric Layer (`ReLU`)** | Stateless | $0$ | $0 \text{ bytes}$ |

## 🏗️ Measuring a Single Layer: Per-Layer Byte Accounting

The helper `_measure_layer_bytes` encapsulates the difference in storage representation between FP32 and quantized layers:

| Layer Category | Parameter Count ($P$) | Byte Footprint Formula | Metadata Overhead ($\mathcal{O}(1)$) |
| :--- | :--- | :--- | :--- |
| **Standard `Linear`** | $N_{\text{weights}} + M_{\text{biases}}$ | $(N + M) \times 4 \text{ bytes}$ | $0 \text{ bytes}$ |
| **`QuantizedLinear`** | $N_{\text{weights}} + M_{\text{biases}}$ | $(N + M) \times 1 \text{ byte} + \text{metadata}$ | $\sim 8\text{--}16 \text{ bytes}$ ($s_W, z_W, s_b, z_b$) |
| **Container (`Sequential`)** | $\sum_{\ell} P_\ell$ | $\sum_{\ell} \text{bytes}(\ell)$ | Sum of child overheads |
"""

# %% nbgrader={"grade": false, "grade_id": "measure_layer_bytes", "solution": true}
#| export
def _measure_layer_bytes(layer, is_quantized: bool = False) -> Tuple[int, int]:
    """
    Measure parameter count and byte usage for a single layer.

    TODO: Count parameters and bytes differently for quantized vs FP32 layers

    APPROACH:
    1. If QuantizedLinear: use layer.memory_usage() for accurate byte count
    2. If regular layer with parameters: count params × BYTES_PER_FLOAT32
    3. If no parameters (e.g., ReLU): return (0, 0)

    Args:
        layer: A single layer (Linear, QuantizedLinear, ReLU, etc.)
        is_quantized: Whether to measure as quantized (uses memory_usage() for QuantizedLinear)

    Returns:
        Tuple of (param_count, byte_count)

    EXAMPLE:
    >>> linear = Linear(100, 50)
    >>> params, bytes_ = _measure_layer_bytes(linear)
    >>> print(f"Params: {params}, Bytes: {bytes_}")
    Params: 5050, Bytes: 20200

    HINT:
    - QuantizedLinear.memory_usage() returns a dict with 'quantized_bytes'
    - Regular layers: sum param.data.size for count, multiply by BYTES_PER_FLOAT32 for bytes
    """
    ### BEGIN SOLUTION role="core"
    if hasattr(layer, 'layers'):
        measurements = [_measure_layer_bytes(child, is_quantized) for child in layer.layers]
        return sum(p for p, _ in measurements), sum(b for _, b in measurements)

    if is_quantized and isinstance(layer, QuantizedLinear):
        memory_info = layer.memory_usage()
        param_count = sum(p.data.size for p in layer.parameters())
        return param_count, memory_info['quantized_bytes']

    if hasattr(layer, 'parameters'):
        params = layer.parameters()
        param_count = sum(p.data.size for p in params)
        byte_count = param_count * BYTES_PER_FLOAT32
        return param_count, byte_count

    return 0, 0
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Measure Layer Bytes

This test validates that we correctly measure bytes for both FP32 and quantized layers.

**What we're testing**: Per-layer byte accounting for different layer types
**Why it matters**: Accurate per-layer measurement is needed for reliable compression metrics
**Expected**: FP32 layers use 4 bytes/param, quantized layers use ~1 byte/param + overhead
"""

# %% nbgrader={"grade": true, "grade_id": "test-measure-layer-bytes", "locked": true, "points": 3}
def test_unit_measure_layer_bytes():
    """🧪 Test per-layer byte measurement for FP32 and quantized layers."""
    print("🧪 Unit Test: Measure Layer Bytes...")

    # Test FP32 Linear layer
    linear = Linear(10, 5)
    linear.weight = Tensor(rng.standard_normal((10, 5)))
    linear.bias = Tensor(rng.standard_normal(5))
    params, bytes_ = _measure_layer_bytes(linear)
    assert params == 55, f"Expected 55 params (10*5 + 5), got {params}"
    assert bytes_ == 55 * BYTES_PER_FLOAT32, f"Expected {55 * BYTES_PER_FLOAT32} bytes, got {bytes_}"

    # Test ReLU (no parameters)
    relu = ReLU()
    params_relu, bytes_relu = _measure_layer_bytes(relu)
    assert params_relu == 0, "ReLU should have 0 params"
    assert bytes_relu == 0, "ReLU should have 0 bytes"

    # Test QuantizedLinear layer
    q_linear = QuantizedLinear(linear)
    params_q, bytes_q = _measure_layer_bytes(q_linear, is_quantized=True)
    assert params_q > 0, "QuantizedLinear should have params"
    assert bytes_q < bytes_, f"Quantized bytes ({bytes_q}) should be less than FP32 ({bytes_})"

    print(f"  FP32: {params} params, {bytes_} bytes")
    print(f"  INT8: {params_q} params, {bytes_q} bytes")
    print(f"  Ratio: {bytes_ / bytes_q:.1f}x")
    print("✅ Measure layer bytes works correctly!")

if __name__ == "__main__":
    test_unit_measure_layer_bytes()

# %% [markdown]
r"""
## 🏗️ Model Size Analysis: The Composition Function

The aggregation function `analyze_model_sizes` traverses original and quantized network graphs in parallel, computing total parameter counts, memory footprints in megabytes, and the overall compression ratio:

| Metric Name | Mathematical Definition | Physical Systems Meaning |
| :--- | :--- | :--- |
| **`original_mb`** | $\sum_{\ell \in \mathcal{M}_{\text{orig}}} \text{bytes}(\ell) / 1024^2$ | Total memory consumed by FP32 baseline weights |
| **`quantized_mb`** | $\sum_{\ell \in \mathcal{M}_{\text{quant}}} \text{bytes}(\ell) / 1024^2$ | Memory consumed by modeled INT8 packed weights + metadata |
| **`compression_ratio`** | $\text{original\_bytes} / \text{quantized\_bytes}$ | Multiplicative factor of model size reduction ($\approx 3.8\times \text{--} 4.0\times$) |
| **`memory_saved_mb`** | $\text{original\_mb} - \text{quantized\_mb}$ | Physical memory footprint liberated for KV cache or other models |
"""

# %% nbgrader={"grade": false, "grade_id": "analyze_model_sizes", "solution": true}
#| export

def analyze_model_sizes(original_model, quantized_model) -> Dict[str, float]:
    """
    Compare memory usage between original and quantized models.

    TODO: Aggregate per-layer measurements and compute compression metrics

    APPROACH:
    1. Iterate original model layers, measure each with _measure_layer_bytes()
    2. Iterate quantized model layers, measure each (with is_quantized=True for QuantizedLinear)
    3. Compute compression ratio and savings from the totals

    Args:
        original_model: Model before quantization
        quantized_model: Model after quantization

    Returns:
        Dictionary with compression metrics

    EXAMPLE:
    >>> layer1 = Linear(100, 50)
    >>> layer2 = Linear(50, 10)
    >>> model = Sequential(layer1, layer2)
    >>> quantize_model(model)
    >>> stats = analyze_model_sizes(model, model)
    >>> print(f"Reduced to {stats['compression_ratio']:.1f}x smaller")

    HINT:
    - Use _measure_layer_bytes(layer) for original FP32 layers
    - Use _measure_layer_bytes(layer, is_quantized=True) for quantized layers
    """
    ### BEGIN SOLUTION role="core"
    # Measure original model
    original_params = 0
    original_bytes = 0
    for layer in original_model.layers:
        p, b = _measure_layer_bytes(layer, is_quantized=False)
        original_params += p
        original_bytes += b

    # Measure quantized model
    quantized_params = 0
    quantized_bytes = 0
    for layer in quantized_model.layers:
        p, b = _measure_layer_bytes(layer, is_quantized=True)
        quantized_params += p
        quantized_bytes += b

    compression_ratio = original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0
    memory_saved = original_bytes - quantized_bytes

    return {
        'original_params': original_params,
        'quantized_params': quantized_params,
        'original_bytes': original_bytes,
        'quantized_bytes': quantized_bytes,
        'compression_ratio': compression_ratio,
        'memory_saved_mb': memory_saved / MB_TO_BYTES,
        'memory_saved_percent': (memory_saved / original_bytes) * 100 if original_bytes > 0 else 0
    }
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Model Size Analysis

This test validates our model size analysis function correctly measures compression.

**What we're testing**: Memory comparison between original and quantized models
**Why it matters**: Need accurate metrics to verify quantization benefits
**Expected**: Compression ratio > 2x and significant memory savings
"""

# %% nbgrader={"grade": true, "grade_id": "test-compare-sizes", "locked": true, "points": 5}
def test_unit_analyze_model_sizes():
    """🧪 Test model size analysis."""
    print("🧪 Unit Test: Model Size Analysis...")

    # Create and quantize a model for testing (using Sequential from tinytorch.core.layers)
    layer1_orig = Linear(100, 50)
    activation_orig = ReLU()
    layer2_orig = Linear(50, 10)
    layer1_orig.weight = Tensor(rng.standard_normal((100, 50)))
    layer1_orig.bias = Tensor(rng.standard_normal(50))
    layer2_orig.weight = Tensor(rng.standard_normal((50, 10)))
    layer2_orig.bias = Tensor(rng.standard_normal(10))
    original_model = Sequential(layer1_orig, activation_orig, layer2_orig)

    # Create quantized copy
    layer1_quant = Linear(100, 50)
    activation_quant = ReLU()
    layer2_quant = Linear(50, 10)
    layer1_quant.weight = Tensor(rng.standard_normal((100, 50)))
    layer1_quant.bias = Tensor(rng.standard_normal(50))
    layer2_quant.weight = Tensor(rng.standard_normal((50, 10)))
    layer2_quant.bias = Tensor(rng.standard_normal(10))
    quantized_model = Sequential(layer1_quant, activation_quant, layer2_quant)

    quantize_model(quantized_model)

    # Analyze sizes
    comparison = analyze_model_sizes(original_model, quantized_model)

    # Verify compression achieved
    assert comparison['compression_ratio'] > 2.0, "Should achieve significant compression"
    assert comparison['memory_saved_percent'] > 50, "Should save >50% memory"

    print(f"  Compression ratio: {comparison['compression_ratio']:.1f}x")
    print(f"  Memory saved: {comparison['memory_saved_percent']:.1f}%")
    print("✅ Model size analysis works correctly!")

if __name__ == "__main__":
    test_unit_analyze_model_sizes()

# %% [markdown]
"""
## 🔧 Integration: The Quantizer Class

Now that we've implemented all quantization components, let's create consolidated classes
for export to the tinytorch package. This allows milestones to use the complete quantization system.
"""

# %% nbgrader={"grade": false, "grade_id": "quantization_export", "solution": false}
#| export
class Quantizer:
    """
    Complete quantization system for milestone use.

    Provides simulated INT8 parameter artifacts and modeled packed storage reports.

    This class delegates to the standalone functions (quantize_int8, dequantize_int8)
    that students implement, providing a clean OOP interface for milestones.

    Two APIs exist for different use cases:
    - Standalone quantize_model(): Modifies model in-place (for learning/testing)
    - Quantizer.quantize_model(): Returns stats dict (for milestones/benchmarking)
    """

    @staticmethod
    def quantize_tensor(tensor: Tensor) -> Tuple[Tensor, float, int]:
        """Quantize FP32 tensor to INT8. Delegates to quantize_int8()."""
        return quantize_int8(tensor)

    @staticmethod
    def dequantize_tensor(q_tensor: Tensor, scale: float, zero_point: int) -> Tensor:
        """Dequantize INT8 tensor back to FP32. Delegates to dequantize_int8()."""
        return dequantize_int8(q_tensor, scale, zero_point)

    @staticmethod
    def quantize_model(model, calibration_data: Optional[List[Tensor]] = None) -> Dict[str, Any]:
        """
        Return quantized parameter artifacts and modeled packed storage statistics.

        This does not replace layers or produce an executable model. Use the
        standalone quantize_model() for that. calibration_data is retained for
        API compatibility; this parameter-only report does not calibrate activations.

        Returns:
            Dict with quantized_layers, original_size_mb, quantized_size_mb, compression_ratio
        """
        quantized_layers = {}
        original_size = 0
        total_elements = 0
        param_idx = 0

        # The container API also handles nested models and shared parameters.
        for param in model.parameters():
            param_size = param.data.nbytes
            original_size += param_size
            total_elements += param.data.size

            # Quantize parameter using the standalone function
            q_param, scale, zp = quantize_int8(param)

            quantized_layers[f'param_{param_idx}'] = {
                'quantized': q_param,
                'scale': scale,
                'zero_point': zp,
                'original_shape': param.data.shape
            }
            param_idx += 1

        # Packed codes plus one float32 scale and int32 zero point per array.
        quantized_size = total_elements * BYTES_PER_INT8 + param_idx * 2 * BYTES_PER_FLOAT32

        return {
            'quantized_layers': quantized_layers,
            'original_size_mb': original_size / MB_TO_BYTES,
            'quantized_size_mb': quantized_size / MB_TO_BYTES,
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 1.0
        }

    @staticmethod
    def compare_models(original_model, quantized_info: Dict) -> Dict[str, float]:
        """Compare memory usage between original and quantized models."""
        return {
            'original_mb': quantized_info['original_size_mb'],
            'quantized_mb': quantized_info['quantized_size_mb'],
            'compression_ratio': quantized_info['compression_ratio'],
            'memory_saved_mb': quantized_info['original_size_mb'] - quantized_info['quantized_size_mb']
        }

# Note: quantize_int8, dequantize_int8, and quantize_model are defined earlier in this module.
# The Quantizer class above delegates to those functions, providing an OOP interface for milestones.

# %% [markdown]
"""
## 📊 Systems Analysis: Quantization in Production

Now let's measure the real-world impact of quantization through systematic analysis.
"""

# %%
def analyze_quantization_memory():
    """Analyze memory reduction across different model sizes."""
    print("Analyzing Quantization Memory Reduction")

    model_sizes = [
        ("Small", 1_000_000),
        ("Medium", 10_000_000),
        ("Large", 100_000_000)
    ]

    print(f"{'Model':<10} {'FP32 (MB)':<12} {'INT8 (MB)':<12} {'Reduction':<12}")
    print("-" * 50)

    for name, params in model_sizes:
        fp32_mb = params * BYTES_PER_FLOAT32 / MB_TO_BYTES
        int8_mb = params * BYTES_PER_INT8 / MB_TO_BYTES
        reduction = fp32_mb / int8_mb

        print(f"{name:<10} {fp32_mb:>10.1f}  {int8_mb:>10.1f}  {reduction:>10.1f}x")

    print("\nKey Insight: packed code storage is 4x smaller, before metadata overhead")
    print("This enables deployment on memory-constrained devices")

if __name__ == "__main__":
    analyze_quantization_memory()

# %%
def analyze_quantization_accuracy():
    """Measure how much quantizing each layer perturbs the model output."""
    print("\nMeasuring Per-Layer Quantization Sensitivity")

    # A small MLP: quantize one Linear layer at a time and compare outputs
    layers = [Linear(64, 32), ReLU(), Linear(32, 16), ReLU(), Linear(16, 10)]
    x = Tensor(rng.standard_normal((32, 64)))
    baseline = Sequential(*layers).forward(x).data
    baseline_power = np.mean(baseline ** 2)

    print(f"{'Layer':<12} {'Params':<10} {'Relative output error'}")
    print("-" * 50)

    for idx, layer in enumerate(layers):
        if not isinstance(layer, Linear):
            continue
        swapped = list(layers)
        swapped[idx] = _quantize_single_layer(layer)
        output = Sequential(*swapped).forward(x).data
        rel_error = np.mean((output - baseline) ** 2) / baseline_power
        params = sum(p.data.size for p in layer.parameters())
        print(f"Linear {idx:<5} {params:<10,} {rel_error:.2e}")

    print("\nKey Insight: per-tensor INT8 perturbs the output by a small, measurable amount.")
    print("Measure sensitivity like this before deciding which layers to keep in FP32.")

if __name__ == "__main__":
    analyze_quantization_accuracy()

# %% [markdown]
r"""
### Advanced Quantization Strategies: Production Techniques

Production inference engines (e.g. TensorRT, llama.cpp, vLLM) employ tailored quantization strategies depending on hardware instruction sets and layer sensitivity:

| Strategy | Granularity | Mathematical Formulation | Hardware / GEMM Cost | Precision Trade-Off |
| :--- | :--- | :--- | :--- | :--- |
| **Per-Tensor (Ours)** | 1 scale & zero-point per entire weight matrix | $s_W = \frac{\max(W) - \min(W)}{255}$ | $\mathcal{O}(1)$ scale metadata; single scalar broadcast | Simple and fast; sensitive to cross-channel outlier values |
| **Per-Channel (Per-Column/Row)** | 1 scale & zero-point per output feature slice | $s_j = \frac{\max(W_{:, j}) - \min(W_{:, j})}{255}$ | $\mathcal{O}(d_{\text{out}})$ scales; vector scaling post-accumulation | Isolates large weight outliers to single channels; standard in LLMs |
| **Mixed Precision** | Selective bitwidths ($32\text{b}, 16\text{b}, 8\text{b}, 4\text{b}$) per module | $q_\ell \in \{\text{FP32}, \text{FP16}, \text{INT8}, \text{INT4}\}$ | Requires mixed-precision dispatch and typecasting buffers | Optimal accuracy-footprint Pareto frontier; protects sensitive layers |

### Granularity Comparison: Per-Tensor vs Per-Channel

$$\text{Per-Tensor}: \quad \hat{W} = (Q_W - z_W) \cdot s_W, \quad s_W \in \mathbb{R}$$

$$\text{Per-Channel}: \quad \hat{W}_{i, j} = (Q_{W, i, j} - z_{W, j}) \cdot s_{W, j}, \quad \mathbf{s}_W \in \mathbb{R}^{d_{\text{out}}}$$

### Mixed Precision Strategy

Empirical sensitivity profiling reveals that certain network components (e.g., token embeddings, self-attention query-key dot products, and final classification projections) degrade catastrophically under 8-bit quantization. A mixed-precision schedule maintains those sensitive boundaries in FP16/FP32 while quantizing bulky feed-forward projections ($W_1, W_2$) to INT8 or INT4:

| Layer Category | Typical Sensitivity | Recommended Precision | System Rationale |
| :--- | :--- | :--- | :--- |
| **Input / Embedding** | High | FP16 / FP32 | Preserves input feature coordinate resolution |
| **Attention Projections ($Q, K, V$)** | Moderate | INT8 / FP16 | Preserves dynamic range for scaled dot-product softmax |
| **MLP / Feed-Forward Bulk** | Low to Moderate | INT8 / INT4 | Dominates parameter count ($>65\%$ of model); maximal compression gain |
| **Output Classification Head** | Very High | FP16 / FP32 | Prevents logit divergence and classification calibration shift |
"""

# %% [markdown]
"""
### Measuring Quantization Savings with Profiler

Now let's use the Profiler tool from Module 14 to measure the actual memory savings from quantization. This demonstrates end-to-end workflow: profile baseline (M14) -> apply quantization (M15) -> measure savings (M14+M15).

This is the production workflow: measure -> compress -> validate -> deploy.
"""

# %% nbgrader={"grade": false, "grade_id": "demo-profiler-quantization", "solution": false}
# Import Profiler from Module 14
from tinytorch.perf.profiling import Profiler

def explore_quantization_with_profiler():
    """Demonstrate memory savings using Profiler from Module 14."""
    print("Measuring Quantization Memory Savings with Profiler")
    print("=" * 70)

    profiler = Profiler()

    # Create a simple model
    model = Linear(512, 256)

    print("\nBEFORE: FP32 Model")
    print("-" * 70)

    # Measure baseline
    param_count = profiler.count_parameters(model)
    input_shape = (32, 512)
    memory_stats = profiler.measure_memory(model, input_shape)

    print(f"   Parameters: {param_count:,}")
    print(f"   Parameter memory: {memory_stats['parameter_memory_mb']:.2f} MB")
    print(f"   Peak memory: {memory_stats['peak_memory_mb']:.2f} MB")
    print(f"   Precision: FP32 (4 bytes per parameter)")

    # Quantize the layer
    print("\nQuantizing to INT8...")
    quantized_model = QuantizedLinear(model)
    usage = quantized_model.memory_usage()

    print("\nAFTER: INT8 Quantized Model")
    print("-" * 70)

    # INT8 storage is simulated here, so memory_usage() reports the analytic byte count
    quantized_param_count = profiler.count_parameters(quantized_model)
    theoretical_memory_mb = usage['quantized_bytes'] / MB_TO_BYTES

    print(f"   Parameters: {quantized_param_count:,} (same count, different precision)")
    print(f"   Parameter memory (theoretical): {theoretical_memory_mb:.2f} MB")
    print(f"   Precision: INT8 (1 byte per parameter)")

    print("\nMEMORY SAVINGS")
    print("=" * 70)
    savings_ratio = memory_stats['parameter_memory_mb'] / theoretical_memory_mb
    savings_percent = (1 - 1/savings_ratio) * 100
    savings_mb = memory_stats['parameter_memory_mb'] - theoretical_memory_mb

    print(f"   Compression ratio: {savings_ratio:.1f}x smaller")
    print(f"   Memory saved: {savings_mb:.2f} MB ({savings_percent:.1f}% reduction)")
    print(f"   Original: {memory_stats['parameter_memory_mb']:.2f} MB -> Quantized: {theoretical_memory_mb:.2f} MB")

    print("\nKey Insight:")
    print(f"   INT8 quantization reduces memory by 4x (FP32 -> INT8)")
    print("   Packed weights save parameter storage; activation memory and runtime costs differ.")
    print(f"   Critical for edge devices with limited memory (mobile, IoT)")
    print("\nThese are modeled packed-storage savings; measure output error separately.")

if __name__ == "__main__":
    explore_quantization_with_profiler()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.
"""

# %% nbgrader={"grade": true, "grade_id": "test_module", "locked": true, "points": 20}
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
    test_unit_quantize_int8()
    test_unit_dequantize_int8()
    test_unit_quantized_linear()
    test_unit_collect_layer_inputs()
    test_unit_quantize_single_layer()
    test_unit_quantize_model()
    test_unit_measure_layer_bytes()
    test_unit_analyze_model_sizes()

    print("\nRunning integration scenarios...")

    # Test realistic usage scenario
    print("Integration Test: End-to-end quantization workflow...")

    # Create a realistic model using explicit composition (Sequential from tinytorch.core.layers)
    layer1 = Linear(784, 128)  # MNIST-like input
    activation1 = ReLU()
    layer2 = Linear(128, 64)
    activation2 = ReLU()
    layer3 = Linear(64, 10)     # 10-class output
    model = Sequential(layer1, activation1, layer2, activation2, layer3)

    # Initialize with realistic weights
    for layer in [layer1, layer2, layer3]:
        if isinstance(layer, Linear):
            # Xavier initialization
            fan_in, fan_out = layer.weight.shape
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight = Tensor(rng.standard_normal((fan_in, fan_out)) * std)
            layer.bias = Tensor(np.zeros(fan_out))

    # Generate realistic calibration data
    calibration_data = [Tensor(rng.standard_normal((1, 784)) * 0.1) for _ in range(20)]

    # Test original model
    test_input = Tensor(rng.standard_normal((8, 784)) * 0.1)
    original_output = model.forward(test_input)

    # Quantize the model
    quantize_model(model, calibration_data)

    # Test quantized model
    quantized_output = model.forward(test_input)

    # Verify functionality is preserved
    assert quantized_output.shape == original_output.shape, "Output shape mismatch"

    # Verify reasonable accuracy preservation
    mse = np.mean((original_output.data - quantized_output.data) ** 2)
    relative_error = np.sqrt(mse) / (np.std(original_output.data) + EPSILON)
    assert relative_error < 0.1, f"Accuracy degradation too high: {relative_error:.3f}"

    # Verify memory savings
    # Create equivalent original model for comparison
    orig_layer1 = Linear(784, 128)
    orig_act1 = ReLU()
    orig_layer2 = Linear(128, 64)
    orig_act2 = ReLU()
    orig_layer3 = Linear(64, 10)
    original_model = Sequential(orig_layer1, orig_act1, orig_layer2, orig_act2, orig_layer3)

    for i, layer in enumerate(model.layers):
        if isinstance(layer, QuantizedLinear):
            # Restore original weights for comparison
            original_model.layers[i].weight = dequantize_int8(
                layer.q_weight, layer.weight_scale, layer.weight_zero_point
            )
            if layer.q_bias is not None:
                original_model.layers[i].bias = dequantize_int8(
                    layer.q_bias, layer.bias_scale, layer.bias_zero_point
                )

    memory_comparison = analyze_model_sizes(original_model, model)
    assert memory_comparison['compression_ratio'] > 2.0, "Insufficient compression achieved"

    print(f"Compression achieved: {memory_comparison['compression_ratio']:.1f}x")
    print(f"Accuracy preserved: {relative_error:.1%} relative error")
    print(f"Memory saved: {memory_comparison['memory_saved_mb']:.1f}MB")

    # Test edge cases
    print("Testing edge cases...")

    # Test constant tensor quantization
    constant_tensor = Tensor([[1.0, 1.0], [1.0, 1.0]])
    q_const, scale_const, zp_const = quantize_int8(constant_tensor)
    assert scale_const == 1.0, "Constant tensor quantization failed"

    # Test zero tensor
    zero_tensor = Tensor([[0.0, 0.0], [0.0, 0.0]])
    q_zero, scale_zero, zp_zero = quantize_int8(zero_tensor)
    restored_zero = dequantize_int8(q_zero, scale_zero, zp_zero)
    assert np.allclose(restored_zero.data, 0.0, atol=1e-6), "Zero tensor restoration failed"

    print("Edge cases handled correctly!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 15")

# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

### Question 1: Memory Architecture Impact
You modeled packed INT8 storage that uses 1 byte per code instead of 4 bytes per FP32 parameter (before metadata).
For a model with 100M parameters:
- **Original memory usage**: $100 \times 10^6 \times 4 \text{ bytes} = 400{,}000{,}000 \text{ bytes} \approx \mathbf{0.400 \text{ GB}}$ ($381.47 \text{ MiB}$).
- **Quantized memory usage**: $100 \times 10^6 \times 1 \text{ byte} + \text{metadata} \approx \mathbf{0.100 \text{ GB}}$ ($95.37 \text{ MiB}$).
- **Memory bandwidth reduction when loading from disk/DRAM**: $\mathbf{4.0\times}$ reduction in bytes transferred ($75\%$ bandwidth savings).
- **Architectural Analysis**: On mobile or edge accelerators (e.g. Apple Neural Engine, Raspberry Pi, Qualcomm Hexagon NPU), disk IO and DRAM bus transfer times dominate cold-start latency. Compressing the model from $400\text{ MB}$ to $100\text{ MB}$ cuts loading latency by $\approx 4\times$, while allowing the model to stay resident in smaller hardware cache tiers (such as system-level L3/LLC cache).

### Question 2: Quantization Error Analysis
Your quantization maps a continuous range to 256 discrete values (INT8).
For weights uniformly distributed in $[-0.1, 0.1]$:
- **Quantization scale**:
  $$s = \frac{x_{\max} - x_{\min}}{255} = \frac{0.1 - (-0.1)}{255} = \frac{0.2}{255} \approx \mathbf{0.00078431} \text{ (or } 7.843 \times 10^{-4}\text{)}$$
- **Maximum quantization error**:
  $$|\epsilon|_{\max} = \frac{s}{2} = \frac{0.00078431}{2} \approx \mathbf{0.00039216} \text{ (or } 3.922 \times 10^{-4}\text{)}$$
- **Signal-to-noise ratio approximately**:
  $$\text{SQNR} \approx 6.02 \times b + 1.76 \text{ dB} = 6.02 \times 8 + 1.76 \approx \mathbf{49.92 \text{ dB}}$$
- **Analytical Confirmation**: For a uniform distribution $U(-A, A)$, signal power is $\sigma_x^2 = \frac{(2A)^2}{12} = \frac{0.04}{12} \approx 0.003333$. Rounding noise power is $\sigma_q^2 = \frac{s^2}{12} = \frac{(0.2/255)^2}{12} \approx 5.126 \times 10^{-8}$. Then $\text{SQNR} = 10 \log_{10}\left(\frac{\sigma_x^2}{\sigma_q^2}\right) = 10 \log_{10}(65025) \approx 48.13 \text{ dB}$, confirming excellent reconstruction fidelity with $< 0.04\%$ maximum pointwise distortion.

### Question 3: Hardware Efficiency
Modern processors have specialized INT8 instructions (such as AVX-512 VNNI `VPDPBUSD` or ARM NEON `SDOT`):
- **SIMD operation packing vs FP32**: $\mathbf{4\times}$ more operations per 512-bit vector instruction ($64 \text{ INT8 operations}$ vs $16 \text{ FP32 operations}$ per vector register). Furthermore, with fused multiply-accumulate primitives like VNNI, 4 pairs of 8-bit integers are multiplied and accumulated into 32-bit registers in a single instruction cycle, achieving up to $\mathbf{2\times \text{--} 4\times}$ compute throughput gains over FP32 FMA.
- **Why actual speedup is less than theoretical maximum**:
  1. *Memory vs Compute Bound*: In memory-bandwidth-bound layers (batch size = 1 autoregressive decoding), throughput is limited by DRAM bus throughput rather than ALU saturation.
  2. *Dequantization / Requantization Overhead*: Rescaling INT32 accumulators back to FP32 or INT8 intermediate activations incurs vector instruction overhead.
  3. *Zero-point compensation*: Asymmetric quantization requires subtracting input/weight zero-point cross terms ($\sum X_i z_w$), adding arithmetic instructions.
  4. *Amdahl's Law*: Non-quantized operators (softmax, LayerNorm, GELU, residual additions) remain in FP32/FP16, bounding total system acceleration.
- **What determines whether quantization improves or hurts performance**:
  Operational intensity (FLOPs/byte). If a layer is compute-bound (large batch GEMM), native INT8 tensor instructions deliver $2\times\text{--}4\times$ speedups. If execution is simulated via explicit dequantization into FP32 registers before calling standard FP32 GEMMs, overhead can degrade wall-clock latency even while reducing memory storage.

### Question 4: Calibration Strategy Trade-offs
Your calibration process finds optimal scales using sample data:
- **Too little calibration data**: Risk of *distribution under-coverage and outlier clipping*. If validation samples do not contain the activation extremes seen during real deployment, runtime values will be clipped to $[-128, 127]$, causing catastrophic saturation errors.
- **Too much calibration data**: Cost of *excessive offline profiling latency and memory overhead*, with diminishing accuracy returns and potential overfitting to calibration dataset anomalies.
- **Per-channel vs per-tensor quantization trades**: *Storage and metadata complexity ($\mathcal{O}(d_{\text{out}})$ scales and offsets vs $\mathcal{O}(1)$ scalar parameters)* for *significantly tighter dynamic range fitting per output channel, isolating channel-specific activation outliers and dramatically reducing overall perplexity degradation*.

### Question 5: Production Deployment
In mobile and edge deployment scenarios:
- **When 4x memory reduction is worth <1% accuracy loss**: In almost all mobile, automotive, and edge AI applications where fitting inside available on-device RAM (e.g., $4\text{ GB}$ mobile shared memory) is the binary gatekeeper between running locally vs failing with an Out-Of-Memory (OOM) operating system termination.
- **Why keep certain layers in FP32**: The *first layer (input token/pixel projection)* and the *final classification projection (output logits before softmax)* have disproportionate impact on output entropy and numerical stability. Discretizing output logits often causes drastic probability distribution shifts and ranking errors.
- **How quantization affects battery life**: DRAM access consumes $100\times \text{--} 1000\times$ more energy per bit than on-chip SRAM register access or ALU operations (e.g. $\sim 20\text{--}50\text{ pJ}$ for DRAM read vs $\sim 0.1\text{ pJ}$ for 8-bit integer add). Streaming $1$ byte instead of $4$ bytes per parameter cuts memory bus energy consumption by up to $\mathbf{75\%}$, extending battery runtime on untethered mobile and edge devices.
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Quantization Shrinks Models

**What you built:** An INT8 quantization simulator with calibration and packed-storage accounting.

**Why it matters:** Packing 400MB of FP32 weights as INT8 codes would use 100MB
before metadata. This can help a model fit on an edge device. Accuracy changes
must be measured on representative inputs; the simulator lets you study them.

You now have a simulator for studying quantization error. Deployment would require packed storage and suitable inference kernels.
"""

# %%
def demo_quantization():
    """🎯 See quantization shrink model size."""
    print("🎯 AHA MOMENT: Quantization Shrinks Models")
    print("=" * 45)

    # Create FP32 weights with concrete values
    weights = Tensor(np.array([
        [0.5, -0.3, 0.8, 0.2],
        [-0.2, 0.6, 0.1, -0.7],
        [0.4, -0.5, 0.3, 0.9]
    ]).astype(np.float32))

    original_bytes = weights.data.nbytes

    # Quantize to INT8
    q_weights, scale, zero_point = quantize_int8(weights)
    quantized_bytes = q_weights.data.size  # 1 byte per INT8 element

    # Restore and verify accuracy preservation
    restored = dequantize_int8(q_weights, scale, zero_point)
    error = np.mean(np.abs(weights.data - restored.data))

    print(f"Original FP32: {original_bytes:,} bytes")
    print(f"Modeled packed INT8: {quantized_bytes:,} bytes")
    print(f"Packed storage model: {original_bytes / quantized_bytes:.0f}x smaller (actual Tensor codes remain float32)")
    print(f"INT8 range: [{q_weights.data.min()}, {q_weights.data.max()}]")
    print(f"Restoration error: {error:.6f}")

    print("\n✨ Rounded values with a modeled 4x reduction in packed code storage!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_quantization()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Quantization

You've built an INT8 quantization simulator that measures rounding error and models packed storage savings!

### Key Accomplishments
- Built INT8 quantization with proper scaling and zero-point calculation
- Implemented QuantizedLinear layer with calibration support
- Created model-level quantization for complete neural networks
- Analyzed quantization trade-offs across different distributions and strategies
- Measured the memory savings analytically (INT8 storage is simulated in float32 here)
- All tests pass (validated by `test_module()`)

### Systems Insights Discovered
- Memory scaling: INT8 reduces storage by 4x (32 bits to 8 bits per parameter)
- Calibration trade-offs: Sample data quality affects quantization accuracy
- Hardware efficiency: Native INT8 kernels can improve speed on supported hardware; this simulator measures rounding effects
- Deployment benefits: Smaller models fit on mobile and edge devices

### Ready for Next Steps
Your quantization pipeline simulates reduced precision without retraining. That
makes it the first optimization you would reach for when a model has to fit on
hardware it was not trained on. Packed INT8 weights can approach 4x storage
savings; the accuracy cost depends on the model and calibration data. This
simulator lets you measure that cost before choosing a deployment format.

Export with: `tito module complete 15`

**Next**: Module 16 will add compression: pruning, distillation, and low-rank approximation to shrink models further!
"""
