# Module 15: Quantization

:::{admonition} Module Info
:class: note

**OPTIMIZATION TIER** | Difficulty: ●●●○ | Time: 4-6 hours | Prerequisites: 01-14

**Prerequisites: Modules 01-14** means you should have:
- Built the complete foundation (Tensor through Training)
- Implemented profiling tools to measure memory usage
- Understanding of neural network parameters and forward passes
- Familiarity with memory calculations and optimization trade-offs

If you can profile a model's memory usage and understand the cost of FP32 storage, you're ready.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F15_quantization%2F15_quantization.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/15_quantization/15_quantization.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/15_quantization.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Modern neural networks face a memory wall problem. A BERT model requires 440 MB, GPT-2 needs 6 GB, and GPT-3 demands 700 GB, yet mobile devices have only 4-8 GB of RAM. The culprit? Every parameter uses 4 bytes of FP32 precision, representing values with 32-bit accuracy when 8 bits often suffice. Quantization solves this by converting FP32 weights to INT8, achieving 4× memory reduction with less than 1% accuracy loss.

In this module, you'll build a production-quality INT8 quantization system. You'll implement the core quantization algorithm, create quantized layer classes, and develop calibration techniques that optimize quantization parameters for minimal accuracy degradation. By the end, you'll compress entire neural networks from hundreds of megabytes to a fraction of their original size, enabling deployment on memory-constrained devices.

This isn't just academic compression. Your implementation uses the same symmetric quantization approach deployed in TensorFlow Lite, PyTorch Mobile, and ONNX Runtime, making models small enough to run on phones, IoT devices, and edge hardware without cloud connectivity.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** INT8 quantization with symmetric scaling and zero-point calculation for 4× memory reduction
- **Master** calibration techniques that optimize quantization parameters using sample data distributions
- **Understand** quantization error propagation and accuracy preservation strategies in compressed models
- **Connect** your implementation to production frameworks like TensorFlow Lite and PyTorch quantization APIs
- **Analyze** memory-accuracy trade-offs across different quantization strategies and model architectures
```

## What You'll Build

```{mermaid}
:align: center
:caption: Quantization System
flowchart TB
    subgraph "Quantization System"
        A["quantize_int8()<br/>FP32 → INT8 conversion"]
        B["dequantize_int8()<br/>INT8 → FP32 restoration"]
        C["QuantizedLinear<br/>Quantized layer class"]
        D["quantize_model()<br/>Full network quantization"]
        E["Calibration<br/>Parameter optimization"]
    end

    A --> C
    B --> C
    C --> D
    E --> D

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
    style E fill:#e2d5f1
```

**Implementation roadmap:**

| Step | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `quantize_int8()` | Scale and zero-point calculation, INT8 mapping |
| 2 | `dequantize_int8()` | FP32 restoration with quantization parameters |
| 3 | `QuantizedLinear` | Quantized linear layer with compressed weights |
| 4 | `calibrate()` | Input quantization optimization using sample data |
| 5 | `quantize_model()` | Full model conversion and memory comparison |

**The pattern you'll enable:**
```python
# Compress a 400MB model to 100MB
quantize_model(model, calibration_data=sample_inputs)
# Now model uses 4× less memory with <1% accuracy loss
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Per-channel quantization (PyTorch supports this for finer-grained precision)
- Mixed precision strategies (keeping sensitive layers in FP16/FP32)
- Quantization-aware training (Module 16: Compression covers this)
- INT8 GEMM kernels (production uses specialized hardware instructions like VNNI)

**You are building symmetric INT8 quantization.** Advanced quantization schemes come in production frameworks.

## API Reference

This section provides a quick reference for the quantization functions and classes you'll build. Use this as your guide while implementing and debugging.

### Core Functions

```python
quantize_int8(tensor: Tensor) -> Tuple[Tensor, float, int]
```
Convert FP32 tensor to INT8 with calculated scale and zero-point.

```python
dequantize_int8(q_tensor: Tensor, scale: float, zero_point: int) -> Tensor
```
Restore INT8 tensor to FP32 using quantization parameters.

### QuantizedLinear Class

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `__init__(linear_layer: Linear)` | Create quantized version of Linear layer |
| `calibrate` | `calibrate(sample_inputs: List[Tensor])` | Optimize input quantization using sample data |
| `forward` | `forward(x: Tensor) -> Tensor` | Compute output with quantized weights |
| `memory_usage` | `memory_usage() -> Dict[str, float]` | Calculate memory savings achieved |

### Model Quantization

| Function | Signature | Description |
|----------|-----------|-------------|
| `quantize_model` | `quantize_model(model, calibration_data=None)` | Quantize all Linear layers in-place |
| `analyze_model_sizes` | `analyze_model_sizes(original, quantized)` | Measure compression ratio and memory saved |

## Core Concepts

This section covers the fundamental ideas behind quantization. Understanding these concepts will help you implement efficient model compression and debug quantization errors.

### Precision and Range

Neural networks use FP32 (32-bit floating point) by default, which can represent approximately 4.3 billion unique values across a vast range from 10⁻³⁸ to 10³⁸. This precision is overkill for most inference tasks. Research shows that neural network weights typically cluster in a narrow range like [-3, 3] after training, and networks are naturally robust to small perturbations due to their continuous optimization.

INT8 quantization maps this continuous FP32 range to just 256 discrete values (from -128 to 127). The key insight is that we can preserve model accuracy by carefully choosing how to map these 256 levels across the actual range of values in each tensor. A tensor with values in [-0.5, 0.5] needs different quantization parameters than one with values in [-10, 10].

Consider the storage implications. A single FP32 parameter requires 4 bytes, while INT8 uses 1 byte. For a model with 100 million parameters, this is the difference between 400 MB (FP32) and 100 MB (INT8). The 4× compression ratio is consistent across all model sizes because we're always reducing from 32 bits to 8 bits per value.

### Quantization Schemes

Symmetric quantization uses a linear mapping where FP32 zero maps to INT8 zero (zero-point = 0). This simplifies hardware implementation and works well for weight distributions centered around zero. Asymmetric quantization allows the zero-point to shift, better capturing ranges like [0, 1] or [-1, 3] where the distribution is not symmetric.

Your implementation uses asymmetric quantization for maximum flexibility:

```python
def quantize_int8(tensor: Tensor) -> Tuple[Tensor, float, int]:
    """Quantize FP32 tensor to INT8 using asymmetric quantization."""
    data = tensor.data

    # Step 1: Find dynamic range
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    # Step 2: Handle edge case (constant tensor)
    if abs(max_val - min_val) < EPSILON:
        scale = 1.0
        zero_point = 0
        quantized_data = np.zeros_like(data, dtype=np.int8)
        return Tensor(quantized_data), scale, zero_point

    # Step 3: Calculate scale and zero_point
    scale = (max_val - min_val) / (INT8_RANGE - 1)
    zero_point = int(np.round(INT8_MIN_VALUE - min_val / scale))
    zero_point = int(np.clip(zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE))

    # Step 4: Apply quantization formula
    quantized_data = np.round(data / scale + zero_point)
    quantized_data = np.clip(quantized_data, INT8_MIN_VALUE, INT8_MAX_VALUE).astype(np.int8)

    return Tensor(quantized_data), scale, zero_point
```

The algorithm finds the minimum and maximum values in the tensor, then calculates a scale that maps this range to [-128, 127]. The zero-point determines which INT8 value represents FP32 zero, ensuring minimal quantization error at zero (important for ReLU activations and sparse patterns).

### Scale and Zero-Point

The scale parameter determines how large each INT8 step is in FP32 space. A scale of 0.01 means each INT8 increment represents 0.01 in the original FP32 values. Smaller scales provide finer precision but can only represent a narrower range; larger scales cover wider ranges but sacrifice precision.

The zero-point is an integer offset that shifts the quantization range. For a symmetric distribution like [-2, 2], the zero-point is 0, mapping FP32 zero to INT8 zero. For an asymmetric range like [-1, 3], the zero-point might be 64, ensuring the quantization levels are distributed optimally across the actual data range.

Here's how dequantization reverses the process:

```python
def dequantize_int8(q_tensor: Tensor, scale: float, zero_point: int) -> Tensor:
    """Dequantize INT8 tensor back to FP32."""
    dequantized_data = (q_tensor.data.astype(np.float32) - zero_point) * scale
    return Tensor(dequantized_data)
```

The formula `(quantized - zero_point) × scale` inverts the quantization mapping. If you quantized 2.5 to INT8 value 85 with scale 0.02 and zero-point 60, dequantization computes `(85 - 60) × 0.02 = 0.5`. The round-trip isn't perfect due to quantization being lossy compression, but the error is bounded by the scale value.

### Post-Training Quantization

Post-training quantization converts a pre-trained FP32 model to INT8 without retraining. This is the approach your implementation uses. The QuantizedLinear class wraps existing Linear layers, quantizing their weights and optionally their inputs:

```python
class QuantizedLinear:
    """Quantized version of Linear layer using INT8 arithmetic."""

    def __init__(self, linear_layer: Linear):
        """Create quantized version of existing linear layer."""
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
```

The forward pass dequantizes weights on-the-fly, performs FP32 matrix multiplication, and returns FP32 outputs. This educational approach makes the code simple to understand, though production implementations use INT8 GEMM (general matrix multiply) operations for speed:

```python
def forward(self, x: Tensor) -> Tensor:
    """Forward pass with quantized computation."""
    # Dequantize weights
    weight_fp32 = dequantize_int8(self.q_weight, self.weight_scale, self.weight_zero_point)

    # Perform computation (same as original layer)
    result = x.matmul(weight_fp32)

    # Add bias if it exists
    if self.q_bias is not None:
        bias_fp32 = dequantize_int8(self.q_bias, self.bias_scale, self.bias_zero_point)
        result = Tensor(result.data + bias_fp32.data)

    return result
```

### Calibration Strategy

Calibration is the process of finding optimal quantization parameters by analyzing sample data. Without calibration, generic quantization parameters may waste precision or clip important values. The calibration method in QuantizedLinear runs sample inputs through the layer and collects statistics:

```python
def calibrate(self, sample_inputs: List[Tensor]):
    """Calibrate input quantization parameters using sample data."""
    # Collect all input values
    all_values = []
    for inp in sample_inputs:
        all_values.extend(inp.data.flatten())

    all_values = np.array(all_values)

    # Calculate input quantization parameters
    min_val = float(np.min(all_values))
    max_val = float(np.max(all_values))

    if abs(max_val - min_val) < EPSILON:
        self.input_scale = 1.0
        self.input_zero_point = 0
    else:
        self.input_scale = (max_val - min_val) / (INT8_RANGE - 1)
        self.input_zero_point = int(np.round(INT8_MIN_VALUE - min_val / self.input_scale))
        self.input_zero_point = np.clip(self.input_zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE)
```

Calibration typically requires 100-1000 representative samples. Too few samples might miss important distribution characteristics; too many waste time with diminishing returns. The goal is capturing the typical range of activations the model will see during inference.

## Production Context

### Your Implementation vs. PyTorch

Your quantization system implements the core algorithms used in production frameworks. The main differences are in scale (production supports many quantization schemes) and performance (production uses INT8 hardware instructions).

| Feature | Your Implementation | PyTorch Quantization |
|---------|---------------------|----------------------|
| **Algorithm** | Asymmetric INT8 quantization | Multiple schemes (INT8, INT4, FP16, mixed) |
| **Calibration** | Min/max statistics | MinMax, histogram, percentile observers |
| **Backend** | NumPy (FP32 compute) | INT8 GEMM kernels (FBGEMM, QNNPACK) |
| **Speed** | 1x (baseline) | 2-4× faster with INT8 ops |
| **Memory** | 4× reduction | 4× reduction (same compression) |
| **Granularity** | Per-tensor | Per-tensor, per-channel, per-group |

### Code Comparison

The following comparison shows quantization in TinyTorch versus PyTorch. The APIs are remarkably similar, reflecting the universal nature of the quantization problem.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.perf.quantization import quantize_model, QuantizedLinear
from tinytorch.core.layers import Linear, Sequential

# Create model
model = Sequential(
    Linear(784, 128),
    Linear(128, 10)
)

# Quantize to INT8
calibration_data = [sample_batch1, sample_batch2, ...]
quantize_model(model, calibration_data)

# Use quantized model
output = model.forward(x)  # 4× less memory!
```
````

````{tab-item} ⚡ PyTorch
```python
import torch
import torch.quantization as quantization

# Create model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.Linear(128, 10)
)

# Quantize to INT8
model.qconfig = quantization.get_default_qconfig('fbgemm')
model_prepared = quantization.prepare(model)
# Run calibration
for batch in calibration_data:
    model_prepared(batch)
model_quantized = quantization.convert(model_prepared)

# Use quantized model
output = model_quantized(x)  # 4× less memory!
```
````
`````

Let's walk through the key differences:

- **Line 1-2 (Import)**: TinyTorch uses `quantize_model()` function; PyTorch uses `torch.quantization` module with prepare/convert API.
- **Lines 4-7 (Model creation)**: Both create identical model architectures. The layer APIs are the same.
- **Lines 9-11 (Quantization)**: TinyTorch uses one-step `quantize_model()` with calibration data. PyTorch uses three-step API: configure (`qconfig`), prepare (insert observers), convert (replace with quantized ops).
- **Lines 13 (Calibration)**: TinyTorch passes calibration data as argument; PyTorch requires explicit calibration loop with forward passes.
- **Lines 15-16 (Inference)**: Both use standard forward pass. The quantized weights are transparent to the user.

```{tip} What's Identical

The core quantization mathematics: scale calculation, zero-point mapping, INT8 range clipping. When you debug PyTorch quantization errors, you'll understand exactly what's happening because you implemented the same algorithms.
```

### Why Quantization Matters at Scale

To appreciate why quantization is critical for production ML, consider these deployment scenarios:

- **Mobile AI**: iPhone has 6 GB RAM shared across all apps. A quantized BERT (110 MB) fits comfortably; FP32 version (440 MB) causes memory pressure and swapping.
- **Edge computing**: IoT devices often have 512 MB RAM. Quantization enables on-device inference for privacy-sensitive applications (medical devices, security cameras).
- **Data centers**: Serving 1000 requests/second requires multiple model replicas. With 4× memory reduction, you fit 4× more models per GPU, reducing serving costs by 75%.
- **Battery life**: INT8 operations consume 2-4× less energy than FP32 on mobile processors. Quantized models drain battery slower, improving user experience.

## Check Your Understanding

Test your quantization knowledge with these systems thinking questions. They're designed to build intuition for memory, precision, and performance trade-offs.

**Q1: Memory Calculation**

A neural network has three Linear layers: 784→256, 256→128, 128→10. How much memory do the weights consume in FP32 vs INT8? Include bias terms.

```{admonition} Answer
:class: dropdown

**Parameter count:**
- Layer 1: (784 × 256) + 256 = 200,960
- Layer 2: (256 × 128) + 128 = 32,896
- Layer 3: (128 × 10) + 10 = 1,290
- **Total: 235,146 parameters**

**Memory usage:**
- FP32: 235,146 × 4 bytes = **940,584 bytes ≈ 0.92 MB**
- INT8: 235,146 × 1 byte = **235,146 bytes ≈ 0.23 MB**
- **Savings: 0.69 MB (75% reduction, 4× compression)**

This shows why quantization matters: even small models benefit significantly.
```

**Q2: Quantization Error Bound**

For FP32 weights uniformly distributed in [-0.5, 0.5], what is the maximum quantization error after INT8 quantization? What is the signal-to-noise ratio in decibels?

```{admonition} Answer
:class: dropdown

**Quantization error:**
- Range: 0.5 - (-0.5) = 1.0
- Scale: 1.0 / 255 = **0.003922**
- Max error: scale / 2 = **±0.001961** (half step size)

**Signal-to-noise ratio:**
- SNR = 20 × log₁₀(signal_range / quantization_step)
- SNR = 20 × log₁₀(1.0 / 0.003922)
- SNR = 20 × log₁₀(255)
- SNR ≈ **48 dB**

This is sufficient for neural networks (typical requirement: >40 dB). The 8-bit quantization provides approximately 6 dB per bit, matching the theoretical limit.
```

**Q3: Calibration Strategy**

You're quantizing a model for deployment. You have 100,000 calibration samples available. How many should you use, and why? What's the trade-off?

```{admonition} Answer
:class: dropdown

**Recommended: 100-1000 samples** (typically 500)

**Reasoning:**
- **Too few (<100)**: Risk missing outliers, suboptimal scale/zero-point
- **Too many (>1000)**: Diminishing returns, calibration time wasted
- **Sweet spot (100-1000)**: Captures distribution, fast calibration

**Trade-off analysis:**
- 10 samples: Fast (1 second), but might miss distribution tails → poor accuracy
- 100 samples: Medium (5 seconds), good representation → 98% accuracy
- 1000 samples: Slow (30 seconds), comprehensive → 98.5% accuracy
- 10000 samples: Very slow (5 minutes), overkill → 98.6% accuracy

**Conclusion**: Calibration accuracy plateaus around 100-1000 samples. Use more only if accuracy is critical (medical, autonomous vehicles).
```

**Q4: Memory Bandwidth Impact**

A model has 100M parameters. Loading from SSD to RAM at 500 MB/s, how long does loading take for FP32 vs INT8? How does this affect user experience?

```{admonition} Answer
:class: dropdown

**Loading time:**
- FP32 size: 100M × 4 bytes = 400 MB
- INT8 size: 100M × 1 byte = 100 MB
- FP32 load time: 400 MB / 500 MB/s = **0.8 seconds**
- INT8 load time: 100 MB / 500 MB/s = **0.2 seconds**
- **Speedup: 4× faster loading**

**User experience impact:**
- Mobile app launch: 0.8s → 0.2s (**0.6s faster startup**)
- Cloud inference: 0.8s latency → 0.2s latency (**4× better throughput**)
- Model updates: 400 MB download → 100 MB download (**75% less data usage**)

**Key insight**: Quantization reduces not just RAM usage, but also disk I/O, network transfer, and cold-start latency. The 4× reduction applies to all memory movement operations.
```

**Q5: Hardware Acceleration**

Modern CPUs have AVX-512 VNNI instructions that can perform INT8 matrix multiply. How many INT8 operations fit in one 512-bit SIMD register vs FP32? Why might actual speedup be less than this ratio?

```{admonition} Answer
:class: dropdown

**SIMD capacity:**
- 512-bit register with FP32: 512 / 32 = **16 values**
- 512-bit register with INT8: 512 / 8 = **64 values**
- **Theoretical speedup: 64/16 = 4×**

**Why actual speedup is 2-3× (not 4×):**

1. **Dequantization overhead**: Converting INT8 → FP32 for activations takes time
2. **Memory bandwidth bottleneck**: INT8 ops are so fast, memory can't feed data fast enough
3. **Mixed precision**: Activations often stay FP32, only weights quantized
4. **Non-compute operations**: Batch norm, softmax, etc. remain FP32 (can't quantize easily)

**Real-world speedup breakdown:**
- Compute-bound workloads (large matmuls): **3-4× speedup**
- Memory-bound workloads (small layers): **1.5-2× speedup**
- Typical mixed models: **2-3× average speedup**

**Key insight**: INT8 quantization shines when matrix multiplications dominate (transformers, large MLPs). For convolutional layers with small kernels, memory bandwidth limits speedup.
```

## Further Reading

For students who want to understand the academic foundations and production implementations of quantization:

### Seminal Papers

- **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference** - Jacob et al. (2018). The foundational paper for symmetric INT8 quantization used in TensorFlow Lite. Introduces quantized training and deployment. [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)

- **Mixed Precision Training** - Micikevicius et al. (2018). NVIDIA's approach to training with FP16/FP32 mixed precision, reducing memory and increasing speed. Concepts extend to INT8 quantization. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

- **Data-Free Quantization Through Weight Equalization and Bias Correction** - Nagel et al. (2019). Techniques for quantizing models without calibration data, using statistical properties of weights. [arXiv:1906.04721](https://arxiv.org/abs/1906.04721)

- **ZeroQ: A Novel Zero Shot Quantization Framework** - Cai et al. (2020). Shows how to quantize models without any calibration data by generating synthetic inputs. [arXiv:2001.00281](https://arxiv.org/abs/2001.00281)

### Additional Resources

- **Blog post**: "[Quantization in PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)" - Official PyTorch quantization tutorial covering eager mode and FX graph mode quantization
- **Documentation**: [TensorFlow Lite Post-Training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) - Production quantization techniques for mobile deployment
- **Survey**: "A Survey of Quantization Methods for Efficient Neural Network Inference" - Gholami et al. (2021) - Comprehensive overview of quantization research. [arXiv:2103.13630](https://arxiv.org/abs/2103.13630)

## What's Next

```{seealso} Coming Up: Module 16 - Compression

Implement model pruning and weight compression techniques. You'll build structured pruning that removes entire neurons and channels, achieving 2-10× speedup by reducing computation, not just memory.
```

**Preview - How Quantization Combines with Future Techniques:**

| Module | What It Does | Quantization In Action |
|--------|--------------|------------------------|
| **16: Compression** | Prune unnecessary weights | `quantize_model(pruned_model)` → 16× total compression |
| **17: Acceleration** | Optimize kernel fusion | `accelerate(quantized_model)` → 8× faster inference |
| **20: Capstone** | Deploy optimized models | Full pipeline: prune → quantize → accelerate → deploy |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/15_quantization/15_quantization.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/15_quantization/15_quantization.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
