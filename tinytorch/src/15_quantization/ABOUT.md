---
file_format: mystnb
kernelspec:
  name: python3
---

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
````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/15_quantization.mp3" type="audio/mpeg">
</audio>
```

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F15_quantization%2Fquantization.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/15_quantization/15_quantization.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #14b8a6; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
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

<div class="slide-viewer-container" id="slide-viewer-15_quantization">
  <div class="slide-header">
    <div class="slide-title">
      <span>🔥</span>
      <span>Slide Deck</span>
      <span class="slide-subtitle">· AI-generated</span>
    </div>
    <div class="slide-toolbar">
      <div class="slide-nav-group">
        <button onclick="slideNav('15_quantization', -1)" title="Previous">‹</button>
        <span class="slide-page-info"><span id="slide-num-15_quantization">1</span> / <span id="slide-count-15_quantization">-</span></span>
        <button onclick="slideNav('15_quantization', 1)" title="Next">›</button>
      </div>
      <div class="slide-zoom-group">
        <button onclick="slideZoom('15_quantization', -0.25)" title="Zoom out">−</button>
        <button onclick="slideZoom('15_quantization', 0.25)" title="Zoom in">+</button>
      </div>
    </div>
  </div>
  <div class="slide-canvas-wrapper">
    <div id="slide-loading-15_quantization" class="slide-loading">Loading slides...</div>
    <canvas id="slide-canvas-15_quantization" class="slide-canvas" style="display:none;"></canvas>
  </div>
  <div class="slide-progress-wrapper">
    <div class="slide-progress-bar" onclick="slideProgress('15_quantization', event)">
      <div class="slide-progress-fill" id="slide-progress-15_quantization" style="width: 0%;"></div>
    </div>
  </div>
  <div class="slide-footer">
    <a href="../_static/slides/15_quantization.pdf" download>⬇ Download</a>
    <a href="#" onclick="slideFullscreen('15_quantization'); return false;" class="secondary">⛶ Fullscreen</a>
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

initSlideViewer('15_quantization', '../_static/slides/15_quantization.pdf');
</script>
```
`````

## Overview

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Model sizes in FP32 (4 bytes per parameter)
bert_params = 110_000_000
gpt2_params = 1_500_000_000
gpt3_params = 175_000_000_000
bert_mb = bert_params * 4 / 1024**2
gpt2_gb = gpt2_params * 4 / 1024**3
gpt3_gb = gpt3_params * 4 / 1024**3
glue("bert_mb", f"{bert_mb:.0f} MB")
glue("gpt2_gb", f"{gpt2_gb:.1f} GB")
glue("gpt3_gb", f"{gpt3_gb:.0f} GB")

# Quantized BERT (INT8 = 1 byte per param)
bert_int8_mb = bert_params * 1 / 1024**2
glue("bert_int8_mb", f"{bert_int8_mb:.0f} MB")
```

Modern neural networks face a memory wall problem. A BERT model requires {glue:text}`bert_mb`, GPT-2 needs {glue:text}`gpt2_gb`, and GPT-3 demands {glue:text}`gpt3_gb`, yet mobile devices have only 4-8 GB of RAM. The culprit? Every parameter uses 4 bytes of FP32 precision, representing values with 32-bit accuracy when 8 bits often suffice. Quantization solves this by converting FP32 weights to INT8, achieving 4× memory reduction with less than 1% accuracy loss.

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

### Quantizer Class

```python
Quantizer()
```

Object-oriented interface wrapping the standalone quantization functions. Provides a convenient API for milestone scripts and production workflows.

| Method | Signature | Description |
|--------|-----------|-------------|
| `quantize_model` | `quantize_model(model, calibration_data=None)` | Quantize model via static method |
| `analyze_model_sizes` | `analyze_model_sizes(original, quantized)` | Compare original vs quantized model sizes |

## Core Concepts

This section covers the fundamental ideas behind quantization. Understanding these concepts will help you implement efficient model compression and debug quantization errors.

### Precision and Range

Neural networks use FP32 (32-bit floating point) by default, which can represent approximately 4.3 billion unique values across a vast range from 10⁻³⁸ to 10³⁸. This precision is overkill for most inference tasks. Research shows that neural network weights typically cluster in a narrow range like [-3, 3] after training, and networks are naturally robust to small perturbations due to their continuous optimization.

INT8 quantization maps this continuous FP32 range to just 256 discrete values (from -128 to 127). The key insight is that we can preserve model accuracy by carefully choosing how to map these 256 levels across the actual range of values in each tensor. A tensor with values in [-0.5, 0.5] needs different quantization parameters than one with values in [-10, 10].

Consider the storage implications. A single FP32 parameter requires 4 bytes, while INT8 uses 1 byte. For a model with 100 million parameters, this is the difference between {glue:text}`q4_fp32_mb` (FP32) and {glue:text}`q4_int8_mb` (INT8). The 4× compression ratio is consistent across all model sizes because we're always reducing from 32 bits to 8 bits per value.

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

The zero-point is an integer offset that shifts the quantization range. For a symmetric distribution like [-2, 2], the zero-point is 0, mapping FP32 zero to INT8 zero. For an asymmetric range like [-1, 3], the zero-point is -64, ensuring the quantization levels are distributed optimally across the actual data range.

Here's how dequantization reverses the process:

```python
def dequantize_int8(q_tensor: Tensor, scale: float, zero_point: int) -> Tensor:
    """Dequantize INT8 tensor back to FP32."""
    dequantized_data = (q_tensor.data.astype(np.float32) - zero_point) * scale
    return Tensor(dequantized_data)
```

The formula `(quantized - zero_point) × scale` inverts the quantization mapping. If you quantized 1.5 to INT8 value 50 with scale 0.02 and zero-point -25, dequantization computes `(50 - (-25)) × 0.02 = 1.5`. The round-trip isn't perfect due to quantization being lossy compression, but the error is bounded by the scale value.

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

- **Mobile AI**: iPhone has 6 GB RAM shared across all apps. A quantized BERT ({glue:text}`bert_int8_mb`) fits comfortably; FP32 version ({glue:text}`bert_mb`) causes memory pressure and swapping.
- **Edge computing**: IoT devices often have 512 MB RAM. Quantization enables on-device inference for privacy-sensitive applications (medical devices, security cameras).
- **Data centers**: Serving 1000 requests/second requires multiple model replicas. With 4× memory reduction, you fit 4× more models per GPU, reducing serving costs by 75%.
- **Battery life**: INT8 operations consume 2-4× less energy than FP32 on mobile processors. Quantized models drain battery slower, improving user experience.

## Check Your Understanding

```{code-cell} python3
:tags: [remove-input, remove-output]
import math

# Q1: 3-layer network parameter counting
q1_l1 = 784 * 256 + 256
q1_l2 = 256 * 128 + 128
q1_l3 = 128 * 10 + 10
q1_total = q1_l1 + q1_l2 + q1_l3
q1_fp32_bytes = q1_total * 4
q1_int8_bytes = q1_total * 1
q1_savings = q1_fp32_bytes - q1_int8_bytes
glue("q1_l1", f"{q1_l1:,}")
glue("q1_l2", f"{q1_l2:,}")
glue("q1_l3", f"{q1_l3:,}")
glue("q1_total", f"{q1_total:,}")
glue("q1_fp32_bytes", f"{q1_fp32_bytes:,}")
glue("q1_fp32_mb", f"{q1_fp32_bytes / 1024**2:.2f} MB")
glue("q1_int8_bytes", f"{q1_int8_bytes:,}")
glue("q1_int8_mb", f"{q1_int8_bytes / 1024**2:.2f} MB")
glue("q1_savings_mb", f"{q1_savings / 1024**2:.2f} MB")

# Q2: Quantization error and SNR
q2_range = 1.0
q2_levels = 255
q2_scale = q2_range / q2_levels
q2_max_error = q2_scale / 2
q2_snr = 20 * math.log10(q2_range / q2_scale)
glue("q2_scale", f"{q2_scale:.6f}")
glue("q2_max_error", f"±{q2_max_error:.6f}")
glue("q2_snr", f"{q2_snr:.0f} dB")

# Q4: Loading time
q4_fp32_mb = 100_000_000 * 4 / 1024**2
q4_int8_mb = 100_000_000 * 1 / 1024**2
q4_bandwidth = 500  # MB/s
q4_fp32_time = q4_fp32_mb / q4_bandwidth
q4_int8_time = q4_int8_mb / q4_bandwidth
glue("q4_fp32_mb", f"{q4_fp32_mb:.0f} MB")
glue("q4_int8_mb", f"{q4_int8_mb:.0f} MB")
glue("q4_fp32_time", f"{q4_fp32_time:.1f} seconds")
glue("q4_int8_time", f"{q4_int8_time:.2f} seconds")
glue("q4_time_saved", f"{q4_fp32_time - q4_int8_time:.1f}s")

# Q5: SIMD register capacity
simd_bits = 512
fp32_per_reg = simd_bits // 32
int8_per_reg = simd_bits // 8
glue("q5_fp32", f"{fp32_per_reg}")
glue("q5_int8", f"{int8_per_reg}")
glue("q5_ratio", f"{int8_per_reg // fp32_per_reg}×")
```

Test your quantization knowledge with these systems thinking questions. They're designed to build intuition for memory, precision, and performance trade-offs.

**Q1: Memory Calculation**

A neural network has three Linear layers: 784→256, 256→128, 128→10. How much memory do the weights consume in FP32 vs INT8? Include bias terms.

```{admonition} Answer
:class: dropdown

**Parameter count:**
- Layer 1: (784 × 256) + 256 = {glue:text}`q1_l1`
- Layer 2: (256 × 128) + 128 = {glue:text}`q1_l2`
- Layer 3: (128 × 10) + 10 = {glue:text}`q1_l3`
- **Total: {glue:text}`q1_total` parameters**

**Memory usage:**
- FP32: {glue:text}`q1_total` × 4 bytes = **{glue:text}`q1_fp32_bytes` bytes ≈ {glue:text}`q1_fp32_mb`**
- INT8: {glue:text}`q1_total` × 1 byte = **{glue:text}`q1_int8_bytes` bytes ≈ {glue:text}`q1_int8_mb`**
- **Savings: {glue:text}`q1_savings_mb` (75% reduction, 4× compression)**

This shows why quantization matters: even small models benefit significantly.
```

**Q2: Quantization Error Bound**

For FP32 weights uniformly distributed in [-0.5, 0.5], what is the maximum quantization error after INT8 quantization? What is the signal-to-noise ratio in decibels?

```{admonition} Answer
:class: dropdown

**Quantization error:**
- Range: 0.5 - (-0.5) = 1.0
- Scale: 1.0 / 255 = **{glue:text}`q2_scale`**
- Max error: scale / 2 = **{glue:text}`q2_max_error`** (half step size)

**Signal-to-noise ratio:**
- SNR = 20 × log₁₀(signal_range / quantization_step)
- SNR = 20 × log₁₀(1.0 / {glue:text}`q2_scale`)
- SNR = 20 × log₁₀(255)
- SNR ≈ **{glue:text}`q2_snr`**

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
- FP32 size: 100M × 4 bytes = {glue:text}`q4_fp32_mb`
- INT8 size: 100M × 1 byte = {glue:text}`q4_int8_mb`
- FP32 load time: {glue:text}`q4_fp32_mb` / 500 MB/s = **{glue:text}`q4_fp32_time`**
- INT8 load time: {glue:text}`q4_int8_mb` / 500 MB/s = **{glue:text}`q4_int8_time`**
- **Speedup: 4× faster loading**

**User experience impact:**
- Mobile app launch: {glue:text}`q4_fp32_time` → {glue:text}`q4_int8_time` (**{glue:text}`q4_time_saved` faster startup**)
- Cloud inference: {glue:text}`q4_fp32_time` latency → {glue:text}`q4_int8_time` latency (**4× better throughput**)
- Model updates: {glue:text}`q4_fp32_mb` download → {glue:text}`q4_int8_mb` download (**75% less data usage**)

**Key insight**: Quantization reduces not just RAM usage, but also disk I/O, network transfer, and cold-start latency. The 4× reduction applies to all memory movement operations.
```

**Q5: Hardware Acceleration**

Modern CPUs have AVX-512 VNNI instructions that can perform INT8 matrix multiply. How many INT8 operations fit in one 512-bit SIMD register vs FP32? Why might actual speedup be less than this ratio?

```{admonition} Answer
:class: dropdown

**SIMD capacity:**
- 512-bit register with FP32: 512 / 32 = **{glue:text}`q5_fp32` values**
- 512-bit register with INT8: 512 / 8 = **{glue:text}`q5_int8` values**
- **Theoretical speedup: {glue:text}`q5_int8`/{glue:text}`q5_fp32` = {glue:text}`q5_ratio`**

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

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/15_quantization/quantization.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/15_quantization/15_quantization.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
