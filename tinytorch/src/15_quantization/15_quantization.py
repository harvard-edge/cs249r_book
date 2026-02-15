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

#| default_exp perf.quantization

# %% [markdown]
"""
# Module 15: Quantization - Reduced Precision for Efficiency

Welcome to Module 15! You're about to build a complete INT8 quantization system that can reduce model size by 4x with minimal accuracy loss.

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML pipeline with profiling (Module 14)
**You'll Build**: INT8 quantization system with calibration and memory savings
**You'll Enable**: 4x memory reduction and 2-4x speedup for production deployment

**Connection Map**:
```
Profiling (14) → Quantization (15)
(measure memory)   (reduce precision)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement INT8 quantization with proper scaling
2. Build quantization-aware training for minimal accuracy loss
3. Apply post-training quantization to existing models
4. Measure actual memory and compute savings
5. Understand quantization error and mitigation strategies

Let's make models 4x smaller!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/15_quantization/quantization_dev.py
**Building Side:** Code exports to tinytorch.perf.quantization

```python
# Final package structure:
from tinytorch.perf.quantization import quantize_int8, QuantizedLinear, quantize_model
```

**Why this matters:**
- **Learning:** Complete quantization system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.quantization with all optimization components together
- **Consistency:** All quantization operations and calibration tools in perf.quantization
- **Integration:** Works seamlessly with existing models for complete optimization pipeline
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| export
import numpy as np
import time
from typing import Tuple, Dict, List, Optional
import warnings

# Import dependencies from other modules
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU

# Constants for INT8 quantization
INT8_MIN_VALUE = -128
INT8_MAX_VALUE = 127
INT8_RANGE = 256  # Number of possible INT8 values (from -128 to 127 inclusive)
EPSILON = 1e-8  # Small value for numerical stability (constant tensor detection)

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
BYTES_PER_INT8 = 1  # INT8 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

if __name__ == "__main__":
    print("Quantization module imports complete")

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Module 14 (Profiling) must be complete

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for performance measurements)
- `typing` (for type annotations)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class from Module 01)
- `tinytorch.core.layers` (Linear, Sequential from Module 03)
- `tinytorch.core.activations` (ReLU from Module 02)
- `tinytorch.perf.profiling` (Profiler from Module 14)

**Dependency Flow**:
```
Module 01 (Tensor) → Module 02 (Activations) → Module 03 (Layers)
     ↓                                              ↓
Module 14 (Profiling) ─────────────────────→ Module 15 (Quantization)
```

Students completing this module will have built a complete
quantization system that achieves 4x memory reduction.
"""

# %% [markdown]
"""
## 💡 Motivation: Why Quantization Matters

Before we learn quantization, let's profile a model to see how much memory
FP32 weights actually consume. This will show us why reduced precision matters.
"""

# %%
def explore_motivation_profiling():
    """Profile model memory usage to discover the quantization problem."""
    from tinytorch.perf.profiling import Profiler

    profiler = Profiler()

    # Create models of increasing size
    print("Profiling Memory Usage (FP32 Precision):\n")
    print("   Parameters   |  FP32 Memory  |  Device Fit?")
    print("   -------------|---------------|---------------")

    model_configs = [
        (256, 256, "Tiny"),
        (512, 512, "Small"),
        (1024, 1024, "Medium"),
        (2048, 2048, "Large"),
    ]

    for in_feat, out_feat, name in model_configs:
        model = Linear(in_feat, out_feat)
        input_data = Tensor(np.random.randn(1, in_feat))

        # Profile the model
        profile = profiler.profile_forward_pass(model, input_data)

        params = profile['parameters']
        memory_fp32_mb = params * BYTES_PER_FLOAT32 / MB_TO_BYTES
        memory_fp32_gb = memory_fp32_mb / 1000

        # Check if it fits on different devices
        fits_mobile = "Y" if memory_fp32_mb < 100 else "N"
        fits_edge = "Y" if memory_fp32_mb < 10 else "N"

        print(f"   {params:>10,}  |  {memory_fp32_mb:7.1f} MB  |  Mobile:{fits_mobile} Edge:{fits_edge}")

    print("\nKey Observations:")
    print("   Every parameter uses 4 bytes (32 bits) in FP32")
    print("   Larger models quickly exceed mobile device memory (~100MB limit)")
    print("   Edge devices have even tighter constraints (~10MB)")
    print("   Memory grows linearly with parameter count")

    print("\nThe Problem:")
    print("   Do we really need 32-bit precision for inference?")
    print("   FP32: Can represent 2^32 = 4.3 billion unique values")
    print("   Neural networks are naturally robust to noise")
    print("   Most weights are in range [-3, 3] after training")

    print("\nThe Solution:")
    print("   Quantize to INT8 (8-bit integers):")
    print("   FP32 -> INT8: 32 bits -> 8 bits (4x compression!)")
    print("   Memory: 100MB -> 25MB (now fits on mobile!)")
    print("   Speed: INT8 operations are 2-4x faster on hardware")
    print("   Accuracy: Minimal loss (<1% typically) with proper calibration\n")

if __name__ == "__main__":
    explore_motivation_profiling()

# %% [markdown]
"""
## 💡 Introduction: The Memory Wall Problem

Imagine trying to fit a library in your backpack. Neural networks face the same challenge - models are getting huge, but devices have limited memory!

### The Precision Paradox

Modern neural networks use 32-bit floating point numbers with incredible precision:

```
FP32 Number: 3.14159265359...
             ^^^^^^^^^^^^^^^^
             32 bits = 4 bytes per weight
```

But here's the surprising truth: **we don't need all that precision for most AI tasks!**

### The Growing Memory Crisis

```
Model Memory Requirements (FP32):
┌─────────────────────────────────────────────────────────────┐
│ BERT-Base:   110M params ×  4 bytes = 440MB                 │
│ GPT-2:       1.5B params ×  4 bytes = 6GB                   │
│ GPT-3:       175B params × 4 bytes = 700GB                  │
│ Your Phone:  Available RAM = 4-8GB                          │
└─────────────────────────────────────────────────────────────┘
                        ↑
                    Problem!
```

### The Quantization Solution

What if we could represent each weight with just 8 bits instead of 32?

```
Before Quantization (FP32):
┌───────────────────────────────┐
│  3.14159265   │  2.71828183   │  32 bits each
└───────────────────────────────┘

After Quantization (INT8):
┌────────┬────────┬────────┬────────┐
│   98   │   85   │   72   │   45   │  8 bits each
└────────┴────────┴────────┴────────┘
         ↑
    4× less memory!
```

### Real-World Impact You'll Achieve

**Memory Reduction:**
- BERT-Base: 440MB → 110MB (4× smaller)
- Fits on mobile devices!
- Faster loading from disk
- More models in GPU memory

**Speed Improvements:**
- 2-4× faster inference (hardware dependent)
- Lower power consumption
- Better user experience

**Accuracy Preservation:**
- <1% accuracy loss with proper techniques
- Sometimes even improves generalization!

**Why This Matters:**
- **Mobile AI:** Deploy powerful models on phones
- **Edge Computing:** Run AI without cloud connectivity
- **Data Centers:** Serve more users with same hardware
- **Environmental:** Reduce energy consumption by 2-4×

Today you'll build the production-quality quantization system that makes all this possible!
"""

# %% [markdown]
"""
## 📐 Foundations: The Mathematics of Compression

### Understanding the Core Challenge

Think of quantization like converting a smooth analog signal to digital steps. We need to map infinite precision (FP32) to just 256 possible values (INT8).

### The Quantization Mapping

```
The Fundamental Problem:

FP32 Numbers (Continuous):        INT8 Numbers (Discrete):
    ∞ possible values         →      256 possible values

  ...  -1.7  -1.2  -0.3  0.0  0.8  1.5  2.1  ...
         ↓     ↓     ↓    ↓    ↓    ↓    ↓
      -128  -95   -38    0   25   48   67   127
```

### The Magic Formula

Every quantization system uses this fundamental relationship:

```
Quantization (FP32 → INT8):
┌─────────────────────────────────────────────────────────┐
│  quantized = round((float_value - zero_point) / scale)  │
└─────────────────────────────────────────────────────────┘

Dequantization (INT8 → FP32):
┌─────────────────────────────────────────────────────────┐
│  float_value = (quantized - zero_point) × scale         │
└─────────────────────────────────────────────────────────┘
```

### The Two Critical Parameters

**1. Scale (s)** - How big each INT8 step is in FP32 space:
```
Small Scale (high precision):       Large Scale (low precision):
 FP32: [0.0, 0.255]                 FP32: [0.0, 25.5]
   ↓     ↓     ↓                       ↓     ↓     ↓
 INT8:  0    128   255              INT8:  0    128   255
        │     │     │                      │     │     │
      0.0   0.127  0.255                 0.0   12.75  25.5

 Scale = 0.001 (very precise)        Scale = 0.1 (less precise)
```

**2. Zero Point (z)** - Which INT8 value represents FP32 zero:
```
Symmetric Range:                    Asymmetric Range:
 FP32: [-2.0, 2.0]                  FP32: [-1.0, 3.0]
   ↓     ↓     ↓                       ↓     ↓     ↓
 INT8: -128    0   127              INT8: -128  -64   127
        │     │     │                      │     │     │
     -2.0    0.0   2.0                  -1.0   0.0   3.0

 Zero Point = 0                     Zero Point = -64
```

### Visual Example: Weight Quantization

```
Original FP32 Weights:           Quantized INT8 Mapping:
┌─────────────────────────┐      ┌─────────────────────────┐
│ -0.8  -0.3   0.0   0.5  │  →   │ -128  -64  -26   38     │
│  0.9   1.2  -0.1   0.7  │      │   89  127  -39   63     │
└─────────────────────────┘      └─────────────────────────┘
     4 bytes each                      1 byte each
   Total: 32 bytes                   Total: 8 bytes
                                    ↑
                              4× compression!
```

### Quantization Error Analysis

```
Perfect Reconstruction (Impossible):  Quantized Reconstruction (Reality):

Original: 0.73                       Original: 0.73
    ↓                                     ↓
INT8: ? (can't represent exactly)     INT8: 93 (closest)
    ↓                                     ↓
Restored: 0.73                        Restored: 0.728
                                           ↑
                                    Error: 0.002
```

**The Quantization Trade-off:**
- **More bits** = Higher precision, larger memory
- **Fewer bits** = Lower precision, smaller memory
- **Goal:** Find the sweet spot where error is acceptable

### Why INT8 is the Sweet Spot

```
Precision vs Memory Trade-offs:

FP32: ████████████████████████████████ (32 bits) - Overkill precision
FP16: ████████████████ (16 bits)                  - Good precision
INT8: ████████ (8 bits)                           - Sufficient precision ← Sweet spot!
INT4: ████ (4 bits)                               - Often too little

Memory:    100%    50%    25%    12.5%
Accuracy:  100%   99.9%  99.5%   95%
```

INT8 gives us 4× memory reduction with <1% accuracy loss - the perfect balance for production systems!
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building the Quantization Engine

### Our Implementation Strategy

We'll build quantization in logical layers, each building on the previous:

```
Quantization System Architecture:

┌─────────────────────────────────────────────────────────────┐
│                    Layer 4: Model Quantization              │
│  quantize_model() - Convert entire neural networks          │
├─────────────────────────────────────────────────────────────┤
│                    Layer 3: Layer Quantization              │
│  QuantizedLinear - Quantized linear transformations         │
├─────────────────────────────────────────────────────────────┤
│                    Layer 2: Tensor Operations               │
│  quantize_int8() - Core quantization algorithm              │
│  dequantize_int8() - Restore to floating point              │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: Foundation                      │
│  Scale & Zero Point Calculation - Parameter optimization    │
└─────────────────────────────────────────────────────────────┘
```

### What We're About to Build

**Core Functions:**
- `quantize_int8()` - Convert FP32 tensors to INT8
- `dequantize_int8()` - Convert INT8 back to FP32
- `QuantizedLinear` - Quantized version of Linear layers
- `quantize_model()` - Quantize entire neural networks

**Key Features:**
- **Automatic calibration** - Find optimal quantization parameters
- **Error minimization** - Preserve accuracy during compression
- **Memory tracking** - Measure actual savings achieved
- **Production patterns** - Industry-standard algorithms

Let's start with the fundamental building block!
"""

# %% [markdown]
"""
## 🏗️ INT8 Quantization - The Foundation

This is the core function that converts any FP32 tensor to INT8. Think of it as a smart compression algorithm that preserves the most important information.

```
Quantization Process Visualization:

Step 1: Analyze Range              Step 2: Calculate Parameters       Step 3: Apply Formula
┌─────────────────────────┐    ┌─────────────────────────┐  ┌─────────────────────────┐
│ Input: [-1.5, 0.2, 2.8] │    │ Min: -1.5               │  │ quantized = round(      │
│                         │    │ Max: 2.8                │  │   value / scale + zp)   │
│ Find min/max values     │ →  │ Range: 4.3              │ →│                         │
│                         │    │ Scale: 4.3/255 = 0.017  │  │                         │
│                         │    │ Zero Point: -39         │  │ Result: [-128,-27, 127] │
└─────────────────────────┘    └─────────────────────────┘  └─────────────────────────┘
```

**Key Challenges This Function Solves:**
- **Dynamic Range:** Each tensor has different min/max values
- **Precision Loss:** Map 4 billion FP32 values to just 256 INT8 values
- **Zero Preservation:** Ensure FP32 zero maps exactly to an INT8 value
- **Symmetric Mapping:** Distribute quantization levels efficiently

**Why This Algorithm:**
- **Linear mapping** preserves relative relationships between values
- **Symmetric quantization** works well for most neural network weights
- **Clipping to [-128, 127]** ensures valid INT8 range
- **Round-to-nearest** minimizes quantization error
"""

# %% nbgrader={"grade": false, "grade_id": "quantize_int8", "solution": true}
#| export
def quantize_int8(tensor: Tensor) -> Tuple[Tensor, float, int]:
    """
    Quantize FP32 tensor to INT8 using symmetric quantization.

    TODO: Implement INT8 quantization with scale and zero_point calculation

    APPROACH:
    1. Find min/max values in tensor data
    2. Calculate scale: (max_val - min_val) / 255 (INT8 range: -128 to 127)
    3. Calculate zero_point: offset that maps min_val to INT8_MIN (-128)
       Formula: zero_point = round(INT8_MIN - min_val / scale)
    4. Apply quantization formula: round(value / scale + zero_point)
    5. Clamp to INT8 range [-128, 127]

    Args:
        tensor: Input FP32 tensor to quantize

    Returns:
        q_tensor: Quantized INT8 tensor
        scale: Scaling factor (float)
        zero_point: Zero point offset (int)

    EXAMPLE:
    >>> tensor = Tensor([[-1.0, 0.0, 2.0], [0.5, 1.5, -0.5]])
    >>> q_tensor, scale, zero_point = quantize_int8(tensor)
    >>> print(f"Scale: {scale:.4f}, Zero point: {zero_point}")
    Scale: 0.0118, Zero point: -43

    HINTS:
    - Use np.round() for quantization
    - Clamp with np.clip(values, -128, 127)
    - Handle edge case where min_val == max_val (set scale=1.0)
    """
    ### BEGIN SOLUTION
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

    # Step 3: Calculate scale and zero_point for standard quantization
    # Map [min_val, max_val] to [INT8_MIN_VALUE, INT8_MAX_VALUE] (INT8 range)
    scale = (max_val - min_val) / (INT8_RANGE - 1)
    zero_point = int(np.round(INT8_MIN_VALUE - min_val / scale))

    # Clamp zero_point to valid INT8 range
    zero_point = int(np.clip(zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE))

    # Step 4: Apply quantization formula: q = (x / scale) + zero_point
    quantized_data = np.round(data / scale + zero_point)

    # Step 5: Clamp to INT8 range and convert to int8
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
    """Test INT8 quantization implementation."""
    print("Unit Test: INT8 Quantization...")

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
    # INT8 quantization has limited precision (256 levels), so error tolerance is higher
    # For a range of 5.0 (1.0 to 6.0), quantization error can be up to ~0.2
    # Using slightly higher tolerance to account for numerical precision variations
    assert error < 0.25, f"Quantization error too high: {error:.4f} (expected < 0.25 for INT8, range=5.0)"

    # Test edge case: constant tensor
    constant_tensor = Tensor([[2.0, 2.0], [2.0, 2.0]])
    q_const, scale_const, zp_const = quantize_int8(constant_tensor)
    assert scale_const == 1.0

    print("INT8 quantization works correctly!")

if __name__ == "__main__":
    test_unit_quantize_int8()

# %% [markdown]
"""
## 🏗️ INT8 Dequantization - Restoring Precision

Dequantization is the inverse process - converting compressed INT8 values back to usable FP32. This is where we "decompress" our quantized data.

```
Dequantization Process:

INT8 Values + Parameters → FP32 Reconstruction

┌───────────────────────────────────┐
│ Quantized: [-128, -27, 127]       │
│ Scale: 0.017                      │
│ Zero Point: -39                   │
└───────────────────────────────────┘
                 │
                 ▼ Apply Formula
┌───────────────────────────────────┐
│ FP32 = (quantized - zero_point)   │
│        × scale                    │
└───────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│ Result: [-1.501, 0.202, 2.799]    │
│ Original: [-1.5, 0.2, 2.8]        │
│ Error: [0.001, 0.002, 0.001]      │
└───────────────────────────────────┘
       ↑
  Excellent approximation!
```

**Why This Step Is Critical:**
- **Neural networks expect FP32** - INT8 values would confuse computations
- **Preserves computation compatibility** - works with existing matrix operations
- **Controlled precision loss** - error is bounded and predictable
- **Hardware flexibility** - can use FP32 or specialized INT8 operations

**When Dequantization Happens:**
- **During forward pass** - before matrix multiplications
- **For gradient computation** - during backward pass
- **Educational approach** - production uses INT8 GEMM directly
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
    # Apply inverse quantization formula
    # This is the correct inverse of: quantized = (value / scale) + zero_point
    # Therefore: value = (quantized - zero_point) * scale
    dequantized_data = (q_tensor.data.astype(np.float32) - zero_point) * scale
    return Tensor(dequantized_data)
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
    """Test INT8 dequantization implementation."""
    print("Unit Test: INT8 Dequantization...")

    # Test round-trip: quantize → dequantize
    original = Tensor([[-1.5, 0.0, 3.2], [1.1, -0.8, 2.7]])
    q_tensor, scale, zero_point = quantize_int8(original)
    restored = dequantize_int8(q_tensor, scale, zero_point)

    # Verify round-trip error is small
    error = np.mean(np.abs(original.data - restored.data))
    assert error < 0.1, f"Round-trip error too high: {error}"

    # Verify output is float32
    assert restored.data.dtype == np.float32

    print("INT8 dequantization works correctly!")

if __name__ == "__main__":
    test_unit_dequantize_int8()

# %% [markdown]
"""
## 🏗️ QuantizedLinear - The Heart of Efficient Networks

### Why We Need Quantized Layers

A quantized model isn't just about storing weights in INT8 - we need layers that can work efficiently with quantized data.

```
Regular Linear Layer:              QuantizedLinear Layer:

┌─────────────────────┐            ┌─────────────────────┐
│ Input: FP32         │            │ Input: FP32         │
│ Weights: FP32       │            │ Weights: INT8       │
│ Computation: FP32   │    VS      │ Computation: Mixed  │
│ Output: FP32        │            │ Output: FP32        │
│ Memory: 4× more     │            │ Memory: 4× less     │
└─────────────────────┘            └─────────────────────┘
```

### The Quantized Forward Pass

```
Quantized Linear Layer Forward Pass:

    Input (FP32)                  Quantized Weights (INT8)
         │                               │
         ▼                               ▼
┌─────────────────┐              ┌─────────────────┐
│    Calibrate    │              │   Dequantize    │
│   (optional)    │              │   Weights       │
└─────────────────┘              └─────────────────┘
         │                               │
         ▼                               ▼
    Input (FP32)                  Weights (FP32)
         │                               │
         └───────────────┬───────────────┘
                         ▼
                ┌─────────────────┐
                │ Matrix Multiply │
                │   (FP32 GEMM)   │
                └─────────────────┘
                         │
                         ▼
                   Output (FP32)

Memory Saved: 4× for weights storage!
Speed: Depends on dequantization overhead vs INT8 GEMM support
```

### Calibration - Finding Optimal Input Quantization

```
Calibration Process:

 Step 1: Collect Sample Inputs    Step 2: Analyze Distribution    Step 3: Optimize Parameters
 ┌─────────────────────────┐      ┌─────────────────────────┐    ┌─────────────────────────┐
 │ input_1: [-0.5, 0.2, ..]│      │   Min: -0.8             │    │ Scale: 0.00627          │
 │ input_2: [-0.3, 0.8, ..]│  →   │   Max: +0.8             │ →  │ Zero Point: 0           │
 │ input_3: [-0.1, 0.5, ..]│      │   Range: 1.6            │    │ Optimal for this data   │
 │ ...                     │      │   Distribution: Normal  │    │ range and distribution  │
 └─────────────────────────┘      └─────────────────────────┘    └─────────────────────────┘
```

**Why Calibration Matters:**
- **Without calibration:** Generic quantization parameters may waste precision
- **With calibration:** Parameters optimized for actual data distribution
- **Result:** Better accuracy preservation with same memory savings
"""

# %% [markdown]
"""
## 🏗️ QuantizedLinear Class - Efficient Neural Network Layer

This class replaces regular Linear layers with quantized versions that use 4× less memory while preserving functionality.

```
QuantizedLinear Architecture:

Creation Time:                       Runtime:
┌───────────────────────────────┐    ┌───────────────────────────────┐
│ Regular Linear Layer          │    │ Input (FP32)                  │
│ ↓                             │    │ ↓                             │
│ Quantize weights → INT8       │    │ Optional: quantize input      │
│ Quantize bias → INT8          │ →  │ ↓                             │
│ Store quantization params     │    │ Dequantize weights            │
│ Ready for deployment!         │    │ ↓                             │
└───────────────────────────────┘    │ Matrix multiply (FP32)        │
      One-time cost                  │ ↓                             │
                                     │ Output (FP32)                 │
                                     └───────────────────────────────┘
                                        Per-inference cost
```

**Key Design Decisions:**

1. **Store original layer reference** - for debugging and comparison
2. **Separate quantization parameters** - weights and bias may need different scales
3. **Calibration support** - optimize input quantization using real data
4. **FP32 computation** - educational approach, production uses INT8 GEMM
5. **Memory tracking** - measure actual compression achieved

**Memory Layout:**

Regular Linear layers store weights in FP32 (4 bytes each), while QuantizedLinear stores them in INT8 (1 byte each) plus a small overhead for quantization parameters (scales and zero points). This achieves approximately 4× memory reduction with minimal overhead.

**Production vs Educational Trade-off:**
- **Our approach:** Dequantize → FP32 computation (easier to understand)
- **Production:** INT8 GEMM operations (faster, more complex)
- **Both achieve:** Same memory savings, similar accuracy
"""

# %% nbgrader={"grade": false, "grade_id": "quantized_linear", "solution": true}
#| export
class QuantizedLinear:
    """Quantized version of Linear layer using INT8 arithmetic."""

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
        >>> original_layer.weight = Tensor(np.random.randn(128, 64) * 0.1)
        >>> original_layer.bias = Tensor(np.random.randn(64) * 0.01)
        >>> quantized_layer = QuantizedLinear(original_layer)
        >>> print(quantized_layer.q_weight.data.dtype)
        int8

        HINTS:
        - Use quantize_int8() to convert weight and bias tensors
        - Store all quantization parameters (scale, zero_point) for later dequantization
        - Initialize input_scale and input_zero_point to None (set during calibration)
        """
        ### BEGIN SOLUTION
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
        >>> sample_data = [Tensor(np.random.randn(1, 64)) for _ in range(10)]
        >>> layer.calibrate(sample_data)
        >>> print(layer.input_scale is not None)
        True

        HINTS:
        - Flatten all sample inputs and find global min/max values
        - Use the same scale/zero_point formula as quantize_int8()
        - Handle edge case where all inputs have the same value (constant tensor)
        """
        ### BEGIN SOLUTION
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
        - Use dequantize_int8() to restore weights to FP32 before computation
        - Use x.matmul() for matrix multiplication
        - Add bias after matmul if it exists (dequantize bias first)

        NOTE: Production quantization uses INT8 GEMM libraries for speed
        """
        ### BEGIN SOLUTION
        # For educational purposes, we dequantize and compute in FP32
        # Production systems use specialized INT8 GEMM operations

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
        """Calculate memory usage in bytes."""
        ### BEGIN SOLUTION
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

        # Add overhead for scales and zero points (small)
        # 2 floats: one scale for weights, one scale for bias (if present)
        overhead_bytes = BYTES_PER_FLOAT32 * 2

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
    """Test QuantizedLinear implementation."""
    print("Unit Test: QuantizedLinear...")

    # Create original linear layer
    original = Linear(4, 3)
    original.weight = Tensor(np.random.randn(4, 3) * 0.5)  # Smaller range for testing
    original.bias = Tensor(np.random.randn(3) * 0.1)

    # Create quantized version
    quantized = QuantizedLinear(original)

    # Test forward pass
    x = Tensor(np.random.randn(2, 4) * 0.5)

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

    # The compression should be close to 4× (allowing for quantization parameter overhead)
    assert memory_info['compression_ratio'] > 2.5, f"Should achieve ~4× compression, got {memory_info['compression_ratio']:.2f}×"

    print(f"  Memory reduction: {memory_info['compression_ratio']:.1f}x")
    print("QuantizedLinear works correctly!")

if __name__ == "__main__":
    test_unit_quantized_linear()

# %% [markdown]
"""
## 🔧 Integration: Scaling to Full Neural Networks

### The Model Quantization Challenge

Quantizing individual tensors is useful, but real applications need to quantize entire neural networks with multiple layers, activations, and complex data flows. The key is replacing standard layers (like Linear) with their quantized equivalents (QuantizedLinear) while keeping activation functions unchanged since they have no parameters.

### Smart Layer Selection

Not all layers benefit equally from quantization. Linear and convolutional layers with many parameters see the largest benefits, while activation functions (which have no parameters) cannot be quantized. Some layers like input/output projections may be sensitive to quantization and should be kept in higher precision for critical applications.

### Calibration Data Flow

Calibration runs sample data through the model layer-by-layer, collecting activation statistics at each layer. These statistics (min/max values, distributions) determine optimal quantization parameters for each layer, ensuring minimal accuracy loss during quantization.

### Memory Impact

Quantization provides consistent 4× memory reduction across all model sizes. The actual impact depends on model architecture, but the compression ratio remains constant since we're reducing precision from 32 bits to 8 bits per parameter.

Now let's implement the functions that make this transformation possible!
"""

# %% [markdown]
"""
## 🔧 Model Quantization - Scaling to Full Networks

Quantizing individual layers is useful, but real applications need to quantize entire neural
networks. We'll build this capability in two steps:

1. **Collect layer inputs** - Forward calibration data through preceding layers to get
   the activation distribution at each layer's input
2. **Quantize a single layer** - Replace one Linear layer with its QuantizedLinear equivalent

Then the composition function `quantize_model()` ties them together to transform a full model.

```
Model Transformation Process:

Input Model:                    Quantized Model:
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ layers[0]: Linear(784, 128) │    │ layers[0]: QuantizedLinear  │
│ layers[1]: ReLU()           │    │ layers[1]: ReLU()           │
│ layers[2]: Linear(128, 64)  │ →  │ layers[2]: QuantizedLinear  │
│ layers[3]: ReLU()           │    │ layers[3]: ReLU()           │
│ layers[4]: Linear(64, 10)   │    │ layers[4]: QuantizedLinear  │
└─────────────────────────────┘    └─────────────────────────────┘
   Memory: 100%                      Memory: ~25%
   Interface: Same                   Interface: Identical
```
"""

# %% [markdown]
"""
## 🏗️ Collecting Layer Inputs - Calibration Data Flow

Before we can calibrate a quantized layer, we need to know what its inputs look like
at runtime. This helper forwards calibration samples through all preceding layers
to collect the activation tensors that arrive at a given layer index.

```
Calibration Data Flow for Layer at Index i:

  Sample Data          Layers 0..i-1          Activations at Layer i
  ┌──────────┐      ┌──────────────────┐      ┌──────────────────┐
  │ sample_0  │ ──→ │ forward through  │ ──→  │ activation_0     │
  │ sample_1  │ ──→ │ preceding layers │ ──→  │ activation_1     │
  │ ...       │     │ (0, 1, ..., i-1) │      │ ...              │
  │ sample_N  │ ──→ │                  │ ──→  │ activation_N     │
  └──────────┘      └──────────────────┘      └──────────────────┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "collect_layer_inputs", "solution": true}
#| export
def _collect_layer_inputs(model, layer_index: int, calibration_data: List[Tensor], max_samples: int = 10) -> List[Tensor]:
    """
    Forward calibration data through preceding layers to collect inputs for a specific layer.

    TODO: Forward each calibration sample through layers 0..layer_index-1

    APPROACH:
    1. Take up to max_samples from calibration_data for efficiency
    2. For each sample, forward through all layers before layer_index
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
    >>> samples = [Tensor(np.random.randn(1, 4)) for _ in range(5)]
    >>> inputs_at_layer2 = _collect_layer_inputs(model, 2, samples)
    >>> print(len(inputs_at_layer2))  # 5 activation tensors
    5

    HINT:
    - Use model.layers[j].forward(x) to pass through each preceding layer
    """
    ### BEGIN SOLUTION
    sample_inputs = []
    for data in calibration_data[:max_samples]:
        x = data
        for j in range(layer_index):
            x = model.layers[j].forward(x)
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
    """Test collecting intermediate activations for calibration."""
    print("Unit Test: Collect Layer Inputs...")

    # Create a simple model
    layer1 = Linear(4, 8)
    layer1.weight = Tensor(np.random.randn(4, 8) * 0.5)
    layer1.bias = Tensor(np.random.randn(8) * 0.1)
    activation = ReLU()
    layer2 = Linear(8, 3)
    layer2.weight = Tensor(np.random.randn(8, 3) * 0.5)
    layer2.bias = Tensor(np.random.randn(3) * 0.1)
    model = Sequential(layer1, activation, layer2)

    samples = [Tensor(np.random.randn(1, 4)) for _ in range(5)]

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

    print("Collect layer inputs works correctly!")

if __name__ == "__main__":
    test_unit_collect_layer_inputs()

# %% [markdown]
"""
## 🏗️ Quantizing a Single Layer - The Replacement Step

This helper takes one Linear layer, wraps it in a QuantizedLinear, and optionally
calibrates it using pre-collected activation samples. This is the atomic operation
that `quantize_model()` applies to each eligible layer.

```
Single Layer Quantization:

  Linear Layer          QuantizedLinear
  ┌──────────────┐      ┌──────────────────────────┐
  │ weight: FP32 │  →   │ q_weight: INT8           │
  │ bias: FP32   │      │ q_bias: INT8             │
  │              │      │ weight_scale, zero_point  │
  └──────────────┘      │ calibrated: Yes/No       │
                        └──────────────────────────┘
       4 bytes/param          1 byte/param + overhead
```
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
    >>> original.weight = Tensor(np.random.randn(8, 3) * 0.5)
    >>> quantized = _quantize_single_layer(original)
    >>> print(quantized.q_weight.data.dtype)
    int8

    HINT:
    - QuantizedLinear(layer) handles weight/bias quantization
    - quantized_layer.calibrate(inputs) sets input quantization parameters
    """
    ### BEGIN SOLUTION
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
    """Test single layer quantization with and without calibration."""
    print("Unit Test: Quantize Single Layer...")

    # Create a linear layer
    layer = Linear(4, 3)
    layer.weight = Tensor(np.random.randn(4, 3) * 0.5)
    layer.bias = Tensor(np.random.randn(3) * 0.1)

    # Quantize without calibration
    q_layer = _quantize_single_layer(layer)
    assert isinstance(q_layer, QuantizedLinear)
    assert q_layer.q_weight is not None, "Quantized weights should exist"
    assert q_layer.input_scale is None, "Without calibration, input_scale should be None"

    # Quantize with calibration
    cal_inputs = [Tensor(np.random.randn(1, 4)) for _ in range(5)]
    q_layer_cal = _quantize_single_layer(layer, calibration_inputs=cal_inputs)
    assert isinstance(q_layer_cal, QuantizedLinear)
    assert q_layer_cal.input_scale is not None, "With calibration, input_scale should be set"

    # Verify forward pass works
    x = Tensor(np.random.randn(2, 4))
    output = q_layer.forward(x)
    assert output.shape == (2, 3), f"Output shape should be (2, 3), got {output.shape}"

    print("Quantize single layer works correctly!")

if __name__ == "__main__":
    test_unit_quantize_single_layer()

# %% [markdown]
"""
## 🔧 Model Quantization - The Composition Function

Now we compose the helpers into the full model quantization function. For each Linear
layer in the model, we collect its calibration inputs and replace it with a quantized version.

```
quantize_model() orchestrates the full pipeline:

  For each layer in model.layers:
      │
      ├── isinstance(layer, Linear)?
      │   ├── YES → _collect_layer_inputs()  → calibration activations
      │   │         _quantize_single_layer()  → QuantizedLinear
      │   │         Replace model.layers[i]
      │   │
      │   └── NO  → Keep unchanged (ReLU, etc.)
```
"""

# %% nbgrader={"grade": false, "grade_id": "quantize_model", "solution": true}
#| export
def quantize_model(model, calibration_data: Optional[List[Tensor]] = None) -> None:
    """
    Quantize all Linear layers in a model in-place.

    TODO: Replace all Linear layers with QuantizedLinear versions

    APPROACH:
    1. Validate model has .layers attribute (Sequential pattern)
    2. Iterate through layers, find Linear layers
    3. For each Linear layer, collect calibration inputs (if data provided)
    4. Replace with quantized version using _quantize_single_layer()

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
    ### BEGIN SOLUTION
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                # Collect calibration inputs if data provided
                cal_inputs = None
                if calibration_data is not None:
                    cal_inputs = _collect_layer_inputs(model, i, calibration_data)

                # Replace with quantized version
                model.layers[i] = _quantize_single_layer(layer, cal_inputs)

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
    """Test model quantization implementation."""
    print("Unit Test: Model Quantization...")

    # Create test model using explicit layer composition (TinyTorch pattern)
    layer1 = Linear(4, 8)
    activation = ReLU()
    layer2 = Linear(8, 3)

    # Initialize weights
    layer1.weight = Tensor(np.random.randn(4, 8) * 0.5)
    layer1.bias = Tensor(np.random.randn(8) * 0.1)
    layer2.weight = Tensor(np.random.randn(8, 3) * 0.5)
    layer2.bias = Tensor(np.random.randn(3) * 0.1)

    # Use Sequential from tinytorch.core.layers
    model = Sequential(layer1, activation, layer2)

    # Test original model
    x = Tensor(np.random.randn(2, 4))
    original_output = model.forward(x)

    # Create calibration data
    calibration_data = [Tensor(np.random.randn(1, 4)) for _ in range(5)]

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

    print("Model quantization works correctly!")

if __name__ == "__main__":
    test_unit_quantize_model()

# %% [markdown]
"""
## 🔧 Model Size Comparison - Measuring the Impact

To compare memory usage between original and quantized models, we need to measure
bytes at the individual layer level first, then aggregate. We'll build this in two steps:

1. **Measure one layer** - Count parameters and bytes for a single layer, handling both
   FP32 (Linear) and INT8 (QuantizedLinear) layers correctly
2. **Aggregate and compare** - Sum across all layers and compute compression metrics

```
Per-Layer Measurement:

  Layer Type          Measurement Strategy
  ┌──────────────┐    ┌───────────────────────────────────┐
  │ Linear       │ →  │ params × 4 bytes (FP32)           │
  │ QuantizedLin │ →  │ memory_usage() dict (INT8 + ovhd) │
  │ ReLU/other   │ →  │ 0 params, 0 bytes (no weights)    │
  └──────────────┘    └───────────────────────────────────┘
```
"""

# %% [markdown]
"""
## 🏗️ Measuring a Single Layer - Per-Layer Byte Accounting

This helper measures the parameter count and byte usage for one layer. It handles
the key distinction: FP32 layers store parameters at 4 bytes each, while QuantizedLinear
layers use INT8 storage with a small overhead for scale/zero_point metadata.

```
Byte Accounting per Layer Type:

  FP32 Linear:                     QuantizedLinear:
  ┌─────────────────────────┐      ┌─────────────────────────────────┐
  │ weight: N × 4 bytes     │      │ q_weight: N × 1 byte            │
  │ bias:   M × 4 bytes     │      │ q_bias:   M × 1 byte            │
  │                         │      │ overhead: ~8 bytes (scale+zp)    │
  │ Total: (N+M) × 4       │      │ Total: (N+M) × 1 + overhead     │
  └─────────────────────────┘      └─────────────────────────────────┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "measure_layer_bytes", "solution": true}
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
    ### BEGIN SOLUTION
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
    """Test per-layer byte measurement for FP32 and quantized layers."""
    print("Unit Test: Measure Layer Bytes...")

    # Test FP32 Linear layer
    linear = Linear(10, 5)
    linear.weight = Tensor(np.random.randn(10, 5))
    linear.bias = Tensor(np.random.randn(5))
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
    print("Measure layer bytes works correctly!")

if __name__ == "__main__":
    test_unit_measure_layer_bytes()

# %% [markdown]
"""
## 🔧 Model Size Analysis - The Composition Function

Now we aggregate per-layer measurements across the full model to produce a comprehensive
comparison between original and quantized versions.

```
Aggregation Flow:

  Original Model                    Quantized Model
  ┌──────────────────────────┐      ┌──────────────────────────┐
  │ Layer 0: _measure(FP32)  │      │ Layer 0: _measure(INT8)  │
  │ Layer 1: _measure(skip)  │      │ Layer 1: _measure(skip)  │
  │ Layer 2: _measure(FP32)  │      │ Layer 2: _measure(INT8)  │
  └──────────────────────────┘      └──────────────────────────┘
           │                                  │
           ▼                                  ▼
     Sum params, bytes                  Sum params, bytes
           │                                  │
           └─────────────┬────────────────────┘
                         ▼
               Compression metrics
```
"""

# %% nbgrader={"grade": false, "grade_id": "analyze_model_sizes", "solution": true}

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
    ### BEGIN SOLUTION
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
        is_q = isinstance(layer, QuantizedLinear)
        p, b = _measure_layer_bytes(layer, is_quantized=is_q)
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
    """Test model size analysis."""
    print("Unit Test: Model Size Analysis...")

    # Create and quantize a model for testing (using Sequential from tinytorch.core.layers)
    layer1_orig = Linear(100, 50)
    activation_orig = ReLU()
    layer2_orig = Linear(50, 10)
    layer1_orig.weight = Tensor(np.random.randn(100, 50))
    layer1_orig.bias = Tensor(np.random.randn(50))
    layer2_orig.weight = Tensor(np.random.randn(50, 10))
    layer2_orig.bias = Tensor(np.random.randn(10))
    original_model = Sequential(layer1_orig, activation_orig, layer2_orig)

    # Create quantized copy
    layer1_quant = Linear(100, 50)
    activation_quant = ReLU()
    layer2_quant = Linear(50, 10)
    layer1_quant.weight = Tensor(np.random.randn(100, 50))
    layer1_quant.bias = Tensor(np.random.randn(50))
    layer2_quant.weight = Tensor(np.random.randn(50, 10))
    layer2_quant.bias = Tensor(np.random.randn(10))
    quantized_model = Sequential(layer1_quant, activation_quant, layer2_quant)

    quantize_model(quantized_model)

    # Analyze sizes
    comparison = analyze_model_sizes(original_model, quantized_model)

    # Verify compression achieved
    assert comparison['compression_ratio'] > 2.0, "Should achieve significant compression"
    assert comparison['memory_saved_percent'] > 50, "Should save >50% memory"

    print(f"  Compression ratio: {comparison['compression_ratio']:.1f}x")
    print(f"  Memory saved: {comparison['memory_saved_percent']:.1f}%")
    print("Model size analysis works correctly!")

if __name__ == "__main__":
    test_unit_analyze_model_sizes()

# %% [markdown]
"""
## 🔧 Consolidated Quantization Classes for Export

Now that we've implemented all quantization components, let's create consolidated classes
for export to the tinytorch package. This allows milestones to use the complete quantization system.
"""

# %% nbgrader={"grade": false, "grade_id": "quantization_export", "solution": true}
#| export
class Quantizer:
    """
    Complete quantization system for milestone use.

    Provides INT8 quantization with calibration for 4× memory reduction.

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
    def quantize_model(model, calibration_data: Optional[List[Tensor]] = None) -> Dict[str, any]:
        """
        Quantize all Linear layers in a model and return stats.

        Unlike the standalone quantize_model() which modifies in-place,
        this returns a dictionary with quantization info for benchmarking.

        Returns:
            Dict with quantized_layers, original_size_mb, quantized_size_mb, compression_ratio
        """
        quantized_layers = {}
        original_size = 0
        total_elements = 0
        param_idx = 0

        # Iterate through model parameters
        for layer in model.layers:
            for param in layer.parameters():
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

        # INT8 uses 1 byte per element
        quantized_size = total_elements

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

    print("\nKey Insight: Memory reduction is consistent at 4x across all model sizes")
    print("This enables deployment on memory-constrained devices")

if __name__ == "__main__":
    analyze_quantization_memory()

# %%
def analyze_quantization_accuracy():
    """Analyze accuracy vs memory trade-off for quantization."""
    print("\nAnalyzing Quantization Accuracy Trade-offs")

    # Simulate quantization impact on different layer types
    layer_types = [
        ("Embeddings", 0.99, "Low impact - lookup tables"),
        ("Attention", 0.97, "Moderate impact - many small ops"),
        ("MLP", 0.98, "Low impact - large matrix muls"),
        ("Output", 0.95, "Higher impact - final predictions")
    ]

    print(f"{'Layer Type':<15} {'Acc Retention':<15} {'Observation'}")
    print("-" * 50)

    for layer, retention, note in layer_types:
        print(f"{layer:<15} {retention:>13.1%}  {note}")

    print("\nKey Insight: Overall model accuracy retention: ~98-99% typical")
    print("Output layers most sensitive to quantization")

if __name__ == "__main__":
    analyze_quantization_accuracy()

# %% [markdown]
"""
## 📊 Advanced Quantization Strategies - Production Techniques

This analysis compares different quantization approaches used in production systems, revealing the trade-offs between accuracy, complexity, and performance.

```
Strategy Comparison Framework:

┌──────────────────────────────────────────────────────────────────────────────────┐
│                          Three Advanced Strategies                             │
├──────────────────────────┬──────────────────────────┬──────────────────────────┤
│       Strategy 1         │       Strategy 2         │       Strategy 3         │
│    Per-Tensor (Ours)     │    Per-Channel Scale     │    Mixed Precision       │
├──────────────────────────┼──────────────────────────┼──────────────────────────┤
│                          │                          │                          │
│ ┌──────────────────────┐ │ ┌──────────────────────┐ │ ┌──────────────────────┐ │
│ │ Weights:             │ │ │ Channel 1: scale₁   │ │ │ Sensitive: FP32      │ │
│ │ [W₁₁ W₁₂ W₁₃]        │ │ │ Channel 2: scale₂   │ │ │ Regular: INT8        │ │
│ │ [W₂₁ W₂₂ W₂₃] scale  │ │ │ Channel 3: scale₃   │ │ │                      │ │
│ │ [W₃₁ W₃₂ W₃₃]        │ │ │                      │ │ │ Input: FP32          │ │
│ └──────────────────────┘ │ │ Better precision     │ │ │ Output: FP32         │ │
│                          │ │ per channel          │ │ │ Hidden: INT8         │ │
│ Simple, fast             │ └──────────────────────┘ │ └──────────────────────┘ │
│ Good baseline            │                          │                          │
│                          │ More complex             │ Optimal accuracy         │
│                          │ Better accuracy          │ Selective compression    │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

**Strategy 1: Per-Tensor Quantization (Our Implementation)**
```
Weight Matrix:                Scale Calculation:
┌─────────────────────────┐     ┌─────────────────────────┐
│ 0.1 -0.3  0.8  0.2      │     │ Global min: -0.5        │
│-0.2  0.5 -0.1  0.7      │ →   │ Global max: +0.8        │
│ 0.4 -0.5  0.3 -0.4      │     │ Scale: 1.3/255 = 0.0051 │
└─────────────────────────┘     └─────────────────────────┘

Pros: Simple, fast           Cons: May waste precision
```

**Strategy 2: Per-Channel Quantization (Advanced)**
```
Weight Matrix:                Scale Calculation:
┌─────────────────────────┐     ┌─────────────────────────┐
│ 0.1 -0.3  0.8  0.2      │     │ Col 1: [-0.2,0.4] → s₁  │
│-0.2  0.5 -0.1  0.7      │ →   │ Col 2: [-0.5,0.5] → s₂  │
│ 0.4 -0.5  0.3 -0.4      │     │ Col 3: [-0.1,0.8] → s₃  │
└─────────────────────────┘     │ Col 4: [-0.4,0.7] → s₄  │
                             └─────────────────────────┘

Pros: Better precision       Cons: More complex
```

**Strategy 3: Mixed Precision (Production)**
```
Model Architecture:            Precision Assignment:
┌─────────────────────────┐     ┌─────────────────────────┐
│ Input Layer  (sensitive) │     │ Keep in FP32 (precision) │
│ Hidden 1     (bulk)     │ →   │ Quantize to INT8        │
│ Hidden 2     (bulk)     │     │ Quantize to INT8        │
│ Output Layer (sensitive)│     │ Keep in FP32 (quality)   │
└─────────────────────────┘     └─────────────────────────┘

Pros: Optimal trade-off      Cons: Requires expertise
```

**Experimental Design:**
```
Comparative Testing Protocol:

1. Create identical test model   →  2. Apply each strategy        →  3. Measure results
   ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
   │ 128 → 64 → 10 MLP      │     │ Per-tensor quantization │     │ MSE error calculation  │
   │ Identical weights       │     │ Per-channel simulation  │     │ Compression measurement│
   │ Same test input         │     │ Mixed precision setup   │     │ Speed comparison       │
   └───────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

**Expected Strategy Rankings:**
1. **Mixed Precision** - Best accuracy, moderate complexity
2. **Per-Channel** - Good accuracy, higher complexity
3. **Per-Tensor** - Baseline accuracy, simplest implementation

This analysis reveals which strategies work best for different deployment scenarios and accuracy requirements.
"""

# %% [markdown]
"""
## 📊 Measuring Quantization Savings with Profiler

Now let's use the Profiler tool from Module 14 to measure the actual memory savings from quantization. This demonstrates end-to-end workflow: profile baseline (M14) -> apply quantization (M15) -> measure savings (M14+M15).

This is the production workflow: measure -> compress -> validate -> deploy.
"""

# %% nbgrader={"grade": false, "grade_id": "demo-profiler-quantization", "solution": true}
# Import Profiler from Module 14
from tinytorch.perf.profiling import Profiler

def explore_quantization_with_profiler():
    """Demonstrate memory savings using Profiler from Module 14."""
    print("Measuring Quantization Memory Savings with Profiler")
    print("=" * 70)

    profiler = Profiler()

    # Create a simple model
    from tinytorch.core.layers import Linear
    model = Linear(512, 256)
    model.name = "baseline_model"

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

    # Quantize the model (in-place modification)
    print("\nQuantizing to INT8...")
    # quantize_model expects a model with .layers attribute, so wrap single layer in Sequential
    wrapped_model = Sequential(model)
    quantize_model(wrapped_model)  # Modifies model in-place, returns None
    quantized_model = wrapped_model.layers[0] if wrapped_model.layers else model
    quantized_model.name = "quantized_model"

    print("\nAFTER: INT8 Quantized Model")
    print("-" * 70)

    # Measure quantized (simulated - in practice INT8 uses 1 byte)
    # For demonstration, we show the theoretical savings
    quantized_param_count = profiler.count_parameters(quantized_model)
    theoretical_memory_mb = param_count * BYTES_PER_INT8 / MB_TO_BYTES

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
    print(f"   This enables: 4x larger models, 4x bigger batches, or 4x lower cost!")
    print(f"   Critical for edge devices with limited memory (mobile, IoT)")
    print("\nThis is the power of quantization: same functionality, 4x less memory!")

if __name__ == "__main__":
    explore_quantization_with_profiler()

# %% [markdown]
"""
## 🔧 Verification: Prove Quantization Works

Before running the full integration test, let's create a verification function that
proves quantization actually reduces memory using real `.nbytes` measurements.
"""

# %%
#| export
def verify_quantization_works(original_model, quantized_model):
    """
    Verify quantization actually reduces memory using real .nbytes measurements.

    This is NOT a theoretical calculation - we measure actual bytes consumed
    by numpy arrays to prove the optimization is real.

    Args:
        original_model: Model with FP32 parameters (Sequential with .parameters())
        quantized_model: Model with INT8 quantized parameters (Sequential with QuantizedLinear layers)

    Returns:
        dict: Verification results with actual_reduction, original_mb, quantized_mb

    Example:
        >>> original = Sequential(Linear(100, 50))
        >>> quantized = Sequential(Linear(100, 50))
        >>> quantize_model(quantized)
        >>> results = verify_quantization_works(original, quantized)
        >>> assert results['actual_reduction'] >= 3.5  # Real 4× reduction
    """
    print("Verifying actual memory reduction with .nbytes...")

    # Collect actual bytes from original FP32 model
    original_bytes = sum(
        param.data.nbytes for param in original_model.parameters()
        if hasattr(param, 'data') and hasattr(param.data, 'nbytes')
    )

    # Collect actual bytes from quantized INT8 model
    quantized_bytes = sum(
        layer.q_weight.data.nbytes + (layer.q_bias.data.nbytes if layer.q_bias is not None else 0)
        for layer in quantized_model.layers
        if isinstance(layer, QuantizedLinear)
    )

    # Calculate actual reduction
    actual_reduction = original_bytes / max(quantized_bytes, 1)

    # Display results
    print(f"   Original model: {original_bytes / MB_TO_BYTES:.2f} MB (FP32)")
    print(f"   Quantized model: {quantized_bytes / MB_TO_BYTES:.2f} MB (INT8)")
    print(f"   Actual reduction: {actual_reduction:.1f}x")
    print(f"   {'PASS' if actual_reduction >= 3.5 else 'FAIL'} Meets 4x reduction target")

    # Verify target met
    assert actual_reduction >= 3.5, f"Expected ~4x reduction, got {actual_reduction:.1f}x"

    print(f"\nVERIFIED: Quantization achieves real {actual_reduction:.1f}x memory reduction!")

    return {
        'actual_reduction': actual_reduction,
        'original_mb': original_bytes / MB_TO_BYTES,
        'quantized_mb': quantized_bytes / MB_TO_BYTES,
        'verified': actual_reduction >= 3.5
    }

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.
"""

# %% nbgrader={"grade": true, "grade_id": "test_module", "locked": true, "points": 20, "solution": false, "schema_version": 3}
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
            layer.weight = Tensor(np.random.randn(fan_in, fan_out) * std)
            layer.bias = Tensor(np.zeros(fan_out))

    # Generate realistic calibration data
    calibration_data = [Tensor(np.random.randn(1, 784) * 0.1) for _ in range(20)]

    # Test original model
    test_input = Tensor(np.random.randn(8, 784) * 0.1)
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
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of quantization and its systems implications:

### Question 1: Memory Architecture Impact
You implemented INT8 quantization that reduces each parameter from 4 bytes to 1 byte.
For a model with 100M parameters:
- Original memory usage: _____ GB
- Quantized memory usage: _____ GB
- Memory bandwidth reduction when loading from disk: _____ ×

### BEGIN SOLUTION
**Answer 1: Memory Architecture Impact**
- Original memory usage: **0.4 GB** (100M parameters × 4 bytes = 400MB = 0.4 GB)
- Quantized memory usage: **0.1 GB** (100M parameters × 1 byte = 100MB = 0.1 GB)
- Memory bandwidth reduction: **4×** (loading 100MB instead of 400MB from disk)

**Key Insight**: Quantization reduces not just RAM usage, but also disk I/O, network transfer time, and memory bandwidth pressure. A 4× reduction in bandwidth means 4× faster model loading and 4× less network traffic when deploying models.
### END SOLUTION

### Question 2: Quantization Error Analysis
Your quantization maps a continuous range to 256 discrete values (INT8).
For weights uniformly distributed in [-0.1, 0.1]:
- Quantization scale: _____
- Maximum quantization error: _____
- Signal-to-noise ratio approximately: _____ dB

### BEGIN SOLUTION
**Answer 2: Quantization Error Analysis**
- Quantization scale: **0.0007843** (range 0.2 / 255 steps = 0.0007843)
- Maximum quantization error: **±0.000392** (scale / 2 = ±0.0003922)
- Signal-to-noise ratio: **~48 dB** (20 × log10(signal_range / quantization_step) ≈ 20 × log10(255) ≈ 48 dB)

**Key Insight**: For 8-bit quantization, theoretical SNR is approximately 6 dB per bit × 8 bits = 48 dB. This is sufficient for neural networks because weights typically have bounded ranges and networks are robust to small perturbations.
### END SOLUTION

### Question 3: Hardware Efficiency
Modern processors have specialized INT8 instructions (like AVX-512 VNNI).
Compared to FP32 operations:
- How many INT8 operations fit in one SIMD instruction vs FP32? _____ × more
- Why might actual speedup be less than this theoretical maximum? _____
- What determines whether quantization improves or hurts performance? _____

### BEGIN SOLUTION
**Answer 3: Hardware Efficiency**
- INT8 operations per SIMD: **4× more** (512-bit register can hold 64 INT8 values vs 16 FP32 values)
- Why actual speedup is less: **Dequantization overhead, memory bandwidth bottlenecks, and non-compute operations** (data movement, activation functions, etc. remain in FP32)
- Performance determinant: **Hardware INT8 support availability** (modern CPUs with VNNI, GPUs with Tensor Cores, mobile chips with Neural Engine) and **compute vs memory-bound workload** (compute-bound benefits more from INT8 ops, memory-bound benefits from reduced bandwidth)

**Key Insight**: Theoretical 4× speedup requires: (1) Hardware with native INT8 instructions, (2) Large matrix multiplications where compute dominates, (3) Minimal dequantization overhead. Real-world speedups are typically 2-3× due to mixed precision operations and data movement costs.
### END SOLUTION

### Question 4: Calibration Strategy Trade-offs
Your calibration process finds optimal scales using sample data.
- Too little calibration data: Risk of _____
- Too much calibration data: Cost of _____
- Per-channel vs per-tensor quantization trades _____ for _____

### BEGIN SOLUTION
**Answer 4: Calibration Strategy Trade-offs**
- Too little calibration data: Risk of **suboptimal quantization parameters that don't represent the true activation distribution**, leading to **clipping of outliers and accuracy degradation**
- Too much calibration data: Cost of **increased calibration time** and **diminishing returns** (accuracy stops improving after ~100-1000 samples typically)
- Per-channel vs per-tensor trades: **Complexity and overhead** (more scales to store/compute) for **better precision** (each channel optimized independently, preserving more information)

**Key Insight**: Calibration is about finding representative data statistics. The rule of thumb: 100-1000 diverse samples usually suffice. Per-channel quantization is worth the complexity for sensitive layers (first/last layers, attention) but overkill for bulk middle layers.
### END SOLUTION

### Question 5: Production Deployment
In mobile/edge deployment scenarios:
- When is 4× memory reduction worth <1% accuracy loss? _____
- Why might you keep certain layers in FP32? _____
- How does quantization affect battery life? _____

### BEGIN SOLUTION
**Answer 5: Production Deployment**
- When 4× reduction worth <1% loss: **Always in memory-constrained environments** (mobile devices with <4GB RAM, edge devices with <512MB, embedded systems). Also when **serving cost matters** (4× smaller = 4× more users per server) or **latency critical** (4× faster loading from disk/network).

- Keep layers in FP32: **First layer** (input quantization loses information), **last layer** (output precision matters for final predictions), **attention layers** (sensitive to precision for softmax stability), and **layers with extreme activation ranges** (quantization error amplifies).

- Battery life impact: **2-4× improvement** due to (1) **less memory access** = lower DRAM power, (2) **INT8 operations use less energy** than FP32 ALUs, (3) **faster inference** = shorter active time. Typical mobile inference: 60% energy from memory, 30% from compute, 10% other.

**Key Insight**: Quantization is essential for edge AI. The 1% accuracy loss is usually imperceptible to users, but 4× memory savings and 2-3× speedup enable entirely new applications (real-time on-device AI, offline functionality, privacy-preserving local inference).
### END SOLUTION
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Quantization Shrinks Models

**What you built:** A complete INT8 quantization system with calibration and memory tracking.

**Why it matters:** A 400MB model becomes 100MB, small enough to run on a phone! Quantization
is how production ML deploys large models to edge devices, achieving 4x memory reduction with
minimal accuracy loss.

Your quantization system is ready for production deployment!
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
    print(f"Quantized INT8: {quantized_bytes:,} bytes")
    print(f"Compression: {original_bytes / quantized_bytes:.0f}x smaller!")
    print(f"INT8 range: [{q_weights.data.min()}, {q_weights.data.max()}]")
    print(f"Restoration error: {error:.6f}")

    print("\n✨ Same values, 4x less memory!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_quantization()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Quantization

Congratulations! You've built a complete INT8 quantization system that can reduce model size by 4x with minimal accuracy loss!

### Key Accomplishments
- Built INT8 quantization with proper scaling and zero-point calculation
- Implemented QuantizedLinear layer with calibration support
- Created model-level quantization for complete neural networks
- Analyzed quantization trade-offs across different distributions and strategies
- Measured real memory savings and performance improvements
- All tests pass (validated by `test_module()`)

### Systems Insights Discovered
- Memory scaling: INT8 reduces storage by 4x (32 bits to 8 bits per parameter)
- Calibration trade-offs: Sample data quality affects quantization accuracy
- Hardware efficiency: Specialized INT8 instructions provide 2-4x speedup
- Deployment benefits: Smaller models fit on mobile and edge devices

Export with: `tito module complete 15`

Quantization is one of the most impactful optimization techniques — reducing precision to INT8 delivers 4x memory savings with minimal accuracy loss.
"""
