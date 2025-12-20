# Module 09: Convolutions

:::{admonition} Module Info
:class: note

**ARCHITECTURE TIER** | Difficulty: ●●●○ | Time: 6-8 hours | Prerequisites: 01-08

**Prerequisites: Modules 01-08** assumes you have:
- Built the complete training pipeline (Modules 01-08)
- Implemented DataLoader for batch processing (Module 05)
- Understanding of parameter initialization, forward/backward passes, and optimization

If you can train an MLP on MNIST using your training loop and DataLoader, you're ready.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F09_convolutions%2F09_convolutions.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/09_convolutions/09_convolutions.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/09_convolutions.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Spatial operations transform machine learning from working with flattened vectors to understanding images and spatial patterns. When you look at a photo, your brain naturally processes spatial relationships: edges connect to form textures, textures form objects. Convolution gives neural networks this same capability by detecting local patterns through sliding filters across images.

This module implements Conv2d, MaxPool2d, and AvgPool2d with explicit loops to reveal the true computational cost of spatial processing. You'll see why a single forward pass through a convolutional layer can require billions of operations, and why efficient implementations are critical for computer vision.

By the end, your spatial operations will enable convolutional neural networks (CNNs) that can classify images, detect objects, and extract hierarchical visual features.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** Conv2d with explicit 7-nested loops revealing O(B×C×H×W×K²×C_in) computational complexity
- **Master** spatial dimension calculations with stride, padding, and kernel size interactions
- **Understand** receptive fields, parameter sharing, and translation equivariance in CNNs
- **Analyze** memory vs computation trade-offs: pooling reduces spatial dimensions 4x while preserving features
- **Connect** your implementations to production CNN architectures like ResNet and VGG
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Spatial Operations
flowchart LR
    subgraph "Your Spatial Operations"
        A["Conv2d<br/>Spatial feature extraction"]
        B["MaxPool2d<br/>Strong feature selection"]
        C["AvgPool2d<br/>Smooth feature averaging"]
    end

    D["Input Image<br/>(B, C_in, H, W)"]
    E["Feature Maps<br/>(B, C_out, H', W')"]
    F["Pooled Features<br/>(B, C, H/2, W/2)"]

    D --> A --> E
    E --> B --> F
    E --> C --> F

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `Conv2d.__init__()` | He initialization for ReLU networks |
| 2 | `Conv2d.forward()` | 7-nested loops for spatial convolution |
| 3 | `MaxPool2d.forward()` | Maximum selection in sliding windows |
| 4 | `AvgPool2d.forward()` | Average pooling for smooth features |

**The pattern you'll enable:**
```python
# Building a CNN block
conv = Conv2d(3, 64, kernel_size=3, padding=1)
pool = MaxPool2d(kernel_size=2, stride=2)

x = Tensor(image_batch)  # (32, 3, 224, 224)
features = pool(ReLU()(conv(x)))  # (32, 64, 112, 112)
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Dilated convolutions (PyTorch supports this with `dilation` parameter)
- Grouped convolutions (that's for efficient architectures like MobileNet)
- Depthwise separable convolutions (advanced optimization technique)
- Transposed convolutions for upsampling (used in GANs and segmentation)
- Optimized implementations (cuDNN uses Winograd algorithm and FFT convolution)

**You are building the foundational spatial operations.** Advanced convolution variants and GPU optimizations come later.

## API Reference

This section provides a quick reference for the spatial operations you'll build. Use it as your guide while implementing and debugging convolution and pooling layers.

### Conv2d Constructor

```python
Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
```

Creates a 2D convolutional layer with learnable filters.

**Parameters:**
- `in_channels`: Number of input channels (e.g., 3 for RGB)
- `out_channels`: Number of output feature maps
- `kernel_size`: Size of convolution kernel (int or tuple)
- `stride`: Stride of convolution (default: 1)
- `padding`: Zero-padding added to input (default: 0)
- `bias`: Whether to add learnable bias (default: True)

**Weight shape:** `(out_channels, in_channels, kernel_h, kernel_w)`

### MaxPool2d Constructor

```python
MaxPool2d(kernel_size, stride=None, padding=0)
```

Creates a max pooling layer for spatial dimension reduction.

**Parameters:**
- `kernel_size`: Size of pooling window (int or tuple)
- `stride`: Stride of pooling (default: same as kernel_size)
- `padding`: Zero-padding added to input (default: 0)

### AvgPool2d Constructor

```python
AvgPool2d(kernel_size, stride=None, padding=0)
```

Creates an average pooling layer for smooth spatial reduction.

**Parameters:**
- `kernel_size`: Size of pooling window (int or tuple)
- `stride`: Stride of pooling (default: same as kernel_size)
- `padding`: Zero-padding added to input (default: 0)

### Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `forward(x: Tensor) -> Tensor` | Apply spatial operation to input |
| `parameters` | `parameters() -> list` | Return trainable parameters (Conv2d only) |
| `__call__` | `__call__(x: Tensor) -> Tensor` | Enable `layer(x)` syntax |

### Output Shape Calculation

For both convolution and pooling:

```
output_height = (input_height + 2×padding - kernel_height) ÷ stride + 1
output_width = (input_width + 2×padding - kernel_width) ÷ stride + 1
```

## Core Concepts

This section covers the fundamental ideas you need to understand spatial operations deeply. These concepts apply to every computer vision system, from simple image classifiers to advanced object detectors.

### Convolution Operation

Convolution detects local patterns by sliding a small filter (kernel) across the entire input, computing weighted sums at each position. Think of it as using a template to find matching patterns everywhere in an image.

Here's how your implementation performs this operation:

```python
def forward(self, x):
    # Calculate output dimensions
    out_height = (in_height + 2 * self.padding - kernel_h) // self.stride + 1
    out_width = (in_width + 2 * self.padding - kernel_w) // self.stride + 1

    # Initialize output
    output = np.zeros((batch_size, out_channels, out_height, out_width))

    # Explicit 7-nested loop convolution
    for b in range(batch_size):
        for out_ch in range(out_channels):
            for out_h in range(out_height):
                for out_w in range(out_width):
                    in_h_start = out_h * self.stride
                    in_w_start = out_w * self.stride

                    conv_sum = 0.0
                    for k_h in range(kernel_h):
                        for k_w in range(kernel_w):
                            for in_ch in range(in_channels):
                                input_val = padded_input[b, in_ch,
                                                       in_h_start + k_h,
                                                       in_w_start + k_w]
                                weight_val = self.weight.data[out_ch, in_ch, k_h, k_w]
                                conv_sum += input_val * weight_val

                    output[b, out_ch, out_h, out_w] = conv_sum
```

The seven nested loops reveal where the computational cost comes from. For a typical CNN layer processing a batch of 32 RGB images (224×224) with 64 output channels and 3×3 kernels, this structure executes **2.8 billion multiply-accumulate operations** per forward pass. This is why optimized implementations matter.

Each output pixel summarizes information from a local neighborhood in the input. A 3×3 convolution looks at 9 pixels to produce each output value, enabling the network to detect local patterns like edges, corners, and textures.

### Stride and Padding

Stride controls how far the kernel moves between positions, and padding adds zeros around the input border. Together, they determine the output spatial dimensions and receptive field coverage.

**Stride = 1** means the kernel moves one pixel at a time, producing an output nearly as large as the input. **Stride = 2** means the kernel jumps two pixels, halving the spatial dimensions and dramatically reducing computation. A stride-2 convolution processes 4× fewer positions than stride-1.

**Padding** solves the border problem. Without padding, a 3×3 convolution on a 224×224 image produces a 222×222 output, shrinking the representation. With `padding=1`, you add a 1-pixel border of zeros, keeping the output at 224×224. This preserves spatial dimensions and ensures edge pixels get processed as many times as center pixels.

```
No Padding (shrinks):           Padding=1 (preserves):
Input: 5×5                      Input: 5×5 → Padded: 7×7
Kernel: 3×3                     Kernel: 3×3
Output: 3×3                     Output: 5×5

┌─────────┐                    ┌───────────┐
│ 1 2 3 4 5│                   │0 0 0 0 0 0 0│
│ 6 7 8 9 0│    3×3 kernel     │0 1 2 3 4 5 0│
│ 1 2 3 4 5│    →              │0 6 7 8 9 0 0│
│ 6 7 8 9 0│    3×3 output     │0 1 2 3 4 5 0│
│ 1 2 3 4 5│                   │0 6 7 8 9 0 0│
└─────────┘                    │0 1 2 3 4 5 0│
                                │0 0 0 0 0 0 0│
                                └───────────┘
                                5×5 output preserved
```

The formula connecting these parameters is:

```
output_size = (input_size + 2×padding - kernel_size) / stride + 1
```

For a 224×224 input with kernel=3, padding=1, stride=1:
```
output_size = (224 + 2×1 - 3) / 1 + 1 = 224
```

For the same input with stride=2:
```
output_size = (224 + 2×1 - 3) / 2 + 1 = 112
```

### Receptive Fields

The receptive field is the region in the original input that influences a particular output neuron. In a single 3×3 convolution, each output pixel has a 3×3 receptive field. But in deep networks, receptive fields grow with each layer.

Consider two stacked 3×3 convolutions. The first layer produces features with 3×3 receptive fields. The second layer takes those features as input, so each output now depends on a 5×5 region of the original input. Stack five 3×3 convolutions and you get an 11×11 receptive field.

This hierarchical growth is why CNNs work. Early layers detect edges and textures (small receptive fields), middle layers detect parts like eyes and wheels (medium receptive fields), and deep layers detect whole objects like faces and cars (large receptive fields).

```
Receptive Field Growth:

Layer 1 (3×3 conv):    Layer 2 (3×3 conv):    Layer 3 (3×3 conv):
┌───┐                  ┌─────┐                ┌───────┐
│▓▓▓│ → 3×3 RF        │▓▓▓▓▓│ → 5×5 RF       │▓▓▓▓▓▓▓│ → 7×7 RF
└───┘                  └─────┘                └───────┘

Stacking N 3×3 convolutions:
Receptive Field = 1 + 2×N

VGG-16 uses this principle: stack many small kernels instead of few large ones.
```

Parameter sharing means the same 3×3 kernel processes every position in the image. This drastically reduces parameters compared to fully connected layers while maintaining translation equivariance: if you shift the input, the output shifts identically.

### Pooling Operations

Pooling reduces spatial dimensions while preserving important features. Max pooling selects the strongest activation in each window, preserving sharp features like edges. Average pooling computes the mean, creating smoother, more general features.

Here's how max pooling works in your implementation:

```python
def forward(self, x):
    # Calculate output dimensions
    out_height = (in_height + 2 * self.padding - kernel_h) // self.stride + 1
    out_width = (in_width + 2 * self.padding - kernel_w) // self.stride + 1

    output = np.zeros((batch_size, channels, out_height, out_width))

    # Explicit nested loop max pooling
    for b in range(batch_size):
        for c in range(channels):
            for out_h in range(out_height):
                for out_w in range(out_width):
                    in_h_start = out_h * self.stride
                    in_w_start = out_w * self.stride

                    # Find maximum in window
                    max_val = -np.inf
                    for k_h in range(kernel_h):
                        for k_w in range(kernel_w):
                            input_val = padded_input[b, c,
                                                   in_h_start + k_h,
                                                   in_w_start + k_w]
                            max_val = max(max_val, input_val)

                    output[b, c, out_h, out_w] = max_val
```

A 2×2 max pooling with stride=2 divides spatial dimensions by 2, reducing memory and computation by 4×. For a 224×224×64 feature map (12.8 MB), pooling produces 112×112×64 (3.2 MB), saving 9.6 MB.

Max pooling provides translation invariance: if a cat's ear moves one pixel, the max in that region remains roughly the same, making the network robust to small shifts. This is crucial for object recognition where precise pixel alignment doesn't matter.

Average pooling smooths features by averaging windows, useful for global feature summarization. Modern architectures often use global average pooling (averaging entire feature maps to single values) instead of fully connected layers, dramatically reducing parameters.

### Output Shape Calculation

Understanding output shapes is critical for building CNNs. A shape mismatch crashes your network, while correct dimensions ensure features flow properly through layers.

The output shape formula applies to both convolution and pooling:

```
H_out = ⌊(H_in + 2×padding - kernel_h) / stride⌋ + 1
W_out = ⌊(W_in + 2×padding - kernel_w) / stride⌋ + 1
```

The floor operation (⌊⌋) ensures integer dimensions. If the calculation doesn't divide evenly, the rightmost and bottommost regions get ignored.

**Example calculations:**

```
Input: (32, 3, 224, 224)  [batch=32, RGB channels, 224×224 image]

Conv2d(3, 64, kernel_size=3, padding=1, stride=1):
H_out = (224 + 2×1 - 3) / 1 + 1 = 224
W_out = (224 + 2×1 - 3) / 1 + 1 = 224
Output: (32, 64, 224, 224)

MaxPool2d(kernel_size=2, stride=2):
H_out = (224 + 0 - 2) / 2 + 1 = 112
W_out = (224 + 0 - 2) / 2 + 1 = 112
Output: (32, 64, 112, 112)

Conv2d(64, 128, kernel_size=3, padding=0, stride=2):
H_out = (112 + 0 - 3) / 2 + 1 = 55
W_out = (112 + 0 - 3) / 2 + 1 = 55
Output: (32, 128, 55, 55)
```

**Common patterns:**
- **Same convolution** (padding=1, stride=1, kernel=3): Preserves spatial dimensions
- **Stride-2 convolution**: Halves dimensions, replaces pooling in some architectures (ResNet)
- **2×2 pooling, stride=2**: Classic dimension reduction, halves H and W

### Computational Complexity

Convolution is expensive. The explicit loops reveal exactly why: you're visiting every position in the output, and for each position, sliding over the entire kernel across all input channels.

For a single Conv2d forward pass:
```
Operations = B × C_out × H_out × W_out × C_in × K_h × K_w
```

**Example:** Batch=32, Input=(3, 224, 224), Conv2d(3→64, kernel=3, padding=1, stride=1)
```
Operations = 32 × 64 × 224 × 224 × 3 × 3 × 3
          = 32 × 64 × 50,176 × 27
          = 2,764,800,000 multiply-accumulate operations
          ≈ 2.8 billion operations per forward pass!
```

This is why kernel size matters enormously. A 7×7 kernel requires (7×7)/(3×3) = 5.4× more computation than 3×3. Modern architectures favor stacking multiple 3×3 convolutions instead of using large kernels.

Pooling operations are cheap by comparison: no learnable parameters, just comparison or addition operations. A 2×2 max pooling visits each output position once and compares 4 values, requiring only 4× comparisons per output.

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Conv2d (K×K) | O(B×C_out×H×W×C_in×K²) | Cubic in spatial dims, quadratic in kernel |
| MaxPool2d (K×K) | O(B×C×H×W×K²) | No channel mixing, just spatial reduction |
| AvgPool2d (K×K) | O(B×C×H×W×K²) | Same as MaxPool but with addition |

Memory consumption follows the output shape. A (32, 64, 224, 224) float32 tensor requires:
```
32 × 64 × 224 × 224 × 4 bytes = 411 MB
```

This is why batch size matters: doubling batch size doubles memory usage. GPUs have limited memory (typically 8-24 GB), constraining how large your batches and feature maps can be.

## Common Errors

These are the errors you'll encounter most often when working with spatial operations. Understanding why they happen will save you hours of debugging CNNs.

### Shape Mismatch in Conv2d

**Error**: `ValueError: Expected 4D input (batch, channels, height, width), got (3, 224, 224)`

Conv2d requires 4D input: (batch, channels, height, width). If you forget the batch dimension, the layer interprets channels as batch, height as channels, causing chaos.

**Fix**: Add batch dimension: `x = x.reshape(1, 3, 224, 224)` or ensure your data pipeline always includes batch dimension.

### Dimension Calculation Errors

**Error**: Output shape is 55 when you expected 56

The floor operation in output dimension calculation can surprise you. If `(input + 2×padding - kernel) / stride` doesn't divide evenly, the result gets floored.

**Example**:
```python
# Input: 224×224, kernel=3, padding=0, stride=2
output_size = (224 + 0 - 3) // 2 + 1 = 221 // 2 + 1 = 110 + 1 = 111
```

**Fix**: Use calculators or test with dummy data to verify dimensions before building full architecture.

### Padding Value Confusion

**Error**: Max pooling produces zeros at borders when using `padding > 0`

If you pad max pooling input with zeros (constant_values=0), and your feature map has negative values, the padded zeros will be selected as maximums at borders, creating artifacts.

**Fix**: Pad max pooling with `-np.inf`:
```python
padded_input = np.pad(x.data, ..., constant_values=-np.inf)
```

### Stride/Kernel Mismatch in Pooling

**Error**: Overlapping pooling windows when stride ≠ kernel_size

By convention, pooling uses non-overlapping windows: `stride = kernel_size`. If you accidentally set stride=1 with kernel=2, windows overlap, creating redundant computation and unexpected behavior.

**Fix**: Ensure `stride = kernel_size` for pooling, or set `stride=None` to use default (equals kernel_size).

### Memory Overflow

**Error**: `RuntimeError: CUDA out of memory` or system hangs

Large feature maps consume enormous memory. A batch of 64 images at 224×224×64 channels = 1.3 GB for a single layer's output. Deep networks with many layers can exceed GPU memory.

**Fix**: Reduce batch size, use smaller images, or add more pooling layers to reduce spatial dimensions faster.

## Production Context

### Your Implementation vs. PyTorch

Your TinyTorch spatial operations and PyTorch's `torch.nn.Conv2d` share the same conceptual foundation: sliding kernels, stride, padding, output shape formulas. The differences lie in optimization and hardware support.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Backend** | NumPy loops (Python) | cuDNN (CUDA C++) |
| **Speed** | 1x (baseline) | 100-1000x faster on GPU |
| **Optimization** | Explicit loops | im2col + GEMM, Winograd, FFT |
| **Memory** | Straightforward allocation | Memory pooling, gradient checkpointing |
| **Features** | Basic conv + pool | Dilated, grouped, transposed, 3D convolutions |

### Code Comparison

The following comparison shows equivalent operations in TinyTorch and PyTorch. Notice how the API mirrors perfectly, making your knowledge transfer directly to production frameworks.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.core.spatial import Conv2d, MaxPool2d, AvgPool2d
from tinytorch.core.activations import ReLU

# Build a CNN block
conv1 = Conv2d(3, 64, kernel_size=3, padding=1)
conv2 = Conv2d(64, 128, kernel_size=3, padding=1)
pool = MaxPool2d(kernel_size=2, stride=2)  # Or use AvgPool2d for smooth features
relu = ReLU()

# Forward pass
x = Tensor(image_batch)  # (32, 3, 224, 224)
x = relu(conv1(x))       # (32, 64, 224, 224)
x = pool(x)              # (32, 64, 112, 112)
x = relu(conv2(x))       # (32, 128, 112, 112)
x = pool(x)              # (32, 128, 56, 56)
```
````

````{tab-item} ⚡ PyTorch
```python
import torch
import torch.nn as nn

# Build a CNN block
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
relu = nn.ReLU()

# Forward pass (identical structure!)
x = torch.tensor(image_batch, dtype=torch.float32)
x = relu(conv1(x))
x = pool(x)
x = relu(conv2(x))
x = pool(x)
```
````
`````

Let's walk through each line to understand the comparison:

- **Lines 1-2 (Imports)**: TinyTorch separates spatial operations and activations into different modules; PyTorch consolidates into `torch.nn`. Same concepts, different organization.
- **Lines 4-7 (Layer creation)**: Identical API. Both use `Conv2d(in, out, kernel_size, padding)` and `MaxPool2d(kernel_size, stride)`. The parameter names and semantics are identical.
- **Line 10 (Input)**: TinyTorch wraps in `Tensor`; PyTorch uses `torch.tensor()` with explicit dtype. Same abstraction.
- **Lines 11-14 (Forward pass)**: Identical call patterns. ReLU activations, pooling for dimension reduction, growing channels (3→64→128). This is the standard CNN building block.
- **Shapes**: Every intermediate shape matches between frameworks because the formulas are identical.

```{tip} What's Identical

Convolution mathematics, stride and padding formulas, receptive field growth, and parameter sharing. The APIs are intentionally identical so your understanding transfers directly to production systems.
```

### Why Spatial Operations Matter at Scale

To appreciate why convolution optimization matters, consider the scale of production vision systems:

- **ResNet-50**: 25 million parameters, **4 billion operations** per image, processes thousands of images per second in production
- **YOLO object detection**: Processes 30 FPS video at 1080p, requiring **60 billion convolution operations per second**
- **Self-driving cars**: Run 10+ CNN models simultaneously on 6 cameras at 30 FPS, consuming **300 billion operations per second** with 50ms latency budget

A single forward pass of your educational Conv2d might take 800ms on CPU. The equivalent PyTorch operation runs in 8ms on GPU using cuDNN optimizations like im2col matrix multiplication and Winograd transforms. This 100× speedup is the difference between research prototypes and production systems.

Modern frameworks achieve this through:
- **im2col + GEMM**: Transforms convolution into matrix multiplication, leveraging highly optimized BLAS libraries
- **Winograd algorithm**: Reduces multiplication count for small kernels (3×3, 5×5) by 2.25×
- **FFT convolution**: For large kernels, Fourier transforms reduce complexity from O(n²) to O(n log n)

## Check Your Understanding

Test yourself with these systems thinking questions. They're designed to build intuition for the spatial operations and performance characteristics you'll encounter in real CNN architectures.

**Q1: Output Shape Calculation**

Given input (32, 3, 128, 128), what's the output shape after Conv2d(3, 64, kernel_size=5, padding=2, stride=2)?

```{admonition} Answer
:class: dropdown

Calculate height and width:
```
H_out = (128 + 2×2 - 5) / 2 + 1 = (128 + 4 - 5) / 2 + 1 = 127 / 2 + 1 = 63 + 1 = 64
W_out = (128 + 2×2 - 5) / 2 + 1 = 64
```

Output shape: **(32, 64, 64, 64)**

Batch and channels change (3→64), spatial dimensions halve due to stride=2.
```

**Q2: Parameter Counting**

How many parameters in Conv2d(3, 64, kernel_size=3, bias=True)?

```{admonition} Answer
:class: dropdown

Weight parameters: out_channels × in_channels × kernel_h × kernel_w
```
Weight: 64 × 3 × 3 × 3 = 1,728 parameters
Bias: 64 parameters
Total: 1,792 parameters
```

Compare this to a fully connected layer for 224×224 RGB images:
```
Dense(224×224×3, 64) = 150,528 × 64 = 9,633,792 parameters!
```

Convolution achieves **5,373× fewer parameters** through parameter sharing!
```

**Q3: Computational Complexity**

For input (16, 64, 56, 56) and Conv2d(64, 128, kernel_size=3, padding=1, stride=1), how many multiply-accumulate operations?

```{admonition} Answer
:class: dropdown

Operations = B × C_out × H_out × W_out × C_in × K_h × K_w

First calculate output dimensions:
```
H_out = (56 + 2×1 - 3) / 1 + 1 = 56
W_out = (56 + 2×1 - 3) / 1 + 1 = 56
```

Then total operations:
```
16 × 128 × 56 × 56 × 64 × 3 × 3
= 16 × 128 × 3,136 × 576
= 3,707,764,736 operations
≈ 3.7 billion operations per forward pass!
```

This is why batch size directly impacts training time: doubling batch doubles operations.
```

**Q4: Memory Calculation**

What's the memory requirement for storing the output of Conv2d(3, 256, kernel_size=7, stride=2, padding=3) on input (64, 3, 224, 224)?

```{admonition} Answer
:class: dropdown

First calculate output dimensions:
```
H_out = (224 + 2×3 - 7) / 2 + 1 = (224 + 6 - 7) / 2 + 1 = 223 / 2 + 1 = 111 + 1 = 112
W_out = 112
```

Output shape: (64, 256, 112, 112)

Memory (float32 = 4 bytes):
```
64 × 256 × 112 × 112 × 4 = 825,753,600 bytes
≈ 826 MB for a single layer's output!
```

This is why deep CNNs require GPUs with large memory (16+ GB). Storing activations for backpropagation across 50+ layers quickly exceeds memory limits.
```

**Q5: Receptive Field Growth**

Starting with 224×224 input, you stack: Conv(3×3, stride=1) → MaxPool(2×2, stride=2) → Conv(3×3, stride=1) → Conv(3×3, stride=1). What's the receptive field of the final layer?

```{admonition} Answer
:class: dropdown

Track receptive field growth through each layer:

Layer 1 - Conv(3×3, stride=1): RF = 3
Layer 2 - MaxPool(2×2, stride=2): RF = 3 + (2-1)×1 = 4
Layer 3 - Conv(3×3, stride=1): RF = 4 + (3-1)×2 = 8  (stride accumulates)
Layer 4 - Conv(3×3, stride=1): RF = 8 + (3-1)×2 = 12

**Receptive field = 12×12**

Each neuron in the final layer sees a 12×12 region of the original input. This is why stacking layers with stride/pooling is crucial: it grows the receptive field so deeper layers can detect larger patterns.

Formula: RF_new = RF_old + (kernel_size - 1) × stride_product

where stride_product is the accumulated stride from all previous layers.
```

## Further Reading

For students who want to understand the academic foundations and explore spatial operations further:

### Seminal Papers

- **Gradient-Based Learning Applied to Document Recognition** - LeCun et al. (1998). The paper that launched convolutional neural networks, introducing LeNet-5 for handwritten digit recognition. Essential reading for understanding why convolution works for vision. [IEEE](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

- **ImageNet Classification with Deep Convolutional Neural Networks** - Krizhevsky et al. (2012). AlexNet, the breakthrough that demonstrated CNNs could win ImageNet. Introduced ReLU, dropout, and data augmentation patterns still used today. [NeurIPS](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

- **Very Deep Convolutional Networks for Large-Scale Image Recognition** - Simonyan & Zisserman (2014). VGG networks showed that stacking many 3×3 convolutions works better than few large kernels. This principle guides modern architecture design. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)

- **Deep Residual Learning for Image Recognition** - He et al. (2015). ResNet introduced skip connections that enable training 100+ layer networks. Revolutionized computer vision and won ImageNet 2015. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

### Additional Resources

- **CS231n: Convolutional Neural Networks for Visual Recognition** - Stanford course notes with excellent visualizations of convolution, receptive fields, and feature maps: [https://cs231n.github.io/convolutional-networks/](https://cs231n.github.io/convolutional-networks/)
- **Textbook**: "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter 9 covers convolutional networks with mathematical depth
- **Distill.pub**: "Feature Visualization" - Interactive article showing what CNN filters learn at different depths: [https://distill.pub/2017/feature-visualization/](https://distill.pub/2017/feature-visualization/)

## What's Next

```{seealso} Coming Up: Module 10 - Tokenization

Shift from spatial processing (images) to sequential processing (text). You'll implement tokenizers that convert text into numeric representations, unlocking natural language processing and transformers.
```

**Preview - How Your Spatial Operations Enable Future Work:**

| Module | What It Does | Your Spatial Ops In Action |
|--------|--------------|---------------------------|
| **Milestone 3: CNN** | Complete CNN for CIFAR-10 | Stack your Conv2d and MaxPool2d for image classification |
| **Module 17: Acceleration** | Optimize convolution | Replace loops with im2col and vectorized operations |
| **Vision Projects** | Object detection, segmentation | Your spatial foundations scale to advanced architectures |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/09_convolutions/09_convolutions.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/09_convolutions/09_convolutions.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
