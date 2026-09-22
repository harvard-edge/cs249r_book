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
# Module 09: Convolutions - Processing Images with Spatial Operations

Welcome to Module 09! You'll implement spatial operations that transform machine learning from working with simple vectors to understanding images and spatial patterns.

## 🔗 Prerequisites & Progress
**You've Built**: Complete training pipeline with MLPs, optimizers, and data loaders (`Tensor`, `Autograd`, `Linear`, `Losses`, `Optimizers`, `Trainer`)
**You'll Build**: Spatial operations (`Conv2d`, `MaxPool2d`, `AvgPool2d`, and `BatchNorm2d`) for image processing
**You'll Enable**: Convolutional Neural Networks (CNNs) for computer vision and spatial representation learning

<div align="center">
  <img src="convolution_blueprint.svg" alt="TinyTorch Execution Datapath: Module 09 Convolutions Highlighted" width="380px">
</div>

### Architectural Roadmap

| Stage | Subsystem | Primitives & Capabilities | Status |
| :--- | :--- | :--- | :--- |
| **Modules 01–08** | Training Foundation | `Tensor`, `Autograd`, `Linear`, `CrossEntropyLoss`, `AdamW`, `Trainer` | Completed |
| **Module 09** | **Spatial Operations** | `Conv2d`, `MaxPool2d`, `AvgPool2d`, `BatchNorm2d` | **Active Subsystem** |
| **Modules 10–13** | Language & Attention | `BPETokenizer`, `Embedding`, `MultiHeadAttention`, `TransformerBlock` | Next Target |

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement `Conv2d` with explicit 7-nested loops to quantify its $\mathcal{O}(B \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot K_h \cdot K_w \cdot C_{\text{in}})$ computational complexity.
2. Build pooling operations (`MaxPool2d` and `AvgPool2d`) for spatial downsampling and translation invariance.
3. Formulate `BatchNorm2d` with running statistics tracking and a three-route backward derivation (direct, mean, and variance paths) across mini-batch axes.
4. Compose complete convolutional pipelines (`SimpleCNN`) demonstrating $>139\times$ parameter efficiency over dense networks.

---

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/09_convolutions/convolutions.ipynb`
**Building Side:** Code exports to `tinytorch.core.spatial`

<div align="center">
  <img src="convolution_margin_source.svg" alt="Source Code Mapping for Module 09 Convolutions" width="240px">
</div>

```python
# How to use this module:
from tinytorch.core.spatial import Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d
```

**Why this matters:**
- **Learning:** Complete spatial processing system in one focused module for deep understanding
- **Production:** Proper organization mirroring PyTorch's `torch.nn.Conv2d` and `torch.nn.BatchNorm2d`
- **Consistency:** All spatial convolution, normalization, and pooling primitives live cleanly in `core.spatial`
- **Integration:** Composes seamlessly with existing layers (`Linear`, `ReLU`) for complete CNN architectures
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01-08 must be complete

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for performance measurements)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (`Tensor` class from Module 01)
- `tinytorch.core.activations` (`ReLU` from Module 02)
- `tinytorch.core.layers` (`Linear` from Module 03)
- `tinytorch.core.autograd` (gradient tracking and reverse-mode AD from Module 06)

This module builds directly upon your training pipeline.
Spatial operations integrate with your existing layers and backpropagation engine.
"""

# %% nbgrader={"grade": false, "grade_id": "spatial-setup", "solution": false}
#| default_exp core.spatial
#| export

import numpy as np
rng = np.random.default_rng(7)
import time

from tinytorch.core.tensor import Tensor, Function
from tinytorch.core.activations import ReLU
from tinytorch.core.layers import Linear
import tinytorch.core.autograd  # completes every operation with its backward half

# %% [markdown]
r"""
## 💡 Introduction: What are Spatial Operations?

Spatial operations transform machine learning from working with simple vectors to understanding images and spatial patterns. When you look at a photo, your brain naturally processes spatial relationships, including edges, textures, and objects. Spatial operations give neural networks this same capability.

### The Two Core Spatial Operations

- **Convolution (`Conv2d`)**: Detects local patterns by sliding learnable filters across the spatial grid.
- **Pooling (`MaxPool2d`, `AvgPool2d`)**: Downsamples spatial dimensions while preserving dominant features and building translation tolerance.

---

### Visual Example: How Convolution Works

<div align="center">
  <img src="convolution_sliding_window.svg" alt="2D Convolution Sliding Window Dot Product" width="680px">
</div>

For a 2D input channel $\mathbf{X}$ and a $K \times K$ kernel $\mathbf{W}$:
$$(\mathbf{X} * \mathbf{W})[i, j] = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \mathbf{X}[i+m, j+n] \cdot \mathbf{W}[m, n]$$

At each valid spatial coordinate $(i, j)$, the kernel overlaps an input patch of size $K \times K$. The overlapping elements are multiplied elementwise and accumulated into a single output scalar.

### Why Spatial Operations Transform Machine Learning

| Characteristic | Fully-Connected Dense Layer (`Linear`) | 2D Convolution Layer (`Conv2d`) | MLSys & Architectural Advantage |
| :--- | :--- | :--- | :--- |
| **Input Shape** | Flattened $(3072,)$ vector | Structured $(3, 32, 32)$ image tensor | Preserves 2D spatial grid topology |
| **Weight Count** | $3{,}072 \times 14{,}400 \approx 44.2\text{M}$ weights | $16 \times (3 \times 3 \times 3) = 432$ weights | **$>100{,}000\times$ parameter reduction** via weight sharing |
| **Connectivity** | All-to-all dense connectivity | $3 \times 3$ local receptive field | Exploits local pixel correlation |
| **Inductive Bias** | Spatial shifts require retraining | Identical kernel applied across all coordinates | Translation equivariance: $f(\text{shift}(x)) = \text{shift}(f(x))$ |

Convolution achieves dramatic parameter reduction while preserving the spatial relationships critical for visual representations.
"""

# %% [markdown]
r"""
## 📐 Foundations: Convolution, Step by Step

### Understanding Convolution Step by Step

Convolution is fundamentally a **sliding window dot product**. Let us trace a $3 \times 4$ input convolved with a $2 \times 2$ kernel:

$$\mathbf{X} = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 0 & 1 & 2 \end{bmatrix}, \qquad \mathbf{K} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

- **Position $(0, 0)$**:
  $$\mathbf{X}_{0:2, 0:2} \odot \mathbf{K} = \begin{bmatrix} 1 & 2 \\ 5 & 6 \end{bmatrix} \odot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = 1(1) + 2(0) + 5(0) + 6(1) = 7$$
- **Position $(0, 1)$**:
  $$\mathbf{X}_{0:2, 1:3} \odot \mathbf{K} = \begin{bmatrix} 2 & 3 \\ 6 & 7 \end{bmatrix} \odot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = 2(1) + 3(0) + 6(0) + 7(1) = 9$$
- **Position $(0, 2)$**:
  $$\mathbf{X}_{0:2, 2:4} \odot \mathbf{K} = \begin{bmatrix} 3 & 4 \\ 7 & 8 \end{bmatrix} \odot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = 3(1) + 4(0) + 7(0) + 8(1) = 11$$
- **Position $(1, 0)$**:
  $$\mathbf{X}_{1:3, 0:2} \odot \mathbf{K} = \begin{bmatrix} 5 & 6 \\ 9 & 0 \end{bmatrix} \odot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = 5(1) + 6(0) + 9(0) + 0(1) = 5$$
- **Position $(1, 1)$**:
  $$\mathbf{X}_{1:3, 1:3} \odot \mathbf{K} = \begin{bmatrix} 6 & 7 \\ 0 & 1 \end{bmatrix} \odot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = 6(1) + 7(0) + 0(0) + 1(1) = 7$$
- **Position $(1, 2)$**:
  $$\mathbf{X}_{1:3, 2:4} \odot \mathbf{K} = \begin{bmatrix} 7 & 8 \\ 1 & 2 \end{bmatrix} \odot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = 7(1) + 8(0) + 1(0) + 2(1) = 9$$

Yielding the output feature map $\mathbf{Y} \in \mathbb{R}^{2 \times 3}$:
$$\mathbf{Y} = \begin{bmatrix} 7 & 9 & 11 \\ 5 & 7 & 9 \end{bmatrix}$$

---

### The Full 4D Convolution Formula

Extending beyond a single channel to batches and multi-channel filter banks:
$$O[b, c_{\text{out}}, i, j] = \text{bias}[c_{\text{out}}] + \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} I[b, c_{\text{in}}, i \cdot S_h + m, j \cdot S_w + n] \times W[c_{\text{out}}, c_{\text{in}}, m, n]$$

Where:
- $S_h, S_w$ denote vertical and horizontal strides.
- $K_h, K_w$ denote kernel height and width.
- $C_{\text{in}}, C_{\text{out}}$ denote input and output channel depths.

#### A Naming Confession: This Is Cross-Correlation

The sum above, and the one every `Conv2d` in this module computes, is *cross-correlation*, not convolution. True convolution flips the kernel before sliding it, reading $W[K_h - 1 - m, K_w - 1 - n]$ where the formula above reads $W[m, n]$. Every production framework, PyTorch and TensorFlow and JAX alike, skips the flip, because a kernel that is learned rather than designed simply absorbs it and the two operations reach identical solutions. The mathematical name stuck to the engineering shortcut, so `Conv2d` everywhere means the un-flipped sum, and the flip only matters when you port a hand-designed filter in from signal processing.

Also note that $S_h, S_w$ and $P_h, P_w$ are written separately here because that is the general case. TinyTorch's `Conv2d` takes one integer for stride and one for padding, so $S_h = S_w$ and $P_h = P_w$ always; handing it a tuple raises a bare `TypeError` from the arithmetic rather than a helpful message. `MaxPool2d` and `AvgPool2d` do accept a tuple stride, so the layers are not interchangeable on that argument.

---

### Pooling: Spatial Summarization

Given the input feature map:
$$\mathbf{X} = \begin{bmatrix} 1 & 3 & 2 & 4 \\ 5 & 6 & 7 & 8 \\ 2 & 9 & 1 & 3 \\ 0 & 1 & 9 & 3 \end{bmatrix}$$

- **Max Pooling** ($2 \times 2$ window, stride $2$):
  $$\mathbf{Y}_{\text{max}} = \begin{bmatrix} \max(\{1, 3, 5, 6\}) & \max(\{2, 4, 7, 8\}) \\ \max(\{2, 9, 0, 1\}) & \max(\{1, 3, 9, 3\}) \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 9 & 9 \end{bmatrix}$$

- **Average Pooling** ($2 \times 2$ window, stride $2$):
  $$\mathbf{Y}_{\text{avg}} = \begin{bmatrix} \frac{1+3+5+6}{4} & \frac{2+4+7+8}{4} \\ \frac{2+9+0+1}{4} & \frac{1+3+9+3}{4} \end{bmatrix} = \begin{bmatrix} 3.75 & 5.25 \\ 3.00 & 4.00 \end{bmatrix}$$

---

### Computational & Memory Budget

For a standard early convolutional layer with input $(1, 3, 224, 224)$, 64 filters of size $(3, 3, 3)$, stride $1$, and padding $1$:

| Component | Tensor Shape | Elements | Precision & Memory |
| :--- | :--- | :--- | :--- |
| **Input Activations** | $(1, 3, 224, 224)$ | $150{,}528$ | $4\text{ B/elem} \to 602.1\text{ KB}$ |
| **Filter Weights** | $(64, 3, 3, 3)$ | $1{,}728$ | $4\text{ B/elem} \to 6.9\text{ KB}$ |
| **Biases** | $(64,)$ | $64$ | $4\text{ B/elem} \to 256\text{ B}$ |
| **Output Activations** | $(1, 64, 224, 224)$ | $3{,}211{,}264$ | $4\text{ B/elem} \to 12.85\text{ MB}$ |
| **Total Forward Memory** | — | $3{,}363{,}584$ | $\mathbf{\approx 13.45\text{ MB}}$ |
| **Compute Operations** | $1 \cdot 64 \cdot 224 \cdot 224 \cdot (3 \cdot 3 \cdot 3)$ | — | **$86.7\text{M MACs}$** ($173.4\text{ MFLOPs}$) |

Memory figures in this module are decimal throughout. KB means $10^3$ bytes and MB means $10^6$ bytes, so $602{,}112$ B reads as $602.1$ KB rather than the $588$ KiB a binary conversion would give.

Notice that kernel size impacts compute quadratically: switching from $3 \times 3$ ($9$ weights/channel) to $7 \times 7$ ($49$ weights/channel) increases compute by $\approx 5.44\times$!

---

### Key Properties That Enable Deep Learning
- **Translation Equivariance**: Translating an object in the input produces an identical translation in the feature map: $f(T_v(x)) = T_v(f(x))$.
- **Weight Sharing**: The same learned detector applies across every spatial location, drastically reducing capacity requirements.
- **Local Connectivity**: Pixels close together carry the highest statistical mutual information.
- **Hierarchical Representations**: Shallow layers detect edges; middle layers combine edges into motifs; deep layers compose motifs into semantic parts.
"""

# %% [markdown]
r"""
## 🏗️ Implementation: Building Spatial Operations

Now we implement convolution step by step using explicit nested loops. Seeing the raw loop mechanics reveals exactly where cache misses, memory bandwidth ceilings, and algorithmic complexity arise.

### The Seven Nested Loops of Convolution

```python
for batch in range(B):              # Loop 1: Process each batch item
    for out_ch in range(C_out):     # Loop 2: Each output channel / filter
        for out_h in range(H_out):  # Loop 3: Each output row
            for out_w in range(W_out):  # Loop 4: Each output column
                for k_h in range(K_h):      # Loop 5: Each kernel row
                    for k_w in range(K_w):      # Loop 6: Each kernel column
                        for in_ch in range(C_in):   # Loop 7: Each input channel
                            result += input[batch, in_ch, out_h * S + k_h, out_w * S + k_w] * weight[out_ch, in_ch, k_h, k_w]
```

| Loop Level | Iterator | Bound | Systems Execution Semantics |
| :--- | :--- | :--- | :--- |
| **Loop 1** | `batch` | $B$ | Independent batch samples (embarrassingly parallel across threads) |
| **Loop 2** | `out_ch` | $C_{\text{out}}$ | Separate filter bank kernels; parallel across CUDA thread blocks |
| **Loop 3** | `out_h` | $H_{\text{out}}$ | Vertical output spatial grid coordinate |
| **Loop 4** | `out_w` | $W_{\text{out}}$ | Horizontal output spatial grid coordinate |
| **Loop 5** | `k_h` | $K_h$ | Vertical kernel spatial coordinate |
| **Loop 6** | `k_w` | $K_w$ | Horizontal kernel spatial coordinate |
| **Loop 7** | `in_ch` | $C_{\text{in}}$ | Channel reduction sum (accumulated across input depth) |

$$\text{Total Operations} = B \times C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}} \times K_h \times K_w \times C_{\text{in}}$$

For standard batch processing ($B=32$, $C_{\text{out}}=64$, $H_{\text{out}}=224$, $W_{\text{out}}=224$, $K_h=3$, $K_w=3$, $C_{\text{in}}=3$):
$$32 \times 64 \times 224 \times 224 \times 3 \times 3 \times 3 = \mathbf{2{,}774{,}532{,}096\text{ operations (\approx 2.8 billion)}}$$
"""

# %% [markdown]
"""
### Shared Input Validation

All spatial operations (Conv2d, MaxPool2d, AvgPool2d) require 4D inputs shaped
as (batch, channels, height, width). Rather than duplicating this validation
logic three times, we define it once here.

This is NOT a student task. It is shared infrastructure.
"""

# %% nbgrader={"grade": false, "grade_id": "validate-4d-input", "solution": false}
#| export

def validate_4d_input(x: Tensor, layer_name: str) -> None:
    """
    Validate that input tensor is 4D (batch, channels, height, width).

    Provides educational error messages that anticipate common student
    mistakes: forgetting the batch dimension, passing flattened data, etc.

    Args:
        x: Input Tensor to validate
        layer_name: Name of the calling layer (for error messages)

    Raises:
        ValueError: If input is not 4D, with specific guidance per case
    """
    if len(x.shape) == 4:
        return  # Valid input

    if len(x.shape) == 3:
        raise ValueError(
            f"{layer_name} expected 4D input (batch, channels, height, width), got 3D: {x.shape}\n"
            f"  One dimension is missing; which one depends on what the tensor holds\n"
            f"  One image with {x.shape[0]} channels: x.reshape(1, {x.shape[0]}, {x.shape[1]}, {x.shape[2]})\n"
            f"  A batch of {x.shape[0]} single-channel images: x.reshape({x.shape[0]}, 1, {x.shape[1]}, {x.shape[2]})"
        )
    elif len(x.shape) == 2:
        raise ValueError(
            f"{layer_name} expected 4D input (batch, channels, height, width), got 2D: {x.shape}\n"
            f"  Got a matrix, expected an image tensor\n"
            f"  {layer_name} needs spatial dimensions (height, width) plus batch and channels\n"
            f"  If this is a flattened image, reshape it: x.reshape(1, channels, height, width)"
        )
    else:
        raise ValueError(
            f"{layer_name} expected 4D input (batch, channels, height, width), got {len(x.shape)}D: {x.shape}\n"
            f"  Wrong number of dimensions\n"
            f"  {layer_name} expects: (batch_size, channels, height, width)\n"
            f"  Reshape your input to 4D with the correct dimensions"
        )


# %% [markdown]
r"""
### Conv2d Implementation: Building the Core of Computer Vision

`Conv2d` is the workhorse of computer vision. It slides learned filter banks across multi-channel feature maps to detect spatial motifs ranging from oriented edges to complex textures.

#### Weight Initialization: He Normal for ReLU Networks

`Conv2d` uses He (Kaiming) normal initialization, calibrated specifically for rectified linear units:
$$\sigma = \sqrt{\frac{2}{\text{fan}_{\text{in}}}}, \qquad \text{fan}_{\text{in}} = C_{\text{in}} \times K_h \times K_w$$

- **Failure Mode**: Standard normal or overly small variance leads to exponential signal decay across deep layers.
- **MLSys Solution**: Scale weights inversely with the receptive field fan-in so variance is preserved across successive layer activations.

---

### Implementation Strategy: Decomposed Helpers

We decompose `Conv2d.forward` into four focused, modular helpers so each mathematical and memory concept can be verified independently:

1. **`_compute_output_shape(in_h, in_w)`** computes output spatial grid dimensions given kernel, stride, and padding.
2. **`_apply_padding(x_data)`** zero-pads spatial dimensions $(H, W)$ while preserving batch and channel axes.
3. **`_convolve_loops(padded, batch, oh, ow)`** executes the sliding window dot products and channel reductions.
4. **`forward(x)`** composes the pipeline and links gradient tracking via `Conv2dFunction`.
"""

# %% [markdown]
r"""
### Step 1: Output Shape Formula

Before allocating output memory buffers, the output dimensions must be determined:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2 P_h - K_h}{S_h} \right\rfloor + 1, \qquad W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} + 2 P_w - K_w}{S_w} \right\rfloor + 1$$

| Input Spatial | Kernel Size | Padding $P$ | Stride $S$ | Output Spatial | MLSys Spatial Effect |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $32 \times 32$ | $3 \times 3$ | 0 | 1 | $30 \times 30$ | Valid convolution (shrinks by $K-1$) |
| $32 \times 32$ | $3 \times 3$ | 1 | 1 | $32 \times 32$ | Same convolution (preserves spatial resolution) |
| $32 \times 32$ | $3 \times 3$ | 0 | 2 | $15 \times 15$ | Strided convolution without padding |
| $32 \times 32$ | $3 \times 3$ | 1 | 2 | $16 \times 16$ | Standard spatial downsampling ($H/2, W/2$) |
"""

# %% [markdown]
r"""
### Step 2: Zero Padding

Padding introduces zero-valued borders around spatial edges $(H, W)$, ensuring perimeter features receive equal filter coverage:

$$\mathbf{X} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \quad \xrightarrow{\text{pad } P=1} \quad \mathbf{X}_{\text{padded}} = \begin{bmatrix} 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 2 & 3 & 0 \\ 0 & 4 & 5 & 6 & 0 \\ 0 & 7 & 8 & 9 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$

Only the spatial dimensions $(H, W)$ are padded. The batch ($B$) and channel ($C$) axes remain completely untouched.
"""

# %% [markdown]
r"""
### Step 3: The Convolution Loops

<div align="center">
  <img src="convolution_margin_im2col.svg" alt="Receptive Field Patch Extraction" width="280px">
</div>

For each output coordinate $(b, c_{\text{out}}, h, w)$, we slice the corresponding multi-channel input patch and compute its inner product against filter kernel $W[c_{\text{out}}]$:

$$\text{Output}[b, c_{\text{out}}, h, w] = \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} X_{\text{pad}}[b, c_{\text{in}}, h \cdot S_h + k_h, w \cdot S_w + k_w] \cdot W[c_{\text{out}}, c_{\text{in}}, k_h, k_w]$$

---

### Backward Propagation Route

During backpropagation, gradients flow from the output tensor back to both input activations and filter weights:

<div align="center">
  <img src="convolution_backward_route.svg" alt="Convolution Backward Route: Gradients to Weights and Inputs" width="560px">
</div>
"""

# %% nbgrader={"grade": false, "grade_id": "conv2d-class", "solution": true}
#| export

class Conv2dFunction(Function):
    """
    The 2D convolution operation: forward runs the layer's sliding-window loops,
    backward computes the gradients.

    Computes gradients for Conv2d backward pass:
    - grad_input: gradient w.r.t. input (for backprop to previous layer)
    - grad_weight: gradient w.r.t. filters (for weight updates)
    - grad_bias: gradient w.r.t. bias (for bias updates)

    This uses explicit loops to show the gradient computation, matching
    the educational approach of the forward pass.
    """

    def forward(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        """
        Convolve the (already validated) input. The layer that owns the weights
        is passed as `layer`, so the student-written helpers below do the work:
        _apply_padding and _convolve_loops.
        """
        batch_size = x.shape[0]
        out_height, out_width = self.layer._compute_output_shape(x.shape[2], x.shape[3])
        padded_input = self.layer._apply_padding(x)
        output = self.layer._convolve_loops(padded_input, batch_size, out_height, out_width)
        if bias is not None:
            for out_ch in range(self.layer.out_channels):
                output[:, out_ch, :, :] += bias[out_ch]
        return output


    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Compute gradients for convolution inputs and parameters.

        Args:
            grad_output: Gradient flowing back from next layer
                        Shape: (batch_size, out_channels, out_height, out_width)

        Returns:
            Tuple of (grad_input, grad_weight, grad_bias)
        """
        x, weight = self.inputs[0], self.inputs[1]
        bias = self.inputs[2] if len(self.inputs) > 2 else None
        stride, padding, kernel_size = self.layer.stride, self.layer.padding, self.layer.kernel_size

        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, in_height, in_width = x.shape
        kernel_h, kernel_w = kernel_size

        # Apply padding to input if needed (for gradient computation)
        if padding > 0:
            padded_input = np.pad(x.data,
                                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                                mode='constant', constant_values=0)
        else:
            padded_input = x.data

        # Initialize gradients
        grad_input_padded = np.zeros_like(padded_input)
        grad_weight = np.zeros_like(weight.data)
        grad_bias = None if bias is None else np.zeros_like(bias.data)

        # Compute gradients using explicit loops (educational approach)
        for b in range(batch_size):
            for out_ch in range(out_channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Position in input
                        in_h_start = out_h * stride
                        in_w_start = out_w * stride

                        # Gradient value flowing back to this position
                        grad_val = grad_output[b, out_ch, out_h, out_w]

                        # Distribute gradient to weight and input
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    # Input position
                                    in_h = in_h_start + k_h
                                    in_w = in_w_start + k_w

                                    # Gradient w.r.t. weight
                                    grad_weight[out_ch, in_ch, k_h, k_w] += (
                                        padded_input[b, in_ch, in_h, in_w] * grad_val
                                    )

                                    # Gradient w.r.t. input
                                    grad_input_padded[b, in_ch, in_h, in_w] += (
                                        weight.data[out_ch, in_ch, k_h, k_w] * grad_val
                                    )

        # Compute gradient w.r.t. bias (sum over batch and spatial dimensions)
        if grad_bias is not None:
            for out_ch in range(out_channels):
                grad_bias[out_ch] = grad_output[:, out_ch, :, :].sum()

        # Remove padding from input gradient
        if padding > 0:
            grad_input = grad_input_padded[:, :,
                                          padding:-padding,
                                          padding:-padding]
        else:
            grad_input = grad_input_padded

        # One gradient per input: (x, weight) or (x, weight, bias).
        if bias is None:
            return grad_input, grad_weight
        return grad_input, grad_weight, grad_bias

class Conv2d:
    """
    2D Convolution layer for spatial feature extraction.

    Implements convolution with explicit loops to demonstrate
    computational complexity and memory access patterns.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output feature maps
        kernel_size: Size of convolution kernel (int or tuple)
        stride: Stride of convolution (default: 1)
        padding: Zero-padding added to input (default: 0)
        bias: Whether to add learnable bias (default: True)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int],
                 stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        """
        Initialize Conv2d layer with proper weight initialization.

        TODO: Complete Conv2d initialization

        APPROACH:
        1. Store hyperparameters (channels, kernel_size, stride, padding)
        2. Initialize weights using He initialization for ReLU compatibility
        3. Initialize bias (if enabled) to zeros
        4. Use proper shapes: weight (out_channels, in_channels, kernel_h, kernel_w)

        WEIGHT INITIALIZATION:
        - He init: std = sqrt(2 / (in_channels * kernel_h * kernel_w))
        - This prevents vanishing/exploding gradients with ReLU

        HINT: Convert kernel_size to tuple if it's an integer
        """
        ### BEGIN SOLUTION role="scaffold"
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        # He initialization for ReLU networks
        kernel_h, kernel_w = self.kernel_size
        fan_in = in_channels * kernel_h * kernel_w
        std = np.sqrt(2.0 / fan_in)

        # Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        self.weight = Tensor(rng.normal(0, std,
                           (out_channels, in_channels, kernel_h, kernel_w)),
                           requires_grad=True)

        # Bias initialization
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
        ### END SOLUTION

    def _compute_output_shape(self, in_h: int, in_w: int) -> tuple[int, int]:
        """
        Calculate output spatial dimensions for convolution.

        TODO: Apply the convolution output size formula

        APPROACH:
        1. Apply the standard formula for each spatial dimension:
           out_dim = (in_dim + 2 * padding - kernel_size) // stride + 1
        2. Return (out_height, out_width) as a tuple

        EXAMPLE:
        >>> conv = Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        >>> oh, ow = conv._compute_output_shape(32, 32)
        >>> print(oh, ow)  # 32, 32 (same padding preserves size)

        >>> conv2 = Conv2d(3, 16, kernel_size=3, padding=0, stride=1)
        >>> oh, ow = conv2._compute_output_shape(32, 32)
        >>> print(oh, ow)  # 30, 30 (shrinks by kernel_size - 1)

        HINT: The formula is the same for height and width, just with
        different input dimensions.
        """
        ### BEGIN SOLUTION role="scaffold"
        kernel_h, kernel_w = self.kernel_size
        out_height = (in_h + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_w + 2 * self.padding - kernel_w) // self.stride + 1
        return out_height, out_width
        ### END SOLUTION

    def _apply_padding(self, x_data: np.ndarray) -> np.ndarray:
        """
        Zero-pad the spatial dimensions of the input numpy array.

        TODO: Add zero-padding around spatial dimensions (height, width)

        APPROACH:
        1. If self.padding > 0, use np.pad to add zeros around spatial dims
        2. Only pad dimensions 2 and 3 (height, width), not batch or channels
        3. If self.padding == 0, return the input unchanged

        EXAMPLE:
        >>> conv = Conv2d(1, 1, kernel_size=3, padding=1)
        >>> x = np.ones((1, 1, 3, 3))
        >>> padded = conv._apply_padding(x)
        >>> print(padded.shape)  # (1, 1, 5, 5), since 3+2*1=5

        HINT: np.pad takes a tuple of (before, after) pairs per dimension.
        Use (0,0) for batch and channel dims, (padding, padding) for spatial.
        """
        ### BEGIN SOLUTION role="scaffold"
        if self.padding > 0:
            return np.pad(x_data,
                         ((0, 0), (0, 0),
                          (self.padding, self.padding),
                          (self.padding, self.padding)),
                         mode='constant', constant_values=0)
        else:
            return x_data
        ### END SOLUTION

    def _convolve_loops(self, padded: np.ndarray, batch_size: int, out_h: int, out_w: int) -> np.ndarray:
        """
        The core convolution: sliding window dot products over the input.

        TODO: Implement the nested loop convolution

        APPROACH:
        1. Initialize output array of shape (batch_size, out_channels, out_h, out_w)
        2. Loop over: batch, output channel, output row, output column
        3. For each output position, accumulate the dot product over:
           kernel height, kernel width, and input channels
        4. Store the accumulated sum at the output position

        LOOP STRUCTURE:
        for b in range(batch_size):
            for out_ch in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        conv_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    conv_sum += padded[b, in_ch, ...] * weight[out_ch, in_ch, ...]
                        output[b, out_ch, oh, ow] = conv_sum

        HINT: The input position for kernel element (k_h, k_w) at output
        position (oh, ow) with stride s is: (oh * s + k_h, ow * s + k_w).
        """
        ### BEGIN SOLUTION
        out_channels = self.out_channels
        in_channels = self.in_channels
        kernel_h, kernel_w = self.kernel_size

        output = np.zeros((batch_size, out_channels, out_h, out_w))

        for b in range(batch_size):
            for out_ch in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_h_start = oh * self.stride
                        in_w_start = ow * self.stride

                        conv_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    input_val = padded[b, in_ch,
                                                      in_h_start + k_h,
                                                      in_w_start + k_w]
                                    weight_val = self.weight.data[out_ch, in_ch, k_h, k_w]
                                    conv_sum += input_val * weight_val

                        output[b, out_ch, oh, ow] = conv_sum

        return output
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through Conv2d layer.

        TODO: Compose input validation, spatial dimension calculation, and Conv2dFunction.apply.

        APPROACH:
        1. Validate input is 4D (shared helper)
        2. Compute output spatial dimensions
        3. Pad input if needed
        4. Run the sliding window convolution loops
        5. Add bias

        Steps 3-5 run inside Conv2dFunction.apply, which records the operation for backward.

        Each step is a separate helper you implement below.
        See the individual helper docstrings for details.

        EXAMPLE:
        >>> conv = Conv2d(3, 16, kernel_size=3, padding=1)
        >>> x = Tensor(rng.standard_normal((2, 3, 32, 32)))  # batch=2, RGB, 32x32
        >>> out = conv(x)
        >>> print(out.shape)  # Should be (2, 16, 32, 32)
        """
        ### BEGIN SOLUTION
        # Step 1: Validate input
        validate_4d_input(x, "Conv2d")

        batch_size, in_channels, in_height, in_width = x.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Conv2d expected {self.in_channels} input channels, got {in_channels}")
        out_height, out_width = self._compute_output_shape(in_height, in_width)
        if out_height <= 0 or out_width <= 0:
            raise ValueError("Conv2d kernel must fit within the padded input")

        # Steps 3-5: pad, convolve, add bias. The operation runs the helpers
        # (via `layer=self`) and Module 06's apply() records it for backward.
        if self.bias is not None:
            return Conv2dFunction.apply(x, self.weight, self.bias, layer=self)
        return Conv2dFunction.apply(x, self.weight, layer=self)
        ### END SOLUTION

    def parameters(self) -> list[Tensor]:
        """Return trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, x: Tensor) -> Tensor:
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
r"""
### 🧪 Unit Test: Conv2d Output Shape Computation

This test validates that `_compute_output_shape` correctly applies the convolution output dimension formula:

$$\text{dim}_{\text{out}} = \left\lfloor \frac{\text{dim}_{\text{in}} + 2 \cdot \text{padding} - \text{kernel}}{\text{stride}} \right\rfloor + 1$$

- **Same padding** ($32 \times 32, K=3, P=1, S=1$): $(32 + 2 - 3)/1 + 1 = 32$
- **Valid padding** ($32 \times 32, K=3, P=0, S=1$): $(32 + 0 - 3)/1 + 1 = 30$
- **Downsampling** ($32 \times 32, K=3, P=0, S=2$): $(32 + 0 - 3)/2 + 1 = 15$

**What we're testing**: Output dimension formula for various configurations
**Why it matters**: Wrong dimensions cause silent shape bugs in CNNs
**Expected**: Matches hand-calculated values
"""

# %% nbgrader={"grade": true, "grade_id": "conv2d-output-shape", "locked": true, "points": 5}
def test_unit_conv2d_output_shape() -> None:
    """🧪 Test Conv2d._compute_output_shape for various configurations."""
    print("🧪 Unit Test: Conv2d Output Shape...")

    # Same padding: output == input
    conv_same = Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
    oh, ow = conv_same._compute_output_shape(32, 32)
    assert (oh, ow) == (32, 32), f"Same padding: expected (32, 32), got ({oh}, {ow})"

    # No padding: output shrinks by (kernel - 1)
    conv_no_pad = Conv2d(3, 16, kernel_size=3, padding=0, stride=1)
    oh, ow = conv_no_pad._compute_output_shape(32, 32)
    assert (oh, ow) == (30, 30), f"No padding: expected (30, 30), got ({oh}, {ow})"

    # Stride 2: output roughly halves
    conv_stride = Conv2d(3, 16, kernel_size=3, padding=0, stride=2)
    oh, ow = conv_stride._compute_output_shape(32, 32)
    assert (oh, ow) == (15, 15), f"Stride 2: expected (15, 15), got ({oh}, {ow})"

    # Non-square input
    conv_rect = Conv2d(1, 8, kernel_size=3, padding=1, stride=1)
    oh, ow = conv_rect._compute_output_shape(28, 14)
    assert (oh, ow) == (28, 14), f"Rectangular: expected (28, 14), got ({oh}, {ow})"

    # Larger kernel
    conv_5x5 = Conv2d(3, 16, kernel_size=5, padding=0, stride=1)
    oh, ow = conv_5x5._compute_output_shape(32, 32)
    assert (oh, ow) == (28, 28), f"5x5 kernel: expected (28, 28), got ({oh}, {ow})"

    print("✅ Conv2d output shape computation works correctly!")

if __name__ == "__main__":
    test_unit_conv2d_output_shape()

# %% [markdown]
r"""
### 🧪 Unit Test: Conv2d Padding

This test validates that `_apply_padding` correctly zero-pads the spatial dimensions while leaving batch and channel dimensions untouched:

$$\mathbf{X}_{(1, 1, 3, 3)} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \quad \xrightarrow{P=1} \quad \mathbf{X}_{\text{pad}(1, 1, 5, 5)} = \begin{bmatrix} 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 2 & 3 & 0 \\ 0 & 4 & 5 & 6 & 0 \\ 0 & 7 & 8 & 9 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$

**What we're testing**: Zero-padding adds correct borders to spatial dims
**Why it matters**: Padding controls whether convolution preserves spatial size
**Expected**: Padded array has correct shape and zero borders
"""

# %% nbgrader={"grade": true, "grade_id": "conv2d-padding", "locked": true, "points": 5}
def test_unit_conv2d_padding() -> None:
    """🧪 Test Conv2d._apply_padding for zero-padding behavior."""
    print("🧪 Unit Test: Conv2d Padding...")

    # No padding: input unchanged
    conv_no_pad = Conv2d(1, 1, kernel_size=3, padding=0)
    x = np.ones((1, 1, 4, 4))
    result = conv_no_pad._apply_padding(x)
    assert result.shape == (1, 1, 4, 4), f"No-pad shape: expected (1,1,4,4), got {result.shape}"
    assert np.array_equal(result, x), "No-pad should return input unchanged"

    # Padding=1: adds 1 pixel border of zeros
    conv_pad1 = Conv2d(1, 1, kernel_size=3, padding=1)
    x = np.ones((1, 1, 3, 3))
    result = conv_pad1._apply_padding(x)
    assert result.shape == (1, 1, 5, 5), f"Pad-1 shape: expected (1,1,5,5), got {result.shape}"
    # Check that borders are zero
    assert np.all(result[:, :, 0, :] == 0), "Top border should be zeros"
    assert np.all(result[:, :, -1, :] == 0), "Bottom border should be zeros"
    assert np.all(result[:, :, :, 0] == 0), "Left border should be zeros"
    assert np.all(result[:, :, :, -1] == 0), "Right border should be zeros"
    # Check that center is preserved
    assert np.all(result[:, :, 1:4, 1:4] == 1), "Center should be preserved"

    # Padding=2: adds 2 pixel border
    conv_pad2 = Conv2d(1, 1, kernel_size=5, padding=2)
    x = np.ones((2, 3, 4, 4))
    result = conv_pad2._apply_padding(x)
    assert result.shape == (2, 3, 8, 8), f"Pad-2 shape: expected (2,3,8,8), got {result.shape}"
    # Batch and channel dims unchanged
    assert result.shape[0] == 2, "Batch dim should be unchanged"
    assert result.shape[1] == 3, "Channel dim should be unchanged"

    print("✅ Conv2d padding works correctly!")

if __name__ == "__main__":
    test_unit_conv2d_padding()

# %% [markdown]
r"""
### 🧪 Unit Test: Conv2d Convolution Loops

This test validates the core sliding window computation in `_convolve_loops`:

$$\text{Output}[b, c_{\text{out}}, h, w] = \sum_{c_{\text{in}}} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} X_{\text{patch}}[b, c_{\text{in}}, m, n] \cdot W[c_{\text{out}}, c_{\text{in}}, m, n]$$

**What we're testing**: The 7-nested-loop convolution produces correct values
**Why it matters**: This is THE core operation of computer vision
**Expected**: Output matches hand-computed convolution results
"""

# %% nbgrader={"grade": true, "grade_id": "conv2d-convolve", "locked": true, "points": 15}
def test_unit_conv2d_convolve_loops() -> None:
    """🧪 Test Conv2d._convolve_loops with known input/weight values."""
    print("🧪 Unit Test: Conv2d Convolution Loops...")

    # Create a Conv2d with known weights (1 input channel, 1 output channel, 2x2 kernel)
    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    # Set weights to known values: [[1, 0], [0, 1]] (identity-like kernel)
    conv.weight = Tensor(np.array([[[[1.0, 0.0],
                                      [0.0, 1.0]]]]), requires_grad=True)

    # Input: 1 batch, 1 channel, 3x3
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    padded = np.array([[[[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]]]])

    # Output should be 2x2 (no padding):
    # pos(0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6
    # pos(0,1): 2*1 + 3*0 + 5*0 + 6*1 = 8
    # pos(1,0): 4*1 + 5*0 + 7*0 + 8*1 = 12
    # pos(1,1): 5*1 + 6*0 + 8*0 + 9*1 = 14
    output = conv._convolve_loops(padded, batch_size=1, out_h=2, out_w=2)

    expected = np.array([[[[6.0, 8.0],
                            [12.0, 14.0]]]])
    assert np.allclose(output, expected), f"Expected:\n{expected}\nGot:\n{output}"

    # Test with multiple output channels
    conv2 = Conv2d(in_channels=1, out_channels=2, kernel_size=2, bias=False)
    # Channel 0: all ones kernel, Channel 1: all twos kernel
    conv2.weight = Tensor(np.array([[[[1.0, 1.0], [1.0, 1.0]]],
                                     [[[2.0, 2.0], [2.0, 2.0]]]]), requires_grad=True)

    output2 = conv2._convolve_loops(padded, batch_size=1, out_h=2, out_w=2)

    # Channel 0 (all-ones kernel): sum of each 2x2 window
    # pos(0,0): 1+2+4+5=12, pos(0,1): 2+3+5+6=16
    # pos(1,0): 4+5+7+8=24, pos(1,1): 5+6+8+9=28
    expected_ch0 = np.array([[12.0, 16.0], [24.0, 28.0]])
    expected_ch1 = expected_ch0 * 2  # All-twos kernel = 2x all-ones
    assert np.allclose(output2[0, 0], expected_ch0), f"Channel 0 mismatch"
    assert np.allclose(output2[0, 1], expected_ch1), f"Channel 1 mismatch"

    print("✅ Conv2d convolution loops work correctly!")

if __name__ == "__main__":
    test_unit_conv2d_convolve_loops()

# %% [markdown]
"""
### 🧪 Unit Test: Conv2d Forward (Composition)

This test validates the complete `forward` method that composes all helpers
together: validation, shape computation, padding, convolution loops, bias,
and gradient tracking.

**What we're testing**: End-to-end Conv2d with shape preservation, padding, stride
**Why it matters**: Convolution is the foundation of computer vision
**Expected**: Correct output shapes and reasonable value ranges
"""

# %% nbgrader={"grade": true, "grade_id": "conv2d-forward", "locked": true, "points": 15}
def test_unit_conv2d() -> None:
    """🧪 Test Conv2d forward pass with multiple configurations."""
    print("🧪 Unit Test: Conv2d Forward...")

    # Test 1: Basic convolution without padding
    print("  Testing basic convolution...")
    conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    x1 = Tensor(rng.standard_normal((2, 3, 32, 32)))
    out1 = conv1(x1)

    expected_h = (32 - 3) + 1  # 30
    expected_w = (32 - 3) + 1  # 30
    assert out1.shape == (2, 16, expected_h, expected_w), f"Expected (2, 16, 30, 30), got {out1.shape}"

    # Test 2: Convolution with padding (same size)
    print("  Testing convolution with padding...")
    conv2 = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    x2 = Tensor(rng.standard_normal((1, 3, 28, 28)))
    out2 = conv2(x2)

    # With padding=1, output should be same size as input
    assert out2.shape == (1, 8, 28, 28), f"Expected (1, 8, 28, 28), got {out2.shape}"

    # Test 3: Convolution with stride
    print("  Testing convolution with stride...")
    conv3 = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2)
    x3 = Tensor(rng.standard_normal((1, 1, 16, 16)))
    out3 = conv3(x3)

    expected_h = (16 - 3) // 2 + 1  # 7
    expected_w = (16 - 3) // 2 + 1  # 7
    assert out3.shape == (1, 4, expected_h, expected_w), f"Expected (1, 4, 7, 7), got {out3.shape}"

    # Test 4: Parameter counting
    print("  Testing parameter counting...")
    conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=True)
    params = conv4.parameters()

    # Weight: (128, 64, 3, 3) = 73,728 parameters
    # Bias: (128,) = 128 parameters
    # Total: 73,856 parameters
    weight_params = 128 * 64 * 3 * 3
    bias_params = 128
    total_params = weight_params + bias_params

    actual_weight_params = np.prod(conv4.weight.shape)
    actual_bias_params = np.prod(conv4.bias.shape) if conv4.bias is not None else 0
    actual_total = actual_weight_params + actual_bias_params

    assert actual_total == total_params, f"Expected {total_params} parameters, got {actual_total}"
    assert len(params) == 2, f"Expected 2 parameter tensors, got {len(params)}"

    # Test 5: No bias configuration
    print("  Testing no bias configuration...")
    conv5 = Conv2d(in_channels=3, out_channels=16, kernel_size=5, bias=False)
    params5 = conv5.parameters()
    assert len(params5) == 1, f"Expected 1 parameter tensor (no bias), got {len(params5)}"
    assert conv5.bias is None, "Bias should be None when bias=False"

    print("✅ Conv2d works correctly!")

if __name__ == "__main__":
    test_unit_conv2d()

# %% [markdown]
r"""
## 🏗️ Pooling Operations: Spatial Dimension Reduction

Pooling operations compress spatial dimensions while preserving dominant feature activations. In MLSys architectures, pooling acts as a spatial summarization mechanism, cutting memory consumption and computational latency by integer factors.

### MaxPool2d vs AvgPool2d: Spatial Aggregation

Consider a $4 \times 4$ input feature map partitioned into four $2 \times 2$ non-overlapping windows (stride $S=2$):

$$\mathbf{X} = \begin{bmatrix} 1 & 3 & 2 & 8 \\ 5 & 6 & 7 & 4 \\ 2 & 9 & 1 & 7 \\ 0 & 1 & 3 & 6 \end{bmatrix}$$

- **Top-Left Window**: $\{1, 3, 5, 6\} \implies \max = 6, \quad \text{mean} = \frac{1+3+5+6}{4} = 3.75$
- **Top-Right Window**: $\{2, 8, 7, 4\} \implies \max = 8, \quad \text{mean} = \frac{2+8+7+4}{4} = 5.25$
- **Bottom-Left Window**: $\{2, 9, 0, 1\} \implies \max = 9, \quad \text{mean} = \frac{2+9+0+1}{4} = 3.00$
- **Bottom-Right Window**: $\{1, 7, 3, 6\} \implies \max = 7, \quad \text{mean} = \frac{1+7+3+6}{4} = 4.25$

$$\mathbf{Y}_{\text{max}} = \begin{bmatrix} 6 & 8 \\ 9 & 7 \end{bmatrix}, \qquad \mathbf{Y}_{\text{avg}} = \begin{bmatrix} 3.75 & 5.25 \\ 3.00 & 4.25 \end{bmatrix}$$

---

### Systems & Memory Footprint Analysis

| Metric | Input Activation | After $2 \times 2$ Pooling ($S=2$) | MLSys Benefit |
| :--- | :--- | :--- | :--- |
| **Spatial Grid** | $224 \times 224$ ($50{,}176$ positions) | $112 \times 112$ ($12{,}544$ positions) | Exact $4\times$ reduction in spatial grid |
| **Activation Buffer** | $1 \times 64 \times 224 \times 224 \times 4\text{ B} = 12.85\text{ MB}$ | $1 \times 64 \times 112 \times 112 \times 4\text{ B} = 3.21\text{ MB}$ | **$9.64\text{ MB}$ saved** per forward/backward pass |
| **Downstream FLOPs** | Evaluates $H \times W$ receptive fields | Evaluates $(H/2) \times (W/2)$ receptive fields | **$4\times$ reduction** in next layer GEMM operations |

---

### Architectural Trade-offs

| Advantage | Systems Rationale | Trade-off / Cost |
| :--- | :--- | :--- |
| **Translation Invariance** | Small pixel shifts do not change window maximum | Loses precise spatial localization |
| **Compute Reduction** | Halves spatial dimensions, cutting subsequent FLOPs by $4\times$ | Small objects (< pooling window) may be lost |
| **Zero Learnable Parameters** | Aggregates without adding weights or optimizer states | Backward still needs each window's argmax, so something has to store it or find it again |

#### Store or Recompute: Max Pooling's Hidden Choice

Max pooling has no parameters, but its backward pass still needs one fact per output element, namely which position in the window won. There are two ways to have it. PyTorch stores the indices during the forward pass, one `int64` per output element, which is $8$ bytes on top of the $4$-byte output itself, and its backward pass is then a single scatter. `MaxPool2dFunction.backward` in this module takes the other side of the trade and stores nothing at all, re-scanning every $K_h \times K_w$ window to find the maximum again. Same gradients, opposite bill. PyTorch triples what the layer holds ($4$ bytes of output plus $8$ bytes of index) to save a pass over the input, and TinyTorch spends that pass to hold nothing extra. Read the backward loop below and you will see the re-scan, not a lookup.
"""

# %% [markdown]
r"""
### MaxPool2d Implementation: Preserving Strong Features

`MaxPool2d` extracts the maximum activation in each window, preserving high-frequency features like edges, corners, and bright intensity peaks while discarding noise.

$$\text{Output}[b, c, h, w] = \max_{0 \le m < K_h, \, 0 \le n < K_w} X_{\text{pad}}[b, c, h \cdot S_h + m, w \cdot S_w + n]$$

During the backward pass, gradients route **exclusively** to the specific coordinate that achieved the maximum during the forward pass. All non-maximal positions receive zero gradient.
"""

# %% nbgrader={"grade": false, "grade_id": "maxpool2d-class", "solution": true}
#| export

class MaxPool2dFunction(Function):
    """
    Forward and backward passes for 2D max pooling.

    Max pooling gradients flow only to the positions that were selected
    as the maximum in the forward pass.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pool the (already validated) input using the layer's helpers, passed as `layer`."""
        batch_size, channels, in_height, in_width = x.shape
        out_height, out_width = self.layer._compute_pool_output_shape(in_height, in_width)
        if self.layer.padding > 0:
            padded_input = np.pad(x,
                                ((0, 0), (0, 0), (self.layer.padding, self.layer.padding), (self.layer.padding, self.layer.padding)),
                                mode='constant', constant_values=-np.inf)
        else:
            padded_input = x
        return self.layer._maxpool_loops(padded_input, batch_size, channels, out_height, out_width)

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray]:
        """
        Route gradients back to max positions.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient w.r.t. input
        """
        x, = self.inputs
        stride_h, stride_w = self.layer.stride
        padding, kernel_size = self.layer.padding, self.layer.kernel_size
        batch_size, channels, in_height, in_width = x.shape
        _, _, out_height, out_width = self.output.shape
        kernel_h, kernel_w = kernel_size

        # Apply padding if needed
        if padding > 0:
            padded_input = np.pad(x.data,
                                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                                mode='constant', constant_values=-np.inf)
            grad_input_padded = np.zeros_like(padded_input)
        else:
            padded_input = x.data
            grad_input_padded = np.zeros_like(x.data)

        # Route gradients to max positions
        for b in range(batch_size):
            for c in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        in_h_start = out_h * stride_h
                        in_w_start = out_w * stride_w

                        # Find max position in this window
                        max_val = -np.inf
                        max_h, max_w = None, None
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                in_h = in_h_start + k_h
                                in_w = in_w_start + k_w
                                # Padding is not an input candidate, even when
                                # real values also equal negative infinity.
                                if not (padding <= in_h < padding + in_height and
                                        padding <= in_w < padding + in_width):
                                    continue
                                val = padded_input[b, c, in_h, in_w]
                                if max_h is None or val > max_val:
                                    max_val = val
                                    max_h, max_w = in_h, in_w

                        # Route gradient to max position
                        if max_h is not None:
                            grad_input_padded[b, c, max_h, max_w] += grad_output[b, c, out_h, out_w]

        # Remove padding
        if padding > 0:
            grad_input = grad_input_padded[:, :,
                                          padding:-padding,
                                          padding:-padding]
        else:
            grad_input = grad_input_padded

        # Return as tuple (following Function protocol)
        return (grad_input,)

class MaxPool2d:
    """
    2D Max Pooling layer for spatial dimension reduction.

    Applies maximum operation over spatial windows, preserving
    the strongest activations while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Integer or (height, width) strides (default: kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] | None = None, padding: int = 0) -> None:
        """
        Initialize MaxPool2d layer.

        TODO: Store pooling parameters

        APPROACH:
        1. Convert kernel_size to tuple if needed
        2. Set stride to kernel_size if not provided (non-overlapping)
        3. Store padding parameter

        HINT: Default stride equals kernel_size for non-overlapping windows
        """
        ### BEGIN SOLUTION role="scaffold"
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Default stride equals kernel_size (non-overlapping)
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)

        self.padding = padding
        ### END SOLUTION

    def _compute_pool_output_shape(self, in_h: int, in_w: int) -> tuple[int, int]:
        """
        Calculate output spatial dimensions for pooling.

        TODO: Apply the pooling output size formula

        APPROACH:
        1. Apply the same formula as convolution:
           out_dim = (in_dim + 2 * padding - kernel_size) // stride + 1
        2. Return (out_height, out_width) as a tuple

        EXAMPLE:
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> oh, ow = pool._compute_pool_output_shape(8, 8)
        >>> print(oh, ow)  # 4, 4 (halved)

        HINT: This formula is identical to convolution's output shape formula.
        """
        ### BEGIN SOLUTION role="scaffold"
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_height = (in_h + 2 * self.padding - kernel_h) // stride_h + 1
        out_width = (in_w + 2 * self.padding - kernel_w) // stride_w + 1
        if out_height <= 0 or out_width <= 0:
            raise ValueError("Pooling kernel must fit within the padded input")
        return out_height, out_width
        ### END SOLUTION

    def _maxpool_loops(self, padded: np.ndarray, batch_size: int, channels: int,
                       out_h: int, out_w: int) -> np.ndarray:
        """
        The core max pooling: find maximum value in each window.

        TODO: Implement the nested loop max pooling

        APPROACH:
        1. Initialize output array of shape (batch_size, channels, out_h, out_w)
        2. Loop over: batch, channel, output row, output column
        3. For each output position, scan the kernel window to find the maximum
        4. Store the maximum at the output position

        LOOP STRUCTURE:
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        max_val = -infinity
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                max_val = max(max_val, padded[b, c, ...])
                        output[b, c, oh, ow] = max_val

        HINT: Initialize max_val to -np.inf so any real value is larger.
        The input position is (oh * stride_h + k_h, ow * stride_w + k_w).
        """
        ### BEGIN SOLUTION role="scaffold"
        kernel_h, kernel_w = self.kernel_size
        output = np.zeros((batch_size, channels, out_h, out_w))

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_h_start = oh * self.stride[0]
                        in_w_start = ow * self.stride[1]

                        max_val = -np.inf
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                input_val = padded[b, c,
                                                  in_h_start + k_h,
                                                  in_w_start + k_w]
                                max_val = max(max_val, input_val)

                        output[b, c, oh, ow] = max_val

        return output
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through MaxPool2d layer.

        This method composes the helpers:
        1. Validate input is 4D
        2. Compute output dimensions
        3. Pad input (with -inf for max pooling)
        4. Run the max pooling loops
        5. Attach gradient tracking

        EXAMPLE:
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> x = Tensor(rng.standard_normal((1, 3, 8, 8)))
        >>> out = pool(x)
        >>> print(out.shape)  # Should be (1, 3, 4, 4)
        """
        ### BEGIN SOLUTION role="scaffold"
        # Step 1: Validate input
        validate_4d_input(x, "MaxPool2d")

        batch_size, channels, in_height, in_width = x.shape

        # Step 2: Compute output dimensions
        out_height, out_width = self._compute_pool_output_shape(in_height, in_width)

        # Steps 3-5: pad and pool inside the operation (via `layer=self`);
        # Module 06's apply() records it for backward.
        return MaxPool2dFunction.apply(x, layer=self)
        ### END SOLUTION

    def parameters(self) -> list[Tensor]:
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x: Tensor) -> Tensor:
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
r"""
### 🧪 Unit Test: MaxPool2d Output Shape

This test validates that `_compute_pool_output_shape` correctly computes the spatial dimensions after max pooling:

$$\text{dim}_{\text{out}} = \left\lfloor \frac{\text{dim}_{\text{in}} + 2 \cdot \text{padding} - \text{kernel}}{\text{stride}} \right\rfloor + 1$$

Common case: $\text{kernel}=2, \text{stride}=2, \text{padding}=0$:
$$(8 + 0 - 2) // 2 + 1 = 4 \quad (\text{spatial dimensions halved})$$

**What we're testing**: Pooling output dimension calculation
**Why it matters**: Wrong dimensions break the CNN dimension chain
**Expected**: Matches hand-calculated values for various configs
"""

# %% nbgrader={"grade": true, "grade_id": "maxpool2d-output-shape", "locked": true, "points": 3}
def test_unit_maxpool2d_output_shape() -> None:
    """🧪 Test MaxPool2d._compute_pool_output_shape."""
    print("🧪 Unit Test: MaxPool2d Output Shape...")

    # Standard 2x2 pooling with stride 2: halves dimensions
    pool = MaxPool2d(kernel_size=2, stride=2)
    oh, ow = pool._compute_pool_output_shape(8, 8)
    assert (oh, ow) == (4, 4), f"2x2 stride 2: expected (4, 4), got ({oh}, {ow})"

    # Non-square input
    oh, ow = pool._compute_pool_output_shape(16, 8)
    assert (oh, ow) == (8, 4), f"Non-square: expected (8, 4), got ({oh}, {ow})"

    # Overlapping pooling: kernel=3, stride=1
    pool_overlap = MaxPool2d(kernel_size=3, stride=1)
    oh, ow = pool_overlap._compute_pool_output_shape(5, 5)
    assert (oh, ow) == (3, 3), f"Overlapping: expected (3, 3), got ({oh}, {ow})"

    # Large kernel
    pool_large = MaxPool2d(kernel_size=4, stride=4)
    oh, ow = pool_large._compute_pool_output_shape(16, 16)
    assert (oh, ow) == (4, 4), f"4x4 stride 4: expected (4, 4), got ({oh}, {ow})"

    print("✅ MaxPool2d output shape computation works correctly!")

if __name__ == "__main__":
    test_unit_maxpool2d_output_shape()

# %% [markdown]
r"""
### 🧪 Unit Test: MaxPool2d Loops

This test validates that `_maxpool_loops` correctly finds the maximum value in each pooling window:

$$\begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix} \quad \xrightarrow{\text{MaxPool } 2 \times 2, \, S=2} \quad \begin{bmatrix} 6 & 8 \\ 14 & 16 \end{bmatrix}$$

**What we're testing**: The max-finding loops produce correct values
**Why it matters**: Max pooling preserves the strongest activations
**Expected**: Output matches hand-computed max values per window
"""

# %% nbgrader={"grade": true, "grade_id": "maxpool2d-loops", "locked": true, "points": 7}
def test_unit_maxpool2d_loops() -> None:
    """🧪 Test MaxPool2d._maxpool_loops with known values."""
    print("🧪 Unit Test: MaxPool2d Loops...")

    pool = MaxPool2d(kernel_size=2, stride=2)

    # Known 4x4 input
    padded = np.array([[[[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0]]]])

    output = pool._maxpool_loops(padded, batch_size=1, channels=1, out_h=2, out_w=2)

    # Window maxes:
    # top-left: max(1,2,5,6) = 6
    # top-right: max(3,4,7,8) = 8
    # bottom-left: max(9,10,13,14) = 14
    # bottom-right: max(11,12,15,16) = 16
    expected = np.array([[[[6.0, 8.0], [14.0, 16.0]]]])
    assert np.allclose(output, expected), f"Expected:\n{expected}\nGot:\n{output}"

    # Test with negative values
    padded_neg = np.array([[[[-5.0, -1.0],
                              [-3.0, -2.0]]]])
    pool_small = MaxPool2d(kernel_size=2, stride=2)
    output_neg = pool_small._maxpool_loops(padded_neg, 1, 1, 1, 1)
    assert output_neg[0, 0, 0, 0] == -1.0, f"Max of negatives: expected -1.0, got {output_neg[0,0,0,0]}"

    print("✅ MaxPool2d loops work correctly!")

if __name__ == "__main__":
    test_unit_maxpool2d_loops()

# %% [markdown]
r"""
### AvgPool2d Implementation: Smoothing and Generalizing Features

`AvgPool2d` computes the arithmetic mean across each spatial window, producing smooth, spatially regularized feature maps less susceptible to localized outlier noise.

#### MaxPool vs AvgPool: Design Philosophies

| Dimension / Characteristic | `MaxPool2d` | `AvgPool2d` |
| :--- | :--- | :--- |
| **Feature Focus** | Detects dominant localized peaks (edges, spots) | Summarizes regional background energy |
| **Mathematical Formulation** | $\max_{(m,n) \in \Omega} x_{m,n}$ | $\frac{1}{\vert \Omega \vert} \sum_{(m,n) \in \Omega} x_{m,n}$ |
| **Sample Window Output** | **0.9** (preserves sharp peak: $\max\{0.1, 0.9, 0.3, 0.3\}$) | **0.4** (smooths regional context: $\text{mean}\{0.1, 0.9, 0.3, 0.3\}$) |
| **Backward Pass Distribution** | 100% gradient routed to the single argmax pixel | Equal gradient $\frac{1}{\vert \Omega \vert}$ distributed to all pixels |

---

#### When to Use Average Pooling

| Architectural Paradigm | MLSys & Training Rationale |
| :--- | :--- |
| **Global Average Pooling (GAP)** | Compresses $(C, H, W) \to (C, 1, 1)$, replacing dense linear layers and eliminating millions of weights |
| **Texture & Background Summarization** | Smooths out high-frequency spatial jitter in shallow networks |
| **Invariance to Precise Localization** | Retains cumulative energy across the receptive field without favoring extreme outliers |
"""

# %% nbgrader={"grade": false, "grade_id": "avgpool2d-class", "solution": true}
#| export

class AvgPool2dFunction(Function):
    """
    Forward and backward passes for 2D average pooling.

    Each output is the mean of the kernel_h*kernel_w inputs in its window, so
    the gradient is distributed equally (1/kernel_area) to every input position
    that contributed, accumulating where windows overlap.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pool the (already validated) input using the layer's helpers, passed as `layer`."""
        batch_size, channels, in_height, in_width = x.shape
        out_height, out_width = self.layer._compute_pool_output_shape(in_height, in_width)
        if self.layer.padding > 0:
            padded_input = np.pad(x,
                                ((0, 0), (0, 0), (self.layer.padding, self.layer.padding), (self.layer.padding, self.layer.padding)),
                                mode='constant', constant_values=0)
        else:
            padded_input = x
        return self.layer._avgpool_loops(padded_input, batch_size, channels, out_height, out_width)

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray]:
        """
        Distribute each output gradient equally across its pooling window.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient w.r.t. input
        """
        x, = self.inputs
        stride_h, stride_w = self.layer.stride
        padding, kernel_size = self.layer.padding, self.layer.kernel_size
        batch_size, channels, in_height, in_width = x.shape
        _, _, out_height, out_width = self.output.shape
        kernel_h, kernel_w = kernel_size
        kernel_area = kernel_h * kernel_w

        # Average pooling pads with zeros, so the gradient buffer is padded with
        # zeros too (matching the forward pass).
        if padding > 0:
            grad_input_padded = np.zeros(
                (batch_size, channels,
                 in_height + 2 * padding,
                 in_width + 2 * padding)
            )
        else:
            grad_input_padded = np.zeros_like(x.data)

        # Spread each output gradient equally over its window, accumulating overlaps.
        for b in range(batch_size):
            for c in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        in_h_start = out_h * stride_h
                        in_w_start = out_w * stride_w
                        share = grad_output[b, c, out_h, out_w] / kernel_area
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                grad_input_padded[b, c, in_h_start + k_h, in_w_start + k_w] += share

        # Remove padding
        if padding > 0:
            grad_input = grad_input_padded[:, :,
                                          padding:-padding,
                                          padding:-padding]
        else:
            grad_input = grad_input_padded

        # Return as tuple (following Function protocol)
        return (grad_input,)

class AvgPool2d:
    """
    2D Average Pooling layer for spatial dimension reduction.

    Applies average operation over spatial windows, smoothing
    features while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Integer or (height, width) strides (default: kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] | None = None, padding: int = 0) -> None:
        """
        Initialize AvgPool2d layer.

        TODO: Store pooling parameters (same as MaxPool2d)

        APPROACH:
        1. Convert kernel_size to tuple if needed
        2. Set stride to kernel_size if not provided
        3. Store padding parameter
        """
        ### BEGIN SOLUTION role="scaffold"
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Default stride equals kernel_size (non-overlapping)
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)

        self.padding = padding
        ### END SOLUTION

    def _compute_pool_output_shape(self, in_h: int, in_w: int) -> tuple[int, int]:
        """
        Calculate output spatial dimensions for pooling.

        TODO: Apply the pooling output size formula

        APPROACH:
        1. Apply the standard formula for each spatial dimension:
           out_dim = (in_dim + 2 * padding - kernel_size) // stride + 1
        2. Return (out_height, out_width) as a tuple

        EXAMPLE:
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> oh, ow = pool._compute_pool_output_shape(8, 8)
        >>> print(oh, ow)  # 4, 4 (halved)

        HINT: This formula is identical to MaxPool2d and Conv2d output shapes.
        """
        ### BEGIN SOLUTION role="scaffold"
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_height = (in_h + 2 * self.padding - kernel_h) // stride_h + 1
        out_width = (in_w + 2 * self.padding - kernel_w) // stride_w + 1
        if out_height <= 0 or out_width <= 0:
            raise ValueError("Pooling kernel must fit within the padded input")
        return out_height, out_width
        ### END SOLUTION

    def _avgpool_loops(self, padded: np.ndarray, batch_size: int, channels: int,
                       out_h: int, out_w: int) -> np.ndarray:
        """
        The core average pooling: compute mean of each window.

        TODO: Implement the nested loop average pooling

        APPROACH:
        1. Initialize output array of shape (batch_size, channels, out_h, out_w)
        2. Loop over: batch, channel, output row, output column
        3. For each output position, sum all values in the kernel window
        4. Divide by window area (kernel_h * kernel_w) to get the average
        5. Store the average at the output position

        LOOP STRUCTURE:
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        window_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                window_sum += padded[b, c, ...]
                        output[b, c, oh, ow] = window_sum / (kernel_h * kernel_w)

        HINT: Unlike max pooling, you accumulate a sum and then divide.
        The input position is (oh * stride_h + k_h, ow * stride_w + k_w).
        """
        ### BEGIN SOLUTION role="scaffold"
        kernel_h, kernel_w = self.kernel_size
        output = np.zeros((batch_size, channels, out_h, out_w))

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_h_start = oh * self.stride[0]
                        in_w_start = ow * self.stride[1]

                        window_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                input_val = padded[b, c,
                                                  in_h_start + k_h,
                                                  in_w_start + k_w]
                                window_sum += input_val

                        output[b, c, oh, ow] = window_sum / (kernel_h * kernel_w)

        return output
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through AvgPool2d layer.

        This method composes the helpers:
        1. Validate input is 4D
        2. Compute output dimensions
        3. Pad input (with zeros for average pooling)
        4. Run the average pooling loops
        5. Return result with gradient tracking

        EXAMPLE:
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> x = Tensor(rng.standard_normal((1, 3, 8, 8)))
        >>> out = pool(x)
        >>> print(out.shape)  # Should be (1, 3, 4, 4)
        """
        ### BEGIN SOLUTION role="scaffold"
        # Step 1: Validate input
        validate_4d_input(x, "AvgPool2d")

        batch_size, channels, in_height, in_width = x.shape

        # Step 2: Compute output dimensions
        out_height, out_width = self._compute_pool_output_shape(in_height, in_width)

        # Steps 3-5: pad and pool inside the operation (via `layer=self`);
        # Module 06's apply() records it for backward.
        return AvgPool2dFunction.apply(x, layer=self)
        ### END SOLUTION

    def parameters(self) -> list[Tensor]:
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x: Tensor) -> Tensor:
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: AvgPool2d Output Shape

This test validates that `_compute_pool_output_shape` correctly computes
the spatial dimensions after average pooling.

**What we're testing**: Pooling output dimension calculation
**Why it matters**: Must match MaxPool2d formula for interchangeability
**Expected**: Same results as MaxPool2d for identical configurations
"""

# %% nbgrader={"grade": true, "grade_id": "avgpool2d-output-shape", "locked": true, "points": 3}
def test_unit_avgpool2d_output_shape() -> None:
    """🧪 Test AvgPool2d._compute_pool_output_shape."""
    print("🧪 Unit Test: AvgPool2d Output Shape...")

    # Standard 2x2 pooling: halves dimensions
    pool = AvgPool2d(kernel_size=2, stride=2)
    oh, ow = pool._compute_pool_output_shape(8, 8)
    assert (oh, ow) == (4, 4), f"2x2 stride 2: expected (4, 4), got ({oh}, {ow})"

    # Non-square input
    oh, ow = pool._compute_pool_output_shape(16, 8)
    assert (oh, ow) == (8, 4), f"Non-square: expected (8, 4), got ({oh}, {ow})"

    # Overlapping pooling: kernel=3, stride=1
    pool_overlap = AvgPool2d(kernel_size=3, stride=1)
    oh, ow = pool_overlap._compute_pool_output_shape(5, 5)
    assert (oh, ow) == (3, 3), f"Overlapping: expected (3, 3), got ({oh}, {ow})"

    print("✅ AvgPool2d output shape computation works correctly!")

if __name__ == "__main__":
    test_unit_avgpool2d_output_shape()

# %% [markdown]
r"""
### 🧪 Unit Test: AvgPool2d Loops

This test validates that `_avgpool_loops` correctly computes the arithmetic mean of each pooling window:

$$\begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix} \quad \xrightarrow{\text{AvgPool } 2 \times 2, \, S=2} \quad \begin{bmatrix} 3.5 & 5.5 \\ 11.5 & 13.5 \end{bmatrix}$$

- Top-left: $\frac{1+2+5+6}{4} = 3.5$
- Top-right: $\frac{3+4+7+8}{4} = 5.5$
- Bottom-left: $\frac{9+10+13+14}{4} = 11.5$
- Bottom-right: $\frac{11+12+15+16}{4} = 13.5$

**What we're testing**: The sum-and-divide loops produce correct averages
**Why it matters**: Average pooling creates smoother features than max pooling
**Expected**: Output matches hand-computed averages per window
"""

# %% nbgrader={"grade": true, "grade_id": "avgpool2d-loops", "locked": true, "points": 7}
def test_unit_avgpool2d_loops() -> None:
    """🧪 Test AvgPool2d._avgpool_loops with known values."""
    print("🧪 Unit Test: AvgPool2d Loops...")

    pool = AvgPool2d(kernel_size=2, stride=2)

    # Known 4x4 input
    padded = np.array([[[[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0]]]])

    output = pool._avgpool_loops(padded, batch_size=1, channels=1, out_h=2, out_w=2)

    # Window averages:
    # top-left: (1+2+5+6)/4 = 3.5
    # top-right: (3+4+7+8)/4 = 5.5
    # bottom-left: (9+10+13+14)/4 = 11.5
    # bottom-right: (11+12+15+16)/4 = 13.5
    expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])
    assert np.allclose(output, expected), f"Expected:\n{expected}\nGot:\n{output}"

    # Test that avg is always <= max for same data
    pool_max = MaxPool2d(kernel_size=2, stride=2)
    max_output = pool_max._maxpool_loops(padded, 1, 1, 2, 2)
    assert np.all(output <= max_output), "Average should always be <= maximum"

    print("✅ AvgPool2d loops work correctly!")

if __name__ == "__main__":
    test_unit_avgpool2d_loops()

# %% [markdown]
r"""
## 🏗️ Batch Normalization: Stabilizing Deep Network Training

Batch Normalization (`BatchNorm2d`) stabilizes the optimization landscape of deep convolutional networks. By standardizing activations to zero mean and unit variance across the mini-batch and spatial dimensions, it smooths the optimization landscape and allows substantially higher learning rates. The original paper credited reduced "internal covariate shift" for that gain, but Santurkar et al. (2018) showed experimentally that the reduction does not account for it, so read the phrase as the historical explanation rather than the mechanism.

### The BatchNorm2d Formulation

For each feature channel $c \in \{0, \dots, C-1\}$, across a mini-batch $\mathcal{B}$ of size $N = B \times H \times W$:

1. **Mini-Batch Statistics** (evaluated during training):
   $$\mu_c = \frac{1}{N} \sum_{b=1}^B \sum_{h=1}^H \sum_{w=1}^W x_{b, c, h, w}, \qquad \sigma_c^2 = \frac{1}{N} \sum_{b=1}^B \sum_{h=1}^H \sum_{w=1}^W (x_{b, c, h, w} - \mu_c)^2$$

2. **Standardization**:
   $$\widehat{x}_{b, c, h, w} = \frac{x_{b, c, h, w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

3. **Affine Transformation** (learnable scale $\gamma_c$ and shift $\beta_c$):
   $$y_{b, c, h, w} = \gamma_c \widehat{x}_{b, c, h, w} + \beta_c$$

4. **Running Statistics Tracking** (exponential moving average):
   $$\mu_{\text{run}} \leftarrow (1 - \alpha) \mu_{\text{run}} + \alpha \mu_c, \qquad \sigma_{\text{run}}^2 \leftarrow (1 - \alpha) \sigma_{\text{run}}^2 + \alpha \sigma_c^2$$

---

### Train vs Eval Mode: A Critical Systems Distinction

| Operational Attribute | Training Mode (`model.train()`) | Evaluation Mode (`model.eval()`) |
| :--- | :--- | :--- |
| **Statistics Used** | Current mini-batch statistics $\mu_c, \sigma_c^2$ | Accumulated running statistics $\mu_{\text{run}}, \sigma_{\text{run}}^2$ |
| **Running Stats Update** | Updated dynamically via exponential moving average | Frozen, strictly read-only |
| **Sample Coupling** | Predictions depend on other samples in the batch | Each sample is normalized independently |
| **Single-Sample ($B=1$)** | Statistics describe one image rather than the data distribution, so they swing sample to sample | Fully deterministic and mathematically stable |
| **Backward Pass Path** | 3 coupled gradient paths through $\mu$ and $\sigma^2$ | 1 direct path: $\frac{\partial \mathcal{L}}{\partial x} = \frac{\gamma}{\sigma_{\text{run}}} \frac{\partial \mathcal{L}}{\partial y}$ |

The $B=1$ row is usually justified by saying the variance across the batch is zero. That holds for `BatchNorm1d` on vectors, which have no spatial extent to reduce over. It does not hold here. This layer reduces over axes $(0, 2, 3)$, so at $B=1$ a single $8 \times 8$ feature map still supplies $N = 1 \cdot 8 \cdot 8 = 64$ samples per channel, the per-channel variance is perfectly finite, and the output comes out with unit standard deviation exactly as it should. The real problem is different and no less fatal. Those statistics characterize one image rather than the data distribution, so a bright image and a dark one get divided by different constants and the network sees neither the way it was trained to see it. Eval mode fixes that by normalizing every sample with the same frozen running statistics, which is why single-sample inference always runs in eval mode.
"""

# %% [markdown]
r"""
### The Backward Pass: Three Coupled Gradient Routes

In training mode, the statistics $\mu_c$ and $\sigma_c^2$ are functions of every input element $x_i$ in channel $c$. Consequently, perturbing an input pixel affects the output through three distinct pathways:

1. **Direct Path**: Perturbation propagating directly through standardized $\widehat{x}_i$.
2. **Mean Path**: Perturbation shifting the batch mean $\mu_c$, which shifts all normalized activations.
3. **Variance Path**: Perturbation altering the sample variance $\sigma_c^2$, which rescales all activations.

Summing all three partial derivatives yields the closed-form gradient w.r.t. input $x_i$:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{N \sqrt{\sigma_c^2 + \epsilon}} \left[ N \frac{\partial \mathcal{L}}{\partial \widehat{x}_i} - \sum_{j=1}^N \frac{\partial \mathcal{L}}{\partial \widehat{x}_j} - \widehat{x}_i \sum_{j=1}^N \left( \frac{\partial \mathcal{L}}{\partial \widehat{x}_j} \cdot \widehat{x}_j \right) \right]$$

In **eval mode**, the running statistics are frozen constants. The mean and variance paths contribute zero gradient, and the expression collapses to:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma_{\text{run}}^2 + \epsilon}} \cdot \frac{\partial \mathcal{L}}{\partial y_i}$$
"""

# %% nbgrader={"grade": false, "grade_id": "batchnorm2d-backward", "solution": false}
#| export
class BatchNorm2dFunction(Function):
    """
    The BatchNorm2d operation: normalize with the given statistics, then scale and shift.

    Computes gradients for x, gamma, and beta in one pass.
    output = gamma * ((x - mean) / sqrt(var + eps)) + beta

    In training mode the batch statistics depend on x, so the gradient for x
    carries three terms. In eval mode the statistics are frozen constants and
    only the direct term survives.
    """

    def forward(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Normalize with self.mean / self.var (chosen by the layer), then scale and shift."""
        channels = x.shape[1]
        mean_reshaped = np.asarray(self.mean).reshape(1, channels, 1, 1)
        var_reshaped = np.asarray(self.var).reshape(1, channels, 1, 1)
        # Keep 1/std: the forward needs it, and so does every term of the backward
        self.inv_std = 1.0 / np.sqrt(var_reshaped + self.eps)
        self.normalized_data = (np.asarray(x) - mean_reshaped) * self.inv_std
        gamma_reshaped = np.asarray(gamma).reshape(1, channels, 1, 1)
        beta_reshaped = np.asarray(beta).reshape(1, channels, 1, 1)
        return gamma_reshaped * self.normalized_data + beta_reshaped


    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Compute gradients for BatchNorm2d (x, gamma, beta)."""
        x, gamma, beta = self.inputs

        grad_x = grad_gamma = grad_beta = None
        normalized = self.normalized_data
        reduce_axes = (0, 2, 3)   # everything except the channel axis

        # Gradient for beta: it was added to every position, so sum them all back
        if isinstance(beta, Tensor) and beta.requires_grad:
            grad_beta = grad_output.sum(axis=reduce_axes)

        # Gradient for gamma: it scaled the normalized values
        if isinstance(gamma, Tensor) and gamma.requires_grad:
            grad_gamma = (grad_output * normalized).sum(axis=reduce_axes)

        # Gradient for x
        if isinstance(x, Tensor) and x.requires_grad:
            gamma_data = gamma.data if isinstance(gamma, Tensor) else gamma
            channels = normalized.shape[1]
            grad_norm = grad_output * np.asarray(gamma_data).reshape(1, channels, 1, 1)

            if self.training:
                # Batch statistics depend on x, so all three paths contribute.
                n = normalized.shape[0] * normalized.shape[2] * normalized.shape[3]
                sum_grad = grad_norm.sum(axis=reduce_axes, keepdims=True)
                sum_grad_norm = (grad_norm * normalized).sum(axis=reduce_axes, keepdims=True)
                grad_x = (self.inv_std / n) * (
                    n * grad_norm - sum_grad - normalized * sum_grad_norm
                )
            else:
                # Frozen statistics are constants: only the direct path survives.
                grad_x = grad_norm * self.inv_std

        return (grad_x, grad_gamma, grad_beta)

# %% [markdown]
r"""
### BatchNorm2d: The Layer Architecture

The `BatchNorm2d` layer manages two learnable parameter tensors (`gamma`, `beta`) and maintains two stateful running buffers (`running_mean`, `running_var`).

- **Training**: Computes mini-batch statistics across axes $(0, 2, 3)$, standardizes, and updates running statistics via EMA ($\text{momentum} = 0.1$).
- **Inference (`eval`)**: Freezes all running statistics, standardizing inputs deterministically with zero batch coupling.

This layer deviates from PyTorch in one place worth knowing about. `running_var` here accumulates the *biased* batch variance, $\frac{1}{N} \sum (x - \mu)^2$, which is what `np.var` returns, while `torch.nn.BatchNorm2d` accumulates the *unbiased* estimate with $\frac{1}{N-1}$. The two differ by a factor of $\frac{N}{N-1}$, invisible at the $N = B \cdot H \cdot W$ in the thousands that a real feature map gives you, and worth a few percent at toy sizes. Load a PyTorch checkpoint into this layer and the eval-mode outputs drift by $\sqrt{\frac{N}{N-1}}$, since the variance enters through a square root.
"""

# %% nbgrader={"grade": false, "grade_id": "batchnorm2d-class", "solution": true}
#| export

class BatchNorm2d:
    """
    Batch Normalization for 2D spatial inputs (images).

    Normalizes activations across batch and spatial dimensions for each channel,
    then applies learnable scale (gamma) and shift (beta) parameters.

    Key behaviors:
    - Training: Uses batch statistics, updates running statistics
    - Eval: Uses frozen running statistics for consistent inference

    Args:
        num_features: Number of channels (C in NCHW format)
        eps: Small constant for numerical stability (default: 1e-5)
        momentum: Momentum for running statistics update (default: 0.1)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        """
        Initialize BatchNorm2d layer.

        TODO: Initialize learnable and running parameters

        APPROACH:
        1. Store hyperparameters (num_features, eps, momentum)
        2. Initialize gamma (scale) to ones - identity at start
        3. Initialize beta (shift) to zeros - no shift at start
        4. Initialize running_mean to zeros
        5. Initialize running_var to ones
        6. Set training mode to True initially

        EXAMPLE:
        >>> bn = BatchNorm2d(64)  # For 64-channel feature maps
        >>> print(bn.gamma.shape)  # (64,)
        >>> print(bn.training)     # True
        """
        ### BEGIN SOLUTION role="scaffold"
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (requires_grad=True for training)
        # gamma (scale): initialized to 1 so output = normalized input initially
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        # beta (shift): initialized to 0 so no shift initially
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Running statistics (not trained, accumulated during training)
        # These are used during evaluation for consistent normalization
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Training mode flag
        self.training = True
        ### END SOLUTION

    def train(self) -> "BatchNorm2d":
        """Set layer to training mode."""
        self.training = True
        return self

    def eval(self) -> "BatchNorm2d":
        """Set layer to evaluation mode."""
        self.training = False
        return self

    def _validate_input(self, x: Tensor) -> None:
        """
        Validate that input tensor has the correct shape for BatchNorm2d.

        TODO: Validate input is 4D with correct channel count

        APPROACH:
        1. Check input is 4D (batch, channels, height, width)
        2. Provide helpful error messages for common mistakes (3D, 2D)
        3. Verify channel dimension matches num_features

        HINTS:
        - Use len(x.shape) to check dimensionality
        - Use ❌ What → 💡 Why → 🔧 Fix error message format
        """
        ### BEGIN SOLUTION role="scaffold"
        if len(x.shape) != 4:
            if len(x.shape) == 3:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch, channels, height, width), got 3D: {x.shape}\n"
                    f"  ❌ Missing batch dimension\n"
                    f"  💡 BatchNorm2d computes statistics over the batch dimension\n"
                    f"  🔧 Add batch dim: x.reshape(1, {x.shape[0]}, {x.shape[1]}, {x.shape[2]})"
                )
            elif len(x.shape) == 2:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch, channels, height, width), got 2D: {x.shape}\n"
                    f"  ❌ Got a matrix, expected an image tensor\n"
                    f"  💡 BatchNorm2d normalizes over spatial dimensions per channel\n"
                    f"  🔧 If this is a flattened image, reshape it: x.reshape(1, channels, height, width)"
                )
            else:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch, channels, height, width), got {len(x.shape)}D: {x.shape}\n"
                    f"  ❌ Wrong number of dimensions\n"
                    f"  💡 BatchNorm2d expects: (batch_size, channels, height, width)\n"
                    f"  🔧 Reshape your input to 4D with the correct dimensions"
                )

        batch_size, channels, height, width = x.shape

        if channels != self.num_features:
            raise ValueError(
                f"BatchNorm2d channel mismatch: expected {self.num_features} channels, got {channels}\n"
                f"  ❌ Input has {channels} channels but BatchNorm2d was created for {self.num_features}\n"
                f"  💡 BatchNorm2d(num_features) must match the channel dimension of your input\n"
                f"  🔧 Either fix your input shape or create BatchNorm2d({channels})"
            )
        ### END SOLUTION

    def _get_stats(self, x: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Get mean and variance for normalization (batch or running stats).

        TODO: Compute or retrieve normalization statistics based on mode

        APPROACH:
        1. If training: compute batch mean/var over axes (0, 2, 3),
           then update running statistics with momentum
        2. If eval: use frozen running_mean and running_var

        HINTS:
        - np.mean(x.data, axis=(0, 2, 3)) gives per-channel mean
        - Running update: running = (1 - momentum) * running + momentum * batch
        """
        ### BEGIN SOLUTION role="scaffold"
        if self.training:
            # Compute batch statistics per channel
            # Mean over batch and spatial dimensions: axes (0, 2, 3)
            batch_mean = np.mean(x.data, axis=(0, 2, 3))  # Shape: (C,)
            batch_var = np.var(x.data, axis=(0, 2, 3))    # Shape: (C,)

            # Update running statistics (exponential moving average)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            return batch_mean, batch_var
        else:
            # Use running statistics (frozen during eval)
            return self.running_mean, self.running_var
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through BatchNorm2d.

        TODO: Compose _validate_input, _get_stats, and normalize+scale

        APPROACH:
        1. Validate input with self._validate_input(x)
        2. Get mean/var with self._get_stats(x)
        3. Normalize: (x - mean) / sqrt(var + eps) with proper broadcasting
        4. Scale by gamma and shift by beta

        EXAMPLE:
        >>> bn = BatchNorm2d(16)
        >>> x = Tensor(rng.standard_normal((2, 16, 8, 8)))
        >>> y = bn(x)
        >>> print(y.shape)  # (2, 16, 8, 8)

        HINTS:
        - Reshape mean/var/gamma/beta to (1, C, 1, 1) for broadcasting
        - Run the normalization through BatchNorm2dFunction.apply, like Conv2d
          runs through Conv2dFunction.apply. A Tensor built by hand would
          advertise no gradients, the optimizer would update nothing, and the
          network would silently not learn
        """
        ### BEGIN SOLUTION role="scaffold"
        self._validate_input(x)

        batch_size, channels, height, width = x.shape
        mean, var = self._get_stats(x)
        # Normalize, scale, and shift inside the operation; Module 06's apply()
        # records it so gamma and beta actually train.
        return BatchNorm2dFunction.apply(x, self.gamma, self.beta,
                                         mean=mean, var=var, eps=self.eps, training=self.training)
        ### END SOLUTION

    def parameters(self) -> list[Tensor]:
        """Return learnable parameters (gamma and beta)."""
        return [self.gamma, self.beta]

    def __call__(self, x: Tensor) -> Tensor:
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: BatchNorm2d._validate_input

**What we're testing**: Input shape validation catches common mistakes
**Why it matters**: Clear errors save hours of debugging wrong tensor shapes
**Expected**: Accepts valid 4D input, rejects 2D/3D/wrong channels with helpful messages
"""

# %% nbgrader={"grade": true, "grade_id": "test-batchnorm2d-validate", "locked": true, "points": 3}
def test_unit_batchnorm2d_validate_input() -> None:
    """🧪 Test BatchNorm2d._validate_input implementation."""
    print("🧪 Unit Test: BatchNorm2d._validate_input...")

    bn = BatchNorm2d(num_features=16)

    # Valid 4D input should not raise
    x_valid = Tensor(rng.standard_normal((2, 16, 8, 8)))
    bn._validate_input(x_valid)  # Should pass silently

    # 3D input should raise
    try:
        bn._validate_input(Tensor(rng.standard_normal((16, 8, 8))))
        assert False, "Should have raised ValueError for 3D input"
    except ValueError as e:
        assert "3D" in str(e), f"Error should mention 3D, got: {e}"

    # Wrong channel count should raise
    try:
        bn._validate_input(Tensor(rng.standard_normal((2, 8, 4, 4))))
        assert False, "Should have raised ValueError for wrong channels"
    except ValueError as e:
        assert "mismatch" in str(e), f"Error should mention mismatch, got: {e}"

    print("✅ BatchNorm2d._validate_input works correctly!")

if __name__ == "__main__":
    test_unit_batchnorm2d_validate_input()

# %% [markdown]
"""
### 🧪 Unit Test: BatchNorm2d._get_stats

**What we're testing**: Statistics computation in training vs eval mode
**Why it matters**: Wrong statistics = wrong normalization = broken model
**Expected**: Training mode computes batch stats and updates running stats; eval mode uses frozen stats
"""

# %% nbgrader={"grade": true, "grade_id": "test-batchnorm2d-get-stats", "locked": true, "points": 3}
def test_unit_batchnorm2d_get_stats() -> None:
    """🧪 Test BatchNorm2d._get_stats implementation."""
    print("🧪 Unit Test: BatchNorm2d._get_stats...")

    bn = BatchNorm2d(num_features=4)
    x = Tensor(rng.standard_normal((8, 4, 6, 6)))

    # Training mode: should return batch stats and update running stats
    bn.train()
    running_mean_before = bn.running_mean.copy()
    mean, var = bn._get_stats(x)

    assert mean.shape == (4,), f"Expected per-channel mean shape (4,), got {mean.shape}"
    assert var.shape == (4,), f"Expected per-channel var shape (4,), got {var.shape}"
    assert not np.allclose(bn.running_mean, running_mean_before), \
        "Running mean should be updated in training mode"

    # Eval mode: should return running stats (frozen)
    bn.eval()
    running_mean_snapshot = bn.running_mean.copy()
    mean_eval, var_eval = bn._get_stats(x)

    assert np.allclose(mean_eval, running_mean_snapshot), \
        "Eval mode should return running mean"
    assert np.allclose(bn.running_mean, running_mean_snapshot), \
        "Running mean should not change in eval mode"

    print("✅ BatchNorm2d._get_stats works correctly!")

if __name__ == "__main__":
    test_unit_batchnorm2d_get_stats()

# %% [markdown]
"""
### 🧪 Unit Test: BatchNorm2d

This test validates batch normalization implementation.

**What we're testing**: Normalization behavior, train/eval mode, running statistics
**Why it matters**: BatchNorm is essential for training deep CNNs effectively
**Expected**: Normalized outputs with proper mean/variance characteristics
"""

# %% nbgrader={"grade": true, "grade_id": "test-batchnorm2d", "locked": true, "points": 10}
def test_unit_batchnorm2d() -> None:
    """🧪 Test BatchNorm2d implementation."""
    print("🧪 Unit Test: BatchNorm2d...")

    # Test 1: Basic forward pass shape
    print("  Testing basic forward pass...")
    bn = BatchNorm2d(num_features=16)
    x = Tensor(rng.standard_normal((4, 16, 8, 8)))  # batch=4, channels=16, 8x8
    y = bn(x)

    assert y.shape == x.shape, f"Output shape should match input, got {y.shape}"

    # Test 2: Training mode normalization
    print("  Testing training mode normalization...")
    bn2 = BatchNorm2d(num_features=8)
    bn2.train()  # Ensure training mode

    # Create input with known statistics per channel
    x2 = Tensor(rng.standard_normal((32, 8, 4, 4)) * 10 + 5)  # Mean~5, std~10
    y2 = bn2(x2)

    # After normalization, each channel should have mean≈0, std≈1
    # (before gamma/beta are applied, since gamma=1, beta=0)
    for c in range(8):
        channel_mean = np.mean(y2.data[:, c, :, :])
        channel_std = np.std(y2.data[:, c, :, :])
        assert abs(channel_mean) < 0.1, f"Channel {c} mean should be ~0, got {channel_mean:.3f}"
        assert abs(channel_std - 1.0) < 0.1, f"Channel {c} std should be ~1, got {channel_std:.3f}"

    # Test 3: Running statistics update
    print("  Testing running statistics update...")
    initial_running_mean = bn2.running_mean.copy()

    # Forward pass updates running stats
    x3 = Tensor(rng.standard_normal((16, 8, 4, 4)) + 3)  # Offset mean
    _ = bn2(x3)

    # Running mean should have moved toward batch mean
    assert not np.allclose(bn2.running_mean, initial_running_mean), \
        "Running mean should update during training"

    # Test 4: Eval mode uses running statistics
    print("  Testing eval mode behavior...")
    bn3 = BatchNorm2d(num_features=4)

    # Train on some data to establish running stats
    for _ in range(10):
        x_train = Tensor(rng.standard_normal((8, 4, 4, 4)) * 2 + 1)
        _ = bn3(x_train)

    saved_running_mean = bn3.running_mean.copy()
    saved_running_var = bn3.running_var.copy()

    # Switch to eval mode
    bn3.eval()

    # Process different data - running stats should NOT change
    x_eval = Tensor(rng.standard_normal((2, 4, 4, 4)) * 5)  # Different distribution
    _ = bn3(x_eval)

    assert np.allclose(bn3.running_mean, saved_running_mean), \
        "Running mean should not change in eval mode"
    assert np.allclose(bn3.running_var, saved_running_var), \
        "Running var should not change in eval mode"

    # Test 5: Parameter counting
    print("  Testing parameter counting...")
    bn4 = BatchNorm2d(num_features=64)
    params = bn4.parameters()

    assert len(params) == 2, f"Should have 2 parameters (gamma, beta), got {len(params)}"
    assert params[0].shape == (64,), f"Gamma shape should be (64,), got {params[0].shape}"
    assert params[1].shape == (64,), f"Beta shape should be (64,), got {params[1].shape}"

    print("✅ BatchNorm2d works correctly!")

if __name__ == "__main__":
    test_unit_batchnorm2d()

# %% [markdown]
"""
### 🧪 Unit Test: BatchNorm2d Gradients

**What we're testing**: That gamma and beta receive gradients, and that the
analytic backward matches a numerical estimate
**Why it matters**: BatchNorm's parameters are handed to the optimizer. If no
gradient reaches them the layer looks fine, trains fine, and learns nothing.
The failure is invisible from the forward pass alone
**Expected**: Non-None gradients of the right shape, matching finite differences
"""

# %% nbgrader={"grade": true, "grade_id": "test-batchnorm2d-grad", "locked": true, "points": 10}
def test_unit_batchnorm2d_gradients() -> None:
    """🧪 Test BatchNorm2d gradient flow."""
    print("🧪 Unit Test: BatchNorm2d Gradients...")

    # Test 1: gradients actually arrive
    print("  Testing gradients reach gamma and beta...")
    bn = BatchNorm2d(num_features=3)
    x = Tensor(rng.standard_normal((4, 3, 2, 2)), requires_grad=True)
    out = bn(x)

    assert out._grad_fn is not None, \
        "BatchNorm output has no _grad_fn: backward() will silently do nothing"

    out.sum().backward()

    assert bn.gamma.grad is not None, "gamma received no gradient, so it will never train"
    assert bn.beta.grad is not None, "beta received no gradient, so it will never train"
    assert x.grad is not None, "no gradient reached the input: the graph is severed here"
    assert bn.gamma.grad.shape == (3,), f"gamma grad shape {bn.gamma.grad.shape}, expected (3,)"
    assert bn.beta.grad.shape == (3,), f"beta grad shape {bn.beta.grad.shape}, expected (3,)"

    # beta is added to every position, so its gradient is that position count
    expected_beta_grad = 4 * 2 * 2
    assert np.allclose(np.asarray(bn.beta.grad.data), expected_beta_grad), \
        f"d(sum)/d(beta) should be N*H*W={expected_beta_grad} per channel"

    # Test 2: analytic gradient matches finite differences
    print("  Testing against numerical gradients...")
    bn2 = BatchNorm2d(num_features=2)
    bn2.gamma = Tensor(np.array([1.3, 0.7]), requires_grad=True)
    bn2.beta = Tensor(np.array([0.2, -0.4]), requires_grad=True)

    x_data = rng.standard_normal((3, 2, 2, 2))
    weights = rng.standard_normal((3, 2, 2, 2))   # random projection to a scalar

    def scalar_loss() -> float:
        probe = BatchNorm2d(num_features=2)
        probe.gamma, probe.beta = bn2.gamma, bn2.beta
        return float((np.asarray(probe(Tensor(x_data)).data) * weights).sum())

    xt = Tensor(x_data, requires_grad=True)
    (bn2(xt) * Tensor(weights)).sum().backward()
    analytic = np.asarray(xt.grad.data)

    # h=1e-2: large enough that float32 round-off does not swamp the difference,
    # small enough that the second-order truncation term stays negligible
    h = 1e-2
    numerical = np.zeros_like(x_data)
    it = np.nditer(x_data, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original = x_data[idx]
        x_data[idx] = original + h
        high = scalar_loss()
        x_data[idx] = original - h
        low = scalar_loss()
        x_data[idx] = original
        numerical[idx] = (high - low) / (2 * h)
        it.iternext()

    assert np.allclose(analytic, numerical, atol=1e-3), \
        f"Analytic and numerical gradients disagree: max diff {np.abs(analytic - numerical).max():.2e}"

    # Test 3: eval mode still delivers gradients, via the simpler path
    print("  Testing eval mode gradients...")
    bn3 = BatchNorm2d(num_features=3)
    bn3.eval()
    x3 = Tensor(rng.standard_normal((2, 3, 2, 2)), requires_grad=True)
    bn3(x3).sum().backward()
    assert bn3.gamma.grad is not None, "gamma should still train when statistics are frozen"
    assert x3.grad is not None, "gradient should still reach the input in eval mode"

    print("✅ BatchNorm2d gradients work correctly!")

if __name__ == "__main__":
    test_unit_batchnorm2d_gradients()

# %% [markdown]
"""
### 🧪 Unit Test: Pooling Operations

This test validates both max and average pooling implementations.

**What we're testing**: Dimension reduction, aggregation correctness
**Why it matters**: Pooling is essential for computational efficiency in CNNs
**Expected**: Correct output shapes and proper value aggregation
"""

# %% nbgrader={"grade": true, "grade_id": "test-pooling", "locked": true, "points": 10}
def test_unit_pooling() -> None:
    """🧪 Test MaxPool2d and AvgPool2d implementations."""
    print("🧪 Unit Test: Pooling Operations...")

    # Test 1: MaxPool2d basic functionality
    print("  Testing MaxPool2d...")
    maxpool = MaxPool2d(kernel_size=2, stride=2)
    x1 = Tensor(rng.standard_normal((1, 3, 8, 8)))
    out1 = maxpool(x1)

    expected_shape = (1, 3, 4, 4)  # 8/2 = 4
    assert out1.shape == expected_shape, f"MaxPool expected {expected_shape}, got {out1.shape}"

    # Test 2: AvgPool2d basic functionality
    print("  Testing AvgPool2d...")
    avgpool = AvgPool2d(kernel_size=2, stride=2)
    x2 = Tensor(rng.standard_normal((2, 16, 16, 16)))
    out2 = avgpool(x2)

    expected_shape = (2, 16, 8, 8)  # 16/2 = 8
    assert out2.shape == expected_shape, f"AvgPool expected {expected_shape}, got {out2.shape}"

    # Test 3: MaxPool vs AvgPool on known data
    print("  Testing max vs avg behavior...")
    # Create simple test case with known values
    test_data = np.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]], dtype=np.float32)
    x3 = Tensor(test_data)

    maxpool_test = MaxPool2d(kernel_size=2, stride=2)
    avgpool_test = AvgPool2d(kernel_size=2, stride=2)

    max_out = maxpool_test(x3)
    avg_out = avgpool_test(x3)

    # For 2x2 windows:
    # Top-left: max([1,2,5,6]) = 6, avg = 3.5
    # Top-right: max([3,4,7,8]) = 8, avg = 5.5
    # Bottom-left: max([9,10,13,14]) = 14, avg = 11.5
    # Bottom-right: max([11,12,15,16]) = 16, avg = 13.5

    expected_max = np.array([[[[6, 8], [14, 16]]]])
    expected_avg = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])

    assert np.allclose(max_out.data, expected_max), f"MaxPool values incorrect: {max_out.data} vs {expected_max}"
    assert np.allclose(avg_out.data, expected_avg), f"AvgPool values incorrect: {avg_out.data} vs {expected_avg}"

    # Test 4: Overlapping pooling (stride < kernel_size)
    print("  Testing overlapping pooling...")
    overlap_pool = MaxPool2d(kernel_size=3, stride=1)
    x4 = Tensor(rng.standard_normal((1, 1, 5, 5)))
    out4 = overlap_pool(x4)

    # Output: (5-3)/1 + 1 = 3
    expected_shape = (1, 1, 3, 3)
    assert out4.shape == expected_shape, f"Overlapping pool expected {expected_shape}, got {out4.shape}"

    # Test 5: No parameters in pooling layers
    print("  Testing parameter counts...")
    assert len(maxpool.parameters()) == 0, "MaxPool should have no parameters"
    assert len(avgpool.parameters()) == 0, "AvgPool should have no parameters"

    print("✅ Pooling operations work correctly!")

if __name__ == "__main__":
    test_unit_pooling()

# %% [markdown]
r"""
## 🔧 Integration: Building a Complete CNN

Now we combine convolution, activation, pooling, and linear classification into an end-to-end Convolutional Neural Network (`SimpleCNN`). You will see how spatial operations extract hierarchical visual representations from raw input pixels.

### SimpleCNN Architecture Datapath

| Layer | Operation | Input Shape | Kernel / Stride / Pad | Output Shape | Parameters | Activations ($B$ samples) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | RGB Image Batch | $(B, 3, 32, 32)$ | — | $(B, 3, 32, 32)$ | 0 | $3{,}072 \times B$ |
| **Conv1** | Spatial Filter Bank | $(B, 3, 32, 32)$ | $K=3, S=1, P=1$ | $(B, 16, 32, 32)$ | $16 \cdot (3 \cdot 3^2) + 16 = \mathbf{448}$ | $16{,}384 \times B$ |
| **ReLU1** | Non-linear Gating | $(B, 16, 32, 32)$ | Elementwise | $(B, 16, 32, 32)$ | 0 | $16{,}384 \times B$ |
| **Pool1** | Spatial Downsampling | $(B, 16, 32, 32)$ | $K=2, S=2$ | $(B, 16, 16, 16)$ | 0 | $4{,}096 \times B$ |
| **Conv2** | Higher-level Features | $(B, 16, 16, 16)$ | $K=3, S=1, P=1$ | $(B, 32, 16, 16)$ | $32 \cdot (16 \cdot 3^2) + 32 = \mathbf{4{,}640}$ | $8{,}192 \times B$ |
| **ReLU2** | Non-linear Gating | $(B, 32, 16, 16)$ | Elementwise | $(B, 32, 16, 16)$ | 0 | $8{,}192 \times B$ |
| **Pool2** | Spatial Downsampling | $(B, 32, 16, 16)$ | $K=2, S=2$ | $(B, 32, 8, 8)$ | 0 | $2{,}048 \times B$ |
| **Flatten** | Spatial Unrolling | $(B, 32, 8, 8)$ | Reshape | $(B, 2048)$ | 0 | $2{,}048 \times B$ |
| **FC** | Classification Logits | $(B, 2048)$ | Module 03 `Linear` | $(B, 10)$ | $2048 \cdot 10 + 10 = \mathbf{20{,}490}$ | $10 \times B$ |
| **Total** | **SimpleCNN Pipeline** | **$(B, 3, 32, 32)$** | — | **$(B, 10)$** | **25,578** | **$60{,}426 \times B$ elements ($\approx 241.7\text{ KB} \times B$)** |

The activation column counts elements, not bytes. Summing it gives $3{,}072 + 16{,}384 + 16{,}384 + 4{,}096 + 8{,}192 + 8{,}192 + 2{,}048 + 2{,}048 + 10 = 60{,}426$ elements per sample, which at $4$ B/element is $241{,}704$ B, or $\approx 241.7$ KB per sample. Autograd has to keep every one of them live until the backward pass reaches it.

---

### The Parameter Efficiency Story: CNN vs Dense MLP

| Subsystem / Layer | CNN Architecture (`SimpleCNN`) | Dense Equivalent (`Linear` MLP) | Parameter Savings |
| :--- | :--- | :--- | :--- |
| **Stage 1** | `Conv2d(3, 16, 3, p=1)`: 448 params | `Linear(3072, 1000)`: 3,073,000 params | **$6{,}859\times$ fewer parameters** |
| **Stage 2** | `Conv2d(16, 32, 3, p=1)`: 4,640 params | `Linear(1000, 500)`: 500,500 params | **$107\times$ fewer parameters** |
| **Classification Head** | `Linear(2048, 10)`: 20,490 params | `Linear(500, 10)`: 5,010 params | Tailored feature routing |
| **Total Parameters** | **25,578 parameters** | **3,578,510 parameters** | **$>139\times$ parameter reduction** |
| **Parameter Buffer** | $\mathbf{\approx 102\text{ KB}}$ (L1/L2 cache resident) | $\mathbf{\approx 14.3\text{ MB}}$ (spills to DRAM) | Dramatically reduced memory bandwidth |

---

### Receptive Field Growth Across Layers

As representations pass through alternating convolutions and pooling layers, the effective receptive field (the input region contributing to an activation) expands rapidly:

| Layer Stage | Kernel Size | Stride | Layer Receptive Field | Cumulative Receptive Field | Visual Semantic Scale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Image** | — | — | $1 \times 1$ | $1 \times 1$ ($1\text{ px}$) | Raw RGB color values |
| **Conv1** | $3 \times 3$ | 1 | $3 \times 3$ | $3 \times 3$ ($9\text{ px}$) | Oriented edges, luminance contrasts |
| **Pool1** | $2 \times 2$ | 2 | $2 \times 2$ | $4 \times 4$ ($16\text{ px}$) | Aggregated corner/junction responses |
| **Conv2** | $3 \times 3$ | 1 | $3 \times 3$ | $8 \times 8$ ($64\text{ px}$) | Combinations of corners, textured patches |
| **Pool2** | $2 \times 2$ | 2 | $2 \times 2$ | $10 \times 10$ ($100\text{ px}$) | Multi-part motifs spanning roughly a tenth of the $32 \times 32$ image |

Each row applies the same recurrence. Carry a jump $j$ (the cumulative stride, starting at $1$) alongside the receptive field $r$, then for a layer with kernel $k$ and stride $s$ set $r \leftarrow r + (k - 1) \cdot j$ and $j \leftarrow j \cdot s$. Pool2 sees $r = 8 + (2 - 1) \cdot 2 = 10$, because by then every step of its window covers $j = 4$ input pixels.
"""

# %% nbgrader={"grade": false, "grade_id": "simple-cnn", "solution": true}
#| export

class SimpleCNN:
    """
    Simple CNN demonstrating spatial operations integration.

    Architecture:
    - Conv2d(3→16, 3×3) + ReLU + MaxPool(2×2)
    - Conv2d(16→32, 3×3) + ReLU + MaxPool(2×2)
    - Flatten + Linear(features→num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        """
        Initialize SimpleCNN.

        TODO: Build CNN architecture with spatial and dense layers

        APPROACH:
        1. Conv layer 1: 3 → 16 channels, 3×3 kernel, padding=1
        2. Pool layer 1: 2×2 max pooling
        3. Conv layer 2: 16 → 32 channels, 3×3 kernel, padding=1
        4. Pool layer 2: 2×2 max pooling
        5. Calculate flattened size and add the final Linear layer (Module 03)

        HINT: For 32×32 input → 32 → 16 → 8 spatial reduction
        Final feature size: 32 channels × 8 × 8 = 2048 features
        Linear(in_features, out_features) maps those 2048 features to num_classes logits
        """
        ### BEGIN SOLUTION role="scaffold"
        # Convolutional layers
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size
        # Input: 32×32 → Conv1+Pool1: 16×16 → Conv2+Pool2: 8×8
        # Final: 32 channels × 8 × 8 = 2048 features
        self.flattened_size = 32 * 8 * 8

        # Classification head: the Linear layer from Module 03
        self.fc = Linear(self.flattened_size, num_classes)
        self.relu = ReLU()
        self.num_classes = num_classes
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through SimpleCNN.

        TODO: Implement CNN forward pass

        APPROACH:
        1. Apply conv1 → ReLU → pool1
        2. Apply conv2 → ReLU → pool2
        3. Flatten spatial dimensions
        4. Apply the final Linear layer to get class logits

        EXAMPLE:
        >>> model = SimpleCNN(num_classes=10)
        >>> logits = model(Tensor(rng.standard_normal((2, 3, 32, 32))))
        >>> print(logits.shape)  # (2, 10)
        """
        ### BEGIN SOLUTION role="scaffold"
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)  # ReLU activation
        x = self.pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)  # ReLU activation
        x = self.pool2(x)

        # Flatten for classification (reshape to 2D)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Classification head
        return self.fc(x)
        ### END SOLUTION

    def parameters(self) -> list[Tensor]:
        """Return all trainable parameters."""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.fc.parameters())
        return params

    def __call__(self, x: Tensor) -> Tensor:
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: SimpleCNN Integration

This test validates that spatial operations work together in a complete CNN architecture.

**What we're testing**: End-to-end spatial processing pipeline
**Why it matters**: Spatial operations must compose correctly for real CNNs
**Expected**: Proper dimension reduction and one logit per class
"""

# %% nbgrader={"grade": true, "grade_id": "test-simple-cnn", "locked": true, "points": 10}


def test_unit_simple_cnn() -> None:
    """🧪 Test SimpleCNN integration with spatial operations."""
    print("🧪 Unit Test: SimpleCNN Integration...")

    # Test 1: Forward pass with CIFAR-10 sized input
    print("  Testing forward pass...")
    model = SimpleCNN(num_classes=10)
    x = Tensor(rng.standard_normal((2, 3, 32, 32)))  # Batch of 2, RGB, 32×32

    logits = model(x)

    # Expected: 2 samples, one logit per class
    expected_shape = (2, 10)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

    # Test 2: Parameter counting
    print("  Testing parameter counting...")
    params = model.parameters()

    # Conv1: (16, 3, 3, 3) + bias (16,) = 432 + 16 = 448
    # Conv2: (32, 16, 3, 3) + bias (32,) = 4608 + 32 = 4640
    # Linear: (2048, 10) + bias (10,) = 20480 + 10 = 20490
    # Total: 448 + 4640 + 20490 = 25578 parameters

    conv1_params = 16 * 3 * 3 * 3 + 16  # weights + bias
    conv2_params = 32 * 16 * 3 * 3 + 32  # weights + bias
    fc_params = 2048 * 10 + 10  # weights + bias
    expected_total = conv1_params + conv2_params + fc_params

    actual_total = sum(np.prod(p.shape) for p in params)
    assert actual_total == expected_total, f"Expected {expected_total} parameters, got {actual_total}"

    # Test 3: Batch processing
    print("  Testing batch processing...")
    x_batch = Tensor(rng.standard_normal((8, 3, 32, 32)))
    logits_batch = model(x_batch)

    expected_batch = (8, 10)
    assert logits_batch.shape == expected_batch, f"Expected {expected_batch}, got {logits_batch.shape}"

    print("✅ SimpleCNN integration works correctly!")

if __name__ == "__main__":
    test_unit_simple_cnn()

# %% [markdown]
"""
## 📊 Systems Analysis: Spatial Operation Performance

Let's understand ONE key systems concept: **computational complexity and memory trade-offs in spatial operations**.

This single analysis reveals why certain design choices matter for real-world performance, and why modern CNNs use specific architectural patterns.
"""

# %% nbgrader={"grade": false, "grade_id": "spatial-analysis", "solution": false}
def analyze_convolution_complexity() -> None:
    """📊 Analyze convolution computational complexity across different configurations."""
    print("📊 Analyzing Convolution Complexity...")

    # Test configurations optimized for educational demonstration (smaller sizes)
    configs = [
        {"input": (1, 3, 16, 16), "conv": (8, 3, 3), "name": "Small (16×16)"},
        {"input": (1, 3, 24, 24), "conv": (12, 3, 3), "name": "Medium (24×24)"},
        {"input": (1, 3, 32, 32), "conv": (16, 3, 3), "name": "Large (32×32)"},
        {"input": (1, 3, 16, 16), "conv": (8, 3, 5), "name": "Large Kernel (5×5)"},
    ]

    print(f"{'Configuration':<20} {'FLOPs':<15} {'Memory (MB)':<12} {'Time (ms)':<10}")
    print("-" * 70)

    for config in configs:
        # Create convolution layer
        in_ch = config["input"][1]
        out_ch, k_size = config["conv"][0], config["conv"][2]
        conv = Conv2d(in_ch, out_ch, kernel_size=k_size, padding=k_size//2)

        # Create input tensor
        x = Tensor(rng.standard_normal(config["input"]))

        # Calculate theoretical FLOPs
        batch, in_channels, h, w = config["input"]
        out_channels, kernel_size = config["conv"][0], config["conv"][2]

        # Each output element requires in_channels * kernel_size² multiply-adds
        flops_per_output = in_channels * kernel_size * kernel_size * 2  # 2 for MAC
        total_outputs = batch * out_channels * h * w  # Assuming same size with padding
        total_flops = flops_per_output * total_outputs

        # Measure memory usage
        input_memory = np.prod(config["input"]) * 4  # float32 = 4 bytes
        weight_memory = out_channels * in_channels * kernel_size * kernel_size * 4
        output_memory = batch * out_channels * h * w * 4
        total_memory = (input_memory + weight_memory + output_memory) / (1024 * 1024)  # MB

        # Measure execution time
        start_time = time.time()
        _ = conv(x)
        end_time = time.time()
        exec_time = (end_time - start_time) * 1000  # ms

        print(f"{config['name']:<20} {total_flops:<15,} {total_memory:<12.2f} {exec_time:<10.2f}")

    print("\n💡 Key Insights:")
    print("🔸 FLOPs scale as O(H×W×C_in×C_out×K²) - quadratic in spatial and kernel size")
    print("🔸 Memory scales linearly with spatial dimensions and channels")
    print("🔸 Large kernels dramatically increase computational cost")
    print("🚀 This motivates more efficient convolution variants that reduce computational cost")

if __name__ == "__main__":
    analyze_convolution_complexity()

# %% nbgrader={"grade": false, "grade_id": "pooling-analysis", "solution": false}
def analyze_pooling_effects() -> None:
    """📊 Analyze pooling's impact on spatial dimensions and features."""
    print("\n📊 Analyzing Pooling Effects...")

    # Create sample input with spatial structure
    # Simple edge pattern that pooling should preserve differently
    pattern = np.zeros((1, 1, 8, 8))
    pattern[0, 0, :, 3:5] = 1.0  # Vertical edge
    pattern[0, 0, 3:5, :] = 1.0  # Horizontal edge
    x = Tensor(pattern)

    print("Original 8×8 pattern:")
    print(x.data[0, 0])

    # Test different pooling strategies
    pools = [
        (MaxPool2d(2, stride=2), "MaxPool 2×2"),
        (AvgPool2d(2, stride=2), "AvgPool 2×2"),
        (MaxPool2d(4, stride=4), "MaxPool 4×4"),
        (AvgPool2d(4, stride=4), "AvgPool 4×4"),
    ]

    print(f"\n{'Operation':<15} {'Output Shape':<15} {'Feature Preservation'}")
    print("-" * 60)

    for pool_op, name in pools:
        result = pool_op(x)
        # Measure how much of the original pattern is preserved
        preservation = np.sum(result.data > 0.1) / np.prod(result.shape)
        print(f"{name:<15} {str(result.shape):<15} {preservation:<.2%}")

        print(f"  Output:")
        print(f"  {result.data[0, 0]}")
        print()

    print("💡 Key Insights:")
    print("🔸 MaxPool preserves sharp features better (edge detection)")
    print("🔸 AvgPool smooths features (noise reduction)")
    print("🔸 Larger pooling windows lose more spatial detail")
    print("🚀 Choice depends on task: classification vs detection vs segmentation")

if __name__ == "__main__":
    analyze_pooling_effects()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 15}
def test_module() -> None:
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire spatial module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")

    # Conv2d helper tests
    test_unit_conv2d_output_shape()
    test_unit_conv2d_padding()
    test_unit_conv2d_convolve_loops()
    test_unit_conv2d()

    # BatchNorm2d helper tests
    test_unit_batchnorm2d_validate_input()
    test_unit_batchnorm2d_get_stats()

    # MaxPool2d helper tests
    test_unit_maxpool2d_output_shape()
    test_unit_maxpool2d_loops()

    # AvgPool2d helper tests
    test_unit_avgpool2d_output_shape()
    test_unit_avgpool2d_loops()

    # Remaining unit tests
    test_unit_batchnorm2d()
    test_unit_batchnorm2d_gradients()
    test_unit_pooling()
    test_unit_simple_cnn()

    print("\nRunning integration scenarios...")

    # Test realistic CNN workflow with BatchNorm
    print("🧪 Integration Test: Complete CNN pipeline with BatchNorm...")

    # Create a mini CNN for CIFAR-10 with BatchNorm (modern architecture)
    conv1 = Conv2d(3, 8, kernel_size=3, padding=1)
    bn1 = BatchNorm2d(8)
    pool1 = MaxPool2d(2, stride=2)
    conv2 = Conv2d(8, 16, kernel_size=3, padding=1)
    bn2 = BatchNorm2d(16)
    pool2 = AvgPool2d(2, stride=2)
    relu = ReLU()

    # Process batch of images (training mode)
    batch_images = Tensor(rng.standard_normal((4, 3, 32, 32)))

    # Forward pass: Conv → BatchNorm → ReLU → Pool (modern pattern)
    x = conv1(batch_images)  # (4, 8, 32, 32)
    x = bn1(x)               # (4, 8, 32, 32) - normalized
    x = relu(x)
    x = pool1(x)             # (4, 8, 16, 16)

    x = conv2(x)             # (4, 16, 16, 16)
    x = bn2(x)               # (4, 16, 16, 16) - normalized
    x = relu(x)
    features = pool2(x)      # (4, 16, 8, 8)

    # Validate shapes at each step
    assert features.shape[0] == 4, f"Batch size should be preserved, got {features.shape[0]}"
    assert features.shape == (4, 16, 8, 8), f"Final features shape incorrect: {features.shape}"

    # Test parameter collection across all layers
    all_params = []
    all_params.extend(conv1.parameters())
    all_params.extend(bn1.parameters())
    all_params.extend(conv2.parameters())
    all_params.extend(bn2.parameters())

    # Pooling has no parameters
    assert len(pool1.parameters()) == 0
    assert len(pool2.parameters()) == 0

    # BatchNorm has 2 params each (gamma, beta)
    assert len(bn1.parameters()) == 2, f"BatchNorm should have 2 parameters, got {len(bn1.parameters())}"

    # Total: Conv1 (2) + BN1 (2) + Conv2 (2) + BN2 (2) = 8 parameters
    assert len(all_params) == 8, f"Expected 8 parameter tensors total, got {len(all_params)}"

    # Test train/eval mode switching
    print("🧪 Integration Test: Train/Eval mode switching...")
    bn1.eval()
    bn2.eval()

    # Run single-sample inference using frozen running statistics
    single_image = Tensor(rng.standard_normal((1, 3, 32, 32)))
    x = conv1(single_image)
    x = bn1(x)  # Uses running stats, not batch stats
    assert x.shape == (1, 8, 32, 32), f"Single sample inference should work in eval mode"

    print("✅ CNN pipeline with BatchNorm works correctly!")

    # Test memory efficiency comparison
    print("🧪 Integration Test: Memory efficiency analysis...")

    # Compare different pooling strategies (reduced size for faster execution)
    input_data = Tensor(rng.standard_normal((1, 16, 32, 32)))

    # No pooling: maintain spatial size
    conv_only = Conv2d(16, 32, kernel_size=3, padding=1)
    no_pool_out = conv_only(input_data)
    no_pool_size = np.prod(no_pool_out.shape) * 4  # float32 bytes

    # With pooling: reduce spatial size
    conv_with_pool = Conv2d(16, 32, kernel_size=3, padding=1)
    pool = MaxPool2d(2, stride=2)
    pool_out = pool(conv_with_pool(input_data))
    pool_size = np.prod(pool_out.shape) * 4  # float32 bytes

    memory_reduction = no_pool_size / pool_size
    assert memory_reduction == 4.0, f"2×2 pooling should give 4× memory reduction, got {memory_reduction:.1f}×"

    print(f"  Memory reduction with pooling: {memory_reduction:.1f}×")
    print("✅ Memory efficiency analysis complete!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 09")

# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

Answer these questions to deepen your systems understanding of spatial computing, memory hierarchies, and hardware acceleration:

### Question 1: Conv2d Memory Footprint & VRAM Scaling
A `Conv2d` layer with 64 filters of shape $(3, 3, 3)$, stride $1$, and padding $1$ processes a $(224 \times 224 \times 3)$ input.
- **Input Memory**: $1 \times 3 \times 224 \times 224 \times 4\text{ B} = 602{,}112\text{ B} = 602.1\text{ KB}$
- **Filter Weights**: $64 \times 3 \times 3 \times 3 \times 4\text{ B} = 6{,}912\text{ B} = 6.9\text{ KB}$
- **Biases**: $64 \times 4\text{ B} = 256\text{ B}$
- **Output Activations**: $1 \times 64 \times 224 \times 224 \times 4\text{ B} = 12{,}845{,}056\text{ B} = 12.85\text{ MB}$

**Systems Implication**:
When the batch size scales from $B=1$ to $B=32$, activation memory scales linearly from $12.85\text{ MB} \to \mathbf{411.0\text{ MB}}$ for this single layer (all figures decimal, as declared in 📐)! During training, autograd must retain all intermediate activations in VRAM for the backward pass. In deep 50-layer networks, this causes activation memory to dominate parameter memory by orders of magnitude, motivating techniques like **activation recomputation** (gradient checkpointing).

---

### Question 2: Spatial Locality, Striding & Cache Reuse
Why do convolutional kernels exhibit far superior hardware cache efficiency than fully-connected dense layers of comparable parameter scale?

- **Temporal Locality**: Filter kernel weights ($K \times K \times C_{\text{in}}$) are held resident in high-speed L1/L2 cache and reused across all $H \times W$ spatial positions.
- **Spatial Locality**: Contiguous horizontal pixel sweeps in row-major order (`C-contiguous`) maximize CPU cache line prefetching (64 bytes/line = 16 `float32` elements loaded simultaneously).
- **Access Striding**: Dense MLPs perform large matrix-vector multiplications that touch large parameter arrays once per input sample, placing intense pressure on main memory DRAM bandwidth.

---

### Question 3: The `im2col` Lowering Trade-off

The `im2col` algorithm lowers a multi-channel 2D convolution into a single dense matrix multiplication (GEMM):

<div align="center">
  <img src="im2col_lowering_gemm.svg" alt="Lowering 2D Convolution to Level-3 BLAS GEMM via im2col" width="680px">
</div>

<div align="center">
  <img src="im2col_unfolding.svg" alt="Spatial Unfolding into Level-3 BLAS GEMM" width="580px">
</div>

**Trade-off Analysis**:
- **Memory Cost**: Overlapping spatial patches duplicate input data in memory by a factor of $K_h \cdot K_w$ ($9\times$ for $3 \times 3$ kernels) at stride $1$. The factor is really $\frac{K_h K_w}{S_h S_w}$, so it falls as stride grows and reaches $1\times$ exactly when $S = K$ and the patches stop overlapping.
- **Compute Gain**: Transforming sliding loops into a standardized GEMM matrix ($X_{\text{col}} \in \mathbb{R}^{(C_{\text{in}} K_h K_w) \times (H_{\text{out}} W_{\text{out}})}$) unlocks vendor-tuned Level-3 BLAS (cuBLAS, CUTLASS, oneDNN) and GPU Tensor Cores, which reach a far larger fraction of peak hardware FLOPS than any loop nest you or a compiler would write by hand. That gap, not any reduction in arithmetic, is what pays for the duplicated memory.
- **Mobile / Edge Constraint**: On mobile devices with strict unified RAM budgets, materializing large intermediate `im2col` matrices causes out-of-memory crashes or thermal throttling. Mobile engines therefore favor direct convolutions or fused on-the-fly implicit GEMM.

---

### Question 4: Pooling Systems Benefits Beyond Parameter Reduction
When `MaxPool2d` or `AvgPool2d` downsamples spatial dimensions ($224 \times 224 \to 112 \times 112$):
- **Activation VRAM**: Cuts activation buffer size by $4\times$ ($75\%$ reduction).
- **Backprop Gradient Traffic**: Downstream layers backpropagate through $4\times$ fewer spatial gradient coordinates.
- **Cache Residency**: Working feature tensors stay resident in fast SRAM and L3 cache rather than spilling to off-chip DRAM.
- **Cumulative Compaction**: If five stages each apply $2 \times 2$ pooling, spatial resolution decreases by $2^5 = 32\times$, reducing feature map area by $32^2 = \mathbf{1{,}024\times}$!

---

### Question 5: Depthwise-Separable Convolutions in Mobile Architectures
Why do mobile networks (MobileNet, EfficientNet) replace standard convolutions with depthwise-separable convolutions?

- **Standard $3 \times 3$ Conv FLOPs**:
  $$\text{FLOPs}_{\text{standard}} = 2 \cdot B \cdot H \cdot W \cdot (C_{\text{in}} \cdot C_{\text{out}} \cdot K_h \cdot K_w)$$
- **Depthwise-Separable FLOPs** ($3 \times 3$ Depthwise + $1 \times 1$ Pointwise):
  $$\text{FLOPs}_{\text{separable}} = 2 \cdot B \cdot H \cdot W \cdot (C_{\text{in}} \cdot K_h \cdot K_w) + 2 \cdot B \cdot H \cdot W \cdot (C_{\text{in}} \cdot C_{\text{out}})$$
- **FLOPs Ratio**:
  $$\frac{\text{FLOPs}_{\text{separable}}}{\text{FLOPs}_{\text{standard}}} = \frac{K_h K_w + C_{\text{out}}}{K_h K_w \cdot C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{K_h K_w}$$

For $K=3$ and $C_{\text{out}}=64$:
$$\text{Ratio} = \frac{1}{64} + \frac{1}{9} \approx 0.0156 + 0.1111 = 0.1267 \implies \mathbf{\approx 8\times\text{ compute reduction!}}$$
"""

# %% [markdown]
r"""
## ⭐ Aha Moment: Convolution Extracts Hierarchical Features

**What you built:** Convolutional and pooling layers that transform raw spatial pixels into structured, invariant feature representations.

**Why it matters:** Unlike dense linear layers that treat every input coordinate independently, `Conv2d` leverages weight sharing and local receptive fields to detect patterns regardless of spatial position. Combined with pooling and batch normalization, you have built the foundational engine of modern computer vision.
"""

# %%
def demo_convolutions() -> None:
    """🎯 See Conv2d process spatial data."""
    print("🎯 AHA MOMENT: Convolution Extracts Features")
    print("=" * 45)

    # Create a simple 8x8 "image" with 1 channel
    image = Tensor(rng.standard_normal((1, 1, 8, 8)))

    # Conv2d: 1 input channel → 4 feature maps
    conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3)

    output = conv(image)

    print(f"Input:  {image.shape}  ← 1 image, 1 channel, 8×8")
    print(f"Output: {output.shape}  ← 1 image, 4 features, 6×6")
    print(f"\nConv kernel: 3×3 sliding window")
    print(f"Output smaller: 8 - 3 + 1 = 6 (no padding)")

    print("\n✨ Conv2d detects spatial patterns in images!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_convolutions()

# %% [markdown]
r"""
## 🚀 MODULE SUMMARY: Spatial Operations

Congratulations! You have built the complete spatial computing foundation of TinyTorch.

### Key Accomplishments
- **Implemented `Conv2d`**: Explicit 7-nested loops demonstrating the $\mathcal{O}(B \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot K_h \cdot K_w \cdot C_{\text{in}})$ complexity and spatial weight sharing.
- **Formulated `BatchNorm2d`**: Dynamic mini-batch normalization during training versus frozen EMA tracking during inference, complete with 3-route backward autograd.
- **Constructed Pooling Operators**: `MaxPool2d` (argmax routing) and `AvgPool2d` (uniform spatial dispersion) for $4\times$ memory reduction.
- **Engineered `SimpleCNN`**: End-to-end vision pipeline achieving $>139\times$ parameter reduction compared to fully-connected MLPs.
- **Analyzed MLSys Lowering**: Detailed trade-offs of `im2col` GEMM lowering, spatial memory footprints, and mobile depthwise-separable acceleration.

### Systems Insights Discovered
- **Weight sharing is a memory win before it is a statistical one**: `SimpleCNN` carries 25,578 parameters where the dense equivalent carries 3,578,510. That $139\times$ reduction puts the entire parameter buffer in $\approx 102$ KB, small enough to sit in L1/L2 cache, while the dense model's $\approx 14.3$ MB has to be streamed from DRAM on every forward pass.
- **Activations dominate parameters, and that is what fills the device**: those 25,578 parameters are stored once, but one sample's activations run to 60,426 elements ($\approx 241.7$ KB), and autograd has to keep all of them live until the backward pass consumes them. At $B=32$ the activations outweigh the weights by roughly $75\times$. This is the asymmetry that makes activation recomputation worth its extra forward pass.
- **The seven-loop cost model tells you where the work is**: total work is $B \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot K_h \cdot K_w \cdot C_{\text{in}}$, quadratic in kernel width and quadratic in spatial resolution, so $3 \times 3 \to 7 \times 7$ costs $5.44\times$ more. `im2col` removes none of that arithmetic. It reshapes the loop nest into a GEMM that vendor BLAS can actually run near peak, and pays $K_h \cdot K_w$ input duplication at stride $1$ for the privilege.
- **Pooling buys the budget back, and spatial precision pays for it**: one $2 \times 2$ stride-2 pool cuts the activation buffer $4\times$ and the next layer's FLOPs $4\times$, at the cost of knowing exactly where a feature was. Stack five and the feature map area falls $1{,}024\times$, which is also why objects smaller than the pooling window can vanish.
- **Store or recompute is a design choice, not a detail**: max-pool backward needs each window's argmax. PyTorch stores the indices in the forward pass and scatters, spending $8$ bytes per output element. This module stores nothing and re-scans every window. The gradients are identical and the resource bill is inverted, which is the same trade you will make again for every intermediate a backward pass needs.
- **Train and eval are two different functions**: `BatchNorm2d` normalizes with batch statistics during training and with frozen running statistics during inference, so the same weights on the same input produce different outputs depending on the mode. There is no error and no crash when you forget to switch, only worse numbers, which is why the mode flag is checked in the layer rather than assumed by the caller.

---

### Ready for Next Steps
All spatial primitives export to `tinytorch.core.spatial` and integrate directly into TinyTorch's autograd engine. Images are now a first-class input to TinyTorch, alongside the plain vectors you started with in Module 01.

Export with: `tito module complete 09`

**Next**: Module 10 will turn raw text into token IDs with `BPETokenizer`, opening the second input modality that Modules 11 through 13 build a transformer on!
"""
