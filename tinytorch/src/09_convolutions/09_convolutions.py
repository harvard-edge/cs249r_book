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
"""
# Module 09: Convolutions - Processing Images with Convolutions

Welcome to Module 09! You'll implement spatial operations that transform machine learning from working with simple vectors to understanding images and spatial patterns.

## 🔗 Prerequisites & Progress
**You've Built**: Complete training pipeline with MLPs, optimizers, and data loaders
**You'll Build**: Spatial operations - Conv2d, MaxPool2d, AvgPool2d for image processing
**You'll Enable**: Convolutional Neural Networks (CNNs) for computer vision

**Connection Map**:
```
Training Pipeline → Spatial Operations → CNN (Milestone 03)
    (MLPs)            (Conv/Pool)        (Computer Vision)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement Conv2d with explicit loops to understand O(N²M²K²) complexity
2. Build pooling operations (Max and Average) for spatial reduction
3. Understand receptive fields and spatial feature extraction
4. Analyze memory vs computation trade-offs in spatial operations

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/09_convolutions/convolutions_dev.py`
**Building Side:** Code exports to `tinytorch.core.spatial`

```python
# How to use this module:
from tinytorch.core.spatial import Conv2d, MaxPool2d, AvgPool2d
```

**Why this matters:**
- **Learning:** Complete spatial processing system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn.Conv2d with all spatial operations together
- **Consistency:** All convolution and pooling operations in core.spatial
- **Integration:** Works seamlessly with existing layers for complete CNN architectures
"""

# %% nbgrader={"grade": false, "grade_id": "spatial-setup", "solution": true}


#| default_exp core.spatial

#| export
import numpy as np
import time

from tinytorch.core.tensor import Tensor

# Constants for convolution defaults
DEFAULT_KERNEL_SIZE = 3  # Default kernel size for convolutions
DEFAULT_STRIDE = 1  # Default stride for convolutions
DEFAULT_PADDING = 0  # Default padding for convolutions

# %% [markdown]
"""
## 💡 Introduction - What are Spatial Operations?

Spatial operations transform machine learning from working with simple vectors to understanding images and spatial patterns. When you look at a photo, your brain naturally processes spatial relationships - edges, textures, objects. Spatial operations give neural networks this same capability.

### The Two Core Spatial Operations

**Convolution**: Detects local patterns by sliding filters across the input
**Pooling**: Reduces spatial dimensions while preserving important features

### Visual Example: How Convolution Works

```
Input Image (5×5):        Kernel (3×3):        Output (3×3):
┌─────────────────┐      ┌───────────┐       ┌─────────┐
│ 1  2  3  4  5   │      │  1  0  -1 │       │ ?  ?  ? │
│ 6  7  8  9  0   │  *   │  1  0  -1 │   =   │ ?  ?  ? │
│ 1  2  3  4  5   │      │  1  0  -1 │       │ ?  ?  ? │
│ 6  7  8  9  0   │      └───────────┘       └─────────┘
│ 1  2  3  4  5   │
└─────────────────┘

Sliding Window Process:
Position (0,0): [1,2,3]   Position (0,1): [2,3,4]   Position (0,2): [3,4,5]
               [6,7,8] *               [7,8,9] *               [8,9,0] *
               [1,2,3]                 [2,3,4]                 [3,4,5]
               = Output[0,0]           = Output[0,1]           = Output[0,2]
```

Each output pixel summarizes a local neighborhood, allowing the network to detect patterns like edges, corners, and textures.

### Why Spatial Operations Transform ML

```
Without Convolution:                    With Convolution:
32×32×3 image = 3,072 inputs          32×32×3 → Conv → 32×32×16
↓                                      ↓                     ↓
Dense(3072 → 1000) = 3M parameters    Shared 3×3 kernel = 432 parameters
↓                                      ↓                     ↓
Memory explosion + no spatial awareness Efficient + preserves spatial structure
```

Convolution achieves dramatic parameter reduction (1000× fewer!) while preserving the spatial relationships that matter for visual understanding.
"""

# %% [markdown]
"""
## 📐 Mathematical Foundations

### Understanding Convolution Step by Step

Convolution sounds complex, but it's just "sliding window multiplication and summation." Let's see exactly how it works:

```
Step 1: Position the kernel over input
Input:          Kernel:
┌─────────┐     ┌─────┐
│ 1 2 3 4 │     │ 1 0 │  ← Place kernel at position (0,0)
│ 5 6 7 8 │  ×  │ 0 1 │
│ 9 0 1 2 │     └─────┘
└─────────┘

Step 2: Multiply corresponding elements
Overlap:        Computation:
┌─────┐         1×1 + 2×0 + 5×0 + 6×1 = 1 + 0 + 0 + 6 = 7
│ 1 2 │
│ 5 6 │
└─────┘

Step 3: Slide kernel and repeat
Position (0,1):  Position (1,0):  Position (1,1):
┌─────┐         ┌─────┐          ┌─────┐
│ 2 3 │         │ 5 6 │          │ 6 7 │
│ 6 7 │         │ 9 0 │          │ 0 1 │
└─────┘         └─────┘          └─────┘
Result: 9       Result: 5        Result: 8

Final Output:   ┌─────┐
               │ 7 9 │
               │ 5 8 │
               └─────┘
```

### The Mathematical Formula

For 2D convolution, we slide kernel K across input I:
```
O[i,j] = Σ Σ I[i+m, j+n] × K[m,n]
         m n
```

This formula captures the "multiply and sum" operation for each kernel position.

### Pooling: Spatial Summarization

```
Max Pooling Example (2×2 window):
Input:             Output:
┌───────────────┐  ┌───────┐
│ 1  3  2  4    │  │ 6   8 │  ← max([1,3,5,6])=6, max([2,4,7,8])=8
│ 5  6  7  8    │  │ 9   9 │  ← max([5,2,9,1])=9, max([7,4,9,3])=9
│ 2  9  1  3    │  └───────┘
│ 0  1  9  3    │
└───────────────┘

Average Pooling (same window):
┌─────────────┐
│ 3.75   5.25 │  ← avg([1,3,5,6])=3.75, avg([2,4,7,8])=5.25
│ 2.75   5.75 │  ← avg([5,2,9,1])=4.25, avg([7,4,9,3])=5.75
└─────────────┘
```

### Why This Complexity Matters

For convolution with input (1, 3, 224, 224) and kernel (64, 3, 3, 3):
- **Operations**: 1 × 64 × 3 × 3 × 3 × 224 × 224 = 86.7 million multiply-adds
- **Memory**: Input (600KB) + Weights (6.9KB) + Output (12.8MB) = ~13.4MB

This is why kernel size matters enormously - a 7×7 kernel would require 5.4× more computation!

### Key Properties That Enable Deep Learning

**Translation Equivariance**: Move the cat → detection moves the same way
**Parameter Sharing**: Same edge detector works everywhere in the image
**Local Connectivity**: Each output only looks at nearby inputs (like human vision)
**Hierarchical Features**: Early layers detect edges → later layers detect objects
"""

# %% [markdown]
"""
## 🏗️ Implementation - Building Spatial Operations

Now we'll implement convolution step by step, using explicit loops so you can see and feel the computational complexity. This helps you understand why modern optimizations matter!

### Conv2d: Detecting Patterns with Sliding Windows

Convolution slides a small filter (kernel) across the entire input, computing weighted sums at each position. Think of it like using a template to find matching patterns everywhere in an image.

```
Convolution Visualization:
Input (4×4):              Kernel (3×3):           Output (2×2):
┌─────────────┐          ┌─────────┐             ┌─────────┐
│ a b c d │            │ k1 k2 k3│             │ o1  o2 │
│ e f g h │     ×      │ k4 k5 k6│      =      │ o3  o4 │
│ i j k l │            │ k7 k8 k9│             └─────────┘
│ m n o p │            └─────────┘
└─────────────┘

Computation Details:
o1 = a×k1 + b×k2 + c×k3 + e×k4 + f×k5 + g×k6 + i×k7 + j×k8 + k×k9
o2 = b×k1 + c×k2 + d×k3 + f×k4 + g×k5 + h×k6 + j×k7 + k×k8 + l×k9
o3 = e×k1 + f×k2 + g×k3 + i×k4 + j×k5 + k×k6 + m×k7 + n×k8 + o×k9
o4 = f×k1 + g×k2 + h×k3 + j×k4 + k×k5 + l×k6 + n×k7 + o×k8 + p×k9
```

### The Six Nested Loops of Convolution

Our implementation will use explicit loops to show exactly where the computational cost comes from:

```
for batch in range(B):          # Loop 1: Process each sample
    for out_ch in range(C_out):     # Loop 2: Generate each output channel
        for out_h in range(H_out):      # Loop 3: Each output row
            for out_w in range(W_out):      # Loop 4: Each output column
                for k_h in range(K_h):          # Loop 5: Each kernel row
                    for k_w in range(K_w):          # Loop 6: Each kernel column
                        for in_ch in range(C_in):       # Loop 7: Each input channel
                            # The actual multiply-accumulate operation
                            result += input[...] * kernel[...]
```

Total operations: B × C_out × H_out × W_out × K_h × K_w × C_in

For typical values (B=32, C_out=64, H_out=224, W_out=224, K_h=3, K_w=3, C_in=3):
That's 32 × 64 × 224 × 224 × 3 × 3 × 3 = **2.8 billion operations** per forward pass!
"""

# %% [markdown]
"""
### Conv2d Implementation - Building the Core of Computer Vision

Conv2d is the workhorse of computer vision. It slides learned filters across images to detect patterns like edges, textures, and eventually complex objects.

#### How Conv2d Transforms Machine Learning

```
Before Conv2d (Dense Only):         After Conv2d (Spatial Aware):
Input: 32×32×3 = 3,072 values      Input: 32×32×3 structured as image
         ↓                                   ↓
Dense(3072→1000) = 3M params       Conv2d(3→16, 3×3) = 448 params
         ↓                                   ↓
No spatial awareness               Preserves spatial relationships
Massive parameter count            Parameter sharing across space
```

#### Weight Initialization: He Initialization for ReLU Networks

Our Conv2d uses He initialization, specifically designed for ReLU activations:
- **Problem**: Wrong initialization → vanishing/exploding gradients
- **Solution**: std = sqrt(2 / fan_in) where fan_in = channels × kernel_height × kernel_width
- **Why it works**: Maintains variance through ReLU nonlinearity

#### The 6-Loop Implementation Strategy

We'll implement convolution with explicit loops to show the true computational cost:

```
Nested Loop Structure:
for batch:           ← Process each sample in parallel (in practice)
  for out_channel:   ← Generate each output feature map
    for out_h:       ← Each row of output
      for out_w:     ← Each column of output
        for k_h:     ← Each row of kernel
          for k_w:   ← Each column of kernel
            for in_ch: ← Accumulate across input channels
              result += input[...] * weight[...]
```

This reveals why convolution is expensive: O(B×C_out×H×W×K_h×K_w×C_in) operations!
"""

# %% nbgrader={"grade": false, "grade_id": "conv2d-class", "solution": true}

#| export

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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
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
        super().__init__()

        ### BEGIN SOLUTION
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
        self.weight = Tensor(np.random.normal(0, std,
                           (out_channels, in_channels, kernel_h, kernel_w)))

        # Bias initialization
        if bias:
            self.bias = Tensor(np.zeros(out_channels))
        else:
            self.bias = None
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through Conv2d layer.

        TODO: Implement convolution with explicit loops

        APPROACH:
        1. Extract input dimensions and validate
        2. Calculate output dimensions
        3. Apply padding if needed
        4. Implement 6 nested loops for full convolution
        5. Add bias if present

        LOOP STRUCTURE:
        for batch in range(batch_size):
            for out_ch in range(out_channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        for k_h in range(kernel_height):
                            for k_w in range(kernel_width):
                                for in_ch in range(in_channels):
                                    # Accumulate: out += input * weight

        EXAMPLE:
        >>> conv = Conv2d(3, 16, kernel_size=3, padding=1)
        >>> x = Tensor(np.random.randn(2, 3, 32, 32))  # batch=2, RGB, 32x32
        >>> out = conv(x)
        >>> print(out.shape)  # Should be (2, 16, 32, 32)

        HINTS:
        - Handle padding by creating padded input array
        - Watch array bounds in inner loops
        - Accumulate products for each output position
        """
        ### BEGIN SOLUTION
        # Input validation and shape extraction
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.shape}")

        batch_size, in_channels, in_height, in_width = x.shape
        out_channels = self.out_channels
        kernel_h, kernel_w = self.kernel_size

        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_w) // self.stride + 1

        # Apply padding if needed
        if self.padding > 0:
            padded_input = np.pad(x.data,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=0)
        else:
            padded_input = x.data

        # Initialize output
        output = np.zeros((batch_size, out_channels, out_height, out_width))

        # Explicit 6-nested loop convolution to show complexity
        for b in range(batch_size):
            for out_ch in range(out_channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Calculate input region for this output position
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        # Accumulate convolution result
                        conv_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    # Get input and weight values
                                    input_val = padded_input[b, in_ch,
                                                           in_h_start + k_h,
                                                           in_w_start + k_w]
                                    weight_val = self.weight.data[out_ch, in_ch, k_h, k_w]

                                    # Accumulate
                                    conv_sum += input_val * weight_val

                        # Store result
                        output[b, out_ch, out_h, out_w] = conv_sum

        # Add bias if present
        if self.bias is not None:
            # Broadcast bias across spatial dimensions
            for out_ch in range(out_channels):
                output[:, out_ch, :, :] += self.bias.data[out_ch]

        return Tensor(output)
        ### END SOLUTION

    def parameters(self):
        """Return trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: Conv2d Implementation
This test validates our convolution implementation with different configurations.
**What we're testing**: Shape preservation, padding, stride effects
**Why it matters**: Convolution is the foundation of computer vision
**Expected**: Correct output shapes and reasonable value ranges
"""

# %% nbgrader={"grade": true, "grade_id": "test-conv2d", "locked": true, "points": 15}


def test_unit_conv2d():
    """🔬 Test Conv2d implementation with multiple configurations."""
    print("🔬 Unit Test: Conv2d...")

    # Test 1: Basic convolution without padding
    print("  Testing basic convolution...")
    conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    x1 = Tensor(np.random.randn(2, 3, 32, 32))
    out1 = conv1(x1)

    expected_h = (32 - 3) + 1  # 30
    expected_w = (32 - 3) + 1  # 30
    assert out1.shape == (2, 16, expected_h, expected_w), f"Expected (2, 16, 30, 30), got {out1.shape}"

    # Test 2: Convolution with padding (same size)
    print("  Testing convolution with padding...")
    conv2 = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    x2 = Tensor(np.random.randn(1, 3, 28, 28))
    out2 = conv2(x2)

    # With padding=1, output should be same size as input
    assert out2.shape == (1, 8, 28, 28), f"Expected (1, 8, 28, 28), got {out2.shape}"

    # Test 3: Convolution with stride
    print("  Testing convolution with stride...")
    conv3 = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2)
    x3 = Tensor(np.random.randn(1, 1, 16, 16))
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
"""
## 🏗️ Pooling Operations - Spatial Dimension Reduction

Pooling operations compress spatial information while keeping the most important features. Think of them as creating "thumbnail summaries" of local regions.

### MaxPool2d: Keeping the Strongest Signals

Max pooling finds the strongest activation in each window, preserving sharp features like edges and corners.

```
MaxPool2d Example (2×2 kernel, stride=2):
Input (4×4):              Windows:               Output (2×2):
┌─────────────┐          ┌─────┬─────┐          ┌───────┐
│ 1  3 │ 2  8 │          │ 1 3 │ 2 8 │          │ 6   8 │
│ 5  6 │ 7  4 │    →     │ 5 6 │ 7 4 │    →     │ 9   7 │
├──────┼──────┤          ├─────┼─────┤          └───────┘
│ 2  9 │ 1  7 │          │ 2 9 │ 1 7 │
│ 0  1 │ 3  6 │          │ 0 1 │ 3 6 │
└─────────────┘          └─────┴─────┘

Window Computations:
Top-left: max(1,3,5,6) = 6     Top-right: max(2,8,7,4) = 8
Bottom-left: max(2,9,0,1) = 9  Bottom-right: max(1,7,3,6) = 7
```

### AvgPool2d: Smoothing Local Features

Average pooling computes the mean of each window, creating smoother, more general features.

```
AvgPool2d Example (same 2×2 kernel, stride=2):
Input (4×4):              Output (2×2):
┌─────────────┐          ┌─────────────┐
│ 1  3 │ 2  8 │          │ 3.75   5.25 │
│ 5  6 │ 7  4 │    →     │ 3.0    4.25 │
├──────┼──────┤          └─────────────┘
│ 2  9 │ 1  7 │
│ 0  1 │ 3  6 │
└─────────────┘

Window Computations:
Top-left: (1+3+5+6)/4 = 3.75    Top-right: (2+8+7+4)/4 = 5.25
Bottom-left: (2+9+0+1)/4 = 3.0  Bottom-right: (1+7+3+6)/4 = 4.25
```

### Why Pooling Matters for Computer Vision

```
Memory Impact:
Input: 224×224×64 = 3.2M values    After 2×2 pooling: 112×112×64 = 0.8M values
Memory reduction: 4× less!         Computation reduction: 4× less!

Information Trade-off:
✅ Preserves important features     ⚠️ Loses fine spatial detail
✅ Provides translation invariance  ⚠️ Reduces localization precision
✅ Reduces overfitting             ⚠️ May lose small objects
```

### Sliding Window Pattern

Both pooling operations follow the same sliding window pattern:

```
Sliding 2×2 window with stride=2:
Step 1:     Step 2:     Step 3:     Step 4:
┌──┐        ┌──┐
│▓▓│        │▓▓│
└──┘        └──┘                   ┌──┐        ┌──┐
                                    │▓▓│        │▓▓│
                                    └──┘        └──┘

Non-overlapping windows → Each input pixel used exactly once
Stride=2 → Output dimensions halved in each direction
```

The key difference: MaxPool takes max(window), AvgPool takes mean(window).
"""

# %% [markdown]
"""
### MaxPool2d Implementation - Preserving Strong Features

MaxPool2d finds the strongest activation in each spatial window, creating a compressed representation that keeps the most important information.

#### Why Max Pooling Works for Computer Vision

```
Edge Detection Example:
Input Window (2×2):         Max Pooling Result:
┌─────┬─────┐
│ 0.1 │ 0.8 │ ←  Strong edge signal
├─────┼─────┤
│ 0.2 │ 0.1 │              Output: 0.8 (preserves edge)
└─────┴─────┘

Noise Reduction Example:
Input Window (2×2):
┌─────┬─────┐
│ 0.9 │ 0.1 │ ←  Feature + noise
├─────┼─────┤
│ 0.2 │ 0.1 │              Output: 0.9 (removes noise)
└─────┴─────┘
```

#### The Sliding Window Pattern

```
MaxPool with 2×2 kernel, stride=2:

Input (4×4):                Output (2×2):
┌───┬───┬───┬───┐          ┌───────┬───────┐
│ a │ b │ c │ d │          │max(a,b│max(c,d│
├───┼───┼───┼───┤     →    │   e,f)│   g,h)│
│ e │ f │ g │ h │          ├───────┼───────┤
├───┼───┼───┼───┤          │max(i,j│max(k,l│
│ i │ j │ k │ l │          │   m,n)│   o,p)│
├───┼───┼───┼───┤          └───────┴───────┘
│ m │ n │ o │ p │
└───┴───┴───┴───┘

Benefits:
✓ Translation invariance (cat moved 1 pixel still detected)
✓ Computational efficiency (4× fewer values to process)
✓ Hierarchical feature building (next layer sees larger receptive field)
```

#### Memory and Computation Impact

For input (1, 64, 224, 224) with 2×2 pooling:
- **Input memory**: 64 × 224 × 224 × 4 bytes = 12.8 MB
- **Output memory**: 64 × 112 × 112 × 4 bytes = 3.2 MB
- **Memory reduction**: 4× less memory needed
- **Computation**: No parameters, minimal compute cost
"""

# %% nbgrader={"grade": false, "grade_id": "maxpool2d-class", "solution": true}

#| export

class MaxPool2d:
    """
    2D Max Pooling layer for spatial dimension reduction.

    Applies maximum operation over spatial windows, preserving
    the strongest activations while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initialize MaxPool2d layer.

        TODO: Store pooling parameters

        APPROACH:
        1. Convert kernel_size to tuple if needed
        2. Set stride to kernel_size if not provided (non-overlapping)
        3. Store padding parameter

        HINT: Default stride equals kernel_size for non-overlapping windows
        """
        super().__init__()

        ### BEGIN SOLUTION
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Default stride equals kernel_size (non-overlapping)
        if stride is None:
            self.stride = self.kernel_size[0]
        else:
            self.stride = stride

        self.padding = padding
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through MaxPool2d layer.

        TODO: Implement max pooling with explicit loops

        APPROACH:
        1. Extract input dimensions
        2. Calculate output dimensions
        3. Apply padding if needed
        4. Implement nested loops for pooling windows
        5. Find maximum value in each window

        LOOP STRUCTURE:
        for batch in range(batch_size):
            for channel in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Find max in window [in_h:in_h+k_h, in_w:in_w+k_w]
                        max_val = -infinity
                        for k_h in range(kernel_height):
                            for k_w in range(kernel_width):
                                max_val = max(max_val, input[...])

        EXAMPLE:
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> x = Tensor(np.random.randn(1, 3, 8, 8))
        >>> out = pool(x)
        >>> print(out.shape)  # Should be (1, 3, 4, 4)

        HINTS:
        - Initialize max_val to negative infinity
        - Handle stride correctly when accessing input
        - No parameters to update (pooling has no weights)
        """
        ### BEGIN SOLUTION
        # Input validation and shape extraction
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.shape}")

        batch_size, channels, in_height, in_width = x.shape
        kernel_h, kernel_w = self.kernel_size

        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_w) // self.stride + 1

        # Apply padding if needed
        if self.padding > 0:
            padded_input = np.pad(x.data,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=-np.inf)
        else:
            padded_input = x.data

        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))

        # Explicit nested loop max pooling
        for b in range(batch_size):
            for c in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Calculate input region for this output position
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

                        # Store result
                        output[b, c, out_h, out_w] = max_val

        return Tensor(output)
        ### END SOLUTION

    def parameters(self):
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
"""
### AvgPool2d Implementation - Smoothing and Generalizing Features

AvgPool2d computes the average of each spatial window, creating smoother features that are less sensitive to noise and exact pixel positions.

#### MaxPool vs AvgPool: Different Philosophies

```
Same Input Window (2×2):    MaxPool Output:    AvgPool Output:
┌─────┬─────┐
│ 0.1 │ 0.9 │               0.9              0.425
├─────┼─────┤              (max)             (mean)
│ 0.3 │ 0.3 │
└─────┴─────┘

Interpretation:
MaxPool: "What's the strongest feature here?"
AvgPool: "What's the general feature level here?"
```

#### When to Use Average Pooling

```
Use Cases:
✓ Global Average Pooling (GAP) for classification
✓ When you want smoother, less noisy features
✓ When exact feature location doesn't matter
✓ In shallower networks where sharp features aren't critical

Typical Pattern:
Feature Maps → Global Average Pool → Dense → Classification
(256×7×7)   →        (256×1×1)      → FC   →    (10)
              Replaces flatten+dense with parameter reduction
```

#### Mathematical Implementation

```
Average Pooling Computation:
Window: [a, b]    Result = (a + b + c + d) / 4
        [c, d]

For efficiency, we:
1. Sum all values in window: window_sum = a + b + c + d
2. Divide by window area: result = window_sum / (kernel_h × kernel_w)
3. Store result at output position

Memory access pattern identical to MaxPool, just different aggregation!
```

#### Practical Considerations

- **Memory**: Same 4× reduction as MaxPool
- **Computation**: Slightly more expensive (sum + divide vs max)
- **Features**: Smoother, more generalized than MaxPool
- **Use**: Often in final layers (Global Average Pooling) to reduce parameters
"""

# %% nbgrader={"grade": false, "grade_id": "avgpool2d-class", "solution": true}

#| export

class AvgPool2d:
    """
    2D Average Pooling layer for spatial dimension reduction.

    Applies average operation over spatial windows, smoothing
    features while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initialize AvgPool2d layer.

        TODO: Store pooling parameters (same as MaxPool2d)

        APPROACH:
        1. Convert kernel_size to tuple if needed
        2. Set stride to kernel_size if not provided
        3. Store padding parameter
        """
        super().__init__()

        ### BEGIN SOLUTION
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Default stride equals kernel_size (non-overlapping)
        if stride is None:
            self.stride = self.kernel_size[0]
        else:
            self.stride = stride

        self.padding = padding
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through AvgPool2d layer.

        TODO: Implement average pooling with explicit loops

        APPROACH:
        1. Similar structure to MaxPool2d
        2. Instead of max, compute average of window
        3. Divide sum by window area for true average

        LOOP STRUCTURE:
        for batch in range(batch_size):
            for channel in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Compute average in window
                        window_sum = 0
                        for k_h in range(kernel_height):
                            for k_w in range(kernel_width):
                                window_sum += input[...]
                        avg_val = window_sum / (kernel_height * kernel_width)

        HINT: Remember to divide by window area to get true average
        """
        ### BEGIN SOLUTION
        # Input validation and shape extraction
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.shape}")

        batch_size, channels, in_height, in_width = x.shape
        kernel_h, kernel_w = self.kernel_size

        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_w) // self.stride + 1

        # Apply padding if needed
        if self.padding > 0:
            padded_input = np.pad(x.data,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=0)
        else:
            padded_input = x.data

        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))

        # Explicit nested loop average pooling
        for b in range(batch_size):
            for c in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Calculate input region for this output position
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        # Compute sum in window
                        window_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                input_val = padded_input[b, c,
                                                       in_h_start + k_h,
                                                       in_w_start + k_w]
                                window_sum += input_val

                        # Compute average
                        avg_val = window_sum / (kernel_h * kernel_w)

                        # Store result
                        output[b, c, out_h, out_w] = avg_val

        # Return Tensor with gradient tracking (consistent with MaxPool2d)
        result = Tensor(output, requires_grad=x.requires_grad)
        return result
        ### END SOLUTION

    def parameters(self):
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
"""
## 🏗️ Batch Normalization - Stabilizing Deep Network Training

Batch Normalization (BatchNorm) is one of the most important techniques for training deep networks. It normalizes activations across the batch dimension, dramatically improving training stability and speed.

### Why BatchNorm Matters

```
Without BatchNorm:                  With BatchNorm:
Layer outputs can have              Layer outputs are normalized
wildly varying scales:              to consistent scale:

Layer 1: mean=0.5, std=0.3         Layer 1: mean≈0, std≈1
Layer 5: mean=12.7, std=8.4   →    Layer 5: mean≈0, std≈1
Layer 10: mean=0.001, std=0.0003   Layer 10: mean≈0, std≈1

Result: Unstable gradients         Result: Stable training
        Slow convergence                   Fast convergence
        Careful learning rate              Robust to hyperparameters
```

### The BatchNorm Computation

For each channel c, BatchNorm computes:
```
1. Batch Statistics (during training):
   μ_c = mean(x[:, c, :, :])     # Mean over batch and spatial dims
   σ²_c = var(x[:, c, :, :])     # Variance over batch and spatial dims

2. Normalize:
   x̂_c = (x[:, c, :, :] - μ_c) / sqrt(σ²_c + ε)

3. Scale and Shift (learnable parameters):
   y_c = γ_c * x̂_c + β_c       # γ (gamma) and β (beta) are learned
```

### Train vs Eval Mode

This is a critical systems concept:

```
Training Mode:                      Eval Mode:
┌────────────────────┐             ┌────────────────────┐
│ Use batch stats    │             │ Use running stats  │
│ Update running     │             │ (accumulated from  │
│ mean/variance      │             │  training)         │
└────────────────────┘             └────────────────────┘
   ↓                                  ↓
Computes μ, σ² from                Uses frozen μ, σ² for
current batch                      consistent inference
```

**Why this matters**: During inference, you might process just 1 image. Batch statistics from 1 sample would be meaningless. Running statistics provide stable normalization.
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

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
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
        super().__init__()

        ### BEGIN SOLUTION
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

    def train(self):
        """Set layer to training mode."""
        self.training = True
        return self

    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False
        return self

    def forward(self, x):
        """
        Forward pass through BatchNorm2d.

        TODO: Implement batch normalization forward pass

        APPROACH:
        1. Validate input shape (must be 4D: batch, channels, height, width)
        2. If training:
           a. Compute batch mean and variance per channel
           b. Normalize using batch statistics
           c. Update running statistics with momentum
        3. If eval:
           a. Use running mean and variance
           b. Normalize using frozen statistics
        4. Apply scale (gamma) and shift (beta)

        EXAMPLE:
        >>> bn = BatchNorm2d(16)
        >>> x = Tensor(np.random.randn(2, 16, 8, 8))  # batch=2, channels=16, 8x8
        >>> y = bn(x)
        >>> print(y.shape)  # (2, 16, 8, 8) - same shape

        HINTS:
        - Compute mean/var over axes (0, 2, 3) to get per-channel statistics
        - Reshape gamma/beta to (1, C, 1, 1) for broadcasting
        - Running stat update: running = (1 - momentum) * running + momentum * batch
        """
        ### BEGIN SOLUTION
        # Input validation
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.shape}")

        batch_size, channels, height, width = x.shape

        if channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels, got {channels}")

        if self.training:
            # Compute batch statistics per channel
            # Mean over batch and spatial dimensions: axes (0, 2, 3)
            batch_mean = np.mean(x.data, axis=(0, 2, 3))  # Shape: (C,)
            batch_var = np.var(x.data, axis=(0, 2, 3))    # Shape: (C,)

            # Update running statistics (exponential moving average)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics (frozen during eval)
            mean = self.running_mean
            var = self.running_var

        # Normalize: (x - mean) / sqrt(var + eps)
        # Reshape mean and var for broadcasting: (C,) -> (1, C, 1, 1)
        mean_reshaped = mean.reshape(1, channels, 1, 1)
        var_reshaped = var.reshape(1, channels, 1, 1)

        x_normalized = (x.data - mean_reshaped) / np.sqrt(var_reshaped + self.eps)

        # Apply scale (gamma) and shift (beta)
        # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
        gamma_reshaped = self.gamma.data.reshape(1, channels, 1, 1)
        beta_reshaped = self.beta.data.reshape(1, channels, 1, 1)

        output = gamma_reshaped * x_normalized + beta_reshaped

        # Return Tensor with gradient tracking
        result = Tensor(output, requires_grad=x.requires_grad or self.gamma.requires_grad)

        return result
        ### END SOLUTION

    def parameters(self):
        """Return learnable parameters (gamma and beta)."""
        return [self.gamma, self.beta]

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: BatchNorm2d
This test validates batch normalization implementation.
**What we're testing**: Normalization behavior, train/eval mode, running statistics
**Why it matters**: BatchNorm is essential for training deep CNNs effectively
**Expected**: Normalized outputs with proper mean/variance characteristics
"""

# %% nbgrader={"grade": true, "grade_id": "test-batchnorm2d", "locked": true, "points": 10}


def test_unit_batchnorm2d():
    """🔬 Test BatchNorm2d implementation."""
    print("🔬 Unit Test: BatchNorm2d...")

    # Test 1: Basic forward pass shape
    print("  Testing basic forward pass...")
    bn = BatchNorm2d(num_features=16)
    x = Tensor(np.random.randn(4, 16, 8, 8))  # batch=4, channels=16, 8x8
    y = bn(x)

    assert y.shape == x.shape, f"Output shape should match input, got {y.shape}"

    # Test 2: Training mode normalization
    print("  Testing training mode normalization...")
    bn2 = BatchNorm2d(num_features=8)
    bn2.train()  # Ensure training mode

    # Create input with known statistics per channel
    x2 = Tensor(np.random.randn(32, 8, 4, 4) * 10 + 5)  # Mean~5, std~10
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
    x3 = Tensor(np.random.randn(16, 8, 4, 4) + 3)  # Offset mean
    _ = bn2(x3)

    # Running mean should have moved toward batch mean
    assert not np.allclose(bn2.running_mean, initial_running_mean), \
        "Running mean should update during training"

    # Test 4: Eval mode uses running statistics
    print("  Testing eval mode behavior...")
    bn3 = BatchNorm2d(num_features=4)

    # Train on some data to establish running stats
    for _ in range(10):
        x_train = Tensor(np.random.randn(8, 4, 4, 4) * 2 + 1)
        _ = bn3(x_train)

    saved_running_mean = bn3.running_mean.copy()
    saved_running_var = bn3.running_var.copy()

    # Switch to eval mode
    bn3.eval()

    # Process different data - running stats should NOT change
    x_eval = Tensor(np.random.randn(2, 4, 4, 4) * 5)  # Different distribution
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
### 🧪 Unit Test: Pooling Operations
This test validates both max and average pooling implementations.
**What we're testing**: Dimension reduction, aggregation correctness
**Why it matters**: Pooling is essential for computational efficiency in CNNs
**Expected**: Correct output shapes and proper value aggregation
"""

# %% nbgrader={"grade": true, "grade_id": "test-pooling", "locked": true, "points": 10}


def test_unit_pooling():
    """🔬 Test MaxPool2d and AvgPool2d implementations."""
    print("🔬 Unit Test: Pooling Operations...")

    # Test 1: MaxPool2d basic functionality
    print("  Testing MaxPool2d...")
    maxpool = MaxPool2d(kernel_size=2, stride=2)
    x1 = Tensor(np.random.randn(1, 3, 8, 8))
    out1 = maxpool(x1)

    expected_shape = (1, 3, 4, 4)  # 8/2 = 4
    assert out1.shape == expected_shape, f"MaxPool expected {expected_shape}, got {out1.shape}"

    # Test 2: AvgPool2d basic functionality
    print("  Testing AvgPool2d...")
    avgpool = AvgPool2d(kernel_size=2, stride=2)
    x2 = Tensor(np.random.randn(2, 16, 16, 16))
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
    x4 = Tensor(np.random.randn(1, 1, 5, 5))
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
"""
## 📊 Systems Analysis - Understanding Spatial Operation Performance

Now let's analyze the computational complexity and memory trade-offs of spatial operations. This analysis reveals why certain design choices matter for real-world performance.

### Key Questions We'll Answer:
1. How does convolution complexity scale with input size and kernel size?
2. What's the memory vs computation trade-off in different approaches?
3. How do modern optimizations (like im2col) change the performance characteristics?
"""

# %% nbgrader={"grade": false, "grade_id": "spatial-analysis", "solution": true}


def analyze_convolution_complexity():
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
        out_ch, k_size = config["conv"][0], config["conv"][1]
        conv = Conv2d(in_ch, out_ch, kernel_size=k_size, padding=k_size//2)

        # Create input tensor
        x = Tensor(np.random.randn(*config["input"]))

        # Calculate theoretical FLOPs
        batch, in_channels, h, w = config["input"]
        out_channels, kernel_size = config["conv"][0], config["conv"][1]

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
    print("🚀 This motivates depthwise separable convolutions and attention mechanisms")

# Analysis will be called in main execution

# %% nbgrader={"grade": false, "grade_id": "pooling-analysis", "solution": true}


def analyze_pooling_effects():
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

# Analysis will be called in main execution

# %% [markdown]
"""
## 🔧 Integration - Building a Complete CNN

Now let's combine convolution and pooling into a complete CNN architecture. You'll see how spatial operations work together to transform raw pixels into meaningful features.

### CNN Architecture: From Pixels to Predictions

A CNN processes images through alternating convolution and pooling layers, gradually extracting higher-level features:

```
Complete CNN Pipeline:

Input Image (32×32×3)     Raw RGB pixels
       ↓
Conv2d(3→16, 3×3)        Detect edges, textures
       ↓
ReLU Activation          Remove negative values
       ↓
MaxPool(2×2)             Reduce to (16×16×16)
       ↓
Conv2d(16→32, 3×3)       Detect shapes, patterns
       ↓
ReLU Activation          Remove negative values
       ↓
MaxPool(2×2)             Reduce to (8×8×32)
       ↓
Flatten                  Reshape to vector (2048,)
       ↓
Linear(2048→10)          Final classification
       ↓
Softmax                  Probability distribution
```

### The Parameter Efficiency Story

```
CNN vs Dense Network Comparison:

CNN Approach:                     Dense Approach:
┌─────────────────┐               ┌─────────────────┐
│ Conv1: 3→16     │               │ Input: 32×32×3  │
│ Params: 448     │               │ = 3,072 values  │
├─────────────────┤               ├─────────────────┤
│ Conv2: 16→32    │               │ Hidden: 1,000   │
│ Params: 4,640   │               │ Params: 3M+     │
├─────────────────┤               ├─────────────────┤
│ Linear: 2048→10 │               │ Output: 10      │
│ Params: 20,490  │               │ Params: 10K     │
└─────────────────┘               └─────────────────┘
Total: ~25K params                Total: ~3M params

CNN wins with 120× fewer parameters!
```

### Spatial Hierarchy: Why This Architecture Works

```
Layer-by-Layer Feature Evolution:

Layer 1 (Conv 3→16):              Layer 2 (Conv 16→32):
┌─────┐ ┌─────┐ ┌─────┐           ┌─────┐ ┌─────┐ ┌─────┐
│Edge │ │Edge │ │Edge │           │Shape│ │Corner│ │Texture│
│ \\ /│ │  |  │ │ / \\│           │ ◇  │ │  L  │ │ ≈≈≈ │
└─────┘ └─────┘ └─────┘           └─────┘ └─────┘ └─────┘
Simple features                   Complex combinations

Why pooling between layers:
✓ Reduces computation for next layer
✓ Increases receptive field (each conv sees larger input area)
✓ Provides translation invariance (cat moved 1 pixel still detected)
```

This hierarchical approach mirrors human vision: we first detect edges, then shapes, then objects!
"""

# %% [markdown]
"""
### SimpleCNN Implementation - Putting It All Together

Now we'll build a complete CNN that demonstrates how convolution and pooling work together. This is your first step from processing individual tensors to understanding complete images!

#### The CNN Architecture Pattern

```
SimpleCNN Architecture Visualization:

Input: (batch, 3, 32, 32)     ← RGB images (CIFAR-10 size)
         ↓
┌─────────────────────────┐
│ Conv2d(3→16, 3×3, p=1)  │    ← Detect edges, textures
│ ReLU()                  │    ← Remove negative values
│ MaxPool(2×2)            │    ← Reduce to (batch, 16, 16, 16)
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ Conv2d(16→32, 3×3, p=1) │   ← Detect shapes, patterns
│ ReLU()                  │   ← Remove negative values
│ MaxPool(2×2)            │   ← Reduce to (batch, 32, 8, 8)
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ Flatten()               │   ← Reshape to (batch, 2048)
│ Linear(2048→10)         │   ← Final classification
└─────────────────────────┘
         ↓
Output: (batch, 10)           ← Class probabilities
```

#### Why This Architecture Works

```
Feature Hierarchy Development:

Layer 1 Features (3→16):     Layer 2 Features (16→32):
┌─────┬─────┬─────┬─────┐   ┌─────┬─────┬─────┬─────┐
│Edge │Edge │Edge │Blob │   │Shape│Corner│Tex-│Pat- │
│ \\  │  |  │ /   │  ○  │   │ ◇   │  L  │ture│tern │
└─────┴─────┴─────┴─────┘   └─────┴─────┴─────┴─────┘
Simple features             Complex combinations

Spatial Dimension Reduction:
32×32 → 16×16 → 8×8
 1024    256     64  (per channel)

Channel Expansion:
3 → 16 → 32
More feature types at each level
```

#### Parameter Efficiency Demonstration

```
CNN vs Dense Comparison for 32×32×3 → 10 classes:

CNN Approach:                    Dense Approach:
┌────────────────────┐          ┌────────────────────┐
│ Conv1: 3→16, 3×3   │          │ Input: 3072 values │
│ Params: 448        │          │        ↓           │
├────────────────────┤          │ Dense: 3072→512    │
│ Conv2: 16→32, 3×3  │          │ Params: 1.57M      │
│ Params: 4,640      │          ├────────────────────┤
├────────────────────┤          │ Dense: 512→10      │
│ Dense: 2048→10     │          │ Params: 5,120      │
│ Params: 20,490     │          └────────────────────┘
└────────────────────┘          Total: 1.58M params
Total: 25,578 params

CNN has 62× fewer parameters while preserving spatial structure!
```

#### Receptive Field Growth

```
How each layer sees progressively larger input regions:

Layer 1 Conv (3×3):           Layer 2 Conv (3×3):
Each output pixel sees        Each output pixel sees
3×3 = 9 input pixels         7×7 = 49 input pixels
                             (due to pooling+conv)

Final Result: Layer 2 can detect complex patterns
spanning 7×7 regions of original image!
```
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

    def __init__(self, num_classes=10):
        """
        Initialize SimpleCNN.

        TODO: Build CNN architecture with spatial and dense layers

        APPROACH:
        1. Conv layer 1: 3 → 16 channels, 3×3 kernel, padding=1
        2. Pool layer 1: 2×2 max pooling
        3. Conv layer 2: 16 → 32 channels, 3×3 kernel, padding=1
        4. Pool layer 2: 2×2 max pooling
        5. Calculate flattened size and add final linear layer

        HINT: For 32×32 input → 32→16→8→4 spatial reduction
        Final feature size: 32 channels × 4×4 = 512 features
        """
        super().__init__()

        ### BEGIN SOLUTION
        # Convolutional layers
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size
        # Input: 32×32 → Conv1+Pool1: 16×16 → Conv2+Pool2: 8×8
        # Wait, let's recalculate: 32×32 → Pool1: 16×16 → Pool2: 8×8
        # Final: 32 channels × 8×8 = 2048 features
        self.flattened_size = 32 * 8 * 8

        # Import Linear layer (we'll implement a simple version)
        # For now, we'll use a placeholder that we can replace
        # This represents the final classification layer
        self.num_classes = num_classes
        self.flattened_size = 32 * 8 * 8  # Will be used when we add Linear layer
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through SimpleCNN.

        TODO: Implement CNN forward pass

        APPROACH:
        1. Apply conv1 → ReLU → pool1
        2. Apply conv2 → ReLU → pool2
        3. Flatten spatial dimensions
        4. Apply final linear layer (when available)

        For now, return features before final linear layer
        since we haven't imported Linear from layers module yet.
        """
        ### BEGIN SOLUTION
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
        x_flat = x.data.reshape(batch_size, -1)

        # Return flattened features
        # In a complete implementation, this would go through a Linear layer
        return Tensor(x_flat)
        ### END SOLUTION

    def relu(self, x):
        """Simple ReLU implementation for CNN."""
        return Tensor(np.maximum(0, x.data))

    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        # Linear layer parameters would be added here
        return params

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: SimpleCNN Integration
This test validates that spatial operations work together in a complete CNN architecture.
**What we're testing**: End-to-end spatial processing pipeline
**Why it matters**: Spatial operations must compose correctly for real CNNs
**Expected**: Proper dimension reduction and feature extraction
"""

# %% nbgrader={"grade": true, "grade_id": "test-simple-cnn", "locked": true, "points": 10}


def test_unit_simple_cnn():
    """🔬 Test SimpleCNN integration with spatial operations."""
    print("🔬 Unit Test: SimpleCNN Integration...")

    # Test 1: Forward pass with CIFAR-10 sized input
    print("  Testing forward pass...")
    model = SimpleCNN(num_classes=10)
    x = Tensor(np.random.randn(2, 3, 32, 32))  # Batch of 2, RGB, 32×32

    features = model(x)

    # Expected: 2 samples, 32 channels × 8×8 spatial = 2048 features
    expected_shape = (2, 2048)
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"

    # Test 2: Parameter counting
    print("  Testing parameter counting...")
    params = model.parameters()

    # Conv1: (16, 3, 3, 3) + bias (16,) = 432 + 16 = 448
    # Conv2: (32, 16, 3, 3) + bias (32,) = 4608 + 32 = 4640
    # Total: 448 + 4640 = 5088 parameters

    conv1_params = 16 * 3 * 3 * 3 + 16  # weights + bias
    conv2_params = 32 * 16 * 3 * 3 + 32  # weights + bias
    expected_total = conv1_params + conv2_params

    actual_total = sum(np.prod(p.shape) for p in params)
    assert actual_total == expected_total, f"Expected {expected_total} parameters, got {actual_total}"

    # Test 3: Different input sizes
    print("  Testing different input sizes...")

    # Test with different spatial dimensions
    x_small = Tensor(np.random.randn(1, 3, 16, 16))
    features_small = model(x_small)

    # 16×16 → 8×8 → 4×4, so 32 × 4×4 = 512 features
    expected_small = (1, 512)
    assert features_small.shape == expected_small, f"Expected {expected_small}, got {features_small.shape}"

    # Test 4: Batch processing
    print("  Testing batch processing...")
    x_batch = Tensor(np.random.randn(8, 3, 32, 32))
    features_batch = model(x_batch)

    expected_batch = (8, 2048)
    assert features_batch.shape == expected_batch, f"Expected {expected_batch}, got {features_batch.shape}"

    print("✅ SimpleCNN integration works correctly!")

if __name__ == "__main__":
    test_unit_simple_cnn()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 15}


def test_module():
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
    test_unit_conv2d()
    test_unit_batchnorm2d()
    test_unit_pooling()
    test_unit_simple_cnn()

    print("\nRunning integration scenarios...")

    # Test realistic CNN workflow with BatchNorm
    print("🔬 Integration Test: Complete CNN pipeline with BatchNorm...")

    # Create a mini CNN for CIFAR-10 with BatchNorm (modern architecture)
    conv1 = Conv2d(3, 8, kernel_size=3, padding=1)
    bn1 = BatchNorm2d(8)
    pool1 = MaxPool2d(2, stride=2)
    conv2 = Conv2d(8, 16, kernel_size=3, padding=1)
    bn2 = BatchNorm2d(16)
    pool2 = AvgPool2d(2, stride=2)

    # Process batch of images (training mode)
    batch_images = Tensor(np.random.randn(4, 3, 32, 32))

    # Forward pass: Conv → BatchNorm → ReLU → Pool (modern pattern)
    x = conv1(batch_images)  # (4, 8, 32, 32)
    x = bn1(x)               # (4, 8, 32, 32) - normalized
    x = Tensor(np.maximum(0, x.data))  # ReLU
    x = pool1(x)             # (4, 8, 16, 16)

    x = conv2(x)             # (4, 16, 16, 16)
    x = bn2(x)               # (4, 16, 16, 16) - normalized
    x = Tensor(np.maximum(0, x.data))  # ReLU
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
    print("🔬 Integration Test: Train/Eval mode switching...")
    bn1.eval()
    bn2.eval()

    # Run inference with single sample (would fail with batch stats)
    single_image = Tensor(np.random.randn(1, 3, 32, 32))
    x = conv1(single_image)
    x = bn1(x)  # Uses running stats, not batch stats
    assert x.shape == (1, 8, 32, 32), f"Single sample inference should work in eval mode"

    print("✅ CNN pipeline with BatchNorm works correctly!")

    # Test memory efficiency comparison
    print("🔬 Integration Test: Memory efficiency analysis...")

    # Compare different pooling strategies (reduced size for faster execution)
    input_data = Tensor(np.random.randn(1, 16, 32, 32))

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

# Run module test when this cell is executed
if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🔧 Main Execution Block

Running all module components including systems analysis and final validation.
"""

# %% nbgrader={"grade": false, "grade_id": "main-execution", "solution": true}

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 09: SPATIAL OPERATIONS - TEST EXECUTION")
    print("=" * 70)

    test_module()

    print("\n" + "="*70)
    print("MODULE 09 TESTS COMPLETE!")
    print("="*70)


# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Before completing this module, reflect on what you've learned about spatial operations and their systems implications:

### Question 1: Conv2d Memory Footprint
A Conv2d layer with 64 filters (3×3) processes a (224×224×3) image.
- Calculate the memory footprint during the forward pass
- Consider: input activations, output activations, filter weights, and biases
- What happens when batch size increases from 1 to 32?

**Think about**: Why do modern vision models use techniques like gradient checkpointing?

### Question 2: Spatial Locality and CPU Performance
Why are CNNs faster on CPUs than fully-connected networks of similar parameter count?

**Consider**:
- Cache locality in convolution operations
- Data reuse patterns in sliding windows
- Memory access patterns (sequential vs random)

**Hint**: Think about what happens when the same filter is applied across the image.

### Question 3: Im2col Trade-off
The im2col algorithm transforms convolution into matrix multiplication, using more memory but speeding up computation.

**When is this trade-off worthwhile?**
- Small vs large batch sizes
- Small vs large images
- Training vs inference
- Mobile vs server deployment

**Think about**: Why don't mobile devices always use im2col?

### Question 4: Pooling's Systems Benefits
MaxPool2d reduces spatial dimensions (e.g., 224×224 → 112×112).

**What's the systems benefit beyond reducing parameters?**
- Memory bandwidth requirements
- Computation in subsequent layers
- Gradient memory during backpropagation
- Cache efficiency in deeper layers

**Calculate**: If 5 layers each use 2×2 pooling, what's the total memory reduction?

### Question 5: Mobile ML Deployment
Why do mobile ML models prefer depthwise-separable convolutions over standard Conv2d?

**Analyze the FLOPs**:
- Standard 3×3 conv: C_in × C_out × H × W × 9
- Depthwise + Pointwise: (C_in × H × W × 9) + (C_in × C_out × H × W)

**When does the trade-off favor depthwise separable?**
- As number of channels increases
- As spatial dimensions change
- Energy consumption vs accuracy

**Real-world context**: This is why MobileNet and EfficientNet architectures exist.

---

**These questions help you think like an ML systems engineer, not just an algorithm implementer.**
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Convolution Extracts Features

**What you built:** Convolutional layers that process spatial data like images.

**Why it matters:** Conv2d looks at local neighborhoods, detecting edges, textures, and patterns.
Unlike Linear layers that see pixels independently, Conv2d understands that nearby pixels are
related. This is why CNNs revolutionized computer vision!

In the milestones, you'll use these spatial operations to build a CNN that recognizes digits.
"""

# %%
def demo_spatial():
    """🎯 See Conv2d process spatial data."""
    print("🎯 AHA MOMENT: Convolution Extracts Features")
    print("=" * 45)

    # Create a simple 8x8 "image" with 1 channel
    image = Tensor(np.random.randn(1, 1, 8, 8))

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
    demo_spatial()

# %% [markdown]
"""
## 🎯 Module Summary

## 🚀 MODULE SUMMARY: Spatial Operations

Congratulations! You've built the spatial processing foundation that powers computer vision!

### Key Accomplishments
- Built Conv2d with explicit loops showing O(N²M²K²) complexity ✅
- Implemented BatchNorm2d with train/eval mode and running statistics ✅
- Implemented MaxPool2d and AvgPool2d for spatial dimension reduction ✅
- Created SimpleCNN demonstrating spatial operation integration ✅
- Analyzed computational complexity and memory trade-offs in spatial processing ✅
- All tests pass including complete CNN pipeline validation ✅

### Systems Insights Discovered
- **Convolution Complexity**: Quadratic scaling with spatial size, kernel size significantly impacts cost
- **Batch Normalization**: Train vs eval mode is critical - batch stats during training, running stats during inference
- **Memory Patterns**: Pooling provides 4× memory reduction while preserving important features
- **Architecture Design**: Strategic spatial reduction enables parameter-efficient feature extraction
- **Cache Performance**: Spatial locality in convolution benefits from optimal memory access patterns

### Ready for Next Steps
Your spatial operations enable building complete CNNs for computer vision tasks!
Export with: `tito module complete 09`

**Next**: Milestone 03 will combine your spatial operations with training pipeline to build a CNN for CIFAR-10!

Your implementation shows why:
- Modern CNNs use small kernels (3×3) instead of large ones (computational efficiency)
- Pooling layers are crucial for managing memory in deep networks (4× reduction per layer)
- Explicit loops reveal the true computational cost hidden by optimized implementations
- Spatial operations unlock computer vision - from MLPs processing vectors to CNNs understanding images!
"""
