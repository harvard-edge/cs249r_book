---
title: "Spatial Operations"
description: "Build CNNs from scratch - implement Conv2d, pooling, and spatial processing for computer vision"
difficulty: "⭐⭐⭐"
time_estimate: "6-8 hours"
prerequisites: ["Tensor", "Activations", "Layers", "DataLoader"]
next_steps: ["Tokenization"]
learning_objectives:
  - "Master memory and computation trade-offs in sliding window convolution operations"
  - "Implement Conv2d layers with weight sharing and understand parameter efficiency vs dense layers"
  - "Design hierarchical feature extraction through stacked convolutional architectures"
  - "Connect spatial operations to PyTorch's torch.nn.Conv2d and understand production CNN implementations"
  - "Analyze receptive field growth, translation invariance, and spatial dimension management"
---

# 09. Spatial Operations

**ARCHITECTURE TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 6-8 hours

## Overview

Implement convolutional neural networks (CNNs) from scratch, building the spatial operations that transformed computer vision from hand-crafted features to learned hierarchical representations. You'll discover why weight sharing revolutionizes computer vision by reducing parameters from millions to thousands while achieving superior spatial reasoning that powers everything from image classification to autonomous driving. This module teaches you how Conv2d achieves massive parameter reduction through weight sharing while enabling the spatial structure understanding critical for modern vision systems.

## Learning Objectives

By the end of this module, you will be able to:

- **Implement Conv2d Forward Pass**: Build sliding window convolution with explicit loops showing O(B×C_out×H×W×K²×C_in) complexity, understanding how weight sharing applies the same learned filter across all spatial positions to detect features like edges and textures
- **Master Weight Sharing Mechanics**: Understand how Conv2d(3→32, kernel=3) uses only 896 parameters while a dense layer for the same 32×32 input needs 32,000 parameters—achieving 35× parameter reduction while preserving spatial structure
- **Design Hierarchical Feature Extractors**: Compose Conv → ReLU → Pool blocks into CNN architectures, learning how depth enables complex feature hierarchies from simple local operations (edges → textures → objects)
- **Build Pooling Operations**: Implement MaxPool2d and AvgPool2d for spatial downsampling, understanding the trade-off between spatial resolution and computational efficiency (4× memory reduction per 2×2 pooling layer)
- **Analyze Receptive Field Growth**: Master how stacked 3×3 convolutions build global context from local operations—two Conv2d layers see 5×5 regions, three layers see 7×7, enabling deep networks to detect large-scale patterns

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement Conv2d with explicit sliding window loops to expose computational complexity, create MaxPool2d and AvgPool2d for spatial downsampling, and build Flatten operations connecting spatial and dense layers for complete CNN architectures
2. **Use**: Train CNNs on CIFAR-10 (60K 32×32 color images) to achieve >75% accuracy, visualize learned feature maps showing edges in early layers and complex patterns in deep layers, and compare CNN vs MLP parameter efficiency on spatial data
3. **Reflect**: Analyze why weight sharing reduces parameters by 35-1000× while improving spatial reasoning, how stacked 3×3 convolutions build global context from local receptive fields, and what memory-computation trade-offs exist between large kernels vs deep stacking

## Implementation Guide

### Convolutional Pipeline Flow

Convolution transforms spatial data through learnable filters, pooling, and hierarchical feature extraction:

```{mermaid}
graph LR
    A[Input Image<br/>H×W×C] --> B[Conv2d<br/>k×k filters]
    B --> C[Feature Maps<br/>H'×W'×F]
    C --> D[Activation<br/>ReLU]
    D --> E[Pool 2×2<br/>Downsample]
    E --> F[Output<br/>H'/2×W'/2×F]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#ffe0b2
    style E fill:#fce4ec
    style F fill:#f0fdf4
```

**Flow**: Image → Convolution (weight sharing) → Feature maps → Nonlinearity → Pooling → Downsampled features

### Conv2d Layer - The Heart of Computer Vision

```python
class Conv2d:
    """
    2D Convolutional layer with learnable filters and weight sharing.

    Implements sliding window convolution where the same learned filter
    applies across all spatial positions, achieving massive parameter
    reduction compared to dense layers while preserving spatial structure.

    Key Concepts:
    - Weight sharing: Same filter at all spatial positions
    - Local connectivity: Each output depends on local input region
    - Learnable filters: Each filter learns to detect different features
    - Translation invariance: Detected features independent of position

    Args:
        in_channels: Number of input channels (3 for RGB, 16 for feature maps)
        out_channels: Number of learned filters (feature detectors)
        kernel_size: Spatial size of sliding window (typically 3 or 5)
        stride: Step size when sliding (1 = no downsampling)
        padding: Border padding to preserve spatial dimensions

    Shape:
        Input: (batch, in_channels, height, width)
        Output: (batch, out_channels, out_height, out_width)
        Where: out_height = (height + 2*padding - kernel_size) // stride + 1
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        # Initialize learnable filters: one per output channel
        # Shape: (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = Tensor(shape=(out_channels, in_channels, kernel_size, kernel_size))

        # He initialization for ReLU networks
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weight.data = np.random.normal(0, std, self.weight.shape)

    def forward(self, x):
        """Apply sliding window convolution with explicit loops to show cost."""
        batch, _, H, W = x.shape
        out_h = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        # Apply padding if needed
        if self.padding > 0:
            x = pad(x, self.padding)

        output = Tensor(shape=(batch, self.out_channels, out_h, out_w))

        # Explicit 7-nested loop showing O(B×C_out×H×W×K_h×K_w×C_in) complexity
        for b in range(batch):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Extract local patch from input
                        i_start = i * self.stride
                        j_start = j * self.stride
                        patch = x[b, :, i_start:i_start+self.kernel_size,
                                       j_start:j_start+self.kernel_size]

                        # Convolution: dot product between filter and patch
                        output.data[b, oc, i, j] = (patch.data * self.weight.data[oc]).sum()

        return output
```

**Why Explicit Loops Matter**: Modern frameworks optimize convolution with im2col transformations and cuDNN kernels, achieving 10-100× speedups. But the explicit loops reveal where computational cost lives—helping you understand why kernel size matters enormously and why production systems carefully balance depth vs width.

### MaxPool2d - Spatial Downsampling and Translation Invariance

```python
class MaxPool2d:
    """
    Max pooling for spatial downsampling and translation invariance.

    Extracts maximum value from each local region, providing:
    - Spatial dimension reduction (4× memory reduction per 2×2 pooling)
    - Translation invariance (robustness to small shifts)
    - Feature importance selection (keep strongest activations)

    Args:
        kernel_size: Size of pooling window (typically 2)
        stride: Step size when sliding (defaults to kernel_size)

    Shape:
        Input: (batch, channels, height, width)
        Output: (batch, channels, out_height, out_width)
        Where: out_height = (height - kernel_size) // stride + 1
    """
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        """Extract maximum value from each local region."""
        batch, channels, H, W = x.shape
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        output = Tensor(shape=(batch, channels, out_h, out_w))

        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        i_start = i * self.stride
                        j_start = j * self.stride
                        patch = x.data[b, c, i_start:i_start+self.kernel_size,
                                             j_start:j_start+self.kernel_size]
                        output.data[b, c, i, j] = patch.max()

        return output
```

**MaxPool vs AvgPool**: MaxPool preserves sharp features like edges (takes max activation), while AvgPool creates smoother features (averages the window). Production systems typically use MaxPool for feature extraction and Global Average Pooling for final classification layers.

### SimpleCNN - Complete Architecture

```python
class SimpleCNN:
    """
    Complete CNN for CIFAR-10 image classification.

    Architecture: Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense

    Layer-by-layer transformation:
        Input: (B, 3, 32, 32) RGB images
        Conv1: (B, 32, 32, 32) - 32 filters detect edges/textures
        Pool1: (B, 32, 16, 16) - downsample by 2×
        Conv2: (B, 64, 16, 16) - 64 filters detect shapes/patterns
        Pool2: (B, 64, 8, 8) - downsample by 2×
        Flatten: (B, 4096) - convert spatial to vector
        Dense: (B, 10) - classify into 10 categories

    Parameters: ~500K (vs ~4M for equivalent dense network)
    """
    def __init__(self):
        # Feature extraction backbone
        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2)

        # Classification head
        self.flatten = Flatten()
        self.fc = Linear(64 * 8 * 8, 10)

    def forward(self, x):
        # Hierarchical feature extraction
        x = self.pool1(relu(self.conv1(x)))  # (B, 32, 16, 16)
        x = self.pool2(relu(self.conv2(x)))  # (B, 64, 8, 8)

        # Classification
        x = self.flatten(x)  # (B, 4096)
        x = self.fc(x)       # (B, 10)
        return x
```

**Architecture Design Principles**: This follows the standard CNN pattern—alternating Conv+ReLU (feature extraction) with Pooling (dimension reduction). Each Conv layer learns hierarchical features (Layer 1: edges → Layer 2: shapes), while pooling provides computational efficiency and translation invariance.

## Getting Started

### Prerequisites

Ensure you understand the foundations from previous modules:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules are complete
tito test tensor      # Module 01: Tensor operations
tito test activations # Module 02: ReLU activation
tito test layers      # Module 03: Linear layers
tito test dataloader  # Module 08: Batch loading
```

**Why These Prerequisites**:
- **Tensor**: Conv2d requires tensor indexing, reshaping, and broadcasting for sliding windows
- **Activations**: CNNs use ReLU after each convolution for non-linear feature learning
- **Layers**: Dense classification layers connect to CNN feature extraction
- **DataLoader**: CIFAR-10 training requires batch loading and data augmentation

### Development Workflow

1. **Open the development file**: `modules/09_spatial/spatial_dev.py`
2. **Implement Conv2d forward pass**: Build sliding window convolution with explicit loops showing computational complexity
3. **Create MaxPool2d and AvgPool2d**: Implement spatial downsampling with different aggregation strategies
4. **Build Flatten operation**: Connect spatial feature maps to dense layers
5. **Design SimpleCNN architecture**: Compose spatial and dense layers into complete CNN
6. **Export and verify**: `tito module complete 09 && tito test spatial`

**Development Tips**:
- Start with small inputs (8×8 images) to debug convolution logic before scaling to 32×32
- Print intermediate shapes at each layer to verify dimension calculations
- Visualize feature maps after Conv layers to understand learned filters
- Compare parameter counts: Conv2d(3→32, k=3) = 896 params vs Dense(3072→32) = 98,304 params

## Testing

### Comprehensive Test Suite

Run the full test suite to verify spatial operation functionality:

```bash
# TinyTorch CLI (recommended)
tito test spatial

# Direct pytest execution
python -m pytest tests/ -k spatial -v
```

### Test Coverage Areas

- ✅ **Conv2d Shape Propagation**: Verifies output dimensions match formula (H+2P-K)//S+1 for various kernel sizes, strides, and padding
- ✅ **Weight Sharing Validation**: Confirms same filter applies at all spatial positions, achieving parameter reduction vs dense layers
- ✅ **Pooling Correctness**: Tests MaxPool extracts maximum values and AvgPool computes correct averages across windows
- ✅ **Translation Invariance**: Verifies CNNs detect features regardless of spatial position through weight sharing
- ✅ **Complete CNN Pipeline**: End-to-end test processing CIFAR-10 images through Conv → Pool → Flatten → Dense architecture

### Inline Testing & Validation

The module includes comprehensive inline tests during development:

```python
# Run inline unit tests
cd /Users/VJ/GitHub/TinyTorch/modules/09_spatial
python spatial_dev.py

# Expected output:
🔬 Unit Test: Conv2d...
✅ Sliding window convolution works correctly
✅ Weight sharing applied at all positions
✅ Output shape matches calculated dimensions
✅ Parameter count: 896 (vs 32,000 for dense layer)
📈 Progress: Conv2d forward pass implemented

🔬 Unit Test: Pooling Operations...
✅ MaxPool2d extracts maximum values correctly
✅ AvgPool2d computes averages correctly
✅ Spatial dimensions reduced by factor of kernel_size
✅ Translation invariance property verified
📈 Progress: Pooling layers implemented

🔬 Unit Test: SimpleCNN Integration...
✅ Forward pass through all layers successful
✅ Output shape: (32, 10) for 10 CIFAR-10 classes
✅ Total parameters: ~500K (efficient!)
📈 Progress: CNN architecture complete
```

### Manual Testing Examples

Test individual components interactively:

```python
from spatial_dev import Conv2d, MaxPool2d, SimpleCNN
import numpy as np

# Test Conv2d with small input
conv = Conv2d(3, 16, kernel_size=3, padding=1)
x = Tensor(np.random.randn(2, 3, 8, 8))
out = conv(x)
print(f"Conv2d output shape: {out.shape}")  # (2, 16, 8, 8)

# Test MaxPool dimension reduction
pool = MaxPool2d(kernel_size=2)
pooled = pool(out)
print(f"MaxPool output shape: {pooled.shape}")  # (2, 16, 4, 4)

# Test complete CNN
cnn = SimpleCNN(num_classes=10)
img = Tensor(np.random.randn(4, 3, 32, 32))
logits = cnn(img)
print(f"CNN output shape: {logits.shape}")  # (4, 10)

# Count parameters
params = cnn.parameters()
total = sum(np.prod(p.shape) for p in params)
print(f"Total parameters: {total:,}")  # ~500,000
```

## Systems Thinking Questions

### Real-World Applications

**Autonomous Driving - Tesla Autopilot**

**Challenge**: Tesla's Autopilot processes 8 cameras at 36 FPS with 1280×960 resolution, running CNN backbones to extract features for object detection, lane recognition, and depth estimation. The entire inference must complete in <30ms for real-time control.

**Solution**: Efficient CNN architectures (MobileNet-style depthwise separable convolutions) and aggressive optimization (TensorRT compilation, INT8 quantization) balance accuracy vs latency on embedded hardware (Tesla FSD computer: 144 TOPS).

**Your Implementation Connection**: Understanding Conv2d's computational cost (K²×C_in×C_out×H×W operations) reveals why Tesla optimizes kernel sizes and channel counts carefully—every operation matters at 36 FPS × 8 cameras = 288 frames/second total processing.

**Medical Imaging - Diagnostic Assistance**

**Challenge**: CNN systems analyze X-rays, CT scans, and pathology slides for diagnostic assistance. PathAI's breast cancer detection achieves 97% sensitivity (vs 92% for individual pathologists) by training deep CNNs on millions of annotated slides. Medical deployment requires interpretability—doctors need to understand why the CNN made a prediction.

**Solution**: Visualizing intermediate feature maps and using attention mechanisms to highlight diagnostic regions. Grad-CAM (Gradient-weighted Class Activation Mapping) shows which spatial regions contributed most to the prediction.

**Your Implementation Connection**: Your Conv2d's feature maps can be visualized showing which spatial regions activate strongly for different filters. This interpretability is crucial for medical deployment where "black box" predictions are insufficient for clinical decisions.

**Face Recognition - Apple Face ID**

**Challenge**: Apple's Face ID uses CNNs to generate face embeddings enabling secure device unlock with <1 in 1,000,000 false accept rate. The entire pipeline (detection + alignment + embedding + matching) runs on-device in real-time. Privacy requires on-device processing, demanding lightweight CNN architectures.

**Solution**: MobileNet-style CNNs with depthwise separable convolutions reduce parameters by 8-10× while maintaining accuracy. The entire model fits in <10MB, enabling on-device execution protecting user privacy.

**Your Implementation Connection**: Understanding Conv2d's parameter count (C_out×C_in×K²) reveals why face recognition systems carefully design CNN architectures—fewer parameters enable on-device deployment without sacrificing accuracy.

**Historical Impact - AlexNet to ResNet**

**LeNet-5 (1998)**: Yann LeCun's CNN successfully read handwritten zip codes for the US Postal Service, establishing the Conv → Pool → Conv → Pool → Dense pattern your SimpleCNN follows. Training took days on CPUs, limiting practical deployment.

**AlexNet (2012)**: Won ImageNet with 16% error (vs 26% for hand-crafted features), sparking the deep learning revolution. Key innovation: training deep CNNs on GPUs with massive datasets proved that scale + convolution = breakthrough performance.

**VGG (2014)**: Demonstrated that deeper CNNs with simple 3×3 kernels outperform shallow networks with large kernels. Established that stacking many small convolutions beats few large ones—the computational trade-off analysis below.

**ResNet (2015)**: 152-layer CNN achieved 3.6% ImageNet error (better than human 5% baseline) via skip connections solving vanishing gradients. Your Conv2d is the foundation—ResNet is "just" your layers with residual connections enabling extreme depth.

### Foundations

**Weight Sharing and Parameter Efficiency**

**Question**: A Conv2d(3, 32, kernel_size=3) layer has 32 filters × (3 channels × 3×3 spatial) = 896 parameters. For a 32×32 RGB image, a dense layer producing 32 feature maps of the same resolution needs (3×32×32) × (32×32×32) = 3,072 × 32,768 = ~100 million parameters. Why does convolution reduce parameters by 100,000×? How does weight sharing enable this dramatic reduction? What spatial assumption does convolution make that dense layers don't—and when might this assumption break?

**Key Insights**:
- **Weight Sharing**: Conv2d applies the same 3×3×3 filter at all 32×32 = 1,024 positions, sharing 896 parameters across 1,024 locations. Dense layers learn independent weights for each position.
- **Local Connectivity**: Each conv output depends only on a local 3×3 neighborhood, not the entire image. This inductive bias reduces parameters but assumes nearby pixels are more related than distant ones.
- **When It Breaks**: For tasks where spatial relationships don't follow local patterns (e.g., finding relationships between distant objects), convolution's local connectivity limits expressiveness. This motivates attention mechanisms in Vision Transformers.

**Translation Invariance Through Weight Sharing**

**Question**: A CNN detects a cat regardless of whether it appears in the top-left or bottom-right corner of an image. A dense network trained on top-left cats fails on bottom-right cats. How does weight sharing enable translation invariance? Why does applying the same filter at all spatial positions make detected features position-independent? What's the trade-off: what spatial information does convolution lose by treating all positions equally?

**Key Insights**:
- **Same Filter Everywhere**: Weight sharing means the "cat ear detector" filter slides across the entire image, detecting ears wherever they appear. Dense layers have position-specific weights that don't generalize spatially.
- **Pooling Enhances Invariance**: MaxPool further increases invariance—if the cat moves 1 pixel, the max in each 2×2 window often stays the same, making predictions robust to small shifts.
- **Trade-off**: Convolution loses absolute position information. For tasks requiring precise localization (e.g., object detection), networks must add position embeddings or specialized heads to recover spatial coordinates.

**Hierarchical Feature Learning**

**Question**: Early CNN layers (Conv1) learn to detect edges and simple textures. Deep layers (Conv5) detect complex objects like faces and cars. This feature hierarchy emerges automatically from stacking convolutions—it's not explicitly programmed. How do stacked convolutions build hierarchical representations from local operations? Why don't deep dense networks show this hierarchical organization? What role does the receptive field (the input region affecting each output) play in hierarchical learning?

**Key Insights**:
- **Receptive Field Growth**: A single 3×3 conv sees 9 pixels. Two stacked 3×3 convs see 5×5 (25 pixels). Three see 7×7 (49 pixels). Deeper layers see larger input regions, enabling detection of larger patterns.
- **Compositional Learning**: Early layers learn simple features (edges). Middle layers combine edges into textures and corners. Deep layers combine textures into object parts (eyes, wheels), then complete objects.
- **Why Dense Doesn't**: Dense layers lack spatial structure—each neuron connects to all inputs equally. Without spatial inductive bias (local connectivity + weight sharing), dense networks don't naturally learn hierarchical spatial features.

### Characteristics

**Receptive Field Growth and Global Context**

**Question**: A single Conv2d(kernel_size=3) sees a 3×3 region. Two stacked Conv2d layers see a 5×5 region (center of second layer sees 3×3 of first layer, which each see 3×3 of input). Three layers see 7×7. How many Conv2d(kernel_size=3) layers are needed to see an entire 32×32 image? How do deep CNNs build global context from local operations? What's the trade-off: why not use one large Conv2d(kernel_size=32) instead of stacking many small kernels?

**Key Insights**:
- **Receptive Field Formula**: For N layers with kernel size K, receptive field = 1 + N×(K-1). For K=3: RF = 1+2N. To cover 32×32 requires RF ≥ 32, so N ≥ 15.5 → need 16 Conv2d(3×3) layers.
- **Stacking Benefits**: Three Conv2d(3×3) layers have 3×(C²×9) = 27C² parameters and 3 ReLU nonlinearities. One Conv2d(7×7) has C²×49 parameters and 1 ReLU. Stacking provides parameter efficiency and more non-linear transformations for the same receptive field.
- **Trade-off**: Deeper stacking increases computational cost (more layers to process) and training difficulty (vanishing gradients). But gains from parameter efficiency and expressiveness typically outweigh costs—hence VGG's success with stacked 3×3 convs vs AlexNet's large kernels.

**Computational Cost and Optimization Strategies**

**Question**: A Conv2d(64→64, kernel_size=7) has 64×64×7×7 = 200K parameters and processes (64×7×7) = 3,136 operations per output pixel. Three stacked Conv2d(64→64, kernel_size=3) have 3×(64×64×3×3) = 110K parameters but perform 3×(64×3×3) = 1,728 operations per output pixel at each of 3 layers. Which is better for parameter efficiency? For computational cost? For feature learning? Why did the field shift from AlexNet's 11×11 kernels to VGG/ResNet's 3×3 stacks?

**Key Insights**:
- **Parameter Efficiency**: Stacked 3×3 (110K params) beats single 7×7 (200K params) by 1.8×.
- **Computational Cost**: Stacked approach performs 3×1,728 = 5,184 ops per output pixel vs 3,136 for single 7×7. Stacking costs 1.65× more computation.
- **Feature Learning**: Stacking provides 3 ReLU nonlinearities vs 1, enabling more complex feature transformations. The expressiveness gain from depth outweighs the 1.65× compute cost.
- **Modern Practice**: VGG established that stacked 3×3 convs outperform large kernels. ResNet, EfficientNet, and modern architectures all use 3×3 (or 1×1 for channel mixing) due to better parameter-computation-expressiveness trade-off.

## Ready to Build?

You're about to implement the spatial operations that revolutionized how machines see. Before deep learning, computer vision relied on hand-crafted features like SIFT and HOG—human experts manually designed algorithms to detect edges, corners, and textures. AlexNet's 2012 ImageNet victory proved that learned convolutional features outperform hand-crafted ones, launching the deep learning revolution. Today, CNNs process billions of images daily across Meta's photo tagging (2B photos/day), Tesla's Autopilot (real-time multi-camera processing), and Google Photos (trillion+ image search).

The Conv2d operations you'll implement aren't just educational exercises—they're the same patterns powering production vision systems. Your sliding window convolution reveals why kernel size matters enormously (7×7 kernels cost 5.4× more than 3×3) and why weight sharing enables CNNs to learn from spatial data 100× more efficiently than dense networks. The explicit loops expose computational costs that modern frameworks hide with im2col transformations and cuDNN kernels—understanding the naive implementation reveals where optimizations matter most.

By building CNNs from first principles, you'll understand not just how convolution works, but why it works—why weight sharing provides translation invariance, how stacked small kernels build global context from local operations, and what memory-computation trade-offs govern architecture design. These insights prepare you to design efficient CNN architectures for resource-constrained deployment (mobile, edge devices) and to debug performance bottlenecks in production systems.

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/09_spatial/spatial_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/09_spatial/spatial_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/09_spatial/spatial_dev.ipynb
:class-header: bg-light

Browse the Jupyter notebook source and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.
```

**Local Development**:
```bash
cd /Users/VJ/GitHub/TinyTorch/modules/09_spatial
python spatial_dev.py  # Run inline tests
tito module complete 09  # Export to package
```

---

<div class="prev-next-area">
<a class="left-prev" href="../08_dataloader/ABOUT.html" title="previous page">← Module 08: DataLoader</a>
<a class="right-next" href="../10_tokenization/ABOUT.html" title="next page">Module 10: Tokenization →</a>
</div>
