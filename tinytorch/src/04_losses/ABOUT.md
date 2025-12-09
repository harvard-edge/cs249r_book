---
title: "Loss Functions"
description: "Build MSE and CrossEntropy loss functions with numerical stability for regression and classification"
difficulty: "⭐⭐ (2/4)"
time_estimate: "3-4 hours"
prerequisites: ["01_tensor", "02_activations", "03_layers"]
next_steps: ["05_autograd"]
learning_objectives:
  - "Understand loss function memory allocation patterns and computational costs"
  - "Implement MSE and CrossEntropy losses with proper numerical stability"
  - "Master the log-sum-exp trick and its role in preventing overflow"
  - "Connect loss implementations to PyTorch/TensorFlow loss APIs"
  - "Analyze gradient flow and scaling trade-offs in loss computation"
---

# 04. Loss Functions

**FOUNDATION TIER** | Difficulty: ⭐⭐ (2/4) | Time: 3-4 hours

## Overview

Loss functions are the mathematical conscience of machine learning. They quantify prediction error and provide the scalar signal that drives perf. This module implements MSE for regression and CrossEntropy for classification, with careful attention to numerical stability through the log-sum-exp trick. You'll build the feedback mechanisms used in billions of training runs across GPT models, ResNets, and all production ML systems.

## Learning Objectives

By the end of this module, you will be able to:

- **Implement MSE Loss**: Build mean squared error with proper reduction strategies and understand memory/compute costs
- **Build CrossEntropy Loss**: Create numerically stable classification loss using log-sum-exp trick to prevent overflow
- **Master Numerical Stability**: Understand why naive implementations fail with large logits and implement production-grade solutions
- **Analyze Memory Patterns**: Compute loss function memory footprints across batch sizes and vocabulary dimensions
- **Connect to Frameworks**: Understand how PyTorch's `nn.MSELoss` and `nn.CrossEntropyLoss` implement these same concepts

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement MSE and CrossEntropy loss functions with the log-sum-exp trick for numerical stability
2. **Use**: Apply losses to regression (house prices) and classification (image recognition) problems
3. **Reflect**: Why does CrossEntropy overflow without log-sum-exp? How does loss scale affect gradient magnitudes?

## Getting Started

### Prerequisites
Ensure you understand the foundations from previous modules:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules
tito test tensor
tito test activations
tito test layers
```

### Development Workflow
1. **Open the development file**: `modules/04_losses/losses_dev.ipynb`
2. **Implement log_softmax**: Build numerically stable log-softmax with log-sum-exp trick
3. **Build MSELoss**: Create regression loss with proper reduction
4. **Create CrossEntropyLoss**: Implement classification loss using stable log-softmax
5. **Add BinaryCrossEntropyLoss**: Build binary classification loss with clamping
6. **Export and verify**: `tito module complete 04 && tito test losses`

## Implementation Guide

### MSELoss - Regression Loss

Mean Squared Error is the foundation of regression problems. It measures the average squared distance between predictions and targets, creating a quadratic penalty that grows rapidly with prediction error.

```python
class MSELoss:
    """Mean Squared Error for regression tasks."""

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Compute: (1/n) * Σ(predictions - targets)²
        diff = predictions.data - targets.data
        squared_diff = diff ** 2
        return Tensor(np.mean(squared_diff))
```

**Key Properties**:
- **Quadratic penalty**: error of 2 → loss of 4, error of 10 → loss of 100
- **Outlier sensitivity**: Large errors dominate the loss landscape
- **Smooth gradients**: Differentiable everywhere, nice optimization properties
- **Memory footprint**: ~2 × batch_size × output_dim for intermediate storage

**Mathematical Foundation**: MSE derives from maximum likelihood estimation under Gaussian noise. When you assume prediction errors are normally distributed, minimizing MSE is equivalent to maximizing the likelihood of observing your data.

**Use Cases**: House price prediction, temperature forecasting, stock price regression, image reconstruction in autoencoders, and any continuous value prediction where quadratic error makes sense.

### Log-Softmax with Numerical Stability

Before implementing CrossEntropy, we need a numerically stable way to compute log-softmax. This is the critical building block that prevents overflow in classification losses.

```python
def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable log-softmax using log-sum-exp trick."""
    # Step 1: Subtract max for stability
    max_vals = np.max(x.data, axis=dim, keepdims=True)
    shifted = x.data - max_vals

    # Step 2: Compute log(sum(exp(shifted)))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # Step 3: Return log-softmax
    return Tensor(x.data - max_vals - log_sum_exp)
```

**Why Log-Sum-Exp Matters**:
```
Without trick: exp(1000) = overflow (inf)
With trick: exp(1000 - 1000) = exp(0) = 1.0 ✓
```

**The Mathematics**: Computing `log(Σ exp(xi))` directly causes overflow when logits are large. The log-sum-exp trick factors out the maximum value: `log(Σ exp(xi)) = max(x) + log(Σ exp(xi - max(x)))`. This shifts all exponents into a safe range (≤ 0) before computing exp, preventing overflow while maintaining mathematical equivalence.

**Production Reality**: This exact technique is used in PyTorch's `F.log_softmax`, TensorFlow's `tf.nn.log_softmax`, and JAX's `jax.nn.log_softmax`. It's not an educational simplification—it's production-critical numerical stability.

### CrossEntropyLoss - Classification Loss

CrossEntropy is the standard loss for multi-class classification. It measures how well predicted probability distributions match true class labels, providing strong gradients for confident wrong predictions and gentle gradients for confident correct predictions.

```python
class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Step 1: Compute log-softmax (stable)
        log_probs = log_softmax(logits, dim=-1)

        # Step 2: Select correct class log-probabilities
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

        # Step 3: Return negative mean
        return Tensor(-np.mean(selected_log_probs))
```

**Gradient Behavior**:
- **Confident and correct**: Small gradient (model is right, minimal updates needed)
- **Confident and wrong**: Large gradient (urgent correction signal)
- **Uncertain predictions**: Medium gradient (encourages confidence when correct)
- **Natural confidence weighting**: The loss automatically provides stronger signals when the model needs to change

**Why It Works**: CrossEntropy derives from maximum likelihood estimation under a categorical distribution. Minimizing CrossEntropy is equivalent to maximizing the probability the model assigns to the correct class. The logarithm transforms products into sums (computationally stable) and creates the characteristic gradient behavior.

### BinaryCrossEntropyLoss - Binary Classification

Binary CrossEntropy is specialized for two-class problems. It's more efficient than full CrossEntropy for binary decisions and provides symmetric treatment of positive and negative classes.

```python
class BinaryCrossEntropyLoss:
    """Binary cross-entropy for yes/no decisions."""

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Clamp to prevent log(0)
        eps = 1e-7
        clamped = np.clip(predictions.data, eps, 1 - eps)

        # BCE = -(y*log(p) + (1-y)*log(1-p))
        return Tensor(-np.mean(
            targets.data * np.log(clamped) +
            (1 - targets.data) * np.log(1 - clamped)
        ))
```

**Numerical Stability**: The epsilon clamping (`1e-7` to `1-1e-7`) prevents `log(0)` which would produce `-inf`. This is critical for binary classification where predictions can approach 0 or 1.

**Use Cases**: Spam detection (spam vs not spam), medical diagnosis (disease vs healthy), fraud detection (fraud vs legitimate), content moderation (toxic vs safe), and any yes/no decision problem where both classes matter equally.

## Common Pitfalls

### Numerical Overflow in Softmax/CrossEntropy

**Problem**: Computing `exp(x)` for large logits (e.g., x=1000) causes overflow (inf), leading to NaN losses.

**Solution**: Use the log-sum-exp trick by subtracting the maximum value before exponentiation:

```python
# ❌ Wrong - causes overflow
exp_vals = np.exp(logits)
softmax = exp_vals / np.sum(exp_vals)
log_softmax = np.log(softmax)  # Can produce inf/NaN

# ✅ Correct - numerically stable
max_val = np.max(logits, axis=-1, keepdims=True)
shifted = logits - max_val
log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
log_softmax = logits - max_val - log_sum_exp
```

### Binary CrossEntropy log(0) Error

**Problem**: When predictions are exactly 0 or 1, `log(0) = -inf` causes NaN losses.

**Solution**: Clamp predictions to a small epsilon range [ε, 1-ε]:

```python
# ❌ Wrong - log(0) produces -inf
loss = -np.mean(targets * np.log(predictions) +
                (1 - targets) * np.log(1 - predictions))

# ✅ Correct - epsilon clamping prevents log(0)
eps = 1e-7
clamped = np.clip(predictions, eps, 1 - eps)
loss = -np.mean(targets * np.log(clamped) +
                (1 - targets) * np.log(1 - clamped))
```

### Incorrect Target Format for CrossEntropy

**Problem**: Using one-hot encoded targets instead of class indices causes dimension mismatches.

**Solution**: CrossEntropy expects integer class indices, not one-hot vectors:

```python
# ❌ Wrong - one-hot targets
targets = Tensor([[0, 1, 0], [1, 0, 0]])  # One-hot (batch, classes)
loss = CrossEntropyLoss()(logits, targets)  # Error!

# ✅ Correct - class indices
targets = Tensor([1, 0])  # Class indices (batch,)
loss = CrossEntropyLoss()(logits, targets)
```

### Reduction Strategy Confusion

**Problem**: Using `sum` reduction instead of `mean` causes loss values to scale with batch size, making learning rate tuning difficult.

**Solution**: Always use `mean` reduction for batch-size-independent gradients:

```python
# ❌ Wrong - sum scales with batch size
# Batch 32: loss=320, Batch 64: loss=640 (same data, different scale)
loss = np.sum((predictions - targets) ** 2)

# ✅ Correct - mean is batch-size independent
# Both batches: loss=10 (consistent scale)
loss = np.mean((predictions - targets) ** 2)
```

### MSE for Classification

**Problem**: Using MSE loss for classification problems leads to poor gradient behavior and slower convergence.

**Solution**: Use CrossEntropy for classification (discrete labels), MSE for regression (continuous values):

```python
# ❌ Wrong - MSE for classification
# Gradients don't push confidently wrong predictions hard enough
loss = MSELoss()(class_logits, class_labels)

# ✅ Correct - CrossEntropy for classification
# Provides strong gradients for confidently wrong predictions
loss = CrossEntropyLoss()(class_logits, class_labels)
```

## Testing

### Comprehensive Test Suite
Run the full test suite to verify loss functionality:

```bash
# TinyTorch CLI (recommended)
tito test losses

# Direct pytest execution
python -m pytest tests/ -k losses -v
```

### Test Coverage Areas
- ✅ **MSE Correctness**: Validates known cases, perfect predictions (loss=0), non-negativity
- ✅ **CrossEntropy Stability**: Tests large logits (1000+), verifies no overflow/underflow
- ✅ **Gradient Properties**: Ensures CrossEntropy gradient equals softmax - target
- ✅ **Binary Classification**: Validates BCE with boundary cases and probability constraints
- ✅ **Log-Sum-Exp Trick**: Confirms numerical stability with extreme values

### Inline Testing & Validation
The module includes comprehensive unit tests:
```python
🔬 Unit Test: Log-Softmax...
✅ log_softmax works correctly with numerical stability!

🔬 Unit Test: MSE Loss...
✅ MSELoss works correctly!

🔬 Unit Test: Cross-Entropy Loss...
✅ CrossEntropyLoss works correctly!

📈 Progress: Loss Functions Module ✓
```

### Manual Testing Examples
```python
from losses_dev import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss

# Regression example
mse = MSELoss()
predictions = Tensor([200.0, 250.0, 300.0])  # House prices (thousands)
targets = Tensor([195.0, 260.0, 290.0])
loss = mse(predictions, targets)
print(f"MSE Loss: {loss.data:.2f}")

# Classification example
ce = CrossEntropyLoss()
logits = Tensor([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2]])
labels = Tensor([0, 1])  # Class indices
loss = ce(logits, labels)
print(f"CrossEntropy Loss: {loss.data:.3f}")
```

## Production Context

### Your Implementation vs. Production Frameworks

Understanding what you're building vs. what production frameworks provide:

| Feature | Your Losses (Module 04) | PyTorch torch.nn | TensorFlow tf.keras |
|---------|------------------------|------------------|---------------------|
| **Backend** | NumPy (CPU-only) | C++/CUDA (CPU/GPU) | C++/CUDA/XLA |
| **MSELoss** | Mean reduction | `nn.MSELoss(reduction=)` | `MeanSquaredError()` |
| **CrossEntropyLoss** | Log-sum-exp trick | `nn.CrossEntropyLoss()` | `CategoricalCrossentropy()` |
| **Numerical Stability** | ✅ Manual log-sum-exp | ✅ Fused LogSoftmax+NLL | ✅ Built-in stability |
| **BinaryCrossEntropy** | Epsilon clamping | `nn.BCELoss()` | `BinaryCrossentropy()` |
| **Reduction Options** | Mean only | mean/sum/none | mean/sum/auto |
| **Label Smoothing** | ❌ Not implemented | ✅ Built-in parameter | ✅ Built-in parameter |
| **Weighted Losses** | ❌ Not implemented | ✅ Class weights | ✅ Sample weights |
| **GPU Acceleration** | ❌ CPU-only NumPy | ✅ CUDA kernels | ✅ CUDA/TPU |

**Educational Focus**: Your implementations prioritize clarity and explicit numerical stability handling. Production frameworks use optimized kernels and fused operations but follow the same mathematical principles.

### Side-by-Side Code Comparison

**Your implementation:**
```python
from tinytorch.core.losses import MSELoss, CrossEntropyLoss

# MSE for regression
mse = MSELoss()
predictions = Tensor([[2.5], [3.1], [1.8]])
targets = Tensor([[2.0], [3.0], [2.0]])
loss = mse.forward(predictions, targets)

# CrossEntropy for classification
ce = CrossEntropyLoss()
logits = Tensor([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2]])
labels = Tensor([0, 1])  # Class indices
loss = ce.forward(logits, labels)
```

**Equivalent PyTorch:**
```python
import torch
import torch.nn as nn

# MSE for regression
mse = nn.MSELoss()
predictions = torch.tensor([[2.5], [3.1], [1.8]])
targets = torch.tensor([[2.0], [3.0], [2.0]])
loss = mse(predictions, targets)  # Callable, no .forward()

# CrossEntropy for classification
ce = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2]])
labels = torch.tensor([0, 1])  # Class indices
loss = ce(logits, labels)  # Fused LogSoftmax+NLLLoss
```

**Key Differences:**
1. **Fused Operations**: PyTorch's `CrossEntropyLoss` combines LogSoftmax and NLLLoss in a single CUDA kernel for efficiency
2. **API Design**: PyTorch losses are callable (implement `__call__`), allowing `loss_fn(input, target)` instead of `loss_fn.forward(input, target)`
3. **Reduction Control**: PyTorch supports `reduction='none'` to return per-sample losses (useful for debugging or custom weighting)
4. **GPU Support**: PyTorch losses work seamlessly on CUDA tensors with same API: `loss.cuda()` then all computations on GPU

### Real-World Production Usage

**OpenAI GPT-3/GPT-4**: Uses CrossEntropy loss over 50K+ token vocabulary for next-token prediction during pretraining. The log-sum-exp trick is critical for numerical stability when computing losses over such large vocabularies—naive softmax would overflow.

**Google BERT**: Fine-tuning on downstream tasks uses CrossEntropy for classification heads (sentiment analysis, NER) and MSE for regression tasks (STS similarity scores). Label smoothing is applied to prevent overconfident predictions.

**Tesla Autopilot**: Object detection uses specialized losses (Focal Loss for class imbalance, IoU Loss for bounding boxes), but semantic segmentation uses CrossEntropy per pixel over 20+ classes. Weighted loss accounts for class imbalance (rare classes like pedestrians get higher weights).

**Meta ResNet**: ImageNet classification uses CrossEntropy over 1000 classes with batch size 256. Memory-efficient implementation computes loss on GPU using fused kernels—computing softmax for 256×1000 logits requires careful numerical handling.

**Medical AI (Diagnosis)**: Binary classification for disease detection uses BinaryCrossEntropy with class weights to handle severe class imbalance (1% positive cases). Epsilon clamping prevents `log(0)` when model becomes overconfident.

### Performance Characteristics at Scale

**Memory Overhead**: CrossEntropy with vocabulary size V and batch size B requires storing:
- Logits: B × V (input)
- Log-softmax: B × V (intermediate)
- Per-sample losses: B (before reduction)
- For GPT-3 scale (B=2048, V=50257): ~400MB in FP32, ~200MB in FP16

**Computational Bottleneck**: The expensive operations in CrossEntropy are:
1. **Max reduction**: O(B×V) to find max per sample
2. **Exponential**: O(B×V) for `exp(x - max)`
3. **Sum reduction**: O(B×V) for normalizer
4. **Log**: O(B) for final log-sum-exp
5. **Indexing**: O(B) to select correct class log-probs

For B=256, V=1000, this is ~768K exponential operations per forward pass. GPU parallelization reduces wall-clock time by 100-1000×.

**Numerical Precision**: Mixed precision training (FP16 activations, FP32 loss) is standard practice. Loss computation must happen in FP32 to prevent numerical issues—FP16 has limited dynamic range (~10⁻⁸ to 10⁴), insufficient for extreme log values.

**Label Smoothing Impact**: Production systems use label smoothing (e.g., ε=0.1) to prevent overconfidence. Instead of hard targets [0,1,0], use [ε/V, 1-ε+ε/V, ε/V]. This improves generalization by ~0.5-1% on ImageNet but requires modifying the loss computation.

### How Your Implementation Maps to PyTorch

**What you just built:**
```python
# Your log-sum-exp implementation
def log_softmax(x, dim=-1):
    max_vals = np.max(x.data, axis=dim, keepdims=True)
    shifted = x.data - max_vals
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
    return Tensor(x.data - max_vals - log_sum_exp)

# Your CrossEntropy
class CrossEntropyLoss:
    def forward(self, logits, targets):
        log_probs = log_softmax(logits, dim=-1)
        batch_size = logits.shape[0]
        selected = log_probs.data[np.arange(batch_size), targets.data.astype(int)]
        return Tensor(-np.mean(selected))
```

**How PyTorch does it:**
```python
# PyTorch C++ implementation (simplified)
# aten/src/ATen/native/Loss.cpp
Tensor log_softmax(const Tensor& self, int64_t dim) {
    auto max_val = at::max_values(self, dim, /*keepdim=*/true);
    auto shifted = self - max_val;
    auto log_sum = at::logsumexp(shifted, dim, /*keepdim=*/true);
    return self - log_sum;  // Fused for efficiency
}

Tensor cross_entropy_loss(const Tensor& input, const Tensor& target) {
    auto log_probs = at::log_softmax(input, /*dim=*/1);
    return at::nll_loss(log_probs, target, /*weight=*/{}, /*reduction=*/Mean);
}
```

**Key Insight**: Your implementation uses the **exact same log-sum-exp algorithm** as PyTorch. The difference is execution (NumPy CPU vs CUDA GPU) and optimization (Python loops vs fused kernels), not the fundamental mathematics.

## Systems Thinking Questions

### Real-World Applications
- **Computer Vision**: ImageNet uses CrossEntropy over 1000 classes with 1.2M training images
- **Language Modeling**: GPT models use CrossEntropy over 50K+ token vocabularies for next-token prediction
- **Medical Diagnosis**: BinaryCrossEntropy for disease detection where class imbalance is critical
- **Recommender Systems**: MSE for rating prediction, BCE for click-through rate estimation

### Mathematical Foundations
- **MSE Properties**: Convex loss landscape, quadratic penalty, maximum likelihood under Gaussian noise assumption
- **CrossEntropy Derivation**: Negative log-likelihood of correct class under softmax distribution
- **Log-Sum-Exp Trick**: Prevents overflow by factoring out max value before exponential computation
- **Gradient Behavior**: MSE gradient scales linearly with error; CrossEntropy gradient is confidence-weighted

### Performance Characteristics
- **Memory Scaling**: CrossEntropy uses ~2.5 × batch_size × num_classes; MSE uses ~2 × batch_size × output_dim
- **Computational Cost**: CrossEntropy requires expensive exp/log operations (~10x arithmetic cost)
- **Numerical Precision**: FP16 training requires loss scaling to prevent gradient underflow
- **Batch Size Effects**: Mean reduction provides batch-size-independent gradients; sum reduction scales with batch size

## Ready to Build?

You're about to implement the objectives that drive all machine learning. Loss functions transform abstract learning goals (make good predictions) into concrete mathematical targets that gradient descent can optimize. Every training run in production ML—from GPT to ResNet—relies on the numerical stability techniques you'll implement here.

Understanding loss functions deeply means you'll know why training diverges with large learning rates, how to debug NaN losses, and when to choose MSE versus CrossEntropy for your problem. These aren't just formulas—they're the feedback mechanisms that make learning possible.

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/04_losses/losses_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/04_losses/losses_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/04_losses/losses_dev.ipynb
:class-header: bg-light

Browse the notebook source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.
```

**Local workflow**:
```bash
# Start the module
tito module start 04

# Work in Jupyter
tito jupyter 04

# When complete
tito module complete 04
tito test losses
```

---

<div class="prev-next-area">
<a class="left-prev" href="../03_layers/ABOUT.html" title="previous page">← Module 03: Layers</a>
<a class="right-next" href="../05_autograd/ABOUT.html" title="next page">Module 05: Autograd →</a>
</div>
