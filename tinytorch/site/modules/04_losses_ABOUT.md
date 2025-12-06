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
