# Module 04: Losses

:::{admonition} Module Info
:class: note

**FOUNDATION TIER** | Difficulty: ●●○○ | Time: 4-6 hours | Prerequisites: 01, 02, 03

**Prerequisites:** Modules 01 (Tensor), 02 (Activations), and 03 (Layers) must be completed. This module assumes you understand:
- Tensor operations and broadcasting (Module 01)
- Activation functions and their role in neural networks (Module 02)
- Layers and how they transform data (Module 03)

If you can build a simple neural network that takes input and produces output, you're ready to learn how to measure its quality.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F04_losses%2F04_losses.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/04_losses/04_losses.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/04_losses.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Loss functions are the mathematical conscience of machine learning. Every neural network needs to know when it's right and when it's wrong. Loss functions provide that feedback by measuring the distance between what your model predicts and what actually happened. Without loss functions, models have no way to improve - they're like athletes training without knowing their score.

In this module, you'll implement three essential loss functions: Mean Squared Error (MSE) for regression, Cross-Entropy for multi-class classification, and Binary Cross-Entropy for binary decisions. You'll also master the log-sum-exp trick, a crucial numerical stability technique that prevents computational overflow with large numbers. These implementations will serve as the foundation for Module 06: Autograd, where gradients flow backward from these loss values to update model parameters.

By the end, you'll understand not just how to compute loss, but why different problems require different loss functions, and how numerical stability shapes production ML systems.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** MSELoss for regression, CrossEntropyLoss for multi-class classification, and BinaryCrossEntropyLoss for binary decisions
- **Master** the log-sum-exp trick for numerically stable softmax computation
- **Understand** computational complexity (O(B×C) for cross-entropy with large vocabularies) and memory trade-offs
- **Analyze** loss function behavior across different prediction patterns and confidence levels
- **Connect** your implementation to production PyTorch patterns and engineering decisions at scale
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Loss Functions
flowchart LR
    subgraph "Your Loss Functions"
        A["log_softmax()<br/>Numerical Stability"]
        B["MSELoss<br/>Regression"]
        C["CrossEntropyLoss<br/>Classification"]
        D["BinaryCrossEntropyLoss<br/>Binary Decisions"]
    end

    A --> C

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
```

**Implementation roadmap:**

| Step | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `log_softmax()` | Log-sum-exp trick for numerical stability |
| 2 | `MSELoss.forward()` | Mean squared error for continuous predictions |
| 3 | `CrossEntropyLoss.forward()` | Negative log-likelihood for multi-class classification |
| 4 | `BinaryCrossEntropyLoss.forward()` | Cross-entropy specialized for binary decisions |

**The pattern you'll enable:**
```python
# Measuring prediction quality
loss = criterion(predictions, targets)  # Scalar feedback signal for learning
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Gradient computation (that's Module 06: Autograd)
- Advanced loss variants (Focal Loss, Label Smoothing, Huber Loss)
- Hierarchical or sampled softmax for large vocabularies (PyTorch optimization)
- Custom reduction strategies beyond mean

**You are building the core feedback signal.** Gradient-based learning comes next.

## API Reference

This section provides a quick reference for the loss functions you'll build. Use it as your cheat sheet while implementing and debugging.

### Helper Functions

```python
log_softmax(x: Tensor, dim: int = -1) -> Tensor
```

Computes numerically stable log-softmax using the log-sum-exp trick. This is the foundation for cross-entropy loss.

**Parameters:**
- `x` (Tensor): Input tensor containing logits (raw model outputs, unbounded values)
- `dim` (int): Dimension along which to compute log-softmax (default: -1, last dimension)

**Returns:** Tensor with same shape as input, containing log-probabilities

**Note:** Logits are raw, unbounded scores from your model before any activation function. CrossEntropyLoss expects logits, not probabilities.

### Loss Functions

All loss functions follow the same pattern:

| Loss Function | Constructor | Forward Signature | Use Case |
|--------------|-------------|-------------------|----------|
| `MSELoss` | `MSELoss()` | `forward(predictions: Tensor, targets: Tensor) -> Tensor` | Regression |
| `CrossEntropyLoss` | `CrossEntropyLoss()` | `forward(logits: Tensor, targets: Tensor) -> Tensor` | Multi-class classification |
| `BinaryCrossEntropyLoss` | `BinaryCrossEntropyLoss()` | `forward(predictions: Tensor, targets: Tensor) -> Tensor` | Binary classification |

**Common Pattern:**
```python
loss_fn = MSELoss()
loss = loss_fn(predictions, targets)  # __call__ delegates to forward()
```

### Input/Output Shapes

Understanding input shapes is crucial for correct loss computation:

| Loss | Predictions Shape | Targets Shape | Output Shape |
|------|------------------|---------------|--------------|
| MSE | `(N,)` or `(N, D)` | Same as predictions | `()` scalar |
| CrossEntropy | `(N, C)` logits¹ | `(N,)` class indices² | `()` scalar |
| BinaryCrossEntropy | `(N,)` probabilities³ | `(N,)` binary labels (0 or 1) | `()` scalar |

Where N = batch size, D = feature dimension, C = number of classes

**Notes:**
1. **Logits**: Raw unbounded values from your model (e.g., `[2.3, -1.2, 5.1]`). Do NOT apply softmax before passing to CrossEntropyLoss.
2. **Class indices**: Integer values from 0 to C-1 indicating the correct class (e.g., `[0, 2, 1]` for 3 samples).
3. **Probabilities**: Values between 0 and 1 after applying sigmoid activation. Must be in valid probability range.

## Core Concepts

This section covers the fundamental ideas you need to understand loss functions deeply. These concepts apply to every ML framework, not just TinyTorch.

### Loss as a Feedback Signal

Loss functions transform the abstract question "how good is my model?" into a concrete number that can drive improvement. Consider a simple example: predicting house prices. If your model predicts $250,000 for a house that sold for $245,000, how wrong is that? What about $150,000 when the actual price was $250,000? The loss function quantifies these errors in a way that optimization algorithms can use.

The key insight is that loss functions must be differentiable - you need to know not just the current error, but which direction to move parameters to reduce that error. This is why we use squared differences instead of absolute differences in MSE: the square function has a smooth derivative that points toward improvement.

Every training iteration follows the same pattern: forward pass produces predictions, loss function measures error, backward pass (Module 06) computes how to improve. The loss value itself becomes a single number summarizing model quality across an entire batch of examples.

### Mean Squared Error

MSE is the foundational loss for regression problems. It measures the average squared distance between predictions and targets. The squaring serves three purposes: it makes all errors positive (preventing cancellation), it heavily penalizes large errors, and it creates smooth gradients for optimization.

Here's the complete implementation from your module:

```python
def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
    """Compute mean squared error between predictions and targets."""
    # Step 1: Compute element-wise difference
    diff = predictions.data - targets.data

    # Step 2: Square the differences
    squared_diff = diff ** 2

    # Step 3: Take mean across all elements
    mse = np.mean(squared_diff)

    return Tensor(mse)
```

The beauty of MSE is its simplicity: subtract, square, average. Yet this simple formula creates a quadratic error landscape. An error of 10 contributes 100 to the loss, while an error of 20 contributes 400. This quadratic growth means the loss function cares much more about fixing large errors than small ones, naturally prioritizing the worst predictions during optimization.

Consider predicting house prices. An error of $5,000 on a $200,000 house gets squared to 25,000,000. An error of $50,000 gets squared to 2,500,000,000 - one hundred times worse for an error only ten times larger. This sensitivity to outliers can be both a strength (quickly correcting large errors) and a weakness (vulnerable to noisy labels).

### Cross-Entropy Loss

Cross-entropy measures how wrong your probability predictions are for classification problems. Unlike MSE which measures distance, cross-entropy measures surprise: how unexpected is the true outcome given your model's probability distribution?

The mathematical formula is deceptively simple: negative log-likelihood of the correct class. But implementing it correctly requires careful attention to numerical stability. Here's how your implementation handles it:

```python
def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
    """Compute cross-entropy loss between logits and target class indices."""
    # Step 1: Compute log-softmax for numerical stability
    log_probs = log_softmax(logits, dim=-1)

    # Step 2: Select log-probabilities for correct classes
    batch_size = logits.shape[0]
    target_indices = targets.data.astype(int)

    # Select correct class log-probabilities using advanced indexing
    selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

    # Step 3: Return negative mean (cross-entropy is negative log-likelihood)
    cross_entropy = -np.mean(selected_log_probs)

    return Tensor(cross_entropy)
```

The critical detail is using `log_softmax` instead of computing softmax then taking the log. This seemingly minor choice prevents catastrophic overflow with large logits. Without it, a logit value of 100 would compute `exp(100) = 2.7×10^43`, which exceeds float32 range and becomes infinity.

Cross-entropy's power comes from its asymmetric penalty structure. If your model predicts 0.99 probability for the correct class, the loss is `-log(0.99) = 0.01` - very small. But if you predict 0.01 for the correct class, the loss is `-log(0.01) = 4.6` - much larger. This creates strong pressure to be confident when correct and uncertain when wrong.

### Numerical Stability in Loss Computation

The log-sum-exp trick is one of the most important numerical stability techniques in machine learning. It solves a fundamental problem: computing softmax directly causes overflow, but we need softmax for classification.

Consider what happens without the trick. Standard softmax computes `exp(x) / sum(exp(x))`. With logits `[100, 200, 300]`, you'd compute `exp(300) = 1.97×10^130`, which is infinity in float32. The trick subtracts the maximum value first, making the largest exponent zero:

```python
def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Compute log-softmax with numerical stability."""
    # Step 1: Find max along dimension for numerical stability
    max_vals = np.max(x.data, axis=dim, keepdims=True)

    # Step 2: Subtract max to prevent overflow
    shifted = x.data - max_vals

    # Step 3: Compute log(sum(exp(shifted)))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # Step 4: Return log_softmax = input - max - log_sum_exp
    result = x.data - max_vals - log_sum_exp

    return Tensor(result)
```

After subtracting the max (300), the shifted logits become `[-200, -100, 0]`. Now the largest exponent is `exp(0) = 1.0`, perfectly safe. The smaller values like `exp(-200)` underflow to zero, but that's acceptable - they contribute negligibly to the sum anyway.

This trick is mathematically exact, not an approximation. Subtracting the max from both numerator and denominator cancels out, leaving the result unchanged. But the computational difference is dramatic: infinity versus valid probabilities.

### Reduction Strategies

All three loss functions reduce a batch of per-sample errors to a single scalar by taking the mean. This reduction strategy affects both the loss magnitude and the resulting gradients during backpropagation.

The mean reduction has important properties. First, it normalizes by batch size, making loss values comparable across different batch sizes. A batch of 32 samples and a batch of 128 samples produce similar loss magnitudes if the per-sample errors are similar. Second, it makes gradients inversely proportional to batch size - with 128 samples, each sample contributes 1/128 to the total gradient, preventing gradient explosion with large batches.

Alternative reduction strategies exist but aren't implemented in this module. Sum reduction (`np.sum` instead of `np.mean`) accumulates total error across the batch, making loss scale with batch size. No reduction (`reduction='none'`) returns per-sample losses, useful for weighted sampling or analyzing individual predictions. Production frameworks support all these modes, but mean reduction is the standard choice for stable training.

The choice of reduction interacts with learning rate. If you switch from mean to sum reduction, you must divide your learning rate by batch size to maintain equivalent optimization dynamics. This is why PyTorch defaults to mean reduction - it makes hyperparameters more transferable across different batch sizes.

## Common Errors

### Shape Mismatch in Cross-Entropy

**Error**: `IndexError: index 5 is out of bounds for axis 1 with size 3`

This happens when your target class indices exceed the number of classes in your logits. If you have 3 classes (indices 0, 1, 2) but your targets contain index 5, the indexing operation fails.

**Fix**: Verify your target indices match your model's output dimensions. For a 3-class problem, targets should only contain 0, 1, or 2.

```python
# ❌ Wrong - target index 5 doesn't exist for 3 classes
logits = Tensor([[1.0, 2.0, 3.0]])  # 3 classes
targets = Tensor([5])  # Index out of bounds

# ✅ Correct - target indices match number of classes
logits = Tensor([[1.0, 2.0, 3.0]])
targets = Tensor([2])  # Index 2 is valid for 3 classes
```

### NaN Loss from Numerical Instability

**Error**: `RuntimeWarning: invalid value encountered in log` followed by `loss.data = nan`

This occurs when probabilities reach exactly 0.0 or 1.0, causing `log(0) = -∞`. Binary cross-entropy is particularly vulnerable because it computes both `log(prediction)` and `log(1-prediction)`.

**Fix**: Clamp probabilities to a safe range using epsilon:

```python
# Already implemented in your BinaryCrossEntropyLoss:
eps = 1e-7
clamped_preds = np.clip(predictions.data, eps, 1 - eps)
```

This ensures you never compute `log(0)` while keeping values extremely close to the true probabilities.

### Confusing Logits and Probabilities

**Error**: `loss.data = inf` or unreasonably large loss values

Cross-entropy expects raw logits (unbounded values from your model), while binary cross-entropy expects probabilities (0 to 1 range). Mixing these up causes numerical explosions.

**Fix**: Check what your model outputs:

```python
# ✅ CrossEntropyLoss: Use raw logits (no sigmoid/softmax!)
logits = linear_layer(x)  # Raw outputs like [2.3, -1.2, 5.1]
loss = CrossEntropyLoss()(logits, targets)

# ✅ BinaryCrossEntropyLoss: Use probabilities (apply sigmoid!)
logits = linear_layer(x)
probabilities = sigmoid(logits)  # Converts to [0, 1] range
loss = BinaryCrossEntropyLoss()(probabilities, targets)
```

## Production Context

### Your Implementation vs. PyTorch

Your TinyTorch loss functions and PyTorch's implementations share the same mathematical foundations and numerical stability techniques. The differences are in performance optimizations, GPU support, and additional features for production use.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Backend** | NumPy (Python) | C++/CUDA |
| **Numerical Stability** | Log-sum-exp trick | Same trick, fused kernels |
| **Speed** | 1x (baseline) | 10-100x faster (GPU) |
| **Reduction Modes** | Mean only | mean, sum, none |
| **Advanced Variants** | ✗ | Label smoothing, weights |
| **Memory Efficiency** | Standard | Fused operations reduce copies |

### Code Comparison

The following comparison shows equivalent loss computations in TinyTorch and PyTorch. Notice how the high-level API is nearly identical - you're learning the same patterns used in production.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch import Tensor
from tinytorch.core.losses import MSELoss, CrossEntropyLoss

# Regression
mse_loss = MSELoss()
predictions = Tensor([200.0, 250.0, 300.0])
targets = Tensor([195.0, 260.0, 290.0])
loss = mse_loss(predictions, targets)

# Classification
ce_loss = CrossEntropyLoss()
logits = Tensor([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2]])
labels = Tensor([0, 1])
loss = ce_loss(logits, labels)
```
````

````{tab-item} ⚡ PyTorch
```python
import torch
import torch.nn as nn

# Regression
mse_loss = nn.MSELoss()
predictions = torch.tensor([200.0, 250.0, 300.0])
targets = torch.tensor([195.0, 260.0, 290.0])
loss = mse_loss(predictions, targets)

# Classification
ce_loss = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 0.5, 0.1], [0.3, 1.8, 0.2]])
labels = torch.tensor([0, 1])
loss = ce_loss(logits, labels)
```
````
`````

Let's walk through the key similarities and differences:

- **Line 1 (Imports)**: Both frameworks use modular imports. TinyTorch exposes loss functions from `core.losses`; PyTorch uses `torch.nn`.
- **Line 3 (Construction)**: Both use the same pattern: instantiate the loss function once, then call it multiple times. No parameters needed for basic usage.
- **Line 4-5 (Data)**: TinyTorch wraps Python lists in `Tensor`; PyTorch uses `torch.tensor()`. The data structure concept is identical.
- **Line 6 (Computation)**: Both compute loss by calling the loss function object. Under the hood, this calls the `forward()` method you implemented.
- **Line 9 (Classification)**: Both expect raw logits (not probabilities) for cross-entropy. The `log_softmax` computation happens internally in both frameworks.

```{tip} What's Identical

The mathematical formulas, numerical stability techniques (log-sum-exp trick), and high-level API patterns. When you debug PyTorch loss functions, you'll understand exactly what's happening because you built the same abstractions.
```

### Why Loss Functions Matter at Scale

To appreciate why loss functions matter in production, consider the scale of modern ML systems:

- **Language models**: 50,000 token vocabulary × 128 batch size = **6.4M exponential operations per loss computation**. With sampled softmax, this reduces to ~128K operations (50× speedup).
- **Computer vision**: ImageNet with 1,000 classes processes **256,000 softmax computations** per batch. Fused CUDA kernels reduce this from 15ms to 0.5ms.
- **Recommendation systems**: Billions of items require specialized loss functions. YouTube's recommendation system uses **sampled softmax over 1M+ videos**, making loss computation the primary bottleneck.

Memory pressure is equally significant. A language model forward pass might consume 8GB for activations, 2GB for parameters, but **768MB just for the cross-entropy loss computation** (B=128, C=50000, float32). Using FP16 cuts this to 384MB. Using hierarchical softmax eliminates the materialization entirely.

The loss computation typically accounts for **5-10% of total training time** in well-optimized systems, but can dominate (30-50%) for large vocabularies without optimization. This is why production frameworks invest heavily in fused kernels, specialized data structures, and algorithmic improvements like hierarchical softmax.

## Check Your Understanding

Test yourself with these systems thinking questions. They're designed to build intuition for the performance characteristics you'll encounter in production ML.

**Q1: Memory Calculation - Large Vocabulary Language Model**

A language model with 50,000 token vocabulary uses CrossEntropyLoss with batch size 128. Using float32, how much memory does the loss computation require for logits, softmax probabilities, and log-probabilities?

```{admonition} Answer
:class: dropdown

**Calculation:**
- Logits: 128 × 50,000 × 4 bytes = 25.6 MB
- Softmax probabilities: 128 × 50,000 × 4 bytes = 25.6 MB
- Log-softmax: 128 × 50,000 × 4 bytes = 25.6 MB

**Total: 76.8 MB** just for loss computation (before model activations!)

**Key insight**: Memory scales as B×C. Doubling vocabulary doubles loss computation memory. This is why large language models use techniques like sampled softmax - they literally can't afford to materialize the full vocabulary every forward pass.

**Production solution**: Switch to FP16 (cuts to 38.4 MB) or use hierarchical/sampled softmax (reduces C from 50,000 to ~1,000).
```

**Q2: Complexity Analysis - Softmax Bottleneck**

Your training profile shows: Forward pass 80ms, Loss computation 120ms, Backward pass 150ms. Your model has 1,000 output classes and batch size 64. Why is loss computation so expensive, and what's the fix?

```{admonition} Answer
:class: dropdown

**Problem**: Loss taking 120ms (34% of iteration time) is unusually high. Normal ratio is 5-10%.

**Root cause**: CrossEntropyLoss is O(B×C). With B=64 and C=1,000, that's 64,000 exp/log operations. If implemented naively in Python loops (not vectorized), this becomes a bottleneck.

**Diagnosis steps**:
1. Profile within loss: Is `log_softmax` the bottleneck? (Likely yes)
2. Check vectorization: Are you using NumPy broadcasting or Python loops?
3. Check batch size: Is B=64 too small to utilize vectorization?

**Fixes**:
- **Immediate**: Ensure you're using vectorized NumPy ops (not loops)
- **Better**: Use PyTorch with CUDA - GPU acceleration gives 10-50× speedup
- **Advanced**: For C>10,000, use hierarchical softmax (reduces to O(B×log C))

**Reality check**: In optimized PyTorch on GPU, loss should be ~5ms for this size, not 120ms. Your implementation in pure Python/NumPy is expected to be slower, but vectorization is crucial.
```

**Q3: Numerical Stability - Why Log-Sum-Exp Matters**

Your model outputs logits `[50, 100, 150]`. Without the log-sum-exp trick, what happens when you compute softmax? With the trick, what values are actually computed?

```{admonition} Answer
:class: dropdown

**Without the trick (naive softmax):**
```text
exp_vals = [exp(50), exp(100), exp(150)]
         = [5.2×10²¹, 2.7×10⁴³, 1.4×10⁶⁵]  # Last value overflows to inf!
softmax = exp_vals / sum(exp_vals)  # inf / inf = nan
```
**Result**: NaN loss, training fails.

**With log-sum-exp trick:**
```text
max_val = 150
shifted = [50-150, 100-150, 150-150] = [-100, -50, 0]
exp_shifted = [exp(-100), exp(-50), exp(0)]
            = [3.7×10⁻⁴⁴, 1.9×10⁻²², 1.0]  # All ≤ 1.0, safe!
sum_exp = 1.0 (others negligible)
log_sum_exp = log(1.0) = 0
log_softmax = shifted - log_sum_exp = [-100, -50, 0]
```
**Result**: Valid log-probabilities, stable training.

**Key insight**: Subtracting max makes largest value 0, so `exp(0) = 1.0` is always safe. Smaller values underflow to 0, but that's fine - they contribute negligibly anyway. This is why **you must use log-sum-exp for any softmax computation**.
```

**Q4: Loss Function Selection - Classification Problem**

You're building a medical diagnosis system with 5 disease categories. Should you use BinaryCrossEntropyLoss or CrossEntropyLoss? What if the categories aren't mutually exclusive (patient can have multiple diseases)?

```{admonition} Answer
:class: dropdown

**Case 1: Mutually exclusive diseases** (patient has exactly one)
- **Use**: CrossEntropyLoss
- **Model output**: Logits of shape (batch_size, 5)
- **Why**: Categories are mutually exclusive - softmax ensures probabilities sum to 1.0

**Case 2: Multi-label classification** (patient can have multiple diseases)
- **Use**: BinaryCrossEntropyLoss
- **Model output**: Probabilities of shape (batch_size, 5) after sigmoid
- **Why**: Each disease is an independent binary decision. Softmax would incorrectly force them to sum to 1.

**Example**:
```python
# ✅ Mutually exclusive (one disease)
logits = Linear(features, 5)(x)  # Shape: (B, 5)
loss = CrossEntropyLoss()(logits, targets)  # targets: class index 0-4

# ✅ Multi-label (can have multiple)
logits = Linear(features, 5)(x)  # Shape: (B, 5)
probs = sigmoid(logits)  # Independent probabilities
targets = Tensor([[1, 0, 1, 0, 0], ...])  # Binary labels for each disease
loss = BinaryCrossEntropyLoss()(probs, targets)
```

**Critical medical consideration**: Multi-label is more realistic - patients often have comorbidities!
```

**Q5: Batch Size Impact - Memory and Gradients**

You train with batch size 32, using 4GB GPU memory. You want to increase to batch size 128. Will memory usage be 16GB? What happens to the loss value and gradient quality?

```{admonition} Answer
:class: dropdown

**Memory usage**: Yes, approximately **16GB** (4× increase)
- Loss computation scales linearly: 4× batch → 4× memory
- Activations scale linearly: 4× batch → 4× memory
- Model parameters: Fixed (same regardless of batch size)

**Problem**: If your GPU only has 12GB, training will crash with OOM (out of memory).

**Loss value**: **Stays the same** (assuming similar data)
```python
# Both compute the mean over their batch:
batch_32_loss = mean(losses[:32])   # Average of 32 samples
batch_128_loss = mean(losses[:128]) # Average of 128 samples
# If data is similar, means are similar
```

**Gradient quality**: **Improves with larger batch**
- Batch 32: High variance, noisy gradient estimates
- Batch 128: Lower variance, smoother gradient, more stable convergence
- Trade-off: More computation per step, fewer steps per epoch

**Production solution - Gradient Accumulation**:
```python
# Simulate batch_size=128 with only batch_size=32 memory:
for i in range(4):  # 4 micro-batches
    loss = compute_loss(data[i*32:(i+1)*32])
    loss.backward()  # Accumulate gradients
optimizer.step()  # Update once with accumulated gradients (4×32 = 128 effective batch)
```

This gives you the gradient quality of batch 128 with only the memory cost of batch 32!
```

## Further Reading

For students who want to understand the academic foundations and explore deeper:

### Seminal Papers

- **Improving neural networks by preventing co-adaptation of feature detectors** - Hinton et al. (2012). Introduces dropout, but also discusses cross-entropy loss and its role in preventing overfitting. Understanding why cross-entropy works better than MSE for classification is fundamental. [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)

- **Focal Loss for Dense Object Detection** - Lin et al. (2017). Addresses class imbalance by reshaping the loss curve to down-weight easy examples. Shows how loss function design directly impacts model performance on real problems. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

- **When Does Label Smoothing Help?** - Müller et al. (2019). Analyzes why adding small noise to target labels (label smoothing) improves generalization. Demonstrates that loss function details matter beyond just basic formulation. [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)

### Additional Resources

- **Tutorial**: [Understanding Cross-Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - PyTorch documentation with mathematical details
- **Blog post**: "The Softmax Function and Its Derivative" - Excellent explanation of log-sum-exp trick and numerical stability
- **Textbook**: "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter 5 covers loss functions and maximum likelihood

## What's Next

```{seealso} Coming Up: Module 05 - DataLoader

Build efficient data pipelines that handle batching, shuffling, and iteration over your datasets. DataLoader prepares your training data so that autograd and training loops can consume it efficiently.
```

**Preview - How Your Loss Functions Get Used in Future Modules:**

| Module | What It Does | Your Loss In Action |
|--------|--------------|---------------------|
| **06: Autograd** | Automatic differentiation | `loss.backward()` computes gradients |
| **07: Optimizers** | Parameter updates | `optimizer.step()` uses loss gradients to improve weights |
| **08: Training** | Complete training loop | `loss = criterion(outputs, targets)` measures progress |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/04_losses/04_losses.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/04_losses/04_losses.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
