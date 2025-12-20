# Module 02: Activations

:::{admonition} Module Info
:class: note

**FOUNDATION TIER** | Difficulty: ●○○○ | Time: 3-5 hours | Prerequisites: 01 (Tensor)

**Prerequisites: Module 01 (Tensor)** means you need:
- Completed Tensor implementation with element-wise operations
- Understanding of tensor shapes and broadcasting
- Familiarity with NumPy mathematical functions

If you can create a Tensor and perform element-wise arithmetic (`x + y`, `x * 2`), you're ready.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F02_activations%2F02_activations.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/02_activations/02_activations.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/02_activations.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Activation functions are the nonlinear transformations that give neural networks their power. Without them, stacking multiple layers would be pointless: no matter how many linear transformations you chain together, the result is still just one linear transformation. A 100-layer network without activations is mathematically identical to a single-layer network.

Activations introduce nonlinearity. ReLU zeros out negative values. Sigmoid squashes any input to a probability between 0 and 1. Softmax converts raw scores into a valid probability distribution. These simple mathematical functions are what enable neural networks to learn complex patterns like recognizing faces, translating languages, and playing games at superhuman levels.

In this module, you'll implement five essential activation functions from scratch. By the end, you'll understand why ReLU replaced sigmoid in hidden layers, how numerical stability prevents catastrophic failures in softmax, and when to use each activation in production systems.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** five core activation functions (ReLU, Sigmoid, Tanh, GELU, Softmax) with proper numerical stability
- **Understand** why nonlinearity is essential for neural network expressiveness and how activations enable learning
- **Master** computational trade-offs between activation choices and their impact on training speed
- **Connect** your implementations to production patterns in PyTorch and real-world architecture decisions
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Activation Functions
flowchart LR
    subgraph "Your Activation Functions"
        A["ReLU<br/>max(0, x)"]
        B["Sigmoid<br/>1/(1+e^-x)"]
        C["Tanh<br/>(e^x - e^-x)/(e^x + e^-x)"]
        D["GELU<br/>x·Φ(x)"]
        E["Softmax<br/>e^xi / Σe^xj"]
    end

    F[Input Tensor] --> A
    F --> B
    F --> C
    F --> D
    F --> E

    A --> G[Output Tensor]
    B --> G
    C --> G
    D --> G
    E --> G

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
    style E fill:#e2d5f1
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `ReLU.forward()` | Sparsity through zeroing negatives |
| 2 | `Sigmoid.forward()` | Mapping to (0,1) for probabilities |
| 3 | `Tanh.forward()` | Zero-centered activation for better gradients |
| 4 | `GELU.forward()` | Smooth nonlinearity for transformers |
| 5 | `Softmax.forward()` | Probability distributions with numerical stability |

**The pattern you'll enable:**

```python
# Transforming tensors through nonlinear functions
relu = ReLU()
activated = relu(x)  # Zeros negatives, keeps positives

softmax = Softmax()
probabilities = softmax(logits)  # Converts to probability distribution (sums to 1)
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Gradient computation (that's Module 06: Autograd - `backward()` methods are stubs for now)
- Learnable parameters (activations are fixed mathematical functions)
- Advanced variants (LeakyReLU, ELU, Swish - PyTorch has dozens, you'll build the core five)
- GPU acceleration (your NumPy implementation runs on CPU)

**You are building the nonlinear transformations.** Automatic differentiation comes in Module 06.

## API Reference

This section provides a quick reference for the activation classes you'll build. Each activation is a callable object with a `forward()` method that transforms an input tensor element-wise.

### Activation Pattern

All activations follow this structure:

```python
class ActivationName:
    def forward(self, x: Tensor) -> Tensor:
        # Apply mathematical transformation
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        # Stub for Module 06
        pass
```

### Core Activations

| Activation | Mathematical Form | Output Range | Primary Use Case |
|------------|------------------|--------------|------------------|
| `ReLU` | `max(0, x)` | `[0, ∞)` | Hidden layers (CNNs, MLPs) |
| `Sigmoid` | `1/(1 + e^-x)` | `(0, 1)` | Binary classification output |
| `Tanh` | `(e^x - e^-x)/(e^x + e^-x)` | `(-1, 1)` | RNNs, zero-centered needs |
| `GELU` | `x · Φ(x)` | `(-∞, ∞)` | Transformers (GPT, BERT) |
| `Softmax` | `e^xi / Σe^xj` | `(0, 1)`, sum=1 | Multi-class classification |

### Method Signatures

**ReLU**
```python
ReLU.forward(x: Tensor) -> Tensor
```
Sets negative values to zero, preserves positive values.

**Sigmoid**
```python
Sigmoid.forward(x: Tensor) -> Tensor
```
Maps any real number to (0, 1) range using logistic function.

**Tanh**
```python
Tanh.forward(x: Tensor) -> Tensor
```
Maps any real number to (-1, 1) range using hyperbolic tangent.

**GELU**
```python
GELU.forward(x: Tensor) -> Tensor
```
Smooth approximation to ReLU using Gaussian error function.

**Softmax**
```python
Softmax.forward(x: Tensor, dim: int = -1) -> Tensor
```
Converts vector to probability distribution along specified dimension.

## Core Concepts

This section covers the fundamental ideas you need to understand activation functions deeply. These concepts explain why neural networks need nonlinearity, how each activation behaves differently, and what trade-offs you're making when you choose one over another.

### Why Non-linearity Matters

Consider what happens when you stack linear transformations. If you multiply a matrix by a vector, then multiply the result by another matrix, the composition is still just matrix multiplication. Mathematically:

```
f(x) = W₂(W₁x) = (W₂W₁)x = Wx
```

A 100-layer network of pure matrix multiplications is identical to a single matrix multiplication. The depth buys you nothing.

Activation functions break this linearity. When you insert `f(x) = max(0, x)` between layers, the composition becomes nonlinear:

```
f(x) = max(0, W₂ max(0, W₁x))
```

Now you can't simplify the layers away. Each layer can learn to detect increasingly complex patterns. Layer 1 might detect edges in an image. Layer 2 combines edges into shapes. Layer 3 combines shapes into objects. This hierarchical feature learning is only possible because activations introduce nonlinearity.

Without activations, neural networks are just linear regression, no matter how many layers you stack. With activations, they become universal function approximators capable of learning any pattern from data.

### ReLU and Its Variants

ReLU (Rectified Linear Unit) is deceptively simple: it zeros out negative values and leaves positive values unchanged. Here's the complete implementation from your module:

```python
class ReLU:
    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU activation element-wise."""
        result = np.maximum(0, x.data)
        return Tensor(result)
```

This simplicity is ReLU's greatest strength. The operation is a single comparison per element: O(n) with a tiny constant factor. Modern CPUs can execute billions of comparisons per second. Compare this to sigmoid, which requires computing an exponential for every element.

ReLU creates **sparsity**. When half your activations are exactly zero, computations become faster (multiplying by zero is free) and models generalize better (sparse representations are less prone to overfitting). In a 1000-neuron layer, ReLU typically activates 300-500 neurons, effectively creating a smaller, specialized network for each input.

The discontinuity at zero is both a feature and a bug. During training (Module 08), you'll discover that ReLU's gradient is exactly 1 for positive inputs and exactly 0 for negative inputs. This prevents the vanishing gradient problem that plagued sigmoid-based networks. But it creates a new problem: **dying ReLU**. If a neuron's weights shift such that it always receives negative inputs, it will output zero forever, and the zero gradient means it can never recover.

Despite this limitation, ReLU remains the default choice for hidden layers in CNNs and feedforward networks. Its speed and effectiveness at preventing vanishing gradients make it hard to beat.

### Sigmoid and Tanh

Sigmoid maps any real number to the range (0, 1), making it perfect for representing probabilities:

```python
class Sigmoid:
    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation element-wise."""
        z = np.clip(x.data, -500, 500)  # Prevent overflow
        result_data = np.zeros_like(z)

        # Positive values: 1 / (1 + exp(-x))
        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        # Negative values: exp(x) / (1 + exp(x))
        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result_data)
```

Notice the numerical stability measures. Computing `1 / (1 + exp(-x))` directly fails for `x = 1000` because `exp(-1000)` underflows to zero, giving `1 / 1 = 1`. But the mathematically equivalent `exp(x) / (1 + exp(x))` fails for `x = 1000` because `exp(1000)` overflows to infinity. The solution is to compute different formulas depending on the sign of x, and clip extreme values to prevent overflow entirely.

Sigmoid's smooth S-curve makes it interpretable as a probability, which is why it's still used for binary classification outputs. But for hidden layers, it has fatal flaws. When |x| is large, the output saturates near 0 or 1, and the gradient becomes nearly zero. In deep networks, these tiny gradients multiply together as they backpropagate, vanishing exponentially. This is why sigmoid was largely replaced by ReLU for hidden layers around 2012.

Tanh is sigmoid's zero-centered cousin, mapping inputs to (-1, 1):

```python
class Tanh:
    def forward(self, x: Tensor) -> Tensor:
        """Apply tanh activation element-wise."""
        result = np.tanh(x.data)
        return Tensor(result)
```

The zero-centering matters because it means the output has roughly equal numbers of positive and negative values. This can help with gradient flow in recurrent networks, where the same weights are applied repeatedly. Tanh still suffers from vanishing gradients at extreme values, but the zero-centering makes it preferable to sigmoid when you need bounded outputs.

### Softmax and Numerical Stability

Softmax converts any vector into a valid probability distribution. All outputs are positive, and they sum to exactly 1. This makes it essential for multi-class classification:

```python
class Softmax:
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply softmax activation along specified dimension."""
        # Numerical stability: subtract max to prevent overflow
        x_max_data = np.max(x.data, axis=dim, keepdims=True)
        x_max = Tensor(x_max_data, requires_grad=False)
        x_shifted = x - x_max

        # Compute exponentials
        exp_values = Tensor(np.exp(x_shifted.data), requires_grad=x_shifted.requires_grad)

        # Sum along dimension
        exp_sum_data = np.sum(exp_values.data, axis=dim, keepdims=True)
        exp_sum = Tensor(exp_sum_data, requires_grad=exp_values.requires_grad)

        # Normalize to get probabilities
        result = exp_values / exp_sum
        return result
```

The max subtraction is critical. Without it, `softmax([1000, 1001, 1002])` would compute `exp(1000)`, which overflows to infinity, producing NaN results. Subtracting the max first gives `softmax([0, 1, 2])`, which computes safely. Mathematically, this is identical because the max factor cancels out:

```
exp(x - max) / Σ exp(x - max) = exp(x) / Σ exp(x)
```

Softmax amplifies differences. If the input is `[1, 2, 3]`, the output is approximately `[0.09, 0.24, 0.67]`. The largest input gets 67% of the probability mass, even though it's only 3× larger than the smallest input. This is because exponentials grow superlinearly. In classification, this is desirable: you want the network to be confident when it's confident.

But softmax's coupling is a gotcha. When you change one input, all outputs change because they're normalized by the same sum. This means the gradient involves a Jacobian matrix, not just element-wise derivatives. You'll see this complexity when you implement `backward()` in Module 06.

### Choosing Activations

Here's the decision tree production ML engineers use:

**For hidden layers:**
- Default choice: **ReLU** (fast, prevents vanishing gradients, creates sparsity)
- Modern transformers: **GELU** (smooth, better gradient flow, state-of-the-art results)
- Recurrent networks: **Tanh** (zero-centered helps with recurrence)
- Experimental: LeakyReLU, ELU, Swish (variants that fix dying ReLU problem)

**For output layers:**
- Binary classification: **Sigmoid** (outputs valid probability in [0, 1])
- Multi-class classification: **Softmax** (outputs probability distribution summing to 1)
- Regression: **None** (linear output, no activation)

**Computational cost matters:**
- ReLU: 1× (baseline, just comparisons)
- GELU: 4-5× (exponential in approximation)
- Sigmoid/Tanh: 3-4× (exponentials)
- Softmax: 5×+ (exponentials + normalization)

For a 1 billion parameter model, using GELU instead of ReLU in every hidden layer might increase training time by 20-30%. But if GELU gives you 2% better accuracy, that trade-off is worth it for production systems where model quality matters more than training speed.

### Computational Complexity

All activation functions are element-wise operations, meaning they apply independently to each element of the tensor. This gives O(n) time complexity where n is the total number of elements. But the constant factors differ dramatically:

| Operation | Complexity | Cost Relative to ReLU |
|-----------|------------|----------------------|
| ReLU (`max(0, x)`) | O(n) comparisons | 1× (baseline) |
| Sigmoid/Tanh | O(n) exponentials | 3-4× |
| GELU | O(n) exponentials + multiplies | 4-5× |
| Softmax | O(n) exponentials + O(n) sum + O(n) divisions | 5×+ |

Exponentials are expensive. A modern CPU can execute 1 billion comparisons per second but only 250 million exponentials per second. This is why ReLU is so popular: at scale, a 4× speedup in activation computation can mean the difference between training in 1 day versus 4 days.

Memory complexity is O(n) for all activations because they create an output tensor the same size as the input. Softmax requires small temporary buffers for the exponentials and sum, but this overhead is negligible compared to the tensor sizes in production networks.

## Production Context

### Your Implementation vs. PyTorch

Your TinyTorch activations and PyTorch's `torch.nn.functional` activations implement the same mathematical functions with the same numerical stability measures. The differences are in optimization and GPU support:

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Backend** | NumPy (Python/C) | C++/CUDA kernels |
| **Speed** | 1× (CPU baseline) | 10-100× faster (GPU) |
| **Numerical Stability** | ✓ Max subtraction (Softmax), clipping (Sigmoid) | ✓ Same techniques |
| **Autograd** | Stubs (Module 06) | Full gradient computation |
| **Variants** | 5 core activations | 30+ variants (LeakyReLU, PReLU, Mish, etc.) |

### Code Comparison

The following comparison shows equivalent activation usage in TinyTorch and PyTorch. Notice how the APIs are nearly identical, differing only in import paths and minor syntax.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.core.activations import ReLU, Sigmoid, Softmax
from tinytorch.core.tensor import Tensor

# Element-wise activations
x = Tensor([[-1, 0, 1, 2]])
relu = ReLU()
activated = relu(x)  # [0, 0, 1, 2]

# Binary classification output
sigmoid = Sigmoid()
probability = sigmoid(x)  # All values in (0, 1)

# Multi-class classification output
logits = Tensor([[1, 2, 3]])
softmax = Softmax()
probs = softmax(logits)  # [0.09, 0.24, 0.67], sum = 1
```
````

````{tab-item} ⚡ PyTorch
```python
import torch
import torch.nn.functional as F

# Element-wise activations
x = torch.tensor([[-1, 0, 1, 2]], dtype=torch.float32)
activated = F.relu(x)  # [0, 0, 1, 2]

# Binary classification output
probability = torch.sigmoid(x)  # All values in (0, 1)

# Multi-class classification output
logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)
probs = F.softmax(logits, dim=-1)  # [0.09, 0.24, 0.67], sum = 1
```
````
`````

Let's walk through the key similarities and differences:

- **Line 1 (Import)**: TinyTorch imports activation classes; PyTorch uses functional interface `torch.nn.functional`. Both approaches work; PyTorch also supports class-based activations via `torch.nn.ReLU()`.
- **Line 4-6 (ReLU)**: Identical semantics. Both zero out negative values, preserve positive values.
- **Line 9-10 (Sigmoid)**: Identical mathematical function. Both use numerically stable implementations to prevent overflow.
- **Line 13-15 (Softmax)**: Same mathematical operation. Both require specifying the dimension for multi-dimensional tensors. PyTorch uses `dim` keyword argument; TinyTorch defaults to `dim=-1`.

```{tip} What's Identical

Mathematical functions, numerical stability techniques (max subtraction in softmax), and the concept of element-wise transformations. When you debug PyTorch activation issues, you'll understand exactly what's happening because you implemented the same logic.
```

### Why Activations Matter at Scale

To appreciate why activation choice matters, consider the scale of modern ML systems:

- **Large language models**: GPT-3 has 96 transformer layers, each with 2 GELU activations. That's **192 GELU operations per forward pass** on billions of parameters.
- **Image classification**: ResNet-50 has 49 convolutional layers, each followed by ReLU. Processing a batch of 256 images at 224×224 resolution means **12 billion ReLU operations** per batch.
- **Production serving**: A model serving 1000 requests per second performs **86 million activation computations per day**. A 20% speedup from ReLU vs GELU saves hours of compute time.

Activation functions account for **5-15% of total training time** in typical networks (the rest is matrix multiplication). But in transformer models with many layers and small matrix sizes, activations can account for **20-30% of compute time**. This is why GELU vs ReLU is a real trade-off: slower computation but potentially better accuracy.

## Check Your Understanding

Test yourself with these systems thinking questions. They're designed to build intuition for how activations behave in real neural networks.

**Q1: Memory Calculation**

A batch of 32 samples passes through a hidden layer with 4096 neurons and ReLU activation. How much memory is required to store the activation outputs (float32)?

```{admonition} Answer
:class: dropdown

32 × 4096 × 4 bytes = **524,288 bytes ≈ 512 KB**

This is the activation memory for ONE layer. A 100-layer network needs 50 MB just to store activations for one forward pass. This is why activation memory dominates training memory usage (you'll see this in Module 06 when you cache activations for backpropagation).
```

**Q2: Computational Cost**

If ReLU takes 1ms to activate 1 million neurons, approximately how long will GELU take on the same input?

```{admonition} Answer
:class: dropdown

GELU is approximately **4-5× slower** than ReLU due to exponential computation in the sigmoid approximation.

Expected time: **4-5ms**

At scale, this matters: if you have 100 activation layers in your model, switching from ReLU to GELU adds 300-400ms per forward pass. For training that requires millions of forward passes, this multiplies into hours or days of extra compute time.
```

**Q3: Numerical Stability**

Why does softmax subtract the maximum value before computing exponentials? What would happen without this step?

```{admonition} Answer
:class: dropdown

**Without max subtraction**: Computing `softmax([1000, 1001, 1002])` requires `exp(1000)`, which overflows to infinity in float32/float64, producing NaN.

**With max subtraction**: First compute `x_shifted = x - max(x) = [0, 1, 2]`, then compute `exp([0, 1, 2])` which stays within float range.

**Why this works mathematically**:
```
exp(x - max) / Σ exp(x - max) = [exp(x) / exp(max)] / [Σ exp(x) / exp(max)]
                                = exp(x) / Σ exp(x)
```

The `exp(max)` factor cancels out, so the result is mathematically identical. But numerically, it prevents overflow. This is a classic example of why production ML requires careful numerical engineering, not just correct math.
```

**Q4: Sparsity Analysis**

A ReLU layer processes input tensor with shape (128, 1024) containing values drawn from a normal distribution N(0, 1). Approximately what percentage of outputs will be exactly zero?

```{admonition} Answer
:class: dropdown

For a standard normal distribution N(0, 1), approximately **50% of values are negative**.

ReLU zeros all negative values, so approximately **50% of outputs will be exactly zero**.

Total elements: 128 × 1024 = 131,072
Zeros: ≈ 65,536

This sparsity has major implications:
- **Speed**: Multiplying by zero is free, so downstream computations can skip ~50% of operations
- **Memory**: Sparse formats can compress the output by 2×
- **Generalization**: Sparse representations often generalize better (less overfitting)

This is why ReLU is so effective: it creates natural sparsity without requiring explicit regularization.
```

**Q5: Activation Selection**

You're building a sentiment classifier that outputs "positive" or "negative". Which activation should you use for the output layer, and why?

```{admonition} Answer
:class: dropdown

**Use Sigmoid** for the output layer.

**Reasoning**:
- Binary classification needs a single probability value in [0, 1]
- Sigmoid maps any real number to (0, 1)
- Output can be interpreted as P(positive) where 0.8 means "80% confident this is positive"
- Decision rule: predict positive if sigmoid(output) > 0.5

**Why NOT other activations**:
- **Softmax**: Overkill for binary classification (designed for multi-class), though technically works with 2 outputs
- **ReLU**: Outputs unbounded positive values, not interpretable as probabilities
- **Tanh**: Outputs in (-1, 1), not directly interpretable as probabilities

**Production pattern**:
```
Input → Linear + ReLU → Linear + ReLU → Linear + Sigmoid → Binary Probability
```

For multi-class sentiment (positive/negative/neutral), you'd use Softmax instead to get a 3-element probability distribution.
```

## Further Reading

For students who want to understand the academic foundations and historical development of activation functions:

### Seminal Papers

- **Deep Sparse Rectifier Neural Networks** - Glorot, Bordes, Bengio (2011). The paper that established ReLU as the default activation for deep networks, showing how its sparsity and constant gradient enable training of very deep networks. [AISTATS](http://proceedings.mlr.press/v15/glorot11a.html)

- **Gaussian Error Linear Units (GELUs)** - Hendrycks & Gimpel (2016). Introduced the smooth activation that powers modern transformers like GPT and BERT. Explains the probabilistic interpretation and why smoothness helps optimization. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)

- **Attention Is All You Need** - Vaswani et al. (2017). While primarily about transformers, this paper's use of specific activations (ReLU in position-wise FFN, Softmax in attention) established patterns still used today. [NeurIPS](https://arxiv.org/abs/1706.03762)

### Additional Resources

- **Textbook**: "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter 6.3 covers activation functions with mathematical rigor
- **Blog**: [Understanding Activation Functions](https://mlu-explain.github.io/relu/) - Amazon's MLU visual explanation of ReLU

## What's Next

```{seealso} Coming Up: Module 03 - Layers

Implement Linear layers that combine your Tensor operations with your activation functions. You'll build the building blocks that stack to form neural networks: weights, biases, and the forward pass that transforms inputs to outputs.
```

**Preview - How Your Activations Get Used in Future Modules:**

| Module | What It Does | Your Activations In Action |
|--------|--------------|---------------------------|
| **03: Layers** | Neural network building blocks | `Linear(x)` followed by `ReLU()(output)` |
| **04: Losses** | Training objectives | Softmax probabilities feed into cross-entropy loss |
| **06: Autograd** | Automatic gradients | `relu.backward(grad)` computes activation gradients |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/02_activations/02_activations.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/02_activations/02_activations.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
