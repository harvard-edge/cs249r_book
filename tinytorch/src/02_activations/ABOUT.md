# Activations

**FOUNDATION TIER** | Difficulty: ● (1/4) | Time: 3-4 hours | Prerequisites: Module 01

This module builds directly on Module 01 (Tensor). You should be comfortable with:
- Creating and manipulating Tensors
- Broadcasting semantics
- Element-wise operations

If you can create a Tensor and perform arithmetic on it, you're ready.

## Overview

Every neural network needs activation functions to learn complex patterns. Without them, stacking layers would be mathematically equivalent to having a single layer. No matter how deep your network, linear transformations composed together are still just linear transformations. Activation functions introduce the non-linearity that lets networks curve, bend, and approximate any function.

In this module, you'll build five essential activation functions: Sigmoid, ReLU, Tanh, GELU, and Softmax. Each serves a different purpose in neural networks, from gating probabilities to creating sparsity to producing probability distributions. By implementing them yourself, you'll understand exactly what happens when you call `torch.relu()` or `torch.softmax()` in production code.

These activations operate on the Tensor class you built in Module 01. They take a Tensor as input and return a new Tensor with the transformed values. This pattern of Tensor-in, Tensor-out is fundamental to how neural networks work.

## Learning Objectives

```{admonition} By completing this module, you will:
:class: tip

- **Implement** five core activation functions (Sigmoid, ReLU, Tanh, GELU, Softmax) with proper numerical stability
- **Understand** why non-linearity is essential for neural network expressiveness
- **Analyze** computational costs and numerical stability considerations for each activation
- **Connect** your implementations to production usage patterns in PyTorch and modern architectures
```

## What You'll Build

```{mermaid}
flowchart LR
    subgraph "Your Activation Functions"
        A["Sigmoid<br/>(0, 1) range"]
        B["ReLU<br/>zeros negatives"]
        C["Tanh<br/>(-1, 1) range"]
        D["GELU<br/>smooth ReLU"]
        E["Softmax<br/>probabilities"]
    end

    T["Input Tensor"] --> A & B & C & D & E
    A & B & C & D & E --> O["Output Tensor"]

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#d4edda
    style D fill:#f8d7da
    style E fill:#e2d5f1
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `Sigmoid.forward()` | Squashing to (0, 1) for probabilities |
| 2 | `ReLU.forward()` | Zeroing negatives for sparsity |
| 3 | `Tanh.forward()` | Zero-centered outputs in (-1, 1) |
| 4 | `GELU.forward()` | Smooth approximation for transformers |
| 5 | `Softmax.forward()` | Converting to probability distribution |

**The pattern you'll enable:**
```python
# Applying non-linearity to tensor data
relu = ReLU()
activated = relu(x)  # x is a Tensor, activated is a new Tensor
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Backward pass (that's Module 05: Autograd)
- Learnable parameters (activations are parameter-free)
- GPU-optimized kernels (PyTorch uses CUDA implementations)
- Variants like LeakyReLU, ELU, Swish (we focus on the core five)

**You are building the forward transformations.** Gradient computation comes in Module 05.

## API Reference

This section provides a quick reference for the activation classes you'll build. Each activation follows the same interface: instantiate, then call with a Tensor.

### Common Interface

All activations implement:

```python
class Activation:
    def forward(self, x: Tensor) -> Tensor: ...
    def __call__(self, x: Tensor) -> Tensor: ...  # Delegates to forward()
    def parameters(self) -> list: ...  # Returns [] (no learnable parameters)
```

### Activation Functions

| Class | Formula | Output Range | Primary Use |
|-------|---------|--------------|-------------|
| `Sigmoid` | 1/(1 + e^(-x)) | (0, 1) | Binary classification, gates |
| `ReLU` | max(0, x) | [0, +inf) | Hidden layers (default) |
| `Tanh` | (e^x - e^(-x))/(e^x + e^(-x)) | (-1, 1) | Hidden layers, RNNs |
| `GELU` | x * sigmoid(1.702x) | (-inf, +inf) | Transformers |
| `Softmax` | e^(x_i) / sum(e^(x_j)) | (0, 1), sum=1 | Multi-class output |

## Core Concepts

This section explains the fundamental ideas behind activation functions. Understanding these concepts will help you make informed choices when building neural networks.

### Why Non-linearity Matters

Consider what happens when you stack linear transformations. If layer 1 computes `W1 @ x + b1` and layer 2 computes `W2 @ y + b2`, the combined result is `W2 @ (W1 @ x + b1) + b2 = (W2 @ W1) @ x + (W2 @ b1 + b2)`. This is still just a linear transformation! You could replace both layers with a single layer and get the same result.

Activation functions break this equivalence. When you apply ReLU between layers, you're introducing non-linear behavior that can't be collapsed. Each layer can now learn something that the previous layers couldn't represent. This is why deep networks can approximate complex functions: they're not just stacking linear maps, they're building up increasingly sophisticated non-linear transformations.

The mathematical proof is elegant: without non-linearity, an N-layer network has the same representational power as a 1-layer network. With non-linearity, representational power grows with depth.

### Output Ranges and Use Cases

Each activation maps inputs to a specific output range, and this determines where you use it.

Sigmoid outputs values in (0, 1), making it perfect for probabilities. When you need to predict "spam or not spam," sigmoid gives you a probability like 0.87. But sigmoid saturates for large inputs (both positive and negative), which can slow down learning in hidden layers.

ReLU outputs values in [0, +inf), zeroing out negatives. This creates sparsity: many neurons output exactly zero, which makes computation more efficient and can help prevent overfitting. ReLU is the default choice for hidden layers in most networks.

Tanh outputs values in (-1, 1), similar to sigmoid but centered around zero. This zero-centering can help with the flow of information through networks, especially recurrent networks where the same weights are applied repeatedly.

GELU is like a smooth version of ReLU. Instead of the sharp corner at zero, it curves gently. This smoothness helps with optimization in transformers, which is why GPT and BERT use GELU.

Softmax converts any vector to a probability distribution where all values are positive and sum to 1. It's essential for multi-class classification: given 1000 ImageNet classes, softmax gives you probabilities for each.

### Numerical Stability

Naive implementations of activations can fail catastrophically with extreme inputs. Consider sigmoid with x = 1000: computing e^(-1000) underflows to 0.0, and computing e^(1000) overflows to infinity. Your implementation handles this by clipping inputs and using numerically stable formulas.

Softmax is particularly tricky. Computing e^(1000) for one element while others are e^(1) would give infinity. The standard trick is to subtract the maximum value first: `softmax(x) = softmax(x - max(x))`. This keeps all exponentials in a safe range without changing the result.

These stability tricks are essential for production code. A model that works on small test inputs but crashes on real data is useless.

### Computational Cost

Not all activations are equal in speed. Understanding these costs matters when you're processing billions of activations per training step.

| Activation | Operations | Relative Cost |
|------------|------------|---------------|
| ReLU | 1 comparison per element | 1x (baseline) |
| Sigmoid | 1 exp + 1 division per element | 3-4x |
| Tanh | 2 exp + 1 division per element | 3-4x |
| GELU | 1 exp + 2 multiplications per element | 4-5x |
| Softmax | n exp + n-1 additions + n divisions | 5x+ |

ReLU's simplicity is one reason it became the default. When you apply an activation billions of times per training step, a 4x speedup matters. GELU's extra cost is worth it in transformers because the improved optimization outweighs the computational overhead.

## Common Errors

### Numerical Overflow in Sigmoid

**Symptom**: `RuntimeWarning: overflow encountered in exp`

**Cause**: Computing `exp(-x)` for very large positive x, or `exp(x)` for very large negative x.

**Fix**: Clip inputs to a safe range (approximately -500 to 500), or use the numerically stable formulation that handles positive and negative inputs separately.

### Softmax Dimension Confusion

**Symptom**: Probabilities don't sum to 1, or shape is wrong

**Cause**: Applying softmax along the wrong axis in multi-dimensional tensors

**Fix**: Always specify the `dim` parameter explicitly. For classification with shape `(batch, classes)`, use `dim=-1` to normalize across classes.

```python
# Wrong: softmax over entire tensor
probs = softmax(logits)  # Sum might not be 1 per sample

# Right: softmax over class dimension
probs = softmax(logits, dim=-1)  # Each row sums to 1
```

### Dead ReLU Neurons

**Symptom**: Some neurons always output 0 during training

**Cause**: Large negative input causes ReLU to output 0, gradient is also 0, so weights never update

**Fix**: This is a known issue with ReLU. Monitor the percentage of dead neurons. If too many die, consider using LeakyReLU (not implemented in this module) or reducing learning rate.

## Production Context

Your TinyTorch activations and PyTorch's `torch.nn.functional` activations share the same mathematical definitions. The differences are in implementation: PyTorch uses C++/CUDA kernels optimized for specific hardware, while yours use NumPy operations.

### Your Implementation vs. PyTorch

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Backend** | NumPy (Python) | C++/CUDA kernels |
| **Speed** | 1x (baseline) | 5-20x faster |
| **GPU** | CPU only | CUDA, Metal, ROCm |
| **Fused ops** | Separate operations | Fused kernels (e.g., bias + ReLU) |
| **Autograd** | forward() only | Full backward support |

### Code Comparison

The following comparison shows equivalent operations in TinyTorch and PyTorch. Notice that the API and behavior are identical; only the import changes.

`````{tab-set}
````{tab-item} 🔥 Your TinyTorch
```python
from tinytorch import Tensor
from tinytorch.core.activations import ReLU, Softmax

x = Tensor([[-1, 0, 1], [2, -2, 3]])
relu = ReLU()
activated = relu(x)  # [0, 0, 1], [2, 0, 3]

logits = Tensor([[1, 2, 3]])
softmax = Softmax()
probs = softmax(logits)  # [0.09, 0.24, 0.67]
```
````

````{tab-item} ⚡ PyTorch
```python
import torch
import torch.nn.functional as F

x = torch.tensor([[-1, 0, 1], [2, -2, 3]], dtype=torch.float32)
activated = F.relu(x)  # Same result!

logits = torch.tensor([[1, 2, 3]], dtype=torch.float32)
probs = F.softmax(logits, dim=-1)  # Same result!
```
````
`````

Let's walk through each operation to understand the comparison:

- **Lines 1-2 (Import)**: TinyTorch organizes activations in `core.activations`; PyTorch provides them as both `torch.nn` modules and `torch.nn.functional` functions. The functional interface (`F.relu`) is more common for stateless activations.
- **Lines 4-6 (ReLU)**: TinyTorch uses class instantiation then calling; PyTorch's functional interface is a single function call. Both produce identical output: negative values become 0, positive values unchanged.
- **Lines 8-10 (Softmax)**: Both normalize the input to a probability distribution. Note that PyTorch requires explicit `dim` specification, which is good practice for avoiding bugs.

```{admonition} What's Identical
:class: tip

The mathematical transformations, output ranges, and numerical stability considerations. When you understand how your ReLU handles negative values, you understand exactly what PyTorch's ReLU does. The only difference is speed.
```

### Why Activations Matter at Scale

Activation functions are applied to every neuron in every layer. Consider the scale:

- **GPT-3**: 175 billion parameters means billions of activation function calls per forward pass
- **ResNet-152**: 60 million activations per image, multiplied by batch size
- **Real-time inference**: 30+ frames per second requires activations to complete in microseconds

ReLU's simplicity is a competitive advantage. In a network with 1 billion parameters, using ReLU instead of GELU saves approximately 3 billion floating-point operations per forward pass. At scale, this translates to measurable time and energy savings.

## Check Your Understanding

Test yourself with these questions. They're designed to build intuition for activation behavior and computational characteristics.

**Q1: Output Range Prediction**

What is the output range of `Sigmoid(Tensor([-1000, 0, 1000]))`?

```{admonition} Answer
:class: dropdown

Output: approximately `[0, 0.5, 1]`

Sigmoid(-1000) approaches 0 (but never reaches it)
Sigmoid(0) = exactly 0.5
Sigmoid(1000) approaches 1 (but never reaches it)

The output is always in the open interval (0, 1), never exactly 0 or 1.
```

**Q2: Computational Cost**

A network has 10 hidden layers, each with 1 million neurons. How many more floating-point operations does GELU require compared to ReLU for a single forward pass?

```{admonition} Answer
:class: dropdown

ReLU: 1 operation per element = 10 million operations total
GELU: ~4-5 operations per element = 40-50 million operations total

Difference: **30-40 million extra operations**

For 10 layers: 300-400 million extra operations per forward pass!
```

**Q3: Softmax Properties**

Given input `[10, 10, 10]`, what is the softmax output?

```{admonition} Answer
:class: dropdown

Output: `[0.333..., 0.333..., 0.333...]` (equal probabilities)

When all inputs are equal, softmax produces a uniform distribution regardless of the actual values. This is because `e^10 / (3 * e^10) = 1/3` for each element.

The same result would occur for `[0, 0, 0]` or `[100, 100, 100]`.
```

**Q4: Memory for Activation Caching**

ResNet-50 applies ReLU to approximately 23 million elements per forward pass for a single image. During training with batch size 32, how much memory is required just to cache these activations for backpropagation (assuming float32)?

```{admonition} Answer
:class: dropdown

23 million elements × 32 batch × 4 bytes = **2.94 GB**

This is why activation memory often dominates GPU memory usage during training, and why techniques like gradient checkpointing (recomputing activations instead of storing them) are used for very large models.
```

## Further Reading

For students who want to understand the academic foundations and evolution of activation functions:

### Seminal Papers

- **Deep Sparse Rectifier Neural Networks** - Glorot, Bordes, Bengio (2011). The paper that established ReLU as the default activation for deep networks, showing how its sparsity and constant gradient enable training of very deep networks. [AISTATS](http://proceedings.mlr.press/v15/glorot11a.html)

- **Gaussian Error Linear Units (GELUs)** - Hendrycks & Gimpel (2016). Introduced the smooth activation that powers modern transformers like GPT and BERT. Explains the probabilistic interpretation and why smoothness helps optimization. [arXiv](https://arxiv.org/abs/1606.08415)

- **Attention Is All You Need** - Vaswani et al. (2017). While primarily about transformers, this paper's use of specific activations (ReLU in position-wise FFN, Softmax in attention) established patterns still used today. [NeurIPS](https://arxiv.org/abs/1706.03762)

### Additional Resources

- **Textbook**: "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter 6.3 covers activation functions with mathematical rigor
- **Blog**: [Understanding Activation Functions](https://mlu-explain.github.io/relu/) - Amazon's MLU visual explanation of ReLU

## What's Next

```{admonition} Coming Up: Module 03 - Layers
:class: seealso

Implement Linear layers that combine your Tensor operations with your activation functions. You'll build the building blocks that stack to form neural networks: weights, biases, and the forward pass that transforms inputs to outputs.
```

**Preview - How Your Activations Get Used in Future Modules:**

| Module | What It Does | Your Activations In Action |
|--------|--------------|---------------------------|
| **03: Layers** | Neural network building blocks | `Linear(x)` followed by `ReLU()(output)` |
| **04: Losses** | Training objectives | Softmax + cross-entropy for classification |
| **05: Autograd** | Automatic gradients | `ReLU.backward()` computes gradients |

## Get Started

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=src/02_activations/02_activations.py
:class-header: bg-light

Run interactively in browser - no setup required
```

```{grid-item-card} ☁️ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/src/02_activations/02_activations.py
:class-header: bg-light

Use Google Colab for cloud compute
```

```{grid-item-card} 📄 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/src/02_activations/02_activations.py
:class-header: bg-light

Browse the implementation code
```

````

```{admonition} Save Your Progress
:class: warning

Binder and Colab sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
