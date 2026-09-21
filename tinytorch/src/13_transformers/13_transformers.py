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
# Module 13: Transformers - Complete Transformer Architecture

Welcome to Module 13! You're about to synthesize everything you've built across TinyTorch into the complete Pre-LayerNorm Transformer architecture that powers modern autoregressive language models (GPT, Claude, Llama, and Mistral).

## 🔗 Prerequisites & Progress
**You've Built**: Autograd engine (`Tensor`), non-linear activations (`GELU`), feed-forward primitives (`Linear`), subword tokenization (`BPETokenizer`), positional embeddings (`EmbeddingLayer`), and multi-head attention (`MultiHeadAttention`).
**You'll Build**: Numerically stable layer normalization (`LayerNorm`), 4× expansion feed-forward networks (`MLP`), complete residual blocks (`TransformerBlock`), and standalone autoregressive generation utilities (`sample_next_token`, `generate`).
**You'll Enable**: Profiling and roofline optimization (`14_profiling`), post-training quantization (`15_quantization`), and key-value memory caching (`18_memoization`).

<div align="center">
  <img src="transformer_blueprint.svg" alt="TinyTorch Architecture Blueprint: Module 13 Transformers" width="380px">
</div>

### Architectural Roadmap

| Stage | Subsystem | Primitives & Capabilities | Status |
|:---|:---|:---|:---|
| **Modules 01–08** | Foundation & Training | `Tensor`, `Function`, `Linear`, `GELU`, `SGD`, `Adam`, `Trainer` | Completed |
| **Modules 09–12** | Representations & Attention | `Conv2d`, `BPETokenizer`, `EmbeddingLayer`, `MultiHeadAttention` | Completed |
| **Module 13** | **Transformer Stack & Generation** | `LayerNorm`, `MLP`, `TransformerBlock`, `sample_next_token`, `generate` | **Active Subsystem** |
| **Modules 14–20** | Systems & Acceleration | `Profiler`, `INT8Linear`, `KVCache`, `TinyGPT` | Downstream Consumers |

## 🎯 Learning Objectives
By the end of this module, you will:
1. **Implement Numerically Stable Layer Normalization**: Standardize per-token feature vectors with a two-pass mean-then-variance computation and learnable affine restoration parameters ($\gamma, \beta$).
2. **Construct Two-Layer MLP Expansion Blocks**: Build a $d_{\text{embed}} \to 4d_{\text{embed}} \to d_{\text{embed}}$ feed-forward channel projection with smooth GELU gating.
3. **Assemble the Pre-LN Transformer Block**: Route clean residual highway streams through attention and MLP off-ramps to preserve unobstructed gradient backpropagation.
4. **Implement Autoregressive Generation & Sampling**: Build standalone token sampling with temperature scaling and autoregressive decoding loops.
5. **Analyze LLM Parameter Scaling & Memory Footprints**: Calculate exact parameter allocations across attention and FFN layers and derive the quadratic memory scaling law.

---

## 📦 Where This Code Lives in the Final Package

| Artifact | Path / Location | Export Target | Primary Symbols |
|:---|:---|:---|:---|
| **Notebook** | `modules/13_transformers/transformers.ipynb` | Interactive Learning | Exploratory code, assertions, benchmarks |
| **Core Source** | `src/13_transformers/13_transformers.py` | `tinytorch.core.transformers` | `LayerNorm`, `MLP`, `TransformerBlock`, `create_causal_mask`, `sample_next_token`, `generate` |

```python
# How to use this module:
from tinytorch.core.transformers import LayerNorm, MLP, TransformerBlock, create_causal_mask, sample_next_token, generate
```

---

## 📋 Module Dependencies

| Dependency | Origin | Functionality Utilized | Role in Module 13 |
|:---|:---|:---|:---|
| `tinytorch.core.tensor` | Module 01 | Multi-dimensional arrays & autograd | Base data structure and computation graphs |
| `tinytorch.core.activations`| Module 02 | `GELU` activation function | Non-linear gating inside the MLP expansion layer |
| `tinytorch.core.layers` | Module 03 | `Linear` projection layers | Projection weights for MLP and output LM head |
| `tinytorch.core.embeddings`| Module 11 | `EmbeddingLayer` | Converts token IDs to dense representation vectors |
| `tinytorch.core.attention` | Module 12 | `MultiHeadAttention` | Subspace relational routing across sequence positions |
| `numpy` | External | High-performance array operations | Random number generation, moment reductions, and masking |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp core.transformers
#| export

import numpy as np
from typing import Any, List, Optional, Tuple

rng = np.random.default_rng(7)

# Import from previous modules - following the dependency chain
from tinytorch.core.tensor import Tensor, Function
from tinytorch.core.activations import GELU
from tinytorch.core.layers import Linear
from tinytorch.core.embeddings import EmbeddingLayer
from tinytorch.core.attention import MultiHeadAttention

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
MB_TO_BYTES = 1024 * 1024  # Binary megabyte (mebibyte) in bytes, so printed values read MiB

# %% [markdown]
r"""
## 💡 Introduction: What are Transformers?

Transformers are the architecture behind modern large language models (GPT-4, Claude, Llama, and Gemini). The core breakthrough is the synthesis of **parallel self-attention** (allowing every token in a sequence to dynamically attend to every other token) with **deep residual stream communication** and **dense feed-forward computation**.

### The Transformer Paradigm Shift

Before transformers, sequence modeling relied on recurrent neural networks (RNNs, LSTMs) that processed text sequentially ($\mathcal{O}(S)$ sequential steps), creating severe training bottlenecks and vanishing gradient problems across long distances. Transformers eliminate temporal recurrence, processing all $S$ tokens simultaneously via matrix multiplications ($\mathcal{O}(1)$ sequential depth per layer).

### Complete GPT Architecture Overview

<div align="center">
  <img src="transformer_block_gpt_pipeline.svg" alt="Pre-LN Transformer Block and Complete GPT Forward Pipeline" width="700px">
</div>

#### End-to-End GPT Execution Pipeline

| Pipeline Stage | Subsystem / Layer | Input Shape | Internal Transformation | Output Shape | Systems & Modeling Role |
|:---|:---|:---:|:---|:---:|:---|
| **1. Embedding** | `EmbeddingLayer` | $(B, S)$ | Token index gather + learned positional vector addition | $(B, S, d_{\text{embed}})$ | Converts discrete vocabulary IDs to continuous space |
| **2. Pre-LN Attention** | `LayerNorm` + `MHA` | $(B, S, d_{\text{embed}})$ | Normalize features $\to$ Causal attention $\to$ Add residual | $(B, S, d_{\text{embed}})$ | Routes context dynamically across sequence positions |
| **3. Pre-LN FFN** | `LayerNorm` + `MLP` | $(B, S, d_{\text{embed}})$ | Normalize features $\to 4\times$ GELU expansion $\to$ Add residual | $(B, S, d_{\text{embed}})$ | Pointwise non-linear feature transformation & memory |
| **4. Final Norm** | `LayerNorm` | $(B, S, d_{\text{embed}})$ | Feature standardization across last dimension | $(B, S, d_{\text{embed}})$ | Stabilizes activations prior to un-embedding projection |
| **5. LM Head** | `Linear` (un-embedding) | $(B, S, d_{\text{embed}})$ | Matrix multiplication $x W_{\text{vocab}}^\top$ | $(B, S, V)$ | Produces unnormalized next-token vocabulary logits |

---

## 📐 Foundations: Essential Transformer Mathematics

### Layer Normalization: The Numerical Stability Engine

Deep networks become badly conditioned when activation magnitudes drift across layers, because the loss surface then stretches along some directions and flattens along others, and a single learning rate cannot suit both. (The original account of this, internal covariate shift, was empirically refuted as the mechanism by Santurkar et al., 2018.) In vision architectures, Batch Normalization standardizes activations across the batch dimension. However, language models process sequences of highly variable lengths, and single-token autoregressive inference operates at batch size $B=1$.

**Layer Normalization** (Ba, Kiros, & Hinton, 2016) resolves this by computing moments strictly across the **feature dimension** ($d_{\text{embed}}$) for each token position independently:

#### Mathematical Formulation

For token vector $x \in \mathbb{R}^{d_{\text{embed}}}$:

$$\mu = \frac{1}{d_{\text{embed}}} \sum_{i=1}^{d_{\text{embed}}} x_i, \qquad \sigma^2 = \frac{1}{d_{\text{embed}}} \sum_{i=1}^{d_{\text{embed}}} (x_i - \mu)^2$$

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y = \gamma \odot \hat{x} + \beta$$

where $\gamma, \beta \in \mathbb{R}^{d_{\text{embed}}}$ are learnable scale and shift parameters initialized to $\mathbf{1}$ and $\mathbf{0}$.

| Normalization Paradigm | Normalization Axis | Batch Size Dependence | Sequence Length Dependence | Inference Suitability |
|:---|:---|:---:|:---:|:---|
| **Batch Normalization (BatchNorm)** | Across batch $(B)$ | High (fails if $B=1$) | Requires fixed padding | Requires tracked running statistics ($\mu_{\text{run}}, \sigma^2_{\text{run}}$) |
| **Layer Normalization (LayerNorm)** | Across features $(d_{\text{embed}})$ | **None** ($B=1$ valid) | **None** (per-token) | **Identical to training** (exact sample statistics) |

---

### Residual Connections: The Gradient Highway System

Deep networks without skip connections suffer from catastrophic gradient decay. Backpropagating through $L$ consecutive layers multiplies Jacobian matrices $\prod_{l=0}^{L-1} W_l$, driving gradients exponentially to zero ($\to 0$) or infinity ($\to \infty$).

<div align="center">
  <img src="residual_stream_bus.svg" alt="Residual Highway State Bus and Gradient Flow" width="700px">
</div>

#### The Pre-LN Residual Highway Invariant

In modern Pre-LN transformer blocks, each sublayer reads from the continuous residual stream, computes an update delta, and adds it back to the stream:

$$x_{l+1} = x_l + F_l(\text{LayerNorm}(x_l))$$

Unrolling this recurrence across all $L$ layers yields:

$$x_L = x_0 + \sum_{l=0}^{L-1} F_l(\text{LayerNorm}(x_l))$$

When differentiating with respect to early representation $x_0$, the chain rule produces:

$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \frac{\partial x_L}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum_{l=0}^{L-1} \frac{\partial F_l}{\partial x_l} \right)$$

<div align="center">
  <img src="transformer_pre_ln_highway.svg" alt="Pre-LN Clean Skip Connection" width="300px">
</div>

Because of the identity matrix $I$, the error signal propagates directly from the loss function $\mathcal{L}$ back to initial embeddings $x_0$ **without vanishing**, allowing architectures dozens of layers deep (GPT-3 175B stacks 96) to converge reliably.

---

### Feed-Forward Network (MLP): Channel Expansion & Memory Retrieval

While self-attention routes information across different sequence positions, the Feed-Forward Network (MLP) operates on each token position independently. It provides non-linear representation capacity and acts as key-value associative memory (Geva et al., 2021).

<div align="center">
  <img src="mlp_expansion_funnel.svg" alt="Feed-Forward Network (MLP) Two-Stage Expansion Funnel" width="680px">
</div>

#### Two-Stage Expansion Architecture

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

where $W_1 \in \mathbb{R}^{d_{\text{embed}} \times 4d_{\text{embed}}}$ expands representations into a higher-dimensional manifold, and $W_2 \in \mathbb{R}^{4d_{\text{embed}} \times d_{\text{embed}}}$ contracts them back to the residual bus dimension:

| Sub-Operation | Tensor Input | Parameter Dimension | Tensor Output | Mathematical Operation |
|:---|:---:|:---:|:---:|:---|
| **Up-Projection (Linear 1)** | $(B, S, d_{\text{embed}})$ | $W_1 \in \mathbb{R}^{d \times 4d}, b_1 \in \mathbb{R}^{4d}$ | $(B, S, 4d_{\text{embed}})$ | $h_1 = x W_1 + b_1$ |
| **GELU Non-Linearity** | $(B, S, 4d_{\text{embed}})$ | None (parameterless) | $(B, S, 4d_{\text{embed}})$ | $a = x \cdot \Phi(x)$ |
| **Down-Projection (Linear 2)** | $(B, S, 4d_{\text{embed}})$ | $W_2 \in \mathbb{R}^{4d \times d}, b_2 \in \mathbb{R}^{d}$ | $(B, S, d_{\text{embed}})$ | $y = a W_2 + b_2$ |

---

### The Complete Pre-LN Transformer Block Data Flow

Combining Pre-LN attention and Pre-LN MLP yields the canonical Transformer Block datapath:

$$\begin{aligned}
x_1 &= \text{LayerNorm}_1(x_0) \\
x_2 &= x_0 + \text{MultiHeadAttention}(x_1, \text{mask}) \\
x_3 &= \text{LayerNorm}_2(x_2) \\
x_4 &= x_2 + \text{MLP}(x_3)
\end{aligned}$$

| Step | Operation | Source Input | Sublayer Applied | Residual Addition | Output State |
|:---:|:---|:---:|:---|:---:|:---:|
| **1** | Attention Pre-LN | $x_0$ | $x_1 = \text{LayerNorm}_1(x_0)$ | — | $x_1$ |
| **2** | Attention Skip | $x_0, x_1$ | $\Delta_{\text{attn}} = \text{MHA}(x_1, \text{mask})$ | $x_2 = x_0 + \Delta_{\text{attn}}$ | $x_2$ |
| **3** | MLP Pre-LN | $x_2$ | $x_3 = \text{LayerNorm}_2(x_2)$ | — | $x_3$ |
| **4** | MLP Skip | $x_2, x_3$ | $\Delta_{\text{mlp}} = \text{MLP}(x_3)$ | $x_4 = x_2 + \Delta_{\text{mlp}}$ | $x_4$ (Block Output) |
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Transformer Components

Now we'll implement each transformer component with a clear understanding of their role in the overall architecture. We'll follow the pattern: **Explanation → Implementation → Test** for each component.

Each component serves a specific purpose:
- **LayerNorm**: Stabilizes training and normalizes activations
- **MLP**: Provides non-linear transformation and "thinking" capacity
- **TransformerBlock**: Combines attention with MLP using residual connections
- **Autoregressive Generation**: Standalone token sampling and autoregressive decoding loops
"""

# %% [markdown]
r"""
### Understanding Layer Normalization

Layer Normalization provides the foundation of stable transformer optimization. In deep residual networks, signal variances compound across layers; without normalization, activation values explode or collapse into extreme saturating regions.

<div align="center">
  <img src="layernorm_feature_plane.svg" alt="Layer Normalization Feature-Axis Normalization and Affine Restoration" width="680px">
</div>

#### Three-Stage LayerNorm Formulation

1. **Feature-Axis Statistics**: Compute mean and variance along the trailing hidden dimension ($d_{\text{embed}}$) for each batch and sequence index $(b, s)$:
   $$\mu_{b, s} = \frac{1}{d_{\text{embed}}} \sum_{i=1}^{d_{\text{embed}}} x_{b, s, i}, \qquad \sigma^2_{b, s} = \frac{1}{d_{\text{embed}}} \sum_{i=1}^{d_{\text{embed}}} (x_{b, s, i} - \mu_{b, s})^2$$

2. **Standardization & Numerical Guardrail**: Subtract mean and scale by standard deviation, adding $\epsilon = 10^{-5}$ to prevent division by zero:
   $$\hat{x}_{b, s, i} = \frac{x_{b, s, i} - \mu_{b, s}}{\sqrt{\sigma^2_{b, s} + \epsilon}}$$

3. **Learnable Affine Restoration**: Allow the optimization process to recover arbitrary representation scales and biases:
   $$y_{b, s, i} = \gamma_i \cdot \hat{x}_{b, s, i} + \beta_i, \quad \text{where } \gamma, \beta \in \mathbb{R}^{d_{\text{embed}}}$$

#### Normalization Variants in Deep Learning

| Normalization Scheme | Normalization Axes | Learnable Params | Batch Size Dependence | Autoregressive Inference |
|:---|:---|:---:|:---:|:---|
| **BatchNorm** | Batch + Spatial $(B, H, W)$ | $2 \cdot C$ | Extreme (fails at $B=1$) | Incompatible with autoregressive generation |
| **LayerNorm** | Feature Axis $(d_{\text{embed}})$ | $2 \cdot d_{\text{embed}}$ | **Zero** ($B=1$ native) | Standard across GPT-2, GPT-3, BERT |
| **RMSNorm** | Feature Axis (Root-Mean-Square) | $1 \cdot d_{\text{embed}}$ ($\gamma$ only) | **Zero** ($B=1$ native) | Modern default (Llama 2/3, Mistral, Gemma) |
"""

# %% nbgrader={"grade": false, "grade_id": "layer-norm", "solution": true}
#| export


class LayerNormFunction(Function):
    """
    The layer normalization operation: forward normalizes across the last axis, backward
    computes the gradients.

    Computes gradients for x, gamma, and beta in one pass.
    output = gamma * ((x - mean) / std) + beta

    The gradient for x uses the standard LayerNorm formula, with gamma folded into the
    incoming gradient FIRST (the per-feature scale sits inside the two mean terms, since
    mean(gamma * grad) is not gamma * mean(grad) unless gamma is uniform):
        g = grad_output * gamma
        dx = (1/std) * (g - mean(g) - normalized * mean(g * normalized))
    """

    def forward(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization to a NumPy array.

        TODO: Implement the normalization formula

        APPROACH:
        1. Compute mean and variance across the last dimension
        2. Normalize: (x - mean) / sqrt(variance + eps)
        3. Keep the normalized values and std on self (backward needs them)
        4. Apply learnable scale and shift: gamma * normalized + beta

        MATHEMATICAL FORMULA:
        y = (x - μ) / σ * γ + β
        where μ = mean(x), σ = sqrt(var(x) + ε)

        HINT: Use keepdims=True to maintain tensor dimensions for broadcasting
        """
        ### BEGIN SOLUTION
        # Compute statistics across last dimension (features)
        mean_data = np.mean(x, axis=-1, keepdims=True)
        # Compute variance: E[(x - μ)²]
        diff = x - mean_data
        variance = np.mean(diff * diff, axis=-1, keepdims=True)
        # Normalize: (x - mean) / sqrt(variance + eps)
        self.std_data = np.sqrt(variance + self.eps)
        self.normalized_data = diff / self.std_data
        # Apply learnable transformation: gamma * normalized + beta
        return gamma * self.normalized_data + beta
        ### END SOLUTION


    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute gradients for LayerNorm (x, gamma, beta)."""
        x, gamma, beta = self.inputs

        grad_x = grad_gamma = grad_beta = None
        normalized = self.normalized_data
        std_data = self.std_data

        # Gradient for beta: sum over all dims except last
        if beta.requires_grad:
            # Sum over batch and sequence dimensions
            grad_beta = grad_output.copy()
            while grad_beta.ndim > 1:
                grad_beta = grad_beta.sum(axis=0)

        # Gradient for gamma: sum of (grad_output * normalized) over batch/seq dims
        if gamma.requires_grad:
            grad_gamma = (grad_output * normalized).copy()
            while grad_gamma.ndim > 1:
                grad_gamma = grad_gamma.sum(axis=0)

        # Gradient for x: full LayerNorm backward formula
        if x.requires_grad:
            # grad flowing through gamma: grad_output * gamma
            grad_norm = grad_output * gamma.data

            mean_grad = np.mean(grad_norm, axis=-1, keepdims=True)
            mean_grad_norm = np.mean(grad_norm * normalized, axis=-1, keepdims=True)
            grad_x = (1.0 / std_data) * (grad_norm - mean_grad - normalized * mean_grad_norm)

        return (grad_x, grad_gamma, grad_beta)


class LayerNorm:
    """
    Layer Normalization for transformer blocks.

    Normalizes across the feature dimension (last axis) for each sample independently,
    unlike batch normalization which normalizes across the batch dimension.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Initialize LayerNorm with learnable parameters.

        TODO: Set up normalization parameters

        APPROACH:
        1. Store the shape to normalize over (usually embed_dim)
        2. Initialize learnable scale (gamma) and shift (beta) parameters
        3. Set small epsilon for numerical stability

        EXAMPLE:
        >>> ln = LayerNorm(512)  # For 512-dimensional embeddings
        >>> x = Tensor(rng.standard_normal((2, 10, 512)))  # (batch, seq, features)
        >>> normalized = ln.forward(x)
        >>> # Each (2, 10) sample normalized independently across 512 features

        HINTS:
        - gamma should start at 1.0 (identity scaling)
        - beta should start at 0.0 (no shift)
        - eps prevents division by zero in variance calculation
        """
        ### BEGIN SOLUTION role="scaffold"
        if not isinstance(normalized_shape, (int, np.integer)) or normalized_shape <= 0:
            raise ValueError("TinyTorch LayerNorm normalizes one positive final dimension")
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters: scale and shift
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)  # Scale parameter
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)  # Shift parameter
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization.

        The normalization itself lives in LayerNormFunction.forward above;
        apply() records it so gamma and beta train.
        """
        if not x.shape or x.shape[-1] != self.normalized_shape:
            raise ValueError(f"LayerNorm expected final dimension {self.normalized_shape}, got {x.shape}")
        return LayerNormFunction.apply(x, self.gamma, self.beta, eps=self.eps)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the layer norm to be called like a function."""
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """Return learnable parameters."""
        return [self.gamma, self.beta]

# %% [markdown]
"""
### 🧪 Unit Test: Layer Normalization

This test validates our LayerNorm implementation works correctly.

**What we're testing**: Normalization statistics and parameter learning
**Why it matters**: Essential for transformer stability and training
**Expected**: Mean approximately 0, std approximately 1 after normalization, learnable parameters work
"""

# %% nbgrader={"grade": true, "grade_id": "test-layer-norm", "locked": true, "points": 10}
def test_unit_layer_norm():
    """🧪 Test LayerNorm implementation."""
    print("🧪 Unit Test: Layer Normalization...")

    # Test basic normalization
    ln = LayerNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # (2, 4)

    normalized = ln.forward(x)

    # Check output shape
    assert normalized.shape == (2, 4)

    # Check normalization properties (approximately)
    # For each sample, mean should be close to 0, std close to 1
    for i in range(2):
        sample_mean = np.mean(normalized.data[i])
        sample_std = np.std(normalized.data[i])
        assert abs(sample_mean) < 1e-5, f"Mean should be ~0, got {sample_mean}"
        assert abs(sample_std - 1.0) < 1e-4, f"Std should be ~1, got {sample_std}"

    # Test parameter shapes
    params = ln.parameters()
    assert len(params) == 2
    assert params[0].shape == (4,)  # gamma
    assert params[1].shape == (4,)  # beta

    print("✅ LayerNorm works correctly!")

if __name__ == "__main__":
    test_unit_layer_norm()

# %% [markdown]
r"""
### Understanding the Multi-Layer Perceptron (MLP)

The Multi-Layer Perceptron (also termed the Feed-Forward Network, FFN) provides the primary computational transformation in each transformer block. While self-attention allows tokens to exchange information across positions, the MLP processes each position completely independently, applying identical weights to every token vector.

#### Mathematical Formulation & Two-Layer Funnel

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

where input $x \in \mathbb{R}^{B \times S \times d_{\text{embed}}}$ is expanded to intermediate dimension $d_{\text{ff}} = 4 \cdot d_{\text{embed}}$ before being contracted back:

#### Detailed Parameter and FLOP Breakdown

| Sub-Layer Operation | Input Tensor | Weight Matrix | Bias Vector | Parameters ($d=512$) | Compute FLOPs / Token |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Linear 1 (Up-Projection)** | $(B, S, d)$ | $(d, 4d)$ | $(4d,)$ | $512 \times 2{,}048 + 2{,}048 = 1{,}050{,}624$ | $2 \cdot d \cdot 4d = 2{,}097{,}152$ |
| **GELU Activation** | $(B, S, 4d)$ | — | — | $0$ (activation) | $\approx 8 \cdot 4d = 16{,}384$ |
| **Linear 2 (Down-Projection)** | $(B, S, 4d)$ | $(4d, d)$ | $(d,)$ | $2{,}048 \times 512 + 512 = 1{,}049{,}088$ | $2 \cdot 4d \cdot d = 2{,}097{,}152$ |
| **Total MLP Sub-Layer** | — | — | — | $\mathbf{8d^2 + 5d \approx 2.10\text{M params}}$ | $\approx \mathbf{16 d^2 \text{ FLOPs}}$ |

**Why 4× Expansion Matters**:
- **Representation Capacity**: The MLP contains approximately $\frac{2}{3}$ of all parameters within each transformer block (compared to $4d^2$ in attention).
- **Associative Memory**: Empirical research (Geva et al., 2021) demonstrates that FFN weights function as key-value memories, where first-layer weights detect factual patterns and second-layer weights produce output distributions.

#### Non-Linear Activation Comparison: GELU vs ReLU

| Property | Rectified Linear Unit ($\text{ReLU}$) | Gaussian Error Linear Unit ($\text{GELU}$) |
|:---|:---|:---|
| **Mathematical Definition** | $\text{ReLU}(x) = \max(0, x)$ | $\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{2/\pi}(x + 0.044715 x^3)\right)\right)$ |
| **Curvature & Continuity** | Piecewise linear; non-differentiable cusp at $x = 0$ | Smooth, continuously differentiable across entire real domain $\mathbb{R}$ |
| **Negative Input Handling** | Hard zero clamp ($\frac{d}{dx} = 0$ for $x < 0$) | Small negative curvature; allows gradient recovery |
| **Dead Neuron Vulnerability** | High (negative activations permanently kill gradient flow) | Negligible (smooth probabilistic gating prevents dead units) |
| **Architectural Adoption** | Classical CNNs (AlexNet, ResNet) | Standard in GPT-2, GPT-3, BERT, and modern LLMs |
"""

# %% nbgrader={"grade": false, "grade_id": "mlp", "solution": true}
#| export
class MLP:
    """
    Multi-Layer Perceptron (Feed-Forward Network) for transformer blocks.

    Standard pattern: Linear -> GELU -> Linear with expansion ratio of 4:1.
    This provides the non-linear transformation in each transformer block.
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, dropout_prob: float = 0.0):
        """
        Initialize MLP with two linear layers.

        dropout_prob must be zero: this compact transformer omits dropout.
        Module 03 provides a standalone Dropout exercise.

        TODO: Set up the feed-forward network layers

        APPROACH:
        1. First layer expands from embed_dim to hidden_dim (usually 4x larger)
        2. Second layer projects back to embed_dim
        3. Use GELU activation (smoother than ReLU, preferred in transformers)

        EXAMPLE:
        >>> mlp = MLP(512)  # Will create 512 -> 2048 -> 512 network
        >>> x = Tensor(rng.standard_normal((2, 10, 512)))
        >>> output = mlp.forward(x)
        >>> assert output.shape == (2, 10, 512)

        HINT: Standard transformer MLP uses 4x expansion (hidden_dim = 4 * embed_dim)
        """
        ### BEGIN SOLUTION role="scaffold"
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Standard 4x expansion

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Keep this teaching model deterministic; do not silently ignore dropout.
        if dropout_prob != 0:
            raise ValueError("TinyTorch MLP supports dropout_prob=0 only")
        self.dropout_prob = dropout_prob

        # Two-layer feed-forward network
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.gelu = GELU()  # Use GELU activation from activations module
        self.linear2 = Linear(hidden_dim, embed_dim)
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through MLP.

        TODO: Implement the feed-forward computation

        APPROACH:
        1. First linear transformation: embed_dim -> hidden_dim
        2. Apply GELU activation (smooth, differentiable)
        3. Second linear transformation: hidden_dim -> embed_dim

        COMPUTATION FLOW:
        x -> Linear -> GELU -> Linear -> output

        HINT: self.gelu is the GELU you built in Module 02
        """
        ### BEGIN SOLUTION role="scaffold"
        # First linear layer with expansion
        hidden = self.linear1.forward(x)

        # GELU activation (YOUR activation from Module 02!)
        hidden = self.gelu.forward(hidden)

        # Second linear layer back to original size
        output = self.linear2.forward(hidden)

        return output
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the MLP to be called like a function."""
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """Return all learnable parameters."""
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params

# %% [markdown]
"""
### 🧪 Unit Test: MLP (Feed-Forward Network)

This test validates our MLP implementation works correctly.

**What we're testing**: Shape preservation and parameter counting
**Why it matters**: MLP provides the non-linear transformation in transformers
**Expected**: Input/output shapes match, correct parameter count
"""

# %% nbgrader={"grade": true, "grade_id": "test-mlp", "locked": true, "points": 10}
def test_unit_mlp():
    """🧪 Test MLP implementation."""
    print("🧪 Unit Test: MLP (Feed-Forward Network)...")

    # Test MLP with standard 4x expansion
    embed_dim = 64
    mlp = MLP(embed_dim)

    # Test forward pass
    batch_size, seq_len = 2, 10
    x = Tensor(rng.standard_normal((batch_size, seq_len, embed_dim)))
    output = mlp.forward(x)

    # Check shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim)

    # Check hidden dimension is 4x
    assert mlp.hidden_dim == 4 * embed_dim

    # Test parameter counting
    params = mlp.parameters()
    expected_params = 4  # 2 weights + 2 biases
    assert len(params) == expected_params

    # Test custom hidden dimension
    custom_mlp = MLP(embed_dim, hidden_dim=128)
    assert custom_mlp.hidden_dim == 128

    print("✅ MLP works correctly!")

if __name__ == "__main__":
    test_unit_mlp()

# %% [markdown]
r"""
### Understanding the Complete Transformer Block

The `TransformerBlock` represents the fundamental atomic unit of GPT and modern generative language models. It harmonizes two complementary operations: **spatial information routing** across tokens via multi-head causal self-attention, and **pointwise non-linear feature synthesis** via the two-stage MLP.

#### Pre-Norm vs Post-Norm Architectural Comparison

The original Transformer (Vaswani et al., 2017) utilized **Post-LayerNorm** ($x = \text{LN}(x + \text{Sublayer}(x))$). Modern large language models almost exclusively adopt **Pre-LayerNorm** ($x = x + \text{Sublayer}(\text{LN}(x))$):

| Architectural Trait | Post-LayerNorm (Original 2017) | Pre-LayerNorm (Modern GPT / Llama) |
|:---|:---|:---|
| **Formulation** | $x_{l+1} = \text{LayerNorm}(x_l + F_l(x_l))$ | $x_{l+1} = x_l + F_l(\text{LayerNorm}(x_l))$ |
| **Residual Path** | Passes through non-linear normalization at every layer | Completely clean identity skip highway |
| **Gradient Backprop** | Scales as $\prod_{l=1}^L \frac{1}{\sigma_l}$; prone to vanishing/explosion | Direct addition: $\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} (I + \sum \frac{\partial F_l}{\partial x_l})$ |
| **Warmup Requirement** | Strict linear learning rate warmup required to prevent divergence | Extremely robust; converges reliably with minimal warmup |
| **Scaling Capability** | Difficult to train beyond 16–24 layers without careful initialization scaling (Xiong et al., 2020) | Scalable to 100+ layers without numerical instability |

The clean residual path has one consequence worth naming. Nothing ever normalizes the stream itself, so every sublayer adds into it and its magnitude keeps growing with depth. A post-LN stack renormalizes after each addition, but a pre-LN stack does not, which is why `GPT` ends with a final `ln_f` before the LM head. That last normalization is the only one the residual stream ever receives, and without it the un-embedding projection sees activations whose scale depends on how many layers happened to be stacked.

#### Step-by-Step Data Transformation in TransformerBlock

For an input batch of token embeddings $x_0 \in \mathbb{R}^{B \times S \times d_{\text{embed}}}$:

$$\begin{aligned}
\text{Step 1 (Attention Pre-LN):} \quad & x_1 = \text{LayerNorm}_1(x_0) \\
\text{Step 2 (Causal Attention):} \quad & \Delta_{\text{attn}} = \text{MultiHeadAttention}(x_1, \text{mask}) \\
\text{Step 3 (First Highway Sum):} \quad & x_2 = x_0 + \Delta_{\text{attn}} \\
\text{Step 4 (MLP Pre-LN):} \quad & x_3 = \text{LayerNorm}_2(x_2) \\
\text{Step 5 (MLP Expansion):} \quad & \Delta_{\text{mlp}} = \text{MLP}(x_3) \\
\text{Step 6 (Second Highway Sum):} \quad & x_4 = x_2 + \Delta_{\text{mlp}}
\end{aligned}$$

Every sub-layer acts as an "off-ramp" that reads from the continuous residual stream and writes an additive delta back into it, preserving the identity gradient highway throughout the model.
"""

# %% nbgrader={"grade": false, "grade_id": "transformer-block", "solution": true}
#| export
class TransformerBlock:
    """
    Complete Transformer Block with self-attention, MLP, and residual connections.

    This is the core building block of GPT and other transformer models.
    Each block processes the input sequence and passes it to the next block.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: Optional[float] = None, *,
                 ff_dim: Optional[int] = None, dropout_prob: float = 0.0):
        """
        Initialize a complete transformer block.

        Give the MLP width one way or the other, never both. Use mlp_ratio (default 4)
        as a multiple of embed_dim, or ff_dim as an absolute hidden width. ff_dim is
        keyword-only, so a stray third positional argument can no longer slip into
        mlp_ratio and build a hidden layer many times wider than intended.

        dropout_prob must be zero; this compact block omits dropout.

        TODO: Set up all components of the transformer block

        APPROACH:
        1. Multi-head self-attention for sequence modeling
        2. First layer normalization (pre-norm architecture)
        3. MLP with specified expansion ratio (or explicit ff_dim)
        4. Second layer normalization

        TRANSFORMER BLOCK ARCHITECTURE:
        x → LayerNorm → MultiHeadAttention → + (residual) →
            LayerNorm → MLP → + (residual) → output

        EXAMPLE:
        >>> block = TransformerBlock(embed_dim=512, num_heads=8)
        >>> x = Tensor(rng.standard_normal((2, 10, 512)))  # (batch, seq, embed)
        >>> output = block.forward(x)
        >>> assert output.shape == (2, 10, 512)

        HINT: We use pre-norm architecture (LayerNorm before attention/MLP)
        """
        ### BEGIN SOLUTION role="scaffold"
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Layer normalizations (pre-norm architecture)
        self.ln1 = LayerNorm(embed_dim)  # Before attention
        self.ln2 = LayerNorm(embed_dim)  # Before MLP

        # Feed-forward network. mlp_ratio and ff_dim are two spellings of the same
        # quantity, so accepting both at once would silently honor one and drop the other.
        if ff_dim is not None and mlp_ratio is not None:
            raise ValueError(
                "Give the MLP width once. Pass mlp_ratio or ff_dim, not both "
                f"(got mlp_ratio={mlp_ratio}, ff_dim={ff_dim})."
            )
        if ff_dim is not None:
            hidden_dim = ff_dim
        else:
            hidden_dim = int(embed_dim * (4 if mlp_ratio is None else mlp_ratio))
        self.mlp = MLP(embed_dim, hidden_dim, dropout_prob)
        ### END SOLUTION

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through transformer block.

        TODO: Implement the complete transformer block computation

        APPROACH:
        1. Apply layer norm, then self-attention, then add residual
        2. Apply layer norm, then MLP, then add residual
        3. Return the transformed sequence

        COMPUTATION FLOW:
        x → ln1 → attention → + x → ln2 → mlp → + → output

        RESIDUAL CONNECTIONS:
        These are crucial for training deep networks - they allow gradients
        to flow directly through the network during backpropagation.

        HINT: Store intermediate results to add residual connections properly
        """
        ### BEGIN SOLUTION
        # First sub-layer: Multi-head self-attention with residual connection
        # Pre-norm: LayerNorm before attention
        normed1 = self.ln1.forward(x)
        # Self-attention: query, key, value are all the same (normed1)
        attention_out = self.attention.forward(normed1, mask)

        # Residual connection
        x = x + attention_out

        # Second sub-layer: MLP with residual connection
        # Pre-norm: LayerNorm before MLP
        normed2 = self.ln2.forward(x)
        mlp_out = self.mlp.forward(normed2)

        # Residual connection
        output = x + mlp_out

        return output
        ### END SOLUTION

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Allows the transformer block to be called like a function."""
        return self.forward(x, mask)

    def parameters(self) -> List[Tensor]:
        """Return all learnable parameters."""
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.mlp.parameters())
        return params

# %% [markdown]
"""
### The Causal Mask

GPT is autoregressive, so position i may attend only to positions j ≤ i. The helper
below encodes that rule in the binary convention Module 12's `_apply_mask`
expects (1 = attend, 0 = block). `GPT.forward` builds one for every sequence.
"""

# %% nbgrader={"grade": false, "grade_id": "causal-mask", "solution": false}
#| export
def create_causal_mask(seq_len: int) -> Tensor:
    """
    Create a causal (autoregressive) attention mask.

    This mask ensures that position i can only attend to positions j where j ≤ i.
    Essential for autoregressive language models like GPT.

    Args:
        seq_len: Length of the sequence

    Returns:
        Tensor of shape (1, seq_len, seq_len) with:
        - 1.0 for positions that CAN be attended to (lower triangle)
        - 0.0 for positions that CANNOT be attended to (upper triangle)

    Example:
        For seq_len=4, creates:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Usage:
        >>> from tinytorch.core.transformers import create_causal_mask
        >>> mask = create_causal_mask(seq_len=10)
        >>> output = attention(x, mask=mask)
    """
    # Lower triangular matrix: 1 = can attend, 0 = cannot attend
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return Tensor(mask[np.newaxis, :, :])  # Add batch dimension


# %% [markdown]
"""
### 🧪 Unit Test: Transformer Block

This test validates our complete TransformerBlock implementation.

**What we're testing**: Shape preservation, residual connections, parameter counting
**Why it matters**: This is the core component that will be stacked to create GPT
**Expected**: Input/output shapes match, all components work together
"""

# %% nbgrader={"grade": true, "grade_id": "test-transformer-block", "locked": true, "points": 15}
def test_unit_transformer_block():
    """🧪 Test TransformerBlock implementation."""
    print("🧪 Unit Test: Transformer Block...")

    # Test transformer block
    embed_dim = 64
    num_heads = 4
    block = TransformerBlock(embed_dim, num_heads)

    # Test forward pass
    batch_size, seq_len = 2, 8
    x = Tensor(rng.standard_normal((batch_size, seq_len, embed_dim)))
    output = block.forward(x)

    # Check shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim)

    # Test with causal mask (for autoregressive generation).
    # Use create_causal_mask, which follows the convention _apply_mask expects
    # in Module 12: a BINARY mask where 1 = attend and 0 = block, turned into an
    # additive (1 - mask) * MASK_VALUE. Handing it a pre-built additive -inf mask
    # instead flattens every allowed score to the same constant, so attention
    # stops depending on Q.K at all -- and a shape-only assertion never notices.
    mask = create_causal_mask(seq_len)
    masked_output = block.forward(x, mask)
    assert masked_output.shape == (batch_size, seq_len, embed_dim)

    # Causality is the whole point of the mask, so assert it: changing the last
    # token must not move any earlier position's output. The perturbation has to
    # be non-uniform across the embedding, because LayerNorm subtracts the mean
    # and would erase a constant shift before attention ever sees it.
    x_perturbed = np.array(x.data, copy=True)
    x_perturbed[:, -1, :] = rng.standard_normal(embed_dim) * 5
    perturbed_output = block.forward(Tensor(x_perturbed), mask)
    assert np.allclose(
        np.asarray(masked_output.data)[:, :-1, :],
        np.asarray(perturbed_output.data)[:, :-1, :],
        atol=1e-5,
    ), "Causal mask leaked a future token into an earlier position"

    # Test parameter counting
    params = block.parameters()
    expected_components = 4  # attention, ln1, ln2, mlp parameters
    assert len(params) > expected_components  # Should have parameters from all components

    # Test different configurations
    large_block = TransformerBlock(embed_dim=128, num_heads=8, mlp_ratio=2)
    assert large_block.mlp.hidden_dim == 256  # 128 * 2

    print("✅ TransformerBlock works correctly!")

if __name__ == "__main__":
    test_unit_transformer_block()

# %% [markdown]
r"""
### Understanding the Complete GPT Architecture

GPT (Generative Pre-trained Transformer) integrates token and positional embeddings, stacked pre-LN transformer blocks, a final normalization layer, and an un-embedding projection head into a unified autoregressive generative system.

<div align="center">
  <img src="transformer_gpt_architecture.svg" alt="Complete GPT Architecture: From Token IDs to Generation" width="560px">
</div>

#### Autoregressive Generation Mechanics

At generation time, text is emitted one token at a time. Each predicted token is appended to the sequence prefix to form the input for the next forward pass:

| Generation Step | Context Prefix ($x_{1:t-1}$) | Active Query Token ($x_{t-1}$) | Argmax / Sampled Token ($x_t$) | Updated Prefix ($x_{1:t}$) |
|:---:|:---|:---|:---:|:---|
| **Step 1** | `"The"` | `"The"` | `"cat"` | `"The cat"` |
| **Step 2** | `"The cat"` | `"cat"` | `"sat"` | `"The cat sat"` |
| **Step 3** | `"The cat sat"` | `"sat"` | `"on"` | `"The cat sat on"` |
| **Step 4** | `"The cat sat on"` | `"on"` | `"the"` | `"The cat sat on the"` |
| **Step 5** | `"The cat sat on the"` | `"the"` | `"mat"` | `"The cat sat on the mat"` |

#### Causal Masking: Preserving Autoregressive Integrity

During pre-training, the model receives full sequences in parallel. To prevent position $i$ from attending to future tokens ($j > i$), we inject an upper-triangular mask of $-\infty$ prior to row-wise softmax normalization:

$$S_{\text{masked}}[i, j] = \begin{cases} S[i, j] & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases} \implies A[i, j] = \text{softmax}(S_{\text{masked}})_{i, j} = 0 \quad (\forall j > i)$$

| Query Position ($i$) | Key 0 ("The") | Key 1 ("cat") | Key 2 ("sat") | Key 3 ("on") | Visibility Semantics |
|:---:|:---:|:---:|:---:|:---:|:---|
| **Pos 0 ("The")** | $A_{0,0} = 1.0$ | $0.0$ ($-\infty$) | $0.0$ ($-\infty$) | $0.0$ ($-\infty$) | Token 0 attends strictly to itself |
| **Pos 1 ("cat")** | $A_{1,0}$ | $A_{1,1}$ | $0.0$ ($-\infty$) | $0.0$ ($-\infty$) | Token 1 attends to tokens $\{0, 1\}$ |
| **Pos 2 ("sat")** | $A_{2,0}$ | $A_{2,1}$ | $A_{2,2}$ | $0.0$ ($-\infty$) | Token 2 attends to tokens $\{0, 1, 2\}$ |
| **Pos 3 ("on")** | $A_{3,0}$ | $A_{3,1}$ | $A_{3,2}$ | $A_{3,3}$ | Token 3 attends to all preceding context |

#### Temperature Scaling in Autoregressive Sampling

Temperature $T > 0$ controls the entropy of the output probability distribution:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^V \exp(z_j / T)}$$

| Sampling Regime | Temperature ($T$) | Logit Transformation | Probability Distribution Profile | Behavioral Characteristics |
|:---|:---:|:---|:---|:---|
| **Greedy / Deterministic** | $T \to 0$ | Extreme scaling ($z / \epsilon \to \pm\infty$) | One-hot Dirac delta ($\max z_i \to 1.0$) | Strict argmax decoding; repetitive but factual |
| **Low Temperature** | $T = 0.2 - 0.5$ | Sharpened logits ($z_i$ scaled by $2\times$ to $5\times$) | Peak-concentrated distribution | High confidence; ideal for code and math |
| **Balanced (Default)** | $T = 0.7 - 1.0$ | Unmodified / lightly scaled | Faithful to model's learned distribution | Balanced fluency, diversity, and coherence |
| **High Temperature** | $T \ge 1.5$ | Flattened logits ($z_i / 1.5$) | Near-uniform entropy distribution | Creative, diverse, but prone to hallucinations |

#### Transformer Scaling Laws: Parameter Allocation Across Scales

| Architecture Tier | Parameters | Layers ($L$) | Heads ($H$) | Hidden Dim ($d_{\text{embed}}$) | MLP / FFN Dim ($d_{\text{ff}}$) | Context Window ($S$) | Primary Deployment Profile |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **TinyTorch GPT** | $\approx 200\text{K}$ | $2$ | $4$ | $64$ | $256$ | $128$ | CPU educational inspection |
| **GPT-2 Small** | $124\text{M}$ | $12$ | $12$ | $768$ | $3{,}072$ | $1{,}024$ | Edge devices & consumer GPUs |
| **Llama 3 (8B)** | $8.0\text{B}$ | $32$ | $32$ | $4{,}096$ | $14{,}336$ | $8{,}192$ | Single workstation GPU ($24\text{ GB}$ VRAM at FP16) |
| **GPT-3 (175B)** | $175\text{B}$ | $96$ | $96$ | $12{,}288$ | $49{,}152$ | $2{,}048$ | Multi-GPU node and beyond ($8\times\text{A100}$ per node) |

Two notes on that table. The $4\times$ expansion this module builds is an empirical convention, not a law, and the newest models have already left it. Llama 3's $d_{\text{ff}} = 14{,}336$ is $3.5d$ rather than $4d$ because SwiGLU splits the up-projection across three matrices instead of two, so a smaller width holds the same parameter budget. Llama 3 8B also needs more than $16\text{ GB}$ in practice, since $16\text{ GB}$ is exactly its FP16 weights with nothing left for the KV cache or activations.
"""

# # %% [markdown]
r"""
## 🎲 Autoregressive Generation & Token Sampling

Language models generate text by iteratively predicting the probability distribution of the next token, sampling a token, and appending it to the sequence context.

In accordance with the **LEGO Bricks Principle**, generation logic is decoupled from neural network layers. Any sequence model implementing a `.forward()` method can be driven by these standalone inference utilities.
"""

# %% nbgrader={"grade": false, "grade_id": "sample-next-token", "solution": true}
#| export
def sample_next_token(logits: np.ndarray, temperature: float = 1.0, rng: Any = None) -> int:
    """
    Sample one token from vocabulary logits using temperature scaling.

    Args:
        logits: Unnormalized model logits, shape (1, vocab_size) or (vocab_size,)
        temperature: Sampling temperature (T >= 0). T=0 is greedy argmax.
        rng: Optional random number generator (defaults to module rng)

    Returns:
        Sampled token index as an integer.
    """
    ### BEGIN SOLUTION role="scaffold"
    # Apply temperature scaling
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    flat_logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature == 0:
        return int(np.argmax(flat_logits))
    # Center in float64 before division so tiny temperatures cannot turn
    # the largest logit into infinity. Negative overflow means zero mass.
    centered_logits = flat_logits - np.max(flat_logits)
    with np.errstate(over="ignore", under="ignore"):
        scaled_logits = centered_logits / temperature
        exp_logits = np.exp(scaled_logits)

    # Convert to probabilities (softmax with numerical stability)
    probs = exp_logits / np.sum(exp_logits)

    generator = rng if rng is not None else globals().get("rng", np.random.default_rng(7))
    # Sample next token from probability distribution
    next_token = generator.choice(len(flat_logits), p=probs)
    return int(next_token)
    ### END SOLUTION


# %% [markdown]
r"""
### 🧪 Unit Test: Token Sampling

This test validates `sample_next_token` across greedy decoding (temperature=0) and stochastic temperature regimes.

**What we're testing**: Temperature scaling, softmax probability output, valid token range
**Why it matters**: Sampling quality controls generation coherence and creativity
**Expected**: Valid token indices, probabilities sum to 1, temperature affects distribution
"""

# %% nbgrader={"grade": true, "grade_id": "test-sample-next-token", "locked": true, "points": 10}
def test_unit_sample_next_token():
    """🧪 Test sample_next_token implementation."""
    print("🧪 Unit Test: Token Sampling...")

    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    token = sample_next_token(logits, temperature=1.0)
    assert isinstance(token, (int, np.integer)), f"Expected int, got {type(token)}"
    assert 0 <= token < 5, f"Token {token} out of range [0, 5)"

    from unittest.mock import Mock, patch
    sampler = Mock()
    sampler.choice.return_value = 4
    with patch.dict(sample_next_token.__globals__, {"rng": sampler}):
        for temperature in (0.01, 1.0, 2.0):
            assert sample_next_token(logits, temperature) == 4
            probabilities = sampler.choice.call_args.kwargs["p"]
            expected = np.exp((logits[0] - logits.max()) / temperature)
            expected /= expected.sum()
            np.testing.assert_allclose(probabilities, expected)
            assert sampler.choice.call_args.args == (5,)

        uniform_logits = np.ones((1, 5))
        sample_next_token(uniform_logits, temperature=2.0)
        np.testing.assert_allclose(sampler.choice.call_args.kwargs["p"], np.full(5, 0.2))

    # Zero temperature is deterministic greedy decoding
    assert sample_next_token(logits, temperature=0) == 4

    print("✅ Token sampling works correctly!")

if __name__ == "__main__":
    test_unit_sample_next_token()


# %% [markdown]
r"""
## 🔤 Autoregressive Generation Loop

The `generate()` function drives any causal language model token-by-token.
"""

# %% nbgrader={"grade": false, "grade_id": "generate-tokens", "solution": true}
#| export
def generate(model: Any, prompt_tokens: Tensor, max_new_tokens: int = 50,
             temperature: float = 1.0, max_seq_len: int = 1024, rng: Any = None) -> Tensor:
    """
    Generate text autoregressively by repeatedly sampling next tokens from a model.

    Args:
        model: Any sequence model with a forward(tokens) -> logits method
        prompt_tokens: Prompt token IDs, shape (1, seq_len)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        max_seq_len: Maximum supported sequence length
        rng: Optional random number generator (defaults to module rng)

    Returns:
        Tensor of shape (1, prompt_len + max_new_tokens)
    """
    ### BEGIN SOLUTION role="scaffold"
    if len(prompt_tokens.shape) != 2 or prompt_tokens.shape[0] != 1 or prompt_tokens.shape[1] == 0:
        raise ValueError("generate expects one nonempty prompt with shape (1, sequence)")
    if not isinstance(max_new_tokens, (int, np.integer)) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if prompt_tokens.shape[1] + max_new_tokens > max_seq_len:
        raise ValueError("Prompt plus generated tokens exceeds max_seq_len")
    current_tokens = Tensor(prompt_tokens.data.copy())

    for _ in range(max_new_tokens):
        # Forward pass to get logits for current sequence
        logits = model.forward(current_tokens)

        # Extract last position logits: (1, vocab_size)
        last_logits = logits.data[:, -1, :]

        # Sample next token
        next_token_id = sample_next_token(last_logits, temperature, rng=rng)

        # Append to sequence
        next_token = np.array([[next_token_id]])
        current_tokens = Tensor(np.concatenate([current_tokens.data, next_token], axis=1))

    return current_tokens
    ### END SOLUTION


# %% [markdown]
r"""
### 🧪 Unit Test: Autoregressive Generation

This test validates the standalone `generate` utility using a deterministic mock sequence model.

**What we're testing**: Token-by-token autoregressive loop, sequence expansion, prompt preservation
**Why it matters**: Drives any causal language model to produce continuous text
**Expected**: Output shape is (1, prompt_len + max_new_tokens), prompt prefix preserved
"""

# %% nbgrader={"grade": true, "grade_id": "test-generate", "locked": true, "points": 15}
def test_unit_generate():
    """🧪 Test generate implementation with mock sequence model."""
    print("🧪 Unit Test: Autoregressive Generation...")

    class MockSequenceModel:
        def __init__(self, vocab_size=10):
            self.vocab_size = vocab_size

        def forward(self, tokens, start_pos=0):
            batch_size, seq_len = tokens.shape
            # Predict token 7 deterministically at every position
            logits = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float32)
            logits[:, :, 7] = 10.0
            return Tensor(logits)

    model = MockSequenceModel(vocab_size=10)
    prompt = Tensor([[1, 2, 3]])
    generated = generate(model, prompt, max_new_tokens=4, temperature=0.0)

    assert generated.shape == (1, 7), f"Expected shape (1, 7), got {generated.shape}"
    np.testing.assert_array_equal(generated.data[0], [1, 2, 3, 7, 7, 7, 7])

    print("✅ Autoregressive generation works correctly!")

if __name__ == "__main__":
    test_unit_generate()

# %% nbgrader={"grade": false, "grade_id": "export-models", "solution": false}
#| export
# Re-export model architectures for downstream compatibility
try:
    from tinytorch.models.transformer import GPT, TinyGPT
except ImportError:
    pass

# %% [markdown]
"""
## 🔧 Integration: Complete Transformer Workflow

Now that we've built all the components, let's see how they work together in a complete language modeling pipeline. This demonstrates the full power of the transformer architecture.

### The Language Modeling Pipeline

```
Complete Workflow Visualization:

1. Text Input:
   "hello world" → Tokenization → [15496, 1917]

2. Model Processing:
   [15496, 1917]
        ↓ Token Embedding
   [[0.1, 0.5, ...], [0.3, -0.2, ...]]  # Vector representations
        ↓ + Position Embedding
   [[0.2, 0.7, ...], [0.1, -0.4, ...]]  # With position info
        ↓ Transformer Block 1
   [[0.3, 0.2, ...], [0.5, -0.1, ...]]  # After attention + MLP
        ↓ Transformer Block 2
   [[0.1, 0.9, ...], [0.7, 0.3, ...]]   # Further processed
        ↓ Final LayerNorm + LM Head
   [[0.1, 0.05, 0.8, ...], [...]]       # Probability over vocab

3. Generation:
   Model predicts next token: "!" (token 33)
   New sequence: "hello world!"
```

This integration demo will show:
- **Character-level tokenization** for simplicity
- **Forward pass** through all components
- **Autoregressive generation** in action
- **Temperature effects** on creativity
"""

# %% nbgrader={"grade": false, "grade_id": "integration-demo", "solution": false}
def demonstrate_transformer_integration():
    """
    Demonstrate complete transformer pipeline.

    This simulates training a small language model on a simple vocabulary.
    """
    print("🔗 Integration Demo: Complete Language Model Pipeline")
    print("Building a mini-GPT for character-level text generation")
    from tinytorch.models.transformer import GPT

    # Create a small vocabulary (character-level)
    vocab = list("abcdefghijklmnopqrstuvwxyz .")
    vocab_size = len(vocab)
    char_to_idx = {char: i for i, char in enumerate(vocab)}
    idx_to_char = {i: char for i, char in enumerate(vocab)}

    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {''.join(vocab)}")

    # Create model
    model = GPT(
        vocab_size=vocab_size,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=32
    )

    # Sample text encoding
    text = "hello world."
    tokens = [char_to_idx[char] for char in text]
    input_tokens = Tensor(np.array([tokens]))

    print(f"\nOriginal text: '{text}'")
    print(f"Tokenized: {tokens}")
    print(f"Input shape: {input_tokens.shape}")

    # Forward pass
    logits = model.forward(input_tokens)
    print(f"Output logits shape: {logits.shape}")
    print(f"Each position predicts next token from {vocab_size} possibilities")

    # Generation demo
    prompt_text = "hello"
    prompt_tokens = [char_to_idx[char] for char in prompt_text]
    prompt = Tensor(np.array([prompt_tokens]))

    print("\nGeneration demo:")
    print(f"Prompt: '{prompt_text}'")

    generated = generate(model, prompt, max_new_tokens=8, temperature=1.0)
    generated_text = ''.join([idx_to_char[idx] for idx in generated.data[0]])

    print(f"Generated: '{generated_text}'")
    print("(Note: Untrained model produces random text)")

    return model

if __name__ == "__main__":
    demonstrate_transformer_integration()

# %% [markdown]
r"""
## 📊 Systems Analysis: Parameter Scaling and Memory

Transformer models scale predictably across parameter count, dataset size, and compute budgets. Understanding the mathematical breakdown of weights and intermediate activations is essential for architecting training clusters and production serving engines.

### The Empirical Scaling Laws

Transformer performance follows consistent power-law relationships across multiple orders of magnitude (Kaplan et al., 2020; Hoffmann et al., 2022 [Chinchilla]):

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$$

| Scaling Dimension | Power-Law Exponent | 10× Resource Increase Impact | Optimal Allocation Ratio |
|:---|:---:|:---:|:---|
| **Parameters ($N$)** | $\alpha_N \approx 0.076$ | $\approx 16\%$ reduction in cross-entropy loss | Chinchilla optimum: scale parameters and tokens equally |
| **Training Tokens ($D$)** | $\alpha_D \approx 0.095$ | $\approx 20\%$ reduction in cross-entropy loss | $D \approx 20 \times N$ for compute-optimal pre-training |
| **Compute FLOPs ($C$)** | $\alpha_C \approx 0.050$ | $\approx 11\%$ reduction in cross-entropy loss | $C \approx 6 \times N \times D$ floating-point operations |

---

### Memory Footprint Breakdown by Component

Transformer memory consumption divides into four distinct categories with different asymptotic scaling behaviors:

| Memory Category | Closed-Form Formula (Bytes) | Complexity | Dominant Regime | Systems Mitigation |
|:---|:---|:---:|:---|:---|
| **Model Parameters** | $M_{\text{params}} \approx (V d + 12 L d^2 + V d) \times 4$ | $\mathcal{O}(V d + L d^2)$ | Large models ($d \ge 4{,}096$) | Tensor parallelism, FP16/INT8 quantization |
| **FFN Activations** | $M_{\text{FFN}} = B \times S \times 4d \times L \times 4$ | $\mathcal{O}(B \cdot S \cdot d \cdot L)$ | Short sequences ($S \ll d$) | Activation recomputation (checkpointing) |
| **Attention Logits** | $M_{\text{attn}} = B \times L \times H \times S^2 \times 4$ | $\mathcal{O}(B \cdot L \cdot H \cdot S^2)$ | Long sequences ($S \ge 4{,}096$) | FlashAttention (online softmax tiling) |
| **KV Cache (Serving)** | $M_{\text{KV}} = 2 \times B \times L \times S \times d \times 2$ | $\mathcal{O}(B \cdot L \cdot S \cdot d)$ | Multi-user generation | PagedAttention, Grouped-Query Attention (GQA) |

The parameter row carries a $\approx$ on purpose. It counts the two vocabulary projections and the $12 L d^2$ of block matrices, and it drops the learned positional table, every bias vector, and every LayerNorm scale and shift. For the configuration in Question 3 below those omissions come to $565{,}248$ parameters, about $1.9\%$ of the true total, which is close enough for capacity planning and wrong for an exact count.

---

### The Quadratic Attention Memory Wall

Because attention weights scale as $S^2$, scaling context length from $2\text{K} \to 128\text{K}$ tokens causes attention matrix storage to explode quadratically:

| Sequence Length ($S$) | Head Matrix ($S^2 \times 4\text{ B}$) | Layer Footprint ($32\text{ heads}$) | Model Total ($32\text{ layers}$) | Production Context Feasibility |
|:---:|:---:|:---:|:---:|:---|
| **$1{,}024$ tokens** | $4.19\text{ MB}$ | $134.2\text{ MB}$ | $4.29\text{ GB}$ | Fits easily on commodity consumer GPUs |
| **$2{,}048$ tokens** | $16.78\text{ MB}$ | $536.9\text{ MB}$ | $17.18\text{ GB}$ | GPT-3 (2020) native context limit |
| **$4{,}096$ tokens** | $67.11\text{ MB}$ | $2.15\text{ GB}$ | $68.72\text{ GB}$ | Llama 2 native context window |
| **$8{,}192$ tokens** | $268.4\text{ MB}$ | $8.59\text{ GB}$ | $274.9\text{ GB}$ | Requires Activation Checkpointing |
| **$32{,}768$ tokens** | $4.29\text{ GB}$ | $137.4\text{ GB}$ | $4.40\text{ TB}$ | Intractable without FlashAttention |
| **$131{,}072$ tokens** | $68.72\text{ GB}$ | $2.20\text{ TB}$ | $70.37\text{ TB}$ | Requires Ring Attention + FlashAttention-3 |
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-scaling", "solution": false}
def analyze_parameter_scaling():
    """📊 Analyze how parameter count scales with embedding dimension."""
    print("📊 Analyzing Parameter Scaling in Transformers...")
    print("Understanding why model size affects performance and cost\n")
    from tinytorch.models.transformer import GPT

    # Vary ONE dimension. The earlier version doubled embed_dim and grew num_layers at
    # the same time, so neither variable's effect could be read off the output.
    num_layers = 4
    num_heads = 8
    vocab_size = 50000  # Typical vocabulary size

    print(f"Holding layers = {num_layers}, heads = {num_heads}, vocab = {vocab_size:,}\n")
    print("  d | Total params |  Vocab proj (share) |    Blocks | Growth")
    print("-" * 68)

    previous_total = None
    for embed_dim in (64, 128, 256, 512):
        model = GPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )

        total_params = sum(param.size for param in model.parameters())
        # The two V x d vocabulary projections: the token/position embeddings and the
        # LM head that maps back out to the vocabulary.
        vocab_params = (sum(p.size for p in model.embedding_layer.parameters())
                        + sum(p.size for p in model.lm_head.parameters()))
        block_params = total_params - vocab_params
        vocab_share = 100.0 * vocab_params / total_params

        growth = "-" if previous_total is None else f"{total_params / previous_total:.2f}x"
        print(f"{embed_dim:3d} | {total_params:12,} | {vocab_params:11,} ({vocab_share:4.1f}%) | "
              f"{block_params:9,} | {growth:>6}")
        previous_total = total_params

    memory_mib = (previous_total * BYTES_PER_FLOAT32) / MB_TO_BYTES
    print(f"\nWidest model above: {memory_mib:.1f} MiB of float32 weights")
    print("💡 Doubling d only about doubles the total here, because the 2Vd vocabulary")
    print("   projections dominate every row. Scaling is effectively LINEAR in d at this")
    print("   width; the quadratic 12Ld^2 block term takes over only at production width,")
    print("   where d is large enough for 12Ld^2 to outgrow 2Vd.")
    print("🚀 Real GPT-3 has 175B parameters, requiring ~700 GB in float32 (~350 GB in float16)")

if __name__ == "__main__":
    analyze_parameter_scaling()

# %% nbgrader={"grade": false, "grade_id": "analyze-attention-memory", "solution": false}
def analyze_attention_memory():
    """📊 Analyze attention memory complexity with sequence length."""
    print("📊 Analyzing Attention Memory Complexity...")
    print("Why long context is expensive and how it scales\n")

    num_heads = 8
    batch_size = 4

    # Test different sequence lengths
    sequence_lengths = [128, 256, 512, 1024, 2048]

    print("Attention Matrix Memory Usage:")
    print("Seq Len | Attention Matrix Size | Memory (MiB)")
    print("-" * 45)

    for seq_len in sequence_lengths:
        # Attention matrix is (batch_size, num_heads, seq_len, seq_len)
        attention_elements = batch_size * num_heads * seq_len * seq_len

        # 4 bytes per float32
        memory_bytes = attention_elements * BYTES_PER_FLOAT32
        memory_mb = memory_bytes / MB_TO_BYTES

        print(f"{seq_len:6d} | {seq_len}×{seq_len} × {batch_size}×{num_heads} | {memory_mb:8.1f}")

    print()
    print("💡 Attention memory grows quadratically with sequence length")
    print("🚀 This is why attention efficiency techniques are crucial for long sequences")

if __name__ == "__main__":
    analyze_attention_memory()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 25}
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
    test_unit_layer_norm()
    test_unit_mlp()
    test_unit_transformer_block()
    test_unit_sample_next_token()
    test_unit_generate()

    print("\nRunning integration scenarios...")

    # Test complete transformer pipeline scenario
    print("🧪 Integration Test: Full Generation Pipeline...")

    from tinytorch.models.transformer import GPT

    # Create model and data
    vocab_size = 50
    embed_dim = 64
    num_layers = 2
    num_heads = 4

    model = GPT(vocab_size, embed_dim, num_layers, num_heads)

    # Test batch processing
    batch_size = 3
    seq_len = 16
    tokens = Tensor(rng.integers(0, vocab_size, (batch_size, seq_len)))

    # Forward pass
    logits = model.forward(tokens)
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # Test generation with different temperatures
    prompt = Tensor(rng.integers(0, vocab_size, (1, 8)))

    # Conservative generation
    conservative = generate(model, prompt, max_new_tokens=5, temperature=0.1)
    assert conservative.shape == (1, 13)

    # Creative generation
    creative = generate(model, prompt, max_new_tokens=5, temperature=2.0)
    assert creative.shape == (1, 13)

    # Test parameter counting consistency
    total_params = sum(param.size for param in model.parameters())
    assert total_params > 1000  # Should have substantial parameters

    print("✅ Full transformer pipeline works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 13")

# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

Answer these questions to deepen your systems understanding of transformer parameter allocation, numerical stability, and inference execution dynamics:

### Question 1: Attention Matrix Memory Footprint & Scaling Mechanics
You evaluated multi-head attention activation memory across batch and sequence dimensions.

**1. Exact Activation Memory Derivation**:
For sequence length $S = 1{,}024$, batch size $B = 4$, and $H = 8$ attention heads at float32 precision ($4\text{ bytes/element}$):
$$\text{Elements per Layer} = B \times H \times S \times S = 4 \times 8 \times 1{,}024^2 = 33{,}554{,}432 \text{ elements} \quad (\mathbf{33.55\text{M values}})$$
$$\text{Memory per Layer} = 33{,}554{,}432 \times 4\text{ B} = 134{,}217{,}728\text{ B} = \mathbf{134.22\text{ MB}} \quad (128.0\text{ MiB})$$

**2. Context Doubling Impact ($S \to 2{,}048$)**:
$$\text{Memory}_{S=2048} = 4 \times 8 \times 2{,}048^2 \times 4\text{ B} = 536{,}870{,}912\text{ B} = \mathbf{536.87\text{ MB}} \quad (512.0\text{ MiB})$$
$$\text{Scaling Factor} = \frac{2{,}048^2}{1{,}024^2} = \mathbf{4\times \text{ (quadratic quadrupling)}}$$

**3. Architectural Implications**:
Across a 32-layer transformer model (e.g. Llama-style), storing attention score matrices alone consumes:
$$32 \times 536.87\text{ MB} = \mathbf{17.18\text{ GB of VRAM}}$$
This activation footprint accounts only for the intermediate attention logits, excluding linear activations, MLP expansions, and parameter weights. This quadratic wall is why naive attention cannot scale to $32\text{K} \to 128\text{K}$ context windows without **FlashAttention** (Dao et al., 2022).

---

### Question 2: Residual Connection Mathematics & Pre-LN vs Post-LN Stability
Your `TransformerBlock` routes representations through residual connections ($x + \text{Attention}(\text{LN}(x))$).

**1. Gradient Propagation in Deep Plain Networks**:
In networks without skip connections ($x_{l+1} = F_l(x_l)$), the chain rule expands into an unbroken product of Jacobian matrices:
$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{l=0}^{L-1} \frac{\partial F_l}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{l=0}^{L-1} W_l \cdot \sigma'(\dots)$$
If the singular values of $W_l$ deviate even slightly from unity ($\lambda \ne 1$), the gradient magnitude scales as $\lambda^L$, causing exponential vanishing ($\to 0$) or explosion ($\to \infty$) beyond 10–20 layers.

**2. The Additive Gradient Highway**:
With residual connections ($x_{l+1} = x_l + F_l(x_l)$), unrolling the recurrence yields $x_L = x_0 + \sum_{l=0}^{L-1} F_l(x_l)$. Differentiating produces:
$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum_{l=0}^{L-1} \frac{\partial F_l}{\partial x_l} \right)$$
The identity matrix $I$ carries the gradient at unity gain. Even if all layer Jacobians $\frac{\partial F_l}{\partial x_l}$ vanish, the error signal still propagates directly back to the initial embeddings $x_0$.

**3. Pre-LN vs Post-LN Gradient Mechanics**:
- **Post-LN** ($x_{l+1} = \text{LN}(x_l + F_l(x_l))$): The residual stream is repeatedly normalized by $\frac{1}{\sigma_l}$. Gradients in late layers scale with the norm of earlier activations, causing gradient variance to explode near the input and requiring warm-up schedules.
- **Pre-LN** ($x_{l+1} = x_l + F_l(\text{LN}(x_l))$): The residual stream remains completely clean and unscaled. Gradients flow directly through $I$, allowing models with 100+ layers to train reliably without warmup.

---

### Question 3: Comprehensive Parameter Budget & Distribution Analysis
For a concrete GPT architecture with $d_{\text{embed}} = 512, V = 10{,}000, L = 6, H = 8, S_{\text{max}} = 1{,}024$:

**1. Exact Parameter Allocations**:
- **Embedding Subsystem**:
  - Token Embeddings: $V \times d_{\text{embed}} = 10{,}000 \times 512 = 5{,}120{,}000\text{ params}$ ($5.12\text{M}$).
  - Learned Positional Embeddings: $S_{\text{max}} \times d_{\text{embed}} = 1{,}024 \times 512 = 524{,}288\text{ params}$ ($0.52\text{M}$).
  - Total Embedding: $\mathbf{5{,}644{,}288\text{ parameters}}$.
- **Per Transformer Block**:
  - Self-Attention ($W_Q, W_K, W_V, W_O$ with biases): $4 \times (512 \times 512 + 512) = 1{,}050{,}624\text{ params}$ ($\approx 1.05\text{M}$).
  - Layer Normalizations ($\text{LN}_1, \text{LN}_2$ with $\gamma, \beta$): $2 \times 2 \times 512 = 2{,}048\text{ params}$.
  - MLP ($W_1, b_1, W_2, b_2$ with $4\times$ expansion): $(512 \times 2048 + 2048) + (2048 \times 512 + 512) = 2{,}099{,}712\text{ params}$ ($\approx 2.10\text{M}$).
  - Total per Block: $1{,}050{,}624 + 2{,}048 + 2{,}099{,}712 = \mathbf{3{,}152{,}384\text{ parameters}}$.
  - Block Weight Ratio: The MLP consumes $\frac{2.10\text{M}}{3.15\text{M}} = \mathbf{66.6\% \text{ of all block parameters}}$.
- **Output Subsystem**:
  - Final LayerNorm: $2 \times 512 = 1{,}024\text{ params}$.
  - Language Modeling Head: $d_{\text{embed}} \times V = 512 \times 10{,}000 = 5{,}120{,}000\text{ params}$ ($5.12\text{M}$).

**2. Total Model Footprint**:
$$\text{Total Parameters} = 5{,}644{,}288 + 6 \times 3{,}152{,}384 + 1{,}024 + 5{,}120{,}000 = \mathbf{29{,}679{,}616\text{ parameters}} \quad (\approx \mathbf{29.68\text{M}})$$
In 32-bit floating point ($4\text{ bytes/param}$), this model requires:
$$\text{Storage Footprint} = 29{,}679{,}616 \times 4\text{ B} \approx \mathbf{118.72\text{ MB of VRAM}}$$

---

### Question 4: Autoregressive Generation Inefficiency & The KV Cache Imperative
Your `generate()` method produces text by repeatedly appending sampled tokens and re-running `forward()`.

**1. Algorithmic Complexity of Naive Generation**:
To generate $N$ new tokens given a prompt of length $S$:
- In step $1$, the forward pass computes representations for $S$ tokens.
- In step $2$, the forward pass computes representations for $S + 1$ tokens.
- In step $N$, the forward pass computes representations for $S + N - 1$ tokens.
$$\text{Total Tokens Evaluated} = \sum_{t=S}^{S+N-1} t = S \cdot N + \frac{N(N-1)}{2} = \mathcal{O}(S \cdot N + N^2)$$
Generating $1{,}000$ tokens from an initial prompt requires evaluating over $\mathbf{500{,}000\text{ token forward passes}}$.

**2. Identification of Redundant Compute**:
Because causal masking prohibits future tokens from affecting past tokens, the keys and values computed for tokens $0$ through $t-2$ **never change**. Recomputing $K$ and $V$ projections for historical tokens in every iteration is entirely redundant work.

**3. The Memory Bandwidth Bottleneck & KV Cache**:
During generation with batch size $B=1$, the model operates in a strictly **memory-bandwidth-bound** regime (arithmetic intensity $\approx 1\text{ FLOP/byte}$). In every single token step, all $29.7\text{M}$ weights must be transferred from GPU memory (HBM) to on-chip SRAM.
- **The KV Cache Solution** (Module 18): By storing past key and value vectors in a pre-allocated tensor buffer, each new token step only requires projecting the single newest token:
  $$Q_{\text{new}} = x_{\text{new}} W_Q, \quad K_{\text{new}} = x_{\text{new}} W_K, \quad V_{\text{new}} = x_{\text{new}} W_V$$
  The newest key and value are appended to the cache, cutting per-token generation cost from $\mathcal{O}(t)$ projections to $\mathcal{O}(1)$ projections plus a linear attention gather.
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Transformer Processes Sequences

**What you built:** A complete transformer block with attention, MLPs, and residual connections.

**Why it matters:** This is THE architecture behind GPT, Claude, LLaMA, and every modern
language model. The transformer block combines attention (for relationships) with MLPs
(for processing) and residual connections (for trainability).

In the milestones, you'll stack these blocks to build a working language model!
"""

# %%
def demo_transformers():
    """🎯 See a transformer block process a sequence."""
    print("🎯 AHA MOMENT: Transformer Processes Sequences")
    print("=" * 45)

    # Create a small transformer block (using concrete parameters)
    embed_dim = 64
    num_heads = 4
    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4)

    # Input: batch of 2 sequences, 8 tokens each, 64 dims.
    # The values must VARY across the feature axis. An all-ones token vector has zero
    # feature variance, so LayerNorm returns exactly 0, both sublayers return 0, and the
    # whole block collapses to the identity (a silent, very convincing non-demo).
    demo_rng = np.random.default_rng(13)
    x = Tensor(demo_rng.standard_normal((2, 8, embed_dim)))

    # Forward pass through transformer block
    output = block.forward(x)

    # Show transformation
    print(f"Input shape:  {x.shape}  (2 sequences, 8 tokens, 64 dimensions)")
    print(f"Output shape: {output.shape}")

    # Verify the transformation actually occurred. Compare element by element rather
    # than by summing. Two opposite-signed deltas cancel in a sum, so a dead block can
    # report a matching total and look alive.
    delta = np.abs(output.data - x.data)
    print("\nData transformation:")
    print(f"  Max |output - input|:  {np.max(delta):.4f}  (per-element change)")
    print(f"  Mean |output - input|: {np.mean(delta):.4f}  (after attention + MLP)")

    print("\nTransformerBlock architecture:")
    print(f"  - Multi-head attention ({num_heads} heads)")
    print("  - Layer normalization (before operations)")
    print(f"  - MLP ({embed_dim} -> {4*embed_dim} -> {embed_dim} with GELU)")
    print("  - Residual connections (preserve information flow)")

    print("\n✨ The building block of GPT, Claude, and modern language models!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_transformers()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Transformers

Congratulations! You've built the complete transformer architecture that powers modern language models like GPT, Claude, and ChatGPT!

### Key Accomplishments
- Built LayerNorm for stable training across deep transformer networks
- Implemented MLP (feed-forward) networks with GELU activation and 4x expansion
- Created complete TransformerBlock with self-attention, residual connections, and pre-norm architecture
- Implemented decoupled autoregressive token generation with temperature scaling
- Discovered attention memory scaling and parameter distribution patterns
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Discovered
- **Attention memory scales quadratically**: the score matrix is seq_len^2 per
  head, which is why context length is expensive and not merely inconvenient
- **Parameters concentrate in the MLP**: the 4x expansion means the feed-forward
  block holds roughly two thirds of a transformer block's weights
- **Pre-norm is a systems decision**: normalizing before each sublayer keeps the
  residual path clean, so gradients reach early layers without vanishing
- **Generation is memory-bound, not compute-bound**: each new token re-reads the
  entire model, which is the problem Module 18 exists to solve

### Ready for Next Steps
Your transformer implementation is the capstone of the language modeling pipeline.
Export with: `tito module complete 13`

**Next**: Module 14 will add profiling and optimization techniques to make your transformers production-ready!
"""
