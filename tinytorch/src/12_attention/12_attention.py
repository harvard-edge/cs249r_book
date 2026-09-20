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
# Module 12: Attention - Learning to Focus

Welcome to Module 12! You're about to build the multi-head attention mechanism that forms the computational core of GPT, BERT, Llama, and modern transformer architectures.

## 🔗 Prerequisites & Progress
**You've Built**: Complete neural network stack including autograd, optimizers, data loaders, 2D convolutions, subword BPE tokenization, and vector embeddings (`Tensor`, `Function`, `Linear`, `Conv2d`, `BPETokenizer`, `EmbeddingLayer`).
**You'll Build**: Scaled dot-product attention (`scaled_dot_product_attention`), causal autoregressive masking, and parallel subspace routing (`MultiHeadAttention`).
**You'll Enable**: Complete Transformer Blocks and causal language models (`13_transformers`), KV caching (`18_memoization`), and the Capstone model (`20_capstone`).

<div align="center">
  <img src="attention_blueprint.svg" alt="TinyTorch Architecture Blueprint: Module 12 Attention" width="380px">
</div>

### Architectural Roadmap

| Stage | Subsystem | Primitives & Capabilities | Status |
| :--- | :--- | :--- | :--- |
| **Modules 01–08** | Foundation & Training | `Tensor`, `Function`, `Linear`, `SGD`, `Adam`, `Trainer` | Completed |
| **Modules 09–11** | Spatial & Language Representations | `Conv2d`, `BPETokenizer`, `Embedding`, `PositionalEncoding` | Completed |
| **Module 12** | **Self-Attention & Multi-Head** | `scaled_dot_product_attention`, `MultiHeadAttention` | **Active Subsystem** |
| **Modules 13–20** | Transformers & Acceleration | `TransformerBlock`, `create_causal_mask`, `KVCache`, `TinyGPT` | Downstream Consumers |

## 🎯 Learning Objectives
By the end of this module, you will:
1. **Implement scaled dot-product attention** from mathematical first principles, exposing the quadratic $\mathcal{O}(S^2)$ memory and compute bottleneck.
2. **Derive score normalization variance scaling ($1/\sqrt{d_k}$)** to prevent dot product magnitudes from saturating softmax gradients.
3. **Apply a causal mask** by replacing future logits with $-\infty$ before the softmax, enforcing autoregressive temporal ordering without information leakage.
4. **Implement multi-head coordinate origami** (`_split_heads` and `_merge_heads`), transforming 3D tensors $(B, S, D)$ into 4D subspace planes $(B, H, S, d_k)$ for batched GEMM execution.
5. **Benchmark quadratic memory and FLOP scaling** across sequence lengths and quantify the activation memory footprint during backward pass.

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/12_attention/attention.ipynb`
**Building Side:** Code exports to `tinytorch.core.attention`

```python
# How to use this module:
from tinytorch.core.attention import scaled_dot_product_attention, MultiHeadAttention
```

**Why this matters:**
- **Learning:** Unveils the inner mechanics of attention as soft dynamic routing across queries, keys, and values.
- **Production:** Mirrors PyTorch's `torch.nn.MultiheadAttention` and `torch.nn.functional.scaled_dot_product_attention`. One convention is inverted, so port masks with care. TinyTorch marks the positions to keep (1 = attend, 0 = block), while PyTorch's boolean `attn_mask` marks the positions to block.
- **Consistency:** Unifies tensor transformations and attention projections in `tinytorch.core.attention`.
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

| Dependency | Origin | Purpose in Module 12 | Systems Invariant |
| :--- | :--- | :--- | :--- |
| `Tensor` | Module 01 (`core.tensor`) | Multi-dimensional array container supporting strided matmul and transposition | Manages contiguous memory buffers and backward computational graph |
| `Softmax` | Module 02 (`core.activations`) | Normalizes raw attention scores into probability distributions | Numerically stable along last dimension `dim=-1` |
| `Linear` | Module 03 (`core.layers`) | Parameter projections $W_Q, W_K, W_V$ and output projection $W_O$ | Dense matrix multiplication $(B, S, D) \times (D, D)$ |
| `EmbeddingLayer` | Module 11 (`core.embeddings`) | Upstream provider of position-aware continuous vectors | Produces input tensor $X \in \mathbb{R}^{B \times S \times D}$ |
| `numpy` | External | Fast array operations, upper-triangular masking, and benchmarking | Provides `np.tril` and `np.broadcast_to` for causal masks |

### Attention Information Routing Pipeline

| Step | Operation | Mathematical Formula | Tensor Shape & Semantics |
| :--- | :--- | :--- | :--- |
| **1. Linear Projections**| Query, Key, Value mappings | $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \; \mathbf{K} = \mathbf{X}\mathbf{W}_K, \; \mathbf{V} = \mathbf{X}\mathbf{W}_V$ | $(B, S, D) \rightarrow (B, H, S, d_k)$ |
| **2. Scaled Dot-Product**| Raw attention scores | $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}$ | $(B, H, S, S)$ query-key correlation |
| **3. Causal Masking** | Autoregressive masking | $\mathbf{S}_{\text{masked}} = \mathbf{S} + \mathbf{M}$ (upper triangle $= -\infty$) | Prevents attending to future tokens |
| **4. Probability Weights**| Softmax normalizer | $\mathbf{A} = \text{softmax}(\mathbf{S}_{\text{masked}}, \text{dim}=-1)$ | Non-negative attention distribution |
| **5. Value Mixture** | Context aggregation | $\mathbf{Y} = \mathbf{A}\mathbf{V}$ | Weighted sum of value vectors |
| **6. Output Projection** | Feature projection | $\text{Output} = \mathbf{Y}\mathbf{W}_O$ | Restores model dimension $(B, S, D)$ |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp core.attention
#| export

import numpy as np
rng = np.random.default_rng(7)
import math
import time
from typing import Optional, Tuple, List

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import Softmax

MASK_VALUE = float("-inf")  # Hard exclusion: exp(-inf) is exactly zero

# %% [markdown]
r"""
## 💡 Introduction: What is Attention?

Attention is the foundational routing mechanism that enables neural networks to dynamically focus on relevant coordinates of an input sequence. Instead of compressing an entire sequence into a single fixed-size recurrent bottleneck vector, attention allows every token representation to directly query and gather information from all other tokens in the sequence.

<div align="center">
  <img src="attention_routing.svg" alt="Attention as Dynamic Soft Information Routing" width="700px">
</div>

### Comparing Sequence Processing Paradigms

| Architecture | Information Routing | Path Length Between Tokens | Parallel Work Available | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **Recurrent (RNN/LSTM)** | Sequential hidden state passing | $\mathcal{O}(S)$ steps | $\mathcal{O}(1)$ (strictly sequential) | $\mathcal{O}(S \cdot d_{\text{model}})$ |
| **Convolutional (Conv1D)** | Local receptive field sliding | $\mathcal{O}(S / K)$ layers | $\mathcal{O}(S)$ parallel | $\mathcal{O}(S \cdot d_{\text{model}})$ |
| **Attention (Transformer)** | Pairwise dynamic dot-product | $\mathcal{O}(1)$ direct connection | $\mathcal{O}(S^2)$ fully parallel | $\mathcal{O}(S^2)$ per head |

### The Core Attention Formulation
In the landmark paper *Attention Is All You Need* (Vaswani et al., 2017), scaled dot-product attention is formulated as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

where queries ($Q$), keys ($K$), and values ($V$) represent learned linear projections of the input representations, and $M$ is an optional mask tensor.
"""

# %% [markdown]
r"""
## 📐 Foundations: Attention Mathematics & The Causal Engine

The attention mechanism implements a differentiable soft key-value store. Given a continuous query vector $\mathbf{q}_i$, the model computes compatibility scores against all available key vectors $\mathbf{k}_j$, normalizes these scores into a probability simplex via softmax, and computes a convex combination of value vectors $\mathbf{v}_j$.

<div align="center">
  <img src="causal_attention_engine.svg" alt="The Causal Attention Engine Architecture" width="760px">
</div>

### The Five-Stage Attention Pipeline

1. **Similarity Scoring (Raw Matrix Multiplication)**:
   For query token $i$ and key token $j$:
   $$\text{RawScore}_{i, j} = \mathbf{q}_i^\top \mathbf{k}_j = \sum_{d=1}^{d_k} Q_{i, d} K_{j, d} \implies S = Q K^\top \in \mathbb{R}^{B \times S \times S}$$
2. **Variance Scaling ($1/\sqrt{d_k}$)**:
   Dividing by $\sqrt{d_k}$ preserves unit variance when queries and keys have zero mean and unit variance, preventing dot products from growing excessively large:
   $$S_{\text{scaled}} = \frac{S}{\sqrt{d_k}}$$
3. **Causal Masking ($M$)**:
   In autoregressive language models, tokens must not attend to future positions ($j > i$). An upper-triangular mask sets future positions to $-\infty$:
   $$S_{\text{masked}} = S_{\text{scaled}} + M, \quad \text{where } M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
   This additive $M$ is the paper's notation. TinyTorch's code takes the equivalent boolean keep mask instead, covered under *Applying the Causal Mask* below.
4. **Softmax Normalization**:
   Exponentiating $-\infty$ yields strictly zero probability ($e^{-\infty} = 0$), forming a valid probability distribution over past and present tokens:
   $$A_{i, j} = \frac{\exp(S_{\text{masked}}[i, j])}{\sum_{m=1}^S \exp(S_{\text{masked}}[i, m])}, \quad \sum_{j=1}^S A_{i, j} = 1.0$$
5. **Value Aggregation**:
   The output vector for token $i$ is the expectation over all values weighted by attention probabilities:
   $$\mathbf{y}_i = \sum_{j=1}^S A_{i, j} \mathbf{v}_j \implies Y = A V \in \mathbb{R}^{B \times S \times d_k}$$

---

### Tensor Shapes & Computational Complexity

| Tensor Stage | Symbol | Shape Contract | FLOPs per Head | Primary Memory Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Input Queries** | $Q$ | $(B, S, d_k)$ | — | Activation buffer |
| **Input Keys** | $K$ | $(B, S, d_k)$ | — | Activation buffer |
| **Input Values** | $V$ | $(B, S, d_k)$ | — | Activation buffer |
| **Score Matrix** | $S = Q K^\top$ | $(B, S, S)$ | $2 \cdot B \cdot S^2 \cdot d_k$ | Quadratic $\mathcal{O}(S^2)$ memory wall |
| **Attention Weights** | $A = \text{softmax}(S)$ | $(B, S, S)$ | $3 \cdot B \cdot S^2$ (exp + sum + div) | Stored for backward pass |
| **Attended Output** | $Y = A V$ | $(B, S, d_k)$ | $2 \cdot B \cdot S^2 \cdot d_k$ | Feed-forward input buffer |

$$\text{Total Attention FLOPs} = 4 \cdot B \cdot S^2 \cdot d_k \quad (\text{quadratic in sequence length } S)$$

---

### Attention Weight Matrix ($A \in \mathbb{R}^{S \times S}$)

Under causal masking, the attention matrix forms a lower-triangular probability distribution where every row sums to $1.00$:

| Query Token ($i$) | Key 0 (`"The"`) | Key 1 (`"cat"`) | Key 2 (`"sat"`) | Key 3 (`"down"`) | Row Sum ($\sum_j A_{i, j}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`"The"` ($i=0$)** | $1.00$ | $0.00$ ($-\infty$) | $0.00$ ($-\infty$) | $0.00$ ($-\infty$) | **$1.00$** |
| **`"cat"` ($i=1$)** | $0.35$ | $0.65$ | $0.00$ ($-\infty$) | $0.00$ ($-\infty$) | **$1.00$** |
| **`"sat"` ($i=2$)** | $0.10$ | $0.45$ | $0.45$ | $0.00$ ($-\infty$) | **$1.00$** |
| **`"down"` ($i=3$)** | $0.15$ | $0.25$ | $0.20$ | $0.40$ | **$1.00$** |
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Scaled Dot-Product Attention

Now let's implement the core attention mechanism that powers all transformer models. We'll build it from three small vectorized helpers (scores, scaling, masking) and then compose them, so each step of the O(n²) computation stays visible.

### Understanding the Algorithm Visually

```
Step-by-Step Attention Computation:

1. Score Computation (Q @ K^T):
   For each query position i and key position j:
   score[i,j] = Σ(Q[i,d] × K[j,d]) for d in embedding_dims

   Query i    Key j      Dot Product
   [0.1,0.8] · [0.2,0.7] = 0.1×0.2 + 0.8×0.7 = 0.58

2. Scaling (÷ √d_k):
   scaled_scores = scores / √d_k
   (d_k is the length of each key vector; later in this module, when we
    split attention into heads, it becomes embed_dim // num_heads)
   (Prevents softmax saturation for large dimensions)

3. Masking (optional):
   For causal attention: scores[i,j] = -∞ if j > i
   (masked_fill replaces blocked scores with -infinity while preserving gradients)

   Causal Mask (lower triangular):
   [  OK  -∞  -∞  -∞ ]
   [  OK   OK  -∞  -∞ ]
   [  OK   OK   OK  -∞ ]
   [  OK   OK   OK   OK ]

4. Softmax (normalize each row):
   weights[i,j] = exp(scores[i,j]) / Σ(exp(scores[i,k])) for all k

5. Apply to Values:
   output[i] = Σ(weights[i,j] × V[j]) for all j
```
"""

# %% [markdown]
"""
### Computing Attention Scores

The first step in attention is measuring how similar each query is to each key.
We do this with matrix multiplication: each element scores[i][j] tells us
how much token i should attend to token j.

```
Q (batch, seq, d) @ K^T (batch, d, seq) -> scores (batch, seq, seq)

scores[i][j] = "how relevant is key j to query i?"
```
"""

# %% nbgrader={"grade": false, "grade_id": "attn-compute-scores", "solution": true}
#| export
def _compute_attention_scores(Q: Tensor, K: Tensor) -> Tensor:
    """Compute raw attention scores via Q @ K^T.

    TODO: Transpose K and multiply by Q to get similarity matrix

    APPROACH:
    1. Transpose K: swap last two dims so (batch, seq, d) -> (batch, d, seq)
    2. Matrix multiply: Q @ K^T gives (batch, seq, seq) scores

    EXAMPLE:
    >>> Q = Tensor(rng.standard_normal((1, 3, 4)))  # 3 tokens, dim=4
    >>> K = Tensor(rng.standard_normal((1, 3, 4)))
    >>> scores = _compute_attention_scores(Q, K)
    >>> print(scores.shape)  # (1, 3, 3) -- every token scored against every other

    HINT: Use K.transpose(-2, -1) to swap the last two dimensions
    """
    ### BEGIN SOLUTION role="scaffold"
    K_t = K.transpose(-2, -1)
    return Q.matmul(K_t)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Attention Scores

**What we're testing**: Q @ K^T produces correct similarity matrix shape and values
**Why it matters**: Wrong score shapes cascade into every downstream step
**Expected**: (batch, seq, seq) shape, all-ones input gives d_model as score
"""

# %% nbgrader={"grade": true, "grade_id": "test-attn-scores", "locked": true, "points": 5}
def test_unit_attention_scores():
    """🧪 Test attention score computation."""
    print("🧪 Unit Test: Attention Scores...")
    Q = Tensor(np.ones((1, 3, 4)))
    K = Tensor(np.ones((1, 3, 4)))
    scores = _compute_attention_scores(Q, K)
    assert scores.shape == (1, 3, 3), f"Expected (1,3,3), got {scores.shape}"
    assert np.allclose(scores.data, 4.0), "All-ones Q@K^T should give d_model=4"
    print("✅ Attention scores: correct shapes and values!")

if __name__ == "__main__":
    test_unit_attention_scores()

# %% [markdown]
"""
### Scaling Scores

A dot product sums d_k independent products. For query/key entries with mean 0
and variance 1, that sum has variance d_k -- so its typical magnitude grows with
the SQUARE ROOT of the dimension, not linearly. Going from d_k=1 to d_k=512
inflates scores about sqrt(512) ~= 23x, which is enough to push softmax into
extreme values where nearly all weight falls on a single token.

That square-root growth is exactly why we divide by sqrt(d_k) and not by d_k:
the scale factor has to match how the scores actually grow, which keeps them in
a stable range regardless of dimension.
"""

# %% nbgrader={"grade": false, "grade_id": "attn-scale-scores", "solution": true}
#| export
def _scale_scores(scores: Tensor, d_k: int) -> Tensor:
    """Scale attention scores by 1/sqrt(d_k).

    d_k is the dimension the dot product was taken over. For single-head
    attention that equals d_model; for multi-head attention it is the
    per-head dimension (embed_dim // num_heads).

    TODO: Divide scores by the square root of d_k

    APPROACH:
    1. Compute scale factor: 1.0 / math.sqrt(d_k)
    2. Multiply scores by scale factor

    EXAMPLE:
    >>> scores = Tensor(np.array([[[4.0, 8.0]]]))
    >>> scaled = _scale_scores(scores, d_k=4)
    >>> print(scaled.data)  # [[[ 2.0, 4.0]]] -- divided by sqrt(4)=2

    HINT: Use math.sqrt() for the square root
    """
    ### BEGIN SOLUTION role="scaffold"
    scale_factor = 1.0 / math.sqrt(d_k)
    return scores * scale_factor
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Score Scaling

**What we're testing**: Scores are divided by sqrt(d_k) correctly
**Why it matters**: Without scaling, softmax saturates for large dimensions
**Expected**: Scores reduced by factor of sqrt(d_k)
"""

# %% nbgrader={"grade": true, "grade_id": "test-attn-scale", "locked": true, "points": 5}
def test_unit_scale_scores():
    """🧪 Test attention score scaling."""
    print("🧪 Unit Test: Score Scaling...")
    scores = Tensor(np.array([[[4.0, 8.0]]]))
    # d_k=16 separates sqrt(d_k)=4 from d_k/2=8, so dividing by the wrong factor fails here
    scaled = _scale_scores(scores, d_k=16)
    assert np.allclose(scaled.data, [[[1.0, 2.0]]]), f"Expected /sqrt(16)=4, got {scaled.data}"
    print("✅ Score scaling works correctly!")

if __name__ == "__main__":
    test_unit_scale_scores()

# %% [markdown]
r"""
### Applying the Causal Mask

In autoregressive language models (such as GPT and Llama), generation proceeds token by token. A token at position $i$ must only attend to past and current tokens ($j \le i$), never future tokens ($j > i$). We enforce this temporal causality by replacing future attention logits with $-\infty$ before applying the softmax operator:

<div align="center">
  <img src="attention_margin_mask.svg" alt="Causal Attention Masking Transformation" width="300px">
</div>

#### Two Notations for One Operation

Read the mask twice, because the paper and the code spell it differently and mixing them up is the single most common way to get a `ValueError` out of this module.

The additive form is the one used in $\text{softmax}(QK^\top/\sqrt{d_k} + M)$ above, and it is what the literature means by $M$, a real-valued matrix added to the scores, carrying $0$ where the key is allowed and $-\infty$ where it is blocked.

TinyTorch never asks you to build that matrix. The mask you hand to `_apply_mask` and to `scaled_dot_product_attention` is a **boolean keep mask** with entries in $\{0, 1\}$, and `masked_fill` performs the $+M$ addition for you by writing $-\infty$ into every position where $\text{keep} = 0$. The two are the same operation, and $\text{scores} + M$ is bit-identical to $\texttt{masked\_fill}(\text{keep} = 0, -\infty)$. Passing a matrix that already contains $-\infty$ raises `ValueError: Attention mask must contain only 0 (blocked) or 1 (allowed)`, so build $\text{keep}$ with `np.tril(np.ones(...))`, not with $-\infty$.

$$S_{\text{masked}}[i, j] = \begin{cases} S[i, j] & \text{if } \text{keep}[i, j] = 1 \\ -\infty & \text{if } \text{keep}[i, j] = 0 \end{cases} \qquad \text{where } \text{keep}[i, j] = \begin{cases} 1 & \text{if } j \le i \\ 0 & \text{if } j > i \end{cases}$$

When exponentiated during softmax normalization, $\exp(-\infty) = 0$, guaranteeing mathematically zero probability and severing backward gradient flow from future tokens:

$$A[i, j] = \frac{\exp(S_{\text{masked}}[i, j])}{\sum_{k=1}^S \exp(S_{\text{masked}}[i, k])} = 0 \quad \text{for all } j > i$$

| Query Position $i$ | Allowed Keys $j$ | Keep Value $\text{keep}[i, j]$ | Equivalent Additive $M[i, j]$ | Pre-Softmax Score $S_{\text{masked}}[i, j]$ | Post-Softmax Attention $A[i, j]$ | Temporal Semantics |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| Current / Historical ($j \le i$) | Historical ($t_j \le t_i$) | $1$ | $0$ | $S[i, j]$ | $\frac{\exp(S[i, j])}{\sum_k \exp(S[i, k])}$ | Causally valid context |
| Future ($j > i$) | Lookahead ($t_j > t_i$) | $0$ | $-\infty$ | $-\infty$ | $\exp(-\infty) / \Sigma = 0.0$ | Severed (no future leakage) |
"""

# %% nbgrader={"grade": false, "grade_id": "attn-apply-mask", "solution": true}
#| export
def _apply_mask(scores: Tensor, mask: Tensor) -> Tensor:
    """Apply a binary attention mask by replacing blocked scores with -infinity.

    TODO: Replace scores with negative infinity where mask is 0

    APPROACH:
    1. Broadcast the binary mask to the scores and require an allowed key per query
    2. Use scores.masked_fill to replace blocked positions with MASK_VALUE
       (-infinity); unmasked scores and their gradients pass through unchanged

    EXAMPLE:
    >>> scores = Tensor(np.ones((1, 3, 3)))
    >>> mask = Tensor(np.tril(np.ones((1, 3, 3))))  # lower triangle
    >>> masked = _apply_mask(scores, mask)
    >>> print(masked.data[0, 0, 1])  # -inf (future position masked)

    HINT: mask=0 means "block this position", mask=1 means "allow"
    """
    ### BEGIN SOLUTION role="scaffold"
    # Both guards below sweep the whole broadcast (B, H, S, S) array, so they cost
    # O(B*H*S^2) on EVERY forward call. That is the price of catching a mis-built mask
    # at the point of use. Production kernels validate the mask once where it is
    # constructed and keep the attention inner loop free of checks.
    if np.any((mask.data != 0) & (mask.data != 1)):
        raise ValueError("Attention mask must contain only 0 (blocked) or 1 (allowed)")
    allowed = np.broadcast_to(mask.data, scores.shape)
    if np.any(np.sum(allowed, axis=-1) == 0):
        raise ValueError("Each query must have at least one allowed key")
    return scores.masked_fill(allowed == 0, MASK_VALUE)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Causal Masking

**What we're testing**: Future positions get set to negative infinity
**Why it matters**: Without masking, GPT could "cheat" by looking at future tokens
**Expected**: Masked positions are -infinity, unmasked positions unchanged
"""

# %% nbgrader={"grade": true, "grade_id": "test-attn-mask", "locked": true, "points": 5}
def test_unit_apply_mask():
    """🧪 Test causal mask application."""
    print("🧪 Unit Test: Causal Masking...")
    scores = Tensor(np.ones((1, 3, 3)))
    mask = Tensor(np.tril(np.ones((1, 3, 3))))
    masked = _apply_mask(scores, mask)
    # Future positions must be excluded even for very large scores
    assert np.isneginf(masked.data[0, 0, 1]), "Future position not masked"
    # Past positions should be unchanged
    assert np.allclose(masked.data[0, 0, 0], 1.0), "Past position was modified"
    print("✅ Causal masking works correctly!")

if __name__ == "__main__":
    test_unit_apply_mask()

# %% [markdown]
"""
### Bringing It Together: Scaled Dot-Product Attention

Now that you've built each piece -- scoring, scaling, and masking -- let's compose
them into the complete attention mechanism. Notice how the composition reads like
a recipe: compute scores, scale them, optionally mask, softmax, apply to values.

```
Pipeline: Q,K -> scores -> scale -> mask -> softmax -> weights @ V -> output
```

The sketch below shows how attention works conceptually using explicit
loops. While easier to read, this is NOT the implementation because:
1. It is extremely slow (Python loops vs optimized C/BLAS)
2. It breaks the autograd graph unless we manually implement the backward pass

Conceptually, this is what the vectorized helpers above are doing:

```
batch_size, seq_len, d_k = Q.shape
scores = np.zeros((batch_size, seq_len, seq_len))

for b in range(batch_size):
    for i in range(seq_len):          # Each query
        for j in range(seq_len):      # Attends to each key
            dot_product = 0.0
            for k in range(d_k):
                dot_product += Q[b, i, k] * K[b, j, k]
            scores[b, i, j] = dot_product / math.sqrt(d_k)
```

Count the loops: i and j each run over seq_len, so the score computation
alone is seq_len² dot products. That is the O(n²) you will measure later.
"""

# %% nbgrader={"grade": false, "grade_id": "attn-scaled-dot-product", "solution": true}
#| export
def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Complete scaled dot-product attention.

    TODO: Compose the helpers into the full attention operation

    APPROACH:
    1. Call _compute_attention_scores(Q, K) for raw similarity
    2. Call _scale_scores(scores, Q.shape[-1]) for numerical stability
    3. If mask provided, call _apply_mask(scores, mask); use binary entries
       and at least one allowed key per query
    4. Apply Softmax to get probability weights
    5. Multiply weights @ V for attended values

    SUB-PROBLEMS (you already implemented these):
    - _compute_attention_scores: Q @ K^T similarity matrix
    - _scale_scores: divide by sqrt(d_k) for stable softmax
    - _apply_mask: block future positions with MASK_VALUE (-inf)

    Args:
        Q: Query tensor of shape (..., seq_len, d_k)
        K: Key tensor of shape (..., seq_len, d_k)
        V: Value tensor of shape (..., seq_len, d_k)
        mask: Optional causal mask, 1=allow, 0=mask, shape (..., seq_len, seq_len)
              or any shape that broadcasts against the scores

        The leading "..." is any number of batch-like dimensions. Called
        directly it is (batch_size,) and d_k = d_model; from
        MultiHeadAttention it is (batch_size, num_heads) and d_k = head_dim.
        The helpers only touch the last two axes, so the same code serves both.

    Returns:
        output: Attended values (..., seq_len, d_k)
        attention_weights: Attention matrix (..., seq_len, seq_len)

    EXAMPLE:
    >>> Q = Tensor(rng.standard_normal((2, 4, 64)))
    >>> K = Tensor(rng.standard_normal((2, 4, 64)))
    >>> V = Tensor(rng.standard_normal((2, 4, 64)))
    >>> output, weights = scaled_dot_product_attention(Q, K, V)
    >>> print(output.shape)   # (2, 4, 64)
    >>> print(weights.shape)  # (2, 4, 4)

    HINT: Softmax is already imported -- use Softmax()(scores, dim=-1)
    """
    ### BEGIN SOLUTION
    scores = _compute_attention_scores(Q, K)
    scores = _scale_scores(scores, Q.shape[-1])
    if mask is not None:
        scores = _apply_mask(scores, mask)
    softmax = Softmax()
    attention_weights = softmax(scores, dim=-1)
    output = attention_weights.matmul(V)
    return output, attention_weights
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Scaled Dot-Product Attention

This test validates our complete attention mechanism works correctly with proper shape handling and masking.

**What we're testing**: End-to-end attention: shapes, probability normalization, causal masking
**Why it matters**: This is the core operation powering all transformer models
**Expected**: Correct shapes, weights summing to 1, future positions masked to zero
"""

# %% nbgrader={"grade": true, "grade_id": "test-attention-basic", "locked": true, "points": 15}
def test_unit_scaled_dot_product_attention():
    """🧪 Test scaled dot-product attention implementation."""
    print("🧪 Unit Test: Scaled Dot-Product Attention...")

    # Test basic functionality
    batch_size, seq_len, d_model = 2, 4, 8
    Q = Tensor(rng.standard_normal((batch_size, seq_len, d_model)))
    K = Tensor(rng.standard_normal((batch_size, seq_len, d_model)))
    V = Tensor(rng.standard_normal((batch_size, seq_len, d_model)))

    output, weights = scaled_dot_product_attention(Q, K, V)

    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape {output.shape} incorrect"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Weights shape {weights.shape} incorrect"

    # Check attention weights sum to 1 (probability distribution)
    weights_sum = weights.data.sum(axis=2)  # Sum over last dimension
    expected_sum = np.ones((batch_size, seq_len))
    assert np.allclose(weights_sum, expected_sum, atol=1e-6), "Attention weights don't sum to 1"

    # Test with causal mask
    mask = Tensor(np.tril(np.ones((batch_size, seq_len, seq_len)), k=0))  # Lower triangular
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask)
    assert output_masked.shape == (batch_size, seq_len, d_model), f"Masked output shape {output_masked.shape} incorrect"

    # Check that future positions have zero attention
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):  # Future positions
                assert abs(weights_masked.data[b, i, j]) < 1e-6, f"Future attention not masked at ({i},{j})"

    print("✅ scaled_dot_product_attention works correctly!")

if __name__ == "__main__":
    test_unit_scaled_dot_product_attention()

# %% [markdown]
r"""
## 🏗️ Multi-Head Attention: Subspace Specialization via Coordinate Origami

Single-head attention computes a single convex combination of values for each token, forcing the model to average over distinct grammatical, positional, and semantic relationships. Multi-Head Attention solves this by projecting input representations into $H$ distinct, lower-dimensional subspaces ($d_k = d_{\text{model}} / H$), allowing parallel heads to independently track syntax, long-range coreference, and semantic dependencies simultaneously.

<div align="center">
  <img src="head_split_origami.svg" alt="Multi-Head Coordinate Origami: Subspace Partitioning and Parallel Execution" width="680px">
</div>

### Mathematical Definition of Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_H) W_O$$
$$\text{head}_h = \text{Attention}\left(Q W_h^Q, \, K W_h^K, \, V W_h^V\right)$$

where projection matrices $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and output projection $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

---

### The Coordinate Origami Datapath

To execute all $H$ heads in parallel on hardware accelerators without slow Python loops, TinyTorch uses **tensor coordinate origami** (reshaping and transposing axis strides):

| Stage | Tensor Operation | NumPy / TinyTorch Expression | Resulting Shape Contract |
| :--- | :--- | :--- | :--- |
| **1. Input Projection** | Linear GEMM | `x.matmul(W_q)` | $(B, S, D)$ |
| **2. Subspace Reshape** | Decompose hidden dim $D \to (H, d_k)$ | `x.reshape(B, S, H, d_k)` | $(B, S, H, d_k)$ |
| **3. Head Transposition** | Swap seq and head axes | `x.transpose(1, 2)` | $(B, H, S, d_k)$ |
| **4. Batched Attention** | Parallel $QK^\top$ & $AV$ | `scaled_dot_product_attention(...)` | $(B, H, S, d_k)$ |
| **5. Untangling Swap** | Restore temporal sequence axis | `x.transpose(1, 2)` | $(B, S, H, d_k)$ |
| **6. Recombination** | Flatten heads back into hidden dim | `x.reshape(B, S, D)` | $(B, S, D)$ |
| **7. Output Projection** | Inter-head representation mixing | `x.matmul(W_o)` | $(B, S, D)$ |

### Systems Efficiency Invariant
Notice that $H$ heads each computing attention with head dimension $d_k = D / H$ require:
$$\text{FLOPs} = H \times (4 \cdot B \cdot S^2 \cdot d_k) = 4 \cdot B \cdot S^2 \cdot (H \cdot d_k) = \mathbf{4 \cdot B \cdot S^2 \cdot D}$$
Multi-head attention provides $H$ specialized, independent attention patterns with **identical computational complexity and parameter count** as a single massive head, while enabling tensor cores to run batched Level-3 BLAS GEMMs over $(B \cdot H)$ matrix slices simultaneously!
"""

# %% [markdown]
"""
### Splitting Heads

Multi-head attention processes the same data through multiple independent "heads."
To do this efficiently, we reshape the projected tensor from 3D to 4D, separating
the embedding dimension into (num_heads, head_dim). Then we transpose so the head
dimension comes before the sequence dimension, enabling parallel attention computation.

```
Split heads: (batch, seq, embed_dim) -> (batch, heads, seq, head_dim)

Example with embed_dim=64, num_heads=8, head_dim=8:
  (2, 10, 64) -> reshape -> (2, 10, 8, 8) -> transpose -> (2, 8, 10, 8)
                             batch seq heads dim          batch heads seq dim
```
"""

# %% [markdown]
"""
### Merging Heads

After each head computes its own attention independently, we need to recombine
them back into a single embedding. This is the reverse of splitting: transpose
the head and sequence dimensions back, then reshape to merge (heads, head_dim)
into a single embed_dim.

```
Merge heads: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)

Example with embed_dim=64, num_heads=8, head_dim=8:
  (2, 8, 10, 8) -> transpose -> (2, 10, 8, 8) -> reshape -> (2, 10, 64)
                                batch seq heads dim          batch seq embed_dim
```
"""

# %% nbgrader={"grade": false, "grade_id": "multihead-attention", "solution": true}
#| export
class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Runs multiple attention heads in parallel, each learning different relationships.
    This is the core component of transformer architectures.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initialize multi-head attention.

        TODO: Set up linear projections and validate configuration

        APPROACH:
        1. Validate that embed_dim is divisible by num_heads
        2. Calculate head_dim (embed_dim // num_heads)
        3. Create linear layers for Q, K, V projections
        4. Create output projection layer
        5. Store configuration parameters

        Args:
            embed_dim: Embedding dimension (d_model)
            num_heads: Number of parallel attention heads

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=512, num_heads=8)
        >>> mha.head_dim  # 64 (512 / 8)
        >>> len(mha.parameters())  # 4 linear layers * 2 params each = 8 tensors

        HINTS:
        - head_dim = embed_dim // num_heads must be integer
        - Need 4 Linear layers: q_proj, k_proj, v_proj, out_proj
        - Each projection maps embed_dim → embed_dim
        """
        ### BEGIN SOLUTION role="scaffold"
        if not isinstance(embed_dim, (int, np.integer)) or embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer")
        if not isinstance(num_heads, (int, np.integer)) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Multi-head attention dimension mismatch\n"
                f"  ❌ embed_dim={embed_dim} is not divisible by num_heads={num_heads} (remainder={embed_dim % num_heads})\n"
                f"  💡 Multi-head attention splits embed_dim equally among heads, so embed_dim must be a multiple of num_heads\n"
                f"  🔧 Try: embed_dim={num_heads * (embed_dim // num_heads + 1)} (next valid size) or num_heads=1"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, values
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        # Output projection to mix information across heads
        self.out_proj = Linear(embed_dim, embed_dim)
        ### END SOLUTION

    def _split_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Reshape to separate attention heads for parallel processing.

        TODO: Reshape (batch, seq, embed_dim) to (batch, heads, seq, head_dim)

        APPROACH:
        1. Reshape: (batch, seq, embed) -> (batch, seq, num_heads, head_dim)
        2. Transpose: swap seq and heads dims -> (batch, heads, seq, head_dim)

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> x = Tensor(rng.standard_normal((2, 10, 64)))  # batch=2, seq=10
        >>> split = mha._split_heads(x, 2, 10)
        >>> print(split.shape)  # (2, 8, 10, 8) -- 8 heads of dim 8

        HINT: reshape(batch, seq, heads, head_dim) then transpose(1, 2)
        """
        ### BEGIN SOLUTION role="scaffold"
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
        ### END SOLUTION

    def _merge_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Merge attention heads back into single embedding dimension.

        TODO: Reshape (batch, heads, seq, head_dim) to (batch, seq, embed_dim)

        APPROACH:
        1. Transpose: swap heads and seq -> (batch, seq, heads, head_dim)
        2. Reshape: merge last two dims -> (batch, seq, embed_dim)

        EXAMPLE:
        >>> # After attention with 8 heads of dim 8:
        >>> attended = Tensor(rng.standard_normal((2, 8, 10, 8)))
        >>> merged = mha._merge_heads(attended, 2, 10)
        >>> print(merged.shape)  # (2, 10, 64) -- back to embed_dim

        HINT: transpose(1, 2) then reshape(batch, seq, embed_dim)
        """
        ### BEGIN SOLUTION role="scaffold"
        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.embed_dim)
        ### END SOLUTION

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through multi-head attention.

        TODO: Compose the helpers into the complete multi-head attention forward pass

        APPROACH:
        1. Extract input dimensions and validate embed_dim
        2. Project input to Q, K, V using linear layers
        3. Call _split_heads() to separate into parallel heads
        4. Apply scaled_dot_product_attention to all heads at once
        5. Call _merge_heads() to recombine heads
        6. Apply output projection

        SUB-PROBLEMS (you already implemented these):
        - _split_heads: reshape 3D -> 4D for parallel head processing
        - _merge_heads: reshape 4D -> 3D to recombine head outputs

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask (batch_size, seq_len, seq_len)

        Returns:
            output: Attended representation (batch_size, seq_len, embed_dim)

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> x = Tensor(rng.standard_normal((2, 10, 64)))  # batch=2, seq=10, dim=64
        >>> output = mha.forward(x)
        >>> print(output.shape)  # (2, 10, 64) - same as input

        HINT: Use scaled_dot_product_attention for the attention computation
        """
        ### BEGIN SOLUTION
        # Step 1: Extract dimensions and validate
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"MultiHeadAttention input dimension mismatch\n"
                f"  ❌ Expected embed_dim={self.embed_dim}, got {embed_dim} from input shape {x.shape}\n"
                f"  💡 The last dimension of input must match embed_dim from initialization (MultiHeadAttention({self.embed_dim}, {self.num_heads}))\n"
                f"  🔧 Try: x.reshape({x.shape[0]}, {x.shape[1]}, {self.embed_dim}) or create new MultiHeadAttention({embed_dim}, num_heads)"
            )

        # Step 2: Project to Q, K, V
        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        # Step 3: Split into heads
        Q = self._split_heads(Q, batch_size, seq_len)
        K = self._split_heads(K, batch_size, seq_len)
        V = self._split_heads(V, batch_size, seq_len)

        # Step 4: Apply attention -- reshape mask (batch, seq_q, seq_k) to
        # (batch, 1, seq_q, seq_k) for broadcasting across heads.
        mask_reshaped = mask
        if mask is not None and len(mask.shape) == 3:
            batch_size_mask, seq_q_mask, seq_k_mask = mask.shape
            mask_reshaped = mask.reshape(batch_size_mask, 1, seq_q_mask, seq_k_mask)

        attended, _ = scaled_dot_product_attention(Q, K, V, mask=mask_reshaped)

        # Step 5: Merge heads back together
        concat_output = self._merge_heads(attended, batch_size, seq_len)

        # Step 6: Apply output projection
        output = self.out_proj.forward(concat_output)

        return output
        ### END SOLUTION

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Make MultiHeadAttention callable like attention(x)."""
        return self.forward(x, mask)

    def parameters(self) -> List[Tensor]:
        """
        Return all trainable parameters.

        TODO: Collect parameters from all linear layers

        APPROACH:
        1. Get parameters from q_proj, k_proj, v_proj, out_proj
        2. Combine into single list

        Returns:
            List of all parameter tensors

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> params = mha.parameters()
        >>> print(len(params))  # 8 (4 layers × 2 params each: weight + bias)
        >>> print(params[0].shape)  # (64, 64) - q_proj weight
        >>> print(params[1].shape)  # (64,) - q_proj bias

        HINTS:
        - Each Linear layer has .parameters() method that returns [weight, bias]
        - Use extend() to add all parameters from each layer to the list
        - Total should be 8 tensors: 4 layers × 2 parameters each
        """
        ### BEGIN SOLUTION role="scaffold"
        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Split Heads

**What we're testing**: 3D to 4D reshape correctly separates embedding into heads
**Why it matters**: Wrong reshaping silently produces garbage attention
**Expected**: (batch, heads, seq, head_dim) shape with correct values
"""

# %% nbgrader={"grade": true, "grade_id": "test-split-heads", "locked": true, "points": 5}
def test_unit_split_heads():
    """🧪 Test head splitting reshape."""
    print("🧪 Unit Test: Split Heads...")
    mha = MultiHeadAttention(embed_dim=64, num_heads=8)
    x = Tensor(rng.standard_normal((2, 10, 64)))
    split = mha._split_heads(x, 2, 10)
    assert split.shape == (2, 8, 10, 8), f"Expected (2,8,10,8), got {split.shape}"
    # Values must be regrouped, not shuffled. Head h owns columns [h*head_dim, (h+1)*head_dim)
    for h in range(8):
        assert np.allclose(split.data[:, h, :, :], x.data[:, :, h * 8:(h + 1) * 8]), f"Head {h} holds the wrong slice"
    print("✅ Split heads: correct 4D shape and per-head values!")

if __name__ == "__main__":
    test_unit_split_heads()

# %% [markdown]
"""
### 🧪 Unit Test: Merge Heads

**What we're testing**: 4D to 3D reshape correctly recombines heads into embedding
**Why it matters**: Split then merge must be a round-trip identity operation
**Expected**: (batch, seq, embed_dim) shape matching original input
"""

# %% nbgrader={"grade": true, "grade_id": "test-merge-heads", "locked": true, "points": 5}
def test_unit_merge_heads():
    """🧪 Test head merging reshape."""
    print("🧪 Unit Test: Merge Heads...")
    mha = MultiHeadAttention(embed_dim=64, num_heads=8)
    # Create 4D tensor as if from split_heads
    x_4d = Tensor(rng.standard_normal((2, 8, 10, 8)))
    merged = mha._merge_heads(x_4d, 2, 10)
    assert merged.shape == (2, 10, 64), f"Expected (2,10,64), got {merged.shape}"

    # Verify round-trip: split then merge recovers original data
    original = Tensor(rng.standard_normal((2, 10, 64)))
    split = mha._split_heads(original, 2, 10)
    recovered = mha._merge_heads(split, 2, 10)
    assert np.allclose(original.data, recovered.data), "Split->merge should recover original data"
    print("✅ Merge heads: correct 3D shape and round-trip!")

if __name__ == "__main__":
    test_unit_merge_heads()

# %% [markdown]
"""
### 🧪 Unit Test: Multi-Head Attention (End-to-End)

**What we're testing**: Configuration, parameter counting, shape preservation, and masking support
**Why it matters**: Multi-head attention must correctly split dimensions across heads and recombine them
**Expected**: Proper head dimension calculation, 8 parameters (4 layers x 2), preserved output shapes
"""

# %% nbgrader={"grade": true, "grade_id": "test-multihead", "locked": true, "points": 15}
def test_unit_multihead_attention():
    """🧪 Test multi-head attention implementation."""
    print("🧪 Unit Test: Multi-Head Attention...")

    # Test initialization
    embed_dim, num_heads = 64, 8
    mha = MultiHeadAttention(embed_dim, num_heads)

    # Check configuration
    assert mha.embed_dim == embed_dim
    assert mha.num_heads == num_heads
    assert mha.head_dim == embed_dim // num_heads

    # Test parameter counting (4 linear layers, each has weight + bias)
    params = mha.parameters()
    assert len(params) == 8, f"Expected 8 parameters (4 layers x 2), got {len(params)}"

    # Test forward pass
    batch_size, seq_len = 2, 6
    x = Tensor(rng.standard_normal((batch_size, seq_len, embed_dim)))

    output = mha.forward(x)

    # Check output shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim), f"Output shape {output.shape} incorrect"

    # Test with causal mask
    mask = Tensor(np.tril(np.ones((batch_size, seq_len, seq_len))))
    output_masked = mha.forward(x, mask)
    assert output_masked.shape == (batch_size, seq_len, embed_dim)

    # Test different head configurations
    mha_small = MultiHeadAttention(embed_dim=32, num_heads=4)
    x_small = Tensor(rng.standard_normal((1, 5, 32)))
    output_small = mha_small.forward(x_small)
    assert output_small.shape == (1, 5, 32)

    print("✅ MultiHeadAttention works correctly!")

if __name__ == "__main__":
    test_unit_multihead_attention()

# %% [markdown]
"""
## 🔧 Integration: Attention Patterns in Action

Let's test our complete attention system with realistic scenarios and visualize actual attention patterns.

### Understanding Attention Patterns

Real transformer models learn interpretable attention patterns:

```
Example Attention Patterns in Language:

1. Local Syntax Attention:
   "The quick brown fox"
   The → quick (determiner-adjective)
         quick → brown (adjective-adjective)
                 brown → fox (adjective-noun)

2. Long-Range Coreference:
   "John went to the store. He bought milk."
   He → John (pronoun resolution across sentence boundary)

3. Compositional Structure:
   "The cat in the hat sat"
   sat → cat (verb attending to subject, skipping prepositional phrase)

4. Causal Dependencies:
   "I think therefore I"
   I → think (causal reasoning patterns)
   I → I (self-reference at end)
```

Let's see these patterns emerge in our implementation.
"""

# %%
def run_attention_scenarios():
    """Test attention mechanisms in realistic scenarios."""
    print("🧪 Testing Attention Scenarios...")

    # Scenario 1: Small transformer block setup
    print("\n1. Small Transformer Setup:")
    embed_dim, num_heads, seq_len = 128, 8, 32

    # Create embeddings (simulating token embeddings + positional)
    embeddings = Tensor(rng.standard_normal((2, seq_len, embed_dim)))

    # Multi-head attention
    mha = MultiHeadAttention(embed_dim, num_heads)
    attended = mha.forward(embeddings)

    print(f"   Input shape: {embeddings.shape}")
    print(f"   Output shape: {attended.shape}")
    print(f"   Parameters: {len(mha.parameters())} tensors")

    # Scenario 2: Causal language modeling
    print("\n2. Causal Language Modeling:")

    # Create causal mask (lower triangular)
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    mask = Tensor(np.broadcast_to(causal_mask, (2, seq_len, seq_len)))

    # Apply causal attention
    causal_output = mha.forward(embeddings, mask)

    print(f"   Masked output shape: {causal_output.shape}")
    print(f"   Causal mask applied: {mask.shape}")

    # Scenario 3: Compare attention patterns
    print("\n3. Attention Pattern Analysis:")

    # Create simple test sequence
    simple_embed = Tensor(rng.standard_normal((1, 4, 16)))
    simple_mha = MultiHeadAttention(16, 4)

    # Get attention weights by calling the base function
    Q = simple_mha.q_proj.forward(simple_embed)
    K = simple_mha.k_proj.forward(simple_embed)
    V = simple_mha.v_proj.forward(simple_embed)

    # Reshape for single head analysis
    Q_head = Tensor(Q.data[:, :, :4])  # First head only
    K_head = Tensor(K.data[:, :, :4])
    V_head = Tensor(V.data[:, :, :4])

    _, weights = scaled_dot_product_attention(Q_head, K_head, V_head)

    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Attention weights (first batch, 4x4 matrix):")
    weight_matrix = weights.data[0, :, :].round(3)

    # Format the attention matrix nicely
    print("     Pos→  0     1     2     3")
    for i in range(4):
        row_str = f"   {i}: " + " ".join(f"{weight_matrix[i,j]:5.3f}" for j in range(4))
        print(row_str)

    print(f"   Row sums: {weights.data[0].sum(axis=1).round(3)} (should be ~1.0)")

    # Scenario 4: Attention with masking visualization
    print("\n4. Causal Masking Effect:")

    # Apply causal mask to the simple example
    simple_mask = Tensor(np.tril(np.ones((1, 4, 4))))
    _, masked_weights = scaled_dot_product_attention(Q_head, K_head, V_head, simple_mask)

    print("   Causal attention matrix (lower triangular):")
    masked_matrix = masked_weights.data[0, :, :].round(3)
    print("     Pos→  0     1     2     3")
    for i in range(4):
        row_str = f"   {i}: " + " ".join(f"{masked_matrix[i,j]:5.3f}" for j in range(4))
        print(row_str)

    print("   Notice: Upper triangle is zero (can't attend to future)")

    print("\n✅ All attention scenarios work correctly!")

# %% [markdown]
r"""
## 📊 Systems Analysis: Memory Layout and Performance

Let's understand ONE key systems concept: **attention's O(n^2) memory and compute scaling**.

This single analysis reveals why attention becomes the bottleneck in modern transformers and drives research into efficient attention variants.

### Attention Memory Footprint Scaling (per Layer)

Every memory figure in this module is decimal, so $1\text{ kB} = 10^3$ bytes and $1\text{ MB} = 10^6$ bytes. Powers-of-two units are marked KiB / MiB / GiB where they appear.

| Sequence Length ($T$) | Attention Matrix ($T \times T$) | Element Count | Memory (FP32, 1 Head) | Relative Scaling |
| :--- | :--- | :--- | :--- | :--- |
| **$T = 128$** | $128 \times 128$ | $16\text{K}$ values | $65.5\text{ kB}$ | $1.0\times$ (Reference) |
| **$T = 512$** | $512 \times 512$ | $262\text{K}$ values | $1.05\text{ MB}$ | $16\times$ larger ($4^2$) |
| **$T = 2048$ (GPT-3)** | $2048 \times 2048$ | $4.2\text{M}$ values | $16.78\text{ MB}$ | $256\times$ larger ($16^2$) |

For full-scale models like GPT-3 ($L = 96$ layers, $H = 96$ heads):

$$\text{Total Attention Memory} = L \times H \times (T^2 \times 4\text{ bytes}) = 96 \times 96 \times 16.78\text{ MB} \approx \mathbf{154.62\text{ GB}} \quad (144.0\text{ GiB})$$

Just for raw intermediate attention score matrices for a single sequence!
"""

# %%
def analyze_attention_complexity():
    """📊 Analyze attention computational complexity and memory scaling."""
    print("📊 Analyzing Attention Complexity...")

    # Test different sequence lengths to show O(n²) scaling
    embed_dim = 64
    sequence_lengths = [16, 32, 64, 128, 256]

    print("\nSequence Length vs Attention Matrix Size:")
    print("Seq Len | Attention Matrix | Memory (kB) | FLOPs")
    print("-" * 55)

    for seq_len in sequence_lengths:
        # Calculate attention matrix size
        attention_matrix_size = seq_len * seq_len

        # Memory for attention weights (float32 = 4 bytes), decimal kB
        attention_memory_kb = (attention_matrix_size * 4) / 1000

        # Both matmuls plus the softmax: 2*S^2*d for Q@K^T, 2*S^2*d for weights@V,
        # and 3*S^2 for softmax (exp + sum + divide), matching the Foundations table
        flops = 4 * seq_len * seq_len * embed_dim + 3 * seq_len * seq_len

        print(f"{seq_len:7d} | {attention_matrix_size:14d} | {attention_memory_kb:10.2f} | {flops:10.0f}")

    print(f"\n💡 KEY INSIGHT: Attention memory scales as O(n^2) with sequence length")
    print(f"🚀 For seq_len=1024, attention matrix alone needs {(1024*1024*4)/1e6:.2f} MB")

if __name__ == "__main__":
    analyze_attention_complexity()

# %%
def analyze_attention_timing():
    """📊 Measure attention computation time vs sequence length."""
    print("\n📊 Analyzing Attention Timing...")

    embed_dim, num_heads = 64, 8
    sequence_lengths = [32, 64, 128, 256]

    print("\nSequence Length vs Computation Time:")
    print("Seq Len | Time (ms) | Forward/sec | Scaling")
    print("-" * 45)

    prev_time = None
    for seq_len in sequence_lengths:
        # Create test input
        x = Tensor(rng.standard_normal((1, seq_len, embed_dim)))
        mha = MultiHeadAttention(embed_dim, num_heads)

        # Time multiple runs for stability
        times = []
        for _ in range(5):
            start_time = time.perf_counter()
            _ = mha.forward(x)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        forwards_per_sec = 1000 / avg_time if avg_time > 0 else 0

        # Calculate scaling factor vs previous
        scaling = avg_time / prev_time if prev_time else 1.0

        print(f"{seq_len:7d} | {avg_time:8.2f} | {forwards_per_sec:11.0f} | {scaling:6.2f}x")
        prev_time = avg_time

    print(f"\n💡 KEY INSIGHT: Attention time scales roughly as O(n^2) with sequence length")
    print(f"🚀 This is why attention efficiency techniques are an active area of research")

if __name__ == "__main__":
    analyze_attention_timing()

# %%
def analyze_attention_memory_overhead():
    """📊 Analyze memory overhead during training (forward + backward passes)."""
    print("\n📊 Analyzing Attention Memory Overhead During Training...")

    sequence_lengths = [128, 256, 512, 1024]

    print("\nAttention Activation Memory per Head (Training vs Inference):")
    print("Seq Len | Inference | Saved for backward | Gradient | Training total")
    print("-" * 70)

    for seq_len in sequence_lengths:
        # Inference: the attention matrix lives only while the layer runs
        attention_matrix_mb = (seq_len * seq_len * 4) / 1e6  # decimal MB

        # Training: the softmax weights are saved for backward, and backward
        # materializes a gradient of the same shape
        saved_mb = attention_matrix_mb
        gradient_mb = attention_matrix_mb
        training_total_mb = saved_mb + gradient_mb

        print(f"{seq_len:7d} | {attention_matrix_mb:7.2f}MB | {saved_mb:16.2f}MB | {gradient_mb:6.2f}MB | {training_total_mb:12.2f}MB")

    print("\n💡 KEY INSIGHT: Training roughly doubles attention's activation memory.")
    print("   The softmax weights are saved for backward and their gradient is the same size.")
    print("   Optimizer state (Adam's two moments) scales with parameters, not sequence length.")
    print("🚀 For GPT-3 (96 layers x 96 heads, 2048 context): 16.78 MB per head becomes ~154.62 GB of saved weights per sequence!")

if __name__ == "__main__":
    analyze_attention_memory_overhead()

# %% [markdown]
r"""
### Systems Insights: The $\mathcal{O}(n^2)$ Attention Memory Wall

Our empirical benchmarking reveals the fundamental scalability bottleneck governing modern transformer systems:

#### The Quadratic Memory Scaling Formula

For a transformer model operating on sequence length $S$, batch size $B$, and $H$ attention heads with float32 precision ($4\text{ bytes/element}$):

$$\text{Attention Matrix Memory (per layer)} = B \times H \times S^2 \times 4\text{ bytes}$$

$$\text{Total Attention Activation Memory (all layers)} = B \times L \times H \times S^2 \times 4\text{ bytes}$$

#### Production Reality Across Model Scales

| Model Architecture | Layers ($L$) | Heads ($H$) | Context ($S$) | Per Head | Per Layer | All Layers Total | Hardware Feasibility |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Edge / TinyTorch** | $6$ | $8$ | $512$ | $1.05\text{ MB}$ | $8.39\text{ MB}$ | $\mathbf{50.33\text{ MB}}$ | Fits DRAM easily, but already past a typical $32\text{ MB}$ L3, so every layer's scores round-trip to main memory |
| **Base Model** (BERT-Base) | $12$ | $12$ | $512$ | $1.05\text{ MB}$ | $12.58\text{ MB}$ | $\mathbf{150.99\text{ MB}}$ | Commodity GPU VRAM |
| **GPT-3 (175B)** | $96$ | $96$ | $2{,}048$ | $16.78\text{ MB}$ | $1.61\text{ GB}$ | $\mathbf{154.62\text{ GB}}$ | Exceeds single $80\text{ GB}$ A100 GPU VRAM |
| **Long Context** (32K Tokens) | $32$ | $32$ | $32{,}768$ | $4.29\text{ GB}$ | $137.44\text{ GB}$ | $\mathbf{4.40\text{ TB}}$ | Intractable with standard materialization |

#### The Three Systems Bottlenecks

1. **Memory Capacity Wall**: As context length $S$ scales from $2\text{K} \to 32\text{K} \to 128\text{K}$, the memory consumed by attention matrices dwarfs parameter storage by orders of magnitude.
2. **Arithmetic Intensity Deficit**: Attention logit computation involves $\mathcal{O}(S^2)$ multiply-accumulate operations to produce an $\mathcal{O}(S^2)$ tensor, followed by memory-bandwidth-bound softmax normalization ($\mathcal{O}(1)\text{ FLOPs/byte}$).
3. **The IO Memory Wall**: In naive attention implementations, the full $S \times S$ matrix is repeatedly transferred between off-chip High-Bandwidth Memory (HBM) and fast on-chip SRAM. This memory traffic bottleneck spurred the development of **FlashAttention** (Dao et al., 2022), which tiles the softmax computation to eliminate global memory materialization entirely. Module 17 will build the cache-aware tiling and operator fusion that FlashAttention rests on, and Module 18 will build the inference-time reuse (KV caching) that avoids recomputing keys and values altogether.
"""

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-test", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire attention module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_attention_scores()
    test_unit_scale_scores()
    test_unit_apply_mask()
    test_unit_scaled_dot_product_attention()
    test_unit_split_heads()
    test_unit_merge_heads()
    test_unit_multihead_attention()

    print("\nRunning integration scenarios...")
    run_attention_scenarios()

    print("\nRunning performance analysis...")
    analyze_attention_complexity()
    print("\nRunning memory overhead analysis...")
    analyze_attention_memory_overhead()

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 12")

# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

Questions 1 and 2 are yours to work out, and every quantity they ask for is derivable from the tables in this module with a calculator. Questions 3 through 5 and the bonus are worked systems analyses. Read those once you have committed to your own answers for the first two, then look for where your reasoning diverged from theirs.

### Question 1: Quadratic Complexity, Memory Footprint & Context Scaling
You benchmarked the memory scaling of the attention logit matrix $S = Q K^\top / \sqrt{d_k}$.

**1. Exact Attention Matrix Memory Calculations**:
- At float32 ($4\text{ bytes/element}$), how many bytes does one head's $n \times n$ logit matrix occupy at $n = 1{,}024$, and at $n = 2{,}048$? Give each in decimal MB.
- By what factor does that memory grow when you double $n$? Explain why the factor is the same at every $n$.

**2. Production Context Length Constraints**:
- GPT-3 175B runs $L = 96$ layers and $H = 96$ heads at $n = 2{,}048$. Multiply out the total logit-matrix memory held for a single sequence.
- Set your answer against the $80\text{ GB}$ of an A100 or H100. What does that comparison rule out about running naive dense attention at $32\text{K}$ or $128\text{K}$ context, and which two techniques named in this module buy the headroom back?

---

### Question 2: Attention vs FFN Memory Bottleneck and Crossover Mechanics
In transformer architectures, attention is frequently the primary memory bottleneck, despite Feed-Forward Networks (FFNs) containing more parameters.

**1. Dimensional Scaling Comparison**:
- For hidden dimension $d$ with the standard $4d$ intermediate expansion, write the FFN's parameter count, its compute FLOPs, and its activation memory in terms of $n$ and $d$.
- Do the same for self-attention with $H$ heads of dimension $d_k = d / H$, splitting its FLOP count into the projection term (in $n d^2$) and the score term (in $n^2 d$). Which of the two grows faster as $n$ rises, and which one does the activation memory follow?

**2. Regime Dominance and Crossover Point**:
- Which subsystem dominates activation memory when $n \ll d$, and which dominates when $n \gg d$? State the inequality that decides it.
- Set attention's activation memory equal to the FFN's and solve for the crossover length $n^*$. Express $n^*$ in terms of $d_k$ alone, then evaluate it for the standard $d_k = 64$. Compare that number against the context lengths production models actually serve.

---

### Question 3: Multi-Head Trade-offs: Subspace Diversity vs Hardware Execution
Comparing $H = 8$ heads of dimension $d_k = 64$ versus $H = 1$ head with $d_k = 512$ (where $d_{\text{embed}} = 512$):

**1. Parameter and FLOP Equivalence**:
- **Parameter Count**:
  - Multi-Head ($H = 8, d_k = 64$): $W_Q, W_K, W_V \in \mathbb{R}^{512 \times 512}$ plus $W_O \in \mathbb{R}^{512 \times 512} \implies 4 \times 512^2 = \mathbf{1{,}048{,}576\text{ parameters}}$.
  - Single-Head ($H = 1, d_k = 512$): $W_Q, W_K, W_V, W_O \in \mathbb{R}^{512 \times 512} \implies 4 \times 512^2 = \mathbf{1{,}048{,}576\text{ parameters}}$.
  - Parameter counts are **identical**.
- **Compute FLOPs**:
  - Multi-Head $Q K^\top$: $8 \times (n \times 64 \times n) = n^2 \times 512$ multiply-accumulates ($2 n^2 d\text{ FLOPs}$).
  - Single-Head $Q K^\top$: $1 \times (n \times 512 \times n) = n^2 \times 512$ multiply-accumulates ($2 n^2 d\text{ FLOPs}$).
  - Theoretical FLOP counts are **identical**.

**2. Systems and Representation Differences**:
- **Activation Memory**: The multi-head model materializes $8$ separate $n \times n$ attention maps, requiring $8 \times n^2 \times 4\text{ bytes}$, whereas the single-head model requires only $1 \times n^2 \times 4\text{ bytes}$ (**$8\times$ memory expansion**).
- **GPU Hardware Efficiency**: Single large GEMMs ($(n \times 512) \times (512 \times n)$) maximize systolic array occupancy on Tensor Cores. Splitting into 8 smaller GEMMs ($(n \times 64) \times (64 \times n)$) introduces kernel launch overhead unless batched into a single batched-strided GEMM.
- **Representational Capacity**: Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions (e.g. tracking syntactic dependencies, semantic coreference, and positional offsets independently).

---

### Question 4: Causal Masking Costs & Arithmetic Intensity
Causal autoregressive masking sets future positions ($j > i$) to $-\infty$.

**1. Computational and Memory Savings in Naive Implementations**:
- In standard vectorized attention, the implementation computes the full dense matrix product $S = Q K^\top$ ($n^2$ dot products), adds an $n \times n$ mask tensor, and evaluates softmax over all $n$ columns.
- Compute saved: **0 FLOPs**. In fact, it expends extra memory bandwidth writing and reading $-\infty$ values across the upper triangle ($\frac{n^2 - n}{2}$ positions).
- Memory saved: **0 bytes**. The full $n \times n$ matrix is still allocated in VRAM.

**2. Production GPU Optimization**:
In production kernels (such as FlashAttention-2 causal mode):
- The GPU thread grid tiles the attention computation into blocks (e.g. $128 \times 128$).
- Thread blocks located entirely in the upper triangle ($j > i$) are **never scheduled**, skipping $\approx 50\%$ of matrix multiply FLOPs.
- Only the boundary tiles along the diagonal require causal masking, dramatically reducing both memory traffic and compute cycles.

---

### Question 5: The Quadratic Memory Challenge & The Softmax Barrier
Materializing the full $(S \times S)$ attention matrix is the primary architectural memory bottleneck.

**1. Exact Numerical Memory Footprint**:
- **$4{,}096$-token sequence with $32$ heads at float32**:
  $$\text{Memory} = 32 \times 4{,}096^2 \times 4\text{ B} = 32 \times 16{,}777{,}216 \times 4\text{ B} = 2{,}147{,}483{,}648\text{ B} = \mathbf{2.15\text{ GB}} \quad (2.0\text{ GiB / layer})$$
  Across a 32-layer model: $32 \times 2.15\text{ GB} = \mathbf{68.72\text{ GB}}$ for a single sequence!
- **$512$-token sequence with $8$ heads at float32**:
  $$\text{Memory} = 8 \times 512^2 \times 4\text{ B} = 8 \times 262{,}144 \times 4\text{ B} = 8{,}388{,}608\text{ B} = \mathbf{8.39\text{ MB}} \quad (8.0\text{ MiB / layer})$$
  Across a 6-layer TinyTorch model: $6 \times 8.39\text{ MB} = \mathbf{50.33\text{ MB}}$.
- **Scaling Behavior**: Doubling sequence length increases attention matrix memory by exactly **$4\times$**.

**2. The Softmax Normalizer Barrier**:
Why can't we easily avoid materializing $S$? Because the softmax operator requires a row-wise reduction:
$$A_{i, j} = \frac{\exp(S_{i, j})}{\sum_{k=1}^S \exp(S_{i, k})}$$
Computing the normalizer requires seeing all keys in row $i$ before outputting any probability. Standard implementations must store all $S_{i, j}$ intermediate values in off-chip global memory (HBM). **FlashAttention** overcomes this barrier using the **online softmax trick**, which maintains running row maximums and partial sums in fast SRAM tiles, avoiding global memory materialization altogether.

---

### Bonus Question: Training Memory Overhead & Optimizer Dynamics
Training requires caching activations during the forward pass to evaluate backward gradients.

**1. Forward vs Backward Activation Footprint**:
- **Inference**: Once the context vector $O = A V$ is computed, the attention weights $A \in \mathbb{R}^{B \times H \times S \times S}$ can be immediately deallocated.
- **Training**: Softmax weights $A$ must be retained in memory for the backward pass because the gradient of softmax depends on its forward output:
  $$\frac{\partial L}{\partial S_{i, j}} = A_{i, j} \left( \frac{\partial L}{\partial A_{i, j}} - \sum_k \frac{\partial L}{\partial A_{i, k}} A_{i, k} \right)$$
  During backward execution, the gradient tensor $\nabla_S$ of the same shape ($B \times H \times S \times S$) is also materialized.
- **Training Overhead**: Training requires $\mathbf{2\times}$ the activation memory of inference for the attention matrix ($A$ retained + $\nabla_S$ allocated).

**2. Optimizer Memory Immunity**:
- The attention matrix $A$ contains **$0$ trainable parameters**. It is a transient dynamic activation generated from input embeddings.
- Consequently, optimizer states (such as Adam's first moment $m_t$ and second moment $v_t$, consuming $8\text{ bytes/param}$) do **not** scale with sequence length $S$. Optimizer state memory scales strictly with model parameters ($W_Q, W_K, W_V, W_O$).

**3. GPT-3 Scale Activation Retention**:
In GPT-3 ($L = 96$ layers, $H = 96$ heads, $S = 2{,}048$):
$$\text{Saved Softmax Weights} = 96 \times 96 \times 2{,}048^2 \times 4\text{ B} \approx \mathbf{154.62\text{ GB per sequence}}$$
This immense activation footprint is why modern large-scale training pipelines employ **Activation Checkpointing** (recomputing attention on the fly during backward) to trade $33\%$ extra compute for drastic VRAM savings.
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Attention Finds Relationships

**What you built:** Attention mechanisms that let tokens interact with each other.

**Why it matters:** Before attention, models processed tokens independently. Attention lets
each token "look at" every other token and decide what's relevant. This is how transformers
understand that "it" refers to "the cat" in a sentence!

Nothing is trained yet, so the demo below plants the match by hand. It makes one key an exact
copy of one query, then shows the attention weight collapsing onto that position.

In the next module, you'll combine attention with MLPs to build full transformer blocks.
"""

# %%
def demo_attention():
    """🎯 See attention compute relationships."""
    print("🎯 AHA MOMENT: Attention Finds Relationships")
    print("=" * 45)

    # Its own generator, so this prints the same numbers standalone or in module order
    demo_rng = np.random.default_rng(1234)

    # 4 tokens with 8-dim embeddings
    Q = demo_rng.standard_normal((1, 4, 8))
    K = demo_rng.standard_normal((1, 4, 8))
    V = demo_rng.standard_normal((1, 4, 8))

    # Plant the relationship. Key 2 becomes an exact copy of query 0, standing in for
    # the key that a pronoun like "it" would be hunting for.
    match_pos = 2
    K[0, match_pos] = Q[0, 0]

    output, weights = scaled_dot_product_attention(Tensor(Q), Tensor(K), Tensor(V))
    token0 = weights.data[0, 0, :]

    print("Sequence length: 4 tokens")
    print("Embedding dim:   8")
    print(f"Planted match:   query 0 matches key {match_pos}")
    print(f"\nAttention weights shape: {weights.shape}")

    print(f"\nToken 0 attention: {token0.round(3)}")
    print(f"Strongest match:   position {int(token0.argmax())}, holding {token0.max():.1%} of the weight")
    print(f"Row sum:           {token0.sum():.1f} (a probability distribution over the 4 positions)")

    drift = float(np.abs(output.data[0, 0] - V[0, match_pos]).max())
    print(f"\nToken 0's output lands within {drift:.2f} of value vector {match_pos},")
    print("so it is a near copy of the matched value rather than an average of all four.")

    print("\n✨ Attention routed token 0 to the position we planted!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_attention()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Attention

Congratulations! You've built the attention mechanism that revolutionized deep learning!

### Key Accomplishments
- **Built scaled dot-product attention** with O(n^2) complexity understanding
- **Implemented multi-head attention** for parallel relationship learning
- **Experienced quadratic memory scaling** firsthand through analysis functions
- **Tested causal masking** for language modeling applications
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Quadratic scaling**: Attention memory grows as n^2, limiting context lengths
- **Memory bottlenecks**: Attention matrices dominate memory in transformers (~154.62 GB per sequence at GPT-3 scale)
- **Multi-head parallelism**: Different heads can specialize in different relationship types
- **Production challenges**: Understanding why attention efficiency research is crucial

### Ready for Next Steps
Your attention implementation is the core mechanism that enables modern language models!
Export with: `tito module complete 12`

**Next**: Module 13 will combine attention with feed-forward layers to build complete transformer blocks!
"""
