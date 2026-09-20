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
# Module 11: Embeddings - Converting Tokens to Learnable Representations

Welcome to Module 11! You're about to build embedding layers that convert discrete tokens into dense, learnable vectors — the foundational bridge between symbolic text processing and deep neural representations.

## 🔗 Prerequisites & Progress
**You've Built**: Complete training pipeline with autograd, optimizers, data loaders, 2D convolutions, and subword tokenization (`Tensor`, `Function`, `Linear`, `Conv2d`, `BPETokenizer`).
**You'll Build**: Vector table lookups (`Embedding`), trainable and mathematical positional encodings (`PositionalEncoding`, `create_sinusoidal_embeddings`), and an integrated datapath (`EmbeddingLayer`).
**You'll Enable**: Continuous sequence modeling powering Multi-Head Attention (`12_attention`) and full Transformers (`13_transformers`).

<div align="center">
  <img src="embeddings_blueprint.svg" alt="TinyTorch Architecture Blueprint: Module 11 Embeddings" width="380px">
</div>

### Architectural Roadmap

| Stage | Subsystem | Primitives & Capabilities | Status |
| :--- | :--- | :--- | :--- |
| **Modules 01–08** | Foundation & Training | `Tensor`, `Function`, `Linear`, `SGD`, `Adam`, `Trainer` | Completed |
| **Modules 09–10** | Vision & Tokenization | `Conv2d`, `MaxPool2d`, `BPETokenizer`, `CharTokenizer` | Completed |
| **Module 11** | **Continuous Embeddings** | `Embedding`, `PositionalEncoding`, `create_sinusoidal_embeddings`, `EmbeddingLayer` | **Active Subsystem** |
| **Modules 12–13** | Language & Attention | `MultiHeadAttention`, `TransformerBlock`, `CausalSelfAttention` | Downstream Consumers |

## 🎯 Learning Objectives
By the end of this module, you will:
1. **Implement efficient token-to-vector table lookups** using tensor indexing without constructing wasteful one-hot matrix multiplications.
2. **Derive the backward scatter-add gradient route** and understand why repeated token indices require race-free accumulation (`np.add.at`).
3. **Build learned positional encodings** that allow neural networks to adaptively represent sequence positions through backpropagation.
4. **Construct fixed sinusoidal positional encodings** using Vaswani et al. (2017) frequency harmonics for length generalization without added weights.
5. **Architect an integrated production `EmbeddingLayer`** that handles optional $\sqrt{d_{\text{embed}}}$ scaling, positional injection, and variable batch sequences.
6. **Analyze memory scaling and bandwidth bottlenecks** governing billion-parameter embedding tables in production LLMs.

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/11_embeddings/embeddings.ipynb`
**Building Side:** Code exports to `tinytorch.core.embeddings`

```python
# How to use this module:
from tinytorch.core.embeddings import Embedding, PositionalEncoding, create_sinusoidal_embeddings, EmbeddingLayer
```

**Why this matters:**
- **Learning:** Transforms discrete symbol IDs ($\mathbb{Z}^{B \times T}$) into continuous geometry ($\mathbb{R}^{B \times T \times D}$) where semantic similarity is measured by dot products.
- **Production:** Directly mirrors PyTorch's `torch.nn.Embedding` with gradient accumulation mechanics and production positional encoding strategies.
- **Consistency:** Unifies table lookups and temporal coordinates in `tinytorch.core.embeddings` for downstream attention layers.
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

| Dependency | Origin | Purpose in Module 11 | Systems Invariant |
| :--- | :--- | :--- | :--- |
| `Tensor` | Module 01 (`core.tensor`) | Wraps embedding weights, token indices, and activations | Strided memory buffer and gradient storage container |
| `Function` | Module 06 (`core.tensor`) | Extensible computational graph node base class | Enforces separation of forward evaluation and backward adjoint |
| `autograd` | Module 06 (`core.autograd`) | Backward tape and automatic gradient backpropagation | Accumulates gradients into leaf weight parameter `.grad` |
| `BPETokenizer` | Module 10 (`core.tokenization`) | Upstream text tokenizer producing discrete token IDs | Maps variable-length strings into static vocabulary bounds $[0, V-1]$ |
| `numpy` | External | Array manipulation, fast vectorized slicing, and math | Provides `np.add.at` for race-free scatter-add accumulation |

### Ingestion & Transformation Pipeline

| Pipeline Stage | Subsystem / Operator | Mathematical Mapping | Tensor Space & Dimensions |
| :--- | :--- | :--- | :--- |
| **0. Raw Input** | Upstream Text Source | Raw string token stream | `"machine learning"` |
| **1. Tokenization** | `BPETokenizer` (M10) | Discrete subword vocabulary lookup | $\mathbf{x}_{\text{ids}} = [1042, 3819] \in \mathbb{Z}^{B \times T}$ |
| **2. Embedding Gather**| `EmbeddingLayer` (M11) | Weight lookup matrix indexing | $\mathbf{E}_{\text{token}} \in \mathbb{R}^{B \times T \times D}$ |
| **3. Positional Encoding**| Learned / Sinusoidal (M11) | Coordinate position addition | $\mathbf{H} = \mathbf{E}_{\text{token}} + \mathbf{P} \in \mathbb{R}^{B \times T \times D}$ |
| **4. Attention Mixing**| `MultiHeadAttention` (M12) | Dynamic contextual routing | $\mathbf{Z} \in \mathbb{R}^{B \times T \times D}$ |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp core.embeddings
#| export

import numpy as np
rng = np.random.default_rng(7)
import math
from typing import List, Optional, Tuple

# Import from previous modules - following dependency chain
from tinytorch.core.tensor import Tensor

# Module 06: Function base class and autograd, so embedding lookups record their backward pass
from tinytorch.core.tensor import Function
import tinytorch.core.autograd  # completes every operation with its backward half

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
r"""
## 💡 Introduction: Why Embeddings?

Neural networks operate on dense vector spaces where linear transformations, inner products, and gradient descent can operate. However, human language arrives as discrete, symbolic tokens ($t_i \in \{0, 1, \dots, V-1\}$). Embeddings provide the foundational bridge converting sparse, discrete tokens into dense, continuous vectors where geometric proximity reflects semantic similarity.

### The Token-to-Vector Challenge

Consider token indices emitted by our BPE tokenizer: `[1, 42, 7]`. How do we convert these arbitrary integers into rich, learnable representations that an attention layer can process?

<div align="center">
  <img src="embeddings_pipeline.svg" alt="The Embedding Pipeline: Token Lookup and Positional Addition" width="700px">
</div>

### Comparing Input Representations

| Dimension | Discrete Token IDs | One-Hot Encoding | Continuous Dense Embeddings |
| :--- | :--- | :--- | :--- |
| **Data Type** | Integers $t_i \in \mathbb{Z}$ | Sparse binary vector $\{0, 1\}^V$ | Dense floating-point vector $\mathbb{R}^{d_{\text{embed}}}$ |
| **Memory per Token** | $4\text{ bytes}$ (`int32`) | $V \times 4\text{ bytes}$ ($200\text{ KB}$ for $V=50\text{K}$) | $d_{\text{embed}} \times 4\text{ bytes}$ ($2\text{ KB}$ for $d=512$) |
| **Semantic Similarity** | Undefined (distance is arbitrary) | Orthogonal ($\mathbf{x}_i^\top \mathbf{x}_j = 0$ for all $i \neq j$) | Angle/Dot Product ($\cos(\theta) \in [-1, 1]$) |
| **Computational Mechanism** | Table Gather ($W[t_i, :]$) | Huge Matrix Multiply ($\mathbf{x}^\top W$) | Direct memory indexing $\mathcal{O}(d_{\text{embed}})$ |

### The Four-Stage Embedding Datapath

1. **Token Embedding Gather ($W_{\text{token}}[t_i, :]$)**: Gathers a dense feature vector representing the static lexical and semantic meaning of token ID $t_i$.
2. **Positional Encoding ($P[pos, :]$)**: Injects temporal sequence awareness into the permutation-invariant representation so that `"dog bites man"` $\neq$ `"man bites dog"`.
3. **Magnitude Scaling ($\sqrt{d_{\text{embed}}}$)**: In standard Transformer architectures (Vaswani et al., 2017), token embeddings are scaled by $\sqrt{d_{\text{embed}}}$ to stabilize activation variance relative to position signals.
4. **Element-wise Composition ($H = E \cdot \sqrt{d_{\text{embed}}} + P$)**: Produces the final position-aware contextual tensor $H \in \mathbb{R}^{B \times T \times d_{\text{embed}}}$ dispatched to Multi-Head Attention.
"""

# %% [markdown]
r"""
## 📐 Foundations: Embedding Strategies

Different embedding approaches make distinct architectural trade-offs across memory residency, computational complexity, and sequence extrapolation capabilities.

### Token Embedding Lookup: Forward Gather & Backward Scatter-Add

In mathematical terms, an embedding layer is a parameter weight matrix $W \in \mathbb{R}^{V \times d_{\text{embed}}}$. For an input batch of token IDs $X \in \mathbb{Z}^{B \times T}$ where each index $t \in \{0, 1, \dots, V-1\}$:

<div align="center">
  <img src="embedding_gather_scatter.svg" alt="Embedding Table Mechanics: Forward Gather and Backward Scatter-Add" width="720px">
</div>

1. **Forward Pass (Row Gather)**:
   Rather than constructing an enormous one-hot tensor $\mathbf{x} \in \{0, 1\}^{B \times T \times V}$ and computing a matrix multiplication $\mathbf{x} W$, hardware gathers rows directly by physical pointer indexing:
   $$E_{b, t, :} = W[X_{b, t}, :]$$
   This achieves $\mathcal{O}(1)$ pointer arithmetic per token, avoiding $\mathcal{O}(V)$ zero multiplications and gigabytes of intermediate activation memory.

2. **Backward Pass (Scatter-Add Accumulation)**:
   When a sequence contains repeated tokens (e.g. `[2, 0, 2]`), multiple output rows propagate gradients back to the same row in weight matrix $W$:
   $$\frac{\partial \mathcal{L}}{\partial W[v, :]} = \sum_{(b, t): X_{b, t} = v} \frac{\partial \mathcal{L}}{\partial E_{b, t, :}}$$
   
   > [!WARNING] **The Silent In-Place Race Condition**
   > In Python and NumPy, naive indexing assignment `grad_weight[indices] += grad_output` fails silently when `indices` contains duplicate entries! Because indexed slice assignment executes in unbuffered parallel copies, duplicate row writes overwrite each other rather than accumulating. TinyTorch uses `np.add.at(grad_weight, indices, grad_output)` to guarantee atomic, race-free gradient accumulation.

---

### Positional Encoding Strategies

Because the self-attention mechanism in Transformers is permutation-equivariant (it treats an unordered set of vectors identically regardless of temporal sequence), position information must be explicitly injected into the embeddings:

$$H_{b, t, :} = E_{\text{token}}[X_{b, t}, :] + P[t, :]$$

| Strategy | Mathematical Formulation | Trainable Params | Extrapolation Bound | Real-World Models |
| :--- | :--- | :--- | :--- | :--- |
| **Learned Positional** | $P \in \mathbb{R}^{T_{\text{max}} \times d_{\text{embed}}}$ | $T_{\text{max}} \times d_{\text{embed}}$ | Hard limit at $T_{\text{max}}$ (crashes if $T > T_{\text{max}}$) | BERT, GPT-2, OPT |
| **Sinusoidal Positional** | $P_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d}}\right)$ | $0$ (fixed math function) | Extrapolates to arbitrary $T$ | Original Transformer (Vaswani et al.) |
| **Rotary (RoPE)** | Complex 2D rotation $\mathbf{R}_{\Theta, t}^d \mathbf{q}_t$ | $0$ (applied to Q/K vectors) | Superior relative length generalization | Llama 2/3, Mistral, Gemma |
| **Relative Bias (ALiBi)** | Attention logit bias $-m \cdot \vert i - j \vert$ | $0$ (linear penalty in attention) | Extrapolates to $10\times$ training length | BLOOM, Falcon, MPT |
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Embedding Systems

Let's implement embedding systems from basic token lookup to sophisticated position-aware representations. We'll start with the core embedding layer and work up to complete systems.
"""

# %% [markdown]
"""
### Gradient Computation for Embedding Lookups

Now that you understand how embedding lookups work (index → row of the weight matrix),
let's think about how gradients flow backward through this operation.

The forward pass is a **gather** — we select rows from the weight matrix by index.
The backward pass is a **scatter** — we distribute gradients back to the rows that were selected.

```
Forward (gather):                    Backward (scatter):
Weight Table:                        Gradient Table:
  Row 0: [0.1, 0.2]  ← selected       Row 0: [2, 2]  ← accumulated (selected twice!)
  Row 1: [0.3, 0.4]                    Row 1: [0, 0]  ← not selected
  Row 2: [0.5, 0.6]  ← selected       Row 2: [1, 1]  ← selected once

Indices: [0, 2, 0]                   grad_output: [[1,1], [1,1], [1,1]]
Output:  [[0.1, 0.2],               Row 0 gets grad[0] + grad[2] = [2, 2]
          [0.5, 0.6],               Row 2 gets grad[1] = [1, 1]
          [0.1, 0.2]]
```

**Key insight**: When the same token appears multiple times in a sequence (like word "the"),
its embedding row accumulates gradients from every position. This is why `np.add.at` is
essential — standard indexing would overwrite instead of accumulating.
"""

# %% nbgrader={"grade": false, "grade_id": "embedding-backward", "solution": true}
#| export
class EmbeddingFunction(Function):
    """
    The embedding lookup operation: forward gathers rows, backward scatters gradients back.

    **Mathematical Rule:** If Y = Embedding[indices], then:
    - ∂Loss/∂Embedding[i] = sum of all gradients where index==i

    Embedding lookup is a gather operation. The backward
    is a scatter operation that accumulates gradients to the embedding weights.
    """

    def forward(self, weight):
        """Gather one row per token id: weight[self.indices] (self.indices is an int array)."""
        return weight[self.indices]


    def backward(self, grad_output):
        """
        Compute gradient for embedding lookup.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple with single gradient for weight tensor

        **Mathematical Foundation:**
        - ∂(Embedding[indices])/∂Embedding = scatter gradients to selected rows
        - Multiple indices can point to same embedding → gradients accumulate

        TODO: Implement gradient computation for embedding lookup.

        APPROACH:
        1. Extract weight tensor from self.inputs
        2. Initialize grad_weight to None
        3. If weight requires gradients:
           - Create zeros array: grad_weight = np.zeros_like(weight.data)
           - Flatten indices: indices_flat = np.asarray(self.indices).flatten()
           - Reshape grad_output: match flattened indices with embedding dimension
           - Use np.add.at to accumulate gradients: np.add.at(grad_weight, indices_flat, grad_output_reshaped)
        4. Return tuple (grad_weight,)

        EXAMPLE:
        >>> vocab = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)  # 3 words, 2D
        >>> indices = Tensor([0, 2, 0])  # Select words 0, 2, 0
        >>> output = EmbeddingFunction.apply(vocab, indices=indices.data.astype(int))
        >>> # output = [[0.1, 0.2], [0.5, 0.6], [0.1, 0.2]]
        >>> # During backward: grad_output = [[1, 1], [1, 1], [1, 1]]
        >>> # grad_vocab[0] accumulates twice: [1, 1] + [1, 1] = [2, 2]
        >>> # grad_vocab[2] once: [1, 1]

        HINTS:
        - Embedding lookup is a gather operation; backward is scatter
        - np.add.at accumulates gradients for repeated indices
        - Reshape grad_output to match: (num_indices, embedding_dim)
        - Return as single-element tuple: (grad_weight,)
        """
        ### BEGIN SOLUTION
        weight, = self.inputs
        grad_weight = None

        if weight.requires_grad:
            # Initialize gradient with zeros
            grad_weight = np.zeros_like(weight.data)

            # Scatter gradients back to embedding weights
            # np.add.at accumulates gradients for repeated indices
            indices_flat = np.asarray(self.indices).flatten()
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])

            np.add.at(grad_weight, indices_flat, grad_output_reshaped)

        return (grad_weight,)
        ### END SOLUTION

# %% [markdown]
"""
### Embedding: The Lookup Table

With `EmbeddingFunction.backward` written, the forward direction is almost anticlimactic:
an embedding layer is a matrix, and a lookup is one row of it.

```
weight: (vocab_size, embed_dim)      the whole table
tokens: [7, 3, 7]                    the ids you want
output: weight[[7, 3, 7]]            three rows, one per token
```

Two things make it worth its own class. It validates that every id is in range,
which turns a confusing IndexError deep in NumPy into a message that names the
offending token. And it sets `requires_grad=True` on the weight, without which
the backward you just wrote would never be reached at all.
"""

# %% nbgrader={"grade": false, "grade_id": "embedding-init", "solution": true}
#| export
class Embedding:
    """
    Learnable embedding layer that maps token indices to dense vectors.

    This is the fundamental building block for converting discrete tokens
    into continuous representations that neural networks can process.

    We'll build this in two steps: first initialize the weight matrix,
    then implement the forward lookup.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize embedding layer with Xavier-uniform weights.

        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            embed_dim: Dimension of embedding vectors

        TODO: Initialize the embedding weight matrix

        APPROACH:
        1. Store vocab_size and embed_dim
        2. Create weight matrix of shape (vocab_size, embed_dim)
        3. Use Xavier/Glorot uniform initialization: limit = sqrt(6 / (V + D))

        HINT: rng.uniform(-limit, limit, (vocab_size, embed_dim))
        """
        ### BEGIN SOLUTION role="scaffold"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Xavier initialization for better gradient flow.
        # requires_grad=True is not decoration: apply() only records the
        # lookup when the weight requires grad, so without this flag the
        # backward you wrote above is never reached and the table never learns.
        limit = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Tensor(
            rng.uniform(-limit, limit, (vocab_size, embed_dim)),
            requires_grad=True
        )
        ### END SOLUTION

    def forward(self, indices: Tensor) -> Tensor:
        """
        Forward pass: lookup embeddings for given indices.

        Args:
            indices: Token indices of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Embedded vectors of shape (*indices.shape, embed_dim)

        TODO: Implement embedding lookup with validation and gradient tracking

        APPROACH:
        1. Validate indices are within [0, vocab_size)
        2. Run the lookup through EmbeddingFunction.apply: its forward does the
           numpy advanced indexing weight[indices], and apply() records it for backward

        HINTS:
        - EmbeddingFunction.apply(self.weight, indices=indices.data.astype(int))
        """
        ### BEGIN SOLUTION
        # Handle input validation
        # Tensor stores float32, but token IDs must still be finite integers.
        if not np.all(np.isfinite(indices.data)) or np.any(indices.data != np.floor(indices.data)):
            raise ValueError("Embedding token IDs must be finite integers")
        if np.any(indices.data >= self.vocab_size) or np.any(indices.data < 0):
            min_idx = int(np.min(indices.data))
            max_idx = int(np.max(indices.data))
            raise ValueError(
                f"Embedding index out of range for vocabulary size {self.vocab_size}\n"
                f"  ❌ Found indices: min={min_idx}, max={max_idx} (valid range: 0 to {self.vocab_size - 1})\n"
                f"  💡 Token IDs must be within the vocabulary. IDs >= vocab_size reference non-existent tokens\n"
                f"  🔧 Check your tokenizer output, or increase vocab_size to at least {max_idx + 1}"
            )

        # Perform embedding lookup through the operation (advanced indexing inside
        # EmbeddingFunction.forward, equivalent to one-hot multiplication but much
        # more efficient). Module 06's apply() records it for backward.
        return EmbeddingFunction.apply(self.weight, indices=indices.data.astype(int))
        ### END SOLUTION

    def __call__(self, indices: Tensor) -> Tensor:
        """Allows the embedding to be called like a function."""
        return self.forward(indices)

    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return [self.weight]

    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"

# %% [markdown]
"""
### 🧪 Unit Test: Embedding.__init__

**What we're testing**: Weight matrix initialization with correct shape and Xavier scaling
**Why it matters**: Bad initialization causes vanishing/exploding gradients from the start
**Expected**: Weight shape is (vocab_size, embed_dim), values are within Xavier bounds
"""

# %% nbgrader={"grade": true, "grade_id": "test-embedding-init", "locked": true, "points": 5}
def test_unit_embedding_init():
    """🧪 Test Embedding.__init__ implementation."""
    print("🧪 Unit Test: Embedding.__init__...")

    embed = Embedding(vocab_size=100, embed_dim=64)

    # Check stored attributes
    assert embed.vocab_size == 100, f"Expected vocab_size=100, got {embed.vocab_size}"
    assert embed.embed_dim == 64, f"Expected embed_dim=64, got {embed.embed_dim}"

    # Check weight shape
    assert embed.weight.shape == (100, 64), f"Expected weight shape (100, 64), got {embed.weight.shape}"

    # Check Xavier bounds: limit = sqrt(6 / (100 + 64)) ≈ 0.191
    limit = math.sqrt(6.0 / (100 + 64))
    assert np.all(embed.weight.data >= -limit - 1e-6), "Weights should be >= -limit"
    assert np.all(embed.weight.data <= limit + 1e-6), "Weights should be <= limit"

    print("✅ Embedding.__init__ works correctly!")

if __name__ == "__main__":
    test_unit_embedding_init()

# %% [markdown]
"""
### 🧪 Unit Test: Embedding.forward

This test validates our Embedding class works correctly with various token indices and batch configurations.

**What we're testing**: Token embedding lookup and parameter management
**Why it matters**: Foundation for all NLP models - if embedding fails, nothing works
**Expected**: Correct shape output, consistent lookups, proper parameter access
"""

# %% nbgrader={"grade": true, "grade_id": "test-embedding", "locked": true, "points": 10}
def test_unit_embedding():
    """🧪 Test Embedding layer implementation."""
    print("🧪 Unit Test: Embedding Layer...")

    # Test 1: Basic embedding creation and forward pass
    embed = Embedding(vocab_size=100, embed_dim=64)

    # Single sequence
    tokens = Tensor([1, 2, 3])
    output = embed.forward(tokens)

    assert output.shape == (3, 64), f"Expected shape (3, 64), got {output.shape}"
    assert len(embed.parameters()) == 1, "Should have 1 parameter (weight matrix)"
    assert embed.parameters()[0].shape == (100, 64), "Weight matrix has wrong shape"

    # Test 2: Batch processing
    batch_tokens = Tensor([[1, 2, 3], [4, 5, 6]])
    batch_output = embed.forward(batch_tokens)

    assert batch_output.shape == (2, 3, 64), f"Expected batch shape (2, 3, 64), got {batch_output.shape}"

    # Test 3: Embedding lookup consistency
    single_lookup = embed.forward(Tensor([1]))
    batch_lookup = embed.forward(Tensor([[1]]))

    # Should get same embedding for same token
    assert np.allclose(single_lookup.data[0], batch_lookup.data[0, 0]), "Inconsistent embedding lookup"

    # Test 4: Parameter access
    params = embed.parameters()
    assert len(params) == 1, "Should have 1 parameter"

    print("✅ Embedding layer works correctly!")

if __name__ == "__main__":
    test_unit_embedding()

# %% [markdown]
"""
### 🧪 Unit Test: Embedding gradients

The forward pass is a lookup, so it is hard to get wrong. The backward pass is
where embeddings are actually interesting, and where the bug is invisible: if you
write `grad_weight[indices] = grad` instead of `np.add.at`, a token that appears
twice in a sequence keeps only one of its two gradients. The forward output is
identical, the loss still falls, and the table simply learns more slowly than it
should for exactly the tokens that matter most.

**What we're testing**: Scatter-add accumulation for repeated indices, and that
untouched rows receive no gradient at all
**Why it matters**: Assignment instead of accumulation silently halves the signal
for frequent tokens, which are the ones a language model sees most
**Expected**: Row 0 (used twice) gets twice the gradient of row 2 (used once);
row 1 (unused) stays exactly zero
"""

# %% nbgrader={"grade": true, "grade_id": "test-embedding-backward", "locked": true, "points": 10}
def test_unit_embedding_backward():
    """🧪 Test Embedding gradient accumulation."""
    print("🧪 Unit Test: Embedding gradients...")

    embed = Embedding(vocab_size=4, embed_dim=2)

    # Token 0 appears twice, token 2 once, tokens 1 and 3 not at all.
    tokens = Tensor([0, 2, 0])
    output = embed.forward(tokens)
    output.sum().backward()

    grad = embed.weight.grad
    assert grad is not None, "No gradient reached embed.weight"

    # Every position contributes a gradient of 1 to its row.
    assert np.allclose(grad[0], [2.0, 2.0]), (
        f"Row 0 is used twice so its gradient should be [2, 2], got {grad[0]}. "
        "Indexed assignment overwrites instead of accumulating; use np.add.at."
    )
    assert np.allclose(grad[2], [1.0, 1.0]), (
        f"Row 2 is used once so its gradient should be [1, 1], got {grad[2]}"
    )
    assert np.allclose(grad[1], [0.0, 0.0]), (
        f"Row 1 is never looked up so its gradient must stay zero, got {grad[1]}"
    )
    assert np.allclose(grad[3], [0.0, 0.0]), (
        f"Row 3 is never looked up so its gradient must stay zero, got {grad[3]}"
    )

    print("✅ Embedding gradients accumulate correctly!")

if __name__ == "__main__":
    test_unit_embedding_backward()

# %% [markdown]
r"""
### Learned Positional Encoding: Broadcast Addition & Dual Gradient Routes

In learned positional encoding (utilized by GPT-2 and BERT), position vectors are stored in a trainable matrix $P \in \mathbb{R}^{T_{\text{max}} \times d_{\text{embed}}}$. During forward execution, the position slice $P[0:T, :]$ broadcasts across the batch dimension and adds element-wise to the token embeddings:

$$H_{b, t, d} = E_{b, t, d} + P_{t, d} \quad \text{for } b \in [0, B-1], \, t \in [0, T-1], \, d \in [0, d_{\text{embed}}-1]$$

<div align="center">
  <img src="embedding_gradient_routes.svg" alt="Two Gradient Routes Through an Embedding Layer" width="700px">
</div>

### Contrasting the Two Gradient Routes

When the loss gradient $\frac{\partial \mathcal{L}}{\partial H} \in \mathbb{R}^{B \times T \times d_{\text{embed}}}$ arrives at the embedding layer during backpropagation:

| Property | Token Weight Gradient ($\frac{\partial \mathcal{L}}{\partial W}$) | Position Weight Gradient ($\frac{\partial \mathcal{L}}{\partial P}$) |
| :--- | :--- | :--- |
| **Routing Key** | Token Identity / Vocabulary ID ($X_{b, t}$) | Temporal Position Index ($t \in [0, T-1]$) |
| **Accumulation Mechanism** | Non-contiguous scatter-add (`np.add.at`) | Structured sum across batch dimension (`grad.sum(axis=0)`) |
| **Sparsity** | Highly sparse (only vocabulary IDs present in batch update) | Dense up to sequence length $T$ (all active positions update) |
| **Parameter Bound** | $V \times d_{\text{embed}}$ (typically $50\text{K} \times 768 \approx 38.4\text{M}$ params) | $T_{\text{max}} \times d_{\text{embed}}$ (typically $1{,}024 \times 768 \approx 0.78\text{M}$ params) |

**Why Learned Positions Work**: The optimizer learns task-specific coordinate geometry (e.g. distinguishing sentence start tokens, delimiter boundaries, or rhythmic syntactic dependencies) through standard end-to-end backpropagation.
"""

# %% [markdown]
"""
### Implementing Learned Positional Encoding

Let's build trainable positional embeddings that can learn position-specific patterns for our specific task.
"""

# %% nbgrader={"grade": false, "grade_id": "positional-encoding-init", "solution": true}
#| export
class PositionalEncoding:
    """
    Learnable positional encoding layer.

    Adds trainable position-specific vectors to token embeddings,
    allowing the model to learn positional patterns specific to the task.

    We'll build this in two steps: initialize the position matrix,
    then implement the forward pass that adds positions to embeddings.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        """
        Initialize learnable positional encoding.

        Args:
            max_seq_len: Maximum sequence length to support
            embed_dim: Embedding dimension (must match token embeddings)

        TODO: Create the position embedding matrix

        APPROACH:
        1. Store max_seq_len and embed_dim
        2. Create position_embeddings matrix of shape (max_seq_len, embed_dim)
        3. Initialize positions with uniform bounds ±sqrt(2 / embed_dim)

        HINT: limit = sqrt(2.0 / embed_dim), then uniform(-limit, limit)
        """
        ### BEGIN SOLUTION role="scaffold"
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Initialize position embedding matrix
        # This scale depends on embed_dim, not vocabulary size. For large
        # vocabularies it can exceed the token initialization scale; optional
        # token scaling in EmbeddingLayer changes their relative magnitudes.
        # Learned positions are returned by parameters(), so the optimizer will
        # try to update them -- they have to carry gradients for that to mean
        # anything. (Sinusoidal encodings are the opposite: fixed by design.)
        limit = math.sqrt(2.0 / embed_dim)
        self.position_embeddings = Tensor(
            rng.uniform(-limit, limit, (max_seq_len, embed_dim)),
            requires_grad=True
        )
        ### END SOLUTION

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """
        Add positional encodings to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, embed_dim)
            start_pos: Position of the first token in x. 0 for a whole sequence;
                       Module 18's KV cache will feed one token at a time and pass
                       the number of tokens already cached.

        Returns:
            Position-encoded embeddings of same shape

        TODO: Validate input and add position embeddings

        APPROACH:
        1. Validate input is 3D with correct embed_dim and seq_len <= max
        2. Slice position_embeddings[start_pos:start_pos + seq_len] (start_pos is 0 except during cached generation)
        3. Reshape to (1, seq_len, embed_dim) for batch broadcasting
        4. Add to input embeddings

        HINTS:
        - Use pos_embeddings.reshape(1, seq_len, embed_dim) to add the batch dimension.
          Do NOT write Tensor(pos_embeddings.data[np.newaxis]): reading .data and
          re-wrapping it builds a new leaf tensor, which cuts these positions out of
          the graph. The forward output looks identical and the gradient silently
          never reaches the parameter.
        - Use x + pos_embeddings_batched for element-wise addition
        """
        ### BEGIN SOLUTION role="scaffold"
        if len(x.shape) == 2:
            raise ValueError(
                f"Expected 3D input (batch, seq, embed), got 2D: {x.shape}\n"
                f"  ❌ Missing batch dimension\n"
                f"  💡 PositionalEncoding expects batched embeddings, not single sequences\n"
                f"  🔧 Add batch dim: x.reshape(1, {x.shape[0]}, {x.shape[1]})"
            )
        elif len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq, embed), got {len(x.shape)}D: {x.shape}\n"
                f"  ❌ Input must have exactly 3 dimensions\n"
                f"  💡 PositionalEncoding expects shape (batch_size, sequence_length, embedding_dim)"
            )

        batch_size, seq_len, embed_dim = x.shape
        if not isinstance(start_pos, (int, np.integer)) or start_pos < 0:
            raise ValueError("start_pos must be a nonnegative integer")
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence runs past the maximum: positions {start_pos}..{start_pos + seq_len - 1} with max_seq_len={self.max_seq_len}\n"
                f"  ❌ Input has {seq_len} positions starting at {start_pos}, but only {self.max_seq_len} are available\n"
                f"  💡 Learned positional encodings have a fixed maximum length set at initialization\n"
                f"  🔧 Either truncate input to {self.max_seq_len} tokens, or create a new PositionalEncoding(max_seq_len={start_pos + seq_len}, ...)"
            )

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {embed_dim}, expected {self.embed_dim}\n"
                f"  ❌ PositionalEncoding was created with embed_dim={self.embed_dim}, but input has embed_dim={embed_dim}\n"
                f"  💡 Token embeddings and positional encodings must have the same dimension to be added together\n"
                f"  🔧 Ensure your Embedding layer uses embed_dim={self.embed_dim}, or create PositionalEncoding(embed_dim={embed_dim}, ...)"
            )

        # Slice position embeddings for this sequence length using Tensor slicing
        pos_embeddings = self.position_embeddings[start_pos:start_pos + seq_len]  # (seq_len, embed_dim)

        # Reshape to add batch dimension: (1, seq_len, embed_dim).
        # Use Tensor.reshape, not Tensor(pos_embeddings.data[np.newaxis]).
        # Reading .data and re-wrapping it builds a brand new leaf tensor, which
        # cuts these positions out of the graph -- the forward output looks
        # identical and the gradient silently never reaches the parameter.
        pos_embeddings_batched = pos_embeddings.reshape(1, seq_len, embed_dim)

        # Add positional information
        result = x + pos_embeddings_batched

        return result
        ### END SOLUTION

    def __call__(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """Allows the positional encoding to be called like a function."""
        return self.forward(x, start_pos)

    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return [self.position_embeddings]

    def __repr__(self):
        return f"PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"

# %% [markdown]
"""
### 🧪 Unit Test: PositionalEncoding.__init__

**What we're testing**: Position embedding matrix initialization with correct shape
**Why it matters**: Wrong shape or scale breaks the additive position signal
**Expected**: Matrix shape is (max_seq_len, embed_dim), values are small (additive)
"""

# %% nbgrader={"grade": true, "grade_id": "test-positional-init", "locked": true, "points": 5}
def test_unit_positional_encoding_init():
    """🧪 Test PositionalEncoding.__init__ implementation."""
    print("🧪 Unit Test: PositionalEncoding.__init__...")

    pos_enc = PositionalEncoding(max_seq_len=512, embed_dim=64)

    # Check stored attributes
    assert pos_enc.max_seq_len == 512, f"Expected max_seq_len=512, got {pos_enc.max_seq_len}"
    assert pos_enc.embed_dim == 64, f"Expected embed_dim=64, got {pos_enc.embed_dim}"

    # Check position embeddings shape
    assert pos_enc.position_embeddings.shape == (512, 64), \
        f"Expected shape (512, 64), got {pos_enc.position_embeddings.shape}"

    # Check values are reasonably small (additive initialization)
    limit = math.sqrt(2.0 / 64)
    assert np.all(pos_enc.position_embeddings.data >= -limit - 1e-6), "Values should be >= -limit"
    assert np.all(pos_enc.position_embeddings.data <= limit + 1e-6), "Values should be <= limit"

    # Check parameters returns the position embeddings
    params = pos_enc.parameters()
    assert len(params) == 1, f"Expected 1 parameter, got {len(params)}"

    print("✅ PositionalEncoding.__init__ works correctly!")

if __name__ == "__main__":
    test_unit_positional_encoding_init()

# %% [markdown]
"""
### 🧪 Unit Test: PositionalEncoding.forward

This test validates our PositionalEncoding class works correctly with various sequence lengths and configurations.

**What we're testing**: Position embedding consistency and shape handling
**Why it matters**: Position awareness is critical for sequence understanding
**Expected**: Consistent encodings, correct shapes, proper parameter management
"""

# %% nbgrader={"grade": true, "grade_id": "test-positional", "locked": true, "points": 10}
def test_unit_positional_encoding():
    """🧪 Test Positional Encoding implementation."""
    print("🧪 Unit Test: Positional Encoding...")

    # Test 1: Basic functionality
    pos_enc = PositionalEncoding(max_seq_len=512, embed_dim=64)

    # Create sample embeddings
    embeddings = Tensor(rng.standard_normal((2, 10, 64)))
    output = pos_enc.forward(embeddings)

    assert output.shape == (2, 10, 64), f"Expected shape (2, 10, 64), got {output.shape}"

    # Test 2: Position consistency
    # Same position should always get same encoding
    emb1 = Tensor(np.zeros((1, 5, 64)))
    emb2 = Tensor(np.zeros((1, 5, 64)))

    out1 = pos_enc.forward(emb1)
    out2 = pos_enc.forward(emb2)

    assert np.allclose(out1.data, out2.data), "Position encodings should be consistent"

    # Test 3: Different positions get different encodings
    short_emb = Tensor(np.zeros((1, 3, 64)))
    long_emb = Tensor(np.zeros((1, 5, 64)))

    short_out = pos_enc.forward(short_emb)
    long_out = pos_enc.forward(long_emb)

    # First 3 positions should match
    assert np.allclose(short_out.data, long_out.data[:, :3, :]), "Position encoding prefix should match"

    # Test 4: Parameters
    params = pos_enc.parameters()
    assert len(params) == 1, "Should have 1 parameter (position embeddings)"
    assert params[0].shape == (512, 64), "Position embedding matrix has wrong shape"

    print("✅ Positional encoding works correctly!")

if __name__ == "__main__":
    test_unit_positional_encoding()

# %% [markdown]
r"""
### Sinusoidal Positional Encoding: Geometric Harmonic Signatures

Rather than allocating millions of trainable parameters, Vaswani et al. (2017) introduced deterministic sinusoidal positional encodings using geometric progressions of trigonometric frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{embed}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{embed}}}}\right)$$

where $pos \in [0, T-1]$ denotes sequence index and $i \in [0, d_{\text{embed}}/2 - 1]$ indexes the 2D orthogonal frequency subchannel.

<div align="center">
  <img src="rope_phase_clock.svg" alt="2D Subspace Harmonic Frequency Signatures" width="560px">
</div>

### Numerical Fingerprint Across Channels ($d_{\text{embed}} = 8$)

The wavelengths form a geometric progression from $2\pi$ to $10{,}000 \cdot 2\pi$. High-frequency channels alternate rapidly, while low-frequency channels change slowly across sequence steps:

| Position | Dim 0 ($\sin, \omega_0$) | Dim 1 ($\cos, \omega_0$) | Dim 2 ($\sin, \omega_1$) | Dim 3 ($\cos, \omega_1$) | Dim 4 ($\sin, \omega_2$) | Dim 5 ($\cos, \omega_2$) | Dim 6 ($\sin, \omega_3$) | Dim 7 ($\cos, \omega_3$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$pos = 0$** | $+0.000$ | $+1.000$ | $+0.000$ | $+1.000$ | $+0.000$ | $+1.000$ | $+0.000$ | $+1.000$ |
| **$pos = 1$** | $+0.841$ | $+0.540$ | $+0.099$ | $+0.995$ | $+0.010$ | $+1.000$ | $+0.001$ | $+1.000$ |
| **$pos = 2$** | $+0.909$ | $-0.416$ | $+0.198$ | $+0.980$ | $+0.020$ | $+1.000$ | $+0.002$ | $+1.000$ |
| **$pos = 3$** | $+0.141$ | $-0.990$ | $+0.295$ | $+0.956$ | $+0.030$ | $+1.000$ | $+0.003$ | $+1.000$ |

### Mathematical Elegance: Relative Linear Projections
By the angle addition theorem:
$$\sin((pos + k)\omega) = \sin(pos \cdot \omega)\cos(k\omega) + \cos(pos \cdot \omega)\sin(k\omega)$$
$$\cos((pos + k)\omega) = \cos(pos \cdot \omega)\cos(k\omega) - \sin(pos \cdot \omega)\sin(k\omega)$$

For any fixed offset $k$, $PE_{pos + k}$ is a **linear transformation** of $PE_{pos}$. This algebraic property allows linear projection matrices in Multi-Head Attention ($W_Q, W_K$) to attend across relative distances $(pos_j - pos_i)$ without learning separate weights for every absolute index.
"""

# %% [markdown]
r"""
### Vectorized Sinusoidal Table Construction

To avoid numerical underflow when computing $10000^{2i/d}$, we compute the frequency divisor in log-space:

$$\omega_i = \exp\left(-\frac{2i}{d_{\text{embed}}} \cdot \ln(10000)\right)$$

The table is constructed via a 4-step vectorized tensor broadcast:

| Step | Operation | Formula | Shape Contract |
| :--- | :--- | :--- | :--- |
| **1. Temporal Coordinates** | Linear position column | $\mathbf{pos} = [0, 1, \dots, \text{max\_len}-1]^\top$ | $(\text{max\_len}, 1)$ |
| **2. Log-Space Frequencies** | Exponential frequency decay | $\boldsymbol{\omega}_i = \exp\left(-\frac{2i}{d} \ln 10000\right)$ | $(1, d_{\text{embed}}/2)$ |
| **3. Outer Product Phase** | Broadcasting phase angles | $\mathbf{\Theta} = \mathbf{pos} \odot \boldsymbol{\omega}$ | $(\text{max\_len}, d_{\text{embed}}/2)$ |
| **4. Column Interleaving** | Even/Odd trigonometric assignment | $PE_{:, 2i} = \sin(\mathbf{\Theta}), \, PE_{:, 2i+1} = \cos(\mathbf{\Theta})$ | $(\text{max\_len}, d_{\text{embed}})$ |
"""

# %% nbgrader={"grade": false, "grade_id": "posenc-sinusoidal-table", "solution": true}
#| export
def _compute_sinusoidal_table(max_len: int, embed_dim: int) -> np.ndarray:
    """
    Compute the raw sinusoidal positional encoding table as a numpy array.

    This helper builds the (max_len, embed_dim) table of sin/cos values
    using the formula from "Attention Is All You Need":
      PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))

    TODO: Compute the sinusoidal table with alternating sin/cos columns

    APPROACH:
    1. Create position indices as column vector: (max_len, 1)
    2. Compute frequency scaling (div_term) using exponential decay
    3. Initialize zeros matrix of shape (max_len, embed_dim)
    4. Fill even columns with sin(position * div_term)
    5. Fill odd columns with cos(position * div_term)
    6. Handle odd embed_dim gracefully

    EXAMPLE:
    >>> table = _compute_sinusoidal_table(4, 8)
    >>> table.shape
    (4, 8)
    >>> table[0, 0]  # sin(0) = 0.0
    0.0
    >>> table[0, 1]  # cos(0) = 1.0
    1.0

    HINT: The div_term creates geometrically decreasing frequencies across
    dimensions. Use np.exp with negative log(10000) scaling.
    """
    ### BEGIN SOLUTION
    # Create position indices [0, 1, 2, ..., max_len-1]
    position = np.arange(max_len, dtype=np.float32)[:, np.newaxis]  # (max_len, 1)

    # Create dimension indices for calculating frequencies
    div_term = np.exp(
        np.arange(0, embed_dim, 2, dtype=np.float32) *
        -(math.log(10000.0) / embed_dim)
    )  # (embed_dim//2,)

    # Initialize the positional encoding matrix
    pe = np.zeros((max_len, embed_dim), dtype=np.float32)

    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...)
    if embed_dim % 2 == 1:
        # Handle odd embed_dim by only filling available positions
        pe[:, 1::2] = np.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(position * div_term)

    return pe
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Sinusoidal Table Computation

This test validates the helper that builds the raw sin/cos table before it gets
wrapped in a Tensor.

**What we're testing**: Correct sin/cos alternation and frequency decay across dimensions
**Why it matters**: The table is the mathematical core of sinusoidal positional encoding
**Expected**: sin(0)=0 at even dims, cos(0)=1 at odd dims, higher dims change slower
"""

# %% nbgrader={"grade": true, "grade_id": "test-sinusoidal-table", "locked": true, "points": 5}
def test_unit_sinusoidal_table():
    """🧪 Test _compute_sinusoidal_table helper."""
    print("🧪 Unit Test: Sinusoidal Table Computation...")

    # Test 1: Shape and dtype
    table = _compute_sinusoidal_table(10, 8)
    assert table.shape == (10, 8), f"Expected (10, 8), got {table.shape}"
    assert table.dtype == np.float32, f"Expected float32, got {table.dtype}"

    # Test 2: Position 0 pattern (sin(0)=0 at even, cos(0)=1 at odd)
    assert np.allclose(table[0, 0::2], 0, atol=1e-6), "Even dims at pos 0 should be sin(0)=0"
    assert np.allclose(table[0, 1::2], 1, atol=1e-6), "Odd dims at pos 0 should be cos(0)=1"

    # Test 3: Frequency decay (higher dims change slower)
    table_100 = _compute_sinusoidal_table(100, 16)
    fast_changes = np.sum(np.abs(np.diff(table_100[:10, 0])))
    slow_changes = np.sum(np.abs(np.diff(table_100[:10, -1])))
    assert fast_changes > slow_changes, "Lower dims should oscillate faster"

    # Test 4: Odd embed_dim
    table_odd = _compute_sinusoidal_table(5, 7)
    assert table_odd.shape == (5, 7), "Should handle odd embed_dim"

    # Test 5: Returns numpy array (not Tensor)
    assert isinstance(table, np.ndarray), "Helper should return raw numpy array"

    print("✅ Sinusoidal table computation works correctly!")

if __name__ == "__main__":
    test_unit_sinusoidal_table()

# %% [markdown]
"""
### Implementing Sinusoidal Positional Encodings

Now we compose the table computation into the public API that returns a Tensor
ready for use in embedding pipelines.
"""

# %% nbgrader={"grade": false, "grade_id": "sinusoidal-function", "solution": true}
#| export
def create_sinusoidal_embeddings(max_seq_len: int, embed_dim: int) -> Tensor:
    """
    Create sinusoidal positional encodings as used in "Attention Is All You Need".

    These fixed encodings use sine and cosine functions to create unique
    positional patterns that don't require training and can extrapolate
    to longer sequences than seen during training.

    TODO: Use _compute_sinusoidal_table to build the encoding and wrap in Tensor

    APPROACH:
    1. Call _compute_sinusoidal_table(max_seq_len, embed_dim) for the raw table
    2. Wrap the result in a Tensor and return

    EXAMPLE:
    >>> pe = create_sinusoidal_embeddings(512, 64)
    >>> print(pe.shape)
    (512, 64)
    >>> # Position 0: [0, 1, 0, 1, 0, 1, ...] (sin(0)=0, cos(0)=1)
    >>> # Each position gets unique trigonometric signature

    HINT: The heavy lifting is done by _compute_sinusoidal_table. This function
    just wraps the result as a Tensor for use in the embedding pipeline.
    """
    ### BEGIN SOLUTION role="scaffold"
    pe = _compute_sinusoidal_table(max_seq_len, embed_dim)
    return Tensor(pe)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Sinusoidal Embeddings

This test validates our sinusoidal positional encoding function creates correct mathematical patterns.

**What we're testing**: Sinusoidal pattern generation and frequency properties
**Why it matters**: Enables position awareness without trainable parameters
**Expected**: Correct sin/cos patterns, unique positions, frequency decay
"""

# %% nbgrader={"grade": true, "grade_id": "test-sinusoidal", "locked": true, "points": 10}
def test_unit_sinusoidal_embeddings():
    """🧪 Test sinusoidal positional embeddings."""
    print("🧪 Unit Test: Sinusoidal Embeddings...")

    # Test 1: Basic shape and properties
    pe = create_sinusoidal_embeddings(512, 64)

    assert pe.shape == (512, 64), f"Expected shape (512, 64), got {pe.shape}"

    # Test 2: Position 0 should be mostly zeros and ones
    pos_0 = pe.data[0]

    # Even indices should be sin(0) = 0
    assert np.allclose(pos_0[0::2], 0, atol=1e-6), "Even indices at position 0 should be ~0"

    # Odd indices should be cos(0) = 1
    assert np.allclose(pos_0[1::2], 1, atol=1e-6), "Odd indices at position 0 should be ~1"

    # Test 3: Different positions should have different encodings
    pe_small = create_sinusoidal_embeddings(10, 8)

    # Check that consecutive positions are different
    for i in range(9):
        assert not np.allclose(pe_small.data[i], pe_small.data[i+1]), f"Positions {i} and {i+1} are too similar"

    # Test 4: Frequency properties
    # Higher dimensions should have lower frequencies (change more slowly)
    pe_test = create_sinusoidal_embeddings(100, 16)

    # First dimension should change faster than last dimension
    first_dim_changes = np.sum(np.abs(np.diff(pe_test.data[:10, 0])))
    last_dim_changes = np.sum(np.abs(np.diff(pe_test.data[:10, -1])))

    assert first_dim_changes > last_dim_changes, "Lower dimensions should change faster than higher dimensions"

    # Test 5: Odd embed_dim handling
    pe_odd = create_sinusoidal_embeddings(10, 7)
    assert pe_odd.shape == (10, 7), "Should handle odd embedding dimensions"

    # Test 6: Returns Tensor (not numpy array)
    assert isinstance(pe, Tensor), "Should return a Tensor wrapping the sinusoidal table"

    print("✅ Sinusoidal embeddings work correctly!")

if __name__ == "__main__":
    test_unit_sinusoidal_embeddings()

# %% [markdown]
r"""
## 🔧 Integration: Bringing It Together

Now let's assemble the complete production embedding pipeline that combines discrete token table lookups, variance magnitude scaling, and positional injection into a unified, reusable `EmbeddingLayer`.

<div align="center">
  <img src="embeddings_pipeline.svg" alt="Complete Transformer Embedding Datapath" width="700px">
</div>

### End-to-End Embedding Datapath

| Phase | Operation | Mathematical Formula | Tensor Shape Contract |
| :--- | :--- | :--- | :--- |
| **1. Input Ingestion** | Discrete Token IDs | $X \in \mathbb{Z}^{B \times T}, \, X_{b, t} \in [0, V-1]$ | $(B, T)$ |
| **2. Table Gather** | Memory Pointer Lookup | $E = W_{\text{token}}[X]$ | $(B, T, d_{\text{embed}})$ |
| **3. Variance Scaling** | Token Signal Stabilization | $E_{\text{scaled}} = E \cdot \sqrt{d_{\text{embed}}}$ (if enabled) | $(B, T, d_{\text{embed}})$ |
| **4. Positional Slicing** | Coordinate Extraction | $P_{\text{active}} = P[0:T, :]$ | $(T, d_{\text{embed}})$ |
| **5. Broadcast Addition** | Composite Representation | $H = E_{\text{scaled}} + P_{\text{active}}$ | $(B, T, d_{\text{embed}})$ |

### Why Scale by $\sqrt{d_{\text{embed}}}$?
In standard Transformer architectures (Vaswani et al., 2017), weights in $W_{\text{token}}$ are initialized with variance $\sigma^2 \approx \frac{1}{d_{\text{embed}}}$. Consequently, initial token embeddings have vector norms of order $\mathcal{O}(1)$ across all coordinates, with individual elements having typical magnitude $\mathcal{O}(1/\sqrt{d_{\text{embed}}})$.
- Fixed sinusoidal encodings $P$ have amplitude on $[-1, 1]$ with variance $\approx 0.5$.
- Adding $P$ directly to unscaled $E$ would cause position signals to drown out lexical word identity by a factor of $\sqrt{d_{\text{embed}}}$ (e.g. $\sqrt{512} \approx 22.6\times$ stronger!).
- Multiplying $E$ by $\sqrt{d_{\text{embed}}}$ balances the variance of lexical and positional features before they enter the first self-attention block.

---

### Interface Contract Specification: `EmbeddingLayer`

| Configuration Argument | Type | Default | Description & Invariant |
| :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | *required* | Total unique tokens in vocabulary $V$; upper bound for input IDs |
| `embed_dim` | `int` | *required* | Hidden dimension width $d_{\text{embed}}$; must be positive |
| `max_seq_len` | `int` | `512` | Maximum supported sequence length $T_{\text{max}}$ |
| `pos_encoding` | `str` or `None` | `'learned'` | Strategy: `'learned'` (trainable), `'sinusoidal'` (fixed math), or `None` |
| `scale_embeddings` | `bool` | `False` | When `True`, scales token embeddings by $\sqrt{d_{\text{embed}}}$ prior to addition |
"""

# %% nbgrader={"grade": false, "grade_id": "emblayer-init", "solution": true}
#| export
class EmbeddingLayer:
    """
    Complete embedding system combining token and positional embeddings.

    This is the production-ready component that handles the full embedding
    pipeline used in transformers and other sequence models.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 512,
        pos_encoding: str = 'learned',
        scale_embeddings: bool = False
    ):
        """
        Initialize complete embedding system.

        TODO: Create sub-components for token embedding and positional encoding

        APPROACH:
        1. Store configuration (vocab_size, embed_dim, max_seq_len, etc.)
        2. Create token Embedding(vocab_size, embed_dim)
        3. Based on pos_encoding argument, create the appropriate positional encoder:
           - 'learned' -> PositionalEncoding(max_seq_len, embed_dim)
           - 'sinusoidal' -> create_sinusoidal_embeddings(max_seq_len, embed_dim)
           - None -> no positional encoding
        4. Raise ValueError for unknown pos_encoding types

        EXAMPLE:
        >>> layer = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding='learned')
        >>> layer.token_embedding  # Embedding(vocab_size=100, embed_dim=64)
        >>> layer.pos_encoding     # PositionalEncoding(max_seq_len=512, embed_dim=64)

        HINT: The pos_encoding parameter selects the strategy; each strategy
        produces a different type of object stored in self.pos_encoding.
        """
        ### BEGIN SOLUTION role="scaffold"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pos_encoding_type = pos_encoding
        self.scale_embeddings = scale_embeddings

        # Token embedding layer
        self.token_embedding = Embedding(vocab_size, embed_dim)

        # Positional encoding
        if pos_encoding == 'learned':
            self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        elif pos_encoding == 'sinusoidal':
            # Create fixed sinusoidal encodings (no parameters)
            self.pos_encoding = create_sinusoidal_embeddings(max_seq_len, embed_dim)
        elif pos_encoding is None:
            self.pos_encoding = None
        else:
            raise ValueError(
                f"Unknown positional encoding type: '{pos_encoding}'\n"
                f"  ❌ pos_encoding must be 'learned', 'sinusoidal', or None\n"
                f"  💡 'learned' = trainable position embeddings (task-specific but fixed max length)\n"
                f"     'sinusoidal' = mathematical sin/cos patterns (no parameters, can extrapolate)\n"
                f"     None = no positional encoding (order-agnostic model)\n"
                f"  🔧 Use: EmbeddingLayer(..., pos_encoding='learned') or pos_encoding='sinusoidal'"
            )
        ### END SOLUTION

    def __call__(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
        """Allows the embedding layer to be called like a function."""
        return self.forward(tokens, start_pos)

    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        params = self.token_embedding.parameters()
        if self.pos_encoding_type == 'learned':
            params.extend(self.pos_encoding.parameters())
        return params

    def __repr__(self):
        return (f"EmbeddingLayer(vocab_size={self.vocab_size}, "
                f"embed_dim={self.embed_dim}, "
                f"pos_encoding='{self.pos_encoding_type}')")

# %% [markdown]
"""
### 🧪 Unit Test: EmbeddingLayer Initialization

This test validates that `__init__` correctly assembles sub-components for each
positional encoding strategy.

**What we're testing**: Sub-component creation and configuration storage
**Why it matters**: Incorrect initialization cascades into broken forward passes
**Expected**: Correct component types, parameter counts, and error on invalid strategy
"""

# %% nbgrader={"grade": true, "grade_id": "test-emblayer-init", "locked": true, "points": 5}
def test_unit_emblayer_init():
    """🧪 Test EmbeddingLayer.__init__ component assembly."""
    print("🧪 Unit Test: EmbeddingLayer Initialization...")

    # Test 1: Learned PE creates PositionalEncoding
    layer_learned = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding='learned')
    assert isinstance(layer_learned.token_embedding, Embedding), "Should create Embedding"
    assert isinstance(layer_learned.pos_encoding, PositionalEncoding), "Should create PositionalEncoding"
    assert len(layer_learned.parameters()) == 2, "Learned PE: 2 param tensors (token + position)"

    # Test 2: Sinusoidal PE creates fixed Tensor
    layer_sin = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding='sinusoidal')
    assert isinstance(layer_sin.pos_encoding, Tensor), "Sinusoidal PE should be a Tensor"
    assert len(layer_sin.parameters()) == 1, "Sinusoidal PE: 1 param tensor (token only)"

    # Test 3: None PE stores None
    layer_none = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding=None)
    assert layer_none.pos_encoding is None, "No PE should store None"
    assert len(layer_none.parameters()) == 1, "No PE: 1 param tensor (token only)"

    # Test 4: Invalid PE raises ValueError
    try:
        EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding='invalid')
        assert False, "Should raise ValueError for invalid pos_encoding"
    except ValueError:
        pass  # Expected

    # Test 5: Configuration stored correctly
    assert layer_learned.vocab_size == 100
    assert layer_learned.embed_dim == 64
    assert layer_learned.scale_embeddings == False

    print("✅ EmbeddingLayer initialization works correctly!")

if __name__ == "__main__":
    test_unit_emblayer_init()

# %% [markdown]
r"""
### EmbeddingLayer Forward Pass

The `forward` method composes the full embedding pipeline: token lookup,
optional scaling, positional encoding addition, and batch dimension handling.

### `EmbeddingLayer.forward` Execution Pipeline

| Execution Step | Operation | Condition / Scope | Shape Transformation |
| :--- | :--- | :--- | :--- |
| **1. Rank Alignment** | Add batch dimension | If input is 1D vector `(seq,)` | `(seq,)` $\rightarrow$ `(1, seq)` |
| **2. Token Lookup** | Direct index gather | Gather embedding weights | `(batch, seq)` $\rightarrow$ `(batch, seq, embed_dim)` |
| **3. Embedding Scaling**| Multiply by $\sqrt{d_{\text{embed}}}$ | Optional (`scale_embeddings=True`) | Normalized variance preservation |
| **4. Positional Encoding**| Add coordinate vectors | Learned (`pos_encoding`) or sinusoidal table | $\mathbf{E}_{\text{pos}} = \mathbf{E}_{\text{tok}} + \mathbf{P}$ |
| **5. Output Squeeze** | Squeeze batch dimension | If input was originally 1D | `(1, seq, embed_dim)` $\rightarrow$ `(seq, embed_dim)` |
"""

# %% nbgrader={"grade": false, "grade_id": "emblayer-forward", "solution": true}
#| exporti

# Continue the EmbeddingLayer class with forward and utility methods
def emblayer_forward(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
    """
    Forward pass through complete embedding system.

    start_pos is the position of the first token in `tokens`. It is 0 for a whole
    sequence; Module 18's KV cache will feed one token at a time and pass how many
    tokens are already cached, so each new token gets its true position.

    TODO: Compose token embed + optional scaling + positional encoding

    APPROACH:
    1. Handle 1D input by reshaping to (1, seq_len)
    2. Look up token embeddings via self.token_embedding.forward(tokens)
    3. If scale_embeddings, multiply by sqrt(embed_dim)
    4. Add positional encoding based on self.pos_encoding_type
    5. Squeeze batch dim if it was added in step 1

    EXAMPLE:
    >>> layer = EmbeddingLayer(vocab_size=100, embed_dim=64)
    >>> tokens = Tensor([[1, 2, 3], [4, 5, 6]])
    >>> output = layer.forward(tokens)
    >>> output.shape
    (2, 3, 64)

    HINTS:
    - For sinusoidal PE, slice the table from start_pos to start_pos + seq_len and add a batch dim with np.newaxis
    - For learned PE, just call self.pos_encoding.forward(token_embeds, start_pos)
    - Remember to squeeze the batch dim for 1D inputs at the end
    """
    ### BEGIN SOLUTION role="scaffold"
    # Handle 1D input by adding batch dimension
    if len(tokens.shape) == 1:
        # NOTE: Tensor reshape preserves gradients
        tokens = tokens.reshape(1, -1)
        squeeze_batch = True
    else:
        squeeze_batch = False

    # Get token embeddings
    token_embeds = self.token_embedding.forward(tokens)  # (batch, seq, embed)

    # Scale embeddings if requested (transformer convention)
    if self.scale_embeddings:
        scale_factor = math.sqrt(self.embed_dim)
        token_embeds = token_embeds * scale_factor  # Use Tensor multiplication to preserve gradients

    # Add positional encoding
    if self.pos_encoding_type == 'learned':
        # Use learnable positional encoding
        output = self.pos_encoding.forward(token_embeds, start_pos)
    elif self.pos_encoding_type == 'sinusoidal':
        # Use fixed sinusoidal encoding (not learnable)
        batch_size, seq_len, embed_dim = token_embeds.shape
        if not isinstance(start_pos, (int, np.integer)) or start_pos < 0:
            raise ValueError("start_pos must be a nonnegative integer")
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError("Sequence runs past the sinusoidal position table")
        pos_embeddings = self.pos_encoding[start_pos:start_pos + seq_len]  # Slice using Tensor slicing

        # Reshape to add batch dimension
        # Deliberately a fresh constant tensor: sinusoidal encodings are fixed,
        # so they take no gradient. The addition below still carries gradients
        # back through token_embeds, which is the part that learns.
        pos_data = pos_embeddings.data[np.newaxis, :, :]
        pos_embeddings_batched = Tensor(pos_data)

        output = token_embeds + pos_embeddings_batched
    else:
        # No positional encoding
        output = token_embeds

    # Remove batch dimension if it was added
    if squeeze_batch:
        # Tensor slicing from Module 01
        output = output[0]

    return output
    ### END SOLUTION

# Attach forward to EmbeddingLayer class (other methods defined in class body above)
EmbeddingLayer.forward = emblayer_forward

# %% [markdown]
"""
### 🧪 Unit Test: Complete Embedding System

This test validates our EmbeddingLayer combines all components correctly for production use.

**What we're testing**: Token + positional embedding integration, scaling, and batch processing
**Why it matters**: Production transformers use this exact pattern
**Expected**: Correct shapes, proper scaling, flexible position encoding support
"""

# %% nbgrader={"grade": true, "grade_id": "test-complete-system", "locked": true, "points": 15}
def test_unit_complete_embedding_system():
    """🧪 Test complete embedding system."""
    print("🧪 Unit Test: Complete Embedding System...")

    # Test 1: Learned positional encoding
    embed_learned = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        max_seq_len=128,
        pos_encoding='learned'
    )

    tokens = Tensor([[1, 2, 3], [4, 5, 6]])
    output_learned = embed_learned.forward(tokens)

    assert output_learned.shape == (2, 3, 64), f"Expected shape (2, 3, 64), got {output_learned.shape}"

    # Test 2: Sinusoidal positional encoding
    embed_sin = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        pos_encoding='sinusoidal'
    )

    output_sin = embed_sin.forward(tokens)
    assert output_sin.shape == (2, 3, 64), "Sinusoidal embedding should have same shape"

    # Test 3: No positional encoding
    embed_none = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        pos_encoding=None
    )

    output_none = embed_none.forward(tokens)
    assert output_none.shape == (2, 3, 64), "No pos encoding should have same shape"

    # Test 4: 1D input handling
    tokens_1d = Tensor([1, 2, 3])
    output_1d = embed_learned.forward(tokens_1d)

    assert output_1d.shape == (3, 64), f"Expected shape (3, 64) for 1D input, got {output_1d.shape}"

    # Test 5: Embedding scaling
    embed_scaled = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        pos_encoding=None,
        scale_embeddings=True
    )

    # Use same weights to ensure fair comparison
    embed_scaled.token_embedding.weight = embed_none.token_embedding.weight

    output_scaled = embed_scaled.forward(tokens)
    output_unscaled = embed_none.forward(tokens)

    # Scaled version should be sqrt(64) times larger
    scale_factor = math.sqrt(64)
    expected_scaled = output_unscaled.data * scale_factor
    assert np.allclose(output_scaled.data, expected_scaled, rtol=1e-5), "Embedding scaling not working correctly"

    # Test 6: Parameter counting
    params_learned = embed_learned.parameters()
    params_sin = embed_sin.parameters()
    params_none = embed_none.parameters()

    assert len(params_learned) == 2, "Learned encoding should have 2 parameter tensors"
    assert len(params_sin) == 1, "Sinusoidal encoding should have 1 parameter tensor"
    assert len(params_none) == 1, "No pos encoding should have 1 parameter tensor"

    print("✅ Complete embedding system works correctly!")

if __name__ == "__main__":
    test_unit_complete_embedding_system()

# %% [markdown]
r"""
## 📊 Systems Analysis: Embedding Trade-offs & Hardware Scaling

In large-scale language models, the embedding layer sits at the interface between host CPU tokenizers and GPU tensor accelerators. Understanding its memory footprint, memory bandwidth limitations, and parameter distribution is critical for production deployment.

### Industry Model Embedding Architectures

| Architecture | Vocab Size $V$ | Hidden Dim $d_{\text{embed}}$ | Context $T_{\text{max}}$ | Embedding Parameters | VRAM (FP16) | Positional Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT-Base** | $30{,}522$ | $768$ | $512$ | $23.44\text{M}$ | $46.88\text{ MB}$ | Learned Absolute |
| **GPT-2 Small** | $50{,}257$ | $768$ | $1{,}024$ | $38.60\text{M}$ | $77.19\text{ MB}$ | Learned Absolute |
| **GPT-3 (175B)** | $50{,}257$ | $12{,}288$ | $2{,}048$ | $617.56\text{M}$ | $1{,}235.1\text{ MB}$ ($1.2\text{ GB}$) | Learned Absolute |
| **Llama 2 (7B)** | $32{,}000$ | $4{,}096$ | $4{,}096$ | $131.07\text{M}$ | $262.14\text{ MB}$ | Rotary (RoPE) |
| **Llama 3 (8B)** | $128{,}256$ | $4{,}096$ | $8{,}192$ | $525.34\text{M}$ | $1{,}050.7\text{ MB}$ ($1.05\text{ GB}$) | Rotary (RoPE) |
| **Gemma 2 (2B)** | $256{,}000$ | $2{,}304$ | $8{,}192$ | $589.82\text{M}$ | $1{,}179.6\text{ MB}$ ($1.18\text{ GB}$) | Rotary (RoPE) |

### Memory Scaling Invariants
$$\text{Memory}_{\text{token}} = V \cdot d_{\text{embed}} \cdot (\text{bytes per parameter})$$
$$\text{Memory}_{\text{pos}} = T_{\text{max}} \cdot d_{\text{embed}} \cdot (\text{bytes per parameter})$$

Notice that in modern multilingual models like Llama 3 and Gemma 2, expanding vocabulary from $32\text{K} \to 128\text{K}$ or $256\text{K}$ causes the embedding table alone to exceed **$1\text{ GB}$ of VRAM** in FP16 precision.
"""

# %%
def analyze_embedding_memory_scaling():
    """📊 Compare embedding memory requirements across different model scales."""
    print("📊 Analyzing Embedding Memory Requirements...")
    print("=" * 60)

    # Vocabulary and embedding dimension scenarios
    scenarios = [
        ("Small Model", 10_000, 256),
        ("Medium Model", 50_000, 512),
        ("Large Model", 100_000, 1024),
        ("GPT-3 Scale", 50_257, 12_288),
    ]

    print(f"{'Model':<15} {'Vocab Size':<12} {'Embed Dim':<12} {'Memory (MB)':<15} {'Parameters (M)':<15}")
    print("-" * 80)

    for name, vocab_size, embed_dim in scenarios:
        # Calculate memory for FP32 (4 bytes per parameter)
        params = vocab_size * embed_dim
        memory_mb = params * BYTES_PER_FLOAT32 / MB_TO_BYTES
        params_m = params / 1_000_000

        print(f"{name:<15} {vocab_size:<12,} {embed_dim:<12} {memory_mb:<15.1f} {params_m:<15.2f}")

    print("\n💡 Key Insights:")
    print("• Embedding tables often dominate model memory (especially for large vocabularies)")
    print("• Memory scales linearly with vocab_size × embed_dim")
    print("• Consider vocabulary pruning for memory-constrained environments")

    # Positional encoding memory comparison
    print(f"\n📊 Positional Encoding Memory Comparison (embed_dim=512, max_seq_len=2048):")

    learned_params = 2048 * 512
    learned_memory = learned_params * BYTES_PER_FLOAT32 / MB_TO_BYTES

    print(f"Learned PE:     {learned_memory:.1f} MB ({learned_params:,} trainable parameters)")
    print(f"Sinusoidal PE:  {learned_memory:.1f} MB stored table, 0 trainable parameters")
    print(f"No PE:          0.0 MB (0 parameters)")

    print("\n🚀 Production Implications:")
    print("• GPT-3's embedding table: ~2.4GB (50K vocab × 12K dims)")
    print("• Learned PE adds memory but may improve task-specific performance")
    print("• Sinusoidal PE avoids gradient/optimizer storage; extend its table for longer sequences")

if __name__ == "__main__":
    analyze_embedding_memory_scaling()

# %%
def analyze_embedding_performance():
    """📊 Compare embedding lookup performance across different configurations."""
    print("\n📊 Analyzing Embedding Lookup Performance...")
    print("=" * 60)

    import time

    # Test different vocabulary sizes and batch configurations
    vocab_sizes = [1_000, 10_000, 100_000]
    embed_dim = 512
    seq_len = 128
    batch_sizes = [1, 16, 64, 256]

    print(f"{'Vocab Size':<12} {'Batch Size':<12} {'Lookup Time (ms)':<18} {'Throughput (tokens/s)':<20}")
    print("-" * 70)

    for vocab_size in vocab_sizes:
        # Create embedding layer
        embed = Embedding(vocab_size, embed_dim)

        for batch_size in batch_sizes:
            # Create random token batch
            tokens = Tensor(rng.integers(0, vocab_size, (batch_size, seq_len)))

            # Warmup
            for _ in range(5):
                _ = embed.forward(tokens)

            # Time the lookup
            start_time = time.perf_counter()
            iterations = 100

            for _ in range(iterations):
                output = embed.forward(tokens)

            end_time = time.perf_counter()

            # Calculate metrics
            total_time = end_time - start_time
            avg_time_ms = (total_time / iterations) * 1000
            total_tokens = batch_size * seq_len * iterations
            throughput = total_tokens / total_time

            print(f"{vocab_size:<12,} {batch_size:<12} {avg_time_ms:<18.2f} {throughput:<20,.0f}")

    print("\n💡 Performance Insights:")
    print("• Lookup time is O(1) per token - vocabulary size doesn't affect individual lookups")
    print("• Larger batches improve throughput due to vectorization")
    print("• Memory bandwidth becomes bottleneck for large embedding dimensions")
    print("• Cache locality important for repeated token patterns")

if __name__ == "__main__":
    analyze_embedding_performance()

# %%
def analyze_positional_encoding_strategies():
    """📊 Compare different positional encoding approaches and trade-offs."""
    print("\n📊 Analyzing Positional Encoding Trade-offs...")
    print("=" * 60)

    max_seq_len = 512
    embed_dim = 256

    # Create both types of positional encodings
    learned_pe = PositionalEncoding(max_seq_len, embed_dim)
    sinusoidal_pe = create_sinusoidal_embeddings(max_seq_len, embed_dim)

    # Analyze memory footprint
    learned_params = max_seq_len * embed_dim
    learned_memory = learned_params * BYTES_PER_FLOAT32 / MB_TO_BYTES

    print(f"📈 Memory Comparison:")
    print(f"Learned PE:     {learned_memory:.2f} MB ({learned_params:,} trainable parameters)")
    print(f"Sinusoidal PE:  {learned_memory:.2f} MB stored table, 0 trainable parameters")

    # Analyze encoding patterns
    print(f"\n📈 Encoding Pattern Analysis:")

    # Test sample sequences
    test_input = Tensor(rng.standard_normal((1, 10, embed_dim)))

    learned_output = learned_pe.forward(test_input)

    # For sinusoidal, manually add to match learned interface
    sin_encodings = sinusoidal_pe.data[:10][np.newaxis, :, :]  # (1, 10, embed_dim)
    sinusoidal_output = Tensor(test_input.data + sin_encodings)

    # Analyze variance across positions
    learned_var = np.var(learned_output.data, axis=1).mean()  # Variance across positions
    sin_var = np.var(sinusoidal_output.data, axis=1).mean()

    print(f"Position variance (learned):    {learned_var:.4f}")
    print(f"Position variance (sinusoidal): {sin_var:.4f}")

    # Check extrapolation capability
    print(f"\n📈 Extrapolation Analysis:")
    extended_length = max_seq_len + 100

    # Learned PE has a hard ceiling: its table has no rows past max_seq_len
    too_long = Tensor(rng.standard_normal((1, extended_length, embed_dim)))
    try:
        learned_pe.forward(too_long)
        print(f"Learned PE: unexpectedly accepted {extended_length} positions")
    except ValueError:
        print(f"Learned PE: raises ValueError for sequences > {max_seq_len} (a bigger table needs retraining)")

    # Sinusoidal can extrapolate
    extended_sin = create_sinusoidal_embeddings(extended_length, embed_dim)
    print(f"Sinusoidal PE: Can extrapolate to length {extended_length} (smooth continuation)")

    print(f"\n🚀 Production Trade-offs:")
    print(f"Learned PE:")
    print(f"  + Can learn task-specific positional patterns")
    print(f"  + May perform better for tasks with specific position dependencies")
    print(f"  - Requires additional memory and parameters")
    print(f"  - Fixed maximum sequence length")
    print(f"  - Needs training data for longer sequences")

    print(f"\nSinusoidal PE:")
    print(f"  + Zero additional parameters")
    print(f"  + Can extrapolate to any sequence length")
    print(f"  + Provides rich, mathematically grounded position signals")
    print(f"  - Cannot adapt to task-specific position patterns")
    print(f"  - May be suboptimal for highly position-dependent tasks")

if __name__ == "__main__":
    analyze_positional_encoding_strategies()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 20}
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
    test_unit_embedding_init()
    test_unit_embedding()
    test_unit_embedding_backward()
    test_unit_positional_encoding_init()
    test_unit_positional_encoding()
    test_unit_sinusoidal_table()
    test_unit_sinusoidal_embeddings()
    test_unit_emblayer_init()
    test_unit_complete_embedding_system()

    print("\nRunning integration scenarios...")

    # Integration Test 1: Realistic NLP pipeline
    print("🧪 Integration Test: NLP Pipeline Simulation...")

    # Simulate a small transformer setup
    vocab_size = 1000
    embed_dim = 128
    max_seq_len = 64

    # Create embedding layer
    embed_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        pos_encoding='learned',
        scale_embeddings=True
    )

    # Simulate tokenized sentences
    sentences = [
        [1, 15, 42, 7, 99],        # "the cat sat on mat"
        [23, 7, 15, 88],           # "dog chased the ball"
        [1, 67, 15, 42, 7, 99, 34] # "the big cat sat on mat here"
    ]

    # Process each sentence
    outputs = []
    for sentence in sentences:
        tokens = Tensor(sentence)
        embedded = embed_layer.forward(tokens)
        outputs.append(embedded)

        # Verify output shape
        expected_shape = (len(sentence), embed_dim)
        assert embedded.shape == expected_shape, f"Wrong shape for sentence: {embedded.shape} != {expected_shape}"

    print("✅ Variable length sentence processing works!")

    # Integration Test 2: Batch processing with padding
    print("🧪 Integration Test: Batched Processing...")

    # Create padded batch (real-world scenario)
    max_len = max(len(s) for s in sentences)
    batch_tokens = []

    for sentence in sentences:
        # Pad with zeros (assuming 0 is padding token)
        padded = sentence + [0] * (max_len - len(sentence))
        batch_tokens.append(padded)

    batch_tensor = Tensor(batch_tokens)  # (3, 7)
    batch_output = embed_layer.forward(batch_tensor)

    assert batch_output.shape == (3, max_len, embed_dim), f"Batch output shape incorrect: {batch_output.shape}"

    print("✅ Batch processing with padding works!")

    # Integration Test 3: Different positional encoding types
    print("🧪 Integration Test: Position Encoding Variants...")

    test_tokens = Tensor([[1, 2, 3, 4, 5]])

    # Test all position encoding types
    for pe_type in ['learned', 'sinusoidal', None]:
        embed_test = EmbeddingLayer(
            vocab_size=100,
            embed_dim=64,
            pos_encoding=pe_type
        )

        output = embed_test.forward(test_tokens)
        assert output.shape == (1, 5, 64), f"PE type {pe_type} failed shape test"

        # Check parameter counts
        if pe_type == 'learned':
            assert len(embed_test.parameters()) == 2, f"Learned PE should have 2 param tensors"
        else:
            assert len(embed_test.parameters()) == 1, f"PE type {pe_type} should have 1 param tensor"

    print("✅ All positional encoding variants work!")

    # Integration Test 4: Repeated forward passes
    print("🧪 Integration Test: Repeated Forward Passes...")

    # Check output shapes stay consistent across repeated calls
    large_embed = EmbeddingLayer(vocab_size=10000, embed_dim=512)
    test_batch = Tensor(rng.integers(0, 10000, (32, 128)))

    # This smoke test checks shape consistency, not memory allocation or retention.
    for _ in range(5):
        output = large_embed.forward(test_batch)
        assert output.shape == (32, 128, 512), "Large batch processing failed"

    print("✅ Repeated forward passes preserve output shape!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 11")

# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

Answer these questions to deepen your systems understanding of embedding memory footprints, hardware memory bandwidth bottlenecks, and positional extrapolation mechanics:

### Question 1: Memory Scaling, VRAM Footprint & Untied Weights
You implemented an embedding table with $V = 50{,}000$ and $d_{\text{embed}} = 512$.

**1. Exact Parameter and Memory Footprint**:
- Parameter count:
  $$\text{Parameters} = V \times d_{\text{embed}} = 50{,}000 \times 512 = 25{,}600{,}000 \quad (\mathbf{25.6\text{M parameters}})$$
- Float32 VRAM footprint ($4\text{ bytes/param}$):
  $$\text{Memory}_{\text{FP32}} = 25{,}600{,}000 \times 4\text{ B} = 102{,}400{,}000\text{ B} = \mathbf{102.4\text{ MB}} \quad (97.66\text{ MiB})$$
- Doubling hidden dimension to $d_{\text{embed}} = 1{,}024$:
  $$\text{Memory}_{\text{double}} = 50{,}000 \times 1{,}024 \times 4\text{ B} = \mathbf{204.8\text{ MB}} \quad (2\times\text{ linear scaling})$$

**2. The Untied Weights and Optimizer State Tax**:
- **Untied Weights**: In architectures with separate input embedding and output classification heads (e.g. Llama models), this memory cost is paid twice, consuming $409.6\text{ MB}$ in FP32.
- **Adam Optimizer State**: For each parameter, Adam stores a 32-bit first moment ($m_t$) and 32-bit second moment ($v_t$), plus the gradient ($g_t$) and FP32 master weight ($16\text{ bytes/param}$ total during mixed-precision training):
  $$\text{Optimizer Memory} = 25.6\text{M} \times 16\text{ B} = \mathbf{409.6\text{ MB}}$$
  The optimizer states for the embedding table alone consume $4\times$ the storage of the forward weights!

---

### Question 2: Memory Bandwidth vs Arithmetic Intensity in Gather Operations
Your embedding layer performs table lookups for token indices across batches.

**1. Batch Lookup Mechanics**:
For a batch of $B = 32$ sequences of length $T = 128$:
$$\text{Total Lookups} = B \times T = 32 \times 128 = \mathbf{4{,}096\text{ token vectors}}$$
$$\text{Data Transferred} = 4{,}096 \times 512 \times 4\text{ B} = 8{,}388{,}608\text{ B} = \mathbf{8.39\text{ MB}}$$

**2. Algorithmic Complexity & Pointer Arithmetic**:
- Time complexity of single-token lookup: $\mathcal{O}(1)$ pointer indexing + $\mathcal{O}(d_{\text{embed}})$ contiguous memory copy.
- Why vocabulary size $V$ does not affect individual lookup time: Hardware locates table rows via base-pointer offset calculation:
  $$\text{Address}(t) = \text{base\_pointer} + t \cdot d_{\text{embed}} \cdot \text{sizeof(float)}$$
  Access time is invariant to whether $V$ is $100$ or $100{,}000$.

**3. The Arithmetic Intensity Bottleneck**:
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} = \frac{0\text{ FLOPs}}{8.39\text{ MB}} = \mathbf{0\text{ FLOPs/byte}}$$
Because table gather performs pointer dereferencing with zero multiply-accumulate operations, embedding lookup is **100% memory-bandwidth bound**. Modern GPUs (e.g. NVIDIA H100 with $>3{,}000\text{ TFLOPS}$ compute but $3.35\text{ TB/s}$ HBM bandwidth) severely underutilize their tensor cores during embedding lookups.

---

### Question 3: Positional Encoding Architectural Comparison: Learned vs Sinusoidal vs RoPE
You evaluated both learned and sinusoidal positional encodings.

**1. Parameter Budget**:
- Learned PE ($T_{\text{max}} = 2{,}048$, $d_{\text{embed}} = 512$):
  $$\text{Parameters} = 2{,}048 \times 512 = \mathbf{1{,}048{,}576\text{ parameters}} \quad (4.19\text{ MB in FP32})$$
- Sinusoidal PE: Exactly **$0$ trainable parameters** (generated deterministically on the fly).

**2. Failure Modes & Extrapolation Ceiling**:
- When fed a sequence of length $T = 2{,}049$, learned positional encoding crashes (`ValueError: Sequence length exceeds max_seq_len`). The model has no parameter vector for position index $2{,}048$ and cannot process the token without architectural surgery and retraining.
- Sinusoidal PE smoothly generates position angles for arbitrary $T$. However, empirical attention weights decay unpredictably at untested lengths.

**3. Why Modern LLMs Replaced Both with RoPE**:
Instead of additive vectors ($E + P$), Rotary Position Embedding (RoPE) rotates 2D coordinate pairs of query ($\mathbf{q}_m$) and key ($\mathbf{k}_n$) vectors by angle multiples $m\theta$ and $n\theta$. The attention inner product $(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$ depends strictly on **relative distance** $(n - m)$, allowing techniques like YaRN and position interpolation to extend context windows from $4\text{K} \to 128\text{K}$ tokens without retraining from scratch.

---

### Question 4: Production LLM Serving & Model Parallelism
Embedding tables exhibit unique systems trade-offs across model scales.

**1. Embedding Parameter Distribution Across Scales**:
- In **GPT-3 (175B)** ($V \approx 50\text{K}$, $d \approx 12\text{K}$):
  $$\text{Embedding Parameters} = 50{,}257 \times 12{,}288 \approx 617.56\text{M} \implies \mathbf{0.35\% \text{ of total model parameters}}$$
- In **Gemma 2 (2B)** ($V = 256\text{K}$, $d = 2{,}048$):
  $$\text{Embedding Parameters} = 256{,}000 \times 2{,}048 \approx 524.29\text{M} \implies \mathbf{26.2\% \text{ of total model parameters!}}$$
  In edge and compact models, vocabulary expansion causes the embedding layer to dominate device RAM.

**2. Strategic Optimization: Halving $V$ vs Halving $d_{\text{embed}}$**:
- Halving vocabulary size $V$ cuts embedding memory by $50\%$ while leaving every Transformer self-attention and MLP layer completely intact.
- Halving hidden dimension $d_{\text{embed}}$ reduces parameter count across all self-attention matrices ($W_Q, W_K, W_V, W_O$) and feed-forward layers ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) quadratically ($\mathcal{O}(d^2)$), devastating the model's expressive capacity.

**3. Distributed Tensor Parallelism (TP)**:
In production multi-GPU inference engines (vLLM, TensorRT-LLM), a $256\text{K}$ vocabulary table cannot fit on a single GPU's fast SRAM. Engines apply **Vocabulary Tensor Parallelism**, sharding $W$ along the vocabulary dimension across $N$ GPUs ($V / N$ rows per device). An `All-Reduce` or `All-Gather` operation reconstructs the batch representations before feeding the first attention block.
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Tokens Become Vectors

**What you built:** An embedding layer that converts token IDs to dense vectors.

**Why it matters:** Tokens are just integers (like word IDs), but embeddings give them meaning!
Each token gets a learned vector that captures its semantic properties. Similar words end up
with similar vectors—this is how models understand language.

In the next module, you'll use attention to let these embeddings interact with each other.
"""

# %%
def demo_embeddings():
    """🎯 See tokens become vectors."""
    print("🎯 AHA MOMENT: Tokens Become Vectors")
    print("=" * 45)

    # Create embedding layer: 100 vocab, 32-dimensional embeddings
    embed = Embedding(vocab_size=100, embed_dim=32)

    # Some token IDs
    tokens = Tensor(np.array([5, 10, 15]))

    # Look up embeddings
    vectors = embed(tokens)

    print(f"Token IDs: {tokens.data}")
    print(f"Embedding shape: {vectors.shape}  ← 3 tokens, 32 dims each")
    print(f"\nToken 5 vector (first 5 dims): {vectors.data[0, :5].round(3)}")
    print(f"Token 10 vector (first 5 dims): {vectors.data[1, :5].round(3)}")

    print("\n✨ Each token has its own learned representation!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_embeddings()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Embeddings

Congratulations! You've built a complete embedding system that transforms discrete tokens into learnable representations!

### Key Accomplishments
- **Built Embedding class** with efficient token-to-vector lookup and Xavier initialization
- **Implemented PositionalEncoding** for learnable position-specific patterns
- **Created sinusoidal embeddings** using the Transformer paper formula for extrapolation
- **Developed EmbeddingLayer** combining token and positional embeddings (production-ready)
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory scaling**: Embedding tables grow linearly with vocab_size x embed_dim
- **Lookup efficiency**: O(1) per token regardless of vocabulary size
- **Positional trade-offs**: Learned PE is task-specific; sinusoidal PE extrapolates to longer sequences
- **Production patterns**: GPT-3's embedding table alone uses ~2.4GB of memory

### Ready for Next Steps
Your embeddings implementation enables attention mechanisms and transformer architectures.
Export with: `tito module complete 11`

**Next**: Module 12 will add attention mechanisms for context-aware representations!
"""
