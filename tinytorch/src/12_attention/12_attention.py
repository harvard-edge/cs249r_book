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

#| default_exp core.attention
#| export

# %% [markdown]
"""
# Module 12: Attention - Learning to Focus

Welcome to Module 12! You're about to build the attention mechanism that revolutionized deep learning and powers GPT, BERT, and modern transformers.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor, activations, layers, losses, autograd, optimizers, training, dataloaders, spatial layers, tokenization, and embeddings
**You'll Build**: Scaled dot-product attention and multi-head attention mechanisms
**You'll Enable**: Transformer architectures, GPT-style language models, and sequence-to-sequence processing

**Connection Map**:
```
Embeddings → Attention → Transformers → Language Models
(representations) (focus mechanism) (complete architecture) (text generation)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement scaled dot-product attention with explicit O(n²) complexity
2. Build multi-head attention for parallel processing streams
3. Understand attention weight computation and interpretation
4. Experience attention's quadratic memory scaling firsthand
5. Test attention mechanisms with masking and sequence processing

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/12_attention/attention_dev.py`
**Building Side:** Code exports to `tinytorch.core.attention`

```python
# How to use this module:
from tinytorch.core.attention import scaled_dot_product_attention, MultiHeadAttention
```

**Why this matters:**
- **Learning:** Complete attention system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn.functional and torch.nn with attention operations
- **Consistency:** All attention computations and multi-head mechanics in core.attention
- **Integration:** Works seamlessly with embeddings for complete sequence processing pipelines
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| export

import numpy as np
import math
import time
from typing import Optional, Tuple, List

# Import dependencies from previous modules - following TinyTorch dependency chain
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import Softmax

# Constants for attention computation
MASK_VALUE = -1e9  # Large negative value used for attention masking (becomes ~0 after softmax)

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01-11 must be complete
- Module 01: Tensor (core data structure)
- Module 03: Layers (Linear for projections)
- Module 04: Activations (Softmax for attention weights)

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `math` (for square root in scaling)
- `time` (for performance analysis)
- `typing` (for type hints)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor.Tensor` - Core tensor operations
- `tinytorch.core.layers.Linear` - For Q, K, V projections
- `tinytorch.core.activations.Softmax` - For attention weight normalization

**Dependency Flow**:
```
Tensor → Layers → Activations → Attention
   ↓                                  ↓
Foundation for              Core mechanism for
all operations             transformer models
```

Students completing this module will have built the attention mechanism
that powers GPT, BERT, and all modern transformer architectures.
"""

# %% [markdown]
"""
## 💡 Introduction - What is Attention?

Attention is the mechanism that allows models to focus on relevant parts of the input when processing sequences. Think of it as a search engine inside your neural network - given a query, attention finds the most relevant keys and retrieves their associated values.

### The Attention Intuition

When you read "The cat sat on the ___", your brain automatically focuses on "cat" and "sat" to predict "mat". This selective focus is exactly what attention mechanisms provide to neural networks.

Imagine attention as a library research system:
- **Query (Q)**: "I need information about machine learning"
- **Keys (K)**: Index cards describing each book's content
- **Values (V)**: The actual books on the shelves
- **Attention Process**: Find books whose descriptions match your query, then retrieve those books

### Why Attention Changed Everything

Before attention, RNNs processed sequences step-by-step, creating an information bottleneck:

```
RNN Processing (Sequential):
Token 1 → Hidden → Token 2 → Hidden → ... → Final Hidden
         ↓              ↓                      ↓
    Limited Info   Compressed State    All Information Lost
```

Attention allows direct connections between any two positions:

```
Attention Processing (Parallel):
Token 1 ←─────────→ Token 2 ←─────────→ Token 3 ←─────────→ Token 4
   ↑                   ↑                   ↑                   ↑
   └─────────────── Direct Connections ──────────────────────┘
```

This enables:
- **Long-range dependencies**: Connecting words far apart
- **Parallel computation**: No sequential dependencies
- **Interpretable focus patterns**: We can see what the model attends to

### The Mathematical Foundation

Attention computes a weighted sum of values, where weights are determined by the similarity between queries and keys:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

This simple formula powers GPT, BERT, and virtually every modern language model.
"""

# %% [markdown]
"""
## 📐 Foundations - Attention Mathematics

### The Three Components Visualized

Think of attention like a sophisticated address book lookup:

```
Query: "What information do I need?"
┌─────────────────────────────────────┐
│ Q: [0.1, 0.8, 0.3, 0.2]             │ ← Query vector (what we're looking for)
└─────────────────────────────────────┘

Keys: "What information is available at each position?"
┌─────────────────────────────────────┐
│ K₁: [0.2, 0.7, 0.1, 0.4]            │ ← Key 1 (description of position 1)
│ K₂: [0.1, 0.9, 0.2, 0.1]            │ ← Key 2 (description of position 2)
│ K₃: [0.3, 0.1, 0.8, 0.3]            │ ← Key 3 (description of position 3)
│ K₄: [0.4, 0.2, 0.1, 0.9]            │ ← Key 4 (description of position 4)
└─────────────────────────────────────┘

Values: "What actual content can I retrieve?"
┌─────────────────────────────────────┐
│ V₁: [content from position 1]       │ ← Value 1 (actual information)
│ V₂: [content from position 2]       │ ← Value 2 (actual information)
│ V₃: [content from position 3]       │ ← Value 3 (actual information)
│ V₄: [content from position 4]       │ ← Value 4 (actual information)
└─────────────────────────────────────┘
```

### The Attention Process Step by Step

```
Step 1: Compute Similarity Scores
Q · K₁ = 0.64    Q · K₂ = 0.81    Q · K₃ = 0.35    Q · K₄ = 0.42
  ↓               ↓               ↓               ↓
Raw similarity scores (higher = more relevant)

Step 2: Scale and Normalize
Scores / √d_k = [0.32, 0.41, 0.18, 0.21]  ← Scale for stability
     ↓
Softmax = [0.20, 0.45, 0.15, 0.20]        ← Convert to probabilities

Step 3: Weighted Combination
Output = 0.20×V₁ + 0.45×V₂ + 0.15×V₃ + 0.20×V₄
```

### Dimensions and Shapes

```
Input Shapes:
Q: (batch_size, seq_len, d_model)  ← Each position has a query
K: (batch_size, seq_len, d_model)  ← Each position has a key
V: (batch_size, seq_len, d_model)  ← Each position has a value

Intermediate Shapes:
QK^T: (batch_size, seq_len, seq_len)  ← Attention matrix (the O(n²) part!)
Weights: (batch_size, seq_len, seq_len)  ← After softmax
Output: (batch_size, seq_len, d_model)  ← Weighted combination of values
```

### Why O(n²) Complexity?

For sequence length n and embedding dimension d, we compute:
1. **QK^T**: n queries × n keys, each a d-dimensional dot product = O(n² × d) operations
2. **Softmax**: n² weights to normalize = O(n²) operations
3. **Weights×V**: n² weights applied to d-dimensional values = O(n² × d) operations

The total **time complexity** is **O(n² × d)** per attention head. The **memory complexity** is **O(n²)** for storing the attention weight matrix. This quadratic scaling in sequence length is attention's blessing (global connectivity) and curse (memory/compute limits).

### The Attention Matrix Visualization

For a 4-token sequence "The cat sat down":

```
Attention Matrix (after softmax):
        The   cat   sat  down
The   [0.30  0.20  0.15  0.35]  ← "The" attends mostly to "down"
cat   [0.10  0.60  0.25  0.05]  ← "cat" focuses on itself and "sat"
sat   [0.05  0.40  0.50  0.05]  ← "sat" attends to "cat" and itself
down  [0.25  0.15  0.10  0.50]  ← "down" focuses on itself and "The"

Each row sums to 1.0 (probability distribution)
```
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Scaled Dot-Product Attention

Now let's implement the core attention mechanism that powers all transformer models. We'll use explicit loops first to make the O(n²) complexity visible and educational.

### Understanding the Algorithm Visually

```
Step-by-Step Attention Computation:

1. Score Computation (Q @ K^T):
   For each query position i and key position j:
   score[i,j] = Σ(Q[i,d] × K[j,d]) for d in embedding_dims

   Query i    Key j      Dot Product
   [0.1,0.8] · [0.2,0.7] = 0.1×0.2 + 0.8×0.7 = 0.58

2. Scaling (÷ √d_k):
   scaled_scores = scores / √embedding_dim
   (Prevents softmax saturation for large dimensions)

3. Masking (optional):
   For causal attention: scores[i,j] = -∞ if j > i

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
### Helper: Computing Attention Scores

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
    >>> Q = Tensor(np.random.randn(1, 3, 4))  # 3 tokens, dim=4
    >>> K = Tensor(np.random.randn(1, 3, 4))
    >>> scores = _compute_attention_scores(Q, K)
    >>> print(scores.shape)  # (1, 3, 3) -- every token scored against every other

    HINT: Use K.transpose(-2, -1) to swap the last two dimensions
    """
    ### BEGIN SOLUTION
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
### Helper: Scaling Scores

Raw dot products grow proportionally with dimension size. For d_model=512,
scores would be ~500x larger than for d_model=1 -- pushing softmax into extreme
values where most weight falls on a single token. Dividing by sqrt(d_model) keeps
scores in a stable range regardless of dimension.
"""

# %% nbgrader={"grade": false, "grade_id": "attn-scale-scores", "solution": true}
#| export
def _scale_scores(scores: Tensor, d_model: int) -> Tensor:
    """Scale attention scores by 1/sqrt(d_model).

    TODO: Divide scores by the square root of the model dimension

    APPROACH:
    1. Compute scale factor: 1.0 / math.sqrt(d_model)
    2. Multiply scores by scale factor

    EXAMPLE:
    >>> scores = Tensor(np.array([[[4.0, 8.0]]]))
    >>> scaled = _scale_scores(scores, d_model=4)
    >>> print(scaled.data)  # [[[ 2.0, 4.0]]] -- divided by sqrt(4)=2

    HINT: Use math.sqrt() for the square root
    """
    ### BEGIN SOLUTION
    scale_factor = 1.0 / math.sqrt(d_model)
    return scores * scale_factor
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Score Scaling

**What we're testing**: Scores are divided by sqrt(d_model) correctly
**Why it matters**: Without scaling, softmax saturates for large dimensions
**Expected**: Scores reduced by factor of sqrt(d_model)
"""

# %% nbgrader={"grade": true, "grade_id": "test-attn-scale", "locked": true, "points": 5}
def test_unit_scale_scores():
    """🧪 Test attention score scaling."""
    print("🧪 Unit Test: Score Scaling...")
    scores = Tensor(np.array([[[4.0, 8.0]]]))
    scaled = _scale_scores(scores, d_model=4)
    assert np.allclose(scaled.data, [[[2.0, 4.0]]]), f"Expected /sqrt(4)=2, got {scaled.data}"
    print("✅ Score scaling works correctly!")

if __name__ == "__main__":
    test_unit_scale_scores()

# %% [markdown]
"""
### Helper: Applying Causal Mask

In autoregressive models (like GPT), each token can only attend to tokens
that came before it -- not future tokens. We enforce this by setting future
positions to -infinity before softmax, which makes their attention weight
exactly zero.

```
Causal Mask (4 tokens):       After masking:
+---+---+---+---+            +----+----+----+----+
| 1 | 0 | 0 | 0 |            | s1 |-inf|-inf|-inf|
| 1 | 1 | 0 | 0 |     ->     | s2 | s3 |-inf|-inf|
| 1 | 1 | 1 | 0 |            | s4 | s5 | s6 |-inf|
| 1 | 1 | 1 | 1 |            | s7 | s8 | s9 | s10|
+---+---+---+---+            +----+----+----+----+
```
"""

# %% nbgrader={"grade": false, "grade_id": "attn-apply-mask", "solution": true}
#| export
def _apply_mask(scores: Tensor, mask: Tensor) -> Tensor:
    """Apply causal mask by setting masked positions to -infinity.

    TODO: Add large negative values to positions where mask is 0

    APPROACH:
    1. Compute additive mask: (1 - mask) * MASK_VALUE
    2. Add to scores (masked positions become -inf, unmasked unchanged)

    EXAMPLE:
    >>> scores = Tensor(np.ones((1, 3, 3)))
    >>> mask = Tensor(np.tril(np.ones((1, 3, 3))))  # lower triangle
    >>> masked = _apply_mask(scores, mask)
    >>> print(masked.data[0, 0, 1])  # -1e9 (future position masked)

    HINT: mask=0 means "block this position", mask=1 means "allow"
    """
    ### BEGIN SOLUTION
    adder = (1.0 - mask.data) * MASK_VALUE
    return scores + Tensor(adder)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Causal Masking

**What we're testing**: Future positions get set to large negative values
**Why it matters**: Without masking, GPT could "cheat" by looking at future tokens
**Expected**: Masked positions ~ -1e9, unmasked positions unchanged
"""

# %% nbgrader={"grade": true, "grade_id": "test-attn-mask", "locked": true, "points": 5}
def test_unit_apply_mask():
    """🧪 Test causal mask application."""
    print("🧪 Unit Test: Causal Masking...")
    scores = Tensor(np.ones((1, 3, 3)))
    mask = Tensor(np.tril(np.ones((1, 3, 3))))
    masked = _apply_mask(scores, mask)
    # Future positions should be large negative
    assert masked.data[0, 0, 1] < -1e8, "Future position not masked"
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

The following commented-out code shows how attention works conceptually
using explicit loops. While easier to understand, this approach is
NOT used here because:
1. It is extremely slow (Python loops vs optimized C/BLAS)
2. It breaks the autograd graph unless we manually implement the backward pass

Conceptually, this is what the vectorized helpers above are doing:

```
batch_size, n_heads, seq_len, d_k = Q.shape
scores = np.zeros((batch_size, n_heads, seq_len, seq_len))

for b in range(batch_size):
    for h in range(n_heads):
        for i in range(seq_len):          # Each query
            for j in range(seq_len):      # Attends to each key
                dot_product = 0.0
                for k in range(d_k):
                    dot_product += Q[b, h, i, k] * K[b, h, j, k]
                scores[b, h, i, j] = dot_product / math.sqrt(d_k)
```
"""

# %% nbgrader={"grade": false, "grade_id": "attn-scaled-dot-product", "solution": true}
#| export
def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Complete scaled dot-product attention.

    TODO: Compose the helpers into the full attention operation

    APPROACH:
    1. Call _compute_attention_scores(Q, K) for raw similarity
    2. Call _scale_scores(scores, Q.shape[-1]) for numerical stability
    3. If mask provided, call _apply_mask(scores, mask)
    4. Apply Softmax to get probability weights
    5. Multiply weights @ V for attended values

    SUB-PROBLEMS (you already implemented these):
    - _compute_attention_scores: Q @ K^T similarity matrix
    - _scale_scores: divide by sqrt(d) for stable softmax
    - _apply_mask: block future positions with -inf

    Args:
        Q: Query tensor of shape (batch_size, seq_len, d_model)
        K: Key tensor of shape (batch_size, seq_len, d_model)
        V: Value tensor of shape (batch_size, seq_len, d_model)
        mask: Optional causal mask, True=allow, False=mask (batch_size, seq_len, seq_len)

    Returns:
        output: Attended values (batch_size, seq_len, d_model)
        attention_weights: Attention matrix (batch_size, seq_len, seq_len)

    EXAMPLE:
    >>> Q = Tensor(np.random.randn(2, 4, 64))
    >>> K = Tensor(np.random.randn(2, 4, 64))
    >>> V = Tensor(np.random.randn(2, 4, 64))
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
    Q = Tensor(np.random.randn(batch_size, seq_len, d_model))
    K = Tensor(np.random.randn(batch_size, seq_len, d_model))
    V = Tensor(np.random.randn(batch_size, seq_len, d_model))

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

    # Check that future positions have zero attention
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):  # Future positions
                assert abs(weights_masked.data[b, i, j]) < 1e-6, f"Future attention not masked at ({i},{j})"

    print("✅ scaled_dot_product_attention works correctly!")

if __name__ == "__main__":
    test_unit_scaled_dot_product_attention()

# %% [markdown]
"""
## 🏗️ Implementation: Multi-Head Attention

Multi-head attention runs multiple attention "heads" in parallel, each learning to focus on different types of relationships. Think of it as having multiple specialists: one for syntax, one for semantics, one for long-range dependencies, etc.

### Understanding Multi-Head Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ SINGLE-HEAD vs MULTI-HEAD ATTENTION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ SINGLE HEAD ATTENTION (Limited Representation):                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Input (512) → [Linear] → Q,K,V (512) → [Attention] → Output (512)   │ │
│ │                  ↑           ↑            ↑            ↑            │ │
│ │            Single proj  Full dimensions  One head   Limited focus   │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ MULTI-HEAD ATTENTION (Rich Parallel Processing):                        │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Input (512)                                                         │ │
│ │      ↓                                                              │ │
│ │ [Q/K/V Projections] → 512 dimensions each                           │ │
│ │      ↓                                                              │ │
│ │ [Split into 8 heads] → 8 × 64 dimensions per head                   │ │
│ │      ↓                                                              │ │
│ │ Head₁: Q₁(64) ⊗ K₁(64) → Attention₁ → Output₁(64)  │ Syntax focus   │ │
│ │ Head₂: Q₂(64) ⊗ K₂(64) → Attention₂ → Output₂(64)  │ Semantic       │ │
│ │ Head₃: Q₃(64) ⊗ K₃(64) → Attention₃ → Output₃(64)  │ Position       │ │
│ │ Head₄: Q₄(64) ⊗ K₄(64) → Attention₄ → Output₄(64)  │ Long-range     │ │
│ │ Head₅: Q₅(64) ⊗ K₅(64) → Attention₅ → Output₅(64)  │ Local deps     │ │
│ │ Head₆: Q₆(64) ⊗ K₆(64) → Attention₆ → Output₆(64)  │ Coreference    │ │
│ │ Head₇: Q₇(64) ⊗ K₇(64) → Attention₇ → Output₇(64)  │ Composition    │ │
│ │ Head₈: Q₈(64) ⊗ K₈(64) → Attention₈ → Output₈(64)  │ Global view    │ │
│ │      ↓                                                              │ │
│ │ [Concatenate] → 8 × 64 = 512 dimensions                             │ │
│ │      ↓                                                              │ │
│ │ [Output Linear] → Final representation (512)                        │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ Key Benefits of Multi-Head:                                             │
│ • Parallel specialization across different relationship types           │
│ • Same total parameters, distributed across multiple focused heads      │
│ • Each head can learn distinct attention patterns                       │
│ • Enables rich, multifaceted understanding of sequences                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Multi-Head Process Detailed

```
Step 1: Project to Q, K, V
Input (512 dims) → Linear → Q, K, V (512 dims each)

Step 2: Split into Heads
Q (512) → Reshape → 8 heads × 64 dims per head
K (512) → Reshape → 8 heads × 64 dims per head
V (512) → Reshape → 8 heads × 64 dims per head

Step 3: Parallel Attention (for each of 8 heads)
Head 1: Q₁(64) attends to K₁(64) → weights₁ → output₁(64)
Head 2: Q₂(64) attends to K₂(64) → weights₂ → output₂(64)
...
Head 8: Q₈(64) attends to K₈(64) → weights₈ → output₈(64)

Step 4: Concatenate and Mix
[output₁ ∥ output₂ ∥ ... ∥ output₈] (512) → Linear → Final(512)
```

### Why Multiple Heads Are Powerful

Each head can specialize in different patterns:
- **Head 1**: Short-range syntax ("the cat" → subject-article relationship)
- **Head 2**: Long-range coreference ("John...he" → pronoun resolution)
- **Head 3**: Semantic similarity ("dog" ↔ "pet" connections)
- **Head 4**: Positional patterns (attending to specific distances)

This parallelization allows the model to attend to different representation subspaces simultaneously.
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
        ### BEGIN SOLUTION
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Multi-head attention dimension mismatch\n"
                f"  ❌ embed_dim={embed_dim} is not divisible by num_heads={num_heads} (remainder={embed_dim % num_heads})\n"
                f"  💡 Multi-head attention splits embed_dim equally among heads, so embed_dim must be a multiple of num_heads\n"
                f"  🔧 Try: embed_dim={num_heads * (embed_dim // num_heads + 1)} (next valid size) or num_heads={embed_dim // (embed_dim // num_heads)} (fewer heads)"
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
        >>> x = Tensor(np.random.randn(2, 10, 64))  # batch=2, seq=10
        >>> split = mha._split_heads(x, 2, 10)
        >>> print(split.shape)  # (2, 8, 10, 8) -- 8 heads of dim 8

        HINT: reshape(batch, seq, heads, head_dim) then transpose(1, 2)
        """
        ### BEGIN SOLUTION
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
        >>> attended = Tensor(np.random.randn(2, 8, 10, 8))
        >>> merged = mha._merge_heads(attended, 2, 10)
        >>> print(merged.shape)  # (2, 10, 64) -- back to embed_dim

        HINT: transpose(1, 2) then reshape(batch, seq, embed_dim)
        """
        ### BEGIN SOLUTION
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
        >>> x = Tensor(np.random.randn(2, 10, 64))  # batch=2, seq=10, dim=64
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

        # Step 4: Apply attention (reshape mask for head broadcasting)
        mask_reshaped = mask
        if mask is not None and len(mask.shape) == 3:
            batch_size_mask, seq_len_mask, _ = mask.shape
            mask_data = mask.data.reshape(batch_size_mask, 1, seq_len_mask, seq_len_mask)
            mask_reshaped = Tensor(mask_data)

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
        ### BEGIN SOLUTION
        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params
        ### END SOLUTION

# %% [markdown]
"""
### Helper: Splitting Heads

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
    x = Tensor(np.random.randn(2, 10, 64))
    split = mha._split_heads(x, 2, 10)
    assert split.shape == (2, 8, 10, 8), f"Expected (2,8,10,8), got {split.shape}"
    print("✅ Split heads: correct 4D shape!")

if __name__ == "__main__":
    test_unit_split_heads()

# %% [markdown]
"""
### Helper: Merging Heads

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
    x_4d = Tensor(np.random.randn(2, 8, 10, 8))
    merged = mha._merge_heads(x_4d, 2, 10)
    assert merged.shape == (2, 10, 64), f"Expected (2,10,64), got {merged.shape}"

    # Verify round-trip: split then merge recovers original data
    original = Tensor(np.random.randn(2, 10, 64))
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
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

    output = mha.forward(x)

    # Check output shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim), f"Output shape {output.shape} incorrect"

    # Test with causal mask
    mask = Tensor(np.tril(np.ones((batch_size, seq_len, seq_len))))
    output_masked = mha.forward(x, mask)
    assert output_masked.shape == (batch_size, seq_len, embed_dim)

    # Test different head configurations
    mha_small = MultiHeadAttention(embed_dim=32, num_heads=4)
    x_small = Tensor(np.random.randn(1, 5, 32))
    output_small = mha_small.forward(x_small)
    assert output_small.shape == (1, 5, 32)

    print("✅ MultiHeadAttention works correctly!")

if __name__ == "__main__":
    test_unit_multihead_attention()

# %% [markdown]
"""
## 📊 Systems Analysis: Memory Layout and Performance

Let's understand ONE key systems concept: **attention's O(n^2) memory and compute scaling**.

This single analysis reveals why attention becomes the bottleneck in modern transformers and drives research into efficient attention variants.

### Memory Complexity Visualization

```
Attention Memory Scaling (per layer):

Sequence Length = 128:
┌────────────────────────────────┐
│ Attention Matrix: 128x128      │ = 16K values
│ Memory: 64 KB (float32)        │
└────────────────────────────────┘

Sequence Length = 512:
┌────────────────────────────────┐
│ Attention Matrix: 512x512      │ = 262K values
│ Memory: 1 MB (float32)         │ <- 16x larger!
└────────────────────────────────┘

Sequence Length = 2048 (GPT-3):
┌────────────────────────────────┐
│ Attention Matrix: 2048x2048    │ = 4.2M values
│ Memory: 16 MB (float32)        │ <- 256x larger than 128!
└────────────────────────────────┘

For a 96-layer model (GPT-3):
Total Attention Memory = 96 layers x 16 MB = 1.5 GB
Just for attention matrices!
```
"""

# %%
def analyze_attention_complexity():
    """📊 Analyze attention computational complexity and memory scaling."""
    print("📊 Analyzing Attention Complexity...")

    # Test different sequence lengths to show O(n²) scaling
    embed_dim = 64
    sequence_lengths = [16, 32, 64, 128, 256]

    print("\nSequence Length vs Attention Matrix Size:")
    print("Seq Len | Attention Matrix | Memory (KB) | Complexity")
    print("-" * 55)

    for seq_len in sequence_lengths:
        # Calculate attention matrix size
        attention_matrix_size = seq_len * seq_len

        # Memory for attention weights (float32 = 4 bytes)
        attention_memory_kb = (attention_matrix_size * 4) / 1024

        # Total complexity (Q@K + softmax + weights@V)
        complexity = 2 * seq_len * seq_len * embed_dim + seq_len * seq_len

        print(f"{seq_len:7d} | {attention_matrix_size:14d} | {attention_memory_kb:10.2f} | {complexity:10.0f}")

    print(f"\n💡 KEY INSIGHT: Attention memory scales as O(n^2) with sequence length")
    print(f"🚀 For seq_len=1024, attention matrix alone needs {(1024*1024*4)/1024/1024:.1f} MB")

# Run the analysis
if __name__ == "__main__":
    analyze_attention_complexity()

# %%
def analyze_attention_timing():
    """📊 Measure attention computation time vs sequence length."""
    print("\n📊 Analyzing Attention Timing...")

    embed_dim, num_heads = 64, 8
    sequence_lengths = [32, 64, 128, 256]

    print("\nSequence Length vs Computation Time:")
    print("Seq Len | Time (ms) | Ops/sec | Scaling")
    print("-" * 40)

    prev_time = None
    for seq_len in sequence_lengths:
        # Create test input
        x = Tensor(np.random.randn(1, seq_len, embed_dim))
        mha = MultiHeadAttention(embed_dim, num_heads)

        # Time multiple runs for stability
        times = []
        for _ in range(5):
            start_time = time.time()
            _ = mha.forward(x)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        ops_per_sec = 1000 / avg_time if avg_time > 0 else 0

        # Calculate scaling factor vs previous
        scaling = avg_time / prev_time if prev_time else 1.0

        print(f"{seq_len:7d} | {avg_time:8.2f} | {ops_per_sec:7.0f} | {scaling:6.2f}x")
        prev_time = avg_time

    print(f"\n💡 KEY INSIGHT: Attention time scales roughly as O(n^2) with sequence length")
    print(f"🚀 This is why attention efficiency techniques are an active area of research")

# Run the analysis
if __name__ == "__main__":
    analyze_attention_timing()

# %%
def analyze_attention_memory_overhead():
    """📊 Analyze memory overhead during training (forward + backward passes)."""
    print("\n📊 Analyzing Attention Memory Overhead During Training...")

    embed_dim, num_heads = 128, 8
    sequence_lengths = [128, 256, 512, 1024]

    print("\nMemory Overhead Analysis (Training vs Inference):")
    print("Seq Len | Forward | + Gradients | + Optimizer | Total Memory")
    print("-" * 65)

    for seq_len in sequence_lengths:
        # Forward pass memory (attention matrix)
        attention_matrix_mb = (seq_len * seq_len * 4) / (1024 * 1024)

        # Backward pass adds gradient storage (2× forward)
        backward_memory_mb = 2 * attention_matrix_mb

        # Optimizer state (Adam: +2× for momentum and velocity)
        optimizer_memory_mb = backward_memory_mb + 2 * attention_matrix_mb

        # Total = forward + gradients + optimizer state
        total_memory_mb = attention_matrix_mb + backward_memory_mb + optimizer_memory_mb

        print(f"{seq_len:7d} | {attention_matrix_mb:6.2f}MB | {backward_memory_mb:10.2f}MB | {optimizer_memory_mb:10.2f}MB | {total_memory_mb:11.2f}MB")

    print(f"\n💡 KEY INSIGHT: Training requires ~7x memory of inference (1x forward + 2x gradients + 4x optimizer state)")
    print(f"🚀 For GPT-3 (96 layers, 2048 context): ~6GB just for attention gradients!")

# Run the analysis
if __name__ == "__main__":
    analyze_attention_memory_overhead()

# %% [markdown]
"""
### Systems Insights: The O(n^2) Reality

Our analysis reveals the fundamental challenge that drives modern attention research:

**Memory Scaling Crisis:**
- Attention matrix grows as n^2 with sequence length
- For GPT-3 context (2048 tokens): 16MB just for attention weights per layer
- With 96 layers: 1.5GB just for attention matrices!
- This excludes activations, gradients, and other tensors

**Time Complexity Validation:**
- Each sequence length doubling roughly quadruples computation time
- This matches the theoretical O(n^2) complexity we implemented
- Real bottleneck shifts from computation to memory at scale

**The Production Reality:**
```
Model Scale Impact:

Small Model (6 layers, 512 context):
Attention Memory = 6 x 1MB = 6MB - Manageable

GPT-3 Scale (96 layers, 2048 context):
Attention Memory = 96 x 16MB = 1.5GB - Significant

GPT-4 Scale (hypothetical: 120 layers, 32K context):
Attention Memory = 120 x 4GB = 480GB - Impossible on single GPU!
```

**Why This Matters:**
This quadratic wall motivates active research into more efficient attention mechanisms (linear attention, sparse attention, Flash Attention).

The quadratic wall is why long-context AI is an active research frontier, not a solved problem.
"""

# %% [markdown]
"""
## 🔧 Integration - Attention Patterns in Action

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
def test_unit_attention_scenarios():
    """Test attention mechanisms in realistic scenarios."""
    print("🧪 Testing Attention Scenarios...")

    # Scenario 1: Small transformer block setup
    print("\n1. Small Transformer Setup:")
    embed_dim, num_heads, seq_len = 128, 8, 32

    # Create embeddings (simulating token embeddings + positional)
    embeddings = Tensor(np.random.randn(2, seq_len, embed_dim))

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
    simple_embed = Tensor(np.random.randn(1, 4, 16))
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
    test_unit_attention_scenarios()

    print("\nRunning performance analysis...")
    analyze_attention_complexity()
    print("\nRunning memory overhead analysis...")
    analyze_attention_memory_overhead()

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 12")

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of attention operations and their systems implications:

### 1. Quadratic Complexity and Memory
**Question**: For sequence length 1024, how much memory does attention's O(n^2) use? What about length 2048?

**Consider**:
- For float32 (4 bytes per value), the attention matrix for seq_len=n requires n^2 x 4 bytes
- Memory for seq_len=1024: 1024^2 x 4 bytes = _____ MB
- Memory for seq_len=2048: 2048^2 x 4 bytes = _____ MB
- Scaling factor when doubling sequence length: _____x
- Why this limits transformer context lengths in production

**Real-world context**: GPT-3's 2048 token context was chosen partly due to this memory constraint. Longer contexts require specialized efficiency techniques (KV caching, sparse attention, Flash Attention).

---

### 2. Attention vs FFN Bottleneck
**Question**: In production transformers, attention is often the memory bottleneck, not the FFN (feed-forward network). Why?

**Consider**:
- A typical transformer has attention + FFN layers
- FFN parameters scale as O(n x d^2) where d is embed_dim
- Attention activations scale as O(n^2)
- For short sequences (n << d): Which dominates? _____
- For long sequences (n >> d): Which dominates? _____
- At what sequence length does attention become the bottleneck?

**Think about**:
- Why does this matter for models like GPT-3 (96 layers, 2048 context)?
- How does this inform architecture choices for different use cases?

---

### 3. Multi-Head Trade-offs
**Question**: 8 attention heads vs 1 head with 8x dimensions - same parameters, different performance. What's the systems difference?

**Consider**:
- Your MultiHeadAttention splits embed_dim=512 into 8 heads of 64 dims each
- Alternative: one head with full 512 dims
- Parameter count: 8 heads x 64 dims vs 1 head x 512 dims = _____ (same or different?)
- Memory access patterns: Multiple small heads vs one large head
- Parallelization: Can heads run in parallel? _____

**Think about**:
- Specialization: Why might diverse small heads learn better than one large head?
- Cache efficiency: Smaller head_dim vs larger single dimension
- Why did the original Transformer paper choose multiple heads?

---

### 4. Masking Costs
**Question**: Causal masking (for autoregressive models) zeros out half the attention matrix. Do we save computation or just correctness?

**Consider**:
- You set masked positions to -infinity before softmax
- In a seq_len=n causal mask, roughly n^2/2 positions are masked (upper triangle)
- Does your implementation skip computation for masked positions? _____
- Does setting scores to -1e9 before softmax save compute? _____

**Think about**:
- What would you need to change to actually skip masked computation?
- In production, does sparse attention (skipping masked positions) help?
- Memory saved: Can we avoid storing masked attention weights?

---

### 5. The Quadratic Memory Challenge
**Question**: Your implementation computes the full (seq_len x seq_len) attention matrix. Why is this the primary memory bottleneck?

**Calculate**:
- For a 4096-token sequence with 32 heads at float32: attention memory = _____ GB
- For a 512-token sequence with 8 heads: attention memory = _____ MB
- How does doubling sequence length affect attention memory?

**Think about**:
- Why is the attention matrix the dominant memory cost (vs. Q, K, V projections)?
- What property of the softmax operation makes it hard to avoid materializing the full matrix?
- Techniques like KV caching and Flash Attention address this in practice.

---

### Bonus: Training Memory Overhead
**Question**: Training requires storing activations for backward pass. How much extra memory does backprop through attention need?

**Calculate**:
- Forward memory: attention matrix = n^2 values
- Backward memory: gradients also n^2 values
- Total training memory: forward + backward = _____ x inference memory
- With Adam optimizer (stores momentum + velocity): _____ x inference memory
- For GPT-3 scale (96 layers, 2048 context): _____ GB just for attention gradients

**Key insight**: Training requires 4x memory of inference (forward + grad + 2x optimizer state).
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Attention Finds Relationships

**What you built:** Attention mechanisms that let tokens interact with each other.

**Why it matters:** Before attention, models processed tokens independently. Attention lets
each token "look at" every other token and decide what's relevant. This is how transformers
understand that "it" refers to "the cat" in a sentence!

In the next module, you'll combine attention with MLPs to build full transformer blocks.
"""

# %%
def demo_attention():
    """🎯 See attention compute relationships."""
    print("🎯 AHA MOMENT: Attention Finds Relationships")
    print("=" * 45)

    # Create Q, K, V for 4 tokens with 8-dim embeddings
    Q = Tensor(np.random.randn(1, 4, 8))
    K = Tensor(np.random.randn(1, 4, 8))
    V = Tensor(np.random.randn(1, 4, 8))

    # Compute attention
    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"Sequence length: 4 tokens")
    print(f"Embedding dim:   8")
    print(f"\nAttention weights shape: {weights.shape}")
    print(f"Each token attends to all 4 positions!")

    print(f"\nToken 0 attention: {weights.data[0, 0, :].round(2)}")
    print("(sums to 1.0 - it's a probability distribution)")

    print("\n✨ Attention lets tokens communicate!")

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
- **Memory bottlenecks**: Attention matrices dominate memory in transformers (1.5GB+ for GPT-3)
- **Multi-head parallelism**: Different heads can specialize in different relationship types
- **Production challenges**: Understanding why attention efficiency research is crucial

### Ready for Next Steps
Your attention implementation is the core mechanism that enables modern language models!
Export with: `tito module complete 12`

**Next**: Module 13 will combine attention with feed-forward layers to build complete transformer blocks!
"""
