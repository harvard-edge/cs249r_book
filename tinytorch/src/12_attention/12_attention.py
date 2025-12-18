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

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
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

For sequence length n, we compute:
1. **QK^T**: n queries × n keys = n² similarity scores
2. **Softmax**: n² weights to normalize
3. **Weights×V**: n² weights × n values = n² operations for aggregation

This quadratic scaling is attention's blessing (global connectivity) and curse (memory/compute limits).

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
## 🏗️ Implementation - Building Scaled Dot-Product Attention

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

# %% nbgrader={"grade": false, "grade_id": "attention-function", "solution": true}
#| export
def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """
    Compute scaled dot-product attention.

    This is the fundamental attention operation that powers all transformer models.
    We'll implement it with explicit loops first to show the O(n²) complexity.

    TODO: Implement scaled dot-product attention step by step

    APPROACH:
    1. Extract dimensions and validate inputs
    2. Compute attention scores with explicit nested loops (show O(n²) complexity)
    3. Scale by 1/√d_k for numerical stability
    4. Apply causal mask if provided (set masked positions to -inf)
    5. Apply softmax to get attention weights
    6. Apply values with attention weights (another O(n²) operation)
    7. Return output and attention weights

    Args:
        Q: Query tensor of shape (batch_size, seq_len, d_model)
        K: Key tensor of shape (batch_size, seq_len, d_model)
        V: Value tensor of shape (batch_size, seq_len, d_model)
        mask: Optional causal mask, True=allow, False=mask (batch_size, seq_len, seq_len)

    Returns:
        output: Attended values (batch_size, seq_len, d_model)
        attention_weights: Attention matrix (batch_size, seq_len, seq_len)

    EXAMPLE:
    >>> Q = Tensor(np.random.randn(2, 4, 64))  # batch=2, seq=4, dim=64
    >>> K = Tensor(np.random.randn(2, 4, 64))
    >>> V = Tensor(np.random.randn(2, 4, 64))
    >>> output, weights = scaled_dot_product_attention(Q, K, V)
    >>> print(output.shape)  # (2, 4, 64)
    >>> print(weights.shape)  # (2, 4, 4)
    >>> print(weights.data[0].sum(axis=1))  # Each row sums to ~1.0

    HINTS:
    - Use explicit nested loops to compute Q[i] @ K[j] for educational purposes
    - Scale factor is 1/√d_k where d_k is the last dimension of Q
    - Masked positions should be set to -1e9 before softmax
    - Remember that softmax normalizes along the last dimension
    """
    ### BEGIN SOLUTION
    # Step 1: Extract dimensions and validate
    # Note: Q, K, V can be 3D (batch, seq, dim) or 4D (batch, heads, seq, dim)
    # We use shape[-1] for d_model to handle both cases
    d_model = Q.shape[-1]

    # Step 2: Compute attention scores using matrix multiplication
    # Q: (..., seq_len, d_model)
    # K: (..., seq_len, d_model) -> K.T: (..., d_model, seq_len)
    # scores = Q @ K.T -> (..., seq_len, seq_len)

    # Transpose K for matrix multiplication
    # For 3D/4D tensors, transpose swaps the last two dimensions
    K_t = K.transpose(-2, -1)

    scores = Q.matmul(K_t)

    # Step 3: Scale by 1/√d_k for numerical stability
    scale_factor = 1.0 / math.sqrt(d_model)
    scores = scores * scale_factor

    # Step 4: Apply causal mask if provided
    if mask is not None:
        # Mask values of 0 indicate positions to mask out (set to -inf)
        # We use (1 - mask) * MASK_VALUE to add large negative values to masked positions
        # mask is expected to be 0 for masked, 1 for unmasked

        # Ensure mask is broadcastable
        mask_data = mask.data
        adder_mask = (1.0 - mask_data) * MASK_VALUE
        adder_mask_tensor = Tensor(adder_mask)
        scores = scores + adder_mask_tensor

    # Step 5: Apply softmax to get attention weights
    softmax = Softmax()
    attention_weights = softmax(scores, dim=-1)

    # Step 6: Apply values with attention weights
    # weights: (..., seq_len, seq_len)
    # V: (..., seq_len, d_model)
    # output = weights @ V -> (..., seq_len, d_model)
    output = attention_weights.matmul(V)

    # ------------------------------------------------------------------
    # PEDAGOGICAL NOTE: Explicit Loop Implementation
    # ------------------------------------------------------------------
    # The following commented-out code shows how attention works conceptually
    # using explicit loops. While easier to understand, this approach is
    # NOT used here because:
    # 1. It is extremely slow (Python loops vs optimized C/BLAS)
    # 2. It breaks the autograd graph unless we manually implement the backward pass
    #
    # Conceptually, this is what the vectorized code above is doing:
    #
    # batch_size, n_heads, seq_len, d_k = Q.shape
    # scores = Tensor(np.zeros((batch_size, n_heads, seq_len, seq_len)), requires_grad=True)
    #
    # for b in range(batch_size):
    #     for h in range(n_heads):
    #         for i in range(seq_len):
    #             for j in range(seq_len):
    #                 # Dot product of query i and key j
    #                 dot_product = 0.0
    #                 for k in range(d_k):
    #                     dot_product += Q.data[b, h, i, k] * K.data[b, h, j, k]
    #
    #                 # Scale and store
    #                 scores.data[b, h, i, j] = dot_product / math.sqrt(d_k)
    #
    # # ... apply mask ...
    # # ... apply softmax ...
    #
    # output = Tensor(np.zeros((batch_size, n_heads, seq_len, d_k)), requires_grad=True)
    # for b in range(batch_size):
    #     for h in range(n_heads):
    #         for i in range(seq_len):
    #             for k in range(d_k):
    #                 # Weighted sum of values
    #                 weighted_sum = 0.0
    #                 for j in range(seq_len):
    #                     weighted_sum += attention_weights.data[b, h, i, j] * V.data[b, h, j, k]
    #                 output.data[b, h, i, k] = weighted_sum
    # ------------------------------------------------------------------

    return output, attention_weights
    ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-attention-basic", "locked": true, "points": 10}
def test_unit_scaled_dot_product_attention():
    """🔬 Unit Test: Scaled Dot-Product Attention"""
    print("🔬 Unit Test: Scaled Dot-Product Attention...")

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

# %% [markdown]
"""
### 🧪 Unit Test: Scaled Dot-Product Attention

This test validates our core attention mechanism:
- **Output shapes**: Ensures attention preserves sequence dimensions
- **Probability constraint**: Attention weights must sum to 1 per query
- **Causal masking**: Future positions should have zero attention weight

**Why attention weights sum to 1**: Each query position creates a probability distribution over all key positions. This ensures the output is a proper weighted average of values.

**Why causal masking matters**: In language modeling, positions shouldn't attend to future tokens (information they wouldn't have during generation).

**The O(n²) complexity you just witnessed**: Our explicit loops show exactly why attention scales quadratically - every query position must compare with every key position.
"""

# %% [markdown]
"""
## 🏗️ Implementation - Multi-Head Attention

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
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).\n"
                f"  Issue: Multi-head attention splits embed_dim into num_heads heads.\n"
                f"  Fix: Choose embed_dim and num_heads such that embed_dim % num_heads == 0.\n"
                f"  Example: embed_dim=512, num_heads=8 works (512/8=64 per head)."
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

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through multi-head attention.

        TODO: Implement the complete multi-head attention forward pass

        APPROACH:
        1. Extract input dimensions (batch_size, seq_len, embed_dim)
        2. Project input to Q, K, V using linear layers
        3. Reshape projections to separate heads: (batch, seq, heads, head_dim)
        4. Transpose to (batch, heads, seq, head_dim) for parallel processing
        5. Apply scaled dot-product attention to each head
        6. Transpose back and reshape to merge heads
        7. Apply output projection

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

        HINTS:
        - Reshape: (batch, seq, embed_dim) → (batch, seq, heads, head_dim)
        - Transpose: (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        - After attention: reverse the process to merge heads
        - Use scaled_dot_product_attention for each head
        """
        ### BEGIN SOLUTION
        # Step 1: Extract dimensions
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Input dimension mismatch in MultiHeadAttention.forward().\n"
                f"  Expected: embed_dim={self.embed_dim} (set during initialization)\n"
                f"  Got: embed_dim={embed_dim} from input shape {x.shape}\n"
                f"  Fix: Ensure input tensor's last dimension matches the embed_dim used when creating MultiHeadAttention."
            )

        # Step 2: Project to Q, K, V
        Q = self.q_proj.forward(x)  # (batch, seq, embed_dim)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        # Step 3: Reshape to separate heads
        # From (batch, seq, embed_dim) to (batch, seq, num_heads, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Step 4: Transpose to (batch, num_heads, seq, head_dim) for parallel processing
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Step 5: Apply attention
        # We can apply attention to all heads at once because scaled_dot_product_attention
        # supports broadcasting or 4D tensors if implemented correctly.

        # Reshape mask if necessary to broadcast over heads
        mask_reshaped = mask
        if mask is not None and len(mask.shape) == 3:
             # Add head dimension: (batch, seq, seq) -> (batch, 1, seq, seq)
             # This allows the mask to broadcast across all attention heads
             batch_size_mask, seq_len_mask, _ = mask.shape
             mask_data = mask.data.reshape(batch_size_mask, 1, seq_len_mask, seq_len_mask)
             mask_reshaped = Tensor(mask_data)

        attended, _ = scaled_dot_product_attention(Q, K, V, mask=mask_reshaped)

        # Step 6: Concatenate heads back together
        # Transpose back: (batch, num_heads, seq, head_dim) → (batch, seq, num_heads, head_dim)
        attended = attended.transpose(1, 2)

        # Reshape: (batch, seq, num_heads, head_dim) → (batch, seq, embed_dim)
        concat_output = attended.reshape(batch_size, seq_len, self.embed_dim)

        # Step 7: Apply output projection
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

# %% nbgrader={"grade": true, "grade_id": "test-multihead", "locked": true, "points": 15}
def test_unit_multihead_attention():
    """🔬 Unit Test: Multi-Head Attention"""
    print("🔬 Unit Test: Multi-Head Attention...")

    # Test initialization
    embed_dim, num_heads = 64, 8
    mha = MultiHeadAttention(embed_dim, num_heads)

    # Check configuration
    assert mha.embed_dim == embed_dim
    assert mha.num_heads == num_heads
    assert mha.head_dim == embed_dim // num_heads

    # Test parameter counting (4 linear layers, each has weight + bias)
    params = mha.parameters()
    assert len(params) == 8, f"Expected 8 parameters (4 layers × 2), got {len(params)}"

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

# %% [markdown]
"""
### 🧪 Unit Test: Multi-Head Attention

This test validates our multi-head attention implementation:
- **Configuration**: Correct head dimension calculation and parameter setup
- **Parameter counting**: 4 linear layers × 2 parameters each = 8 total
- **Shape preservation**: Output maintains input dimensions
- **Masking support**: Causal masks work correctly with multiple heads

**Why multi-head attention works**: Different heads can specialize in different types of relationships (syntactic, semantic, positional), providing richer representations than single-head attention.

**Architecture insight**: The split → attend → concat pattern allows parallel processing of different representation subspaces, dramatically increasing the model's capacity to understand complex relationships.
"""

# %% [markdown]
"""
## 📊 Systems Analysis - Attention's Computational Reality

Now let's analyze the computational and memory characteristics that make attention both powerful and challenging at scale.

### Memory Complexity Visualization

```
Attention Memory Scaling (per layer):

Sequence Length = 128:
┌────────────────────────────────┐
│ Attention Matrix: 128×128      │ = 16K values
│ Memory: 64 KB (float32)        │
└────────────────────────────────┘

Sequence Length = 512:
┌────────────────────────────────┐
│ Attention Matrix: 512×512      │ = 262K values
│ Memory: 1 MB (float32)         │ ← 16× larger!
└────────────────────────────────┘

Sequence Length = 2048 (GPT-3):
┌────────────────────────────────┐
│ Attention Matrix: 2048×2048    │ = 4.2M values
│ Memory: 16 MB (float32)        │ ← 256× larger than 128!
└────────────────────────────────┘

For a 96-layer model (GPT-3):
Total Attention Memory = 96 layers × 16 MB = 1.5 GB
Just for attention matrices!
```
"""

# %% nbgrader={"grade": false, "grade_id": "attention-complexity", "solution": true}
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

    print(f"\n💡 Attention memory scales as O(n²) with sequence length")
    print(f"🚀 For seq_len=1024, attention matrix alone needs {(1024*1024*4)/1024/1024:.1f} MB")

# %% nbgrader={"grade": false, "grade_id": "attention-timing", "solution": true}
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

    print(f"\n💡 Attention time scales roughly as O(n²) with sequence length")
    print(f"🚀 This is why efficient attention (FlashAttention) is crucial for long sequences")

# %% nbgrader={"grade": false, "grade_id": "attention-memory-overhead", "solution": true}
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

        print(f"{seq_len:7d} | {attention_matrix_mb:6.2f}MB | {backward_memory_mb:10.2f}MB | {optimizer_memory_mb:10.2f}MB | {optimizer_memory_mb:11.2f}MB")

    print(f"\n💡 Training requires 4× memory of inference (forward + grad + 2× optimizer state)")
    print(f"🚀 For GPT-3 (96 layers, 2048 context): ~6GB just for attention gradients!")

# %% [markdown]
"""
### 📊 Systems Analysis: The O(n²) Reality

Our analysis reveals the fundamental challenge that drives modern attention research:

**Memory Scaling Crisis:**
- Attention matrix grows as n² with sequence length
- For GPT-3 context (2048 tokens): 16MB just for attention weights per layer
- With 96 layers: 1.5GB just for attention matrices!
- This excludes activations, gradients, and other tensors

**Time Complexity Validation:**
- Each sequence length doubling roughly quadruples computation time
- This matches the theoretical O(n²) complexity we implemented with explicit loops
- Real bottleneck shifts from computation to memory at scale

**The Production Reality:**
```
Model Scale Impact:

Small Model (6 layers, 512 context):
Attention Memory = 6 × 1MB = 6MB ✅ Manageable

GPT-3 Scale (96 layers, 2048 context):
Attention Memory = 96 × 16MB = 1.5GB ⚠️ Significant

GPT-4 Scale (hypothetical: 120 layers, 32K context):
Attention Memory = 120 × 4GB = 480GB ❌ Impossible on single GPU!
```

**Why This Matters:**
- **FlashAttention**: Reformulates computation to reduce memory without changing results
- **Sparse Attention**: Only compute attention for specific patterns (local, strided)
- **Linear Attention**: Approximate attention with linear complexity
- **State Space Models**: Alternative architectures that avoid attention entirely

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

# %% nbgrader={"grade": false, "grade_id": "attention-scenarios", "solution": true}
def test_attention_scenarios():
    """Test attention mechanisms in realistic scenarios."""
    print("🔬 Testing Attention Scenarios...")

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
### 🧪 Integration Test: Attention Scenarios

This comprehensive test validates attention in realistic use cases:

**Transformer Setup**: Standard configuration matching real architectures
- 128-dimensional embeddings with 8 attention heads
- 16 dimensions per head (128 ÷ 8 = 16)
- Proper parameter counting and shape preservation

**Causal Language Modeling**: Essential for GPT-style models
- Lower triangular mask ensures autoregressive property
- Position i cannot attend to positions j > i (future tokens)
- Critical for language generation and training stability

**Attention Pattern Visualization**: Understanding what the model "sees"
- Each row sums to 1.0 (valid probability distribution)
- Patterns reveal which positions the model finds relevant
- Causal masking creates structured sparsity in attention

**Real-World Implications**:
- These patterns are interpretable in trained models
- Attention heads often specialize (syntax, semantics, position)
- Visualization tools like BertViz use these matrices for model interpretation

The attention matrices you see here are the foundation of model interpretability in transformers.
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
    test_unit_scaled_dot_product_attention()
    test_unit_multihead_attention()

    print("\nRunning integration scenarios...")
    test_attention_scenarios()

    print("\nRunning performance analysis...")
    analyze_attention_complexity()
    print("\nRunning memory overhead analysis...")
    analyze_attention_memory_overhead()

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 12")

# %% nbgrader={"grade": false, "grade_id": "main-execution", "solution": false}
# Run comprehensive module test when executed directly
if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

These questions help you connect your implementation to production ML systems and real-world trade-offs.

### Question 1: Quadratic Complexity
For sequence length 1024, how much memory does attention's O(n²) use? What about length 2048?

**Context**: You implemented attention with explicit nested loops showing the quadratic scaling. For float32 (4 bytes per value), the attention matrix for seq_len=n requires n² × 4 bytes.

**Think about**:
- Memory for seq_len=1024: 1024² × 4 bytes = _____ MB
- Memory for seq_len=2048: 2048² × 4 bytes = _____ MB
- Scaling factor when doubling sequence length: _____×
- Why this limits transformer context lengths in production

### Question 2: Attention Bottleneck
In production transformers, attention is often the memory bottleneck, not the FFN (feed-forward network). Why?

**Context**: A typical transformer has attention + FFN layers. FFN parameters scale as O(n × d²) where d is embed_dim, while attention activations scale as O(n²).

**Think about**:
- For short sequences (n << d): Which dominates, attention or FFN? _____
- For long sequences (n >> d): Which dominates? _____
- At what sequence length does attention become the bottleneck?
- Why does this matter for models like GPT-3 (96 layers, 2048 context)?

### Question 3: Multi-Head Trade-off
8 attention heads vs 1 head with 8× dimensions - same parameters, different performance. What's the systems difference?

**Context**: Your MultiHeadAttention splits embed_dim=512 into 8 heads of 64 dims each. Alternative: one head with full 512 dims.

**Think about**:
- Parameter count: 8 heads × 64 dims vs 1 head × 512 dims = _____ (same or different?)
- Memory access patterns: Multiple small heads vs one large head
- Parallelization: Can heads run in parallel? _____
- Specialization: Why might diverse small heads learn better than one large head?
- Cache efficiency: Smaller head_dim vs larger single dimension

### Question 4: Masking Cost
Causal masking (for autoregressive models) zeros out half the attention matrix. Do we save computation or just correctness?

**Context**: You set masked positions to -∞ before softmax. In a seq_len=n causal mask, roughly n²/2 positions are masked (upper triangle).

**Think about**:
- Does your implementation skip computation for masked positions? _____
- Does setting scores to -1e9 before softmax save compute? _____
- What would you need to change to actually skip masked computation?
- In production, does sparse attention (skipping masked positions) help? _____
- Memory saved: Can we avoid storing masked attention weights?

### Question 5: Flash Attention
Modern systems use "flash attention" to reduce attention's memory from O(n²) to O(n). How might this work conceptually?

**Context**: Your implementation computes full attention matrix (batch, seq_len, seq_len), then applies it to values. FlashAttention reformulates this to never materialize the full matrix.

**Think about**:
- Your implementation: stores (seq_len × seq_len) attention weights
- FlashAttention idea: Compute attention in _____? (blocks, tiles, chunks)
- Recomputation trade-off: Save memory by _____ during backward pass
- Why does this enable longer context windows?
- Is this an algorithm change or just implementation optimization?

### Question 6: Gradient Memory (Bonus)
Training requires storing activations for backward pass. How much extra memory does backprop through attention need?

**Context**: Your forward pass creates attention matrices. Backward pass needs these for gradients.

**Think about**:
- Forward memory: attention matrix = n² values
- Backward memory: gradients also n² values
- Total training memory: forward + backward = _____ × inference memory
- With Adam optimizer (stores momentum + velocity): _____ × inference memory
- For GPT-3 scale (96 layers, 2048 context): _____ GB just for attention gradients
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
- Built scaled dot-product attention with explicit O(n²) complexity demonstration
- Implemented multi-head attention for parallel relationship learning
- Experienced attention's quadratic memory scaling firsthand through analysis
- Tested causal masking for language modeling applications
- Visualized actual attention patterns and weight distributions
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Gained
- **Computational Complexity**: Witnessed O(n²) scaling in both memory and time through explicit loops
- **Memory Bottlenecks**: Attention matrices dominate memory usage in transformers (1.5GB+ for GPT-3 scale)
- **Parallel Processing**: Multi-head attention enables diverse relationship learning across representation subspaces
- **Production Challenges**: Understanding why FlashAttention and efficient attention research are crucial
- **Interpretability Foundation**: Attention matrices provide direct insight into model focus patterns

### Ready for Next Steps
Your attention implementation is the core mechanism that enables modern language models!
Export with: `tito module complete 12`

**Next**: Module 13 will combine attention with feed-forward layers to build complete transformer blocks!

### What You Just Built Powers
- **GPT models**: Your attention mechanism is the exact pattern used in ChatGPT and GPT-4
- **BERT and variants**: Bidirectional attention for understanding tasks
- **Vision Transformers**: The same attention applied to image patches
- **Modern AI systems**: Nearly every state-of-the-art language and multimodal model

The mechanism you just implemented with explicit loops is mathematically identical to the attention in production language models - you've built the foundation of modern AI!
"""
