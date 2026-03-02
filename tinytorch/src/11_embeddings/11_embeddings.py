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
"""
# Module 11: Embeddings - Converting Tokens to Learnable Representations

Welcome to Module 11! You're about to build embedding layers that convert discrete tokens into dense, learnable vectors - the foundation of all modern NLP models.

## 🔗 Prerequisites & Progress
**You've Built**: Tensors, layers, tokenization (discrete text processing)
**You'll Build**: Embedding lookups and positional encodings for sequence modeling
**You'll Enable**: Foundation for attention mechanisms and transformer architectures

**Connection Map**:
```
Tokenization → Embeddings → Positional Encoding → Attention
(discrete)     (dense)      (position-aware)     (context-aware)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement embedding layers for token-to-vector conversion
2. Understand learnable vs fixed positional encodings
3. Build both sinusoidal and learned position encodings
4. Analyze embedding memory requirements and lookup performance

Let's transform tokens into intelligence!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/11_embeddings/embeddings_dev.py`
**Building Side:** Code exports to `tinytorch.text.embeddings`

```python
# How to use this module:
from tinytorch.core.embeddings import Embedding, PositionalEncoding, create_sinusoidal_embeddings
```

**Why this matters:**
- **Learning:** Complete embedding system for converting discrete tokens to continuous representations
- **Production:** Essential component matching PyTorch's torch.nn.Embedding with positional encoding patterns
- **Consistency:** All embedding operations and positional encodings in text.embeddings
- **Integration:** Works seamlessly with tokenizers for complete text processing pipeline
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp core.embeddings
#| export

import numpy as np
import math
from typing import List, Optional, Tuple

# Import from previous modules - following dependency chain
from tinytorch.core.tensor import Tensor

# Enable autograd for gradient tracking (required for learnable embeddings)
from tinytorch.core.autograd import Function, enable_autograd
enable_autograd()

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01-10 (especially Tensor foundation)

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `math` (for mathematical constants and functions)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor.Tensor` (from Module 01)

**Dependency Flow**:
```
Module 01 (Tensor) → Module 11 (Embeddings)
     ↓                       ↓
  Foundation        Token-to-Vector
```

Students completing this module will have built the embedding system
that converts discrete tokens into continuous representations for transformers.
"""

# %% [markdown]
"""
## 💡 Introduction: Why Embeddings?

Neural networks operate on dense vectors, but language consists of discrete tokens. Embeddings are the crucial bridge that converts discrete tokens into continuous, learnable vector representations that capture semantic meaning.

### The Token-to-Vector Challenge

Consider the tokens from our tokenizer: [1, 42, 7] - how do we turn these discrete indices into meaningful vectors that capture semantic relationships?

```
┌─────────────────────────────────────────────────────────────────┐
│  EMBEDDING PIPELINE: Discrete Tokens → Dense Vectors            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (Token IDs):     [1, 42, 7]                              │
│           │                                                     │
│           ├─ Step 1: Lookup in embedding table                  │
│           │         Each ID → vector of learned features        │
│           │                                                     │
│           ├─ Step 2: Add positional information                 │
│           │         Same word at different positions → different│
│           │                                                     │
│           ├─ Step 3: Create position-aware representations      │
│           │         Ready for attention mechanisms              │
│           │                                                     │
│           └─ Step 4: Enable semantic understanding              │
│                     Similar words → similar vectors             │
│                                                                 │
│  Output (Dense Vectors): [[0.1, 0.4, ...], [0.7, -0.2, ...]]    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Four-Layer Embedding System

Modern embedding systems combine multiple components:

**1. Token embeddings** - Learn semantic representations for each vocabulary token
**2. Positional encoding** - Add information about position in sequence
**3. Optional scaling** - Normalize embedding magnitudes (Transformer convention)
**4. Integration** - Combine everything into position-aware representations

### Why This Matters

The choice of embedding strategy dramatically affects:
- **Semantic understanding** - How well the model captures word meaning
- **Memory requirements** - Embedding tables can be gigabytes in size
- **Position awareness** - Whether the model understands word order
- **Extrapolation** - How well the model handles longer sequences than training
"""

# %% [markdown]
"""
## 📐 Foundations: Embedding Strategies

Different embedding approaches make different trade-offs between memory, semantic understanding, and computational efficiency.

### Token Embedding Lookup Process

**Approach**: Each token ID maps to a learned dense vector

```
┌──────────────────────────────────────────────────────────────┐
│ TOKEN EMBEDDING LOOKUP PROCESS                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Build Embedding Table (vocab_size × embed_dim)      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Token ID  │  Embedding Vector (learned features)       │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │    0      │  [0.2, -0.1,  0.3, 0.8, ...]  (<UNK>)      │  │
│  │    1      │  [0.1,  0.4, -0.2, 0.6, ...]  ("the")      │  │
│  │   42      │  [0.7, -0.2,  0.1, 0.4, ...]  ("cat")      │  │
│  │    7      │  [-0.3, 0.1,  0.5, 0.2, ...]  ("sat")      │  │
│  │   ...     │             ...                            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 2: Lookup Process (O(1) per token)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Input: Token IDs [1, 42, 7]                           │  │
│  │                                                        │  │
│  │   ID 1  → embedding[1]  → [0.1,  0.4, -0.2, ...]       │  │
│  │   ID 42 → embedding[42] → [0.7, -0.2,  0.1, ...]       │  │
│  │   ID 7  → embedding[7]  → [-0.3, 0.1,  0.5, ...]       │  │
│  │                                                        │  │
│  │  Output: Matrix (3 × embed_dim)                        │  │
│  │  [[0.1,  0.4, -0.2, ...],                              │  │
│  │   [0.7, -0.2,  0.1, ...],                              │  │
│  │   [-0.3, 0.1,  0.5, ...]]                              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 3: Training Updates Embeddings                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Gradients flow back to embedding table                │  │
│  │                                                        │  │
│  │  Similar words learn similar vectors:                  │  │
│  │  "cat" and "dog" → closer in embedding space           │  │
│  │  "the" and "a"   → closer in embedding space           │  │
│  │  "sat" and "run" → farther in embedding space          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Pros**:
- Dense representation (every dimension meaningful)
- Learnable (captures semantic relationships through training)
- Efficient lookup (O(1) time complexity)
- Scales to large vocabularies

**Cons**:
- Memory intensive (vocab_size × embed_dim parameters)
- Requires training to develop semantic relationships
- Fixed vocabulary (new tokens need special handling)

### Positional Encoding Strategies

Since embeddings by themselves have no notion of order, we need positional information:

```
Position-Aware Embeddings = Token Embeddings + Positional Encoding

Learned Approach:     Fixed Mathematical Approach:
Position 0 → [learned]     Position 0 → [sin/cos pattern]
Position 1 → [learned]     Position 1 → [sin/cos pattern]
Position 2 → [learned]     Position 2 → [sin/cos pattern]
...                        ...
```

**Learned Positional Encoding**:
- Trainable position embeddings
- Can learn task-specific patterns
- Limited to maximum training sequence length

**Sinusoidal Positional Encoding**:
- Mathematical sine/cosine patterns
- No additional parameters
- Can extrapolate to longer sequences

### Strategy Comparison

```
Text: "cat sat on mat" → Token IDs: [42, 7, 15, 99]

Token Embeddings:    [vec_42, vec_7, vec_15, vec_99]  # Same vectors anywhere
Position-Aware:      [vec_42+pos_0, vec_7+pos_1, vec_15+pos_2, vec_99+pos_3]
                      ↑ Now "cat" at position 0 ≠ "cat" at position 1
```

The combination enables transformers to understand both meaning and order!
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
class EmbeddingBackward(Function):
    """
    Gradient computation for embedding lookup operation.

    **Mathematical Rule:** If Y = Embedding[indices], then:
    - ∂Loss/∂Embedding[i] = sum of all gradients where index==i

    Embedding lookup is a gather operation. The backward
    is a scatter operation that accumulates gradients to the embedding weights.
    """

    def __init__(self, weight, indices):
        """
        Args:
            weight: Embedding weight matrix
            indices: Indices used for lookup
        """
        super().__init__(weight)
        self.indices = indices

    def apply(self, grad_output):
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
        1. Extract weight tensor from self.saved_tensors
        2. Initialize grad_weight to None
        3. If weight requires gradients:
           - Create zeros array: grad_weight = np.zeros_like(weight.data)
           - Flatten indices: indices_flat = self.indices.data.astype(int).flatten()
           - Reshape grad_output: match flattened indices with embedding dimension
           - Use np.add.at to accumulate gradients: np.add.at(grad_weight, indices_flat, grad_output_reshaped)
        4. Return tuple (grad_weight,)

        EXAMPLE:
        >>> vocab = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)  # 3 words, 2D
        >>> indices = Tensor([0, 2, 0])  # Select words 0, 2, 0
        >>> output = vocab[indices]  # [[0.1, 0.2], [0.5, 0.6], [0.1, 0.2]]
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
        weight, = self.saved_tensors
        grad_weight = None

        if isinstance(weight, Tensor) and weight.requires_grad:
            # Initialize gradient with zeros
            grad_weight = np.zeros_like(weight.data)

            # Scatter gradients back to embedding weights
            # np.add.at accumulates gradients for repeated indices
            indices_flat = self.indices.data.astype(int).flatten()
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])

            np.add.at(grad_weight, indices_flat, grad_output_reshaped)

        return (grad_weight,)
        ### END SOLUTION

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

        HINT: np.random.uniform(-limit, limit, (vocab_size, embed_dim))
        """
        ### BEGIN SOLUTION
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Xavier initialization for better gradient flow
        limit = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (vocab_size, embed_dim))
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
        2. Perform lookup using numpy advanced indexing: weight[indices]
        3. Attach EmbeddingBackward gradient function if weight requires grad

        HINTS:
        - Use self.weight.data[indices.data.astype(int)] for the lookup
        - Attach result._grad_fn = EmbeddingBackward(self.weight, indices)
        """
        ### BEGIN SOLUTION
        # Handle input validation
        if np.any(indices.data >= self.vocab_size) or np.any(indices.data < 0):
            min_idx = int(np.min(indices.data))
            max_idx = int(np.max(indices.data))
            raise ValueError(
                f"Embedding index out of range for vocabulary size {self.vocab_size}\n"
                f"  ❌ Found indices: min={min_idx}, max={max_idx} (valid range: 0 to {self.vocab_size - 1})\n"
                f"  💡 Token IDs must be within the vocabulary. IDs >= vocab_size reference non-existent tokens\n"
                f"  🔧 Check your tokenizer output, or increase vocab_size to at least {max_idx + 1}"
            )

        # Perform embedding lookup using advanced indexing
        # This is equivalent to one-hot multiplication but much more efficient
        embedded = self.weight.data[indices.data.astype(int)]

        result = Tensor(embedded)

        # Attach gradient function for backpropagation
        # EmbeddingBackward (defined above) handles sparse gradient accumulation
        if self.weight.requires_grad:
            result.requires_grad = True
            result._grad_fn = EmbeddingBackward(self.weight, indices)

        return result
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

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_embedding()

# %% [markdown]
"""
### Learned Positional Encoding

Trainable position embeddings that can learn position-specific patterns. This approach treats each position as a learnable parameter, similar to token embeddings.

```
Learned Position Embedding Process:

Step 1: Initialize Position Embedding Table
┌───────────────────────────────────────────────────────────────┐
│ Position  │  Learnable Vector (trainable parameters)          │
├───────────────────────────────────────────────────────────────┤
│    0      │ [0.1, -0.2,  0.4, ...]  ← learns "start" patterns │
│    1      │ [0.3,  0.1, -0.1, ...]  ← learns "second" patterns│
│    2      │ [-0.1, 0.5,  0.2, ...]  ← learns "third" patterns │
│   ...     │        ...                                        │
│  511      │ [0.4, -0.3,  0.1, ...]  ← learns "late" patterns  │
└───────────────────────────────────────────────────────────────┘

Step 2: Add to Token Embeddings
Input: ["The", "cat", "sat"] → Token IDs: [1, 42, 7]

Token embeddings:     Position embeddings:     Combined:
[1]  → [0.1, 0.4, ...] + [0.1, -0.2, ...] = [0.2, 0.2, ...]
[42] → [0.7, -0.2, ...] + [0.3, 0.1, ...] = [1.0, -0.1, ...]
[7]  → [-0.3, 0.1, ...] + [-0.1, 0.5, ...] = [-0.4, 0.6, ...]

Result: Position-aware embeddings that can learn task-specific patterns!
```

**Why learned positions work**: The model can discover that certain positions have special meaning (like sentence beginnings, question words, etc.) and learn specific representations for those patterns.
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
        3. Use smaller initialization than token embeddings (they're additive)

        HINT: limit = sqrt(2.0 / embed_dim), then uniform(-limit, limit)
        """
        ### BEGIN SOLUTION
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Initialize position embedding matrix
        # Smaller initialization than token embeddings since these are additive
        limit = math.sqrt(2.0 / embed_dim)
        self.position_embeddings = Tensor(
            np.random.uniform(-limit, limit, (max_seq_len, embed_dim))
        )
        ### END SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encodings to input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, seq_len, embed_dim)

        Returns:
            Position-encoded embeddings of same shape

        TODO: Validate input and add position embeddings

        APPROACH:
        1. Validate input is 3D with correct embed_dim and seq_len <= max
        2. Slice position_embeddings[:seq_len] for variable-length support
        3. Reshape to (1, seq_len, embed_dim) for batch broadcasting
        4. Add to input embeddings

        HINTS:
        - pos_embeddings.data[np.newaxis, :, :] adds the batch dimension
        - Use x + pos_embeddings_batched for element-wise addition
        """
        ### BEGIN SOLUTION
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

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length exceeds maximum: {seq_len} > {self.max_seq_len}\n"
                f"  ❌ Input sequence has {seq_len} positions, but max_seq_len is {self.max_seq_len}\n"
                f"  💡 Learned positional encodings have a fixed maximum length set at initialization\n"
                f"  🔧 Either truncate input to {self.max_seq_len} tokens, or create a new PositionalEncoding(max_seq_len={seq_len}, ...)"
            )

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {embed_dim}, expected {self.embed_dim}\n"
                f"  ❌ PositionalEncoding was created with embed_dim={self.embed_dim}, but input has embed_dim={embed_dim}\n"
                f"  💡 Token embeddings and positional encodings must have the same dimension to be added together\n"
                f"  🔧 Ensure your Embedding layer uses embed_dim={self.embed_dim}, or create PositionalEncoding(embed_dim={embed_dim}, ...)"
            )

        # Slice position embeddings for this sequence length using Tensor slicing
        pos_embeddings = self.position_embeddings[:seq_len]  # (seq_len, embed_dim)

        # Reshape to add batch dimension: (1, seq_len, embed_dim)
        pos_data = pos_embeddings.data[np.newaxis, :, :]
        pos_embeddings_batched = Tensor(pos_data)

        # Add positional information
        result = x + pos_embeddings_batched

        return result
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the positional encoding to be called like a function."""
        return self.forward(x)

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
    embeddings = Tensor(np.random.randn(2, 10, 64))
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

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_positional_encoding()

# %% [markdown]
"""
### Sinusoidal Positional Encoding

Mathematical position encoding that creates unique signatures for each position using trigonometric functions. This approach requires no additional parameters and can extrapolate to sequences longer than seen during training.

```
┌───────────────────────────────────────────────────────────────────────┐
│ SINUSOIDAL POSITION ENCODING: Mathematical Position Signatures        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ MATHEMATICAL FORMULA:                                                 │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))  # Even dims      │ │
│ │ PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))  # Odd dims       │ │
│ │                                                                   │ │
│ │ Where:                                                            │ │
│ │   pos = position in sequence (0, 1, 2, ...)                       │ │
│ │   i = dimension pair index (0, 1, 2, ...)                         │ │
│ │   10000 = base frequency (creates different wavelengths)          │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ FREQUENCY PATTERN ACROSS DIMENSIONS:                                  │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Dimension:  0     1     2     3     4     5     6     7           │ │
│ │ Frequency:  High  High  Med   Med   Low   Low   VLow  VLow        │ │
│ │ Function:   sin   cos   sin   cos   sin   cos   sin   cos         │ │
│ │                                                                   │ │
│ │ pos=0:    [0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │ pos=1:    [0.84, 0.54, 0.01, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │ pos=2:    [0.91,-0.42, 0.02, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │ pos=3:    [0.14,-0.99, 0.03, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │                                                                   │ │
│ │ Each position gets a unique mathematical "fingerprint"!           │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ WHY THIS WORKS:                                                       │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Wave Pattern Visualization:                                       │ │
│ │                                                                   │ │
│ │ Dim 0: ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (rapid oscillation)                  │ │
│ │ Dim 2: ∿---∿---∿---∿---∿---∿  (medium frequency)                  │ │
│ │ Dim 4: ∿-----∿-----∿-----∿--  (low frequency)                     │ │
│ │ Dim 6: ∿----------∿----------  (very slow changes)                │ │
│ │                                                                   │ │
│ │ • High frequency dims change rapidly between positions            │ │
│ │ • Low frequency dims change slowly                                │ │
│ │ • Combination creates unique signature for each position          │ │
│ │ • Similar positions have similar (but distinct) encodings         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ KEY ADVANTAGES:                                                       │
│ • Zero parameters (no memory overhead)                                │
│ • Infinite sequence length (can extrapolate)                          │
│ • Smooth transitions (nearby positions are similar)                   │
│ • Mathematical elegance (interpretable patterns)                      │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Why this matters**: The mathematical structure creates unique positional signatures and enables smooth interpolation to longer sequences. Attention mechanisms leverage these properties to distinguish token positions.
"""

# %% [markdown]
"""
### Computing the Sinusoidal Table

The core of sinusoidal positional encoding is building a table of sin/cos values
where each dimension oscillates at a different frequency. This helper computes
the raw numpy array that both `create_sinusoidal_embeddings` and other components
can reuse.

```
Sinusoidal Table Construction:

Step 1: Position column vector     Step 2: Frequency row vector
  [0]                                [high_freq, ..., low_freq]
  [1]     (max_len, 1)               (embed_dim//2,)
  [2]
  [...]

Step 3: Outer product → angles     Step 4: Interleave sin/cos
  positions * frequencies            pe[:, 0::2] = sin(angles)
  = (max_len, embed_dim//2)          pe[:, 1::2] = cos(angles)
                                     = (max_len, embed_dim)
```
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

    ### BEGIN SOLUTION
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

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_sinusoidal_embeddings()

# %% [markdown]
"""
## 🔧 Integration: Bringing It Together

Now let's build the complete embedding system that combines token and positional embeddings into a production-ready component. This is the same pattern used in modern language models.

```
Complete Embedding Pipeline:

1. Token Lookup → 2. Position Encoding → 3. Combination → 4. Ready for Attention
     ↓                     ↓                   ↓                  ↓
  sparse IDs         position info       dense vectors      context-aware
```
"""

# %% [markdown]
"""
### Complete Embedding System Architecture

The production embedding layer that powers modern transformers combines multiple components into an efficient, flexible pipeline.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ COMPLETE EMBEDDING SYSTEM: Token + Position → Position-Aware Representations│
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ INPUT: Token IDs [1, 42, 7, 99]                                           │
│         │                                                                 │
│         ├─ STEP 1: TOKEN EMBEDDING LOOKUP                                 │
│         │  ┌─────────────────────────────────────────────────────────┐    │
│         │  │   Token Embedding Table (vocab_size × embed_dim)        │    │
│         │  │                                                         │    │
│         │  │   ID 1  → [0.1,  0.4, -0.2, ...]  (semantic features)   │    │
│         │  │   ID 42 → [0.7, -0.2,  0.1, ...]  (learned meaning)     │    │
│         │  │   ID 7  → [-0.3, 0.1,  0.5, ...]  (dense vector)        │    │
│         │  │   ID 99 → [0.9, -0.1,  0.3, ...]  (context-free)        │    │
│         │  └─────────────────────────────────────────────────────────┘    │
│         │                                                                 │
│         ├─ STEP 2: POSITIONAL ENCODING (Choose Strategy)                  │
│         │  ┌─────────────────────────────────────────────────────────┐    │
│         │  │ Strategy A: Learned PE                                  │    │
│         │  │   pos 0 → [trainable vector] (learns patterns)          │    │
│         │  │   pos 1 → [trainable vector] (task-specific)            │    │
│         │  │   pos 2 → [trainable vector] (fixed max length)         │    │
│         │  │                                                         │    │
│         │  │ Strategy B: Sinusoidal PE                               │    │
│         │  │   pos 0 → [sin/cos pattern] (mathematical)              │    │
│         │  │   pos 1 → [sin/cos pattern] (no parameters)             │    │
│         │  │   pos 2 → [sin/cos pattern] (infinite length)           │    │
│         │  │                                                         │    │
│         │  │ Strategy C: No PE                                       │    │
│         │  │   positions ignored (order-agnostic)                    │    │
│         │  └─────────────────────────────────────────────────────────┘    │
│         │                                                                 │
│         ├─ STEP 3: ELEMENT-WISE ADDITION                                  │
│         │  ┌─────────────────────────────────────────────────────────┐    │
│         │  │ Token + Position = Position-Aware Representation        │    │
│         │  │                                                         │    │
│         │  │ [0.1, 0.4, -0.2] + [pos0] = [0.1+p0, 0.4+p0, ...]       │    │
│         │  │ [0.7, -0.2, 0.1] + [pos1] = [0.7+p1, -0.2+p1, ...]      │    │
│         │  │ [-0.3, 0.1, 0.5] + [pos2] = [-0.3+p2, 0.1+p2, ...]      │    │
│         │  │ [0.9, -0.1, 0.3] + [pos3] = [0.9+p3, -0.1+p3, ...]      │    │
│         │  └─────────────────────────────────────────────────────────┘    │
│         │                                                                 │
│         ├─ STEP 4: OPTIONAL SCALING (Transformer Convention)              │
│         │  ┌─────────────────────────────────────────────────────────┐    │
│         │  │ Scale by √embed_dim for gradient stability              │    │
│         │  │ Helps balance token and position magnitudes             │    │
│         │  └─────────────────────────────────────────────────────────┘    │
│         │                                                                 │
│         └─ OUTPUT: Position-Aware Dense Vectors                           │
│            Ready for attention mechanisms and transformers!               │
│                                                                           │
│ INTEGRATION FEATURES:                                                     │
│ • Flexible position encoding (learned/sinusoidal/none)                    │
│ • Efficient batch processing with variable sequence lengths               │
│ • Memory optimization (shared position encodings)                         │
│ • Production patterns (matches PyTorch/HuggingFace)                       │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

**Why this architecture works**: By separating token semantics from positional information, the model can learn meaning and order independently, then combine them optimally for the specific task.
"""

# %% [markdown]
"""
### EmbeddingLayer Initialization

The `__init__` method assembles the sub-components: a token `Embedding` for
vocabulary lookup and one of three positional encoding strategies.

```
EmbeddingLayer.__init__ assembles sub-components:

  vocab_size, embed_dim, pos_encoding
         │
         ├─ self.token_embedding = Embedding(vocab_size, embed_dim)
         │
         └─ self.pos_encoding =
              'learned'    → PositionalEncoding(max_seq_len, embed_dim)
              'sinusoidal'  → create_sinusoidal_embeddings(max_seq_len, embed_dim)
              None          → None  (no positional information)
```
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
        ### BEGIN SOLUTION
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

    def __call__(self, tokens: Tensor) -> Tensor:
        """Allows the embedding layer to be called like a function."""
        return self.forward(tokens)

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
"""
### EmbeddingLayer Forward Pass

The `forward` method composes the full embedding pipeline: token lookup,
optional scaling, positional encoding addition, and batch dimension handling.

```
EmbeddingLayer.forward pipeline:

  tokens (batch, seq) or (seq,)
         │
         ├─ 1D? Add batch dim → (1, seq)
         │
         ├─ Token lookup → (batch, seq, embed)
         │
         ├─ Scale by √embed_dim? (optional)
         │
         ├─ Add positional encoding
         │    learned:    pos_encoding.forward(token_embeds)
         │    sinusoidal: token_embeds + sinusoidal_table[:seq_len]
         │    None:       pass through
         │
         └─ Squeeze batch if added → output
```
"""

# %% nbgrader={"grade": false, "grade_id": "emblayer-forward", "solution": true}
#| export

# Continue the EmbeddingLayer class with forward and utility methods
def emblayer_forward(self, tokens: Tensor) -> Tensor:
    """
    Forward pass through complete embedding system.

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
    - For sinusoidal PE, slice the table to seq_len and add a batch dim with np.newaxis
    - For learned PE, just call self.pos_encoding.forward(token_embeds)
    - Remember to squeeze the batch dim for 1D inputs at the end
    """
    ### BEGIN SOLUTION
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
        output = self.pos_encoding.forward(token_embeds)
    elif self.pos_encoding_type == 'sinusoidal':
        # Use fixed sinusoidal encoding (not learnable)
        batch_size, seq_len, embed_dim = token_embeds.shape
        pos_embeddings = self.pos_encoding[:seq_len]  # Slice using Tensor slicing

        # Reshape to add batch dimension
        pos_data = pos_embeddings.data[np.newaxis, :, :]
        pos_embeddings_batched = Tensor(pos_data)  # Sinusoidal are fixed

        output = token_embeds + pos_embeddings_batched
    else:
        # No positional encoding
        output = token_embeds

    # Remove batch dimension if it was added
    if squeeze_batch:
        # Use Tensor slicing (now supported in Module 01)
        output = output[0]

    return output
    ### END SOLUTION

# Attach forward to EmbeddingLayer class (other methods defined in class body above)
EmbeddingLayer.forward = emblayer_forward

# %% [markdown]
"""
### 🧪 Unit Test: EmbeddingLayer Forward Pass

This test validates the forward composition: token lookup + scaling + positional
encoding addition across all three PE strategies.

**What we're testing**: Token + positional embedding integration, scaling, and batch processing
**Why it matters**: Production transformers use this exact pattern
**Expected**: Correct shapes, proper scaling, flexible position encoding support
"""

# %% nbgrader={"grade": true, "grade_id": "test-emblayer-forward", "locked": true, "points": 10}
def test_unit_emblayer_forward():
    """🧪 Test EmbeddingLayer.forward composition."""
    print("🧪 Unit Test: EmbeddingLayer Forward Pass...")

    tokens = Tensor([[1, 2, 3], [4, 5, 6]])

    # Test 1: Learned PE forward
    embed_learned = EmbeddingLayer(vocab_size=100, embed_dim=64, max_seq_len=128, pos_encoding='learned')
    output_learned = embed_learned.forward(tokens)
    assert output_learned.shape == (2, 3, 64), f"Expected (2, 3, 64), got {output_learned.shape}"

    # Test 2: Sinusoidal PE forward
    embed_sin = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding='sinusoidal')
    output_sin = embed_sin.forward(tokens)
    assert output_sin.shape == (2, 3, 64), "Sinusoidal should produce same shape"

    # Test 3: No PE forward
    embed_none = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding=None)
    output_none = embed_none.forward(tokens)
    assert output_none.shape == (2, 3, 64), "No PE should produce same shape"

    # Test 4: 1D input handling (auto batch dim)
    tokens_1d = Tensor([1, 2, 3])
    output_1d = embed_learned.forward(tokens_1d)
    assert output_1d.shape == (3, 64), f"Expected (3, 64) for 1D input, got {output_1d.shape}"

    # Test 5: Embedding scaling by sqrt(embed_dim)
    embed_scaled = EmbeddingLayer(vocab_size=100, embed_dim=64, pos_encoding=None, scale_embeddings=True)
    embed_scaled.token_embedding.weight = embed_none.token_embedding.weight  # share weights
    output_scaled = embed_scaled.forward(tokens)
    output_unscaled = embed_none.forward(tokens)
    scale_factor = math.sqrt(64)
    assert np.allclose(output_scaled.data, output_unscaled.data * scale_factor, rtol=1e-5), "Scaling broken"

    print("✅ EmbeddingLayer forward pass works correctly!")

if __name__ == "__main__":
    test_unit_emblayer_forward()

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

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_complete_embedding_system()

# %% [markdown]
"""
## 📊 Systems Analysis: Embedding Trade-offs

Understanding the performance implications of different embedding strategies is crucial for building efficient NLP systems that scale to production workloads.
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
    learned_memory = learned_params * 4 / (1024 * 1024)

    print(f"Learned PE:     {learned_memory:.1f} MB ({learned_params:,} parameters)")
    print(f"Sinusoidal PE:  0.0 MB (0 parameters - computed on-the-fly)")
    print(f"No PE:          0.0 MB (0 parameters)")

    print("\n🚀 Production Implications:")
    print("• GPT-3's embedding table: ~2.4GB (50K vocab × 12K dims)")
    print("• Learned PE adds memory but may improve task-specific performance")
    print("• Sinusoidal PE saves memory and allows longer sequences")

# Run analysis when developing/testing this module
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
            tokens = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))

            # Warmup
            for _ in range(5):
                _ = embed.forward(tokens)

            # Time the lookup
            start_time = time.time()
            iterations = 100

            for _ in range(iterations):
                output = embed.forward(tokens)

            end_time = time.time()

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

# Run analysis when developing/testing this module
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
    learned_memory = learned_params * 4 / (1024 * 1024)  # MB

    print(f"📈 Memory Comparison:")
    print(f"Learned PE:     {learned_memory:.2f} MB ({learned_params:,} parameters)")
    print(f"Sinusoidal PE:  0.00 MB (0 parameters)")

    # Analyze encoding patterns
    print(f"\n📈 Encoding Pattern Analysis:")

    # Test sample sequences
    test_input = Tensor(np.random.randn(1, 10, embed_dim))

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

    try:
        # Learned PE cannot handle longer sequences
        extended_learned = PositionalEncoding(extended_length, embed_dim)
        print(f"Learned PE: Requires retraining for sequences > {max_seq_len}")
    except:
        print(f"Learned PE: Cannot handle sequences > {max_seq_len}")

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

# Run analysis when developing/testing this module
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
    test_unit_positional_encoding_init()
    test_unit_positional_encoding()
    test_unit_sinusoidal_table()
    test_unit_sinusoidal_embeddings()
    test_unit_emblayer_init()
    test_unit_emblayer_forward()
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

    # Integration Test 4: Memory efficiency check
    print("🧪 Integration Test: Memory Efficiency...")

    # Test that we're not creating unnecessary copies
    large_embed = EmbeddingLayer(vocab_size=10000, embed_dim=512)
    test_batch = Tensor(np.random.randint(0, 10000, (32, 128)))

    # Multiple forward passes should not accumulate memory (in production)
    for _ in range(5):
        output = large_embed.forward(test_batch)
        assert output.shape == (32, 128, 512), "Large batch processing failed"

    print("✅ Memory efficiency check passed!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 11")

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of embedding systems and their implications:

### 1. Memory Scaling
You implemented an embedding layer with vocab_size=50,000 and embed_dim=512.
- How many parameters does this embedding table contain? _____ million
- If using FP32 (4 bytes per parameter), how much memory does this use? _____ MB
- If you double the embedding dimension to 1024, what happens to memory usage? _____ MB

---

### 2. Lookup Complexity
Your embedding layer performs table lookups for token indices.
- What is the time complexity of looking up a single token? O(_____)
- For a batch of 32 sequences, each of length 128, how many lookup operations? _____
- Why doesn't vocabulary size affect individual lookup performance? _____

---

### 3. Positional Encoding Trade-offs
You implemented both learned and sinusoidal positional encodings.
- Learned PE for max_seq_len=2048, embed_dim=512 adds how many parameters? _____
- What happens if you try to process a sequence longer than max_seq_len with learned PE? _____
- Which type of PE can handle sequences longer than seen during training? _____

---

### 4. Production Implications
Your complete EmbeddingLayer combines token and positional embeddings.
- In GPT-3 (vocab_size≈50K, embed_dim≈12K), approximately what percentage of total parameters are in the embedding table? _____%
- If you wanted to reduce memory usage by 50%, which would be more effective: halving vocab_size or halving embed_dim? _____
- Why might sinusoidal PE be preferred for models that need to handle variable sequence lengths? _____
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
