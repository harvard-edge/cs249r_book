---
title: "Embeddings - Token to Vector Representations"
description: "Build embedding layers that convert discrete tokens to dense, learnable vector representations powering modern NLP"
difficulty: "⭐⭐"
time_estimate: "4-5 hours"
prerequisites: ["Tensor", "Tokenization"]
next_steps: ["Attention"]
learning_objectives:
  - "Implement embedding layers with efficient lookup table operations and proper initialization"
  - "Design both learned and sinusoidal positional encodings to capture sequence order information"
  - "Understand memory scaling relationships with vocabulary size and embedding dimensions"
  - "Optimize embedding lookups for cache efficiency and sparse gradient updates"
  - "Apply dimensionality principles to semantic vector space design and trade-offs"
---

# 11. Embeddings - Token to Vector Representations

**ARCHITECTURE TIER** | Difficulty: ⭐⭐ (2/4) | Time: 4-5 hours

## Overview

Build the embedding systems that transform discrete token IDs into dense, learnable vector representations - the bridge between symbolic text and neural computation. This module implements lookup tables, positional encodings, and the optimization techniques that power every modern language model from word2vec to GPT-4's input layers.

You'll discover why embeddings aren't just "lookup tables" but sophisticated parameter spaces where semantic meaning emerges through training. By implementing both token embeddings and positional encodings from scratch, you'll understand the architectural choices that shape how transformers process language and why certain design decisions (sinusoidal vs learned positions, embedding dimensions, initialization strategies) have profound implications for model capacity, memory usage, and inference performance.

## Learning Objectives

By the end of this module, you will be able to:

- **Implement embedding layers**: Build efficient lookup tables for token-to-vector conversion with proper Xavier initialization and gradient flow
- **Design positional encodings**: Create both sinusoidal (Transformer-style) and learned (GPT-style) position representations with different extrapolation capabilities
- **Understand memory scaling**: Analyze how vocabulary size and embedding dimensions impact parameter count, memory bandwidth, and serving costs
- **Optimize embedding lookups**: Implement sparse gradient updates that avoid computing gradients for 99% of vocabulary during training
- **Apply dimensionality principles**: Balance semantic expressiveness with computational efficiency in vector space design and initialization

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement embedding lookup tables with trainable parameters, sinusoidal positional encodings using mathematical patterns, learned position embeddings, and complete token+position combination systems
2. **Use**: Convert tokenized text sequences to dense vectors, add positional information for sequence order awareness, and prepare embeddings for attention mechanisms
3. **Reflect**: Analyze memory scaling with vocabulary size (why GPT-3's embeddings use 2.4GB), understand sparse gradient efficiency for large vocabularies, and explore semantic geometry in learned embedding spaces

```{admonition} Systems Reality Check
:class: tip

**Production Context**: GPT-3's embedding table contains 50,257 vocabulary × 12,288 dimensions = 617M parameters (about 20% of the model's 175B total). Every token lookup requires reading 48KB of memory - making embedding access a major bandwidth bottleneck during inference, especially for long sequences.

**Performance Note**: During training, only ~1% of vocabulary appears in each batch. Sparse gradient updates avoid computing gradients for the other 99% of embedding parameters, saving massive computation and memory bandwidth. This is why frameworks like PyTorch implement specialized sparse gradient operations for embeddings.
```

## Implementation Guide

### Embedding Layer - The Token Lookup Table

The fundamental building block that maps discrete token IDs to continuous dense vectors. This is where semantic meaning will eventually be learned through training.

**Core Implementation Pattern:**

```python
class Embedding:
    """Learnable embedding layer for token-to-vector conversion.

    Implements efficient lookup table that maps token IDs to dense vectors.
    The foundation of all language models and sequence processing.

    Args:
        vocab_size: Size of vocabulary (e.g., 50,000 for GPT-2)
        embedding_dim: Dimension of dense vectors (e.g., 768 for BERT-base)

    Memory Cost: vocab_size × embedding_dim parameters
    Example: 50K vocab × 768 dim = 38.4M parameters (153MB at FP32)
    """
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Xavier/Glorot initialization for stable gradients
        limit = math.sqrt(6.0 / (vocab_size + embedding_dim))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (vocab_size, embedding_dim)),
            requires_grad=True
        )

    def forward(self, indices):
        """Look up embeddings for token IDs.

        Args:
            indices: (batch_size, seq_len) tensor of token IDs

        Returns:
            embeddings: (batch_size, seq_len, embedding_dim) dense vectors
        """
        # Advanced indexing: O(1) per token lookup
        embedded = self.weight.data[indices.data.astype(int)]

        result = Tensor(embedded, requires_grad=self.weight.requires_grad)

        # Attach gradient computation (sparse updates during backward)
        if self.weight.requires_grad:
            result._grad_fn = EmbeddingBackward(self.weight, indices)

        return result
```

**Why This Design Works:**
- **Xavier initialization** ensures gradients don't explode or vanish during early training
- **Advanced indexing** provides O(1) lookup complexity regardless of vocabulary size
- **Sparse gradients** mean only embeddings for tokens in the current batch receive updates
- **Trainable weights** allow the model to learn semantic relationships through backpropagation

### Sinusoidal Positional Encoding (Transformer-Style)

Fixed mathematical encodings that capture position without learned parameters. The original "Attention is All You Need" approach that enables extrapolation to longer sequences.

**Mathematical Foundation:**

```python
def create_sinusoidal_embeddings(max_seq_len, embedding_dim):
    """Create sinusoidal positional encodings from Vaswani et al. (2017).

    Uses sine/cosine functions of different frequencies to encode position.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))  # Even indices
        PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))  # Odd indices

    Where:
        pos = position in sequence (0, 1, 2, ...)
        i = dimension pair index
        10000 = base frequency (creates wavelengths from 2π to 10000·2π)

    Advantages:
        - Zero parameters (no memory overhead)
        - Generalizes to sequences longer than training
        - Smooth transitions (nearby positions similar)
        - Rich frequency spectrum across dimensions
    """
    # Position indices: [0, 1, 2, ..., max_seq_len-1]
    position = np.arange(max_seq_len, dtype=np.float32)[:, np.newaxis]

    # Frequency term for each dimension pair
    div_term = np.exp(
        np.arange(0, embedding_dim, 2, dtype=np.float32) *
        -(math.log(10000.0) / embedding_dim)
    )

    # Initialize positional encoding matrix
    pe = np.zeros((max_seq_len, embedding_dim), dtype=np.float32)

    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...)
    pe[:, 1::2] = np.cos(position * div_term)

    return Tensor(pe)
```

**Why Sinusoidal Patterns Work:**
- **Different frequencies** per dimension: high frequencies change rapidly between positions, low frequencies change slowly
- **Unique signatures** for each position through combination of frequencies
- **Linear combinations** allow the model to learn relative position offsets through attention
- **No length limit** - can compute encodings for any sequence length at inference time

### Learned Positional Encoding (GPT-Style)

Trainable position embeddings that can adapt to task-specific patterns. Used in GPT models and other architectures where positional patterns may be learnable.

**Implementation Pattern:**

```python
class PositionalEncoding:
    """Learnable positional encoding layer.

    Trainable position-specific vectors added to token embeddings.

    Args:
        max_seq_len: Maximum sequence length to support
        embedding_dim: Dimension matching token embeddings

    Advantages:
        - Can learn task-specific position patterns
        - May capture regularities like sentence structure
        - Often performs slightly better than sinusoidal

    Disadvantages:
        - Requires additional parameters (max_seq_len × embedding_dim)
        - Cannot extrapolate beyond training sequence length
        - Needs sufficient training data to learn position patterns
    """
    def __init__(self, max_seq_len, embedding_dim):
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # Smaller initialization than token embeddings (additive combination)
        limit = math.sqrt(2.0 / embedding_dim)
        self.position_embeddings = Tensor(
            np.random.uniform(-limit, limit, (max_seq_len, embedding_dim)),
            requires_grad=True
        )

    def forward(self, x):
        """Add positional encodings to input embeddings.

        Args:
            x: (batch_size, seq_len, embedding_dim) input embeddings

        Returns:
            Position-aware embeddings of same shape
        """
        batch_size, seq_len, embedding_dim = x.shape

        # Get position embeddings for this sequence length
        pos_embeddings = self.position_embeddings.data[:seq_len]

        # Broadcast to batch dimension: (1, seq_len, embedding_dim)
        pos_embeddings = pos_embeddings[np.newaxis, :, :]

        # Element-wise addition combines token and position information
        result = x + Tensor(pos_embeddings, requires_grad=True)

        return result
```

**Design Rationale:**
- **Learned parameters** can capture task-specific patterns (e.g., sentence beginnings, clause boundaries)
- **Smaller initialization** because positions add to token embeddings (not replace them)
- **Fixed max length** is a limitation but acceptable for many production use cases
- **Element-wise addition** preserves both token semantics and position information

### Complete Embedding System

Production-ready integration of token and positional embeddings used in real transformer implementations.

**Full Pipeline:**

```python
class EmbeddingLayer:
    """Complete embedding system combining token and positional embeddings.

    Production component matching PyTorch/HuggingFace transformer patterns.
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len=512,
                 pos_encoding='learned', scale_embeddings=False):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.scale_embeddings = scale_embeddings

        # Token embedding table
        self.token_embedding = Embedding(vocab_size, embed_dim)

        # Positional encoding strategy
        if pos_encoding == 'learned':
            self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        elif pos_encoding == 'sinusoidal':
            self.pos_encoding = create_sinusoidal_embeddings(max_seq_len, embed_dim)
        elif pos_encoding is None:
            self.pos_encoding = None

    def forward(self, tokens):
        """Convert tokens to position-aware embeddings.

        Args:
            tokens: (batch_size, seq_len) token indices

        Returns:
            (batch_size, seq_len, embed_dim) position-aware vectors
        """
        # Token lookup
        token_embeds = self.token_embedding.forward(tokens)

        # Optional scaling (Transformer convention: √embed_dim)
        if self.scale_embeddings:
            token_embeds = Tensor(token_embeds.data * math.sqrt(self.embed_dim))

        # Add positional information
        if self.pos_encoding is not None:
            output = self.pos_encoding.forward(token_embeds)
        else:
            output = token_embeds

        return output
```

**Integration Benefits:**
- **Flexible positional encoding** supports learned, sinusoidal, or none
- **Embedding scaling** (multiply by √d) is Transformer convention for gradient stability
- **Batch processing** handles variable sequence lengths efficiently
- **Parameter management** tracks all trainable components for optimization

## Getting Started

### Prerequisites

Before starting this module, ensure you have completed:

- **Module 01 (Tensor)**: Provides the foundational Tensor class with gradient tracking and operations
- **Module 10 (Tokenization)**: Required for converting text to token IDs that embeddings consume

Verify your prerequisites:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules
tito test tensor
tito test tokenization
```

### Development Workflow

1. **Open the development notebook**: `modules/11_embeddings/embeddings_dev.ipynb`
2. **Implement Embedding class**: Create lookup table with Xavier initialization and efficient indexing
3. **Build sinusoidal encodings**: Compute sine/cosine position representations using mathematical formula
4. **Create learned positions**: Add trainable position embedding table with proper initialization
5. **Integrate complete system**: Combine token and position embeddings with flexible encoding strategies
6. **Export and verify**: `tito module complete 11 && tito test embeddings`

## Testing

### Comprehensive Test Suite

Run the full test suite to verify embedding functionality:

```bash
# TinyTorch CLI (recommended)
tito test embeddings

# Direct pytest execution
python -m pytest tests/ -k embeddings -v
```

### Test Coverage Areas

- ✅ **Embedding lookup correctness**: Verify token IDs map to correct vector rows in weight table
- ✅ **Gradient flow verification**: Ensure sparse gradient updates accumulate properly during backpropagation
- ✅ **Positional encoding math**: Validate sinusoidal formula implementation with correct frequencies
- ✅ **Shape broadcasting**: Test token + position combination across batch dimensions
- ✅ **Memory efficiency profiling**: Verify parameter count and lookup performance characteristics

### Inline Testing & Validation

The module includes comprehensive unit tests during development:

```python
# Example inline test output
🔬 Unit Test: Embedding layer...
✅ Lookup table created: 10K vocab × 256 dims = 2.56M parameters
✅ Forward pass shape correct: (32, 20, 256) for batch of 32 sequences
✅ Backward pass sparse gradients accumulate correctly
✅ Xavier initialization keeps variance stable
📈 Progress: Embedding Layer ✓

🔬 Unit Test: Sinusoidal positional encoding...
✅ Encodings computed for 512 positions × 256 dimensions
✅ Sine/cosine patterns verified (pos 0: [0, 1, 0, 1, ...])
✅ Different positions have unique signatures
✅ Frequency spectrum correct (high to low across dimensions)
📈 Progress: Sinusoidal Positions ✓

🔬 Unit Test: Learned positional encoding...
✅ Trainable position embeddings initialized
✅ Addition with token embeddings preserves gradients
✅ Batch broadcasting handled correctly
📈 Progress: Learned Positions ✓

🔬 Unit Test: Complete embedding system...
✅ Token + position combination works for all strategies
✅ Embedding scaling (√d) applied correctly
✅ Variable sequence lengths handled gracefully
✅ Parameter counting correct for each configuration
📈 Progress: Complete System ✓
```

### Manual Testing Examples

Test your embedding implementation interactively:

```python
from tinytorch.core.embeddings import Embedding, PositionalEncoding, create_sinusoidal_embeddings

# Create embedding layer
vocab_size, embed_dim = 10000, 256
token_emb = Embedding(vocab_size, embed_dim)

# Test token lookup
token_ids = Tensor([[1, 5, 23], [42, 7, 19]])  # (2, 3) - batch of 2 sequences
embeddings = token_emb.forward(token_ids)      # (2, 3, 256)
print(f"Token embeddings shape: {embeddings.shape}")

# Add learned positional encodings
pos_emb = PositionalEncoding(max_seq_len=512, embed_dim=256)
token_embeddings_3d = embeddings  # Already (batch, seq, embed)
pos_aware = pos_emb.forward(token_embeddings_3d)
print(f"Position-aware shape: {pos_aware.shape}")  # (2, 3, 256)

# Try sinusoidal encodings
sin_pe = create_sinusoidal_embeddings(max_seq_len=512, embed_dim=256)
sin_positions = sin_pe.data[:3][np.newaxis, :, :]  # (1, 3, 256)
combined = Tensor(embeddings.data + sin_positions)
print(f"Sinusoidal combined: {combined.shape}")  # (2, 3, 256)

# Verify position 0 pattern (should be [0, 1, 0, 1, ...])
print(f"Position 0 pattern: {sin_pe.data[0, :8]}")
# Expected: [~0.0, ~1.0, ~0.0, ~1.0, ~0.0, ~1.0, ~0.0, ~1.0]
```

## Systems Thinking Questions

### Real-World Applications

- **Large Language Models (GPT-4, Claude, Llama)**: Embedding tables often contain 20-40% of total model parameters. GPT-3's 50K vocab × 12K dims = 617M embedding parameters alone (2.4GB at FP32). This makes embeddings a major memory consumer in serving infrastructure.

- **Recommendation Systems (YouTube, Netflix, Spotify)**: Billion-scale item embeddings for personalized content retrieval. YouTube's embedding space contains hundreds of millions of video embeddings, enabling fast nearest-neighbor search for recommendations in milliseconds.

- **Multilingual Models (Google Translate, mBERT)**: Shared embedding spaces across 100+ languages enable zero-shot cross-lingual transfer. Words with similar meanings across languages cluster together in the learned vector space, allowing translation without parallel data.

- **Search Engines (Google, Bing)**: Query and document embeddings power semantic search beyond keyword matching. BERT-style embeddings capture meaning, letting "how to fix a leaky faucet" match "plumbing repair for dripping tap" even with no shared words.

### Mathematical Foundations

- **Embedding Geometry**: Why do word embeddings exhibit linear relationships like "king - man + woman ≈ queen"? The training objective (predicting context words in word2vec, or next tokens in language models) creates geometric structure where semantic relationships become linear vector operations. This emerges without explicit supervision.

- **Dimensionality Trade-offs**: Higher dimensions increase expressiveness (more capacity to separate distinct concepts) but require more memory and computation. BERT-base uses 768 dimensions, BERT-large uses 1024 - carefully chosen based on performance-cost Pareto analysis. Doubling dimensions doubles memory but may only improve accuracy by a few percentage points.

- **Positional Encoding Mathematics**: Sinusoidal encodings use different frequencies (wavelengths from 2π to 10,000·2π) so each position gets a unique pattern. The model can learn relative positions through attention: the dot product of position encodings at offsets k captures periodic patterns the attention mechanism learns to use.

- **Sparse Gradient Efficiency**: During training with vocabulary size V and batch containing b unique tokens, dense gradients would update all V embeddings. Sparse gradients only update b embeddings - when b << V (typical: 1000 tokens vs 50K vocab), this saves ~98% of gradient computation and memory bandwidth.

### Performance Characteristics

- **Memory Scaling**: Embedding tables grow as O(vocab_size × embedding_dim). At FP32 (4 bytes per parameter): 50K vocab × 768 dims = 153MB, 100K vocab × 1024 dims = 410MB. Mixed precision (FP16) cuts this in half, but vocabulary size dominates scaling for large models.

- **Bandwidth Bottleneck**: Every token lookup reads embedding_dim × sizeof(dtype) bytes from memory. With 768 dims at FP32, that's 3KB per token. Processing a 2048-token context requires reading 6MB from the embedding table - memory bandwidth becomes the bottleneck, not compute.

- **Cache Efficiency**: Sequential token access has poor cache locality because tokens are typically non-sequential in the embedding table (token IDs [1, 42, 7, 99] means random jumps through the weight matrix). Batching improves throughput by amortizing cache misses, but embedding access remains memory-bound, not compute-bound.

- **Inference Optimization**: Embedding quantization (INT8 or even INT4) reduces memory footprint and bandwidth by 2-4×, critical for deployment. KV-caching in transformers makes embedding lookup happen only once per token (not per layer), so optimizing this cold start is important for latency-sensitive applications.

## Ready to Build?

You're about to implement the embedding systems that power modern AI language understanding. These lookup tables and positional encodings are the bridge between discrete tokens (words, subwords, characters) and the continuous vector spaces where neural networks operate. What seems like a simple "array lookup" is actually the foundation of how language models represent meaning.

What makes this module special is understanding not just *how* embeddings work, but *why* certain design choices matter. Why do we need positional encodings when embeddings already contain token information? Why sparse gradients instead of dense updates? How does embedding dimension affect model capacity versus memory footprint? These aren't just implementation details - they're fundamental design principles that shape every production language model's architecture.

By building embeddings from scratch, you'll gain intuition for memory-computation trade-offs in deep learning systems. You'll understand why GPT-3's embedding table consumes 2.4GB of memory, and why that matters for serving costs at scale (more memory = more expensive GPUs = higher operational costs). You'll see how sinusoidal encodings allow transformers to process sequences longer than training data, while learned positions might perform better on specific tasks with known maximum lengths.

This is where theory meets the economic realities of deploying AI at scale. Every architectural choice - vocabulary size, embedding dimension, positional encoding strategy - has both technical implications (accuracy, generalization) and business implications (memory costs, inference latency, serving throughput). Understanding these trade-offs is what separates machine learning researchers from machine learning systems engineers.

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/11_embeddings/embeddings_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/11_embeddings/embeddings_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/11_embeddings/embeddings_dev.ipynb
:class-header: bg-light

Browse the Jupyter notebook source and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.
```

---

<div class="prev-next-area">
<a class="left-prev" href="../modules/10_tokenization_ABOUT.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../modules/12_attention_ABOUT.html" title="next page">Next Module →</a>
</div>
