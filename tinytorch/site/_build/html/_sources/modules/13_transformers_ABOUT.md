# Module 13: Transformers

:::{admonition} Module Info
:class: note

**ARCHITECTURE TIER** | Difficulty: ●●●● | Time: 8-10 hours | Prerequisites: 01-08, 10-12

**Prerequisites: Modules 01-08 and 10-12** means you need a strong foundation across three domains. This module assumes you've implemented tensors, layers, training loops, tokenization, embeddings, and attention mechanisms. If you can explain how multi-head attention processes queries, keys, and values to compute weighted representations, you're ready.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F13_transformers%2F13_transformers.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/13_transformers/13_transformers.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/13_transformers.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

The Transformer architecture revolutionized machine learning and powers every major language model you interact with today: GPT, Claude, LLaMA, and countless others. At its core, transformers combine self-attention mechanisms with feed-forward networks using residual connections and layer normalization to process sequences of any length. This module brings together everything you've built to create a complete autoregressive language model capable of generating coherent text.

Unlike recurrent networks that process tokens sequentially, transformers process all tokens in parallel while maintaining relationships through attention. This enables both faster training and superior modeling of long-range dependencies. By stacking multiple transformer blocks, the architecture creates increasingly abstract representations of language, from surface patterns to semantic meaning.

You'll implement the complete GPT architecture, from token embeddings through multiple transformer blocks to the final output projection. This is not just an academic exercise: the patterns you implement here are identical to those running in production systems processing billions of tokens daily.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** layer normalization to stabilize training across deep networks with learnable scale and shift parameters
- **Design** complete transformer blocks combining self-attention, feed-forward networks, and residual connections using pre-norm architecture
- **Build** a full GPT model with token embeddings, positional encoding, stacked transformer blocks, and autoregressive generation
- **Analyze** parameter scaling and memory requirements, understanding why attention memory grows quadratically with sequence length
- **Master** causal masking to enable autoregressive generation while preventing information leakage from future tokens
```

## What You'll Build

```{mermaid}
:align: center
:caption: Complete GPT Architecture
flowchart TB
    subgraph "Complete GPT Architecture"
        A["Token IDs<br/>[15496, 1917]"]
        B["Embeddings<br/>Token + Position"]
        C["Transformer Block 1<br/>Attention + MLP"]
        D["Transformer Block 2<br/>Attention + MLP"]
        E["... N Blocks ..."]
        F["Final LayerNorm"]
        G["Language Head<br/>Vocabulary Logits"]
    end

    A --> B --> C --> D --> E --> F --> G

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#d4edda
    style D fill:#d4edda
    style F fill:#f8d7da
    style G fill:#e2d5f1
```

**Implementation roadmap:**

| Step | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `LayerNorm` with learnable gamma/beta | Stabilizes training by normalizing activations |
| 2 | `MLP` with 4x expansion and GELU | Provides non-linear transformation capacity |
| 3 | `TransformerBlock` with pre-norm architecture | Combines attention and MLP with residual connections |
| 4 | `GPT` model with embeddings and blocks | Complete autoregressive language model |
| 5 | Autoregressive generation with temperature | Text generation with controllable randomness |

**The pattern you'll enable:**
```python
# Building and using a complete language model
model = GPT(vocab_size=50000, embed_dim=768, num_layers=12, num_heads=12)
logits = model.forward(tokens)  # Process input sequence
generated = model.generate(prompt, max_new_tokens=50)  # Generate text
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- KV caching for efficient generation (production systems cache keys/values to avoid recomputation)
- FlashAttention or other memory-efficient attention (PyTorch uses specialized CUDA kernels)
- Mixture of Experts or sparse transformers (advanced scaling techniques)
- Multi-query or grouped-query attention (used in modern LLMs for efficiency)

**You are building the canonical transformer architecture.** Optimizations come later.

## API Reference

This section documents the transformer components you'll implement. Each class builds on the previous, culminating in a complete language model.

### Helper Functions

#### create_causal_mask

```python
create_causal_mask(seq_len: int) -> Tensor
```

Creates a causal (autoregressive) attention mask that prevents positions from attending to future positions. Returns a lower triangular matrix where position `i` can only attend to positions `j ≤ i`.

**Returns**: Tensor of shape `(1, seq_len, seq_len)` with 1.0 for allowed positions, 0.0 for masked positions.

### LayerNorm

```python
LayerNorm(normalized_shape: int, eps: float = 1e-5) -> LayerNorm
```

Normalizes activations across features for each sample independently. Essential for stable training of deep transformer networks.

**Core Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `forward(x: Tensor) -> Tensor` | Normalize across last dimension with learnable scale/shift |
| `parameters` | `parameters() -> List[Tensor]` | Returns `[gamma, beta]` learnable parameters |

### MLP (Multi-Layer Perceptron)

```python
MLP(embed_dim: int, hidden_dim: int = None, dropout_prob: float = 0.1) -> MLP
```

Feed-forward network with 4x expansion, GELU activation, and projection back to original dimension.

**Core Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `forward(x: Tensor) -> Tensor` | Apply Linear → GELU → Linear transformation |
| `parameters` | `parameters() -> List[Tensor]` | Returns weights and biases from both layers |

### TransformerBlock

```python
TransformerBlock(embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout_prob: float = 0.1) -> TransformerBlock
```

Complete transformer block with self-attention, MLP, layer normalization, and residual connections using pre-norm architecture.

**Core Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `forward(x: Tensor, mask: Tensor = None) -> Tensor` | Process sequence through attention and MLP sub-layers |
| `parameters` | `parameters() -> List[Tensor]` | Returns all parameters from attention, norms, and MLP |

### GPT

```python
GPT(vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int = 1024) -> GPT
```

Complete GPT model for autoregressive language modeling with token embeddings, positional encoding, stacked transformer blocks, and generation capability. The architecture combines token and positional embeddings, processes through multiple transformer blocks with causal masking, applies final layer normalization, and projects to vocabulary logits.

**Core Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `forward(tokens: Tensor) -> Tensor` | Compute vocabulary logits for each position with causal masking |
| `generate` | `generate(prompt_tokens: Tensor, max_new_tokens: int = 50, temperature: float = 1.0) -> Tensor` | Autoregressively generate text using temperature-controlled sampling |
| `parameters` | `parameters() -> List[Tensor]` | Returns all model parameters from embeddings, blocks, and output head |
| `_create_causal_mask` | `_create_causal_mask(seq_len: int) -> Tensor` | Internal method creating upper triangular mask for autoregressive attention |

## Core Concepts

This section explores the architectural innovations that make transformers the dominant deep learning architecture. Understanding these concepts deeply will prepare you for both implementing transformers and designing novel architectures.

### Layer Normalization: The Stability Foundation

Layer normalization is the unsung hero of deep transformer training. Without it, training networks with dozens or hundreds of layers becomes nearly impossible due to internal covariate shift, where the distribution of activations shifts dramatically during training.

Unlike batch normalization which normalizes across the batch dimension, layer norm normalizes each sample independently across its features. This independence is crucial for transformers processing variable-length sequences. Consider a batch containing both short and long sequences: batch normalization would compute statistics mixing these fundamentally different inputs, while layer norm treats each position independently.

Here's the complete implementation showing how normalization stabilizes training:

```python
class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters initialized to identity transform
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)

    def forward(self, x):
        # Compute statistics across last dimension (features)
        mean = x.mean(axis=-1, keepdims=True)
        diff = x - mean
        variance = (diff * diff).mean(axis=-1, keepdims=True)

        # Normalize to zero mean, unit variance
        std = Tensor(np.sqrt(variance.data + self.eps))
        normalized = (x - mean) / std

        # Apply learnable transformation
        return normalized * self.gamma + self.beta
```

The mathematical formula is deceptively simple: `output = (x - μ) / σ * γ + β`. But this simplicity enables profound effects. By forcing activations to have consistent statistics, layer norm prevents the vanishing and exploding gradient problems that plague deep networks. The learnable `gamma` and `beta` parameters let the model recover any distribution it needs, so normalization does not restrict expressiveness.

The `eps = 1e-5` term prevents division by zero when computing standard deviation. In sequences where all features have identical values (rare but possible), variance approaches zero, and without epsilon, you would divide by zero. This tiny constant ensures numerical stability without affecting normal operation.

### Pre-Norm Architecture and Residual Connections

Modern transformers use pre-norm architecture where layer normalization comes before the sub-layer, not after. This seemingly minor change dramatically improves trainability of deep networks. The pattern is: normalize, transform, add residual. This creates clean normalized inputs to each operation while preserving gradient flow through residual connections.

Residual connections are the gradient highways that make deep learning possible. When you add the input directly to the output (`x + f(x)`), gradients during backpropagation have two paths: through the transformation `f` and directly through the residual connection. This direct path ensures gradients reach early layers even in 100-layer networks.

Here's how the transformer block implements pre-norm with residuals:

```python
def forward(self, x, mask=None):
    # First sub-layer: attention with pre-norm
    normed1 = self.ln1.forward(x)
    attention_out = self.attention.forward(normed1, mask)
    x = x + attention_out  # Residual connection

    # Second sub-layer: MLP with pre-norm
    normed2 = self.ln2.forward(x)
    mlp_out = self.mlp.forward(normed2)
    output = x + mlp_out  # Residual connection

    return output
```

Notice the pattern: each sub-layer receives normalized input but adds its contribution to the unnormalized residual stream. This separation of concerns creates remarkable stability. The normalized path provides consistent inputs for learning, while the residual path preserves information flow.

### The MLP: Computational Capacity Through Expansion

The multi-layer perceptron provides the non-linear transformation capacity in each transformer block. While attention handles relationships between tokens, the MLP processes each position independently, adding computational depth. The standard pattern expands to 4x the embedding dimension, applies GELU activation, then contracts back.

Why 4x expansion? This creates an information bottleneck that forces the model to learn useful transformations. The expansion phase creates a high-dimensional space where features can be separated and transformed, while the contraction phase forces compression of useful information back to the original dimension.

```python
class MLP:
    def __init__(self, embed_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Standard 4x expansion

        self.linear1 = Linear(embed_dim, hidden_dim)
        self.gelu = GELU()
        self.linear2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        hidden = self.linear1.forward(x)
        hidden = self.gelu.forward(hidden)
        output = self.linear2.forward(hidden)
        return output
```

GELU (Gaussian Error Linear Unit) activation replaced ReLU in transformer models because it provides smoother gradients. Where ReLU has a hard cutoff at zero, GELU smoothly gates values based on their magnitude, creating better training dynamics for language modeling.

The parameter count in the MLP is substantial. For `embed_dim = 512`, the first layer has `512 × 2048 + 2048 ≈ 1.05M` parameters, and the second has `2048 × 512 + 512 ≈ 1.05M`, totaling 2.1M parameters per block. In a 12-layer model, MLPs alone contribute 25M parameters.

### Causal Masking for Autoregressive Generation

GPT is an autoregressive model: it predicts each token based only on previous tokens. During training, the model sees the entire sequence, but causal masking ensures position `i` cannot attend to positions `j > i`. This prevents information leakage from the future.

The causal mask is an upper triangular matrix filled with negative infinity:

```python
def create_causal_mask(seq_len: int) -> Tensor:
    # Lower triangle = 1 (can attend), upper triangle = 0 (cannot attend)
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return Tensor(mask[np.newaxis, :, :])
```

For a 4-token sequence, this creates:
```
[[1, 0, 0, 0],   # Position 0 only sees itself
 [1, 1, 0, 0],   # Position 1 sees 0, 1
 [1, 1, 1, 0],   # Position 2 sees 0, 1, 2
 [1, 1, 1, 1]]   # Position 3 sees everything
```

In the attention mechanism, these zeros become `-inf` in the logits before softmax. After softmax, `-inf` becomes exactly 0 probability, completely preventing attention to future positions. This elegant mechanism enables parallel training on entire sequences while maintaining autoregressive constraints.

### Complete Transformer Block Architecture

The transformer block is where all components unite into a coherent processing unit. Each block transforms the input sequence through two sub-layers: multi-head self-attention and MLP, each wrapped with layer normalization and residual connections.

```python
class TransformerBlock:
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)  # Before attention
        self.ln2 = LayerNorm(embed_dim)  # Before MLP
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim)

    def forward(self, x, mask=None):
        # First sub-layer: attention with residual
        normed1 = self.ln1.forward(x)
        attention_out = self.attention.forward(normed1, mask)
        x = x + attention_out  # Residual connection

        # Second sub-layer: MLP with residual
        normed2 = self.ln2.forward(x)
        mlp_out = self.mlp.forward(normed2)
        output = x + mlp_out  # Residual connection

        return output
```

The data flow creates a residual stream that accumulates information. Input embeddings enter the first block and flow through attention (adding relationship information) and MLP (adding transformation), then continue to the next block. By the final block, the residual stream contains the original embeddings plus contributions from every attention and MLP sub-layer in the stack.

This residual stream perspective explains why transformers can be trained to hundreds of layers. Each layer makes a small additive contribution rather than completely transforming the representation. Gradients flow backward through these contributions, reaching early layers with minimal degradation.

### Parameter Scaling and Memory Requirements

Understanding parameter distribution and memory requirements is essential for designing and deploying transformers. Parameters scale roughly quadratically with embedding dimension, while attention memory scales quadratically with sequence length. These scaling laws determine the feasibility of training and deploying transformer models.

For a single transformer block with `embed_dim = 512` and `num_heads = 8`:

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Multi-Head Attention | ~1.5M | 4 × (512 × 512) for Q, K, V, O projections |
| Layer Norm 1 | 1K | 2 × 512 for gamma, beta |
| MLP | ~2.1M | (512 × 2048 + 2048) + (2048 × 512 + 512) |
| Layer Norm 2 | 1K | 2 × 512 for gamma, beta |
| **Total per block** | **~3.6M** | Dominated by MLP and attention |

For a complete GPT model, add embeddings and output projection:

```
Embeddings: vocab_size × embed_dim (e.g., 50000 × 512 = 25.6M)
Position Embeddings: max_seq_len × embed_dim (e.g., 2048 × 512 = 1M)
Transformer Blocks: num_layers × 3.6M (e.g., 12 × 3.6M = 43.2M)
Output Projection: embed_dim × vocab_size (often tied to embeddings)

Total: ~70M parameters for this configuration
```

Memory requirements have three components:

1. **Parameter Memory**: Linear with model size, stored once
2. **Activation Memory**: Needed for backpropagation, grows with batch size and sequence length
3. **Attention Memory**: Quadratic with sequence length, the primary bottleneck

The attention memory wall explains why extending context length is expensive. For a batch of 4 sequences, 8 attention heads, and varying sequence lengths:

| Sequence Length | Attention Matrix Size | Memory (MB) |
|-----------------|----------------------|-------------|
| 512 | 4 × 8 × 512 × 512 | 33.6 |
| 1024 | 4 × 8 × 1024 × 1024 | 134.2 |
| 2048 | 4 × 8 × 2048 × 2048 | 536.9 |
| 4096 | 4 × 8 × 4096 × 4096 | 2147.5 |

Doubling sequence length quadruples attention memory. This quadratic scaling drove innovations like sparse attention, linear attention, and FlashAttention that make long context tractable.

## Production Context

### Your Implementation vs. PyTorch

Your transformer implementation and PyTorch's production transformers share the same architectural principles. The differences lie in optimization: PyTorch uses fused CUDA kernels, memory-efficient attention, and various tricks for speed and scale.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Architecture** | Pre-norm transformer blocks | Pre-norm (modern) or post-norm (legacy) |
| **Attention** | Standard scaled dot-product | FlashAttention, sparse attention |
| **Memory** | Full attention matrices | KV caching, memory-efficient attention |
| **Precision** | Float32 | Mixed precision (FP16/BF16) |
| **Parallelism** | Single device | Model parallel, pipeline parallel |
| **Efficiency** | Educational clarity | Production optimization |

### Code Comparison

The following comparison shows equivalent transformer usage in TinyTorch and PyTorch. The API patterns are nearly identical because your implementation follows production design principles.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.core.transformer import TransformerBlock, GPT

# Create transformer block
block = TransformerBlock(embed_dim=512, num_heads=8)
output = block.forward(x)

# Create complete GPT model
model = GPT(vocab_size=50000, embed_dim=768, num_layers=12, num_heads=12)
logits = model.forward(tokens)
generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
```
````

````{tab-item} ⚡ PyTorch
```python
import torch.nn as nn

# PyTorch transformer block (using nn.TransformerEncoderLayer)
block = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
output = block(x)

# Complete model (using HuggingFace transformers)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
outputs = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
```
````
`````

Let's walk through the key similarities and differences:

- **Line 1-2 (Block creation)**: Both create transformer blocks with identical parameters. PyTorch uses `TransformerEncoderLayer` while you built `TransformerBlock` from scratch.
- **Line 3 (Forward pass)**: Both process sequences with identical semantics. Your implementation explicitly shows attention and MLP; PyTorch's is identical internally.
- **Line 5-6 (Model creation)**: Both create complete language models. PyTorch typically uses pre-trained models via HuggingFace; you build from scratch.
- **Line 7 (Generation)**: Both support autoregressive generation with temperature control. PyTorch adds beam search, top-k/top-p sampling, and other advanced techniques.

```{tip} What's Identical

The core architecture, pre-norm pattern, residual connections, and causal masking are identical. When you debug transformer models in PyTorch, you'll understand exactly what's happening because you built it yourself.
```

### Why Transformers Matter at Scale

To appreciate transformer impact, consider the scale of modern deployments:

- **GPT-3 (175B parameters)**: Requires 350GB just to store weights, 700GB for mixed-precision training
- **Training cost**: GPT-3 training cost approximately $4.6M in compute, using 10,000 GPUs for weeks
- **Inference latency**: Processing 2048 tokens through a 175B model takes 100-200ms on optimized hardware
- **Context scaling**: Extending from 2K to 32K context requires 256× more attention memory per layer

These numbers explain why transformer optimization is a multi-billion dollar industry. Techniques like FlashAttention (reducing attention memory from O(n²) to O(n)), model parallelism (splitting models across GPUs), and quantization (reducing precision to 8-bit or 4-bit) are essential for making transformers practical at scale.

## Check Your Understanding

Test your understanding of transformer architecture and scaling with these systems thinking questions.

**Q1: Attention Memory Calculation**

A transformer with `batch_size=8`, `num_heads=16`, `seq_len=2048` computes attention matrices at each layer. How much memory does one layer's attention matrices consume? How does this scale if you double the sequence length to 4096?

```{admonition} Answer
:class: dropdown

Attention matrix size: `batch_size × num_heads × seq_len × seq_len`
= `8 × 16 × 2048 × 2048 = 536,870,912 elements`

Memory: `536,870,912 × 4 bytes (float32) = 2,147,483,648 bytes ≈ 2.15 GB`

Doubling sequence length to 4096:
= `8 × 16 × 4096 × 4096 = 2,147,483,648 elements ≈ 8.6 GB`

**Scaling**: Doubling sequence length quadruples memory (4× increase). This quadratic scaling is why long context is expensive and drove innovations like sparse attention.
```

**Q2: Parameter Distribution Analysis**

For a GPT model with `vocab_size=50000`, `embed_dim=768`, `num_layers=12`, `num_heads=12`, calculate approximate total parameters. Which component dominates the parameter count: embeddings or transformer blocks?

```{admonition} Answer
:class: dropdown

**Token Embeddings**: `50000 × 768 = 38.4M`

**Position Embeddings**: `2048 × 768 = 1.6M` (assuming max_seq_len=2048)

**Transformer Blocks**: Each block has approximately 3.6M parameters with embed_dim=768
- Attention: `4 × (768 × 768) ≈ 2.4M`
- MLP: `(768 × 3072 + 3072) + (3072 × 768 + 768) ≈ 4.7M`
- Layer norms: negligible
- **Per block**: approximately 7.1M
- **Total blocks**: `12 × 7.1M ≈ 85M`

**Output Projection**: Usually tied to embeddings (0 additional)

**Total**: `38.4M + 1.6M + 85M ≈ 125M parameters`

**Dominant component**: Transformer blocks (85M) > Embeddings (40M). As models scale, transformer blocks dominate because they scale with `embed_dim²` while embeddings scale linearly.
```

**Q3: Residual Connection Benefits**

Why do transformers use residual connections (`x + f(x)`) rather than just `f(x)`? How do residual connections affect gradient flow during backpropagation in a 24-layer transformer?

```{admonition} Answer
:class: dropdown

**Without residual connections** (`y = f(x)`):
- Gradients must flow through all transformation layers
- Each layer multiplication can shrink gradients (vanishing) or amplify them (exploding)
- In 24 layers, gradients might become effectively zero or infinity

**With residual connections** (`y = x + f(x)`):
- During backpropagation: `∂y/∂x = 1 + ∂f/∂x`
- The "+1" term provides a direct gradient path
- Even if `∂f/∂x` is small, gradients still flow through the "+1" path
- This creates "gradient highways" through the network

**24-layer impact**: Without residuals, gradient might decay by factor of 0.9²⁴ ≈ 0.08. With residuals, the "+1" path ensures gradients reach early layers at full strength. This is why transformers can scale to 100+ layers while vanilla networks struggle beyond 10.
```

**Q4: Autoregressive Generation Efficiency**

Your `generate()` method processes the entire sequence for each new token. For generating 100 tokens with prompt length 50, how many total forward passes occur? Why is this inefficient?

```{admonition} Answer
:class: dropdown

**Current implementation**: For each of 100 new tokens, reprocess the entire sequence
- Token 1: Process 50 tokens (prompt)
- Token 2: Process 51 tokens (prompt + 1)
- Token 3: Process 52 tokens
- ...
- Token 100: Process 149 tokens

**Total forward passes**: `50 + 51 + 52 + ... + 149 = Σ(50 to 149) = 9,950 token processings`

**Why inefficient**: Attention recomputes key/value projections for all previous tokens every step, even though they don't change. For position 50, we recompute the same key/value vectors 100 times.

**KV Caching optimization**: Store computed key/value projections for previous tokens
- Each new token only computes its own key/value
- Attention uses cached keys/values from previous tokens
- Total computation: `50 (initial) + 100 (new tokens) = 150 token processings`

**Speedup**: `9,950 / 150 ≈ 66× faster` for this example. The speedup increases with generation length, making KV caching essential for production systems.
```

**Q5: Layer Normalization vs Batch Normalization**

Why do transformers use layer normalization instead of batch normalization? Consider a batch with sequences of varying lengths: [10 tokens, 50 tokens, 100 tokens].

```{admonition} Answer
:class: dropdown

**Batch Normalization** normalizes across the batch dimension:
- For position 5, would compute statistics mixing all three sequences
- But sequence 1 has no position 50, sequence 2 has no position 100
- With padding, statistics contaminated by pad tokens
- Depends on batch composition; different batches → different statistics

**Layer Normalization** normalizes across features for each sample:
- Each position normalized independently: `(x - mean(x)) / std(x)`
- Position 5 of sequence 1 does not affect position 50 of sequence 2
- No dependency on batch composition
- Works naturally with variable-length sequences

**Example**: For a tensor `(batch=3, seq=10, features=768)`:
- Batch norm: Compute 10 × 768 statistics across batch dimension (problematic)
- Layer norm: Compute 3 × 10 statistics across feature dimension (independent)

**Why it matters**: Transformers process variable-length sequences. Layer norm treats each position independently, making it robust to sequence length variation and batch composition.
```

## Further Reading

For students who want to understand the theoretical foundations and explore advanced transformer architectures:

### Seminal Papers

- **Attention Is All You Need** - Vaswani et al. (2017). The paper that introduced the transformer architecture, revolutionizing sequence modeling. Describes multi-head attention, positional encoding, and the encoder-decoder structure. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- **Language Models are Few-Shot Learners (GPT-3)** - Brown et al. (2020). Demonstrates scaling laws and emergent capabilities of large language models. Shows how transformer performance improves predictably with scale. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

- **FlashAttention: Fast and Memory-Efficient Exact Attention** - Dao et al. (2022). Reduces attention memory from O(n²) to O(n) through IO-aware algorithms, enabling long context processing. Essential for understanding modern attention optimization. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

- **On Layer Normalization in the Transformer Architecture** - Xiong et al. (2020). Analyzes pre-norm vs post-norm architectures and why pre-norm enables training deeper transformers. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)

### Additional Resources

- **Blog post**: "The Illustrated Transformer" by Jay Alammar - Visual walkthrough of transformer architecture with clear diagrams
- **Paper**: "Scaling Laws for Neural Language Models" - Kaplan et al. (2020) - Mathematical analysis of how performance scales with parameters, data, and compute
- **Implementation**: HuggingFace Transformers library - Production transformer implementations to compare with your code

## What's Next

```{seealso} Coming Up: Module 14 - Profiling

Profile your transformer to identify performance bottlenecks. You'll learn to measure forward pass time, memory allocation, and computation distribution across layers, preparing for optimization in later modules.
```

**Preview - How Transformers Get Used in Future Modules:**

| Module | What It Does | Your Transformer In Action |
|--------|--------------|---------------------------|
| **14: Profiling** | Measure performance bottlenecks | `profiler.analyze(model.forward(x))` identifies slow layers |
| **15: Quantization** | Reduce precision to 8-bit | `quantize_model(gpt)` compresses 175B → 44B parameters |
| **20: Capstone** | Complete production system | Deploy transformer for real-time inference |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/13_transformers/13_transformers.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/13_transformers/13_transformers.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
