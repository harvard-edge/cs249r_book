---
title: "Transformers - Complete GPT Architecture"
description: "Build decoder-only transformer architecture for autoregressive text generation"
difficulty: "⭐⭐⭐⭐"
time_estimate: "6-8 hours"
prerequisites: ["Embeddings", "Attention"]
next_steps: ["Profiling (Optimization Tier)"]
learning_objectives:
  - "Implement complete transformer blocks with multi-head attention, feed-forward networks, layer normalization, and residual connections"
  - "Build decoder-only GPT architecture with causal masking for autoregressive text generation"
  - "Understand pre-norm architecture and residual connections for training deep networks (12+ layers)"
  - "Analyze parameter scaling, memory complexity, and attention quadratic growth with sequence length"
  - "Apply transformer architecture to language modeling tasks using patterns from PyTorch and production systems"
---

# 13. Transformers - Complete GPT Architecture

**ARCHITECTURE TIER** | Difficulty: ⭐⭐⭐⭐ (4/4) | Time: 6-8 hours

## Overview

You'll build the complete GPT transformer architecture—the decoder-only foundation powering ChatGPT, GPT-4, Claude, and virtually all modern large language models. This module combines everything you've learned about attention, embeddings, and neural networks into a production-ready autoregressive language model capable of text generation. You'll implement layer normalization, feed-forward networks, transformer blocks with residual connections, and the complete GPT model that matches PyTorch's `nn.TransformerDecoder` design.

## Learning Objectives

By the end of this module, you will be able to:

- **Implement complete transformer blocks** with multi-head self-attention, position-wise feed-forward networks (4x expansion), layer normalization, and residual connections for gradient highways enabling deep networks (12+ layers)
- **Build decoder-only GPT architecture** with causal masking preventing future token leakage, autoregressive generation with temperature sampling, and embeddings combining token and positional information
- **Understand pre-norm architecture and residual connections** critical for training stability—pre-norm placement before sub-layers (not after) enables 100+ layer networks by providing clean normalized inputs and direct gradient paths
- **Analyze parameter scaling and memory complexity** including quadratic attention memory growth O(n²) with sequence length, linear parameter scaling with layers, and techniques like gradient checkpointing for memory reduction
- **Apply transformer architecture to language modeling** using real-world patterns from PyTorch `nn.Transformer`, understanding decoder-only vs encoder-only vs encoder-decoder choices, and production optimizations like KV caching

## Build → Use → Reflect

This module follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement LayerNorm with learnable scale/shift, MLP feed-forward networks with 4x expansion and GELU activation, TransformerBlock combining attention+MLP with pre-norm residual connections, complete GPT decoder with causal masking and generation
2. **Use**: Train GPT-style decoder on character-level text generation, implement autoregressive generation with temperature sampling (conservative vs creative), analyze parameter scaling across model sizes (Tiny → GPT-3 scale), measure attention memory quadratic growth
3. **Reflect**: Why are residual connections critical for deep transformers (gradient vanishing without them)? How does pre-norm differ from post-norm (training stability for >12 layers)? What's the compute/memory trade-off in stacking layers vs widening dimensions? Why does attention memory scale quadratically with sequence length (O(n²d) cost)?

```{admonition} Systems Reality Check
:class: tip

**Production Context**: The decoder-only GPT architecture you're implementing powers virtually all modern LLMs. GPT-4 uses a 120-layer decoder stack, ChatGPT is based on GPT-3.5 with 96 layers, Claude uses decoder-only architecture, Llama 2 has 80 layers—all are transformer decoders with causal attention. This architecture dominated because it scales predictably with parameters and data.

**Performance Note**: Transformer depth has O(n²d) attention cost per layer (n=sequence length, d=model dimension). For GPT-3 with 2048 tokens, each attention layer processes 4M token pairs. Memory scales linearly with layers but quadratically with sequence length. Production systems use KV caching (reuse key-value pairs during generation), FlashAttention (memory-efficient attention), and gradient checkpointing (trade compute for memory) to manage this. Understanding these trade-offs is critical for ML systems engineering.
```

## Getting Started

### Prerequisites

Ensure you understand the foundations from previous modules:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Verify prerequisite modules
tito test embeddings
tito test attention
```

**Required Background:**
- **Module 11 (Embeddings)**: Token and positional embeddings for input representation
- **Module 12 (Attention)**: Multi-head attention mechanism for sequence modeling
- **Module 05 (Autograd)**: Automatic differentiation for training deep networks
- **Module 02 (Activations)**: GELU activation used in MLP layers

### Development Workflow

1. **Open the development file**: `modules/13_transformers/transformers.py`
2. **Implement LayerNorm**: Normalize across feature dimension with learnable scale/shift parameters (gamma, beta)
3. **Build MLP**: Two linear layers with 4x expansion ratio and GELU activation (position-wise transformation)
4. **Create TransformerBlock**: Combine attention and MLP with pre-norm residual connections (LayerNorm before sub-layers)
5. **Add GPT model**: Stack transformer blocks with token+positional embeddings, causal masking, and generation
6. **Export and verify**: `tito module complete 13 && tito test transformers`

## Implementation Guide

### LayerNorm - Training Stability for Deep Networks

Layer normalization stabilizes training by normalizing activations across the feature dimension for each sample independently. Unlike batch normalization (normalizes across batch), LayerNorm works with any batch size including batch=1 during inference—essential for variable-length sequences.

```python
class LayerNorm:
    """Layer normalization for transformer training stability.

    Normalizes across feature dimension (last axis) for each sample independently.
    Includes learnable scale (gamma) and shift (beta) parameters.

    Formula: output = gamma * (x - mean) / sqrt(variance + eps) + beta

    Why LayerNorm for Transformers:
    - Batch-independent: Works with any batch size (good for inference)
    - Variable-length sequences: Each sample normalized independently
    - Better gradients: Empirically superior to BatchNorm for NLP tasks
    """
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = Tensor(np.ones(normalized_shape))   # Learnable scale (starts at 1.0)
        self.beta = Tensor(np.zeros(normalized_shape))   # Learnable shift (starts at 0.0)
        self.eps = eps  # Numerical stability in variance calculation

    def forward(self, x):
        # Compute statistics across last dimension (features)
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)

        # Normalize: (x - μ) / σ
        normalized = (x - mean) / sqrt(variance + self.eps)

        # Apply learnable transformation: γ * norm + β
        return self.gamma * normalized + self.beta
```

**Key Design Decisions:**
- **Per-sample normalization**: Each sequence position normalized independently across features (batch-independent)
- **Learnable parameters**: Gamma/beta allow model to recover any desired distribution after normalization
- **Epsilon for stability**: Small constant (1e-5) prevents division by zero in variance calculation

**LayerNorm vs BatchNorm:**
| Aspect | LayerNorm | BatchNorm |
|--------|-----------|-----------|
| Normalizes across | Features (per sample) | Batch (per feature) |
| Batch size dependency | Independent | Dependent |
| Inference behavior | Same as training | Requires running statistics |
| Best for | Transformers, NLP | CNNs, Computer Vision |

### MLP - Position-Wise Feed-Forward Network

The MLP provides non-linear transformation capacity in each transformer block. It's a simple two-layer network with a 4x expansion pattern applied identically to each sequence position.

```python
class MLP:
    """Multi-Layer Perceptron (Feed-Forward Network) for transformer blocks.

    Standard pattern: Linear(expand) → GELU → Linear(contract)
    Expansion ratio: 4:1 (embed_dim → 4*embed_dim → embed_dim)

    This provides the "thinking" capacity after attention computes relationships.
    """
    def __init__(self, embed_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Standard 4x expansion

        self.linear1 = Linear(embed_dim, hidden_dim)    # Expansion: 512 → 2048
        self.gelu = GELU()                              # Smooth activation
        self.linear2 = Linear(hidden_dim, embed_dim)    # Contraction: 2048 → 512

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = self.linear1(x)      # Expand to hidden_dim
        x = self.gelu(x)         # Nonlinearity (smoother than ReLU)
        x = self.linear2(x)      # Contract back to embed_dim
        return x
```

**Why 4x Expansion?**
- **Parameter capacity**: More parameters = more representation power (MLP typically has more params than attention)
- **Information bottleneck**: Expansion → contraction forces model to compress useful information
- **Empirical success**: 4x ratio found to work well across model sizes (some models experiment with 2x-8x)

**GELU vs ReLU:**
- **ReLU**: Hard cutoff at zero `max(0, x)` - simple but non-smooth
- **GELU**: Smooth probabilistic activation `x * Φ(x)` where Φ is Gaussian CDF
- **Why GELU**: Smoother gradients, better performance for language modeling tasks

### TransformerBlock - Complete Layer with Attention and MLP

A single transformer layer combining multi-head self-attention with feed-forward processing using pre-norm residual architecture. This is the core building block stacked 12-120 times in production models.

```python
class TransformerBlock:
    """Complete transformer layer with self-attention, MLP, and residual connections.

    Pre-Norm Architecture (Modern Standard):
        x → LayerNorm → MultiHeadAttention → Add(x) →
            LayerNorm → MLP → Add → Output

    Each sub-layer (attention, MLP) gets normalized input but adds to residual stream.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        # Attention sub-layer components
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)  # Pre-norm: before attention

        # MLP sub-layer components
        self.mlp = MLP(embed_dim, hidden_dim=int(embed_dim * mlp_ratio))
        self.ln2 = LayerNorm(embed_dim)  # Pre-norm: before MLP

    def forward(self, x, mask=None):
        """Forward pass with residual connections.

        Args:
            x: (batch, seq_len, embed_dim) input
            mask: Optional attention mask (causal mask for GPT)

        Returns:
            output: (batch, seq_len, embed_dim) transformed sequence
        """
        # Attention sub-layer with residual
        normed = self.ln1(x)                          # Normalize input
        attended = self.attention(normed, mask)       # Self-attention
        x = x + attended                              # Residual connection

        # MLP sub-layer with residual
        normed = self.ln2(x)                          # Normalize again
        mlp_out = self.mlp(normed)                    # Feed-forward
        x = x + mlp_out                               # Residual connection

        return x
```

**Pre-Norm vs Post-Norm:**

**Pre-Norm (What We Implement):**
```
x → LayerNorm → Attention → Add(x) → output
```
- LayerNorm **before** sub-layers (attention, MLP)
- Better gradient flow for deep models (>12 layers)
- Modern standard in GPT-3, GPT-4, LLaMA, Claude

**Post-Norm (Original Transformer Paper):**
```
x → Attention → Add(x) → LayerNorm → output
```
- LayerNorm **after** sub-layers
- Used in original "Attention is All You Need" paper
- Struggles with very deep networks (gradient issues)

**Why Pre-Norm Wins:**
1. **Clean inputs**: Each sub-layer receives normalized input (stable mean/variance)
2. **Direct gradient path**: Residual connections bypass normalization during backprop
3. **Deeper networks**: Enables training 100+ layer transformers (GPT-4 has ~120 layers)

### GPT - Complete Decoder-Only Architecture

GPT (Generative Pre-trained Transformer) is the complete autoregressive language model combining embeddings, transformer blocks, and generation capability. It's **decoder-only** with causal masking preventing future token leakage.

```python
class GPT:
    """Complete GPT decoder for autoregressive language modeling.

    Architecture:
        Input tokens → Token Embedding + Positional Embedding →
        TransformerBlocks (with causal masking) →
        LayerNorm → Linear(embed_dim → vocab_size) → Logits

    Key Feature: Causal masking ensures position i only attends to positions ≤ i
    """
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len=1024):
        # Embedding layers
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.position_embedding = Embedding(max_seq_len, embed_dim)

        # Stack of transformer blocks
        self.blocks = [TransformerBlock(embed_dim, num_heads)
                      for _ in range(num_layers)]

        # Output layers
        self.ln_f = LayerNorm(embed_dim)              # Final layer norm
        self.lm_head = Linear(embed_dim, vocab_size)  # Vocab projection

    def forward(self, tokens):
        """Forward pass through GPT decoder.

        Args:
            tokens: (batch, seq_len) token indices

        Returns:
            logits: (batch, seq_len, vocab_size) unnormalized predictions
        """
        batch_size, seq_len = tokens.shape

        # Embeddings: tokens + positions
        token_emb = self.token_embedding(tokens)
        positions = Tensor(np.arange(seq_len).reshape(1, seq_len))
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb  # (batch, seq_len, embed_dim)

        # Causal mask: prevent attending to future positions
        mask = self._create_causal_mask(seq_len)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits

    def _create_causal_mask(self, seq_len):
        """Create causal mask: upper triangular matrix with -inf.

        Mask ensures position i can only attend to positions j where j ≤ i.
        After softmax, -inf becomes probability 0.
        """
        mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        return Tensor(mask)

    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        """Autoregressive text generation.

        Args:
            prompt_tokens: (batch, prompt_len) initial sequence
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            generated: (batch, prompt_len + max_new_tokens) full sequence
        """
        current = Tensor(prompt_tokens.data.copy())

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(current)

            # Get last position logits
            next_logits = logits.data[:, -1, :] / temperature

            # Sample from distribution
            probs = softmax(next_logits)
            next_token = sample(probs)

            # Append to sequence
            current = concat([current, next_token], axis=1)

        return current
```

**Causal Masking Visualization:**
```
Sequence: ["The", "cat", "sat", "on"]
Positions:   0      1      2     3

Attention Matrix (✓ = can attend, ✗ = masked):
       To:  0   1   2   3
From 0:   [ ✓   ✗   ✗   ✗ ]  ← "The" only sees itself
From 1:   [ ✓   ✓   ✗   ✗ ]  ← "cat" sees "The" + itself
From 2:   [ ✓   ✓   ✓   ✗ ]  ← "sat" sees all previous
From 3:   [ ✓   ✓   ✓   ✓ ]  ← "on" sees everything

Implementation: Upper triangular with -∞
[[  0, -∞, -∞, -∞],
 [  0,   0, -∞, -∞],
 [  0,   0,   0, -∞],
 [  0,   0,   0,   0]]

After softmax: -∞ → probability 0
```

**Temperature Sampling:**
- **Low temperature (0.1-0.5)**: Conservative, deterministic (picks highest probability)
- **Medium temperature (1.0)**: Balanced sampling from probability distribution
- **High temperature (1.5-2.0)**: Creative, random (flattens distribution)

### Decoder-Only Architecture Choice

This module implements **decoder-only GPT architecture**. Here's why this choice dominates modern LLMs:

**Decoder-Only (GPT) - What We Build:**
- **Attention**: Causal masking (position i only sees positions ≤ i)
- **Training**: Next-token prediction (autoregressive objective)
- **Use cases**: Text generation, code completion, dialogue, instruction following
- **Examples**: GPT-3/4, ChatGPT, Claude, LLaMA, PaLM, Gemini LLMs

**Encoder-Only (BERT) - Not Implemented:**
- **Attention**: Bidirectional (all positions see all positions)
- **Training**: Masked language modeling (predict masked tokens)
- **Use cases**: Classification, NER, question answering, search ranking
- **Examples**: BERT, RoBERTa (Google Search uses BERT for ranking)

**Encoder-Decoder (T5) - Not Implemented:**
- **Attention**: Encoder is bidirectional, decoder is causal
- **Training**: Sequence-to-sequence tasks
- **Use cases**: Translation, summarization
- **Examples**: T5, BART (Google Translate uses encoder-decoder)

**Why Decoder-Only Won:**
1. **Simplicity**: Single architecture type (no encoder-decoder coordination)
2. **Scalability**: Predictable scaling laws with parameters and data
3. **Versatility**: Handles both understanding and generation tasks
4. **Efficiency**: Simpler to implement and optimize than encoder-decoder

## Common Pitfalls

### Forgetting Residual Connections in Transformer Blocks

**Problem**: Implementing transformer blocks without residual connections causes gradient vanishing in deep networks (12+ layers)

**Solution**: Always add residual connections around both attention and MLP sub-layers using `x = x + sublayer(x)` pattern

```python
# Wrong: No residual connections
def forward(self, x, mask=None):
    x = self.ln1(x)
    x = self.attention(x, mask)  # Gradient vanishes after 12+ layers
    x = self.ln2(x)
    x = self.mlp(x)
    return x

# Correct: Pre-norm with residual connections
def forward(self, x, mask=None):
    x = x + self.attention(self.ln1(x), mask)  # Residual around attention
    x = x + self.mlp(self.ln2(x))              # Residual around MLP
    return x
```

### Incorrect Pre-Norm vs Post-Norm Architecture

**Problem**: Applying LayerNorm after sub-layers (post-norm) instead of before (pre-norm) causes training instability in deep transformers

**Solution**: Use pre-norm architecture where LayerNorm is applied before attention and MLP, not after

```python
# Wrong: Post-norm (original Transformer paper, unstable for deep models)
x = self.ln1(x + self.attention(x))  # LayerNorm after residual
x = self.ln2(x + self.mlp(x))

# Correct: Pre-norm (modern standard, stable for 100+ layers)
x = x + self.attention(self.ln1(x))  # LayerNorm before sub-layer
x = x + self.mlp(self.ln2(x))
```

### Missing Causal Mask in GPT Forward Pass

**Problem**: Forgetting to create or apply causal mask allows future token leakage, breaking autoregressive generation property

**Solution**: Always create upper triangular mask with -inf and pass to all transformer blocks during training and inference

```python
# Wrong: No causal mask
logits = self.forward(tokens)  # Position i can see all positions (invalid for GPT)

# Correct: Create and apply causal mask
mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
for block in self.blocks:
    x = block(x, mask=mask)  # Causal masking applied
```

### Incorrect Temperature Sampling Implementation

**Problem**: Applying temperature after softmax or forgetting to apply it entirely results in incorrect generation behavior

**Solution**: Divide logits by temperature before softmax to control randomness in sampling

```python
# Wrong: Temperature after softmax (no effect)
probs = softmax(logits)
probs = probs / temperature  # Doesn't change distribution shape

# Correct: Temperature before softmax
logits = logits / temperature  # temperature > 1 flattens, < 1 sharpens
probs = softmax(logits)
next_token = sample(probs)
```

### LayerNorm Numerical Instability Without Epsilon

**Problem**: Computing variance without epsilon causes division by zero or NaN when all features are identical

**Solution**: Always add small epsilon (1e-5) to variance before taking square root

```python
# Wrong: No epsilon protection
normalized = (x - mean) / sqrt(variance)  # NaN if variance = 0

# Correct: Epsilon for numerical stability
normalized = (x - mean) / sqrt(variance + self.eps)  # Safe for variance = 0
```

## Production Context

### Your Implementation vs Production Frameworks

Understanding what you are building vs what production frameworks provide:

| Feature | Your Transformers | PyTorch nn.Transformer | HuggingFace Transformers |
|---------|-------------------|------------------------|--------------------------|
| **Backend** | NumPy (CPU-only) | C++/CUDA (CPU/GPU/TPU) | PyTorch + optimizations |
| **Algorithm** | Exact: Pre-norm GPT | Same architecture | Same + variants (BERT, T5) |
| **Causal Masking** | Manual upper triangular | Built-in mask parameter | Auto-generated per model |
| **Memory** | O(n²) attention per layer | Same + gradient checkpointing | Same + FlashAttention option |
| **Generation** | Temperature sampling | Built-in generate() | Advanced (beam search, top-k, nucleus) |
| **KV Caching** | Not implemented | Manual implementation | Automatic during generation |
| **Model Sizes** | Educational scale | Any scale | Pre-trained (GPT-2 to 175B) |
| **Optimizations** | None | CUDA kernels, mixed precision | All + model parallelism |

**Educational Focus**: Your implementation prioritizes understanding the architecture fundamentals. The O(n²) attention cost and linear parameter scaling you experience are the SAME bottlenecks production systems face—they just optimize with specialized hardware and algorithms.

### Side-by-Side Code Comparison

**Your TinyTorch GPT:**
```python
from tinytorch.models.transformer import GPT

# Create GPT model
gpt = GPT(
    vocab_size=50257,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    max_seq_len=1024
)

# Forward pass
tokens = Tensor([[15496, 1917, 318]])  # "This is a"
logits = gpt.forward(tokens)  # YOUR implementation

# Generation
generated = gpt.generate(
    prompt_tokens=tokens,
    max_new_tokens=50,
    temperature=0.8
)
```

**Equivalent HuggingFace (Production):**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Forward pass (automatic GPU, KV caching, optimizations)
tokens = tokenizer("This is a", return_tensors="pt")
logits = model(**tokens).logits

# Generation (beam search, top-k, nucleus sampling, KV cache)
generated = model.generate(
    **tokens,
    max_new_tokens=50,
    temperature=0.8,
    do_sample=True
)
```

**Key Differences:**
1. **Pre-trained Weights**: HuggingFace loads GPT-2 trained on 40GB of text; your model starts from random initialization
2. **GPU Acceleration**: HuggingFace moves model to GPU with `.to("cuda")` for 10-100× speedup
3. **KV Caching**: HuggingFace caches key-value pairs during generation, reducing O(n²) to O(n) per token
4. **Advanced Sampling**: HuggingFace supports beam search, top-k, nucleus (top-p), and other sampling strategies
5. **Production Optimizations**: Mixed precision (FP16/BF16), model parallelism, gradient checkpointing

### Real-World Applications

**OpenAI GPT-4**: Uses 120-layer decoder with estimated 1.8 trillion parameters. Architecture identical to your implementation—pre-norm transformer blocks with causal attention. Training required 25,000 A100 GPUs for 3-4 months. Inference serves millions of users requiring KV caching, quantization (FP16/INT8), and distributed serving across thousands of GPUs.

**Meta LLaMA 2**: 70B parameter open-source model uses your exact architecture with Grouped-Query Attention optimization. Replaces 80 independent KV heads with 10 shared heads, reducing KV cache from 640GB to 80GB (8× memory saving) with minimal quality loss. Released with pre-trained weights enabling fine-tuning for custom applications.

**Anthropic Claude**: Uses decoder-only architecture with 100K+ token context window. Extended context requires sparse attention patterns and memory optimizations—full dense attention would need 38TB for attention weights across layers. Production systems use FlashAttention and custom CUDA kernels for feasibility.

**Google Gemini**: Multimodal transformer processing text, images, audio, and video using unified decoder architecture. Same self-attention mechanism you implemented, but operates on mixed token types. Demonstrates versatility of transformer architecture beyond pure language modeling.

### Performance Characteristics at Scale

**Parameter Scaling**: Your GPT with embed_dim=768, num_layers=12, num_heads=12 has approximately 117M parameters (similar to GPT-2 Small). Scaling to GPT-3: embed_dim=12,288, num_layers=96, num_heads=96 yields 175B parameters. Storage: 175B × 2 bytes (FP16) = 350GB. Training requires 4× memory for gradients/optimizer states = 1.4TB per GPU.

**Attention Memory Growth**: For GPT-3 with seq_len=2048, batch=4, heads=96: attention weights per layer = 4 × 96 × 2048² = 1.6B elements × 4 bytes = 6.4GB. Across 96 layers: 614GB (infeasible!). FlashAttention solves this by computing attention on-the-fly with O(n) memory = 12MB per layer.

**Generation Efficiency**: Autoregressive generation without KV caching: 100 tokens = 5,050 attention computations (1+2+...+100). With KV cache: 100 tokens = 100 attention computations (10-15× speedup). All production LLMs use KV caching—ChatGPT would be economically infeasible without it.

**Training Compute**: GPT-3 training required 3.14×10²³ FLOPs, consuming 1,300 MWh of electricity. At commercial GPU rates, training cost approximately 5-10 million USD. This explains why few organizations train models from scratch—most fine-tune existing models like LLaMA 2 or GPT-3.5.

## Testing

### Comprehensive Test Suite

Run the full test suite to verify transformer functionality:

```bash
# TinyTorch CLI (recommended)
tito test transformers

# Direct pytest execution
python -m pytest tests/ -k transformers -v
```

### Test Coverage Areas

- ✅ **LayerNorm**: Feature-wise normalization (mean≈0, std≈1), learnable gamma/beta parameters, numerical stability with epsilon
- ✅ **MLP**: 4x expansion ratio (embed_dim → 4*embed_dim → embed_dim), GELU activation, shape preservation
- ✅ **TransformerBlock**: Pre-norm architecture (LayerNorm before sub-layers), residual connections (x + sublayer), attention+MLP composition
- ✅ **GPT Model**: Forward pass shape correctness (batch, seq, vocab_size), causal masking preventing future leakage, autoregressive generation
- ✅ **Generation**: Temperature sampling (conservative vs creative), sequence extension, parameter counting validation

### Inline Testing & Architecture Validation

The module includes comprehensive architecture validation:

```python
# Example inline test output
🔬 Unit Test: LayerNorm...
✅ Mean ≈ 0, std ≈ 1 after normalization
✅ Learnable gamma/beta parameters work
📈 Progress: LayerNorm ✓

🔬 Unit Test: MLP...
✅ 4x expansion ratio correct (embed_dim → 4*embed_dim)
✅ Shape preserved (input: [2,10,64] → output: [2,10,64])
✅ GELU activation applied
📈 Progress: MLP ✓

🔬 Unit Test: TransformerBlock...
✅ Pre-norm residual connections work
✅ Attention + MLP sub-layers compose correctly
✅ Causal mask prevents future information leak
📈 Progress: TransformerBlock ✓

🔬 Unit Test: GPT Model...
✅ Forward pass: [2,8] tokens → [2,8,100] logits
✅ Generation: [1,5] prompt + 3 new → [1,8] sequence
✅ Parameter counting validates all components
📈 Progress: GPT Model ✓
```

### Manual Testing Examples

```python
from transformers import GPT, TransformerBlock, LayerNorm, MLP

# Test LayerNorm
ln = LayerNorm(512)
x = Tensor(np.random.randn(2, 10, 512))  # (batch, seq, features)
normalized = ln.forward(x)
print(f"Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")  # ≈ 0, ≈ 1

# Test MLP
mlp = MLP(embed_dim=512)
output = mlp.forward(x)
assert output.shape == (2, 10, 512)  # Shape preserved

# Test TransformerBlock
block = TransformerBlock(embed_dim=512, num_heads=8)
mask = Tensor(np.triu(np.ones((10, 10)) * -np.inf, k=1))  # Causal mask
transformed = block.forward(x, mask=mask)

# Test GPT
gpt = GPT(vocab_size=50000, embed_dim=768, num_layers=12, num_heads=12)
tokens = Tensor(np.random.randint(0, 50000, (4, 512)))  # Batch of sequences
logits = gpt.forward(tokens)  # (4, 512, 50000)

# Test generation
prompt = Tensor(np.array([[15496, 1917]]))  # "Hello world"
generated = gpt.generate(prompt, max_new_tokens=50, temperature=0.8)
print(f"Generated {generated.shape[1] - prompt.shape[1]} new tokens")
```

## Where This Code Lives in the Final Package

**Package Export:** Code exports to `tinytorch.models.transformer`

```python
# When students install tinytorch, they import your work like this:
from tinytorch.core.transformer import GPT, TransformerBlock
from tinytorch.nn import LayerNorm, MLP  # Your normalization and feed-forward implementations
from tinytorch.core.tensor import Tensor  # Foundation from Module 01
from tinytorch.core.attention import MultiHeadAttention  # From Module 12
from tinytorch.core.embeddings import Embedding  # From Module 11

# Example: Build a GPT-2 scale model
gpt2 = GPT(
    vocab_size=50257,      # GPT-2 BPE vocabulary
    embed_dim=768,         # GPT-2 Small dimension
    num_layers=12,         # 12 transformer blocks
    num_heads=12,          # 12 attention heads
    max_seq_len=1024       # 1K token context
)

# Forward pass
tokens = Tensor([[15496, 1917, 318, 281]])  # "This is a"
logits = gpt2.forward(tokens)  # (1, 4, 50257)

# Autoregressive generation
generated = gpt2.generate(
    prompt_tokens=tokens,
    max_new_tokens=100,
    temperature=0.7  # Balanced creativity
)

# Example: Build transformer components directly
block = TransformerBlock(embed_dim=512, num_heads=8, mlp_ratio=4)
ln = LayerNorm(512)
mlp = MLP(embed_dim=512, hidden_dim=2048)
```

**Package Structure:**
```
tinytorch/
├── models/
│   └── transformer.py      # GPT, TransformerBlock
├── nn/
│   ├── feedforward.py      # MLP implementation
│   └── normalization.py    # LayerNorm implementation
├── core/
│   ├── attention.py        # MultiHeadAttention (Module 12)
│   └── layers.py           # Linear layers
└── text/
    └── embeddings.py       # Embedding, PositionalEncoding
```

## Systems Thinking Questions

### Real-World Applications

- **Large Language Models (OpenAI, Anthropic, Google)**: GPT-4 uses ~120-layer decoder stack trained on trillions of tokens. ChatGPT is GPT-3.5 with 96 layers and RLHF fine-tuning. Claude uses decoder-only architecture with constitutional AI training. All modern LLMs are transformer decoders because decoder-only architecture scales predictably with parameters and data—every 10× parameter increase yields ~5× better performance.

- **Code Generation Systems (GitHub, Google, Meta)**: Copilot uses GPT-based decoder trained on billions of lines of GitHub code. AlphaCode uses transformer decoder for competitive programming. CodeLlama specialized 70B decoder for code completion. All leverage causal attention for autoregressive generation because programming requires left-to-right token prediction matching code syntax.

- **Conversational AI (ChatGPT, Claude, Gemini)**: All modern chatbots use decoder-only transformers fine-tuned with RLHF (reinforcement learning from human feedback). Architecture is identical to base GPT—conversation formatted as single sequence with special tokens. Production systems serve billions of queries daily requiring efficient KV caching to avoid recomputing past tokens.

- **Production Scaling Challenges**: Training GPT-3 (175B parameters) required 3.14×10²³ FLOPs (floating point operations), consuming ~1,300 MWh of electricity. Inference costs dominate at scale—ChatGPT serves millions of users requiring thousands of GPUs. Memory is primary bottleneck: 175B parameters × 2 bytes (FP16) = 350GB just for model weights, plus activation memory during inference.

### Architectural Foundations

- **Residual Connections Enable Deep Networks**: Without residuals, gradients vanish exponentially with depth—in a 12-layer network without residuals, gradients at layer 1 are ~0.1¹² ≈ 10⁻¹² smaller than output gradients. Residuals create gradient highways: ∂Loss/∂x = ∂Loss/∂output × (1 + ∂F(x)/∂x), ensuring gradient magnitude ≥ output gradient. This enables 100+ layer transformers (GPT-4 has ~120 layers).

- **Pre-Norm vs Post-Norm Architecture**: Pre-norm (LayerNorm before sub-layers) provides better gradient flow for deep models. In post-norm, gradients must flow through LayerNorm's division operation which can amplify small gradient differences. Pre-norm gives each sub-layer clean normalized inputs (mean=0, var=1) while residuals bypass the normalization during backprop. GPT-3, GPT-4, LLaMA all use pre-norm.

- **Layer Normalization vs Batch Normalization**: LayerNorm normalizes across features per sample (batch-independent), BatchNorm normalizes across batch per feature (batch-dependent). Transformers use LayerNorm because: (1) Variable sequence lengths make batch statistics unstable, (2) Inference requires batch=1 support, (3) Empirically better for NLP. BatchNorm works for CNNs because spatial dimensions provide consistent normalization axis.

- **MLP Expansion Ratio Trade-offs**: Standard 4× expansion (embed_dim=512 → hidden=2048) balances capacity with compute. MLP parameters dominate transformers: per layer, MLP has 8×embed_dim² parameters vs attention's 4×embed_dim². Larger expansion (8×) increases capacity but quadratically increases memory and FLOPs. Some models experiment with 2× (faster) or gated MLPs (SwiGLU in LLaMA uses 5.33× effective expansion).

### Performance Characteristics

- **Quadratic Attention Memory Growth**: Attention computes (batch, heads, seq_len, seq_len) matrix requiring batch×heads×seq_len² elements. For GPT-3 with seq_len=2048, batch=4, heads=96: 4×96×2048² ≈ 1.6B elements × 4 bytes = 6.4GB per layer just for attention matrices. Doubling sequence length quadruples attention memory. This is why 8K context requires 4× memory vs 4K context.

- **Parameter Scaling**: Total parameters ≈ vocab_size×embed_dim (embeddings) + num_layers×[4×embed_dim² (attention) + 8×embed_dim² (MLP)] ≈ num_layers×12×embed_dim². GPT-3 has embed_dim=12,288, num_layers=96 → 96×12×12,288² ≈ 175B parameters. Storage: 175B × 2 bytes (FP16) = 350GB. Training requires 4× memory for gradients and optimizer states = 1.4TB per GPU.

- **Computational Complexity**: Per layer: O(batch×seq_len²×embed_dim) for attention + O(batch×seq_len×embed_dim²) for MLP. For short sequences (seq_len < embed_dim), MLP dominates. For long sequences (seq_len > embed_dim), attention dominates. GPT-3 with seq_len=2048, embed_dim=12,288: attention is 2048²×12,288 ≈ 51B FLOPs vs MLP 2048×12,288² ≈ 309B FLOPs—MLP dominates even at 2K tokens.

- **Generation Efficiency**: Autoregressive generation requires one forward pass per token. For 100 tokens through 96-layer network: 100×96 = 9,600 layer evaluations. KV caching optimizes this: cache key-value pairs from previous positions, reducing attention from O(n²) to O(n) during generation. Without KV cache, 100-token generation takes ~10× longer. Production systems always use KV caching.

- **Memory-Compute Trade-offs**: Gradient checkpointing trades compute for memory by recomputing activations during backward pass instead of storing them. Saves ~50% activation memory but increases training time ~20%. Mixed precision training (FP16/BF16 forward, FP32 gradients) reduces memory by 50% and increases throughput by 2-3× on modern GPUs with tensor cores.

## Reflection Questions

1. **Residual Connection Necessity**: Remove residual connections from a 12-layer transformer. What happens during training? Calculate gradient flow: if each layer multiplies gradients by 0.5, what's the gradient at layer 1 after 12 layers? (0.5¹² ≈ 0.0002). How do residuals solve this by providing gradient highways that bypass layer computations?

2. **Pre-Norm vs Post-Norm Trade-offs**: Original Transformer paper used post-norm (LayerNorm after sub-layers). Modern transformers use pre-norm (LayerNorm before). Why? Consider gradient flow: in post-norm, gradients pass through LayerNorm's division which can amplify noise. In pre-norm, residuals bypass normalization. When does pre-norm become critical (how many layers)?

3. **Attention Memory Quadratic Growth**: For seq_len=1024, batch=4, heads=8, attention matrix is 4×8×1024×1024 = 33.5M elements × 4 bytes = 134MB per layer. What happens at seq_len=4096? (×16 memory = 2.1GB per layer). Why is this quadratic growth the primary bottleneck for long-context models? How does FlashAttention address this?

4. **Parameter Scaling Analysis**: GPT-3 has embed_dim=12,288, num_layers=96. Calculate approximate parameters: embeddings ≈ 50K vocab × 12,288 = 614M. Per layer: attention ≈ 4×12,288² = 604M, MLP ≈ 8×12,288² = 1.2B. Total per layer ≈ 1.8B. 96 layers × 1.8B = 173B. Compare to measured 175B. What's the parameter distribution?

5. **Decoder-Only vs Encoder-Decoder**: Why did decoder-only (GPT) dominate over encoder-decoder (T5) for LLMs? Consider: (1) Simplicity of single architecture, (2) Scaling laws holding predictably, (3) Versatility handling both understanding and generation. When would you still choose encoder-decoder (translation, summarization)?

6. **Generation Efficiency**: Generating 100 tokens through 96-layer GPT-3 without KV caching requires 100 forward passes through all 96 layers = 9,600 layer evaluations. With KV caching, only new token processed through layers = 96 evaluations per token = 9,600 total. Same compute! But KV cache requires storing keys and values for all positions. Calculate memory for seq_len=2048: 2×(num_layers×batch×heads×seq_len×head_dim) elements. What's the memory-compute trade-off?

## Ready to Build?

You're about to implement the transformer architecture that powers virtually all modern AI systems! The decoder-only GPT architecture you'll build is the exact design used in ChatGPT, GPT-4, Claude, and every major language model. This isn't a simplified educational version—it's the real production architecture that revolutionized AI.

Understanding transformers from first principles—implementing layer normalization, feed-forward networks, residual connections, and causal attention yourself—will give you deep insight into how production ML systems work. You'll understand why GPT-4 has 120 layers, why residual connections prevent gradient vanishing in deep networks, why pre-norm architecture enables training very deep models, and how attention memory scales quadratically with sequence length.

This module is the culmination of your Architecture Tier journey. You've built tensors (Module 01), activations (Module 02), layers (Module 03), embeddings (Module 11), and attention (Module 12). Now you'll compose them into the complete transformer model that matches PyTorch's `nn.TransformerDecoder` and powers billion-dollar AI systems. Take your time, test thoroughly, and enjoy building the architecture behind ChatGPT, Claude, and the AI revolution!

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/13_transformers/transformers.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/13_transformers/transformers.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/13_transformers/transformers.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

```

---

<div class="prev-next-area">
<a class="left-prev" href="../modules/12_attention/ABOUT.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../modules/14_profiling/ABOUT.html" title="next page">Next Module →</a>
</div>
