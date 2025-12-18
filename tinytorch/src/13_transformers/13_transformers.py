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
# Module 13: Transformers - Complete Transformer Architecture

Welcome to Module 13! You're about to build the complete transformer architecture that powers modern language models like GPT, Claude, and ChatGPT.

## 🔗 Prerequisites & Progress
**You've Built**: Tokenization, embeddings, attention mechanisms, and all foundational components
**You'll Build**: TransformerBlock, complete GPT architecture, and autoregressive generation
**You'll Enable**: Full language model training and text generation capabilities

**Connection Map**:
```
Tokenization + Embeddings + Attention → Transformers → Language Generation
(text→numbers)  (learnable vectors) (sequence modeling)  (complete models)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement complete TransformerBlock with attention, MLP, and layer normalization
2. Build full GPT architecture with multiple transformer blocks
3. Add autoregressive text generation capability
4. Understand parameter scaling in large language models
5. Test transformer components and generation pipeline

Let's get started!
"""

# %%
#| default_exp core.transformer

# %%
#| export
import numpy as np
import math
from typing import Optional, List

# Import from previous modules - following proper dependency chain
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.attention import MultiHeadAttention
from tinytorch.core.activations import GELU
from tinytorch.core.embeddings import Embedding, PositionalEncoding

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


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
        >>> from tinytorch.core.transformer import create_causal_mask
        >>> mask = create_causal_mask(seq_len=10)
        >>> output = attention(x, mask=mask)
    """
    # Lower triangular matrix: 1 = can attend, 0 = cannot attend
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return Tensor(mask[np.newaxis, :, :])  # Add batch dimension


# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/13_transformers/transformers_dev.py`
**Building Side:** Code exports to `tinytorch.core.transformer`

```python
# How to use this module:
from tinytorch.core.transformer import TransformerBlock, GPT, LayerNorm, MLP
```

**Why this matters:**
- **Learning:** Complete transformer system showcasing how all components work together
- **Production:** Matches PyTorch's transformer implementation with proper model organization
- **Consistency:** All transformer components and generation logic in core.transformer
- **Integration:** Demonstrates the power of modular design by combining all previous modules
"""


# %% [markdown]
"""
## 💡 Introduction: What are Transformers?

Transformers are the revolutionary architecture that powers modern AI language models like GPT, ChatGPT, and Claude. The key breakthrough is **self-attention**, which allows every token in a sequence to directly interact with every other token, creating rich contextual understanding.

### The Transformer Revolution

Before transformers, language models used RNNs or CNNs that processed text sequentially or locally. Transformers changed everything by processing all positions in parallel while maintaining global context.

### Complete GPT Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  COMPLETE GPT ARCHITECTURE: From Text to Generation             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: "Hello world"  →  Token IDs: [15496, 1917]              │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                EMBEDDING LAYER                            │  │
│  │                                                           │  │
│  │  ┌─────────────┐       ┌─────────────────────────────┐    │  │
│  │  │Token Embed  │   +   │ Positional Embedding        │    │  │
│  │  │15496→[0.1,  │       │ pos_0→[0.05, -0.02, ...]    │    │  │
│  │  │     0.3,..]│       │ pos_1→[0.12,  0.08, ...]     │    │  │
│  │  │1917→[0.2,   │       │                             │    │  │
│  │  │    -0.1,..]│       │                              │    │  │
│  │  └─────────────┘       └─────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              TRANSFORMER BLOCK 1                          │  │
│  │                                                           │  │
│  │  x → LayerNorm → MultiHeadAttention → + x → result        │  │
│  │  │                                      ↑                 │  │
│  │  │              residual connection     │                 │  │
│  │  └──────────────────────────────────────┘                 │  │
│  │  │                                                        │  │
│  │  result → LayerNorm → MLP (Feed Forward) → + result       │  │
│  │  │                                           ↑            │  │
│  │  │                residual connection        │            │  │
│  │  └───────────────────────────────────────────┘            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│              TRANSFORMER BLOCK 2 (same pattern)                 │
│                                ↓                                │
│                      ... (more blocks) ...                      │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   OUTPUT HEAD                             │  │
│  │                                                           │  │
│  │  final_hidden → LayerNorm → Linear(embed_dim, vocab_size) │  │
│  │                              ↓                            │  │
│  │               Vocabulary Logits: [0.1, 0.05, 0.8, ...]    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│  OUTPUT: Next Token Probabilities                               │
│  "Hello" → 10%,  "world" → 5%,  "!" → 80%,  ...                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Transformers Dominate

**Parallel Processing**: Unlike RNNs that process tokens one by one, transformers process all positions simultaneously. This makes training much faster.

**Global Context**: Every token can directly attend to every other token in the sequence, capturing long-range dependencies that RNNs struggle with.

**Scalability**: Performance predictably improves with more parameters and data. This enabled the scaling laws that led to GPT-3, GPT-4, and beyond.

**Residual Connections**: Allow training very deep networks (100+ layers) by providing gradient highways.

### The Building Blocks We'll Implement

1. **LayerNorm**: Stabilizes training by normalizing activations
2. **Multi-Layer Perceptron (MLP)**: Provides non-linear transformation
3. **TransformerBlock**: Combines attention + MLP with residuals
4. **GPT**: Complete model with embeddings and generation capability
"""

# %% [markdown]
"""
## 📐 Foundations: Essential Transformer Mathematics

### Layer Normalization: The Stability Engine

Layer Normalization is crucial for training deep transformer networks. Unlike batch normalization (which normalizes across the batch), layer norm normalizes across the feature dimension for each individual sample.

```
Mathematical Formula:
output = (x - μ) / σ * γ + β

where:
  μ = mean(x, axis=features)     # Mean across feature dimension
  σ = sqrt(var(x) + ε)          # Standard deviation + small epsilon
  γ = learnable scale parameter  # Initialized to 1.0
  β = learnable shift parameter  # Initialized to 0.0
```

**Why Layer Norm Works:**
- **Independence**: Each sample normalized independently (good for variable batch sizes)
- **Stability**: Prevents internal covariate shift that breaks training
- **Gradient Flow**: Helps gradients flow better through deep networks

### Residual Connections: The Gradient Highway

Residual connections are the secret to training deep networks. They create "gradient highways" that allow information to flow directly through the network.

```
┌─────────────────────────────────────────────────────────────────┐
│  RESIDUAL CONNECTIONS: The Gradient Highway System              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PRE-NORM ARCHITECTURE (Modern Standard):                       │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                ATTENTION SUB-LAYER                        │  │
│  │                                                           │  │
│  │  Input (x) ────┬─→ LayerNorm ─→ MultiHeadAttention ─┐     │  │
│  │                │                                    │     │  │
│  │                │         ┌──────────────────────────┘     │  │
│  │                │         ▼                                │  │
│  │                └────→ ADD ─→ Output to next sub-layer     │  │
│  │                      (x + attention_output)               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   MLP SUB-LAYER                           │  │
│  │                                                           │  │
│  │  Input (x) ────┬─→ LayerNorm ─→ MLP (Feed Forward)  ─┐    │  │
│  │                │                                     │    │  │
│  │                │         ┌───────────────────────────┘    │  │
│  │                │         ▼                                │  │
│  │                └────→ ADD ─→ Final Output                 │  │
│  │                      (x + mlp_output)                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  KEY INSIGHT: Each sub-layer ADDS to the residual stream        │
│  rather than replacing it, preserving information flow!         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Gradient Flow Visualization:**
```
Backward Pass Without Residuals:    With Residuals:
Loss                                Loss
 │ gradients get smaller             │ gradients stay strong
 ↓ at each layer                    ↓ via residual paths
Layer N  ← tiny gradients          Layer N  ← strong gradients
 │                                  │     ↗ (direct path)
 ↓                                  ↓   ↗
Layer 2  ← vanishing                Layer 2  ← strong gradients
 │                                  │     ↗
 ↓                                  ↓   ↗
Layer 1  ← gone!                   Layer 1  ← strong gradients
```

### Feed-Forward Network (MLP): The Thinking Layer

The MLP provides the actual "thinking" in each transformer block. It's a simple two-layer network with a specific expansion pattern.

```
MLP Architecture:
Input (embed_dim) → Linear → GELU → Linear → Output (embed_dim)
       512           2048      2048    512
                   (4x expansion)

Mathematical Formula:
FFN(x) = Linear₂(GELU(Linear₁(x)))
       = W₂ · GELU(W₁ · x + b₁) + b₂

where:
  W₁: (embed_dim, 4*embed_dim)  # Expansion matrix
  W₂: (4*embed_dim, embed_dim)  # Contraction matrix
  GELU: smooth activation function (better than ReLU for language)
```

**Why 4x Expansion?**
- **Capacity**: More parameters = more representation power
- **Non-linearity**: GELU activation creates complex transformations
- **Information Bottleneck**: Forces the model to compress useful information

### The Complete Transformer Block Data Flow

```
Input Tensor (batch, seq_len, embed_dim)
          ↓
    ┌─────────────────────────────────────┐
    │ ATTENTION SUB-LAYER                 │
    │                                     │
    │ x₁ = LayerNorm(x₀)                  │
    │ attention_out = MultiHeadAttn(x₁)   │
    │ x₂ = x₀ + attention_out  (residual) │
    └─────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────┐
    │ MLP SUB-LAYER                       │
    │                                     │
    │ x₃ = LayerNorm(x₂)                  │
    │ mlp_out = MLP(x₃)                   │
    │ x₄ = x₂ + mlp_out    (residual)     │
    └─────────────────────────────────────┘
          ↓
Output Tensor (batch, seq_len, embed_dim)
```

**Key Insight**: Each sub-layer (attention and MLP) gets a "clean" normalized input but adds its contribution to the residual stream. This creates a stable training dynamic.
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building Transformer Components

Now we'll implement each transformer component with a clear understanding of their role in the overall architecture. We'll follow the pattern: **Explanation → Implementation → Test** for each component.

Each component serves a specific purpose:
- **LayerNorm**: Stabilizes training and normalizes activations
- **MLP**: Provides non-linear transformation and "thinking" capacity
- **TransformerBlock**: Combines attention with MLP using residual connections
- **GPT**: Complete autoregressive language model for text generation
"""

# %% [markdown]
"""
### Understanding Layer Normalization

Layer Normalization is the foundation of stable transformer training. Unlike batch normalization, it normalizes each sample independently across its feature dimensions.

#### Why Layer Norm is Essential

Without normalization, deep networks suffer from "internal covariate shift" - the distribution of inputs to each layer changes during training, making learning unstable.

#### Layer Norm Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER NORMALIZATION: Stabilizing Deep Networks                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT TENSOR: (batch=2, seq=3, features=4)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Sample 1: [[1.0,  2.0,  3.0,  4.0],     ← Position 0      │  │
│  │            [5.0,  6.0,  7.0,  8.0],     ← Position 1      │  │
│  │            [9.0, 10.0, 11.0, 12.0]]     ← Position 2      │  │
│  │                                                           │  │
│  │ Sample 2: [[13., 14., 15., 16.],         ← Position 0     │  │
│  │            [17., 18., 19., 20.],         ← Position 1     │  │
│  │            [21., 22., 23., 24.]]         ← Position 2     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│           NORMALIZE ACROSS FEATURES (per position)              │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ AFTER NORMALIZATION: Each position → mean=0, std=1        │  │
│  │                                                           │  │
│  │ Sample 1: [[-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34]]                  │  │
│  │                                                           │  │
│  │ Sample 2: [[-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34]]                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│            APPLY LEARNABLE PARAMETERS: γ * norm + β             │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ FINAL OUTPUT: Model can learn any desired distribution    │  │
│  │ γ (scale) and β (shift) are learned during training       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  KEY INSIGHT: Unlike batch norm, each sample normalized         │
│  independently - perfect for variable-length sequences!         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Properties
- **Per-sample normalization**: Each sequence position normalized independently
- **Learnable parameters**: γ (scale) and β (shift) allow the model to recover any desired distribution
- **Gradient friendly**: Helps gradients flow smoothly through deep networks
"""

# %% nbgrader={"grade": false, "grade_id": "layer-norm", "solution": true}
#| export
class LayerNorm:
    """
    Layer Normalization for transformer blocks.

    Normalizes across the feature dimension (last axis) for each sample independently,
    unlike batch normalization which normalizes across the batch dimension.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize LayerNorm with learnable parameters.

        TODO: Set up normalization parameters

        APPROACH:
        1. Store the shape to normalize over (usually embed_dim)
        2. Initialize learnable scale (gamma) and shift (beta) parameters
        3. Set small epsilon for numerical stability

        EXAMPLE:
        >>> ln = LayerNorm(512)  # For 512-dimensional embeddings
        >>> x = Tensor(np.random.randn(2, 10, 512))  # (batch, seq, features)
        >>> normalized = ln.forward(x)
        >>> # Each (2, 10) sample normalized independently across 512 features

        HINTS:
        - gamma should start at 1.0 (identity scaling)
        - beta should start at 0.0 (no shift)
        - eps prevents division by zero in variance calculation
        """
        ### BEGIN SOLUTION
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters: scale and shift
        self.gamma = Tensor(np.ones(normalized_shape))  # Scale parameter
        self.beta = Tensor(np.zeros(normalized_shape))  # Shift parameter
        ### END SOLUTION

    def forward(self, x):
        """
        Apply layer normalization.

        TODO: Implement layer normalization formula

        APPROACH:
        1. Compute mean and variance across the last dimension
        2. Normalize: (x - mean) / sqrt(variance + eps)
        3. Apply learnable scale and shift: gamma * normalized + beta

        MATHEMATICAL FORMULA:
        y = (x - μ) / σ * γ + β
        where μ = mean(x), σ = sqrt(var(x) + ε)

        HINT: Use keepdims=True to maintain tensor dimensions for broadcasting
        """
        ### BEGIN SOLUTION
        # Compute statistics across last dimension (features)
        mean = x.mean(axis=-1, keepdims=True)

        # Compute variance: E[(x - μ)²]
        # Use Tensor operations to preserve computation graph!
        diff = x - mean
        variance = (diff * diff).mean(axis=-1, keepdims=True)

        # Normalize
        std = Tensor(np.sqrt(variance.data + self.eps))
        normalized = (x - mean) / std

        # Apply learnable transformation
        output = normalized * self.gamma + self.beta
        return output
        ### END SOLUTION

    def __call__(self, x):
        """Allows the layer norm to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return learnable parameters."""
        return [self.gamma, self.beta]

# %% [markdown]
"""
### 🔬 Unit Test: Layer Normalization
This test validates our LayerNorm implementation works correctly.
**What we're testing**: Normalization statistics and parameter learning
**Why it matters**: Essential for transformer stability and training
**Expected**: Mean ≈ 0, std ≈ 1 after normalization, learnable parameters work
"""

# %% nbgrader={"grade": true, "grade_id": "test-layer-norm", "locked": true, "points": 10}
def test_unit_layer_norm():
    """🔬 Test LayerNorm implementation."""
    print("🔬 Unit Test: Layer Normalization...")

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

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_layer_norm()  # Moved after implementation

# %% [markdown]
"""
### Understanding the Multi-Layer Perceptron (MLP)

The MLP is where the "thinking" happens in each transformer block. It's a simple feed-forward network that provides non-linear transformation capacity.

#### The Role of MLP in Transformers

While attention handles relationships between tokens, the MLP processes each position independently, adding computational depth and non-linearity.

#### MLP Architecture and Information Flow

```
Information Flow Through MLP:

Input: (batch, seq_len, embed_dim=512)
         ↓
┌─────────────────────────────────────────────┐
│ Linear Layer 1: Expansion                   │
│ Weight: (512, 2048)  Bias: (2048,)          │
│ Output: (batch, seq_len, 2048)              │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ GELU Activation                             │
│ Smooth, differentiable activation           │
│ Better than ReLU for language modeling      │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ Linear Layer 2: Contraction                 │
│ Weight: (2048, 512)  Bias: (512,)           │
│ Output: (batch, seq_len, 512)               │
└─────────────────────────────────────────────┘
         ↓
Output: (batch, seq_len, embed_dim=512)
```

#### Why 4x Expansion?

```
Parameter Count Analysis:

Embed Dim: 512
MLP Hidden: 2048 (4x expansion)

Parameters:
- Linear1: 512 × 2048 + 2048 = 1,050,624
- Linear2: 2048 × 512 + 512 = 1,049,088
- Total MLP: ~2.1M parameters

For comparison:
- Attention (same embed_dim): ~1.5M parameters
- MLP has MORE parameters → more computational capacity
```

#### GELU vs ReLU

```
Activation Function Comparison:

ReLU(x) = max(0, x)        # Hard cutoff at 0
         ┌────
         │
    ─────┘
         0

GELU(x) ≈ x * Φ(x)         # Smooth, probabilistic
         ╭────
        ╱
    ───╱
      ╱
     0

GELU is smoother and provides better gradients for language modeling.
```
"""

# %% nbgrader={"grade": false, "grade_id": "mlp", "solution": true}
#| export
class MLP:
    """
    Multi-Layer Perceptron (Feed-Forward Network) for transformer blocks.

    Standard pattern: Linear -> GELU -> Linear with expansion ratio of 4:1.
    This provides the non-linear transformation in each transformer block.
    """

    def __init__(self, embed_dim, hidden_dim=None, dropout_prob=0.1):
        """
        Initialize MLP with two linear layers.

        TODO: Set up the feed-forward network layers

        APPROACH:
        1. First layer expands from embed_dim to hidden_dim (usually 4x larger)
        2. Second layer projects back to embed_dim
        3. Use GELU activation (smoother than ReLU, preferred in transformers)

        EXAMPLE:
        >>> mlp = MLP(512)  # Will create 512 -> 2048 -> 512 network
        >>> x = Tensor(np.random.randn(2, 10, 512))
        >>> output = mlp.forward(x)
        >>> assert output.shape == (2, 10, 512)

        HINT: Standard transformer MLP uses 4x expansion (hidden_dim = 4 * embed_dim)
        """
        ### BEGIN SOLUTION
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Standard 4x expansion

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Two-layer feed-forward network
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.gelu = GELU()  # Use GELU activation from activations module
        self.linear2 = Linear(hidden_dim, embed_dim)
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through MLP.

        TODO: Implement the feed-forward computation

        APPROACH:
        1. First linear transformation: embed_dim -> hidden_dim
        2. Apply GELU activation (smooth, differentiable)
        3. Second linear transformation: hidden_dim -> embed_dim

        COMPUTATION FLOW:
        x -> Linear -> GELU -> Linear -> output

        HINT: GELU activation is implemented above as a function
        """
        ### BEGIN SOLUTION
        # First linear layer with expansion
        hidden = self.linear1.forward(x)

        # GELU activation (YOUR activation from Module 03!)
        hidden = self.gelu.forward(hidden)

        # Second linear layer back to original size
        output = self.linear2.forward(hidden)

        return output
        ### END SOLUTION

    def __call__(self, x):
        """Allows the MLP to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params

# %% [markdown]
"""
### 🔬 Unit Test: MLP (Feed-Forward Network)
This test validates our MLP implementation works correctly.
**What we're testing**: Shape preservation and parameter counting
**Why it matters**: MLP provides the non-linear transformation in transformers
**Expected**: Input/output shapes match, correct parameter count
"""

# %% nbgrader={"grade": true, "grade_id": "test-mlp", "locked": true, "points": 10}
def test_unit_mlp():
    """🔬 Test MLP implementation."""
    print("🔬 Unit Test: MLP (Feed-Forward Network)...")

    # Test MLP with standard 4x expansion
    embed_dim = 64
    mlp = MLP(embed_dim)

    # Test forward pass
    batch_size, seq_len = 2, 10
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
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

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_mlp()  # Moved after implementation

# %% [markdown]
"""
### Understanding the Complete Transformer Block

The TransformerBlock is the core building unit of GPT and other transformer models. It combines self-attention with feed-forward processing using a carefully designed residual architecture.

#### Pre-Norm vs Post-Norm Architecture

Modern transformers use "pre-norm" architecture where LayerNorm comes BEFORE the sub-layers, not after. This provides better training stability.

```
Pre-Norm Architecture (What We Implement):
┌────────────────────────────────────────────────────────┐
│                     INPUT (x)                          │
│                       │                                │
│       ┌───────────────┴───────────────┐                │
│       │                               │                │
│       ▼                               │                │
│  LayerNorm                            │                │
│       │                               │                │
│       ▼                               │                │
│ MultiHeadAttention                    │                │
│       │                               │                │
│       └───────────────┬───────────────┘                │
│                       │          (residual connection) │
│                       ▼                                │
│                  x + attention                         │
│                       │                                │
│       ┌───────────────┴───────────────┐                │
│       │                               │                │
│       ▼                               │                │
│  LayerNorm                            │                │
│       │                               │                │
│       ▼                               │                │
│      MLP                              │                │
│       │                               │                │
│       └───────────────┬───────────────┘                │
│                       │          (residual connection) │
│                       ▼                                │
│                   x + mlp                              │
│                       │                                │
│                       ▼                                │
│                    OUTPUT                              │
└────────────────────────────────────────────────────────┘
```

#### Why Pre-Norm Works Better

**Training Stability**: LayerNorm before operations provides clean, normalized inputs to attention and MLP layers.

**Gradient Flow**: Residual connections carry gradients directly from output to input, bypassing the normalized operations.

**Deeper Networks**: Pre-norm enables training much deeper networks (100+ layers) compared to post-norm.

#### Information Processing in Transformer Block

```
Step-by-Step Data Transformation:

1. Input Processing:
   x₀: (batch, seq_len, embed_dim) # Original input

2. Attention Sub-layer:
   x₁ = LayerNorm(x₀)               # Normalize input
   attn_out = MultiHeadAttn(x₁)     # Self-attention
   x₂ = x₀ + attn_out               # Residual connection

3. MLP Sub-layer:
   x₃ = LayerNorm(x₂)               # Normalize again
   mlp_out = MLP(x₃)                # Feed-forward
   x₄ = x₂ + mlp_out                # Final residual

4. Output:
   return x₄                        # Ready for next block
```

#### Residual Stream Concept

Think of the residual connections as a "stream" that carries information through the network:

```
Residual Stream Flow:

Layer 1: [original embeddings] ─┐
                                 ├─→ + attention info ─┐
Attention adds information ──────┘                      │
                                                        ├─→ + MLP info ─┐
MLP adds information ───────────────────────────────────┘               │
                                                                        │
Layer 2: carries accumulated information ───────────────────────────────┘
```

Each layer adds information to this stream rather than replacing it, creating a rich representation.
"""

# %% nbgrader={"grade": false, "grade_id": "transformer-block", "solution": true}
#| export
class TransformerBlock:
    """
    Complete Transformer Block with self-attention, MLP, and residual connections.

    This is the core building block of GPT and other transformer models.
    Each block processes the input sequence and passes it to the next block.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout_prob=0.1):
        """
        Initialize a complete transformer block.

        TODO: Set up all components of the transformer block

        APPROACH:
        1. Multi-head self-attention for sequence modeling
        2. First layer normalization (pre-norm architecture)
        3. MLP with specified expansion ratio
        4. Second layer normalization

        TRANSFORMER BLOCK ARCHITECTURE:
        x → LayerNorm → MultiHeadAttention → + (residual) →
            LayerNorm → MLP → + (residual) → output

        EXAMPLE:
        >>> block = TransformerBlock(embed_dim=512, num_heads=8)
        >>> x = Tensor(np.random.randn(2, 10, 512))  # (batch, seq, embed)
        >>> output = block.forward(x)
        >>> assert output.shape == (2, 10, 512)

        HINT: We use pre-norm architecture (LayerNorm before attention/MLP)
        """
        ### BEGIN SOLUTION
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Layer normalizations (pre-norm architecture)
        self.ln1 = LayerNorm(embed_dim)  # Before attention
        self.ln2 = LayerNorm(embed_dim)  # Before MLP

        # Feed-forward network
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim)
        ### END SOLUTION

    def forward(self, x, mask=None):
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

    def __call__(self, x, mask=None):
        """Allows the transformer block to be called like a function."""
        return self.forward(x, mask)

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.mlp.parameters())
        return params

# %% [markdown]
"""
### 🔬 Unit Test: Transformer Block
This test validates our complete TransformerBlock implementation.
**What we're testing**: Shape preservation, residual connections, parameter counting
**Why it matters**: This is the core component that will be stacked to create GPT
**Expected**: Input/output shapes match, all components work together
"""

# %% nbgrader={"grade": true, "grade_id": "test-transformer-block", "locked": true, "points": 15}
def test_unit_transformer_block():
    """🔬 Test TransformerBlock implementation."""
    print("🔬 Unit Test: Transformer Block...")

    # Test transformer block
    embed_dim = 64
    num_heads = 4
    block = TransformerBlock(embed_dim, num_heads)

    # Test forward pass
    batch_size, seq_len = 2, 8
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    output = block.forward(x)

    # Check shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim)

    # Test with causal mask (for autoregressive generation)
    mask = Tensor(np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1))
    masked_output = block.forward(x, mask)
    assert masked_output.shape == (batch_size, seq_len, embed_dim)

    # Test parameter counting
    params = block.parameters()
    expected_components = 4  # attention, ln1, ln2, mlp parameters
    assert len(params) > expected_components  # Should have parameters from all components

    # Test different configurations
    large_block = TransformerBlock(embed_dim=128, num_heads=8, mlp_ratio=2)
    assert large_block.mlp.hidden_dim == 256  # 128 * 2

    print("✅ TransformerBlock works correctly!")

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_transformer_block()  # Moved after implementation

# %% [markdown]
"""
### Understanding the Complete GPT Architecture

GPT (Generative Pre-trained Transformer) is the complete language model that combines all our components into a text generation system. It's designed for **autoregressive** generation - predicting the next token based on all previous tokens.

#### GPT's Autoregressive Nature

GPT generates text one token at a time, using all previously generated tokens as context:

```
Autoregressive Generation Process:

Step 1: "The cat" → model predicts → "sat"
Step 2: "The cat sat" → model predicts → "on"
Step 3: "The cat sat on" → model predicts → "the"
Step 4: "The cat sat on the" → model predicts → "mat"

Result: "The cat sat on the mat"
```

#### Complete GPT Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GPT ARCHITECTURE            |
│                                                 |
│ Input: Token IDs [15496, 1917, ...]             |
│                    │                            |
│ ┌──────────────────┴──────────────────┐         |
│ │          EMBEDDING LAYER            │         |
│ │  ┌─────────────┐  ┌─────────────────┐│        |
│ │  │Token Embed  │+│Position Embed   ││         |
│ │  │vocab→vector ││ │sequence→vector  ││        |
│ │  └─────────────┘  └─────────────────┘│        |
│ └──────────────────┬──────────────────┘         |
│                    │                            |
│ ┌──────────────────┴──────────────────┐         │
│ │        TRANSFORMER BLOCK 1          │         │
│ │ ┌─────────┐  ┌─────────┐  ┌───────┐ │         │
│ │ │LayerNorm│→│Attention│→│  +x   │ │           │
│ │ └─────────┘  └─────────┘  └───┬───┘ │         │
│ │                               │     │         │
│ │ ┌─────────┐  ┌─────────┐  ┌───▼───┐ │         │
│ │ │LayerNorm│→│   MLP   │→│  +x   │ │           │
│ │ └─────────┘  └─────────┘  └───────┘ │         |
│ └──────────────────┬──────────────────┘         │
│                    │                            │
│         ... (more transformer blocks) ...       │
│                    │                            │
│ ┌──────────────────┴──────────────────┐         │
│ │         OUTPUT HEAD                 │         │
│ │ ┌─────────┐  ┌─────────────────────┐ │        │
│ │ │LayerNorm│→│Linear(embed→vocab)  │ │         │
│ │ └─────────┘  └─────────────────────┘ │        │
│ └──────────────────┬──────────────────┘         │
│                    │                            │
│ Output: Vocabulary Logits [0.1, 0.05, 0.8, ...] │
└─────────────────────────────────────────────────┘
```

#### Causal Masking for Autoregressive Training

During training, GPT sees the entire sequence but must not "cheat" by looking at future tokens:

```
┌─────────────────────────────────────────────────────────────────┐
│  CAUSAL MASKING: Preventing Future Information Leakage          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SEQUENCE: ["The", "cat", "sat", "on"]                          │
│  POSITIONS:   0      1      2     3                             │
│                                                                 │
│  ATTENTION MATRIX (what each position can see):                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │      Pos:  0   1   2   3                                 │   │
│  │  Pos 0:  [ ✓   ✗   ✗   ✗ ]  ← "The" only sees itself     │   │
│  │  Pos 1:  [ ✓   ✓   ✗   ✗ ]  ← "cat" sees "The" + self    │   │
│  │  Pos 2:  [ ✓   ✓   ✓   ✗ ]  ← "sat" sees all previous    │   │
│  │  Pos 3:  [ ✓   ✓   ✓   ✓ ]  ← "on" sees everything       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  IMPLEMENTATION: Upper triangular matrix with -∞                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ [[  0, -∞, -∞, -∞],                                      │   │
│  │  [  0,   0, -∞, -∞],                                     │   │
│  │  [  0,   0,   0, -∞],                                    │   │
│  │  [  0,   0,   0,   0]]                                   │   │
│  │                                                          │   │
│  │ After softmax: -∞ becomes 0 probability                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  WHY THIS WORKS: During training, model sees entire sequence    │
│  but mask ensures position i only attends to positions ≤ i      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Generation Temperature Control

Temperature controls the randomness of generation:

```
Temperature Effects:

Original logits: [1.0, 2.0, 3.0]

Temperature = 0.1 (Conservative):
Scaled: [10.0, 20.0, 30.0] → Sharp distribution
Probs: [0.00, 0.00, 1.00] → Always picks highest

Temperature = 1.0 (Balanced):
Scaled: [1.0, 2.0, 3.0] → Moderate distribution
Probs: [0.09, 0.24, 0.67] → Weighted sampling

Temperature = 2.0 (Creative):
Scaled: [0.5, 1.0, 1.5] → Flatter distribution
Probs: [0.18, 0.33, 0.49] → More random
```

#### Model Scaling and Parameters

```
GPT Model Size Scaling:

Tiny GPT (our implementation):
- embed_dim: 64, layers: 2, heads: 4
- Parameters: ~50K
- Use case: Learning and experimentation

GPT-2 Small:
- embed_dim: 768, layers: 12, heads: 12
- Parameters: 117M
- Use case: Basic text generation

GPT-3:
- embed_dim: 12,288, layers: 96, heads: 96
- Parameters: 175B
- Use case: Advanced language understanding

GPT-4 (estimated):
- embed_dim: ~16,384, layers: ~120, heads: ~128
- Parameters: ~1.7T
- Use case: Reasoning and multimodal tasks
```
"""

# %% nbgrader={"grade": false, "grade_id": "gpt", "solution": true}
#| export
class GPT:
    """
    Complete GPT (Generative Pre-trained Transformer) model.

    This combines embeddings, positional encoding, multiple transformer blocks,
    and a language modeling head for text generation.
    """

    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len=1024):
        """
        Initialize complete GPT model.

        TODO: Set up all components of the GPT architecture

        APPROACH:
        1. Token embedding layer to convert tokens to vectors
        2. Positional embedding to add position information
        3. Stack of transformer blocks (the main computation)
        4. Final layer norm and language modeling head

        GPT ARCHITECTURE:
        tokens → embedding → + pos_embedding →
                transformer_blocks → layer_norm → lm_head → logits

        EXAMPLE:
        >>> model = GPT(vocab_size=1000, embed_dim=256, num_layers=6, num_heads=8)
        >>> tokens = Tensor(np.random.randint(0, 1000, (2, 10)))  # (batch, seq)
        >>> logits = model.forward(tokens)
        >>> assert logits.shape == (2, 10, 1000)  # (batch, seq, vocab)

        HINTS:
        - Positional embeddings are learned, not fixed sinusoidal
        - Final layer norm stabilizes training
        - Language modeling head shares weights with token embedding (tie_weights)
        """
        ### BEGIN SOLUTION
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.position_embedding = Embedding(max_seq_len, embed_dim)

        # Stack of transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads)
            self.blocks.append(block)

        # Final layer normalization
        self.ln_f = LayerNorm(embed_dim)

        # Language modeling head (projects to vocabulary)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        ### END SOLUTION

    def forward(self, tokens):
        """
        Forward pass through GPT model.

        TODO: Implement the complete GPT forward pass

        APPROACH:
        1. Get token embeddings and positional embeddings
        2. Add them together (broadcasting handles different shapes)
        3. Pass through all transformer blocks sequentially
        4. Apply final layer norm and language modeling head

        COMPUTATION FLOW:
        tokens → embed + pos_embed → blocks → ln_f → lm_head → logits

        CAUSAL MASKING:
        For autoregressive generation, we need to prevent tokens from
        seeing future tokens. This is handled by the attention mask.

        HINT: Create position indices as range(seq_len) for positional embedding
        """
        ### BEGIN SOLUTION
        batch_size, seq_len = tokens.shape

        # Token embeddings
        token_emb = self.token_embedding.forward(tokens)

        # Positional embeddings
        positions = Tensor(np.arange(seq_len).reshape(1, seq_len))
        pos_emb = self.position_embedding.forward(positions)

        # Combine embeddings
        x = token_emb + pos_emb

        # Create causal mask for autoregressive generation
        mask = self._create_causal_mask(seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer normalization
        x = self.ln_f.forward(x)

        # Language modeling head
        logits = self.lm_head.forward(x)

        return logits
        ### END SOLUTION

    def __call__(self, tokens):
        """Allows the GPT model to be called like a function."""
        return self.forward(tokens)

    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent attending to future positions."""
        ### BEGIN SOLUTION
        # Upper triangular matrix filled with -inf
        mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        return Tensor(mask)
        ### END SOLUTION

    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        """
        Generate text autoregressively.

        TODO: Implement autoregressive text generation

        APPROACH:
        1. Start with prompt tokens
        2. For each new position:
           - Run forward pass to get logits
           - Sample next token from logits
           - Append to sequence
        3. Return generated sequence

        AUTOREGRESSIVE GENERATION:
        At each step, the model predicts the next token based on all
        previous tokens. This is how GPT generates coherent text.

        EXAMPLE:
        >>> model = GPT(vocab_size=100, embed_dim=64, num_layers=2, num_heads=4)
        >>> prompt = Tensor([[1, 2, 3]])  # Some token sequence
        >>> generated = model.generate(prompt, max_new_tokens=5)
        >>> assert generated.shape[1] == 3 + 5  # original + new tokens

        HINT: Use np.random.choice with temperature for sampling
        """
        ### BEGIN SOLUTION
        current_tokens = Tensor(prompt_tokens.data.copy())

        for _ in range(max_new_tokens):
            # Get logits for current sequence
            logits = self.forward(current_tokens)

            # Get logits for last position (next token prediction)
            last_logits = logits.data[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature scaling
            scaled_logits = last_logits / temperature

            # Convert to probabilities (softmax)
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Sample next token
            next_token = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])

            # Append to sequence
            current_tokens = Tensor(np.concatenate([current_tokens.data, next_token], axis=1))

        return current_tokens
        ### END SOLUTION

    def parameters(self):
        """Return all learnable parameters."""
        params = []
        params.extend(self.token_embedding.parameters())
        params.extend(self.position_embedding.parameters())

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())

        return params

# %% [markdown]
"""
### 🔬 Unit Test: GPT Model
This test validates our complete GPT implementation.
**What we're testing**: Model forward pass, shape consistency, generation capability
**Why it matters**: This is the complete language model that ties everything together
**Expected**: Correct output shapes, generation works, parameter counting
"""

# %% nbgrader={"grade": true, "grade_id": "test-gpt", "locked": true, "points": 20}
def test_unit_gpt():
    """🔬 Test GPT model implementation."""
    print("🔬 Unit Test: GPT Model...")

    # Test small GPT model
    vocab_size = 100
    embed_dim = 64
    num_layers = 2
    num_heads = 4

    model = GPT(vocab_size, embed_dim, num_layers, num_heads)

    # Test forward pass
    batch_size, seq_len = 2, 8
    tokens = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    logits = model.forward(tokens)

    # Check output shape
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape

    # Test generation
    prompt = Tensor(np.random.randint(0, vocab_size, (1, 5)))
    generated = model.generate(prompt, max_new_tokens=3)

    # Check generation shape
    assert generated.shape == (1, 8)  # 5 prompt + 3 new tokens

    # Test parameter counting
    params = model.parameters()
    assert len(params) > 10  # Should have many parameters from all components

    # Test different model sizes
    larger_model = GPT(vocab_size=200, embed_dim=128, num_layers=4, num_heads=8)
    test_tokens = Tensor(np.random.randint(0, 200, (1, 10)))
    larger_logits = larger_model.forward(test_tokens)
    assert larger_logits.shape == (1, 10, 200)

    print("✅ GPT model works correctly!")

# Run test immediately when developing this module
if __name__ == "__main__":
    test_unit_gpt()  # Moved after implementation

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

# %% nbgrader={"grade": false, "grade_id": "integration-demo", "solution": true}
def demonstrate_transformer_integration():
    """
    Demonstrate complete transformer pipeline.

    This simulates training a small language model on a simple vocabulary.
    """
    print("🔗 Integration Demo: Complete Language Model Pipeline")
    print("Building a mini-GPT for character-level text generation")

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

    print(f"\nGeneration demo:")
    print(f"Prompt: '{prompt_text}'")

    generated = model.generate(prompt, max_new_tokens=8, temperature=1.0)
    generated_text = ''.join([idx_to_char[idx] for idx in generated.data[0]])

    print(f"Generated: '{generated_text}'")
    print("(Note: Untrained model produces random text)")

    return model

# demonstrate_transformer_integration()  # Moved to __main__ block below

# %% [markdown]
"""
## 📊 Systems Analysis: Parameter Scaling and Memory

Transformer models scale dramatically with size, leading to both opportunities and challenges. Let's analyze the computational and memory requirements to understand why training large language models requires massive infrastructure.

### The Scaling Laws Revolution

One of the key discoveries in modern AI is that transformer performance follows predictable scaling laws:

```
Scaling Laws Pattern:
Performance ∝ Parameters^α × Data^β × Compute^γ

where α ≈ 0.7, β ≈ 0.8, γ ≈ 0.5

This means:
- 10× more parameters → ~5× better performance
- 10× more data → ~6× better performance
- 10× more compute → ~3× better performance
```

### Memory Scaling Analysis

Memory requirements grow in different ways for different components:

```
Memory Scaling by Component:

1. Parameter Memory (Linear with model size):
   - Embeddings: vocab_size × embed_dim
   - Transformer blocks: ~4 × embed_dim²
   - Total: O(embed_dim²)

2. Attention Memory (Quadratic with sequence length):
   - Attention matrices: batch × heads × seq_len²
   - This is why long context is expensive!
   - Total: O(seq_len²)

3. Activation Memory (Linear with batch size):
   - Forward pass activations for backprop
   - Scales with: batch × seq_len × embed_dim
   - Total: O(batch_size)
```

### The Attention Memory Wall

```
┌─────────────────────────────────────────────────────────────────┐
│  ATTENTION MEMORY WALL: Why Long Context is Expensive          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MEMORY USAGE BY SEQUENCE LENGTH (Quadratic Growth):           │
│                                                                 │
│  1K tokens:   [▓] 16 MB                ← Manageable            │
│  2K tokens:   [▓▓▓▓] 64 MB             ← 4× memory (quadratic) │
│  4K tokens:   [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 256 MB   ← 16× memory        │
│  8K tokens:   [████████████████████████████████] 1 GB          │
│  16K tokens:  [████████████████████████████████████████] 4 GB  │
│  32K tokens:  [████████████████████████████████████████████] → │
│               ← extends to 16 GB (off the chart!)              │
│                                                                 │
│  REAL-WORLD CONTEXT LIMITS:                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ GPT-3:     2K tokens  (limited by memory)                 │ │
│  │ GPT-4:     8K tokens  (32K with optimizations)            │ │
│  │ Claude-3:  200K tokens (special techniques required!)     │ │
│  │ GPT-4o:    128K tokens (efficient attention)              │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  MATHEMATICAL SCALING:                                         │
│  Memory = batch_size × num_heads × seq_len² × 4 bytes          │
│                                   ↑                            │
│                          This is the killer!                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-scaling", "solution": true}
def analyze_parameter_scaling():
    """📊 Analyze how parameter count scales with model dimensions."""
    print("📊 Analyzing Parameter Scaling in Transformers...")
    print("Understanding why model size affects performance and cost\n")

    # Test different model sizes
    configs = [
        {"name": "Tiny", "embed_dim": 64, "num_layers": 2, "num_heads": 4},
        {"name": "Small", "embed_dim": 128, "num_layers": 4, "num_heads": 8},
        {"name": "Medium", "embed_dim": 256, "num_layers": 8, "num_heads": 16},
        {"name": "Large", "embed_dim": 512, "num_layers": 12, "num_heads": 16},
    ]

    vocab_size = 50000  # Typical vocabulary size

    for config in configs:
        model = GPT(
            vocab_size=vocab_size,
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"]
        )

        # Count parameters
        total_params = 0
        for param in model.parameters():
            total_params += param.size

        # Calculate memory requirements (4 bytes per float32 parameter)
        memory_mb = (total_params * BYTES_PER_FLOAT32) / MB_TO_BYTES

        print(f"{config['name']} Model:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Memory: {memory_mb:.1f} MB")
        print(f"  Embed dim: {config['embed_dim']}, Layers: {config['num_layers']}")
        print()

    print("💡 Parameter scaling is roughly quadratic with embedding dimension")
    print("🚀 Real GPT-3 has 175B parameters, requiring ~350GB memory!")

if __name__ == "__main__":
    analyze_parameter_scaling()

# %% nbgrader={"grade": false, "grade_id": "analyze-attention-memory", "solution": true}
def analyze_attention_memory():
    """📊 Analyze attention memory complexity with sequence length."""
    print("📊 Analyzing Attention Memory Complexity...")
    print("Why long context is expensive and how it scales\n")

    embed_dim = 512
    num_heads = 8
    batch_size = 4

    # Test different sequence lengths
    sequence_lengths = [128, 256, 512, 1024, 2048]

    print("Attention Matrix Memory Usage:")
    print("Seq Len | Attention Matrix Size | Memory (MB)")
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
    print("🚀 This is why techniques like FlashAttention are crucial for long sequences")

if __name__ == "__main__":
    analyze_attention_memory()

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-module", "locked": true, "points": 25}
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
    test_unit_gpt()

    print("\nRunning integration scenarios...")

    # Test complete transformer training scenario
    print("🔬 Integration Test: Full Training Pipeline...")

    # Create model and data
    vocab_size = 50
    embed_dim = 64
    num_layers = 2
    num_heads = 4

    model = GPT(vocab_size, embed_dim, num_layers, num_heads)

    # Test batch processing
    batch_size = 3
    seq_len = 16
    tokens = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))

    # Forward pass
    logits = model.forward(tokens)
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # Test generation with different temperatures
    prompt = Tensor(np.random.randint(0, vocab_size, (1, 8)))

    # Conservative generation
    conservative = model.generate(prompt, max_new_tokens=5, temperature=0.1)
    assert conservative.shape == (1, 13)

    # Creative generation
    creative = model.generate(prompt, max_new_tokens=5, temperature=2.0)
    assert creative.shape == (1, 13)

    # Test parameter counting consistency
    total_params = sum(param.size for param in model.parameters())
    assert total_params > 1000  # Should have substantial parameters

    print("✅ Full transformer pipeline works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 13")

# Call the comprehensive test
# test_module()  # Only run in __main__ block below

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Transformer Architecture Foundations

### Question 1: Attention Memory Complexity
You implemented multi-head attention that computes attention matrices of size (batch, heads, seq_len, seq_len).

For a model with seq_len=1024, batch_size=4, num_heads=8:
- How many elements in the attention matrix? _____
- If each element is 4 bytes (float32), how much memory per layer? _____ MB
- Why does doubling sequence length quadruple attention memory? _____

### Question 2: Residual Connection Benefits
Your TransformerBlock uses residual connections (x + attention_output, x + mlp_output).

- What happens to gradients during backpropagation without residual connections? _____
- How do residual connections help train deeper networks? _____
- Why is pre-norm (LayerNorm before operations) preferred over post-norm? _____

### Question 3: Parameter Scaling Analysis
Your GPT model combines embeddings, transformer blocks, and output projection.

For embed_dim=512, vocab_size=10000, num_layers=6:
- Token embedding parameters: _____ (vocab_size × embed_dim)
- Approximate parameters per transformer block: _____ (hint: ~4 × embed_dim²)
- Total model parameters: approximately _____ million

### Question 4: Autoregressive Generation Efficiency
Your generate() method processes the full sequence for each new token.

- Why is this inefficient for long sequences? _____
- What optimization caches key-value pairs to avoid recomputation? _____
- How would this change the computational complexity from O(n²) to O(n)? _____
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

    # Input: batch of 2 sequences, 8 tokens each, 64 dims (concrete values)
    x = Tensor(np.ones((2, 8, embed_dim)))

    # Forward pass through transformer block
    output = block.forward(x)

    # Show transformation
    print(f"Input shape:  {x.shape}  (2 sequences, 8 tokens, 64 dimensions)")
    print(f"Output shape: {output.shape}")

    # Verify transformation occurred (values changed)
    input_sum = np.sum(x.data)
    output_sum = np.sum(output.data)
    print(f"\nData transformation:")
    print(f"  Input sum:  {input_sum:.1f}  (initial values: all 1s)")
    print(f"  Output sum: {output_sum:.1f}  (after attention + MLP)")

    print(f"\nTransformerBlock architecture:")
    print(f"  • Multi-head attention ({num_heads} heads)")
    print(f"  • Layer normalization (before operations)")
    print(f"  • MLP ({embed_dim} → {4*embed_dim} → {embed_dim} with GELU)")
    print(f"  • Residual connections (preserve information flow)")

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
- Built full GPT model with embeddings, positional encoding, and autoregressive generation
- Discovered attention memory scaling and parameter distribution patterns
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your transformer implementation is the capstone of the language modeling pipeline.
Export with: `tito module complete 13`

**Next**: Module 14 will add profiling and optimization techniques to make your transformers production-ready!
"""
