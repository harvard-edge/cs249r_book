---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell} python3
:tags: [remove-input, remove-output]
import math
from myst_nb import glue

# --- Multi-Head Attention: head dimension ---
mha_embed_dim = 512
mha_num_heads = 8
mha_head_dim = mha_embed_dim // mha_num_heads
glue("mha_head_dim", f"{mha_embed_dim}/{mha_num_heads}={mha_head_dim}")

# --- Computational Complexity (prose): GPT-3 scale ---
complexity_seq = 2048
complexity_elements = complexity_seq ** 2
complexity_bytes = complexity_elements * 4
complexity_mb = complexity_bytes / 1024**2
complexity_gpt3_layers = 96
complexity_gpt3_attn_gb = complexity_gpt3_layers * complexity_mb / 1024

glue("complexity_elements", f"{complexity_elements:,}")
glue("complexity_mb", f"{complexity_mb:.0f} MB")
glue("complexity_gpt3_attn_gb", f"{complexity_gpt3_attn_gb:.1f} GB")

# --- Computational Complexity (prose): GPT-3 training (5x inference) ---
complexity_train_multiplier = 5
complexity_gpt3_train_gb = complexity_train_multiplier * complexity_gpt3_attn_gb
glue("complexity_gpt3_train_gb", f"~{complexity_gpt3_train_gb:.1f} GB")

# --- Computational Complexity (prose): GPT-4 estimate ---
complexity_gpt4_layers = 120
complexity_gpt4_ctx = 32768
complexity_gpt4_gb = (complexity_gpt4_layers * (complexity_gpt4_ctx ** 2) * 4) / 1024**3
glue("complexity_gpt4_gb", f"~{complexity_gpt4_gb:.0f} GB")

# --- Q1: Memory Calculation ---
q1_seq_a = 1024
q1_elements_a = q1_seq_a ** 2
q1_bytes_a = q1_elements_a * 4
q1_mb_a = q1_bytes_a / 1024**2

q1_seq_b = 2048
q1_elements_b = q1_seq_b ** 2
q1_bytes_b = q1_elements_b * 4
q1_mb_b = q1_bytes_b / 1024**2

q1_scale_factor = (q1_seq_b // q1_seq_a) ** 2
q1_gpt3_layers = 96
q1_gpt3_gb = q1_gpt3_layers * q1_mb_b / 1024

glue("q1_elements_a", f"{q1_elements_a:,}")
glue("q1_mb_a", f"{q1_mb_a:.1f} MB")
glue("q1_elements_b", f"{q1_elements_b:,}")
glue("q1_mb_b", f"{q1_mb_b:.1f} MB")
glue("q1_scale_factor", f"{q1_scale_factor}")
glue("q1_gpt3_layers", f"{q1_gpt3_layers}")
glue("q1_gpt3_mb_b", f"{q1_mb_b:.1f} MB")
glue("q1_gpt3_total_gb", f"{q1_gpt3_gb:.1f} GB")

# --- Q2: Attention Bottleneck ---
q2_d = 512
q2_d_squared = q2_d ** 2
q2_crossover = q2_d_squared // q2_d

glue("q2_d_squared", f"{q2_d_squared:,}")
glue("q2_crossover", f"{q2_crossover}")

# --- Q5: Gradient Memory ---
q5_multiplier = 5
q5_layers = 96
q5_attn_mb = q1_mb_b  # reuse 2048-context value: 16.0 MB
q5_inference_gb = q5_layers * q5_attn_mb / 1024
q5_training_gb = q5_layers * q5_attn_mb * q5_multiplier / 1024

glue("q5_multiplier", f"{q5_multiplier}")
glue("q5_attn_mb", f"{q5_attn_mb:.0f} MB")
glue("q5_inference_gb", f"{q5_inference_gb:.1f} GB")
glue("q5_training_gb", f"{q5_training_gb:.1f} GB")
```

# Module 12: Attention

:::{admonition} Module Info
:class: note

**ARCHITECTURE TIER** | Difficulty: ●●●○ | Time: 5-7 hours | Prerequisites: 01-08, 10-11

**Prerequisites: Modules 01-08 and 10-11** means you should understand:
- Tensor operations and shape manipulation (Module 01)
- Activations, particularly softmax (Module 02)
- Linear layers and weight projections (Module 03)
- Autograd for gradient computation (Module 06)
- Tokenization and embeddings (Modules 10-11)

If you can explain why `softmax(x).sum(axis=-1)` equals 1.0 and how embeddings convert token IDs to dense vectors, you're ready.
:::

`````{only} html
````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/12_attention.mp3" type="audio/mpeg">
</audio>
```

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F12_attention%2Fattention.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/12_attention/12_attention.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #14b8a6; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

````

```{raw} html
<style>
.slide-viewer-container {
  margin: 0.5rem 0 1.5rem 0;
  background: #0f172a;
  border-radius: 1rem;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.slide-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 1rem;
  background: rgba(255,255,255,0.03);
}
.slide-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #94a3b8;
  font-weight: 500;
  font-size: 0.85rem;
}
.slide-subtitle {
  color: #64748b;
  font-weight: 400;
  font-size: 0.75rem;
}
.slide-toolbar {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}
.slide-toolbar button {
  background: transparent;
  border: none;
  color: #64748b;
  width: 32px;
  height: 32px;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 1.1rem;
  transition: all 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
}
.slide-toolbar button:hover {
  background: rgba(249, 115, 22, 0.15);
  color: #f97316;
}
.slide-nav-group {
  display: flex;
  align-items: center;
}
.slide-page-info {
  color: #64748b;
  font-size: 0.75rem;
  padding: 0 0.5rem;
  font-weight: 500;
}
.slide-zoom-group {
  display: flex;
  align-items: center;
  margin-left: 0.25rem;
  padding-left: 0.5rem;
  border-left: 1px solid rgba(255,255,255,0.1);
}
.slide-canvas-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0.5rem 1rem 1rem 1rem;
  min-height: 380px;
  background: #0f172a;
}
.slide-canvas {
  max-width: 100%;
  max-height: 350px;
  height: auto;
  border-radius: 0.5rem;
  box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.slide-progress-wrapper {
  padding: 0 1rem 0.5rem 1rem;
}
.slide-progress-bar {
  height: 3px;
  background: rgba(255,255,255,0.08);
  border-radius: 1.5px;
  overflow: hidden;
  cursor: pointer;
}
.slide-progress-fill {
  height: 100%;
  background: #f97316;
  border-radius: 1.5px;
  transition: width 0.2s ease;
}
.slide-loading {
  color: #f97316;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.slide-loading::before {
  content: '';
  width: 18px;
  height: 18px;
  border: 2px solid rgba(249, 115, 22, 0.2);
  border-top-color: #f97316;
  border-radius: 50%;
  animation: slide-spin 0.8s linear infinite;
}
@keyframes slide-spin {
  to { transform: rotate(360deg); }
}
.slide-footer {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.6rem 1rem;
  background: rgba(255,255,255,0.02);
  border-top: 1px solid rgba(255,255,255,0.05);
}
.slide-footer a {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  background: #f97316;
  color: white;
  padding: 0.4rem 0.9rem;
  border-radius: 2rem;
  text-decoration: none;
  font-weight: 500;
  font-size: 0.75rem;
  transition: all 0.15s;
}
.slide-footer a:hover {
  background: #ea580c;
  color: white;
}
.slide-footer a.secondary {
  background: transparent;
  color: #94a3b8;
  border: 1px solid rgba(255,255,255,0.15);
}
.slide-footer a.secondary:hover {
  background: rgba(255,255,255,0.05);
  color: #f8fafc;
}
@media (max-width: 600px) {
  .slide-header { flex-direction: column; gap: 0.5rem; padding: 0.5rem 0.75rem; }
  .slide-toolbar button { width: 28px; height: 28px; }
  .slide-canvas-wrapper { min-height: 260px; padding: 0.5rem; }
  .slide-canvas { max-height: 220px; }
}
</style>

<div class="slide-viewer-container" id="slide-viewer-12_attention">
  <div class="slide-header">
    <div class="slide-title">
      <span>🔥</span>
      <span>Slide Deck</span>
      <span class="slide-subtitle">· AI-generated</span>
    </div>
    <div class="slide-toolbar">
      <div class="slide-nav-group">
        <button onclick="slideNav('12_attention', -1)" title="Previous">‹</button>
        <span class="slide-page-info"><span id="slide-num-12_attention">1</span> / <span id="slide-count-12_attention">-</span></span>
        <button onclick="slideNav('12_attention', 1)" title="Next">›</button>
      </div>
      <div class="slide-zoom-group">
        <button onclick="slideZoom('12_attention', -0.25)" title="Zoom out">−</button>
        <button onclick="slideZoom('12_attention', 0.25)" title="Zoom in">+</button>
      </div>
    </div>
  </div>
  <div class="slide-canvas-wrapper">
    <div id="slide-loading-12_attention" class="slide-loading">Loading slides...</div>
    <canvas id="slide-canvas-12_attention" class="slide-canvas" style="display:none;"></canvas>
  </div>
  <div class="slide-progress-wrapper">
    <div class="slide-progress-bar" onclick="slideProgress('12_attention', event)">
      <div class="slide-progress-fill" id="slide-progress-12_attention" style="width: 0%;"></div>
    </div>
  </div>
  <div class="slide-footer">
    <a href="../_static/slides/12_attention.pdf" download>⬇ Download</a>
    <a href="#" onclick="slideFullscreen('12_attention'); return false;" class="secondary">⛶ Fullscreen</a>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<script>
(function() {
  if (window.slideViewersInitialized) return;
  window.slideViewersInitialized = true;

  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

  window.slideViewers = {};

  window.initSlideViewer = function(id, pdfUrl) {
    const viewer = { pdf: null, page: 1, scale: 1.3, rendering: false, pending: null };
    window.slideViewers[id] = viewer;

    const canvas = document.getElementById('slide-canvas-' + id);
    const ctx = canvas.getContext('2d');

    function render(num) {
      viewer.rendering = true;
      viewer.pdf.getPage(num).then(function(page) {
        const viewport = page.getViewport({scale: viewer.scale});
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        page.render({canvasContext: ctx, viewport: viewport}).promise.then(function() {
          viewer.rendering = false;
          if (viewer.pending !== null) { render(viewer.pending); viewer.pending = null; }
        });
      });
      document.getElementById('slide-num-' + id).textContent = num;
      document.getElementById('slide-progress-' + id).style.width = (num / viewer.pdf.numPages * 100) + '%';
    }

    function queue(num) { if (viewer.rendering) viewer.pending = num; else render(num); }

    pdfjsLib.getDocument(pdfUrl).promise.then(function(pdf) {
      viewer.pdf = pdf;
      document.getElementById('slide-count-' + id).textContent = pdf.numPages;
      document.getElementById('slide-loading-' + id).style.display = 'none';
      canvas.style.display = 'block';
      render(1);
    }).catch(function() {
      document.getElementById('slide-loading-' + id).innerHTML = 'Unable to load. <a href="' + pdfUrl + '" style="color:#f97316;">Download PDF</a>';
    });

    viewer.queue = queue;
  };

  window.slideNav = function(id, dir) {
    const v = window.slideViewers[id];
    if (!v || !v.pdf) return;
    const newPage = v.page + dir;
    if (newPage >= 1 && newPage <= v.pdf.numPages) { v.page = newPage; v.queue(newPage); }
  };

  window.slideZoom = function(id, delta) {
    const v = window.slideViewers[id];
    if (!v) return;
    v.scale = Math.max(0.5, Math.min(3, v.scale + delta));
    v.queue(v.page);
  };

  window.slideProgress = function(id, event) {
    const v = window.slideViewers[id];
    if (!v || !v.pdf) return;
    const bar = event.currentTarget;
    const pct = (event.clientX - bar.getBoundingClientRect().left) / bar.offsetWidth;
    const newPage = Math.max(1, Math.min(v.pdf.numPages, Math.ceil(pct * v.pdf.numPages)));
    if (newPage !== v.page) { v.page = newPage; v.queue(newPage); }
  };

  window.slideFullscreen = function(id) {
    const el = document.getElementById('slide-viewer-' + id);
    if (el.requestFullscreen) el.requestFullscreen();
    else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
  };
})();

initSlideViewer('12_attention', '../_static/slides/12_attention.pdf');
</script>
```
`````

## Overview

The attention mechanism revolutionized deep learning by solving a fundamental problem: how can models focus on relevant information when processing sequences? Before attention, models like RNNs compressed entire sequences into fixed-size hidden states, creating an information bottleneck. Attention changes this by allowing every position in a sequence to directly access information from every other position, weighted by relevance.

You'll build scaled dot-product attention and multi-head attention from scratch, the exact mechanisms powering GPT, BERT, and modern transformers. By implementing the core formula `Attention(Q, K, V) = softmax(QK^T / √d_k) V` with vectorized matrix operations, you'll witness the O(n²) memory complexity that makes attention both powerful and challenging at scale. This hands-on implementation reveals why research into efficient attention variants like FlashAttention is crucial for production systems.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** scaled dot-product attention with vectorized operations that reveal O(n²) memory complexity
- **Build** multi-head attention for parallel processing of different relationship types across representation subspaces
- **Master** attention weight computation, normalization, and the query-key-value paradigm
- **Understand** quadratic memory scaling and why attention becomes the bottleneck in long-context transformers
- **Connect** your implementation to production frameworks and understand why efficient attention research matters at scale
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Attention System
flowchart LR
    subgraph "Your Attention System"
        A["Query (Q)<br/>What to look for"]
        B["Key (K)<br/>What's available"]
        C["Value (V)<br/>What to retrieve"]
        D["Similarity Scores<br/>QK^T / √d_k"]
        E["Attention Weights<br/>softmax(scores)"]
        F["Weighted Output<br/>weights @ V"]
    end

    A --> D
    B --> D
    D --> E
    E --> F
    C --> F

    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D fill:#fff3cd
    style E fill:#f8d7da
    style F fill:#d4edda
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `scaled_dot_product_attention()` | Core attention mechanism with QK^T similarity |
| 2 | Attention weight normalization | Softmax converts scores to probability distribution |
| 3 | Causal masking support | Preventing attention to future positions |
| 4 | `MultiHeadAttention.__init__()` | Linear projections and head configuration |
| 5 | `MultiHeadAttention.forward()` | Split, attend, concatenate pattern |

**The pattern you'll enable:**
```python
# Multi-head attention for sequence processing
mha = MultiHeadAttention(embed_dim=512, num_heads=8)
output = mha(embeddings, mask)  # Learn different relationship types in parallel
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Full transformer blocks (that's Module 13: Transformers)
- Positional encoding (you built this in Module 11: Embeddings)
- Efficient attention variants like FlashAttention (production optimization beyond scope)
- Cross-attention for encoder-decoder models (PyTorch does this with separate Q vs K/V inputs)

**You are building the core attention mechanism.** Complete transformer architectures come next.

## API Reference

This section provides a quick reference for the attention functions and classes you'll build. Use this as your implementation guide and debugging reference.

### Scaled Dot-Product Attention Function

```python
scaled_dot_product_attention(Q, K, V, mask=None) -> (output, attention_weights)
```

Computes the fundamental attention operation that powers all transformers.

**Parameters:**
- `Q`: Query tensor `(batch_size, seq_len, d_model)` - what each position is looking for
- `K`: Key tensor `(batch_size, seq_len, d_model)` - what's available at each position
- `V`: Value tensor `(batch_size, seq_len, d_model)` - actual content to retrieve
- `mask`: Optional `(batch_size, seq_len, seq_len)` - 1.0 for allowed positions, 0.0 for masked

**Returns:**
- `output`: Attended values `(batch_size, seq_len, d_model)`
- `attention_weights`: Attention matrix `(batch_size, seq_len, seq_len)` showing focus patterns

### MultiHeadAttention Class

Multi-head attention runs multiple attention mechanisms in parallel, each learning to focus on different types of relationships.

**Constructor:**
```python
MultiHeadAttention(embed_dim, num_heads) -> MultiHeadAttention
```

Creates multi-head attention with `embed_dim // num_heads` dimensions per head.

**Core Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `forward(x, mask=None) -> Tensor` | Apply multi-head attention to input |
| `parameters` | `parameters() -> List[Tensor]` | Return all trainable parameters |

**Attributes:**
- `embed_dim`: Total embedding dimension
- `num_heads`: Number of parallel attention heads
- `head_dim`: Dimension per head (embed_dim // num_heads)
- `q_proj`, `k_proj`, `v_proj`: Linear projections for queries, keys, values
- `out_proj`: Output linear layer to mix information across heads

## Core Concepts

This section covers the fundamental ideas you need to understand attention deeply. These concepts apply to every transformer-based model in production today.

### Query, Key, Value: The Information Retrieval Paradigm

Attention models sequence processing as an information retrieval problem. Think of it like searching a database: you have a query describing what you need, keys that describe what each entry contains, and values representing the actual data. When you search "machine learning papers," the search engine compares your query against document descriptions (keys) to determine relevance, then returns the actual documents (values) weighted by how well they match.

The same pattern applies to transformers. For each position in a sequence, the query asks "what information do I need?", keys describe "what information do I have?", and values contain the actual representations to retrieve. The beauty is that queries, keys, and values are all learned projections of the same input embeddings, allowing the model to discover what aspects of meaning to search for and retrieve.

Here's how your implementation creates these three components through linear projections:

```python
# From MultiHeadAttention.__init__
self.q_proj = Linear(embed_dim, embed_dim)  # Learn what to search for
self.k_proj = Linear(embed_dim, embed_dim)  # Learn what to index by
self.v_proj = Linear(embed_dim, embed_dim)  # Learn what to retrieve

# From MultiHeadAttention.forward
Q = self.q_proj.forward(x)  # Transform input to queries
K = self.k_proj.forward(x)  # Transform input to keys
V = self.v_proj.forward(x)  # Transform input to values
```

Each linear projection learns a different perspective on the input. During training, the model discovers that queries might emphasize semantic meaning, keys might emphasize syntactic roles, and values might emphasize contextual information, all optimized end-to-end for the task.

### Scaled Dot-Product Attention: Similarity as Relevance

The core attention computation answers a simple question: how similar is each query to each key? The dot product between vectors naturally measures similarity, where higher values indicate more aligned directions in embedding space. For a query at position i and key at position j, the dot product `Q[i] · K[j]` quantifies their relevance.

But raw dot products grow with embedding dimension, creating numerical instability in softmax. With 512-dimensional embeddings, dot products can reach hundreds, causing softmax to saturate (output probabilities near 0 or 1 with tiny gradients). Scaling by `1/√d_k` normalizes the variance, keeping values in a stable range regardless of embedding dimension.

Your implementation computes this using vectorized matrix operations:

```python
# From scaled_dot_product_attention (lines 303-319)
d_model = Q.shape[-1]

# Compute all query-key similarities at once using matmul
# This is mathematically equivalent to nested loops computing Q[i] · K[j]
# for all i,j pairs, but vectorized for efficiency
K_t = K.transpose(-2, -1)  # Transpose to align dimensions
scores = Q.matmul(K_t)     # (batch, seq_len, seq_len) - the O(n²) matrix

# Scale by 1/√d_k for numerical stability
scale_factor = 1.0 / math.sqrt(d_model)
scores = scores * scale_factor
```

The resulting `scores` tensor is the attention matrix before normalization. Element `[i,j]` represents how much position i should attend to position j. The vectorized `matmul` operation computes all n² query-key pairs simultaneously—while much faster than Python loops, it still creates the full O(n²) attention matrix that dominates memory usage at scale.

### Attention Weights and Softmax Normalization

Raw similarity scores need to become a probability distribution. Softmax transforms scores into positive values that sum to 1.0 along each row, creating a proper weighted average. This ensures that for each query position, the attention weights over all key positions form valid mixing coefficients.

The softmax operation `exp(scores[i,j]) / Σ_k exp(scores[i,k])` has important properties. It's differentiable, allowing gradients to flow during training. It amplifies differences: a score of 2.0 becomes much more prominent than 1.0 after exponentiation. And it's translation-invariant: adding the same constant to all scores doesn't change the output (exploited for numerical stability).

Here's the complete attention weight computation with masking support:

```python
# From scaled_dot_product_attention

# Apply causal mask if provided (set masked positions to large negative)
if mask is not None:
    mask_data = mask.data
    adder_mask = (1.0 - mask_data) * MASK_VALUE  # MASK_VALUE = -1e9
    adder_mask_tensor = Tensor(adder_mask, requires_grad=False)
    scores = scores + adder_mask_tensor

# Softmax converts scores to probability distribution
softmax = Softmax()
attention_weights = softmax(scores, dim=-1)  # Normalize along last dimension

# Apply to values: weighted combination
output = attention_weights.matmul(V)
```

The mask addition is clever: for positions where `mask=0` (masked), we add -1e9 to the score. After softmax, `exp(-1e9)` is effectively zero, so masked positions get zero attention weight. For positions where `mask=1` (allowed), adding zero leaves scores unchanged. This preserves differentiability while enforcing hard constraints.

### Multi-Head Attention: Parallel Relationship Learning

Single-head attention learns one similarity function between queries and keys. But sequences have multiple types of relationships: syntactic dependencies, semantic similarity, positional patterns, long-range coreference. Multi-head attention addresses this by running multiple attention mechanisms in parallel, each with different learned projections.

The key insight is splitting the embedding dimension across heads rather than duplicating it. For `embed_dim=512` and `num_heads=8`, each head operates on {glue:text}`mha_head_dim` dimensions. This keeps parameter count constant while allowing diverse specialization. One head might learn to focus on adjacent tokens (local syntax), another on semantically similar words (meaning), another on specific positional offsets (structured patterns).

Your implementation handles this through reshape and transpose operations:

```python
# From MultiHeadAttention.forward

# Project to Q, K, V (each is batch, seq, embed_dim)
Q = self.q_proj.forward(x)
K = self.k_proj.forward(x)
V = self.v_proj.forward(x)

# Reshape to separate heads: (batch, seq, num_heads, head_dim)
Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

# Transpose to (batch, num_heads, seq, head_dim) for parallel processing
Q = Q.transpose(1, 2)
K = K.transpose(1, 2)
V = V.transpose(1, 2)

# Apply attention to all heads at once
attended, _ = scaled_dot_product_attention(Q, K, V, mask=mask_reshaped)

# Transpose back and concatenate heads
attended = attended.transpose(1, 2)  # (batch, seq, num_heads, head_dim)
concat_output = attended.reshape(batch_size, seq_len, self.embed_dim)

# Mix information across heads with output projection
output = self.out_proj.forward(concat_output)
```

The reshape-transpose-attend-transpose-reshape dance separates heads for independent processing, then recombines their outputs. The final output projection learns how to mix information discovered by different heads, creating a rich representation that captures multiple relationship types simultaneously.

### Causal Masking: Preventing Information Leakage

In language modeling, predicting the next token requires a strict causality constraint: position i can only attend to positions 0 through i, never to future positions. Without this, the model could "cheat" by looking at the answer during training. Causal masking enforces this by zeroing attention weights for all positions j > i.

The implementation uses a lower triangular mask: ones below and on the diagonal, zeros above. For a sequence length of 4, the mask looks like:

```
[[1, 0, 0, 0],   # Position 0 can only see itself
 [1, 1, 0, 0],   # Position 1 sees 0 and 1
 [1, 1, 1, 0],   # Position 2 sees 0, 1, 2
 [1, 1, 1, 1]]   # Position 3 sees all positions
```

When combined with the masking logic in attention (adding -1e9 to masked scores before softmax), this creates a structured sparsity pattern: exactly half the attention matrix becomes zero. This is crucial for autoregressive models like GPT, where generation must proceed left-to-right without access to future tokens.

### Computational Complexity: The O(n²) Reality

Attention's power comes from all-to-all connectivity: every position can attend to every other position. But this creates quadratic scaling in both computation and memory. For sequence length n, the attention matrix has n² elements. The vectorized `Q @ K^T` operation computes all n² similarity scores in one matrix multiplication, softmax normalizes n² values, and applying attention to values multiplies n² weights by the value vectors.

The memory cost is particularly severe. For GPT-3 with 2048-token context, a single attention matrix stores 2048² = {glue:text}`complexity_elements` float32 values, requiring {glue:text}`complexity_mb`. With 96 layers, attention matrices alone need {glue:text}`complexity_gpt3_attn_gb`, excluding activations, gradients, and other tensors. This quadratic wall is why long-context AI remains an active research challenge.

| Operation | Time Complexity | Memory Complexity | Dominates When |
|-----------|----------------|-------------------|----------------|
| QK^T | O(n² × d) | O(n²) | Long sequences |
| Softmax | O(n²) | O(n²) | Always stores full matrix |
| Weights @ V | O(n² × d) | O(n × d) | Output reuses attention weights |
| **Total** | **O(n² × d)** | **O(n²)** | n > d (long sequences) |

For comparison, feed-forward networks in transformers have O(n × d²) complexity. When sequence length n exceeds embedding dimension d (common in modern models), attention's O(n²) term dominates, making it the primary bottleneck. This explains why research into efficient attention variants like sparse attention, linear attention, and FlashAttention is crucial for production systems.

## Common Errors

These are the errors you'll encounter most often when implementing attention. Understanding them will save hours of debugging.

### Shape Mismatch in Attention

**Error**: `ValueError: Cannot perform matrix multiplication: (2, 4, 64) @ (2, 4, 64). Inner dimensions must match`

When computing `Q @ K^T`, the key tensor needs transposing. The matrix multiplication `Q @ K` has shape `(batch, seq_len, d_model) @ (batch, seq_len, d_model)`, which fails because the inner dimensions are both `d_model`. You need `Q @ K.transpose()` to get `(batch, seq_len, d_model) @ (batch, d_model, seq_len)`, producing the correct `(batch, seq_len, seq_len)` attention matrix.

**Fix**: Always transpose K before the matmul: `scores = Q.matmul(K.transpose(-2, -1))`

### Attention Weights Don't Sum to 1

**Error**: `AssertionError: Attention weights don't sum to 1`

This happens when softmax is applied to the wrong axis. Attention weights must form a probability distribution over key positions for each query position. If you apply softmax along the wrong dimension, you'll get values that don't sum to 1.0 per row.

**Fix**: Use `softmax(scores, dim=-1)` to normalize along the last dimension (across keys for each query)

### Multi-Head Dimension Mismatch

**Error**: `ValueError: embed_dim (512) must be divisible by num_heads (7)`

Multi-head attention splits the embedding dimension across heads. If `embed_dim=512` and `num_heads=7`, you'd get `512/7=73.14` dimensions per head, which doesn't work with integer tensor shapes. The architecture requires exact divisibility.

**Fix**: Choose num_heads that evenly divides embed_dim. Common pairs: (512, 8), (768, 12), (1024, 16)

### Mask Broadcasting Errors

**Error**: `ValueError: operands could not be broadcast together with shapes (2,1,4,4) (2,4,4)`

Multi-head attention expects masks with a head dimension. If you pass a 3D mask `(batch, seq, seq)` but the implementation expects 4D `(batch, heads, seq, seq)`, broadcasting fails. The mask needs reshaping to add a dimension that broadcasts across all heads.

**Fix**: Reshape mask: `mask.reshape(batch, 1, seq_len, seq_len)` to broadcast over heads

### Gradient Flow Issues

**Error**: Loss doesn't decrease during training despite correct forward pass

This can happen if you create new Tensor objects incorrectly, breaking the autograd graph. When applying masks or performing intermediate computations, ensure tensors maintain `requires_grad` appropriately.

**Fix**: Check that operations preserve gradient flow: `Tensor(result, requires_grad=True)` when needed

## Production Context

### Your Implementation vs. PyTorch

Your TinyTorch attention and PyTorch's `nn.MultiheadAttention` implement the same mathematical operations. The differences are in implementation efficiency, features, and flexibility. PyTorch uses highly optimized C++ kernels, supports additional attention variants, and integrates with production training systems.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Core Algorithm** | Scaled dot-product attention | Same mathematical operation |
| **Multi-Head** | Split-attend-concat pattern | Identical architecture |
| **Backend** | NumPy (Python loops) | C++ CUDA kernels |
| **Speed** | 1x (baseline) | 50-100x faster on GPU |
| **Memory Optimization** | Stores full attention matrix | Optional FlashAttention integration |
| **Batch First** | `(batch, seq, embed)` | Configurable via `batch_first=True` |
| **Cross-Attention** | Self-attention only | Separate Q vs K/V inputs supported |
| **Key Padding Mask** | Manual mask creation | Built-in mask utilities |

### Code Comparison

The following comparison shows equivalent attention operations in TinyTorch and PyTorch. Notice how the high-level API and shape conventions match almost exactly.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.core.attention import MultiHeadAttention
from tinytorch.core.tensor import Tensor
import numpy as np

# Create multi-head attention
mha = MultiHeadAttention(embed_dim=512, num_heads=8)

# Input embeddings (batch=2, seq=10, dim=512)
x = Tensor(np.random.randn(2, 10, 512))

# Apply attention
output = mha.forward(x)  # (2, 10, 512)

# With causal masking
mask = Tensor(np.tril(np.ones((2, 10, 10))))
output_masked = mha.forward(x, mask)
```
````

````{tab-item} ⚡ PyTorch
```python
import torch
import torch.nn as nn

# Create multi-head attention
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8,
                             batch_first=True)

# Input embeddings (batch=2, seq=10, dim=512)
x = torch.randn(2, 10, 512)

# Apply attention (PyTorch returns output + weights)
output, weights = mha(x, x, x)  # Self-attention: Q=K=V=x

# With causal masking (upper triangle = -inf)
mask = torch.triu(torch.ones(10, 10) * float('-inf'), diagonal=1)
output_masked, _ = mha(x, x, x, attn_mask=mask)
```
````
`````

Let's walk through the key differences:

- **Line 1-2 (Imports)**: TinyTorch separates attention into its own module; PyTorch includes it in `torch.nn`. Both follow modular design patterns.
- **Line 4-5 (Construction)**: API is nearly identical. PyTorch adds `batch_first=True` for compatibility with older code that expected `(seq, batch, embed)` order.
- **Line 8 (Input)**: Shape conventions match exactly: `(batch, seq, embed)`. This is the modern standard.
- **Line 11 (Forward Pass)**: TinyTorch uses `mha.forward(x)` with x as both Q, K, V (self-attention). PyTorch makes this explicit with `mha(x, x, x)`, allowing cross-attention where Q differs from K/V.
- **Line 14-15 (Masking)**: TinyTorch uses 0/1 masks (0=masked). PyTorch uses additive masks (-inf=masked). Both work, but PyTorch's convention integrates better with certain optimizations.

```{tip} What's Identical

The mathematical operations, architectural patterns, and shape conventions are identical. Multi-head attention works the same way in production. Understanding your implementation means understanding PyTorch's attention.
```

### Why Attention Matters at Scale

To appreciate why attention research is crucial, consider the scaling characteristics of modern language models:

- **GPT-3** (96 layers, 2048 context): ~{glue:text}`complexity_gpt3_attn_gb` just for attention matrices during forward pass, {glue:text}`complexity_gpt3_train_gb` with gradients during training
- **GPT-4** (estimated 120 layers, 32K context): Would require {glue:text}`complexity_gpt4_gb` for attention alone without optimization, exceeding single-GPU memory
- **Long-context models** (100K+ tokens): Attention becomes computationally prohibitive without algorithmic improvements

These constraints drive modern attention research:

- **FlashAttention**: Reformulates computation to reduce memory from O(n²) to O(n) without approximation, enabling 8x longer contexts
- **Sparse Attention**: Only compute attention for specific patterns (local windows, strided access), reducing complexity to O(n log n) or O(n√n)
- **Linear Attention**: Approximate attention with linear complexity O(n), trading accuracy for scale
- **State Space Models**: Alternative architectures (Mamba, RWKV) that avoid attention's quadratic cost entirely

The attention mechanism you built is mathematically identical to production systems, but the O(n²) wall explains why so much research focuses on making it tractable at scale.

## Check Your Understanding

Test yourself with these systems thinking questions. They're designed to build intuition for the performance characteristics you'll encounter in production ML.

**Q1: Memory Calculation**

For sequence length 1024, how much memory does a single attention matrix require (float32)? What about sequence length 2048?

```{admonition} Answer
:class: dropdown

**Sequence length 1024:**
- Attention matrix: 1024 × 1024 = {glue:text}`q1_elements_a` elements
- Memory: {glue:text}`q1_elements_a` × 4 bytes = **{glue:text}`q1_mb_a`**

**Sequence length 2048:**
- Attention matrix: 2048 × 2048 = {glue:text}`q1_elements_b` elements
- Memory: {glue:text}`q1_elements_b` × 4 bytes = **{glue:text}`q1_mb_b`**

**Scaling factor:** Doubling sequence length quadruples memory (2² = {glue:text}`q1_scale_factor`×)

For GPT-3 ({glue:text}`q1_gpt3_layers` layers, 2048 context):
- {glue:text}`q1_gpt3_layers` layers × {glue:text}`q1_gpt3_mb_b` = **{glue:text}`q1_gpt3_total_gb`** just for attention matrices!
- This excludes Q/K/V projections, gradients, and all other tensors.
```

**Q2: Attention Bottleneck**

A transformer layer has attention (O(n² × d)) and feed-forward network (O(n × d²)). For embed_dim=512, at what sequence length does attention dominate?

```{admonition} Answer
:class: dropdown

**Complexity comparison:**
- Attention: O(n² × d) = O(n² × 512)
- FFN: O(n × d²) = O(n × 512²) = O(n × {glue:text}`q2_d_squared`)

**Crossover point:** n² × 512 > n × {glue:text}`q2_d_squared`
- Simplify: n > {glue:text}`q2_d_squared` / 512 = **{glue:text}`q2_crossover`**

**When n > {glue:text}`q2_crossover`**, attention becomes the memory bottleneck.

**Real-world implications:**
- Short sequences (n=128): FFN dominates, 262K vs 8K operations
- Medium sequences (n=512): Break-even point
- Long sequences (n=2048): Attention dominates, 2M vs 262K operations
- **This is why GPT-3 (2048 context) needed attention optimization!**
```

**Q3: Multi-Head Efficiency**

Why use 8 heads of 64 dimensions instead of 1 head of 512 dimensions? Parameters are the same—what's the systems difference?

```{admonition} Answer
:class: dropdown

**Parameter count (both are identical):**
- 8 heads × 64 dims: Linear(512→512) for Q, K, V, Out = 4 × (512×512 + 512) weights+biases
- 1 head × 512 dims: Same projection parameters

**Key differences:**

**1. Parallelization:**
- 8 heads can process in parallel on modern GPUs (separate CUDA streams)
- Each head's smaller matmul operations utilize GPU cores more efficiently

**2. Representation diversity:**
- 8 heads learn 8 different similarity functions (syntax, semantics, position, etc.)
- 1 head learns a single monolithic similarity function
- Training discovers specialization automatically

**3. Cache efficiency:**
- Smaller head_dim (64) fits better in GPU cache/shared memory
- Larger single head (512) causes more cache misses

**4. Gradient flow:**
- Multiple heads provide diverse gradient signals during backpropagation
- Single head has one gradient path, slower learning

**Empirical result:** 8 heads consistently outperform 1 head despite same parameter count. Diversity matters!
```

**Q4: Causal Masking Computation**

Causal masking zeros out the upper triangle (roughly half the attention matrix). Do we save computation, or just ensure correctness?

````{admonition} Answer
:class: dropdown

**In your implementation: NO computation saved**

Your code computes the full attention matrix, then adds -1e9 to masked positions:
```python
scores = Q.matmul(K_t)  # Full n² computation
scores = scores + adder_mask_tensor  # Masking happens after
```

**Why no savings:**
- `Q.matmul(K_t)` computes all n² scores
- Masking only affects softmax, not the initial computation
- We still store and normalize the full matrix

**To actually save computation, you'd need:**
1. Sparse matrix multiplication (skip masked positions in matmul)
2. Only compute lower triangle of scores
3. Specialized CUDA kernels that exploit sparsity

**Production optimizations:**
- PyTorch's standard attention: Also computes full matrix (same as yours)
- FlashAttention: Uses tiling to avoid full matrix but doesn't exploit sparsity
- Sparse attention (BigBird, Longformer): Actually skips computation for sparse patterns

**Memory could be saved:** Store only lower triangle (n²/2 elements), but requires custom indexing
````

**Q5: Gradient Memory**

Training attention requires storing activations for backpropagation. How much memory does training need compared to inference?

```{admonition} Answer
:class: dropdown

**Forward pass (inference):**
- Attention matrix: n² values

**Backward pass (training) additional memory:**
- Gradient of attention weights: n² values
- Gradient of Q, K, V: 3 × (n × d) values
- Intermediate gradients from softmax: n² values

**With Adam optimizer (standard for transformers):**
- First moment (momentum): n² values
- Second moment (velocity): n² values

**Total multiplier for attention matrix alone:**
- Forward: 1× (attention weights)
- Backward: +2× (gradients)
- Optimizer: +2× (Adam state)
- **Total: {glue:text}`q5_multiplier`× inference memory**

**For GPT-3 scale (96 layers, 2048 context):**
- Inference: 96 × {glue:text}`q5_attn_mb` = {glue:text}`q5_inference_gb`
- Training: 96 × {glue:text}`q5_attn_mb` × {glue:text}`q5_multiplier` = **{glue:text}`q5_training_gb`** just for attention gradients and optimizer state!

This excludes Q/K/V matrices, feed-forward networks, embeddings, and activations from other layers. Full GPT-3 training requires 350+ GB.
```

## Further Reading

For students who want to understand the academic foundations and explore the research that created modern transformers:

### Seminal Papers

- **Attention Is All You Need** - Vaswani et al. (2017). The paper that introduced transformers and the multi-head attention mechanism you just built. Shows how attention alone, without recurrence, achieves state-of-the-art results. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- **BERT: Pre-training of Deep Bidirectional Transformers** - Devlin et al. (2018). Demonstrates how bidirectional attention (no causal mask) enables powerful language understanding. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

- **Language Models are Unsupervised Multitask Learners (GPT-2)** - Radford et al. (2019). Shows how causal attention with your masking pattern enables autoregressive language modeling at scale. [OpenAI](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- **FlashAttention: Fast and Memory-Efficient Exact Attention** - Dao et al. (2022). Addresses the O(n²) memory bottleneck you experienced, achieving 2-4× speedups without approximation. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

### Additional Resources

- **Blog post**: "The Illustrated Transformer" by Jay Alammar - Visual explanations of attention mechanics that complement your implementation
- **Interactive tool**: BertViz - Visualize attention patterns in trained models to see the specialization you enabled with multi-head attention
- **Textbook**: "Speech and Language Processing" (Jurafsky & Martin, Chapter 9) - Formal treatment of attention in sequence-to-sequence models

## What's Next

```{seealso} Coming Up: Module 13 - Transformers

Build complete transformer blocks by combining your attention mechanism with feed-forward networks, layer normalization, and residual connections. You'll assemble the architecture behind GPT, BERT, and modern language models.
```

**Preview - How Your Attention Gets Used in Future Modules:**

| Module | What It Does | Your Attention In Action |
|--------|--------------|--------------------------|
| **13: Transformers** | Complete transformer blocks | `TransformerLayer(attention + FFN + LayerNorm)` |
| **13: Transformers** | Positional encoding | Attention on position-aware embeddings |
| **13: Transformers** | Stacked layers | `attention → FFN → attention → FFN...` |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/12_attention/attention.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/12_attention/12_attention.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
