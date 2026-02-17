---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# --- MLP section: embed_dim=512, hidden_dim=2048 ---
mlp_linear1 = 512 * 2048 + 2048
mlp_linear2 = 2048 * 512 + 512
mlp_total = mlp_linear1 + mlp_linear2
mlp_total_12layers = 12 * mlp_total

glue("mlp_linear1", f"{mlp_linear1:,}")
glue("mlp_linear1_approx", f"{mlp_linear1 / 1e6:.2f}M")
glue("mlp_linear2", f"{mlp_linear2:,}")
glue("mlp_linear2_approx", f"{mlp_linear2 / 1e6:.2f}M")
glue("mlp_total", f"{mlp_total:,}")
glue("mlp_total_approx", f"{mlp_total / 1e6:.1f}M")
glue("mlp_12layer_approx", f"{mlp_total_12layers / 1e6:.1f}M")

# --- Parameter table: embed_dim=512, num_heads=8 ---
attn_params_512 = 4 * (512 * 512)
ln_params_512 = 2 * 512
mlp_params_512 = (512 * 2048 + 2048) + (2048 * 512 + 512)
block_total_512 = attn_params_512 + ln_params_512 + mlp_params_512 + ln_params_512

glue("attn_params_512", f"~{attn_params_512 / 1e6:.2f}M")
glue("attn_params_512_raw", f"{attn_params_512:,}")
glue("ln_params_512", f"{ln_params_512:,}")
glue("ln_params_512_approx", f"{ln_params_512 / 1e3:.0f}K")
glue("mlp_params_512", f"~{mlp_params_512 / 1e6:.1f}M")
glue("block_total_512", f"~{block_total_512 / 1e6:.1f}M")

# --- GPT model totals: vocab=50000, embed_dim=512, seq=2048, layers=12 ---
tok_emb_512 = 50000 * 512
pos_emb_512 = 2048 * 512
blocks_total_512 = 12 * block_total_512
gpt_total_512 = tok_emb_512 + pos_emb_512 + blocks_total_512

glue("tok_emb_512", f"{tok_emb_512 / 1e6:.1f}M")
glue("tok_emb_512_raw", f"{tok_emb_512:,}")
glue("pos_emb_512", f"{pos_emb_512 / 1e6:.1f}M")
glue("pos_emb_512_raw", f"{pos_emb_512:,}")
glue("blocks_total_512", f"{blocks_total_512 / 1e6:.1f}M")
glue("blocks_total_512_formula", f"12 x {block_total_512 / 1e6:.1f}M = {blocks_total_512 / 1e6:.1f}M")
glue("gpt_total_512", f"~{gpt_total_512 / 1e6:.0f}M")

# --- Attention memory table: batch=4, heads=8, float32 ---
KB = 1024
MB = 1024 ** 2
GB = 1024 ** 3

def attn_mem_mb(batch, heads, seq):
    return batch * heads * seq * seq * 4 / MB

attn_512 = attn_mem_mb(4, 8, 512)
attn_1024 = attn_mem_mb(4, 8, 1024)
attn_2048 = attn_mem_mb(4, 8, 2048)
attn_4096 = attn_mem_mb(4, 8, 4096)

glue("attn_mem_512", f"{attn_512:.1f}")
glue("attn_mem_1024", f"{attn_1024:.1f}")
glue("attn_mem_2048", f"{attn_2048:.1f}")
glue("attn_mem_4096", f"{attn_4096:.1f}")

# --- Q1: Attention memory calc: batch=8, heads=16, seq=2048/4096 ---
q1_elements_2048 = 8 * 16 * 2048 * 2048
q1_bytes_2048 = q1_elements_2048 * 4
q1_gb_2048 = q1_bytes_2048 / GB

q1_elements_4096 = 8 * 16 * 4096 * 4096
q1_bytes_4096 = q1_elements_4096 * 4
q1_gb_4096 = q1_bytes_4096 / GB

glue("q1_elements_2048", f"{q1_elements_2048:,}")
glue("q1_bytes_2048", f"{q1_bytes_2048:,}")
glue("q1_gb_2048", f"{q1_gb_2048:.1f}")
glue("q1_elements_4096", f"{q1_elements_4096:,}")
glue("q1_gb_4096", f"{q1_gb_4096:.1f}")

# --- Q2: Parameter distribution: vocab=50000, embed=768, layers=12, heads=12 ---
q2_tok_emb = 50000 * 768
q2_pos_emb = 2048 * 768
q2_attn_per_block = 4 * (768 * 768)
q2_mlp_per_block = (768 * 3072 + 3072) + (3072 * 768 + 768)
q2_per_block = q2_attn_per_block + q2_mlp_per_block
q2_total_blocks = 12 * q2_per_block
q2_total_emb = q2_tok_emb + q2_pos_emb
q2_grand_total = q2_tok_emb + q2_pos_emb + q2_total_blocks

glue("q2_tok_emb", f"{q2_tok_emb / 1e6:.1f}M")
glue("q2_tok_emb_raw", f"{q2_tok_emb:,}")
glue("q2_pos_emb", f"{q2_pos_emb / 1e6:.1f}M")
glue("q2_pos_emb_raw", f"{q2_pos_emb:,}")
glue("q2_attn_per_block", f"~{q2_attn_per_block / 1e6:.1f}M")
glue("q2_attn_per_block_raw", f"{q2_attn_per_block:,}")
glue("q2_mlp_per_block", f"~{q2_mlp_per_block / 1e6:.1f}M")
glue("q2_mlp_per_block_raw", f"{q2_mlp_per_block:,}")
glue("q2_per_block", f"{q2_per_block / 1e6:.1f}M")
glue("q2_total_blocks", f"{q2_total_blocks / 1e6:.0f}M")
glue("q2_total_blocks_raw", f"{q2_total_blocks:,}")
glue("q2_total_emb", f"{q2_total_emb / 1e6:.0f}M")
glue("q2_grand_total", f"~{q2_grand_total / 1e6:.0f}M")

# --- Q4: Generation efficiency: prompt=50, gen=100 ---
q4_total_processings = sum(range(50, 150))
q4_optimized = 50 + 100
q4_speedup = q4_total_processings / q4_optimized

glue("q4_total_processings", f"{q4_total_processings:,}")
glue("q4_optimized", f"{q4_optimized:,}")
glue("q4_speedup", f"{q4_speedup:.0f}")
```

# Module 13: Transformers

:::{admonition} Module Info
:class: note

**ARCHITECTURE TIER** | Difficulty: ●●●● | Time: 8-10 hours | Prerequisites: 01-08, 10-12

**Prerequisites: Modules 01-08 and 10-12** means you need a strong foundation across three domains. This module assumes you've implemented tensors, layers, training loops, tokenization, embeddings, and attention mechanisms. If you can explain how multi-head attention processes queries, keys, and values to compute weighted representations, you're ready.
:::

`````{only} html
````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/13_transformers.mp3" type="audio/mpeg">
</audio>
```

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F13_transformers%2Ftransformers.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/13_transformers/13_transformers.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #14b8a6; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
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

<div class="slide-viewer-container" id="slide-viewer-13_transformers">
  <div class="slide-header">
    <div class="slide-title">
      <span>🔥</span>
      <span>Slide Deck</span>
      <span class="slide-subtitle">· AI-generated</span>
    </div>
    <div class="slide-toolbar">
      <div class="slide-nav-group">
        <button onclick="slideNav('13_transformers', -1)" title="Previous">‹</button>
        <span class="slide-page-info"><span id="slide-num-13_transformers">1</span> / <span id="slide-count-13_transformers">-</span></span>
        <button onclick="slideNav('13_transformers', 1)" title="Next">›</button>
      </div>
      <div class="slide-zoom-group">
        <button onclick="slideZoom('13_transformers', -0.25)" title="Zoom out">−</button>
        <button onclick="slideZoom('13_transformers', 0.25)" title="Zoom in">+</button>
      </div>
    </div>
  </div>
  <div class="slide-canvas-wrapper">
    <div id="slide-loading-13_transformers" class="slide-loading">Loading slides...</div>
    <canvas id="slide-canvas-13_transformers" class="slide-canvas" style="display:none;"></canvas>
  </div>
  <div class="slide-progress-wrapper">
    <div class="slide-progress-bar" onclick="slideProgress('13_transformers', event)">
      <div class="slide-progress-fill" id="slide-progress-13_transformers" style="width: 0%;"></div>
    </div>
  </div>
  <div class="slide-footer">
    <a href="../_static/slides/13_transformers.pdf" download>⬇ Download</a>
    <a href="#" onclick="slideFullscreen('13_transformers'); return false;" class="secondary">⛶ Fullscreen</a>
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

initSlideViewer('13_transformers', '../_static/slides/13_transformers.pdf');
</script>
```
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
TransformerBlock(embed_dim: int, num_heads: int, mlp_ratio: int = 4, ff_dim: int = None, dropout_prob: float = 0.1) -> TransformerBlock
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

The parameter count in the MLP is substantial. For `embed_dim = 512`, the first layer has `512 x 2048 + 2048 =` {glue:text}`mlp_linear1` ({glue:text}`mlp_linear1_approx`) parameters, and the second has `2048 x 512 + 512 =` {glue:text}`mlp_linear2` ({glue:text}`mlp_linear2_approx`) parameters, totaling {glue:text}`mlp_total_approx` parameters per block. In a 12-layer model, MLPs alone contribute {glue:text}`mlp_12layer_approx` parameters.

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
| Multi-Head Attention | {glue:text}`attn_params_512` | 4 x (512 x 512) for Q, K, V, O projections |
| Layer Norm 1 | {glue:text}`ln_params_512_approx` | 2 x 512 for gamma, beta |
| MLP | {glue:text}`mlp_params_512` | (512 x 2048 + 2048) + (2048 x 512 + 512) |
| Layer Norm 2 | {glue:text}`ln_params_512_approx` | 2 x 512 for gamma, beta |
| **Total per block** | **{glue:text}`block_total_512`** | Dominated by MLP and attention |

For a complete GPT model, add embeddings and output projection:

Embeddings: vocab_size x embed_dim (e.g., 50000 x 512 = {glue:text}`tok_emb_512`)
Position Embeddings: max_seq_len x embed_dim (e.g., 2048 x 512 = {glue:text}`pos_emb_512`)
Transformer Blocks: num_layers x {glue:text}`block_total_512` (e.g., {glue:text}`blocks_total_512_formula`)
Output Projection: embed_dim x vocab_size (often tied to embeddings)

Total: {glue:text}`gpt_total_512` parameters for this configuration

Memory requirements have three components:

1. **Parameter Memory**: Linear with model size, stored once
2. **Activation Memory**: Needed for backpropagation, grows with batch size and sequence length
3. **Attention Memory**: Quadratic with sequence length, the primary bottleneck

The attention memory wall explains why extending context length is expensive. For a batch of 4 sequences, 8 attention heads, and varying sequence lengths:

| Sequence Length | Attention Matrix Size | Memory (MB) |
|-----------------|----------------------|-------------|
| 512 | 4 x 8 x 512 x 512 | {glue:text}`attn_mem_512` |
| 1024 | 4 x 8 x 1024 x 1024 | {glue:text}`attn_mem_1024` |
| 2048 | 4 x 8 x 2048 x 2048 | {glue:text}`attn_mem_2048` |
| 4096 | 4 x 8 x 4096 x 4096 | {glue:text}`attn_mem_4096` |

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
from tinytorch.core.transformers import TransformerBlock, GPT

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

Attention matrix size: `batch_size x num_heads x seq_len x seq_len`
= `8 x 16 x 2048 x 2048 = ` {glue:text}`q1_elements_2048` elements

Memory: {glue:text}`q1_elements_2048` ` x 4 bytes (float32) = ` {glue:text}`q1_bytes_2048` ` bytes =` {glue:text}`q1_gb_2048` GB

Doubling sequence length to 4096:
= `8 x 16 x 4096 x 4096 = ` {glue:text}`q1_elements_4096` ` elements =` {glue:text}`q1_gb_4096` GB

**Scaling**: Doubling sequence length quadruples memory (4x increase). This quadratic scaling is why long context is expensive and drove innovations like sparse attention.
```

**Q2: Parameter Distribution Analysis**

For a GPT model with `vocab_size=50000`, `embed_dim=768`, `num_layers=12`, `num_heads=12`, calculate approximate total parameters. Which component dominates the parameter count: embeddings or transformer blocks?

```{admonition} Answer
:class: dropdown

**Token Embeddings**: `50000 x 768 = ` {glue:text}`q2_tok_emb`

**Position Embeddings**: `2048 x 768 = ` {glue:text}`q2_pos_emb` (assuming max_seq_len=2048)

**Transformer Blocks**: Each block has approximately {glue:text}`q2_per_block` parameters with embed_dim=768
- Attention: `4 x (768 x 768) = ` {glue:text}`q2_attn_per_block`
- MLP: `(768 x 3072 + 3072) + (3072 x 768 + 768) = ` {glue:text}`q2_mlp_per_block`
- Layer norms: negligible
- **Per block**: approximately {glue:text}`q2_per_block`
- **Total blocks**: `12 x ` {glue:text}`q2_per_block` ` = ` {glue:text}`q2_total_blocks`

**Output Projection**: Usually tied to embeddings (0 additional)

**Total**: {glue:text}`q2_tok_emb` ` + ` {glue:text}`q2_pos_emb` ` + ` {glue:text}`q2_total_blocks` ` = ` {glue:text}`q2_grand_total` parameters

**Dominant component**: Transformer blocks ({glue:text}`q2_total_blocks`) > Embeddings ({glue:text}`q2_total_emb`). As models scale, transformer blocks dominate because they scale with `embed_dim²` while embeddings scale linearly.
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

**Total forward passes**: `50 + 51 + 52 + ... + 149 = ` {glue:text}`q4_total_processings` token processings

**Why inefficient**: Attention recomputes key/value projections for all previous tokens every step, even though they don't change. For position 50, we recompute the same key/value vectors 100 times.

**KV Caching optimization**: Store computed key/value projections for previous tokens
- Each new token only computes its own key/value
- Attention uses cached keys/values from previous tokens
- Total computation: `50 (initial) + 100 (new tokens) = ` {glue:text}`q4_optimized` token processings

**Speedup**: {glue:text}`q4_total_processings` ` / ` {glue:text}`q4_optimized` ` = ` {glue:text}`q4_speedup` `x faster` for this example. The speedup increases with generation length, making KV caching essential for production systems.
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
- Batch norm: Compute 10 x 768 statistics across batch dimension (problematic)
- Layer norm: Compute 3 x 10 statistics across feature dimension (independent)

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

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/13_transformers/transformers.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/13_transformers/13_transformers.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
