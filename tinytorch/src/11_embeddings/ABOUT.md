---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# --- Embedding Dimension Trade-offs section ---

# Prose: 50,000 vocab x 512 embed_dim memory (approximate)
tradeoffs_50k_512_bytes = 50000 * 512 * 4
tradeoffs_50k_512_mb = tradeoffs_50k_512_bytes / 1024**2
glue("tradeoffs_50k_512_mb", f"{tradeoffs_50k_512_mb:.0f} MB")

# Prose: doubled to 1024 embed_dim
tradeoffs_50k_1024_bytes = 50000 * 1024 * 4
tradeoffs_50k_1024_mb = tradeoffs_50k_1024_bytes / 1024**2
glue("tradeoffs_50k_1024_mb", f"{tradeoffs_50k_1024_mb:.0f} MB")

# GPT-3 embedding memory in prose
gpt3_embed_bytes = 50257 * 12288 * 4
gpt3_embed_gb = gpt3_embed_bytes / 1024**3
glue("tradeoffs_gpt3_embed_gb", f"{gpt3_embed_gb:.1f} GB")

# --- Production table ---

# Small BERT: 30,000 x 768
table_bert_bytes = 30000 * 768 * 4
table_bert_mb = table_bert_bytes / 1024**2
glue("table_bert_mb", f"{table_bert_mb:.0f} MB")

# GPT-2: 50,257 x 1,024
table_gpt2_bytes = 50257 * 1024 * 4
table_gpt2_mb = table_gpt2_bytes / 1024**2
glue("table_gpt2_mb", f"{table_gpt2_mb:.0f} MB")

# GPT-3: 50,257 x 12,288
table_gpt3_bytes = 50257 * 12288 * 4
table_gpt3_mb = table_gpt3_bytes / 1024**2
glue("table_gpt3_mb", f"{table_gpt3_mb:,.0f} MB")

# Large Transformer: 100,000 x 1,024
table_large_bytes = 100000 * 1024 * 4
table_large_mb = table_large_bytes / 1024**2
glue("table_large_mb", f"{table_large_mb:.0f} MB")

# --- Scale section ---

# GPT-3 parameter count
gpt3_params = 50257 * 12288
glue("scale_gpt3_params", f"{gpt3_params / 1e6:.0f} million parameters")
glue("scale_gpt3_gb", f"{gpt3_embed_gb:.1f} GB")

# Batch lookups: 32 sequences x 2048 tokens
batch_lookups = 32 * 2048
glue("scale_batch_lookups", f"{batch_lookups:,}")

# --- Q1: Memory Calculation ---

q1_params = 50000 * 512
q1_bytes = q1_params * 4
q1_mb = q1_bytes / 1024**2
glue("q1_total_bytes", f"{q1_bytes:,} bytes")
glue("q1_mb", f"{q1_mb:.1f} MB")
glue("q1_params", f"{q1_params:,}")
glue("q1_bytes_full", f"{q1_bytes:,}")

# --- Q2: Positional Encoding Memory ---

q2_params = 2048 * 512
q2_bytes = q2_params * 4
q2_mb = q2_bytes / 1024**2
glue("q2_bytes", f"{q2_bytes:,} bytes")
glue("q2_mb", f"{q2_mb:.1f} MB")
glue("q2_params", f"{q2_params:,}")

# GPT-3 learned PE memory
q2_gpt3_bytes = 2048 * 12288 * 4
q2_gpt3_mb = q2_gpt3_bytes / 1024**2
glue("q2_gpt3_pe_mb", f"{q2_gpt3_mb:.0f} MB")

# --- Q3: Lookup Complexity ---

q3_total = 32 * 128
glue("q3_total_lookups", f"{q3_total:,}")

# --- Q4: Embedding Dimension Scaling ---

q4_original_bytes = 50000 * 512 * 4
q4_original_mb = q4_original_bytes / 1024**2
q4_doubled_bytes = 50000 * 1024 * 4
q4_doubled_mb = q4_doubled_bytes / 1024**2
glue("q4_original_mb", f"{q4_original_mb:.0f} MB")
glue("q4_doubled_mb", f"{q4_doubled_mb:.0f} MB")
```

# Module 11: Embeddings

:::{admonition} Module Info
:class: note

**ARCHITECTURE TIER** | Difficulty: ●●○○ | Time: 3-5 hours | Prerequisites: 01-08, 10

**Prerequisites: Modules 01-08 and 10** means you should understand:
- Tensor operations (shape manipulation, matrix operations, broadcasting)
- Training fundamentals (forward/backward, optimization)
- Tokenization (converting text to token IDs, vocabularies)

If you can explain how a tokenizer converts "hello" to token IDs and how to multiply matrices, you're ready.
:::

`````{only} html
````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/11_embeddings.mp3" type="audio/mpeg">
</audio>
```

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F11_embeddings%2Fembeddings.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/11_embeddings/11_embeddings.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #14b8a6; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
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

<div class="slide-viewer-container" id="slide-viewer-11_embeddings">
  <div class="slide-header">
    <div class="slide-title">
      <span>🔥</span>
      <span>Slide Deck</span>
      <span class="slide-subtitle">· AI-generated</span>
    </div>
    <div class="slide-toolbar">
      <div class="slide-nav-group">
        <button onclick="slideNav('11_embeddings', -1)" title="Previous">‹</button>
        <span class="slide-page-info"><span id="slide-num-11_embeddings">1</span> / <span id="slide-count-11_embeddings">-</span></span>
        <button onclick="slideNav('11_embeddings', 1)" title="Next">›</button>
      </div>
      <div class="slide-zoom-group">
        <button onclick="slideZoom('11_embeddings', -0.25)" title="Zoom out">−</button>
        <button onclick="slideZoom('11_embeddings', 0.25)" title="Zoom in">+</button>
      </div>
    </div>
  </div>
  <div class="slide-canvas-wrapper">
    <div id="slide-loading-11_embeddings" class="slide-loading">Loading slides...</div>
    <canvas id="slide-canvas-11_embeddings" class="slide-canvas" style="display:none;"></canvas>
  </div>
  <div class="slide-progress-wrapper">
    <div class="slide-progress-bar" onclick="slideProgress('11_embeddings', event)">
      <div class="slide-progress-fill" id="slide-progress-11_embeddings" style="width: 0%;"></div>
    </div>
  </div>
  <div class="slide-footer">
    <a href="../_static/slides/11_embeddings.pdf" download>⬇ Download</a>
    <a href="#" onclick="slideFullscreen('11_embeddings'); return false;" class="secondary">⛶ Fullscreen</a>
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

initSlideViewer('11_embeddings', '../_static/slides/11_embeddings.pdf');
</script>
```
`````

## Overview

Embeddings are the crucial bridge between discrete tokens and continuous neural network operations. Your tokenizer from Module 10 converts text into token IDs like `[42, 7, 15]`, but neural networks operate on dense vectors of real numbers. Embeddings transform each integer token ID into a learned dense vector that captures semantic meaning. By the end of this module, you'll implement the embedding systems that power modern language models, from basic lookup tables to sophisticated positional encodings.

Every transformer model, from BERT to GPT-4, relies on embeddings to convert language into learnable representations. You'll build the exact patterns used in production, understanding not just how they work but why they're designed this way.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** embedding layers that convert token IDs to dense vectors through efficient table lookup
- **Master** positional encoding strategies including learned and sinusoidal approaches
- **Understand** memory scaling for embedding tables and the trade-offs between vocabulary size and embedding dimension
- **Connect** your implementation to production transformer architectures used in GPT and BERT
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Embedding System
flowchart LR
    subgraph "Your Embedding System"
        A["Embedding<br/>Token → Vector"]
        B["PositionalEncoding<br/>Learned Positions"]
        C["Sinusoidal PE<br/>Math Patterns"]
        D["EmbeddingLayer<br/>Complete System"]
    end

    A --> D
    B --> D
    C --> D

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `Embedding` class | Token ID to vector lookup via indexing |
| 2 | `PositionalEncoding` class | Learnable position embeddings |
| 3 | `create_sinusoidal_embeddings()` | Mathematical position encoding |
| 4 | `EmbeddingLayer` class | Complete token + position system |

**The pattern you'll enable:**
```python
# Converting tokens to position-aware dense vectors
embed_layer = EmbeddingLayer(vocab_size=50000, embed_dim=512)
tokens = Tensor([[1, 42, 7]])  # Token IDs from tokenizer
embeddings = embed_layer(tokens)  # (1, 3, 512) dense vectors ready for attention
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Attention mechanisms (that's Module 12: Attention)
- Full transformer architectures (that's Module 13: Transformers)
- Word2Vec or GloVe pretrained embeddings (you're building learnable embeddings)
- Subword embedding composition (PyTorch handles this at the tokenization level)

**You are building the foundation for sequence models.** Context-aware representations come next.

## API Reference

This section documents the embedding components you'll build. Use this as your reference while implementing and debugging.

### Embedding Class

```python
Embedding(vocab_size, embed_dim)
```

Learnable embedding layer that maps token indices to dense vectors through table lookup.

**Constructor Parameters:**
- `vocab_size` (int): Size of vocabulary (number of unique tokens)
- `embed_dim` (int): Dimension of embedding vectors

**Core Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `forward(indices: Tensor) -> Tensor` | Lookup embeddings for token indices |
| `parameters` | `parameters() -> List[Tensor]` | Return weight matrix for optimization |

**Properties:**
- `weight`: Tensor of shape `(vocab_size, embed_dim)` containing learnable embeddings

### PositionalEncoding Class

```python
PositionalEncoding(max_seq_len, embed_dim)
```

Learnable positional encoding that adds trainable position-specific vectors to embeddings.

**Constructor Parameters:**
- `max_seq_len` (int): Maximum sequence length to support
- `embed_dim` (int): Embedding dimension (must match token embeddings)

**Core Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `forward(x: Tensor) -> Tensor` | Add positional encodings to input embeddings |
| `parameters` | `parameters() -> List[Tensor]` | Return position embedding matrix |

### Sinusoidal Embeddings Function

```python
create_sinusoidal_embeddings(max_seq_len, embed_dim) -> Tensor
```

Creates fixed sinusoidal positional encodings using trigonometric functions. No parameters to learn.

**Mathematical Formula:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))
PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))
```

### EmbeddingLayer Class

```python
EmbeddingLayer(vocab_size, embed_dim, max_seq_len=512,
               pos_encoding='learned', scale_embeddings=False)
```

Complete embedding system combining token embeddings and positional encoding.

**Constructor Parameters:**
- `vocab_size` (int): Size of vocabulary
- `embed_dim` (int): Embedding dimension
- `max_seq_len` (int): Maximum sequence length for positional encoding
- `pos_encoding` (str): Type of positional encoding ('learned', 'sinusoidal', or None)
- `scale_embeddings` (bool): Whether to scale by sqrt(embed_dim) (transformer convention)

**Core Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `forward(tokens: Tensor) -> Tensor` | Complete embedding pipeline |
| `parameters` | `parameters() -> List[Tensor]` | All trainable parameters |

## Core Concepts

This section explores the fundamental ideas behind embeddings and positional encoding. These concepts apply to every modern language model, from small research experiments to production systems.

### From Indices to Vectors

Neural networks operate on continuous vectors, but language consists of discrete tokens. After your tokenizer converts "the cat sat" to token IDs `[1, 42, 7]`, you need a way to represent these discrete integers as dense vectors that can capture semantic meaning.

The embedding layer solves this through a simple but powerful idea: maintain a learnable table where each token ID maps to a vector. For a vocabulary of 50,000 tokens with 512-dimensional embeddings, you create a matrix of shape `(50000, 512)` initialized with small random values. During training, these vectors adjust to capture semantic relationships: similar words learn similar vectors.

Here's the core implementation showing how efficiently this works:

```python
def forward(self, indices: Tensor) -> Tensor:
    """Forward pass: lookup embeddings for given indices."""
    # Validate indices are in range
    if np.any(indices.data >= self.vocab_size) or np.any(indices.data < 0):
        raise ValueError(
            f"Index out of range. Expected 0 <= indices < {self.vocab_size}, "
            f"got min={np.min(indices.data)}, max={np.max(indices.data)}"
        )

    # Perform embedding lookup using advanced indexing
    # This is equivalent to one-hot multiplication but much more efficient
    embedded = self.weight.data[indices.data.astype(int)]

    # Create result tensor with gradient tracking
    result = Tensor(embedded, requires_grad=self.weight.requires_grad)

    if result.requires_grad:
        result._grad_fn = EmbeddingBackward(self.weight, indices)

    return result
```

The beauty is in the simplicity: `self.weight.data[indices.data.astype(int)]` uses NumPy's advanced indexing (also called fancy indexing) to look up multiple embeddings simultaneously. For input indices `[1, 42, 7]`, this single operation retrieves rows 1, 42, and 7 from the weight matrix in one efficient step, automatically handling batched inputs of any shape. While conceptually equivalent to creating one-hot vectors and matrix multiplication, direct indexing is orders of magnitude faster and requires no intermediate allocations.

### Embedding Table Mechanics

The embedding table is a learnable parameter matrix initialized with small random values. For vocabulary size V and embedding dimension D, the table has shape `(V, D)`. Each row represents one token's learned representation.

Initialization matters for training stability. The implementation uses Xavier initialization:

```python
# Xavier initialization for better gradient flow
limit = math.sqrt(6.0 / (vocab_size + embed_dim))
self.weight = Tensor(
    np.random.uniform(-limit, limit, (vocab_size, embed_dim)),
    requires_grad=True
)
```

This initialization scale ensures gradients neither explode nor vanish at the start of training. The limit is computed from both vocabulary size and embedding dimension, balancing the fan-in and fan-out of the embedding layer.

During training, gradients flow back through the lookup operation to update only the embeddings that were accessed. If your batch contains tokens `[5, 10, 10, 5]`, only rows 5 and 10 of the embedding table receive gradient updates. This sparse gradient pattern is handled by the `EmbeddingBackward` gradient function, making embedding updates extremely efficient even for vocabularies with millions of tokens.

### Learned vs Fixed Embeddings

Positional information can be added to token embeddings in two fundamentally different ways: learned position embeddings and fixed sinusoidal encodings.

**Learned positional encoding** treats each position as a trainable parameter, just like token embeddings. For maximum sequence length M and embedding dimension D, you create a second embedding table of shape `(M, D)`:

```python
# Initialize position embedding matrix
# Smaller initialization than token embeddings since these are additive
limit = math.sqrt(2.0 / embed_dim)
self.position_embeddings = Tensor(
    np.random.uniform(-limit, limit, (max_seq_len, embed_dim)),
    requires_grad=True
)
```

During forward pass, you slice position embeddings and add them to token embeddings:

```python
def forward(self, x: Tensor) -> Tensor:
    """Add positional encodings to input embeddings."""
    batch_size, seq_len, embed_dim = x.shape

    # Slice position embeddings for this sequence length
    # Tensor slicing preserves gradient flow (from Module 01's __getitem__)
    pos_embeddings = self.position_embeddings[:seq_len]

    # Reshape to add batch dimension: (1, seq_len, embed_dim)
    pos_data = pos_embeddings.data[np.newaxis, :, :]
    pos_embeddings_batched = Tensor(pos_data, requires_grad=pos_embeddings.requires_grad)

    # Copy gradient function to preserve backward connection
    if hasattr(pos_embeddings, '_grad_fn') and pos_embeddings._grad_fn is not None:
        pos_embeddings_batched._grad_fn = pos_embeddings._grad_fn

    # Add positional information - gradients flow through both x and pos_embeddings!
    result = x + pos_embeddings_batched
    return result
```

The slicing operation `self.position_embeddings[:seq_len]` preserves gradient tracking because TinyTorch's Tensor `__getitem__` (from Module 01) maintains the connection to the original parameter. This allows backpropagation to update only the position embeddings actually used in the forward pass.

The advantage is flexibility: the model can learn task-specific positional patterns. The disadvantage is memory cost and a hard maximum sequence length.

**Fixed sinusoidal encoding** uses mathematical patterns requiring no parameters. The formula creates unique position signatures using sine and cosine at different frequencies:

```python
# Create position indices [0, 1, 2, ..., max_seq_len-1]
position = np.arange(max_seq_len, dtype=np.float32)[:, np.newaxis]

# Create dimension indices for calculating frequencies
div_term = np.exp(
    np.arange(0, embed_dim, 2, dtype=np.float32) *
    -(math.log(10000.0) / embed_dim)
)

# Initialize the positional encoding matrix
pe = np.zeros((max_seq_len, embed_dim), dtype=np.float32)

# Apply sine to even indices, cosine to odd indices
pe[:, 0::2] = np.sin(position * div_term)
pe[:, 1::2] = np.cos(position * div_term)
```

This creates a unique vector for each position where low-index dimensions oscillate rapidly (high frequency) and high-index dimensions change slowly (low frequency). The pattern allows the model to learn relative positions through dot products, and crucially, can extrapolate to sequences longer than seen during training.

### Positional Encodings

Without positional information, embeddings have no notion of order. The sentence "cat sat on mat" would have identical representation to "mat on sat cat" because you'd sum or average the same four token embeddings regardless of order.

Positional encodings solve this by adding position-specific information to each token's embedding. After embedding lookup, token 42 at position 0 gets different positional information than token 42 at position 5, making the model position-aware.

The Transformer paper introduced sinusoidal positional encoding with a clever mathematical structure. For position `pos` and dimension `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))  # Even dimensions
PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))  # Odd dimensions
```

The `10000` base creates different wavelengths across dimensions. Dimension 0 oscillates rapidly (frequency ≈ 1), while dimension 510 changes extremely slowly (frequency ≈ 1/10000). This multiscale structure gives each position a unique "fingerprint" and enables the model to learn relative position through simple vector arithmetic.

At position 0, all sine terms equal 0 and all cosine terms equal 1: `[0, 1, 0, 1, 0, 1, ...]`. At position 1, the pattern shifts based on each dimension's frequency. The combination of many frequencies creates distinct encodings where nearby positions have similar (but not identical) vectors, providing smooth positional gradients.

The trigonometric identity enables learning relative positions: `PE(pos+k)` can be expressed as a linear function of `PE(pos)` using sine and cosine addition formulas. This allows attention mechanisms to implicitly learn positional offsets (like "the token 3 positions ahead") through learned weights on the position encodings, without the model needing separate relative position parameters.

### Embedding Dimension Trade-offs

The embedding dimension D controls the capacity of your learned representations. Larger D provides more expressiveness but costs memory and compute. The choice involves several interacting factors.

**Memory scaling**: Embedding tables scale as `vocab_size × embed_dim × 4 bytes` (for float32). A vocabulary of 50,000 tokens with 512-dimensional embeddings requires {glue:text}`tradeoffs_50k_512_mb`. Double the dimension to 1024, and memory doubles to {glue:text}`tradeoffs_50k_1024_mb`. For large vocabularies, the embedding table often dominates total model memory. GPT-3's 50,257 token vocabulary with 12,288-dimensional embeddings uses approximately {glue:text}`tradeoffs_gpt3_embed_gb` just for token embeddings.

**Semantic capacity**: Higher dimensions allow finer-grained semantic distinctions. With 64 dimensions, you might capture basic categories (animals, actions, objects). With 512 dimensions, you can encode subtle relationships (synonyms, antonyms, part-of-speech, contextual variations). With 1024+ dimensions, you have capacity for highly nuanced semantic features discovered through training.

**Computational cost**: Every attention head in transformers performs operations over the embedding dimension. Memory bandwidth becomes the bottleneck: transferring embedding vectors from RAM to cache dominates the time to process them. Larger embeddings mean more memory traffic per token, reducing throughput.

**Typical scales in production**:

| Model | Vocabulary | Embed Dim | Embedding Memory |
|-------|-----------|-----------|------------------|
| Small BERT | 30,000 | 768 | {glue:text}`table_bert_mb` |
| GPT-2 | 50,257 | 1,024 | {glue:text}`table_gpt2_mb` |
| GPT-3 | 50,257 | 12,288 | {glue:text}`table_gpt3_mb` |
| Large Transformer | 100,000 | 1,024 | {glue:text}`table_large_mb` |

The embedding dimension typically matches the model's hidden dimension since embeddings feed directly into the first transformer layer. You rarely see models with embedding dimension different from hidden dimension (though it's technically possible with a projection layer).

## Common Errors

These are the errors you'll encounter most often when working with embeddings. Understanding why they happen will save you hours of debugging.

### Index Out of Range

**Error**: `ValueError: Index out of range. Expected 0 <= indices < 50000, got max=50001`

Embedding lookup expects token IDs in the range `[0, vocab_size-1]`. If your tokenizer produces an ID of 50001 but your embedding layer has `vocab_size=50000`, the lookup fails.

**Cause**: Mismatch between tokenizer vocabulary size and embedding layer vocabulary size. This often happens when you train a tokenizer separately and forget to sync the vocabulary size when creating the embedding layer.

**Fix**: Ensure `embed_layer.vocab_size` matches your tokenizer's vocabulary size exactly:

```python
tokenizer = Tokenizer(vocab_size=50000)
embed = Embedding(vocab_size=tokenizer.vocab_size, embed_dim=512)
```

### Sequence Length Exceeds Maximum

**Error**: `ValueError: Sequence length 1024 exceeds maximum 512`

Learned positional encodings have a fixed maximum sequence length set during initialization. If you try to process a sequence longer than this maximum, the forward pass fails because there are no position embeddings for those positions.

**Cause**: Input sequences longer than `max_seq_len` parameter used when creating the positional encoding layer.

**Fix**: Either increase `max_seq_len` during initialization, truncate your sequences, or use sinusoidal positional encoding which can handle arbitrary lengths:

```python
# Option 1: Increase max_seq_len
pos_enc = PositionalEncoding(max_seq_len=2048, embed_dim=512)

# Option 2: Use sinusoidal (no length limit)
embed_layer = EmbeddingLayer(vocab_size=50000, embed_dim=512,
                             pos_encoding='sinusoidal')
```

### Embedding Dimension Mismatch

**Error**: `ValueError: Embedding dimension mismatch: expected 512, got 768`

When adding positional encodings to token embeddings, the dimensions must match exactly. If your token embeddings are 512-dimensional but your positional encoding expects 768-dimensional inputs, element-wise addition fails.

**Cause**: Creating embedding components with inconsistent `embed_dim` values.

**Fix**: Use the same `embed_dim` for all embedding components:

```python
embed_dim = 512
token_embed = Embedding(vocab_size=50000, embed_dim=embed_dim)
pos_enc = PositionalEncoding(max_seq_len=512, embed_dim=embed_dim)
```

### Shape Errors with Batching

**Error**: `ValueError: Expected 3D input (batch, seq, embed), got shape (128, 512)`

Positional encoding expects 3D tensors with batch dimension. If you pass a 2D tensor (sequence, embedding), the forward pass fails.

**Cause**: Forgetting to add batch dimension when processing single sequences, or using raw embedding output without reshaping.

**Fix**: Ensure inputs have batch dimension, even for single sequences:

```python
# Wrong: 2D input
tokens = Tensor([1, 2, 3])
embeddings = embed(tokens)  # Shape: (3, 512) - missing batch dim

# Right: 3D input
tokens = Tensor([[1, 2, 3]])  # Added batch dimension
embeddings = embed(tokens)  # Shape: (1, 3, 512) - correct
```

## Production Context

### Your Implementation vs. PyTorch

Your TinyTorch embedding system and PyTorch's `torch.nn.Embedding` share the same conceptual design and API patterns. The differences are in scale, optimization, and device support.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Backend** | NumPy (CPU only) | C++/CUDA (CPU/GPU) |
| **Lookup Speed** | 1x (baseline) | 10-100x faster on GPU |
| **Max Vocabulary** | Limited by RAM | Billions (with techniques) |
| **Positional Encoding** | Learned + sinusoidal | Must implement yourself* |
| **Sparse Gradients** | Via custom backward | Native sparse gradient support |
| **Memory Optimization** | Standard | Quantization, pruning, sharing |

*PyTorch provides building blocks but you implement positional encoding patterns yourself (as you did here)

### Code Comparison

The following comparison shows equivalent embedding operations in TinyTorch and PyTorch. Notice how the APIs mirror each other closely.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.core.embeddings import Embedding, EmbeddingLayer

# Create embedding layer
embed = Embedding(vocab_size=50000, embed_dim=512)

# Token lookup
tokens = Tensor([[1, 42, 7, 99]])
embeddings = embed(tokens)  # (1, 4, 512)

# Complete system with position encoding
embed_layer = EmbeddingLayer(
    vocab_size=50000,
    embed_dim=512,
    max_seq_len=2048,
    pos_encoding='learned'
)
position_aware = embed_layer(tokens)
```
````

````{tab-item} PyTorch
```python
import torch
import torch.nn as nn

# Create embedding layer
embed = nn.Embedding(num_embeddings=50000, embedding_dim=512)

# Token lookup
tokens = torch.tensor([[1, 42, 7, 99]])
embeddings = embed(tokens)  # (1, 4, 512)

# Complete system (you implement positional encoding yourself)
class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, tokens):
        seq_len = tokens.shape[1]
        positions = torch.arange(seq_len).unsqueeze(0)
        return self.token_embed(tokens) + self.pos_embed(positions)

embed_layer = EmbeddingWithPosition(50000, 512, 2048)
position_aware = embed_layer(tokens)
```
````
`````

Let's walk through each section to understand the comparison:

- **Line 1-2 (Import)**: Both frameworks provide dedicated embedding modules. TinyTorch packages everything in `embeddings`; PyTorch uses `nn.Embedding` as the base class.
- **Line 4-5 (Embedding Creation)**: Your `Embedding` class closely mirrors PyTorch's `nn.Embedding`. The parameter names differ (`vocab_size` vs `num_embeddings`) but the concept is identical.
- **Line 7-9 (Token Lookup)**: Both use identical calling patterns. The embedding layer acts as a function, taking token IDs and returning dense vectors. Shape semantics are identical.
- **Line 11-20 (Complete System)**: Your `EmbeddingLayer` provides a complete system in one class. In PyTorch, you implement this pattern yourself by composing `nn.Embedding` layers for tokens and positions. The HuggingFace Transformers library implements this exact pattern for BERT, GPT, and other models.
- **Line 22-24 (Forward Pass)**: Both systems add token and position embeddings element-wise. Your implementation handles this internally; PyTorch requires you to manage position indices explicitly.

```{tip} What's Identical

Embedding lookup semantics, gradient flow patterns, and the addition of positional information. When you debug PyTorch transformer models, you'll recognize these exact patterns because you built them yourself.
```

### Why Embeddings Matter at Scale

To appreciate embedding systems, consider the scale of modern language models:

- **GPT-3 embeddings**: 50,257 token vocabulary × 12,288 dimensions = **{glue:text}`scale_gpt3_params`** = {glue:text}`scale_gpt3_gb` of memory (just for token embeddings, not counting position embeddings)
- **Lookup throughput**: Processing 32 sequences of 2048 tokens requires **{glue:text}`scale_batch_lookups` embedding lookups** per batch. At 1000 batches per second (typical training), that's 65 million lookups per second.
- **Memory bandwidth**: Each lookup transfers 512-1024 dimensions × 4 bytes = **2-4 KB from RAM to cache**. At scale, memory bandwidth (not compute) becomes the bottleneck.
- **Gradient sparsity**: In a batch with {glue:text}`scale_batch_lookups` tokens, only a small fraction of the 50,257 vocabulary is accessed. Efficient training exploits this sparsity, updating only the accessed embeddings' gradients.

Modern transformer training spends approximately **10-15% of total time** in embedding operations (lookup + position encoding). The remaining 85-90% goes to attention and feedforward layers. However, embeddings consume **30-40% of model memory** for models with large vocabularies, making them critical for deployment.

## Check Your Understanding

Test yourself with these systems thinking questions. They're designed to build intuition for the performance and memory characteristics you'll encounter in production.

**Q1: Memory Calculation**

An embedding layer has `vocab_size=50000` and `embed_dim=512`. How much memory does the embedding table use (in MB)?

```{admonition} Answer
:class: dropdown

50,000 × 512 × 4 bytes = **{glue:text}`q1_total_bytes` = {glue:text}`q1_mb`**

Calculation breakdown:
- Parameters: 50,000 × 512 = {glue:text}`q1_params`
- Memory: {glue:text}`q1_params` × 4 bytes (float32) = {glue:text}`q1_bytes_full` bytes
- In MB: {glue:text}`q1_bytes_full` / (1024 × 1024) = {glue:text}`q1_mb`

This is why vocabulary size matters for model deployment!
```

**Q2: Positional Encoding Memory**

Compare memory requirements for learned vs sinusoidal positional encoding with `max_seq_len=2048` and `embed_dim=512`.

```{admonition} Answer
:class: dropdown

**Learned PE**: 2,048 × 512 × 4 = **{glue:text}`q2_bytes` = {glue:text}`q2_mb`** ({glue:text}`q2_params` parameters)

**Sinusoidal PE**: **0 bytes** (0 parameters - computed mathematically)

For large models, learned PE adds significant memory. GPT-3 uses learned PE with 2048 positions × 12,288 dimensions = {glue:text}`q2_gpt3_pe_mb` additional memory. Some models use sinusoidal to save this memory.
```

**Q3: Lookup Complexity**

What is the time complexity of looking up embeddings for a batch of 32 sequences, each with 128 tokens?

```{admonition} Answer
:class: dropdown

**O(1) per token**, or **O(batch_size × seq_len)** = O(32 × 128) = O({glue:text}`q3_total_lookups`) total

The lookup operation is constant time per token because it's just array indexing: `weight[token_id]`. For {glue:text}`q3_total_lookups` tokens, you perform {glue:text}`q3_total_lookups` constant-time lookups.

Importantly, vocabulary size does NOT affect lookup time. Looking up tokens from a 1,000 word vocabulary is the same speed as from a 100,000 word vocabulary (assuming cache effects are comparable). The memory access is direct indexing, not search.
```

**Q4: Embedding Dimension Scaling**

You have an embedding layer with `vocab_size=50000, embed_dim=512` using {glue:text}`q4_original_mb`. If you double `embed_dim` to 1024, what happens to memory?

```{admonition} Answer
:class: dropdown

Memory **doubles to {glue:text}`q4_doubled_mb`**

Embedding memory scales linearly with embedding dimension:
- Original: 50,000 × 512 × 4 = {glue:text}`q4_original_mb`
- Doubled: 50,000 × 1,024 × 4 = {glue:text}`q4_doubled_mb`

This is why you can't arbitrarily increase embedding dimensions. Each doubling doubles memory and memory bandwidth requirements. Large models carefully balance embedding dimension against available memory.
```

**Q5: Sinusoidal Extrapolation**

You trained a model with sinusoidal positional encoding and `max_seq_len=512`. Can you process sequences of length 1024 at inference time? What about with learned positional encoding?

````{admonition} Answer
:class: dropdown

**Sinusoidal PE: Yes** - can extrapolate to length 1024 (or any length)

The mathematical formula creates unique encodings for any position. You simply compute:
```python
pe_1024 = create_sinusoidal_embeddings(max_seq_len=1024, embed_dim=512)
```

**Learned PE: No** - cannot handle sequences longer than training maximum

Learned PE creates a fixed embedding table of shape `(max_seq_len, embed_dim)`. For positions beyond 512, there are no learned embeddings. You must either:
- Retrain with larger `max_seq_len`
- Interpolate position embeddings (advanced technique)
- Truncate sequences to 512 tokens

This is why many production models use sinusoidal or relative positional encodings that can handle variable lengths.
````

## Further Reading

For students who want to understand the academic foundations and mathematical underpinnings of embeddings and positional encoding:

### Seminal Papers

- **Word2Vec** - Mikolov et al. (2013). Introduced efficient learned word embeddings through context prediction. Though your implementation learns embeddings end-to-end, Word2Vec established the idea that similar words should have similar vectors. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)

- **Attention Is All You Need** - Vaswani et al. (2017). Introduced sinusoidal positional encoding and demonstrated that learned embeddings combined with positional information enable powerful sequence models. Section 3.5 explains the positional encoding formula you implemented. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- **BERT: Pre-training of Deep Bidirectional Transformers** - Devlin et al. (2018). Shows how embeddings combine with positional and segment encodings for language understanding tasks. BERT uses learned positional embeddings rather than sinusoidal. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

### Additional Resources

- **Blog Post**: "The Illustrated Word2Vec" by Jay Alammar - Visual explanation of learned word embeddings and semantic relationships
- **Documentation**: [PyTorch nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) - See production embedding implementation
- **Paper**: "GloVe: Global Vectors for Word Representation" - Pennington et al. (2014) - Alternative embedding approach based on co-occurrence statistics

## What's Next

```{seealso} Coming Up: Module 12 - Attention

Implement attention mechanisms that let embeddings interact with each other. You'll build the scaled dot-product attention that enables transformers to learn which tokens should influence each other, creating context-aware representations.
```

**Preview - How Your Embeddings Get Used in Future Modules:**

| Module | What It Does | Your Embeddings In Action |
|--------|--------------|--------------------------|
| **12: Attention** | Context-aware representations | `attention(embed_layer(tokens))` creates query, key, value |
| **13: Transformers** | Complete sequence-to-sequence | `transformer(embed_layer(src), embed_layer(tgt))` |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/11_embeddings/embeddings.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/11_embeddings/11_embeddings.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
