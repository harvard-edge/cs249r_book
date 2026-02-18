---
file_format: mystnb
kernelspec:
  name: python3
---

# Module 16: Compression

:::{admonition} Module Info
:class: note

**OPTIMIZATION TIER** | Difficulty: ●●●○ | Time: 5-7 hours | Prerequisites: 01-14

**Prerequisites: Modules 01-14** means you should have:
- Built tensors, layers, and the complete training pipeline (Modules 01-08)
- Implemented profiling tools to measure model characteristics (Module 14)
- Comfort with weight distributions, parameter counting, and memory analysis

If you can profile a model's parameters and understand weight distributions, you're ready.
:::

`````{only} html
````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/16_compression.mp3" type="audio/mpeg">
</audio>
```

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F16_compression%2Fcompression.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/16_compression/16_compression.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #14b8a6; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
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

<div class="slide-viewer-container" id="slide-viewer-16_compression">
  <div class="slide-header">
    <div class="slide-title">
      <span>🔥</span>
      <span>Slide Deck</span>
      <span class="slide-subtitle">· AI-generated</span>
    </div>
    <div class="slide-toolbar">
      <div class="slide-nav-group">
        <button onclick="slideNav('16_compression', -1)" title="Previous">‹</button>
        <span class="slide-page-info"><span id="slide-num-16_compression">1</span> / <span id="slide-count-16_compression">-</span></span>
        <button onclick="slideNav('16_compression', 1)" title="Next">›</button>
      </div>
      <div class="slide-zoom-group">
        <button onclick="slideZoom('16_compression', -0.25)" title="Zoom out">−</button>
        <button onclick="slideZoom('16_compression', 0.25)" title="Zoom in">+</button>
      </div>
    </div>
  </div>
  <div class="slide-canvas-wrapper">
    <div id="slide-loading-16_compression" class="slide-loading">Loading slides...</div>
    <canvas id="slide-canvas-16_compression" class="slide-canvas" style="display:none;"></canvas>
  </div>
  <div class="slide-progress-wrapper">
    <div class="slide-progress-bar" onclick="slideProgress('16_compression', event)">
      <div class="slide-progress-fill" id="slide-progress-16_compression" style="width: 0%;"></div>
    </div>
  </div>
  <div class="slide-footer">
    <a href="../_static/slides/16_compression.pdf" download>⬇ Download</a>
    <a href="#" onclick="slideFullscreen('16_compression'); return false;" class="secondary">⛶ Fullscreen</a>
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

initSlideViewer('16_compression', '../_static/slides/16_compression.pdf');
</script>
```
`````

## Overview

Model compression is the art of making neural networks smaller and faster while preserving their intelligence. Modern language models occupy 100GB+ of storage, but mobile devices have less than 1GB available for models. Edge devices have even tighter constraints at under 100MB. Compression techniques bridge this gap, enabling deployment of powerful models on resource-constrained devices.

In this module, you'll implement four fundamental compression techniques: magnitude-based pruning removes small weights, structured pruning eliminates entire channels for hardware efficiency, knowledge distillation trains compact student models from large teachers, and low-rank approximation factors matrices to reduce parameters. By the end, you'll achieve 80-90% sparsity with minimal accuracy loss, understanding the systems trade-offs between model size, inference speed, and prediction quality.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** magnitude-based pruning to remove 80-90% of small weights while preserving accuracy
- **Master** structured pruning that creates hardware-friendly sparsity patterns by removing entire channels
- **Build** knowledge distillation systems that compress models 10x through teacher-student training
- **Understand** compression trade-offs between sparsity ratio, inference speed, memory footprint, and accuracy preservation
- **Analyze** when to apply different compression techniques based on deployment constraints and performance requirements
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Compression System
flowchart LR
    subgraph "Your Compression System"
        A["Sparsity<br/>Measurement"]
        B["Magnitude<br/>Pruning"]
        C["Structured<br/>Pruning"]
        D["Knowledge<br/>Distillation"]
        E["Low-Rank<br/>Approximation"]
    end

    A --> B --> C --> D --> E

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
    style E fill:#e2d5f1
```

**Implementation roadmap:**

| Step | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `measure_sparsity()` | Calculate percentage of zero weights |
| 2 | `magnitude_prune()` | Remove weights below threshold |
| 3 | `structured_prune()` | Remove entire channels by importance |
| 4 | `KnowledgeDistillation` | Train small model from large teacher |
| 5 | `low_rank_approximate()` | Compress matrices via SVD |

**The pattern you'll enable:**
```python
# Compress a model by removing 80% of smallest weights
magnitude_prune(model, sparsity=0.8)
sparsity = measure_sparsity(model)  # Returns ~80%
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Sparse storage formats like CSR (scipy.sparse handles this in production)
- Fine-tuning after pruning (iterative pruning schedules)
- Dynamic pruning during training (PyTorch does this with hooks and callbacks)
- Combined quantization and pruning (advanced technique for maximum compression)

**You are building compression algorithms.** Sparse execution optimizations come from specialized libraries.

## API Reference

This section provides a quick reference for the compression functions and classes you'll build. Use it as your guide while implementing and debugging.

### Sparsity Measurement

```python
measure_sparsity(model) -> float
```

Calculate the percentage of zero weights in a model. Essential for tracking compression effectiveness.

### Pruning Methods

| Function | Signature | Description |
|----------|-----------|-------------|
| `magnitude_prune` | `magnitude_prune(model, sparsity=0.9)` | Remove smallest weights to achieve target sparsity |
| `structured_prune` | `structured_prune(model, prune_ratio=0.5)` | Remove entire channels based on L2 norm importance |

### Knowledge Distillation

```python
KnowledgeDistillation(teacher_model, student_model, temperature=3.0, alpha=0.7)
```

**Constructor Parameters:**
- `teacher_model`: Large pre-trained model with high accuracy
- `student_model`: Smaller model to train via distillation
- `temperature`: Softening parameter for probability distributions (typical: 3-5)
- `alpha`: Weight for soft targets (0.7 = 70% teacher, 30% hard labels)

**Key Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `distillation_loss` | `distillation_loss(student_logits, teacher_logits, true_labels) -> float` | Combined soft and hard target loss |

### Low-Rank Approximation

```python
low_rank_approximate(weight_matrix, rank_ratio=0.5) -> Tuple[ndarray, ndarray, ndarray]
```

**Parameters:**
- `weight_matrix`: Weight matrix to compress (e.g., (512, 256) Linear layer weights)
- `rank_ratio`: Fraction of original rank to keep (0.5 = keep 50% of singular values)

**Returns:**
- `U`: Left singular vectors (shape: m × k)
- `S`: Singular values (shape: k)
- `V`: Right singular vectors (shape: k × n)

Where k = rank_ratio × min(m, n). Reconstruct approximation with `U @ diag(S) @ V`.

## Core Concepts

This section covers the fundamental ideas you need to understand model compression deeply. These concepts apply across all ML frameworks and deployment scenarios.

### Pruning Fundamentals

Neural networks are remarkably over-parameterized. Research shows that 50-90% of weights in trained models contribute minimally to predictions. Pruning exploits this redundancy by removing unimportant weights, creating sparse networks that maintain accuracy while using dramatically fewer parameters.

The core insight is simple: weights with small magnitudes have little effect on outputs. When you compute `y = W @ x`, a weight of 0.001 contributes almost nothing compared to weights of magnitude 2.0 or 3.0. Pruning identifies and zeros out these negligible weights.

Here's how your magnitude pruning implementation works:

```python
def magnitude_prune(model, sparsity=0.9):
    """Remove weights with smallest magnitudes to achieve target sparsity."""
    # Collect all weights from model (excluding biases)
    all_weights = []
    weight_params = []

    for param in model.parameters():
        if len(param.shape) > 1:  # Only weight matrices, not bias vectors
            all_weights.extend(param.data.flatten())
            weight_params.append(param)

    # Calculate magnitude threshold at desired percentile
    magnitudes = np.abs(all_weights)
    threshold = np.percentile(magnitudes, sparsity * 100)

    # Apply pruning mask: zero out weights below threshold
    for param in weight_params:
        mask = np.abs(param.data) >= threshold
        param.data = param.data * mask  # In-place zeroing

    return model
```

The elegance is in the percentile-based threshold. Setting `sparsity=0.9` means "remove the bottom 90% of weights by magnitude." NumPy's `percentile` function finds the exact value that splits the distribution, and then a binary mask zeros out everything below that threshold.

To understand why this works, consider a typical weight distribution after training:

```
Weight Magnitudes (sorted):
[0.001, 0.002, 0.003, ..., 0.085, 0.087, ..., 0.95, 1.2, 2.3, 3.1]
 └──────────────── 90% ────────────────┘  └────── 10% ──────┘
        Small, removable                    Large, important

90th percentile = 0.087
Threshold mask: magnitude >= 0.087
Result: Keep only weights >= 0.087 (top 10%)
```

The critical insight is that weight distributions in trained networks are heavily skewed toward zero. Most weights contribute minimally, so removing them preserves the essential computation while dramatically reducing storage and compute.

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Pruning fundamentals: 10M parameter model at 90% sparsity
prune_total = 10_000_000
prune_sparsity = 0.9
prune_active = int(prune_total * (1 - prune_sparsity))
glue("prune_total", f"{prune_total / 1_000_000:.0f} million")
glue("prune_active", f"{prune_active / 1_000_000:.0f} million")
glue("prune_pct", f"{prune_sparsity * 100:.0f}%")
```

The memory impact is immediate. A model with {glue:text}`prune_total` parameters at {glue:text}`prune_pct` sparsity has only {glue:text}`prune_active` active weights. With sparse storage formats (like scipy's CSR matrix), this translates directly to 90% memory reduction. The compute savings come from skipping zero multiplications, though realizing this speedup requires sparse computation libraries.

### Structured vs Unstructured Pruning

Magnitude pruning creates unstructured sparsity: zeros scattered randomly throughout weight matrices. This achieves high compression ratios but creates irregular memory access patterns that modern hardware struggles to accelerate. Structured pruning solves this by removing entire computational units like channels, neurons, or attention heads.

Think of the difference like editing text. Unstructured pruning removes random letters from words, making them hard to read quickly. Structured pruning removes entire words or sentences, preserving readability while reducing length.

```
Unstructured Sparsity (Magnitude Pruning):
┌─────────────────────────────────────────┐
│ Channel 0: [2.1, 0.0, 1.8, 0.0, 3.2]    │ ← Scattered zeros
│ Channel 1: [0.0, 2.8, 0.0, 2.1, 0.0]    │ ← Irregular pattern
│ Channel 2: [1.5, 0.0, 2.4, 0.0, 1.9]    │ ← Hard to optimize
└─────────────────────────────────────────┘

Structured Sparsity (Channel Pruning):
┌─────────────────────────────────────────┐
│ Channel 0: [2.1, 1.3, 1.8, 0.9, 3.2]    │ ← Fully dense
│ Channel 1: [0.0, 0.0, 0.0, 0.0, 0.0]    │ ← Fully removed
│ Channel 2: [1.5, 2.2, 2.4, 1.1, 1.9]    │ ← Fully dense
└─────────────────────────────────────────┘
```

Structured pruning requires deciding which channels to remove. Your implementation uses L2 norm as an importance metric:

```python
def structured_prune(model, prune_ratio=0.5):
    """Remove entire channels based on L2 norm importance."""
    for layer in model.layers:
        if isinstance(layer, Linear):
            weight = layer.weight.data

            # Calculate L2 norm for each output channel (column)
            channel_norms = np.linalg.norm(weight, axis=0)

            # Find channels to prune (lowest importance)
            num_channels = weight.shape[1]
            num_to_prune = int(num_channels * prune_ratio)

            if num_to_prune > 0:
                # Get indices of smallest channels
                prune_indices = np.argpartition(channel_norms, num_to_prune)[:num_to_prune]

                # Zero out entire channels
                weight[:, prune_indices] = 0

                # Also zero corresponding bias elements
                if layer.bias is not None:
                    layer.bias.data[prune_indices] = 0

    return model
```

The L2 norm `||W[:,i]||_2 = sqrt(sum(w_j^2))` measures the total magnitude of all weights in a channel. Channels with small L2 norms contribute less to the output because all their weights are small. Removing entire channels creates block sparsity that hardware can exploit through vectorized operations on the remaining dense channels.

The key insight in structured pruning is that you remove entire computational units, not scattered weights. When you zero out channel `i` in layer `l`, you're eliminating:
- All connections from that channel to the next layer (forward propagation)
- All gradient computation for that channel (backward propagation)
- The entire channel's activation storage (memory savings)

This creates contiguous blocks of zeros that enable:
1. **Memory coalescing**: Hardware accesses dense remaining channels sequentially
2. **SIMD operations**: CPUs/GPUs process multiple channels in parallel
3. **No indexing overhead**: Don't need sparse matrix formats to track zero locations
4. **Cache efficiency**: Better spatial locality from accessing dense blocks

The trade-off is clear: structured pruning achieves lower sparsity (typically 30-50%) than magnitude pruning (80-90%), but the sparsity it creates enables real hardware acceleration. On GPUs and specialized accelerators, structured sparsity can provide 2-3x speedup, while unstructured sparsity requires custom sparse kernels to see any speedup at all.

### Knowledge Distillation

Knowledge distillation takes a different approach to compression: instead of removing weights from an existing model, train a smaller model to mimic a larger one's behavior. The large "teacher" model transfers its knowledge to the compact "student" model, achieving similar accuracy with dramatically fewer parameters.

The key innovation is using soft targets instead of hard labels. Traditional training uses one-hot labels: for a cat image, the label is `[0, 0, 1, 0]` (100% cat). But the teacher's predictions are softer: `[0.02, 0.05, 0.85, 0.08]` (85% cat, but some uncertainty about similar classes). These soft predictions contain richer information about class relationships that helps the student learn more effectively.

Temperature scaling controls how soft the distributions become:

```python
def _softmax(self, logits):
    """Compute softmax with numerical stability."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def distillation_loss(self, student_logits, teacher_logits, true_labels):
    """Calculate combined distillation loss."""
    # Soften distributions with temperature
    student_soft = self._softmax(student_logits / self.temperature)
    teacher_soft = self._softmax(teacher_logits / self.temperature)

    # Soft target loss (KL divergence)
    soft_loss = self._kl_divergence(student_soft, teacher_soft)

    # Hard target loss (cross-entropy)
    student_hard = self._softmax(student_logits)
    hard_loss = self._cross_entropy(student_hard, true_labels)

    # Combined loss
    total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    return total_loss
```

Dividing logits by temperature before softmax spreads probability mass across classes. With `temperature=1`, you get standard softmax with sharp peaks. With `temperature=3`, the distribution flattens, revealing the teacher's uncertainty. The student learns both what the teacher predicts (highest probability class) and what it considers similar (non-zero probabilities on other classes).

The combined loss balances two objectives. The soft loss (with `alpha=0.7`) teaches the student to match the teacher's reasoning process. The hard loss (with `1-alpha=0.3`) ensures the student still learns correct classifications. This combination typically achieves 10x compression with only 2-5% accuracy loss.

### Low-Rank Approximation Theory

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Low-rank example 1: (512, 256) matrix with rank_ratio=0.5
lr1_m, lr1_n = 512, 256
lr1_original = lr1_m * lr1_n
lr1_rank_ratio = 0.5
lr1_k = int(lr1_rank_ratio * min(lr1_m, lr1_n))
lr1_compressed = (lr1_m * lr1_k) + lr1_k + (lr1_k * lr1_n)
lr1_ratio = lr1_original / lr1_compressed
lr1_reduction_pct = (1 - lr1_compressed / lr1_original) * 100
glue("lr1_original", f"{lr1_original:,}")
glue("lr1_k", f"{lr1_k}")
glue("lr1_u", f"{lr1_m * lr1_k:,}")
glue("lr1_s", f"{lr1_k:,}")
glue("lr1_v", f"{lr1_k * lr1_n:,}")
glue("lr1_compressed", f"{lr1_compressed:,}")
glue("lr1_ratio", f"{lr1_ratio:.2f}x")
glue("lr1_reduction_pct", f"{lr1_reduction_pct:.0f}%")

# Low-rank example 2: (1024, 1024) matrix with rank_ratio=0.1
lr2_m, lr2_n = 1024, 1024
lr2_original = lr2_m * lr2_n
lr2_rank_ratio = 0.1
lr2_k = int(lr2_rank_ratio * min(lr2_m, lr2_n))
lr2_compressed = (lr2_m * lr2_k) + lr2_k + (lr2_k * lr2_n)
lr2_ratio = lr2_original / lr2_compressed
lr2_reduction_pct = (1 - lr2_compressed / lr2_original) * 100
glue("lr2_original", f"{lr2_original:,}")
glue("lr2_k", f"{lr2_k}")
glue("lr2_u", f"{lr2_m * lr2_k:,}")
glue("lr2_s", f"{lr2_k:,}")
glue("lr2_v", f"{lr2_k * lr2_n:,}")
glue("lr2_compressed", f"{lr2_compressed:,}")
glue("lr2_ratio", f"{lr2_ratio:.1f}x")
glue("lr2_reduction_pct", f"{lr2_reduction_pct:.0f}%")
```

Weight matrices in neural networks often contain redundancy that can be captured through low-rank approximations. Singular Value Decomposition (SVD) provides the mathematically optimal way to approximate a matrix with fewer parameters while minimizing reconstruction error.

The core idea is matrix factorization. Instead of storing a full (512, 256) weight matrix with {glue:text}`lr1_original` parameters, you decompose it into smaller factors that capture the essential structure:

```python
def low_rank_approximate(weight_matrix, rank_ratio=0.5):
    """Approximate weight matrix using SVD-based low-rank decomposition."""
    m, n = weight_matrix.shape

    # Perform SVD: W = U @ diag(S) @ V
    U, S, V = np.linalg.svd(weight_matrix, full_matrices=False)

    # Keep only top-k singular values
    max_rank = min(m, n)
    target_rank = max(1, int(rank_ratio * max_rank))

    # Truncate to target rank
    U_truncated = U[:, :target_rank]     # (m, k)
    S_truncated = S[:target_rank]         # (k,)
    V_truncated = V[:target_rank, :]      # (k, n)

    return U_truncated, S_truncated, V_truncated
```

SVD identifies the most important "directions" in the weight matrix through singular values. Larger singular values capture more variance, so keeping only the top k values preserves most of the matrix's information while dramatically reducing parameters.

For a (512, 256) matrix with rank_ratio=0.5:
- Original: 512 × 256 = {glue:text}`lr1_original` parameters
- Compressed: (512 × {glue:text}`lr1_k`) + {glue:text}`lr1_s` + ({glue:text}`lr1_k` × 256) = {glue:text}`lr1_compressed` parameters
- Compression ratio: {glue:text}`lr1_ratio` ({glue:text}`lr1_reduction_pct` reduction)

The compression ratio improves with larger matrices. For a (1024, 1024) matrix at rank_ratio=0.1:
- Original: {glue:text}`lr2_original` parameters
- Compressed: (1024 × {glue:text}`lr2_k`) + {glue:text}`lr2_s` + ({glue:text}`lr2_k` × 1024) = {glue:text}`lr2_compressed` parameters
- Compression ratio: {glue:text}`lr2_ratio` ({glue:text}`lr2_reduction_pct` reduction)

Low-rank approximation trades accuracy for size. The reconstruction error depends on the discarded singular values. Choosing the right rank_ratio balances compression and accuracy preservation.

### Compression Trade-offs

Every compression technique trades accuracy for efficiency, but different techniques make different trade-offs. Understanding these helps you choose the right approach for your deployment constraints.

| Technique | Compression Ratio | Accuracy Loss | Hardware Speedup | Training Required |
|-----------|-------------------|---------------|------------------|-------------------|
| **Magnitude Pruning** | 5-10x | 1-3% | Minimal (needs sparse libs) | No (prune pretrained) |
| **Structured Pruning** | 2-3x | 2-5% | 2-3x (hardware-friendly) | No (prune pretrained) |
| **Knowledge Distillation** | 10-50x | 5-10% | Proportional to size | Yes (train student) |
| **Low-Rank Approximation** | 2-5x | 3-7% | Minimal (depends on impl) | No (SVD decomposition) |

The systems insight is that compression ratio alone doesn't determine deployment success. A 10x compressed model with magnitude pruning might run slower than a 3x compressed model with structured pruning because hardware can't accelerate irregular sparsity. Similarly, knowledge distillation requires training infrastructure but achieves the best compression for a given accuracy target.

## Production Context

### Your Implementation vs. PyTorch

Your TinyTorch compression functions and PyTorch's pruning utilities share the same core algorithms. The differences are in integration depth: PyTorch provides hooks for pruning during training, automatic mask management, and integration with quantization-aware training. But the fundamental magnitude-based and structured pruning logic is identical.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Magnitude Pruning** | Global threshold via percentile | `torch.nn.utils.prune.l1_unstructured` |
| **Structured Pruning** | L2 norm channel removal | `torch.nn.utils.prune.ln_structured` |
| **Knowledge Distillation** | Manual loss calculation | User-implemented (same approach) |
| **Sparse Execution** | ✗ Dense NumPy arrays | ✓ Sparse tensors + kernels |
| **Pruning Schedules** | One-shot pruning | Iterative + fine-tuning |

### Code Comparison

The following comparison shows equivalent compression operations in TinyTorch and PyTorch. Notice how the core concepts translate directly while PyTorch provides additional automation for production workflows.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.perf.compression import magnitude_prune, measure_sparsity

# Create model
model = Sequential(Linear(100, 50), ReLU(), Linear(50, 10))

# Apply magnitude pruning
magnitude_prune(model, sparsity=0.8)

# Measure results
sparsity = measure_sparsity(model)  # Returns 80.0 (percentage)
print(f"Sparsity: {sparsity:.1f}%")
```
````

````{tab-item} ⚡ PyTorch
```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Create model
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# Apply magnitude pruning
for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.8)
        prune.remove(module, 'weight')  # Make pruning permanent

# Measure results
total_params = sum(p.numel() for p in model.parameters())
zero_params = sum((p == 0).sum().item() for p in model.parameters())
sparsity = zero_params / total_params
print(f"Sparsity: {sparsity:.1%}")
```
````
`````

Let's walk through the key differences:

- **Line 1 (Import)**: TinyTorch provides compression in a dedicated `perf.compression` module. PyTorch's `torch.nn.utils.prune` offers similar functionality with additional hooks.
- **Line 4-5 (Model)**: Both create identical model architectures. PyTorch's `nn.Sequential` matches TinyTorch's explicit layer composition.
- **Line 8 (Pruning)**: TinyTorch uses a simple function call that operates on the entire model. PyTorch requires iterating over modules and applying pruning individually, offering finer control.
- **Line 13 (Permanence)**: TinyTorch immediately zeros weights. PyTorch uses masks that can be removed or made permanent, enabling experimentation with different sparsity levels.
- **Line 16-19 (Measurement)**: TinyTorch provides a dedicated `measure_sparsity()` function. PyTorch requires manual counting, giving you full control over what counts as "sparse."

```{tip} What's Identical

The core algorithms for magnitude thresholding, L2 norm channel ranking, and knowledge distillation loss are identical. When you understand TinyTorch compression, you understand PyTorch compression. The production differences are in automation, not algorithms.
```

### Why Compression Matters at Scale

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Deployment example: 100MB model pruned to 90% sparsity
deploy_model_mb = 100
deploy_sparsity = 0.9
deploy_sparse_mb = deploy_model_mb * (1 - deploy_sparsity)
glue("deploy_model_mb", f"{deploy_model_mb} MB")
glue("deploy_sparse_mb", f"{deploy_sparse_mb:.0f} MB")
```

To appreciate compression's impact, consider real deployment constraints:

- **Mobile apps**: Models must fit in <10MB for reasonable download sizes and <50MB runtime memory
- **Edge devices**: Raspberry Pi 4 has 4GB RAM total, shared across OS and all applications
- **Cloud cost**: GPT-3 inference at scale costs $millions/month; 10x compression = $millions saved
- **Latency targets**: Self-driving cars need <100ms inference time; compression enables real-time decisions
- **Energy efficiency**: Smartphones have ~3000mAh batteries; model size directly impacts battery life

A {glue:text}`deploy_model_mb` model pruned to 90% sparsity becomes {glue:text}`deploy_sparse_mb` with sparse storage, fitting mobile constraints. The same model distilled to a 1MB student runs 10x faster, meeting latency requirements. These aren't theoretical gains; they're necessary for deployment.

## Check Your Understanding

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q1: Sparsity calculation for (512, 256) layer at 80% pruning
q1_rows, q1_cols = 512, 256
q1_total = q1_rows * q1_cols
q1_sparsity = 0.8
q1_active = int(q1_total * (1 - q1_sparsity))
q1_zeroed = int(q1_total * q1_sparsity)
glue("q1_total", f"{q1_total:,}")
glue("q1_active", f"{q1_active:,}")
glue("q1_zeroed", f"{q1_zeroed:,}")

# Q2: Sequential pruning (magnitude 90% then structured 50%)
q2_after_mag = 0.10  # 10% active after 90% magnitude
q2_after_struct = 0.50  # 50% channels remain
q2_final_active_pct = q2_after_mag * q2_after_struct * 100
q2_final_sparse_pct = 100 - q2_final_active_pct
glue("q2_final_active", f"{q2_final_active_pct:.0f}%")
glue("q2_final_sparse", f"{q2_final_sparse_pct:.0f}%")

# Q3: Knowledge distillation efficiency
q3_teacher_params = 100_000_000
q3_student_params = 10_000_000
q3_teacher_acc = 95
q3_student_acc = 92
q3_teacher_ms = 500
q3_student_ms = 50
q3_compression = q3_teacher_params / q3_student_params
q3_speedup = q3_teacher_ms / q3_student_ms
q3_acc_loss = q3_teacher_acc - q3_student_acc
glue("q3_compression", f"{q3_compression:.0f}x")
glue("q3_speedup", f"{q3_speedup:.0f}x")
glue("q3_acc_loss", f"{q3_acc_loss:.0f}%")

# Q4: Low-rank decomposition for (1000, 1000) with rank=100
q4_m, q4_n = 1000, 1000
q4_original = q4_m * q4_n
q4_rank = 100
q4_u_params = q4_m * q4_rank
q4_s_params = q4_rank
q4_v_params = q4_rank * q4_n
q4_compressed = q4_u_params + q4_s_params + q4_v_params
q4_ratio = q4_original / q4_compressed
q4_savings_bytes = (q4_original - q4_compressed) * 4
q4_savings_mb = q4_savings_bytes / 1024**2
glue("q4_original", f"{q4_original:,}")
glue("q4_u_params", f"{q4_u_params:,}")
glue("q4_s_params", f"{q4_s_params:,}")
glue("q4_v_params", f"{q4_v_params:,}")
glue("q4_compressed", f"{q4_compressed:,}")
glue("q4_ratio", f"~{q4_ratio:.0f}x")
glue("q4_savings_mb", f"{q4_savings_mb:.1f} MB")
```

Test yourself with these systems thinking questions. They're designed to build intuition for compression trade-offs you'll encounter in production.

**Q1: Sparsity Calculation**

A Linear layer with shape (512, 256) undergoes 80% magnitude pruning. How many weights remain active?

```{admonition} Answer
:class: dropdown

Total parameters: 512 × 256 = **{glue:text}`q1_total`**

After 80% pruning: 20% remain active = {glue:text}`q1_total` × 0.2 = **{glue:text}`q1_active` active weights**

Zeroed weights: {glue:text}`q1_total` × 0.8 = **{glue:text}`q1_zeroed` zeros**

This is why sparsity creates memory savings - 80% of parameters are literally zero!
```

**Q2: Compression Ratio Analysis**

You apply magnitude pruning (90% sparsity) and structured pruning (50% channels) sequentially. What's the final sparsity?

```{admonition} Answer
:class: dropdown

**Trick question!** Structured pruning zeros entire channels, which may already be partially sparse from magnitude pruning.

Approximation:
- After magnitude: 90% sparse → 10% active weights
- Structured removes 50% of channels → removes 50% of rows/columns
- Final active weights ≈ 10% × 50% = **{glue:text}`q2_final_active` active → {glue:text}`q2_final_sparse` sparse**

Actual result depends on which channels structured pruning removes. If it removes already-sparse channels, sparsity increases less.
```

**Q3: Knowledge Distillation Efficiency**

Teacher model: 100M parameters, 95% accuracy, 500ms inference
Student model: 10M parameters, 92% accuracy, 50ms inference

What's the compression ratio and speedup?

```{admonition} Answer
:class: dropdown

**Compression ratio**: 100M / 10M = **{glue:text}`q3_compression` smaller**

**Speedup**: 500ms / 50ms = **{glue:text}`q3_speedup` faster**

**Accuracy loss**: 95% - 92% = **{glue:text}`q3_acc_loss` degradation**

Why speedup matches compression: Student has 10x fewer parameters, so 10x fewer operations. Linear scaling!

Is this good? **Yes** - {glue:text}`q3_compression` compression with only {glue:text}`q3_acc_loss` accuracy loss is excellent for mobile deployment.
```

**Q4: Low-Rank Decomposition Math**

A (1000, 1000) weight matrix gets low-rank approximation with rank=100. Calculate parameter reduction.

```{admonition} Answer
:class: dropdown

Original: 1000 × 1000 = **{glue:text}`q4_original` parameters**

SVD decomposition: W ≈ U @ S @ V
- U: (1000, 100) = {glue:text}`q4_u_params` parameters
- S: (100,) = {glue:text}`q4_s_params` parameters (diagonal)
- V: (100, 1000) = {glue:text}`q4_v_params` parameters

Compressed: {glue:text}`q4_u_params` + {glue:text}`q4_s_params` + {glue:text}`q4_v_params` = **{glue:text}`q4_compressed` parameters**

Compression ratio: {glue:text}`q4_original` / {glue:text}`q4_compressed` = **{glue:text}`q4_ratio` reduction**

Memory savings: ({glue:text}`q4_original` - {glue:text}`q4_compressed`) × 4 bytes = **{glue:text}`q4_savings_mb` saved** (float32)
```

**Q5: Structured vs Unstructured Trade-offs**

For mobile deployment with tight latency constraints, would you choose magnitude pruning (90% sparsity) or structured pruning (30% sparsity)? Why?

```{admonition} Answer
:class: dropdown

**Choose structured pruning (30% sparsity)** despite lower compression.

Reasoning:
1. **Hardware acceleration**: Mobile CPUs/GPUs can execute dense channels 2-3x faster than sparse patterns
2. **Latency guarantee**: Structured sparsity gives predictable speedup; magnitude sparsity needs sparse libraries (often unavailable on mobile)
3. **Real speedup**: 30% structured = ~1.5x actual speedup; 90% magnitude = no speedup without custom kernels
4. **Memory**: Both save memory, but latency requirement dominates

**Production insight**: High sparsity ≠ high speedup. Hardware capabilities matter more than compression ratio for latency-critical applications.
```

## Further Reading

For students who want to understand the academic foundations and explore compression techniques further:

### Seminal Papers

- **Learning both Weights and Connections for Efficient Neural Networks** - Han et al. (2015). Introduced magnitude-based pruning and demonstrated 90% sparsity with minimal accuracy loss. Foundation for modern pruning research. [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)

- **The Lottery Ticket Hypothesis** - Frankle & Carlin (2019). Showed that dense networks contain sparse subnetworks trainable to full accuracy from initialization. Changed how we think about pruning and network over-parameterization. [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)

- **Distilling the Knowledge in a Neural Network** - Hinton et al. (2015). Introduced knowledge distillation with temperature scaling. Enables training compact models that match large model accuracy. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

- **Pruning Filters for Efficient ConvNets** - Li et al. (2017). Demonstrated structured pruning by removing entire convolutional filters. Showed that L1-norm ranking identifies unimportant channels effectively. [arXiv:1608.08710](https://arxiv.org/abs/1608.08710)

### Additional Resources

- **Survey**: "Model Compression and Hardware Acceleration for Neural Networks" by Deng et al. (2020) - Comprehensive overview of compression techniques and hardware implications
- **Tutorial**: [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) - See how production frameworks implement these concepts
- **Blog**: "The State of Sparsity in Deep Neural Networks" by Uber Engineering - Practical experiences deploying sparse models at scale

## What's Next

```{seealso} Coming Up: Module 17 - Acceleration

Implement hardware-aware optimization techniques including vectorized matrix operations, fused kernels, and cache-friendly tiling. You'll combine multiple operations to reduce memory bandwidth bottlenecks and maximize hardware utilization.
```

**Preview - How Your Compression Gets Used in Future Modules:**

| Module | What It Does | Your Compression In Action |
|--------|--------------|---------------------------|
| **17: Acceleration** | Optimize computation kernels | Structured sparsity enables vectorized operations |
| **18: Memoization** | Cache repeated computations | `compress_model()` before caching for memory efficiency |
| **19: Benchmarking** | Measure end-to-end performance | Compare dense vs sparse model throughput |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/16_compression/compression.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/16_compression/16_compression.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
