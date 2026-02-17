---
file_format: mystnb
kernelspec:
  name: python3
---

# Module 07: Optimizers

:::{admonition} Module Info
:class: note

**FOUNDATION TIER** | Difficulty: ●●○○ | Time: 3-5 hours | Prerequisites: 01-06

**Prerequisites: Modules 01-06** means you need:
- Tensor operations and parameter storage
- DataLoader for efficient batch processing
- Understanding of forward/backward passes (autograd)
- Why gradients point toward higher loss

If you understand how `loss.backward()` computes gradients and why we need to update parameters to minimize loss, you're ready.
:::

`````{only} html
````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/07_optimizers.mp3" type="audio/mpeg">
</audio>
```

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F07_optimizers%2Foptimizers.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/07_optimizers/07_optimizers.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #14b8a6; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
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

<div class="slide-viewer-container" id="slide-viewer-07_optimizers">
  <div class="slide-header">
    <div class="slide-title">
      <span>🔥</span>
      <span>Slide Deck</span>
      <span class="slide-subtitle">· AI-generated</span>
    </div>
    <div class="slide-toolbar">
      <div class="slide-nav-group">
        <button onclick="slideNav('07_optimizers', -1)" title="Previous">‹</button>
        <span class="slide-page-info"><span id="slide-num-07_optimizers">1</span> / <span id="slide-count-07_optimizers">-</span></span>
        <button onclick="slideNav('07_optimizers', 1)" title="Next">›</button>
      </div>
      <div class="slide-zoom-group">
        <button onclick="slideZoom('07_optimizers', -0.25)" title="Zoom out">−</button>
        <button onclick="slideZoom('07_optimizers', 0.25)" title="Zoom in">+</button>
      </div>
    </div>
  </div>
  <div class="slide-canvas-wrapper">
    <div id="slide-loading-07_optimizers" class="slide-loading">Loading slides...</div>
    <canvas id="slide-canvas-07_optimizers" class="slide-canvas" style="display:none;"></canvas>
  </div>
  <div class="slide-progress-wrapper">
    <div class="slide-progress-bar" onclick="slideProgress('07_optimizers', event)">
      <div class="slide-progress-fill" id="slide-progress-07_optimizers" style="width: 0%;"></div>
    </div>
  </div>
  <div class="slide-footer">
    <a href="../_static/slides/07_optimizers.pdf" download>⬇ Download</a>
    <a href="#" onclick="slideFullscreen('07_optimizers'); return false;" class="secondary">⛶ Fullscreen</a>
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

initSlideViewer('07_optimizers', '../_static/slides/07_optimizers.pdf');
</script>
```
`````

## Overview

Optimizers are the engines that drive neural network learning. After your autograd system computes gradients that point uphill toward higher loss, optimizers use those gradients to move parameters downhill toward lower loss. Think of optimization as hiking in dense fog where you can only feel the slope under your feet but can't see where you're going. Different optimizers represent different hiking strategies, from simple gradient descent to sophisticated algorithms that adapt their step size for each parameter.

In this module, you'll build three production-grade optimizers: SGD with momentum (the foundation algorithm), Adam with adaptive learning rates (the workhorse of modern deep learning), and AdamW with decoupled weight decay (the state-of-the-art for transformers). These optimizers differ dramatically in memory usage, convergence speed, and numerical behavior.

By the end, you'll understand not just how optimizers work but also the systems trade-offs between them: SGD uses 2x parameter memory while Adam uses 3x, but Adam often converges in fewer steps.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** SGD with momentum to reduce oscillations and accelerate convergence in narrow valleys
- **Master** Adam's adaptive learning rate mechanism with first and second moment estimation
- **Understand** memory trade-offs (SGD: 2x memory vs Adam: 3x memory) and computational complexity per step
- **Connect** optimizer state management to checkpointing and distributed training considerations
```

## What You'll Build

```{mermaid}
:align: center
:caption: Your Optimizer Classes
flowchart LR
    subgraph "Your Optimizer Classes"
        A["Optimizer Base<br/>zero_grad(), step()"]
        B["SGD<br/>momentum buffers"]
        C["Adam<br/>m, v buffers"]
        D["AdamW<br/>decoupled decay"]
    end

    A --> B --> C --> D

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
```

**Implementation roadmap:**

| Step | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `Optimizer` base class | Common interface: zero_grad(), step() |
| 2 | `SGD` with momentum | Velocity buffers to reduce oscillations |
| 3 | `Adam` optimizer | First and second moment estimation with bias correction |
| 4 | `AdamW` optimizer | Decoupled weight decay for proper regularization |

**The pattern you'll enable:**
```python
# Training loop with optimizer
optimizer = Adam(model.parameters(), lr=0.001)
loss.backward()  # Compute gradients (Module 06)
optimizer.step()  # Update parameters using gradients
optimizer.zero_grad()  # Clear gradients for next iteration
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Learning rate schedules (that's Module 08: Training)
- Gradient clipping (PyTorch provides this via `torch.nn.utils.clip_grad_norm_`)
- Second-order optimizers like L-BFGS (rarely used in deep learning due to memory cost)
- Distributed optimizer sharding (production frameworks use techniques like ZeRO)

**You are building the core optimization algorithms.** Advanced training techniques come in Module 08.

## API Reference

This section provides a quick reference for the Optimizer classes you'll build. Use this as your guide while implementing and debugging.

### Optimizer Base Class

```python
Optimizer(params: List[Tensor])
```

Base class defining the optimizer interface. All optimizers inherit from this.

| Method | Signature | Description |
|--------|-----------|-------------|
| `zero_grad` | `zero_grad() -> None` | Clear gradients from all parameters |
| `step` | `step() -> None` | Update parameters (implemented by subclasses) |

### SGD Optimizer

```python
SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)
```

Stochastic Gradient Descent with optional momentum and weight decay.

**Parameters:**
- `params`: List of Tensor parameters to optimize
- `lr`: Learning rate (step size, default: 0.01)
- `momentum`: Momentum factor (0.0-1.0, typically 0.9, default: 0.0)
- `weight_decay`: L2 penalty coefficient (default: 0.0)

**Update rule:**
- Without momentum: `param = param - lr * grad`
- With momentum: `v = momentum * v + grad; param = param - lr * v`

**State management methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `has_momentum` | `has_momentum() -> bool` | Check if optimizer uses momentum (momentum > 0) |
| `get_momentum_state` | `get_momentum_state() -> Optional[List]` | Get momentum buffers for checkpointing |
| `set_momentum_state` | `set_momentum_state(state: Optional[List]) -> None` | Restore momentum buffers from checkpoint |

### Adam Optimizer

```python
Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
```

Adaptive Moment Estimation with per-parameter learning rates.

**Parameters:**
- `params`: List of Tensor parameters to optimize
- `lr`: Learning rate (default: 0.001)
- `betas`: Tuple of coefficients (β₁, β₂) for computing running averages (default: (0.9, 0.999))
- `eps`: Small constant for numerical stability (default: 1e-8)
- `weight_decay`: L2 penalty coefficient (default: 0.0)

**State:**
- `m_buffers`: First moment estimates (momentum of gradients)
- `v_buffers`: Second moment estimates (momentum of squared gradients)

### AdamW Optimizer

```python
AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
```

Adam with decoupled weight decay regularization.

**Parameters:**
- `params`: List of Tensor parameters to optimize
- `lr`: Learning rate (default: 0.001)
- `betas`: Tuple of coefficients (β₁, β₂) for computing running averages (default: (0.9, 0.999))
- `eps`: Small constant for numerical stability (default: 1e-8)
- `weight_decay`: L2 penalty coefficient (default: 0.01, higher than Adam)

**Key difference from Adam:** Weight decay is applied directly to parameters after gradient update, not mixed into the gradient.

## Core Concepts

This section covers the fundamental ideas you need to understand optimization deeply. These concepts apply across all ML frameworks and will serve you throughout your career in machine learning systems.

### Gradient Descent Fundamentals

Gradient descent is conceptually simple: gradients point uphill toward higher loss, so we step downhill by moving in the opposite direction. The gradient ∇L tells us the direction of steepest ascent, so -∇L points toward steepest descent.

The basic update rule is: **θ_new = θ_old - α * ∇L**, where θ represents parameters and α is the learning rate (step size). This simple formula hides important challenges. How large should steps be? What if different parameters need different step sizes? What about noisy gradients or narrow valleys that cause oscillation?

Here's how your SGD implementation handles the basic case without momentum:

```python
def step(self):
    """Perform SGD update step with momentum."""
    for i, param in enumerate(self.params):
        if param.grad is None:
            continue

        # Get gradient data
        grad = param.grad
        if isinstance(grad, Tensor):
            grad_data = grad.data
        else:
            grad_data = grad

        # Apply weight decay if specified
        if self.weight_decay != 0:
            grad_data = grad_data + self.weight_decay * param.data

        # Update parameter: param = param - lr * grad
        param.data = param.data - self.lr * grad_data

    self.step_count += 1
```

The code reveals the simplicity of basic SGD: subtract learning rate times gradient from each parameter. But this simplicity comes with a cost: plain SGD can oscillate wildly in narrow valleys of the loss landscape.

### Momentum and Acceleration

Momentum solves the oscillation problem by remembering previous update directions. Think of a ball rolling down a hill: it doesn't immediately change direction when it hits a small bump because it has momentum carrying it forward. In optimization, momentum accumulates velocity in directions that gradients consistently agree on, while oscillations in perpendicular directions cancel out.

The momentum update maintains a velocity buffer v for each parameter: **v = β * v_prev + grad** and then **param = param - lr * v**. The momentum coefficient β (typically 0.9) controls how much previous direction we remember. With β=0.9, we keep 90% of the old velocity and add 10% of the current gradient.

Here's how your SGD implementation adds momentum:

```python
# Update momentum buffer
if self.momentum != 0:
    if self.momentum_buffers[i] is None:
        # Initialize momentum buffer on first use
        self.momentum_buffers[i] = np.zeros_like(param.data)

    # Update momentum: v = momentum * v_prev + grad
    self.momentum_buffers[i] = self.momentum * self.momentum_buffers[i] + grad_data
    grad_data = self.momentum_buffers[i]

# Update parameter: param = param - lr * grad
param.data = param.data - self.lr * grad_data
```

The momentum buffer is initialized lazily (only when first needed) to save memory for optimizers without momentum. Once initialized, each step accumulates 90% of the previous velocity plus the current gradient, creating a smoothed update direction that's less susceptible to noise and oscillation.

### Adam and Adaptive Learning Rates

Adam solves a fundamental problem: different parameters often need different learning rates. Consider a neural network with embedding weights ranging from -0.01 to 0.01 and output weights ranging from -10 to 10. A learning rate that works well for embeddings might cause output weights to explode, while a rate that's safe for output weights makes embeddings learn too slowly.

Adam addresses this by maintaining two statistics for each parameter: a first moment m (exponential moving average of gradients) and a second moment v (exponential moving average of squared gradients). The ratio m/√v gives an adaptive step size: parameters with large gradients get smaller effective learning rates, while parameters with small gradients get larger effective rates.

The algorithm tracks: **m = β₁ * m_prev + (1-β₁) * grad** and **v = β₂ * v_prev + (1-β₂) * grad²**. Then it corrects for initialization bias (m and v start at zero) and updates: **param = param - lr * m̂ / (√v̂ + ε)**, where m̂ and v̂ are bias-corrected moments.

Here's the complete Adam update from your implementation:

```python
def step(self):
    """Perform Adam update step."""
    self.step_count += 1

    for i, param in enumerate(self.params):
        if param.grad is None:
            continue

        grad = param.grad
        if isinstance(grad, Tensor):
            grad_data = grad.data
        else:
            grad_data = grad

        # Initialize buffers if needed
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(param.data)
            self.v_buffers[i] = np.zeros_like(param.data)

        # Update biased first moment estimate
        self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data

        # Update biased second moment estimate
        self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)

        # Compute bias correction
        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count

        # Compute bias-corrected moments
        m_hat = self.m_buffers[i] / bias_correction1
        v_hat = self.v_buffers[i] / bias_correction2

        # Update parameter
        param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

The bias correction terms (1 - β^t) are crucial in the first few steps. Without correction, m and v start at zero and take many steps to reach reasonable values, causing the optimizer to take tiny steps initially. The correction divides by increasingly large values: at step 1, divide by 0.1; at step 2, divide by 0.19; eventually the correction approaches 1.0 and has no effect.

### AdamW and Decoupled Weight Decay

AdamW fixes a subtle but important bug in Adam's weight decay implementation. In standard Adam, weight decay is added to the gradient before adaptive scaling: **grad = grad + λ * param**, then proceed with normal Adam. This seems reasonable but creates a problem: the weight decay effect gets scaled by the adaptive learning rate mechanism, making regularization inconsistent across parameters.

Parameters with large gradients get small adaptive learning rates, which also makes their weight decay small. Parameters with small gradients get large adaptive learning rates, which amplifies their weight decay. This is backwards: we want consistent regularization regardless of gradient magnitudes.

AdamW decouples weight decay from the gradient by applying it directly to parameters after the gradient update: first update using pure gradients with Adam's adaptive mechanism, then separately shrink parameters by a fixed proportion. This ensures regularization strength is consistent across all parameters.

Here's how your AdamW implementation achieves decoupling:

```python
# Update moments using pure gradients (NO weight decay mixed in)
self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)

# Compute bias correction and bias-corrected moments
bias_correction1 = 1 - self.beta1 ** self.step_count
bias_correction2 = 1 - self.beta2 ** self.step_count
m_hat = self.m_buffers[i] / bias_correction1
v_hat = self.v_buffers[i] / bias_correction2

# Apply gradient-based update
param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Apply decoupled weight decay (separate from gradient update)
if self.weight_decay != 0:
    param.data = param.data * (1 - self.lr * self.weight_decay)
```

Notice that weight decay appears only at the end, multiplying parameters by (1 - lr * weight_decay) to shrink them slightly. This shrinkage happens after the gradient update and is completely independent of gradient magnitudes or adaptive scaling.

### Learning Rate Selection

Learning rate is the single most important hyperparameter in optimization. Too large, and parameters oscillate or diverge. Too small, and training takes forever or gets stuck in poor local minima. The optimal learning rate depends on the optimizer, network architecture, dataset, and batch size.

For SGD, learning rates typically range from 0.001 to 0.1. SGD is very sensitive to learning rate choice and often requires careful tuning or learning rate schedules. Momentum helps but doesn't eliminate the sensitivity.

For Adam and AdamW, the default learning rate of 0.001 works well across many problems. The adaptive mechanism provides some robustness to learning rate choice. However, transformers often use smaller rates (0.0001 to 0.0003) with warmup periods where the rate gradually increases from zero.

The relationship between learning rate and batch size matters for distributed training. Larger batches provide less noisy gradients, allowing larger learning rates. A common heuristic is to scale learning rate linearly with batch size: if you double the batch size from 32 to 64, double the learning rate from 0.001 to 0.002.

## Production Context

### Your Implementation vs. PyTorch

Your TinyTorch optimizers and PyTorch's `torch.optim` share the same algorithmic foundations and API patterns. The differences lie in implementation details: PyTorch uses optimized C++/CUDA kernels, supports mixed precision training, and includes specialized optimizers for specific domains.

| Feature | Your Implementation | PyTorch |
|---------|---------------------|---------|
| **Backend** | NumPy (Python) | C++/CUDA kernels |
| **Speed** | 1x (baseline) | 10-50x faster |
| **Memory** | Same asymptotic cost | Same (3x for Adam) |
| **State management** | Manual buffers | Automatic state_dict() |
| **Optimizers** | SGD, Adam, AdamW | 10+ algorithms (RMSprop, Adagrad, etc.) |

### Code Comparison

The following comparison shows how optimizer usage looks nearly identical in TinyTorch and PyTorch. This similarity is intentional: by learning TinyTorch's patterns, you're simultaneously learning production PyTorch patterns.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.core.optimizers import Adam

# Create optimizer for model parameters
optimizer = Adam(model.parameters(), lr=0.001)

# Training step
loss = criterion(predictions, targets)
loss.backward()  # Compute gradients
optimizer.step()  # Update parameters
optimizer.zero_grad()  # Clear gradients
```
````

````{tab-item} ⚡ PyTorch
```python
import torch.optim as optim

# Create optimizer for model parameters
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training step
loss = criterion(predictions, targets)
loss.backward()  # Compute gradients
optimizer.step()  # Update parameters
optimizer.zero_grad()  # Clear gradients
```
````
`````

Let's walk through each line to understand the comparison:

- **Line 1 (Import)**: TinyTorch exposes optimizers from `tinytorch.core.optimizers`; PyTorch uses `torch.optim`. The namespace structure mirrors production frameworks.
- **Line 4 (Creation)**: Both use identical syntax: `Adam(model.parameters(), lr=0.001)`. The `model.parameters()` method returns an iterable of tensors with `requires_grad=True`.
- **Line 7-8 (Training)**: The loss computation and backward pass are identical. Your autograd system from Module 06 computes gradients just like PyTorch.
- **Line 9 (Update)**: Both call `optimizer.step()` to update parameters using computed gradients. The update rules are mathematically identical.
- **Line 10 (Clear)**: Both call `optimizer.zero_grad()` to clear gradients before the next iteration. Without this, gradients would accumulate across batches.

```{tip} What's Identical

The optimizer API, update algorithms, and memory patterns are identical. When you debug Adam's learning rate or analyze optimizer memory usage in production, you'll understand exactly what's happening because you built these mechanisms yourself.
```

### Why Optimizers Matter at Scale

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# 175B-parameter model: optimizer state with Adam
scale_params = 175_000_000_000
scale_bytes_per_param = 4
scale_param_bytes = scale_params * scale_bytes_per_param
scale_param_gb = scale_param_bytes / 1024**3
scale_state_bytes = 2 * scale_param_bytes  # 2 Adam buffers (m, v)
scale_state_tb = scale_state_bytes / 1024**4
scale_multiplier = 3  # params + 2 state buffers

glue("scale_param_gb", f"{scale_param_gb:.1f} GB")
glue("scale_state_tb", f"{scale_state_tb:.2f} TB")
glue("scale_multiplier", f"{scale_multiplier}x")
```

To appreciate optimizer importance, consider production training scenarios:

- **Large language models (175B parameters)**: Optimizer state alone consumes **{glue:text}`scale_state_tb`** with Adam ({glue:text}`scale_multiplier` x {glue:text}`scale_param_gb` parameters), requiring multi-GPU state sharding
- **Transformer training**: AdamW with weight_decay=0.01 is standard, improving generalization over plain Adam by 2-5% accuracy
- **Convergence speed**: Adam typically converges in **30-50% fewer steps** than SGD on vision and language tasks, saving hours of GPU time despite higher memory cost

The optimizer choice directly impacts training feasibility. For models that barely fit in memory with SGD, switching to Adam might require distributed training or gradient checkpointing to handle the 1.5x memory increase.

## Check Your Understanding

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q1: Memory calculation for 10B-parameter model (float32)
q1_params = 10_000_000_000
q1_bytes_per_param = 4
q1_param_bytes = q1_params * q1_bytes_per_param
q1_param_gb = q1_param_bytes / 1024**3

q1_adam_state_gb = 2 * q1_param_gb       # 2 buffers (m, v)
q1_total_adam_gb = q1_param_gb + q1_adam_state_gb

q1_sgd_state_gb = q1_param_gb            # 1 buffer (velocity)
q1_total_sgd_gb = q1_param_gb + q1_sgd_state_gb

q1_diff_gb = q1_total_adam_gb - q1_total_sgd_gb

glue("q1_param_gb", f"{q1_param_gb:.2f} GB")
glue("q1_adam_state_gb", f"{q1_adam_state_gb:.2f} GB")
glue("q1_total_adam_gb", f"{q1_total_adam_gb:.2f} GB")
glue("q1_sgd_state_gb", f"{q1_sgd_state_gb:.2f} GB")
glue("q1_total_sgd_gb", f"{q1_total_sgd_gb:.2f} GB")
glue("q1_diff_gb", f"{q1_diff_gb:.2f} GB")
```

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q2: Convergence trade-off
q2_adam_steps = 100_000
q2_adam_overhead = 1.2
q2_sgd_steps = 200_000
q2_sgd_overhead = 1.0

q2_adam_time = q2_adam_steps * q2_adam_overhead
q2_sgd_time = q2_sgd_steps * q2_sgd_overhead
q2_speedup = q2_sgd_time / q2_adam_time

glue("q2_adam_time", f"{q2_adam_time:,.0f}")
glue("q2_sgd_time", f"{q2_sgd_time:,.0f}")
glue("q2_speedup", f"{q2_speedup:.2f}x")
```

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q3: Bias correction impact
q3_beta1 = 0.9

q3_corr_step1 = 1 - q3_beta1 ** 1
q3_corr_step10 = 1 - q3_beta1 ** 10
q3_corr_step100 = 1 - q3_beta1 ** 100

q3_mult_step1 = 1 / q3_corr_step1
q3_mult_step10 = 1 / q3_corr_step10
q3_mult_step100 = 1 / q3_corr_step100

glue("q3_corr_step1", f"{q3_corr_step1:.1f}")
glue("q3_corr_step10", f"{q3_corr_step10:.3f}")
glue("q3_corr_step100", f"{q3_corr_step100:.4f}")
glue("q3_mult_step1", f"{q3_mult_step1:.0f}x")
glue("q3_mult_step10", f"{q3_mult_step10:.2f}x")
glue("q3_mult_step100", f"{q3_mult_step100:.1f}x")
```

Test yourself with these systems thinking questions designed to build intuition for optimization trade-offs in production ML.

**Q1: Memory Calculation**

A language model has 10 billion float32 parameters. Using Adam optimizer, how much total memory does optimizer state require? How does this compare to SGD with momentum?

```{admonition} Answer
:class: dropdown

**Parameters:** 10B x 4 bytes = **{glue:text}`q1_param_gb`**

**Adam state:** 2 buffers (m, v) = 2 x {glue:text}`q1_param_gb` = **{glue:text}`q1_adam_state_gb`**
**Total with Adam:** {glue:text}`q1_param_gb` (params) + {glue:text}`q1_adam_state_gb` (state) = **{glue:text}`q1_total_adam_gb`**

**SGD with momentum:** 1 buffer (velocity) = **{glue:text}`q1_sgd_state_gb`**
**Total with SGD:** {glue:text}`q1_param_gb` (params) + {glue:text}`q1_sgd_state_gb` (state) = **{glue:text}`q1_total_sgd_gb`**

**Difference:** Adam uses **{glue:text}`q1_diff_gb` more** than SGD (50% increase). This might force you to use fewer GPUs or implement optimizer state sharding.
```

**Q2: Convergence Trade-off**

If Adam converges in 100,000 steps and SGD needs 200,000 steps, but Adam's per-step time is 1.2x slower due to additional computations, which optimizer finishes training faster?

```{admonition} Answer
:class: dropdown

**Adam:** 100,000 steps x 1.2 = **{glue:text}`q2_adam_time` time units**
**SGD:** 200,000 steps x 1.0 = **{glue:text}`q2_sgd_time` time units**

**Adam finishes {glue:text}`q2_speedup` faster** despite higher per-step cost. The convergence advantage (2x fewer steps) outweighs the computational overhead (1.2x slower steps).

This illustrates why Adam is popular despite higher memory and compute: wall-clock time to convergence often matters more than per-step efficiency.
```

**Q3: Bias Correction Impact**

In Adam, bias correction divides first moment by (1 - β₁^t). At step 1 with β₁=0.9, this correction factor is {glue:text}`q3_corr_step1`. At step 10, it's {glue:text}`q3_corr_step10`. How does this affect early vs late training?

```{admonition} Answer
:class: dropdown

**Step 1:** Divide by {glue:text}`q3_corr_step1` = multiply by **{glue:text}`q3_mult_step1`** (huge correction)
**Step 10:** Divide by {glue:text}`q3_corr_step10` = multiply by **{glue:text}`q3_mult_step10`** (moderate correction)
**Step 100:** Divide by {glue:text}`q3_corr_step100` ≈ multiply by **{glue:text}`q3_mult_step100`** (negligible correction)

**Early training:** Large corrections amplify small moment estimates to reasonable magnitudes, enabling effective learning from the first step.

**Late training:** Corrections approach 1.0 and have minimal effect, so the algorithm uses raw moment estimates.

**Without correction:** First moment m starts at 0, making initial steps tiny (learning rate effectively 0.1x intended). Training would be very slow initially.
```

**Q4: Weight Decay Comparison**

Adam adds weight decay to gradients before adaptive scaling. AdamW applies it after. For a parameter with grad=0.001 and param=1.0, which experiences stronger regularization with weight_decay=0.01 and lr=0.1?

```{admonition} Answer
:class: dropdown

**Adam approach:**
- Modified grad = 0.001 + 0.01 × 1.0 = 0.011
- This gradient gets adaptively scaled (divided by √v, which is small for small gradients)
- Effective decay is amplified by adaptive scaling

**AdamW approach:**
- Pure gradient update uses grad=0.001 (small adaptive step)
- Then param = param × (1 - 0.1 × 0.01) = param × 0.999 (fixed 0.1% shrinkage)

**AdamW has consistent 0.1% weight decay** regardless of gradient magnitude. Adam's decay strength varies with adaptive learning rate scaling, making it inconsistent across parameters. AdamW's consistency leads to better regularization behavior.
```

**Q5: Optimizer State Checkpointing**

You're training with Adam and checkpoint every 1000 steps. The checkpoint saves parameters and optimizer state (m, v buffers). If you resume from step 5000 but change learning rate from 0.001 to 0.0001, should you restore old optimizer state or reset it?

```{admonition} Answer
:class: dropdown

**Restore state (recommended):** The m and v buffers contain valuable information about gradient statistics accumulated over 5000 steps. Resetting loses this and causes the optimizer to "forget" learned gradient scales.

**Impact of restoring:**
- Keeps adaptive learning rates calibrated to parameter-specific gradient magnitudes
- Prevents slow re-convergence that happens when resetting
- Learning rate change affects step size but not the adaptive scaling

**When to reset:**
- If switching optimizer types (SGD → Adam)
- If gradient distribution has fundamentally changed (switching datasets)
- If debugging and suspecting corrupted state

**Production practice:** Always restore optimizer state when resuming training unless you have specific reasons to reset. The state is part of what makes Adam effective.
```

## Further Reading

For students who want to understand the academic foundations and mathematical underpinnings of optimization algorithms:

### Seminal Papers

- **Adam: A Method for Stochastic Optimization** - Kingma & Ba (2015). The original Adam paper introducing adaptive moment estimation with bias correction. Explains the motivation and derivation. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

- **Decoupled Weight Decay Regularization (AdamW)** - Loshchilov & Hutter (2019). Identifies the weight decay bug in Adam and proposes the decoupled fix. Shows significant improvements on image classification and language modeling. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

- **On the Importance of Initialization and Momentum in Deep Learning** - Sutskever et al. (2013). Classic paper explaining why momentum works and how it accelerates convergence in deep networks. [ICML 2013](http://proceedings.mlr.press/v28/sutskever13.pdf)

### Additional Resources

- **Tutorial**: "An overview of gradient descent optimization algorithms" by Sebastian Ruder - Comprehensive survey covering SGD variants, momentum methods, and adaptive learning rate algorithms
- **Documentation**: [PyTorch Optimization Documentation](https://pytorch.org/docs/stable/optim.html) - See how production frameworks organize and document optimization algorithms

## What's Next

```{seealso} Coming Up: Module 08 - Training

Combine optimizers with training loops to actually train neural networks. You'll implement learning rate scheduling, checkpointing, and the complete training/validation workflow that makes everything work together.
```

**Preview - How Your Optimizers Get Used in Future Modules:**

| Module | What It Does | Your Optimizers In Action |
|--------|--------------|---------------------------|
| **08: Training** | Complete training loops | `for epoch in range(10): loss.backward(); optimizer.step()` |
| **09: Convolutions** | Convolutional networks | `AdamW` optimizes millions of CNN parameters efficiently |
| **13: Transformers** | Attention mechanisms | Large models require careful optimizer selection |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/07_optimizers/optimizers.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/07_optimizers/07_optimizers.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
