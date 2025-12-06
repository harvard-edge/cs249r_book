<!-- Main heading -->
<h1 style="text-align: center; font-size: 3rem; margin: 0rem 0 0.5rem 0; font-weight: 700;">
Build Your Own ML Framework
</h1>

<p style="text-align: center; margin: 0 0 1.5rem 0;">
<a href="https://mlsysbook.ai" target="_blank" class="textbook-link" style="color: #64748b; font-size: 0.95rem; text-decoration: none; border-bottom: 1px solid #cbd5e1; transition: all 0.2s ease;">
Hands-on labs for the <span style="font-weight: 600; color: #475569;">Machine Learning Systems</span> textbook
</a>
</p>

<h2 style="background: linear-gradient(135deg, #E74C3C 0%, #E67E22 50%, #F39C12 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; font-size: 2.5rem; margin: 1.5rem 0 1rem 0; font-weight: 700;">
Don't import it. Build it.
</h2>

<!-- Enhanced description: Added "machine learning (ML)" clarification and "under the hood"
     to emphasize deep understanding of framework internals -->
<p style="text-align: center; font-size: 1.2rem; margin: 0 auto 2rem auto; max-width: 800px; color: #374151;">
Build a complete machine learning (ML) framework from tensors to systems—understand how PyTorch, TensorFlow, and JAX really work under the hood.
</p>

```{raw} html
<!-- Hero GIF Carousel - Compact Design -->
<div class="hero-carousel-compact">
  <div class="carousel-track">
    <div class="carousel-item active">
      <div class="gif-preview">
        <img src="_static/demos/01-clone-setup.gif" alt="Clone & Setup workflow" loading="lazy" />
        <div class="preview-fallback">💻</div>
      </div>
    </div>

    <div class="carousel-item">
      <div class="gif-preview">
        <img src="_static/demos/02-build-jupyter.gif" alt="Build in Jupyter workflow" loading="lazy" />
        <div class="preview-fallback">📓</div>
      </div>
    </div>

    <div class="carousel-item">
      <div class="gif-preview">
        <img src="_static/demos/03-export-tito.gif" alt="Export with TITO workflow" loading="lazy" />
        <div class="preview-fallback">🛠️</div>
      </div>
    </div>

    <div class="carousel-item">
      <div class="gif-preview">
        <img src="_static/demos/04-validate-history.gif" alt="Validate with History workflow" loading="lazy" />
        <div class="preview-fallback">🏆</div>
      </div>
    </div>
  </div>

  <div class="carousel-nav">
    <button class="nav-arrow prev" onclick="moveCarousel(-1)">←</button>
    <button class="nav-arrow next" onclick="moveCarousel(1)">→</button>
  </div>
</div>
```

<div style="text-align: center; margin: 2rem 0;">
  <a href="quickstart-guide.html" style="display: inline-block; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; padding: 0.875rem 2rem; border-radius: 0.5rem; text-decoration: none; font-weight: 600; font-size: 1rem; margin: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
    Start Building in 15 Minutes →
  </a>
</div>

## Getting Started

TinyTorch is organized into **four progressive tiers** that take you from mathematical foundations to production-ready systems. Each tier builds on the previous one, teaching you not just how to code ML components, but how they work together as a complete system.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2rem 0 2.5rem 0; max-width: 1100px;">

<a href="tiers/foundation.html" class="tier-card" style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #1976d2; text-decoration: none; display: block; transition: transform 0.2s ease, box-shadow 0.2s ease;">
<h3 style="margin: 0 0 0.75rem 0; color: #0d47a1; font-size: 1.15rem; font-weight: 600;">🏗 Foundation (Modules 01-07)</h3>
<p style="margin: 0 0 0.75rem 0; color: #1565c0; font-size: 0.95rem; line-height: 1.6;">Build the mathematical core that makes neural networks learn.</p>
<p style="margin: 0.75rem 0 0 0; color: #0d47a1; font-size: 0.85rem; font-style: italic;">
Unlocks: Perceptron (1957) • XOR Crisis (1969) • MLP (1986)
</p>
</a>

<a href="tiers/architecture.html" class="tier-card" style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #7b1fa2; text-decoration: none; display: block; transition: transform 0.2s ease, box-shadow 0.2s ease;">
<h3 style="margin: 0 0 0.75rem 0; color: #4a148c; font-size: 1.15rem; font-weight: 600;">🏛️ Architecture (Modules 08-13)</h3>
<p style="margin: 0 0 0.75rem 0; color: #6a1b9a; font-size: 0.95rem; line-height: 1.6;">Build modern neural architectures—from computer vision to language models.</p>
<p style="margin: 0.75rem 0 0 0; color: #4a148c; font-size: 0.85rem; font-style: italic;">
Unlocks: CNN Revolution (1998) • Transformer Era (2017)
</p>
</a>

<a href="tiers/optimization.html" class="tier-card" style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #f57c00; text-decoration: none; display: block; transition: transform 0.2s ease, box-shadow 0.2s ease;">
<h3 style="margin: 0 0 0.75rem 0; color: #e65100; font-size: 1.15rem; font-weight: 600;">⏱️ Optimization (Modules 14-19)</h3>
<p style="margin: 0 0 0.75rem 0; color: #ef6c00; font-size: 0.95rem; line-height: 1.6;">Transform research prototypes into production-ready systems.</p>
<p style="margin: 0.75rem 0 0 0; color: #e65100; font-size: 0.85rem; font-style: italic;">
Unlocks: MLPerf Torch Olympics (2018) • 8-16× compression • 12-40× speedup
</p>
</a>

<a href="tiers/olympics.html" class="tier-card" style="background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #c2185b; text-decoration: none; display: block; transition: transform 0.2s ease, box-shadow 0.2s ease;">
<h3 style="margin: 0 0 0.75rem 0; color: #880e4f; font-size: 1.15rem; font-weight: 600;">🏅 Torch Olympics (Module 20)</h3>
<p style="margin: 0 0 0.75rem 0; color: #ad1457; font-size: 0.95rem; line-height: 1.6;">The ultimate test: Build a complete, competition-ready ML system.</p>
<p style="margin: 0.75rem 0 0 0; color: #880e4f; font-size: 0.85rem; font-style: italic;">
Capstone: Vision • Language • Speed • Compression tracks
</p>
</a>

</div>

**[Complete course structure](chapters/00-introduction)** • **[Getting started guide](getting-started)** • **[Join the community](community)**

## Recreate ML History

Walk through ML history by rebuilding its greatest breakthroughs with YOUR TinyTorch implementations. Click each milestone to see what you'll build and how it shaped modern AI.

```{raw} html
<div class="ml-timeline-container">
    <div class="ml-timeline-line"></div>

    <div class="ml-timeline-item left perceptron">
        <div class="ml-timeline-dot"></div>
        <div class="ml-timeline-content">
            <div class="ml-timeline-year">1957</div>
            <div class="ml-timeline-title">The Perceptron</div>
            <div class="ml-timeline-desc">The first trainable neural network</div>
            <div class="ml-timeline-tech">Input → Linear → Sigmoid → Output</div>
        </div>
    </div>

    <div class="ml-timeline-item right xor">
        <div class="ml-timeline-dot"></div>
        <div class="ml-timeline-content">
            <div class="ml-timeline-year">1969</div>
            <div class="ml-timeline-title">XOR Crisis Solved</div>
            <div class="ml-timeline-desc">Hidden layers unlock non-linear learning</div>
            <div class="ml-timeline-tech">Input → Linear → ReLU → Linear → Output</div>
        </div>
    </div>

    <div class="ml-timeline-item left mlp">
        <div class="ml-timeline-dot"></div>
        <div class="ml-timeline-content">
            <div class="ml-timeline-year">1986</div>
            <div class="ml-timeline-title">MLP Revival</div>
            <div class="ml-timeline-desc">Backpropagation enables deep learning (95%+ MNIST)</div>
            <div class="ml-timeline-tech">Images → Flatten → Linear → ... → Classes</div>
        </div>
    </div>

    <div class="ml-timeline-item right cnn">
        <div class="ml-timeline-dot"></div>
        <div class="ml-timeline-content">
            <div class="ml-timeline-year">1998</div>
            <div class="ml-timeline-title">CNN Revolution 🎯</div>
            <div class="ml-timeline-desc">Spatial intelligence unlocks computer vision (75%+ CIFAR-10)</div>
            <div class="ml-timeline-tech">Images → Conv → Pool → ... → Classes</div>
        </div>
    </div>

    <div class="ml-timeline-item left transformer">
        <div class="ml-timeline-dot"></div>
        <div class="ml-timeline-content">
            <div class="ml-timeline-year">2017</div>
            <div class="ml-timeline-title">Transformer Era</div>
            <div class="ml-timeline-desc">Attention launches the LLM revolution</div>
            <div class="ml-timeline-tech">Tokens → Attention → FFN → Output</div>
        </div>
    </div>

    <div class="ml-timeline-item right olympics">
        <div class="ml-timeline-dot"></div>
        <div class="ml-timeline-content">
            <div class="ml-timeline-year">2018</div>
            <div class="ml-timeline-title">MLPerf Benchmarks </div>
            <div class="ml-timeline-desc">Production optimization (8-16× smaller, 12-40× faster)</div>
            <div class="ml-timeline-tech">Profile → Compress → Accelerate</div>
        </div>
    </div>
</div>
```

**[View complete milestone details](chapters/milestones)** to see full technical requirements and learning objectives.

## Why Build Instead of Use?

Understanding the difference between using a framework and building one is the difference between being limited by tools and being empowered to create them.

<div class="comparison-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; margin: 3rem 0 2.5rem 0; max-width: 1100px;">

<div style="background: #fef2f2; padding: 2rem; border-radius: 0.5rem; border-left: 4px solid #ef4444;">
<h3 style="margin: 0 0 1.25rem 0; color: #991b1b; font-size: 1.15rem;">Traditional ML Education</h3>

```python
import torch
model = torch.nn.Linear(784, 10)
output = model(input)
# When this breaks, you're stuck
```

<p style="margin: 1.25rem 0 0 0; line-height: 1.6;"><strong>Problem</strong>: OOM errors, NaN losses, slow training—you can't debug what you don't understand.</p>
</div>

<div style="background: #f0fdf4; padding: 2rem; border-radius: 0.5rem; border-left: 4px solid #22c55e;">
<h3 style="margin: 0 0 1.25rem 0; color: #166534; font-size: 1.15rem;">TinyTorch Approach</h3>

```python
from tinytorch import Linear  # YOUR code
model = Linear(784, 10)       # YOUR implementation
output = model(input)
# You know exactly how this works
```

<p style="margin: 1.25rem 0 0 0; line-height: 1.6;"><strong>Advantage</strong>: You understand memory layouts, gradient flows, and performance bottlenecks because you implemented them.</p>
</div>

</div>

**Systems Thinking**: TinyTorch emphasizes understanding how components interact—memory hierarchies, computational complexity, and optimization trade-offs—not just isolated algorithms. Every module connects mathematical theory to systems understanding.

**See [Course Philosophy](chapters/00-introduction)** for the full origin story and pedagogical approach.

## The Build → Use → Reflect Approach

Every module follows a proven learning cycle that builds deep understanding:

```{mermaid}
graph LR
    B[Build<br/>Implement from scratch] --> U[Use<br/>Real data, real problems]
    U --> R[Reflect<br/>Systems thinking questions]
    R --> B

    style B fill:#FFC107,color:#000
    style U fill:#4CAF50,color:#fff
    style R fill:#2196F3,color:#fff
```

1. **Build**: Implement each component yourself—tensors, autograd, optimizers, attention
2. **Use**: Apply your implementations to real problems—MNIST, CIFAR-10, text generation
3. **Reflect**: Answer systems thinking questions—memory usage, scaling behavior, trade-offs

This approach develops not just coding ability, but systems engineering intuition essential for production ML.

## Is This For You?

Perfect if you want to **debug ML systems**, **implement custom operations**, or **understand how PyTorch actually works**.

**Prerequisites**: Python + basic linear algebra. No prior ML experience required.

---

## 🌍 Join the Community

<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 2rem; border-radius: 1rem; margin: 2rem 0; text-align: center;">
  <p style="color: #f1f5f9; font-size: 1.25rem; margin: 0 0 0.5rem 0; font-weight: 600;">
    See learners building ML systems worldwide
  </p>
  <p style="color: #94a3b8; margin: 0 0 1.5rem 0;">
    Add yourself to the map • Share your progress • Connect with builders
  </p>
  <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
  <a href="https://tinytorch.ai/join" target="_blank" 
     style="display: inline-block; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); 
            color: white; padding: 0.75rem 2rem; border-radius: 0.5rem; 
            text-decoration: none; font-weight: 600; font-size: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
    Join the Map →
  </a>
    <a href="#" onclick="event.preventDefault(); if(window.openSubscribeModal) openSubscribeModal();" 
       style="display: inline-block; background: rgba(255,255,255,0.1); 
              border: 1px solid rgba(255,255,255,0.2);
              color: #f1f5f9; padding: 0.75rem 2rem; border-radius: 0.5rem; 
              text-decoration: none; font-weight: 600; font-size: 1rem;
              transition: all 0.2s ease;">
      ✉️ Subscribe
    </a>
  </div>
</div>

---

**Next Steps**: **[Quick Start Guide](quickstart-guide)** (15 min) • **[Course Structure](chapters/00-introduction)** • **[FAQ](faq.md)**

<div style="text-align: center; padding: 1.5rem 0; margin-top: 2rem; border-top: 1px solid #e2e8f0; color: #64748b; font-size: 0.9rem;">
  <span style="color: #f97316;">🔥</span> <strong>TinyTorch</strong> 
  <span style="margin: 0 0.75rem;">•</span> 
  <a href="https://mlsysbook.ai" style="color: #64748b; text-decoration: none;">MLSysBook</a>
  <span style="margin: 0 0.75rem;">•</span>
  <a href="https://github.com/mlsysbook/TinyTorch" style="color: #64748b; text-decoration: none;">GitHub</a>
  <span style="margin: 0 0.75rem;">•</span>
  <a href="https://tinytorch.ai/leaderboard" style="color: #64748b; text-decoration: none;">Leaderboard</a>
</div>
