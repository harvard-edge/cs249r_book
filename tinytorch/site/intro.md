---
title: "Don't import torch. Build it."
og:title: "Don't import torch. Build it."
---

# Don't import torch. Build it.

```{raw} html
<p style="text-align: center; font-size: 2.5rem; margin: 1rem 0 0.5rem 0; font-weight: 700;">
Build Your Own ML Framework
</p>
<p style="text-align: center; margin: 0 0 1rem 0;">
  <span style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 1rem; padding: 0.25rem 0.75rem; font-size: 0.75rem; color: #92400e; font-weight: 600;">🚧 Preview · Classroom ready 2026</span>
</p>
```

<h2 style="background: linear-gradient(135deg, #E74C3C 0%, #E67E22 50%, #F39C12 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: center; font-size: 2.5rem; margin: 1rem 0; font-weight: 700;">
Don't import it. Build it.
</h2>

<p style="text-align: center; font-size: 1.15rem; margin: 0 auto 1.5rem auto; max-width: 750px; color: #374151;">
From tensors to systems. An educational framework for building and optimizing ML—understand how PyTorch, TensorFlow, and JAX really work.
</p>

<div style="text-align: center; margin: 0 0 1rem 0;">
  <a href="getting-started.html" style="display: inline-block; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white; padding: 0.875rem 2.5rem; border-radius: 0.5rem; text-decoration: none; font-weight: 600; font-size: 1.1rem; box-shadow: 0 4px 12px rgba(249,115,22,0.3);">
    Start Building →
  </a>
</div>
<div style="text-align: center; margin: 0 0 2.5rem 0;">
  <a href="https://github.com/harvard-edge/cs249r_book?tab=readme-ov-file#support-this-work" target="_blank" style="display: inline-flex; align-items: center; gap: 0.4rem; color: #6b7280; font-size: 0.85rem; text-decoration: none;">
    <span>⭐</span>
    <span id="star-count-hero" style="font-weight: 600;">...</span>
    <span>learners · every ⭐ helps support free ML education</span>
  </a>
</div>

```{raw} html
<style>
.approach-box {
  max-width: 720px;
  margin: 0 auto 2rem auto;
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border-radius: 12px;
  padding: 2rem;
  border: 1px solid rgba(249, 115, 22, 0.3);
  box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
.approach-title {
  color: #f97316;
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 1rem 0;
  text-align: center;
}
.approach-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
}
.approach-item {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}
.approach-icon {
  font-size: 1.25rem;
  flex-shrink: 0;
}
.approach-text {
  color: #e2e8f0;
  font-size: 0.9rem;
  line-height: 1.5;
}
.approach-text strong {
  color: #fbbf24;
}
@media (max-width: 600px) {
  .approach-grid { grid-template-columns: 1fr; }
}
</style>

<div class="approach-box">
  <div class="approach-grid">
    <div class="approach-item">
      <span class="approach-icon">🔧</span>
      <span class="approach-text"><strong>Build each piece</strong> — Tensors, autograd, attention. No magic imports.</span>
    </div>
    <div class="approach-item">
      <span class="approach-icon">📚</span>
      <span class="approach-text"><strong>Recreate history</strong> — Perceptron → CNN → Transformers → MLPerf.</span>
    </div>
    <div class="approach-item">
      <span class="approach-icon">⚡</span>
      <span class="approach-text"><strong>Understand systems</strong> — Memory, compute, optimization trade-offs.</span>
    </div>
    <div class="approach-item">
      <span class="approach-icon">🎯</span>
      <span class="approach-text"><strong>Debug anything</strong> — OOM, NaN, slow training—because you built it.</span>
    </div>
  </div>
</div>
```

## Recreate ML History

Walk through ML history by rebuilding its greatest breakthroughs with YOUR TinyTorch implementations. Click each milestone to see what you'll build and how it shaped modern AI.

```{raw} html
<div class="ml-timeline-container">
    <div class="ml-timeline-line"></div>

    <div class="ml-timeline-item left perceptron">
        <div class="ml-timeline-dot"></div>
        <div class="ml-timeline-content">
            <div class="ml-timeline-year">1958</div>
            <div class="ml-timeline-title">The Perceptron</div>
            <div class="ml-timeline-desc">The first trainable neural network</div>
            <div class="ml-timeline-tech">Input → Linear → Sigmoid → Output</div>
        </div>
    </div>

    <div class="ml-timeline-item right xor">
        <div class="ml-timeline-dot"></div>
        <div class="ml-timeline-content">
            <div class="ml-timeline-year">1969</div>
            <div class="ml-timeline-title">XOR Crisis</div>
            <div class="ml-timeline-desc">Minsky & Papert expose limits of single-layer networks</div>
            <div class="ml-timeline-tech">Input → Linear → Sigmoid → FAIL!</div>
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
            <div class="ml-timeline-year">2018–Present</div>
            <div class="ml-timeline-title">MLPerf Benchmarks</div>
            <div class="ml-timeline-desc">Production optimization (8-16× smaller, 12-40× faster)</div>
            <div class="ml-timeline-tech">Profile → Compress → Accelerate</div>
        </div>
    </div>
</div>
```

## Why Build Instead of Use?

<p style="text-align: center; font-size: 1.3rem; font-style: italic; color: #475569; margin: 0 0 1.5rem 0; max-width: 600px; margin-left: auto; margin-right: auto;">
"Building systems creates irreversible understanding."
</p>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1.5rem 0 2rem 0; max-width: 1000px;">

<div style="background: #fef2f2; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #ef4444;">
<h3 style="margin: 0 0 1rem 0; color: #991b1b; font-size: 1.05rem;">Traditional ML Education</h3>

```python
import torch
model = torch.nn.Linear(784, 10)
output = model(input)
# When this breaks, you're stuck
```

<p style="margin: 1rem 0 0 0; line-height: 1.5; font-size: 0.9rem;"><strong>Problem</strong>: You can't debug what you don't understand.</p>
</div>

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e;">
<h3 style="margin: 0 0 1rem 0; color: #166534; font-size: 1.05rem;">TinyTorch: Build → Use → Reflect</h3>

```python
# BUILD it yourself
class Linear:
    def forward(self, x):
        return x @ self.weight + self.bias

# USE it on real data
loss.backward()  # YOUR autograd
```

<p style="margin: 1rem 0 0 0; line-height: 1.5; font-size: 0.9rem;"><strong>Advantage</strong>: You can debug it because you built it.</p>
</div>

</div>

## Learning Path

Four progressive tiers take you from foundations to production systems:

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin: 1.5rem 0 2rem 0; max-width: 1100px;">

<a href="tiers/foundation.html" style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.25rem; border-radius: 0.5rem; border-left: 4px solid #1976d2; text-decoration: none; display: block;">
<h3 style="margin: 0 0 0.5rem 0; color: #0d47a1; font-size: 1rem; font-weight: 600;">Foundation (01-08)</h3>
<p style="margin: 0; color: #1565c0; font-size: 0.9rem;">Tensors, autograd, layers, training loops</p>
</a>

<a href="tiers/architecture.html" style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 1.25rem; border-radius: 0.5rem; border-left: 4px solid #7b1fa2; text-decoration: none; display: block;">
<h3 style="margin: 0 0 0.5rem 0; color: #4a148c; font-size: 1rem; font-weight: 600;">Architecture (09-13)</h3>
<p style="margin: 0; color: #6a1b9a; font-size: 0.9rem;">CNNs, attention, transformers, GPT</p>
</a>

<a href="tiers/optimization.html" style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1.25rem; border-radius: 0.5rem; border-left: 4px solid #f57c00; text-decoration: none; display: block;">
<h3 style="margin: 0 0 0.5rem 0; color: #e65100; font-size: 1rem; font-weight: 600;">Optimization (14-19)</h3>
<p style="margin: 0; color: #ef6c00; font-size: 0.9rem;">Profiling, quantization, acceleration</p>
</a>

<a href="tiers/olympics.html" style="background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%); padding: 1.25rem; border-radius: 0.5rem; border-left: 4px solid #c2185b; text-decoration: none; display: block;">
<h3 style="margin: 0 0 0.5rem 0; color: #880e4f; font-size: 1rem; font-weight: 600;">Torch Olympics (20)</h3>
<p style="margin: 0; color: #ad1457; font-size: 0.9rem;">Competition-ready capstone project</p>
</a>

</div>

**[The Big Picture](big-picture)** • **[Getting Started](getting-started)** • **[Preface](preface)**

## Is This For You?

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.5rem 0;">

<div style="background: #f8fafc; padding: 1.25rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
<p style="margin: 0 0 0.5rem 0; font-weight: 600; color: #1e293b;">🎓 Students</p>
<p style="margin: 0; font-size: 0.9rem; color: #64748b;">Taking ML courses, want to understand what's behind <code>import torch</code></p>
</div>

<div style="background: #f8fafc; padding: 1.25rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
<p style="margin: 0 0 0.5rem 0; font-weight: 600; color: #1e293b;">👩‍🏫 Instructors</p>
<p style="margin: 0; font-size: 0.9rem; color: #64748b;">Teaching ML systems with ready-made hands-on labs</p>
</div>

<div style="background: #f8fafc; padding: 1.25rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
<p style="margin: 0 0 0.5rem 0; font-weight: 600; color: #1e293b;">🚀 Self-learners</p>
<p style="margin: 0; font-size: 0.9rem; color: #64748b;">Career changers or hobbyists going deeper than tutorials</p>
</div>

</div>

**Prerequisites**: Python + basic linear algebra. No ML experience required.


##  Join the Community

<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 2rem; border-radius: 1rem; margin: 2rem 0; text-align: center;">
  <p style="color: #f1f5f9; font-size: 1.25rem; margin: 0 0 0.5rem 0; font-weight: 600;">
    See learners building ML systems worldwide
  </p>
  <p style="color: #94a3b8; margin: 0 0 0.75rem 0;">
    Add yourself to the map • Share your progress • Connect with builders
  </p>
  <p style="color: #fbbf24; margin: 0 0 1.5rem 0; font-size: 0.9rem;">
    Part of the <a href="https://github.com/harvard-edge/cs249r_book?tab=readme-ov-file#support-this-work" target="_blank" style="color: #fbbf24; text-decoration: underline;">MLSysBook</a> project — every ⭐ helps support free ML education
  </p>
  <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; align-items: center;">
    <a href="community/"
       style="display: inline-block; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
              color: white; padding: 0.75rem 2rem; border-radius: 0.5rem;
              text-decoration: none; font-weight: 600; font-size: 1rem;
              box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
      🌍 Join the Community
    </a>
    <a href="https://github.com/harvard-edge/cs249r_book" target="_blank"
       style="display: inline-flex; align-items: center; gap: 0.5rem;
              background: rgba(255,255,255,0.15);
              border: 1px solid rgba(255,255,255,0.3);
              color: #ffffff; padding: 0.75rem 1.5rem; border-radius: 0.5rem;
              text-decoration: none; font-weight: 600; font-size: 1rem;
              transition: all 0.2s ease;">
      ⭐ Star on GitHub
      <span id="star-count" style="background: rgba(249,115,22,0.3); padding: 0.2rem 0.6rem; border-radius: 1rem; font-size: 0.85rem; color: #fbbf24;">...</span>
    </a>
    <a href="https://github.com/harvard-edge/cs249r_book/discussions/1076" target="_blank"
       style="display: inline-block; background: rgba(255,255,255,0.15);
              border: 1px solid rgba(255,255,255,0.3);
              color: #ffffff; padding: 0.75rem 2rem; border-radius: 0.5rem;
              text-decoration: none; font-weight: 600; font-size: 1rem;
              transition: all 0.2s ease;">
      💬 Discuss on GitHub
    </a>
    <a href="#" onclick="event.preventDefault(); if(window.openSubscribeModal) openSubscribeModal();"
       style="display: inline-block; background: rgba(255,255,255,0.15);
              border: 1px solid rgba(255,255,255,0.3);
              color: #ffffff; padding: 0.75rem 2rem; border-radius: 0.5rem;
              text-decoration: none; font-weight: 600; font-size: 1rem;
              transition: all 0.2s ease;">
      📬 Get Updates
    </a>
  </div>
</div>

<script>
async function fetchGitHubStars() {
  const starElement = document.getElementById('star-count');
  const starElementHero = document.getElementById('star-count-hero');

  try {
    const response = await fetch('https://api.github.com/repos/harvard-edge/cs249r_book');
    const data = await response.json();
    const starCount = data.stargazers_count;
    const formattedCount = starCount.toLocaleString();

    if (starElement) starElement.textContent = formattedCount;
    if (starElementHero) starElementHero.textContent = formattedCount;
  } catch (error) {
    console.error('Failed to fetch GitHub stars:', error);
    if (starElement) starElement.textContent = '10k+';
    if (starElementHero) starElementHero.textContent = '10k+';
  }
}

document.addEventListener('DOMContentLoaded', fetchGitHubStars);
</script>


**Next Steps**: **[Quick Start](getting-started)** (15 min) • **[The Big Picture](big-picture)** • **[Community](community)**
