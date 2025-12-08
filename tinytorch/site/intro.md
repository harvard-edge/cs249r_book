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
<style>
.demo-terminal {
  max-width: 640px;
  margin: 0 auto 1.5rem auto;
}
.demo-window {
  background: #0d1117;
  border-radius: 10px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  overflow: hidden;
}
.demo-titlebar {
  background: linear-gradient(90deg, #21262d 0%, #161b22 100%);
  padding: 10px 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  border-bottom: 1px solid #30363d;
}
.demo-dots {
  display: flex;
  gap: 6px;
}
.demo-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}
.demo-dot.red { background: #ff5f57; }
.demo-dot.yellow { background: #febc2e; }
.demo-dot.green { background: #28c840; }
.demo-titlebar-text {
  flex: 1;
  text-align: center;
  color: #8b949e;
  font-size: 0.85rem;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}
.demo-body {
  padding: 1.5rem 1.5rem;
  font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
  font-size: 0.9rem;
  line-height: 1.8;
  min-height: 280px;
}
.demo-line {
  color: #c9d1d9;
  margin-bottom: 0.25rem;
  opacity: 0;
  transform: translateY(5px);
  animation: fadeInLine 0.3s ease forwards;
}
.demo-line.visible {
  opacity: 1;
  transform: translateY(0);
}
@keyframes fadeInLine {
  to { opacity: 1; transform: translateY(0); }
}
.demo-prompt {
  color: #7ee787;
}
.demo-output {
  color: #8b949e;
  padding-left: 1rem;
}
.demo-success {
  color: #7ee787;
}
.demo-faded {
  color: #484f58;
  font-style: italic;
  margin-top: 1rem;
}
.demo-progress {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1rem;
}
.demo-step {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.4rem 0.75rem;
  background: #1e293b;
  border-radius: 20px;
  font-size: 0.8rem;
  color: #64748b;
  transition: all 0.3s;
  cursor: pointer;
  border: none;
}
.demo-step.active {
  background: #1e3a8a;
  color: #93c5fd;
}
.demo-step:hover {
  background: #334155;
  color: #94a3b8;
}
.demo-step.active:hover {
  background: #1e40af;
}
</style>

<div class="demo-terminal">
  <div class="demo-window">
    <div class="demo-titlebar">
      <div class="demo-dots">
        <span class="demo-dot red"></span>
        <span class="demo-dot yellow"></span>
        <span class="demo-dot green"></span>
      </div>
      <span class="demo-titlebar-text">TinyTorch Terminal</span>
    </div>
    <div class="demo-body" id="demo-body">
      <!-- Lines rendered by JS -->
    </div>
  </div>
  <div class="demo-progress">
    <button class="demo-step active" data-step="0" onclick="goStep(0)">Setup</button>
    <button class="demo-step" data-step="1" onclick="goStep(1)">Build</button>
    <button class="demo-step" data-step="2" onclick="goStep(2)">Milestone</button>
    <button class="demo-step" data-step="3" onclick="goStep(3)">Compete</button>
  </div>
</div>

<script>
(function() {
  const scenes = [
    { name: 'Setup', lines: [
      { type: 'cmd', text: 'git clone https://github.com/mlsysbook/TinyTorch' },
      { type: 'out', text: 'Cloning into \'TinyTorch\'...' },
      { type: 'cmd', text: 'cd TinyTorch && ./setup.sh' },
      { type: 'success', text: 'Environment ready!' },
      { type: 'faded', text: 'Demos coming soon...' }
    ]},
    { name: 'Build', lines: [
      { type: 'cmd', text: 'tito start tensor' },
      { type: 'out', text: 'Opening Module 01: Tensor...' },
      { type: 'cmd', text: 'tito test' },
      { type: 'success', text: 'All tests passed!' },
      { type: 'faded', text: 'Demos coming soon...' }
    ]},
    { name: 'Milestone', lines: [
      { type: 'cmd', text: 'tito milestone' },
      { type: 'success', text: 'Perceptron (1957) unlocked!' },
      { type: 'success', text: 'XOR Crisis (1969) solved!' },
      { type: 'success', text: 'MLP (1986) achieved!' },
      { type: 'faded', text: 'Demos coming soon...' }
    ]},
    { name: 'Compete', lines: [
      { type: 'cmd', text: 'tito olympics submit' },
      { type: 'out', text: 'Benchmarking model...' },
      { type: 'success', text: 'Score: 847/1000' },
      { type: 'success', text: 'Rank #42 on leaderboard!' },
      { type: 'faded', text: 'Demos coming soon...' }
    ]}
  ];

  let current = 0;
  let interval;

  function renderScene(idx) {
    const body = document.getElementById('demo-body');
    const scene = scenes[idx];
    body.innerHTML = '';

    scene.lines.forEach((line, i) => {
      const div = document.createElement('div');
      div.className = 'demo-line';
      if (line.type === 'cmd') {
        div.innerHTML = '<span class="demo-prompt">$ </span>' + line.text;
      } else if (line.type === 'success') {
        div.innerHTML = '<span class="demo-success">' + line.text + '</span>';
      } else if (line.type === 'faded') {
        div.innerHTML = '<span class="demo-faded">' + line.text + '</span>';
      } else {
        div.innerHTML = '<span class="demo-output">' + line.text + '</span>';
      }
      body.appendChild(div);
      setTimeout(() => div.classList.add('visible'), i * 400);
    });

    document.querySelectorAll('.demo-step').forEach((btn, i) => {
      btn.classList.toggle('active', i === idx);
    });
  }

  function nextScene() {
    current = (current + 1) % scenes.length;
    renderScene(current);
  }

  window.goStep = function(idx) {
    current = idx;
    renderScene(current);
    clearInterval(interval);
    interval = setInterval(nextScene, 4000);
  };

  renderScene(0);
  interval = setInterval(nextScene, 4000);
})();
</script>
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
<h3 style="margin: 0 0 0.75rem 0; color: #0d47a1; font-size: 1.15rem; font-weight: 600;"> Foundation (Modules 01-07)</h3>
<p style="margin: 0 0 0.75rem 0; color: #1565c0; font-size: 0.95rem; line-height: 1.6;">Build the mathematical core that makes neural networks learn.</p>
<p style="margin: 0.75rem 0 0 0; color: #0d47a1; font-size: 0.85rem; font-style: italic;">
Unlocks: Perceptron (1957) • XOR Crisis (1969) • MLP (1986)
</p>
</a>

<a href="tiers/architecture.html" class="tier-card" style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #7b1fa2; text-decoration: none; display: block; transition: transform 0.2s ease, box-shadow 0.2s ease;">
<h3 style="margin: 0 0 0.75rem 0; color: #4a148c; font-size: 1.15rem; font-weight: 600;"> Architecture (Modules 08-13)</h3>
<p style="margin: 0 0 0.75rem 0; color: #6a1b9a; font-size: 0.95rem; line-height: 1.6;">Build modern neural architectures—from computer vision to language models.</p>
<p style="margin: 0.75rem 0 0 0; color: #4a148c; font-size: 0.85rem; font-style: italic;">
Unlocks: CNN Revolution (1998) • Transformer Era (2017)
</p>
</a>

<a href="tiers/optimization.html" class="tier-card" style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #f57c00; text-decoration: none; display: block; transition: transform 0.2s ease, box-shadow 0.2s ease;">
<h3 style="margin: 0 0 0.75rem 0; color: #e65100; font-size: 1.15rem; font-weight: 600;"> Optimization (Modules 14-19)</h3>
<p style="margin: 0 0 0.75rem 0; color: #ef6c00; font-size: 0.95rem; line-height: 1.6;">Transform research prototypes into production-ready systems.</p>
<p style="margin: 0.75rem 0 0 0; color: #e65100; font-size: 0.85rem; font-style: italic;">
Unlocks: MLPerf Torch Olympics (2018) • 8-16× compression • 12-40× speedup
</p>
</a>

<a href="tiers/olympics.html" class="tier-card" style="background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%); padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #c2185b; text-decoration: none; display: block; transition: transform 0.2s ease, box-shadow 0.2s ease;">
<h3 style="margin: 0 0 0.75rem 0; color: #880e4f; font-size: 1.15rem; font-weight: 600;"> Torch Olympics (Module 20)</h3>
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

##  Join the Community

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
      ✉ Subscribe
    </a>
  </div>
</div>

---

**Next Steps**: **[Quick Start Guide](quickstart-guide)** (15 min) • **[Course Structure](chapters/00-introduction)** • **[FAQ](faq.md)**

<div style="text-align: center; padding: 1.5rem 0; margin-top: 2rem; border-top: 1px solid #e2e8f0; color: #64748b; font-size: 0.9rem;">
  <span style="color: #f97316;">\raisebox{-0.1em}{\includegraphics[height=1em]{../_static/logos/fire-emoji.png}}</span> <strong>TinyTorch</strong> 
  <span style="margin: 0 0.75rem;">•</span> 
  <a href="https://mlsysbook.ai" style="color: #64748b; text-decoration: none;">MLSysBook</a>
  <span style="margin: 0 0.75rem;">•</span>
  <a href="https://github.com/mlsysbook/TinyTorch" style="color: #64748b; text-decoration: none;">GitHub</a>
  <span style="margin: 0 0.75rem;">•</span>
  <a href="https://tinytorch.ai/leaderboard" style="color: #64748b; text-decoration: none;">Leaderboard</a>
</div>
