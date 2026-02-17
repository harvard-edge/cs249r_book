---
file_format: mystnb
kernelspec:
  name: python3
---

# Module 20: Capstone

:::{admonition} Module Info
:class: note

**OPTIMIZATION TIER** | Difficulty: ●●●● | Time: 6-8 hours | Prerequisites: All modules (01-19)

**Prerequisites: All modules** means you've built a complete ML framework. This capstone assumes:
- Complete TinyTorch framework (Modules 01-13) - **Required**
- Optimization techniques (Modules 14-18) - **Optional but recommended**
- Benchmarking methodology (Module 19) - **Required**

The core benchmarking functionality (Parts 1-4) works with just Modules 01-13 and 19. Modules 14-18 enable the advanced optimization workflow (Part 4b), which demonstrates how to integrate all TinyTorch components. If optimization modules aren't available, the system gracefully degrades to baseline benchmarking only.
:::

`````{only} html
````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/20_capstone.mp3" type="audio/mpeg">
</audio>
```

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F20_capstone%2Fcapstone.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/20_capstone/20_capstone.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #14b8a6; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
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

<div class="slide-viewer-container" id="slide-viewer-20_capstone">
  <div class="slide-header">
    <div class="slide-title">
      <span>🔥</span>
      <span>Slide Deck</span>
      <span class="slide-subtitle">· AI-generated</span>
    </div>
    <div class="slide-toolbar">
      <div class="slide-nav-group">
        <button onclick="slideNav('20_capstone', -1)" title="Previous">‹</button>
        <span class="slide-page-info"><span id="slide-num-20_capstone">1</span> / <span id="slide-count-20_capstone">-</span></span>
        <button onclick="slideNav('20_capstone', 1)" title="Next">›</button>
      </div>
      <div class="slide-zoom-group">
        <button onclick="slideZoom('20_capstone', -0.25)" title="Zoom out">−</button>
        <button onclick="slideZoom('20_capstone', 0.25)" title="Zoom in">+</button>
      </div>
    </div>
  </div>
  <div class="slide-canvas-wrapper">
    <div id="slide-loading-20_capstone" class="slide-loading">Loading slides...</div>
    <canvas id="slide-canvas-20_capstone" class="slide-canvas" style="display:none;"></canvas>
  </div>
  <div class="slide-progress-wrapper">
    <div class="slide-progress-bar" onclick="slideProgress('20_capstone', event)">
      <div class="slide-progress-fill" id="slide-progress-20_capstone" style="width: 0%;"></div>
    </div>
  </div>
  <div class="slide-footer">
    <a href="../_static/slides/20_capstone.pdf" download>⬇ Download</a>
    <a href="#" onclick="slideFullscreen('20_capstone'); return false;" class="secondary">⛶ Fullscreen</a>
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

initSlideViewer('20_capstone', '../_static/slides/20_capstone.pdf');
</script>
```
`````

## Overview

You've built an entire machine learning framework from scratch across 19 modules. You can train neural networks, implement transformers, optimize models with quantization and pruning, and measure performance with profiling tools. But there's one critical piece missing: proving your work with reproducible results.

In production ML systems, claims without measurements are worthless. This capstone module teaches you to benchmark models comprehensively, document optimizations systematically, and generate standardized submissions that mirror industry practices like MLPerf and Papers with Code. You'll learn the complete workflow from baseline measurement through optimization to final submission, using the same patterns employed by ML research labs and production engineering teams.

By the end, you'll have a professional benchmarking system that demonstrates your framework's capabilities and enables fair comparisons with others' implementations.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** comprehensive benchmarking infrastructure measuring accuracy, latency, throughput, and memory
- **Master** the three pillars of reliable benchmarking: repeatability, comparability, and completeness
- **Understand** performance measurement traps (variance, cold starts, batch effects) and how to avoid them
- **Connect** your TinyTorch implementation to production ML workflows (experiment tracking, A/B testing, regression detection)
- **Generate** schema-validated JSON submissions that enable reproducible comparisons and community sharing
```

## What You'll Build

```{mermaid}
:align: center
:caption: Benchmarking System
flowchart LR
    subgraph "Benchmarking System"
        A["BenchmarkReport<br/>measure performance"]
        B["Submission<br/>standardized format"]
        C["Validation<br/>schema compliance"]
    end

    subgraph "Workflow"
        D["Baseline Model"]
        E["Optimization"]
        F["Comparison"]
    end

    D --> A
    E --> A
    A --> B
    B --> C
    A --> F

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
    style E fill:#e2d5f1
    style F fill:#ffe4e1
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `SimpleMLP` | Baseline model for benchmarking demonstrations |
| 2 | `BenchmarkReport` | Comprehensive performance measurement with statistical rigor |
| 3 | `generate_submission()` | Standardized JSON generation with schema compliance |
| 4 | `validate_submission_schema()` | Automated validation ensuring data quality |
| 5 | Complete workflows | Baseline, optimization, comparison, submission pipeline |

**The pattern you'll enable:**
```python
# Professional ML workflow
report = BenchmarkReport(model_name="my_model")
report.benchmark_model(model, X_test, y_test, num_runs=100)

submission = generate_submission(
    baseline_report=baseline_report,
    optimized_report=optimized_report,
    techniques_applied=["quantization", "pruning"]
)
save_submission(submission, "results.json")
```

### What You're NOT Building (Yet)

To keep this capstone focused, you will **not** implement:

- Automated CI/CD pipelines (production systems run benchmarks on every commit)
- Multi-hardware comparison (benchmarking across CPU/GPU/TPU)
- Visualization dashboards (plotting accuracy vs latency trade-off curves)
- Leaderboard aggregation (combining community submissions)

**You are building the measurement and reporting foundation.** Automation and visualization come later in production MLOps.

## API Reference

This section provides a quick reference for the benchmarking classes and functions you'll build. Use this while implementing and debugging.

### BenchmarkReport Constructor

```python
BenchmarkReport(model_name: str = "model") -> BenchmarkReport
```

Creates a benchmark report instance that measures and stores model performance metrics along with system context for reproducibility.

### BenchmarkReport Properties

| Property | Type | Description |
|----------|------|-------------|
| `model_name` | `str` | Identifier for the model being benchmarked |
| `metrics` | `dict` | Performance measurements (accuracy, latency, etc.) |
| `system_info` | `dict` | Platform, Python version, NumPy version |
| `timestamp` | `str` | When benchmark was run (ISO format) |

### Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `benchmark_model` | `benchmark_model(model, X_test, y_test, num_runs=100) -> dict` | Measures accuracy, latency (mean ± std), throughput, memory |
| `generate_submission` | `generate_submission(baseline_report, optimized_report=None, ...) -> dict` | Creates standardized JSON with baseline, optimized, improvements |
| `save_submission` | `save_submission(submission, filepath="submission.json") -> str` | Writes JSON to file with validation |
| `validate_submission_schema` | `validate_submission_schema(submission) -> bool` | Validates structure and value ranges |

### Module Dependencies and Imports

This capstone integrates components from across TinyTorch:

**Core dependencies (required):**
```python
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU
from tinytorch.core.losses import CrossEntropyLoss
```

**Optimization modules (optional):**
```python
# These imports use try/except blocks for graceful degradation
try:
    from tinytorch.perf.profiling import Profiler, quick_profile
    from tinytorch.perf.compression import magnitude_prune, structured_prune
    from tinytorch.perf.benchmarking import Benchmark, BenchmarkResult
except ImportError:
    # Core benchmarking still works without optimization modules
    pass
```

The advanced optimization workflow (Part 4b) demonstrates these optional integrations, but the core benchmarking system (Parts 1-4) works with just the foundation modules (01-13) and basic benchmarking (19).

## Core Concepts

This section covers the fundamental principles of professional ML benchmarking. These concepts apply to every ML system, from research papers to production deployments.

### The Reproducibility Crisis in ML

Modern machine learning faces a credibility problem. Many published results cannot be replicated because researchers omit critical details. When a paper claims "92% accuracy with 10ms latency," can you reproduce that result? Often not, because the paper doesn't specify hardware platform, software versions, batch size, measurement methodology, or variance across runs.

Industry standards like MLPerf and Papers with Code emerged to address this crisis by requiring:
- **Standardized tasks** with fixed datasets
- **Hardware specifications** documented completely
- **Measurement protocols** defined precisely
- **Code submissions** for automated verification

Your benchmarking system implements these same principles. When you generate a submission, it captures everything needed for someone else to verify your claims or build on your work.

### The Three Pillars of Reliable Benchmarking

Professional benchmarking rests on three foundational principles: repeatability, comparability, and completeness.

**Repeatability** means running the same experiment multiple times produces the same result. This requires fixed random seeds, consistent test datasets, and measuring variance across runs. A single measurement of "10.3ms" is worthless because you don't know if that's typical or an outlier. Measuring 100 times and reporting "10.0ms ± 0.5ms" tells you the true performance and its variability.

Here's how your implementation ensures repeatability:

```python
# Measure latency with statistical rigor
latencies = []
for _ in range(num_runs):
    start = time.time()
    _ = model.forward(X_test[:1])  # Single sample inference
    latencies.append((time.time() - start) * 1000)  # Convert to ms

avg_latency = np.mean(latencies)
std_latency = np.std(latencies)
```

The loop runs inference 100 times (by default) to capture variance. The first few runs may be slower due to cold caches, and occasional runs may hit garbage collection pauses. By aggregating many measurements, you get a statistically valid estimate.

**Comparability** means different people can fairly compare results. This requires documenting the environment completely:

```python
def _get_system_info(self):
    """Collect system information for reproducibility."""
    return {
        'platform': platform.platform(),
        'python_version': sys.version.split()[0],
        'numpy_version': np.__version__
    }
```

When someone sees your submission claiming 10ms latency, they need to know if that was measured on a MacBook or a server with 32 CPU cores. Platform differences can cause 10x performance variations, making cross-platform comparisons meaningless without context.

**Completeness** means capturing all relevant metrics, not cherry-picking favorable ones. Your `benchmark_model` method measures six distinct metrics:

```python
self.metrics = {
    'parameter_count': int(param_count),
    'model_size_mb': float(model_size_mb),
    'accuracy': float(accuracy),
    'latency_ms_mean': float(avg_latency),
    'latency_ms_std': float(std_latency),
    'throughput_samples_per_sec': float(1000 / avg_latency)
}
```

Each metric answers a different question. Parameter count indicates model capacity. Model size determines deployment cost. Accuracy measures task performance. Latency affects user experience. Throughput determines batch processing capacity. Optimizations create trade-offs between these dimensions, so measuring all of them prevents hiding downsides.

### Latency vs Throughput: A Critical Distinction

Many beginners confuse latency and throughput because both relate to speed. They measure fundamentally different things.

**Latency** measures per-sample speed: how long does it take to process one input? This matters for real-time applications where users wait for results. Your implementation measures latency by timing single-sample inference:

```python
# Latency: time for ONE sample
start = time.time()
_ = model.forward(X_test[:1])  # Shape: (1, features)
latency_ms = (time.time() - start) * 1000
```

A model with 10ms latency processes one input in 10 milliseconds. If a user submits a query, they wait 10ms for a response. This directly impacts user experience.

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Latency vs Throughput: derived metrics
lt_latency_ms = 10
lt_throughput = 1000 / lt_latency_ms
glue("lt_throughput", f"{lt_throughput:.0f}")

lt_batch_size = 32
lt_batch_time_ms = 50
lt_batch_throughput = lt_batch_size * 1000 / lt_batch_time_ms
glue("lt_batch_throughput", f"{lt_batch_throughput:.0f}")
```

**Throughput** measures batch capacity: how many inputs can you process per second? This matters for offline batch jobs processing millions of examples. Your implementation derives throughput from latency:

```python
throughput_samples_per_sec = 1000 / avg_latency
```

If latency is 10ms per sample, throughput is 1000ms / 10ms = {glue:text}`lt_throughput` samples/second. But this assumes processing samples one at a time. In practice, batching increases throughput significantly while adding latency. Processing a batch of 32 samples might take 50ms total, giving {glue:text}`lt_batch_throughput` samples/second throughput but 50ms per-request latency.

The trade-off: **Batching increases throughput but hurts latency.** A production API serving individual user requests optimizes for latency. A batch processing pipeline optimizes for throughput.

### Statistical Rigor: Why Variance Matters

Single measurements lie. Variance tells the truth about performance consistency.

Consider two models, both with mean latency of 10.0ms. Model A has standard deviation of 0.5ms. Model B has standard deviation of 4.2ms. Which would you deploy?

Model A's predictable performance (9.5-10.5ms range) provides consistent user experience. Model B's erratic performance (sometimes 6ms, sometimes 15ms) creates frustration. Users prefer reliable slowness over unpredictable speed.

Your implementation captures this variance:

```python
latencies = []
for _ in range(num_runs):
    start = time.time()
    _ = model.forward(X_test[:1])
    latencies.append((time.time() - start) * 1000)

avg_latency = np.mean(latencies)
std_latency = np.std(latencies)  # Captures variance
```

Running 100 iterations isn't just for accuracy of the mean. It also characterizes the distribution. High standard deviation indicates performance varies significantly run-to-run, perhaps due to garbage collection pauses, cache effects, or OS scheduling.

In production systems, engineers track percentiles (p50, p90, p99) to understand tail latency. The p99 latency tells you "99% of requests complete within this time," which matters more for user experience than mean latency. One user experiencing a 100ms delay (because they hit p99) has a worse experience than if all users consistently saw 20ms.

### The Optimization Trade-off Triangle

Every optimization involves trade-offs between three competing objectives: speed (latency), size (memory), and accuracy. You can optimize for any two, but achieving all three simultaneously is impossible with current techniques.

**Fast + Small** means aggressive optimization. Quantizing to INT8 reduces model size 4x and speeds up inference 2x, but typically costs 1-2% accuracy. Pruning 50% of weights halves memory and adds another speedup, but may lose another 1-2% accuracy. You've traded accuracy for efficiency.

**Fast + Accurate** means careful optimization. You might quantize only certain layers, or use INT16 instead of INT8. You preserve accuracy but achieve less compression. The model is faster but not dramatically smaller.

**Small + Accurate** means conservative techniques. Knowledge distillation transfers accuracy from a large teacher to a small student. The student is smaller and maintains accuracy, but may be slower than aggressive quantization because it still operates in FP32.

Your submission captures these trade-offs automatically:

```python
submission['improvements'] = {
    'speedup': float(baseline_latency / optimized_latency),
    'compression_ratio': float(baseline_size / optimized_size),
    'accuracy_delta': float(
        optimized_report.metrics['accuracy'] - baseline_report.metrics['accuracy']
    )
}
```

A speedup of 2.3x with compression of 4.1x but accuracy delta of -0.01 (-1%) shows you chose the "fast + small" corner of the triangle. A speedup of 1.2x with compression of 1.5x but accuracy delta of 0.00 shows you chose "accurate + moderately fast."

### Schema Validation: Enabling Automation

Your submission format uses a structured JSON schema that enforces completeness and type safety. This isn't bureaucracy—it enables powerful automation.

Without schema validation, submissions become inconsistent. One person reports accuracy as a percentage string ("92%"), another as a float (0.92), another as an integer (92). Aggregating these results requires manual cleaning. With schema validation, every submission uses the same format:

```python
# Schema-enforced format
'accuracy': float(accuracy)  # Always 0.0-1.0 float

# Validation catches errors
assert 0 <= metrics['accuracy'] <= 1, "Accuracy must be in [0, 1]"
```

This enables automated processing:

```python
# Aggregate community results automatically
all_submissions = [load_json(f) for f in submission_files]
avg_accuracy = np.mean([s['baseline']['metrics']['accuracy']
                        for s in all_submissions])

# Build leaderboards
sorted_by_speedup = sorted(all_submissions,
                           key=lambda s: s['improvements']['speedup'],
                           reverse=True)

# Detect regressions in CI/CD
if new_latency > baseline_latency * 1.1:
    raise Exception("Performance regression: 10% slower!")
```

The schema also enables forward compatibility. When you add new optional fields, old submissions remain valid. When you require new fields, the version number increments, and validation enforces the migration.

### Performance Measurement Traps

Real-world benchmarking is full of subtle traps that invalidate measurements. Understanding these pitfalls is crucial for accurate results.

**Trap 1: Measuring the Wrong Thing.** If you time model creation instead of just inference, you're measuring initialization overhead, not runtime performance. If you include data loading in the timing loop, you're measuring I/O speed, not model speed. The fix is isolating exactly what you want to measure:

```python
# Prepare data BEFORE timing
X = create_test_input()

# Time ONLY the operation you care about
start = time.time()
output = model.forward(X)  # Only this is timed
latency = time.time() - start

# Process output AFTER timing
predictions = postprocess(output)
```

**Trap 2: Ignoring System Noise.** Operating systems multitask. Your benchmark might get interrupted by background processes, garbage collection, or CPU thermal throttling. Single measurements capture noise. Multiple measurements average it out. Your implementation runs 100 iterations by default to handle this.

**Trap 3: Cold Start Effects.** The first inference is often slower because caches are cold and JIT compilers haven't optimized yet. Production benchmarks typically discard the first N runs as "warm-up." Your implementation includes warm-up inherently by averaging all runs—the few slow cold starts get averaged with many fast warm runs.

**Trap 4: Batch Size Confusion.** Measuring latency on batch_size=32 then dividing by 32 doesn't give per-sample latency. Batching amortizes overhead, so batch latency / batch_size underestimates per-sample latency. Always measure with the same batch size as production deployment.

### System Integration: The Complete ML Lifecycle

This capstone represents the final stage of the ML systems lifecycle, but it's also the beginning of the next iteration. Production ML systems operate in a never-ending loop:

1. **Research & Development** - Build models (Modules 01-13)
2. **Baseline Measurement** - Benchmark unoptimized performance (Module 19)
3. **Optimization** - Apply techniques like quantization and pruning (Modules 14-18)
4. **Validation** - Benchmark optimized version (Module 19)
5. **Decision** - Keep optimization if improvements outweigh costs (Module 20)
6. **Deployment** - Serve model in production
7. **Monitoring** - Track performance over time, detect regressions
8. **Iteration** - When performance degrades or requirements change, loop back to step 3

Your submission captures a snapshot of this cycle. The baseline metrics document performance before optimization. The optimized metrics show results after applying techniques. The improvements section quantifies the delta. The techniques_applied list enables reproducibility.

In production, engineers maintain this documentation across hundreds of experiments. When a deployment's latency increases from 10ms to 30ms three months later, they consult the original benchmark to understand what changed. Without system_info and reproducible measurements, debugging becomes guesswork.

## Production Context

### Your Implementation vs. Industry Standards

Your TinyTorch benchmarking system implements the same principles used by production ML frameworks and research competitions, just at educational scale.

| Feature | Your Implementation | Production Systems |
|---------|---------------------|-------------------|
| **Metrics** | 6 core metrics (accuracy, latency, etc.) | 20+ metrics including p99 latency, memory bandwidth |
| **Runs** | 100 iterations for variance | 1000+ runs, discard outliers |
| **Validation** | Python assertions | JSON Schema, automated CI checks |
| **Format** | Simple JSON | Protobuf, versioned schemas |
| **Scale** | Single model benchmarks | Automated pipelines tracking 1000s of experiments |

### Code Comparison

The following comparison shows how your educational implementation translates to production tools.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.olympics import BenchmarkReport, generate_submission

# Benchmark baseline
baseline_report = BenchmarkReport(model_name="my_model")
baseline_report.benchmark_model(model, X_test, y_test, num_runs=100)

# Benchmark optimized
optimized_report = BenchmarkReport(model_name="optimized_model")
optimized_report.benchmark_model(opt_model, X_test, y_test, num_runs=100)

# Generate submission
submission = generate_submission(
    baseline_report=baseline_report,
    optimized_report=optimized_report,
    techniques_applied=["quantization", "pruning"]
)

save_submission(submission, "results.json")
```
````

````{tab-item} Production MLflow
```python
import mlflow

# Track baseline experiment
with mlflow.start_run(run_name="baseline"):
    mlflow.log_params({"model": "my_model"})
    metrics = benchmark_model(model, X_test, y_test)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("model.pkl")

# Track optimized experiment
with mlflow.start_run(run_name="optimized"):
    mlflow.log_params({"model": "optimized_model",
                       "techniques": ["quantization", "pruning"]})
    metrics = benchmark_model(opt_model, X_test, y_test)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("optimized_model.pkl")

# Compare experiments in MLflow UI
```
````
`````

Let's walk through the comparison line by line:

- **Line 1 (Import)**: TinyTorch uses a simple module import. MLflow provides enterprise-grade experiment tracking with databases and web UIs.
- **Line 4 (Benchmark baseline)**: TinyTorch's `BenchmarkReport` mirrors MLflow's experiment runs. Both capture metrics and system context.
- **Line 8 (Benchmark optimized)**: Same API in both—create report, benchmark model. This consistency makes transitioning to production tools natural.
- **Line 12 (Generate submission)**: TinyTorch generates JSON. MLflow logs to a database that supports querying, visualization, and comparison.
- **Line 18 (Save)**: TinyTorch saves to file. MLflow persists to SQL database with version control and artifact storage.

```{tip} What's Identical

The workflow pattern: baseline → optimize → benchmark → compare → decide. Whether you use TinyTorch or MLflow, this cycle is fundamental to production ML. The tools scale, but the methodology remains the same.
```

### Why Benchmarking Matters at Scale

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Model serving cost calculation
scale_requests_per_day = 10_000_000
scale_latency_before_ms = 20
scale_latency_after_ms = 10
scale_saved_ms = scale_latency_before_ms - scale_latency_after_ms
scale_seconds_saved = scale_requests_per_day * scale_saved_ms / 1000
scale_days_saved = scale_seconds_saved / 86400
scale_cost_reduction_pct = (scale_saved_ms / scale_latency_before_ms) * 100

glue("scale_seconds_saved", f"{scale_seconds_saved:,.0f}")
glue("scale_days_saved", f"{scale_days_saved:.2f}")
glue("scale_cost_reduction", f"{scale_cost_reduction_pct:.0f}%")

# Training pipeline savings
scale_training_cost = 1_000_000
scale_data_loading_pct = 60
scale_pipeline_savings = scale_training_cost * scale_data_loading_pct / 100

glue("scale_pipeline_savings", f"${scale_pipeline_savings:,.0f}")
```

To appreciate why professional benchmarking matters, consider the scale of production ML systems:

- **Model serving**: A recommendation system handles 10 million requests/day. If you reduce latency from 20ms to 10ms, you save {glue:text}`scale_seconds_saved` seconds of compute daily = {glue:text}`scale_days_saved` days of compute per day = {glue:text}`scale_cost_reduction` cost reduction.
- **Training efficiency**: Training a large language model costs $1 million in GPU time. Profiling reveals 60% of time is spent in data loading. Optimizing the data pipeline saves {glue:text}`scale_pipeline_savings`.
- **Deployment constraints**: A mobile app's model must fit in 50MB. Quantization compresses a 200MB model to 50MB with 1% accuracy loss. The app ships; without benchmarking, you wouldn't know the trade-off was acceptable.

Systematic benchmarking with reproducible results isn't academic exercise—it's how engineers justify technical decisions and demonstrate business impact.

## Check Your Understanding

Test yourself with these systems thinking questions about benchmarking and performance measurement.

**Q1: Memory Calculation**

A model has 5 million parameters stored as FP32. After INT8 quantization, how much memory is saved?

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q1: Memory calculation (FP32 vs INT8 quantization)
# Using binary units: 1 MB = 1024^2 = 1,048,576 bytes
q1_params = 5_000_000
q1_fp32_bytes_per_param = 4
q1_int8_bytes_per_param = 1
q1_bytes_per_mb = 1024 ** 2

q1_fp32_bytes = q1_params * q1_fp32_bytes_per_param
q1_int8_bytes = q1_params * q1_int8_bytes_per_param
q1_fp32_mb = q1_fp32_bytes / q1_bytes_per_mb
q1_int8_mb = q1_int8_bytes / q1_bytes_per_mb
q1_savings_mb = q1_fp32_mb - q1_int8_mb
q1_reduction_pct = q1_savings_mb / q1_fp32_mb * 100
q1_compression = q1_fp32_mb / q1_int8_mb

glue("q1_fp32_bytes", f"{q1_fp32_bytes:,}")
glue("q1_fp32_mb", f"{q1_fp32_mb:.2f}")
glue("q1_int8_bytes", f"{q1_int8_bytes:,}")
glue("q1_int8_mb", f"{q1_int8_mb:.2f}")
glue("q1_savings_mb", f"{q1_savings_mb:.2f}")
glue("q1_reduction_pct", f"{q1_reduction_pct:.0f}%")
glue("q1_compression", f"{q1_compression:.1f}x")
```

```{admonition} Answer
:class: dropdown

FP32: 5,000,000 parameters x 4 bytes = {glue:text}`q1_fp32_bytes` bytes = **{glue:text}`q1_fp32_mb` MB**

INT8: 5,000,000 parameters x 1 byte = {glue:text}`q1_int8_bytes` bytes = **{glue:text}`q1_int8_mb` MB**

Savings: {glue:text}`q1_fp32_mb` MB - {glue:text}`q1_int8_mb` MB = **{glue:text}`q1_savings_mb` MB** ({glue:text}`q1_reduction_pct` reduction)

Compression ratio: {glue:text}`q1_fp32_mb` MB / {glue:text}`q1_int8_mb` MB = **{glue:text}`q1_compression`**

This is why quantization is standard in mobile deployment—models must fit in tight memory budgets.
```

**Q2: Latency Variance Analysis**

Model A: 10.0ms +/- 0.3ms latency. Model B: 10.0ms +/- 3.0ms latency. Both have same accuracy. Which do you deploy and why?

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q2: Latency variance analysis (95% confidence = +/- 2 std)
q2_mean = 10.0
q2_std_a = 0.3
q2_std_b = 3.0
q2_variance_ratio = q2_std_b / q2_std_a
q2_a_lo = q2_mean - 2 * q2_std_a
q2_a_hi = q2_mean + 2 * q2_std_a
q2_b_lo = q2_mean - 2 * q2_std_b
q2_b_hi = q2_mean + 2 * q2_std_b

glue("q2_variance_ratio", f"{q2_variance_ratio:.0f}x")
glue("q2_a_lo", f"{q2_a_lo:.1f}")
glue("q2_a_hi", f"{q2_a_hi:.1f}")
glue("q2_b_lo", f"{q2_b_lo:.1f}")
glue("q2_b_hi", f"{q2_b_hi:.1f}")
```

```{admonition} Answer
:class: dropdown

**Deploy Model A.**

Same mean latency (10.0ms) but Model A has {glue:text}`q2_variance_ratio` lower variance (0.3ms vs 3.0ms std).

Model A's latency range: ~{glue:text}`q2_a_lo`-{glue:text}`q2_a_hi`ms (95% confidence: +/- 2 std)
Model B's latency range: ~{glue:text}`q2_b_lo`-{glue:text}`q2_b_hi`ms (95% confidence: +/- 2 std)

**Why consistency matters:**
- Users prefer predictable performance over erratic speed
- High variance suggests GC pauses, cache misses, or resource contention
- Production SLAs commit to p99 latency—Model B's p99 could be 16ms vs Model A's 11ms

In production, **reliability > mean performance**. A consistently decent experience beats an unreliable fast one.
```

**Q3: Batch Size Trade-off**

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q3: Batch size trade-off — why amortized != actual per-sample latency
# Given: batch=32 takes 100ms total, batch=1 takes 8ms actual
# Solve system of equations:
#   batch_total = fixed + batch_size * variable_per_sample
#   batch1_total = fixed + 1 * variable_per_sample
# => fixed = (batch_total - batch_size * batch1) / (1 - batch_size)
q3_batch_size = 32
q3_batch_total_ms = 100
q3_batch1_actual_ms = 8.0

q3_amortized = q3_batch_total_ms / q3_batch_size
q3_fixed = (q3_batch_total_ms - q3_batch_size * q3_batch1_actual_ms) / (1 - q3_batch_size)
q3_var_total = q3_batch_total_ms - q3_fixed
q3_var_per_sample = q3_var_total / q3_batch_size
q3_batch1_check = q3_fixed + q3_var_per_sample

glue("q3_amortized", f"{q3_amortized:.1f}")
glue("q3_fixed", f"{q3_fixed:.1f}")
glue("q3_var_total", f"{q3_var_total:.1f}")
glue("q3_var_per_sample", f"{q3_var_per_sample:.1f}")
glue("q3_batch1_check", f"{q3_batch1_check:.1f}")
```

Measuring latency with batch_size=32 gives 100ms total. Can you claim 100ms / 32 = {glue:text}`q3_amortized`ms per-sample latency?

```{admonition} Answer
:class: dropdown

**No.** This underestimates per-sample latency.

Batching amortizes fixed overhead (data transfer, kernel launch). Per-sample latency at batch=1 is higher than batch=32 divided by 32.

Example reality:
- Batch=32: 100ms total → {glue:text}`q3_amortized`ms per sample (amortized)
- Batch=1: 8ms total → 8ms per sample (actual)

**Why the discrepancy?**
- Fixed overhead: {glue:text}`q3_fixed`ms (data transfer, setup)
- Variable cost: {glue:text}`q3_var_total`ms / 32 = {glue:text}`q3_var_per_sample`ms per sample
- At batch=1: {glue:text}`q3_fixed`ms fixed + {glue:text}`q3_var_per_sample`ms variable = {glue:text}`q3_batch1_check`ms

**Always benchmark at deployment batch size.** If production serves single requests, measure with batch=1.
```

**Q4: Speedup Calculation**

```{code-cell} python3
:tags: [remove-input, remove-output]
from myst_nb import glue

# Q4: Speedup and real-world impact
q4_baseline_ms = 20
q4_optimized_ms = 5
q4_speedup = q4_baseline_ms / q4_optimized_ms

q4_baseline_rps = 100  # requests/sec (given scenario)
q4_optimized_rps = q4_baseline_rps * q4_speedup
q4_baseline_cost = 1000  # $/month (given scenario)
q4_optimized_cost = q4_baseline_cost / q4_speedup

q4_baseline_util = 60  # % utilization (given scenario)
q4_optimized_util = q4_baseline_util / q4_speedup
q4_headroom = 100 - q4_optimized_util

glue("q4_speedup", f"{q4_speedup:.1f}x")
glue("q4_times_faster", f"{q4_speedup:.0f}")
glue("q4_optimized_rps", f"{q4_optimized_rps:.0f}")
glue("q4_optimized_cost", f"${q4_optimized_cost:.0f}")
glue("q4_headroom", f"{q4_headroom:.0f}%")
```

Baseline: 20ms latency. Optimized: 5ms latency. What is the speedup and what does it mean?

```{admonition} Answer
:class: dropdown

Speedup = baseline_latency / optimized_latency = 20ms / 5ms = **{glue:text}`q4_speedup`**

**What it means:**
- Optimized model is **{glue:text}`q4_times_faster` times faster**
- Processes same input in 1/{glue:text}`q4_times_faster` the time
- Can handle {glue:text}`q4_times_faster`x more traffic with same hardware

**Real-world impact:**
- If baseline served 100 requests/sec, optimized serves {glue:text}`q4_optimized_rps` requests/sec
- If baseline cost $1000/month in compute, optimized costs {glue:text}`q4_optimized_cost`/month
- If baseline met latency SLA at 60% utilization, optimized has {glue:text}`q4_headroom` headroom

**Note:** Speedup alone doesn't tell the full story. Check accuracy_delta and compression_ratio to understand trade-offs.
```

**Q5: Schema Validation Value**

Why does the submission schema require `accuracy` as float in [0, 1] instead of allowing any format?

````{admonition} Answer
:class: dropdown

**Type safety enables automation.**

Without schema:
```python
# Different submissions, different formats (breaks aggregation)
{"accuracy": "92%"}      # String
{"accuracy": 92}         # Integer (out of 100)
{"accuracy": 0.92}       # Float
{"accuracy": "good"}     # Non-numeric
```

Aggregating these requires manual parsing and error handling.

With schema:
```python
# All submissions use same format (aggregation works)
{"accuracy": 0.92}  # Always float in [0.0, 1.0]
```

**Benefits:**
1. **Automated validation** - Reject invalid submissions immediately
2. **Aggregation** - `np.mean([s['accuracy'] for s in submissions])` just works
3. **Comparison** - Sort by accuracy without parsing different formats
4. **APIs** - Other tools can consume submissions without custom parsers

**Real example:** Papers with Code leaderboards require strict schemas. Thousands of submissions from different teams aggregate automatically because everyone follows the same format.
````

## Further Reading

For students who want to understand the academic foundations and industry standards for ML benchmarking:

### Seminal Papers

- **MLPerf: An Industry Standard Benchmark Suite for Machine Learning Performance** - Mattson et al. (2020). Defines standardized ML benchmarks for hardware comparison. The gold standard for fair performance comparisons. [arXiv:1910.01500](https://arxiv.org/abs/1910.01500)

- **A Step Toward Quantifying Independently Reproducible Machine Learning Research** - Pineau et al. (2021). Analyzes reproducibility crisis in ML and proposes requirements for verifiable claims. Introduces reproducibility checklist adopted by NeurIPS. [arXiv:2104.05563](https://arxiv.org/abs/2104.05563)

- **Hidden Technical Debt in Machine Learning Systems** - Sculley et al. (2015). Identifies systems challenges in production ML, including monitoring, versioning, and reproducibility. Required reading for ML systems engineers. [NeurIPS 2015](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)

### Additional Resources

- **MLflow Documentation**: [https://mlflow.org/](https://mlflow.org/) - Production experiment tracking system implementing patterns from this module
- **Papers with Code**: [https://paperswithcode.com/](https://paperswithcode.com/) - See how research papers submit benchmarks with reproducible code
- **Weights & Biases Best Practices**: [https://wandb.ai/site/experiment-tracking](https://wandb.ai/site/experiment-tracking) - Industry standard for ML experiment management

## What's Next

```{seealso} Congratulations: You've Completed TinyTorch!

You've built a complete ML framework from scratch—from basic tensors to production-ready benchmarking. You understand how PyTorch works under the hood, how optimizations affect performance, and how to measure and document results professionally. These skills transfer directly to production ML engineering.
```

**Next Steps - Applying Your Knowledge:**

| Direction | What To Build | Skills Applied |
|-----------|--------------|----------------|
| **Advanced Optimizations** | Benchmark milestone models (MNIST CNN, Transformer) with Modules 14-18 techniques | Apply learned optimizations to real models |
| **Production Systems** | Integrate MLflow or Weights & Biases into your projects | Scale benchmarking to team workflows |
| **Research Contributions** | Submit to Papers with Code using your schema validation patterns | Share reproducible results with community |
| **MLOps Automation** | Build CI/CD pipelines that run benchmarks on every commit | Detect performance regressions automatically |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/20_capstone/capstone.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/20_capstone/20_capstone.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
