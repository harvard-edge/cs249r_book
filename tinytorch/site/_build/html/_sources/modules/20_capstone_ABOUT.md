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
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F20_capstone%2F20_capstone.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/20_capstone/20_capstone.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/20_capstone.mp3" type="audio/mpeg">
</audio>
```

````
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
    from tinytorch.benchmarking import Benchmark, BenchmarkResult
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

**Throughput** measures batch capacity: how many inputs can you process per second? This matters for offline batch jobs processing millions of examples. Your implementation derives throughput from latency:

```python
throughput_samples_per_sec = 1000 / avg_latency
```

If latency is 10ms per sample, throughput is 1000ms / 10ms = 100 samples/second. But this assumes processing samples one at a time. In practice, batching increases throughput significantly while adding latency. Processing a batch of 32 samples might take 50ms total, giving 640 samples/second throughput but 50ms per-request latency.

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

To appreciate why professional benchmarking matters, consider the scale of production ML systems:

- **Model serving**: A recommendation system handles 10 million requests/day. If you reduce latency from 20ms to 10ms, you save 100,000 seconds of compute daily = 1.16 days of compute per day = 42% cost reduction.
- **Training efficiency**: Training a large language model costs $1 million in GPU time. Profiling reveals 60% of time is spent in data loading. Optimizing the data pipeline saves $600,000.
- **Deployment constraints**: A mobile app's model must fit in 50MB. Quantization compresses a 200MB model to 50MB with 1% accuracy loss. The app ships; without benchmarking, you wouldn't know the trade-off was acceptable.

Systematic benchmarking with reproducible results isn't academic exercise—it's how engineers justify technical decisions and demonstrate business impact.

## Check Your Understanding

Test yourself with these systems thinking questions about benchmarking and performance measurement.

**Q1: Memory Calculation**

A model has 5 million parameters stored as FP32. After INT8 quantization, how much memory is saved?

```{admonition} Answer
:class: dropdown

FP32: 5,000,000 parameters × 4 bytes = 20,000,000 bytes = **20 MB**

INT8: 5,000,000 parameters × 1 byte = 5,000,000 bytes = **5 MB**

Savings: 20 MB - 5 MB = **15 MB** (75% reduction)

Compression ratio: 20 MB / 5 MB = **4.0x**

This is why quantization is standard in mobile deployment—models must fit in tight memory budgets.
```

**Q2: Latency Variance Analysis**

Model A: 10.0ms ± 0.3ms latency. Model B: 10.0ms ± 3.0ms latency. Both have same accuracy. Which do you deploy and why?

```{admonition} Answer
:class: dropdown

**Deploy Model A.**

Same mean latency (10.0ms) but Model A has 10x lower variance (0.3ms vs 3.0ms std).

Model A's latency range: ~9.4-10.6ms (95% confidence: ± 2 std)
Model B's latency range: ~4.0-16.0ms (95% confidence: ± 2 std)

**Why consistency matters:**
- Users prefer predictable performance over erratic speed
- High variance suggests GC pauses, cache misses, or resource contention
- Production SLAs commit to p99 latency—Model B's p99 could be 16ms vs Model A's 11ms

In production, **reliability > mean performance**. A consistently decent experience beats an unreliable fast one.
```

**Q3: Batch Size Trade-off**

Measuring latency with batch_size=32 gives 100ms total. Can you claim 100ms / 32 = 3.1ms per-sample latency?

```{admonition} Answer
:class: dropdown

**No.** This underestimates per-sample latency.

Batching amortizes fixed overhead (data transfer, kernel launch). Per-sample latency at batch=1 is higher than batch=32 divided by 32.

Example reality:
- Batch=32: 100ms total → 3.1ms per sample (amortized)
- Batch=1: 8ms total → 8ms per sample (actual)

**Why the discrepancy?**
- Fixed overhead: 10ms (data transfer, setup)
- Variable cost: 90ms / 32 = 2.8ms per sample
- At batch=1: 10ms fixed + 2.8ms variable = 12.8ms

**Always benchmark at deployment batch size.** If production serves single requests, measure with batch=1.
```

**Q4: Speedup Calculation**

Baseline: 20ms latency. Optimized: 5ms latency. What is the speedup and what does it mean?

```{admonition} Answer
:class: dropdown

Speedup = baseline_latency / optimized_latency = 20ms / 5ms = **4.0x**

**What it means:**
- Optimized model is **4 times faster**
- Processes same input in 1/4 the time
- Can handle 4x more traffic with same hardware

**Real-world impact:**
- If baseline served 100 requests/sec, optimized serves 400 requests/sec
- If baseline cost $1000/month in compute, optimized costs $250/month
- If baseline met latency SLA at 60% utilization, optimized has 85% headroom

**Note:** Speedup alone doesn't tell the full story. Check accuracy_delta and compression_ratio to understand trade-offs.
```

**Q5: Schema Validation Value**

Why does the submission schema require `accuracy` as float in [0, 1] instead of allowing any format?

```{admonition} Answer
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
```

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

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/20_capstone/20_capstone.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/20_capstone/20_capstone.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
