# Module 19: Benchmarking

:::{admonition} Module Info
:class: note

**OPTIMIZATION TIER** | Difficulty: ●●●○ | Time: 5-7 hours | Prerequisites: 01-18

This module assumes familiarity with the complete TinyTorch stack (Modules 01-13), profiling (Module 14), and optimization techniques (Modules 15-18). You should understand how to build, profile, and optimize models before tackling systematic benchmarking and statistical comparison of optimizations.
:::

`````{only} html
````{grid} 1 2 3 3
:gutter: 3

```{grid-item-card} 🚀 Launch Binder

Run interactively in your browser.

<a href="https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F19_benchmarking%2F19_benchmarking.ipynb" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #f97316; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">Open in Binder →</a>
```

```{grid-item-card} 📄 View Source

Browse the source code on GitHub.

<a href="https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/19_benchmarking/19_benchmarking.py" target="_blank" style="display: flex; align-items: center; justify-content: center; width: 100%; height: 54px; margin-top: auto; background: #6b7280; color: white; text-align: center; text-decoration: none; border-radius: 27px; font-size: 14px; box-sizing: border-box;">View on GitHub →</a>
```

```{grid-item-card} 🎧 Audio Overview

Listen to an AI-generated overview.

<audio controls style="width: 100%; height: 54px; margin-top: auto;">
  <source src="https://github.com/harvard-edge/cs249r_book/releases/download/tinytorch-audio-v0.1.1/19_benchmarking.mp3" type="audio/mpeg">
</audio>
```

````
`````

## Overview

Benchmarking transforms performance optimization from guesswork into engineering discipline. You have learned individual optimization techniques in Modules 14-18, but how do you know which optimizations actually work? How do you compare a quantized model against a pruned one? How do you ensure your measurements are statistically valid rather than random noise?

In this module, you will build the infrastructure that powers TorchPerf Olympics, the capstone competition framework. You will implement professional benchmarking tools that measure latency, accuracy, and memory with statistical rigor, generate Pareto frontiers showing optimization trade-offs, and produce reproducible results that guide real engineering decisions.

By the end, you will have the evaluation framework needed to systematically combine optimizations and compete in the capstone challenge.

## Learning Objectives

```{tip} By completing this module, you will:

- **Implement** professional benchmarking infrastructure with statistical analysis including confidence intervals and variance control
- **Master** multi-objective optimization trade-offs between accuracy, latency, and memory through Pareto frontier analysis
- **Understand** measurement uncertainty, warmup protocols, and reproducibility requirements for fair model comparison
- **Connect** optimization techniques from Modules 14-18 into systematic evaluation workflows for the TorchPerf Olympics capstone
```

## What You'll Build

```{mermaid}
:align: center
:caption: Benchmarking Infrastructure
flowchart TB
    subgraph "Benchmarking Infrastructure"
        A["BenchmarkResult<br/>Statistical container"]
        B["Benchmark<br/>Single metric runner"]
        C["BenchmarkSuite<br/>Multi-metric evaluation"]
        D["TorchPerf Olympics<br/>Competition framework"]
    end

    E["Models to Compare"] --> B
    F["Test Datasets"] --> B
    B --> A
    A --> C
    C --> D

    style A fill:#e1f5ff
    style B fill:#fff3cd
    style C fill:#f8d7da
    style D fill:#d4edda
```

**Implementation roadmap:**

| Part | What You'll Implement | Key Concept |
|------|----------------------|-------------|
| 1 | `BenchmarkResult` dataclass | Statistical analysis with confidence intervals |
| 2 | `precise_timer()` context manager | High-precision timing for microsecond measurements |
| 3 | `Benchmark.run_latency_benchmark()` | Warmup protocols and latency measurement |
| 4 | `Benchmark.run_accuracy_benchmark()` | Model quality evaluation across datasets |
| 5 | `Benchmark.run_memory_benchmark()` | Peak memory tracking during inference |
| 6 | `BenchmarkSuite` for multi-metric analysis | Pareto frontier generation and trade-off visualization |

**The pattern you'll enable:**

```python
# Compare baseline vs optimized model with statistical rigor
benchmark = Benchmark([baseline_model, optimized_model])
latency_results = benchmark.run_latency_benchmark()
# Output: baseline: 12.3ms ± 0.8ms, optimized: 4.1ms ± 0.3ms (67% reduction, p < 0.01)
```

### What You're NOT Building (Yet)

To keep this module focused, you will **not** implement:

- Hardware-specific benchmarks (GPU profiling requires CUDA, covered in production frameworks)
- Energy measurement (requires specialized hardware like power meters)
- Distributed benchmarking (multi-node coordination is beyond scope)
- Automated hyperparameter tuning for optimization (that is Module 20: Capstone)

**You are building the statistical foundation for fair comparison.** Advanced deployment scenarios come in production systems.

## API Reference

This section documents the benchmarking API you will implement. Use it as your reference while building the statistical analysis and measurement infrastructure.

### BenchmarkResult Dataclass

```python
BenchmarkResult(metric_name: str, values: List[float], metadata: Dict[str, Any] = {})
```

Statistical container for benchmark measurements with automatic computation of mean, standard deviation, median, and 95% confidence intervals.

**Properties computed in `__post_init__`:**

| Property | Type | Description |
|----------|------|-------------|
| `mean` | `float` | Average of all measurements |
| `std` | `float` | Standard deviation (0 if single measurement) |
| `median` | `float` | Middle value, less sensitive to outliers |
| `min_val` | `float` | Minimum observed value |
| `max_val` | `float` | Maximum observed value |
| `count` | `int` | Number of measurements |
| `ci_lower` | `float` | Lower bound of 95% confidence interval |
| `ci_upper` | `float` | Upper bound of 95% confidence interval |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_dict` | `to_dict() -> Dict[str, Any]` | Serialize to dictionary for JSON export |
| `__str__` | Returns formatted summary | `"metric: mean ± std (n=count)"` |

### Timing Context Manager

```python
with precise_timer() as timer:
    # Your code to measure
    ...
# Access elapsed time
elapsed_seconds = timer.elapsed
```

High-precision timing context manager using `time.perf_counter()` for monotonic, nanosecond-resolution measurements.

### Benchmark Class

```python
Benchmark(models: List[Any], datasets: List[Any],
          warmup_runs: int = 5, measurement_runs: int = 10)
```

Core benchmarking engine for single-metric evaluation across multiple models.

**Parameters:**
- `models`: List of models to benchmark (supports any object with forward/predict/__call__)
- `datasets`: List of datasets for accuracy benchmarking (required)
- `warmup_runs`: Number of warmup iterations before measurement (default: 5)
- `measurement_runs`: Number of measurement iterations for statistical analysis (default: 10)

**Core Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `run_latency_benchmark` | `run_latency_benchmark(input_shape=(1,28,28)) -> Dict[str, BenchmarkResult]` | Measure inference time per model |
| `run_accuracy_benchmark` | `run_accuracy_benchmark() -> Dict[str, BenchmarkResult]` | Measure prediction accuracy on datasets |
| `run_memory_benchmark` | `run_memory_benchmark(input_shape=(1,28,28)) -> Dict[str, BenchmarkResult]` | Track peak memory usage during inference |
| `compare_models` | `compare_models(metric: str = "latency") -> List[Dict]` | Compare models across a specific metric |

### BenchmarkSuite Class

```python
BenchmarkSuite(models: List[Any], datasets: List[Any], output_dir: str = "benchmark_results")
```

Comprehensive multi-metric evaluation suite for generating Pareto frontiers and optimization trade-off analysis.

| Method | Signature | Description |
|--------|-----------|-------------|
| `run_full_benchmark` | `run_full_benchmark() -> Dict[str, Dict[str, BenchmarkResult]]` | Run all benchmark types and aggregate results |
| `plot_pareto_frontier` | `plot_pareto_frontier(x_metric='latency', y_metric='accuracy')` | Plot Pareto frontier for two competing objectives |
| `plot_results` | `plot_results(save_plots=True)` | Generate visualization plots for benchmark results |
| `generate_report` | `generate_report() -> str` | Generate comprehensive benchmark report |

## Core Concepts

This section covers the fundamental principles that make benchmarking scientifically valid. Understanding these concepts separates professional performance engineering from naive timing measurements.

### Benchmarking Methodology

Single measurements are meaningless in performance engineering. Consider timing a model inference once: you might get 1.2ms. Run it again and you might get 3.1ms because a background process started. Which number represents the model's true performance? Neither.

Professional benchmarking treats measurements as samples from a noisy distribution. The true performance is the distribution's mean, and your job is to estimate it with statistical confidence. This requires understanding measurement variance and controlling for confounding factors.

The methodology follows a structured protocol:

**Warmup Phase**: Modern systems adapt to workloads. JIT compilers optimize hot code paths after several executions. CPU frequency scales up under sustained load. Caches fill with frequently accessed data. Without warmup, your first measurements capture cold-start behavior, not steady-state performance.

```python
# Warmup protocol in action
for _ in range(warmup_runs):
    _ = model(input_data)  # Run but discard measurements
```

**Measurement Phase**: After warmup, run the operation multiple times and collect timing data. The Central Limit Theorem tells us that with enough samples, the sample mean approaches the true mean, and we can compute confidence intervals.

Here is how the Benchmark class implements the complete protocol:

```python
def run_latency_benchmark(self, input_shape: Tuple[int, ...] = (1, 28, 28)) -> Dict[str, BenchmarkResult]:
    """Benchmark model inference latency using Profiler."""
    results = {}

    for i, model in enumerate(self.models):
        model_name = getattr(model, 'name', f'model_{i}')
        latencies = []

        # Create input tensor for this benchmark
        input_data = Tensor(np.random.randn(*input_shape).astype(np.float32))

        # Warmup runs (discard results)
        for _ in range(self.warmup_runs):
            if hasattr(model, 'forward'):
                _ = model.forward(input_data)
            elif callable(model):
                _ = model(input_data)

        # Measurement runs (collect statistics)
        for _ in range(self.measurement_runs):
            with precise_timer() as timer:
                if hasattr(model, 'forward'):
                    _ = model.forward(input_data)
                elif callable(model):
                    _ = model(input_data)
            latencies.append(timer.elapsed)

        results[model_name] = BenchmarkResult(
            metric_name=f"{model_name}_latency",
            values=latencies,
            metadata={'input_shape': input_shape, **self.system_info}
        )

    return results
```

Notice the clear separation between warmup (discarded) and measurement (collected) phases. This ensures measurements reflect steady-state performance.

**Statistical Analysis**: Raw measurements get transformed into confidence intervals that quantify uncertainty. The 95% confidence interval tells you: "If we ran this benchmark 100 times, 95 of those runs would produce a mean within this range."

### Metrics Selection

Different optimization goals require different metrics. Choosing the wrong metric leads to optimizing for the wrong objective.

**Latency (Time per Inference)**: Measures how fast a model processes a single input. Critical for real-time systems like autonomous vehicles where a prediction must complete in 30ms before the next camera frame arrives. Measured in milliseconds or microseconds.

**Throughput (Inputs per Second)**: Measures total processing capacity. Critical for batch processing systems like translating millions of documents. Higher throughput means more efficient hardware utilization. Measured in samples per second or frames per second.

**Accuracy (Prediction Quality)**: Measures how often the model makes correct predictions. The fundamental quality metric. No point having a 1ms model if it is wrong 50% of the time. Measured as percentage of correct predictions on a held-out test set.

**Memory Footprint**: Measures peak RAM usage during inference. Critical for edge devices with limited memory. A 100MB model cannot run on a device with 64MB RAM. Measured in megabytes.

**Model Size**: Measures storage size of model parameters. Critical for over-the-air updates and storage-constrained devices. A 500MB model takes minutes to download on slow networks. Measured in megabytes.

The key insight: **these metrics trade off against each other**. Quantization reduces memory but may reduce accuracy. Pruning reduces latency but may require retraining. Professional benchmarking reveals these trade-offs quantitatively.

### Statistical Validity

Statistics transforms noisy measurements into reliable conclusions. Without statistical validity, you cannot distinguish true performance differences from random noise.

**Why Variance Matters**: Consider two models. Model A runs in 10.2ms, 10.3ms, 10.1ms (low variance). Model B runs in 9.5ms, 12.1ms, 8.9ms (high variance). Is B faster? Looking at means (10.2ms vs 10.2ms) suggests they are identical, but B's high variance makes it unpredictable. Statistical analysis reveals both the mean and the reliability.

The BenchmarkResult class computes statistical properties automatically:

```python
@dataclass
class BenchmarkResult:
    metric_name: str
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute statistics after initialization."""
        if not self.values:
            raise ValueError("BenchmarkResult requires at least one measurement.")

        self.mean = statistics.mean(self.values)
        self.std = statistics.stdev(self.values) if len(self.values) > 1 else 0.0
        self.median = statistics.median(self.values)
        self.min_val = min(self.values)
        self.max_val = max(self.values)
        self.count = len(self.values)

        # 95% confidence interval for the mean
        if len(self.values) > 1:
            t_score = 1.96  # Approximate for large samples
            margin_error = t_score * (self.std / np.sqrt(self.count))
            self.ci_lower = self.mean - margin_error
            self.ci_upper = self.mean + margin_error
        else:
            self.ci_lower = self.ci_upper = self.mean
```

The confidence interval calculation uses the standard error of the mean (std / sqrt(n)) scaled by the t-score. For large samples, a t-score of 1.96 corresponds to 95% confidence. This means: "We are 95% confident the true mean latency lies between ci_lower and ci_upper."

**Coefficient of Variation**: The ratio std / mean measures relative noise. A CV of 0.05 means standard deviation is 5% of the mean, indicating stable measurements. A CV of 0.30 means 30% relative noise, indicating unstable or noisy measurements that need more samples.

**Outlier Detection**: Extreme values can skew the mean. The median is robust to outliers. If mean and median differ significantly, investigate outliers. They might indicate thermal throttling, background processes, or measurement errors.

### Reproducibility

Reproducibility means another engineer can run your benchmark and get the same results. Without reproducibility, your optimization insights are not transferable.

**System Metadata**: Recording system configuration ensures results are interpreted correctly. A benchmark on a 2020 laptop will differ from a 2024 server, not because the model changed, but because the hardware did.

```python
def __init__(self, models: List[Any], datasets: List[Any] = None,
             warmup_runs: int = DEFAULT_WARMUP_RUNS,
             measurement_runs: int = DEFAULT_MEASUREMENT_RUNS):
    self.models = models
    self.datasets = datasets or []
    self.warmup_runs = warmup_runs
    self.measurement_runs = measurement_runs

    # Capture system information for reproducibility
    self.system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count() or 1,
    }
```

This metadata gets embedded in every BenchmarkResult, allowing you to understand why a benchmark run produced specific numbers.

**Controlled Environment**: Background processes, thermal state, and power settings all affect measurements. Professional benchmarking controls these factors:

- Close unnecessary applications before benchmarking
- Let the system reach thermal equilibrium (run warmup)
- Use the same hardware configuration across runs
- Document any anomalies or environmental changes

**Versioning**: Record the versions of all dependencies. A NumPy update might change BLAS library behavior, affecting performance. Recording versions ensures results remain interpretable months later.

### Reporting Results

Raw data is useless without clear communication. Professional benchmarking generates reports that guide engineering decisions.

**Comparison Tables**: Show mean, standard deviation, and confidence intervals for each model on each metric. This lets stakeholders quickly identify winners and understand uncertainty.

**Pareto Frontiers**: When metrics trade off, visualize the Pareto frontier to show which models are optimal for different constraints. A model is Pareto-optimal if no other model is better on all metrics simultaneously.

Consider three models:
- Model A: 10ms latency, 90% accuracy
- Model B: 15ms latency, 95% accuracy
- Model C: 12ms latency, 91% accuracy

Model C is dominated by Model A (faster and only 1% less accurate). It is not Pareto-optimal. Models A and B are both Pareto-optimal: if you want maximum accuracy, choose B; if you want minimum latency, choose A.

**Visualization**: Scatter plots reveal relationships between metrics. Plot latency vs accuracy and you immediately see the trade-off frontier. Add annotations showing model names and you have an actionable decision tool.

**Statistical Significance**: Report not just means but confidence intervals. Saying "Model A is 2ms faster" is incomplete. Saying "Model A is 2ms faster with 95% confidence interval [1.5ms, 2.5ms], p < 0.01" provides the statistical rigor needed for engineering decisions.

## Common Errors

These are the mistakes students commonly make when implementing benchmarking infrastructure. Understanding these patterns will save you debugging time.

### Insufficient Measurement Runs

**Error**: Running a benchmark only 3 times produces unreliable statistics.

**Symptom**: Results change dramatically between benchmark runs. Confidence intervals are extremely wide.

**Cause**: The Central Limit Theorem requires sufficient samples. With only 3 measurements, the sample mean is a poor estimator of the true mean.

**Fix**: Use at least 10 measurement runs (the default in this module). For high-variance operations, increase to 30+ runs.

```python
# BAD: Only 3 measurements
benchmark = Benchmark(models, measurement_runs=3)  # Unreliable!

# GOOD: 10+ measurements
benchmark = Benchmark(models, measurement_runs=10)  # Statistical confidence
```

### Skipping Warmup

**Error**: Measuring performance without warmup captures cold-start behavior, not steady-state performance.

**Symptom**: First measurement is much slower than subsequent ones. Results do not reflect production performance.

**Cause**: JIT compilation, cache warming, and CPU frequency scaling all require several iterations to stabilize.

**Fix**: Always run warmup iterations before measurement.

```python
# Already handled in Benchmark class
for _ in range(self.warmup_runs):
    _ = model(input_data)  # Warmup (discarded)

for _ in range(self.measurement_runs):
    # Now measure steady-state performance
```

### Comparing Different Input Shapes

**Error**: Benchmarking Model A on 28x28 images and Model B on 224x224 images, then comparing latency.

**Symptom**: Misleading conclusions about which model is faster.

**Cause**: Larger inputs require more computation. You are measuring input size effects, not model efficiency.

**Fix**: Use identical input shapes across all models in a benchmark.

```python
# Ensure all models use same input shape
results = benchmark.run_latency_benchmark(input_shape=(1, 28, 28))
```

### Ignoring Variance

**Error**: Reporting only the mean, ignoring standard deviation and confidence intervals.

**Symptom**: Cannot determine if performance differences are statistically significant or just noise.

**Cause**: Treating measurements as deterministic when they are actually stochastic.

**Fix**: Always report confidence intervals, not just means.

```python
# BenchmarkResult automatically computes confidence intervals
result = BenchmarkResult("latency", measurements)
print(f"{result.mean:.3f}ms ± {result.std:.3f}ms, 95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

## Production Context

### Your Implementation vs. Industry Benchmarks

Your TinyTorch benchmarking infrastructure implements the same statistical principles used in production ML benchmarking frameworks. The difference is scale and automation.

| Feature | Your Implementation | MLPerf / Industry |
|---------|---------------------|-------------------|
| **Statistical Analysis** | Mean, std, 95% CI | Same + hypothesis testing, ANOVA |
| **Metrics** | Latency, accuracy, memory | Same + energy, throughput, tail latency |
| **Warmup Protocol** | Fixed warmup runs | Same + adaptive warmup until convergence |
| **Reproducibility** | System metadata | Same + hardware specs, thermal state |
| **Automation** | Manual benchmark runs | CI/CD integration, regression detection |
| **Scale** | Single machine | Distributed benchmarks across clusters |

### Code Comparison

The following shows equivalent benchmarking patterns in TinyTorch and production frameworks like MLPerf.

`````{tab-set}
````{tab-item} Your TinyTorch
```python
from tinytorch.benchmarking import Benchmark

# Setup models and benchmark
benchmark = Benchmark(
    models=[baseline_model, optimized_model],
    warmup_runs=5,
    measurement_runs=10
)

# Run latency benchmark
results = benchmark.run_latency_benchmark(input_shape=(1, 28, 28))

# Analyze results
for model_name, result in results.items():
    print(f"{model_name}: {result.mean*1000:.2f}ms ± {result.std*1000:.2f}ms")
```
````

````{tab-item} MLPerf (Industry Standard)
```python
import mlperf_loadgen as lg

# Configure benchmark scenario
settings = lg.TestSettings()
settings.scenario = lg.TestScenario.SingleStream
settings.mode = lg.TestMode.PerformanceOnly

# Run standardized benchmark
sut = SystemUnderTest(model)
lg.StartTest(sut, qsl, settings)

# Results include latency percentiles, throughput, accuracy
```
````
`````

Let's understand the comparison:

- **Line 1-3 (Setup)**: TinyTorch uses a simple class-based API. MLPerf uses a loadgen library with standardized scenarios (SingleStream, Server, MultiStream, Offline). Both ensure fair comparison.
- **Line 6-8 (Configuration)**: TinyTorch exposes warmup_runs and measurement_runs directly. MLPerf abstracts this into TestSettings with scenario-specific defaults. Same concept, different abstraction level.
- **Line 11-13 (Execution)**: TinyTorch returns BenchmarkResult objects with statistics. MLPerf logs results to standardized formats that compare across hardware vendors. Both provide statistical analysis.
- **Statistical Rigor**: Both use repeated measurements, warmup, and confidence intervals. TinyTorch teaches the foundations; MLPerf adds industry-specific requirements.

```{tip} What's Identical

The statistical methodology, warmup protocols, and reproducibility requirements are identical. Production frameworks add automation, standardization across organizations, and hardware-specific optimizations. Understanding TinyTorch benchmarking gives you the foundation to work with any industry benchmarking framework.
```

### Why Benchmarking Matters at Scale

Production ML systems operate at scales where small performance differences compound into massive resource consumption:

- **Cost**: A data center running 10,000 GPUs 24/7 consumes $50 million in electricity annually. Reducing latency 10% saves $5 million per year.
- **User Experience**: Search engines must return results in under 200ms. A 50ms latency reduction is the difference between keeping or losing users.
- **Sustainability**: Training GPT-3 consumed 1,287 MWh of energy, equivalent to the annual energy use of 120 US homes. Optimization reduces carbon footprint.

Fair benchmarking ensures optimization efforts focus on changes that produce measurable, statistically significant improvements.

## Check Your Understanding

Test your understanding of benchmarking statistics and methodology with these quantitative questions.

**Q1: Statistical Significance**

You benchmark a baseline model and an optimized model 10 times each. Baseline: mean=12.5ms, std=1.2ms. Optimized: mean=11.8ms, std=1.5ms. Is the optimized model statistically significantly faster?

```{admonition} Answer
:class: dropdown

**Calculate 95% confidence intervals:**

Baseline: CI = mean ± 1.96 * (std / sqrt(n)) = 12.5 ± 1.96 * (1.2 / sqrt(10)) = 12.5 ± 0.74 = [11.76, 13.24]

Optimized: CI = 11.8 ± 1.96 * (1.5 / sqrt(10)) = 11.8 ± 0.93 = [10.87, 12.73]

**Result**: The confidence intervals OVERLAP (baseline goes as low as 11.76, optimized goes as high as 12.73). This means the difference is **NOT statistically significant** at the 95% confidence level. You cannot confidently claim the optimized model is faster.

**Lesson**: Always compute confidence intervals. A 0.7ms difference in means might seem meaningful, but with these variances and sample sizes, it could be random noise.
```

**Q2: Sample Size Calculation**

You measure latency with standard deviation of 2.0ms. How many measurements do you need to achieve a 95% confidence interval width of ±0.5ms?

```{admonition} Answer
:class: dropdown

**Confidence interval formula**: margin = 1.96 * (std / sqrt(n))

**Solve for n**: 0.5 = 1.96 * (2.0 / sqrt(n))

sqrt(n) = 1.96 * 2.0 / 0.5 = 7.84

n = 7.84² = **61.5 ≈ 62 measurements**

**Lesson**: Achieving tight confidence intervals requires many measurements. Quadrupling precision (from ±1.0ms to ±0.5ms) requires 4x more samples (15 to 60). This is why professional benchmarks run hundreds of iterations.
```

**Q3: Warmup Impact**

Without warmup, your measurements are: [15.2, 12.1, 10.8, 10.5, 10.6, 10.4] ms. With 3 warmup runs discarded, your measurements are: [10.5, 10.6, 10.4, 10.7, 10.5, 10.6] ms. How much does warmup reduce measured latency and variance?

```{admonition} Answer
:class: dropdown

**Without warmup:**
- Mean = (15.2 + 12.1 + 10.8 + 10.5 + 10.6 + 10.4) / 6 = **11.6ms**
- Std = 1.8ms (high variance due to warmup effects)

**With warmup:**
- Mean = (10.5 + 10.6 + 10.4 + 10.7 + 10.5 + 10.6) / 6 = **10.55ms**
- Std = 0.1ms (low variance, stable measurements)

**Impact:**
- Latency reduced: 11.6 - 10.55 = **1.05ms (9% reduction)**
- Variance reduced: 1.8 → 0.1ms = **95% reduction in noise**

**Lesson**: Warmup eliminates cold-start effects and dramatically reduces measurement variance. Without warmup, you are measuring system startup behavior, not steady-state performance.
```

**Q4: Pareto Frontier**

You have three models with (latency, accuracy): A=(5ms, 88%), B=(8ms, 92%), C=(6ms, 89%). Which are Pareto-optimal?

```{admonition} Answer
:class: dropdown

**Pareto-optimal definition**: A model is Pareto-optimal if no other model is better on ALL metrics simultaneously.

**Analysis:**
- Model A vs B: A is faster (5ms < 8ms) but less accurate (88% < 92%). Neither dominates. Both Pareto-optimal.
- Model A vs C: A is faster (5ms < 6ms) but less accurate (88% < 89%). Neither dominates. Both Pareto-optimal.
- Model B vs C: B is slower (8ms > 6ms) but more accurate (92% > 89%). Neither dominates. Both Pareto-optimal.

**Result**: **All three models are Pareto-optimal**. None is strictly dominated by another.

**Lesson**: The Pareto frontier shows trade-off options. If you need minimum latency, choose A. If you need maximum accuracy, choose B. If you want balanced performance, choose C.
```

**Q5: Measurement Overhead**

Your timer has 1μs overhead per measurement. You measure a 50μs operation 1000 times. What percentage of measured time is overhead?

```{admonition} Answer
:class: dropdown

**Total true operation time**: 50μs × 1000 = 50,000μs = 50ms

**Total timer overhead**: 1μs × 1000 = 1,000μs = 1ms

**Total measured time**: 50ms + 1ms = 51ms

**Overhead percentage**: (1ms / 51ms) × 100% = **1.96%**

**Lesson**: Timer overhead is negligible for operations longer than ~50μs, but becomes significant for microsecond-scale operations. This is why we use `time.perf_counter()` with nanosecond resolution and minimal overhead. For operations under 10μs, consider measuring batches and averaging.
```

## Further Reading

For students who want to understand the academic and industry foundations of ML benchmarking:

### Seminal Papers

- **MLPerf: An Industry Standard Benchmark Suite for Machine Learning Performance** - Mattson et al. (2020). The definitive industry benchmark framework that establishes fair comparison methodology across hardware vendors. Your benchmarking infrastructure follows the same statistical principles MLPerf uses for standardized evaluation. [arXiv:1910.01500](https://arxiv.org/abs/1910.01500)

- **How to Benchmark Code Execution Times on Intel IA-32 and IA-64 Instruction Set Architectures** - Paoloni (2010). White paper explaining low-level timing mechanisms, measurement overhead, and sources of timing variance. Essential reading for understanding what happens when you call `time.perf_counter()`. [Intel](https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/ia-32-ia-64-benchmark-code-execution-paper.pdf)

- **Statistical Tests for Comparing Performance** - Georges et al. (2007). Academic treatment of statistical methodology for benchmarking, including hypothesis testing, sample size calculation, and handling measurement noise. [ACM OOPSLA](https://doi.org/10.1145/1297027.1297033)

- **Benchmarking Deep Learning Inference on Mobile Devices** - Ignatov et al. (2018). Comprehensive study of mobile ML benchmarking challenges including thermal throttling, battery constraints, and heterogeneous hardware. Shows how benchmarking methodology changes for resource-constrained devices. [arXiv:1812.01328](https://arxiv.org/abs/1812.01328)

### Additional Resources

- **Industry Standard**: [MLCommons MLPerf](https://mlcommons.org/en/inference-edge-11/) - Browse actual benchmark results across hardware vendors to see professional benchmarking in practice
- **Textbook**: "The Art of Computer Systems Performance Analysis" by Raj Jain - Comprehensive treatment of experimental design, statistical analysis, and benchmarking methodology for systems engineering

## What's Next

```{seealso} Coming Up: Module 20 - Capstone

Apply everything you have learned in Modules 01-19 to compete in the TorchPerf Olympics! You will strategically combine optimization techniques, benchmark your results, and compete for the fastest, smallest, or most accurate model on the leaderboard.
```

**Preview - How Your Benchmarking Gets Used in the Capstone:**

| Competition Event | Metric Optimized | Your Benchmark In Action |
|-------------------|------------------|--------------------------|
| **Latency Sprint** | Minimize inference time | `benchmark.run_latency_benchmark()` determines winners |
| **Memory Challenge** | Minimize model size | `benchmark.run_memory_benchmark()` tracks footprint |
| **Accuracy Contest** | Maximize accuracy under constraints | `benchmark.run_accuracy_benchmark()` validates quality |
| **All-Around** | Balanced Pareto frontier | `suite.generate_pareto_frontier()` finds optimal trade-offs |

## Get Started

```{tip} Interactive Options

- **[Launch Binder](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?urlpath=lab/tree/tinytorch/modules/19_benchmarking/19_benchmarking.ipynb)** - Run interactively in browser, no setup required
- **[View Source](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/19_benchmarking/19_benchmarking.py)** - Browse the implementation code
```

```{warning} Save Your Progress

Binder sessions are temporary. Download your completed notebook when done, or clone the repository for persistent local work.
```
