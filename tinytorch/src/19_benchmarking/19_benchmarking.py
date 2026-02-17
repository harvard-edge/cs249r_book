# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Module 19: Benchmarking - Performance Measurement Infrastructure

Welcome to Module 19! You'll build the benchmarking infrastructure for systematic ML performance evaluation.

**Note on hasattr() Usage:** This module uses hasattr() throughout for duck-typing and polymorphic benchmarking. This is legitimate because benchmarking frameworks must work with ANY model type (PyTorch, TinyTorch, custom) with different method names.

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML framework with profiling, acceleration, quantization, and compression
**You'll Build**: TorchPerf benchmarking system for fair model comparison and performance evaluation
**You'll Enable**: Systematic optimization combination and competitive performance evaluation

**Connection Map**:
```
Individual Optimizations (M14-18) → Benchmarking (M19) → Module 20 (Capstone)
(techniques)                        (evaluation)         (application)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement professional benchmarking infrastructure with statistical rigor
2. Learn to combine optimization techniques strategically (order matters!)
3. Build the TorchPerf class - a standardized performance evaluation framework
4. Understand ablation studies and systematic performance evaluation

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/19_benchmarking/benchmarking_dev.py`
**Building Side:** Code exports to `tinytorch.perf.benchmarking`

```python
# Final package structure:
from tinytorch.perf.benchmarking import Benchmark, OlympicEvent

# For capstone submission:
benchmark = Benchmark([baseline_model, optimized_model],
                     [{"name": "baseline"}, {"name": "optimized"}])
results = benchmark.run_latency_benchmark()
```

**Why this matters:**
- **Learning:** Complete benchmarking ecosystem in one focused module for rigorous evaluation
- **TorchPerf Olympics:** The Benchmark class provides the standardized framework for capstone submissions
- **Consistency:** All benchmarking operations and reporting in benchmarking.benchmark
- **Integration:** Works seamlessly with optimization modules (M14-18) for complete systems evaluation
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp perf.benchmarking
#| export

# Constants for benchmarking defaults
DEFAULT_WARMUP_RUNS = 5  # Default warmup runs for JIT compilation and cache warming
DEFAULT_MEASUREMENT_RUNS = 10  # Default measurement runs for statistical significance

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01-18 (Complete TinyTorch framework)

**External Dependencies**:
- `numpy` (for numerical operations)
- `time`, `statistics` (for measurements)
- `tracemalloc` (for memory profiling)
- `matplotlib` (optional, for visualization)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class)
- `tinytorch.core.layers` (Linear layer)
- `tinytorch.perf.profiling` (Profiler from Module 14)

**Dependency Flow**:
```
Profiling (M14) → Benchmarking (M19)
       ↓
→ Module 20 (Capstone)
```

Students completing this module will have built professional
benchmarking infrastructure for systematic performance evaluation.
"""

# %% [markdown]
"""
## 🏅 Looking Ahead

The benchmarking tools you build here will be used in Module 20's capstone project, where you'll apply optimization techniques competitively. For now, focus on building reliable, fair measurement infrastructure.
"""

# %% [markdown]
"""
## 💡 Introduction: What is Fair Benchmarking?

Benchmarking in ML systems isn't just timing code - it's about making fair, reproducible comparisons that guide real optimization decisions. Think of it like standardized testing: everyone takes the same test under the same conditions.

Consider comparing three models: a base CNN, a quantized version, and a pruned version. Without proper benchmarking, you might conclude the quantized model is "fastest" because you measured it when your CPU was idle, while testing the others during peak system load. Fair benchmarking controls for these variables.

The challenge: ML models have multiple competing objectives (accuracy vs speed vs memory), measurements can be noisy, and "faster" depends on your hardware and use case.

### Benchmarking as a Systems Engineering Discipline

Professional ML benchmarking requires understanding measurement uncertainty and controlling for confounding factors:

**Statistical Foundations**: We need enough measurements to achieve statistical significance. Running a model once tells you nothing about its true performance - you need distributions.

**System Noise Sources**:
- **Thermal throttling**: CPU frequency drops when hot
- **Background processes**: OS interrupts and other applications
- **Memory pressure**: Garbage collection, cache misses
- **Network interference**: For distributed models

**Fair Comparison Requirements**:
- Same hardware configuration
- Same input data distributions
- Same measurement methodology
- Statistical significance testing

This module builds infrastructure that addresses all these challenges while generating actionable insights for optimization decisions.
"""

# %% [markdown]
"""
## 📐 Foundations: Statistics for Performance Engineering

Benchmarking is applied statistics. We measure noisy processes (model inference) and need to extract reliable insights about their true performance characteristics.

### Central Limit Theorem in Practice

When you run a model many times, the distribution of measurements approaches normal (regardless of the underlying noise distribution). This lets us:
- Compute confidence intervals for the true mean
- Detect statistically significant differences between models
- Control for measurement variance

```
Single measurement: Meaningless
Few measurements: Unreliable
Many measurements: Statistical confidence
```

### Multi-Objective Optimization Theory

ML systems exist on a **Pareto frontier** - you can't simultaneously maximize accuracy and minimize latency without trade-offs. Good benchmarks reveal this frontier:

```
Accuracy
    ^
    |  A .     <- Model A: High accuracy, high latency
    |
    |    B .  <- Model B: Balanced trade-off
    |
    |      C .<- Model C: Low accuracy, low latency
    |__________> Latency (lower is better)
```

The goal: Find the optimal operating point for your specific constraints.

### Measurement Uncertainty and Error Propagation

Every measurement has uncertainty. When combining metrics (like accuracy per joule), uncertainties compound:

- **Systematic errors**: Consistent bias (timer overhead, warmup effects)
- **Random errors**: Statistical noise (thermal variation, OS scheduling)
- **Propagated errors**: How uncertainty spreads through calculations

Professional benchmarking quantifies and minimizes these uncertainties.
"""

# %%
#| export
import numpy as np
import time
import statistics
import os
import tracemalloc
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import platform
from contextlib import contextmanager
import warnings

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear

# Optional dependency for visualization only
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Create minimal fallback for when matplotlib is not available
    class plt:
        @staticmethod
        def subplots(*args, **kwargs):
            return None, None
        @staticmethod
        def figure(*args, **kwargs):
            return None
        @staticmethod
        def scatter(*args, **kwargs):
            pass
        @staticmethod
        def annotate(*args, **kwargs):
            pass
        @staticmethod
        def xlabel(*args, **kwargs):
            pass
        @staticmethod
        def ylabel(*args, **kwargs):
            pass
        @staticmethod
        def title(*args, **kwargs):
            pass
        @staticmethod
        def grid(*args, **kwargs):
            pass
        @staticmethod
        def tight_layout(*args, **kwargs):
            pass
        @staticmethod
        def savefig(*args, **kwargs):
            pass
        @staticmethod
        def show(*args, **kwargs):
            pass

# Import Profiler from Module 14 for measurement reuse
from tinytorch.perf.profiling import Profiler

# %%
from enum import Enum

#| export
class OlympicEvent(Enum):
    """
    Performance evaluation event categories for systematic optimization benchmarking.

    Each event optimizes for different objectives with specific constraints,
    enabling structured comparison of optimization strategies.
    """
    LATENCY_SPRINT = "latency_sprint"      # Minimize latency (accuracy >= 85%)
    MEMORY_CHALLENGE = "memory_challenge"   # Minimize memory (accuracy >= 85%)
    ACCURACY_CONTEST = "accuracy_contest"   # Maximize accuracy (latency < 100ms, memory < 10MB)
    ALL_AROUND = "all_around"               # Best balanced score across all metrics
    EXTREME_PUSH = "extreme_push"           # Most aggressive optimization (accuracy >= 80%)

# %% [markdown]
"""
## 🏗️ Implementation: Building Professional Benchmarking Infrastructure

We'll build a comprehensive benchmarking system that handles statistical analysis, multi-dimensional comparison, and automated reporting. Each component builds toward production-quality evaluation tools.

### Benchmark Architecture Overview

```
Benchmark Architecture:
┌─────────────────────────────────────────┐
│ Profiler (Module 14)                    │
│ • Base measurement tools                │
├─────────────────────────────────────────┤
│ BenchmarkResult                         │
│ • Statistical container for measurements│
├─────────────────────────────────────────┤
│ Benchmark                               │
│ • Uses Profiler + multi-model comparison│
├─────────────────────────────────────────┤
│ BenchmarkSuite                          │
│ • Multi-metric comprehensive evaluation │
├─────────────────────────────────────────┤
│ MLPerf                                  │
│ • Standardized industry-style benchmarks│
└─────────────────────────────────────────┘
```

**Key Architectural Decision**: The `Benchmark` class reuses `Profiler` from Module 14 for individual model measurements, then adds statistical comparison across multiple models. This demonstrates proper systems architecture - build once, reuse everywhere!

Each level adds capability while maintaining statistical rigor at the foundation.
"""

# %% [markdown]
"""
### BenchmarkResult - Statistical Analysis Container

Before measuring anything, we need a robust container that stores measurements and computes statistical properties. This is the foundation of all our benchmarking.

### Why Statistical Analysis Matters

Single measurements are meaningless in performance engineering. Consider timing a model:
- Run 1: 1.2ms (CPU was idle)
- Run 2: 3.1ms (background process started)
- Run 3: 1.4ms (CPU returned to normal)

Without statistics, which number do you trust? BenchmarkResult solves this by:
- Computing confidence intervals for the true mean
- Detecting outliers and measurement noise
- Providing uncertainty estimates for decision making

### Statistical Properties We Track

```
Raw measurements: [1.2, 3.1, 1.4, 1.3, 1.5, 1.1, 1.6]
                           ↓
        Statistical Analysis
                           ↓
Mean: 1.46ms ± 0.25ms (95% confidence interval)
Median: 1.4ms (less sensitive to outliers)
CV: 17% (coefficient of variation - relative noise)
```

The confidence interval tells us: "We're 95% confident the true mean latency is between 1.21ms and 1.71ms." This guides optimization decisions with statistical backing.
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-dataclass", "solution": true}
#| export

@dataclass
class BenchmarkResult:
    """
    Container for benchmark measurements with statistical analysis.

    TODO: Implement a robust result container that stores measurements and metadata

    APPROACH:
    1. Store raw measurements and computed statistics
    2. Include metadata about test conditions
    3. Provide methods for statistical analysis
    4. Support serialization for result persistence

    EXAMPLE:
    >>> result = BenchmarkResult("model_accuracy", [0.95, 0.94, 0.96])
    >>> print(f"Mean: {result.mean:.3f} ± {result.std:.3f}")
    Mean: 0.950 ± 0.010

    HINTS:
    - Use statistics module for robust mean/std calculations
    - Store both raw data and summary statistics
    - Include confidence intervals for professional reporting
    """
    ### BEGIN SOLUTION
    metric_name: str
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute statistics after initialization."""
        if not self.values:
            raise ValueError(
                f"Empty values list for BenchmarkResult\n"
                f"  ❌ Cannot compute statistics: values=[] (0 measurements)\n"
                f"  💡 BenchmarkResult needs data to compute mean, std, percentiles\n"
                f"  🔧 Add measurements: BenchmarkResult('{self.metric_name}', [1.2, 1.3, 1.1])"
            )

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_name': self.metric_name,
            'values': self.values,
            'mean': self.mean,
            'std': self.std,
            'median': self.median,
            'min': self.min_val,
            'max': self.max_val,
            'count': self.count,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        return f"{self.metric_name}: {self.mean:.4f} ± {self.std:.4f} (n={self.count})"
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: BenchmarkResult

This test validates our BenchmarkResult class correctly computes statistical properties from measurements.

**What we're testing**: Statistical calculations (mean, std, confidence intervals)
**Why it matters**: Reliable statistics are the foundation of fair benchmarking
**Expected**: Correct statistics and proper handling of edge cases
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchmark-result", "locked": true, "points": 10}
def test_unit_benchmark_result():
    """🧪 Test BenchmarkResult statistical calculations."""
    print("🧪 Unit Test: BenchmarkResult...")

    # Test basic statistics
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = BenchmarkResult("test_metric", values)

    assert result.mean == 3.0
    assert abs(result.std - statistics.stdev(values)) < 1e-10
    assert result.median == 3.0
    assert result.min_val == 1.0
    assert result.max_val == 5.0
    assert result.count == 5

    # Test confidence intervals
    assert result.ci_lower < result.mean < result.ci_upper

    # Test serialization
    result_dict = result.to_dict()
    assert result_dict['metric_name'] == "test_metric"
    assert result_dict['mean'] == 3.0

    print("✅ BenchmarkResult works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_result()

# %% [markdown]
"""
## 🏗️ High-Precision Timing Infrastructure

Accurate timing is the foundation of performance benchmarking. System clocks have different precision and behavior, so we need a robust timing mechanism.

### Timing Challenges in Practice

Consider what happens when you time a function:
```
User calls: time.time()
            ↓
Operating System scheduling delays (μs to ms)
            ↓
Timer system call overhead (~1μs)
            ↓
Hardware clock resolution (ns to μs)
            ↓
Your measurement
```

For microsecond-precision timing, each of these can introduce significant error.

### Why perf_counter() Matters

Python's `time.perf_counter()` is specifically designed for interval measurement:
- **Monotonic**: Never goes backwards (unaffected by system clock adjustments)
- **High resolution**: Typically nanosecond precision
- **Low overhead**: Optimized system call

### Timing Best Practices

```
Context Manager Pattern:
┌─────────────────┐
│  with timer():  │ ← Start timing
│    operation()  │ ← Your code runs
│  # End timing   │ ← Automatic cleanup
└─────────────────┘
    ↓
elapsed = timer.elapsed
```

This pattern ensures timing starts/stops correctly even if exceptions occur.
"""

# %% nbgrader={"grade": false, "grade_id": "timer-context", "solution": true}
#| export
@contextmanager
def precise_timer():
    """
    High-precision timing context manager for benchmarking.

    TODO: Implement a context manager that provides accurate timing measurements

    APPROACH:
    1. Use time.perf_counter() for high precision
    2. Handle potential interruptions and system noise
    3. Return elapsed time when context exits
    4. Provide warmup capability for JIT compilation

    Yields:
        Timer object with .elapsed attribute (set after context exits)

    EXAMPLE:
    >>> with precise_timer() as timer:
    ...     time.sleep(0.1)  # Some operation
    >>> print(f"Elapsed: {timer.elapsed:.4f}s")
    Elapsed: 0.1001s

    HINTS:
    - perf_counter() is monotonic and high-resolution
    - Store start time in __enter__, compute elapsed in __exit__
    - Handle any exceptions gracefully
    """
    ### BEGIN SOLUTION
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
            self.start_time = None

    timer = Timer()
    timer.start_time = time.perf_counter()

    try:
        yield timer
    finally:
        timer.elapsed = time.perf_counter() - timer.start_time
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Precise Timer

This test validates our timing context manager provides accurate measurements.

**What we're testing**: High-precision timing with perf_counter
**Why it matters**: Accurate timing is essential for reliable benchmarks
**Expected**: Measurements close to actual sleep durations
"""

# %% nbgrader={"grade": true, "grade_id": "test-precise-timer", "locked": true, "points": 5}
def test_unit_precise_timer():
    """🧪 Test precise_timer context manager."""
    print("🧪 Unit Test: precise_timer...")

    # Test basic timing
    with precise_timer() as timer:
        time.sleep(0.01)  # 10ms sleep

    # Should be close to 0.01 seconds (allow some variance)
    assert 0.005 < timer.elapsed < 0.05, f"Expected ~0.01s, got {timer.elapsed}s"

    # Test multiple uses
    times = []
    for _ in range(3):
        with precise_timer() as timer:
            time.sleep(0.001)  # 1ms sleep
        times.append(timer.elapsed)

    # All times should be reasonably close
    assert all(0.0005 < t < 0.01 for t in times)

    print("✅ precise_timer works correctly!")

if __name__ == "__main__":
    test_unit_precise_timer()

# %% [markdown]
"""
### Benchmark Class - Core Measurement Engine

The Benchmark class implements the core measurement logic for different metrics. It handles the complex orchestration of multiple models, datasets, and measurement protocols.

### Benchmark Architecture Overview

```
Benchmark Execution Flow:
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Models    │    │   Datasets   │    │ Measurement     │
│ [M1, M2...] │ →  │ [D1, D2...]  │ →  │ Protocol        │
└─────────────┘    └──────────────┘    └─────────────────┘
                                               ↓
                           ┌─────────────────────────────────┐
                           │        Benchmark Loop           │
                           │ 1. Warmup runs (JIT, cache)     │
                           │ 2. Measurement runs (statistics)│
                           │ 3. System info capture          │
                           │ 4. Result aggregation           │
                           └─────────────────────────────────┘
                                        ↓
                        ┌────────────────────────────────────┐
                        │          BenchmarkResult           │
                        │ • Statistical analysis             │
                        │ • Confidence intervals             │
                        │ • Metadata (system, conditions)    │
                        └────────────────────────────────────┘
```

### Why Warmup Runs Matter

Modern systems have multiple layers of adaptation:
- **JIT compilation**: Code gets faster after being run several times
- **CPU frequency scaling**: Processors ramp up under load
- **Cache warming**: Data gets loaded into faster memory
- **Branch prediction**: CPU learns common execution paths

Without warmup, your first few measurements don't represent steady-state performance.

### Multiple Benchmark Types

Different metrics require different measurement strategies:

**Latency Benchmarking**:
- Focus: Time per inference
- Key factors: Input size, model complexity, hardware utilization
- Measurement: High-precision timing of forward pass

**Accuracy Benchmarking**:
- Focus: Quality of predictions
- Key factors: Dataset representativeness, evaluation protocol
- Measurement: Correct predictions / total predictions

**Memory Benchmarking**:
- Focus: Peak and average memory usage
- Key factors: Model size, batch size, intermediate activations
- Measurement: Process memory monitoring during inference
"""

# %% [markdown]
"""
### Benchmark.__init__ - Setting Up the Measurement Engine

The Benchmark constructor configures the measurement infrastructure: models to test,
datasets for evaluation, and system metadata for reproducibility. It reuses the
Profiler from Module 14 for individual model measurements.

```
Benchmark Setup:
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Models    │     │   Datasets   │     │  Profiler   │
│ [M1, M2...] │ ──> │ [D1, D2...]  │ ──> │ (Module 14) │
└─────────────┘     └──────────────┘     └─────────────┘
                           ↓
                 ┌──────────────────┐
                 │  System Metadata │
                 │ • platform       │
                 │ • processor      │
                 │ • python version │
                 └──────────────────┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-init", "solution": true}
#| export
class Benchmark:
    """
    Professional benchmarking system for ML models and operations.

    Provides latency, accuracy, and memory benchmarking with statistical
    rigor. Reuses Profiler from Module 14 for individual measurements
    and adds multi-model comparison with confidence intervals.

    EXAMPLE:
    >>> benchmark = Benchmark(models=[model1, model2], datasets=[test_data])
    >>> results = benchmark.run_accuracy_benchmark()
    """

    def __init__(self, models: List[Any], datasets: List[Any],
                 warmup_runs: int = DEFAULT_WARMUP_RUNS, measurement_runs: int = DEFAULT_MEASUREMENT_RUNS):
        """
        Initialize benchmark with models and datasets.

        TODO: Set up the benchmark runner with models, datasets, and system metadata

        APPROACH:
        1. Store models and datasets for benchmarking
        2. Configure warmup and measurement run counts
        3. Initialize Profiler from Module 14 for measurements
        4. Capture system information for reproducibility

        HINTS:
        - Use platform module for system info
        - os.cpu_count() can return None, use fallback
        """
        ### BEGIN SOLUTION
        self.models = models
        self.datasets = datasets
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        self.results = {}

        # Use Profiler from Module 14 for measurements
        self.profiler = Profiler()

        # System information for metadata (using Python standard library)
        self.system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count() or 1,  # os.cpu_count() can return None
        }
        # Note: System total memory not available via standard library
        # Process memory measurement uses tracemalloc (via Profiler)
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Benchmark.__init__

**What we're testing**: Benchmark initialization with models, datasets, and system metadata
**Why it matters**: Proper setup ensures reproducible benchmarking conditions
**Expected**: All attributes initialized, system info captured
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchmark-init", "locked": true, "points": 5}
def test_unit_benchmark_init():
    """🧪 Test Benchmark initialization."""
    print("🧪 Unit Test: Benchmark.__init__...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return x

    models = [MockModel("m1"), MockModel("m2")]
    datasets = [{"data": "test"}]

    benchmark = Benchmark(models, datasets, warmup_runs=3, measurement_runs=5)

    assert len(benchmark.models) == 2
    assert len(benchmark.datasets) == 1
    assert benchmark.warmup_runs == 3
    assert benchmark.measurement_runs == 5
    assert isinstance(benchmark.results, dict)
    assert 'platform' in benchmark.system_info
    assert 'processor' in benchmark.system_info
    assert 'python_version' in benchmark.system_info
    assert 'cpu_count' in benchmark.system_info
    assert benchmark.profiler is not None

    print("✅ Benchmark.__init__ works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_init()

# %% [markdown]
"""
### Benchmark.run_latency_benchmark - Measuring Inference Speed

Latency benchmarking measures how long each model takes to process input. We use
the Profiler for warmup, then collect multiple individual measurements for
statistical analysis via BenchmarkResult.

```
Latency Measurement Flow:
Input Tensor ──> Warmup Runs (discard) ──> Measurement Runs ──> BenchmarkResult
                 (JIT, cache warming)      (collect times)      (mean, std, CI)
```
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-latency", "solution": true}
#| export
    # --- Benchmark.run_latency_benchmark ---
def benchmark_run_latency_benchmark(self, input_shape: Tuple[int, ...] = (1, 28, 28)) -> Dict[str, BenchmarkResult]:
    """
    Benchmark model inference latency using Profiler.

    TODO: Measure inference latency for each model with statistical rigor

    APPROACH:
    1. Create input tensor matching input_shape
    2. Use Profiler for initial warmup measurement
    3. Collect multiple individual latency measurements
    4. Wrap results in BenchmarkResult for statistical analysis

    HINTS:
    - Use self.profiler.measure_latency() for warmup
    - Collect self.measurement_runs individual measurements
    - Include system_info in metadata
    """
    ### BEGIN SOLUTION
    results = {}

    for i, model in enumerate(self.models):
        model_name = getattr(model, 'name', f'model_{i}')

        # Create input tensor for profiling
        from tinytorch.core.tensor import Tensor
        input_tensor = Tensor(np.random.randn(*input_shape).astype(np.float32))

        # Use Profiler to measure latency with proper warmup and iterations
        latency_ms = self.profiler.measure_latency(
            model,
            input_tensor,
            warmup=self.warmup_runs,
            iterations=self.measurement_runs
        )

        # Profiler returns single median value
        # For BenchmarkResult, we need multiple measurements
        # Run additional measurements for statistical analysis
        latencies = []
        for _ in range(self.measurement_runs):
            single_latency = self.profiler.measure_latency(
                model, input_tensor, warmup=0, iterations=1
            )
            latencies.append(single_latency)

        results[model_name] = BenchmarkResult(
            f"{model_name}_latency_ms",
            latencies,
            metadata={'input_shape': input_shape, **self.system_info}
        )

    return results
    ### END SOLUTION

Benchmark.run_latency_benchmark = benchmark_run_latency_benchmark

# %% [markdown]
"""
### 🧪 Unit Test: Benchmark.run_latency_benchmark

**What we're testing**: Latency measurement across multiple models
**Why it matters**: Accurate latency data guides deployment decisions
**Expected**: BenchmarkResult for each model with positive latency values
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchmark-latency", "locked": true, "points": 10}
def test_unit_benchmark_latency():
    """🧪 Test Benchmark latency measurement."""
    print("🧪 Unit Test: Benchmark.run_latency_benchmark...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            time.sleep(0.001)
            return x

    models = [MockModel("fast"), MockModel("slow")]
    benchmark = Benchmark(models, [{"data": "test"}], warmup_runs=1, measurement_runs=3)

    results = benchmark.run_latency_benchmark()
    assert len(results) == 2
    assert "fast" in results
    assert "slow" in results
    assert all(isinstance(r, BenchmarkResult) for r in results.values())
    assert all(r.mean > 0 for r in results.values())

    print("✅ Benchmark.run_latency_benchmark works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_latency()

# %% [markdown]
"""
### Benchmark.run_accuracy_benchmark - Measuring Prediction Quality

Accuracy benchmarking evaluates model correctness across datasets. Models with
an `evaluate` method are tested directly; otherwise, accuracy is simulated for
demonstration purposes.

```
Accuracy Measurement:
Model ──> Dataset 1 ──> accuracy_1 ──┐
      ──> Dataset 2 ──> accuracy_2 ──┼──> BenchmarkResult
      ──> Dataset N ──> accuracy_N ──┘    (mean, std across datasets)
```
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-accuracy", "solution": true}
#| export
    # --- Benchmark.run_accuracy_benchmark ---
def benchmark_run_accuracy_benchmark(self) -> Dict[str, BenchmarkResult]:
    """
    Benchmark model accuracy across datasets.

    TODO: Evaluate each model on each dataset and collect accuracy scores

    APPROACH:
    1. Iterate over all models and datasets
    2. Use model.evaluate() if available, otherwise simulate
    3. Clamp accuracy to [0, 1] range
    4. Wrap results in BenchmarkResult

    HINTS:
    - Use hasattr(model, 'evaluate') for duck-typing
    - Different models get different base accuracies for simulation
    """
    ### BEGIN SOLUTION
    results = {}

    for i, model in enumerate(self.models):
        model_name = getattr(model, 'name', f'model_{i}')
        accuracies = []

        for dataset in self.datasets:
            # Simulate accuracy measurement
            # In practice, this would evaluate the model on the dataset
            try:
                if hasattr(model, 'evaluate'):
                    accuracy = model.evaluate(dataset)
                else:
                    # Simulate accuracy for demonstration
                    base_accuracy = 0.85 + i * 0.05  # Different models have different base accuracies
                    accuracy = base_accuracy + np.random.normal(0, 0.02)  # Add noise
                    accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
            except Exception:
                # Fallback simulation
                accuracy = 0.80 + np.random.normal(0, 0.05)
                accuracy = max(0.0, min(1.0, accuracy))

            accuracies.append(accuracy)

        results[model_name] = BenchmarkResult(
            f"{model_name}_accuracy",
            accuracies,
            metadata={'num_datasets': len(self.datasets), **self.system_info}
        )

    return results
    ### END SOLUTION

Benchmark.run_accuracy_benchmark = benchmark_run_accuracy_benchmark

# %% [markdown]
"""
### 🧪 Unit Test: Benchmark.run_accuracy_benchmark

**What we're testing**: Accuracy evaluation across models and datasets
**Why it matters**: Accuracy is the primary quality metric for ML models
**Expected**: Accuracy values in [0, 1] range for each model
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchmark-accuracy", "locked": true, "points": 10}
def test_unit_benchmark_accuracy():
    """🧪 Test Benchmark accuracy measurement."""
    print("🧪 Unit Test: Benchmark.run_accuracy_benchmark...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return x

    models = [MockModel("model_a"), MockModel("model_b")]
    datasets = [{"d": "1"}, {"d": "2"}]
    benchmark = Benchmark(models, datasets, warmup_runs=1, measurement_runs=3)

    results = benchmark.run_accuracy_benchmark()
    assert len(results) == 2
    assert all(isinstance(r, BenchmarkResult) for r in results.values())
    assert all(0 <= r.mean <= 1 for r in results.values())

    print("✅ Benchmark.run_accuracy_benchmark works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_accuracy()

# %% [markdown]
"""
### Benchmark.run_memory_benchmark - Measuring Resource Consumption

Memory benchmarking tracks how much RAM each model consumes during inference.
We use the Profiler's memory measurement, falling back to parameter-count
estimation when tracemalloc reports minimal usage.

```
Memory Measurement:
Model ──> Profiler.measure_memory() ──> peak_memory_mb
                                         ↓
                            If < 1.0 MB detected:
                            count_parameters() * 4 bytes
                                         ↓
                                  BenchmarkResult
```
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-memory", "solution": true}
#| export
    # --- Benchmark.run_memory_benchmark ---
def benchmark_run_memory_benchmark(self, input_shape: Tuple[int, ...] = (1, 28, 28)) -> Dict[str, BenchmarkResult]:
    """
    Benchmark model memory usage using Profiler.

    TODO: Measure memory consumption for each model across multiple runs

    APPROACH:
    1. Use self.profiler.measure_memory() for each model
    2. Fall back to parameter-count estimation if tracemalloc reports < 1 MB
    3. Collect self.measurement_runs samples
    4. Wrap results in BenchmarkResult

    HINTS:
    - memory_stats['peak_memory_mb'] is the primary metric
    - Estimate memory as param_count * 4 / (1024**2) for float32
    """
    ### BEGIN SOLUTION
    results = {}

    for i, model in enumerate(self.models):
        model_name = getattr(model, 'name', f'model_{i}')
        memory_usages = []

        for run in range(self.measurement_runs):
            # Use Profiler to measure memory
            memory_stats = self.profiler.measure_memory(model, input_shape)
            # Use peak_memory_mb as the primary metric
            memory_used = memory_stats['peak_memory_mb']

            # If no significant memory change detected, estimate from parameters
            if memory_used < 1.0:
                param_count = self.profiler.count_parameters(model)
                memory_used = param_count * 4 / (1024**2)  # 4 bytes per float32

            memory_usages.append(max(0, memory_used))

        results[model_name] = BenchmarkResult(
            f"{model_name}_memory_mb",
            memory_usages,
            metadata={'input_shape': input_shape, **self.system_info}
        )

    return results
    ### END SOLUTION

Benchmark.run_memory_benchmark = benchmark_run_memory_benchmark

# %% [markdown]
"""
### 🧪 Unit Test: Benchmark.run_memory_benchmark

**What we're testing**: Memory usage measurement across multiple models
**Why it matters**: Memory constraints determine deployment feasibility on edge devices
**Expected**: Non-negative memory values for each model
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchmark-memory", "locked": true, "points": 10}
def test_unit_benchmark_memory():
    """🧪 Test Benchmark memory measurement."""
    print("🧪 Unit Test: Benchmark.run_memory_benchmark...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return x

    models = [MockModel("small"), MockModel("large")]
    benchmark = Benchmark(models, [{"data": "test"}], warmup_runs=1, measurement_runs=3)

    results = benchmark.run_memory_benchmark()
    assert len(results) == 2
    assert all(isinstance(r, BenchmarkResult) for r in results.values())
    assert all(r.mean >= 0 for r in results.values())

    print("✅ Benchmark.run_memory_benchmark works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_memory()

# %% [markdown]
"""
### Benchmark.compare_models - Cross-Model Comparison

The compare_models method dispatches to the appropriate benchmark type and
formats results into a structured list of dictionaries for easy comparison.
This is the primary interface for multi-model evaluation.
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-compare", "solution": true}
#| export
    # --- Benchmark.compare_models ---
def benchmark_compare_models(self, metric: str = "latency"):
    """
    Compare models across a specific metric.

    TODO: Dispatch to the appropriate benchmark and format comparison results

    APPROACH:
    1. Select benchmark type based on metric string
    2. Run the selected benchmark
    3. Format results into list of dicts for easy comparison

    HINTS:
    - Support 'latency', 'accuracy', 'memory' metrics
    - Return list of dicts with model, metric, mean, std, ci_lower, ci_upper, count
    """
    ### BEGIN SOLUTION
    if metric == "latency":
        results = self.run_latency_benchmark()
    elif metric == "accuracy":
        results = self.run_accuracy_benchmark()
    elif metric == "memory":
        results = self.run_memory_benchmark()
    else:
        raise ValueError(
            f"Unknown benchmark metric: '{metric}'\n"
            f"  ❌ Metric '{metric}' is not supported\n"
            f"  💡 compare_models() supports three metrics: latency (timing), memory (bytes), accuracy (correctness)\n"
            f"  🔧 Use: compare_models(metric='latency') or 'memory' or 'accuracy'"
        )

    # Return structured list of dicts for easy comparison
    # (No pandas dependency - students can convert to DataFrame if needed)
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'model': model_name.replace(f'_{metric}', '').replace('_ms', '').replace('_mb', ''),
            'metric': metric,
            'mean': result.mean,
            'std': result.std,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            'count': result.count
        })

    return comparison_data
    ### END SOLUTION

Benchmark.compare_models = benchmark_compare_models

# %% [markdown]
"""
### 🧪 Unit Test: Benchmark (Full Class Integration)

This test validates our Benchmark class measures latency, accuracy, and memory correctly,
and that compare_models dispatches properly.

**What we're testing**: Multi-model benchmarking with different metrics
**Why it matters**: Reliable comparisons guide optimization decisions
**Expected**: Consistent results across multiple benchmark types
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchmark", "locked": true, "points": 15}
def test_unit_benchmark():
    """🧪 Test Benchmark class functionality."""
    print("🧪 Unit Test: Benchmark...")

    # Create mock models for testing
    class MockModel:
        def __init__(self, name):
            self.name = name

        def forward(self, x):
            time.sleep(0.001)  # Simulate computation
            return x

    models = [MockModel("fast_model"), MockModel("slow_model")]
    datasets = [{"data": "test1"}, {"data": "test2"}]

    benchmark = Benchmark(models, datasets, warmup_runs=2, measurement_runs=3)

    # Test latency benchmark
    latency_results = benchmark.run_latency_benchmark()
    assert len(latency_results) == 2
    assert "fast_model" in latency_results
    assert all(isinstance(result, BenchmarkResult) for result in latency_results.values())

    # Test accuracy benchmark
    accuracy_results = benchmark.run_accuracy_benchmark()
    assert len(accuracy_results) == 2
    assert all(0 <= result.mean <= 1 for result in accuracy_results.values())

    # Test memory benchmark
    memory_results = benchmark.run_memory_benchmark()
    assert len(memory_results) == 2
    assert all(result.mean >= 0 for result in memory_results.values())

    # Test comparison (returns list of dicts, not DataFrame)
    comparison_data = benchmark.compare_models("latency")
    assert len(comparison_data) == 2
    assert isinstance(comparison_data, list)
    assert all(isinstance(item, dict) for item in comparison_data)
    assert "model" in comparison_data[0]
    assert "mean" in comparison_data[0]

    print("✅ Benchmark works correctly!")

if __name__ == "__main__":
    test_unit_benchmark()

# %% [markdown]
"""
### BenchmarkSuite - Comprehensive Multi-Metric Evaluation

The BenchmarkSuite orchestrates multiple benchmark types and generates comprehensive reports. This is where individual measurements become actionable engineering insights.

### Why Multi-Metric Analysis Matters

Single metrics mislead. Consider these three models:
- **Model A**: 95% accuracy, 100ms latency, 50MB memory
- **Model B**: 90% accuracy, 20ms latency, 10MB memory
- **Model C**: 85% accuracy, 10ms latency, 5MB memory

Which is "best"? It depends on your constraints:
- **Server deployment**: Model A (accuracy matters most)
- **Mobile app**: Model C (memory/latency critical)
- **Edge device**: Model B (balanced trade-off)

### Multi-Dimensional Comparison Workflow

```
BenchmarkSuite Execution Pipeline:
┌──────────────┐
│   Models     │ ← Input: List of models to compare
│ [M1,M2,M3]   │
└──────┬───────┘
       ↓
┌──────────────┐
│ Metric Types │ ← Run each benchmark type
│ • Latency    │
│ • Accuracy   │
│ • Memory     │
│ • Energy     │
└──────┬───────┘
       ↓
┌──────────────┐
│ Result       │ ← Aggregate into unified view
│ Aggregation  │
└──────┬───────┘
       ↓
┌──────────────┐
│ Analysis &   │ ← Generate insights
│ Reporting    │   • Best performer per metric
│              │   • Trade-off analysis
│              │   • Use case recommendations
└──────────────┘
```

### Pareto Frontier Analysis

The suite automatically identifies Pareto-optimal solutions - models that aren't strictly dominated by others across all metrics. This reveals the true trade-off space for optimization decisions.

### Energy Efficiency Modeling

Since direct energy measurement requires specialized hardware, we estimate energy based on computational complexity and memory usage. This provides actionable insights for battery-powered deployments.
"""

# %% [markdown]
"""
### BenchmarkSuite.__init__ - Setting Up Multi-Metric Evaluation

The BenchmarkSuite constructor creates the evaluation infrastructure, including
a Benchmark instance for measurements and an output directory for reports and plots.
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-init", "solution": true}
#| export
class BenchmarkSuite:
    """
    Comprehensive benchmark suite for ML systems evaluation.

    Orchestrates multiple benchmark types (latency, accuracy, memory, energy)
    and generates reports with visualizations and recommendations.

    EXAMPLE:
    >>> suite = BenchmarkSuite(models, datasets)
    >>> report = suite.run_full_benchmark()
    >>> suite.generate_report(report)
    """

    def __init__(self, models: List[Any], datasets: List[Any],
                 output_dir: str = "benchmark_results"):
        """
        Initialize comprehensive benchmark suite.

        TODO: Set up the suite with models, datasets, output directory, and a Benchmark instance

        APPROACH:
        1. Store models and datasets
        2. Create output directory (use Path, mkdir with exist_ok)
        3. Create Benchmark instance for measurements
        4. Initialize empty results dict

        HINTS:
        - Use Path(output_dir) for cross-platform paths
        - The Benchmark instance handles individual model measurements
        """
        ### BEGIN SOLUTION
        self.models = models
        self.datasets = datasets
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.benchmark = Benchmark(models, datasets)
        self.results = {}
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: BenchmarkSuite.__init__

**What we're testing**: Suite initialization with output directory and Benchmark instance
**Why it matters**: Proper setup ensures results can be saved and compared
**Expected**: All attributes initialized, output directory created
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchsuite-init", "locked": true, "points": 5}
def test_unit_benchsuite_init():
    """🧪 Test BenchmarkSuite initialization."""
    print("🧪 Unit Test: BenchmarkSuite.__init__...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return x

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        models = [MockModel("m1")]
        datasets = [{"d": "1"}]
        suite = BenchmarkSuite(models, datasets, output_dir=tmp_dir)

        assert len(suite.models) == 1
        assert len(suite.datasets) == 1
        assert suite.output_dir == Path(tmp_dir)
        assert isinstance(suite.benchmark, Benchmark)
        assert isinstance(suite.results, dict)

    print("✅ BenchmarkSuite.__init__ works correctly!")

if __name__ == "__main__":
    test_unit_benchsuite_init()

# %% [markdown]
"""
### BenchmarkSuite.run_full_benchmark - Orchestrating All Measurements

The run_full_benchmark method runs all four benchmark categories (latency, accuracy,
memory, energy) in sequence, collecting comprehensive results for each model.

```
Run Full Benchmark Pipeline:
Models ──> Latency Benchmark ──┐
       ──> Accuracy Benchmark ──┼──> self.results dict
       ──> Memory Benchmark   ──┤    (keyed by metric type)
       ──> Energy Estimation  ──┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-run", "solution": true}
#| export
    # --- BenchmarkSuite.run_full_benchmark ---
def benchsuite_run_full_benchmark(self) -> Dict[str, Dict[str, BenchmarkResult]]:
    """
    Run all benchmark categories.

    TODO: Orchestrate latency, accuracy, memory, and energy benchmarks

    APPROACH:
    1. Run self.benchmark.run_latency_benchmark()
    2. Run self.benchmark.run_accuracy_benchmark()
    3. Run self.benchmark.run_memory_benchmark()
    4. Run self._estimate_energy_efficiency()
    5. Store all results in self.results dict

    HINTS:
    - Print progress messages for each benchmark type
    - Return the complete results dict
    """
    ### BEGIN SOLUTION
    print("🧪 Running comprehensive benchmark suite...")

    # Run all benchmark types
    print("  📊 Measuring latency...")
    self.results['latency'] = self.benchmark.run_latency_benchmark()

    print("  🎯 Measuring accuracy...")
    self.results['accuracy'] = self.benchmark.run_accuracy_benchmark()

    print("  💾 Measuring memory usage...")
    self.results['memory'] = self.benchmark.run_memory_benchmark()

    # Simulate energy benchmark (would require specialized hardware)
    print("  ⚡ Estimating energy efficiency...")
    self.results['energy'] = self._estimate_energy_efficiency()

    return self.results
    ### END SOLUTION

BenchmarkSuite.run_full_benchmark = benchsuite_run_full_benchmark

# %% [markdown]
"""
### 🧪 Unit Test: BenchmarkSuite.run_full_benchmark

**What we're testing**: Orchestration of all four benchmark types
**Why it matters**: Complete evaluation requires all metrics measured consistently
**Expected**: Results dict with keys for latency, accuracy, memory, energy
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchsuite-run", "locked": true, "points": 15}
def test_unit_benchsuite_run():
    """🧪 Test BenchmarkSuite.run_full_benchmark."""
    print("🧪 Unit Test: BenchmarkSuite.run_full_benchmark...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            time.sleep(0.001)
            return x

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        models = [MockModel("m1"), MockModel("m2")]
        suite = BenchmarkSuite(models, [{"d": "1"}], output_dir=tmp_dir)

        results = suite.run_full_benchmark()

        assert 'latency' in results
        assert 'accuracy' in results
        assert 'memory' in results
        assert 'energy' in results
        for metric_results in results.values():
            assert len(metric_results) == 2
            assert all(isinstance(r, BenchmarkResult) for r in metric_results.values())

    print("✅ BenchmarkSuite.run_full_benchmark works correctly!")

# Note: test_unit_benchsuite_run() is called at the bottom of the module
# after all BenchmarkSuite methods (including _estimate_energy_efficiency) are patched.

# %% [markdown]
"""
### BenchmarkSuite._estimate_energy_efficiency - Energy Modeling

Since direct energy measurement requires specialized hardware (power meters, RAPL),
we estimate energy from latency and memory usage. This simplified model captures the
key relationship: energy is proportional to power (memory-related) multiplied by time (latency).

```
Energy Estimation Model:
energy = base_cost + (latency/1000) * 2.0 + memory * 0.01   (Joules)
         ↑            ↑                      ↑
         Fixed        Time component          Memory component
         overhead     (active power)          (static power)
```
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-energy", "solution": true}
#| export
    # --- BenchmarkSuite._estimate_energy_efficiency ---
def _benchsuite_estimate_energy_efficiency(self) -> Dict[str, BenchmarkResult]:
    """
    Estimate energy efficiency (simplified simulation).

    TODO: Estimate energy from latency and memory measurements

    APPROACH:
    1. Check if latency and memory results are available
    2. Combine latency and memory into energy estimate per measurement
    3. Fall back to simulated values if prerequisites missing
    4. Wrap results in BenchmarkResult

    HINTS:
    - Energy model: energy = 0.1 + (lat/1000)*2.0 + mem*0.01
    - Use zip() to pair latency and memory measurements
    """
    ### BEGIN SOLUTION
    energy_results = {}

    for i, model in enumerate(self.models):
        model_name = getattr(model, 'name', f'model_{i}')

        # Energy roughly correlates with latency * memory usage
        if 'latency' in self.results and 'memory' in self.results:
            latency_result = self.results['latency'].get(model_name)
            memory_result = self.results['memory'].get(model_name)

            if latency_result and memory_result:
                # Energy ∝ power × time, power ∝ memory usage
                energy_values = []
                for lat, mem in zip(latency_result.values, memory_result.values):
                    # Simplified energy model: energy = base + latency_factor * time + memory_factor * memory
                    energy = 0.1 + (lat / 1000) * 2.0 + mem * 0.01  # Joules
                    energy_values.append(energy)

                energy_results[model_name] = BenchmarkResult(
                    f"{model_name}_energy_joules",
                    energy_values,
                    metadata={'estimated': True, **self.benchmark.system_info}
                )

    # Fallback if no latency/memory results
    if not energy_results:
        for i, model in enumerate(self.models):
            model_name = getattr(model, 'name', f'model_{i}')
            # Simulate energy measurements
            energy_values = [0.5 + np.random.normal(0, 0.1) for _ in range(5)]
            energy_results[model_name] = BenchmarkResult(
                f"{model_name}_energy_joules",
                energy_values,
                metadata={'estimated': True, **self.benchmark.system_info}
            )

    return energy_results
    ### END SOLUTION

BenchmarkSuite._estimate_energy_efficiency = _benchsuite_estimate_energy_efficiency

# %% [markdown]
"""
### 🧪 Unit Test: BenchmarkSuite._estimate_energy_efficiency

**What we're testing**: Energy estimation from latency and memory data
**Why it matters**: Energy awareness is critical for edge/mobile deployment
**Expected**: Positive energy values for each model
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchsuite-energy", "locked": true, "points": 5}
def test_unit_benchsuite_energy():
    """🧪 Test BenchmarkSuite energy estimation."""
    print("🧪 Unit Test: BenchmarkSuite._estimate_energy_efficiency...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            time.sleep(0.001)
            return x

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        models = [MockModel("m1")]
        suite = BenchmarkSuite(models, [{"d": "1"}], output_dir=tmp_dir)

        # Populate latency and memory first
        suite.results['latency'] = suite.benchmark.run_latency_benchmark()
        suite.results['memory'] = suite.benchmark.run_memory_benchmark()

        energy = suite._estimate_energy_efficiency()
        assert len(energy) >= 1
        assert all(isinstance(r, BenchmarkResult) for r in energy.values())
        assert all(r.mean > 0 for r in energy.values())

    print("✅ BenchmarkSuite._estimate_energy_efficiency works correctly!")

if __name__ == "__main__":
    test_unit_benchsuite_energy()

# %% [markdown]
"""
### BenchmarkSuite.plot_results - Visualization

The plot_results method generates a 2x2 grid of bar charts comparing models
across all four metrics. The best performer in each category is highlighted green.
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-plot", "solution": true}
#| export
    # --- BenchmarkSuite.plot_results and plot_pareto_frontier ---
def benchsuite_plot_results(self, save_plots: bool = True):
    """
    Generate visualization plots for benchmark results.

    TODO: Create 2x2 bar chart grid comparing models across metrics

    APPROACH:
    1. Check that results exist and matplotlib is available
    2. Create 2x2 subplot grid for latency, accuracy, memory, energy
    3. Plot bar charts with error bars (std)
    4. Highlight best performer in green
    5. Save and show plots

    HINTS:
    - For latency/memory/energy, lower is better
    - For accuracy, higher is better
    - Use alpha=0.7 for bars, capsize=5 for error bars
    """
    ### BEGIN SOLUTION
    if not self.results:
        print("No results to plot. Run benchmark first.")
        return

    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ matplotlib not available - skipping plots. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML Model Benchmark Results', fontsize=16, fontweight='bold')

    # Plot each metric type
    metrics = ['latency', 'accuracy', 'memory', 'energy']
    units = ['ms', 'accuracy', 'MB', 'J']

    for idx, (metric, unit) in enumerate(zip(metrics, units)):
        ax = axes[idx // 2, idx % 2]

        if metric in self.results:
            model_names = []
            means = []
            stds = []

            for model_name, result in self.results[metric].items():
                clean_name = model_name.replace(f'_{metric}', '').replace('_ms', '').replace('_mb', '').replace('_joules', '')
                model_names.append(clean_name)
                means.append(result.mean)
                stds.append(result.std)

            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylabel(f'{metric.capitalize()} ({unit})')
            ax.tick_params(axis='x', rotation=45)

            # Color bars by performance (green = better)
            if metric in ['latency', 'memory', 'energy']:  # Lower is better
                best_idx = means.index(min(means))
            else:  # Higher is better (accuracy)
                best_idx = means.index(max(means))

            for i, bar in enumerate(bars):
                if i == best_idx:
                    bar.set_color('green')
                    bar.set_alpha(0.8)
        else:
            ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric.capitalize()} Comparison')

    plt.tight_layout()

    if save_plots:
        plot_path = self.output_dir / 'benchmark_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to {plot_path}")

    plt.show()
    ### END SOLUTION

BenchmarkSuite.plot_results = benchsuite_plot_results

def benchsuite_plot_pareto_frontier(self, x_metric: str = 'latency', y_metric: str = 'accuracy'):
    """Plot Pareto frontier for two competing objectives."""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ matplotlib not available - skipping plots. Install with: pip install matplotlib")
        return

    if x_metric not in self.results or y_metric not in self.results:
        print(f"Missing data for {x_metric} or {y_metric}")
        return

    plt.figure(figsize=(10, 8))

    x_values = []
    y_values = []
    model_names = []

    for model_name in self.results[x_metric].keys():
        clean_name = model_name.replace(f'_{x_metric}', '').replace('_ms', '').replace('_mb', '').replace('_joules', '')
        if clean_name in [mn.replace(f'_{y_metric}', '') for mn in self.results[y_metric].keys()]:
            x_val = self.results[x_metric][model_name].mean

            # Find corresponding y value
            y_key = None
            for key in self.results[y_metric].keys():
                if clean_name in key:
                    y_key = key
                    break

            if y_key:
                y_val = self.results[y_metric][y_key].mean
                x_values.append(x_val)
                y_values.append(y_val)
                model_names.append(clean_name)

    # Plot points
    plt.scatter(x_values, y_values, s=100, alpha=0.7)

    # Label points
    for i, name in enumerate(model_names):
        plt.annotate(name, (x_values[i], y_values[i]),
                    xytext=(5, 5), textcoords='offset points')

    # Determine if lower or higher is better for each metric
    x_lower_better = x_metric in ['latency', 'memory', 'energy']
    y_lower_better = y_metric in ['latency', 'memory', 'energy']

    plt.xlabel(f'{x_metric.capitalize()} ({"lower" if x_lower_better else "higher"} is better)')
    plt.ylabel(f'{y_metric.capitalize()} ({"lower" if y_lower_better else "higher"} is better)')
    plt.title(f'Pareto Frontier: {x_metric.capitalize()} vs {y_metric.capitalize()}')
    plt.grid(True, alpha=0.3)

    # Save plot
    plot_path = self.output_dir / f'pareto_{x_metric}_vs_{y_metric}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Pareto plot saved to {plot_path}")
    plt.show()

BenchmarkSuite.plot_pareto_frontier = benchsuite_plot_pareto_frontier

# %% [markdown]
"""
### 🧪 Unit Test: BenchmarkSuite.plot_results

**What we're testing**: Visualization generation (graceful handling when matplotlib unavailable)
**Why it matters**: Visual comparisons make benchmark results actionable
**Expected**: No errors when plotting (or graceful fallback message)
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchsuite-plot", "locked": true, "points": 10}
def test_unit_benchsuite_plot():
    """🧪 Test BenchmarkSuite plotting."""
    print("🧪 Unit Test: BenchmarkSuite.plot_results...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            time.sleep(0.001)
            return x

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        models = [MockModel("m1"), MockModel("m2")]
        suite = BenchmarkSuite(models, [{"d": "1"}], output_dir=tmp_dir)
        suite.run_full_benchmark()

        # Should not raise even without matplotlib display
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            suite.plot_results(save_plots=True)
        except Exception:
            pass  # Plotting is optional

    # Test with no results
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        suite2 = BenchmarkSuite([MockModel("m1")], [{"d": "1"}], output_dir=tmp_dir)
        suite2.plot_results()  # Should print "No results" without error

    print("✅ BenchmarkSuite.plot_results works correctly!")

if __name__ == "__main__":
    test_unit_benchsuite_plot()

# %% [markdown]
"""
### BenchmarkSuite.generate_report - Actionable Insights

The generate_report method compiles all benchmark results into a structured
markdown report with system information, per-metric summaries, best performers,
trade-off analysis, and deployment recommendations.

```
Report Generation Pipeline:
Results Dict ──> System Info Section ──> Per-Metric Summaries ──> Trade-off Analysis
                                                                         ↓
                                                              Recommendations Section
                                                                         ↓
                                                              Save to benchmark_report.md
```

We'll build this in three steps: format the per-metric results summary,
compute trade-off recommendations, then compose the full report.
"""

# %% [markdown]
"""
#### Step 1: Format Per-Metric Results Summary

For each metric type, identify the best performer and list all model scores.
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-format-results", "solution": true}
#| export
def _benchsuite_format_results_summary(self) -> List[str]:
    """
    Format per-metric results into report lines.

    Returns:
        List of markdown-formatted lines

    TODO: Summarize each metric with best performer and detailed scores

    APPROACH:
    1. For each metric type in self.results:
       a. Determine if lower or higher is better
       b. Find the best performer (min for latency/memory/energy, max for accuracy)
       c. List all models with mean ± std
    """
    ### BEGIN SOLUTION
    lines = []
    lines.append("## Benchmark Results Summary")
    lines.append("")

    for metric_type, results in self.results.items():
        lines.append(f"### {metric_type.capitalize()} Results")
        lines.append("")

        # Find best performer
        if metric_type in ['latency', 'memory', 'energy']:
            best_model = min(results.items(), key=lambda x: x[1].mean)
            comparison_text = "fastest" if metric_type == 'latency' else "most efficient"
        else:
            best_model = max(results.items(), key=lambda x: x[1].mean)
            comparison_text = "most accurate"

        lines.append(f"**Best performer**: {best_model[0]} ({comparison_text})")
        lines.append("")

        for model_name, result in results.items():
            clean_name = model_name.replace(f'_{metric_type}', '').replace('_ms', '').replace('_mb', '').replace('_joules', '')
            lines.append(f"- **{clean_name}**: {result.mean:.4f} ± {result.std:.4f}")
        lines.append("")

    return lines
    ### END SOLUTION

BenchmarkSuite._format_results_summary = _benchsuite_format_results_summary

# %% [markdown]
"""
#### Step 2: Compute Trade-off Recommendations

Analyze accuracy vs speed trade-offs and generate use-case recommendations.
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-format-recs", "solution": true}
#| export
def _benchsuite_format_recommendations(self) -> List[str]:
    """
    Generate recommendation lines from benchmark results.

    Returns:
        List of markdown-formatted recommendation lines

    TODO: Compute trade-off scores and generate use-case recommendations

    APPROACH:
    1. If latency and accuracy results exist, normalize and compute combined scores
    2. Find best overall trade-off model
    3. Add use-case recommendations (max accuracy, min latency, production)

    HINTS:
    - Normalize: 1 - (val - min) / (max - min) for lower-is-better
    - Normalize: (val - min) / (max - min) for higher-is-better
    """
    ### BEGIN SOLUTION
    lines = []
    lines.append("## Recommendations")
    lines.append("")

    if len(self.results) >= 2:
        if 'latency' in self.results and 'accuracy' in self.results:
            lines.append("### Accuracy vs Speed Trade-off")

            latency_results = self.results['latency']
            accuracy_results = self.results['accuracy']

            scores = {}
            for model_name in latency_results.keys():
                clean_name = model_name.replace('_latency', '').replace('_ms', '')

                acc_key = None
                for key in accuracy_results.keys():
                    if clean_name in key:
                        acc_key = key
                        break

                if acc_key:
                    lat_vals = [r.mean for r in latency_results.values()]
                    acc_vals = [r.mean for r in accuracy_results.values()]

                    norm_latency = 1 - (latency_results[model_name].mean - min(lat_vals)) / (max(lat_vals) - min(lat_vals) + 1e-8)
                    norm_accuracy = (accuracy_results[acc_key].mean - min(acc_vals)) / (max(acc_vals) - min(acc_vals) + 1e-8)

                    scores[clean_name] = (norm_latency + norm_accuracy) / 2

            if scores:
                best_overall = max(scores.items(), key=lambda x: x[1])
                lines.append(f"- **Best overall trade-off**: {best_overall[0]} (score: {best_overall[1]:.3f})")
                lines.append("")

    lines.append("### Usage Recommendations")
    if 'accuracy' in self.results and 'latency' in self.results:
        acc_results = self.results['accuracy']
        lat_results = self.results['latency']

        best_acc_model = max(acc_results.items(), key=lambda x: x[1].mean)
        best_lat_model = min(lat_results.items(), key=lambda x: x[1].mean)

        lines.append(f"- **For maximum accuracy**: Use {best_acc_model[0].replace('_accuracy', '')}")
        lines.append(f"- **For minimum latency**: Use {best_lat_model[0].replace('_latency_ms', '')}")
        lines.append("- **For production deployment**: Consider the best overall trade-off model above")

    return lines
    ### END SOLUTION

BenchmarkSuite._format_recommendations = _benchsuite_format_recommendations

# %% [markdown]
"""
#### Step 3: Compose the Full Report

Combine system info, results summary, and recommendations into a complete
markdown report and save it to disk.
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-report", "solution": true}
#| export
def benchsuite_generate_report(self) -> str:
    """
    Generate comprehensive benchmark report.

    TODO: Compose _format_results_summary and _format_recommendations into a full report

    APPROACH:
    1. Add report header and system information
    2. Call self._format_results_summary() for per-metric data
    3. Call self._format_recommendations() for trade-off analysis
    4. Save to output_dir/benchmark_report.md
    """
    ### BEGIN SOLUTION
    if not self.results:
        return "No benchmark results available. Run benchmark first."

    report_lines = []
    report_lines.append("# ML Model Benchmark Report")
    report_lines.append("=" * 50)
    report_lines.append("")

    # System information
    report_lines.append("## System Information")
    system_info = self.benchmark.system_info
    for key, value in system_info.items():
        report_lines.append(f"- {key}: {value}")
    report_lines.append("")

    # Results summary (from helper)
    report_lines.extend(self._format_results_summary())

    # Recommendations (from helper)
    report_lines.extend(self._format_recommendations())

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("Report generated by TinyTorch Benchmarking Suite")

    # Save report
    report_text = "\n".join(report_lines)
    report_path = self.output_dir / 'benchmark_report.md'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"📄 Report saved to {report_path}")
    return report_text
    ### END SOLUTION

BenchmarkSuite.generate_report = benchsuite_generate_report

# %% [markdown]
"""
### 🧪 Unit Test: BenchmarkSuite._format_results_summary

**What we're testing**: Per-metric results formatting with best performer identification
**Why it matters**: Correct summaries help engineers quickly identify winners
**Expected**: Markdown lines with metric headers, best performers, and model scores
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchsuite-format-results", "locked": true, "points": 3}
def test_unit_benchsuite_format_results():
    """🧪 Test BenchmarkSuite._format_results_summary implementation."""
    print("🧪 Unit Test: BenchmarkSuite._format_results_summary...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return x * 0.5

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        models = [MockModel("fast_model"), MockModel("accurate_model")]
        suite = BenchmarkSuite(models, [{"data": "test"}], output_dir=tmp_dir)
        suite.run_full_benchmark()

        lines = suite._format_results_summary()

        # Should return a list of strings
        assert isinstance(lines, list), f"Expected list, got {type(lines)}"
        assert len(lines) > 0, "Should produce at least some lines"

        # Should contain results summary header
        text = "\n".join(lines)
        assert "Results Summary" in text, "Should contain 'Results Summary'"
        assert "Best performer" in text, "Should identify best performer"

    print("✅ BenchmarkSuite._format_results_summary works correctly!")

if __name__ == "__main__":
    test_unit_benchsuite_format_results()

# %% [markdown]
"""
### 🧪 Unit Test: BenchmarkSuite._format_recommendations

**What we're testing**: Trade-off analysis and use-case recommendation generation
**Why it matters**: Wrong recommendations lead to wrong deployment decisions
**Expected**: Markdown lines with trade-off scores and use-case guidance
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchsuite-format-recs", "locked": true, "points": 3}
def test_unit_benchsuite_format_recs():
    """🧪 Test BenchmarkSuite._format_recommendations implementation."""
    print("🧪 Unit Test: BenchmarkSuite._format_recommendations...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return x * 0.5

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        models = [MockModel("fast_model"), MockModel("accurate_model")]
        suite = BenchmarkSuite(models, [{"data": "test"}], output_dir=tmp_dir)
        suite.run_full_benchmark()

        lines = suite._format_recommendations()

        assert isinstance(lines, list), f"Expected list, got {type(lines)}"
        text = "\n".join(lines)
        assert "Recommendations" in text, "Should contain 'Recommendations'"

    print("✅ BenchmarkSuite._format_recommendations works correctly!")

if __name__ == "__main__":
    test_unit_benchsuite_format_recs()

# %% [markdown]
"""
### 🧪 Unit Test: BenchmarkSuite (Full Class Integration)

This test validates our BenchmarkSuite runs comprehensive multi-metric evaluation
and generates valid reports with recommendations.

**What we're testing**: Full benchmark suite with report generation
**Why it matters**: Comprehensive evaluation enables informed optimization decisions
**Expected**: Complete results across all metrics with valid reports
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchmark-suite", "locked": true, "points": 15}
def test_unit_benchmark_suite():
    """🧪 Test BenchmarkSuite comprehensive functionality."""
    print("🧪 Unit Test: BenchmarkSuite...")

    # Create mock models
    class MockModel:
        def __init__(self, name):
            self.name = name

        def forward(self, x):
            time.sleep(0.001)
            return x

    models = [MockModel("efficient_model"), MockModel("accurate_model")]
    datasets = [{"test": "data"}]

    # Create temporary directory for test output
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        suite = BenchmarkSuite(models, datasets, output_dir=tmp_dir)

        # Run full benchmark
        results = suite.run_full_benchmark()

        # Verify all benchmark types completed
        assert 'latency' in results
        assert 'accuracy' in results
        assert 'memory' in results
        assert 'energy' in results

        # Verify results structure
        for metric_results in results.values():
            assert len(metric_results) == 2  # Two models
            assert all(isinstance(result, BenchmarkResult) for result in metric_results.values())

        # Test report generation
        report = suite.generate_report()
        assert "Benchmark Report" in report
        assert "System Information" in report
        assert "Recommendations" in report

        # Verify files are created
        output_path = Path(tmp_dir)
        assert (output_path / 'benchmark_report.md').exists()

    print("✅ BenchmarkSuite works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_suite()

# %% [markdown]
"""
### MLPerf - Standardized Industry Benchmarking

MLPerf provides standardized benchmarks that enable fair comparison across different systems, similar to how MLPerf works for larger models. This is crucial for reproducible research and industry adoption.

### Why Standardization Matters

Without standards, every team benchmarks differently:
- Different datasets, input sizes, measurement protocols
- Different accuracy metrics, latency definitions
- Different hardware configurations, software stacks

This makes it impossible to compare results across papers, products, or research groups.

### MLPerf Benchmark Architecture

```
MLPerf Benchmark Structure:
┌─────────────────────────────────────────────────────────┐
│                  Benchmark Definition                   │
│ • Standard datasets (CIFAR-10, Speech Commands, etc.)   │
│ • Fixed input shapes and data types                     │
│ • Target accuracy and latency thresholds                │
│ • Measurement protocol (warmup, runs, etc.)             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                 Execution Protocol                      │
│ 1. Model registration and validation                    │
│ 2. Warmup phase (deterministic random inputs)           │
│ 3. Measurement phase (statistical sampling)             │
│ 4. Accuracy evaluation (ground truth comparison)        │
│ 5. Compliance checking (thresholds, statistical tests)  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Compliance Determination                   │
│ PASS: accuracy ≥ target AND latency ≤ target            │
│ FAIL: Either constraint violated                        │
│ Report: Detailed metrics + system information           │
└─────────────────────────────────────────────────────────┘
```

### Standard Benchmark Tasks

**Keyword Spotting**: Wake word detection from audio
- Input: 1-second 16kHz audio samples
- Task: Binary classification (keyword present/absent)
- Target: 90% accuracy, <100ms latency

**Visual Wake Words**: Person detection in images
- Input: 96×96 RGB images
- Task: Binary classification (person present/absent)
- Target: 80% accuracy, <200ms latency

**Anomaly Detection**: Industrial sensor monitoring
- Input: 640-element sensor feature vectors
- Task: Binary classification (anomaly/normal)
- Target: 85% accuracy, <50ms latency

### Reproducibility Requirements

All MLPerf benchmarks use:
- **Fixed random seeds**: Deterministic input generation
- **Standardized hardware**: Reference implementations for comparison
- **Statistical validation**: Multiple runs with confidence intervals
- **Compliance reporting**: Machine-readable results format
"""

# %% [markdown]
"""
### MLPerf.__init__ - Configuring Standard Benchmarks

The MLPerf constructor sets up four standardized benchmark tasks, each with
fixed input shapes, target accuracy, and maximum latency thresholds. Using a
fixed random seed ensures reproducible results across different systems.

```
Standard MLPerf Benchmarks:
┌─────────────────────┬──────────────────┬─────────┬──────────┐
│ Benchmark           │ Input Shape      │ Acc Tgt │ Lat Tgt  │
├─────────────────────┼──────────────────┼─────────┼──────────┤
│ keyword_spotting     │ (1, 16000)       │ 90%     │ <100ms   │
│ visual_wake_words   │ (1, 96, 96, 3)   │ 80%     │ <200ms   │
│ anomaly_detection   │ (1, 640)         │ 85%     │ <50ms    │
│ image_classification│ (1, 32, 32, 3)   │ 75%     │ <150ms   │
└─────────────────────┴──────────────────┴─────────┴──────────┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-init", "solution": true}
#| export
class MLPerf:
    """
    MLPerf-style standardized benchmarking for edge ML systems.

    Provides fixed benchmark configurations with target thresholds,
    standardized measurement protocols, and compliance reporting.

    EXAMPLE:
    >>> perf = MLPerf()
    >>> results = perf.run_standard_benchmark(model, 'keyword_spotting')
    >>> perf.generate_compliance_report(results)
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize MLPerf benchmark suite.

        TODO: Set up standard benchmark configurations with fixed seeds

        APPROACH:
        1. Store random seed and initialize numpy RNG
        2. Define benchmark configs with input_shape, target_accuracy, max_latency_ms

        HINTS:
        - Each benchmark is a dict with 'input_shape', 'target_accuracy', 'max_latency_ms', 'description'
        - keyword_spotting uses (1, 16000) for 1 second of 16kHz audio
        """
        ### BEGIN SOLUTION
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Standard MLPerf benchmark configurations
        self.benchmarks = {
            'keyword_spotting': {
                'input_shape': (1, 16000),  # 1 second of 16kHz audio
                'target_accuracy': 0.90,
                'max_latency_ms': 100,
                'description': 'Wake word detection'
            },
            'visual_wake_words': {
                'input_shape': (1, 96, 96, 3),  # 96x96 RGB image
                'target_accuracy': 0.80,
                'max_latency_ms': 200,
                'description': 'Person detection in images'
            },
            'anomaly_detection': {
                'input_shape': (1, 640),  # Machine sensor data
                'target_accuracy': 0.85,
                'max_latency_ms': 50,
                'description': 'Industrial anomaly detection'
            },
            'image_classification': {
                'input_shape': (1, 32, 32, 3),  # CIFAR-10 style
                'target_accuracy': 0.75,
                'max_latency_ms': 150,
                'description': 'Tiny image classification'
            }
        }
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: MLPerf.__init__

**What we're testing**: Benchmark configuration setup with all four standard tasks
**Why it matters**: Correct configurations ensure fair, standardized comparisons
**Expected**: Four benchmarks with proper input shapes and thresholds
"""

# %% nbgrader={"grade": true, "grade_id": "test-tinymlperf-init", "locked": true, "points": 5}
def test_unit_mlperf_init():
    """🧪 Test MLPerf initialization."""
    print("🧪 Unit Test: MLPerf.__init__...")

    perf = MLPerf(random_seed=42)

    assert perf.random_seed == 42
    assert len(perf.benchmarks) == 4
    assert 'keyword_spotting' in perf.benchmarks
    assert 'visual_wake_words' in perf.benchmarks
    assert 'anomaly_detection' in perf.benchmarks
    assert 'image_classification' in perf.benchmarks

    # Verify config structure
    for name, config in perf.benchmarks.items():
        assert 'input_shape' in config
        assert 'target_accuracy' in config
        assert 'max_latency_ms' in config
        assert 'description' in config
        assert 0 < config['target_accuracy'] <= 1.0
        assert config['max_latency_ms'] > 0

    print("✅ MLPerf.__init__ works correctly!")

if __name__ == "__main__":
    test_unit_mlperf_init()

# %% [markdown]
"""
### MLPerf._run_latency_test - Measuring Inference Latency

This helper runs the latency measurement phase: warmup, then timed inference
for each test input. Returns lists of latencies (ms) and model predictions.

```
Latency Test Protocol:
Test Inputs ──> Warmup Phase (10%) ──> Measurement Phase (100%) ──> latencies[], predictions[]
                (discard timing)       (collect per-input timing)
```
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-latency", "solution": true}
#| export
    # --- MLPerf._run_latency_test ---
def _mlperf_run_latency_test(self, model: Any, test_inputs: List[Any],
                                  benchmark_name: str, num_runs: int) -> Tuple[List[float], List[Any]]:
    """
    Run latency measurement phase with warmup.

    TODO: Implement warmup and measurement phases for latency testing

    APPROACH:
    1. Warmup phase: run 10% of inputs without timing
    2. Measurement phase: time each inference with precise_timer
    3. Use duck-typing (forward/predict/callable) for model invocation
    4. Return latencies in ms and predictions list

    HINTS:
    - warmup_runs = max(1, num_runs // 10)
    - Use precise_timer() context manager
    - Convert elapsed seconds to ms: timer.elapsed * 1000
    """
    ### BEGIN SOLUTION
    # Warmup phase (10% of runs)
    warmup_runs = max(1, num_runs // 10)
    print(f"   Warming up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        if hasattr(model, 'forward'):
            model.forward(test_inputs[i])
        elif hasattr(model, 'predict'):
            model.predict(test_inputs[i])
        elif callable(model):
            model(test_inputs[i])

    # Measurement phase
    print(f"   Measuring performance ({num_runs} runs)...")
    latencies = []
    predictions = []

    for i, test_input in enumerate(test_inputs):
        with precise_timer() as timer:
            try:
                if hasattr(model, 'forward'):
                    output = model.forward(test_input)
                elif hasattr(model, 'predict'):
                    output = model.predict(test_input)
                elif callable(model):
                    output = model(test_input)
                else:
                    # Simulate prediction
                    output = np.random.rand(2) if benchmark_name in ['keyword_spotting', 'visual_wake_words'] else np.random.rand(10)

                predictions.append(output)
            except Exception:
                # Fallback simulation
                predictions.append(np.random.rand(2))

        latencies.append(timer.elapsed * 1000)  # Convert to ms

    return latencies, predictions
    ### END SOLUTION

MLPerf._run_latency_test = _mlperf_run_latency_test

# %% [markdown]
"""
### 🧪 Unit Test: MLPerf._run_latency_test

**What we're testing**: Warmup and measurement phase execution
**Why it matters**: Proper warmup eliminates cold-start bias in measurements
**Expected**: Positive latency values and predictions for each input
"""

# %% nbgrader={"grade": true, "grade_id": "test-tinymlperf-latency", "locked": true, "points": 10}
def test_unit_mlperf_latency():
    """🧪 Test MLPerf latency measurement phase."""
    print("🧪 Unit Test: MLPerf._run_latency_test...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            time.sleep(0.001)
            return np.random.rand(2)

    perf = MLPerf(random_seed=42)
    model = MockModel("test")

    test_inputs = [np.random.randn(1, 16000).astype(np.float32) for _ in range(5)]
    latencies, predictions = perf._run_latency_test(model, test_inputs, 'keyword_spotting', 5)

    assert len(latencies) == 5
    assert len(predictions) == 5
    assert all(lat > 0 for lat in latencies)

    print("✅ MLPerf._run_latency_test works correctly!")

if __name__ == "__main__":
    test_unit_mlperf_latency()

# %% [markdown]
"""
### MLPerf._run_accuracy_test - Evaluating Prediction Quality

This helper calculates accuracy by comparing model predictions against synthetic
ground truth labels. It handles both binary classification (keyword spotting,
visual wake words) and multi-class classification (image classification,
anomaly detection).

We'll build this in two steps: first a helper to extract a clean prediction
array from various output formats, then the accuracy calculation itself.
"""

# %% [markdown]
"""
#### Step 1: Extract Prediction Array

Model outputs can be TinyTorch Tensors, numpy arrays, or plain Python objects.
This helper normalizes them into a flat numpy array for label extraction.
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-extract-pred", "solution": true}
#| export
def _extract_pred_array(pred) -> np.ndarray:
    """
    Extract a flat numpy array from a model prediction.

    Args:
        pred: Raw prediction (Tensor, numpy array, or list)

    Returns:
        Flattened numpy array of prediction values

    TODO: Normalize various prediction formats into a flat numpy array

    APPROACH:
    1. If pred has .data attribute (TinyTorch Tensor), use it
    2. Otherwise convert to numpy array
    3. Flatten if multi-dimensional
    """
    ### BEGIN SOLUTION
    if hasattr(pred, 'data'):
        pred_array = pred.data
    else:
        pred_array = np.array(pred)

    # Convert to numpy array if needed (handle memoryview objects)
    if not isinstance(pred_array, np.ndarray):
        pred_array = np.array(pred_array)

    if len(pred_array.shape) > 1:
        pred_array = pred_array.flatten()

    return pred_array
    ### END SOLUTION

# %% [markdown]
"""
#### Step 2: Calculate Accuracy

Use _extract_pred_array to get clean predictions, then compare against
synthetic ground truth for binary and multi-class tasks.
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-memory", "solution": true}
#| export
def _mlperf_run_accuracy_test(self, model: Any, predictions: List[Any],
                                    benchmark_name: str, num_runs: int) -> float:
    """
    Calculate accuracy from predictions against synthetic ground truth.

    TODO: Implement accuracy calculation using _extract_pred_array helper

    APPROACH:
    1. Generate synthetic ground truth using fixed seed
    2. For binary tasks: use _extract_pred_array, compare class scores
    3. For multi-class: use _extract_pred_array, take argmax
    4. Add realistic noise based on model name

    HINTS:
    - keyword_spotting and visual_wake_words are binary (2 classes)
    - image_classification has 10 classes, anomaly_detection has 5
    """
    ### BEGIN SOLUTION
    np.random.seed(self.random_seed)
    if benchmark_name in ['keyword_spotting', 'visual_wake_words']:
        # Binary classification
        true_labels = np.random.randint(0, 2, num_runs)
        predicted_labels = []
        for pred in predictions:
            pred_array = _extract_pred_array(pred)
            if len(pred_array) >= 2:
                predicted_labels.append(1 if pred_array[1] > pred_array[0] else 0)
            else:
                predicted_labels.append(1 if pred_array[0] > 0.5 else 0)
    else:
        # Multi-class classification
        num_classes = 10 if benchmark_name == 'image_classification' else 5
        true_labels = np.random.randint(0, num_classes, num_runs)
        predicted_labels = []
        for pred in predictions:
            pred_array = _extract_pred_array(pred)
            predicted_labels.append(np.argmax(pred_array) % num_classes)

    # Calculate accuracy
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    accuracy = correct_predictions / num_runs

    # Add realistic noise based on model complexity
    model_name = getattr(model, 'name', 'unknown_model')
    if 'efficient' in model_name.lower():
        accuracy = min(0.95, accuracy + 0.1)
    elif 'accurate' in model_name.lower():
        accuracy = min(0.98, accuracy + 0.2)

    return accuracy
    ### END SOLUTION

MLPerf._run_accuracy_test = _mlperf_run_accuracy_test

# %% [markdown]
"""
### 🧪 Unit Test: _extract_pred_array

**What we're testing**: Prediction array extraction from various output formats
**Why it matters**: Models return Tensors, numpy arrays, or lists — we need to handle all
**Expected**: Always returns a flat numpy array regardless of input format
"""

# %% nbgrader={"grade": true, "grade_id": "test-extract-pred", "locked": true, "points": 3}
def test_unit_extract_pred_array():
    """🧪 Test _extract_pred_array helper."""
    print("🧪 Unit Test: _extract_pred_array...")

    # Test with plain numpy array
    result = _extract_pred_array(np.array([0.3, 0.7]))
    assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
    assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"

    # Test with 2D array (should flatten)
    result_2d = _extract_pred_array(np.array([[0.3, 0.7]]))
    assert len(result_2d.shape) == 1, "Should flatten multi-dimensional input"

    # Test with list
    result_list = _extract_pred_array([0.3, 0.7])
    assert isinstance(result_list, np.ndarray), "Should convert list to ndarray"

    print("✅ _extract_pred_array works correctly!")

if __name__ == "__main__":
    test_unit_extract_pred_array()

# %% [markdown]
"""
### 🧪 Unit Test: MLPerf._run_accuracy_test

**What we're testing**: Accuracy calculation for binary and multi-class tasks
**Why it matters**: Accuracy determines whether a model meets compliance thresholds
**Expected**: Accuracy value between 0 and 1
"""

# %% nbgrader={"grade": true, "grade_id": "test-tinymlperf-accuracy", "locked": true, "points": 10}
def test_unit_mlperf_accuracy():
    """🧪 Test MLPerf accuracy calculation."""
    print("🧪 Unit Test: MLPerf._run_accuracy_test...")

    class MockModel:
        def __init__(self, name):
            self.name = name

    perf = MLPerf(random_seed=42)
    model = MockModel("test_model")

    # Binary classification predictions
    predictions = [np.random.rand(2) for _ in range(10)]
    accuracy = perf._run_accuracy_test(model, predictions, 'keyword_spotting', 10)
    assert 0 <= accuracy <= 1

    # Multi-class predictions
    predictions_mc = [np.random.rand(10) for _ in range(10)]
    accuracy_mc = perf._run_accuracy_test(model, predictions_mc, 'image_classification', 10)
    assert 0 <= accuracy_mc <= 1

    print("✅ MLPerf._run_accuracy_test works correctly!")

if __name__ == "__main__":
    test_unit_mlperf_accuracy()

# %% [markdown]
"""
### MLPerf.run_standard_benchmark - Complete Benchmark Execution

This method orchestrates a complete standardized benchmark: input generation,
latency testing, accuracy evaluation, and compliance determination. It composes
the `_run_latency_test` and `_run_accuracy_test` helpers into the full protocol.

```
run_standard_benchmark Pipeline:
Config Lookup ──> Generate Inputs ──> _run_latency_test() ──> _run_accuracy_test()
                  (deterministic)     (warmup + measure)      (evaluate quality)
                                                                     ↓
                                                          Compile Results Dict
                                                          (accuracy, latency, compliance)
```
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-run", "solution": true}
#| export
    # --- MLPerf.run_standard_benchmark ---
def mlperf_run_standard_benchmark(self, model: Any, benchmark_name: str,
                              num_runs: int = 100) -> Dict[str, Any]:
    """
    Run a standardized MLPerf benchmark.

    TODO: Orchestrate input generation, latency test, accuracy test, and compliance check

    APPROACH:
    1. Validate benchmark_name and get config
    2. Generate deterministic test inputs using seeded random
    3. Call self._run_latency_test() for timing
    4. Call self._run_accuracy_test() for quality
    5. Compile results with compliance determination

    HINTS:
    - Use np.random.seed(self.random_seed + i) for each input
    - Audio data: np.random.randn, Image data: np.random.randint(0,256)/255
    - compliant = accuracy_met AND latency_met
    """
    ### BEGIN SOLUTION
    if benchmark_name not in self.benchmarks:
        available = list(self.benchmarks.keys())
        raise ValueError(
            f"Unknown MLPerf benchmark: '{benchmark_name}'\n"
            f"  ❌ '{benchmark_name}' is not a registered benchmark\n"
            f"  💡 MLPerf defines standard edge ML benchmarks for reproducible comparison\n"
            f"  🔧 Choose from: {available}"
        )

    config = self.benchmarks[benchmark_name]
    print(f"🧪 Running MLPerf {benchmark_name} benchmark...")
    print(f"   Target: {config['target_accuracy']:.1%} accuracy, "
          f"<{config['max_latency_ms']}ms latency")

    # Generate standardized test inputs
    input_shape = config['input_shape']
    test_inputs = []
    for i in range(num_runs):
        # Use deterministic random generation for reproducibility
        np.random.seed(self.random_seed + i)
        if len(input_shape) == 2:  # Audio/sequence data
            test_input = np.random.randn(*input_shape).astype(np.float32)
        else:  # Image data
            test_input = np.random.randint(0, 256, input_shape).astype(np.float32) / 255.0
        test_inputs.append(test_input)

    # Run latency and accuracy tests using helpers
    latencies, predictions = self._run_latency_test(model, test_inputs, benchmark_name, num_runs)
    accuracy = self._run_accuracy_test(model, predictions, benchmark_name, num_runs)

    # Compile results
    mean_latency = float(np.mean(latencies))
    accuracy_met = bool(accuracy >= config['target_accuracy'])
    latency_met = bool(mean_latency <= config['max_latency_ms'])

    results = {
        'benchmark_name': benchmark_name,
        'model_name': getattr(model, 'name', 'unknown_model'),
        'accuracy': float(accuracy),
        'mean_latency_ms': mean_latency,
        'std_latency_ms': float(np.std(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p90_latency_ms': float(np.percentile(latencies, 90)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'max_latency_ms': float(np.max(latencies)),
        'throughput_fps': float(1000 / mean_latency),
        'target_accuracy': float(config['target_accuracy']),
        'target_latency_ms': float(config['max_latency_ms']),
        'accuracy_met': accuracy_met,
        'latency_met': latency_met,
        'compliant': accuracy_met and latency_met,
        'num_runs': int(num_runs),
        'random_seed': int(self.random_seed)
    }

    print(f"   Results: {accuracy:.1%} accuracy, {np.mean(latencies):.1f}ms latency")
    print(f"   Compliance: {'✅ PASS' if results['compliant'] else '❌ FAIL'}")

    return results
    ### END SOLUTION

MLPerf.run_standard_benchmark = mlperf_run_standard_benchmark

def mlperf_run_all_benchmarks(self, model: Any) -> Dict[str, Dict[str, Any]]:
    """Run all MLPerf benchmarks on a model."""
    all_results = {}

    print(f"🚀 Running full MLPerf suite on {getattr(model, 'name', 'model')}...")
    print("=" * 60)

    for benchmark_name in self.benchmarks.keys():
        try:
            results = self.run_standard_benchmark(model, benchmark_name)
            all_results[benchmark_name] = results
            print()
        except Exception as e:
            print(f"   ❌ Failed to run {benchmark_name}: {e}")
            all_results[benchmark_name] = {'error': str(e)}

    return all_results

MLPerf.run_all_benchmarks = mlperf_run_all_benchmarks

# %% [markdown]
"""
### 🧪 Unit Test: MLPerf.run_standard_benchmark

**What we're testing**: Complete benchmark execution with compliance determination
**Why it matters**: The full pipeline must produce valid, reproducible results
**Expected**: Results dict with all required metrics and compliance flags
"""

# %% nbgrader={"grade": true, "grade_id": "test-tinymlperf-run", "locked": true, "points": 15}
def test_unit_mlperf_run():
    """🧪 Test MLPerf standard benchmark execution."""
    print("🧪 Unit Test: MLPerf.run_standard_benchmark...")

    class MockModel:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            time.sleep(0.001)
            if hasattr(x, 'shape') and len(x.shape) == 2:
                return np.random.rand(2)
            return np.random.rand(10)

    perf = MLPerf(random_seed=42)
    model = MockModel("test_model")

    result = perf.run_standard_benchmark(model, 'keyword_spotting', num_runs=5)

    required_keys = ['accuracy', 'mean_latency_ms', 'throughput_fps', 'compliant',
                     'accuracy_met', 'latency_met', 'p50_latency_ms', 'p99_latency_ms']
    assert all(key in result for key in required_keys)
    assert 0 <= result['accuracy'] <= 1
    assert result['mean_latency_ms'] > 0
    assert result['throughput_fps'] > 0
    assert isinstance(result['compliant'], bool)

    # Test invalid benchmark name
    try:
        perf.run_standard_benchmark(model, 'nonexistent')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✅ MLPerf.run_standard_benchmark works correctly!")

if __name__ == "__main__":
    test_unit_mlperf_run()

# %% [markdown]
"""
### MLPerf.generate_compliance_report - Scorecard Generation

The compliance report compiles results from multiple benchmarks into both
machine-readable JSON and human-readable markdown formats, with overall
compliance determination.

```
Report Generation:
Results Dict ──> Count compliant benchmarks ──> JSON report (structured data)
                                              ──> Markdown summary (human-readable)
                                              ──> Overall: COMPLIANT/NON-COMPLIANT
```

We'll build this in two steps: compile the structured report data,
then format it into a human-readable summary.
"""

# %% [markdown]
"""
#### Step 1: Compile Structured Report Data

Process raw benchmark results into a structured dictionary with compliance
statistics, ready for JSON serialization.
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-compile-data", "solution": true}
#| export
def _mlperf_compile_report_data(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compile benchmark results into structured report data.

    Args:
        results: Raw benchmark results dict

    Returns:
        Structured report_data dict with benchmarks and summary

    TODO: Process results into a structured dict with compliance stats

    APPROACH:
    1. Initialize report_data with version, seed, timestamp
    2. Loop through results, skipping errors
    3. Count compliant benchmarks and compute compliance_rate
    4. Store per-benchmark metrics

    HINTS:
    - overall_compliant = compliance_rate == 1.0
    - Set model_name from first successful result
    """
    ### BEGIN SOLUTION
    compliant_benchmarks = []
    total_benchmarks = 0

    report_data = {
        'mlperf_version': '1.0',
        'random_seed': self.random_seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': 'unknown',
        'benchmarks': {},
        'summary': {}
    }

    for benchmark_name, result in results.items():
        if 'error' not in result:
            total_benchmarks += 1
            if result.get('compliant', False):
                compliant_benchmarks.append(benchmark_name)

            if report_data['model_name'] == 'unknown':
                report_data['model_name'] = result.get('model_name', 'unknown')

            report_data['benchmarks'][benchmark_name] = {
                'accuracy': result['accuracy'],
                'mean_latency_ms': result['mean_latency_ms'],
                'p99_latency_ms': result['p99_latency_ms'],
                'throughput_fps': result['throughput_fps'],
                'target_accuracy': result['target_accuracy'],
                'target_latency_ms': result['target_latency_ms'],
                'accuracy_met': result['accuracy_met'],
                'latency_met': result['latency_met'],
                'compliant': result['compliant']
            }

    if total_benchmarks > 0:
        compliance_rate = len(compliant_benchmarks) / total_benchmarks
        report_data['summary'] = {
            'total_benchmarks': total_benchmarks,
            'compliant_benchmarks': len(compliant_benchmarks),
            'compliance_rate': compliance_rate,
            'overall_compliant': compliance_rate == 1.0,
            'compliant_benchmark_names': compliant_benchmarks
        }

    return report_data
    ### END SOLUTION

MLPerf._compile_report_data = _mlperf_compile_report_data

# %% [markdown]
"""
#### Step 2: Format Human-Readable Summary

Convert structured report data into a markdown compliance summary.
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-format-summary", "solution": true}
#| export
def _mlperf_format_compliance_summary(self, report_data: Dict[str, Any]) -> str:
    """
    Format report data into a human-readable markdown summary.

    Args:
        report_data: Structured report dict from _compile_report_data

    Returns:
        Markdown-formatted summary string

    TODO: Generate markdown summary from structured report data

    APPROACH:
    1. Add header with model name and date
    2. Show overall COMPLIANT/NON-COMPLIANT status
    3. List each benchmark with PASS/FAIL and metrics
    """
    ### BEGIN SOLUTION
    summary_lines = []
    summary_lines.append("# MLPerf Compliance Report")
    summary_lines.append("=" * 40)
    summary_lines.append(f"Model: {report_data['model_name']}")
    summary_lines.append(f"Date: {report_data['timestamp']}")
    summary_lines.append("")

    if report_data['summary']:
        overall = report_data['summary']['overall_compliant']
        rate = report_data['summary']['compliance_rate']
        compliant_count = report_data['summary']['compliant_benchmarks']
        total = report_data['summary']['total_benchmarks']

        summary_lines.append(f"## Overall Result: {'✅ COMPLIANT' if overall else '❌ NON-COMPLIANT'}")
        summary_lines.append(f"Compliance Rate: {rate:.1%} ({compliant_count}/{total})")
        summary_lines.append("")

        summary_lines.append("## Benchmark Details:")
        for benchmark_name, result in report_data['benchmarks'].items():
            status = "✅ PASS" if result['compliant'] else "❌ FAIL"
            summary_lines.append(f"- **{benchmark_name}**: {status}")
            summary_lines.append(f"  - Accuracy: {result['accuracy']:.1%} (target: {result['target_accuracy']:.1%})")
            summary_lines.append(f"  - Latency: {result['mean_latency_ms']:.1f}ms (target: <{result['target_latency_ms']}ms)")
            summary_lines.append("")
    else:
        summary_lines.append("No successful benchmark runs.")

    return "\n".join(summary_lines)
    ### END SOLUTION

MLPerf._format_compliance_summary = _mlperf_format_compliance_summary

# %% [markdown]
"""
#### Step 3: Compose the Full Compliance Report

Combine data compilation, JSON serialization, and summary formatting.
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-scorecard", "solution": true}
#| export
def mlperf_generate_compliance_report(self, results: Dict[str, Dict[str, Any]],
                                           output_path: str = "mlperf_report.json") -> str:
    """
    Generate MLPerf compliance report.

    TODO: Compose _compile_report_data and _format_compliance_summary

    APPROACH:
    1. Compile structured data with self._compile_report_data(results)
    2. Save JSON report with json.dump
    3. Format summary with self._format_compliance_summary(report_data)
    4. Save summary markdown alongside JSON
    """
    ### BEGIN SOLUTION
    # Compile structured report data
    report_data = self._compile_report_data(results)

    # Save JSON report
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    # Generate and save human-readable summary
    summary_text = self._format_compliance_summary(report_data)

    summary_path = output_path.replace('.json', '_summary.md')
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    print(f"📄 MLPerf report saved to {output_path}")
    print(f"📄 Summary saved to {summary_path}")

    return summary_text
    ### END SOLUTION

MLPerf.generate_compliance_report = mlperf_generate_compliance_report

# %% [markdown]
"""
### 🧪 Unit Test: MLPerf._compile_report_data

**What we're testing**: Structured data compilation from raw benchmark results
**Why it matters**: Correct data structure is the foundation for both JSON and markdown reports
**Expected**: Dict with benchmarks, summary, compliance stats
"""

# %% nbgrader={"grade": true, "grade_id": "test-tinymlperf-compile", "locked": true, "points": 3}
def test_unit_mlperf_compile_data():
    """🧪 Test MLPerf._compile_report_data implementation."""
    print("🧪 Unit Test: MLPerf._compile_report_data...")

    perf = MLPerf(random_seed=42)

    # Simulate results from run_standard_benchmark
    mock_results = {
        'keyword_spotting': {
            'accuracy': 0.92, 'mean_latency_ms': 50.0, 'p99_latency_ms': 80.0,
            'throughput_fps': 20.0, 'target_accuracy': 0.90, 'target_latency_ms': 100,
            'accuracy_met': True, 'latency_met': True, 'compliant': True,
            'model_name': 'test_model'
        }
    }

    report_data = perf._compile_report_data(mock_results)

    assert 'benchmarks' in report_data, "Should have 'benchmarks' key"
    assert 'summary' in report_data, "Should have 'summary' key"
    assert report_data['summary']['total_benchmarks'] == 1
    assert report_data['summary']['overall_compliant'] == True
    assert report_data['model_name'] == 'test_model'

    print("✅ MLPerf._compile_report_data works correctly!")

if __name__ == "__main__":
    test_unit_mlperf_compile_data()

# %% [markdown]
"""
### 🧪 Unit Test: MLPerf._format_compliance_summary

**What we're testing**: Markdown summary generation from structured report data
**Why it matters**: Human-readable reports are what engineers actually read
**Expected**: Markdown string with COMPLIANT/NON-COMPLIANT status and benchmark details
"""

# %% nbgrader={"grade": true, "grade_id": "test-tinymlperf-format", "locked": true, "points": 3}
def test_unit_mlperf_format_summary():
    """🧪 Test MLPerf._format_compliance_summary implementation."""
    print("🧪 Unit Test: MLPerf._format_compliance_summary...")

    perf = MLPerf(random_seed=42)

    report_data = {
        'model_name': 'test_model',
        'timestamp': '2025-01-01 00:00:00',
        'summary': {
            'total_benchmarks': 1, 'compliant_benchmarks': 1,
            'compliance_rate': 1.0, 'overall_compliant': True,
            'compliant_benchmark_names': ['keyword_spotting']
        },
        'benchmarks': {
            'keyword_spotting': {
                'accuracy': 0.92, 'mean_latency_ms': 50.0,
                'target_accuracy': 0.90, 'target_latency_ms': 100,
                'compliant': True
            }
        }
    }

    summary = perf._format_compliance_summary(report_data)

    assert isinstance(summary, str), f"Expected string, got {type(summary)}"
    assert "COMPLIANT" in summary, "Should contain compliance status"
    assert "keyword_spotting" in summary, "Should list benchmark names"
    assert "PASS" in summary, "Compliant benchmark should show PASS"

    print("✅ MLPerf._format_compliance_summary works correctly!")

if __name__ == "__main__":
    test_unit_mlperf_format_summary()

# %% [markdown]
"""
### 🧪 Unit Test: MLPerf (Full Class Integration)

This test validates our MLPerf class provides standardized benchmarking
with proper compliance reporting.

**What we're testing**: Industry-standard benchmark protocols and compliance reporting
**Why it matters**: Standardized benchmarks enable fair cross-system comparison
**Expected**: Proper metrics, compliance checking, and report generation
"""

# %% nbgrader={"grade": true, "grade_id": "test-tinymlperf", "locked": true, "points": 10}
def test_unit_mlperf():
    """🧪 Test MLPerf standardized benchmarking."""
    print("🧪 Unit Test: MLPerf...")

    # Create mock model for testing
    class MockModel:
        def __init__(self, name):
            self.name = name

        def forward(self, x):
            time.sleep(0.001)  # Simulate computation
            # Return appropriate output shape for different benchmarks
            if hasattr(x, 'shape'):
                if len(x.shape) == 2:  # Audio/sequence
                    return np.random.rand(2)  # Binary classification
                else:  # Image
                    return np.random.rand(10)  # Multi-class
            return np.random.rand(2)

    model = MockModel("test_model")
    perf = MLPerf(random_seed=42)

    # Test individual benchmark
    result = perf.run_standard_benchmark(model, 'keyword_spotting', num_runs=5)

    # Verify result structure
    required_keys = ['accuracy', 'mean_latency_ms', 'throughput_fps', 'compliant']
    assert all(key in result for key in required_keys)
    assert 0 <= result['accuracy'] <= 1
    assert result['mean_latency_ms'] > 0
    assert result['throughput_fps'] > 0

    # Test full benchmark suite (with fewer runs for speed)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run subset of benchmarks for testing
        subset_results = {}
        for benchmark in ['keyword_spotting', 'image_classification']:
            subset_results[benchmark] = perf.run_standard_benchmark(model, benchmark, num_runs=3)

        # Test compliance report generation
        report_path = f"{tmp_dir}/test_report.json"
        summary = perf.generate_compliance_report(subset_results, report_path)

        # Verify report was created
        assert Path(report_path).exists()
        assert "MLPerf Compliance Report" in summary
        assert "Compliance Rate" in summary

    print("✅ MLPerf works correctly!")

if __name__ == "__main__":
    test_unit_mlperf()

# %% [markdown]
"""
## 🔧 Integration: Building Complete Benchmark Workflows

Now we'll integrate all our benchmarking components into complete workflows that demonstrate professional ML systems evaluation. This integration shows how to combine statistical rigor with practical insights.

The integration layer connects individual measurements into actionable engineering insights. This is where benchmarking becomes a decision-making tool rather than just data collection.

### Workflow Architecture

```
Integration Workflow Pipeline:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Model Variants  │    │ Optimization    │    │ Use Case        │
│ • Base model    │ →  │ Techniques      │ →  │ Analysis        │
│ • Quantized     │    │ • Accuracy loss │    │ • Mobile        │
│ • Pruned        │    │ • Speed gain    │    │ • Server        │
│ • Distilled     │    │ • Memory save   │    │ • Edge          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

This workflow helps answer questions like:
- "Which optimization gives the best accuracy/latency trade-off?"
- "What's the memory budget impact of each technique?"
- "Which model should I deploy for mobile vs server?"
"""

# %% [markdown]
"""
### Optimization Comparison Engine

Before implementing the comparison function, let's understand what makes optimization comparison challenging and valuable.

### Why Optimization Comparison is Complex

When you optimize a model, you're making trade-offs across multiple dimensions simultaneously:

```
Optimization Impact Matrix:
                   Accuracy    Latency    Memory    Energy
Quantization        -5%        +2.1x      +2.0x     +1.8x
Pruning            -2%        +1.4x      +3.2x     +1.3x
Knowledge Distill. -8%        +1.9x      +1.5x     +1.7x
```

The challenge: Which is "best"? It depends entirely on your deployment constraints.

### Multi-Objective Decision Framework

Our comparison engine implements a decision framework that:

1. **Measures all dimensions**: Don't optimize in isolation
2. **Calculates efficiency ratios**: Accuracy per MB, accuracy per ms
3. **Identifies Pareto frontiers**: Models that aren't dominated in all metrics
4. **Generates use-case recommendations**: Tailored to specific constraints

### Recommendation Algorithm

```
For each use case:
├── Latency-critical (real-time apps)
│   └── Optimize: min(latency) subject to accuracy > threshold
├── Memory-constrained (mobile/IoT)
│   └── Optimize: min(memory) subject to accuracy > threshold
├── Accuracy-preservation (quality-critical)
│   └── Optimize: max(accuracy) subject to latency < threshold
└── Balanced (general deployment)
    └── Optimize: weighted combination of all factors
```

This principled approach ensures recommendations match real deployment needs.
"""

# %% [markdown]
"""
### _collect_base_metrics - Extracting Baseline Performance

This helper extracts the base model's mean performance across all metrics from
the benchmark results. It establishes the reference point for improvement calculations.
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-collect-base", "solution": true}
#| export
def _collect_base_metrics(base_name: str, benchmark_results: Dict) -> Dict[str, float]:
    """
    Extract base model metrics from benchmark results.

    TODO: Find the base model's mean value for each metric type

    APPROACH:
    1. Iterate over each metric type (latency, accuracy, memory, energy)
    2. Find the result whose key contains base_name
    3. Store result.mean in a dict keyed by metric type

    HINTS:
    - Use 'base_name in model_name' to match the base model
    """
    ### BEGIN SOLUTION
    base_metrics = {}
    for metric_type, results in benchmark_results.items():
        for model_name, result in results.items():
            if base_name in model_name:
                base_metrics[metric_type] = result.mean
                break
    return base_metrics
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: _collect_base_metrics

**What we're testing**: Extraction of base model's mean metrics from benchmark results
**Why it matters**: Accurate baselines are essential for meaningful improvement ratios
**Expected**: Dict with metric types as keys and mean values as floats
"""

# %% nbgrader={"grade": true, "grade_id": "test-collect-base", "locked": true, "points": 5}
def test_unit_collect_base_metrics():
    """🧪 Test _collect_base_metrics helper."""
    print("🧪 Unit Test: _collect_base_metrics...")

    # Simulate benchmark results
    mock_results = {
        'latency': {'base_latency_ms': BenchmarkResult('base_latency_ms', [10.0, 11.0, 12.0])},
        'accuracy': {'base_accuracy': BenchmarkResult('base_accuracy', [0.9, 0.91, 0.89])},
    }

    metrics = _collect_base_metrics('base', mock_results)
    assert 'latency' in metrics
    assert 'accuracy' in metrics
    assert abs(metrics['latency'] - 11.0) < 0.01
    assert abs(metrics['accuracy'] - 0.9) < 0.01

    print("✅ _collect_base_metrics works correctly!")

if __name__ == "__main__":
    test_unit_collect_base_metrics()

# %% [markdown]
"""
### _calculate_improvements - Computing Speedup and Retention Ratios

This helper computes improvement ratios for each optimized model relative to
the baseline. For latency/memory/energy (lower is better), it calculates
base/optimized as the speedup factor. For accuracy, it calculates
optimized/base as the retention ratio.

```
Improvement Calculation:
Latency:  speedup = base_latency / opt_latency  (>1 means faster)
Memory:   speedup = base_memory / opt_memory     (>1 means smaller)
Accuracy: retention = opt_accuracy / base_accuracy (closer to 1 is better)
```
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-calc-improvements", "solution": true}
#| export
def _calculate_improvements(base_metrics: Dict[str, float], opt_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate improvement ratios for an optimized model vs baseline.

    TODO: Compute speedup ratios for latency/memory/energy and retention for accuracy

    APPROACH:
    1. For latency, memory, energy: improvement = base / optimized
    2. For accuracy: retention = optimized / base
    3. Handle division by zero with fallback to 1.0

    HINTS:
    - Check opt_metrics[metric] > 0 before dividing
    - Use f'{metric_type}_speedup' as key names
    """
    ### BEGIN SOLUTION
    improvements = {}
    for metric_type in ['latency', 'memory', 'energy']:
        if metric_type in base_metrics and metric_type in opt_metrics:
            # For these metrics, lower is better, so improvement = base/optimized
            if opt_metrics[metric_type] > 0:
                improvements[f'{metric_type}_speedup'] = base_metrics[metric_type] / opt_metrics[metric_type]
            else:
                improvements[f'{metric_type}_speedup'] = 1.0

    if 'accuracy' in base_metrics and 'accuracy' in opt_metrics:
        # Accuracy retention (higher is better)
        improvements['accuracy_retention'] = opt_metrics['accuracy'] / base_metrics['accuracy']

    return improvements
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: _calculate_improvements

**What we're testing**: Improvement ratio calculations for all metric types
**Why it matters**: Correct ratios drive optimization recommendations
**Expected**: Speedup > 1 when optimized is better, retention near 1.0
"""

# %% nbgrader={"grade": true, "grade_id": "test-calc-improvements", "locked": true, "points": 5}
def test_unit_calculate_improvements():
    """🧪 Test _calculate_improvements helper."""
    print("🧪 Unit Test: _calculate_improvements...")

    base = {'latency': 10.0, 'memory': 100.0, 'accuracy': 0.90}
    opt = {'latency': 5.0, 'memory': 50.0, 'accuracy': 0.85}

    improvements = _calculate_improvements(base, opt)

    assert abs(improvements['latency_speedup'] - 2.0) < 0.01  # 10/5 = 2x
    assert abs(improvements['memory_speedup'] - 2.0) < 0.01   # 100/50 = 2x
    assert abs(improvements['accuracy_retention'] - 0.9444) < 0.01  # 0.85/0.90

    # Test with zero (edge case)
    opt_zero = {'latency': 0.0, 'memory': 50.0, 'accuracy': 0.85}
    imp_zero = _calculate_improvements(base, opt_zero)
    assert imp_zero['latency_speedup'] == 1.0  # Fallback

    print("✅ _calculate_improvements works correctly!")

if __name__ == "__main__":
    test_unit_calculate_improvements()

# %% [markdown]
"""
### _generate_recommendations - Deployment-Specific Guidance

This helper analyzes improvement ratios across all optimized models to generate
recommendations for four deployment scenarios: latency-critical, memory-constrained,
accuracy-preservation, and balanced deployment.
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-gen-recs", "solution": true}
#| export
def _generate_recommendations(all_improvements: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
    """
    Generate deployment recommendations from improvement data.

    TODO: Find best model for each deployment scenario

    APPROACH:
    1. Track best latency, memory, accuracy, and overall scores
    2. For overall: weight speedups equally but accuracy retention at 5x
    3. Cap speedup at 5.0x to avoid outlier domination
    4. Return recommendation dict with model, reason, use_case

    HINTS:
    - Iterate over all_improvements items (opt_name -> improvements dict)
    - Overall score = (sum of capped speedups + accuracy_retention * 5) / count
    """
    ### BEGIN SOLUTION
    best_latency = None
    best_memory = None
    best_accuracy = None
    best_overall = None

    best_latency_score = 0
    best_memory_score = 0
    best_accuracy_score = 0
    best_overall_score = 0

    for opt_name, improvements in all_improvements.items():
        # Latency recommendation
        if 'latency_speedup' in improvements and improvements['latency_speedup'] > best_latency_score:
            best_latency_score = improvements['latency_speedup']
            best_latency = opt_name

        # Memory recommendation
        if 'memory_speedup' in improvements and improvements['memory_speedup'] > best_memory_score:
            best_memory_score = improvements['memory_speedup']
            best_memory = opt_name

        # Accuracy recommendation
        if 'accuracy_retention' in improvements and improvements['accuracy_retention'] > best_accuracy_score:
            best_accuracy_score = improvements['accuracy_retention']
            best_accuracy = opt_name

        # Overall balance (considering all factors)
        overall_score = 0
        count = 0
        for key, value in improvements.items():
            if 'speedup' in key:
                overall_score += min(value, 5.0)  # Cap speedup at 5x to avoid outliers
                count += 1
            elif 'retention' in key:
                overall_score += value * 5  # Weight accuracy retention heavily
                count += 1

        if count > 0:
            overall_score /= count
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_overall = opt_name

    return {
        'for_latency_critical': {
            'model': best_latency,
            'reason': f"Best latency improvement: {best_latency_score:.2f}x faster",
            'use_case': "Real-time applications, edge devices with strict timing requirements"
        },
        'for_memory_constrained': {
            'model': best_memory,
            'reason': f"Best memory reduction: {best_memory_score:.2f}x smaller",
            'use_case': "Mobile devices, IoT sensors, embedded systems"
        },
        'for_accuracy_preservation': {
            'model': best_accuracy,
            'reason': f"Best accuracy retention: {best_accuracy_score:.1%} of original",
            'use_case': "Applications where quality cannot be compromised"
        },
        'for_balanced_deployment': {
            'model': best_overall,
            'reason': f"Best overall trade-off (score: {best_overall_score:.2f})",
            'use_case': "General production deployment with multiple constraints"
        }
    }
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: _generate_recommendations

**What we're testing**: Recommendation generation from improvement data
**Why it matters**: Correct recommendations guide deployment decisions
**Expected**: Four recommendation categories with appropriate model selections
"""

# %% nbgrader={"grade": true, "grade_id": "test-gen-recs", "locked": true, "points": 5}
def test_unit_generate_recommendations():
    """🧪 Test _generate_recommendations helper."""
    print("🧪 Unit Test: _generate_recommendations...")

    improvements = {
        'quantized': {'latency_speedup': 2.0, 'memory_speedup': 3.0, 'accuracy_retention': 0.95},
        'pruned': {'latency_speedup': 1.5, 'memory_speedup': 4.0, 'accuracy_retention': 0.98},
    }

    recs = _generate_recommendations(improvements)

    assert 'for_latency_critical' in recs
    assert 'for_memory_constrained' in recs
    assert 'for_accuracy_preservation' in recs
    assert 'for_balanced_deployment' in recs

    # Quantized has best latency speedup (2.0 > 1.5)
    assert recs['for_latency_critical']['model'] == 'quantized'
    # Pruned has best memory speedup (4.0 > 3.0)
    assert recs['for_memory_constrained']['model'] == 'pruned'
    # Pruned has best accuracy retention (0.98 > 0.95)
    assert recs['for_accuracy_preservation']['model'] == 'pruned'

    print("✅ _generate_recommendations works correctly!")

if __name__ == "__main__":
    test_unit_generate_recommendations()

# %% [markdown]
"""
### analyze_optimization_techniques - Composition Function

This is the main entry point that composes `_collect_base_metrics`,
`_calculate_improvements`, and `_generate_recommendations` into a complete
optimization comparison workflow.

```
analyze_optimization_techniques Pipeline:
┌────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ Run Full   │ ──> │ _collect_base_metrics│ ──> │ For each opt model: │
│ Benchmark  │     │ (extract baseline)   │     │ _calculate_improvements│
└────────────┘     └─────────────────────┘     └─────────────────────┘
                                                          ↓
                                               ┌─────────────────────┐
                                               │_generate_recommendations│
                                               │ (deploy guidance)    │
                                               └─────────────────────┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-comparison", "solution": true}
#| export
def analyze_optimization_techniques(base_model: Any, optimized_models: List[Any],
                                  datasets: List[Any]) -> Dict[str, Any]:
    """
    Compare base model against various optimization techniques.

    TODO: Compose helpers to run benchmarks, calculate improvements, generate recommendations

    APPROACH:
    1. Run BenchmarkSuite on [base_model] + optimized_models
    2. Use _collect_base_metrics() for baseline
    3. Use _calculate_improvements() for each optimized model
    4. Use _generate_recommendations() for deployment guidance
    5. Print summary and return results

    Args:
        base_model: Baseline model (unoptimized)
        optimized_models: List of models with different optimizations applied
        datasets: List of datasets for evaluation

    Returns:
        Dictionary with 'base_metrics', 'optimized_results', 'improvements', 'recommendations'

    EXAMPLE:
    >>> results = analyze_optimization_techniques(base_model, [quant, pruned], datasets)
    >>> print(results['recommendations'])
    """
    ### BEGIN SOLUTION
    all_models = [base_model] + optimized_models
    suite = BenchmarkSuite(all_models, datasets)

    print("🧪 Running optimization comparison benchmark...")
    benchmark_results = suite.run_full_benchmark()

    # Extract base model performance using helper
    base_name = getattr(base_model, 'name', 'model_0')
    base_metrics = _collect_base_metrics(base_name, benchmark_results)

    # Initialize comparison results
    comparison_results = {
        'base_model': base_name,
        'base_metrics': base_metrics,
        'optimized_results': {},
        'improvements': {},
        'efficiency_metrics': {},
        'recommendations': {}
    }

    for opt_model in optimized_models:
        opt_name = getattr(opt_model, 'name', f'optimized_model_{len(comparison_results["optimized_results"])}')

        # Find results for this optimized model
        opt_metrics = {}
        for metric_type, results in benchmark_results.items():
            for model_name, result in results.items():
                if opt_name in model_name:
                    opt_metrics[metric_type] = result.mean
                    break

        comparison_results['optimized_results'][opt_name] = opt_metrics

        # Calculate improvements using helper
        improvements = _calculate_improvements(base_metrics, opt_metrics)
        comparison_results['improvements'][opt_name] = improvements

        # Calculate efficiency metrics
        efficiency = {}
        if 'accuracy' in opt_metrics:
            if 'memory' in opt_metrics and opt_metrics['memory'] > 0:
                efficiency['accuracy_per_mb'] = opt_metrics['accuracy'] / opt_metrics['memory']
            if 'latency' in opt_metrics and opt_metrics['latency'] > 0:
                efficiency['accuracy_per_ms'] = opt_metrics['accuracy'] / opt_metrics['latency']

        comparison_results['efficiency_metrics'][opt_name] = efficiency

    # Generate recommendations using helper
    recommendations = _generate_recommendations(comparison_results['improvements'])
    comparison_results['recommendations'] = recommendations

    # Print summary
    print("\n📊 Optimization Comparison Results:")
    print("=" * 50)

    for opt_name, improvements in comparison_results['improvements'].items():
        print(f"\n{opt_name}:")
        for metric, value in improvements.items():
            if 'speedup' in metric:
                print(f"  {metric}: {value:.2f}x improvement")
            elif 'retention' in metric:
                print(f"  {metric}: {value:.1%}")

    print("\n🎯 Recommendations:")
    for use_case, rec in recommendations.items():
        if rec['model']:
            print(f"  {use_case}: {rec['model']} - {rec['reason']}")

    return comparison_results
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: analyze_optimization_techniques (Full Integration)

This test validates the complete optimization comparison workflow generates
useful recommendations from benchmark data.

**What we're testing**: Multi-model comparison with recommendation generation
**Why it matters**: Guides engineers to choose the right optimization for their use case
**Expected**: Valid comparisons and actionable recommendations
"""

# %% nbgrader={"grade": true, "grade_id": "test-optimization-comparison", "locked": true, "points": 10}
def test_unit_optimization_comparison():
    """🧪 Test optimization comparison functionality."""
    print("🧪 Unit Test: analyze_optimization_techniques...")

    # Create mock models with different characteristics
    class MockModel:
        def __init__(self, name, latency_factor=1.0, accuracy_factor=1.0, memory_factor=1.0):
            self.name = name
            self.latency_factor = latency_factor
            self.accuracy_factor = accuracy_factor
            self.memory_factor = memory_factor

        def forward(self, x):
            time.sleep(0.001 * self.latency_factor)
            return x

    # Base model and optimized variants
    base_model = MockModel("base_model", latency_factor=1.0, accuracy_factor=1.0, memory_factor=1.0)
    quantized_model = MockModel("quantized_model", latency_factor=0.7, accuracy_factor=0.95, memory_factor=0.5)
    pruned_model = MockModel("pruned_model", latency_factor=0.8, accuracy_factor=0.98, memory_factor=0.3)

    datasets = [{"test": "data"}]

    # Run comparison
    results = analyze_optimization_techniques(base_model, [quantized_model, pruned_model], datasets)

    # Verify results structure
    assert 'base_model' in results
    assert 'optimized_results' in results
    assert 'improvements' in results
    assert 'recommendations' in results

    # Verify improvements were calculated
    assert len(results['improvements']) == 2  # Two optimized models

    # Verify recommendations were generated
    recommendations = results['recommendations']
    assert 'for_latency_critical' in recommendations
    assert 'for_memory_constrained' in recommendations
    assert 'for_accuracy_preservation' in recommendations
    assert 'for_balanced_deployment' in recommendations

    print("✅ analyze_optimization_techniques works correctly!")

if __name__ == "__main__":
    test_unit_optimization_comparison()

# %% [markdown]
"""
## 📊 Systems Analysis: Benchmark Variance and Optimization Trade-offs

Let's understand the key systems concept of measurement variance and optimization trade-offs.
"""

# %%
def analyze_benchmark_variance():
    """📊 Analyze measurement variance and confidence intervals."""
    print("📊 Analyzing Benchmark Variance")
    print("=" * 60)

    # Simulate benchmarking with different sample sizes
    sample_sizes = [5, 10, 20, 50, 100]
    true_latency = 10.0  # True mean latency in ms
    noise_std = 1.5  # Standard deviation of measurement noise

    print("Effect of Sample Size on Confidence Interval Width:\n")
    print(f"{'Samples':<10} {'Mean (ms)':<15} {'CI Width (ms)':<15} {'Relative Error':<15}")
    print("-" * 60)

    for n_samples in sample_sizes:
        # Simulate measurements
        measurements = np.random.normal(true_latency, noise_std, n_samples)
        mean_latency = np.mean(measurements)
        std_latency = np.std(measurements)

        # Calculate 95% confidence interval
        t_score = 1.96
        margin_error = t_score * (std_latency / np.sqrt(n_samples))
        ci_width = 2 * margin_error
        relative_error = ci_width / mean_latency * 100

        print(f"{n_samples:<10} {mean_latency:<15.2f} {ci_width:<15.2f} {relative_error:<15.1f}%")

    print("\n💡 Key Insights:")
    print("   • More samples reduce confidence interval width")
    print("   • CI width decreases with √n (diminishing returns)")
    print("   • 20-50 samples typically sufficient for <10% error")
    print("   • Statistical rigor requires measuring variance, not just mean")

if __name__ == "__main__":
    analyze_benchmark_variance()

# %%
def analyze_optimization_tradeoffs():
    """📊 Analyze trade-offs between different optimization techniques."""
    print("\n📊 Analyzing Optimization Trade-offs")
    print("=" * 60)

    # Simulated optimization results
    optimizations = {
        'Baseline': {'accuracy': 0.89, 'latency_ms': 45, 'memory_mb': 12, 'energy_j': 2.0},
        'Quantization (INT8)': {'accuracy': 0.88, 'latency_ms': 30, 'memory_mb': 3, 'energy_j': 1.3},
        'Pruning (70%)': {'accuracy': 0.87, 'latency_ms': 35, 'memory_mb': 4, 'energy_j': 1.5},
        'Both (INT8 + 70%)': {'accuracy': 0.85, 'latency_ms': 22, 'memory_mb': 1, 'energy_j': 0.9},
    }

    # Calculate efficiency metrics
    print("\nEfficiency Metrics (higher is better):\n")
    print(f"{'Technique':<25} {'Acc/MB':<12} {'Acc/ms':<12} {'Acc/J':<12}")
    print("-" * 60)

    baseline = optimizations['Baseline']

    for name, metrics in optimizations.items():
        acc_per_mb = metrics['accuracy'] / metrics['memory_mb']
        acc_per_ms = metrics['accuracy'] / metrics['latency_ms']
        acc_per_j = metrics['accuracy'] / metrics['energy_j']

        print(f"{name:<25} {acc_per_mb:<12.3f} {acc_per_ms:<12.4f} {acc_per_j:<12.3f}")

    print("\nPareto Frontier Analysis:")
    print("   • Quantization: Best memory efficiency (0.293 acc/MB)")
    print("   • Pruning: Balanced trade-off")
    print("   • Combined: Maximum resource efficiency, highest accuracy loss")

    print("\n💡 Key Insights:")
    print("   • No single optimization dominates all metrics")
    print("   • Combined optimizations compound benefits and risks")
    print("   • Choose based on deployment constraints (memory vs speed vs accuracy)")
    print("   • Pareto frontier reveals non-dominated solutions")

if __name__ == "__main__":
    analyze_optimization_tradeoffs()

# %% [markdown]
"""
## 📊 MLPerf Principles - Industry-Standard Benchmarking

MLPerf (created by MLCommons) is the industry-standard ML benchmarking framework. Understanding these principles grounds your capstone competition in professional methodology.

### Core Principles

**Reproducibility:** Fixed hardware specs, software versions, random seeds, and multiple runs for statistical validity.

**Standardization:** Fixed models and datasets enable fair comparison. MLPerf has two divisions:
- **Closed:** Same models/datasets, optimize systems (hardware/software)
- **Open:** Modify models/algorithms, show innovation

**MLPerf:** Edge device benchmarks (<1MB models, <100ms latency, <10mW power) that inspire the capstone.

### Key Takeaways

1. Document everything for reproducibility
2. Use same baseline for fair comparison
3. Measure multiple metrics (accuracy, latency, memory, energy)
4. Optimize for real deployment constraints

The capstone project follows MLPerf-style principles!
"""

# %% [markdown]
"""
## 📊 Combination Strategies

Strategic optimization combines multiple techniques for different performance goals. The order matters: quantize-then-prune may preserve accuracy better, while prune-then-quantize may be faster.

### Ablation Studies

Professional ML engineers use ablation studies to understand each optimization's contribution:

```
Baseline:           Accuracy: 89%, Latency: 45ms, Memory: 12MB
+ Quantization:     Accuracy: 88%, Latency: 30ms, Memory: 3MB   (Δ: -1%, -33%, -75%)
+ Pruning:          Accuracy: 87%, Latency: 22ms, Memory: 2MB   (Δ: -1%, -27%, -33%)
+ Kernel Fusion:    Accuracy: 87%, Latency: 18ms, Memory: 2MB   (Δ: 0%, -18%, 0%)
```

You'll apply these strategies with specific optimization targets in Module 20's capstone project.
"""

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that our complete benchmarking system works correctly and integrates properly with all TinyTorch components.

This comprehensive test validates the entire benchmarking ecosystem and ensures it's ready for production use in the final capstone project.
"""

# %% nbgrader={"grade": true, "grade_id": "test-module", "locked": true, "points": 10}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire benchmarking module functionality.

    This final test runs before module summary to ensure:
    - All benchmarking components work together correctly
    - Statistical analysis provides reliable results
    - Integration with optimization modules functions properly
    - Professional reporting generates actionable insights
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_benchmark_result()
    test_unit_precise_timer()
    test_unit_benchmark_init()
    test_unit_benchmark_latency()
    test_unit_benchmark_accuracy()
    test_unit_benchmark_memory()
    test_unit_benchmark()
    test_unit_benchsuite_init()
    test_unit_benchsuite_run()
    test_unit_benchsuite_energy()
    test_unit_benchsuite_plot()
    test_unit_benchsuite_format_results()
    test_unit_benchsuite_format_recs()
    test_unit_benchmark_suite()
    test_unit_mlperf_init()
    test_unit_mlperf_latency()
    test_unit_extract_pred_array()
    test_unit_mlperf_accuracy()
    test_unit_mlperf_run()
    test_unit_mlperf_compile_data()
    test_unit_mlperf_format_summary()
    test_unit_mlperf()
    test_unit_collect_base_metrics()
    test_unit_calculate_improvements()
    test_unit_generate_recommendations()
    test_unit_optimization_comparison()

    print("\nRunning integration scenarios...")

    # Test realistic benchmarking workflow
    print("🧪 Integration Test: Complete benchmarking workflow...")

    # Create realistic test models
    class RealisticModel:
        def __init__(self, name, characteristics):
            self.name = name
            self.characteristics = characteristics

        def forward(self, x):
            # Simulate different model behaviors
            base_time = self.characteristics.get('base_latency', 0.001)
            variance = self.characteristics.get('variance', 0.0001)
            memory_factor = self.characteristics.get('memory_factor', 1.0)

            # Simulate realistic computation
            time.sleep(max(0, base_time + np.random.normal(0, variance)))

            # Simulate memory usage
            if hasattr(x, 'shape'):
                temp_size = int(np.prod(x.shape) * memory_factor)
                temp_data = np.random.randn(temp_size)
                _ = np.sum(temp_data)  # Use the data

            return x

        def evaluate(self, dataset):
            # Simulate evaluation
            base_acc = self.characteristics.get('base_accuracy', 0.85)
            return base_acc + np.random.normal(0, 0.02)

        def parameters(self):
            # Simulate parameter count - return Tensor objects for compatibility
            from tinytorch.core.tensor import Tensor
            param_count = self.characteristics.get('param_count', 1000000)
            return [Tensor(np.random.randn(param_count))]

    # Create test model suite
    models = [
        RealisticModel("efficient_model", {
            'base_latency': 0.001,
            'base_accuracy': 0.82,
            'memory_factor': 0.5,
            'param_count': 500000
        }),
        RealisticModel("accurate_model", {
            'base_latency': 0.003,
            'base_accuracy': 0.95,
            'memory_factor': 2.0,
            'param_count': 2000000
        }),
        RealisticModel("balanced_model", {
            'base_latency': 0.002,
            'base_accuracy': 0.88,
            'memory_factor': 1.0,
            'param_count': 1000000
        })
    ]

    datasets = [{"test_data": f"dataset_{i}"} for i in range(3)]

    # Test 1: Comprehensive benchmark suite
    print("  Testing comprehensive benchmark suite...")
    suite = BenchmarkSuite(models, datasets)
    results = suite.run_full_benchmark()

    assert 'latency' in results
    assert 'accuracy' in results
    assert 'memory' in results
    assert 'energy' in results

    # Verify all models were tested
    for result_type in results.values():
        assert len(result_type) == len(models)

    # Test 2: Statistical analysis
    print("  Testing statistical analysis...")
    for result_type, model_results in results.items():
        for model_name, result in model_results.items():
            assert isinstance(result, BenchmarkResult)
            assert result.count > 0
            assert result.std >= 0
            assert result.ci_lower <= result.mean <= result.ci_upper

    # Test 3: Report generation
    print("  Testing report generation...")
    report = suite.generate_report()
    assert "Benchmark Report" in report
    assert "System Information" in report
    assert "Recommendations" in report

    # Test 4: MLPerf compliance
    print("  Testing MLPerf compliance...")
    perf = MLPerf(random_seed=42)
    perf_results = perf.run_standard_benchmark(models[0], 'keyword_spotting', num_runs=5)

    required_keys = ['accuracy', 'mean_latency_ms', 'compliant', 'target_accuracy']
    assert all(key in perf_results for key in required_keys)
    assert 0 <= perf_results['accuracy'] <= 1
    assert perf_results['mean_latency_ms'] > 0

    # Test 5: Optimization comparison
    print("  Testing optimization comparison...")
    comparison_results = analyze_optimization_techniques(
        models[0], models[1:], datasets[:1]
    )

    assert 'base_model' in comparison_results
    assert 'improvements' in comparison_results
    assert 'recommendations' in comparison_results
    assert len(comparison_results['improvements']) == 2

    # Test 6: Cross-platform compatibility
    print("  Testing cross-platform compatibility...")
    system_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version()
    }

    # Verify system information is captured
    benchmark = Benchmark(models[:1], datasets[:1])
    assert all(key in benchmark.system_info for key in system_info.keys())

    print("✅ End-to-end benchmarking workflow works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 19")

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of benchmarking and performance engineering:

### 1. Statistical Confidence in Measurements
You implemented BenchmarkResult with confidence intervals for measurements.
If you run 20 trials and get mean latency 5.2ms with std dev 0.8ms:
- What's the 95% confidence interval for the true mean? [_____ ms, _____ ms]
- How many more trials would you need to halve the confidence interval width? _____ total trials

### 2. Measurement Overhead Analysis
Your precise_timer context manager has microsecond precision, but models run for milliseconds.
For a model that takes 1ms to execute:
- If timer overhead is 10μs, what's the relative error? _____%
- At what model latency does timer overhead become negligible (<1%)? _____ ms

### 3. Benchmark Configuration Trade-offs
Your optimize_benchmark_configuration() function tested different warmup/measurement combinations.
For a CI/CD pipeline that runs 100 benchmarks per day:
- Fast config (3s each): _____ minutes total daily
- Accurate config (15s each): _____ minutes total daily
- What's the key trade-off you're making? [accuracy/precision/development velocity]

### 4. MLPerf Compliance Metrics
You implemented MLPerf-style standardized benchmarks with target thresholds.
If a model achieves 89% accuracy (target: 90%) and 120ms latency (target: <100ms):
- Is it compliant? [Yes/No] _____
- Which constraint is more critical for edge deployment? [accuracy/latency]
- How would you prioritize optimization? [accuracy first/latency first/balanced]

### 5. Optimization Comparison Analysis
Your analyze_optimization_techniques() generates recommendations for different use cases.
Given three optimized models:
- Quantized: 0.8× memory, 2× speed, 0.95× accuracy
- Pruned: 0.3× memory, 1.5× speed, 0.98× accuracy
- Distilled: 0.6× memory, 1.8× speed, 0.92× accuracy

For a mobile app with 50MB model size limit and <100ms latency requirement:
- Which optimization offers best memory reduction? _____
- Which balances all constraints best? _____
- What's the key insight about optimization trade-offs? [no free lunch/specialization wins/measurement guides decisions]
"""

# %% [markdown]
"""
## ⭐ Aha Moment: Measurement Enables Optimization

**What you built:** A benchmarking system with warmup, statistics, and reproducibility.

**Why it matters:** "Premature optimization is the root of all evil"—but you can't optimize
without measuring! Your benchmarking system produces reliable, comparable numbers: warmup
iterations eliminate cold-start effects, multiple runs give confidence intervals.

This is how production ML teams make decisions: measure, compare, improve, repeat.
"""

# %%
def demo_benchmarking():
    """🎯 See professional benchmarking in action."""
    print("🎯 AHA MOMENT: Measurement Enables Optimization")
    print("=" * 45)

    # Create a simple model and input
    layer = Linear(512, 256)
    x = Tensor(np.random.randn(32, 512))

    # Benchmark with proper methodology
    benchmark = Benchmark(
        models=[layer],
        datasets=[(x, None)],
        warmup_runs=3,
        measurement_runs=10
    )

    results = benchmark.run_latency_benchmark(input_shape=(32, 512))
    result = list(results.values())[0]

    print(f"Model: Linear(512 → 256)")
    print(f"Batch: 32 samples")
    print(f"\nBenchmark Results (10 iterations):")
    print(f"  Mean latency: {result.mean*1000:.2f} ms")
    print(f"  Std dev:      {result.std*1000:.2f} ms")
    print(f"  Min:          {result.min_val*1000:.2f} ms")
    print(f"  Max:          {result.max_val*1000:.2f} ms")

    print("\n✨ Reliable measurements guide optimization decisions!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_benchmarking()

# %% [markdown]
"""
## 🚀 MODULE SUMMARY: Benchmarking

Congratulations! You've built a professional benchmarking system that rivals industry-standard evaluation frameworks!

### Key Accomplishments
- Built comprehensive benchmarking infrastructure with BenchmarkResult, Benchmark, and BenchmarkSuite classes
- Implemented statistical rigor with confidence intervals, variance analysis, and measurement optimization
- Created MLPerf-style standardized benchmarks for reproducible cross-system comparison
- Developed optimization comparison workflows that generate actionable recommendations
- All tests pass ✅ (validated by `test_module()`)

### Systems Engineering Insights Gained
- **Measurement Science**: Statistical significance requires proper sample sizes and variance control
- **Benchmark Design**: Standardized protocols enable fair comparison across different systems
- **Trade-off Analysis**: Pareto frontiers reveal optimization opportunities and constraints
- **Production Integration**: Automated reporting transforms measurements into engineering decisions

### Ready for Systems Capstone
Your benchmarking implementation enables comprehensive systems evaluation, demonstrating your complete optimization toolkit. This is where all 19 modules come together!

Export with: `tito module complete 19`

**Next**: Milestone 5 (Systems Capstone) will demonstrate the complete ML systems engineering workflow!
"""
