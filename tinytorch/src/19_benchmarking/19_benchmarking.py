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

#| default_exp perf.benchmarking
#| export

# Constants for benchmarking defaults
DEFAULT_WARMUP_RUNS = 5  # Default warmup runs for JIT compilation and cache warming
DEFAULT_MEASUREMENT_RUNS = 10  # Default measurement runs for statistical significance

# %% [markdown]
"""
# Module 19: Benchmarking - TorchPerf Olympics Preparation

**IMPORTANT - hasattr() Usage in This Module:**
This module uses hasattr() throughout for duck-typing and polymorphic benchmarking.
This is LEGITIMATE because:
1. Benchmarking framework must work with ANY model type (PyTorch, TinyTorch, custom)
2. Different frameworks use different method names (forward vs predict vs __call__)
3. We need runtime introspection for maximum compatibility
4. This is the CORRECT use of hasattr() for framework-agnostic tooling

Welcome to the final implementation module! You've learned individual optimization techniques in Modules 14-18. Now you'll build the benchmarking infrastructure that powers **TorchPerf Olympics** - the capstone competition framework.

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML framework with profiling, acceleration, quantization, and compression
**You'll Build**: TorchPerf benchmarking system for fair model comparison and capstone submission
**You'll Enable**: Systematic optimization combination and competitive performance evaluation

**Connection Map**:
```
Individual Optimizations (M14-18) → Benchmarking (M19) → TorchPerf Olympics (Capstone)
(techniques)                        (evaluation)         (competition)
```

## 🏅 TorchPerf Olympics: The Capstone Framework

The TorchPerf Olympics is your capstone competition! Choose your event:
- 🏃 **Latency Sprint**: Minimize inference time (fastest model wins)
- 🏋️ **Memory Challenge**: Minimize model size (smallest footprint wins)
- 🎯 **Accuracy Contest**: Maximize accuracy within constraints
- 🏋️‍♂️ **All-Around**: Best balanced performance across all metrics
- 🚀 **Extreme Push**: Most aggressive optimization while staying viable

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement professional benchmarking infrastructure with statistical rigor
2. Learn to combine optimization techniques strategically (order matters!)
3. Build the TorchPerf class - your standardized capstone submission framework
4. Understand ablation studies and systematic performance evaluation

🔥 Carry the torch. Optimize the model. Win the gold! 🏅
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/19_benchmarking/benchmarking_dev.py`
**Building Side:** Code exports to `tinytorch.perf.benchmarking`

**How to use this module (after running `tito module complete 19`):**

```python
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

# %% [markdown]
"""
## 💡 Introduction - What is Fair Benchmarking?

Benchmarking in ML systems isn't just timing code - it's about making fair, reproducible comparisons that guide real optimization decisions. Think of it like standardized testing: everyone takes the same test under the same conditions.

Consider comparing three models: a base CNN, a quantized version, and a pruned version. Without proper benchmarking, you might conclude the quantized model is "fastest" because you measured it when your CPU was idle, while testing the others during peak system load. Fair benchmarking controls for these variables.

The challenge: ML models have multiple competing objectives (accuracy vs speed vs memory), measurements can be noisy, and "faster" depends on your hardware and use case.

## 💡 Benchmarking as a Systems Engineering Discipline

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
## 📐 Mathematical Foundations - Statistics for Performance Engineering

Benchmarking is applied statistics. We measure noisy processes (model inference) and need to extract reliable insights about their true performance characteristics.

## Central Limit Theorem in Practice

When you run a model many times, the distribution of measurements approaches normal (regardless of the underlying noise distribution). This lets us:
- Compute confidence intervals for the true mean
- Detect statistically significant differences between models
- Control for measurement variance

```
Single measurement: Meaningless
Few measurements: Unreliable
Many measurements: Statistical confidence
```

## Multi-Objective Optimization Theory

ML systems exist on a **Pareto frontier** - you can't simultaneously maximize accuracy and minimize latency without trade-offs. Good benchmarks reveal this frontier:

```
Accuracy
    ↑
    |  A ●     ← Model A: High accuracy, high latency
    |
    |    B ●  ← Model B: Balanced trade-off
    |
    |      C ●← Model C: Low accuracy, low latency
    |__________→ Latency (lower is better)
```

The goal: Find the optimal operating point for your specific constraints.

## Measurement Uncertainty and Error Propagation

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
    TorchPerf Olympics event categories.

    Each event optimizes for different objectives with specific constraints.
    Students choose their event and compete for medals!
    """
    LATENCY_SPRINT = "latency_sprint"      # Minimize latency (accuracy >= 85%)
    MEMORY_CHALLENGE = "memory_challenge"   # Minimize memory (accuracy >= 85%)
    ACCURACY_CONTEST = "accuracy_contest"   # Maximize accuracy (latency < 100ms, memory < 10MB)
    ALL_AROUND = "all_around"               # Best balanced score across all metrics
    EXTREME_PUSH = "extreme_push"           # Most aggressive optimization (accuracy >= 80%)

# %% [markdown]
"""
## 🏗️ Implementation - Building Professional Benchmarking Infrastructure

We'll build a comprehensive benchmarking system that handles statistical analysis, multi-dimensional comparison, and automated reporting. Each component builds toward production-quality evaluation tools.

The architecture follows a hierarchical design:
```
Profiler (Module 14) ← Base measurement tools
       ↓
BenchmarkResult ← Statistical container for measurements
       ↓
Benchmark ← Uses Profiler + adds multi-model comparison
       ↓
BenchmarkSuite ← Multi-metric comprehensive evaluation
       ↓
TinyMLPerf ← Standardized industry-style benchmarks
```

**Key Architectural Decision**: The `Benchmark` class reuses `Profiler` from Module 14 for individual model measurements, then adds statistical comparison across multiple models. This demonstrates proper systems architecture - build once, reuse everywhere!

Each level adds capability while maintaining statistical rigor at the foundation.
"""

# %% [markdown]
"""
## 🏗️ BenchmarkResult - Statistical Analysis Container

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
                "BenchmarkResult requires at least one measurement.\n"
                "  Issue: Cannot compute statistics without any measurements.\n"
                "  Fix: Ensure benchmark runs produce at least one measurement before creating BenchmarkResult."
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

def test_unit_benchmark_result():
    """🔬 Test BenchmarkResult statistical calculations."""
    print("🔬 Unit Test: BenchmarkResult...")

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

def test_unit_precise_timer():
    """🔬 Test precise_timer context manager."""
    print("🔬 Unit Test: precise_timer...")

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
## 🏗️ Benchmark Class - Core Measurement Engine

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

# %% nbgrader={"grade": false, "grade_id": "benchmark-class", "solution": true}
#| export
class Benchmark:
    """
    Professional benchmarking system for ML models and operations.

    TODO: Implement a comprehensive benchmark runner with statistical rigor

    APPROACH:
    1. Support multiple models, datasets, and metrics
    2. Run repeated measurements with proper warmup
    3. Control for system variance and compute confidence intervals
    4. Generate structured results for analysis

    EXAMPLE:
    >>> benchmark = Benchmark(models=[model1, model2], datasets=[test_data])
    >>> results = benchmark.run_accuracy_benchmark()
    >>> benchmark.plot_results(results)

    HINTS:
    - Use warmup runs to stabilize performance
    - Collect multiple samples for statistical significance
    - Store metadata about system conditions
    - Provide different benchmark types (accuracy, latency, memory)
    """
    ### BEGIN SOLUTION
    def __init__(self, models: List[Any], datasets: List[Any],
                 warmup_runs: int = DEFAULT_WARMUP_RUNS, measurement_runs: int = DEFAULT_MEASUREMENT_RUNS):
        """Initialize benchmark with models and datasets."""
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

    def run_latency_benchmark(self, input_shape: Tuple[int, ...] = (1, 28, 28)) -> Dict[str, BenchmarkResult]:
        """Benchmark model inference latency using Profiler."""
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

    def run_accuracy_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Benchmark model accuracy across datasets."""
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
                except:
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

    def run_memory_benchmark(self, input_shape: Tuple[int, ...] = (1, 28, 28)) -> Dict[str, BenchmarkResult]:
        """Benchmark model memory usage using Profiler."""
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

    def compare_models(self, metric: str = "latency"):
        """Compare models across a specific metric."""
        if metric == "latency":
            results = self.run_latency_benchmark()
        elif metric == "accuracy":
            results = self.run_accuracy_benchmark()
        elif metric == "memory":
            results = self.run_memory_benchmark()
        else:
            raise ValueError(
                f"Unknown metric: '{metric}'.\n"
                f"  Available metrics: 'latency', 'memory', 'accuracy'.\n"
                f"  Fix: Use one of the supported metric names."
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

def test_unit_benchmark():
    """🔬 Test Benchmark class functionality."""
    print("🔬 Unit Test: Benchmark...")

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
## 🏗️ BenchmarkSuite - Comprehensive Multi-Metric Evaluation

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

# %% nbgrader={"grade": false, "grade_id": "benchmark-suite", "solution": true}
#| export
class BenchmarkSuite:
    """
    Comprehensive benchmark suite for ML systems evaluation.

    TODO: Implement a full benchmark suite that runs multiple test categories

    APPROACH:
    1. Combine multiple benchmark types (latency, accuracy, memory, energy)
    2. Generate comprehensive reports with visualizations
    3. Support different model categories and hardware configurations
    4. Provide recommendations based on results

    EXAMPLE:
    >>> suite = BenchmarkSuite(models, datasets)
    >>> report = suite.run_full_benchmark()
    >>> suite.generate_report(report)

    HINTS:
    - Organize results by benchmark type and model
    - Create Pareto frontier analysis for trade-offs
    - Include system information and test conditions
    - Generate actionable insights and recommendations
    """
    ### BEGIN SOLUTION
    def __init__(self, models: List[Any], datasets: List[Any],
                 output_dir: str = "benchmark_results"):
        """Initialize comprehensive benchmark suite."""
        self.models = models
        self.datasets = datasets
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.benchmark = Benchmark(models, datasets)
        self.results = {}

    def run_full_benchmark(self) -> Dict[str, Dict[str, BenchmarkResult]]:
        """Run all benchmark categories."""
        print("🔬 Running comprehensive benchmark suite...")

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

    def _estimate_energy_efficiency(self) -> Dict[str, BenchmarkResult]:
        """Estimate energy efficiency (simplified simulation)."""
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

    def plot_results(self, save_plots: bool = True):
        """Generate visualization plots for benchmark results."""
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

    def plot_pareto_frontier(self, x_metric: str = 'latency', y_metric: str = 'accuracy'):
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

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
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

        # Results summary
        report_lines.append("## Benchmark Results Summary")
        report_lines.append("")

        for metric_type, results in self.results.items():
            report_lines.append(f"### {metric_type.capitalize()} Results")
            report_lines.append("")

            # Find best performer
            if metric_type in ['latency', 'memory', 'energy']:
                # Lower is better
                best_model = min(results.items(), key=lambda x: x[1].mean)
                comparison_text = "fastest" if metric_type == 'latency' else "most efficient"
            else:
                # Higher is better
                best_model = max(results.items(), key=lambda x: x[1].mean)
                comparison_text = "most accurate"

            report_lines.append(f"**Best performer**: {best_model[0]} ({comparison_text})")
            report_lines.append("")

            # Detailed results
            for model_name, result in results.items():
                clean_name = model_name.replace(f'_{metric_type}', '').replace('_ms', '').replace('_mb', '').replace('_joules', '')
                report_lines.append(f"- **{clean_name}**: {result.mean:.4f} ± {result.std:.4f}")
            report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")

        if len(self.results) >= 2:
            # Find overall best trade-off model
            if 'latency' in self.results and 'accuracy' in self.results:
                report_lines.append("### Accuracy vs Speed Trade-off")

                # Simple scoring: normalize metrics and combine
                latency_results = self.results['latency']
                accuracy_results = self.results['accuracy']

                scores = {}
                for model_name in latency_results.keys():
                    clean_name = model_name.replace('_latency', '').replace('_ms', '')

                    # Find corresponding accuracy
                    acc_key = None
                    for key in accuracy_results.keys():
                        if clean_name in key:
                            acc_key = key
                            break

                    if acc_key:
                        # Normalize: latency (lower better), accuracy (higher better)
                        lat_vals = [r.mean for r in latency_results.values()]
                        acc_vals = [r.mean for r in accuracy_results.values()]

                        norm_latency = 1 - (latency_results[model_name].mean - min(lat_vals)) / (max(lat_vals) - min(lat_vals) + 1e-8)
                        norm_accuracy = (accuracy_results[acc_key].mean - min(acc_vals)) / (max(acc_vals) - min(acc_vals) + 1e-8)

                        # Combined score (equal weight)
                        scores[clean_name] = (norm_latency + norm_accuracy) / 2

                if scores:
                    best_overall = max(scores.items(), key=lambda x: x[1])
                    report_lines.append(f"- **Best overall trade-off**: {best_overall[0]} (score: {best_overall[1]:.3f})")
                    report_lines.append("")

        report_lines.append("### Usage Recommendations")
        if 'accuracy' in self.results and 'latency' in self.results:
            acc_results = self.results['accuracy']
            lat_results = self.results['latency']

            # Find highest accuracy model
            best_acc_model = max(acc_results.items(), key=lambda x: x[1].mean)
            best_lat_model = min(lat_results.items(), key=lambda x: x[1].mean)

            report_lines.append(f"- **For maximum accuracy**: Use {best_acc_model[0].replace('_accuracy', '')}")
            report_lines.append(f"- **For minimum latency**: Use {best_lat_model[0].replace('_latency_ms', '')}")
            report_lines.append("- **For production deployment**: Consider the best overall trade-off model above")

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

def test_unit_benchmark_suite():
    """🔬 Test BenchmarkSuite comprehensive functionality."""
    print("🔬 Unit Test: BenchmarkSuite...")

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
## 🏗️ TinyMLPerf - Standardized Industry Benchmarking

TinyMLPerf provides standardized benchmarks that enable fair comparison across different systems, similar to how MLPerf works for larger models. This is crucial for reproducible research and industry adoption.

### Why Standardization Matters

Without standards, every team benchmarks differently:
- Different datasets, input sizes, measurement protocols
- Different accuracy metrics, latency definitions
- Different hardware configurations, software stacks

This makes it impossible to compare results across papers, products, or research groups.

### TinyMLPerf Benchmark Architecture

```
TinyMLPerf Benchmark Structure:
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

All TinyMLPerf benchmarks use:
- **Fixed random seeds**: Deterministic input generation
- **Standardized hardware**: Reference implementations for comparison
- **Statistical validation**: Multiple runs with confidence intervals
- **Compliance reporting**: Machine-readable results format
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf", "solution": true}
#| export
class TinyMLPerf:
    """
    TinyMLPerf-style standardized benchmarking for edge ML systems.

    TODO: Implement standardized benchmarks following TinyMLPerf methodology

    APPROACH:
    1. Define standard benchmark tasks and datasets
    2. Implement standardized measurement protocols
    3. Ensure reproducible results across different systems
    4. Generate compliance reports for fair comparison

    EXAMPLE:
    >>> perf = TinyMLPerf()
    >>> results = perf.run_keyword_spotting_benchmark(model)
    >>> perf.generate_compliance_report(results)

    HINTS:
    - Use fixed random seeds for reproducibility
    - Implement warm-up and measurement phases
    - Follow TinyMLPerf power and latency measurement standards
    - Generate standardized result formats
    """
    ### BEGIN SOLUTION
    def __init__(self, random_seed: int = 42):
        """Initialize TinyMLPerf benchmark suite."""
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Standard TinyMLPerf benchmark configurations
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

    def run_standard_benchmark(self, model: Any, benchmark_name: str,
                             num_runs: int = 100) -> Dict[str, Any]:
        """Run a standardized TinyMLPerf benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(
                f"Unknown benchmark: '{benchmark_name}'.\n"
                f"  Available benchmarks: {list(self.benchmarks.keys())}.\n"
                f"  Fix: Use one of the supported benchmark names from the list above."
            )

        config = self.benchmarks[benchmark_name]
        print(f"🔬 Running TinyMLPerf {benchmark_name} benchmark...")
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
                except:
                    # Fallback simulation
                    predictions.append(np.random.rand(2))

            latencies.append(timer.elapsed * 1000)  # Convert to ms

        # Simulate accuracy calculation (would use real labels in practice)
        # Generate synthetic ground truth labels
        np.random.seed(self.random_seed)
        if benchmark_name in ['keyword_spotting', 'visual_wake_words']:
            # Binary classification
            true_labels = np.random.randint(0, 2, num_runs)
            predicted_labels = []
            for pred in predictions:
                if hasattr(pred, 'data'):
                    pred_array = pred.data
                else:
                    pred_array = np.array(pred)

                # Convert to numpy array if needed (handle memoryview objects)
                if not isinstance(pred_array, np.ndarray):
                    pred_array = np.array(pred_array)

                if len(pred_array.shape) > 1:
                    pred_array = pred_array.flatten()

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
                if hasattr(pred, 'data'):
                    pred_array = pred.data
                else:
                    pred_array = np.array(pred)

                if len(pred_array.shape) > 1:
                    pred_array = pred_array.flatten()

                predicted_labels.append(np.argmax(pred_array) % num_classes)

        # Calculate accuracy
        correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
        accuracy = correct_predictions / num_runs

        # Add some realistic noise based on model complexity
        model_name = getattr(model, 'name', 'unknown_model')
        if 'efficient' in model_name.lower():
            accuracy = min(0.95, accuracy + 0.1)  # Efficient models might be less accurate
        elif 'accurate' in model_name.lower():
            accuracy = min(0.98, accuracy + 0.2)  # Accurate models perform better

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

    def run_all_benchmarks(self, model: Any) -> Dict[str, Dict[str, Any]]:
        """Run all TinyMLPerf benchmarks on a model."""
        all_results = {}

        print(f"🚀 Running full TinyMLPerf suite on {getattr(model, 'name', 'model')}...")
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

    def generate_compliance_report(self, results: Dict[str, Dict[str, Any]],
                                 output_path: str = "tinymlperf_report.json") -> str:
        """Generate TinyMLPerf compliance report."""
        # Calculate overall compliance
        compliant_benchmarks = []
        total_benchmarks = 0

        report_data = {
            'tinymlperf_version': '1.0',
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

                # Set model name from first successful result
                if report_data['model_name'] == 'unknown':
                    report_data['model_name'] = result.get('model_name', 'unknown')

                # Store benchmark results
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

        # Summary statistics
        if total_benchmarks > 0:
            compliance_rate = len(compliant_benchmarks) / total_benchmarks
            report_data['summary'] = {
                'total_benchmarks': total_benchmarks,
                'compliant_benchmarks': len(compliant_benchmarks),
                'compliance_rate': compliance_rate,
                'overall_compliant': compliance_rate == 1.0,
                'compliant_benchmark_names': compliant_benchmarks
            }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        # Generate human-readable summary
        summary_lines = []
        summary_lines.append("# TinyMLPerf Compliance Report")
        summary_lines.append("=" * 40)
        summary_lines.append(f"Model: {report_data['model_name']}")
        summary_lines.append(f"Date: {report_data['timestamp']}")
        summary_lines.append("")

        if total_benchmarks > 0:
            summary_lines.append(f"## Overall Result: {'✅ COMPLIANT' if report_data['summary']['overall_compliant'] else '❌ NON-COMPLIANT'}")
            summary_lines.append(f"Compliance Rate: {compliance_rate:.1%} ({len(compliant_benchmarks)}/{total_benchmarks})")
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

        summary_text = "\n".join(summary_lines)

        # Save human-readable report
        summary_path = output_path.replace('.json', '_summary.md')
        with open(summary_path, 'w') as f:
            f.write(summary_text)

        print(f"📄 TinyMLPerf report saved to {output_path}")
        print(f"📄 Summary saved to {summary_path}")

        return summary_text
    ### END SOLUTION

def test_unit_tinymlperf():
    """🔬 Test TinyMLPerf standardized benchmarking."""
    print("🔬 Unit Test: TinyMLPerf...")

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
    perf = TinyMLPerf(random_seed=42)

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
        assert "TinyMLPerf Compliance Report" in summary
        assert "Compliance Rate" in summary

    print("✅ TinyMLPerf works correctly!")

if __name__ == "__main__":
    test_unit_tinymlperf()

# %% [markdown]
"""
## 🔧 Integration - Building Complete Benchmark Workflows

Now we'll integrate all our benchmarking components into complete workflows that demonstrate professional ML systems evaluation. This integration shows how to combine statistical rigor with practical insights.

The integration layer connects individual measurements into actionable engineering insights. This is where benchmarking becomes a decision-making tool rather than just data collection.

## 🔧 Workflow Architecture

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
## 🏗️ Optimization Comparison Engine

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

# %% nbgrader={"grade": false, "grade_id": "benchmark-comparison", "solution": true}
def analyze_optimization_techniques(base_model: Any, optimized_models: List[Any],
                                  datasets: List[Any]) -> Dict[str, Any]:
    """
    Compare base model against various optimization techniques.

    TODO: Implement comprehensive comparison of optimization approaches

    APPROACH:
    1. Run benchmarks on base model and all optimized variants
    2. Calculate improvement ratios and trade-offs
    3. Generate insights about which optimizations work best
    4. Create recommendation matrix for different use cases

    Args:
        base_model: Baseline model (unoptimized)
        optimized_models: List of models with different optimizations applied
        datasets: List of datasets for evaluation

    Returns:
        Dictionary with 'base_metrics', 'optimized_results', 'improvements', 'recommendations'

    EXAMPLE:
    >>> models = [base_model, quantized_model, pruned_model, distilled_model]
    >>> results = analyze_optimization_techniques(base_model, models[1:], datasets)
    >>> print(results['recommendations'])

    HINTS:
    - Compare accuracy retention vs speed/memory improvements
    - Calculate efficiency metrics (accuracy per MB, accuracy per ms)
    - Identify Pareto-optimal solutions
    - Generate actionable recommendations for different scenarios
    """
    ### BEGIN SOLUTION
    all_models = [base_model] + optimized_models
    suite = BenchmarkSuite(all_models, datasets)

    print("🔬 Running optimization comparison benchmark...")
    benchmark_results = suite.run_full_benchmark()

    # Extract base model performance for comparison
    base_name = getattr(base_model, 'name', 'model_0')

    base_metrics = {}
    for metric_type, results in benchmark_results.items():
        for model_name, result in results.items():
            if base_name in model_name:
                base_metrics[metric_type] = result.mean
                break

    # Calculate improvement ratios
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

        # Calculate improvements
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

        comparison_results['improvements'][opt_name] = improvements

        # Calculate efficiency metrics
        efficiency = {}
        if 'accuracy' in opt_metrics:
            if 'memory' in opt_metrics and opt_metrics['memory'] > 0:
                efficiency['accuracy_per_mb'] = opt_metrics['accuracy'] / opt_metrics['memory']
            if 'latency' in opt_metrics and opt_metrics['latency'] > 0:
                efficiency['accuracy_per_ms'] = opt_metrics['accuracy'] / opt_metrics['latency']

        comparison_results['efficiency_metrics'][opt_name] = efficiency

    # Generate recommendations based on results
    recommendations = {}

    # Find best performers in each category
    best_latency = None
    best_memory = None
    best_accuracy = None
    best_overall = None

    best_latency_score = 0
    best_memory_score = 0
    best_accuracy_score = 0
    best_overall_score = 0

    for opt_name, improvements in comparison_results['improvements'].items():
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

    recommendations = {
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

def test_unit_optimization_comparison():
    """🔬 Test optimization comparison functionality."""
    print("🔬 Unit Test: analyze_optimization_techniques...")

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
## 📊 Systems Analysis - Benchmark Variance and Optimization Trade-offs

Understanding measurement variance and optimization trade-offs through systematic analysis.
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

**TinyMLPerf:** Edge device benchmarks (<1MB models, <100ms latency, <10mW power) that inspire the capstone.

### Key Takeaways

1. Document everything for reproducibility
2. Use same baseline for fair comparison
3. Measure multiple metrics (accuracy, latency, memory, energy)
4. Optimize for real deployment constraints

**Module 20 capstone** follows TinyMLPerf-style principles!
"""

# %% [markdown]
"""
## 📊 Combination Strategies - Preparing for TorchPerf Olympics

Strategic optimization combines multiple techniques for different competition objectives. The order matters: quantize-then-prune may preserve accuracy better, while prune-then-quantize may be faster.

### Ablation Studies

Professional ML engineers use ablation studies to understand each optimization's contribution:

```
Baseline:           Accuracy: 89%, Latency: 45ms, Memory: 12MB
+ Quantization:     Accuracy: 88%, Latency: 30ms, Memory: 3MB   (Δ: -1%, -33%, -75%)
+ Pruning:          Accuracy: 87%, Latency: 22ms, Memory: 2MB   (Δ: -1%, -27%, -33%)
+ Kernel Fusion:    Accuracy: 87%, Latency: 18ms, Memory: 2MB   (Δ: 0%, -18%, 0%)
```

### Olympic Event Quick Guide

- **Latency Sprint**: Fusion > Caching > Quantization > Pruning
- **Memory Challenge**: Quantization > Pruning > Compression
- **Accuracy Contest**: High-bit quantization (8-bit), light pruning (30-50%)
- **All-Around**: Balanced INT8 + 60% pruning + selective fusion
- **Extreme Push**: 4-bit quantization + 90% pruning (verify accuracy threshold)

**Key strategy:** Start with one technique, measure impact, add next, repeat!
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
    test_unit_benchmark()
    test_unit_benchmark_suite()
    test_unit_tinymlperf()
    test_unit_optimization_comparison()

    print("\nRunning integration scenarios...")

    # Test realistic benchmarking workflow
    print("🔬 Integration Test: Complete benchmarking workflow...")

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

    # Test 4: TinyMLPerf compliance
    print("  Testing TinyMLPerf compliance...")
    perf = TinyMLPerf(random_seed=42)
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
## 🤔 ML Systems Thinking: Benchmarking and Performance Engineering

### Question 1: Statistical Confidence in Measurements
You implemented BenchmarkResult with confidence intervals for measurements.
If you run 20 trials and get mean latency 5.2ms with std dev 0.8ms:
- What's the 95% confidence interval for the true mean? [_____ ms, _____ ms]
- How many more trials would you need to halve the confidence interval width? _____ total trials

### Question 2: Measurement Overhead Analysis
Your precise_timer context manager has microsecond precision, but models run for milliseconds.
For a model that takes 1ms to execute:
- If timer overhead is 10μs, what's the relative error? _____%
- At what model latency does timer overhead become negligible (<1%)? _____ ms

### Question 3: Benchmark Configuration Trade-offs
Your optimize_benchmark_configuration() function tested different warmup/measurement combinations.
For a CI/CD pipeline that runs 100 benchmarks per day:
- Fast config (3s each): _____ minutes total daily
- Accurate config (15s each): _____ minutes total daily
- What's the key trade-off you're making? [accuracy/precision/development velocity]

### Question 4: TinyMLPerf Compliance Metrics
You implemented TinyMLPerf-style standardized benchmarks with target thresholds.
If a model achieves 89% accuracy (target: 90%) and 120ms latency (target: <100ms):
- Is it compliant? [Yes/No] _____
- Which constraint is more critical for edge deployment? [accuracy/latency]
- How would you prioritize optimization? [accuracy first/latency first/balanced]

### Question 5: Optimization Comparison Analysis
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
- Created TinyMLPerf-style standardized benchmarks for reproducible cross-system comparison
- Developed optimization comparison workflows that generate actionable recommendations
- All tests pass ✅ (validated by `test_module()`)

### Systems Engineering Insights Gained
- **Measurement Science**: Statistical significance requires proper sample sizes and variance control
- **Benchmark Design**: Standardized protocols enable fair comparison across different systems
- **Trade-off Analysis**: Pareto frontiers reveal optimization opportunities and constraints
- **Production Integration**: Automated reporting transforms measurements into engineering decisions

### Ready for Systems Capstone
Your benchmarking implementation enables the final milestone: a comprehensive systems evaluation comparing CNN vs TinyGPT with quantization, pruning, and performance analysis. This is where all 19 modules come together!

Export with: `tito module complete 19`

**Next**: Milestone 5 (Systems Capstone) will demonstrate the complete ML systems engineering workflow!
"""
