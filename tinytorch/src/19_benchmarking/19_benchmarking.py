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
r"""
# Module 19: Benchmarking - Performance Measurement Infrastructure

Welcome to Module 19! In this module, we transition from individual optimization algorithms to systemic evaluation: constructing statistically rigorous, reproducible benchmarking infrastructure that quantifies latency, accuracy, memory, and energy trade-offs across neural network architectures.

## 🔗 Prerequisites & Progress

<img src="benchmarking_blueprint.svg" width="100%" alt="TinyTorch Framework Blueprint: Module 19 Benchmarking" />

### Architectural Roadmap

| Optimization Stage | Core Technique | Hardware & Algorithmic Focus | Primary Target |
|:---|:---|:---|:---|
| **14. Profiling** | Microsecond Benchmarks & Tracing | Profiler timer loops, Roofline bounds | Identify compute vs memory bottlenecks |
| **15. Quantization** | Symmetric/Asymmetric INT8 | 8-bit scale & zero-point arithmetic | 4× weight footprint & memory bus bandwidth |
| **16. Compression** | Magnitude Pruning & Distillation | Weight sparsity & student distillation | Redundant parameter elimination |
| **17. Acceleration** | SIMD GEMM, Fusion, `im2col` | Memory traffic elimination & systolic arrays | Kernel overhead & hardware utilization |
| **18. Memoization** | Static KV Cache Buffers | $\mathcal{O}(1)$ decode steps & zero recomputation | Autoregressive decoding latency |
| **19. Benchmarking** *(Active)* | **Statistical Evaluation & MLPerf** | **Variance control, confidence intervals, Pareto frontiers** | **Rigorous cross-system evaluation** |
| **20. Capstone** | End-to-End Pipeline Optimization | Capstone deployment & system integration | Production serving pipeline |

## 🎯 Learning Objectives
By the end of this module, you will:
1. Construct high-precision benchmarking infrastructure with warmup discard and statistical variance control.
2. Formulate sample distribution statistics: reporting median $P_{50}$, tail latency $P_{95}/P_{99}$, and student-$t$ confidence intervals.
3. Build the `Benchmark` and `BenchmarkSuite` evaluation engines to compare baseline and optimized models across multiple hardware axes.
4. Implement an MLPerf Tiny standardized compliance runner enforcing deterministic input seeds and hard accuracy/latency thresholds.
5. Derive empirical Pareto frontiers to identify non-dominated model variants across competing systems objectives.

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/19_benchmarking/benchmarking.ipynb`  
**Building Side:** Code exports to `tinytorch.perf.benchmarking`

<img src="benchmarking_source_card.svg" width="100%" alt="Source Code Mapping Card for Module 19 Benchmarking" />

```python
# Final package structure:
from tinytorch.perf.benchmarking import Benchmark, BenchmarkSuite, BenchmarkResult, MLPerf, precise_timer
```

## 📋 Module Dependencies

| Dependency Module | Exported Abstraction | Consumed Functional Role | Memory & Evaluation Invariant |
|:---|:---|:---|:---|
| **Module 01 (`01_tensor`)** | `Tensor` | Contiguous N-D numerical array representation | Evaluation inputs and outputs without autograd overhead |
| **Module 07 (`07_layers`)** | `Linear` | Fully connected layer primitive | Reference workloads for single-layer benchmarking |
| **Module 14 (`14_profiling`)** | `Profiler` | High-resolution microsecond timer | Core latency and memory probe reused by `Benchmark` |
| **Modules 15–18** | Quantized, Pruned, Accelerated Models | Optimized model variants | Inputs to multi-dimensional comparative benchmarking |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp perf.benchmarking
#| export

import json
import os
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
rng = np.random.default_rng(7)

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.perf.profiling import Profiler  # Module 14: reuse its measurements

# Optional dependency, for plots only
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Constants for benchmarking defaults
DEFAULT_WARMUP_RUNS = 5  # Default warmup runs: cache warming and CPU clock ramp (NumPy has no JIT)
DEFAULT_MEASUREMENT_RUNS = 10  # Default measurement runs for statistical significance

# Illustrative energy model (no power meter here): a fixed cost per inference, an
# active-power term proportional to time, and a static term proportional to memory
ENERGY_BASE_JOULES = 0.1
ENERGY_JOULES_PER_SECOND = 2.0  # about 2 W while the model runs
ENERGY_JOULES_PER_MB = 0.01

# %% [markdown]
r"""
### Looking Ahead

The benchmarking tools you build here will be used in Module 20's capstone project, where you'll apply optimization techniques competitively. For now, focus on building reliable, fair measurement infrastructure.
"""

# %% [markdown]
r"""
## 💡 Introduction: What is Fair Benchmarking?

Benchmarking in ML systems is not merely recording wall-clock time—it is an empirical science requiring controlled experimental conditions to enable fair, reproducible comparisons that guide production deployment decisions.

<img src="benchmarking_methodology_overview.svg" width="100%" alt="Benchmarking Methodology Pipeline" />

### Confounding Factors & Controlled Experimental Variables

| Noise Source | Physical Hardware Mechanism | Benchmarking Defense Strategy | Controlled Variable |
|:---|:---|:---|:---|
| **Cold Starts** | Dynamic library loading & page faults | Warmup iterations discarded before recording | Memory state |
| **Thermal Throttling** | DVFS frequency scaling when silicon overheats | Cooldown pauses & randomized trial interleaving | CPU / GPU clock frequency |
| **OS Interrupts** | Background scheduler preemption & context switches | Multi-trial sampling with median and percentile reporting | CPU core pinning |
| **Cache Pollution** | Shared L2/L3 cache evictions by OS daemons | Contiguous tensor layout & deterministic array strides | SRAM cache residency |
| **Memory Pressure** | Python garbage collection pauses | Explicit GC disabled during inner timing loop | Heap allocation state |

---

## 📐 Foundations: Statistics for Performance Engineering

Inference latency on modern superscalar processors is an inherently non-deterministic, right-skewed stochastic process.

<img src="latency_anatomy_distribution.svg" width="100%" alt="The Anatomy of Latency Distributions and Tail Percentiles" />

<img src="benchmarking_latency_card.svg" width="100%" alt="Latency Distribution Card" />

### Central Limit Theorem & Confidence Intervals

While individual latency measurements exhibit heavy-tailed distributions due to system hiccups, the sample mean $\bar{X}$ over $n$ independent trials converges toward a normal distribution:

$$\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i, \quad s = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2}$$

The standard error of the mean ($\text{SE}$) and the two-sided $95\%$ Student-$t$ confidence interval are given by:

$$\text{SE} = \frac{s}{\sqrt{n}}, \quad \text{CI}_{95\%} = \left[ \bar{X} - t_{0.025, \, n-1} \frac{s}{\sqrt{n}}, \quad \bar{X} + t_{0.025, \, n-1} \frac{s}{\sqrt{n}} \right]$$

### Multi-Objective Optimization & Pareto Dominance

Model optimization is multi-objective: latency, accuracy, memory, and energy represent competing physical trade-offs.

**Mathematical Definition of Pareto Dominance**:
Let $\mathcal{M}$ be the set of evaluation metrics. A model variant $\theta_A$ strictly Pareto-dominates variant $\theta_B$ ($\theta_A \succ \theta_B$) if and only if:

$$\forall m \in \mathcal{M}, \quad \text{score}_m(\theta_A) \ge \text{score}_m(\theta_B) \quad \land \quad \exists m \in \mathcal{M}, \quad \text{score}_m(\theta_A) > \text{score}_m(\theta_B)$$

| Optimization Dimension | Preferred Direction | Systems Constraint | Typical Hardware Boundary |
|:---|:---|:---|:---|
| **Latency ($ms$)** | Minimize ($\downarrow$) | SLA / Real-time interactivity ($<100 \text{ ms}$) | ALU compute & DRAM bandwidth |
| **Accuracy ($\%$)** | Maximize ($\uparrow$) | Task fidelity & quality threshold | Model representational capacity |
| **Memory ($MB$)** | Minimize ($\downarrow$) | Embedded / GPU VRAM capacity limit | SRAM / DRAM capacity |
| **Energy ($Joules$)** | Minimize ($\downarrow$) | Mobile battery life & thermal TDP envelope | Power delivery & dynamic voltage |

---

## 🏗️ Implementation: Building Professional Benchmarking Infrastructure

### Architectural Components

| Infrastructure Component | Role & Scope | Core Inputs | Produced Output Abstraction |
|:---|:---|:---|:---|
| `precise_timer` | High-precision monotonic interval timing | Code block context manager | `timer.elapsed` (seconds) |
| `Profiler` (Module 14) | Hardware timer and memory probe | Model + input tensor | Raw latency and memory readings |
| `Benchmark` | Multi-model evaluation across single metrics | Models, datasets, warmup/trial counts | `Dict[str, BenchmarkResult]` |
| `BenchmarkResult` | Statistical analysis container | Raw measurements list | Mean, std, median, $P_{90}$, $P_{99}$, CI |
| `BenchmarkSuite` | Multi-dimensional evaluation engine | Models, datasets, metric configurations | Comparative trade-off tables & Pareto analysis |
| `MLPerf` | Standardized edge compliance harness | Reference tasks, deterministic seeds | Pass/Fail compliance report |

### Statistical Metrics Tracked by BenchmarkResult

| Statistical Metric | Mathematical Estimator | Systems Interpretation | Robustness Against Outliers |
|:---|:---|:---|:---|
| **Mean ($\mu$)** | $\frac{1}{n} \sum X_i$ | Average throughput expectation | Sensitive to tail stalls |
| **Median ($P_{50}$)** | 50th percentile rank | Typical steady-state latency | Highly robust |
| **Tail Latency ($P_{95}, P_{99}$)** | 95th / 99th percentile rank | Worst-case SLA compliance bound | Captures OS scheduling spikes |
| **Std Deviation ($s$)** | $\sqrt{\frac{1}{n-1} \sum (X_i - \bar{X})^2}$ | Measurement dispersion | Sensitive to extreme outliers |
| **Coeff. of Variation (CV)** | $\frac{s}{\bar{X}} \times 100\%$ | Relative measurement noise | Normalized noise index |
| **$95\%$ Confidence Interval** | $\bar{X} \pm t_{n-1} \cdot \text{SE}$ | True population mean bound | Requisite for statistical claims |
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
    ### BEGIN SOLUTION role="scaffold"
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
            z_score = 1.96  # normal approximation for 95%; accurate once count is large
            margin_error = z_score * (self.std / np.sqrt(self.count))
            self.ci_lower = self.mean - margin_error
            self.ci_upper = self.mean + margin_error
        else:
            self.ci_lower = self.ci_upper = self.mean

    def percentile(self, p: float) -> float:
        """The value p percent of the way through the sorted measurements (NumPy's linear rule)."""
        return float(np.percentile(self.values, p))

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
r"""
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
r"""
## 🏗️ High-Precision Timing Infrastructure

Accurate timing is the foundation of performance benchmarking. System clocks have different precision and behavior, so we need a robust timing mechanism.

### Timing Challenges in Practice

When timing a function call in high-level languages like Python, several layers of operating system and hardware indirection intervene between the software invocation and the physical timer:

$$\Delta t_{\text{measured}} = \Delta t_{\text{true}} + \delta_{\text{call}} + \delta_{\text{OS}} + \delta_{\text{quantization}}$$

| Latency Component | Typical Magnitude | Root Cause / System Source | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Kernel / Forward Pass** ($\Delta t_{\text{true}}$) | $\mu\text{s}$ to $\text{ms}$ | Actual computational operations (FLOPs, memory loads) | Target metric under benchmark |
| **Syscall Overhead** ($\delta_{\text{call}}$) | $10\text{--}50\text{ ns}$ | User-to-kernel context switch for clock sampling | Use monotonic userspace vDSO clock |
| **OS Scheduling** ($\delta_{\text{OS}}$) | $\mu\text{s}$ to $\text{ms}$ | Thread preemption, core migration, page faults | Discard warmup runs, sample distributions |
| **Timer Quantization** ($\delta_{\text{quantization}}$) | $1\text{ ns}$ to $1\text{ }\mu\text{s}$ | Hardware counter frequency resolution limits | Use nanosecond-resolution monotonic counter |

For microsecond-precision timing, each of these can introduce significant error.

### Why `perf_counter()` Matters

Python's `time.perf_counter()` is specifically designed for interval measurement:
- **Monotonic**: Never goes backwards (unaffected by NTP time sync or system clock adjustments)
- **High resolution**: Nanosecond resolution backed by hardware counters (`RDTSC` on x86, `CNTVCT_EL0` on ARM)
- **Low overhead**: Optimized system call via virtual Dynamic Shared Object (vDSO) avoiding kernel trapping

### Timing Best Practices: The Context Manager Pattern

| Context Phase | Program Action | System State / Effect |
| :--- | :--- | :--- |
| `__enter__` | `t_start = time.perf_counter()` | Sample monotonic nanosecond counter prior to workload |
| Yield Block | Execute operation (forward pass) | Target compute runs; CPU registers and cache active |
| `__exit__` (`finally`) | `t_end = time.perf_counter()` | Guaranteed sample even if an unexpected exception occurs |
| Post-Context | `elapsed = t_end - t_start` | Monotonic interval $\Delta t \ge 0$ recorded reliably |

This pattern ensures timing starts and stops correctly with deterministic resource handling even if exceptions occur.
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
    2. Yield a small Timer object and fill in .elapsed when the block exits
    3. Use try/finally so .elapsed is set even if the block raises

    Yields:
        Timer object with .elapsed attribute (set after context exits)

    EXAMPLE:
    >>> with precise_timer() as timer:
    ...     time.sleep(0.1)  # Some operation
    >>> print(f"Elapsed: {timer.elapsed:.4f}s")
    Elapsed: 0.1001s

    HINTS:
    - perf_counter() is monotonic and high-resolution
    - Record the start before yield and compute elapsed after it
    - try/finally, not try/except: errors propagate but the time is still recorded
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
r"""
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
    assert timer.elapsed >= 0.005, f"Expected ~0.01s, got {timer.elapsed}s"

    # Test multiple uses
    times = []
    for _ in range(3):
        with precise_timer() as timer:
            time.sleep(0.001)  # 1ms sleep
        times.append(timer.elapsed)

    # All times should be reasonably close
    assert all(t >= 0.0005 for t in times)

    print("✅ precise_timer works correctly!")

if __name__ == "__main__":
    test_unit_precise_timer()

# %% [markdown]
r"""
### Benchmark Class: Core Measurement Engine

The `Benchmark` class implements the core measurement logic for different metrics. It handles the orchestration of multiple models, datasets, and measurement protocols.

<img src="benchmarking_methodology_overview.svg" alt="Benchmarking Methodology Overview" width="100%">

### Benchmark Architecture Execution Flow

| Stage | Input Artifacts | Processing Step | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **1. Ingestion** | Models $[M_1, M_2, \dots]$, Datasets $[D_1, D_2, \dots]$ | Register candidate architectures and evaluation datasets | Model registry with validated callable interfaces |
| **2. Warmup** | Synthetic or unmeasured batches | Execute $W$ iterations to warm CPU caches and JIT tables | Discarded startup latencies, stabilized hardware clocks |
| **3. Measurement** | Fixed-seed inputs | Sample $N$ independent forward passes with `perf_counter` | Raw latency samples $[t_1, t_2, \dots, t_N]$ |
| **4. Profiling** | Model instance | Trace memory allocations and peak buffer usage | Traced peak memory (MB) and FLOP counts |
| **5. Synthesis** | Raw timing and memory metrics | Compute $\mu, s, \text{SE}$, and confidence intervals $[CI_{\text{low}}, CI_{\text{high}}]$ | `BenchmarkResult` container with system metadata |

### Why Warmup Runs Matter

Modern operating systems and processors have multiple layers of runtime adaptation:
- **JIT compilation**: Specialized machine code and branch paths stabilize after initial iterations
- **CPU frequency scaling**: Dynamic Voltage and Frequency Scaling (DVFS) ramps execution cores to performance governors
- **Cache warming**: Instruction and weight caches ($L_1/L_2/L_3$) achieve steady-state hit rates
- **Memory frame allocation**: OS page faults occur during initial virtual memory touches

<img src="latency_anatomy_distribution.svg" alt="Latency Anatomy and Warmup" width="100%">

### Multiple Benchmark Dimensions

Different metrics require distinct measurement strategies:

| Dimension | Primary Focus | Key Governing Factors | Measurement Technique |
| :--- | :--- | :--- | :--- |
| **Latency** | Milliseconds per forward pass ($\text{ms}$) | Batch size, compute depth, memory bandwidth | High-precision timing via `perf_counter` |
| **Accuracy** | Fraction of correct inferences ($[0, 1]$) | Quantization noise, pruning sparsity, model capacity | Ground-truth evaluation over validation set |
| **Memory** | Allocator footprint during execution ($\text{MB}$) | Parameter storage, activation tensors, workspace | Traced allocator peak vs OS process RSS |
| **Energy** | Joules per inference pass ($\text{mJ}$) | FLOP complexity, SRAM transfers, DRAM accesses | Empirical analytical modeling or hardware PMUs |
"""

# %% [markdown]
r"""
### Benchmark.__init__: Setting Up the Measurement Engine

The `Benchmark` constructor configures the measurement infrastructure: models to test,
datasets for evaluation, and system metadata for reproducibility. It reuses the
`Profiler` from Module 14 for individual model measurements.

| Configuration Field | Source / Inspection Method | Architectural Role |
| :--- | :--- | :--- |
| `models` | Candidate architectures ($M_i$) | Models under comparative evaluation |
| `datasets` | Labeled or synthetic batches | Validation data slices for evaluation |
| `profiler` | `Profiler()` (Module 14) | Memory tracing and execution instrumentation |
| `system_info` | `platform.platform()`, `cpu_count()` | Hardware and runtime metadata for reproducibility |
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
        ### BEGIN SOLUTION role="scaffold"
        if warmup_runs < 0 or measurement_runs <= 0:
            raise ValueError("warmup_runs must be nonnegative and measurement_runs positive")
        self.models = list(models)
        # Keep display names when possible, but never let copied names overwrite results.
        self.model_names = []
        for i, model in enumerate(self.models):
            stem = str(getattr(model, 'name', None) or f'model_{i}')
            name = stem
            suffix = 1
            while name in self.model_names:
                name = f'{stem}_{suffix}'
                suffix += 1
            self.model_names.append(name)
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
r"""
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
r"""
### Benchmark.run_latency_benchmark: Measuring Inference Speed

Latency benchmarking measures how long each model takes to process input. We use
the Profiler for warmup, then collect multiple individual measurements for
statistical analysis via `BenchmarkResult`.

<img src="benchmarking_latency_card.svg" alt="Tail Latency Percentiles" width="100%">

### Latency Measurement Pipeline

| Pipeline Stage | Implementation Action | Purpose & Guarantees |
| :--- | :--- | :--- |
| **Input Synthesis** | `Tensor(rng.standard_normal(shape))` | Allocates representative evaluation tensor matching hardware target |
| **Warmup Phase** | `profiler.measure_latency(warmup=W)` | Triggers initial JIT compilation, cache warming, and frame allocation (discarded) |
| **Measurement Sampling**| `precise_timer()` loop ($N$ trials) | Gathers independent steady-state execution latencies $[t_1, t_2, \dots, t_N]$ in ms |
| **Statistical Wrapping**| `BenchmarkResult("latency", samples)`| Computes mean, variance, percentiles (p50/p95/p99), and 95% confidence interval |
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-latency", "solution": true}
#| exporti
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
        model_name = self.model_names[i]

        # Create input tensor for profiling
        input_tensor = Tensor(rng.standard_normal(input_shape).astype(np.float32))

        # Warm up through the Profiler (that one timing is discarded), then
        # record every measured run on its own so BenchmarkResult sees the tail
        self.profiler.measure_latency(model, input_tensor, warmup=self.warmup_runs, iterations=1)
        latencies = []
        for _ in range(self.measurement_runs):
            latencies.append(self.profiler.measure_latency(model, input_tensor, warmup=0, iterations=1))

        results[model_name] = BenchmarkResult(
            f"{model_name}_latency_ms",
            latencies,
            metadata={'input_shape': input_shape, **self.system_info}
        )

    return results
    ### END SOLUTION

Benchmark.run_latency_benchmark = benchmark_run_latency_benchmark

# %% [markdown]
r"""
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
r"""
### Simulated Accuracy: The Honest Stand-In

Some models handed to a benchmark harness have no `evaluate` method and no
labeled data behind them. The tempting move is to make up a plausible score.
Do not. A fabricated number propagates into every table, chart, and comparison
downstream, and nothing further along can tell it apart from a measurement.

The stand-in below is deliberately modest. It runs the model on a small,
seeded **proxy task** -- inputs drawn from a fixed generator, labels defined by
a rule the inputs actually determine -- and reports how often the model's
output agrees. That is a real score of a fake task, which is a very different
claim from a fake score of a real task, and the caller marks it `simulated`.

Two properties make it defensible:

- **Deterministic**: the same model and dataset always give the same number.
- **Output-dependent**: it measures what the model *computes*. It never reads
  the model's name or its position in the list.
"""

# %% nbgrader={"grade": false, "grade_id": "simulated-accuracy", "solution": true}
#| export
def _simulated_accuracy(model: Any, dataset: Any, num_samples: int = 32) -> float:
    """
    Score a model on a seeded proxy task when no ground truth is available.

    This is NOT a measurement of the model's accuracy on `dataset`. It is a
    reproducible, output-dependent stand-in for demos and smoke tests. Callers
    must mark any result built from it as simulated.

    TODO: Score the model on a small seeded synthetic classification task.

    APPROACH:
    1. Derive a seed from the dataset so different datasets give different probes
    2. Draw num_samples probe vectors; the label is 1 when the probe sums positive
    3. Run the model on each probe and predict 1 when its output sums positive
    4. Return the fraction of agreements; a model we cannot invoke gets 0.5

    Args:
        model: Any object with forward(), predict(), or __call__
        dataset: Used only to vary the probe seed, never as ground truth
        num_samples: Number of probe vectors (default: 32)

    Returns:
        float: Agreement rate in [0, 1]; 0.5 means chance

    HINTS:
    - A model that passes signal through scores near 1.0; one that destroys it
      lands near 0.5, which is exactly the discrimination we want
    - Never let the score depend on getattr(model, 'name') or the model's index
    """
    ### BEGIN SOLUTION role="scaffold"
    # Vary the probe per dataset, deterministically and without hashing.
    seed = 7 + sum(repr(dataset).encode()) % 1000
    probe_rng = np.random.default_rng(seed)

    probes = probe_rng.standard_normal((num_samples, 4)).astype(np.float32)
    true_labels = (probes.sum(axis=1) > 0).astype(int)

    predicted = []
    for probe in probes:
        try:
            x = Tensor(probe.reshape(1, -1))
            if hasattr(model, 'forward'):
                out = model.forward(x)
            elif hasattr(model, 'predict'):
                out = model.predict(x)
            elif callable(model):
                out = model(x)
            else:
                raise TypeError("Model needs forward(), predict(), or __call__")
        except Exception as exc:
            raise RuntimeError("Synthetic accuracy probe failed; no score was measured") from exc

        out_data = out.data if hasattr(out, 'data') else out
        predicted.append(int(np.asarray(out_data, dtype=np.float64).sum() > 0))

    return float(np.mean(np.asarray(predicted) == true_labels))
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: _simulated_accuracy

**What we're testing**: Determinism, output-dependence, and identity-independence
**Why it matters**: A stand-in score is only defensible if it cannot be gamed by
renaming a model or reordering the list
**Expected**: Same input gives the same score; a pass-through model beats a model
that destroys the signal; renaming changes nothing
"""

# %% nbgrader={"grade": true, "grade_id": "test-simulated-accuracy", "locked": true, "points": 5}
def test_unit_simulated_accuracy():
    """🧪 Test the simulated-accuracy stand-in."""
    print("🧪 Unit Test: _simulated_accuracy...")

    class PassThrough:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return x

    class Deaf:
        """Destroys the input signal, so it should land near chance."""
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return Tensor(np.zeros_like(x.data) - 1.0)

    dataset = {"d": "1"}

    # Deterministic
    a = _simulated_accuracy(PassThrough("a"), dataset)
    assert a == _simulated_accuracy(PassThrough("a"), dataset), "Score must be reproducible"

    # Identity-independent: the name must not move the number
    assert a == _simulated_accuracy(PassThrough("zzz_accurate_efficient"), dataset), \
        "Score must not depend on the model's name"

    # Output-dependent: passing signal through beats destroying it
    deaf = _simulated_accuracy(Deaf("b"), dataset)
    assert a > deaf, f"Pass-through ({a}) should beat signal-destroying ({deaf})"
    assert 0.0 <= deaf <= 1.0 and 0.0 <= a <= 1.0

    print("✅ _simulated_accuracy works correctly!")

if __name__ == "__main__":
    test_unit_simulated_accuracy()

# %% [markdown]
r"""
### Benchmark.run_accuracy_benchmark: Measuring Prediction Quality

Accuracy benchmarking evaluates model correctness across datasets. A model that
exposes an `evaluate(dataset)` method is **measured**: whatever score it returns
is what gets reported.

There is no honest way to measure accuracy without ground truth, so a model that
has no `evaluate` method cannot be scored. Rather than invent a plausible number,
the default raises an error. For classroom demonstrations only, explicitly pass
`simulate=True` to score against a **seeded synthetic label set**; the benchmark
marks the result `simulated=True` so nobody mistakes it for a measurement. The
score still depends on what the model actually outputs -- never on its name or
its position in the list.

> A benchmark that reports a number nobody measured is worse than a benchmark
> that reports nothing. If you take one habit from this module, take that one.

$$\text{Accuracy}(M) = \frac{1}{|D_{\text{val}}|} \sum_{(x, y) \in D_{\text{val}}} \mathbf{1}\big(\arg\max f_\theta(x) = y\big)$$

| Evaluation Mode | Required Interface | Dataset Source | Scientific Rigor Guarantee |
| :--- | :--- | :--- | :--- |
| **Empirical Evaluation** | `model.evaluate(dataset)` | Labeled validation partitions | Measures true task performance against verified ground truth |
| **Classroom Probe** (`simulate=True`) | Model forward pass outputs | Seeded synthetic label probe | Tagged with `simulated=True` metadata to prevent reporting unmeasured numbers |
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-accuracy", "solution": true}
#| exporti
def benchmark_run_accuracy_benchmark(self, simulate: bool = False) -> Dict[str, BenchmarkResult]:
    """
    Benchmark model accuracy across datasets.

    TODO: Evaluate each model on each dataset and collect accuracy scores

    APPROACH:
    1. Iterate over all models and datasets
    2. If the model has evaluate(), use its score -- that is a real measurement
    3. Otherwise raise unless simulate=True was explicitly requested; then use
       the seeded classroom probe and flag its result as simulated
    4. Reject non-finite/out-of-range accuracy and wrap the scores in BenchmarkResult

    HINTS:
    - Use hasattr(model, 'evaluate') for duck-typing
    - The simulated path must depend on the model's OUTPUT, never on its name or
      its index -- a score keyed to identity is a fabricated benchmark
    - Record simulated=True in metadata so a reader can tell the two apart
    """
    ### BEGIN SOLUTION role="scaffold"
    results = {}

    for i, model in enumerate(self.models):
        model_name = self.model_names[i]
        accuracies = []
        simulated = False

        for dataset in self.datasets:
            if hasattr(model, 'evaluate'):
                # Real measurement: the model tells us how it did on this dataset
                accuracy = float(model.evaluate(dataset))
            else:
                if not simulate:
                    raise ValueError("Accuracy requires model.evaluate(dataset); use simulate=True only for a classroom probe")
                # No ground truth available. Score the model's actual outputs
                # against a seeded reference label set so the number is at least
                # reproducible and output-dependent -- but flag it as simulated.
                simulated = True
                accuracy = _simulated_accuracy(model, dataset)

            if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError("Accuracy must be finite and in [0, 1]")
            accuracies.append(accuracy)

        if simulated:
            print(f"   ⚠️  {model_name}: no evaluate() method -- accuracy is SIMULATED, not measured")

        results[model_name] = BenchmarkResult(
            f"{model_name}_accuracy",
            accuracies,
            metadata={
                'num_datasets': len(self.datasets),
                'simulated': simulated,
                **self.system_info,
            }
        )

    return results
    ### END SOLUTION

Benchmark.run_accuracy_benchmark = benchmark_run_accuracy_benchmark

# %% [markdown]
r"""
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

    results = benchmark.run_accuracy_benchmark(simulate=True)
    assert len(results) == 2
    assert all(isinstance(r, BenchmarkResult) for r in results.values())
    assert all(0 <= r.mean <= 1 for r in results.values())

    print("✅ Benchmark.run_accuracy_benchmark works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_accuracy()

# %% [markdown]
r"""
### Benchmark.run_memory_benchmark: Measuring Resource Consumption

Memory benchmarking tracks how much RAM each model consumes during inference.
We retain the Profiler's traced allocation peak even when it is small. This is
distinguished from total process RSS or the static size of all model parameters:

| Memory Dimension | Measurement Target | Scope & Definition | Systems Significance |
| :--- | :--- | :--- | :--- |
| **Traced Allocator Peak** | `memory_stats['peak_memory_mb']` | High-water mark of live forward tensor buffers | Determines minimal physical DRAM/SRAM working footprint |
| **Static Weights** | $\sum_l \lvert W_l \rvert \times \text{sizeof}(\text{dtype})$ | Persistent model parameters in storage | Dictates flash storage requirements and transfer latency |
| **Transient Activations** | Layer intermediate shapes $\mathcal{O}(B \times S \times D)$ | Execution buffers during layer evaluation | Opportunities for memory pooling and inplace reuse |
| **Process RSS** | Resident Set Size reported by OS | Entire Python runtime, shared libraries, and heap | High-level system footprint influenced by OS page allocation |
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-memory", "solution": true}
#| exporti
def benchmark_run_memory_benchmark(self, input_shape: Tuple[int, ...] = (1, 28, 28)) -> Dict[str, BenchmarkResult]:
    """
    Benchmark model memory usage using Profiler.

    TODO: Measure memory consumption for each model across multiple runs

    APPROACH:
    1. Use self.profiler.measure_memory() for each model
    2. Keep the measured peak, including values below 1 MB
    3. Collect self.measurement_runs samples
    4. Wrap results in BenchmarkResult

    HINTS:
    - memory_stats['peak_memory_mb'] is the primary metric
    - Parameter storage is a different metric; do not substitute it for allocator peaks
    """
    ### BEGIN SOLUTION role="scaffold"
    results = {}

    for i, model in enumerate(self.models):
        model_name = self.model_names[i]
        memory_usages = []

        for run in range(self.measurement_runs):
            # Use Profiler to measure memory
            memory_stats = self.profiler.measure_memory(model, input_shape)
            # Use peak_memory_mb as the primary metric
            memory_used = memory_stats['peak_memory_mb']

            # Preserve the allocator peak, even for a small model.
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
r"""
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
r"""
### Benchmark.compare_models: Cross-Model Comparison

The compare_models method dispatches to the appropriate benchmark type and
formats results into a structured list of dictionaries for easy comparison.
This is the primary interface for multi-model evaluation.
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-compare", "solution": true}
#| exporti
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
    ### BEGIN SOLUTION role="scaffold"
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
            'model': model_name,
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
r"""
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
    accuracy_results = benchmark.run_accuracy_benchmark(simulate=True)
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
r"""
### BenchmarkSuite: Comprehensive Multi-Metric Evaluation

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

<img src="benchmarking_source_card.svg" alt="Source Code Mapping" width="100%">

| Evaluation Phase | Component Operation | Collected Data | Systems Decision Role |
| :--- | :--- | :--- | :--- |
| **Model Ingestion** | `models = [M1, M2, ...]` | Model architectures and weight buffers | Candidates for comparative deployment profiling |
| **Metric Execution** | Latency, Accuracy, Memory, Energy | Sample distributions and allocator peaks | Multi-objective empirical measurement vectors |
| **Aggregation** | Unified dictionary indexing | Synchronized per-metric `BenchmarkResult` | Cross-model normalization and variance alignment |
| **Pareto Analysis** | Non-dominated sorting | Pareto frontiers, best-in-class flags | Eliminates strictly sub-optimal candidate variants |
| **Deployment Synthesis** | Markdown & JSON report generator | Quantitative tradeoff recommendations | Concrete deployment mapping (Server, Mobile, IoT) |

### Pareto Frontier Analysis

The suite automatically identifies Pareto-optimal solutions - models that aren't strictly dominated by others across all metrics. This reveals the true trade-off space for optimization decisions.

### Energy Efficiency Modeling

Since direct energy measurement requires specialized hardware, we estimate energy based on computational complexity and memory usage. This provides actionable insights for battery-powered deployments.
"""

# %% [markdown]
r"""
### BenchmarkSuite.__init__: Setting Up Multi-Metric Evaluation

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
        ### BEGIN SOLUTION role="scaffold"
        self.models = models
        self.datasets = datasets
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.benchmark = Benchmark(models, datasets)
        self.results = {}
        ### END SOLUTION

# %% [markdown]
r"""
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
r"""
### BenchmarkSuite.run_full_benchmark: Orchestrating All Measurements

The `run_full_benchmark` method runs all four benchmark categories (latency, accuracy,
memory, energy) in sequence, assembling comprehensive empirical results for each candidate model:

| Evaluation Phase | Invoked Subsystem | Metric Output | Result Dictionary Key |
| :--- | :--- | :--- | :--- |
| **1. Latency** | `Benchmark.run_latency_benchmark()` | Inference duration ($\text{ms}$) with warmup discard | `results['latency']` |
| **2. Accuracy** | `Benchmark.run_accuracy_benchmark()` | Empirical evaluation accuracy score ($[0, 1]$) | `results['accuracy']` |
| **3. Memory** | `Benchmark.run_memory_benchmark()` | Traced peak memory allocator allocation ($\text{MB}$) | `results['memory']` |
| **4. Energy** | `_estimate_energy_efficiency()` | Relative hardware energy proxy score | `results['energy']` |
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-run", "solution": true}
#| exporti
def benchsuite_run_full_benchmark(self, simulate: bool = False,
                                 input_shape: Tuple[int, ...] = (1, 28, 28)) -> Dict[str, Dict[str, BenchmarkResult]]:
    """
    Run all benchmark categories using input_shape for latency and memory.

    Choose the shape of a representative inference batch for your model.

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
    ### BEGIN SOLUTION role="scaffold"
    print("🧪 Running comprehensive benchmark suite...")

    # Run all benchmark types
    print("  📊 Measuring latency...")
    self.results['latency'] = self.benchmark.run_latency_benchmark(input_shape=input_shape)

    print("  🎯 Measuring accuracy...")
    self.results['accuracy'] = self.benchmark.run_accuracy_benchmark(simulate=simulate)

    print("  💾 Measuring memory usage...")
    self.results['memory'] = self.benchmark.run_memory_benchmark(input_shape=input_shape)

    # Simulate energy benchmark (would require specialized hardware)
    print("  ⚡ Estimating energy efficiency...")
    self.results['energy'] = self._estimate_energy_efficiency()

    return self.results
    ### END SOLUTION

BenchmarkSuite.run_full_benchmark = benchsuite_run_full_benchmark

# %% [markdown]
r"""
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

        results = suite.run_full_benchmark(simulate=True)

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
r"""
### BenchmarkSuite._estimate_energy_efficiency: Energy Modeling

Since direct energy measurement requires specialized hardware (power meters, RAPL),
we estimate energy from latency and memory usage. This simplified model captures the
key relationship: energy is proportional to power (memory-related) multiplied by time (latency).

```
Energy Estimation Model:
energy = base_cost + (latency/1000) * 2.0 + memory * 0.01   (Joules; illustrative constants)
         ↑            ↑                      ↑
         Fixed        Time component          Memory component
         overhead     (active power)          (static power)
```
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-energy", "solution": true}
#| export
def _benchsuite_estimate_energy_efficiency(self) -> Dict[str, BenchmarkResult]:
    """
    Estimate energy efficiency (simplified simulation).

    TODO: Estimate energy from latency and memory measurements

    APPROACH:
    1. Check if latency and memory results are available
    2. Combine latency and memory into energy estimate per measurement
    3. Raise a clear error if the latency or memory results are missing
    4. Wrap results in BenchmarkResult

    HINTS:
    - Energy model: ENERGY_BASE_JOULES + (lat/1000) * ENERGY_JOULES_PER_SECOND + mem * ENERGY_JOULES_PER_MB
    - Use zip() to pair latency and memory measurements
    """
    ### BEGIN SOLUTION role="scaffold"
    energy_results = {}

    for i, model in enumerate(self.models):
        model_name = self.benchmark.model_names[i]

        # Energy roughly correlates with latency * memory usage
        if 'latency' in self.results and 'memory' in self.results:
            latency_result = self.results['latency'].get(model_name)
            memory_result = self.results['memory'].get(model_name)

            if latency_result and memory_result:
                # Energy ∝ power × time, power ∝ memory usage
                energy_values = []
                for lat, mem in zip(latency_result.values, memory_result.values):
                    energy = ENERGY_BASE_JOULES + (lat / 1000) * ENERGY_JOULES_PER_SECOND + mem * ENERGY_JOULES_PER_MB
                    energy_values.append(energy)

                energy_results[model_name] = BenchmarkResult(
                    f"{model_name}_energy_joules",
                    energy_values,
                    metadata={'estimated': True, **self.benchmark.system_info}
                )

    if not energy_results:
        raise RuntimeError(
            "Energy estimation needs latency and memory results first\n"
            "  💡 Run the latency and memory benchmarks (or run_full_benchmark) before estimating energy"
        )

    return energy_results
    ### END SOLUTION

BenchmarkSuite._estimate_energy_efficiency = _benchsuite_estimate_energy_efficiency

# %% [markdown]
r"""
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
r"""
### BenchmarkSuite.plot_results: Visualization

The plot_results method generates a 2x2 grid of bar charts comparing models
across all four metrics. The best performer in each category is highlighted green.
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-plot", "solution": true}
#| exporti
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
    ### BEGIN SOLUTION role="scaffold"
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
                model_names.append(model_name)
                means.append(result.mean)
                stds.append(result.std)

            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f'{metric.capitalize()} Comparison' +
                         (' (estimated)' if metric == 'energy' else
                          ' (synthetic probe)' if any(r.metadata.get('simulated', False)
                          for r in self.results[metric].values()) else ''))
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

    # Both result dicts are keyed by model name, so a plain lookup pairs them
    for model_name, x_result in self.results[x_metric].items():
        y_result = self.results[y_metric].get(model_name)
        if y_result is None:
            continue
        x_values.append(x_result.mean)
        y_values.append(y_result.mean)
        model_names.append(model_name)

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
r"""
### 🧪 Unit Test: BenchmarkSuite.plot_results

**What we're testing**: That plot_results actually writes a comparison chart, and stays quiet when there is nothing to plot
**Why it matters**: A visualization step that silently produces no file is worse than none at all, because the report still claims a chart exists
**Expected**: benchmark_comparison.png exists and is non-empty after a run; an empty suite prints a message instead of raising
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
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        models = [MockModel("m1"), MockModel("m2")]
        suite = BenchmarkSuite(models, [{"d": "1"}], output_dir=tmp_dir)
        suite.run_full_benchmark(simulate=True)

        if MATPLOTLIB_AVAILABLE:
            # Agg is the headless backend: it writes files and never opens a
            # window, so plt.show() inside plot_results becomes a no-op here.
            plt.switch_backend("Agg")
            suite.plot_results(save_plots=True)

            # Deliberately no try/except around the call above. The whole point
            # of this test is that the chart is produced, and swallowing the
            # exception would let a plot_results that draws nothing pass.
            plot_path = Path(tmp_dir) / "benchmark_comparison.png"
            assert plot_path.exists(), (
                f"plot_results(save_plots=True) wrote no {plot_path.name}. "
                f"Output directory holds: "
                f"{sorted(f.name for f in Path(tmp_dir).iterdir())}"
            )
            assert plot_path.stat().st_size > 0, (
                f"{plot_path.name} was created but is empty"
            )
        else:
            # Without matplotlib the method must degrade, not raise.
            suite.plot_results(save_plots=True)

    # An empty suite reports that there is nothing to plot, and writes no file.
    with tempfile.TemporaryDirectory() as tmp_dir:
        suite2 = BenchmarkSuite([MockModel("m1")], [{"d": "1"}], output_dir=tmp_dir)
        suite2.plot_results()
        assert not (Path(tmp_dir) / "benchmark_comparison.png").exists(), (
            "plot_results wrote a chart even though the suite held no results"
        )

    print("✅ BenchmarkSuite.plot_results works correctly!")

if __name__ == "__main__":
    test_unit_benchsuite_plot()

# %% [markdown]
r"""
### BenchmarkSuite.generate_report: Actionable Insights

The `generate_report` method compiles all benchmark results into a structured
markdown report with system information, per-metric summaries, best performers,
trade-off analysis, and deployment recommendations.

| Report Generation Stage | Input Data | Generated Section | Key Technical Content |
| :--- | :--- | :--- | :--- |
| **1. System Metadata** | `system_info` | Environment Header | OS, CPU architecture, core count, Python runtime |
| **2. Per-Metric Summaries** | `results[metric]` | Score Breakdown | Mean, standard deviation, 95% CI, best performer |
| **3. Trade-Off Analysis** | Cross-metric vectors | Multi-Objective Ranking | Pareto-optimal models, efficiency ratios ($\text{Acc}/\text{ms}$) |
| **4. Recommendations** | Decision rules | Deployment Guidance | Recommended variants per deployment target |
| **5. File Persistence** | Formatted report buffer | Markdown Artifact | Saved to `output_dir / "benchmark_report.md"` |

We will construct this in three modular steps: formatting the per-metric results summary,
computing trade-off recommendations, and composing the complete report.
"""

# %% [markdown]
r"""
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
    ### BEGIN SOLUTION role="scaffold"
    lines = []
    lines.append("## Benchmark Results Summary")
    lines.append("")

    for metric_type, results in self.results.items():
        qualifier = " (estimated)" if any(r.metadata.get('estimated', False) for r in results.values()) else ""
        if any(r.metadata.get('simulated', False) for r in results.values()):
            qualifier += " (synthetic probe)"
        lines.append(f"### {metric_type.capitalize()} Results{qualifier}")
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
            lines.append(f"- **{model_name}**: {result.mean:.4f} ± {result.std:.4f}")
        lines.append("")

    return lines
    ### END SOLUTION

BenchmarkSuite._format_results_summary = _benchsuite_format_results_summary

# %% [markdown]
r"""
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
    ### BEGIN SOLUTION role="scaffold"
    lines = []
    lines.append("## Recommendations")
    lines.append("")
    if any(result.metadata.get('simulated', False)
           for result in self.results.get('accuracy', {}).values()):
        lines.append("Synthetic accuracy probe: no deployment recommendations.")
        return lines


    if len(self.results) >= 2:
        if 'latency' in self.results and 'accuracy' in self.results:
            lines.append("### Accuracy vs Speed Trade-off")

            latency_results = self.results['latency']
            accuracy_results = self.results['accuracy']

            scores = {}
            for model_name in latency_results.keys():
                acc_key = model_name if model_name in accuracy_results else None

                if acc_key:
                    lat_vals = [r.mean for r in latency_results.values()]
                    acc_vals = [r.mean for r in accuracy_results.values()]

                    norm_latency = 1 - (latency_results[model_name].mean - min(lat_vals)) / (max(lat_vals) - min(lat_vals) + 1e-8)
                    norm_accuracy = (accuracy_results[acc_key].mean - min(acc_vals)) / (max(acc_vals) - min(acc_vals) + 1e-8)

                    scores[model_name] = (norm_latency + norm_accuracy) / 2

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

        lines.append(f"- **For maximum accuracy**: Use {best_acc_model[0]}")
        lines.append(f"- **For minimum latency**: Use {best_lat_model[0]}")
        lines.append("- **For production deployment**: Consider the best overall trade-off model above")

    return lines
    ### END SOLUTION

BenchmarkSuite._format_recommendations = _benchsuite_format_recommendations

# %% [markdown]
r"""
#### Step 3: Compose the Full Report

Combine system info, results summary, and recommendations into a complete
markdown report and save it to disk.
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-report", "solution": true}
#| exporti
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
    ### BEGIN SOLUTION role="scaffold"
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
r"""
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
        suite.run_full_benchmark(simulate=True)

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
r"""
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
        suite.run_full_benchmark(simulate=True)

        lines = suite._format_recommendations()

        assert isinstance(lines, list), f"Expected list, got {type(lines)}"
        text = "\n".join(lines)
        assert "Recommendations" in text, "Should contain 'Recommendations'"

    print("✅ BenchmarkSuite._format_recommendations works correctly!")

if __name__ == "__main__":
    test_unit_benchsuite_format_recs()

# %% [markdown]
r"""
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
        results = suite.run_full_benchmark(simulate=True)

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
r"""
### MLPerf: Standardized Industry Benchmarking

MLPerf® is a trademark of MLCommons. This module provides MLPerf-style standardized
benchmarks that enable fair comparison across different systems, similar to how the
official MLPerf suite works for larger models. This is important for reproducible
research and industry adoption.

### Why Standardization Matters

Without standards, every team benchmarks differently:
- Different datasets, input sizes, measurement protocols
- Different accuracy metrics, latency definitions
- Different hardware configurations, software stacks

This makes it impossible to compare results across papers, products, or research groups.

### MLPerf Benchmark Architecture

| Architecture Layer | Core Responsibilities | Operational Specification |
| :--- | :--- | :--- |
| **1. Benchmark Definition** | Standardized task configuration | Fixed input tensors, evaluation datasets, accuracy targets ($\text{Acc}_{\text{target}}$), latency constraints ($\text{Lat}_{\text{max}}$) |
| **2. Execution Protocol** | Controlled measurement environment | Seeded input generation, untimed warmup iterations, monotonic interval sampling via `perf_counter` |
| **3. Compliance Engine** | Objective threshold gating | Evaluates $\text{Acc} \ge \text{Acc}_{\text{target}} \land \text{Lat} \le \text{Lat}_{\text{max}}$ to produce pass/fail verification |

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

**Image Classification**: Tiny image recognition (CIFAR-style)
- Input: 32×32 RGB images
- Task: Multi-class classification (10 classes)
- Target: 75% accuracy, <150ms latency

### Reproducibility Requirements

All MLPerf benchmarks use:
- **Fixed random seeds**: Deterministic input generation
- **Standardized hardware**: Reference implementations for comparison
- **Statistical validation**: Multiple runs with confidence intervals
- **Compliance reporting**: Machine-readable results format
"""

# %% [markdown]
r"""
### MLPerf.__init__: Configuring Standard Benchmarks

The `MLPerf` constructor sets up four standardized benchmark tasks, each with
fixed input shapes, target accuracy, and maximum latency thresholds. Using a
fixed random seed ensures reproducible results across different systems:

| Benchmark Task | Input Tensor Shape | Domain & Modality | Accuracy Target | Latency Ceiling |
| :--- | :--- | :--- | :--- | :--- |
| `keyword_spotting` | `(1, 16000)` | 1-second 16kHz audio stream | $\ge 90\%$ | $< 100\text{ ms}$ |
| `visual_wake_words` | `(1, 96, 96, 3)` | 96×96 RGB vision camera | $\ge 80\%$ | $< 200\text{ ms}$ |
| `anomaly_detection` | `(1, 640)` | Multi-channel acoustic sensor | $\ge 85\%$ | $< 50\text{ ms}$ |
| `image_classification`| `(1, 32, 32, 3)` | 32×32 CIFAR-10 RGB stream | $\ge 75\%$ | $< 150\text{ ms}$ |
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-init", "solution": true}
#| export
class MLPerf:
    """
    MLPerf-style standardized benchmarking for edge ML systems.

    MLPerf® is a trademark of MLCommons. Used here purely for educational purposes.
    This module teaches the principles of MLPerf-style benchmarking through a
    simplified suite inspired by MLPerf Tiny.

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
        1. Store random_seed; every phase rebuilds its generator from it
        2. Define benchmark configs with input_shape, target_accuracy, max_latency_ms

        HINTS:
        - Each benchmark is a dict with 'input_shape', 'target_accuracy', 'max_latency_ms', 'description'
        - keyword_spotting uses (1, 16000) for 1 second of 16kHz audio
        - Store the seed itself, not a generator. Each phase calls
          np.random.default_rng(self.random_seed), so running the same
          benchmark twice draws the same inputs and the same synthetic labels.
          A seed that no phase reads makes random_seed a lie, and a benchmark
          whose seed does nothing is not reproducible no matter what it prints
        """
        ### BEGIN SOLUTION role="scaffold"
        self.random_seed = random_seed

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
r"""
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
r"""
### MLPerf._run_latency_test: Measuring Inference Latency

This helper runs the latency measurement phase: warmup, then timed inference
for each test input. Returns lists of latencies (ms) and model predictions:

| Test Protocol Step | Execution Action | Statistical / Systems Impact |
| :--- | :--- | :--- |
| **Warmup Phase** | Execute $\max(1, \lfloor N / 10 \rfloor)$ inputs | Heats instruction caches, registers, and memory controllers (untimed) |
| **Monotonic Timing** | `with precise_timer() as timer:` | Samples userspace monotonic clock around single model invocation |
| **Result Logging** | `latencies.append(timer.elapsed * 1000)` | Converts elapsed seconds to milliseconds for per-sample distribution |
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-latency", "solution": true}
#| export
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
    ### BEGIN SOLUTION role="scaffold"
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
                    raise TypeError(f"{type(model).__name__} has no forward(), predict(), or __call__ to benchmark")
            except Exception as exc:
                raise RuntimeError(f"{benchmark_name}: model failed on input {i}: {exc}") from exc

            predictions.append(output)

        latencies.append(timer.elapsed * 1000)  # Convert to ms

    return latencies, predictions
    ### END SOLUTION

MLPerf._run_latency_test = _mlperf_run_latency_test

# %% [markdown]
r"""
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
            return rng.random(2)

    perf = MLPerf(random_seed=42)
    model = MockModel("test")

    test_inputs = [rng.standard_normal((1, 16000)).astype(np.float32) for _ in range(5)]
    latencies, predictions = perf._run_latency_test(model, test_inputs, 'keyword_spotting', 5)

    assert len(latencies) == 5
    assert len(predictions) == 5
    assert all(lat > 0 for lat in latencies)

    print("✅ MLPerf._run_latency_test works correctly!")

if __name__ == "__main__":
    test_unit_mlperf_latency()

# %% [markdown]
r"""
### MLPerf._run_accuracy_test: Evaluating Prediction Quality

This helper calculates accuracy by comparing model predictions against synthetic
ground truth labels. It handles both binary classification (keyword spotting,
visual wake words, anomaly detection) and multi-class classification (image
classification).

We'll build this in two steps: first a helper to extract a clean prediction
array from various output formats, then the accuracy calculation itself.
"""

# %% [markdown]
r"""
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
    ### BEGIN SOLUTION role="scaffold"
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
r"""
#### Step 2: Calculate Accuracy

Use _extract_pred_array to get clean predictions, then compare against
synthetic ground truth for binary and multi-class tasks.

Expect chance-level numbers here (50% binary, 10% multi-class) when the model
under test has nothing to do with the synthetic labels. Resist the urge to
"fix" that by nudging the score toward the compliance target. A benchmark that
adjusts its output until the result looks plausible has stopped measuring.
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-accuracy", "solution": true}
#| export
def _mlperf_run_accuracy_test(self, model: Any, predictions: List[Any],
                                    benchmark_name: str, num_runs: int,
                                    labels: Optional[np.ndarray] = None) -> float:
    """
    Calculate accuracy from predictions against the reference labels.

    Pass `labels` (one per test input) to score against a real test set. When
    none are given the ground truth is SYNTHETIC: a model with no relationship
    to those labels scores at chance -- 50% on the binary tasks, 10% on the
    10-class task -- and that is the correct, informative result. A real
    MLPerf submission uses the benchmark's own labeled dataset; the protocol
    around it does not change.

    TODO: Implement accuracy calculation using _extract_pred_array helper

    APPROACH:
    1. Use the labels given, or draw synthetic ones from self.random_seed so the run repeats
    2. For binary tasks: use _extract_pred_array, compare class scores
    3. For multi-class: use _extract_pred_array, take argmax
    4. Return the agreement rate. Nothing else.

    HINTS:
    - keyword_spotting, visual_wake_words, and anomaly_detection are binary (2 classes)
    - image_classification has 10 classes
    - Do not adjust the score by the model's name. A benchmark that rewards a
      model for calling itself 'efficient' measures marketing, not the model
    """
    ### BEGIN SOLUTION role="scaffold"
    binary = benchmark_name in ['keyword_spotting', 'visual_wake_words', 'anomaly_detection']
    num_classes = 2 if binary else 10
    if num_runs <= 0 or len(predictions) != num_runs:
        raise ValueError("Accuracy needs one prediction per test input")
    rng = np.random.default_rng(self.random_seed)
    true_labels = np.asarray(labels) if labels is not None else rng.integers(0, num_classes, num_runs)
    if (true_labels.shape != (num_runs,)
            or not np.issubdtype(true_labels.dtype, np.integer)
            or np.any(true_labels < 0) or np.any(true_labels >= num_classes)):
        raise ValueError("labels must contain one integer class index in the task's range per input")

    predicted_labels = []
    for pred in predictions:
        raw = np.asarray(pred.data if isinstance(pred, Tensor) else pred)
        # Each timed input represents one example: accept a score vector or a
        # singleton batch, never flatten several examples into one prediction.
        if raw.ndim == 2 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim == 0 and binary:
            raw = raw.reshape(1)
        allowed_widths = (1, 2) if binary else (10,)
        if raw.ndim != 1 or raw.size not in allowed_widths:
            raise ValueError(f"Prediction must contain {allowed_widths} class scores for one example")
        if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw) or not np.all(np.isfinite(raw)):
            raise ValueError("Prediction class scores must be finite real numbers")
        raw = _extract_pred_array(raw)
        if binary and raw.size == 1:
            if not 0 <= raw[0] <= 1:
                raise ValueError("A single binary score must be a probability in [0, 1]")
            predicted_labels.append(int(raw[0] > 0.5))
        else:
            predicted_labels.append(int(np.argmax(raw)))

    return float(np.mean(true_labels == predicted_labels))
    ### END SOLUTION

MLPerf._run_accuracy_test = _mlperf_run_accuracy_test

# %% [markdown]
r"""
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
r"""
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
    predictions = [rng.random(2) for _ in range(10)]
    accuracy = perf._run_accuracy_test(model, predictions, 'keyword_spotting', 10)
    assert 0 <= accuracy <= 1

    # Multi-class predictions
    predictions_mc = [rng.random(10) for _ in range(10)]
    accuracy_mc = perf._run_accuracy_test(model, predictions_mc, 'image_classification', 10)
    assert 0 <= accuracy_mc <= 1

    print("✅ MLPerf._run_accuracy_test works correctly!")

if __name__ == "__main__":
    test_unit_mlperf_accuracy()

# %% [markdown]
r"""
### MLPerf.run_standard_benchmark: Complete Benchmark Execution

This method orchestrates a complete standardized benchmark: input generation,
latency testing, accuracy evaluation, and compliance determination. It composes
the `_run_latency_test` and `_run_accuracy_test` helpers into the full protocol:

| Pipeline Step | Mechanism | Verification Target |
| :--- | :--- | :--- |
| **Config Resolution** | Task lookup in `self.benchmarks` | Retrieves target accuracy and latency threshold |
| **Input Synthesis** | Seeded random generation (`seed=42`) | Generates $N$ reproducible synthetic tensors or consumes empirical inputs |
| **Latency Sampling** | `_run_latency_test()` | 10% warmup discard, per-run latency distribution |
| **Quality Evaluation** | `_run_accuracy_test()` | Top-1 accuracy score across measured predictions |
| **Compliance Gating** | Threshold comparison | $\text{compliant} \iff (\text{accuracy} \ge \text{target}) \land (\text{mean\_latency} \le \text{limit})$ |
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-run", "solution": true}
#| exporti
def mlperf_run_standard_benchmark(self, model: Any, benchmark_name: str,
                              num_runs: int = 100,
                              test_inputs: Optional[List[Any]] = None,
                              labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Run a standardized MLPerf benchmark.

    Pass your own `test_inputs` (a list of Tensors of the task's input shape)
    and `labels` to measure a real test set; without them the inputs are
    deterministic random data and the labels synthetic, so the protocol runs
    but the accuracy number means nothing.

    TODO: Orchestrate input generation, latency test, accuracy test, and compliance check

    APPROACH:
    1. Validate benchmark_name and get config
    2. Use the test inputs given, or generate deterministic ones using seeded random
    3. Call self._run_latency_test() for timing
    4. Call self._run_accuracy_test() for quality
    5. Compile results with compliance determination: the latency bar is checked
       at the 99th percentile, as MLPerf's server scenario bounds the tail, not the mean

    HINTS:
    - Seed one generator from self.random_seed, and draw every input from it
    - Audio data: rng.standard_normal, Image data: rng.integers(0,256)/255
    - compliant = accuracy_met AND latency_met
    """
    ### BEGIN SOLUTION role="scaffold"
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

    # Use the caller's test set, or generate standardized test inputs
    # (as Tensors for TinyTorch model compatibility)
    input_shape = config['input_shape']
    if test_inputs is not None:
        test_inputs = list(test_inputs)
        num_runs = len(test_inputs)
    else:
        # Seeded from self.random_seed, not a hardcoded constant: the same
        # inputs on every run of this benchmark, and a different set only when
        # the caller asks for a different seed. A hardcoded seed here would make
        # the constructor's random_seed argument decorative.
        input_rng = np.random.default_rng(self.random_seed)
        test_inputs = []
        for _ in range(num_runs):
            if len(input_shape) == 2:  # Audio/sequence data (keyword_spotting, anomaly_detection)
                arr = input_rng.standard_normal(input_shape).astype(np.float32)
            else:  # Image data (visual_wake_words, image_classification) - use CHW for Conv2d
                arr = input_rng.integers(0, 256, input_shape).astype(np.float32) / 255.0
                if arr.ndim == 4 and arr.shape[-1] == 3:  # (B,H,W,C) -> (B,C,H,W)
                    arr = np.transpose(arr, (0, 3, 1, 2))
            test_inputs.append(Tensor(arr))

    # Run latency and accuracy tests using helpers
    if num_runs <= 0:
        raise ValueError("Benchmark needs at least one input")
    if labels is not None and np.asarray(labels).shape != (num_runs,):
        raise ValueError("labels must contain one class index per test input")
    latencies, predictions = self._run_latency_test(model, test_inputs, benchmark_name, num_runs)
    accuracy = self._run_accuracy_test(model, predictions, benchmark_name, num_runs, labels)

    # Compile results. The latency bar is checked at the tail (p99), because a
    # user waits on the slow requests; the mean is reported beside it.
    mean_latency = float(np.mean(latencies))
    p99_latency = float(np.percentile(latencies, 99))
    accuracy_met = bool(accuracy >= config['target_accuracy'])
    latency_met = bool(p99_latency <= config['max_latency_ms'])

    results = {
        'synthetic_labels': labels is None,
        'official_mlperf': False,
        'benchmark_name': benchmark_name,
        'model_name': getattr(model, 'name', 'unknown_model'),
        'accuracy': float(accuracy),
        'mean_latency_ms': mean_latency,
        'std_latency_ms': float(np.std(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p90_latency_ms': float(np.percentile(latencies, 90)),
        'p99_latency_ms': p99_latency,
        'max_latency_ms': float(np.max(latencies)),
        'throughput_fps': float(1000 / mean_latency),
        'target_accuracy': float(config['target_accuracy']),
        'target_latency_ms': float(config['max_latency_ms']),
        'accuracy_met': accuracy_met,
        'latency_met': latency_met,
        'compliant': accuracy_met and latency_met and labels is not None,
        'num_runs': int(num_runs),
        'random_seed': int(self.random_seed)
    }

    print(f"   Results: {accuracy:.1%} accuracy, {mean_latency:.1f}ms mean latency, {p99_latency:.1f}ms p99")
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
r"""
### 🧪 Unit Test: MLPerf.run_standard_benchmark

**What we're testing**: Complete benchmark execution, and that random_seed actually controls the data
**Why it matters**: A benchmark that prints a seed it never uses reports a number nobody can reproduce
**Expected**: Results dict with all required metrics and compliance flags; equal seeds give identical inputs, different seeds give different ones
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
                return rng.random(2)
            return rng.random(10)

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

    # random_seed must control the generated inputs, not merely appear in the
    # report. This model records what it was actually fed, which is the only way
    # to see the difference: a benchmark that hardcodes its own seed still
    # returns a plausible-looking results dict.
    class RecordingModel:
        def __init__(self):
            self.seen = []
        def forward(self, x):
            self.seen.append(np.asarray(x.data).copy())
            return Tensor(np.zeros(2, dtype=np.float32))

    same_a, same_b, different = RecordingModel(), RecordingModel(), RecordingModel()
    MLPerf(random_seed=42).run_standard_benchmark(same_a, 'keyword_spotting', num_runs=3)
    MLPerf(random_seed=42).run_standard_benchmark(same_b, 'keyword_spotting', num_runs=3)
    MLPerf(random_seed=1234).run_standard_benchmark(different, 'keyword_spotting', num_runs=3)

    assert np.array_equal(same_a.seen[0], same_b.seen[0]), (
        "Two runs at random_seed=42 were fed different inputs, so the benchmark "
        "does not repeat"
    )
    assert not np.array_equal(same_a.seen[0], different.seen[0]), (
        "Changing random_seed from 42 to 1234 did not change the benchmark "
        "inputs. The seed is decorative and the reported number is not "
        "reproducible by anyone who passes a different one"
    )

    print("✅ MLPerf.run_standard_benchmark works correctly!")

if __name__ == "__main__":
    test_unit_mlperf_run()

# %% [markdown]
r"""
### MLPerf.generate_compliance_report: Scorecard Generation

The compliance report compiles results from multiple benchmarks into both
machine-readable JSON and human-readable markdown formats, with overall
compliance determination:

| Report Stage | Processing Operation | Generated Artifact |
| :--- | :--- | :--- |
| **Statistical Aggregation**| Count compliant vs non-compliant tasks | Compliance ratio: $\frac{N_{\text{compliant}}}{N_{\text{total}}}$ |
| **Structured Output** | `_compile_report_data()` | Machine-readable JSON dictionary |
| **Human-Readable Summary** | `_format_summary_markdown()` | Formatted Markdown scorecard table |
| **Overall Verdict** | Evaluate unanimous compliance | Status tag: `COMPLIANT` vs `NON-COMPLIANT` |

We will build this in two modular steps: compiling structured report data,
then formatting it into a human-readable summary.
"""

# %% [markdown]
r"""
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
    2. Loop through results; failed runs remain in the total count
    3. Count compliant benchmarks and compute compliance_rate
    4. Store per-benchmark metrics

    HINTS:
    - overall_compliant = compliance_rate == 1.0
    - Set model_name from first successful result
    """
    ### BEGIN SOLUTION role="scaffold"
    compliant_benchmarks = []
    total_benchmarks = len(results)

    report_data = {
        'benchmark_suite': 'TinyTorch classroom benchmark (not official MLPerf)',
        'random_seed': self.random_seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': 'unknown',
        'benchmarks': {},
        'summary': {}
    }

    for benchmark_name, result in results.items():
        if 'error' not in result:
            if result.get('compliant', False):
                compliant_benchmarks.append(benchmark_name)

            if report_data['model_name'] == 'unknown':
                report_data['model_name'] = result.get('model_name', 'unknown')

            report_data['benchmarks'][benchmark_name] = {
                'synthetic_labels': result.get('synthetic_labels', False),
                'official_mlperf': False,
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
r"""
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
    ### BEGIN SOLUTION role="scaffold"
    summary_lines = []
    summary_lines.append("# TinyTorch Classroom Benchmark Report (not official MLPerf)")
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
r"""
#### Step 3: Compose the Full Compliance Report

Combine data compilation, JSON serialization, and summary formatting.
"""

# %% nbgrader={"grade": false, "grade_id": "tinymlperf-scorecard", "solution": true}
#| exporti
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
    ### BEGIN SOLUTION role="scaffold"
    # Compile structured report data
    report_data = self._compile_report_data(results)

    # Save JSON report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)

    # Generate and save human-readable summary
    summary_text = self._format_compliance_summary(report_data)

    summary_path = output_path.replace('.json', '_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"📄 MLPerf report saved to {output_path}")
    print(f"📄 Summary saved to {summary_path}")

    return summary_text
    ### END SOLUTION

MLPerf.generate_compliance_report = mlperf_generate_compliance_report

# %% [markdown]
r"""
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
r"""
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
r"""
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
                    return rng.random(2)  # Binary classification
                else:  # Image
                    return rng.random(10)  # Multi-class
            return rng.random(2)

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
        assert "Classroom Benchmark Report" in summary
        assert "Compliance Rate" in summary

    print("✅ MLPerf works correctly!")

if __name__ == "__main__":
    test_unit_mlperf()

# %% [markdown]
r"""
## 🔧 Integration: Building Complete Benchmark Workflows

Now we'll integrate all our benchmarking components into complete workflows that demonstrate professional ML systems evaluation. This integration shows how to combine statistical rigor with practical insights.

The integration layer connects individual measurements into actionable engineering insights. This is where benchmarking becomes a decision-making tool rather than just data collection.

### Workflow Architecture

| Stage | Input Entities | Transformations | Practical Deployment Questions |
| :--- | :--- | :--- | :--- |
| **1. Model Variants** | Base, Quantized, Pruned, Distilled | Standardized test execution | "What are the baseline performance envelopes?" |
| **2. Optimization Profiling** | Multi-metric benchmarks | Compute $\Delta\text{Acc}$, Speedup, Memory savings | "Which optimization yields the highest efficiency ratio?" |
| **3. Use-Case Mapping** | Hardware & SLA constraints | Multi-objective optimization | "Which model satisfies edge vs server deployment constraints?" |
"""

# %% [markdown]
r"""
### Optimization Comparison Engine

Before implementing the comparison function, let's understand what makes optimization comparison challenging and valuable.

### Why Optimization Comparison is Complex

When you optimize a model, you make trade-offs across multiple dimensions simultaneously:

| Optimization Technique | Accuracy Impact ($\Delta \text{Acc}$) | Latency Speedup | Memory Reduction | Energy Savings | Primary Trade-Off |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Quantization** (INT8 / FP8) | $-5\%$ | $2.1\times$ | $2.0\times$ | $1.8\times$ | Precision loss in low dynamic range |
| **Structured Pruning** | $-2\%$ | $1.4\times$ | $3.2\times$ | $1.3\times$ | Sparse memory access vs density |
| **Knowledge Distillation** | $-8\%$ | $1.9\times$ | $1.5\times$ | $1.7\times$ | Dark knowledge transfer fidelity |

The challenge: Which is "best"? It depends entirely on your deployment constraints.

### Multi-Objective Decision Framework

Our comparison engine implements a decision framework that:

1. **Measures all dimensions**: Don't optimize in isolation
2. **Calculates efficiency ratios**: Accuracy per MB, accuracy per ms
3. **Identifies Pareto frontiers**: Models that aren't dominated in all metrics
4. **Generates use-case recommendations**: Tailored to specific constraints

### Formal Recommendation Objectives

| Deployment Regime | Optimization Formulation | Constraint Boundary | Target Domain |
| :--- | :--- | :--- | :--- |
| **Latency-Critical** | $\min_{\theta} \text{Latency}(\theta)$ | $\text{Accuracy}(\theta) \ge \text{Acc}_{\text{target}}$ | Autonomous driving, real-time audio |
| **Memory-Constrained** | $\min_{\theta} \text{PeakMemory}(\theta)$ | $\text{Accuracy}(\theta) \ge \text{Acc}_{\text{target}}$ | Microcontrollers, wearable IoT |
| **Accuracy-Preservation** | $\max_{\theta} \text{Accuracy}(\theta)$ | $\text{Latency}(\theta) \le \text{Lat}_{\text{max}}$ | Medical diagnostics, legal review |
| **Balanced Deployment** | $\max_{\theta} \big[\alpha \frac{\text{Acc}}{\text{Acc}_0} + \beta \frac{\text{Lat}_0}{\text{Lat}} + \gamma \frac{\text{Mem}_0}{\text{Mem}}\big]$ | Multi-objective budget | Edge mobile, client-side web |

This principled approach ensures recommendations match real deployment needs.
"""

# %% [markdown]
r"""
### _collect_base_metrics: Extracting Baseline Performance

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
    2. Find the result keyed by base_name, or base_name plus a metric suffix
    3. Store result.mean in a dict keyed by metric type

    HINTS:
    - Keys are model names, sometimes with a suffix such as base_latency_ms
    """
    ### BEGIN SOLUTION role="scaffold"
    base_metrics = {}
    for metric_type, results in benchmark_results.items():
        if base_name in results:
            base_metrics[metric_type] = results[base_name].mean
            continue
        for model_name, result in results.items():
            if model_name.startswith(base_name + "_"):
                base_metrics[metric_type] = result.mean
                break
    return base_metrics
    ### END SOLUTION

# %% [markdown]
r"""
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
r"""
### _calculate_improvements: Computing Speedup and Retention Ratios

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
    ### BEGIN SOLUTION role="scaffold"
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
        if base_metrics['accuracy'] > 0:
            improvements['accuracy_retention'] = opt_metrics['accuracy'] / base_metrics['accuracy']
        else:
            improvements['accuracy_retention'] = 1.0

    return improvements
    ### END SOLUTION

# %% [markdown]
r"""
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
r"""
### _generate_recommendations: Deployment-Specific Guidance

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
    ### BEGIN SOLUTION role="scaffold"
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
r"""
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
r"""
### analyze_optimization_techniques: Composition Function

This is the main entry point that composes `_collect_base_metrics`,
`_calculate_improvements`, and `_generate_recommendations` into a complete
optimization comparison workflow:

| Pipeline Step | Module Subroutine | Computed Metric / Transformation |
| :--- | :--- | :--- |
| **1. Full Benchmark** | `BenchmarkSuite.run_full_benchmark()` | Runs latency, accuracy, memory, and energy across all models |
| **2. Baseline Extraction** | `_collect_base_metrics()` | Isolates baseline vector $\mathbf{v}_{\text{base}} = [\mu_{\text{lat}}, \mu_{\text{acc}}, \mu_{\text{mem}}, \mu_{\text{eng}}]$ |
| **3. Relative Deltas** | `_calculate_improvements()` | Computes speedup ratios, accuracy changes, and memory reduction factors |
| **4. Policy Recommendation** | `_generate_recommendations()` | Evaluates constrained optimization criteria to assign models to targets |
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-comparison", "solution": true}
#| export
def analyze_optimization_techniques(base_model: Any, optimized_models: List[Any],
                                  datasets: List[Any], simulate: bool = False,
                                  input_shape: Tuple[int, ...] = (1, 28, 28)) -> Dict[str, Any]:
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
        input_shape: Representative inference batch shape for latency and memory

    Returns:
        Dictionary with 'base_metrics', 'optimized_results', 'improvements', 'recommendations'

    EXAMPLE:
    >>> results = analyze_optimization_techniques(base_model, [quant, pruned], datasets)
    >>> print(results['recommendations'])
    """
    ### BEGIN SOLUTION role="scaffold"
    all_models = [base_model] + optimized_models
    suite = BenchmarkSuite(all_models, datasets)

    print("🧪 Running optimization comparison benchmark...")
    benchmark_results = suite.run_full_benchmark(simulate=simulate, input_shape=input_shape)

    # Extract base model performance using helper
    base_name = suite.benchmark.model_names[0]
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

    for opt_index, opt_model in enumerate(optimized_models, start=1):
        opt_name = suite.benchmark.model_names[opt_index]

        # Find results for this optimized model
        opt_metrics = {}
        for metric_type, results in benchmark_results.items():
            for model_name, result in results.items():
                if model_name == opt_name:
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
    simulated = any(r.metadata.get('simulated', False) for r in benchmark_results['accuracy'].values())
    recommendations = {} if simulated else _generate_recommendations(comparison_results['improvements'])
    comparison_results['simulated'] = simulated
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
r"""
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
    results = analyze_optimization_techniques(base_model, [quantized_model, pruned_model], datasets, simulate=True)

    # Verify results structure
    assert 'base_model' in results
    assert 'optimized_results' in results
    assert 'improvements' in results
    assert 'recommendations' in results

    # Verify improvements were calculated
    assert len(results['improvements']) == 2  # Two optimized models

    # Synthetic probe results must not become deployment recommendations.
    assert results['simulated'] is True
    assert results['recommendations'] == {}

    print("✅ analyze_optimization_techniques works correctly!")

if __name__ == "__main__":
    test_unit_optimization_comparison()

# %% [markdown]
r"""
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
        measurements = rng.normal(true_latency, noise_std, n_samples)
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

    import copy
    from tinytorch.perf.quantization import QuantizedLinear
    from tinytorch.perf.compression import magnitude_prune

    # One dense layer, then the two optimizations you built in Modules 15 and 16
    profiler = Profiler()
    base = Linear(512, 256)
    x = Tensor(rng.standard_normal((32, 512)).astype(np.float32))
    reference = base.forward(x).data

    quantized = QuantizedLinear(base)
    pruned = magnitude_prune(copy.deepcopy(base), sparsity=0.7)

    def stored_mb(model):
        """Modeled packed INT8 bytes; actual dense parameter bytes otherwise."""
        if isinstance(model, QuantizedLinear):
            return model.memory_usage()['quantized_bytes'] / (1024 * 1024)
        return sum(p.data.nbytes for p in model.parameters()) / (1024 * 1024)

    print("\nMeasured on one Linear(512, 256) layer:\n")
    print(f"{'Technique':<20} {'Latency (ms)':<14} {'Payload (MB)':<13} {'Output error'}")
    print("-" * 60)

    for name, model in [('Baseline', base), ('Quantization (INT8)', quantized), ('Pruning (70%)', pruned)]:
        latency = profiler.measure_latency(model, x, warmup=3, iterations=10)
        output = model.forward(x).data
        rel_error = np.mean((output - reference) ** 2) / np.mean(reference ** 2)
        print(f"{name:<20} {latency:<14.3f} {stored_mb(model):<13.3f} {rel_error:.2e}")

    print("\n💡 Key Insights:")
    print("   • Baseline/pruned payloads count actual dense parameter bytes; zeros still occupy storage")
    print("   • INT8 payload is a modeled packed representation, not current Tensor storage")
    print("   • Latency is measured here; NumPy neither uses INT8 kernels nor skips pruned zeros")
    print("   • Output error is the price; accuracy on a task is what you must measure next")
    print("   • No single optimization dominates: pick by the deployment constraint that binds")

if __name__ == "__main__":
    analyze_optimization_tradeoffs()

# %% [markdown]
r"""
### MLPerf Principles: Industry-Standard Benchmarking

MLPerf (created by MLCommons) is the industry-standard ML benchmarking framework. Understanding these principles grounds your capstone competition in professional methodology.

### Core Principles

**Reproducibility:** Fixed hardware specs, software versions, random seeds, and multiple runs for statistical validity.

**Standardization:** Fixed models and datasets enable fair comparison. MLPerf has two divisions:
- **Closed:** Same models/datasets, optimize systems (hardware/software)
- **Open:** Modify models/algorithms, show innovation

**MLPerf Tiny:** Edge-device benchmarks (<1MB models, <100ms latency, <10mW power) that inspire the capstone.

### Key Takeaways

1. Document everything for reproducibility
2. Use same baseline for fair comparison
3. Measure multiple metrics (accuracy, latency, memory, energy)
4. Optimize for real deployment constraints

The capstone project follows MLPerf-style principles!
"""

# %% [markdown]
r"""
### Combination Strategies

Strategic optimization combines multiple techniques for different performance goals. The order matters: quantize-then-prune may preserve accuracy better, while prune-then-quantize may be faster.

### Ablation Studies

Professional ML engineers use ablation studies to understand each optimization's contribution (illustrative numbers):

```
Baseline:           Accuracy: 89%, Latency: 45ms, Memory: 12MB
+ Quantization:     Accuracy: 88%, Latency: 30ms, Memory: 3MB   (Δ: -1%, -33%, -75%)
+ Pruning:          Accuracy: 87%, Latency: 22ms, Memory: 2MB   (Δ: -1%, -27%, -33%)
+ Kernel Fusion:    Accuracy: 87%, Latency: 18ms, Memory: 2MB   (Δ: 0%, -18%, 0%)
```

You'll apply these strategies with specific optimization targets in Module 20's capstone project.
"""

# %% [markdown]
r"""
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
    test_unit_simulated_accuracy()
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
            time.sleep(max(0, base_time + rng.normal(0, variance)))

            # Simulate memory usage
            if hasattr(x, 'shape'):
                temp_size = int(np.prod(x.shape) * memory_factor)
                temp_data = rng.standard_normal(temp_size)
                _ = np.sum(temp_data)  # Use the data

            return x

        def evaluate(self, dataset):
            # Simulate evaluation
            base_acc = self.characteristics.get('base_accuracy', 0.85)
            return base_acc + rng.normal(0, 0.02)

        def parameters(self):
            # Simulate parameter count - return Tensor objects for compatibility
            from tinytorch.core.tensor import Tensor
            param_count = self.characteristics.get('param_count', 1000000)
            return [Tensor(rng.standard_normal(param_count))]

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
    results = suite.run_full_benchmark(simulate=True)

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
    class BinaryClassifier:
        def forward(self, x):
            return Tensor([[0.25, 0.75]])
    perf_results = perf.run_standard_benchmark(BinaryClassifier(), 'keyword_spotting', num_runs=5)

    required_keys = ['accuracy', 'mean_latency_ms', 'compliant', 'target_accuracy']
    assert all(key in perf_results for key in required_keys)
    assert 0 <= perf_results['accuracy'] <= 1
    assert perf_results['mean_latency_ms'] > 0

    # Test 5: Optimization comparison
    print("  Testing optimization comparison...")
    comparison_results = analyze_optimization_techniques(
        models[0], models[1:], datasets[:1], simulate=True
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
r"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of benchmarking and performance engineering:

### Question 1: Statistical Confidence in Measurements
You implemented `BenchmarkResult` with confidence intervals for measurements.
If you run 20 trials and get mean latency $\mu = 5.2\text{ ms}$ with sample standard deviation $s = 0.8\text{ ms}$:

- **What's the 95% confidence interval for the true mean?**
  $$\text{SE} = \frac{s}{\sqrt{n}} = \frac{0.8}{\sqrt{20}} = \frac{0.8}{4.4721} \approx 0.1789\text{ ms}$$
  Using the normal critical value ($z_{0.975} = 1.96$):
  $$\text{Margin of Error} = 1.96 \times 0.1789 \approx 0.3506\text{ ms} \implies [4.85\text{ ms}, 5.55\text{ ms}]$$
  *(Using Student's $t$ distribution with $\nu = 19$ degrees of freedom, $t_{19, 0.975} \approx 2.093$, yielding margin $2.093 \times 0.1789 \approx 0.3744\text{ ms} \implies [4.83\text{ ms}, 5.57\text{ ms}]$.)*
- **How many more trials would you need to halve the confidence interval width?**
  $$\text{Width} \propto \frac{1}{\sqrt{n}} \implies \frac{\text{Width}_{\text{new}}}{\text{Width}_{\text{old}}} = \frac{1}{2} \implies \sqrt{\frac{n_{\text{new}}}{n_{\text{old}}}} = 2 \implies n_{\text{new}} = 4 \times n_{\text{old}} = 4 \times 20 = \mathbf{80\text{ total trials}}$$
  *(You would need $80 - 20 = \mathbf{60\text{ additional trials}}$).*

---

### Question 2: Measurement Overhead Analysis
Your `precise_timer` context manager has microsecond precision, but models run for milliseconds.
For a model that takes $1.0\text{ ms}$ to execute:

- **If timer overhead is $10\text{ }\mu\text{s}$, what's the relative error?**
  $$\text{Relative Error} = \frac{\delta_{\text{timer}}}{T_{\text{exec}}} = \frac{10\text{ }\mu\text{s}}{1000\text{ }\mu\text{s}} \times 100\% = \mathbf{1.0\%}$$
- **At what model latency does timer overhead become negligible ($<1\%$)?**
  $$\frac{\delta_{\text{timer}}}{T_{\text{exec}}} < 0.01 \implies T_{\text{exec}} > \frac{10\text{ }\mu\text{s}}{0.01} = 1000\text{ }\mu\text{s} = \mathbf{1.0\text{ ms}}$$
  *Systems implication: For micro-benchmarking sub-millisecond operators (such as individual activation layers or tensor slicing taking $< 50\text{ }\mu\text{s}$), timing individual iterations introduces unacceptable distortion ($> 20\%$). In such regimes, you must amortize overhead by executing an internal loop of $K = 100\text{--}1000$ iterations inside a single timer block and dividing.*

---

### Question 3: Benchmark Configuration Trade-offs
The `BenchmarkSuite` class uses configurable `warmup_runs` and `measurement_runs` parameters
(with `DEFAULT_WARMUP_RUNS = 5` and `DEFAULT_MEASUREMENT_RUNS = 10` as defaults).
For a CI/CD regression testing pipeline that executes 100 model benchmarks per daily build:

- **Fast config ($3\text{ s}$ each):** $100 \times 3\text{ s} = 300\text{ s} = \mathbf{5.0\text{ minutes}}$ total daily pipeline execution.
- **Accurate config ($15\text{ s}$ each):** $100 \times 15\text{ s} = 1500\text{ s} = \mathbf{25.0\text{ minutes}}$ total daily pipeline execution.
- **What's the key trade-off you're making?**
  **Statistical precision vs development velocity** (detection threshold for small performance regressions vs rapid engineer feedback cycle). A 5-minute suite allows commit-level pre-merge gating; a 25-minute suite is typically reserved for nightly integration builds.

---

### Question 4: MLPerf Compliance Metrics
You implemented MLPerf-style standardized benchmarks with target thresholds.
If an edge candidate model achieves 89% accuracy (target: 90%) and 120ms latency (target: <100ms):

- **Is it compliant?** **No**. MLPerf compliance is a strict conjunction ($\text{Acc} \ge \text{Target} \land \text{Lat} \le \text{Threshold}$). Both constraints are violated ($89\% < 90\%$ and $120\text{ ms} > 100\text{ ms}$).
- **Which constraint is more critical for edge deployment?** **Latency**. Latency on edge devices is a hard real-time physical deadline dictated by sensor sampling rates (e.g. 10 fps camera stream requires $< 100\text{ ms}$ processing), UI responsiveness, or watchdog timeouts. Dropping below the latency ceiling causes dropped sensor frames or system lockup, whereas an accuracy delta of $1\%$ is typically tolerable.
- **How would you prioritize optimization?** **Latency-first**. First compress/accelerate the model to reliably meet the $\le 100\text{ ms}$ deadline with buffer room, then tune hyper-parameters or calibration datasets to recover the remaining accuracy gap.

---

### Question 5: Optimization Comparison Analysis
Your `analyze_optimization_techniques()` generates recommendations for different use cases.
Given three optimized models:
- **Quantized**: $0.8\times$ memory footprint (20% reduction), $2.0\times$ speedup, $0.95\times$ accuracy
- **Pruned**: $0.3\times$ memory footprint (70% reduction), $1.5\times$ speedup, $0.98\times$ accuracy
- **Distilled**: $0.6\times$ memory footprint (40% reduction), $1.8\times$ speedup, $0.92\times$ accuracy

For a mobile app with a 50MB model size limit and a strict $< 100\text{ ms}$ latency requirement:
- **Which optimization offers best memory reduction?** **Pruned** ($0.3\times$ original memory footprint, yielding a $70\%$ reduction).
- **Which balances all constraints best?** **Pruned**. It achieves the greatest memory compression ($0.3\times$), provides a respectable $1.5\times$ speedup, and preserves $98\%$ of base accuracy ($0.98\times$).
- **What's the key insight about optimization trade-offs?** **No free lunch / empirical Pareto measurement guides decisions**. No single technique dominates across all axes simultaneously; empirical Pareto frontiers reveal non-dominated configurations that match specific hardware budget envelopes.
"""

# %% [markdown]
r"""
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
    x = Tensor(rng.standard_normal((32, 512)))

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
    print(f"  Mean latency: {result.mean:.2f} ms")
    print(f"  Std dev:      {result.std:.2f} ms")
    print(f"  Min:          {result.min_val:.2f} ms")
    print(f"  Max:          {result.max_val:.2f} ms")

    print("\n✨ Reliable measurements guide optimization decisions!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_benchmarking()

# %% [markdown]
r"""
## 🚀 MODULE SUMMARY: Benchmarking

Congratulations! You have built a professional, statistically rigorous benchmarking framework that mirrors production ML evaluation suites like MLPerf!

### Systems Milestone Scorecard

| Milestone Capability | Mathematical / Systems Mechanism | TinyTorch Implementation | Production Parallel |
| :--- | :--- | :--- | :--- |
| **Statistical Rigor** | Sample mean $\mu$, variance $s^2$, standard error $\frac{s}{\sqrt{n}}$, and Student's $t$ CI | `BenchmarkResult` | Google Benchmark, Criterion.rs |
| **Monotonic Timing** | Monotonic userspace vDSO clock with nanosecond counter | `precise_timer()` | `clock_gettime(CLOCK_MONOTONIC)` |
| **Warmup Discard** | Cold-start page fault & cache warming isolation | `Benchmark.run_latency_benchmark()` | MLPerf Tiny warmup harness |
| **Memory Accounting** | Peak allocator buffer tracking vs process RSS | `Benchmark.run_memory_benchmark()` | PyTorch CUDA Caching Allocator profiler |
| **Standardized Tasks** | Fixed seeds, input shapes, and multi-objective thresholds | `MLPerf` class | MLPerf Inference & Mobile Benchmark Suite |
| **Multi-Objective Tradeoffs**| Constrained Pareto optimization ($\min \text{Lat}, \min \text{Mem}$ s.t. $\text{Acc} \ge \tau$) | `analyze_optimization_techniques()` | Optuna, Neural Network Intelligence (NNI) |

### Key Systems Insights Discovered
- **Measurement Science**: Single-run latency numbers are noise; true systems characterization requires isolated warmup and statistical confidence intervals.
- **Metric Dimensionality**: Optimizing for speed without tracking memory or accuracy creates brittle models that fail silent SLA requirements.
- **Hardware Realities**: Micro-benchmarks on sub-millisecond kernels must account for syscall overhead ($\delta_{\text{timer}} \sim 1\text{ }\mu\text{s}$) through iteration amortization.
- **Production Integration**: Objective compliance checks and machine-readable JSON reports bridge experimental training to operational deployment.

### Ready for Next Steps
Your benchmarking framework completes the Optimization Tier! All components—from Tensors, Autograd, Convolutions, and Transformers to Quantization, Acceleration, Memoization, and Benchmarking—are now verified.

Export with: `tito dev export 19`

**Next**: Module 20 (Capstone) will integrate every single concept into an end-to-end production ML system!
"""
