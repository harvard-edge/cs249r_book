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

**You've Built**: The full neural network stack (Modules 01–13) and the optimization tier including profiling (`14_profiling`), INT8 quantization (`15_quantization`), pruning (`16_compression`), vector acceleration (`17_acceleration`), and KV caching (`18_memoization`).
**You'll Build**: Statistically sound benchmarking harnesses (`Benchmark`, `BenchmarkSuite`, `BenchmarkResult`), standard error and Student-$t$ confidence intervals, an MLPerf compliance runner, and multi-objective Pareto frontier analysis (`pareto_frontier`).
**You'll Enable**: Empirical, reproducible performance verification that rigorously proves the speedup, memory, and accuracy trade-offs across all your optimizations before the capstone synthesis.

<div align="center">
  <img src="benchmarking_blueprint.svg" width="380px" alt="TinyTorch Framework Blueprint: Module 19 Benchmarking" />
</div>

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
4. Implement an MLPerf Tiny standardized compliance runner enforcing deterministic input seeds, MLPerf Tiny's published accuracy targets, and a $p_{90}$ single-stream latency gate (that ceiling being TinyTorch's own, since MLPerf Tiny measures latency rather than gating on it).
5. Derive empirical Pareto frontiers with `pareto_frontier()` to identify non-dominated model variants across competing systems objectives.

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/19_benchmarking/benchmarking.ipynb`  
**Building Side:** Code exports to `tinytorch.perf.benchmarking`

<div align="center">
  <img src="benchmarking_source_card.svg" width="260px" alt="Source Code Mapping Card for Module 19 Benchmarking" />
</div>

```python
# Final package structure:
from tinytorch.perf.benchmarking import Benchmark, BenchmarkSuite, BenchmarkResult, MLPerf, precise_timer, pareto_frontier
```

## 📋 Module Dependencies

| Dependency Module | Exported Abstraction | Consumed Functional Role | Memory & Evaluation Invariant |
|:---|:---|:---|:---|
| **Module 01 (`01_tensor`)** | `Tensor` | Contiguous N-D numerical array representation | Evaluation inputs and outputs without autograd overhead |
| **Module 03 (`03_layers`)** | `Linear` | Fully connected layer primitive | Reference workloads for single-layer benchmarking |
| **Module 14 (`14_profiling`)** | `Profiler` | High-resolution microsecond timer | Core latency and memory probe reused by `Benchmark` |
| **Modules 15–16** | `QuantizedLinear`, `magnitude_prune` | Optimized model variants | Inputs to multi-dimensional comparative benchmarking |
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp perf.benchmarking
#| export

import json
import os
import platform
import statistics
import tempfile
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

# Two-sided 95% Student-t critical values t_{0.975, nu}, indexed by
# nu = n - 1 degrees of freedom for nu = 1..30. scipy is not a TinyTorch
# dependency and a benchmark loop needs nothing more than this, because the
# table covers exactly the range where t and the normal limit disagree most.
T_CRITICAL_95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)
Z_CRITICAL_95 = 1.96  # the nu -> infinity limit


def t_critical_95(dof: int) -> float:
    """
    Two-sided 95% critical value t_{0.975, nu} for `dof` degrees of freedom.

    Table lookup up to nu = 30. Past that, the Cornish-Fisher expansion
    t ~ z + (z^3 + z) / (4 nu) is within 0.15% of the true value and converges
    to z from above, so the interval never comes out narrower than it should.
    """
    if dof < 1:
        raise ValueError("A confidence interval needs at least 1 degree of freedom (n >= 2)")
    if dof <= len(T_CRITICAL_95):
        return T_CRITICAL_95[dof - 1]
    z = Z_CRITICAL_95
    return z + (z ** 3 + z) / (4 * dof)

# %% [markdown]
r"""
## 💡 Introduction: What is Fair Benchmarking?

Benchmarking in ML systems is not merely recording wall-clock time. It is an empirical science requiring controlled experimental conditions to enable fair, reproducible comparisons that guide production deployment decisions.

<div align="center">
  <img src="benchmarking_methodology_overview.svg" width="680px" alt="Benchmarking Methodology Pipeline" />
</div>

### Confounding Factors: What This Harness Controls

Every noise source below is real. Only two of them are controlled by the code in
this module, and the table says which. Read the last column as the list of things
a number from this harness does **not** account for.

| Noise Source | Physical Hardware Mechanism | What this harness does | What a production harness adds |
|:---|:---|:---|:---|
| **Cold Starts** | Dynamic library loading & page faults | Discards warmup iterations before recording (implemented) | Pre-faults and locks pages so the first timed run is already resident |
| **OS Interrupts** | Background scheduler preemption & context switches | Samples many trials and reports median and percentiles (implemented) | Pins threads to cores (`sched_setaffinity`) and raises scheduling priority |
| **Thermal Throttling** | DVFS frequency scaling when silicon overheats | Nothing; a throttled trial simply shows up in the tail of the distribution | Cooldown pauses between trials and randomized trial interleaving |
| **Cache Pollution** | Shared L2/L3 cache evictions by other processes | Nothing; array layout cannot stop another process evicting your lines | An isolated core with a partitioned cache, or a quiesced machine |
| **Memory Pressure** | Python garbage collection pauses | Nothing; a collection during a timed run lands in the tail | `gc.disable()` around the inner timing loop, then a forced collection between trials |

### Looking Ahead

The benchmarking tools you build here will be used in Module 20's capstone project, where you'll apply optimization techniques competitively. For now, focus on building reliable, fair measurement infrastructure.

---

## 📐 Foundations: Statistics for Performance Engineering

Inference latency on modern superscalar processors is an inherently non-deterministic, right-skewed stochastic process.

<div align="center">
  <img src="latency_anatomy_distribution.svg" width="680px" alt="The Anatomy of Latency Distributions and Tail Percentiles" />
</div>

<div align="center">
  <img src="benchmarking_latency_card.svg" width="320px" alt="Latency Distribution Card" />
</div>

### Central Limit Theorem & Confidence Intervals

While individual latency measurements exhibit heavy-tailed distributions due to system hiccups, the sample mean $\bar{X}$ over $n$ independent trials converges toward a normal distribution:

$$\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i, \quad s = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2}$$

The standard error of the mean ($\text{SE}$) and the two-sided $95\%$ Student-$t$ confidence interval are given by:

$$\text{SE} = \frac{s}{\sqrt{n}}, \quad \text{CI}_{95\%} = \left[ \bar{X} - t_{0.025, \, n-1} \frac{s}{\sqrt{n}}, \quad \bar{X} + t_{0.025, \, n-1} \frac{s}{\sqrt{n}} \right]$$

The critical value is Student's $t$, not the normal $z_{0.975} = 1.96$, because $s$
is an estimate from the same $n$ samples rather than a known population value. The
distinction is not cosmetic at benchmarking sample sizes. At the module default of
$n = 10$, $t_{0.975, 9} = 2.262$ against $z = 1.96$, so quoting $1.96$ reports an
interval $13\%$ narrower than the data supports. The two agree to within $2\%$ only
past $n \approx 60$, which is well beyond what most benchmark loops run.

### Latency, Throughput, and the Batching Precondition

The reciprocal identity $\text{throughput} = 1 / \text{latency}$ holds **only when
each timed call processes exactly one example**. That single-stream regime is the one
this harness enforces, because `MLPerf._run_accuracy_test` rejects a prediction
carrying more than one example's scores.

Batching breaks the identity in both directions at once. A batch of $B$ examples takes
longer per call, so latency rises, while per-example cost falls as fixed overhead
amortizes and the matmul reaches a better shape. Throughput becomes
$B / T_{\text{batch}}$, which can exceed $1 / T_{\text{single}}$ by an order of
magnitude while every individual user waits longer. Reporting one of those numbers and
calling it the other is the most common way a benchmark misleads.

### Multi-Objective Optimization & Pareto Dominance

Model optimization is multi-objective: latency, accuracy, memory, and energy represent competing physical trade-offs.

**Mathematical Definition of Pareto Dominance**:
Let $\mathcal{M}$ be the set of evaluation metrics. A model variant $\theta_A$ strictly Pareto-dominates variant $\theta_B$ ($\theta_A \succ \theta_B$) if and only if:

$$\forall m \in \mathcal{M}, \quad \text{score}_m(\theta_A) \ge \text{score}_m(\theta_B) \quad \land \quad \exists m \in \mathcal{M}, \quad \text{score}_m(\theta_A) > \text{score}_m(\theta_B)$$

The definition is written for scores where more is better. Latency, memory, and
energy are the opposite, so an implementation either negates them or carries one
direction flag per objective. `pareto_frontier()` below takes the flags, because
negating measured milliseconds makes the debugging output unreadable.

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
| `BenchmarkSuite` | Multi-dimensional evaluation engine | Models, datasets, metric configurations | Comparative trade-off tables, plus the non-dominated set from `pareto_frontier` |
| `pareto_frontier` | Non-dominated filtering over measured points | Per-model metric vectors, one direction flag per objective | Names of the variants no other variant dominates |
| `MLPerf` | Standardized edge compliance harness | Reference tasks, deterministic seeds | Pass/Fail compliance report |

### Statistical Metrics Tracked by BenchmarkResult

| Statistical Metric | Mathematical Estimator | Systems Interpretation | Robustness Against Outliers |
|:---|:---|:---|:---|
| **Mean ($\mu$)** | $\frac{1}{n} \sum X_i$ | Average cost per timed call; it inverts to throughput only at one example per call | Sensitive to tail stalls |
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

        # 95% Student-t confidence interval for the mean. The critical value
        # depends on n through the degrees of freedom, so it cannot be the
        # constant 1.96: at n = 10 that constant reports an interval 13% too
        # narrow. One sample has no degrees of freedom and therefore no
        # interval at all, so the bounds are None rather than the mean itself.
        # A zero-width 95% interval printed beside a result is a false claim.
        if self.count > 1:
            margin_error = t_critical_95(self.count - 1) * (self.std / np.sqrt(self.count))
            self.ci_lower = self.mean - margin_error
            self.ci_upper = self.mean + margin_error
        else:
            self.ci_lower = self.ci_upper = None

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

    # Test confidence intervals. The critical value must be Student's t for
    # n - 1 = 4 degrees of freedom (2.776), not the normal 1.96, which would
    # give a half-width of 1.386 instead of 1.964.
    assert result.ci_lower < result.mean < result.ci_upper
    expected_half_width = 2.776 * (result.std / np.sqrt(result.count))
    assert abs((result.ci_upper - result.mean) - expected_half_width) < 1e-3, (
        f"Half-width {result.ci_upper - result.mean:.4f} does not match the "
        f"Student-t interval {expected_half_width:.4f}; a constant 1.96 would "
        f"give {1.96 * result.std / np.sqrt(result.count):.4f}"
    )

    # A single measurement has no degrees of freedom, so it has no interval.
    single = BenchmarkResult("one_shot", [7.5])
    assert single.mean == 7.5 and single.std == 0.0
    assert single.ci_lower is None and single.ci_upper is None, (
        "n=1 must report no interval; a zero-width 95% CI claims a precision "
        "one sample cannot support"
    )

    # Test serialization
    result_dict = result.to_dict()
    assert result_dict['metric_name'] == "test_metric"
    assert result_dict['mean'] == 3.0

    print("✅ BenchmarkResult works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_result()

# %% [markdown]
r"""
## 🏗️ Timing: High-Precision Interval Measurement

Accurate timing is the foundation of performance benchmarking. System clocks have different precision and behavior, so we need a robust timing mechanism.

### Timing Challenges in Practice

When timing a function call in high-level languages like Python, several layers of operating system and hardware indirection intervene between the software invocation and the physical timer:

$$\Delta t_{\text{measured}} = \Delta t_{\text{true}} + \delta_{\text{call}} + \delta_{\text{OS}} + \delta_{\text{quantization}}$$

| Latency Component | Typical Magnitude | Root Cause / System Source | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Kernel / Forward Pass** ($\Delta t_{\text{true}}$) | $\mu\text{s}$ to $\text{ms}$ | Actual computational operations (FLOPs, memory loads) | Target metric under benchmark |
| **Syscall Overhead** ($\delta_{\text{call}}$) | $10\text{--}50\text{ ns}$ (one `perf_counter()` call measured at $\approx 35\text{ ns}$ here) | User-to-kernel context switch for clock sampling | Use monotonic userspace vDSO clock |
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

<div align="center">
  <img src="benchmarking_methodology_overview.svg" alt="Benchmarking Methodology Overview" width="680px">
</div>

### Benchmark Architecture Execution Flow

| Stage | Input Artifacts | Processing Step | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **1. Ingestion** | Models $[M_1, M_2, \dots]$, Datasets $[D_1, D_2, \dots]$ | Register candidate architectures and evaluation datasets | Model registry with validated callable interfaces |
| **2. Warmup** | Synthetic or unmeasured batches | Execute $W$ iterations to warm CPU caches and fault in pages | Discarded startup latencies, stabilized hardware clocks |
| **3. Measurement** | Fixed-seed inputs | Sample $N$ independent forward passes with `perf_counter` | Raw latency samples $[t_1, t_2, \dots, t_N]$ |
| **4. Profiling** | Model instance | Trace memory allocations and peak buffer usage | Traced peak memory (MB) and FLOP counts |
| **5. Synthesis** | Raw timing and memory metrics | Compute $\mu, s, \text{SE}$, and confidence intervals $[CI_{\text{low}}, CI_{\text{high}}]$ | `BenchmarkResult` container with system metadata |

### Why Warmup Runs Matter

Modern operating systems and processors have multiple layers of runtime adaptation:
- **CPU frequency scaling**: Dynamic Voltage and Frequency Scaling (DVFS) ramps execution cores to performance governors
- **Cache warming**: Instruction and weight caches ($L_1/L_2/L_3$) achieve steady-state hit rates
- **Memory frame allocation**: OS page faults occur during initial virtual memory touches
- **Branch predictor and TLB training**: The first pass through a loop mispredicts and misses; later passes do not

TinyTorch runs on NumPy, which has **no JIT compiler**, so none of the warmup
benefit here comes from code specialization. On a JIT-backed stack (PyTorch's
`torch.compile`, JAX, Numba) the first call additionally pays for tracing and
compilation, which is often orders of magnitude larger than everything above and
is the reason warmup counts are so much higher there.

<div align="center">
  <img src="latency_anatomy_distribution.svg" alt="Latency Anatomy and Warmup" width="680px">
</div>

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

<div align="center">
  <img src="benchmarking_latency_card.svg" alt="Tail Latency Percentiles" width="320px">
</div>

### Latency Measurement Pipeline

| Pipeline Stage | Implementation Action | Purpose & Guarantees |
| :--- | :--- | :--- |
| **Input Synthesis** | `Tensor(rng.standard_normal(shape))` | Allocates representative evaluation tensor matching hardware target |
| **Warmup Phase** | `profiler.measure_latency(warmup=W)` | Absorbs page faults, cache misses, and DVFS ramp (discarded) |
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

    # A mean above zero passes even if only one run was timed, or if the timer
    # returns a constant. Pin the sample count and the known floor instead:
    # each forward() sleeps 1 ms, so every sample must clear 0.9 ms.
    for name, r in results.items():
        assert r.count == 3, f"{name}: timed {r.count} runs, expected measurement_runs=3"
        assert r.min_val >= 0.9, f"{name}: a 1ms sleep measured {r.min_val:.3f}ms"
        assert r.mean >= 0.9, f"{name}: mean {r.mean:.3f}ms is below the 1ms sleep"
        assert r.ci_lower <= r.mean <= r.ci_upper

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
    assert all(r.count == 3 for r in results.values())

    # `mean >= 0` is vacuous: the implementation clamps at zero, so it holds even
    # if the method reports nothing it measured. Pin the value to the profiler's
    # peak, using a sub-1MB peak that must survive rather than be rounded away.
    benchmark.profiler.measure_memory = lambda model, shape: {'peak_memory_mb': 0.125}
    stubbed = benchmark.run_memory_benchmark()
    for name, r in stubbed.items():
        assert r.values == [0.125, 0.125, 0.125], (
            f"{name}: recorded {r.values}, not the profiler's 0.125 MB peak"
        )

    print("✅ Benchmark.run_memory_benchmark works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_memory()

# %% [markdown]
r"""
### 🧪 Unit Test: Benchmark (Full Class Integration)

This test validates our Benchmark class measures latency, accuracy, and memory correctly.

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
    assert all(result.count == 3 for result in memory_results.values())

    print("✅ Benchmark works correctly!")

if __name__ == "__main__":
    test_unit_benchmark()

# %% [markdown]
r"""
## 🏗️ Benchmark Suite: Multi-Metric Evaluation and Reporting

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

<div align="center">
  <img src="benchmarking_source_card.svg" alt="Source Code Mapping" width="260px">
</div>

| Evaluation Phase | Component Operation | Collected Data | Systems Decision Role |
| :--- | :--- | :--- | :--- |
| **Model Ingestion** | `models = [M1, M2, ...]` | Model architectures and weight buffers | Candidates for comparative deployment profiling |
| **Metric Execution** | Latency, Accuracy, Memory, Energy | Sample distributions and allocator peaks | Multi-objective empirical measurement vectors |
| **Aggregation** | Unified dictionary indexing | Synchronized per-metric `BenchmarkResult` | Cross-model normalization and variance alignment |
| **Pareto Analysis** | `pareto_frontier()` non-dominated filtering | The set of variants no other variant dominates | Eliminates strictly sub-optimal candidate variants |
| **Deployment Synthesis** | Markdown & JSON report generator | Quantitative tradeoff recommendations | Concrete deployment mapping (Server, Mobile, IoT) |

### Pareto Frontier Analysis

`generate_report` names the Pareto-optimal models, the ones no other model beats on
every metric at once. That is a filter rather than a ranking. It narrows the candidate
set without deciding among what survives, and it never collapses the trade-off to a
single score. Collapsing is a separate, weighted choice, and this module keeps the
two steps apart so you can see which one is doing the deciding.

### Energy Efficiency Modeling

Since direct energy measurement requires specialized hardware, we estimate energy based on computational complexity and memory usage. This provides actionable insights for battery-powered deployments.
"""

# %% [markdown]
r"""
### BenchmarkSuite.__init__: Setting Up Multi-Metric Evaluation

The BenchmarkSuite constructor creates the evaluation infrastructure, including
a Benchmark instance for measurements and an output directory for reports and plots.
It also forwards `warmup_runs` and `measurement_runs` to that Benchmark, which is
the only knob that trades measurement wall-clock time against confidence-interval
width. A suite that hard-coded the defaults would leave a CI pipeline no way to run
a fast smoke configuration and a slow nightly one from the same code.
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
                 output_dir: str = "benchmark_results",
                 warmup_runs: int = DEFAULT_WARMUP_RUNS,
                 measurement_runs: int = DEFAULT_MEASUREMENT_RUNS):
        """
        Initialize comprehensive benchmark suite.

        TODO: Set up the suite with models, datasets, output directory, and a Benchmark instance

        APPROACH:
        1. Store models and datasets
        2. Create output directory (use Path, mkdir with exist_ok)
        3. Create Benchmark instance for measurements, forwarding the run counts
        4. Initialize empty results dict

        HINTS:
        - Use Path(output_dir) for cross-platform paths
        - The Benchmark instance handles individual model measurements
        - Forward warmup_runs and measurement_runs; a suite that swallows them
          leaves no way to trade measurement time against interval width
        """
        ### BEGIN SOLUTION role="scaffold"
        self.models = models
        self.datasets = datasets
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.benchmark = Benchmark(models, datasets,
                                   warmup_runs=warmup_runs,
                                   measurement_runs=measurement_runs)
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
        assert suite.benchmark.warmup_runs == DEFAULT_WARMUP_RUNS
        assert suite.benchmark.measurement_runs == DEFAULT_MEASUREMENT_RUNS

        # The run counts must actually reach the Benchmark, or a CI pipeline has
        # no way to pick a fast or an accurate configuration.
        tuned = BenchmarkSuite(models, datasets, output_dir=tmp_dir,
                               warmup_runs=1, measurement_runs=3)
        assert tuned.benchmark.warmup_runs == 1
        assert tuned.benchmark.measurement_runs == 3

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
### BenchmarkSuite._estimate_energy_efficiency: Energy Modeling

Since direct energy measurement requires specialized hardware (power meters, RAPL),
we estimate energy from latency and memory usage. The model is a sum of three terms,
not a product: a fixed cost per inference, an active-power term that is the only one
multiplied by time, and a static term charged per megabyte resident.

```
Energy Estimation Model:
energy = base_cost + (latency/1000) * 2.0 + memory * 0.01   (Joules; illustrative constants)
         ↑            ↑                      ↑
         Fixed        Active power x time    Static cost per MB
         overhead     (the only t term)      (added, not scaled by t)
```

Charging memory additively rather than as $P_{\text{static}} \times t$ is a
simplification, and an inspectable one. A model that is slow and small gets no
static-power credit for finishing quickly. Say so in a report rather than letting a
reader assume the constants came off a power meter.
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
                # Three additive terms: fixed overhead, active power x time, and a
                # static per-MB charge. Only the middle term scales with latency.
                #
                # zip() pairs the i-th latency with the i-th memory peak, but those
                # came from two separate sweeps over the model, so the pairing is an
                # artifact. The MEAN of energy_values is still correct (the mean of a
                # sum is the sum of the means); its STD is not, because it assumes a
                # within-run correlation that was never measured.
                energy_values = []
                for lat, mem in zip(latency_result.values, memory_result.values):
                    energy = ENERGY_BASE_JOULES + (lat / 1000) * ENERGY_JOULES_PER_SECOND + mem * ENERGY_JOULES_PER_MB
                    energy_values.append(energy)

                energy_results[model_name] = BenchmarkResult(
                    f"{model_name}_energy_joules",
                    energy_values,
                    metadata={'estimated': True, 'std_is_meaningful': False,
                              **self.benchmark.system_info}
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

if __name__ == "__main__":
    test_unit_benchsuite_run()

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

    # Agg is the headless backend: it can write a file but has no window to
    # open, and calling show() on it emits a UserWarning instead of a plot.
    if plt.get_backend().lower() != 'agg':
        plt.show()
    plt.close(fig)
    ### END SOLUTION

BenchmarkSuite.plot_results = benchsuite_plot_results

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
            # Agg is the headless backend: it writes files and has no window to
            # open, so plot_results skips plt.show() rather than triggering
            # "UserWarning: FigureCanvasAgg is non-interactive".
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
### pareto_frontier: Filtering Out the Dominated Variants

The dominance definition from 📐 is one function. A variant is on the frontier when
no other variant matches it on every objective and beats it on at least one. Nothing
is weighted, nothing is normalized, and no single winner is produced. The frontier is
a **set**, and its size is the honest answer to "how many real choices do I have?"

| Input | Meaning | Example |
| :--- | :--- | :--- |
| `points` | One metric vector per variant, all in the same order | `{'small': (5.0, 0.80), 'big': (40.0, 0.94)}` |
| `lower_is_better` | One flag per objective, in that same order | `(True, False)` for (latency ms, accuracy) |

With those two inputs, `small` and `big` are both on the frontier: `small` wins on
latency, `big` wins on accuracy, and neither dominates. Add `{'bad': (40.0, 0.70)}`
and it drops out, because `big` is no slower and more accurate.

The cost is $O(n^2 m)$ for $n$ variants and $m$ objectives, which is the right
algorithm here. $n$ is the number of variants you are choosing between, so it is
single digits, and the $O(n \log n)$ divide-and-conquer alternatives only pay off in
the thousands.
"""

# %% nbgrader={"grade": false, "grade_id": "pareto-frontier", "solution": true}
#| export
def pareto_frontier(points: Dict[str, Tuple[float, ...]],
                    lower_is_better: Tuple[bool, ...]) -> List[str]:
    """
    Return the names of the non-dominated points, in input order.

    TODO: Implement Pareto dominance filtering over measured metric vectors

    APPROACH:
    1. Write a dominates(a, b) predicate straight from the 📐 definition:
       a is at least as good as b on EVERY objective, and strictly better on ONE
    2. Respect lower_is_better per objective, so latency and accuracy can mix
    3. Keep a point when nothing else dominates it

    Args:
        points: {name: metric vector}, every vector the same length
        lower_is_better: one flag per objective, in the vectors' order

    Returns:
        List of names on the frontier, in the order they appeared in `points`

    EXAMPLE:
    >>> pareto_frontier({'a': (5.0, 0.80), 'b': (40.0, 0.94), 'c': (40.0, 0.70)},
    ...                 (True, False))
    ['a', 'b']

    HINTS:
    - "At least as good" is <= for a minimized objective and >= for a maximized one
    - Both conditions are required. Without the strict part, two identical points
      would dominate each other and the frontier would come back empty
    """
    ### BEGIN SOLUTION
    width = len(lower_is_better)
    if any(len(vector) != width for vector in points.values()):
        raise ValueError(f"Every metric vector must carry {width} objectives")

    def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        no_worse = all((x <= y) if low else (x >= y)
                       for x, y, low in zip(a, b, lower_is_better))
        better_somewhere = any((x < y) if low else (x > y)
                               for x, y, low in zip(a, b, lower_is_better))
        return no_worse and better_somewhere

    return [name for name, vector in points.items()
            if not any(dominates(other, vector)
                       for rival, other in points.items() if rival != name)]
    ### END SOLUTION

# %% [markdown]
r"""
### 🧪 Unit Test: pareto_frontier

**What we're testing**: Dominance filtering, including the strictness condition and mixed objective directions
**Why it matters**: A frontier is the claim this module makes seven times; an implementation that returns everything (or nothing) makes the claim vacuous
**Expected**: Dominated variants are dropped, tied variants both survive, and a single objective reduces to the best value
"""

# %% nbgrader={"grade": true, "grade_id": "test-pareto-frontier", "locked": true, "points": 5}
def test_unit_pareto_frontier():
    """🧪 Test pareto_frontier dominance filtering."""
    print("🧪 Unit Test: pareto_frontier...")

    # (latency ms, accuracy): minimize the first, maximize the second.
    points = {
        'fast':      (5.0, 0.80),
        'accurate': (40.0, 0.94),
        'dominated': (40.0, 0.70),   # no faster than 'accurate', less accurate
        'balanced': (12.0, 0.90),
    }
    frontier = pareto_frontier(points, (True, False))
    assert frontier == ['fast', 'accurate', 'balanced'], frontier
    assert 'dominated' not in frontier, (
        "'dominated' is beaten by 'accurate' on accuracy and tied on latency"
    )

    # Ties must both survive. Identical points do not dominate each other,
    # because dominance requires being strictly better somewhere.
    twins = pareto_frontier({'a': (1.0, 0.5), 'b': (1.0, 0.5)}, (True, False))
    assert sorted(twins) == ['a', 'b'], twins

    # One objective collapses to "the best value wins", ties included.
    single = pareto_frontier({'a': (3.0,), 'b': (1.0,), 'c': (1.0,)}, (True,))
    assert sorted(single) == ['b', 'c'], single

    # A frontier is never empty: something always survives.
    assert pareto_frontier({'only': (1.0, 1.0)}, (True, True)) == ['only']

    # Mismatched widths are a caller bug, not a silently wrong frontier.
    try:
        pareto_frontier({'a': (1.0, 2.0), 'b': (1.0,)}, (True, False))
        assert False, "Should have raised ValueError for a short metric vector"
    except ValueError:
        pass

    print("✅ pareto_frontier works correctly!")

if __name__ == "__main__":
    test_unit_pareto_frontier()

# %% [markdown]
r"""
### BenchmarkSuite.generate_report: Actionable Insights

The `generate_report` method compiles all benchmark results into a structured
markdown report with system information, per-metric summaries, best performers,
trade-off analysis, and deployment recommendations.

| Report Generation Stage | Input Data | Generated Section | Key Technical Content |
| :--- | :--- | :--- | :--- |
| **1. System Metadata** | `system_info` | Environment Header | OS, CPU architecture, core count, Python runtime |
| **2. Per-Metric Summaries** | `results[metric]` | Score Breakdown | Mean with its unit, standard deviation, 95% CI, best performer |
| **3. Trade-Off Analysis** | Cross-metric vectors | Non-dominated filtering | The Pareto frontier over (latency, accuracy) from `pareto_frontier()` |
| **4. Recommendations** | Per-axis winners | Deployment Guidance | Fastest and most accurate variant, drawn from the frontier |
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
       c. List all models with mean ± std, THE UNIT, and the 95% CI
    2. For n = 1 print "n=1, no interval" rather than a zero-width interval

    HINTS:
    - METRIC_UNITS below carries one unit per metric; a bare four-decimal number
      is not comparable, which is the one thing this whole module is about
    """
    ### BEGIN SOLUTION role="scaffold"
    METRIC_UNITS = {'latency': 'ms', 'accuracy': '', 'memory': 'MB', 'energy': 'J'}

    lines = []
    lines.append("## Benchmark Results Summary")
    lines.append("")

    for metric_type, results in self.results.items():
        qualifier = " (estimated)" if any(r.metadata.get('estimated', False) for r in results.values()) else ""
        if any(r.metadata.get('simulated', False) for r in results.values()):
            qualifier += " (synthetic probe)"
        unit = METRIC_UNITS.get(metric_type, '')
        unit_label = f" [{unit}]" if unit else " [fraction of 1]"
        lines.append(f"### {metric_type.capitalize()} Results{unit_label}{qualifier}")
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
            suffix = f" {unit}" if unit else ""
            # One measurement has no degrees of freedom, so it has no interval.
            # Printing [x, x] as a 95% CI would claim a precision n=1 cannot give.
            if result.ci_lower is None:
                interval = "n=1, no interval"
            else:
                interval = (f"95% CI [{result.ci_lower:.4f}, {result.ci_upper:.4f}]{suffix}"
                            f" (n={result.count})")
            lines.append(f"- **{model_name}**: {result.mean:.4f}{suffix} "
                         f"± {result.std:.4f}{suffix}, {interval}")
        if any(r.metadata.get('std_is_meaningful') is False for r in results.values()):
            lines.append("")
            lines.append("> **Note:** the spread here is not a measured spread. Each value "
                         "pairs the i-th latency with the i-th memory peak from two separate "
                         "sweeps, so the mean is sound and the standard deviation and interval "
                         "assume a within-run correlation nobody measured.")
        lines.append("")

    return lines
    ### END SOLUTION

BenchmarkSuite._format_results_summary = _benchsuite_format_results_summary

# %% [markdown]
r"""
#### Step 2: Report the Trade-off Space, Then the Per-Axis Winners

This step reports the Pareto frontier over (latency, accuracy) and names the winner
on each axis. It deliberately does **not** compute a "best overall" score.

The tempting alternative is to min-max normalize both metrics and average them. Do not.
With two models where each is worst on the other's axis, the fastest normalizes to
$(1, 0)$ and the most accurate to $(0, 1)$, so both score exactly $0.500$ and the
winner is whichever the dictionary happens to yield first. A tie broken by insertion
order, printed to three decimals, reads as a measurement. It is not one.

Weighing latency against accuracy needs weights, and weights come from the deployment,
not from the benchmark. `_generate_recommendations` in 🔧 Integration is the one place
in this module that applies weights, and it states them. Here we narrow the field and
stop.
"""

# %% nbgrader={"grade": false, "grade_id": "benchsuite-format-recs", "solution": true}
#| export
def _benchsuite_format_recommendations(self) -> List[str]:
    """
    Generate recommendation lines from benchmark results.

    Returns:
        List of markdown-formatted recommendation lines

    TODO: Report the Pareto frontier over (latency, accuracy), then the per-axis winners

    APPROACH:
    1. Bail out early when the accuracy numbers came from the synthetic probe
    2. Build one (latency, accuracy) vector per model that has both
    3. Call pareto_frontier(points, (True, False)) and list what survives
    4. Name the fastest and the most accurate model, both unambiguous

    HINTS:
    - Do NOT average normalized metrics into a single score; see the lead-in above
    - The frontier is a set. Reporting its size is the useful line, because it says
      how many genuine choices the measurements left open
    """
    ### BEGIN SOLUTION role="scaffold"
    lines = []
    lines.append("## Recommendations")
    lines.append("")
    if any(result.metadata.get('simulated', False)
           for result in self.results.get('accuracy', {}).values()):
        lines.append("Synthetic accuracy probe: no deployment recommendations.")
        return lines

    if 'latency' in self.results and 'accuracy' in self.results:
        latency_results = self.results['latency']
        accuracy_results = self.results['accuracy']

        # One vector per model, in a fixed objective order: (latency, accuracy).
        points = {name: (result.mean, accuracy_results[name].mean)
                  for name, result in latency_results.items()
                  if name in accuracy_results}

        if points:
            lines.append("### Accuracy vs Speed Trade-off")
            frontier = pareto_frontier(points, (True, False))
            lines.append(f"- **Pareto frontier** (latency ms down, accuracy up): "
                         f"{', '.join(frontier)}")
            dominated = [name for name in points if name not in frontier]
            if dominated:
                lines.append(f"- **Dominated** (another model is no slower and no "
                             f"less accurate): {', '.join(dominated)}")
            else:
                lines.append(f"- No model dominates another: all {len(points)} are "
                             f"real choices, and the weights are yours to set")
            lines.append("")

        lines.append("### Usage Recommendations")
        best_acc_model = max(accuracy_results.items(), key=lambda x: x[1].mean)
        best_lat_model = min(latency_results.items(), key=lambda x: x[1].mean)

        lines.append(f"- **For maximum accuracy**: Use {best_acc_model[0]} "
                     f"({best_acc_model[1].mean:.4f})")
        lines.append(f"- **For minimum latency**: Use {best_lat_model[0]} "
                     f"({best_lat_model[1].mean:.4f} ms)")
        lines.append("- **For production deployment**: Pick from the frontier above "
                     "using your own latency budget; this report will not pick for you")

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
    with open(report_path, 'w', encoding='utf-8') as f:
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

**What we're testing**: Both paths, the synthetic-probe bail-out and the measured path that computes the frontier and names the per-axis winners
**Why it matters**: Wrong recommendations lead to wrong deployment decisions, and a test that only runs the bail-out never sees the recommendation code at all
**Expected**: Probe results yield no recommendations; measured results name the frontier, drop the dominated model, and pick the right winner on each axis
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchsuite-format-recs", "locked": true, "points": 3}
def test_unit_benchsuite_format_recs():
    """🧪 Test BenchmarkSuite._format_recommendations implementation."""
    print("🧪 Unit Test: BenchmarkSuite._format_recommendations...")

    class ProbeModel:
        """No evaluate(), so accuracy can only come from the synthetic probe."""
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return x * 0.5

    class MeasuredModel:
        """Has evaluate(), so the measured recommendation path runs."""
        def __init__(self, name, score, delay):
            self.name, self.score, self.delay = name, score, delay
        def forward(self, x):
            time.sleep(self.delay)
            return x * 0.5
        def evaluate(self, dataset):
            return self.score

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        suite = BenchmarkSuite([ProbeModel("a"), ProbeModel("b")],
                               [{"data": "test"}], output_dir=tmp_dir,
                               warmup_runs=0, measurement_runs=2)
        suite.run_full_benchmark(simulate=True)

        lines = suite._format_recommendations()
        assert isinstance(lines, list), f"Expected list, got {type(lines)}"
        text = "\n".join(lines)
        assert "Recommendations" in text, "Should contain 'Recommendations'"
        assert "Synthetic accuracy probe" in text, (
            "A probe score must not turn into a deployment recommendation"
        )

    # Measured path: 'quick' is fastest, 'sharp' is most accurate, and 'weak' is
    # dominated (slower than 'sharp' and less accurate than either).
    with tempfile.TemporaryDirectory() as tmp_dir:
        suite = BenchmarkSuite(
            [MeasuredModel("quick", 0.80, 0.001),
             MeasuredModel("sharp", 0.95, 0.010),
             MeasuredModel("weak", 0.60, 0.020)],
            [{"data": "test"}], output_dir=tmp_dir,
            warmup_runs=0, measurement_runs=2)
        suite.run_full_benchmark()

        text = "\n".join(suite._format_recommendations())
        assert "Synthetic accuracy probe" not in text, (
            "Models with evaluate() are measured, so the bail-out must not fire"
        )
        assert "Pareto frontier" in text, "Measured path must report the frontier"
        assert "quick" in text and "sharp" in text
        assert "For maximum accuracy**: Use sharp" in text, text
        assert "For minimum latency**: Use quick" in text, text
        assert "Dominated" in text and "weak" in text, (
            f"'weak' is dominated by 'sharp' and must be named as such:\n{text}"
        )

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
## 🏗️ MLPerf Harness: Standardized Edge Compliance

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

**What MLPerf Tiny actually specifies, and what this module adds.** MLPerf Tiny
defines a **quality target** per task and then **measures** latency and energy. It
does not impose a latency threshold a submission must clear; a slow submission is a
valid submission with a slow number. The millisecond ceilings below are **TinyTorch
classroom thresholds**, invented here so that `run_standard_benchmark` has something
to gate on and you can see a pass/fail harness work. The accuracy targets are the
real ones.

**Keyword Spotting**: Wake word detection from audio
- Input: 1-second 16kHz audio samples
- Task: Binary classification (keyword present/absent)
- Quality target (MLPerf Tiny): 90% top-1 accuracy
- Latency ceiling: <100ms *(TinyTorch classroom threshold, not an MLPerf rule)*

**Visual Wake Words**: Person detection in images
- Input: 96×96 RGB images
- Task: Binary classification (person present/absent)
- Quality target (MLPerf Tiny): 80% top-1 accuracy
- Latency ceiling: <200ms *(TinyTorch classroom threshold, not an MLPerf rule)*

**Anomaly Detection**: Industrial sensor monitoring
- Input: 640-element sensor feature vectors
- Task: MLPerf Tiny scores this by **AUC ≥ 0.85** on an autoencoder's reconstruction
  error, not by classification accuracy. This module simplifies it to a binary
  accuracy ≥ 0.85 so that all four tasks share one scoring path. AUC is
  threshold-free and accuracy is not, so the two are not interchangeable; the
  simplification is ours
- Latency ceiling: <50ms *(TinyTorch classroom threshold, not an MLPerf rule)*

**Image Classification**: Tiny image recognition (CIFAR-10)
- Input: 32×32 RGB images
- Task: Multi-class classification (10 classes)
- Quality target (MLPerf Tiny): 85% top-1 accuracy (ResNet-8 on CIFAR-10)
- Latency ceiling: <150ms *(TinyTorch classroom threshold, not an MLPerf rule)*

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
fixed random seed ensures reproducible results across different systems.

The accuracy column is MLPerf Tiny's real quality target. The latency column is a
**TinyTorch classroom threshold**: MLPerf Tiny measures latency and does not gate on
it, so these four ceilings exist only to give this harness something to check.

| Benchmark Task | Input Tensor Shape | Domain & Modality | Accuracy Target (MLPerf Tiny) | Latency Ceiling (TinyTorch) |
| :--- | :--- | :--- | :--- | :--- |
| `keyword_spotting` | `(1, 16000)` | 1-second 16kHz audio stream | $\ge 90\%$ | $< 100\text{ ms}$ |
| `visual_wake_words` | `(1, 96, 96, 3)` | 96×96 RGB vision camera | $\ge 80\%$ | $< 200\text{ ms}$ |
| `anomaly_detection` | `(1, 640)` | Multi-channel acoustic sensor | $\ge 0.85$ **AUC** upstream, simplified to $\ge 85\%$ accuracy here | $< 50\text{ ms}$ |
| `image_classification`| `(1, 32, 32, 3)` | 32×32 CIFAR-10 RGB stream | $\ge 85\%$ (ResNet-8) | $< 150\text{ ms}$ |
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
        - target_accuracy carries MLPerf Tiny's real quality targets (0.90 / 0.80 /
          0.85 / 0.85). max_latency_ms carries TinyTorch classroom thresholds:
          MLPerf Tiny measures latency, it does not gate on it
        - Store the seed itself, not a generator. Each phase calls
          np.random.default_rng(self.random_seed), so running the same
          benchmark twice draws the same inputs and the same synthetic labels.
          A seed that no phase reads makes random_seed a lie, and a benchmark
          whose seed does nothing is not reproducible no matter what it prints
        """
        ### BEGIN SOLUTION role="scaffold"
        self.random_seed = random_seed

        # Benchmark configurations. 'target_accuracy' is MLPerf Tiny's published
        # quality target; 'max_latency_ms' is a TinyTorch classroom threshold, since
        # MLPerf Tiny measures latency rather than gating on it.
        self.benchmarks = {
            'keyword_spotting': {
                'input_shape': (1, 16000),  # 1 second of 16kHz audio
                'target_accuracy': 0.90,
                'max_latency_ms': 100,  # classroom threshold, not an MLPerf rule
                'description': 'Wake word detection'
            },
            'visual_wake_words': {
                'input_shape': (1, 96, 96, 3),  # 96x96 RGB image
                'target_accuracy': 0.80,
                'max_latency_ms': 200,  # classroom threshold, not an MLPerf rule
                'description': 'Person detection in images'
            },
            'anomaly_detection': {
                'input_shape': (1, 640),  # Machine sensor data
                # MLPerf Tiny scores this task by AUC >= 0.85. This harness has one
                # scoring path (accuracy), so the target is carried as an accuracy
                # of 0.85. The number matches; the statistic does not.
                'target_accuracy': 0.85,
                'max_latency_ms': 50,  # classroom threshold, not an MLPerf rule
                'description': 'Industrial anomaly detection (AUC upstream, accuracy here)'
            },
            'image_classification': {
                'input_shape': (1, 32, 32, 3),  # CIFAR-10 style
                'target_accuracy': 0.85,  # MLPerf Tiny: ResNet-8 on CIFAR-10
                'max_latency_ms': 150,  # classroom threshold, not an MLPerf rule
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

    # MLPerf Tiny's published quality targets, which are the numbers a reader will
    # take away from this module. Image classification is 85% (ResNet-8 on
    # CIFAR-10), not 75%.
    assert perf.benchmarks['keyword_spotting']['target_accuracy'] == 0.90
    assert perf.benchmarks['visual_wake_words']['target_accuracy'] == 0.80
    assert perf.benchmarks['anomaly_detection']['target_accuracy'] == 0.85
    assert perf.benchmarks['image_classification']['target_accuracy'] == 0.85

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
    # anomaly_detection sits in the binary branch because this harness has one
    # scoring path. Real MLPerf Tiny scores it by AUC over an autoencoder's
    # reconstruction error, which needs the full score distribution rather than a
    # thresholded label, so the simplification is TinyTorch's and not the standard's.
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
**Why it matters**: Models return Tensors, numpy arrays, or lists, and we need to handle all of them
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

    # `0 <= acc <= 1` passes on a function that returns a constant. Pin the actual
    # agreement rate against labels we choose, at both ends of the range.
    labels = np.array([0, 1, 0, 1])
    perfect = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
               np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    assert perf._run_accuracy_test(model, perfect, 'keyword_spotting', 4, labels) == 1.0
    inverted = [p[::-1] for p in perfect]
    assert perf._run_accuracy_test(model, inverted, 'keyword_spotting', 4, labels) == 0.0
    half = perfect[:2] + inverted[2:]
    assert perf._run_accuracy_test(model, half, 'keyword_spotting', 4, labels) == 0.5

    # Multi-class must take the argmax, not the first or the largest index.
    mc_labels = np.array([3, 7])
    mc_preds = [np.eye(10)[3], np.eye(10)[7]]
    assert perf._run_accuracy_test(model, mc_preds, 'image_classification', 2, mc_labels) == 1.0

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
| **Compliance Gating** | Threshold comparison | $\text{compliant} \iff (\text{accuracy} \ge \text{target}) \land (p_{90} \le \text{limit})$ |

### Which Latency Statistic Gates, and How Many Samples It Needs

This harness is **single-stream**: one example per timed call, enforced by
`_run_accuracy_test`. MLPerf Inference's single-stream metric is the **90th
percentile**, so that is what the gate uses. The 99th percentile is the **server**
scenario's rule, and borrowing it here would import a tail bound from a workload this
harness does not run. The gated statistic and the printed statistic must be the same
one, or a reader sees a number beside a verdict it did not produce.

A percentile needs samples to mean anything, and this is where classroom benchmarks
quietly break:

| Samples $n$ | What `np.percentile(latencies, 99)` actually returns |
| :--- | :--- |
| $3$ | An interpolation $98\%$ of the way from the second value to the maximum |
| $5$ | Essentially the maximum: for `[1, 1, 1, 1, 50]` it returns $48.04$ |
| $100$ | The first honest estimate, and still a noisy one |
| $\ge 10{,}000$ | A stable tail estimate |

The unit tests below run $n = 3$ and $n = 5$ because they test the protocol, not
performance. Read any percentile from them as the maximum wearing a percentile's name.
MLPerf's server scenario mandates hundreds of thousands of queries for exactly this
reason, and it uses a **non-interpolating order statistic** (the
$\lceil 0.99 n \rceil$-th sorted sample) where `np.percentile` interpolates between
neighbors, so the two disagree on small $n$.
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
       at the 90th percentile, which is MLPerf Inference's single-stream metric.
       (p99 is the server scenario's rule and does not apply to a batch-1 harness.)

    HINTS:
    - Seed one generator from self.random_seed, and draw every input from it
    - Audio data: rng.standard_normal, Image data: rng.integers(0,256)/255
    - compliant = accuracy_met AND latency_met
    - Print the statistic you gated on. Printing the mean beside a p90 verdict
      leaves a reader staring at "50.0ms (target: <100ms)" next to FAIL
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
    print(f"   Target: {config['target_accuracy']:.1%} accuracy (MLPerf Tiny), "
          f"p90 < {config['max_latency_ms']}ms (TinyTorch classroom ceiling)")

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

    # Compile results. The gate is p90, MLPerf Inference's single-stream metric,
    # because every timed call here processes exactly one example. The mean and
    # p99 are reported alongside it but neither decides the verdict.
    mean_latency = float(np.mean(latencies))
    p90_latency = float(np.percentile(latencies, 90))
    p99_latency = float(np.percentile(latencies, 99))
    accuracy_met = bool(accuracy >= config['target_accuracy'])
    latency_met = bool(p90_latency <= config['max_latency_ms'])

    results = {
        'synthetic_labels': labels is None,
        'official_mlperf': False,
        'benchmark_name': benchmark_name,
        'model_name': getattr(model, 'name', 'unknown_model'),
        'accuracy': float(accuracy),
        'mean_latency_ms': mean_latency,
        'std_latency_ms': float(np.std(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p90_latency_ms': p90_latency,
        'p99_latency_ms': p99_latency,
        # The OBSERVED maximum, not a ceiling. The ceiling is target_latency_ms.
        'max_observed_latency_ms': float(np.max(latencies)),
        # One example per timed call, so 1/latency is a real rate. Not 'fps':
        # keyword spotting scores audio windows and there are no frames here.
        'inferences_per_second': float(1000 / mean_latency),
        'gating_statistic': 'p90_latency_ms',
        'target_accuracy': float(config['target_accuracy']),
        'target_latency_ms': float(config['max_latency_ms']),
        'accuracy_met': accuracy_met,
        'latency_met': latency_met,
        'compliant': accuracy_met and latency_met and labels is not None,
        'num_runs': int(num_runs),
        'random_seed': int(self.random_seed)
    }

    print(f"   Results: {accuracy:.1%} accuracy, {p90_latency:.1f}ms p90 latency (gated), "
          f"{mean_latency:.1f}ms mean")
    print(f"   Compliance: {'✅ PASS' if results['compliant'] else '❌ FAIL'}")

    return results
    ### END SOLUTION

MLPerf.run_standard_benchmark = mlperf_run_standard_benchmark

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

    required_keys = ['accuracy', 'mean_latency_ms', 'inferences_per_second', 'compliant',
                     'accuracy_met', 'latency_met', 'p50_latency_ms', 'p90_latency_ms',
                     'p99_latency_ms', 'max_observed_latency_ms', 'gating_statistic']
    assert all(key in result for key in required_keys), [k for k in required_keys if k not in result]
    assert 0 <= result['accuracy'] <= 1
    assert result['mean_latency_ms'] > 0
    assert result['inferences_per_second'] > 0
    assert isinstance(result['compliant'], bool)

    # The verdict must come from the statistic the harness says it gates on.
    assert result['gating_statistic'] == 'p90_latency_ms'
    assert result['latency_met'] == (result['p90_latency_ms'] <= result['target_latency_ms'])
    # One example per timed call, so the rate is exactly the reciprocal of the mean.
    assert abs(result['inferences_per_second'] - 1000 / result['mean_latency_ms']) < 1e-6
    # The observed maximum is a measurement; the ceiling is target_latency_ms.
    assert result['max_observed_latency_ms'] >= result['p99_latency_ms']

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
                # p90 is the gated statistic; the others are context.
                'p90_latency_ms': result.get('p90_latency_ms'),
                'p99_latency_ms': result['p99_latency_ms'],
                'inferences_per_second': result.get('inferences_per_second'),
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
            # Print the statistic the gate used (p90, single-stream), then the mean
            # as context. Printing only the mean beside a p90 verdict leaves the
            # reader with a passing-looking number next to FAIL and no explanation.
            gated = result.get('p90_latency_ms')
            gated_text = f"{gated:.1f}ms p90" if gated is not None else "p90 unavailable"
            summary_lines.append(
                f"  - Latency: {gated_text} (gated, target: <{result['target_latency_ms']}ms); "
                f"mean {result['mean_latency_ms']:.1f}ms")
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
            'accuracy': 0.92, 'mean_latency_ms': 50.0,
            'p90_latency_ms': 72.0, 'p99_latency_ms': 80.0,
            'inferences_per_second': 20.0, 'target_accuracy': 0.90, 'target_latency_ms': 100,
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
    # The gated statistic must survive into the report, or the scorecard cannot
    # show the number the verdict came from.
    assert report_data['benchmarks']['keyword_spotting']['p90_latency_ms'] == 72.0

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
                'accuracy': 0.92, 'mean_latency_ms': 50.0, 'p90_latency_ms': 72.0,
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
    # The scorecard must show the statistic the gate used, not only the mean.
    assert "72.0ms p90 (gated" in summary, (
        f"Summary reports no gated statistic:\n{summary}"
    )

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
    required_keys = ['accuracy', 'mean_latency_ms', 'inferences_per_second', 'compliant']
    assert all(key in result for key in required_keys)
    assert 0 <= result['accuracy'] <= 1
    assert result['mean_latency_ms'] > 0
    assert result['inferences_per_second'] > 0

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
| **Quantization** (INT8 from FP32) | $-5\%$ | $2.1\times$ | $4.0\times$ | $1.8\times$ | Precision loss in low dynamic range |
| **Structured Pruning** | $-2\%$ | $1.4\times$ | $3.2\times$ | $1.3\times$ | Sparse memory access vs density |
| **Knowledge Distillation** | $-8\%$ | $1.9\times$ | $1.5\times$ | $1.7\times$ | Dark knowledge transfer fidelity |

The challenge: Which is "best"? It depends entirely on your deployment constraints.

### Multi-Objective Decision Framework

Our comparison engine implements a decision framework that:

1. **Measures all dimensions**: Don't optimize in isolation
2. **Calculates efficiency ratios**: Accuracy per MB, accuracy per ms
3. **Identifies the Pareto frontier**: `pareto_frontier()` keeps a variant when no
   other variant is at least as good on *every* metric and strictly better on one
4. **Generates use-case recommendations**: Tailored to specific constraints, using
   stated weights (see `_generate_recommendations`, the one weighted step here)

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

Three of the four are unambiguous, each picking the maximum of one ratio. The fourth,
balanced deployment, is the **only** weighted decision in this module, and the weights
are worth stating because they are a choice and not a measurement:

$$\text{score} = \frac{\sum_{k \in \text{speedups}} \min(r_k,\, 5) \;+\; 5\,\rho_{\text{acc}}}{\lvert \text{speedups} \rvert + 1}$$

Accuracy retention $\rho_{\text{acc}}$ carries a weight of $5$ against $1$ per
speedup, so a variant that gives up $10\%$ of the baseline's accuracy must win roughly
$0.5\times$ of speedup somewhere to break even. Speedups are capped at $5\times$ so one
outlier ratio cannot outvote everything else. Change either number and the
recommendation changes, which is why the reason string carries both.

Compare this with `pareto_frontier()`, which weighs nothing and therefore decides
nothing. The frontier narrows the field; a weighted score picks from it. Keep the two
steps separate, so you always know which one moved the answer.
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
    - This is the module's ONE weighted decision. Put the weights in the reason
      string: an unexplained score reads like a measurement
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
            'reason': (f"Best weighted trade-off (score: {best_overall_score:.2f}; "
                       f"speedups capped at 5x, accuracy retention weighted 5x)"),
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
| **4. Frontier** | `pareto_frontier()` | Drops variants another variant beats on every metric at once |
| **5. Policy Recommendation** | `_generate_recommendations()` | Applies the one stated weighting to pick a balanced variant |
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-comparison", "solution": true}
#| export
def analyze_optimization_techniques(base_model: Any, optimized_models: List[Any],
                                  datasets: List[Any], simulate: bool = False,
                                  input_shape: Tuple[int, ...] = (1, 28, 28),
                                  output_dir: Optional[str] = None) -> Dict[str, Any]:
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
        output_dir: Where the suite may write reports and plots. None uses a
            temporary directory, so importing and calling this never creates
            a folder in the caller's working directory.

    Returns:
        Dictionary with 'base_metrics', 'optimized_results', 'improvements',
        'pareto_frontier', and 'recommendations'

    EXAMPLE:
    >>> results = analyze_optimization_techniques(base_model, [quant, pruned], datasets)
    >>> print(results['recommendations'])
    """
    ### BEGIN SOLUTION role="scaffold"
    all_models = [base_model] + optimized_models

    # A function that leaves ./benchmark_results/ behind on every import-and-call
    # is a side effect nobody asked for. Default to a directory that cleans itself.
    scratch = tempfile.TemporaryDirectory() if output_dir is None else None
    suite = BenchmarkSuite(all_models, datasets,
                           output_dir=output_dir if scratch is None else scratch.name)

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

    # Narrow the field before weighting it: the frontier over the four measured
    # objectives, baseline included, since the baseline can itself be non-dominated.
    frontier_points = {}
    for model_name in suite.benchmark.model_names:
        vector = tuple(benchmark_results[metric][model_name].mean
                       for metric in ('latency', 'accuracy', 'memory', 'energy')
                       if model_name in benchmark_results.get(metric, {}))
        if len(vector) == 4:
            frontier_points[model_name] = vector
    # Minimize latency, memory and energy; maximize accuracy.
    frontier = (pareto_frontier(frontier_points, (True, False, True, True))
                if frontier_points else [])
    comparison_results['pareto_frontier'] = frontier

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

    if frontier:
        dominated = [n for n in frontier_points if n not in frontier]
        print(f"\n🧭 Pareto frontier over (latency, accuracy, memory, energy): "
              f"{', '.join(frontier)}")
        print(f"   Dominated: {', '.join(dominated) if dominated else 'none'}")

    print("\n🎯 Recommendations:")
    for use_case, rec in recommendations.items():
        if rec['model']:
            print(f"  {use_case}: {rec['model']} - {rec['reason']}")

    if scratch is not None:
        scratch.cleanup()

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

    # Run comparison from inside a scratch directory. output_dir defaults to a
    # temporary directory, so the call must leave that scratch directory empty:
    # importing a function should never create a folder where you are standing.
    import tempfile
    with tempfile.TemporaryDirectory() as scratch_cwd:
        original_cwd = os.getcwd()
        try:
            os.chdir(scratch_cwd)
            results = analyze_optimization_techniques(
                base_model, [quantized_model, pruned_model], datasets, simulate=True)
            leftovers = sorted(p.name for p in Path(scratch_cwd).iterdir())
        finally:
            os.chdir(original_cwd)
    assert leftovers == [], f"The call wrote into the caller's directory: {leftovers}"

    # Verify results structure
    assert 'base_model' in results
    assert 'optimized_results' in results
    assert 'improvements' in results
    assert 'recommendations' in results
    assert 'pareto_frontier' in results

    # Verify improvements were calculated
    assert len(results['improvements']) == 2  # Two optimized models

    # The frontier covers all three models and is never empty.
    assert 1 <= len(results['pareto_frontier']) <= 3
    assert set(results['pareto_frontier']) <= (
        set(results['optimized_results']) | {results['base_model']})

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

    print(f"⚠️  SYNTHETIC DATA: draws from a known Normal(mu={true_latency} ms, "
          f"sigma={noise_std} ms), not timings of any model. Nothing was benchmarked.")
    print("   Holding the true distribution fixed is the only way to isolate the")
    print("   sample-size effect, and a real measurement cannot do it.\n")

    print("Effect of Sample Size on Confidence Interval Width:\n")
    print(f"{'Samples':<10} {'Mean (ms)':<12} {'t_0.975':<10} {'CI Width (ms)':<15} {'Relative Width':<15}")
    print("-" * 66)

    for n_samples in sample_sizes:
        # Draw from the known distribution
        measurements = rng.normal(true_latency, noise_std, n_samples)
        mean_latency = np.mean(measurements)
        std_latency = np.std(measurements, ddof=1)

        # 95% Student-t interval, the same critical value BenchmarkResult uses
        t_score = t_critical_95(n_samples - 1)
        margin_error = t_score * (std_latency / np.sqrt(n_samples))
        ci_width = 2 * margin_error
        relative_width = ci_width / mean_latency * 100

        print(f"{n_samples:<10} {mean_latency:<12.2f} {t_score:<10.3f} "
              f"{ci_width:<15.2f} {relative_width:<15.1f}%")

    # The sample size needed for a target width follows from the ratio the table
    # shows, so compute it rather than quoting a remembered rule of thumb. For a
    # relative CI width w: w = 2 * t * (sigma/mu) / sqrt(n).
    cv = noise_std / true_latency
    needed = next(n for n in range(2, 1000)
                  if 2 * t_critical_95(n - 1) * cv / np.sqrt(n) < 0.10)
    at_twenty = 2 * t_critical_95(19) * cv / np.sqrt(20) * 100

    print("\n💡 Key Insights:")
    print("   • More samples reduce confidence interval width")
    print("   • CI width decreases with √n (diminishing returns)")
    print(f"   • At sigma/mu = {cv:.2f} the EXPECTED width at n=20 is {at_twenty:.1f}%, and")
    print(f"     n={needed} is the first size expected under 10%. A single draw lands")
    print("     either side of that, which is what the table's scatter shows")
    print("   • So 'use 20 samples' is not a rule, it is a number that depends on")
    print("     the noise you actually have. Measure the noise, then pick n")
    print("   • The critical value is Student's t, not 1.96: at n=10 the constant")
    print("     1.96 reports an interval 13% narrower than the samples support")

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

**MLPerf Tiny:** Four reference benchmarks for microcontroller-class devices (keyword spotting, visual wake words, anomaly detection, image classification) that inspire the capstone. Each defines a **quality target** and then **measures** latency and energy. It sets no model-size, latency, or power limit a submission must clear, which is why the millisecond ceilings in this module are labeled as TinyTorch's own.

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

Strategic optimization combines multiple techniques for different performance goals. The order matters. Quantize-then-prune may preserve accuracy better, while prune-then-quantize may be faster.

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
    test_unit_pareto_frontier()
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
            # Deterministic per (model, dataset), and clipped into [0, 1].
            # Drawing from the shared module-level rng made this mock
            # order-dependent: the score changed with how many draws earlier
            # cells had taken, and a 0.95 base could wander above 1.0 and trip
            # run_accuracy_benchmark's range check.
            base_acc = self.characteristics.get('base_accuracy', 0.85)
            seed = (sum(self.name.encode()) + sum(repr(dataset).encode())) % 100_000
            probe = np.random.default_rng(seed)
            return float(np.clip(base_acc + probe.normal(0, 0.02), 0.0, 1.0))

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

    # Test 1: Comprehensive benchmark suite. The output directory is a temporary
    # one: running a module top to bottom must not leave ./benchmark_results/
    # behind in the student's working directory.
    import tempfile
    tmp_output = tempfile.TemporaryDirectory()
    print("  Testing comprehensive benchmark suite...")
    suite = BenchmarkSuite(models, datasets, output_dir=tmp_output.name)
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
            # n = 1 has no interval at all; anything larger must bracket the mean.
            if result.count == 1:
                assert result.ci_lower is None and result.ci_upper is None
            else:
                assert result.ci_lower <= result.mean <= result.ci_upper

    # Test 3: Report generation
    print("  Testing report generation...")
    report = suite.generate_report()
    assert "Benchmark Report" in report
    assert "System Information" in report
    assert "Recommendations" in report
    # Every reported number carries its unit and its interval, or the table is
    # four decimal places of nothing comparable.
    assert "Latency Results [ms]" in report and "Memory Results [MB]" in report
    assert "95% CI [" in report, "The report claims a 95% CI it never prints"

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
        models[0], models[1:], datasets[:1], simulate=True, output_dir=tmp_output.name
    )

    assert 'base_model' in comparison_results
    assert 'improvements' in comparison_results
    assert 'recommendations' in comparison_results
    assert 'pareto_frontier' in comparison_results
    assert len(comparison_results['improvements']) == 2
    # The frontier is a subset of the models measured, and never empty.
    frontier = comparison_results['pareto_frontier']
    assert 1 <= len(frontier) <= 3
    assert set(frontier) <= set(comparison_results['optimized_results']) | {comparison_results['base_model']}

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

    tmp_output.cleanup()

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
  $s$ is estimated from the same 20 samples, so the critical value is Student's $t$
  with $\nu = 19$ degrees of freedom, $t_{0.975, 19} = 2.093$:
  $$\text{Margin of Error} = 2.093 \times 0.1789 \approx 0.3744\text{ ms} \implies \mathbf{[4.83\text{ ms},\ 5.57\text{ ms}]}$$
  *(Substituting the normal $z_{0.975} = 1.96$ gives margin $0.3506\text{ ms}$ and
  $[4.85, 5.55]$, an interval $6.4\%$ too narrow. `BenchmarkResult` uses the $t$
  value, which is why `t_critical_95()` carries a table rather than a constant.)*
- **How many more trials would you need to halve the confidence interval width?**
  $$\text{Width} \propto \frac{1}{\sqrt{n}} \implies \frac{\text{Width}_{\text{new}}}{\text{Width}_{\text{old}}} = \frac{1}{2} \implies \sqrt{\frac{n_{\text{new}}}{n_{\text{old}}}} = 2 \implies n_{\text{new}} = 4 \times n_{\text{old}} = 4 \times 20 = \mathbf{80\text{ total trials}}$$
  *(You would need $80 - 20 = \mathbf{60\text{ additional trials}}$).*

---

### Question 2: Measurement Overhead Analysis
`precise_timer` samples `time.perf_counter()` twice per measurement. On this machine a
single `perf_counter()` call costs about $35\text{ ns}$ (see the timing table in 🏗️),
so the overhead a measurement pays is $\delta_{\text{timer}} \approx 70\text{ ns}$.

- **For a model that takes $1.0\text{ ms}$, what's the relative error?**
  $$\text{Relative Error} = \frac{\delta_{\text{timer}}}{T_{\text{exec}}} = \frac{0.07\text{ }\mu\text{s}}{1000\text{ }\mu\text{s}} \times 100\% = \mathbf{0.007\%}$$
  Negligible, and this is the usual case. Timer overhead is almost never what
  distorts a benchmark; OS scheduling ($\delta_{\text{OS}}$, microseconds to
  milliseconds) is, and it is three to five orders of magnitude larger.
- **Below what execution time does timer overhead exceed $1\%$?**
  $$T_{\text{exec}} < \frac{\delta_{\text{timer}}}{0.01} = \frac{70\text{ ns}}{0.01} = 7\text{ }\mu\text{s}$$
  *Systems implication: individual operator timing only becomes clock-limited in the
  single-digit-microsecond range, which is roughly one $256 \times 256$ matmul. Below
  that, amortize by running an internal loop of $K = 100\text{--}1000$ iterations
  inside one timer block and dividing. Note what the amortization does and does not
  buy. It removes the clock cost, and it leaves the $\delta_{\text{OS}}$ term
  untouched, because a preemption inside the loop is still inside the interval.*

---

### Question 3: Benchmark Configuration Trade-offs
`BenchmarkSuite` forwards `warmup_runs` and `measurement_runs` to its `Benchmark`
(defaults `DEFAULT_WARMUP_RUNS = 5`, `DEFAULT_MEASUREMENT_RUNS = 10`), so a CI pipeline
can pick a configuration per stage. Latency cost per model is roughly
$(\text{warmup} + \text{measurement}) \times T_{\text{forward}}$.

For a model with $T_{\text{forward}} = 20\text{ ms}$, benchmarked across 100 models:

- **Fast config** (`warmup_runs=2, measurement_runs=5`): $7 \times 20\text{ ms} = 0.14\text{ s}$ per model, $\mathbf{14\text{ s}}$ for the suite.
- **Accurate config** (`warmup_runs=5, measurement_runs=40`): $45 \times 20\text{ ms} = 0.9\text{ s}$ per model, $\mathbf{90\text{ s}}$ for the suite.
- **What does the extra $76\text{ s}$ actually buy?** Interval width, and only as
  $1/\sqrt{n}$. Going from $n=5$ to $n=40$ narrows the confidence interval by
  $\sqrt{40/5} = 2.83\times$, and the $t$ critical value drops from $2.776$ to
  $2.023$, for a total narrowing of about $3.9\times$. Run
  `analyze_benchmark_variance()` to see this table for your own noise level.
- **What's the key trade-off you're making?** **Regression detection threshold vs
  feedback latency.** At the $\sigma/\mu = 0.15$ noise level
  `analyze_benchmark_variance()` uses, the fast config's own interval is
  $\pm 18.6\%$, so it cannot resolve a $5\%$ regression. It will neither catch it nor
  report it as unresolvable, and a gate that
  cannot see the thing it guards against is worse than no gate, because it grants
  confidence. Size $n$ from the smallest regression you need to catch, not from the
  wall-clock budget, and if the two disagree, benchmark fewer models rather than
  benchmarking all of them badly.

---

### Question 4: MLPerf Compliance Metrics
You implemented MLPerf-style standardized benchmarks with target thresholds.
If an edge candidate model achieves 89% accuracy (MLPerf Tiny target: 90%) and a $p_{90}$
latency of 120ms (TinyTorch classroom ceiling: <100ms):

- **Is it compliant?** **No**. This harness gates on a strict conjunction ($\text{Acc} \ge \text{Target} \land p_{90} \le \text{Threshold}$). Both are violated ($89\% < 90\%$ and $120\text{ ms} > 100\text{ ms}$). Note which half of that is a real MLPerf rule. The accuracy target is; the latency ceiling is ours. In official MLPerf Tiny this submission misses the quality target and is simply reported with its latency, whatever that latency is.
- **Which constraint is more critical for edge deployment?** **Latency**. Latency on edge devices is a hard real-time physical deadline dictated by sensor sampling rates (e.g. 10 fps camera stream requires $< 100\text{ ms}$ processing), UI responsiveness, or watchdog timeouts. Dropping below the latency ceiling causes dropped sensor frames or system lockup, whereas an accuracy delta of $1\%$ is typically tolerable.
- **How would you prioritize optimization?** **Latency-first**. First compress/accelerate the model to reliably meet the $\le 100\text{ ms}$ deadline with buffer room, then tune hyper-parameters or calibration datasets to recover the remaining accuracy gap.

---

### Question 5: Optimization Comparison Analysis
Your `analyze_optimization_techniques()` generates recommendations for different use cases.
Start from an FP32 baseline measuring $120\text{ MB}$ and $180\text{ ms}$, and three
optimized variants:
- **Quantized** (INT8 from FP32): $0.25\times$ memory footprint (75% reduction, the exact $4\times$ ratio of 4 bytes to 1), $2.0\times$ speedup, $0.95\times$ accuracy
- **Pruned**: $0.3\times$ memory footprint (70% reduction), $1.5\times$ speedup, $0.98\times$ accuracy
- **Distilled**: $0.6\times$ memory footprint (40% reduction), $1.8\times$ speedup, $0.92\times$ accuracy

For a mobile app with a $50\text{ MB}$ model size limit and a strict $< 100\text{ ms}$ latency requirement:
- **Which optimization offers best memory reduction?** **Quantized** ($0.25\times$, a $75\%$ reduction). INT8 from FP32 stores one byte where four were stored, so the ratio is exactly $4\times$ and not a tunable number. Pruning's $0.3\times$ is close but is a *modeled* figure, because zeroed weights still occupy dense storage unless you also change the format, which is why `analyze_optimization_tradeoffs()` above reports the same payload bytes for baseline and pruned.
- **Which variants actually satisfy both constraints?** Apply them:

  | Variant | Size | $\le 50\text{ MB}$? | Latency | $< 100\text{ ms}$? |
  | :--- | :--- | :--- | :--- | :--- |
  | Quantized | $120 \times 0.25 = 30\text{ MB}$ | ✅ | $180 / 2.0 = 90\text{ ms}$ | ✅ |
  | Pruned | $120 \times 0.3 = 36\text{ MB}$ | ✅ | $180 / 1.5 = 120\text{ ms}$ | ❌ |
  | Distilled | $120 \times 0.6 = 72\text{ MB}$ | ❌ | $180 / 1.8 = 100\text{ ms}$ | ❌ (not strictly under) |

  Only **Quantized** clears both, so the answer is Quantized despite Pruned retaining
  more accuracy. This is what a hard constraint does: it removes candidates from
  consideration before any trade-off is weighed, and the $0.98$ retention that would
  have won an unconstrained comparison never gets to compete.
- **What's the key insight about optimization trade-offs?** **The frontier filters,
  constraints filter again, and only then do weights choose.** Run

  ```python
  pareto_frontier({'Quantized': (0.25, 2.0, 0.95),
                   'Pruned':    (0.30, 1.5, 0.98),
                   'Distilled': (0.60, 1.8, 0.92)},
                  (True, False, False))   # minimize memory, maximize speedup and accuracy
  ```

  and you get `['Quantized', 'Pruned']`. **Distilled is dominated**: Quantized is
  smaller *and* faster *and* more accurate, so nothing about any deployment could
  make Distilled the right answer, and no weighting needs to be argued about. That is
  what the frontier is for. It does not rank Quantized against Pruned, because
  neither beats the other on every axis; the $50\text{ MB}$ and $100\text{ ms}$
  constraints do that, and they remove Pruned. Had both survived, you would still owe
  an explicit weighting, and the weights would be a statement about your deployment
  rather than a fact about the models.
"""

# %% [markdown]
r"""
## ⭐ Aha Moment: Measurement Enables Optimization

**What you built:** A benchmarking system with warmup, statistics, and reproducibility.

**Why it matters:** "Premature optimization is the root of all evil", but you cannot
optimize what you have not measured. Your benchmarking system produces reliable,
comparable numbers. Warmup iterations discard cold-start effects, repeated runs give a
Student-$t$ confidence interval, and every reported number carries its unit.

This is how production ML teams make decisions. Measure, compare, improve, repeat.
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

    print("Model: Linear(512 → 256)")
    print("Batch: 32 samples")
    print(f"\nBenchmark Results ({result.count} iterations):")
    print(f"  Mean latency: {result.mean:.2f} ms")
    print(f"  Std dev:      {result.std:.2f} ms")
    print(f"  Min:          {result.min_val:.2f} ms")
    print(f"  Max:          {result.max_val:.2f} ms")
    print(f"  95% CI:       [{result.ci_lower:.2f}, {result.ci_upper:.2f}] ms "
          f"(Student-t, {result.count - 1} dof)")

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

### Key Accomplishments

| Milestone Capability | Mathematical / Systems Mechanism | TinyTorch Implementation | Production Parallel |
| :--- | :--- | :--- | :--- |
| **Statistical Rigor** | Sample mean $\mu$, variance $s^2$, standard error $\frac{s}{\sqrt{n}}$, and a Student-$t$ CI from a 30-entry $t_{0.975,\nu}$ table | `BenchmarkResult` | Google Benchmark, Criterion.rs |
| **Monotonic Timing** | Monotonic userspace vDSO clock with nanosecond counter | `precise_timer()` | `clock_gettime(CLOCK_MONOTONIC)` |
| **Warmup Discard** | Cold-start page fault & cache warming isolation | `Benchmark.run_latency_benchmark()` | MLPerf Tiny warmup harness |
| **Memory Accounting** | Peak allocator buffer tracking vs process RSS | `Benchmark.run_memory_benchmark()` | PyTorch CUDA Caching Allocator profiler |
| **Standardized Tasks** | Fixed seeds, input shapes, real MLPerf Tiny quality targets, and a $p_{90}$ single-stream latency gate | `MLPerf` class | MLPerf Inference & Mobile Benchmark Suite |
| **Pareto Filtering** | Non-dominated set under one direction flag per objective | `pareto_frontier()` | Optuna, Neural Network Intelligence (NNI) |
| **Weighted Selection** | Capped speedups plus $5\times$ accuracy retention, weights stated in the output | `analyze_optimization_techniques()` | Ax, Vizier multi-objective schedulers |

### Systems Insights Discovered
- **Measurement Science**: Single-run latency numbers are noise; true systems characterization requires isolated warmup and statistical confidence intervals.
- **The Critical Value Matters**: At $n = 10$ the normal constant $1.96$ reports an interval $13\%$ narrower than $t_{0.975,9} = 2.262$ allows. A CI is a claim, and the claim has to match the estimator.
- **Metric Dimensionality**: Optimizing for speed without tracking memory or accuracy creates brittle models that fail silent SLA requirements.
- **Filtering Is Not Choosing**: A Pareto frontier removes the variants nothing could justify and ranks nothing. Any single "best overall" number is a weighting, and a weighting that is not printed is a decision nobody reviewed.
- **Hardware Realities**: A `perf_counter()` call costs about $35\text{ ns}$, so timer overhead only matters below roughly $7\text{ }\mu\text{s}$ of work. Scheduling jitter, three to five orders of magnitude larger, is what actually distorts a benchmark.
- **Percentiles Need Samples**: A $p_{99}$ over five samples is the maximum wearing a percentile's name, which is why MLPerf's server scenario mandates hundreds of thousands of queries.
- **Production Integration**: Objective compliance checks and machine-readable JSON reports bridge experimental training to operational deployment.

### Ready for Next Steps
Your benchmarking framework completes the Optimization Tier. Every component, from Tensors, Autograd, Convolutions, and Transformers to Quantization, Acceleration, Memoization, and Benchmarking, is now verified.

Export with: `tito module complete 19`

**Next**: Module 20 (Capstone) will integrate every single concept into an end-to-end production ML system!
"""
