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
# Module 20: Capstone - Benchmarking & Submission

Welcome to the TinyTorch capstone! You've built an entire deep learning framework from scratch across 19 foundational modules. Now it is time to demonstrate your engineering achievements by systematically benchmarking a neural architecture, applying compound optimizations, and generating a verified submission that proves framework performance.

<img src="capstone_blueprint.svg" alt="Capstone Framework Blueprint" width="100%">

## 🔗 Prerequisites & Progress

| Systems Dimension | Previous Modules (01–19) | Capstone Synthesis (Module 20) | Production Parallel |
| :--- | :--- | :--- | :--- |
| **Foundation Engine** | Tensors, Autograd, Optimizers, Training | Complete verified neural pipeline execution | PyTorch / LibTorch C++ Core |
| **Architectures** | Convolutions (M09), Attention & Transformers (M12–M13) | Multi-layer perceptron & vision baselines | Hugging Face Transformers |
| **Optimization Stack**| INT8 Quant (M15), Pruning (M16), Fusion (M17), KV Cache (M18) | Compound optimization stacking & Pareto frontiers | TensorRT, ONNX Runtime |
| **Measurement Rigor** | Monotonic timing, warmup discard, confidence intervals (M19) | Schema-validated standardized benchmark submission | MLPerf Inference & Mobile Suite |

### Architectural Roadmap

| Stage | Module Focus | Core Deliverable | Verification Target |
| :--- | :--- | :--- | :--- |
| **1. Measurement** | Latency vs Throughput separation | Monotonic interval timing via `precise_timer()` | Zero cold-start cache distortion |
| **2. Optimization** | Precision & sparsity transforms | Combined INT8 quantization and structured pruning | Compounding memory & latency gains |
| **3. Validation** | Standardized JSON schema | Strict type checking & range constraints | Schema-certified reproducible report |
| **4. Submission** | Olympic events & qualification | Multi-objective tradeoff documentation | Pareto dominance over baseline |

## 🎯 Learning Objectives
By the end of this capstone, you will:
1. Use Module 19's `precise_timer` to measure latency and throughput as two distinct physical measurements
2. Apply optimization techniques from Modules 15 and 16 to improve a baseline model
3. Generate standardized JSON submissions following industry best practices
4. Validate submissions against a strict schema for end-to-end reproducibility
5. Compare baseline vs. optimized models with quantitative Pareto efficiency metrics
6. Share your results with the TinyTorch community in an automated, machine-readable format

## 📦 Where This Code Lives in the Final Package

<img src="capstone_source_card.svg" alt="Source Code Mapping" width="100%">

**Learning Side:** You work in `modules/20_capstone/capstone.ipynb`  
**Building Side:** Code exports to `tinytorch.olympics`

```python
# Final package structure:
from tinytorch.olympics import BenchmarkReport, generate_submission, save_submission, validate_submission_schema
from tinytorch.olympics import OlympicEvent, qualifies_event

# Benchmark your model
report = BenchmarkReport(model_name="my_model")
report.benchmark_model(my_model, X_test, y_test)

# Generate, validate, and save the submission (a plain dict, written as JSON)
submission = generate_submission(report)
validate_submission_schema(submission)
# Eligibility is separate from a well-formed submission
qualifies_event(report.metrics, OlympicEvent.LATENCY_SPRINT)
save_submission(submission, "my_submission.json")
```

**Why this matters:**
- **Learning:** Complete workflow from raw model weights to shareable, reproducible evidence
- **Production:** Standardized submission protocol mirroring MLPerf and Papers with Code standards
- **Community:** Objective apples-to-apples performance comparisons across distinct hardware architectures
- **Reproducibility:** Strict schema validation ensures experimental findings can be replicated and audited
"""

# %% [markdown]
r"""
## 📋 Module Dependencies

| Dependency Type | Component / Module | Systems Function | Technical Role |
| :--- | :--- | :--- | :--- |
| **External Runtime** | `numpy`, `time`, `json`, `pathlib`, `platform`, `enum` | OS interface & serialization | High-resolution timing, platform detection, JSON encoding |
| **Core Primitives** | `tinytorch.core.tensor`, `tinytorch.core.layers` | Computational substrate | Tensor allocations, affine projections (`Linear`), activations (`ReLU`) |
| **Optimization Stack**| `tinytorch.perf.profiling`, `quantization`, `compression` | Efficiency transformations | Memory tracing (M14), INT8 conversion (M15), magnitude pruning (M16) |
| **Benchmarking Engine**| `tinytorch.perf.benchmarking.precise_timer` | High-precision measurement | Monotonic nanosecond timing with guaranteed `finally` cleanup |

Students completing this module demonstrate their complete framework's capabilities through reproducible benchmarking and professional submission generation.
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": false}
#| default_exp olympics
#| export
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import platform
from enum import Enum
import sys

# TinyTorch modules the capstone builds on
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU
from tinytorch.perf.benchmarking import precise_timer  # Module 19's timing context manager

# One generator for the two example workflows below. The unit tests seed their
# own generators so a test's numbers never depend on which cells ran before it.
rng = np.random.default_rng(7)

# %% [markdown]
"""
## 💡 Introduction: From Framework to Reproducible Results

Over the past 19 modules, you built a complete ML framework from the ground up. You implemented tensors, layers, optimizers, loss functions, and advanced optimization techniques. But building a framework is only half the story.

**The Missing Piece: Proving It Works**

In production ML systems, claims without measurements are worthless. When researchers publish papers or engineers deploy models, they need to answer fundamental questions:
- How fast is inference on this hardware?
- How much memory does the model consume?
- What's the accuracy-latency trade-off?
- How do optimizations affect these metrics?

### The Reproducibility Crisis in ML

Modern ML faces a reproducibility crisis. Many published results can't be replicated because:
- **Missing system details** - What hardware? What software versions?
- **Inconsistent metrics** - Different ways to measure "accuracy" or "latency"
- **Cherry-picked results** - Showing best runs without variance
- **Incomplete reporting** - Omitting negative results or failed optimizations

#### Industry Standard: Benchmarking Frameworks

Professional ML systems rely on standardized, community-verified benchmarking frameworks:

| Benchmark Consortium | Primary Domain | Core Standards & Methodologies | Quality Guarantees |
| :--- | :--- | :--- | :--- |
| **MLPerf (MLCommons)** | AI Hardware & Systems | Fixed tasks, datasets, warmup rules, and percentile latency constraints | Level-playing-field evaluation across microarchitectures |
| **Papers with Code** | Academic Research | Public leaderboards, tracked checkpoints, mandatory open reproducibility | Verifiable algorithmic claims backed by code artifacts |

### What This Capstone Teaches You

This module shows you how to:
1. **Measure comprehensively** - Not just accuracy, but latency, memory, throughput
2. **Report systematically** - Following a schema that ensures completeness
3. **Enable comparison** - Using standardized metrics others can verify
4. **Document optimizations** - Tracking what techniques were applied and their impact
5. **Share professionally** - Generating submission files that work like research papers

Let's build the benchmarking and submission system.
"""

# %% [markdown]
r"""
## 📐 Foundations: The Science of Benchmarking

Before we build our submission system, let's understand what makes a good benchmark and why standardized reporting matters.

### The Three Pillars of Good Benchmarking

| Pillar | Operational Principle | Required Controls | Systems Failure Mode Prevented |
| :--- | :--- | :--- | :--- |
| **Repeatability** | Identical inputs produce statistically consistent distributions | Seeded PRNG (`default_rng`), fixed datasets, $N=100$ trials | Single-run outlier distortion (GC pauses, OS jitter) |
| **Comparability** | Fair apples-to-apples evaluation across model variants | Identical hardware platform, shared evaluation sets | Incomparable heterogeneous execution environments |
| **Completeness** | Multi-dimensional characterization | Quality, Latency, Memory, Throughput | Blind optimization sacrificing latency for parameter size |

### What Metrics Actually Matter?

Different engineering stakeholders care about distinct system dimensions:

| Engineering Persona | Primary Metric | Secondary Metric | Production SLA Focus |
| :--- | :--- | :--- | :--- |
| **ML Researcher** | Task accuracy ($\text{Top-1}$), $F_1$, BLEU | Training epochs, loss convergence | Generalization and model capacity |
| **Systems Engineer** | Latency ($p_{50}, p_{99}$ tail), throughput | Allocator peak memory, cache misses | Real-time interactive deadlines ($< 100\text{ ms}$) |
| **Product Manager** | End-to-end user perceived latency | Cloud infrastructure cost per query | Retention, conversion, and service reliability |
| **DevOps / MLOps** | Binary artifact footprint (MB), DRAM RSS | Concurrency headroom, thermal throttle | Resource density in serverless or edge silicon |

**Key Insight**: A complete benchmark captures ALL perspectives, not just one.

### Benchmark Report Components

Our `BenchmarkReport` class tracks everything required for end-to-end reproducibility:

| Metadata Category | Tracked Attributes | Systems Role |
| :--- | :--- | :--- |
| **Model Characteristics** | `parameter_count`, `model_size_mb` | Parameter storage and hardware memory footprint |
| **Inference Latency** | `latency_ms_mean`, `latency_ms_std`, `median` | Execution speed and statistical dispersion across runs |
| **Throughput** | `throughput` ($\text{samples}/\text{sec}$) | Batch processing capacity under sustained load |
| **Model Quality** | `accuracy` ($\% \text{ or ratio}$) | Preservation of functional predictive accuracy |
| **System Context** | Platform, processor, Python / NumPy version, timestamp | Exact hardware and software environment for reproducibility |

### Latency vs. Throughput: A Critical Distinction

Many engineers conflate latency and throughput. They quantify distinct physical characteristics:

$$\text{Latency} = T_{\text{batch}=1} \quad [\text{ms}]$$

$$\text{Throughput} = \frac{N_{\text{batch}}}{T_{\text{batch}}} \quad \left[\frac{\text{samples}}{\text{second}}\right]$$

By Little's Law, the maximum concurrency $L$ that an engine can sustain without queuing delay is:

$$L = \text{Throughput} \times \text{Latency}$$

| Performance Metric | Physical Definition | Primary Use Case | Scaling Behavior |
| :--- | :--- | :--- | :--- |
| **Latency ($T_1$)** | Time to process a single sample from arrival to return | Interactive web APIs, autonomous control, robotics | Bounded by memory bandwidth and kernel launch latency |
| **Throughput ($\Phi$)** | Total input samples evaluated per unit time | Offline indexing, batch inference, dataset preprocessing | Scales with tensor parallelism and larger GEMM tile sizes |

Because the two pull in opposite directions, `BenchmarkReport` measures them with two different calls: latency times `model.forward` on one sample, and throughput times `model.forward` on the whole test batch and divides the batch size by that time. Deriving one from the other ($1000 / \text{latency\_ms}$) would erase exactly the trade-off this relationship exposes.

### Why Variance Matters

Single measurements lie. Reporting statistical dispersion reveals true system predictability:

| Measurement Sequence | Individual Trials (ms) | Mean ($\mu$) | Standard Deviation ($s$) | Production Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Engine A (Consistent)** | $9.2, 10.1, 9.8, 10.5, 10.4$ | $10.0\text{ ms}$ | $0.5\text{ ms}$ | Predictable real-time compliance; low tail jitter |
| **Engine B (Erratic)** | $5.2, 14.8, 8.1, 15.3, 6.6$ | $10.0\text{ ms}$ | $4.2\text{ ms}$ | Unacceptable tail spikes; causes frame drops and timeouts |

**Which model would you deploy?** Engine A, because deterministic execution is essential for production SLAs.

### The Submission Schema: Enforcing Standards

Our submission format follows a JSON schema that ensures:
- **Required fields** can't be omitted (no incomplete results)
- **Type safety** prevents errors (accuracy is float, not string)
- **Version tracking** allows format evolution
- **Nested structure** organizes related data logically

```
Submission JSON Schema:
{
  "tinytorch_version": "0.1.0",           ← Version tracking
  "submission_type": "capstone_benchmark", ← Classification
  "timestamp": "2025-01-15 14:30:00",     ← When run
  "system_info": {                         ← Environment
    "platform": "macOS-14.0-arm64",
    "python_version": "3.11.0",
    "numpy_version": "1.24.0"
  },
  "baseline": {                            ← Required baseline
    "model_name": "simple_mlp",
    "metrics": {
      "parameter_count": 1000,
      "model_size_mb": 0.004,
      "accuracy": 0.92,
      "latency_ms_mean": 0.15,
      "latency_ms_std": 0.02,
      "throughput_samples_per_sec": 6666.67
    }
  },
  "optimized": {                           ← Optional optimization
    "model_name": "quantized_mlp",
    "metrics": { ... },
    "techniques_applied": ["int8_quantization", "pruning"]
  },
  "improvements": {                        ← Auto-calculated
    "speedup": 2.3,
    "compression_ratio": 4.1,
    "accuracy_delta": -0.01
  }
}
```

This structure makes it trivial to:
- **Validate** submissions programmatically
- **Compare** different models objectively
- **Aggregate** results across the community
- **Visualize** trends and trade-offs

Now let's build it!
"""

# %% [markdown]
"""
## 🏗️ Implementation: Building a Simple Benchmark Model

For this capstone, we'll use a simple MLP model. This keeps the focus on the benchmarking workflow rather than model complexity.

**Why a Simple Model?**
- **Focus on workflow** - The submission process is the learning goal, not model architecture
- **Fast iteration** - Quick benchmarks let you experiment with the pipeline
- **Extensible pattern** - Same workflow applies to complex models from milestones

Students can later apply this exact workflow to more sophisticated models (CNNs, Transformers, etc.) from milestone projects!
"""

# %% nbgrader={"grade": false, "grade_id": "toy-model", "solution": true}
#| export
class SimpleMLP:
    """
    Simple 2-layer MLP for benchmarking demonstration.

    This is a toy model to demonstrate the benchmarking workflow.
    Students can later apply the same workflow to milestone models.

    Architecture:
        Input → Linear(in, hidden) → ReLU → Linear(hidden, out) → Output

    Why this design:
    - Two layers: Enough to show optimization impact (quantization, pruning)
    - ReLU activation: Common pattern students recognize
    - Small by default: Fast benchmarking during development
    - Configurable sizes: Can scale up for experiments
    """
    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        """
        Initialize simple MLP with random weights.

        TODO: Create a 2-layer MLP with ReLU activation

        APPROACH:
        1. Create fc1 Linear layer (input_size -> hidden_size)
        2. Create ReLU activation
        3. Create fc2 Linear layer (hidden_size -> output_size)

        HINTS:
        - Use Linear(in_features, out_features) for layers; Module 03's Linear
          already initializes weights sensibly and biases to zero
        """
        ### BEGIN SOLUTION
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through the network.

        TODO: Implement forward pass through fc1 -> ReLU -> fc2

        APPROACH:
        1. Pass input through fc1 (first linear layer)
        2. Apply ReLU activation
        3. Pass through fc2 (second linear layer)
        4. Return output

        HINTS:
        - Call layer.forward(x) for each layer
        - Order matters: linear -> activation -> linear
        """
        ### BEGIN SOLUTION
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        return x
        ### END SOLUTION

    def parameters(self):
        """Return every parameter (fc1 first, then fc2) so count_parameters,
        Module 14's Profiler, and Module 16's magnitude_prune can walk them."""
        return self.fc1.parameters() + self.fc2.parameters()

    def count_parameters(self):
        """Count total number of parameters."""
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total

# %% [markdown]
r"""
### Understanding SimpleMLP Parameter Counting

Let's break down where the parameter allocations originate in the network hierarchy:

| Layer Index & Module | Tensor Dimension & Shape | Parameter Arithmetic | Weight + Bias Subtotal |
| :--- | :--- | :--- | :--- |
| **Layer 1: `Linear(10, 20)`** | Weights: $(10, 20)$, Biases: $(20,)$ | $10 \times 20 + 20$ | $220\text{ parameters}$ |
| **Layer 2: `ReLU()`** | Element-wise $\max(0, x)$ | Non-parametric | $0\text{ parameters}$ |
| **Layer 3: `Linear(20, 3)`** | Weights: $(20, 3)$, Biases: $(3,)$ | $20 \times 3 + 3$ | $63\text{ parameters}$ |
| **Total Architecture** | Full multi-layer perceptron | $220 + 0 + 63$ | $\mathbf{283\text{ parameters}}$ |

### Memory Footprint Derivations

The storage requirement scales directly with the numerical data type ($\text{dtype}$):

$$\text{Memory}_{\text{FP32}} = 283\text{ params} \times 4\text{ bytes} = 1,132\text{ bytes} \approx 1.13\text{ KB} \quad (0.00113\text{ MB})$$

Quantizing the parameter tensors to INT8 (Module 15):

$$\text{Memory}_{\text{INT8}} = 283\text{ params} \times 1\text{ byte} = 283\text{ bytes} \approx 0.28\text{ KB} \quad (0.00028\text{ MB})$$

$$\text{Compression Factor} = \frac{1,132\text{ bytes}}{283\text{ bytes}} = \mathbf{4.0\times\text{ memory reduction}}$$

This compact architecture provides an ideal benchmark probe to measure optimization compounding without lengthy test suite runtimes.
"""

# %% [markdown]
"""
### 🧪 Unit Test: SimpleMLP

This test validates the SimpleMLP model works correctly for benchmarking demonstrations.

**What we're testing**: Model creation, parameter counting, and forward pass
**Why it matters**: The model must work correctly before we can benchmark it
**Expected**: Correct output shapes and no NaN values
"""

# %% nbgrader={"grade": true, "grade_id": "test-simple-mlp", "locked": true, "points": 10}
def test_unit_simple_mlp():
    """🧪 Test SimpleMLP model creation and forward pass."""
    print("🧪 Unit Test: SimpleMLP...")

    # Test model creation with default parameters
    model = SimpleMLP()
    assert model is not None, "Model should be created"

    # Test with custom parameters
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)

    # Test parameter count
    param_count = model.count_parameters()
    expected_params = (10 * 20 + 20) + (20 * 3 + 3)  # fc1 + fc2
    assert param_count == expected_params, f"Expected {expected_params} parameters, got {param_count}"

    # Test forward pass
    rng = np.random.default_rng(7)
    X = Tensor(rng.standard_normal((5, 10)))  # 5 samples, 10 features
    output = model.forward(X)

    assert output.shape == (5, 3), f"Expected output shape (5, 3), got {output.shape}"
    assert not np.isnan(output.data).any(), "Output should not contain NaN values"

    print("✅ SimpleMLP works correctly!")

if __name__ == "__main__":
    test_unit_simple_mlp()

# %% [markdown]
"""
## 🏗️ Benchmark Report Class

The BenchmarkReport class encapsulates all benchmark results and provides methods for comprehensive measurement and professional reporting.

**Design Philosophy:**
1. **Separation of concerns** - Measurement logic separate from model logic
2. **Comprehensive metrics** - Capture model characteristics AND performance
3. **System context** - Record environment for reproducibility
4. **Statistical rigor** - Multiple runs for latency, report mean + std
5. **JSON-serializable** - All data types compatible with JSON export
"""

# %% nbgrader={"grade": false, "grade_id": "benchmark-report", "solution": true}
#| export
class BenchmarkReport:
    """
    Benchmark report for model performance.

    Measures and stores:
    - Model characteristics (parameters, size)
    - Performance metrics (accuracy, latency, throughput)
    - System context (platform, versions)
    - Optimization info (techniques applied)

    Usage:
        report = BenchmarkReport(model_name="my_model")
        report.benchmark_model(model, X_test, y_test, num_runs=100)
        print(report.metrics)
    """
    def __init__(self, model_name="model"):
        self.model_name = model_name
        self.metrics = {}
        self.system_info = self._get_system_info()
        self.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    def _get_system_info(self):
        """Collect system information for reproducibility."""
        return {
            'platform': platform.platform(),
            'python_version': sys.version.split()[0],
            'numpy_version': np.__version__
        }

    def benchmark_model(self, model, X_test, y_test, num_runs=100):
        """
        Benchmark model performance comprehensively.

        Args:
            model: Model to benchmark (must have .forward() and .count_parameters())
            X_test: Test inputs (Tensor)
            y_test: Test labels (numpy array of class indices)
            num_runs: Number of inference runs for latency measurement (default: 100)

        Returns:
            Dictionary of metrics

        Measurements:
        1. Parameter count - Model capacity indicator
        2. Model size (MB) - Deployment cost (assumes FP32)
        3. Accuracy - Task performance (classification accuracy)
        4. Latency (mean ± std, and the median) - Inference speed and consistency,
           timed with Module 19's precise_timer after a few untimed warmup runs
        5. Throughput - Samples per second when the whole test batch goes through
           one forward call, timed separately from latency (see Foundations)
        """
        if len(X_test.shape) == 0 or X_test.shape[0] == 0:
            raise ValueError("X_test must contain at least one sample")
        y_test = np.asarray(y_test)
        if y_test.shape != (X_test.shape[0],):
            raise ValueError("y_test must contain one class index per sample")
        if num_runs <= 0:
            raise ValueError("num_runs must be positive")
        # Count parameters and stored size (see measure_memory)
        param_count = (
            model.count_parameters()
            if hasattr(model, "count_parameters")
            else sum(p.data.size for p in model.parameters())
        )
        model_size_mb = self.measure_memory(model)

        # Validate classification outputs before argmax: NaN scores can otherwise
        # silently select class zero and make a diverged model look accurate.
        predictions = model.forward(X_test)
        scores = np.asarray(predictions.data)
        if scores.ndim != 2 or scores.shape[0] != X_test.shape[0] or scores.shape[1] == 0:
            raise ValueError("Predictions must have shape (samples, classes) with at least one class")
        if not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores) or not np.all(np.isfinite(scores)):
            raise ValueError("Prediction scores must be finite real numbers")
        if not np.issubdtype(y_test.dtype, np.integer):
            raise ValueError("y_test must contain integer class indices")
        if np.any(y_test < 0) or np.any(y_test >= scores.shape[1]):
            raise ValueError("y_test class indices must be within the prediction classes")
        pred_labels = np.argmax(scores, axis=1)
        accuracy = np.mean(pred_labels == y_test)

        # Latency: untimed warmup, then num_runs single-sample calls timed one by one
        # Why multiple runs? See "Variance" section in Foundations
        latencies = self.measure_latency(model, X_test, num_runs)

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        median_latency = np.median(latencies)

        # Throughput: time the WHOLE batch through one forward call, num_runs times,
        # after its own untimed warmup. This is a separate measurement, not
        # 1000 / latency: batching raises samples/second without making any one
        # sample faster (Foundations, "Latency vs. Throughput")
        batch_size = X_test.shape[0]
        for _ in range(min(5, num_runs)):
            _ = model.forward(X_test)
        batch_seconds = []
        for _ in range(num_runs):
            with precise_timer() as timer:
                _ = model.forward(X_test)
            batch_seconds.append(timer.elapsed)
        # Median batch time, floored so a batch that measures 0.0 s cannot divide by zero
        throughput = batch_size / max(np.median(batch_seconds), 1e-9)

        # Store metrics (all as Python native types for JSON serialization)
        self.metrics = {
            'parameter_count': int(param_count),
            'model_size_mb': float(model_size_mb),
            'accuracy': float(accuracy),
            'latency_ms_mean': float(avg_latency),
            'latency_ms_std': float(std_latency),
            'latency_ms_median': float(median_latency),
            'throughput_samples_per_sec': float(throughput)
        }

        print(f"\n📊 Benchmark Results for {self.model_name}:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Size: {model_size_mb:.2f} MB")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Latency: {avg_latency:.2f}ms ± {std_latency:.2f}ms (median {median_latency:.2f}ms)")
        print(f"  Throughput: {throughput:,.0f} samples/sec (batch of {batch_size})")

        return self.metrics

    def measure_latency(self, model, X_batch, num_runs=100):
        """
        Measure single-sample inference latency over multiple runs.

        Args:
            model: Model with a .forward() method
            X_batch: Test inputs (Tensor); only the first sample, X_batch[:1], is timed
            num_runs: Number of timed runs (default: 100)

        Returns:
            List of per-run latencies in milliseconds

        TODO: Time single-sample inference over multiple runs

        APPROACH:
        1. Run a few untimed warmup calls first (Module 19)
        2. Run inference num_runs times
        3. Time each run with Module 19's precise_timer() context manager
        4. Convert seconds to milliseconds
        5. Return the list of latencies

        HINTS:
        - `with precise_timer() as timer:` around model.forward(); timer.elapsed
          holds the seconds once the block exits
        - Multiply by 1000 to convert seconds to milliseconds
        - Use X_batch[:1] so each call sees exactly one sample
        """
        ### BEGIN SOLUTION role="scaffold"
        if num_runs <= 0 or X_batch.shape[0] == 0:
            raise ValueError("Latency measurement needs samples and positive num_runs")
        for _ in range(min(5, num_runs)):
            _ = model.forward(X_batch[:1])
        latencies = []
        for _ in range(num_runs):
            with precise_timer() as timer:
                _ = model.forward(X_batch[:1])
            latencies.append(timer.elapsed * 1000)
        return latencies
        ### END SOLUTION

    def measure_memory(self, model):
        """
        Measure model memory footprint.

        TODO: Calculate model size in MB

        APPROACH:
        1. If the model reports its array storage via size_bytes(), use it
        2. Otherwise sum parameter array nbytes (zeros still occupy storage)
        3. Convert to MB (divide by 1024*1024)

        HINTS:
        - model.count_parameters() gives total param count
        - FP32 = 4 bytes per parameter
        - 1 MB = 1024 * 1024 bytes
        """
        ### BEGIN SOLUTION role="scaffold"
        if hasattr(model, 'size_bytes'):
            return model.size_bytes() / (1024 * 1024)
        return sum(param.data.nbytes for param in model.parameters()) / (1024 * 1024)
        ### END SOLUTION

# %% [markdown]
r"""
### Why These Metrics?

Each metric answers a specific production engineering and deployment question:

| Production Inquiry | Monitored Metric | Physical Dimension | Systems Decision SLA |
| :--- | :--- | :--- | :--- |
| **"Will it fit on the target device?"** | `model_size_mb` | Parameter storage & DRAM footprint | SRAM bounds in microcontroller or embedded cache |
| **"Is prediction quality preserved?"** | `accuracy` | Top-1 accuracy score ($[0, 1]$) | Task accuracy threshold ($\ge \tau_{\text{acc}}$) |
| **"Is interactive inference fast enough?"**| `latency_ms_mean` | Forward pass duration ($\text{ms}$) | Frame rate budget ($< 16\text{ ms}$ or $< 100\text{ ms}$) |
| **"Is execution latency predictable?"** | `latency_ms_std` | Latency dispersion / jitter | Elimination of tail stalls ($p_{99}$) |
| **"Can the service scale under load?"** | `throughput` | Processed samples per second | Cloud server concurrency headroom |
| **"How compute-heavy is the graph?"** | `parameter_count` | Total active scalar weights | Memory bus transfer volume |

### Design Choice: Warmup, Then $N=100$ Iterations

`measure_latency` executes untimed warmup iterations first, followed by 100 timed trials:
- **Warmup is untimed**, ensuring first-call costs (physical frame allocations, instruction cache warming, and JIT compilation) stay out of the numbers
- **100 timed runs** average out OS scheduler interrupts and garbage collection cycles
- **The median is reported alongside mean $\pm$ std** because asymmetric tail outliers drag the mean; the submission compares medians for robust qualification
- **Standard deviation ($s$) is a distribution spread, not a confidence interval**: it quantifies runtime variance, while standard error ($\text{SE} = \frac{s}{\sqrt{n}}$) bounds precision

| Benchmark Strategy | Sample Size | Reported Metrics | Systems Reliability & Risk |
| :--- | :--- | :--- | :--- |
| **Single Run (Unreliable)** | $N=1$ | Single scalar (e.g. $12.3\text{ ms}$) | Unknown variance; highly vulnerable to cold starts or GC spikes |
| **Sampled Distribution (Reliable)**| $N=100$ | $\mu \pm s$ (e.g. $10.0\text{ ms} \pm 0.5\text{ ms}$), $\text{median} = 10.1\text{ ms}$ | Statistically grounded; captures true tail behavior and ensures reproducibility |

### Design Choice: Python Native Types

All metrics are explicitly cast to native Python primitives (`int`, `float`) before report assembly:

```python
'parameter_count': int(param_count),  # NumPy int64 -> Python int
'accuracy': float(accuracy),          # NumPy float64 -> Python float
```

**Why?** Standard JSON libraries cannot serialize NumPy scalar types directly:
```python
# Fails with TypeError: Object of type int64 is not JSON serializable
json.dumps({"value": np.int64(42)})

# Succeeds seamlessly across all JSON parsers
json.dumps({"value": int(42)})
```

This design decision ensures generated submissions are universally machine-readable across any programming language without proprietary serializer dependencies.
"""

# %% [markdown]
"""
### 🧪 Unit Test: BenchmarkReport

This test validates the BenchmarkReport class captures all required metrics.

**What we're testing**: Report initialization, metric collection, and value ranges
**Why it matters**: Benchmarks must be comprehensive and accurate for reproducibility
**Expected**: All required metrics present with valid types and ranges
"""

# %% nbgrader={"grade": true, "grade_id": "test-benchmark-report", "locked": true, "points": 15}
def test_unit_benchmark_report():
    """🧪 Test BenchmarkReport class functionality."""
    print("🧪 Unit Test: BenchmarkReport...")

    # Create report
    report = BenchmarkReport(model_name="test_model")

    # Check initialization
    assert report.model_name == "test_model", "Model name should be set correctly"
    assert report.timestamp is not None, "Timestamp should be set"
    assert report.system_info is not None, "System info should be collected"
    assert 'platform' in report.system_info, "Should have platform info"
    assert 'python_version' in report.system_info, "Should have Python version"

    # Create test data
    rng = np.random.default_rng(7)
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(rng.standard_normal((50, 10)))
    y_test = rng.integers(0, 3, 50)

    # Benchmark model
    metrics = report.benchmark_model(model, X_test, y_test, num_runs=10)

    # Check metrics exist
    required_metrics = [
        'parameter_count', 'model_size_mb', 'accuracy',
        'latency_ms_mean', 'latency_ms_std', 'throughput_samples_per_sec'
    ]
    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"

    # Check metric types and ranges
    assert isinstance(metrics['parameter_count'], int), "Parameter count should be int"
    assert metrics['parameter_count'] > 0, "Should have positive parameter count"
    assert metrics['model_size_mb'] > 0, "Model size should be positive"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"
    assert metrics['latency_ms_mean'] > 0, "Latency should be positive"
    assert metrics['latency_ms_std'] >= 0, "Standard deviation should be non-negative"
    assert metrics['throughput_samples_per_sec'] > 0, "Throughput should be positive"

    print("✅ BenchmarkReport works correctly!")

if __name__ == "__main__":
    test_unit_benchmark_report()

# %% [markdown]
"""
### OlympicEvent: Applying the Capstone's Rules

Module 19 measures models under a shared protocol. This capstone decides whether
those measurements meet a classroom event's requirements. Keep that decision
separate from schema validation: a valid report can describe a model that does
not qualify. These thresholds are classroom rules, not official MLPerf criteria.

`OlympicEvent` gives each event one stable name. `qualifies_event` reads the
accuracy, median single-sample latency, and model array storage already collected
by `BenchmarkReport`. The legacy key `model_size_mb` stores MiB (bytes / 2**20).
Latency and memory events require at least 85% accuracy; the accuracy event
requires latency below 100 ms and storage below 10 MiB. Extreme push lowers the
accuracy floor to 80%. All-around has no eligibility floor: it keeps the separate
metrics for discussion and does not invent a combined ranking.

The supplied policy is short so every student applies the same rules. It never
changes the measurements, and it cannot verify the experiment that produced them.
"""

# %% nbgrader={"grade": false, "grade_id": "olympic-event", "solution": false}
#| export
class OlympicEvent(Enum):
    """Stable names for the five classroom capstone events."""
    LATENCY_SPRINT = "latency_sprint"
    MEMORY_CHALLENGE = "memory_challenge"
    ACCURACY_CONTEST = "accuracy_contest"
    ALL_AROUND = "all_around"
    EXTREME_PUSH = "extreme_push"


def qualifies_event(metrics: Dict[str, float], event: OlympicEvent) -> bool:
    """Check classroom eligibility without modifying or combining measurements.

    Requires accuracy in [0, 1], positive median latency in milliseconds, and
    positive array storage in MiB. Unknown events and invalid values raise;
    missing measurements raise KeyError rather than receiving default values.
    A False result means a valid measurement failed the selected event's rule.
    """
    event = OlympicEvent(event)
    accuracy = metrics['accuracy']
    latency = metrics['latency_ms_median']
    size = metrics['model_size_mb']
    if not all(np.isfinite(value) for value in (accuracy, latency, size)):
        raise ValueError("Event measurements must be finite")
    if not 0 <= accuracy <= 1 or latency <= 0 or size <= 0:
        raise ValueError("Event measurements require valid accuracy and positive latency/storage")

    if event in (OlympicEvent.LATENCY_SPRINT, OlympicEvent.MEMORY_CHALLENGE):
        return bool(accuracy >= 0.85)
    if event == OlympicEvent.ACCURACY_CONTEST:
        return bool(latency < 100.0 and size < 10.0)
    if event == OlympicEvent.EXTREME_PUSH:
        return bool(accuracy >= 0.80)
    return True  # All-around compares the separate metrics, without a floor.

# %% [markdown]
"""
## 🏗️ Submission Generation

The core function that generates a standardized JSON submission from benchmark results.

**Design Goals:**
1. **Baseline-first** - Always require baseline results (comparison reference)
2. **Optimization optional** - Support baseline-only OR baseline+optimized submissions
3. **Auto-calculate improvements** - Automatically compute speedup, compression, accuracy delta
4. **Schema compliance** - Generate structure that passes validation
5. **Extensible** - Easy to add new fields without breaking existing code
"""

# %% nbgrader={"grade": false, "grade_id": "generate-submission", "solution": false}
#| export
def generate_submission(
    baseline_report: BenchmarkReport,
    optimized_report: Optional[BenchmarkReport] = None,
    student_name: Optional[str] = None,
    techniques_applied: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a standardized benchmark submission.

    Args:
        baseline_report: Benchmark results for baseline model (REQUIRED)
        optimized_report: Optional benchmark results for optimized model
        student_name: Optional student/submitter name
        techniques_applied: List of optimization techniques used (e.g., ["quantization", "pruning"])

    Returns:
        Dictionary containing submission data (ready for JSON export)

    Submission Structure:
        {
          "tinytorch_version": "0.1.0",
          "submission_type": "capstone_benchmark",
          "timestamp": "...",
          "system_info": {...},
          "baseline": {
            "model_name": "...",
            "metrics": {...}
          },
          "optimized": {...},        # Optional
          "improvements": {...}      # Auto-calculated if optimized present
        }
    """
    submission = {
        'tinytorch_version': '0.1.0',
        'submission_type': 'capstone_benchmark',
        'timestamp': baseline_report.timestamp,
        'system_info': baseline_report.system_info,
        'baseline': {
            'model_name': baseline_report.model_name,
            'metrics': baseline_report.metrics
        }
    }

    # Add student name if provided
    if student_name:
        submission['student_name'] = student_name

    # Add optimization results if provided
    if optimized_report:
        submission['optimized'] = {
            'model_name': optimized_report.model_name,
            'metrics': optimized_report.metrics,
            'techniques_applied': techniques_applied or []
        }

        # Calculate improvement metrics
        # Compare medians when both reports carry one (Module 19: the median is
        # the honest center of a skewed latency distribution); a report that
        # recorded only a mean still compares
        key = 'latency_ms_median' if all('latency_ms_median' in r.metrics for r in (baseline_report, optimized_report)) else 'latency_ms_mean'
        baseline_latency = baseline_report.metrics[key]
        optimized_latency = optimized_report.metrics[key]
        baseline_size = baseline_report.metrics['model_size_mb']
        optimized_size = optimized_report.metrics['model_size_mb']

        submission['improvements'] = {
            # Floor both denominators: an empty model measures 0.0 ms, and a model
            # pruned to sparsity 1.0 stores 0 bytes
            'speedup': float(baseline_latency / max(optimized_latency, 1e-6)),
            'compression_ratio': float(baseline_size / max(optimized_size, 1e-9)),
            'accuracy_delta': float(
                optimized_report.metrics['accuracy'] - baseline_report.metrics['accuracy']
            )
        }

    return submission

def save_submission(submission: Dict[str, Any], filepath: str = "submission.json"):
    """
    Save submission to JSON file.

    Args:
        submission: Submission dictionary from generate_submission()
        filepath: Output path (default: "submission.json")

    Returns:
        Path to saved file
    """
    Path(filepath).write_text(json.dumps(submission, indent=2))
    print(f"\n✅ Submission saved to: {filepath}")
    return filepath

# %% [markdown]
r"""
### Understanding the Improvements Calculation

When you provide both baseline and optimized reports, the submission system auto-calculates three foundational improvement metrics:

$$\text{Speedup } (S) = \frac{\text{Latency}_{\text{base}}}{\text{Latency}_{\text{opt}}}$$

$$\text{Compression Ratio } (C) = \frac{\text{ModelSize}_{\text{base}}}{\text{ModelSize}_{\text{opt}}}$$

$$\text{Accuracy Delta } (\Delta \text{Acc}) = \text{Accuracy}_{\text{opt}} - \text{Accuracy}_{\text{base}}$$

| Quantitative Improvement Metric | Mathematical Definition | Numerical Example | Systems Engineering Interpretation |
| :--- | :--- | :--- | :--- |
| **Speedup ($S$)** | $\frac{T_{\text{base}}}{T_{\text{opt}}}$ | $\frac{10.0\text{ ms}}{5.0\text{ ms}} = 2.0\times$ | $2.0\times = \text{Optimized model is twice as fast}$; $< 1.0\times = \text{Regression}$ |
| **Compression Ratio ($C$)** | $\frac{M_{\text{base}}}{M_{\text{opt}}}$ | $\frac{4.0\text{ MB}}{1.0\text{ MB}} = 4.0\times$ | $4.0\times = \text{Model footprint reduced to } 25\%$; $< 1.0\times = \text{Bloat}$ |
| **Accuracy Delta ($\Delta \text{Acc}$)** | $\text{Acc}_{\text{opt}} - \text{Acc}_{\text{base}}$ | $91.5\% - 92.0\% = -0.5\%$ | $\ge 0.0\% = \text{Lossless quality}$; $-0.5\% = \text{Acceptable degradation}$ |

### The Optimization Trade-off Space: Pareto Frontiers

Every physical optimization technique trades off across the three vertices of systems engineering: speed, storage, and predictive accuracy:

<img src="capstone_pareto_frontier.svg" alt="Capstone Optimization Pareto Frontier" width="100%">

| Optimization Strategy | Prioritized Dimensions | Incurred Cost | Typical Target Deployment |
| :--- | :--- | :--- | :--- |
| **Aggressive Quantization** | Speedup ($2.8\times$) + Compression ($4\times$) | Small accuracy delta ($-0.5\text{ to } -1.0\%$) | Edge mobile, embedded microcontrollers |
| **Structured Pruning** | Compression ($2\times$) + Memory bandwidth | Small accuracy delta, indexing overhead | Cache-constrained serverless runtimes |
| **Operator Fusion + KV Caching**| Latency speedup ($4.8\times$) | Transient SRAM working memory | Interactive LLM generation & real-time audio |
| **Balanced Optimization** | Pareto frontier sweet spot | Co-designed constraint envelope | General production microservices |

### Why JSON Schema Validation Matters

Our submission format is designed to be validated:

```python
# Valid submission (passes validation):
{
  "tinytorch_version": "0.1.0",      # ✓ Required, string
  "timestamp": "2025-01-15 14:30",   # ✓ Required, string
  "baseline": {                       # ✓ Required, object
    "metrics": {                      # ✓ Required, object
      "accuracy": 0.92                # ✓ Required, float in [0, 1]
    }
  }
}

# Invalid submission (fails validation):
{
  "tinytorch_version": 0.1,          # ✗ Wrong type (number not string)
  # ✗ Missing timestamp
  "baseline": {
    "metrics": {
      "accuracy": "92%"                # ✗ Wrong type (string not float)
    }
  }
}
```

This prevents common mistakes:
- Forgetting required fields
- Using wrong data types
- Invalid value ranges (accuracy > 1.0)
- Inconsistent structure

In production ML, schema validation is what makes benchmarks trustworthy and comparable!
"""

# %% [markdown]
"""
### 🧪 Unit Test: Submission Generation

This test validates the submission generation creates proper JSON structure.

**What we're testing**: Submission structure, required fields, and optional fields
**Why it matters**: Submissions must be schema-compliant for community sharing
**Expected**: Valid JSON structure with all required fields
"""

# %% nbgrader={"grade": true, "grade_id": "test-submission-generation", "locked": true, "points": 15}
def test_unit_submission_generation():
    """🧪 Test generate_submission() function."""
    print("🧪 Unit Test: Submission Generation...")

    # Create baseline report
    rng = np.random.default_rng(7)
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(rng.standard_normal((50, 10)))
    y_test = rng.integers(0, 3, 50)

    baseline_report = BenchmarkReport(model_name="baseline_model")
    baseline_report.benchmark_model(model, X_test, y_test, num_runs=10)

    # Generate submission with baseline only
    submission = generate_submission(baseline_report)

    # Check submission structure
    assert isinstance(submission, dict), "Submission should be a dictionary"
    assert 'tinytorch_version' in submission, "Should have version field"
    assert 'submission_type' in submission, "Should have submission type"
    assert 'timestamp' in submission, "Should have timestamp"
    assert 'system_info' in submission, "Should have system info"
    assert 'baseline' in submission, "Should have baseline results"

    # Check baseline structure
    baseline = submission['baseline']
    assert 'model_name' in baseline, "Baseline should have model name"
    assert 'metrics' in baseline, "Baseline should have metrics"
    assert baseline['model_name'] == "baseline_model", "Model name should match"

    # Test with student name
    submission_with_name = generate_submission(baseline_report, student_name="Test Student")
    assert 'student_name' in submission_with_name, "Should include student name when provided"
    assert submission_with_name['student_name'] == "Test Student", "Student name should match"

    print("✅ Submission generation works correctly!")

if __name__ == "__main__":
    test_unit_submission_generation()

# %% [markdown]
"""
### The Submission Schema

Before a submission is worth comparing against anyone else's, it has to be
readable by the tooling that aggregates it. The validator below is the contract:
it names the required fields and the ranges their values must fall in. It ships
with the package so the graders and your own scripts can apply the same rules.
"""

# %% nbgrader={"grade": false, "grade_id": "validate-submission-schema", "solution": false}
#| export
def validate_submission_schema(submission: Dict[str, Any]) -> bool:
    """
    Validate submission JSON conforms to required schema.

    This function ensures submissions are:
    - Complete (no missing required fields)
    - Type-safe (correct data types)
    - Valid (values in acceptable ranges)

    Used for automated validation before accepting community submissions.
    """
    # Check required top-level fields
    required_fields = ['tinytorch_version', 'submission_type', 'timestamp', 'system_info', 'baseline']
    for field in required_fields:
        if field not in submission:
            raise AssertionError(f"Missing required field: {field}")

    # Check field types
    assert isinstance(submission['tinytorch_version'], str), "Version should be string"
    assert isinstance(submission['submission_type'], str), "Submission type should be string"
    assert isinstance(submission['timestamp'], str), "Timestamp should be string"
    assert isinstance(submission['system_info'], dict), "System info should be dict"
    assert isinstance(submission['baseline'], dict), "Baseline should be dict"

    # Apply the same contract to every reported model, including optimizations.
    required_metrics = ['parameter_count', 'model_size_mb', 'accuracy', 'latency_ms_mean']
    for section in ('baseline', 'optimized'):
        if section not in submission:
            continue
        report = submission[section]
        assert isinstance(report, dict), f"{section} should be a dict"
        assert isinstance(report.get('model_name'), str), f"{section} missing model_name"
        assert isinstance(report.get('metrics'), dict), f"{section} missing metrics"
        metrics = report['metrics']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric in {section}: {metric}"
            value = metrics[metric]
            assert type(value) in (int, float) and np.isfinite(value), f"{section}.{metric} must be finite numeric data"
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy must be in [0, 1]"
        assert type(metrics['parameter_count']) is int and metrics['parameter_count'] > 0, "Parameter count must be a positive integer"
        assert metrics['model_size_mb'] > 0, "Model size must be positive"
        assert metrics['latency_ms_mean'] > 0, "Latency must be positive"
        # Older reports may omit these fields; present values must still obey
        # the measurement contract because medians drive ranking and speedup.
        for metric in ('latency_ms_median', 'latency_ms_std', 'throughput_samples_per_sec'):
            if metric in metrics:
                value = metrics[metric]
                assert type(value) in (int, float) and np.isfinite(value), f"{section}.{metric} must be finite numeric data"
                if metric == 'latency_ms_std':
                    assert value >= 0, "Latency standard deviation must be nonnegative"
                else:
                    assert value > 0, f"{section}.{metric} must be positive"

    # Improvements are optional, but any supplied comparison must agree with
    # the two reports. Use the same median/mean fallback as generate_submission.
    if 'improvements' in submission:
        assert 'optimized' in submission, "Improvements require an optimized report"
        improvements = submission['improvements']
        assert isinstance(improvements, dict), "Improvements should be a dict"
        baseline = submission['baseline']['metrics']
        optimized = submission['optimized']['metrics']
        latency_key = 'latency_ms_median' if all('latency_ms_median' in m for m in (baseline, optimized)) else 'latency_ms_mean'
        expected = {
            'speedup': baseline[latency_key] / max(optimized[latency_key], 1e-6),
            'compression_ratio': baseline['model_size_mb'] / max(optimized['model_size_mb'], 1e-9),
            'accuracy_delta': optimized['accuracy'] - baseline['accuracy'],
        }
        for metric, expected_value in expected.items():
            value = improvements.get(metric)
            assert type(value) in (int, float) and np.isfinite(value), f"improvements.{metric} must be finite numeric data"
            assert np.isclose(value, expected_value, rtol=1e-6, atol=1e-12), f"improvements.{metric} disagrees with reported measurements"

    # Check system info
    system_info = submission['system_info']
    assert 'platform' in system_info, "System info missing platform"
    assert 'python_version' in system_info, "System info missing python_version"

    return True


# %% [markdown]
"""
### 🧪 Unit Test: Schema Validation

This test validates submissions conform to the required schema.

**What we're testing**: Required fields, type safety, value constraints
**Why it matters**: Schema validation enables automated aggregation and comparison
**Expected**: Valid submissions pass, invalid submissions fail with clear errors
"""

# %% nbgrader={"grade": true, "grade_id": "test-submission-schema", "locked": true, "points": 10}
def test_unit_submission_schema():
    """🧪 Test submission schema validation."""
    print("🧪 Unit Test: Submission Schema...")

    # Create valid submission
    rng = np.random.default_rng(7)
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(rng.standard_normal((50, 10)))
    y_test = rng.integers(0, 3, 50)

    report = BenchmarkReport(model_name="test_model")
    report.benchmark_model(model, X_test, y_test, num_runs=10)

    submission = generate_submission(report)

    # Validate schema
    assert validate_submission_schema(submission), "Submission should pass schema validation"

    # Test with optimized results
    optimized_model = SimpleMLP(input_size=10, hidden_size=15, output_size=3)
    optimized_report = BenchmarkReport(model_name="optimized_model")
    optimized_report.benchmark_model(optimized_model, X_test, y_test, num_runs=10)

    submission_with_opt = generate_submission(
        report,
        optimized_report,
        techniques_applied=["pruning"]
    )

    # Validate optimized submission
    assert validate_submission_schema(submission_with_opt), "Optimized submission should pass validation"
    assert 'optimized' in submission_with_opt, "Should have optimized section"
    assert 'improvements' in submission_with_opt, "Should have improvements section"

    print("✅ Submission schema validation works correctly!")

if __name__ == "__main__":
    test_unit_submission_schema()

# %% [markdown]
"""
### 🧪 Unit Test: Submission with Optimization

This test validates submissions with both baseline and optimized results.

**What we're testing**: Optimized section, techniques list, improvements calculation
**Why it matters**: Comparing baseline vs optimized is the core value of benchmarking
**Expected**: Proper improvements calculation with speedup, compression, accuracy delta
"""

# %% nbgrader={"grade": true, "grade_id": "test-submission-with-optimization", "locked": true, "points": 10}
def test_unit_submission_with_optimization():
    """🧪 Test submission with baseline + optimized comparison."""
    print("🧪 Unit Test: Submission with Optimization...")

    # Create baseline
    rng = np.random.default_rng(7)
    baseline_model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(rng.standard_normal((50, 10)))
    y_test = rng.integers(0, 3, 50)

    baseline_report = BenchmarkReport(model_name="baseline")
    baseline_report.benchmark_model(baseline_model, X_test, y_test, num_runs=10)

    # Create optimized version (smaller model for demo)
    optimized_model = SimpleMLP(input_size=10, hidden_size=15, output_size=3)
    optimized_report = BenchmarkReport(model_name="optimized")
    optimized_report.benchmark_model(optimized_model, X_test, y_test, num_runs=10)

    # Generate submission with both
    techniques = ["model_sizing", "pruning"]
    submission = generate_submission(
        baseline_report,
        optimized_report,
        student_name="Test Student",
        techniques_applied=techniques
    )

    # Check optimized section exists
    assert 'optimized' in submission, "Should have optimized section"
    optimized = submission['optimized']
    assert 'model_name' in optimized, "Optimized section should have model name"
    assert 'metrics' in optimized, "Optimized section should have metrics"
    assert 'techniques_applied' in optimized, "Should have techniques list"
    assert optimized['techniques_applied'] == techniques, "Techniques should match"

    # Check improvements section
    assert 'improvements' in submission, "Should have improvements section"
    improvements = submission['improvements']
    assert 'speedup' in improvements, "Should have speedup metric"
    assert 'compression_ratio' in improvements, "Should have compression ratio"
    assert 'accuracy_delta' in improvements, "Should have accuracy delta"

    # Check improvement values are reasonable
    assert improvements['speedup'] > 0, "Speedup should be positive"
    assert improvements['compression_ratio'] > 0, "Compression ratio should be positive"
    assert -1 <= improvements['accuracy_delta'] <= 1, "Accuracy delta should be in [-1, 1]"

    print("✅ Submission with optimization works correctly!")

if __name__ == "__main__":
    test_unit_submission_with_optimization()

# %% [markdown]
"""
### 🧪 Unit Test: Improvements Calculation

This test validates the mathematical correctness of improvement metrics.

**What we're testing**: Speedup, compression ratio, accuracy delta formulas
**Why it matters**: Incorrect calculations would invalidate all comparisons
**Expected**: Exact match with manual calculations
"""

# %% nbgrader={"grade": true, "grade_id": "test-improvements-calculation", "locked": true, "points": 10}
def test_unit_improvements_calculation():
    """🧪 Test speedup/compression/accuracy calculations are correct."""
    print("🧪 Unit Test: Improvements Calculation...")

    # Create baseline with known metrics
    baseline_report = BenchmarkReport(model_name="baseline")
    baseline_report.metrics = {
        'parameter_count': 1000,
        'model_size_mb': 4.0,
        'accuracy': 0.80,
        'latency_ms_mean': 10.0,
        'latency_ms_std': 1.0,
        'throughput_samples_per_sec': 100.0
    }
    baseline_report.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    baseline_report.system_info = {'platform': 'test', 'python_version': '3.9', 'numpy_version': '1.20'}

    # Create optimized with 2x speedup, 2x compression, 5% accuracy loss
    optimized_report = BenchmarkReport(model_name="optimized")
    optimized_report.metrics = {
        'parameter_count': 500,
        'model_size_mb': 2.0,
        'accuracy': 0.75,
        'latency_ms_mean': 5.0,
        'latency_ms_std': 0.5,
        'throughput_samples_per_sec': 200.0
    }
    optimized_report.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    optimized_report.system_info = baseline_report.system_info

    # Generate submission
    submission = generate_submission(baseline_report, optimized_report)

    improvements = submission['improvements']

    # Verify calculations
    # Speedup = baseline_latency / optimized_latency = 10.0 / 5.0 = 2.0
    assert abs(improvements['speedup'] - 2.0) < 0.01, f"Expected speedup 2.0, got {improvements['speedup']}"

    # Compression = baseline_size / optimized_size = 4.0 / 2.0 = 2.0
    assert abs(improvements['compression_ratio'] - 2.0) < 0.01, f"Expected compression 2.0, got {improvements['compression_ratio']}"

    # Accuracy delta = 0.75 - 0.80 = -0.05
    assert abs(improvements['accuracy_delta'] - (-0.05)) < 0.001, f"Expected accuracy delta -0.05, got {improvements['accuracy_delta']}"

    print("✅ Improvements calculation is correct!")

if __name__ == "__main__":
    test_unit_improvements_calculation()

# %% [markdown]
"""
### 🧪 Unit Test: JSON Serialization

This test validates save_submission() creates valid, round-trip compatible JSON.

**What we're testing**: File creation, JSON validity, round-trip preservation
**Why it matters**: Submissions must be loadable and shareable
**Expected**: Valid JSON that loads with identical structure
"""

# %% nbgrader={"grade": true, "grade_id": "test-json-serialization", "locked": true, "points": 10}
def test_unit_json_serialization():
    """🧪 Test save_submission() creates valid JSON files."""
    print("🧪 Unit Test: JSON Serialization...")

    # Create submission
    rng = np.random.default_rng(7)
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(rng.standard_normal((50, 10)))
    y_test = rng.integers(0, 3, 50)

    report = BenchmarkReport(model_name="test_model")
    report.benchmark_model(model, X_test, y_test, num_runs=10)

    submission = generate_submission(report, student_name="Test Student")

    # Save to file
    test_file = "/tmp/test_submission_unit.json"
    filepath = save_submission(submission, test_file)

    # Check file exists
    assert Path(filepath).exists(), "Submission file should exist"

    # Load and verify JSON is valid
    loaded_json = json.loads(Path(test_file).read_text())

    # Verify structure is preserved
    assert loaded_json['tinytorch_version'] == submission['tinytorch_version'], "Version should match"
    assert loaded_json['student_name'] == submission['student_name'], "Student name should match"
    assert loaded_json['baseline']['model_name'] == submission['baseline']['model_name'], "Model name should match"

    # Verify metrics are preserved
    baseline_metrics = loaded_json['baseline']['metrics']
    original_metrics = submission['baseline']['metrics']
    assert baseline_metrics['accuracy'] == original_metrics['accuracy'], "Accuracy should match"
    assert baseline_metrics['parameter_count'] == original_metrics['parameter_count'], "Parameter count should match"

    # Verify JSON can be dumped again (round-trip test)
    round_trip = json.dumps(loaded_json, indent=2)
    assert len(round_trip) > 0, "JSON should serialize again"

    # Clean up
    Path(test_file).unlink()

    print("✅ JSON serialization works correctly!")

if __name__ == "__main__":
    test_unit_json_serialization()

# %% [markdown]
"""
## 🔧 Integration: From Model to Submission

This section demonstrates the complete workflow from model to submission.
Students can modify this to benchmark their own models!

**Workflow Steps:**
1. Create test dataset (or load from milestone)
2. Create baseline model
3. Benchmark baseline performance
4. (Optional) Apply optimizations
5. (Optional) Benchmark optimized version
6. Generate submission with comparisons
7. Save to JSON file

This is the EXACT workflow used in production ML systems!
"""

# %% nbgrader={"grade": false, "grade_id": "example-workflow", "solution": false}
def run_example_benchmark():
    """
    Complete example showing the full benchmarking workflow.

    Students can modify this to benchmark their own models!
    """
    print("="*70)
    print("TINYTORCH CAPSTONE: BENCHMARKING WORKFLOW EXAMPLE")
    print("="*70)

    # Step 1: Create toy dataset
    print("\n🔧 Step 1: Creating toy dataset...")
    X_test = Tensor(rng.standard_normal((100, 10)))
    y_test = rng.integers(0, 3, 100)
    print(f"  Dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features, 3 classes")

    # Step 2: Create baseline model
    print("\n🔧 Step 2: Creating baseline model...")
    baseline_model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    print(f"  Model: {baseline_model.count_parameters():,} parameters")

    # Step 3: Benchmark baseline
    print("\n📊 Step 3: Benchmarking baseline model...")
    baseline_report = BenchmarkReport(model_name="baseline_mlp")
    baseline_report.benchmark_model(baseline_model, X_test, y_test, num_runs=50)

    # Step 4: Generate submission
    print("\n📝 Step 4: Generating submission...")
    submission = generate_submission(
        baseline_report=baseline_report,
        student_name="TinyTorch Student"
    )

    # Step 5: Save submission
    print("\n💾 Step 5: Saving submission...")
    save_submission(submission, "capstone_submission.json")

    print("\n" + "="*70)
    print("🎉 WORKFLOW COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Try optimizing the model (quantization, pruning, etc.)")
    print("  2. Benchmark the optimized version")
    print("  3. Generate a new submission with both baseline and optimized results")
    print("  4. Share your submission.json with the TinyTorch community!")

    return submission


if __name__ == "__main__":
    run_example_benchmark()

# %% [markdown]
r"""
### Understanding the Workflow Pattern

This workflow follows industry best practices for iterative optimization and deployment qualification:

<img src="capstone_pipeline_overview.svg" alt="Capstone Optimization Pipeline Overview" width="100%">

### Production ML Systems Optimization Lifecycle

| Lifecycle Stage | Implementation Action | Output Artifact / Decision | Systems Verification |
| :--- | :--- | :--- | :--- |
| **1. Define Task & SLA** | Select dataset and evaluation metrics | Fixed test split, latency ceiling, minimum accuracy | Controlled experimental environment |
| **2. Baseline Measurement**| Profile unoptimized reference model | `BenchmarkReport` ($\mu_{\text{base}}, s_{\text{base}}, M_{\text{base}}$) | True unoptimized reference baseline |
| **3. Apply Candidate Transform**| Apply one optimization technique (e.g. INT8) | Transformed model graph (e.g. `QuantizedLinear`) | Isolate individual transformation impact |
| **4. Benchmark & Recheck** | Sample latency and evaluate accuracy | Candidate `BenchmarkReport` | Detect immediate accuracy regressions |
| **5. Multi-Objective Comparison**| Calculate Speedup ($S$), Compression ($C$), $\Delta \text{Acc}$ | Pareto frontier assignment | Keep if dominating, revert if regressing |
| **6. Submission & Audit** | Serialized JSON submission schema | `submission.json` with system environment metadata | Machine-readable reproducible artifact |

**Key Insight**: Professional ML systems engineers iterate methodically on Step 3 and Step 4, verifying each single modification before compounding optimizations.
"""

# %% [markdown]
"""
### Advanced Optimization Workflow

This section demonstrates using the complete optimization pipeline from Modules 14-19:
- Module 14 (Profiling): Measure baseline performance and identify bottlenecks
- Module 15 (Quantization): Reduce precision from FP32 to INT8
- Module 16 (Compression): Prune low-magnitude weights
- Module 17 (Acceleration): Use optimized kernels
- Module 18 (Memoization): Cache repeated computations
- Module 19 (Benchmarking): Professional measurement infrastructure

This is the COMPLETE story: Profile → Optimize → Benchmark → Submit

**What Students Learn:**
- How to import and use APIs from previous modules
- How to combine multiple optimizations (quantization + pruning)
- How to measure cumulative impact (memory savings from pruning and INT8 compound; latency does not, in NumPy)
- How to document techniques for reproducibility
"""

# %% nbgrader={"grade": false, "grade_id": "optimization-workflow", "solution": false}
def run_optimization_workflow_example():
    """
    Advanced example showing the complete optimization workflow.

    This demonstrates:
    1. Profiling baseline model (Module 14)
    2. Applying optimizations (Modules 15, 16)
    3. Benchmarking with best practices (Module 19)
    4. Generating submission with before/after comparison

    Students learn how to use TinyTorch as a complete framework!
    """
    print("="*70)
    print("TINYTORCH CAPSTONE: OPTIMIZATION WORKFLOW")
    print("="*70)
    print("\nThis workflow uses Modules 14, 15, 16, 19, and 20 together:")
    print("  📊 Module 14: Profiling (parameter count)")
    print("  ✂️  Module 16: Compression (magnitude pruning)")
    print("  🔢 Module 15: Quantization (INT8 weights)")
    print("  📈 Module 19: Benchmarking (warmup, repeated timing)")
    print("  📝 Module 20: Submission Generation")

    import copy
    from tinytorch.perf.profiling import Profiler
    from tinytorch.perf.quantization import QuantizedLinear
    from tinytorch.perf.compression import magnitude_prune

    # Step 1: Create dataset
    print("\n" + "="*70)
    print("STEP 1: Create Test Dataset")
    print("="*70)
    X_test = Tensor(rng.standard_normal((100, 10)))
    y_test = rng.integers(0, 3, 100)
    print(f"  Dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features, 3 classes")

    # Step 2: Profile and benchmark the baseline (Modules 14 and 19)
    print("\n" + "="*70)
    print("STEP 2: Baseline Model - Profile & Benchmark")
    print("="*70)
    baseline_model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    profiler = Profiler()
    print(f"  Model: {profiler.count_parameters(baseline_model):,} parameters (Module 14's count)")

    baseline_report = BenchmarkReport(model_name="baseline_mlp")
    baseline_report.benchmark_model(baseline_model, X_test, y_test, num_runs=50)

    # Step 3: Optimize with Module 16 (pruning) and Module 15 (INT8 weights)
    print("\n" + "="*70)
    print("STEP 3: Optimize - Prune, then Quantize")
    print("="*70)
    optimized_model = copy.deepcopy(baseline_model)
    magnitude_prune(optimized_model, sparsity=0.5)  # zero the smallest half of the weights
    nonzero_params = sum(int(np.count_nonzero(p.data)) for p in optimized_model.parameters())
    optimized_model.fc1 = QuantizedLinear(optimized_model.fc1)  # INT8 weights, FP32 arithmetic
    optimized_model.fc2 = QuantizedLinear(optimized_model.fc2)
    # A deployment stores one INT8 byte per surviving weight; BenchmarkReport.measure_memory uses this
    # QuantizedLinear retains both FP32 reference weights and FP32 tensors of
    # rounded values. Count those actual arrays; pruning does not pack zeros.
    def stored_bytes():
        return sum(p.data.nbytes
                   for layer in (optimized_model.fc1, optimized_model.fc2)
                   for p in layer.parameters() + layer.original_layer.parameters())
    optimized_model.size_bytes = stored_bytes
    print(f"  Kept {nonzero_params:,} of {baseline_model.count_parameters():,} nonzero parameters before simulated quantization")

    optimized_report = BenchmarkReport(model_name="optimized_mlp")
    optimized_report.benchmark_model(optimized_model, X_test, y_test, num_runs=50)

    # Step 4: Generate submission with before/after comparison
    print("\n" + "="*70)
    print("STEP 4: Generate Submission with Improvements")
    print("="*70)

    submission = generate_submission(
        baseline_report=baseline_report,
        optimized_report=optimized_report,
        student_name="TinyTorch Optimizer",
        techniques_applied=["magnitude_pruning_0.5", "int8_quantization"]
    )

    # Display improvement summary
    if 'improvements' in submission:
        improvements = submission['improvements']
        print("\n  📈 Optimization Results:")
        print(f"     Speedup: {improvements['speedup']:.2f}x")
        print(f"     Compression: {improvements['compression_ratio']:.2f}x")
        print(f"     Accuracy change: {improvements['accuracy_delta']*100:+.1f}%")

    # Step 5: Save submission
    print("\n" + "="*70)
    print("STEP 5: Save Submission")
    print("="*70)
    filepath = save_submission(submission, "optimization_submission.json")

    print("\n" + "="*70)
    print("🎉 OPTIMIZATION WORKFLOW COMPLETE!")
    print("="*70)
    print("\n📚 What students learned:")
    print("  ✅ How to import and use optimization APIs from Modules 14-19")
    print("  ✅ How to benchmark before and after optimization")
    print("  ✅ How to generate professional submissions with improvement metrics")
    print("  ✅ How TinyTorch modules work together as a complete framework")
    print("\n💡 Next steps:")
    print("  - Try other sparsities, calibration data, or leaving a sensitive layer in FP32")
    print("  - Benchmark milestone models (XOR, TinyDigits MLP/CNN, Transformer, etc.)")
    print("  - Share your optimized results with the community!")

    return submission


if __name__ == "__main__":
    run_optimization_workflow_example()

# %% [markdown]
r"""
#### Combining Multiple Optimizations: Stacking & Amdahl's Law

In production ML systems, optimizations are rarely deployed in isolation. Rather, systems engineers stack multiple complementary transformations (vectorization, operator fusion, INT8 quantization, weight pruning, KV-caching) for compounding returns:

<img src="stacking_waterfall_amdahl.svg" alt="Stacking Optimizations Waterfall & Amdahl's Law" width="100%">

### Compounding Optimization Stacking

| Optimization Pass | Transformation Mechanism | Memory Footprint | Latency | Compounding Benefit | Accuracy Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0. Unoptimized Baseline** | Naive Python / FP32 arrays | $4.0\text{ MB}$ ($1.0\times$) | $10.0\text{ ms}$ ($1.0\times$) | Reference baseline | $92.0\%$ (Baseline) |
| **1. INT8 Quantization** | Symmetrically scaled 8-bit integers | $1.0\text{ MB}$ ($4.0\times$) | $5.0\text{ ms}$ ($2.0\times$) | $4.0\times\text{ memory, } 2.0\times\text{ speed}$ | $91.8\%$ ($-0.2\%$) |
| **2. Magnitude Pruning** | 50% structured weight sparsity | $0.5\text{ MB}$ ($2.0\times$) | $3.5\text{ ms}$ ($1.4\times$) | $2.0\times\text{ memory, } 1.4\times\text{ speed}$ | $91.5\%$ ($-0.3\%$) |
| **Cumulative Optimized**| **Compounded Pipeline** | $\mathbf{0.5\text{ MB}}$ ($\mathbf{8.0\times}$) | $\mathbf{3.5\text{ ms}}$ ($\mathbf{2.9\times}$) | **$\mathbf{8.0\times}$ compression, $\mathbf{2.9\times}$ speedup** | $\mathbf{91.5\%}$ ($-0.5\%$) |

$$\text{Total Compression} = C_{\text{quant}} \times C_{\text{prune}} = 4.0\times \times 2.0\times = \mathbf{8.0\times\text{ reduction}}$$

The submission's `techniques_applied` list documents each applied pass for automated audit and community replication:

```json
"techniques_applied": ["int8_quantization", "magnitude_pruning_0.5"]
```

This ensures downstream deployment pipelines know the exact sequence of graph lowering transformations required to reproduce the numbers.
"""

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.
"""

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 20}
def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_simple_mlp()
    test_unit_benchmark_report()
    test_unit_submission_generation()
    test_unit_submission_schema()
    test_unit_submission_with_optimization()
    test_unit_improvements_calculation()
    test_unit_json_serialization()

    print("\nRunning integration scenarios...")

    # Test complete workflow
    print("🧪 Integration Test: Complete Workflow...")
    rng = np.random.default_rng(7)
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    X_test = Tensor(rng.standard_normal((50, 10)))
    y_test = rng.integers(0, 3, 50)

    report = BenchmarkReport(model_name="integration_test")
    report.benchmark_model(model, X_test, y_test, num_runs=10)

    submission = generate_submission(report, student_name="Integration Test")
    assert validate_submission_schema(submission), "Submission should pass validation"

    print("🧪 Integration Test: Capstone Eligibility...")
    metrics = submission['baseline']['metrics']
    assert qualifies_event(metrics, OlympicEvent.LATENCY_SPRINT) == (metrics['accuracy'] >= 0.85)
    # Known boundaries: schema validity does not imply event eligibility.
    candidate = dict(metrics, accuracy=0.85, latency_ms_median=99.0, model_size_mb=9.0)
    assert qualifies_event(candidate, OlympicEvent.LATENCY_SPRINT)
    assert qualifies_event(candidate, OlympicEvent.ACCURACY_CONTEST)
    assert not qualifies_event(dict(candidate, accuracy=0.84), OlympicEvent.LATENCY_SPRINT)
    assert not qualifies_event(dict(candidate, latency_ms_median=100.0), OlympicEvent.ACCURACY_CONTEST)

    print("✅ Complete workflow works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 20")

# %% [markdown]
r"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of benchmarking, reproducibility, and ML systems integration:

You've built an entire ML framework across 20 modules. This capstone asks you to step back and reflect on the complete systems journey—from tensors to production-ready benchmarking.

### Question 1: End-to-End System Integration

Modern ML systems aren't just individual components working in isolation—they're carefully orchestrated pipelines where each piece connects to form a cohesive whole.

**The Complete Pipeline You Built:**

| Tier | Modules | Systems Responsibility | Artifacts & Dependencies |
| :--- | :--- | :--- | :--- |
| **Tier 1: Foundations** | `01_tensor` $\rightarrow$ `08_training` | N-D strided memory, autodiff DAG engine, loss optimization | `Tensor`, computational graph, SGD/Adam, mini-batch loops |
| **Tier 2: Architectures** | `09_convolutions` $\rightarrow$ `13_transformers` | Spatial filtering, tokenization, multi-head causal attention | `Conv2d`, `BPETokenizer`, `MultiHeadAttention`, `TransformerLM` |
| **Tier 3: Optimization** | `14_profiling` $\rightarrow$ `18_memoization` | Bottleneck isolation, dynamic range mapping, state reuse | Call trees, INT8 symmetric affine quantization, KV-Cache |
| **Tier 4: Evaluation** | `19_benchmarking` $\rightarrow$ `20_capstone` | Statistical rigor, Pareto frontiers, standardized schemas | `BenchmarkReport`, Olympic submission schemas, leaderboards |

**Systems Integration Lessons:**

1. **Dependency Management** — Each module imports cleanly from previous modules, creating an acyclic dependency graph.
2. **API Consistency** — Tensor operations (`+`, `@`, `.reshape()`) behave identically whether executed in Module 01 or Module 20.
3. **Composability** — Complex architectures (Transformers) are composed cleanly from elementary primitives (linear projections, softmax, residuals).
4. **Progressive Enhancement** — Module 06 activated autograd hooks dormant since Module 01 without breaking existing tensor callers.

**Reflection Question:** When you imported `from tinytorch.core.tensor import Tensor` in Module 15 (Quantization), the Tensor already had gradient tracking from Module 06. How does this "single source of truth" design simplify system integration compared to having separate BasicTensor and GradTensor classes?

**Systems Analysis & Solution:**

The "single source of truth" design provides decisive architectural and systems advantages over maintaining dual `BasicTensor` and `GradTensor` class hierarchies:

1. **Zero-Copy Memory Layout and Uniform Buffers**:
   A single unified `Tensor` class encapsulates both the underlying storage buffer (`self.data: np.ndarray`) and execution metadata (`self.grad: Optional[np.ndarray]`, `self.requires_grad: bool`, `self._backward: Callable`). If separate `BasicTensor` and `GradTensor` classes existed:
   - Transitioning between inference/quantization and training would require wrapping, unwrapping, or copying the underlying raw data buffer, introducing $O(N)$ allocation overheads and cache thrashing.
   - In-place quantization passes (e.g., `tensor.quantize_int8()`) or weight packing can mutate or alias the underlying array in-place without invalidating the container identity or DAG connections.

2. **Computational Graph Discovery and Polymorphic Dispatch**:
   When autograd traverses the DAG during backpropagation, every node in the graph expects identical interfaces. If an intermediate layer output were a `BasicTensor` while its weights were a `GradTensor`, every forward and backward operator would need combinatoric type checking:
   $$\text{Op}(\text{Basic}, \text{Grad}) \rightarrow ?, \quad \text{Op}(\text{Grad}, \text{Basic}) \rightarrow ?, \quad \text{Op}(\text{Grad}, \text{Grad}) \rightarrow ?$$
   With a single unified `Tensor`, the dispatch logic reduces to a single boolean flag:
   $$\text{requires\_grad}_{\text{out}} = A.\text{requires\_grad} \lor B.\text{requires\_grad}$$
   Quantization scales and zero-points can thus be attached directly to tensors without disrupting autograd lineage.

3. **Elimination of Dual Operator Registrations**:
   Every mathematical operator (`__add__`, `__matmul__`, `relu`, `conv2d`) only needs to be implemented once against the `Tensor` interface. Separate classes would double the required test and maintenance surface across all 20 modules.

4. **Production Framework Alignment (The PyTorch 0.4 Unification)**:
   Early versions of PyTorch (pre-0.4.0) separated `torch.Tensor` (raw multidimensional arrays) from `torch.autograd.Variable` (DAG node wrappers). This caused ubiquitous boilerplate (`x = Variable(tensor)`), type conversion bugs, and double allocations. PyTorch 0.4 completely merged `Variable` into `torch.Tensor` with a `.requires_grad` attribute—exactly the architectural pattern TinyTorch embodies.

---

### Question 2: Benchmarking Methodology: Science Meets Engineering

Effective benchmarking requires rigorous methodology that bridges scientific measurement with engineering pragmatism.

**The Three Pillars of Reliable Benchmarking:**

| Pillar | Core Principle | Technical Mechanism | Failure Mode Without It |
| :--- | :--- | :--- | :--- |
| **1. Repeatability** | Same Experiment $\rightarrow$ Same Result | Fixed PRNG seeds (`default_rng(42)`), cache warmup ($N_{\text{warmup}} \ge 10$), multi-trial statistical sampling ($N \ge 100$) | Single measurement lies: a transient OS interrupt can turn a $10.0\text{ ms}$ kernel into a $15.2\text{ ms}$ false regression. |
| **2. Comparability** | Fair, Apples-to-Apples Evaluation | Standardized evaluation splits, identical batch dimensions, documented hardware environment (`platform`, CPU architecture) | Comparing inference latency on an AVX-512 server against a low-power laptop core invalidates performance claims. |
| **3. Completeness** | Multidimensional Trade-off Capture | Joint Pareto logging of Latency ($T$), Throughput ($\Phi$), Memory ($M$), and Accuracy ($\text{Acc}$) | Degenerative solutions: an aggressive pruning pass achieving $10\times$ speedup but dropping accuracy to random chance ($10\%$). |

**The Variance Story: Why Single Runs Lie:**

A single latency sample $T_{\text{single}} = 12.3\text{ ms}$ is uninterpretable: was it delayed by an OS scheduler context switch, Python garbage collection pause, or CPU thermal frequency drop? In contrast, reporting $N=100$ runs with Student's $t$ confidence bounds provides rigorous statistical confidence:

$$\bar{T} = \frac{1}{N}\sum_{i=1}^N T_i, \quad s = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (T_i - \bar{T})^2}, \quad \text{CI}_{95\%} = \bar{T} \pm t_{0.975, N-1} \frac{s}{\sqrt{N}}$$

| Measurement Strategy | Reported Latency | Statistical Validity | Systems Utility |
| :--- | :--- | :--- | :--- |
| **Single Measurement ($N=1$)** | $12.3\text{ ms}$ | Non-reproducible (Zero CI) | Useless: cannot detect whether a code change introduced a regression or caught a quiet OS tick. |
| **Statistical Aggregate ($N=100$)** | $10.0\text{ ms} \pm 0.5\text{ ms}$ | Statistically sound ($p < 0.05$) | Actionable: regressions $> 0.5\text{ ms}$ can be flagged with high confidence in automated CI pipelines. |

**Reflection Question:** Your benchmark runs inference 100 times and reports mean latency. A production API serves 1 million requests/day. Which percentile (p50, p90, p99) matters more for user experience, and why isn't mean sufficient?

**Systems Analysis & Solution:**

For user experience in production, **tail latency percentiles ($p_{90}$, $p_{99}$, and $p_{99.9}$) matter vastly more than the mean**, and mean is actively deceptive for several fundamental systems reasons:

1. **The Scale of $p_{99}$ at One Million Requests**:
   At $1{,}000{,}000$ requests per day, the 99th percentile represents the experience of **$10{,}000$ user requests every single day**:
   $$\text{Affected Users} = 1{,}000{,}000 \times (1.0 - 0.99) = \mathbf{10{,}000\text{ daily degraded interactions}}$$
   If those $10{,}000$ requests time out, stall interactive UI rendering, or exceed user patience thresholds ($>200\text{ ms}$), the service suffers severe user abandonment and customer churn, even if the mean latency looks pristine.

2. **Asymmetric Fat Tails in Real Systems**:
   Execution latency is bounded on the left by the speed of light and peak hardware FLOPs ($T > T_{\text{hardware\_min}} > 0$), but is completely unbounded on the right. Systems events—such as page faults, thread descheduling, network TCP retransmissions, container CPU throttle CFS credits, and GC collection passes—cause asymmetric, multimodal spikes.
   $$\mu = \frac{99 \times 5\text{ ms} + 1 \times 500\text{ ms}}{100} = \mathbf{9.95\text{ ms}}$$
   Here, reporting $\mu = 9.95\text{ ms}$ conceals the catastrophic reality that $1\%$ of requests suffered a **$100\times$ latency blowout** ($500\text{ ms}$).

3. **Fan-Out Amplification in Distributed Microservices**:
   Modern production architectures rarely serve a user request with a single inference. A user query typically fans out across $K$ parallel model inferences or retrieval workers (e.g., $K=20$ candidate rankers or ensemble models). The user's total waiting time is determined by the *slowest* worker:
   $$P(\text{User Request Complete in } \le t) = [P(T_{\text{worker}} \le t)]^K$$
   If each individual worker meets its SLA $99\%$ of the time ($p_{99}$), the probability that a user request involving $K=50$ backend worker calls completes without experiencing a tail spike drops dramatically:
   $$P(\text{No Tail Delay}) = 0.99^{50} \approx 0.605 \implies \mathbf{39.5\%\text{ of user requests suffer tail latency!}}$$
   Monitoring and optimizing $p_{99}$ and $p_{99.9}$ is therefore the only way to prevent distributed service degradation.

---

### Question 3: Performance Measurement Traps and How to Avoid Them

Real-world benchmarking is full of subtle traps that can invalidate your measurements.

**Common Measurement Pitfalls and Systems Countermeasures:**

| Measurement Trap | Flawed Practice | Root Cause Mechanism | Systems Mitigation |
| :--- | :--- | :--- | :--- |
| **Trap 1: Scope Leakage** | Timing input tensor creation, disk I/O, or metric logging inside the timed loop | Interleaving host I/O and memory allocations with compute kernels | Isolate purely the inference kernel: `start = perf_counter(); out = model.forward(x); latency = perf_counter() - start`. |
| **Trap 2: Transient Noise & Cold Caches** | Benchmarking on first run with background apps active | CPU cold instruction/data caches, OS page faults, thermal throttling | Execute $N_{\text{warmup}} \ge 10$ discard iterations, pin CPU thread affinity, disable background daemons. |
| **Trap 3: Cherry-Picking Best Runs** | Reporting only $\min(T)$ from 10 runs | Minimum captures unrealistic peak CPU turbo boost without thermal equilibrium | Report complete empirical distribution: median ($p_{50}$), mean $\pm$ standard deviation, and $p_{95}/p_{99}$ tail percentiles. |
| **Trap 4: Architectural Mismatch** | Benchmarking on an Apple Silicon M-series laptop and deploying to an x86 server | Differing memory bandwidth, instruction sets (NEON vs AVX-512), and cache hierarchies | Benchmark directly on production-equivalent target instances and record hardware metadata in `system_info`. |
| **Trap 5: Conflating Latency & Throughput** | Dividing batch inference time by batch size and calling it "user latency" | Amortized batch compute does not reflect single-stream interactive arrival latency | Separately report single-sample online latency ($T_{\text{online}}$, $\text{batch}=1$) and batched throughput ($\Phi = N / T$). |

**Real Example from TinyTorch:**

```python
# ❌ WRONG: Measures memory allocation, data synthesis, and formatting overhead
def bad_benchmark(model):
    start = time.perf_counter()
    x = Tensor(np.random.randn(32, 10))  # Synthesis allocation included!
    out = model.forward(x)
    preds = np.argmax(out.data, axis=1)  # Post-processing included!
    return time.perf_counter() - start

# ✅ CORRECT: Isolates pure inference compute on warm cache
def good_benchmark(model, x_cached):
    # Warmup
    for _ in range(10):
        _ = model.forward(x_cached)
    # Timed region: strictly model compute
    start = time.perf_counter()
    out = model.forward(x_cached)
    latency = time.perf_counter() - start
    return latency
```

**Reflection Question:** You benchmark a model at batch_size=32 and report 50ms latency (1.56ms per sample). A production API serves requests one at a time. Will real users experience 1.56ms latency? Why or why not?

**Systems Analysis & Solution:**

**No, real online users will NOT experience $1.56\text{ ms}$ latency.** They will experience substantially higher latency—typically $8\text{ ms}$ to $15\text{ ms}$ or worse—for fundamental systems reasons:

1. **Fixed Kernel Launch and Dispatch Overheads**:
   Every model invocation incurs fixed runtime costs: Python interpreter function call dispatch, NumPy C-API binding overhead, CPU thread pool synchronization, and memory allocator tracking:
   $$T_{\text{total}}(\text{batch}) = T_{\text{fixed\_overhead}} + \text{batch} \times T_{\text{marginal\_compute}}$$
   At $\text{batch}=32$, $T_{\text{fixed\_overhead}}$ (e.g., $3.0\text{ ms}$) is amortized over 32 samples ($\approx 0.09\text{ ms/sample}$). At $\text{batch}=1$, that entire $3.0\text{ ms}$ overhead falls squarely upon that single user sample, establishing a hard latency floor that cannot be amortized.

2. **Under-Subscribed Hardware Vectorization & Memory Bandwidth**:
   Modern CPUs and GPUs rely on wide SIMD registers (AVX-512, NEON) and parallel matrix multiplication engines (Tensor Cores). At $\text{batch}=32$, GEMM operations achieve high computational density, keeping memory buses saturated and compute pipelines full. At $\text{batch}=1$, matrix-matrix multiplication degrades to a memory-bandwidth-bound matrix-vector product (GEMV). The processor spends the majority of its clock cycles waiting for weights to travel from DRAM/L3 cache into registers, yielding poor operational intensity.

3. **Queueing Latency and Little's Law**:
   In an online API serving real users, requests arrive asynchronously according to a Poisson process with arrival rate $\lambda$. To achieve a batch size of 32 in production, incoming requests must sit in an ingress buffer:
   $$T_{\text{user\_latency}} = T_{\text{queue\_wait}} + T_{\text{inference}}$$
   The first user to arrive must wait for 31 other distinct users to send requests before the dynamic batch is triggered. If requests arrive every $1.0\text{ ms}$, the earliest user waits $31\text{ ms}$ in queue plus $50\text{ ms}$ for batch execution—experiencing an unacceptable **$81\text{ ms}$ turnaround latency**, over $50\times$ higher than the advertised $1.56\text{ ms}$ throughput metric!

---

### Question 4: Schema Validation: Making Results Machine-Readable

Your submission format uses JSON Schema validation—a powerful pattern for ensuring data quality and enabling automation.

**Unstructured vs Schema-Validated Payloads:**

| Dimension | Without Schema (Unstructured Ad-Hoc Dictionaries) | With Schema (Standardized Specification) |
| :--- | :--- | :--- |
| **Data Types** | Brittle string coercion: `"accuracy": "92%"`, `"latency": "10ms"` | Strict native typing: `"accuracy": 0.92` (float), `"latency_ms_mean": 10.0` |
| **Value Constraints** | Undetected bugs: negative memory size, accuracy $> 1.0$ | Automated range enforcement: `minimum: 0.0`, `maximum: 1.0` |
| **Field Semantics** | Ambiguous terminology: `"time": 10.5` (seconds or milliseconds?) | Unambiguous unit contracts: `"latency_ms_mean": 10.5` |
| **Automation & CI** | Manual review required; regex parsing scripts frequently break | Zero-touch automated leaderboard ingestion, verification, and regression gating |
| **Replication** | Missing hardware context prevents reproduction | Enforced `system_info` schema captures platform, architecture, and libraries |

**Schema Design Principles:**

1. **Required vs Optional Fields** — Baseline metrics and system context are strictly mandatory; optimization metadata is conditionally structured.
2. **Strict Type Safety** — Enforce numeric primitives (`number`, `integer`) rather than freeform strings.
3. **Value Boundary Verification** — Enforce mathematical limits ($\text{accuracy} \in [0.0, 1.0]$, $\text{latency} > 0.0$).
4. **Hierarchical Encapsulation** — Cleanly isolate `baseline`, `optimized`, and `improvements` sub-objects.
5. **Contract Versioning** — Explicit `tinytorch_version` field enables backward-compatible schema evolution.

**Reflection Question:** Your submission schema requires `model_size_mb` as a float. Why is this better than allowing users to write "4MB" or "4.0 megabytes" as strings? Think about aggregation and comparison.

**Systems Analysis & Solution:**

Enforcing `model_size_mb` as a floating-point number is fundamentally superior to accepting freeform strings for four critical systems reasons:

1. **Deterministic Machine Ordering and $O(N \log N)$ Sorting**:
   A competitive leaderboard or automated regression gate must sort hundreds of submissions by model footprint. Numeric floats sort in native machine registers with zero overhead. Freeform strings sort lexicographically, leading to disastrous ordering bugs where `"10.0MB" < "2.0MB"` because character `'1'` precedes `'2'`.

2. **Elimination of Binary vs Decimal Unit Ambiguity**:
   The string `"4MB"` is notorious for creating conflicting interpretations across software stacks:
   $$\text{Decimal (SI)}: 4 \times 10^6 = 4{,}000{,}000\text{ bytes} \quad \text{vs} \quad \text{Binary (IEC)}: 4 \times 2^{20} = 4{,}194{,}304\text{ bytes}$$
   A $4.86\%$ discrepancy corrupts compression ratio calculations ($C = M_{\text{base}} / M_{\text{opt}}$). By enforcing a float defined explicitly as megabytes ($M / 10^6$) or mebibytes ($M / 2^{20}$), all submissions adhere to the identical unit baseline.

3. **Automated Mathematical Validation and Constraint Checking**:
   JSON Schema can directly validate floats using mathematical boundary conditions:
   ```json
   "model_size_mb": { "type": "number", "minimum": 0.000001 }
   ```
   Accepting strings would allow invalid, unparsable entries (such as `"-4MB"`, `"four megs"`, `"N/A"`, or `"4.0 GB"`) to bypass schema validation, causing runtime exceptions during downstream analytics.

4. **Zero-Overhead Vectorized Aggregation**:
   When computing community benchmark statistics (e.g., average compression ratio across all submissions), a column of floats loads directly into a contiguous NumPy array (`np.float64`) for SIMD aggregation. Parsing strings requires regex tokenization, exception handling for malformed units, and dynamic string allocations in Python.

---

### Question 5: The Complete ML Systems Lifecycle

This capstone represents the final stage of the ML systems lifecycle—but it's also the beginning of the next iteration.

**The Seven-Stage ML Systems Lifecycle:**

| Stage | Name | Systems Focus | Primary TinyTorch Milestone |
| :--- | :--- | :--- | :--- |
| **1** | **Research & Modeling** | Architectural design, forward/backward differentiation, loss convergence | Modules 01–13 (`tensor`, `autograd`, `transformers`) |
| **2** | **Baseline Profiling** | Identifying compute and memory bottlenecks, call tree analysis | Modules 14 (`profiling`) & 19 (`benchmarking`) |
| **3** | **Systems Optimization** | Quantization, structured pruning, kernel fusion, cache memoization | Modules 15–18 (`quantization`, `acceleration`, `memoization`) |
| **4** | **Validation & Comparison** | Multi-metric Pareto evaluation, empirical speedup & compression verification | Module 20 (`capstone` BenchmarkReport) |
| **5** | **Packaging & Contract** | Schema validation, hardware context capture, deployment qualification | Module 20 Olympic Submission Schema |
| **6** | **Production Serving** | Model serving, dynamic batching, hardware execution | Milestones 05 & 06 (Serving & Inference) |
| **7** | **Monitoring & Drift Loop** | Tail latency tracking, SLA compliance, continuous profiling | Continuous feedback triggering Stage 2 or 3 |

**Key Insight:** Production ML is never "write once and forget." Hardware, libraries, and traffic distributions continuously evolve. Standardized benchmark reports capture an immutable snapshot of systems performance so any future regression can be pinpointed instantly.

**Reflection Question:** You deploy a model with 92% accuracy and 10ms latency. Three months later, users complain it's slow. Monitoring shows 30ms latency now (same model, same code). You didn't save system_info in your original benchmark. What went wrong, and how does proper benchmarking prevent this?

**Systems Analysis & Solution:**

When latency triples from $10\text{ ms}$ to $30\text{ ms}$ without any code changes, **the underlying systems environment has drifted**. Because `system_info` was not recorded, engineers face a blind troubleshooting nightmare.

**Probable Systems Root Causes:**

1. **BLAS / Linear Algebra Library Drift**:
   A routine host update or container image rebuild may have replaced an optimized BLAS library (Intel MKL or OpenBLAS compiled with AVX-512 vectorization) with an unoptimized generic BLAS fallback, or unset the environment variable `OMP_NUM_THREADS=1`, causing catastrophic OpenMP thread over-subscription and lock thrashing.
2. **CPU Dynamic Voltage and Frequency Scaling (DVFS) / Thermal Throttling**:
   The initial benchmark was performed when the server was cool and idle. Under sustained production load or elevated rack ambient temperature, the CPU downclocked from a $3.8\text{ GHz}$ turbo frequency to a $1.2\text{ GHz}$ thermal protection throttle.
3. **Container Resource Contention and CFS Throttling**:
   The deployment was moved into a container (Docker/K8s) where CPU quota limits (`cpu.cfs_quota_us`) were exhausted by sibling processes on the same host ("noisy neighbors"), pausing the inference thread mid-kernel.
4. **Memory Swapping and Major Page Faults**:
   Another service on the node consumed host memory, forcing model weight buffers out of high-speed DDR RAM into swap space on disk. Every forward pass incurred millisecond-scale page fault interrupts.

**How Proper Benchmarking Prevents and Resolves This:**

By capturing a comprehensive `system_info` snapshot in the benchmark schema:
- `platform`: OS version, kernel release (`uname -r`)
- `cpu`: Exact processor model, microarchitecture, core count, cache sizes (L1/L2/L3)
- `python_version` & `numpy_version`: Runtime environment
- `blas_info`: Linked BLAS/LAPACK libraries and vector instruction flags (AVX2, AVX-512, NEON)
- `thread_count`: Configured active thread pool

When the regression occurs, engineers simply execute a **diff between the current deployment node and the benchmark baseline**. Within seconds, they discover:
$$\Delta\text{Config}: \quad \text{Baseline BLAS} = \text{libmkl\_avx512.so} \quad \longleftrightarrow \quad \text{Production BLAS} = \text{libopenblas\_generic.so}$$
This turns a multi-day blind investigation into an immediate, deterministic fix.

---

### Question 6: Your Path Forward: From Learning to Production

You've completed an educational framework, but the patterns you learned apply directly to production systems.

**Translating TinyTorch Skills to Production:**

| TinyTorch Educational Pattern | Production Enterprise Equivalent | Industry Standard / Tooling |
| :--- | :--- | :--- |
| `BenchmarkReport` | Centralized Experiment Tracking | MLflow, Weights & Biases, Neptune.ai |
| `generate_submission()` | Automated Artifact & Metric Packaging | BentoML, TorchScript, ONNX Model Cards |
| `validate_submission_schema()` | Schema Enforcement & API Contracts | Pydantic, Protobuf, JSON Schema |
| `system_info` collection | Reproducible Containerized Environments | Docker, OCI Containers, Kubernetes |
| Baseline vs Optimized comparison | A/B Testing & Shadow Deployments | Triton Inference Server, TorchServe, vLLM |
| `calculate_improvements()` | Automated Regression CI/CD Gates | GitHub Actions Performance Regression Testing |

**Real-World Applications:**

1. **Model Comparison** — Same workflow as Module 20, scaled across dozens of candidate checkpoints and quantization bit-widths.
2. **Continuous Performance CI/CD** — Triggering automated benchmarks on every Pull Request to catch latency regressions before merging.
3. **Reproducible Research Standards** — Aligning with MLPerf and Papers with Code submission guidelines.
4. **Engineering Alignment** — Providing an unambiguous, cross-functional scorecard for researchers, systems engineers, and product teams.

Congratulations! You've gone from implementing basic multidimensional array striding to architecting, optimizing, and benchmark-validating an end-to-end deep learning framework. 🚀
"""

# %% [markdown]
"""
## ⭐ Aha Moment: You Built a Complete ML System

**What you built:** A professional benchmarking and submission system for your TinyTorch models.

**Why it matters:** You've gone from raw tensors to complete ML systems! Your capstone ties
together everything: models, training, optimization, profiling, and benchmarking. The
submission format you created is how real ML competitions and production deployments work.

Congratulations - you've built a deep learning framework from scratch!
"""

# %%
def demo_capstone():
    """🎯 See your complete system come together."""
    print("🎯 AHA MOMENT: You Built a Complete ML System")
    print("=" * 45)

    print("\n📚 Your TinyTorch Journey:")
    print()
    print("  Modules 01-08: Foundation")
    print("    Tensor -> Activations -> Layers -> Losses")
    print("    -> DataLoader -> Autograd -> Optimizers -> Training")
    print()
    print("  Modules 09-13: Neural Architectures")
    print("    Conv2d -> Tokenization -> Embeddings")
    print("    -> Attention -> Transformers")
    print()
    print("  Modules 14-19: Production Optimization")
    print("    Profiling -> Quantization -> Compression")
    print("    -> Acceleration -> KV Caching -> Benchmarking")
    print()
    print("  Module 20: Capstone")
    print("    Complete benchmarking and submission system")

    print("\n✨ From np.array to production ML - congratulations!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_capstone()

# %% [markdown]
r"""
## 🚀 MODULE SUMMARY: Capstone

Congratulations! You've completed the TinyTorch capstone by building a professional benchmarking and submission system!

### Key Accomplishments
- **Built a complete BenchmarkReport class** with comprehensive performance measurement (accuracy, latency, throughput, memory)
- **Implemented submission generation** with standardized JSON format and schema validation
- **Created comparison infrastructure** for automatic calculation of speedup, compression, and accuracy delta
- **Demonstrated complete workflows** from baseline to optimized models with reproducible results
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Benchmarking science**: Repeatability, comparability, and completeness principles
- **Metrics that matter**: Latency vs throughput, mean vs variance, accuracy vs efficiency trade-offs
- **Reproducibility requirements**: System context, schema validation, and standardized reporting
- **Production patterns**: How real ML systems measure and compare model performance

### 20-Module Grand Systems Milestone Scorecard

| Curriculum Tier | Modules Completed | Core Systems Primitives Built | Systems Capabilities Unlocked |
| :--- | :--- | :--- | :--- |
| **Tier 1: Foundations** | `01_tensor` $\rightarrow$ `08_training` | Strided N-D array memory, activations, modular layers, cross-entropy loss, data batching, autograd DAG engine, SGD/Adam, training loop | Deterministic striding, dynamic reverse-mode autodiff, loss convergence, backpropagation |
| **Tier 2: Architectures** | `09_convolutions` $\rightarrow$ `13_transformers` | 2D im2col convolutions, BPE tokenizer, token/positional embeddings, scaled dot-product multi-head causal attention, full decoder Transformer | Vision filtering, subword tokenization, sequence autoregression, language modeling |
| **Tier 3: Optimization** | `14_profiling` $\rightarrow$ `18_memoization` | cProfile call trees, INT8 symmetric affine quantization, magnitude/structured pruning, SIMD vectorization & operator fusion, KV-cache memoization | $4\times$ weight compression, $2.9\times$ latency speedup, eliminating duplicate autoregressive attention compute |
| **Tier 4: Evaluation** | `19_benchmarking` $\rightarrow$ `20_capstone` | High-resolution statistical timers, Student's $t$ confidence bounds, Pareto frontier trade-offs, JSON Schema contract enforcement | Publication-grade empirical measurement, automated regression gating, TinyTorch Olympic qualification |

### Ready for Next Steps

You started Module 01 with a simple Tensor class. Now you have:
- A complete ML framework built from first principles
- Advanced hardware-aware optimization techniques
- Professional benchmarking and evaluation infrastructure
- Reproducible, shareable JSON Schema-validated results

**You didn't just learn ML systems — you BUILT one from scratch.**

Export with: `tito module complete 20`

**Next**: The TinyTorch Olympics in `milestones/06_2018_mlperf/` pit your submission against everyone else's. Congratulations on completing TinyTorch!
"""
