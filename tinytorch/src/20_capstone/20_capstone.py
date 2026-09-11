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
# Module 20: Capstone - Benchmarking & Submission

Welcome to the TinyTorch capstone! You've built an entire ML framework from scratch across 19 modules. Now it's time to demonstrate your work by benchmarking a model and generating a submission that showcases your framework's capabilities.

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML framework with profiling (M14), quantization (M15), compression (M16), acceleration (M17), memoization (M18), and benchmarking (M19)
**You'll Build**: Professional benchmark submission workflow with standardized reporting
**You'll Enable**: Shareable, reproducible results demonstrating framework performance

**Connection Map**:
```
Modules 01-13 → Optimization Suite (14-18) → Benchmarking (19) → Submission (20)
(Framework)     (Performance Tools)            (Measurement)       (Results)
```

## 🎯 Learning Objectives
By the end of this capstone, you will:
1. Use Module 19's `precise_timer` to measure latency and throughput as two separate measurements
2. Apply optimization techniques from Modules 15 and 16 to improve a baseline model
3. Generate standardized JSON submissions following industry best practices
4. Validate submissions against a schema for reproducibility
5. Compare baseline vs. optimized models with quantitative metrics
6. Share your results with the TinyTorch community in a professional format

Let's get started!

## 📦 Where This Code Lives in the Final Package

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
# Eligibility is separate from a well-formed submission.
qualifies_event(report.metrics, OlympicEvent.LATENCY_SPRINT)
save_submission(submission, "my_submission.json")
```

**Why this matters:**
- **Learning:** Complete workflow from model to shareable results
- **Production:** Professional submission format mirroring MLPerf and Papers with Code standards
- **Community:** Share and compare results with other builders using standardized metrics
- **Reproducibility:** Schema-validated submissions ensure results can be verified and trusted
"""

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01-19 must be complete

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for the report timestamp)
- `json` (for submission serialization)
- `pathlib` (for file path handling)
- `platform` (for system information)
- `enum` (for the capstone event names)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class from Module 01)
- `tinytorch.core.layers` (Linear layer from Module 03)
- `tinytorch.core.activations` (ReLU from Module 02)
- `tinytorch.perf.benchmarking` (`precise_timer` from Module 19, used for every timing)
- `tinytorch.perf.profiling`, `tinytorch.perf.quantization`, `tinytorch.perf.compression` (Modules 14, 15, 16; imported only inside the optimization workflow example)

**Dependency Flow**:
```
Modules 01-13 → Modules 14-18 → Module 19 → Module 20 (Capstone)
(Framework)     (Optimization)   (Benchmark)  (Submission)
```

Students completing this module will demonstrate their complete framework's capabilities through reproducible benchmarking and professional submission generation.
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

### Industry Standard: Benchmarking Frameworks

Professional ML systems use standardized benchmarking frameworks:

```
Industry Benchmarking Standards:
┌──────────────────────────────────────────────────────────────┐
│ MLPerf (AI Hardware)     │ Papers with Code (Research)       │
├──────────────────────────┼───────────────────────────────────┤
│ • Standardized tasks     │ • Leaderboards for all datasets   │
│ • Hardware specifications│ • Reproducible results required   │
│ • Measurement protocols  │ • Code submission mandatory       │
│ • Fair comparisons       │ • Automated verification          │
└──────────────────────────┴───────────────────────────────────┘
```

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
"""
## 📐 Foundations: The Science of Benchmarking

Before we build our submission system, let's understand what makes a good benchmark and why standardized reporting matters.

### The Three Pillars of Good Benchmarking

```
Good Benchmarks Rest on Three Pillars:
┌─────────────────┬─────────────────┬─────────────────┐
│ Repeatability   │ Comparability   │ Completeness    │
├─────────────────┼─────────────────┼─────────────────┤
│ Same result     │ Apples-to-apples│ All relevant    │
│ every time      │ comparisons     │ metrics captured│
│                 │                 │                 │
│ • Fixed seeds   │ • Same hardware │ • Accuracy      │
│ • Same data     │ • Same metrics  │ • Latency       │
│ • Same config   │ • Same protocol │ • Memory        │
│ • Variance      │ • Documented    │ • Throughput    │
└─────────────────┴─────────────────┴─────────────────┘
```

### What Metrics Actually Matter?

Different stakeholders care about different metrics:

```
Stakeholder View:
┌──────────────────────────────────────────────────────────────┐
│ ML Researcher:                                               │
│   Primary   → Accuracy, F1, BLEU (task-specific)             │
│   Secondary → Training time, convergence                     │
│                                                              │
│ Systems Engineer:                                            │
│   Primary   → Latency (p50, p99), throughput                 │
│   Secondary → Memory usage, CPU/GPU utilization              │
│                                                              │
│ Product Manager:                                             │
│   Primary   → User experience (latency < 100ms?)             │
│   Secondary → Cost per request, scalability                  │
│                                                              │
│ DevOps/MLOps:                                                │
│   Primary   → Model size (deployment), inference cost        │
│   Secondary → Batch throughput, hardware utilization         │
└──────────────────────────────────────────────────────────────┘
```

**Key Insight**: A complete benchmark captures ALL perspectives, not just one.

### Benchmark Report Components

Our BenchmarkReport class will track everything needed for reproducibility:

```
BenchmarkReport Structure:
┌─────────────────────────────────────────────────────────────┐
│ Model Characteristics:                                      │
│   • Parameter count     → Model capacity                    │
│   • Model size (MB)     → Deployment cost                   │
│                                                             │
│ Performance Metrics:                                        │
│   • Accuracy           → Task performance                   │
│   • Latency (mean/std) → Inference speed + variance         │
│   • Throughput         → Samples/second capacity            │
│                                                             │
│ System Context:                                             │
│   • Platform           → Hardware/OS environment            │
│   • Python version     → Language runtime                   │
│   • NumPy version      → Numerical library version          │
│   • Timestamp          → When benchmark was run             │
└─────────────────────────────────────────────────────────────┘
```

### Latency vs. Throughput: A Critical Distinction

Many beginners confuse latency and throughput. They measure different things:

```
Latency vs. Throughput:

Latency (Per-Sample Speed):
┌──────────────────────────────────────────────────┐
│  Input → Model → Output                          │
│   ↑              ↓                               │
│   └──── 10ms ────┘                               │
│                                                  │
│  "How fast can I get ONE result?"                │
│  Critical for: Real-time apps, user experience   │
└──────────────────────────────────────────────────┘

Throughput (Batch Capacity):
┌──────────────────────────────────────────────────┐
│  [Input1, Input2, ... Input100]                  │
│           ↓                                      │
│        Model                                     │
│           ↓                                      │
│  [Out1, Out2, ... Out100] in 200ms               │
│                                                  │
│  "How many samples per second?"                  │
│  Critical for: Batch jobs, data processing       │
└──────────────────────────────────────────────────┘

Example:
  Latency:     10ms per sample   → "Fast" for users
  Throughput:  500 samples/sec   → "Fast" for batches

Trade-off: Batching increases throughput but adds latency!
```

Because the two pull in opposite directions, `BenchmarkReport` measures them with two different calls: latency times `model.forward` on one sample, and throughput times `model.forward` on the whole test batch and divides the batch size by that time. Deriving one from the other (`1000 / latency_ms`) would erase exactly the trade-off this box describes.

### Why Variance Matters

Single measurements lie. Variance tells the truth:

```
Why We Report Mean ± Std:

Measurement 1: 9.2ms    ┐
Measurement 2: 10.1ms   │ Mean = 10.0ms
Measurement 3: 9.8ms    │ Std  = 0.5ms
Measurement 4: 10.5ms   │
Measurement 5: 10.4ms   ┘

vs.

Measurement 1: 5.2ms    ┐
Measurement 2: 14.8ms   │ Mean = 10.0ms ← Same mean!
Measurement 3: 8.1ms    │ Std  = 4.2ms  ← Different variance!
Measurement 4: 15.3ms   │
Measurement 5: 6.6ms    ┘
           ↑
    Unpredictable performance!
```

**Which model would you deploy?** The first one, because consistent performance matters in production.

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
"""
### Understanding SimpleMLP Parameter Counting

Let's break down where the parameters come from:

```
SimpleMLP Parameter Breakdown:
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Linear(10, 20)                                     │
│   Weight matrix: (10, 20) = 200 parameters                  │
│   Bias vector:   (20,)    = 20 parameters                   │
│   Subtotal: 220 parameters                                  │
│                                                             │
│ Layer 2: ReLU                                               │
│   No parameters (just max(0, x))                            │
│   Subtotal: 0 parameters                                    │
│                                                             │
│ Layer 3: Linear(20, 3)                                      │
│   Weight matrix: (20, 3)  = 60 parameters                   │
│   Bias vector:   (3,)     = 3 parameters                    │
│   Subtotal: 63 parameters                                   │
│                                                             │
│ TOTAL: 220 + 0 + 63 = 283 parameters                        │
└─────────────────────────────────────────────────────────────┘

Memory Calculation (FP32):
  283 parameters × 4 bytes/param = 1,132 bytes ≈ 0.001 MB

If we quantize to INT8:
  283 parameters × 1 byte/param = 283 bytes ≈ 0.0003 MB
  → 4× memory reduction!
```

This small model is perfect for demonstrating optimization impact without long benchmark times.
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
        if X_test.shape[0] == 0:
            raise ValueError("X_test must contain at least one sample")
        y_test = np.asarray(y_test)
        if y_test.shape != (X_test.shape[0],):
            raise ValueError("y_test must contain one class index per sample")
        if num_runs <= 0:
            raise ValueError("num_runs must be positive")
        # Count parameters and stored size (see measure_memory)
        param_count = model.count_parameters()
        model_size_mb = self.measure_memory(model)

        # Measure accuracy
        predictions = model.forward(X_test)
        pred_labels = np.argmax(predictions.data, axis=1)
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
        ### BEGIN SOLUTION
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
        ### BEGIN SOLUTION
        if hasattr(model, 'size_bytes'):
            return model.size_bytes() / (1024 * 1024)
        return sum(param.data.nbytes for param in model.parameters()) / (1024 * 1024)
        ### END SOLUTION

# %% [markdown]
"""
### Why These Metrics?

Each metric answers a specific production question:

```
Metric Decision Tree:
┌─────────────────────────────────────────────────────────────┐
│ Question                 │ Metric              │ Why        │
├──────────────────────────┼─────────────────────┼────────────┤
│ "Will it fit on device?" │ model_size_mb       │ Memory     │
│ "Is it accurate enough?" │ accuracy            │ Quality    │
│ "Is it fast enough?"     │ latency_ms_mean     │ UX         │
│ "Is it consistent?"      │ latency_ms_std      │ Reliability│
│ "Can it scale?"          │ throughput          │ Capacity   │
│ "How complex is it?"     │ parameter_count     │ Capacity   │
└─────────────────────────────────────────────────────────────┘
```

### Design Choice: Warmup, Then num_runs=100

`measure_latency` makes a few untimed calls first, then times 100 calls:
- **Warmup is untimed**, so first-call costs (allocations, cold caches) stay out of the numbers
- **100 timed runs** average out OS interrupts and GC pauses, and the std shows how consistent the model is
- **The median is reported alongside mean ± std** because a few slow outliers drag the mean; the submission compares medians for that reason
- **Std is a spread, not a confidence interval**: it says how wide the distribution is, not how sure you are of the mean

```
Single Run (Unreliable):        Multiple Runs (Reliable):
┌─────────────────────────┐     ┌─────────────────────────┐
│ Run 1: 12.3ms           │     │ Run 1: 12.3ms           │
│                         │     │ Run 2: 9.8ms            │
│ Result: 12.3ms          │     │ Run 3: 10.1ms           │
│ Spread: unknown         │     │ ...                     │
│ (Could be outlier!)     │     │ Run 100: 10.2ms         │
│                         │     │                         │
│                         │     │ Result: 10.0ms ± 0.5ms  │
│                         │     │ Spread: visible         │
│                         │     │ (median 10.1ms)         │
└─────────────────────────┘     └─────────────────────────┘
```

### Design Choice: Python Native Types

Notice we convert all metrics to Python native types (int, float):

```python
'parameter_count': int(param_count),  # NumPy int64 → Python int
'accuracy': float(accuracy),          # NumPy float64 → Python float
```

**Why?** JSON can't serialize NumPy types directly:
```python
# ❌ This fails:
json.dumps({"value": np.int64(42)})  # TypeError!

# ✅ This works:
json.dumps({"value": int(42)})  # Success!
```

This design decision makes our submissions JSON-compatible without custom encoders.
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
"""
### Understanding the Improvements Calculation

When you provide both baseline and optimized results, the submission auto-calculates three key improvement metrics:

```
Improvement Metrics Explained:

1. Speedup (Latency Ratio):
   ┌────────────────────────────────────────────────┐
   │ Speedup = baseline_latency / optimized_latency │
   │                                                │
   │ Example:                                       │
   │   Baseline:  10.0ms                            │
   │   Optimized: 5.0ms                             │
   │   Speedup:   10.0 / 5.0 = 2.0x                 │
   │                                                │
   │ Interpretation:                                │
   │   2.0x = Optimized model is 2× faster          │
   │   1.0x = No change                             │
   │   0.5x = Optimized model is slower (bad!)      │
   └────────────────────────────────────────────────┘

2. Compression Ratio (Size Reduction):
   ┌────────────────────────────────────────────────┐
   │ Compression = baseline_size / optimized_size   │
   │                                                │
   │ Example:                                       │
   │   Baseline:  4.0 MB                            │
   │   Optimized: 1.0 MB                            │
   │   Compression: 4.0 / 1.0 = 4.0x                │
   │                                                │
   │ Interpretation:                                │
   │   4.0x = Model is 4× smaller                   │
   │   1.0x = Same size                             │
   │   0.8x = Larger after "optimization" (bad!)    │
   └────────────────────────────────────────────────┘

3. Accuracy Delta (Quality Impact):
   ┌────────────────────────────────────────────────┐
   │ Delta = optimized_accuracy - baseline_accuracy │
   │                                                │
   │ Example:                                       │
   │   Baseline:  92.0%                             │
   │   Optimized: 91.5%                             │
   │   Delta:     91.5 - 92.0 = -0.5%               │
   │                                                │
   │ Interpretation:                                │
   │   +0.5% = Improved accuracy (rare but good!)   │
   │    0.0% = Maintained accuracy (ideal!)         │
   │   -0.5% = Slight loss (acceptable)             │
   │   -5.0% = Major loss (unacceptable)            │
   └────────────────────────────────────────────────┘
```

### The Optimization Trade-off Triangle

Every optimization involves trade-offs:

```
The Impossible Triangle:
         Fast (Speedup)
              ▲
             /│\
            / │ \
           /  │  \
          /   │   \
         /  Good  \
        /  Balance \
       ▼─────────────▼
    Small         Accurate
  (Compression)   (Delta)

You can pick TWO:
• Fast + Small   → Aggressive optimization, some accuracy loss
• Fast + Accurate → Careful optimization, less compression
• Small + Accurate → Conservative quantization, slower

The goal: Find the sweet spot for YOUR use case!
```

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
            assert isinstance(value, (int, float)) and np.isfinite(value), f"{section}.{metric} must be finite"
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy must be in [0, 1]"
        assert metrics['parameter_count'] > 0, "Parameter count must be positive"
        assert metrics['model_size_mb'] > 0, "Model size must be positive"
        assert metrics['latency_ms_mean'] > 0, "Latency must be positive"

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
"""
### Understanding the Workflow Pattern

This workflow follows industry best practices:

```
Production ML Workflow:
┌─────────────────────────────────────────────────────────────┐
│ 1. Define Task                                              │
│    ↓ What are we solving? What's the test set?              │
│                                                             │
│ 2. Baseline Model                                           │
│    ↓ Simplest reasonable model                              │
│                                                             │
│ 3. Baseline Benchmark                                       │
│    ↓ Measure: accuracy, latency, memory                     │
│                                                             │
│ 4. Optimization (ITERATIVE)                                 │
│    ↓ Try technique → Benchmark → Compare → Keep or revert   │
│    ↓ Quantization? Pruning? Distillation?                   │
│                                                             │
│ 5. Final Submission                                         │
│    ↓ Document: baseline, optimized, improvements            │
│    ↓ Share: JSON file, metrics, techniques                  │
│                                                             │
│ 6. Community Comparison                                     │
│    ↓ How do your results compare to others?                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Professional ML engineers iterate on step 4, trying different optimizations and measuring their impact. The submission captures the BEST result after this exploration.
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
"""
#### Combining Multiple Optimizations

In production ML, you often stack optimizations for cumulative benefits:

```
Stacking Optimizations (illustrative numbers):
┌─────────────────────────────────────────────────────────────┐
│ Baseline Model                                              │
│   Size: 4.0 MB, Latency: 10.0ms, Accuracy: 92.0%            │
│                                                             │
│ ↓ Apply Quantization (INT8)                                 │
│   Size: 1.0 MB (4.0×), Latency: 5.0ms (2.0×), Acc: 91.8%    │
│                                                             │
│ ↓ Apply Pruning (50% sparsity)                              │
│   Size: 0.5 MB (2.0×), Latency: 3.5ms (1.4×), Acc: 91.5%    │
│                                                             │
│ Final Optimized Model                                       │
│   Total compression: 8.0× (4.0 MB → 0.5 MB)                 │
│   Total speedup: 2.9× (10.0ms → 3.5ms)                      │
│   Accuracy loss: -0.5% (92.0% → 91.5%)                      │
└─────────────────────────────────────────────────────────────┘

Key Insight: Effects multiply!
  Quant (4.0×) × Pruning (2.0×) = 8.0× total compression
```

The submission's `techniques_applied` list documents this for reproducibility:
```json
"techniques_applied": ["int8_quantization", "magnitude_pruning_0.5"]
```

This tells other engineers EXACTLY what you did, so they can reproduce or build on your work!
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
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of benchmarking, reproducibility, and ML systems integration:

You've built an entire ML framework across 20 modules. This capstone asks you to step back and reflect on the complete systems journey—from tensors to production-ready benchmarking.

### Question 1: End-to-End System Integration

Modern ML systems aren't just individual components working in isolation—they're carefully orchestrated pipelines where each piece connects to form a cohesive whole.

**The Complete Pipeline You Built:**

```
Data → Tensor (M01) → Layers (M03) → Model → Training (M08)
                ↓                      ↓           ↓
          Activations (M02)      DataLoader (M05) Spatial Ops (M09)
                ↓                      ↓
          Losses (M04)           Autograd (M06) → Optimizers (M07)
                                       ↓
                              Advanced Architectures
                         (Tokenization, Embeddings, Attention,
                          Transformers: M10-M13)
                                       ↓
                              Optimization Pipeline
                         (Profiling, Quantization, Compression,
                          KV Cache, Acceleration: M14-M18)
                                       ↓
                           Measurement & Validation
                         (Benchmarking M19, Submission M20)
```

**Systems Integration Lessons:**

1. **Dependency Management** - Each module imports from previous modules, creating a proper dependency graph
2. **API Consistency** - Tensor operations work the same whether in Module 01 or Module 20
3. **Composability** - Complex systems (transformers) built from simple primitives (linear layers)
4. **Progressive Enhancement** - Module 06 activated gradients dormant since Module 01

**Reflection Question:** When you imported `from tinytorch.core.tensor import Tensor` in Module 15 (Quantization), the Tensor already had gradient tracking from Module 06. How does this "single source of truth" design simplify system integration compared to having separate BasicTensor and GradTensor classes?

### Question 2: Benchmarking Methodology: Science Meets Engineering

Effective benchmarking requires rigorous methodology that bridges scientific measurement with engineering pragmatism.

**The Three Pillars of Reliable Benchmarking:**

```
1. REPEATABILITY (Same Experiment → Same Result)
   ┌─────────────────────────────────────────┐
   │ • Fixed random seeds (default_rng)      │
   │ • Same test dataset across runs         │
   │ • Consistent environment (same hardware)│
   │ • Multiple runs to capture variance     │
   │                                         │
   │ Why: Single measurements lie            │
   │ 10.3ms once vs 10.0ms ± 0.5ms (100×)    │
   └─────────────────────────────────────────┘

2. COMPARABILITY (Fair Comparisons)
   ┌─────────────────────────────────────────┐
   │ • Same hardware platform                │
   │ • Same test data for baseline/optimized │
   │ • Same metrics (latency, accuracy)      │
   │ • Documented environment (sys.platform) │
   │                                         │
   │ Why: Apples-to-apples decisions         │
   │ Can't compare GPU timing to CPU timing  │
   └─────────────────────────────────────────┘

3. COMPLETENESS (Capture All Dimensions)
   ┌─────────────────────────────────────────┐
   │ • Accuracy (quality metric)             │
   │ • Latency (speed metric)                │
   │ • Memory (resource metric)              │
   │ • Throughput (capacity metric)          │
   │                                         │
   │ Why: Optimizations have trade-offs      │
   │ Fast + Small might mean Less Accurate   │
   └─────────────────────────────────────────┘
```

**Measurement Best Practices You Implemented:**

1. **Warm-up runs** - First inference is often slower (cold cache)
2. **Statistical aggregation** - Report mean ± std, not single values
3. **Multiple metrics** - Never optimize for just one dimension
4. **System context** - Platform, Python version, library versions matter

**The Variance Story:**

```python
# Why we run 100 iterations instead of 1:

Single measurement: 12.3ms
  → Could be outlier (GC pause? OS interrupt?)
  → No confidence interval
  → Can't detect performance regressions

100 measurements: 10.0ms ± 0.5ms
  → Statistically valid
  → Confidence: "Next run will likely be 9.5-10.5ms"
  → Can detect if update made things worse
```

**Reflection Question:** Your benchmark runs inference 100 times and reports mean latency. A production API serves 1 million requests/day. Which percentile (p50, p90, p99) matters more for user experience, and why isn't mean sufficient?

### Question 3: Performance Measurement Traps and How to Avoid Them

Real-world benchmarking is full of subtle traps that can invalidate your measurements.

**Common Measurement Pitfalls:**

```
TRAP 1: Measuring the Wrong Thing
  ❌ Timing model creation instead of inference
  ❌ Including data loading in latency measurement
  ❌ Measuring batch=32 when production uses batch=1

  ✅ FIX: Isolate exactly what you're measuring
     start = time.perf_counter()
     output = model.forward(x)  # ONLY this
     latency = time.perf_counter() - start

TRAP 2: Ignoring System Noise
  ❌ Running benchmarks while streaming video
  ❌ Single measurement (affected by GC, OS)
  ❌ Not warming up (first run is slow)

  ✅ FIX: Multiple runs, discard outliers
     for _ in range(100):  # Warm up + measure
         measure_latency()
     report mean ± std

TRAP 3: Cherry-Picking Results
  ❌ "Ran 10 times, best was 8.2ms!" (reporting min)
  ❌ Rerunning until you get good numbers
  ❌ Omitting variance in reporting

  ✅ FIX: Report full distribution
     "10.0ms ± 0.5ms (n=100, p99=11.2ms)"

TRAP 4: Wrong Hardware Baseline
  ❌ Benchmarking on MacBook, deploying to server
  ❌ Comparing GPU results to CPU results
  ❌ Not documenting hardware (can't reproduce)

  ✅ FIX: Benchmark on deployment hardware
     submission['system_info'] = {
       'platform': platform.platform(),
       'cpu': 'Intel Xeon Gold',
       'gpu': 'NVIDIA A100'
     }

TRAP 5: Confusing Latency and Throughput
  ❌ "Processes 1000 samples in 10s = 0.01s per sample"
     (Batch processing != per-sample latency!)
  ❌ Optimizing throughput hurts latency (big batches)

  ✅ FIX: Measure both separately
     latency = measure_single_sample()
     throughput = measure_batch_processing()
```

**Real Example from TinyTorch:**

```python
# ❌ WRONG: Measures more than inference
def bad_benchmark():
    start = time.time()
    x = create_random_input()      # Includes data generation!
    output = model.forward(x)
    result = postprocess(output)   # Includes postprocessing!
    return time.time() - start

# ✅ CORRECT: Isolates inference
def good_benchmark():
    x = create_random_input()      # Setup (not timed)

    start = time.time()
    output = model.forward(x)      # ONLY inference
    latency = time.time() - start

    postprocess(output)            # Cleanup (not timed)
    return latency
```

**Reflection Question:** You benchmark a model at batch_size=32 and report 50ms latency (1.56ms per sample). A production API serves requests one at a time. Will real users experience 1.56ms latency? Why or why not?

### Question 4: Schema Validation: Making Results Machine-Readable

Your submission format uses JSON Schema validation—a powerful pattern for ensuring data quality and enabling automation.

**Why Schema Validation Matters:**

```
WITHOUT Schema:                     WITH Schema:
┌──────────────────────────┐       ┌──────────────────────────┐
│ {                        │       │ {                        │
│   "accuracy": "92%",     │  bad  │   "accuracy": 0.92,      │  ok
│   "latency": 10.5,       │  bad  │   "latency_ms_mean": 10.5│  ok
│   "time": "today"        │  bad  │   "timestamp": "2025..." │  ok
│ }                        │       │ }                        │
│                          │       │                          │
│ Problems:                │       │ Benefits:                │
│ • Wrong type (string %)  │       │ • Enforced types (float) │
│ • Ambiguous name         │       │ • Clear field names      │
│ • Unparsable time        │       │ • Standard format        │
│ • Can't aggregate        │       │ • Automated validation   │
│ • No automation possible │       │ • Aggregation works      │
└──────────────────────────┘       └──────────────────────────┘
```

**Schema Design Principles:**

1. **Required fields** - Baseline metrics are mandatory, optimized optional
2. **Type safety** - `accuracy: float` not `accuracy: any`
3. **Value constraints** - `accuracy in [0.0, 1.0]` catches errors
4. **Nested structure** - Group related fields (`baseline: {metrics: {...}}`)
5. **Version tracking** - `tinytorch_version: "0.1.0"` enables evolution

**The Power of Machine-Readable Data:**

```python
# With schema-validated submissions, you can:

# 1. Automatically aggregate community results
all_submissions = load_all_submissions()
avg_accuracy = np.mean([s['baseline']['metrics']['accuracy']
                       for s in all_submissions])

# 2. Build leaderboards
sorted_by_speedup = sorted(all_submissions,
                          key=lambda s: s['improvements']['speedup'],
                          reverse=True)

# 3. Detect regressions
if new_latency > baseline_latency * 1.1:
    alert("Performance regression detected!")

# 4. Generate visualizations
plot_accuracy_vs_speedup(all_submissions)
```

**Reflection Question:** Your submission schema requires `model_size_mb` as a float. Why is this better than allowing users to write "4MB" or "4.0 megabytes" as strings? Think about aggregation and comparison.

### Question 5: The Complete ML Systems Lifecycle

This capstone represents the final stage of the ML systems lifecycle—but it's also the beginning of the next iteration.

**The Never-Ending Loop:**

```
            ┌──────────────────────────────────┐
            │    1. RESEARCH & DEVELOPMENT     │
            │  (Modules 01-13: Build framework)│
            └────────────┬─────────────────────┘
                         ↓
            ┌──────────────────────────────────┐
            │     2. BASELINE MEASUREMENT      │
            │   (Module 19: Benchmark baseline)│
            └────────────┬─────────────────────┘
                         ↓
            ┌──────────────────────────────────┐
            │      3. OPTIMIZATION PHASE       │
            │ (Modules 14-18: Apply techniques)│
            └────────────┬─────────────────────┘
                         ↓
            ┌──────────────────────────────────┐
            │    4. VALIDATION & COMPARISON    │
            │  (Module 20: Benchmark optimized)│
            └────────────┬─────────────────────┘
                         ↓
            ┌──────────────────────────────────┐
            │     5. DECISION & SUBMISSION     │
            │  (Keep? Deploy? Iterate? Share?) │
            └────────────┬─────────────────────┘
                         ↓
                   Did we meet goals?
                         ↓
                    No ─────→ (Loop back to step 3)
                         ↓ Yes
            ┌──────────────────────────────────┐
            │      6. PRODUCTION DEPLOY        │
            │   (Model serves real traffic)    │
            └────────────┬─────────────────────┘
                         ↓
            ┌──────────────────────────────────┐
            │     7. MONITORING & FEEDBACK     │
            │  (Is performance degrading? New  │
            │   optimization opportunities?)   │
            └────────────┬─────────────────────┘
                         ↓
                   (Loop back to step 1)
```

**Key Insight:** Production ML is iterative. Your submission captures a snapshot, but the system keeps evolving. This is why reproducibility (schema, environment documentation) is critical—you need to know what changed when performance shifts.

**Reflection Question:** You deploy a model with 92% accuracy and 10ms latency. Three months later, users complain it's slow. Monitoring shows 30ms latency now (same model, same code). You didn't save system_info in your original benchmark. What went wrong, and how does proper benchmarking prevent this?

### Question 6: Your Path Forward: From Learning to Production

You've completed an educational framework, but the patterns you learned apply directly to production systems.

**Translating TinyTorch Skills to Production:**

```
TinyTorch Pattern          →  Production Equivalent
─────────────────────────────────────────────────────
BenchmarkReport            →  MLflow Tracking
generate_submission()      →  Experiment logging
validate_schema()          →  JSON Schema / Protobuf
system_info collection     →  Environment containers (Docker)
baseline vs optimized      →  A/B testing framework
improvements calculation   →  Regression detection
```

**Real-World Applications:**

1. **Model Comparison** - Same workflow as Module 20, scaled to dozens of experiments
2. **Performance Monitoring** - Continuous benchmarking in CI/CD pipelines
3. **Reproducible Research** - Papers with Code submissions use similar schemas
4. **Team Collaboration** - Shared benchmark format enables comparison across engineers

**Next Steps for Production Systems:**

- **Scale beyond toy models** - Apply to CNNs, Transformers from milestones
- **Automated pipelines** - Trigger benchmarks on every commit (CI/CD)
- **Visualization dashboards** - Plot accuracy vs latency trade-off curves
- **Multi-hardware comparison** - Benchmark on CPU, GPU, TPU
- **Production monitoring** - Track deployed model performance over time

Congratulations! You've gone from implementing basic tensors to understanding end-to-end ML systems. The benchmarking methodology and systems thinking you learned here will serve you throughout your career in ML engineering. 🚀
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
"""
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

The complete journey:

```
Module 01: Tensor          -> Built foundation
Modules 02-13: Framework   -> Implemented ML components
Modules 14-18: Optimization -> Learned performance techniques
Module 19: Benchmarking    -> Measured performance
Module 20: Submission      -> Proved it works!
```

### Ready for Next Steps

You started Module 01 with a simple Tensor class. Now you have:
- A complete ML framework
- Advanced optimization techniques
- Professional benchmarking infrastructure
- Reproducible, shareable results

**You didn't just learn ML systems - you BUILT one from scratch.**

Export with: `tito module complete 20`

**Next**: The TorchPerf Olympics in `milestones/06_2018_mlperf/` pit your submission against everyone else's. Congratulations on completing TinyTorch!
"""
