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
1. Use Module 19's benchmarking tools to measure model performance comprehensively
2. Apply optimization techniques from Modules 14-18 to improve baseline models
3. Generate standardized JSON submissions following industry best practices
4. Validate submissions against a schema for reproducibility
5. Compare baseline vs. optimized models with quantitative metrics
6. Share your results with the TinyTorch community in a professional format

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `src/20_capstone/20_capstone.py`
**Building Side:** Code exports to `tinytorch.olympics`

```python
# Final package structure:
from tinytorch.olympics import generate_submission, BenchmarkReport

# Benchmark your model
report = BenchmarkReport()
report.benchmark_model(my_model, X_test, y_test)

# Generate submission
submission = generate_submission(report)
submission.save("my_submission.json")
```

**Why this matters:**
- **Learning:** Complete workflow from model to shareable results
- **Production:** Professional submission format mirroring MLPerf and Papers with Code standards
- **Community:** Share and compare results with other builders using standardized metrics
- **Reproducibility:** Schema-validated submissions ensure results can be verified and trusted
"""

# %% nbgrader={"grade": false, "grade_id": "exports", "solution": true}
#| default_exp olympics
#| export

# %% [markdown]
"""
## 📋 Module Dependencies

**Prerequisites**: Modules 01-19 must be complete

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for latency measurements)
- `json` (for submission serialization)
- `pathlib` (for file path handling)
- `platform` (for system information)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class from Module 01)
- `tinytorch.core.layers` (Linear layer from Module 03)
- `tinytorch.core.activations` (ReLU from Module 02)
- `tinytorch.core.losses` (CrossEntropyLoss from Module 04)
- Optimization modules 14-18 (optional, for advanced workflows)

**Dependency Flow**:
```
Modules 01-13 → Modules 14-18 → Module 19 → Module 20 (Capstone)
(Framework)     (Optimization)   (Benchmark)  (Submission)
```

Students completing this module will demonstrate their complete framework's capabilities through reproducible benchmarking and professional submission generation.
"""

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

Let's build a benchmarking and submission system worthy of production ML!
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| export
import numpy as np
rng = np.random.default_rng(7)
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import platform
import sys

# %% nbgrader={"grade": false, "grade_id": "imports2", "solution": false}
# Import TinyTorch modules (not exported - used for module development only)
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU
from tinytorch.core.losses import CrossEntropyLoss

if __name__ == "__main__":
    print("✅ Capstone modules imported!")
    print("📊 Ready to benchmark and submit results")

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

### Why Variance Matters

Single measurements lie. Variance tells the truth:

```
Why We Report Mean ± Std:

Measurement 1: 9.2ms    ┐
Measurement 2: 10.1ms   │ Mean = 10.0ms
Measurement 3: 9.8ms    │ Std  = 0.5ms
Measurement 4: 10.5ms   │
Measurement 5: 9.4ms    ┘

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
        4. Initialize weights with small random values (scale 0.01)
        5. Initialize biases to zeros

        HINTS:
        - Use Linear(in_features, out_features) for layers
        - Weight shape is (in_features, out_features)
        - Small initial weights (0.01 scale) help training stability
        """
        ### BEGIN SOLUTION
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)

        # Initialize with small random weights
        # Linear layer expects weight shape: (in_features, out_features)
        self.fc1.weight.data = rng.standard_normal((input_size, hidden_size)) * 0.01
        self.fc1.bias.data = np.zeros(hidden_size)
        self.fc2.weight.data = rng.standard_normal((hidden_size, output_size)) * 0.01
        self.fc2.bias.data = np.zeros(output_size)
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
        """Return model parameters for perf."""
        return [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]

    def count_parameters(self):
        """Count total number of parameters."""
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total

if __name__ == "__main__":
    print("✅ SimpleMLP model defined")

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
## 🏗️ Implementation: Benchmark Report Class

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
        4. Latency (mean ± std) - Inference speed and consistency
        5. Throughput - Maximum samples/second capacity
        """
        # Count parameters
        param_count = model.count_parameters()
        model_size_mb = (param_count * 4) / (1024 * 1024)  # Assuming FP32

        # Measure accuracy
        predictions = model.forward(X_test)
        pred_labels = np.argmax(predictions.data, axis=1)
        accuracy = np.mean(pred_labels == y_test)

        # Measure latency (average over multiple runs)
        # Why multiple runs? See "Variance" section in Foundations
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            _ = model.forward(X_test[:1])  # Single sample inference
            latencies.append((time.time() - start) * 1000)  # Convert to ms

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        # Store metrics (all as Python native types for JSON serialization)
        self.metrics = {
            'parameter_count': int(param_count),
            'model_size_mb': float(model_size_mb),
            'accuracy': float(accuracy),
            'latency_ms_mean': float(avg_latency),
            'latency_ms_std': float(std_latency),
            'throughput_samples_per_sec': float(1000 / avg_latency)
        }

        print(f"\n📊 Benchmark Results for {self.model_name}:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Size: {model_size_mb:.2f} MB")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Latency: {avg_latency:.2f}ms ± {std_latency:.2f}ms")

        return self.metrics

    def measure_latency(self, model, X_sample, num_runs=100):
        """
        Measure inference latency over multiple runs.

        TODO: Time single-sample inference over multiple runs

        APPROACH:
        1. Run inference num_runs times
        2. Measure each run with time.time()
        3. Convert to milliseconds
        4. Return list of latencies

        HINTS:
        - Use time.time() before and after model.forward()
        - Multiply by 1000 to convert seconds to milliseconds
        - Use X_sample[:1] for single-sample timing
        """
        ### BEGIN SOLUTION
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            _ = model.forward(X_sample[:1])
            latencies.append((time.time() - start) * 1000)
        return latencies
        ### END SOLUTION

    def measure_memory(self, model):
        """
        Measure model memory footprint.

        TODO: Calculate model size in MB assuming FP32 weights

        APPROACH:
        1. Count total parameters
        2. Multiply by 4 bytes (FP32)
        3. Convert to MB (divide by 1024*1024)

        HINTS:
        - model.count_parameters() gives total param count
        - FP32 = 4 bytes per parameter
        - 1 MB = 1024 * 1024 bytes
        """
        ### BEGIN SOLUTION
        param_count = model.count_parameters()
        return (param_count * 4) / (1024 * 1024)
        ### END SOLUTION

if __name__ == "__main__":
    print("✅ BenchmarkReport class defined")

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

### Design Choice: Why num_runs=100?

We run inference 100 times by default to:
- **Warm up** the system (first runs are often slower)
- **Capture variance** (some runs hit cache, others miss)
- **Average out noise** (OS interrupts, GC pauses)
- **Get confidence intervals** (via std dev)

```
Single Run (Unreliable):        Multiple Runs (Reliable):
┌─────────────────────────┐     ┌─────────────────────────┐
│ Run 1: 12.3ms           │     │ Run 1: 12.3ms           │
│                         │     │ Run 2: 9.8ms            │
│ Result: 12.3ms          │     │ Run 3: 10.1ms           │
│ Confidence: Low         │     │ ...                     │
│ (Could be outlier!)     │     │ Run 100: 10.2ms         │
│                         │     │                         │
│                         │     │ Result: 10.0ms ± 0.5ms  │
│                         │     │ Confidence: High        │
│                         │     │ (Statistically sound)   │
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
## 🏗️ Implementation: Submission Generation

The core function that generates a standardized JSON submission from benchmark results.

**Design Goals:**
1. **Baseline-first** - Always require baseline results (comparison reference)
2. **Optimization optional** - Support baseline-only OR baseline+optimized submissions
3. **Auto-calculate improvements** - Automatically compute speedup, compression, accuracy delta
4. **Schema compliance** - Generate structure that passes validation
5. **Extensible** - Easy to add new fields without breaking existing code
"""

# %% nbgrader={"grade": false, "grade_id": "generate-submission", "solution": true}
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
        baseline_latency = baseline_report.metrics['latency_ms_mean']
        optimized_latency = optimized_report.metrics['latency_ms_mean']
        baseline_size = baseline_report.metrics['model_size_mb']
        optimized_size = optimized_report.metrics['model_size_mb']

        submission['improvements'] = {
            'speedup': float(baseline_latency / optimized_latency),
            'compression_ratio': float(baseline_size / optimized_size),
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

if __name__ == "__main__":
    print("✅ Submission generation functions defined")

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
## 🔧 Integration: Complete Example Workflow

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

# %% nbgrader={"grade": false, "grade_id": "example-workflow", "solution": true}
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
    rng = np.random.default_rng(7)
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
    print("✅ Example workflow defined")

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
## 🔧 Integration: Advanced Optimization Workflow

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
- How to measure cumulative impact (2× from quant + 1.5× from pruning = 3× total)
- How to document techniques for reproducibility
"""

# %% nbgrader={"grade": false, "grade_id": "optimization-workflow", "solution": true}
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
    print("\nThis workflow demonstrates using Modules 14-19 together:")
    print("  📊 Module 14: Profiling")
    print("  🔢 Module 15: Quantization (optional - API imported for demonstration)")
    print("  ✂️  Module 16: Compression (optional - API imported for demonstration)")
    print("  ⚡ Module 17: Acceleration (optional - API imported for demonstration)")
    print("  💾 Module 18: Memoization (optional - API imported for demonstration)")
    print("  📈 Module 19: Benchmarking")
    print("  📝 Module 20: Submission Generation")

    # Demonstrate API imports (students can use these for their own optimizations)
    print("\n🔧 Importing optimization APIs...")
    try:
        from tinytorch.perf.profiling import Profiler, quick_profile
        print("  ✅ Module 14 (Profiling) imported")
    except ImportError:
        print("  ⚠️  Module 14 (Profiling) not available - using basic profiling")
        Profiler = None

    try:
        from tinytorch.perf.compression import magnitude_prune, structured_prune
        print("  ✅ Module 16 (Compression) imported")
    except ImportError:
        print("  ⚠️  Module 16 (Compression) not available - skipping pruning demo")
        magnitude_prune = None

    try:
        from tinytorch.perf.benchmarking import BenchmarkSuite, BenchmarkResult
        print("  ✅ Module 19 (Benchmarking) imported")
    except ImportError:
        print("  ⚠️  Module 19 (Benchmarking) not available - using basic benchmarking")
        BenchmarkSuite = None

    # Step 1: Create dataset
    print("\n" + "="*70)
    print("STEP 1: Create Test Dataset")
    print("="*70)
    rng = np.random.default_rng(7)
    X_test = Tensor(rng.standard_normal((100, 10)))
    y_test = rng.integers(0, 3, 100)
    print(f"  Dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features, 3 classes")

    # Step 2: Create and profile baseline model
    print("\n" + "="*70)
    print("STEP 2: Baseline Model - Profile & Benchmark")
    print("="*70)
    baseline_model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    print(f"  Model: {baseline_model.count_parameters():,} parameters")

    # Benchmark baseline using BenchmarkReport
    baseline_report = BenchmarkReport(model_name="baseline_mlp")
    baseline_metrics = baseline_report.benchmark_model(baseline_model, X_test, y_test, num_runs=50)

    # Optional: Demonstrate using Module 14's Profiler if available
    if Profiler:
        print("\n  📊 Optional: Using Module 14's Profiler for detailed analysis...")
        profiler = Profiler()
        # Note: Profiler integration would go here
        # This demonstrates the API is available for students to use

    # Step 3: (DEMO ONLY) Show optimization APIs available
    print("\n" + "="*70)
    print("STEP 3: Optimization APIs Available (Demo)")
    print("="*70)
    print("\n  📚 Students can apply these optimizations:")
    print("     - Module 15: quantize_model(model, bits=8)")
    print("     - Module 16: magnitude_prune(model, sparsity=0.5)")
    print("     - Module 17: Use accelerated ops (vectorized_matmul, etc.)")
    print("     - Module 18: enable_kv_cache(model)  # For transformers")
    print("\n  💡 For this demo, we'll simulate an optimized model")
    print("     (Students can replace this with real optimizations!)")

    # Create "optimized" model (students would apply real optimizations here)
    optimized_model = SimpleMLP(input_size=10, hidden_size=15, output_size=3)  # Smaller for demo
    optimized_report = BenchmarkReport(model_name="optimized_mlp")
    optimized_metrics = optimized_report.benchmark_model(optimized_model, X_test, y_test, num_runs=50)

    # Step 4: Generate submission with before/after comparison
    print("\n" + "="*70)
    print("STEP 4: Generate Submission with Improvements")
    print("="*70)

    ### BEGIN SOLUTION
    submission = generate_submission(
        baseline_report=baseline_report,
        optimized_report=optimized_report,
        student_name="TinyTorch Optimizer",
        techniques_applied=["model_sizing", "architecture_search"]  # Students list real techniques
    )
    ### END SOLUTION

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
    print("  - Apply real optimizations (quantization, pruning, etc.)")
    print("  - Benchmark milestone models (XOR, MNIST, CNN, etc.)")
    print("  - Share your optimized results with the community!")

    return submission

if __name__ == "__main__":
    print("✅ Optimization workflow example defined")

# %% [markdown]
"""
### Combining Multiple Optimizations

In production ML, you often stack optimizations for cumulative benefits:

```
Stacking Optimizations:
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
## 🧪 Unit Tests

Individual unit tests for each component, following TinyTorch testing patterns.

**Testing Strategy:**
1. **Unit tests** - Test each class/function in isolation
2. **Integration test** - Test complete workflow end-to-end (in test_module)
3. **Schema validation** - Ensure submissions conform to standard
4. **Edge cases** - Test with missing optional fields, extreme values

Each test validates one specific aspect and provides clear feedback.
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

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_simple_mlp()

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

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_benchmark_report()

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

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_submission_generation()

# %% [markdown]
"""
### 🧪 Unit Test: Schema Validation

This test validates submissions conform to the required schema.

**What we're testing**: Required fields, type safety, value constraints
**Why it matters**: Schema validation enables automated aggregation and comparison
**Expected**: Valid submissions pass, invalid submissions fail with clear errors
"""

# %% nbgrader={"grade": true, "grade_id": "test-submission-schema", "locked": true, "points": 10}
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

    # Check baseline structure
    baseline = submission['baseline']
    assert 'model_name' in baseline, "Baseline missing model_name"
    assert 'metrics' in baseline, "Baseline missing metrics"

    # Check metrics structure and types
    metrics = baseline['metrics']
    required_metrics = ['parameter_count', 'model_size_mb', 'accuracy', 'latency_ms_mean']
    for metric in required_metrics:
        if metric not in metrics:
            raise AssertionError(f"Missing metric in baseline: {metric}")

    # Check metric value ranges
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy must be in [0, 1]"
    assert metrics['parameter_count'] > 0, "Parameter count must be positive"
    assert metrics['model_size_mb'] > 0, "Model size must be positive"
    assert metrics['latency_ms_mean'] > 0, "Latency must be positive"

    # Check system info
    system_info = submission['system_info']
    assert 'platform' in system_info, "System info missing platform"
    assert 'python_version' in system_info, "System info missing python_version"

    return True

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

# Run test immediately when developing
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

# Run test immediately when developing
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

# Run test immediately when developing
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

# Run test immediately when developing
if __name__ == "__main__":
    test_unit_json_serialization()

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

    print("✅ Complete workflow works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 20")

# Run comprehensive module test
if __name__ == "__main__":
    test_module()

# %% [markdown]
"""
## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of benchmarking, reproducibility, and ML systems integration:

### Reflecting on the Complete ML Systems Journey

You've built an entire ML framework across 20 modules. This capstone asks you to step back and reflect on the complete systems journey—from tensors to production-ready benchmarking.

### End-to-End System Integration

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

### Benchmarking Methodology: Science Meets Engineering

Effective benchmarking requires rigorous methodology that bridges scientific measurement with engineering pragmatism.

**The Three Pillars of Reliable Benchmarking:**

```
1. REPEATABILITY (Same Experiment → Same Result)
   ┌─────────────────────────────────────────┐
   │ • Fixed random seeds (default_rng)       │
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

### Performance Measurement Traps and How to Avoid Them

Real-world benchmarking is full of subtle traps that can invalidate your measurements.

**Common Measurement Pitfalls:**

```
TRAP 1: Measuring the Wrong Thing
  ❌ Timing model creation instead of inference
  ❌ Including data loading in latency measurement
  ❌ Measuring batch=32 when production uses batch=1

  ✅ FIX: Isolate exactly what you're measuring
     start = time.time()
     output = model.forward(x)  # ONLY this
     latency = time.time() - start

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

### Schema Validation: Making Results Machine-Readable

Your submission format uses JSON Schema validation—a powerful pattern for ensuring data quality and enabling automation.

**Why Schema Validation Matters:**

```
WITHOUT Schema:                     WITH Schema:
┌──────────────────────────┐       ┌──────────────────────────┐
│ {                        │       │ {                        │
│   "accuracy": "92%",     │ ❌    │   "accuracy": 0.92,      │ ✅
│   "latency": 10.5,       │ ❌    │   "latency_ms_mean": 10.5│ ✅
│   "time": "today"        │ ❌    │   "timestamp": "2025..." │ ✅
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

### The Complete ML Systems Lifecycle

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

### Your Path Forward: From Learning to Production

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
## 🚀 MODULE SUMMARY: Capstone - Benchmarking & Submission

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

### The Complete TinyTorch Journey

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

**Congratulations on completing TinyTorch!**
"""
