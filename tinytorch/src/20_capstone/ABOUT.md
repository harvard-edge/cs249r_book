---
title: "Capstone - Submission Infrastructure"
description: "Build professional benchmarking workflows that generate standardized submissions for ML competitions"
difficulty: "●●●●"
time_estimate: "5-8 hours"
prerequisites: ["Module 19 - Benchmarking", "Modules 14-18 - Optimization Techniques"]
next_steps: ["Milestone 05 - TinyGPT", "Wake Vision Competition"]
learning_objectives:
  - "Apply Module 19's benchmarking tools to measure baseline and optimized model performance"
  - "Generate standardized JSON submissions following MLPerf-inspired industry formats"
  - "Calculate normalized improvement metrics (speedup, compression ratio, accuracy delta)"
  - "Execute complete optimization workflows integrating profiling, optimization, and benchmarking"
  - "Build submission infrastructure that enables future ML competitions and community challenges"
---

# Capstone - Submission Infrastructure

**OPTIMIZATION TIER CAPSTONE** | Difficulty: ●●●● (4/4) | Time: 5-8 hours

## Overview

Build professional submission infrastructure that brings together everything you've learned in the Optimization Tier. This capstone teaches you how to benchmark models using TinyTorch's optimization APIs (Modules 14-19), generate standardized JSON submissions, and create shareable results for ML competitions.

**What You Learn**: Complete optimization workflow from profiling to submission—how to use TinyTorch as a cohesive framework, not just individual modules. You'll apply the optimization pipeline (Profile → Optimize → Benchmark → Submit) to demonstrate your framework's capabilities with measurable, reproducible results.

**The Focus**: Using TinyTorch's APIs together in a professional workflow. This isn't about building new optimization techniques—it's about orchestrating existing tools to generate competition-ready submissions.

## Learning Objectives

By the end of this capstone, you will be able to:

- **Apply benchmarking tools systematically**: Use Module 19's `BenchmarkReport` class to measure model performance with statistical rigor
- **Generate standardized submissions**: Create MLPerf-inspired JSON submissions with system info, metrics, and reproducibility metadata
- **Calculate improvement metrics**: Compute normalized speedup, compression ratio, and accuracy delta for hardware-independent comparison
- **Execute optimization workflows**: Integrate profiling (M14), optimization techniques (M15-18), and benchmarking (M19) in complete pipeline
- **Build competition infrastructure**: Create submission formats that enable future challenges (Wake Vision, custom competitions)

## Build → Use → Reflect

This capstone follows TinyTorch's **Build → Use → Reflect** framework:

1. **Build**: Implement `BenchmarkReport` class, `generate_submission()` function, and standardized JSON schema with system metadata
2. **Use**: Benchmark baseline and optimized models using TinyTorch's optimization APIs; generate submissions comparing before/after performance
3. **Reflect**: How do optimization techniques combine in practice? What makes submissions reproducible? How does standardized infrastructure enable fair competition?

## Implementation Guide

### Core Infrastructure Components

The submission infrastructure implements three key systems:

#### 1. BenchmarkReport Class

**Comprehensive Performance Measurement**

The `BenchmarkReport` class encapsulates all metrics needed for competition submissions:

```python
class BenchmarkReport:
    """
    Benchmark report for model performance measurement.

    Collects and stores:
    - Model characteristics (parameters, size)
    - Performance metrics (accuracy, latency, throughput)
    - System information (hardware, software versions)
    - Timestamps for reproducibility
    """

    def __init__(self, model_name="model"):
        self.model_name = model_name
        self.metrics = {}
        self.system_info = self._get_system_info()
        self.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    def benchmark_model(self, model, X_test, y_test, num_runs=100):
        """
        Benchmark model with statistical rigor.

        Measures:
        - Parameter count and model size (MB)
        - Accuracy on test set
        - Latency with mean ± std (100 runs for statistics)
        - Throughput (samples/second)

        Returns:
            Dict with all metrics for submission generation
        """
        # Count parameters
        param_count = model.count_parameters()
        model_size_mb = (param_count * 4) / (1024 * 1024)  # FP32

        # Measure accuracy
        predictions = model.forward(X_test)
        pred_labels = np.argmax(predictions.data, axis=1)
        accuracy = np.mean(pred_labels == y_test)

        # Measure latency (statistical rigor with 100 runs)
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            _ = model.forward(X_test[:1])  # Single-sample inference
            latencies.append((time.time() - start) * 1000)  # ms

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        # Store comprehensive metrics
        self.metrics = {
            'parameter_count': int(param_count),
            'model_size_mb': float(model_size_mb),
            'accuracy': float(accuracy),
            'latency_ms_mean': float(avg_latency),
            'latency_ms_std': float(std_latency),
            'throughput_samples_per_sec': float(1000 / avg_latency)
        }

        return self.metrics
```

**Why This Design**: Single class captures all necessary information for reproducible benchmarking. System info ensures results can be contextualized. Multiple latency runs provide statistical confidence.

#### 2. Submission Generation Function

**Standardized JSON Schema Following MLPerf Format**

```python
def generate_submission(
    baseline_report: BenchmarkReport,
    optimized_report: Optional[BenchmarkReport] = None,
    student_name: Optional[str] = None,
    techniques_applied: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate standardized benchmark submission.

    Creates MLPerf-inspired JSON with:
    - Version and timestamp metadata
    - System information for reproducibility
    - Baseline performance metrics
    - Optional optimized metrics with techniques
    - Automatic improvement calculation

    Args:
        baseline_report: BenchmarkReport for baseline model
        optimized_report: Optional BenchmarkReport for optimized version
        student_name: Optional submitter name
        techniques_applied: List of optimization techniques used

    Returns:
        Dictionary ready for JSON serialization
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

    # Add optional student name
    if student_name:
        submission['student_name'] = student_name

    # Add optimization results if provided
    if optimized_report:
        submission['optimized'] = {
            'model_name': optimized_report.model_name,
            'metrics': optimized_report.metrics,
            'techniques_applied': techniques_applied or []
        }

        # Automatically calculate improvement metrics
        baseline_lat = baseline_report.metrics['latency_ms_mean']
        optimized_lat = optimized_report.metrics['latency_ms_mean']
        baseline_size = baseline_report.metrics['model_size_mb']
        optimized_size = optimized_report.metrics['model_size_mb']

        submission['improvements'] = {
            'speedup': float(baseline_lat / optimized_lat),
            'compression_ratio': float(baseline_size / optimized_size),
            'accuracy_delta': float(
                optimized_report.metrics['accuracy'] -
                baseline_report.metrics['accuracy']
            )
        }

    return submission

def save_submission(submission: Dict[str, Any], filepath: str):
    """Save submission to JSON file with proper formatting."""
    Path(filepath).write_text(json.dumps(submission, indent=2))
    print(f" Submission saved to: {filepath}")
    return filepath
```

**Submission Schema Example**:
```json
{
  "tinytorch_version": "0.1.0",
  "submission_type": "capstone_benchmark",
  "timestamp": "2025-01-15 14:23:41",
  "system_info": {
    "platform": "macOS-14.2-arm64",
    "python_version": "3.11.6",
    "numpy_version": "1.24.3"
  },
  "baseline": {
    "model_name": "baseline_mlp",
    "metrics": {
      "parameter_count": 263,
      "model_size_mb": 0.001,
      "accuracy": 0.35,
      "latency_ms_mean": 0.042,
      "latency_ms_std": 0.008,
      "throughput_samples_per_sec": 23809.52
    }
  },
  "optimized": {
    "model_name": "optimized_mlp",
    "metrics": {
      "parameter_count": 198,
      "model_size_mb": 0.00075,
      "accuracy": 0.33,
      "latency_ms_mean": 0.031,
      "latency_ms_std": 0.006,
      "throughput_samples_per_sec": 32258.06
    },
    "techniques_applied": ["pruning", "quantization"]
  },
  "improvements": {
    "speedup": 1.35,
    "compression_ratio": 1.33,
    "accuracy_delta": -0.02
  }
}
```

**Why This Schema**: MLPerf-inspired format ensures reproducibility. System info enables verification. Normalized metrics (speedup, compression ratio) work across hardware. Automatic improvement calculation prevents manual errors.

#### 3. Complete Optimization Workflow

**Bringing Modules 14-19 Together**

This workflow demonstrates the full optimization pipeline:

```python
def run_optimization_workflow_example():
    """
    Complete optimization workflow using Modules 14-19.

    Pipeline:
    1. Profile baseline (Module 14)
    2. Apply optimizations (Modules 15-18)
    3. Benchmark with rigor (Module 19)
    4. Generate submission (Module 20)

    This is the COMPLETE story of TinyTorch optimization!
    """
    print("="*70)
    print("TINYTORCH OPTIMIZATION WORKFLOW")
    print("="*70)

    # Import optimization APIs
    from tinytorch.perf.profiling import Profiler, quick_profile
    from tinytorch.perf.compression import magnitude_prune
    from tinytorch.benchmarking import Benchmark, BenchmarkResult

    # Step 1: Profile baseline model (Module 14)
    print("\n[STEP 1] Profile Baseline - Module 14")
    baseline_model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)

    profiler = Profiler()
    # Optional: Use Module 14's profiler for detailed analysis
    # profile_data = quick_profile(baseline_model, input_tensor)

    # Step 2: Benchmark baseline (Module 19)
    print("\n[STEP 2] Benchmark Baseline - Module 19")
    baseline_report = BenchmarkReport(model_name="baseline_mlp")
    baseline_report.benchmark_model(baseline_model, X_test, y_test, num_runs=50)

    # Step 3: Apply optimizations (Modules 15-18)
    print("\n[STEP 3] Apply Optimizations - Modules 15-18")
    print("  Available APIs:")
    print("    - Module 15: quantize_model(model, bits=8)")
    print("    - Module 16: magnitude_prune(model, sparsity=0.5)")
    print("    - Module 17: enable_kv_cache(model)  # For transformers")
    print("    - Module 18: Use accelerated ops")

    # Example: Apply pruning (students can add quantization, etc.)
    optimized_model = baseline_model  # Apply real optimizations here
    # optimized_model = magnitude_prune(baseline_model, sparsity=0.3)
    # optimized_model = quantize_model(optimized_model, bits=8)

    # Step 4: Benchmark optimized version (Module 19)
    print("\n[STEP 4] Benchmark Optimized - Module 19")
    optimized_report = BenchmarkReport(model_name="optimized_mlp")
    optimized_report.benchmark_model(optimized_model, X_test, y_test, num_runs=50)

    # Step 5: Generate submission (Module 20)
    print("\n[STEP 5] Generate Submission - Module 20")
    submission = generate_submission(
        baseline_report=baseline_report,
        optimized_report=optimized_report,
        student_name="TinyTorch Optimizer",
        techniques_applied=["pruning", "quantization"]
    )

    # Display improvements
    if 'improvements' in submission:
        imp = submission['improvements']
        print(f"\n   Results:")
        print(f"     Speedup: {imp['speedup']:.2f}x")
        print(f"     Compression: {imp['compression_ratio']:.2f}x")
        print(f"     Accuracy Δ: {imp['accuracy_delta']*100:+.1f}%")

    # Step 6: Save submission
    save_submission(submission, "optimization_submission.json")

    print("\n Complete optimization workflow demonstrated!")
    return submission
```

**Why This Workflow Matters**: Shows how TinyTorch modules work together as a cohesive framework. Students see the complete optimization story: measure → optimize → validate → submit. This workflow pattern applies to real production ML perf.

### Connection to TinyTorch Optimization Tier

This capstone brings together the entire Optimization Tier:

**The Complete Optimization Story**:
```
Module 14 (Profiling)
    ↓
  Identify bottlenecks
    ↓
Modules 15-18 (Optimization Techniques)
    ↓
  Apply targeted optimizations
    ↓
Module 19 (Benchmarking)
    ↓
  Measure improvements with statistics
    ↓
Module 20 (Submission)
    ↓
  Package results for sharing
```

**How Modules Work Together**:

1. **Module 14 (Profiling)**: Identifies bottlenecks (memory-bound vs compute-bound)
2. **Module 15 (Quantization)**: Reduces precision to save memory and improve throughput
3. **Module 16 (Compression)**: Prunes parameters to reduce model size
4. **Module 17 (Memoization)**: Caches computations to avoid redundant work
5. **Module 18 (Acceleration)**: Applies operator fusion and vectorization
6. **Module 19 (Benchmarking)**: Validates optimizations with statistical rigor
7. **Module 20 (Submission)**: Packages everything into shareable format

**Real-World Application**: This workflow mirrors how production ML teams optimize models:
- Google TPU teams profile → optimize → benchmark → deploy
- OpenAI profiles GPT training → applies gradient checkpointing → validates memory savings
- Meta benchmarks PyTorch inference → fuses operators → measures latency improvements

### Enabling Future Competitions

The submission infrastructure you build enables future TinyTorch challenges:

**Wake Vision Competition (Coming Soon)**:
- Optimize computer vision models for edge deployment
- Constraints: Latency < 100ms, Model size < 5MB, Accuracy > 85%
- Rankings based on normalized submissions (same format you're building)

**Custom Competitions**:
- Educational settings: Classroom competitions using TinyTorch
- Research benchmarks: Reproducible optimization studies
- Community challenges: Open-source ML optimization contests

**Extensibility**: The submission format you implement can be extended with:
- Additional metrics (energy consumption, memory bandwidth)
- Constraint validation (checking competition requirements)
- Leaderboard integration (automated ranking systems)

## Getting Started

### Prerequisites

Ensure you understand benchmarking and optimization techniques:

```bash
# Activate TinyTorch environment
source scripts/activate-tinytorch

# Required: Benchmarking methodology (Module 19)
tito test benchmarking

# Helpful: Optimization techniques (Modules 14-18)
tito test profiling        # Module 14: Find bottlenecks
tito test quantization     # Module 15: Reduce precision
tito test compression      # Module 16: Prune parameters
tito test memoization      # Module 17: Cache computations
tito test acceleration     # Module 18: Operator fusion
```

**Why Module 19 is Essential**: This capstone uses Module 19's `BenchmarkReport` class as the foundation. Understanding statistical measurement methodology from Module 19 is critical for generating valid submissions.

### Development Workflow

1. **Open the development file**: `modules/20_capstone/20_capstone.py`
2. **Implement SimpleMLP**: Simple demonstration model for benchmarking
3. **Build BenchmarkReport**: Class to collect and store metrics
4. **Create generate_submission()**: Function to create standardized JSON
5. **Add save_submission()**: JSON serialization with proper formatting
6. **Implement workflow examples**: Basic and optimization workflow demonstrations
7. **Export and verify**: `tito module complete 20 && tito test capstone`

**Development Tips**:
```python
# Test BenchmarkReport with toy model
model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
report = BenchmarkReport(model_name="test_model")
metrics = report.benchmark_model(model, X_test, y_test, num_runs=10)

# Verify all required metrics are present
required = ['parameter_count', 'model_size_mb', 'accuracy',
            'latency_ms_mean', 'latency_ms_std', 'throughput_samples_per_sec']
assert all(metric in metrics for metric in required)

# Test submission generation
submission = generate_submission(report, student_name="Test")
assert 'baseline' in submission
assert 'system_info' in submission
assert submission['submission_type'] == 'capstone_benchmark'

# Test with optimization comparison
optimized_report = BenchmarkReport(model_name="optimized")
# ... benchmark optimized model ...
submission_with_opt = generate_submission(report, optimized_report,
                                         techniques_applied=["pruning"])
assert 'improvements' in submission_with_opt
assert 'speedup' in submission_with_opt['improvements']
```

## Testing

### Comprehensive Test Suite

Run the full test suite to verify submission infrastructure:

```bash
# TinyTorch CLI (recommended)
tito test capstone

# Direct pytest execution
python -m pytest tests/ -k capstone -v

# Expected output:
#  test_simple_mlp - Model creation and forward pass
#  test_benchmark_report - Metrics collection and storage
#  test_submission_generation - JSON creation
#  test_submission_schema - Schema validation
#  test_submission_with_optimization - Before/after comparison
#  test_improvements_calculation - Speedup/compression/accuracy
#  test_json_serialization - File saving and loading
```

### Test Coverage Areas

- ✓ **SimpleMLP Model**: Forward pass, parameter counting, output shape validation
- ✓ **BenchmarkReport**: Metric collection, system info capture, statistical measurement
- ✓ **Submission Generation**: Schema structure, field presence, type validation
- ✓ **Schema Validation**: Required fields, value ranges, type correctness
- ✓ **Optimization Comparison**: Improvements calculation, technique tracking
- ✓ **JSON Serialization**: File writing, round-trip preservation, formatting

### Inline Testing & Validation

The module includes comprehensive unit tests:

```python
 Unit Test: SimpleMLP...
 Model creation with custom parameters
 Parameter count: 263 (10×20 + 20 + 20×3 + 3)
 Forward pass output shape: (5, 3)
 No NaN values in output
 Progress: SimpleMLP ✓

 Unit Test: BenchmarkReport...
 Model name and timestamp set correctly
 System info collected (platform, python_version, numpy_version)
 Metrics: parameter_count, model_size_mb, accuracy, latency, throughput
 Metric types and ranges validated
 Progress: BenchmarkReport ✓

 Unit Test: Submission Generation...
 Baseline submission structure complete
 Version, type, timestamp, system_info, baseline present
 Student name included when provided
 Progress: generate_submission() ✓

 Unit Test: Submission Schema...
 Required fields present
 Field types correct (str, dict, float, int)
 Baseline and metrics structure validated
 System info contains platform and python_version
 Progress: Schema validation ✓

 Unit Test: Submission with Optimization...
 Optimized section present with techniques
 Improvements section with speedup, compression, accuracy_delta
 Techniques list matches input
 Progress: Optimization comparison ✓

 Unit Test: Improvements Calculation...
 Speedup: 2.0x (baseline 10.0ms / optimized 5.0ms)
 Compression: 2.0x (baseline 4.0MB / optimized 2.0MB)
 Accuracy delta: -0.05 (0.75 - 0.80)
 Progress: Improvements math ✓

 Unit Test: JSON Serialization...
 File created and exists
 JSON valid and loadable
 Structure preserved (version, student_name, metrics)
 Round-trip serialization successful
 Progress: File I/O ✓
```

### Manual Testing Examples

```python
from tinytorch.capstone import SimpleMLP, BenchmarkReport, generate_submission, save_submission
from tinytorch.core.tensor import Tensor
import numpy as np

# Example 1: Basic benchmark workflow
np.random.seed(42)
X_test = Tensor(np.random.randn(100, 10))
y_test = np.random.randint(0, 3, 100)

model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
report = BenchmarkReport(model_name="simple_mlp")
metrics = report.benchmark_model(model, X_test, y_test, num_runs=50)

submission = generate_submission(report, student_name="Your Name")
save_submission(submission, "my_submission.json")

# Example 2: Optimization comparison workflow
baseline_model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
baseline_report = BenchmarkReport(model_name="baseline")
baseline_report.benchmark_model(baseline_model, X_test, y_test, num_runs=50)

# Apply optimizations (example: smaller model)
optimized_model = SimpleMLP(input_size=10, hidden_size=15, output_size=3)
optimized_report = BenchmarkReport(model_name="optimized")
optimized_report.benchmark_model(optimized_model, X_test, y_test, num_runs=50)

# Generate comparison submission
submission = generate_submission(
    baseline_report=baseline_report,
    optimized_report=optimized_report,
    student_name="Optimizer",
    techniques_applied=["architecture_search", "pruning"]
)

print(f"Speedup: {submission['improvements']['speedup']:.2f}x")
print(f"Compression: {submission['improvements']['compression_ratio']:.2f}x")
print(f"Accuracy Δ: {submission['improvements']['accuracy_delta']*100:+.1f}%")

save_submission(submission, "optimization_comparison.json")
```

## Systems Thinking Questions

### Optimization Workflow Integration

**Question 1: Optimization Interaction**

You apply INT8 quantization (4× memory reduction) followed by 75% magnitude pruning (4× parameter reduction). Should you expect 16× total memory reduction?

**Reflection Structure**:
- Quantization affects: Precision per parameter (FP32 → INT8 = 4 bytes → 1 byte)
- Pruning affects: Parameter count (75% zeroed out)
- Combined effect: Depends on sparse storage format
- Why not multiplicative: Dense storage still allocates space for zeros

**Systems Insight**: Quantization reduces bits per parameter. Pruning zeros out weights but doesn't automatically reduce memory in dense format. For true 16× reduction, you need sparse storage (CSR/COO format) that doesn't allocate space for zeros. This is why Module 16 teaches both pruning AND sparse representations.

### Submission Reproducibility

**Question 2: System Information Requirements**

Why does the submission schema require `system_info` with platform, Python version, and NumPy version? What breaks if this is omitted?

**Systems Insight**: Reproducibility requires environment specification. NumPy 1.24 vs 2.0 can produce different results due to algorithm changes. Platform affects performance (ARM vs x86, SIMD instruction sets). Python version impacts library behavior. Without system info, results aren't verifiable—claims of "2× speedup" are meaningless if hardware isn't specified. Production ML teams learned this the hard way when "optimizations" only worked on specific configurations.

### Statistical Measurement Validity

**Question 3: Measurement Rigor**

Your optimized model shows 5% latency improvement with standard deviation of 8%. Is this a real improvement or measurement noise?

**Reflection Points**:
- Mean improvement: 5% faster
- Standard deviation: 8% of baseline latency
- Confidence interval: Likely overlapping
- Statistical significance: Requires hypothesis testing

**Systems Insight**: When std > improvement magnitude, difference could be noise. Proper approach: run t-test with p < 0.05 threshold. Module 19's benchmarking teaches this—multiple runs + confidence intervals prevent false claims. Production teams don't deploy "optimizations" without statistical confidence because regressions cost money.

### Workflow Scalability

**Question 4: Production Scaling**

How does this submission workflow scale to production models with millions of parameters and hours-long training runs?

**Reflection**:
- SimpleMLP benchmarks in milliseconds → GPT-2 trains for days
- Toy dataset (100 samples) → Production (billions of tokens)
- Single metric focus → Multi-objective optimization (latency + memory + throughput + cost)

**Systems Insight**: The workflow patterns are identical—profile, optimize, benchmark, submit—but tools must scale. Production uses distributed profiling (across GPUs/nodes), long-running benchmarks (days not minutes), and comprehensive metrics (MLPerf includes 20+ metrics). TinyTorch teaches the workflow; PyTorch provides production-scale infrastructure.

### Competition Fairness

**Question 5: Normalized Metrics Design**

Why use speedup ratios (baseline_time / optimized_time) instead of absolute times (10ms → 5ms) for competition ranking?

**Systems Insight**: Hardware variability makes absolute times meaningless for comparison. M1 Mac vs Intel i9 vs AMD Threadripper all produce different absolute times. But 2× speedup is meaningful across hardware—same relative improvement. Speedup ratios enable fair comparison and focus on optimization quality, not hardware access. MLPerf competitions use normalized metrics for exactly this reason.

## Ready to Complete the Optimization Tier?

You've reached the capstone of TinyTorch's Optimization Tier. This submission infrastructure brings together everything from Modules 14-19, transforming individual optimization techniques into a cohesive workflow that mirrors production ML engineering.

**What You'll Achieve**:
- Complete optimization workflow: Profile → Optimize → Benchmark → Submit
- Professional submission infrastructure enabling future competitions
- Understanding of how TinyTorch modules work together as a framework
- Reproducible, shareable results demonstrating your optimization skills

**The Capstone Mindset**:
> "Individual modules teach techniques. The capstone teaches workflow. Production ML isn't about knowing tools—it's about orchestrating them effectively."
> — Every ML systems engineer

**What's Next**:
- **Milestone 05**: Build TinyGPT using your complete framework
- **Wake Vision Competition**: Apply optimization skills to real challenges
- **Community Sharing**: Submit your results, compare with others

This capstone demonstrates you don't just understand optimization techniques—you can apply them systematically to produce measurable, reproducible improvements. That's the difference between knowing tools and being an ML systems engineer.

Choose your preferred way to engage with this capstone:

````{grid} 1 2 3 3

```{grid-item-card}  Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/20_capstone/20_capstone.py
:class-header: bg-light

Run this capstone interactively in your browser. No installation required!
```

```{grid-item-card}  Open in Colab
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/20_capstone/20_capstone.ipynb
:class-header: bg-light

Use Google Colab for cloud compute power and easy sharing.
```

```{grid-item-card}  View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/20_capstone/20_capstone.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition}  Local Development Recommended
:class: tip
This capstone involves benchmarking workflows that benefit from consistent hardware and persistent results. Local setup provides better control over measurement conditions and faster iteration cycles.

**Setup**: `git clone https://github.com/mlsysbook/TinyTorch.git && source scripts/activate-tinytorch && cd modules/20_capstone`
```

---

<div class="prev-next-area">
<a class="left-prev" href="19_benchmarking_ABOUT.html" title="previous page">← Module 19: Benchmarking</a>
</div>
