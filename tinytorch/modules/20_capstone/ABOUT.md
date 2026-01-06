---
title: "Torch Olympics - ML Systems Competition"
description: "Learn competition workflow: use Benchmark harness to measure performance and generate standardized submissions"
difficulty: "⭐⭐⭐⭐"
time_estimate: "5-8 hours"
prerequisites: ["Benchmarking (Module 19)", "Optimization techniques (Modules 14-18)"]
next_steps: []
learning_objectives:
  - "Understand competition events: Know how different Olympic events (Latency Sprint, Memory Challenge, All-Around) have different constraints and optimization strategies"
  - "Use Benchmark harness: Apply Module 19's Benchmark class to measure performance with statistical rigor (confidence intervals, multiple runs)"
  - "Generate submissions: Create standardized submission formats following MLPerf-style industry standards"
  - "Validate submissions: Check that submissions meet event constraints (accuracy thresholds, latency limits) and flag unrealistic improvements"
  - "Workflow integration: Understand how benchmarking tools (Module 19) and optimization techniques (Modules 14-18) work together in competition context"
---

# MLPerf® Edu Competition

```{admonition} Try it Now
:class: tip

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?labpath=tinytorch%2Fmodules%2F20_capstone%2F20_capstone.ipynb)

Open this module in your browser. No installation required.
```

**CAPSTONE PROJECT** | Difficulty: ⭐⭐⭐⭐ (4/4) | Time: 5-8 hours

## Overview

The TinyTorch Olympics capstone teaches you how to participate in professional ML competitions. You've learned benchmarking methodology in Module 19—now apply those tools in a competition workflow. This module focuses on understanding competition events, using the Benchmark harness to measure performance, generating standardized submissions, and validating results meet competition requirements.

**What You Learn**: Competition workflow and submission packaging—how to use benchmarking tools (Module 19) and optimization techniques (Modules 14-18) to create competition-ready submissions following industry standards (MLPerf-style).

**The Focus**: Understanding how professional ML competitions work—from measurement to submission—not building TinyGPT (that's Milestone 05).

## Learning Objectives

By the end of this capstone, you will be able to:

- **Understand Competition Events**: Know how different Olympic events (Latency Sprint, Memory Challenge, All-Around) have different constraints and optimization strategies
- **Use Benchmark Harness**: Apply Module 19's Benchmark class to measure performance with statistical rigor (confidence intervals, multiple runs)
- **Generate Submissions**: Create standardized submission formats following MLPerf-style industry standards
- **Validate Submissions**: Check that submissions meet event constraints (accuracy thresholds, latency limits) and flag unrealistic improvements
- **Workflow Integration**: Understand how benchmarking tools (Module 19) and optimization techniques (Modules 14-18) work together in competition context

## The Five Olympic Events

Choose your competition event based on optimization goals:

### 🏃 Event 1: Latency Sprint
**Objective**: Minimize inference latency
**Constraints**: Accuracy ≥ 85%
**Strategy Focus**: Operator fusion, quantization, efficient data flow
**Winner**: Fastest average inference time (with confidence intervals)

### 🏋️ Event 2: Memory Challenge
**Objective**: Minimize model memory footprint
**Constraints**: Accuracy ≥ 85%
**Strategy Focus**: Quantization, pruning, weight sharing
**Winner**: Smallest model size maintaining accuracy

### 🎯 Event 3: Accuracy Contest
**Objective**: Maximize model accuracy
**Constraints**: Latency < 100ms, Memory < 10MB
**Strategy Focus**: Balanced optimization, selective precision
**Winner**: Highest accuracy within constraints

### 🏋️‍♂️ Event 4: All-Around
**Objective**: Best balanced performance
**Scoring**: Composite score across latency, memory, accuracy
**Strategy Focus**: Multi-objective optimization, Pareto efficiency
**Winner**: Highest composite score

### 🚀 Event 5: Extreme Push
**Objective**: Most aggressive optimization
**Constraints**: Accuracy ≥ 80% (lower threshold)
**Strategy Focus**: Maximum compression, aggressive quantization
**Winner**: Best compression-latency product

## Competition Workflow

This module teaches the workflow of professional ML competitions. You'll learn how to use benchmarking tools (Module 19) to measure performance and generate standardized submissions.

### Stage 1: Understand Competition Events

Different Olympic events have different constraints and optimization strategies:

```python
from tinytorch.competition import OlympicEvent

# Event types
event = OlympicEvent.LATENCY_SPRINT      # Minimize latency, accuracy ≥ 85%
event = OlympicEvent.MEMORY_CHALLENGE   # Minimize memory, accuracy ≥ 85%
event = OlympicEvent.ALL_AROUND         # Best balanced performance
event = OlympicEvent.EXTREME_PUSH       # Most aggressive, accuracy ≥ 80%
```

**Event Constraints:**
- **Latency Sprint**: Accuracy ≥ 85%, optimize for speed
- **Memory Challenge**: Accuracy ≥ 85%, optimize for size
- **All-Around**: Balanced optimization across metrics
- **Extreme Push**: Accuracy ≥ 80%, maximum optimization

### Stage 2: Measure Baseline Performance

Use Module 19's Benchmark harness to measure baseline:

```python
from tinytorch.benchmarking import Benchmark

# Measure baseline performance
benchmark = Benchmark([baseline_model], [test_data], ["latency", "memory", "accuracy"])
baseline_results = benchmark.run()

# Results include statistical rigor (confidence intervals)
print(f"Baseline - Latency: {baseline_results['latency'].mean:.2f}ms")
print(f"  95% CI: [{baseline_results['latency'].ci_lower:.2f}, {baseline_results['latency'].ci_upper:.2f}]")
print(f"Baseline - Memory: {baseline_results['memory'].mean:.2f}MB")
print(f"Baseline - Accuracy: {baseline_results['accuracy'].mean:.2%}")
```

**Key Insight**: Module 19 provides statistical rigor—multiple runs, confidence intervals, warmup periods. This ensures fair comparison.

### Stage 3: Measure Optimized Performance

Apply optimization techniques (from Modules 14-18), then measure:

```python
# Apply optimizations (using techniques from Modules 14-18)
optimized_model = apply_optimizations(baseline_model)

# Measure optimized performance with same Benchmark harness
optimized_results = benchmark.run()  # Same benchmark, different model
```

**Fair Comparison**: Same Benchmark harness, same test data, same hardware—ensures apples-to-apples comparison.

### Stage 4: Calculate Normalized Scores

Compute hardware-independent metrics:

```python
from tinytorch.competition import calculate_normalized_scores

# Convert to normalized scores (hardware-independent)
scores = calculate_normalized_scores(
    baseline_results={'latency': 100.0, 'memory': 12.0, 'accuracy': 0.85},
    optimized_results={'latency': 40.0, 'memory': 3.0, 'accuracy': 0.83}
)

# Results: speedup=2.5×, compression_ratio=4.0×, accuracy_delta=-0.02
print(f"Speedup: {scores['speedup']:.2f}×")
print(f"Compression: {scores['compression_ratio']:.2f}×")
print(f"Accuracy change: {scores['accuracy_delta']:+.2%}")
```

**Why Normalized**: Speedup ratios work on any hardware. "2.5× faster" is meaningful whether you have M1 Mac or Intel i9.

### Stage 5: Generate Submission

Create standardized submission following MLPerf-style format:

```python
from tinytorch.competition import generate_submission, validate_submission

# Generate submission
submission = generate_submission(
    baseline_results=baseline_results,
    optimized_results=optimized_results,
    event=OlympicEvent.LATENCY_SPRINT,
    athlete_name="YourName",
    github_repo="https://github.com/yourname/tinytorch",
    techniques=["INT8 Quantization", "70% Pruning", "KV Cache"]
)

# Validate submission meets requirements
validation = validate_submission(submission)
if validation['valid']:
    print("✅ Submission valid!")
    print(f"   Checks passed: {len([c for c in validation['checks'] if c['passed']])}")
else:
    print("❌ Submission invalid:")
    for issue in validation['issues']:
        print(f"   - {issue}")

# Save submission
import json
with open('submission.json', 'w') as f:
    json.dump(submission, f, indent=2)
```

**Submission Format**: Includes normalized scores, system info, event constraints, statistical confidence—everything needed for fair competition ranking.

## Getting Started

### Prerequisites

This capstone requires understanding of benchmarking (Module 19) and optimization techniques (Modules 14-18):

```bash
# Activate TinyTorch environment
source bin/activate-tinytorch.sh

# Required: Benchmarking methodology (Module 19)
tito test --module benchmarking     # Module 19: Statistical measurement, fair comparison

# Helpful: Optimization techniques (Modules 14-18)
tito test --module profiling        # Module 14: Find bottlenecks
tito test --module quantization     # Module 15: Reduce precision
tito test --module compression      # Module 16: Prune parameters
tito test --module memoization      # Module 17: Cache computations
tito test --module acceleration     # Module 18: Operator fusion
```

**Why You Need Module 19:**
- Module 19 teaches benchmarking methodology (statistical rigor, fair comparison)
- Module 20 teaches how to use Benchmark harness in competition workflow
- You use Benchmark class from Module 19 to measure performance

**The Focus**: Understanding competition workflow—how to use benchmarking tools to generate submissions—not building models from scratch (that's Milestones 05-06).

### Development Workflow

1. **Understand Competition Events** (`Stage 1`):
   - Review OlympicEvent enum and event constraints
   - Understand how different events require different strategies
   - Learn event-specific accuracy thresholds

2. **Measure Baseline** (`Stage 2`):
   - Use Benchmark harness from Module 19 to measure baseline performance
   - Understand statistical rigor (confidence intervals, multiple runs)
   - Learn fair comparison protocols

3. **Measure Optimized** (`Stage 3`):
   - Apply optimization techniques (from Modules 14-18)
   - Use same Benchmark harness to measure optimized performance
   - Ensure fair comparison (same data, hardware, methodology)

4. **Calculate Normalized Scores** (`Stage 4`):
   - Compute hardware-independent metrics (speedup, compression ratio)
   - Understand why normalized scores enable fair comparison
   - Learn how to combine multiple metrics

5. **Generate Submission** (`Stage 5`):
   - Create standardized submission format (MLPerf-style)
   - Validate submission meets event constraints
   - Understand submission structure and requirements

6. **Export and verify**:
   ```bash
   tito module complete 20
   tito test --module capstone
   ```

## Testing

### Comprehensive Test Suite

Run the full test suite to verify your competition submission:

```bash
# TinyTorch CLI (recommended)
tito test --module capstone

# Direct pytest execution
python -m pytest tests/ -k capstone -v

# Expected output:
# ✅ test_baseline_establishment - Verifies baseline measurement
# ✅ test_optimization_pipeline - Tests combined optimizations
# ✅ test_event_constraints - Validates constraint satisfaction
# ✅ test_statistical_significance - Ensures improvements are real
# ✅ test_submission_generation - Verifies report creation
```

### Test Coverage Areas

- ✅ **OlympicEvent Enum**: Event types and constraints work correctly
- ✅ **Normalized Scoring**: Speedup and compression ratios calculated correctly
- ✅ **Submission Generation**: Creates valid MLPerf-style submissions
- ✅ **Submission Validation**: Checks event constraints and flags issues
- ✅ **Workflow Integration**: Complete workflow demonstration executes

## Systems Thinking Questions

### Integration Complexity

**Question 1: Optimization Interaction**
You apply INT8 quantization (4× memory reduction) followed by 75% pruning (4× parameter reduction). Should you expect 16× total memory reduction?

**Answer Structure:**
- Quantization affects: _____
- Pruning affects: _____
- Combined effect: _____
- Why not multiplicative: _____

**Systems Insight**: Quantization reduces bits per parameter (4 bytes → 1 byte). Pruning reduces parameter count (but zero values still stored in dense format). Combined effect depends on sparse matrix representation. For true 16× reduction, need sparse storage format that doesn't store zeros.

### Measurement Validity

**Question 2: Statistical Significance**
Your optimized model shows 5% latency improvement with p-value = 0.12. Competitor shows 8% improvement with p-value = 0.02. Who wins?

**Systems Insight**: With p=0.12, your 5% could be noise (not statistically significant at α=0.05). Competitor's 8% with p=0.02 is significant. Always report p-values—bigger speedup doesn't mean better if not statistically valid!

### Event Strategy

**Question 3: All-Around Optimization**
For All-Around event, should you: (a) Optimize each metric separately, then combine? (b) Optimize all metrics simultaneously from start?

**Systems Insight**: Simultaneous optimization risks sub-optimal trade-offs. Better strategy: (1) Profile to find bottlenecks, (2) Apply technique targeting worst metric, (3) Re-measure all metrics, (4) Repeat. Iterative refinement with full measurement prevents over-optimization of one metric at expense of others.

### Production Relevance

**Question 4: Real-World Connection**
How does Torch Olympics competition preparation translate to production ML systems work?

**Reflection**: Production deployment requires the exact skills you're practicing: profiling to find bottlenecks, applying targeted optimizations, validating improvements statistically, balancing trade-offs based on constraints (latency SLA, memory budget, accuracy requirements), and documenting decisions. The Olympic events mirror real scenarios: mobile deployment (Memory Challenge), real-time inference (Latency Sprint), high-accuracy requirements (Accuracy Contest).

## Ready for Competition?

This capstone teaches you how professional ML competitions work. You've learned benchmarking methodology in Module 19—now understand how to use those tools in a competition workflow. Module 20 focuses on:

- **Competition Workflow**: How to participate in ML competitions (MLPerf-style)
- **Submission Packaging**: How to format results for fair comparison and validation
- **Event Understanding**: How different events require different optimization strategies
- **Workflow Integration**: How benchmarking tools (Module 19) + optimization techniques (Modules 14-18) work together

**What's Next**:
- Build TinyGPT in Milestone 05 (historical achievement)
- Compete in Torch Olympics (Milestone 06) using this workflow
- Check your status with `tito olympics status`!

This module teaches workflow and packaging—you use existing tools, not rebuild them. The competition workflow demonstrates how professional ML competitions are structured and participated in.

Choose your preferred way to engage with this capstone:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/harvard-edge/cs249r_book/main?filepath=modules/20_capstone/capstone_dev.ipynb
:class-header: bg-light

Run this capstone interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab
:link: https://colab.research.google.com/github/harvard-edge/cs249r_book/blob/main/modules/20_capstone/capstone_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/harvard-edge/cs249r_book/blob/main/modules/20_capstone/capstone.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💡 Competition Recommendation
:class: tip
**Local development recommended!** This capstone involves extended optimization experiments, profiling sessions, and benchmarking runs. Local setup provides better debugging, faster iteration, and persistent results. Cloud sessions may timeout during long benchmark runs.

**Setup**: `git clone https://github.com/harvard-edge/cs249r_book.git && source bin/activate-tinytorch.sh && cd modules/20_capstone`
```

---

<div class="prev-next-area">
<a class="left-prev" href="../19_benchmarking/ABOUT.html" title="previous page">← Module 19: Benchmarking</a>
</div>
