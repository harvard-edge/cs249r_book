# MLPerf EDU — Getting Started Guide

## Setup

```bash
# Clone the repository
git clone https://github.com/harvard-edge/mlperf-edu.git
cd mlperf-edu

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start: Your First Benchmark

### 1. Train NanoGPT (5 minutes)

```bash
mlperf run cloud --task nanogpt-12m
```

This trains an 85.9M-parameter GPT-2 variant on TinyShakespeare. You'll see:
- Training loss converging from ~4.3 to ~2.25
- Inference latency measured at the end
- A JSON submission file saved to `submissions/`

### 2. Generate a Report

```bash
mlperf report --submission submissions/<your_file>.json
```

Open the generated HTML report in your browser. It shows:
- Metrics summary (loss, latency, throughput)
- Hardware fingerprint (for auditability)
- Convergence behavior
- SHA-256 hashes (anti-tampering)

### 3. Run All Workloads

```bash
mlperf train --all          # Train all 16 workloads
mlperf train --division cloud   # Just the cloud suite
```

## Lab Structure

### Lab 1: Training Optimization (Closed Division)

**Goal**: Reduce ResNet-18 training time by 20% without dropping below the quality target.

```bash
# Baseline run
mlperf run edge --task resnet18

# Your optimized run
python examples/lab1_optimization.py
```

**What you'll learn**:
- Batch size vs. convergence tradeoffs
- Data loading bottlenecks (num_workers)
- Learning rate scheduling

### Lab 2: Inference Architecture (Open Division)

**Goal**: Build a System Under Test (SUT) that handles the load generator's query stream.

```bash
python examples/lab2_inference_sut.py
```

**What you'll learn**:
- Latency percentiles (p50/p90/p99)
- Throughput vs. latency tradeoffs
- Batching strategies

### Lab 3: Architecture Comparison

**Goal**: Compare dense (NanoGPT) vs. sparse (Nano-MoE) architectures.

```bash
python examples/lab3_arch_comparison.py
```

**What you'll learn**:
- Expert specialization in MoE
- Routing overhead vs. quality improvement
- Parameter efficiency

## Declarative Interface (YAML)

```yaml
# experiment.yaml
workload: nanogpt-12m      # S.Model
dataset: tinyshakespeare    # S.Data
target_quality: 2.3         # S.Constraints
epochs: 25                  # S.Constraints
```

```bash
mlperf config experiment.yaml
```

## Available Workloads

| Division | Workload | Time | Key Concept |
|----------|----------|------|-------------|
| Cloud | NanoGPT | 89s | O(N²) attention scaling |
| Cloud | Nano-MoE | 158s | Conditional compute |
| Cloud | DLRM | 5s | Sparse vs. dense memory |
| Cloud | Diffusion | 41s | Denoising step count |
| Cloud | GCN | 2s | Message passing |
| Cloud | BERT | 45s | Bidirectional attention |
| Cloud | LSTM | 20s | Sequential bottleneck |
| Cloud | RL | 1s | Policy gradient variance |
| Edge | ResNet-18 | 64s | Skip connections + batch norm |
| Edge | MobileNetV2 | 60s | Depthwise-sep. convolutions |
| Tiny | DS-CNN | 51s | Spectrogram features |
| Tiny | Anomaly AE | 6s | Reconstruction error |
| Tiny | VWW | 10s | Sub-10K model compression |

## Submission & Grading

After each run, the harness produces a JSON submission:

```bash
# Verify your submission
mlperf verify --submission submissions/your_run.json

# Generate a grading artifact (for TAs)
mlperf submit
```

## Need Help?

- `mlperf about` — Architecture overview
- `mlperf list` — All available workloads
- `mlperf --help` — Full CLI reference
