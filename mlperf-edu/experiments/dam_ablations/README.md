# DAM Taxonomy Experimental Ablations Suite (MLPerf EDU)

This directory contains standalone, non-invasive experimental scripts evaluating the **DAM Taxonomy** (Data, Algorithm, Machine) across MLPerf EDU benchmark workloads.

> **Immutability Invariant:** These experiment scripts import the reference benchmark models and dataset loaders from `src/mlperf` in **read-only** mode. The original benchmark source code (`src/`), registry contracts (`registry/`), and reference evidence (`provisional_results/`) are strictly untouched and unmodified.

---

## 🔬 Taxonomy Dimensions & Experiment Breakdown

### 1. Data Ablations (D)
* **Dataset Sample Budget Scaling:** Evaluates model performance, training latency, and Fail-Closed semantic target admission under sample budget constraints ($100\%, 50\%, 25\%, 10\%$).
* **Target Metric:** Time-to-accuracy trade-offs against inherited quality gates.

### 2. Algorithm Ablations (A)
* **Quantization & Precision Sweeps:** Evaluates FP32, FP16, and INT8 precision formats across vision (ResNet8) and text (DistilBERT) models.
* **Target Metrics:** Weight footprint (MB), inference throughput (samples/sec), DRAM memory bandwidth demand (GB/s), and task accuracy delta.

### 3. Machine Ablations (M)
* **Hardware Backend Execution (CPU vs. MPS GPU):** Compares execution latency, kernel dispatch overhead, and speedup ratios across Apple Silicon CPU cores and Metal Performance Shaders (MPS) GPU acceleration.
* **Target Metrics:** GEMM speedup vs. irregular/sparse memory access bottlenecks (gather/scatter and warp divergence).

---

## 🚀 Execution Instructions

Run all DAM ablation experiments using `uv run`:

```bash
# Execute full DAM ablation suite
uv run python experiments/dam_ablations/run_dam_experiments.py
```

Results are exported to `experiments/dam_ablations/results/`:
* `dam_ablation_results.json` — Structured JSON execution records with device fingerprints and SHA256 digests.
* `dam_summary.csv` — Scannable CSV table of all DAM experiment metrics.
