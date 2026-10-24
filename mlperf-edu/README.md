# MLPerf EDU

A benchmark suite for machine learning systems that runs on standard laptops and workstations without compromising on the quality gate.

Production benchmark suites teach essential empirical discipline. They hold the task, dataset, metric, and quality target fixed so that different systems, compilers, and hardware backends can be fairly compared. However, production suites typically demand multi-node GPU clusters and complex submission pipelines, putting them out of reach for courses and lightweight local experimentation. The common alternative—informal folders of toy models—solves accessibility at the cost of quietly discarding rigorous quality floors.

MLPerf EDU preserves both accessibility and rigor. Fourteen workloads run locally across CPU, Apple Silicon (MPS), and NVIDIA CUDA, and **no quality target is set arbitrarily by this project**. Every workload inherits its quality target from an upstream paper, official leaderboard, or reference evaluator.

> **Note:** MLPerf EDU is an independent research project and working name. It is not an official MLCommons benchmark and is not endorsed by MLCommons.

---

## Quickstart

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). An accelerator (Apple Silicon GPU or NVIDIA CUDA) is supported automatically when present; otherwise, CPU execution is used.

```bash
# 1. Enter the directory and sync dependencies
cd mlperf-edu
uv sync --locked --extra dev

# 2. Inspect your local environment, hardware, and registry
uv run mlperf doctor

# 3. Run the four-workload starter smoke test
uv run mlperf init --profile min --output-dir submissions/first-run

# 4. View the generated report
uv run mlperf report submissions/first-run --format html --open
```

`doctor` checks your Python interpreter, PyTorch installation, available accelerators (MPS/CUDA/CPU), caches, and workload registry.

`init` runs the representative four-workload `min` profile (image classification, language modeling, graph node classification, and time-series forecasting), verifies SHA-256 provenance manifests, and generates HTML, JSON, and CSV reports.

Next, run a full quality-gated evaluation:

```bash
# Fetch assets before timing
uv run mlperf fetch --workload image-classification --profile max

# Execute the quality-bearing benchmark
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/image-review
```

This runs the official MLPerf Tiny ResNet8 on CIFAR-10 and grades accuracy against the inherited 85.0% top-1 target in seconds.

---

## Core Concepts

Understanding three concepts makes navigating the suite straightforward:

### 1. Workload vs. Configuration
A workload represents a fixed task and evaluation contract (e.g., `causal-language-modeling`), not a specific hardware or software configuration. Execution modes (`training` vs. `inference`), inference phases (`prefill` vs. `decode`), quantization, batch sizes, and compilation strategies are recorded as execution configurations in the result report.

### 2. Execution Profiles
Profiles define the purpose and contract depth of a run:

| Profile | Purpose | Contract & Claims |
|:---|:---|:---|
| `min` | Install check, sanity verification, CI | Fast functional path (default runs 4 starter workloads; `--collection all` runs all 14). Never score-bearing. |
| `max` | Authoritative reference path | Executes against the inherited quality target on local hardware. |
| `pro` | Research and ablation envelope | Exposes controlled knobs (batch size, precision, backend, compiler flags) via YAML experiment plans. |

### 3. Cryptographic Provenance
Every benchmark execution generates:
- A detailed structured report (`*_report.json`)
- Aggregate HTML and CSV summaries
- A cryptographic provenance manifest (`*.provd.json`) binding code, model weights, dataset partitions, and hardware telemetry by SHA-256

Verify any run manifest with:
```bash
uv run mlperf verify submissions/image-review/image-classification_max.provd.json
```

---

## Workload Catalog

MLPerf EDU packages fourteen workloads organized across seven domain suites. Each workload targets a distinct ML systems bottleneck:

### 1. Language Suite (`language`)

* **`causal-language-modeling`**
  * **Model & Data:** `nanogpt-shakespeare-char` (10.77M params) on Tiny Shakespeare.
  * **Modes:** `training` (pinned 5,001-step training loop) and `inference` (`full`, `prefill`, `decode` phases).
  * **Inherited Gate:** Validation cross-entropy loss ≤ 1.4697 (from nanoGPT).
  * **Systems Lens:** Pinned training exercises compute-bound matrix multiplications (GEMMs). Autoregressive inference isolates compute-bound prompt evaluation (`prefill` / TTFT) from memory-bandwidth-bound token generation (`decode` / ITL) and demonstrates KV-cache mechanics.

* **`text-classification`**
  * **Model & Data:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`, 66.96M params) on SST-2 (Stanford Sentiment Treebank).
  * **Modes:** `inference`.
  * **Inherited Gate:** Accuracy ≥ 0.91055 (from model card).
  * **Systems Lens:** Standard bidirectional transformer encoder inference, batch sizing trade-offs, and tokenization overhead.

* **`information-retrieval`**
  * **Model & Data:** Cross-Encoder MiniLM (`cross-encoder/ms-marco-MiniLM-L6-v2`) on NanoBEIR reranking benchmark.
  * **Modes:** `inference`.
  * **Inherited Gate:** Mean nDCG@10 ≥ 0.60717 (from BEIR benchmark).
  * **Systems Lens:** Pairwise multi-document cross-encoding, query-document concatenation, and high-throughput batch scoring.

* **`code-generation`**
  * **Model & Data:** `Qwen/Qwen2.5-Coder-0.5B-Instruct` on HumanEval+.
  * **Modes:** `inference`.
  * **Inherited Gate:** pass@1 ≥ 0.573 (from model card · **recorded miss**).
  * **Systems Lens:** Long-context autoregressive code generation, token decoding latency, and unit-test execution verification.

* **`function-calling`**
  * **Model & Data:** `Qwen/Qwen3-1.7B` on Berkeley Function Calling Leaderboard (BFCL V4) non-live AST.
  * **Modes:** `inference`.
  * **Inherited Gate:** AST accuracy ≥ 82.92% (from BFCL · **recorded miss**).
  * **Systems Lens:** Structured JSON decoding, tool-use schema parsing, and grammar-constrained decoding efficiency.

### 2. Vision Suite (`vision`)

* **`image-classification`**
  * **Model & Data:** ResNet8 (MLPerf Tiny v1.0 reference) on CIFAR-10 (200-sample official evaluation set).
  * **Modes:** `inference`.
  * **Inherited Gate:** Top-1 accuracy ≥ 0.850 (from MLPerf Tiny).
  * **Systems Lens:** 2D convolutions, residual skip connections, and compute-bound GEMM/im2col on compact activations.

* **`image-generation`**
  * **Model & Data:** NVIDIA EDM (Elucidated Diffusion Model) on CIFAR-10.
  * **Modes:** `inference`.
  * **Inherited Gate:** FID ≤ 1.79 (from NVIDIA EDM paper · **recorded miss**).
  * **Systems Lens:** Iterative multi-step score-based denoising loops, latency vs. sample quality, and activation memory footprints.

### 3. Tiny / Edge Suite (`tiny`)

* **`keyword-spotting`**
  * **Model & Data:** DS-CNN (Depthwise Separable CNN) on EEMBC / Speech Commands (1,000 1-second clips).
  * **Modes:** `inference`.
  * **Inherited Gate:** Top-1 accuracy ≥ 0.900 (from MLPerf Tiny).
  * **Systems Lens:** Depthwise separable convolutions (memory-bound spatial filter + compute-bound 1×1 pointwise filter) and audio MFCC feature extraction pipelines.

* **`visual-wake-words`**
  * **Model & Data:** MobileNetV1 0.25 on Visual Wake Words / COCO (1,000 images).
  * **Modes:** `inference`.
  * **Inherited Gate:** Top-1 accuracy ≥ 0.800 (from MLPerf Tiny).
  * **Systems Lens:** Ultra-constrained vision for microcontrollers, strict SRAM/Flash capacity limits, and integer quantization.

* **`anomaly-detection`**
  * **Model & Data:** ToyCar Deep Autoencoder on ToyADMOS machine operating sounds (248 recordings).
  * **Modes:** `inference`.
  * **Inherited Gate:** ROC AUC ≥ 0.850 (from MLPerf Tiny).
  * **Systems Lens:** Fully connected layer inference, spectral feature pipelines, and reconstruction error loss calculation.

### 4. Graph Suite (`graph`)

* **`graph-node-classification`**
  * **Model & Data:** OGB GCN (2-layer Graph Convolutional Network via PyG) on `ogbn-arxiv` (169,343 nodes, 1.16M citation edges).
  * **Modes:** `training`.
  * **Inherited Gate:** Test accuracy ≥ 0.7174 (from official OGB benchmark).
  * **Systems Lens:** Sparse matrix multiplication (SpMM), irregular memory access patterns, and graph neighbor aggregation.

### 5. Time-Series Suite (`timeseries`)

* **`time-series-forecasting`**
  * **Model & Data:** PatchTST on ETTm1 (Electricity Transformer Temperature, 15-minute intervals).
  * **Modes:** `training`.
  * **Inherited Gate:** Test MSE ≤ 0.290 (from PatchTST paper · **recorded miss**).
  * **Systems Lens:** Channel-independent patching, multivariate sequence forecasting, and temporal attention mechanisms.

### 6. Recommendation Suite (`recommendation`)

* **`recommendation`**
  * **Model & Data:** NCF (Neural Collaborative Filtering: Generalized Matrix Factorization + MLP) on MovieLens-20M.
  * **Modes:** `training`.
  * **Inherited Gate:** Hit Rate @ 10 (HR@10) ≥ 0.635 (from MLPerf Training v0.5 · **recorded miss**).
  * **Systems Lens:** High-cardinality sparse embedding table lookups, memory-bandwidth bottleneck, non-uniform memory access, and embedding cache misses.

### 7. Reinforcement Learning Suite (`reinforcement`)

* **`reinforcement-learning`**
  * **Model & Data:** MiniGo (AlphaGo Zero dual policy-value convolutional architecture) on self-play game records.
  * **Modes:** `training`.
  * **Inherited Gate:** Professional move prediction accuracy ≥ 0.400 (from MLPerf Training v0.5 · **recorded miss**).
  * **Systems Lens:** Monte Carlo Tree Search (MCTS) coupled with asynchronous neural network evaluation and self-play data generation.

---

## Workload Summary & Evidence Scope

| Workload | Suite | Model | Dataset | Mode | Inherited Gate | Status |
|:---|:---|:---|:---|:---|:---|:---|
| `image-classification` | vision | ResNet8 | CIFAR-10 (200) | inference | Top-1 ≥ 0.850 | Pass |
| `keyword-spotting` | tiny | DS-CNN | EEMBC Audio (1,000) | inference | Top-1 ≥ 0.900 | Pass |
| `visual-wake-words` | tiny | MobileNetV1 0.25 | VWW / COCO (1,000) | inference | Top-1 ≥ 0.800 | Pass |
| `anomaly-detection` | tiny | ToyCar Autoencoder | ToyADMOS (248) | inference | ROC AUC ≥ 0.850 | Pass |
| `causal-language-modeling` | language | nanoGPT (10.77M) | Tiny Shakespeare | training / inference | Val Loss ≤ 1.4697 | Pass |
| `text-classification` | language | DistilBERT (67M) | SST-2 (872) | inference | Accuracy ≥ 0.91055 | Pass |
| `information-retrieval` | language | CrossEncoder MiniLM | NanoBEIR Reranking | inference | Mean nDCG@10 ≥ 0.60717 | Pass |
| `graph-node-classification` | graph | OGB GCN | ogbn-arxiv | training | Test Acc ≥ 0.7174 | Pass |
| `time-series-forecasting` | timeseries | PatchTST | ETTm1 | training | Test MSE ≤ 0.290 | Recorded miss |
| `recommendation` | recommendation | NCF (GMF + MLP) | MovieLens-20M | training | HR@10 ≥ 0.635 | Recorded miss |
| `code-generation` | language | Qwen2.5-Coder-0.5B | HumanEval+ | inference | pass@1 ≥ 0.573 | Recorded miss |
| `function-calling` | language | Qwen3-1.7B | BFCL V4 non-live AST | inference | AST Acc ≥ 82.92% | Recorded miss |
| `image-generation` | vision | NVIDIA EDM | CIFAR-10 | inference | FID ≤ 1.79 | Recorded miss |
| `reinforcement-learning` | reinforcement | MiniGo Policy/Value | Self-play games | training | Move Pred ≥ 0.400 | Recorded miss |

### The Value of Recorded Misses
In MLPerf EDU, missed targets are explicitly recorded rather than silently relaxed or adjusted. Each miss marks an interesting systems boundary where running a published recipe on local consumer hardware requires targeted architectural, hyperparameter, or numerical adjustments. Detailed investigations of each shortfall live in [`docs/internal/MISS_DIAGNOSIS.md`](docs/internal/MISS_DIAGNOSIS.md).

The committed reference evidence covers nine workloads across twelve evidence cases (treating training and the `full`, `prefill`, and `decode` inference phases of `causal-language-modeling` as distinct evidence cases). The remaining five workloads run end-to-end at a functional stage.

---

## Clean CLI Command Reference

All operations run through the `mlperf` CLI via `uv run mlperf`.

### Discovery & Inspection

```bash
# List all 14 workloads
uv run mlperf list

# List available profiles (min, max, pro)
uv run mlperf list profiles

# Inspect a specific workload contract and quality target
uv run mlperf show image-classification
uv run mlperf info --workload causal-language-modeling

# Inspect workload prerequisites and device compatibility
uv run mlperf doctor --workload graph-node-classification
```

### Fetching Assets Before Measurement

Downloading assets beforehand ensures timing measurements include zero network latency:

```bash
# Preview required downloads without downloading
uv run mlperf fetch --workload image-classification --profile max --dry-run

# Download and verify assets for a workload
uv run mlperf fetch --workload image-classification --profile max

# Download assets for an entire suite
uv run mlperf fetch --suite tiny --profile max
```

### Running Benchmarks

```bash
# Run the fast 4-workload starter smoke test
uv run mlperf run --profile min --output-dir submissions/smoke

# Run all 14 workloads in min profile (complete health check)
uv run mlperf health
# or explicitly:
uv run mlperf run --profile min --collection all --output-dir submissions/health-check

# Run a single workload in max profile
uv run mlperf run --workload image-classification --profile max --output-dir submissions/img-cls

# Run an entire suite in max profile
uv run mlperf run --suite tiny --profile max --output-dir submissions/tiny-suite

# Run training vs. inference modes
uv run mlperf run --workload causal-language-modeling --mode training --profile max --output-dir submissions/clm-train

# Run specific inference phases (full, prefill, or decode)
uv run mlperf run --workload causal-language-modeling --mode inference --phase decode --profile max --output-dir submissions/clm-decode

# Collect hardware power and energy telemetry (macOS powermetrics or NVIDIA NVML)
uv run mlperf run --workload keyword-spotting --profile max --power --output-dir submissions/kws-power
```

### Viewing and Verifying Results

```bash
# Print a clean terminal summary table
uv run mlperf report submissions/first-run

# Open the interactive HTML dashboard in your browser
uv run mlperf report submissions/first-run --format html --open

# Verify SHA-256 provenance manifest
uv run mlperf verify submissions/first-run/image-classification_min.provd.json
```

---

## Building the Research Paper

The companion research paper explains the design, methodology, and empirical evaluation of MLPerf EDU. It is written in LaTeX and located in [`paper/`](paper/).

The PDF is compiled from source and automatically validated against the live benchmark registry:

```bash
# Build the paper and run automated checks
uv run make -C paper clean all check
```

The compiled PDF is written to:
```text
paper/output/pdf/mlperf-edu-review-draft.pdf
```

### How the Paper Build Works
1. `generate_registry_snapshot.py` queries the live workload registry (`registry/` and `workloads.yaml`) and provisional results to export synchronized LaTeX tables (`generated_registry.tex`, `generated_dam_table.tex`, `generated_appendix_data.tex`).
2. `pdflatex` and `bibtex` compile the 16-page document with full figures and citations.
3. `check_paper_pdf.py` performs automated quality verification on the compiled PDF, ensuring exact page count (16 pages), valid citation linkages, and zero unresolved placeholders or overflow boxes.

---

## Repository Structure

```text
mlperf-edu/
├── bench/                 # Benchmark runners and execution harnesses
├── conformance_results/   # Conformance run artifacts
├── data/                  # Local dataset cache (populated by fetch)
├── datasets.yaml          # Dataset catalog with upstream URLs and checksums
├── docs/                  # In-depth guides, design docs, and internal notes
├── examples/              # Research plans and classroom lab exercises
├── paper/                 # Companion research paper LaTeX source and build scripts
│   ├── figures/           # Rendered paper figures
│   ├── output/pdf/        # Generated paper PDF (mlperf-edu-review-draft.pdf)
│   ├── Makefile           # Paper compilation pipeline
│   └── paper.tex          # Paper LaTeX source
├── provisional_results/   # Reference evaluation results and evidence index
├── registry/              # Authoritative YAML workload definitions
├── site/                  # Quarto source for https://mlsysbook.ai/mlperf-edu
├── src/mlperf/            # MLPerf EDU Python core package and CLI implementation
├── tests/                 # Unit and integration test suite
├── tools/                 # Validation, documentation, and auditing utilities
├── tutorials/             # Interactive Marimo and Jupyter notebook tutorials
├── workloads.yaml         # Consolidated workload registry mirror
└── pyproject.toml         # Packaging and dependency declarations
```

---

## Further Reading

- [Installation Guide](INSTALL.md) — Wheel builds, offline setups, and pip options
- [Specification](SPEC.md) — Benchmark specification and contract definition
- [Public Result Rules](PUBLIC_RULES.md) — Disclosure and claims policy
- [Security Review](SECURITY_REVIEW.md) — Sandboxing and network isolation boundaries
- [Miss Diagnosis](docs/internal/MISS_DIAGNOSIS.md) — Technical post-mortems on the recorded misses
- [Quality Target Review](docs/internal/QUALITY_TARGET_REVIEW.md) — Detailed rationale for each inherited target
