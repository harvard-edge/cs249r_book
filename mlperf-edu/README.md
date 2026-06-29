<!-- MLPERF-EDU-STATUS:START -->
> [!WARNING]
> **🚧 Under construction**
>
> This tree is a runnable preview, not a stable public benchmark release. The `mlperf` CLI, registry, reports, and validation paths work locally, but public-result policy, dataset approval, and MLCommons endorsement review are still in progress. **Do not rely on it for production benchmarking** until we publish a stable "1.0" teaching release.

> [!NOTE]
> **📌 Early work (2026)**
>
> MLPerf EDU is being developed in public alongside the **2026** MLSysBook ecosystem. Harness scripts, compliance checks, and teaching notes will keep moving as we align workloads with the core curriculum.
>
> **Feedback** — [GitHub issues](https://github.com/harvard-edge/cs249r_book/issues) or pull requests (especially if something in this README is wrong or outdated).
<!-- MLPERF-EDU-STATUS:END -->

# MLPerf EDU 🎓

**A 30-workload pedagogical ML systems benchmark registry with runnable `min`/`max` coverage and a `pro` research envelope, aligned with [MLCommons MLPerf](https://mlcommons.org/benchmarks/).**

MLPerf EDU brings industry-standard ML benchmarking into teaching and research. The core teaching models are self-contained, white-box PyTorch `nn.Module` implementations, while the SLM suite uses off-the-shelf Hugging Face models for local serving, quantization, LoRA, and backend studies.

📄 **Paper**: See [`paper/paper.tex`](paper/paper.tex) — "MLPerf EDU: Bridging Industry Benchmarking and ML Systems Education"

🧭 **North star**: See [`NORTH_STAR.md`](NORTH_STAR.md) for the two-year goal: MLPerf EDU as the SPEC-like, runnable academic benchmark substrate for ML systems papers.

📦 **Install**: See [`INSTALL.md`](INSTALL.md) for the `uv sync`, `uv tool install`, and `uv build` package workflow.

📋 **Product contract**: See [`SPEC.md`](SPEC.md) for the CLI, suite/profile vocabulary, backend policy, and validation presets that keep this tree runnable from a fresh clone.

🗂️ **Workload registry**: See [`registry/`](registry/) for the native suite/workload/variant metadata layout. [`workloads.yaml`](workloads.yaml) is kept as a generated compatibility mirror.

🔁 **Iteration loop**: See [`ITERATION_LOOP.md`](ITERATION_LOOP.md) for how we collect student, instructor, researcher, MLCommons, and maintainer feedback without confusing the user-facing product.

⚖️ **Public rules**: See [`PUBLIC_RULES.md`](PUBLIC_RULES.md) for score-bearing, performance-bearing, systems-only, and scenario promotion rules.

🧾 **Dataset release review**: See [`DATASET_RELEASE_REVIEW.md`](DATASET_RELEASE_REVIEW.md) for the public dataset decisions that remain before endorsement.

🎯 **Quality target review**: See [`QUALITY_TARGET_REVIEW.md`](QUALITY_TARGET_REVIEW.md) for the expert-review matrix behind score-bearing and performance-bearing rows.

🚢 **Release checklist**: See [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) for the packaging and endorsement release bars.

📝 **MLCommons proposal**: See [`PROPOSAL.md`](PROPOSAL.md) for the endorsement path and staged review plan.

---

## Quick Start: Run A Benchmark

```bash
# Clone and install
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book/mlperf-edu
uv sync --extra dev

# Check that this machine can run MLPerf EDU
uv run mlperf doctor

# See available workloads
uv run mlperf list
uv run mlperf list matrix --profile max
uv run mlperf info --dataset tinyshakespeare
uv run mlperf info --model smollm2-135m

# Run the smallest local confidence path
uv run mlperf init --profile min

# Run the max benchmark profile
uv run mlperf fetch --profile max
uv run mlperf run --profile max --open-report
```

Every run writes JSON, HTML, and CSV reports. `--open-report` opens the HTML
report in your browser. Add `--power` when you want aggregate estimated watts
and joules without privileged hardware counters.
Reports include hardware/software fingerprints, dataset/model asset dossiers,
checkpoint dependencies, quality-required status, and provenance links. Use
`mlperf report <run-directory> --format html --open` to open the latest report
from a run directory without copying the timestamped JSON filename.

Selection rule: a bare `--profile min|max|pro` selects that default profile
path. `--suite` selects a workload domain. `--workload <canonical>` selects all
variants under that workload family, and `--variant <name>` narrows to one.

## Common Runs

```bash
# Minimal smoke profile
mlperf run --profile min --dry-run
mlperf run --profile min

# Max-profile score-bearing training benchmark
mlperf fetch --workload nanogpt-train --profile max
mlperf run --workload nanogpt-train --profile max --open-report

# Verify, inspect, and package a benchmark result
mlperf verify submissions/nanogpt-train_max.provd.json
mlperf report submissions/nanogpt-train_max_report.json --format html --open
mlperf report submissions/nanogpt-train_max_report.json --format csv
mlperf report submissions --format html --open
mlperf package submissions/nanogpt-train_max.provd.json

# Checkpoint-backed max-profile inference on the trained NanoGPT checkpoint
mlperf run --workload nanogpt-inference --variant prefill --profile max
mlperf run --workload nanogpt-inference --variant decode --profile max

# Max-profile recommender benchmark
mlperf fetch --workload micro-dlrm-train --profile max
mlperf run --workload micro-dlrm-train --profile max

# Max-profile tiny anomaly benchmark
mlperf fetch --workload anomaly-ae-train --profile max
mlperf run --workload anomaly-ae-train --profile max

# Max-profile vision benchmark
mlperf fetch --workload resnet18-train --profile max
mlperf run --workload resnet18-train --profile max

# Off-the-shelf SLM decode suite
mlperf fetch --suite slm --profile max --dry-run
mlperf run --workload smollm2-chat-inference --variant baseline --profile max --model smollm2-135m
mlperf run --workload smollm2-chat-inference --variant quantized-int8 --profile max --model smollm2-135m
mlperf run --workload smollm2-chat-inference --variant batched-b4 --profile pro --model smollm2-135m
mlperf run --workload smollm2-chat-inference --variant long-context --profile pro --model smollm2-135m

# Research-envelope profile
MLPERF_EDU_PRO_REPETITIONS=1 mlperf run --workload nanogpt-train --profile pro
```

## Instructor And Maintainer Commands

These commands validate the suite itself, check public-result metadata, or
grade submissions. Most students and paper readers should not need them for a
normal benchmark run.

```bash
# Audit workload labels and public-result contracts; does not run benchmarks
mlperf audit
mlperf audit --policy public

# Run bundled validation presets; these execute workloads and grade artifacts
mlperf validate smoke
mlperf validate coverage
mlperf validate max
mlperf validate release --output-dir submissions/validation

# Grade a submissions directory
mlperf grade submissions --output submissions/grade.json

# Run tests
pytest
```

The installed command is `mlperf`; this package defaults that command to the
`mlperf-edu` benchmark suite. `mlperf-edu` may also be installed as a
compatibility alias, but public instructions should use `mlperf`.

The benchmark profiles are `min`, `max`, and `pro`: MIN checks, MAX
benchmarks, and PRO explores. Validation presets are named by
intent so they do not collide with profile names: `smoke` runs doctor plus the
default fast path at `min` scale, `coverage` runs every workload at `min` scale,
`max` runs every workload at `max` scale, and `release` runs every workload
at both `min` and `max` scale. Each validation writes run reports and grading
summaries under stable directories such as `submissions/validation/min-default`,
plus one top-level validation JSON/HTML/CSV summary.

Public result status is separate from profile and suite. `score-bearing`
workloads have real-data quality targets and can carry public
quality-plus-performance results. `performance-bearing` workloads have
standardized functional checks and can carry public performance results.
`systems-only` workloads are still useful for architecture, kernel, backend,
quantization, pruning, LoRA, distributed, or agent studies, but should not be
advertised as public scores. Run `mlperf audit` for the local development
contract that students and instructors should see as clean. Run
`mlperf audit --policy public` for endorsement/release review; that stricter
policy fails on unresolved public-release warnings such as dataset terms that
require maintainer or MLCommons approval. `mlperf grade` and `mlperf validate`
remain local execution and quality checks, so they do not fail or warn on
public-release policy decisions.

Observed local validation runtimes on an Apple Silicon laptop are:

| Validation | What it checks | Observed Runtime |
|---|---|---:|
| `smoke` | Doctor plus default 12-workload `min` run | 11.9 s |
| `coverage` | All 30 registered `min` manifests | 24.8 s |
| `max` | All 30 registered `max` manifests | 95.3 s |
| `release` | All 30 `min` plus all 30 `max` manifests | 115.9 s |

The validation summaries persist `duration_seconds` at the top level and for each
suite item. They also embed a workload breakdown in the validation JSON/HTML and
write `mlperf_validate_workloads_<preset>_<timestamp>.csv`, so instructors can
track local machine drift, identify bottleneck workloads, and decide whether a
run belongs in setup, lab, or release validation. Validation summaries also
carry local grading status, and workload CSV rows preserve canonical
selectors, dataset terms, and shared checkpoint dependencies.

The default `min` profile path is the fast 12-workload starter run used for setup confidence.
`mlperf validate coverage` runs every registered workload at `min` scale. All
runs automatically write a timestamped JSON report plus paired HTML and CSV
summaries. Use `--open-report` to open the HTML report in the default browser,
or convert any workload or suite report with
`mlperf report <path-or-run-directory> --format json|csv|html`.
Use `--power` to add aggregate estimated watts and joules to the reports without
requiring privileged hardware counters.
Verified manifests can be bundled with `mlperf package`, and
`mlperf grade` scans a submissions directory with the same provenance
verifier used by the standalone `verify` command.

Every registered workload now has a `min` runner. The goal of `min` is
functional confidence: imports, model construction, a tiny deterministic
forward/train/decode loop, report export, provenance, and grading should all
work locally before instructors or researchers scale to `max` and `pro`.

The `max` profile path is the comparable full-suite run intended for assignments,
artifact evaluation, and paper baselines. It currently runs all 30 registered
workloads at `max` scale. Each score-bearing run emits a report, checkpoint
where applicable, and verifiable provenance manifest. Systems-only `max` workloads
use deterministic micro-shards until their real-data quality checks are promoted.
The `pro` profile has a conservative default path that repeats the matching
`max` runner, records sub-run evidence hashes, and can be scaled with
`MLPERF_EDU_PRO_REPETITIONS`. Larger pro-only sweeps are being wired behind the
contract in [`SPEC.md`](SPEC.md).

The SLM suite is exposed as `smollm2-chat-inference` with variants such as
`baseline`, `quantized-int8`, `batched-b4`, and `long-context`. The public CLI,
CSV, JSON, and HTML reports show those canonical selectors; internal runner IDs
remain metadata for compatibility and debugging. The
`min` profiles use a deterministic tiny local model for setup validation; `max`
defaults to `HuggingFaceTB/SmolLM2-135M-Instruct`; `--model` can select aliases
such as `smollm2-135m`, `qwen2.5-0.5b`, or `qwen3-0.6b`.

The vision suite now has `max` coverage for ResNet-18, MobileNetV2, and the
MobileNet compression-composition workload. Tensor-shard overrides keep tests
fast, while normal score-bearing runs use MIT-licensed Fashion-MNIST.

The recommender suite now has `max` coverage for both DLRM memory-system
variants: `micro-dlrm-dram-train` uses real MovieLens data with a scalable
hashed virtual embedding table, and `micro-dlrm-distributed` validates
localhost Gloo DDP against a gradient-accumulation baseline.

The tiny suite now has `max` smoke coverage for DS-CNN keyword spotting and
visual wake-word models using synthetic micro-shards. These validate training,
checkpointing, reports, and provenance without requiring Speech Commands, Wake
Vision, or torchaudio during setup validation.

The agent suite now has `max` coverage for RAG retrieval/generation, iterative
code generation, ReAct-style tool use, and structured tool calling. These runs
use deterministic local PyTorch models and synthetic prompts so students can
profile systems costs without external APIs.

Research-oriented workloads now have `min` and `max` coverage for MoE,
diffusion, GNN, BERT, LSTM, RL, LoRA fine-tuning, fp32/fp16 NanoGPT decode,
and speculative decoding. The current `max` path is a deterministic
micro-shard systems measurement; real-data quality checks should replace
those micro-shards one workload at a time.

## Benchmark Suite

> [!NOTE]
> **Source of truth** — `registry/suites/...` and `mlperf list` define the executable registry. `workloads.yaml` is a generated compatibility mirror. This table is a human-readable catalog of the major workload families.

<table width="100%" style="width:100%">
  <thead>
    <tr>
      <th align="left" width="10%">Suite</th>
      <th align="left" width="12%">Task</th>
      <th align="left" width="14%">Model</th>
      <th align="left" width="10%">Params</th>
      <th align="left" width="26%">Dataset</th>
      <th align="left" width="28%">Quality target</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>language</td><td>Training</td><td>NanoGPT</td><td>11.1M</td><td>TinyShakespeare from Project Gutenberg</td><td>Loss &lt; 2.3</td></tr>
    <tr><td>language</td><td>Optimization</td><td>Nano-MoE</td><td>17.4M</td><td>TinyShakespeare from Project Gutenberg</td><td>Loss &lt; 0.05</td></tr>
    <tr><td>recommender</td><td>Training</td><td>Micro-DLRM</td><td>23K</td><td>MovieLens-100K</td><td>Acc &gt; 0.70</td></tr>
    <tr><td>vision</td><td>Training</td><td>Micro-Diff.</td><td>2.0M</td><td>CIFAR-10</td><td>MSE &lt; 0.002</td></tr>
    <tr><td>graph</td><td>Training</td><td>Micro-GCN</td><td>5.6K</td><td>Cora</td><td>Acc &gt; 0.78</td></tr>
    <tr><td>language</td><td>Training</td><td>Micro-BERT</td><td>432K</td><td>SST-2</td><td>Acc &gt; 0.78</td></tr>
    <tr><td>timeseries</td><td>Training</td><td>Micro-LSTM</td><td>51K</td><td>ETTh1</td><td>MSE &lt; 0.13</td></tr>
    <tr><td>rl</td><td>Training</td><td>Micro-RL</td><td>17K</td><td>CartPole (local)</td><td>Reward &gt; 195</td></tr>
    <tr><td>slm</td><td>Decode</td><td>SmolLM2/Qwen</td><td>135M+</td><td>Local prompts</td><td>Generated tokens &gt;= 8</td></tr>
    <tr><td>slm</td><td>Quant. Decode</td><td>SmolLM2/Qwen int8</td><td>135M+</td><td>Local prompts</td><td>Generated tokens &gt;= 8</td></tr>
    <tr><td>slm</td><td>Batched Decode</td><td>SmolLM2/Qwen</td><td>135M+</td><td>Prompt batch</td><td>Generated tokens &gt;= 8 per request</td></tr>
    <tr><td>slm</td><td>Long Context</td><td>SmolLM2/Qwen</td><td>135M+</td><td>Expanded local prompt</td><td>Generated tokens &gt;= 8</td></tr>
    <tr><td>vision</td><td>Img. Cls.</td><td>ResNet-18</td><td>11.2M</td><td>Fashion-MNIST</td><td>Top1 &gt; 75%</td></tr>
    <tr><td>vision</td><td>Mobile</td><td>MobileNetV2</td><td>2.4M</td><td>Fashion-MNIST</td><td>Top1 &gt; 70%</td></tr>
    <tr><td>tiny</td><td>KWS</td><td>DS-CNN</td><td>20K</td><td>Speech Commands v2</td><td>Top1 &gt; 90%</td></tr>
    <tr><td>tiny</td><td>Anomaly</td><td>Autoencoder</td><td>0.3M</td><td>MNIST</td><td>MSE &lt; 0.04</td></tr>
    <tr><td>tiny</td><td>Person Det.</td><td>MicroNet</td><td>8.5K</td><td>Wake Vision</td><td>Acc &gt; 85%</td></tr>
    <tr><td>agent</td><td>RAG</td><td>NanoRAG</td><td>20.1M</td><td>ReAct Traces</td><td>Retr.+Gen</td></tr>
    <tr><td>agent</td><td>CodeGen</td><td>NanoCodeGen</td><td>13.7M</td><td>MBPP (20 tasks)</td><td>pass@1 &gt; 0.15</td></tr>
    <tr><td>agent</td><td>ReAct</td><td>NanoReAct</td><td>13.7M</td><td>ReAct Traces</td><td>Trace acc &gt; 0.60</td></tr>
    <tr><td>agent</td><td>ToolCall</td><td>NanoToolCall</td><td>13.7M</td><td>ReAct Traces</td><td>JSON validity + dispatch</td></tr>
  </tbody>
</table>

Most local teaching models are inspectable PyTorch modules. The SLM workload uses `transformers` to hydrate off-the-shelf models. Training times were originally measured on Apple M1 MPS and are being re-verified as the runnable harness stabilizes.

## Project Structure

```text
mlperf-edu/
├── paper/                      # Publication source (LaTeX)
│   ├── paper.tex               # Main paper
│   ├── refs.bib                # Bibliography
│   └── figures/                # TikZ + pgfplots figures
├── reference/                  # Reference implementations
│   ├── cloud/                  # NanoGPT, MoE, DLRM, Diffusion, GNN, BERT, LSTM, RL, Agents
│   ├── edge/                   # ResNet-18, MobileNetV2  (fully local)
│   ├── tiny/                   # DS-CNN, Autoencoder, MicroNet
│   ├── dataset_factory.py      # Unified data loading (deterministic, seed=42)
│   └── agent_datasets.py       # MBPP + ReAct trace datasets
├── src/mlperf/                 # Core harness
│   ├── edu_cli.py              # mlperf CLI entry point
│   ├── loadgen.py              # LoadGen proxy (Offline/Server/SingleStream/MultiStream)
│   ├── power.py                # Power profiler (powermetrics / nvidia-smi)
│   └── sut.py                  # System Under Test interface
├── scripts/
│   └── compliance_checker.py   # Quality target validation
├── examples/                   # Student lab exercises
│   ├── lab1_optimization.py    # Systems optimization challenge
│   ├── lab2_inference_sut.py   # Inference SUT plugin
│   └── lab3_arch_comparison.py # Dense vs. sparse architectures
├── registry/                   # Native suite/workload/variant registry source
├── workloads.yaml              # Generated compatibility mirror
└── data/                       # Local datasets (TinyShakespeare, MBPP, etc.)
```

## Lab Exercises

### Lab 1: Systems Optimization Challenge
Students receive a "broken baseline" ResNet-18 (batch_size=8, no workers, no schedule, no augmentation) and must reach >50% accuracy within a 30-second wall-clock budget.

```bash
python examples/lab1_optimization.py
```

### Lab 2: Inference Latency Optimization
Students implement a System Under Test (SUT) plugin for NanoGPT inference. Optimize with KV-cache, `torch.compile()`, or FP16 while the LoadGen measures p90 latency.

```bash
mlperf run --workload nanogpt-inference --variant decode --profile min
```

### Lab 3: Architecture Comparison
Students train NanoGPT (dense) and Nano-MoE (sparse) side-by-side, comparing convergence, memory, and throughput.

```bash
python examples/lab3_arch_comparison.py
```

## How It Works

**Students are "submitters."** They modify model code, training loops, or inference pipelines. The harness measures everything:

1. **Train** → Quality target validation (loss/accuracy thresholds)
2. **Infer** → LoadGen proxy generates Poisson/bulk arrivals, measures latency percentiles
3. **Profile** → Power measurement via `powermetrics` (macOS) or `nvidia-smi` (Linux)
4. **Submit** → JSON artifact with hardware fingerprint, metrics, and SHA-256 hash
5. **Check** → Compliance checker validates quality, parameter counts, convergence bounds

## Dataset Strategy

<table width="100%" style="width:100%">
  <thead>
    <tr>
      <th align="left" width="22%">Strategy</th>
      <th align="left" width="53%">Datasets</th>
      <th align="left" width="25%">Download</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>Bundled/local</b></td><td>Prompt fixtures, synthetic micro-shards, local traces, and small zero-network setup assets</td><td align="center">0 B</td></tr>
    <tr><td><b>Public upstream</b></td><td>Project Gutenberg TinyShakespeare recipe, Fashion-MNIST, MNIST, Hugging Face model weights, and other public assets recorded in dossiers</td><td align="center">On fetch or first run</td></tr>
    <tr><td><b>Restricted/review</b></td><td>MovieLens-100K, Speech Commands v2, Wake Vision, optional CIFAR experiments, and any dataset whose public redistribution policy still needs owner or MLCommons review</td><td align="center">Policy-dependent</td></tr>
  </tbody>
</table>

Each asset has source, license, public-release status, cache behavior, and next-step metadata. Synthetic or micro-sharded data is labeled and is not treated as a public score.

## Requirements

- Python 3.10+
- `uv` for the recommended install path
- PyTorch 2.0+
- `torchvision` (for Fashion-MNIST, MNIST, and optional CIFAR experiments)
- `transformers` (for SLM workloads)
- Optional: `torchaudio` for full Speech Commands experiments

```bash
uv sync --extra dev
uv run mlperf doctor
```

For Apple Silicon: set `PYTORCH_ENABLE_MPS_FALLBACK=1` for full MPS compatibility.

## Citation

```bibtex
@inproceedings{mlperfedu2026,
  title={{MLPerf EDU}: Bridging Industry Benchmarking and {ML} Systems Education},
  author={[Authors]},
  year={2026}
}
```

---

*Built for [Machine Learning Systems](https://mlsysbook.ai) education.*
