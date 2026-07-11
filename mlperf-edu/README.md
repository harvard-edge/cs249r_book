<!-- MLPERF-EDU-STATUS:START -->
> [!WARNING]
> **Independent review preview**
>
> MLPerf EDU is not an official MLCommons benchmark and is not endorsed by
> MLCommons. The repository contains a runnable candidate suite, but a release
> claim requires fresh validation evidence, complete five-seed reference
> packets, closed asset-policy decisions, and external review. Do not describe
> results from this tree as official MLPerf results.
<!-- MLPERF-EDU-STATUS:END -->

# MLPerf EDU

MLPerf EDU is a laptop-scale ML systems benchmark and teaching suite. The
current registry has 30 executable rows in 23 workload families across 10
suites. It separates five score-bearing candidates, three
performance-bearing candidates, and 22 systems-only teaching and research
rows.

The counts and labels are project classifications for review. They are not
MLCommons-approved result categories.

## Current Readiness Boundary

| **State** | **What It Means Here** |
|:---|:---|
| Implemented | The CLI, native registry, `min` and `max` runners, reports, provenance verification, portable packaging, labs, tutorial smoke, generated site, and validation workflows exist in this tree. |
| Must be revalidated | The complete test suite, actual `smoke`, `max`, and `release` presets, clean wheel install, package extraction, generated-file checks, site render, and paper build must all pass on the final source revision. |
| Evidence pending | Every score-bearing target needs one clean, create-once five-seed evidence packet produced from the final source revision. Registry calibration values are not a substitute for those packets. |
| External decision | The component license, MovieLens-100K policy, public result wording, project name, and any MLCommons relationship require decisions outside this repository. |

The executable release ledger is [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
The target-by-target evidence review is
[QUALITY_TARGET_REVIEW.md](QUALITY_TARGET_REVIEW.md).

## Install From a Checkout

The supported preview path is a source checkout with the locked environment.
No package-index release is claimed.

```bash
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book/mlperf-edu
uv sync --locked --extra dev
uv run mlperf doctor
uv run mlperf validate smoke --output-dir submissions/first-smoke
```

Python 3.10 or newer is required. A GPU is optional. Score-bearing `max` runs
fetch public datasets, and the SLM `max` path fetches pinned Hugging Face model
weights. Run `fetch` before measurement so network time is outside the result.

See [INSTALL.md](INSTALL.md) for wheel and clean-environment checks.

## Run One Candidate Benchmark

```bash
uv run mlperf fetch --workload resnet18-train --profile max
uv run mlperf run \
  --workload resnet18-train \
  --profile max \
  --output-dir submissions/resnet-review \
  --open-report
```

Every workload run writes a JSON report, an HTML view, a CSV view, and a
`.provd.json` provenance manifest. A passing run proves that one execution met
its local contract. It does not prove that the release, target evidence, asset
policy, or MLCommons review gates are closed.

Useful discovery commands follow.

```bash
uv run mlperf list
uv run mlperf list matrix --profile max
uv run mlperf list variants --workload nanogpt-inference
uv run mlperf show nanogpt-train
uv run mlperf info --dataset tinyshakespeare
uv run mlperf info --model smollm2-135m
```

A bare `--profile` selects that profile's default collection. `--suite` selects
one workload domain, `--workload` selects one row or family, and `--variant`
narrows a family to one variant.

## Profiles and Result Status

| **Term** | **Contract** |
|:---|:---|
| `min` | Fast deterministic execution and artifact plumbing. Synthetic or tiny local data is allowed. A `min` result is not a quality baseline. |
| `max` | Candidate comparable scale. Public candidates must use eligible real, checkpoint-backed, or pinned-model data and pass the report-level review contract. |
| `pro` | Research repetitions and controlled variants. It does not become public evidence without a declared protocol. |
| `score-bearing` | Candidate quality-plus-performance row with a real-data target and five-seed protocol. |
| `performance-bearing` | Candidate performance row with a nonempty functional or task-quality gate and repeatable timing. |
| `systems-only` | Runnable teaching or research row. Its numbers must not be advertised as public benchmark scores. |

Every systems-only row has a machine-readable `max_execution` boundary in the
registry. Generated pages disclose the actual data mode, whether fetched and
declared assets are consumed, and whether the current runner enforces a quality
target. Several incubation rows intentionally use deterministic micro-shards;
their declared research datasets and targets are not presented as current max
baselines.

`mlperf audit` checks registry structure and local policy without running the
benchmarks. `mlperf audit --policy public` also fails on unresolved public
release warnings. Execution evidence comes from `mlperf validate`, not from
`audit` or `--dry-run`.

## Candidate Quality Targets

The five score-bearing rows use seeds 0 through 4 and the median as the
acceptance statistic.

| **Workload** | **Dataset** | **Required `max` Metric** | **Target** | **Release-Evidence State** |
|:---|:---|:---|---:|:---|
| `nanogpt-train` | Project Gutenberg TinyShakespeare recipe | cross-entropy loss | `<= 2.30` | Seeds 0-4 reached `1.9816`-`2.1345`, median `2.0878`; clean final-revision packet pending. |
| `micro-dlrm-train` | MovieLens-100K | best validation accuracy | `>= 0.70` | Five-seed calibration is recorded; clean packet and MovieLens policy decision pending. |
| `anomaly-ae-train` | MNIST | anomaly AUROC | `>= 0.95` | Five-seed full-test development calibration is recorded; clean packet pending. |
| `resnet18-train` | Fashion-MNIST | top-1 accuracy | `>= 0.85` | Five-seed full-test development calibration is recorded; clean packet pending. |
| `mobilenetv2-train` | Fashion-MNIST | top-1 accuracy | `>= 0.78` | Five-seed full-test development calibration is recorded; clean packet pending. |

The anomaly row grades discrimination across labeled normal and anomalous
examples. Reconstruction MSE remains diagnostic and is not the public target.
The target rationale and rerun rules live in
[QUALITY_TARGET_REVIEW.md](QUALITY_TARGET_REVIEW.md).

## Candidate Inference Methodology

Three inference rows are performance-bearing candidates.

Each registry contract asks for five retained reference executions. The table
below describes the repeated measurements inside each execution.

| **Workload** | **Functional and Quality Gate** | **Default Timing Protocol** |
|:---|:---|:---|
| `nanogpt-inference --variant prefill` | A quality-approved NanoGPT checkpoint with a recorded SHA-256 digest must complete prefill with positive throughput. | Three discarded warmups and ten synchronized measurements; median, p90, and p99 latency. |
| `nanogpt-inference --variant decode` | The same checkpoint lineage must complete 64 decode steps with positive throughput. | One discarded warmup and five synchronized measured requests; TTFT and inter-token median, p90, and p99 latency. |
| `smollm2-chat-inference --variant baseline` | Pinned SmolLM2 revision, at least eight generated tokens, and continuation perplexity `<= 10` on the bundled four-case suite. | One warmup and five measured requests; separate prefill and generation median, p90, and p99 latency. |

The SLM development calibration completed five outer executions. All five
passed the token and perplexity gates; output throughput had a median of
`74.04` tokens/s and a range of `73.67`-`87.12` tokens/s on the recorded MPS
host. A clean-commit public-candidate packet is still required.

The dynamic-int8 SLM variant is systems-only. Its current calibration completes
generation but fails the quality-parity gate, so its latency is not eligible as
a public performance result.

## Provenance and Portable Packages

```bash
uv run mlperf verify submissions/resnet-review/resnet18-train_max.provd.json
uv run mlperf package submissions/resnet-review/resnet18-train_max.provd.json
uv run mlperf grade submissions/resnet-review --output submissions/resnet-review/grade.json
```

The manifest binds the exact report bytes and canonical report contents to
recorded source, data, model weights, seed, hardware, and optional sidecars.
Its digest detects changes but does not authenticate who produced the result.
`mlperf package` creates a package-schema 0.2 archive with relative artifact
paths, SHA-256 and size indexes, and a clean-extraction verification step.
It refuses known restricted or unresolved dataset bytes, including MovieLens,
so a local hash-verifiable run cannot accidentally become a redistribution.

For checkpoint-backed NanoGPT inference, the report carries the checkpoint
digest and training-quality dependency. For the SLM candidate, the report
carries the pinned model revision, bundled quality-suite digest, model metadata,
and task-quality result.

## Validation Commands

```bash
# Fast blocking checks
uv run pytest
uv run mlperf audit
uv run mlperf validate smoke --output-dir submissions/validation-smoke

# Generated-source and documentation drift
uv run python tools/export_registry_layout.py --check
uv run python tools/export_flat_registry.py --check
uv run python tools/check_taxonomy.py
uv run python tools/generate_review_packets.py --check
uv run python tools/generate_docs.py --check

# Evidence-bearing full execution
uv run mlperf validate max --keep-going --output-dir submissions/validation-max
uv run mlperf validate release --keep-going --output-dir submissions/validation-release
```

The full workflow has a five-hour CI timeout and uploads its artifacts. A dry
run shows selection only and never counts as benchmark evidence. The live site
workflow requires both recent development validation and recent full benchmark
validation before a manual publish. Workflow presence does not mean either
workflow has passed or deployed for the current revision.

## Labs and Tutorial

The three command-line labs have deterministic CPU-only, network-free smoke
paths.

```bash
uv run python examples/lab1_optimization.py --smoke
uv run python examples/lab2_inference_sut.py --smoke
uv run python examples/lab3_arch_comparison.py --smoke
uv run python tutorials/smoke_first_benchmark.py
```

Lab 1 compares real ResNet-18 training-loop configurations. Lab 2 compares
naive and KV-cache decode with token parity. Lab 3 compares dense NanoGPT and
Nano-MoE on identical batches. Their JSON outputs are classroom measurements,
not canonical submissions. The product CLI does not currently load an
arbitrary `--sut` plugin path.

Tutorial 01 runs a registered `min` workload through the public command,
inspects its reports, and verifies its provenance. Additional notebooks remain
roadmap items and are labeled that way in [tutorials/README.md](tutorials/README.md).

## Dataset and Governance Documents

- [PUBLIC_RULES.md](PUBLIC_RULES.md) defines the candidate result contract.
- [DATASET_RELEASE_REVIEW.md](DATASET_RELEASE_REVIEW.md) records all 14 dataset
  classifications and the unresolved MovieLens decision.
- [DESIGN_PHILOSOPHY.md](DESIGN_PHILOSOPHY.md) states the implemented design
  boundary without promising unavailable plugin or hydration commands.
- [NORTH_STAR.md](NORTH_STAR.md) separates the long-term ambition from the
  current release state.
- [PROPOSAL.md](PROPOSAL.md) is the review brief for MLCommons stakeholders.
- [site/](site/) contains the Quarto documentation source. Registry and dataset
  pages are generated with `uv run python tools/generate_docs.py`.

## Project Structure

```text
mlperf-edu/
├── registry/                 # Native executable source of truth
├── src/mlperf/               # CLI, policy, provenance, runners, references
├── src/mlperf_edu/           # Packaged registry and bundled prompt fixture
├── tools/                    # Registry, docs, review, and reference-sweep tools
├── examples/                 # Three complete classroom labs
├── tutorials/                # One implemented notebook and CI smoke
├── review_packets/           # Generated public-candidate review packets
├── site/                     # Quarto review site
├── paper/                    # Draft companion paper
├── workloads.yaml            # Generated compatibility mirror
└── datasets.yaml             # Dataset release catalog
```

## Draft Citation

The companion paper is a draft and is not yet an archival benchmark citation.

```bibtex
@misc{mlperfedu2026,
  title  = {{MLPerf EDU}: Bridging Industry Benchmarking and ML Systems Education},
  author = {MLPerf EDU Contributors},
  year   = {2026},
  note   = {Independent review preview}
}
```

MLPerf EDU is developed alongside the
[Machine Learning Systems](https://mlsysbook.ai) textbook.
