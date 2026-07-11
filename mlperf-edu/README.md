<!-- MLPERF-EDU-STATUS:START -->
> [!WARNING]
> **Independent review preview**
>
> MLPerf EDU is not an official MLCommons benchmark and is not endorsed by
> MLCommons. The repository contains a runnable candidate suite, but a release
> claim still requires complete final validation, reviewer access to retained
> raw evidence, closed asset-policy decisions, and external review. Do not
> describe results from this tree as official MLPerf results.
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
| Historical validation snapshot | The promotion revision recorded in `review-ece36ac566/handoff_manifest.json` passed 241 tests, the four validation presets, a clean wheel install, package extraction, generated-file checks, site render, link checks, and the paper build. Subsequent protocol changes require the complete matrix to run again on the final revision. |
| Current reference evidence committed | Eight content-addressed summaries from source commit `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3` are committed and review-eligible for local handoff under the current contracts. |
| Verification limits | The committed summaries do not substitute for hosted CI on a pushed revision, independent reproduction, or representative desktop and narrow browser inspection. Those gates remain recorded in the release ledger. |
| External decision | The component license, MovieLens-100K policy, public result wording, project name, and any MLCommons relationship require decisions outside this repository. |

The executable release ledger is [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
The target-by-target evidence review is
[QUALITY_TARGET_REVIEW.md](QUALITY_TARGET_REVIEW.md).

The current committed reference evidence uses source commit
`86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`; its raw create-once attempts are
available for local handoff.

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
| `nanogpt-train` | Project Gutenberg TinyShakespeare recipe | cross-entropy loss | `<= 2.30` | Current five-run packet committed. |
| `micro-dlrm-train` | MovieLens-100K official `u1.base`/`u1.test` split | fixed-final-epoch ROC AUC | `>= 0.76` | Current five-run packet committed; raw MovieLens-derived artifacts remain local-only pending policy review. |
| `anomaly-ae-train` | MNIST hard-curve-v1 | macro anomaly AUROC | `>= 0.93` | Current five-run packet committed; every run also reaches worst-class AUROC `>= 0.90` and learned-control margin `>= 0.20`. |
| `resnet18-train` | Fashion-MNIST | top-1 accuracy | `>= 0.85` | Current five-run packet committed. |
| `mobilenetv2-train` | Fashion-MNIST | top-1 accuracy | `>= 0.78` | Current five-run packet committed. |

The anomaly row trains on digit 5 and evaluates digits 3, 8, and 9 separately.
Its macro score, worst-class gate, and margin over three no-training controls
must all pass. Reconstruction MSE remains diagnostic and is not the public
target.
The target rationale and rerun rules live in
[QUALITY_TARGET_REVIEW.md](QUALITY_TARGET_REVIEW.md).

The exact IDs, full SHA-256 digests, and aggregates for the current packets are
indexed in `reference_results/index.json`.

## Candidate Inference Methodology

Three inference rows are performance-bearing candidates.

Each registry contract asks for five retained reference executions. The table
below describes the repeated measurements inside each execution.

| **Workload** | **Functional and Quality Gate** | **Default Timing Protocol** |
|:---|:---|:---|
| `nanogpt-inference --variant prefill` | A quality-approved NanoGPT checkpoint with a recorded SHA-256 digest must complete prefill with positive throughput. | One content-addressed fixed prompt, fresh KV-cache materialization, three discarded warmups, and twenty synchronized measurements; median, p90, and p99 latency. |
| `nanogpt-inference --variant decode` | The same checkpoint lineage must complete 64 decode steps with positive throughput. | Single-stream sequential microbenchmark with three discarded warmups and twenty synchronized requests. Request TTFT spans prompt prefill through first-token selection; every subsequent cached-token interval is an ITL, and the first is retained separately as first-decode latency. |
| `smollm2-chat-inference --variant baseline` | Pinned SmolLM2 revision, at least eight generated tokens, token-weighted continuation perplexity `<= 7` overall, and worst-category perplexity `<= 24` on the attributed 28-case v2 suite. | Three warmups and twenty cache-reusing greedy requests. TTFT spans prompt prefill through the first output token; subsequent ITLs and complete request latency are retained separately. |

All three performance-bearing summaries are current five-run packets. NanoGPT
prefill measures fresh cache materialization with twenty samples, decode uses
causal TTFT and ITL boundaries, and SmolLM2 uses the attributed 28-case v2
suite.

## Current Reference Evidence

Source revision: `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`. The public
candidate repeatability limit is `5%` coefficient of variation for timed
performance references.

| **Workload** | **Evidence ID** | **Primary Metric Median** | **Minimum** | **Maximum** | **CV** |
|:---|:---|---:|---:|---:|---:|
| `anomaly-ae-train` | `anomaly-ae-train_max_20260711T204007.498158Z` | `4.3060` | `4.2488` | `4.5012` | n/a |
| `micro-dlrm-train` | `micro-dlrm-train_max_20260711T205839.712863Z` | `2.0394` | `1.9475` | `2.0924` | n/a |
| `mobilenetv2-train` | `mobilenetv2-train_max_20260711T204653.574587Z` | `103.6699` | `103.4276` | `104.7116` | n/a |
| `nanogpt-decode` | `nanogpt-decode_max_20260711T210118.136527Z` | `316.7836` | `313.2178` | `319.4718` | `0.75%` |
| `nanogpt-prefill` | `nanogpt-prefill_max_20260711T205947.060179Z` | `3859.7758` | `3849.9537` | `3868.4721` | `0.19%` |
| `nanogpt-train` | `nanogpt-train_max_20260711T202223.716219Z` | `117.5981` | `117.0730` | `118.2963` | n/a |
| `resnet18-train` | `resnet18-train_max_20260711T204117.913822Z` | `58.4192` | `57.3181` | `59.5123` | n/a |
| `slm-decode` | `slm-decode_max_20260711T210317.517476Z` | `61.5742` | `60.4083` | `61.9703` | `1.11%` |

The dynamic-int8 SLM variant is systems-only. Its historical v1 calibration
completed generation but failed quality parity, and its exact values are now
protocol-superseded. A fresh v2 calibration must pass the overall,
weakest-category, and NLL-parity gates before its latency can become eligible.

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

`tools/build_handoff_manifest.py` closes a reviewer handoff across the committed
index, historical raw attempts, source lock, NanoGPT lineage archive, and every
policy-permitted portable run package. Its output records policy-blocked runs
explicitly and must be written outside the checkout.

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
uv run python tools/sync_verified_baselines.py --check
uv run python tools/check_taxonomy.py
uv run python tools/check_reference_claims.py --check
uv run python tools/generate_review_packets.py --check
uv run python tools/generate_docs.py --check

# Evidence-bearing full execution
uv run mlperf validate coverage --output-dir submissions/validation-coverage
uv run mlperf validate max --keep-going --output-dir submissions/validation-max
uv run mlperf validate release --keep-going --output-dir submissions/validation-release
```

The `max` and `release` presets use reference-protocol seed 0 when no seed
environment variable is set. Each validation record includes the resolved seed
and its source. Set `MLPERF_EDU_SEED` explicitly to audit another declared
reference seed.

The full workflow has a five-hour CI timeout and uploads its artifacts. A dry
run shows selection only and never counts as benchmark evidence. The live site
workflow requires both recent development validation and recent full benchmark
validation before a manual publish. Workflow presence does not mean either
workflow has passed or deployed for the current revision. No hosted
same-revision CI result or representative in-app browser inspection is claimed
by the committed reference summaries.

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
├── reference_results/         # Committed content-addressed evidence summaries
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
