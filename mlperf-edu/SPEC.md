# MLPerf EDU v0.1 Specification

## Status

This document defines the independent MLPerf EDU v0.1 review candidate. It is
not an MLCommons specification and does not create an official MLPerf result
category. The normative words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY
describe conformance within this repository.

The registered portfolio contains fourteen workloads. The current evidence
scope covers nine workloads and twelve evidence cases. Those draft
measurements are content-addressed in `provisional_results/index.json` and
rendered into the website and paper from that source. Six records have complete
five-run project evidence; six are explicitly provisional. The five remaining
workloads are functional-stage integrations and have no draft quality result.
No provisional or functional-stage record is a promoted baseline or an
MLCommons-verified result.

## Purpose and Scope

MLPerf EDU is a locally executable, quality-gated benchmark specification for
teaching and studying single-node ML systems. It supports controlled work on
processors, memory systems, runtimes, compilers, and model execution. Every
canonical workload must run on laptop-class hardware and remain understandable
in a machine-learning systems course.

The following claims are outside v0.1:

- Distributed scaling or communication efficiency
- Datacenter throughput, availability, or serving capacity
- Official MLPerf compliance or MLCommons endorsement
- Capability evaluation for autonomous agents or retrieval-augmented generation
- A replacement for upstream task leaderboards

## Admission Rule

A score-bearing workload MUST inherit all of these elements from an
authoritative upstream source:

1. Task definition
2. Model or reference implementation
3. Dataset and split
4. Evaluator and metric
5. Quality target or published baseline
6. Provenance source and pinned artifacts
7. A practical laptop execution boundary

MLPerf EDU MAY add a thin PyTorch adapter, measurement harness, quality gate,
provenance capture, packaging, and reporting. It MUST NOT invent a reduced
task, substitute dataset, new metric, or arbitrary quality target to fill a
coverage gap.

Every admitted workload also needs a rationale that explains task
significance, benchmark lineage, classroom value, and distinct single-node
systems behavior. Rejected and deferred proposals remain in the selection
ledger with explicit reasons.

## Portfolio

| **Workload** | **Suite** | **Upstream Authority** | **Canonical Boundary** | **Primary Systems Behavior** |
|:---|:---|:---|:---|:---|
| `image-classification` | vision | MLCommons MLPerf Tiny | ResNet8 inference over the official accuracy set | dense convolution, layout, and batching |
| `keyword-spotting` | tiny | MLCommons MLPerf Tiny and EEMBC | DS-CNN inference over the official accuracy set | depthwise convolution and small-tensor dispatch |
| `anomaly-detection` | tiny | MLCommons MLPerf Tiny, ToyADMOS, and DCASE | autoencoder inference over the official ToyCar accuracy set | spectrogram preparation, dense reconstruction, and error scoring |
| `visual-wake-words` | tiny | MLCommons MLPerf Tiny and EEMBC | MobileNetV1 0.25 inference over the official accuracy set | depthwise convolution, image decoding, and small-model dispatch |
| `causal-language-modeling` | language | nanoGPT | training plus full, prefill, and decode inference | attention training, KV-cache execution, and phase transitions |
| `text-classification` | language | DistilBERT and GLUE | pinned SST-2 checkpoint inference | encoder attention and variable-length batching |
| `information-retrieval` | language | Sentence Transformers | CrossEncoder NanoBEIR reranking | pair tokenization, encoder scoring, and ranking |
| `graph-node-classification` | graph | Open Graph Benchmark | official GCN training on `ogbn-arxiv` | sparse gather, scatter, and irregular memory access |
| `time-series-forecasting` | timeseries | PatchTST | official ETTm1 training and evaluation | long-context attention and patch-based sequence processing |
| `code-generation` | language | Qwen and EvalPlus | complete pinned HumanEval+ generation and sandboxed evaluation; executed, target not met | variable-length autoregressive decode and sandboxed evaluation |
| `function-calling` | language | BFCL and Qwen | complete 1,150-case BFCL AST evaluation; executed, target not met | schema-heavy prefill and structured decode |
| `recommendation` | recommendation | MLPerf Training v0.5 NCF | trains locally on MovieLens-20M; published 0.635 HR@10 target, recorded as a miss | embedding lookup and dense interaction |
| `image-generation` | vision | NVIDIA EDM | complete three-trial 50,000-image FID protocol; executed, target not met | repeated denoiser execution and scheduler overhead |
| `reinforcement-learning` | reinforcement | MLPerf Training MiniGo | policy-value self-play probe; authoritative MiniGo contract not executed locally | search-coupled inference and dynamic training data |

The five new workloads MUST retain `experimental` status and MUST set
`promotion_scope` to false until their quality evidence is accepted. Code
generation, function calling, and image generation MAY emit a canonical `max`
quality result and MUST report the measured shortfall against the unchanged
target; they MUST remain ineligible for promotion during this readiness stage.
Recommendation and reinforcement learning MUST state that the authoritative
quality contract was not executed locally.
MiniGo remains the reinforcement-learning identity; a small control task MUST
NOT be substituted under that label.

## Identity and Taxonomy

### Workload

A workload is the stable learning task and upstream quality contract. A
workload ID MUST NOT encode an optimization, hardware backend, batch size, or
serving strategy.

### Mode

Training and inference are modes under the same workload identity. A mode MAY
select a different upstream artifact boundary, but it MUST remain bound to the
same task and declared lineage.

### Phase

An inference mode MAY expose phases such as full, prefill, and decode. A phase
is an independently measured case, not an independent workload.

### Configuration and Scenario

Precision, quantization, compilation, batching, scheduling, context length,
and serving behavior are configurations or scenarios. Reports MUST disclose
them. They MUST NOT appear as workload IDs.

### Result Role

| **Role** | **Meaning** |
|:---|:---|
| `score-bearing` | The canonical task metric and timing are both present, and every quality gate passes. |
| `performance-bearing` | The case reports comparable timing after every functional gate passes. |
| `systems-only` | The run supports systems exploration but cannot establish a public task score or performance baseline. |
| `deferred` | The authoritative contract is known but is not practical locally without changing it. |
| `rejected` | The proposal depends on a project-invented or unstable benchmark definition. |

## Profiles

| **Profile** | **Required Use** | **Data and Quality Boundary** |
|:---|:---|:---|
| `min` | Fast setup, teaching, and CI check | MAY use a deterministic reduced input; MUST NOT be promoted. |
| `max` | Canonical classroom and comparison run | MUST use the real-data contract before promotion; a functional-stage `max` probe MUST remain explicitly nonconformant and nonpromotable. |
| `pro` | Extended single-node research study | MAY expose controlled configurations while retaining workload identity. |

The `pro` profile is not a larger workload collection. It is the research
envelope for the selected workload.

## Quality Contracts

Each score-bearing case MUST declare its metric key, direction, target,
tolerance, target basis, evaluator, and reference protocol. Every fresh process
and the declared aggregate MUST meet the gate. A median pass MUST NOT hide a
failed individual run.

Performance-bearing phases MUST declare a functional gate. Every fresh process
must pass it before timing can be promoted. There is no machine-derived
performance threshold.

The quality contracts are:

| **Workload** | **Metric** | **Gate Basis** |
|:---|:---|:---|
| `image-classification` | top-1 accuracy at least 0.85 | MLPerf Tiny threshold |
| `keyword-spotting` | top-1 accuracy at least 0.90 | MLPerf Tiny threshold |
| `anomaly-detection` | ROC AUC at least 0.85 | MLPerf Tiny threshold |
| `visual-wake-words` | top-1 accuracy at least 0.80 | MLPerf Tiny threshold |
| `causal-language-modeling` | validation cross-entropy at most 1.4697 | nanoGPT published Shakespeare result |
| `text-classification` | SST-2 accuracy at least 0.9105504587155964 | pinned checkpoint model-index metadata |
| `information-retrieval` | mean nDCG@10 equal to the documented evaluator result | Sentence Transformers example |
| `graph-node-classification` | test accuracy within the published GCN tolerance | official OGB GCN reference |
| `time-series-forecasting` | test MSE at most 0.290 | Strict reproduction point from the published PatchTST result |

`QUALITY_TARGET_REVIEW.md` records the numerical details and rationale.

## Measurement Contract

Every canonical case MUST define the measured region, included phases,
excluded phases, synchronization rule, primary timing metric, and
repeatability limit. Asset download, model construction, dataset preparation,
report serialization, and provenance serialization are excluded unless the
case explicitly declares otherwise.

Promotion requires five fresh processes at the canonical seed. Fresh
processes are used to measure execution repeatability, not seed sensitivity.
The sample coefficient of variation of the primary timing metric MUST be at
most 0.05. A failed or interrupted process invalidates the complete attempt.
Individual runs MUST NOT be replaced inside an existing attempt.

Laptop reference campaigns SHOULD use AC power with platform low-power modes
disabled. Power source, power policy, sleep interruptions, and material
background activity MUST be disclosed. An interrupted or power-state-changing
attempt MUST be rejected and rerun in full.

## Evidence Closure

The current twelve evidence cases are one canonical `max` case for each of the
nine workloads in `promotion_scope` plus full, prefill, and decode inference
for `causal-language-modeling`. The five functional-stage workloads are
excluded until quality conformance replaces their bounded probes. The eventual
fourteen-workload closure contains seventeen cases.

Each promoted case summary MUST contain:

- Exactly five passing run records
- Exact source SHA and clean-tree evidence
- Case role, workload, profile, mode, phase, and scenario
- Primary and quality aggregates where applicable
- Repeatability and acceptance decisions
- Comparison fingerprint and artifact digests
- Paths to JSON, CSV, HTML, and provenance artifacts
- A complete create-once retained-file index

All three causal inference phases MUST use one portable package selecting
exactly one committed training run whose quality is the five-run median. The
package, checkpoint, source report, and source provenance digests MUST match
across phases.

The importer independently verifies historical source code, registry
contracts, raw reports, manifests, artifacts, package structure, digests,
quality gates, repeatability, and lineage before writing the committed index.

The draft index MAY retain a one-run or two-run provisional record to document
execution and gate passage before promotion. Such a record MUST state its run
count, MUST set promotion and public-baseline eligibility to false, MUST state
that repeatability is not established, and MUST remain separate from
`reference_results/`. The current draft contains six five-run verified records,
five one-run provisional records, and one two-run provisional record. The
two-run causal-training record passes its quality gate but has a diagnostic
5.19% timing CV, so it is not repeatable under the 5% rule.

## Provenance and Packaging

Every canonical run MUST emit JSON, CSV, HTML, and `.provd.json` outputs. The
manifest binds source, model or checkpoint, dataset files, random state,
hardware fingerprint, and measurement report. SHA-256 provides integrity
checking, not producer authentication.

Portable packages MUST use safe relative paths, index every retained file,
verify after clean extraction, and exclude bytes whose redistribution policy
does not permit packaging. Fetch-only operation is acceptable when the
dataset contract remains reproducible.

## CLI Contract

The public workflow is:

```bash
mlperf doctor
mlperf list
mlperf show WORKLOAD
mlperf fetch --workload WORKLOAD --profile PROFILE
mlperf run --workload WORKLOAD --profile PROFILE [--mode MODE] [--phase PHASE]
mlperf report REPORT_OR_DIRECTORY --format html
mlperf verify MANIFEST
mlperf package MANIFEST
mlperf audit --policy public
mlperf validate PRESET
```

Explicit `--mode` and `--phase` selections MUST override defaults only when the
selected workload declares them. Unsupported combinations MUST fail before
execution.

## Platform Support

The reference implementation targets PyTorch on CPU and Apple Silicon MPS.
CUDA MAY be used where the same runner and quality contract apply. Backend
availability does not imply a published baseline. Reports MUST identify the
requested device, executed device, executed backend, software stack, processor
topology, accelerator runtime, and performance-related environment settings.

Canonical baselines are machine-specific reference results. They are not
universal performance thresholds and MUST NOT be compared without matching
case fingerprints and disclosures.

## Conformance

A v0.1 review candidate is conformant only when:

1. The native registry and packaged flat mirrors agree.
2. All fourteen workload definitions pass schema and taxonomy validation.
3. All twelve draft evidence cases pass import, digest, source-lock, lineage,
   and evidence-class validation.
4. Every public claim is bound to the committed draft index and distinguishes
   five-run verified evidence from provisional evidence.
5. Every `min`, applicable `max`, and applicable `pro` path executes through the public CLI.
6. CPU and Apple Silicon support claims have direct test evidence.
7. Unit, integration, provenance, packaging, site, paper, and clean-install workflows pass.
8. Generated documentation has no drift or retired workload pages.
9. The release checklist distinguishes completed implementation from external governance decisions.

No passing subset can compensate for a missing required case, stale public
claim, synthetic score-bearing input, or unverifiable artifact.
