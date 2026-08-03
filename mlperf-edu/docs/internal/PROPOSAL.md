# MLPerf EDU v0.1 Review Proposal

## Proposed North Star

MLPerf EDU is a locally executable, quality-gated benchmark specification for
teaching and studying single-node ML systems. It transfers the
reproducibility, verification, disclosure, and comparability discipline of
mature benchmark suites to classroom-scale PyTorch workloads. It supports
controlled research on processors, memory systems, runtimes, compilers, and
model execution while explicitly excluding distributed and datacenter-scale
claims.

This is an independent proposal for discussion. It is not an official
MLCommons benchmark and is not endorsed by MLCommons.

## Problem

Machine-learning systems courses need workloads that are small enough to run
locally, transparent enough to teach, and disciplined enough to support
reproducible comparison. Small teaching examples often omit quality gates,
measurement boundaries, provenance, and stable task definitions. Production
benchmark suites provide stronger governance but are often too operationally
heavy for a notebook or a single class period.

MLPerf EDU occupies the space between those extremes. The project packages
established task definitions behind one PyTorch CLI and one artifact contract.
It does not create new learning tasks merely to expand coverage.

## Backward Design

The intended classroom experience determines the architecture:

1. A student installs one locked environment and inspects the registry.
2. The student fetches pinned assets before measurement.
3. A `min` run confirms that the path works.
4. A `max` run executes the canonical real-data task and quality gate.
5. The report exposes timing, quality, configuration, and provenance.
6. A `pro` study changes a controlled system configuration without renaming the workload.
7. An instructor or researcher verifies and packages the resulting artifact.

That workflow requires stable workload identity, explicit modes and phases,
quality-gated timing, portable provenance, and fail-closed public claims.

## Portfolio

The proposed v0.1 portfolio contains fourteen workloads. The current evidence
scope covers nine workloads and twelve evidence cases.

| **Workload** | **Reason for Inclusion** | **Distinct Systems Value** |
|:---|:---|:---|
| `image-classification` | Directly inherits the MLPerf Tiny ResNet8 definition. | Dense convolution, input layout, and offline batching. |
| `keyword-spotting` | Directly inherits the MLPerf Tiny DS-CNN definition. | Depthwise convolution and latency-sensitive small tensors. |
| `anomaly-detection` | Directly inherits the MLPerf Tiny ToyCar autoencoder definition. | Spectrogram construction, dense reconstruction, and anomaly scoring. |
| `visual-wake-words` | Directly inherits the MLPerf Tiny MobileNetV1 0.25 definition. | Depthwise convolution, image decoding, and compact vision dispatch. |
| `causal-language-modeling` | Uses the established nanoGPT Shakespeare recipe without reducing it. | Transformer training, full inference, prefill, decode, and checkpoint lineage. |
| `text-classification` | Uses a pinned published DistilBERT SST-2 checkpoint. | Encoder attention, tokenization, padding, and batching. |
| `information-retrieval` | Reproduces the documented CrossEncoder NanoBEIR example. | Query-document pair scoring and ranking. |
| `graph-node-classification` | Uses the official OGB GCN recipe and evaluator. | Sparse gather, scatter, and irregular memory access. |
| `time-series-forecasting` | Uses the official PatchTST ETTm1 recipe and split. | Patch extraction, long-context attention, and multivariate sequence training. |
| `code-generation` | Preserves Qwen2.5-Coder and the complete HumanEval+ contract; one authoritative local result remains pending. | Variable-length autoregressive decode and sandboxed correctness evaluation. |
| `function-calling` | Preserves Qwen3-1.7B and BFCL V4 Non-Live AST while quality reproduction remains pending. | Schema-heavy prefill and short structured decode. |
| `recommendation` | Preserves Meta DLRM and the Criteo contract while a practical quality boundary remains pending. | Sparse embeddings, memory capacity, and dense-sparse interaction. |
| `image-generation` | Preserves NVIDIA EDM and its official 50,000-image FID contract while exact reproduction remains pending. | Iterative denoising, scheduler overhead, and repeated UNet execution. |
| `reinforcement-learning` | Preserves historical MLPerf MiniGo while the full self-play contract remains impractical locally. | Search-coupled inference, dynamic data generation, and iterative training. |

The first spiral makes every workload runnable through one CLI with reports and
provenance. The current quality-evidence spiral retains twelve cases for the
original nine workloads. Once all fourteen workloads pass their authoritative
quality contracts, one canonical `max` case per workload plus the three extra
causal inference phases will produce seventeen evidence cases. Optimization
choices are configurations, not additional workloads.

## Spiral Delivery Model

1. Functional integration proves that the execution path, report, and
   provenance work without making a quality or timing claim.
2. Quality conformance binds the authoritative model, dataset, evaluator, and
   published target.
3. Stabilization establishes fresh-process repeatability and controls runtime
   variance.
4. Promotion imports one complete source-locked evidence set and enables a
   public baseline only after review.

## Profiles

`min` is the fast functional path. `max` is the canonical classroom and
comparison path. `pro` is the single-node research envelope. The research
profile exists so architecture, runtime, compiler, memory, precision, and
scheduling studies can remain comparable to the same task contract.

## Evidence Discipline

Every promoted case uses five fresh processes at the canonical seed. Every
quality or functional gate must pass, and timing CV must remain within 5%.
The case summary records the complete run set, source revision, comparison
fingerprint, metrics, acceptance decision, and artifact index.

Causal inference phases add a stronger lineage rule. All phases must use one
verified package selecting the median-quality committed training run. This
prevents phase comparisons from silently changing model weights.

The committed index is the source for exact baseline values. Public documents
should not copy mutable result tables by hand.

The current draft index keeps evidence classes explicit. Six cases satisfy the
five-run project quality and repeatability checks. Five cases have one verified
measurement, and causal training has two quality-passing measurements with a
5.19% diagnostic timing CV. Those six cases remain provisional and make no
repeatability or promoted-baseline claim.

## Research Boundary

The suite can support controlled research on:

- Processor and accelerator behavior
- Memory hierarchy and sparse access
- Compiler and graph transformations
- Runtime and kernel selection
- Precision and quantization
- Batch, context, and scheduling configurations
- Training-to-inference lineage

Distributed scaling, datacenter serving claims, agent capability evaluation,
and large-model system claims remain outside v0.1.

## Functional-Stage Reinforcement Learning

MiniGo is the historically correct MLPerf reinforcement-learning workload.
The public CLI now exercises a bounded policy-value self-play and training
path, but the workload remains outside promotion because no authoritative
laptop-scale configuration preserves its complete self-play and checkpoint
quality contract unchanged. A control-environment substitute would create a
different benchmark and weaken the admission rule.

## Requested Review

The first review should focus on five questions:

1. Does the fourteen-workload portfolio, including its nine-workload promotion scope and five functional-stage additions, cover enough distinct single-node behavior for a v0.1 classroom suite?
2. Are the inherited quality targets and laptop execution boundaries defensible?
3. Does the mode, phase, configuration, and profile taxonomy match mature benchmark practice?
4. Are five-run repeatability, provenance, and disclosure rules sufficient for initial comparison?
5. Which naming, governance, licensing, and publication steps are required before any MLCommons association?

Implementation readiness and external governance are separate. The release
checklist reports both without treating pending governance as a technical pass.
