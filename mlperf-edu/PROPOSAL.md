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

The proposed v0.1 portfolio contains seven workloads and ten evidence cases.

| **Workload** | **Reason for Inclusion** | **Distinct Systems Value** |
|:---|:---|:---|
| `image-classification` | Directly inherits the MLPerf Tiny ResNet8 definition. | Dense convolution, input layout, and offline batching. |
| `keyword-spotting` | Directly inherits the MLPerf Tiny DS-CNN definition. | Depthwise convolution and latency-sensitive small tensors. |
| `causal-language-modeling` | Uses the established nanoGPT Shakespeare recipe without reducing it. | Transformer training, full inference, prefill, decode, and checkpoint lineage. |
| `text-classification` | Uses a pinned published DistilBERT SST-2 checkpoint. | Encoder attention, tokenization, padding, and batching. |
| `information-retrieval` | Reproduces the documented CrossEncoder NanoBEIR example. | Query-document pair scoring and ranking. |
| `graph-node-classification` | Uses the official OGB GCN recipe and evaluator. | Sparse gather, scatter, and irregular memory access. |
| `time-series-forecasting` | Uses the official PatchTST ETTm1 recipe and split. | Patch extraction, long-context attention, and multivariate sequence training. |

One canonical `max` case is required for every workload. The causal workload
adds full, prefill, and decode inference, producing ten evidence cases in
total. Optimization choices are configurations, not additional workloads.

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

## Deferred Reinforcement Learning

MiniGo is the historically correct MLPerf reinforcement-learning workload.
The project defers it because its self-play and checkpoint quality contract is
not preserved by any authoritative laptop-scale configuration we could adopt
unchanged. A control-environment substitute would create a different
benchmark and weaken the admission rule.

## Requested Review

The first review should focus on five questions:

1. Does the seven-workload portfolio cover enough distinct single-node behavior for a v0.1 classroom suite?
2. Are the inherited quality targets and laptop execution boundaries defensible?
3. Does the mode, phase, configuration, and profile taxonomy match mature benchmark practice?
4. Are five-run repeatability, provenance, and disclosure rules sufficient for initial comparison?
5. Which naming, governance, licensing, and publication steps are required before any MLCommons association?

Implementation readiness and external governance are separate. The release
checklist reports both without treating pending governance as a technical pass.
