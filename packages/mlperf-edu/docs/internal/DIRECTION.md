# MLPerf EDU Direction

> MLPerf EDU is a locally executable, quality-gated benchmark specification for
> teaching and studying single-node ML systems. It transfers the
> reproducibility, verification, disclosure, and comparability discipline of
> mature benchmark suites to classroom-scale PyTorch workloads. It supports
> controlled research on processors, memory systems, runtimes, compilers, and
> model execution while explicitly excluding distributed and datacenter-scale
> claims.

This is an independent project. It is not an official MLCommons benchmark and
is not endorsed by MLCommons.

This document states why the project exists and what it refuses to do. It
carries no counts and no results. Current status lives in
[STATUS](STATUS.md), the plan of record lives in [SHIP_PLAN](SHIP_PLAN.md), and
the normative rules live in [SPEC](../../SPEC.md).

## The Problem

Machine-learning systems courses need workloads small enough to run locally,
transparent enough to teach, and disciplined enough to support reproducible
comparison. Small teaching examples usually omit quality gates, measurement
boundaries, provenance, and stable task definitions. Production benchmark
suites supply that governance but are operationally too heavy for a notebook or
a single class period.

MLPerf EDU occupies the space between those extremes. It packages established
task definitions behind one PyTorch CLI and one artifact contract. It does not
create new learning tasks to expand coverage.

## Design Commitments

### Curate Rather Than Invent

The suite begins from an authoritative upstream workload that supplies the
task, model or reference implementation, dataset and split, evaluator, quality
contract, and credible baseline. MLPerf EDU adds only the execution adapter,
declared measurement protocol, quality gate, provenance, and report surface
needed to run that contract.

A missing upstream component is not an invitation to substitute a convenient
one. A bounded functional probe may establish integration plumbing, but it
stays experimental and outside promotion until the authoritative contract
executes unchanged. This is why MiniGo remains the reinforcement-learning
identity rather than being replaced by a control task.

### Design Backward From Use

A student installs one locked environment, inspects a workload, fetches pinned
assets, confirms the path with a `min` run, executes the canonical `max` task
and quality gate, and hands an instructor a reviewable artifact. A researcher
alters a controlled single-node configuration without losing task identity or
lineage.

That workflow is what requires stable workload identity, explicit modes and
phases, quality-gated timing, portable provenance, and fail-closed public
claims.

### Keep Workload Identity Stable

A workload ID names the learning task. Training and inference are modes. Full,
prefill, and decode are phases. Precision, quantization, compilation, batching,
context length, scheduling, and serving behavior are configurations that appear
in reports, never in workload IDs.

### Gate Performance With Quality

A fast invalid model is not a benchmark result. Every score-bearing case passes
its inherited task-quality contract before its timing is interpreted. Every
performance-bearing phase passes a functional contract and inherits the
required model lineage. No aggregate may hide a failed individual run.

### Separate Profile Intent From Hardware Envelope

A profile defines the depth of the benchmark contract, not a machine size.

| Profile | Design intent |
|:---|:---|
| `min` | Fast setup, teaching, and CI confidence. Never a public score. |
| `max` | The canonical classroom and comparison contract, unchanged. |
| `pro` | The extended single-node research envelope under the same identity. |

Every `min` path runs on classroom hardware while preserving enough identity to
verify setup, execution, reporting, and provenance. A `max` path runs the
unchanged authoritative contract. Most fit a laptop; the ones that currently do
not are recorded in [STATUS](STATUS.md) as gaps to close rather than as
accepted exceptions. `pro` adds research controls without silently changing the
task, data, evaluator, or target.

### Measure the Declared Boundary

Asset fetching, model construction, and untimed warmup stay outside the
measured region unless an upstream contract includes them. Accelerator
measurements synchronize at each boundary. Reference runs record power source
and power mode, and an intervening sleep or power-state change invalidates the
attempt.

Optional power data is coarse platform telemetry. A roofline claim needs a
measured, digest-checked sidecar. Missing information stays `unmeasured` rather
than being inferred from an architecture name.

### Treat Reports as the Interface

Console output is transient. A registered run writes structured JSON, a flat
CSV view, a human-readable HTML report, and a provenance manifest that binds
the report and retained inputs with SHA-256.

That manifest detects change; it does not authenticate the producer. Portable
packages use relative paths and re-verify every included byte after clean
extraction. Independent reproduction remains necessary.

Because measurements belong to the run that produced them, the website and
these documents carry no results. The report does.

### Preserve Training Lineage

Causal language modeling keeps training, full inference, prefill, and decode
under one identity. Canonical inference requires a checkpoint from a passing
canonical training run, and reports record the checkpoint, source report,
source manifest, and package digests so a serving result cannot silently use
unrelated weights. Other inference workloads use pinned authoritative
checkpoints and record the exact revision, model files, dataset files, and
evaluator contract.

### Keep the Scale Honest

The standard path targets CPU and laptop accelerators and supports controlled
studies of processors, memory systems, runtimes, compilers, and model
execution. It does not represent distributed training, datacenter serving,
cluster scheduling, or fleet economics.

Local execution does not mean zero downloads or identical runtimes everywhere.
Canonical workloads may fetch substantial assets and take tens of minutes. The
project publishes observed hardware and runtime evidence rather than promising
a universal duration.

## Admission Test

A workload may enter the functional stage once questions 1, 2, 5, 6, and 7 are
answered. It may enter quality conformance and the promotion evidence scope
only when every question is answered.

1. Is the task significant and established?
2. Is the upstream model or implementation authoritative?
3. Are the dataset, split, evaluator, metric, and target fixed upstream?
4. Can the unchanged contract run credibly on laptop-class hardware?
5. Does it add distinct classroom value?
6. Does it expose distinct single-node systems behavior?
7. Can all assets, versions, hashes, and adaptations be disclosed?
8. Can the unchanged canonical `max` path pass its quality or functional gate
   on laptop-class hardware?

The portfolio and each workload's rationale are in
[SPEC](../../SPEC.md); the selection ledger records every accepted, deferred,
and rejected proposal with reasons.

## Delivery Stages

1. **Functional integration** proves execution, reporting, and provenance work
   while explicitly withholding quality and timing claims.
2. **Quality conformance** binds the authoritative model, dataset, evaluator,
   and published target.
3. **Stabilization** establishes fresh-process timing repeatability.
4. **Promotion** imports one complete, source-locked evidence set after review.

Each stage is monotonic. A later stage adds evidence; it never relabels an
earlier probe as though it had satisfied the stronger contract.

## Research Boundary

Supported: processor and accelerator behavior, memory hierarchy and sparse
access, compiler and graph transformations, runtime and kernel selection,
precision and quantization, batch/context/scheduling configurations, and
training-to-inference lineage.

Outside v0.1: distributed scaling, datacenter serving claims, agent capability
evaluation, and large-model system claims.

## Open Review Questions

1. Does the portfolio cover enough distinct single-node behavior for a v0.1
   classroom suite?
2. Are the inherited quality targets and laptop execution boundaries
   defensible?
3. Does the mode, phase, configuration, and profile taxonomy match mature
   benchmark practice?
4. Are the repeatability, provenance, and disclosure rules sufficient for
   initial comparison?
5. Which naming, governance, licensing, and publication steps are required
   before any MLCommons association?

## Success Criteria

MLPerf EDU v0.1 succeeds when an ML systems instructor can use it without
explaining away synthetic scores, arbitrary targets, broken setup, or opaque
provenance, and when a systems researcher can reproduce a local result without
guessing which task or weights were measured.

Technical readiness and governance stay separate. A green run cannot settle
dataset rights, grant endorsement, or authenticate a producer. Initial
MLCommons review is the governance milestone; it is never implied by the
project name alone.
