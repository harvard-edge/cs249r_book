# MLPerf EDU Initial Usability Readiness

## Scope

This checklist tracks the first usable version of all fourteen workloads. The
first milestone requires every authoritative quality path to run on a supported
local machine. One complete run is enough to evaluate a quality target. The
milestone does not require five timing repetitions, a promoted performance
baseline, or production publication. The implementation plan is maintained in
[LOCAL_EXECUTION_PLAN.md](LOCAL_EXECUTION_PLAN.md).

The existing provisional importer contains twelve evidence cases across the
original quality scope. That evidence scope contains nine workloads. The five
quality-conformance workloads remain outside that importer until their result
gates close deliberately.

A workload is initially usable when its CLI path, exact assets, quality target,
evaluator, report, and provenance boundary are explicit and its authoritative
path can complete without remote compute. A functional `min` probe establishes
setup only. It never counts as quality evidence. An environment handoff is
useful diagnostic output, but it does not satisfy the local milestone.

## Portfolio Status

Every workload has a `min` runner, an authoritative `max` runner, a target with
an upstream basis, and a one-result acceptance contract. Twelve authoritative
paths run locally. Eight workloads have at least one complete target-passing
result, four record an honest quality gap, and two still require local executor
work.

| **Workload** | **Quality Target** | **Initial Quality Status** | **Next Work** |
|:---|:---|:---|:---|
| `image-classification` | top-1 accuracy ≥ 0.85 | Pass at 0.870000 | Keep the current result. Stability is later work. |
| `keyword-spotting` | top-1 accuracy ≥ 0.90 | Pass at 0.902000 | Keep the initial result. The quality-preserving but nonidentical adapter remains blocked from public promotion. |
| `anomaly-detection` | ROC AUC ≥ 0.85 | Pass at 0.902910 | Keep the current result. Stability is later work. |
| `visual-wake-words` | top-1 accuracy ≥ 0.80 | Pass at 0.851000 | Keep the current result. Stability is later work. |
| `causal-language-modeling` | validation cross-entropy ≤ 1.4697 | Pass with a two-run median of 1.458998 | Keep one passing training result and its checkpoint-backed inference lineage. |
| `text-classification` | accuracy ≥ 0.9105504587 | Pass at 0.9105504751 | Keep the current result. Stability is later work. |
| `information-retrieval` | mean nDCG@10 ≥ 0.6071684099 | Pass at 0.6071684099 | Keep the current result. Stability is later work. |
| `graph-node-classification` | test accuracy ≥ 0.7174 with 0.0029 tolerance | Pass at 0.720964 | Keep the current result. Domain review of the tolerance remains a publication gate. |
| `time-series-forecasting` | test MSE ≤ 0.290 | Complete result missed at 0.292393 | Investigate the gap without transferring an unrelated MLPerf Inference tolerance to the published PatchTST result. |
| `code-generation` | HumanEval+ pass@1 ≥ 0.573, or at least 94 of 164 tasks | Complete result missed at 91 of 164, or 0.554878 | Investigate the three-task gap without weakening the target. |
| `function-calling` | BFCL V4 Non-Live AST accuracy ≥ 0.8292 | Complete provenance-bound 1,150-case result missed at 0.785208 | Review the category gaps without weakening the target or changing the pinned evaluator. |
| `image-generation` | minimum CIFAR-10 FID ≤ 1.79 across three trials | Complete provenance-bound packet rescored at 1.801554 | Review the narrow gap without weakening the target, reducing the image count, or changing the sampler. |
| `recommendation` | Criteo Terabyte ROC AUC ≥ 0.8025 | Reference runner ready; local execution blocked | Add a verified manual-data journey and an out-of-core CPU backend that preserves the full accuracy contract. |
| `reinforcement-learning` | professional-move prediction ≥ 0.40 plus the 0.55 playoff gate | Reference runner ready; local execution blocked | Add a native CPU or MPS backend that preserves MiniGo semantics and both quality gates. |

## Model and Checkpoint Lineage

The suite does not force every workload into an artificial train-then-infer
shape. Its lineage falls into three categories.

- `causal-language-modeling` trains a checkpoint in the suite and runs full,
  prefill, or decode inference from that exact checkpoint.
- `graph-node-classification`, `time-series-forecasting`, and
  `reinforcement-learning` are training benchmarks whose quality evaluation is
  part of the training run.
- The remaining workloads are fixed-checkpoint inference benchmarks. Their
  reports identify the upstream training authority and pinned checkpoint, but
  they do not claim that training occurred during the local run.

The HTML dashboard exposes this distinction as a Training → Checkpoint →
Inference → Evaluation chain. The JSON report carries the same normalized
`execution_lineage` record.

## Dataset and Model Readiness

The benchmark choices, splits, and evaluator inputs are explicit for all
fourteen workloads. Dataset correctness and dataset redistribution are separate
questions. A release-policy review does not mean the selected dataset is wrong.

- Tiny Shakespeare, HumanEval+, the ToyADMOS anomaly set, and the local prompt
  suite are approved for fetch-only or bundled use under their current policy.
- OGB `ogbn-arxiv` is approved for attributed ODC-By 1.0 fetch-only use.
- CIFAR-10, SST-2, NanoBEIR, ETTm1, KWS, visual wake words, BFCL, and the MiniGo
  inputs are pinned and runnable, but their public redistribution or release
  wording still needs review.
- Criteo Terabyte requires manual upstream terms acceptance. The CLI does not
  download or redistribute it and fails closed until the user supplies the
  complete preprocessed accuracy set.

Model sources and revisions are pinned where a fixed checkpoint is required.
DLRM keeps its large checkpoint external. MiniGo and the three local training
benchmarks generate checkpoints during execution. Reports record the relevant
source, revision, path, digest, or checkpoint manifest without treating a
pretrained model as locally trained.

## Student Journey

1. Install the locked environment and run `uv run mlperf health`.
2. Inspect available work with `uv run mlperf list` and `uv run mlperf show`.
3. Run a quick `min` probe to confirm the machine and code path work.
4. Check the workload's declared execution envelope, then fetch and verify the
   exact `max` assets before measurement.
5. Run one `max` workload. The CLI generates an HTML dashboard plus JSON and
   CSV siblings without opening a browser.
6. Read the lead quality card, target decision, run configuration, hardware,
   model lineage, and provenance sections.
7. Verify the `.provd.json` manifest and package the result when an instructor
   or collaborator needs a portable artifact.
8. Use `pro` for the research collection or select an individual workload.
   The default is one max-contract execution. Optional repetition belongs to
   the later stability phase.

Every `min` path is intended for classroom hardware. Twelve `max` paths are
already laptop-capable. DLRM and MiniGo currently fail closed when the required
data, memory, legacy runtime, or GPU is unavailable. They are remaining
milestone blockers rather than accepted research-environment exceptions.

```bash
uv run mlperf doctor
uv run mlperf health
uv run mlperf run --workload image-classification --profile min
uv run mlperf fetch --workload image-classification --profile max
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/image-classification
uv run mlperf verify \
  submissions/image-classification/image-classification_max.provd.json
uv run mlperf run --profile pro --dry-run
```

No browser flag is needed in a headless environment. Use `--open-report` only
when the dashboard should open immediately. `--no-open-report` and
`MLPERF_EDU_NO_BROWSER=1` remain available for explicit automation policy.

## Remaining Initial-Usability Work

- [x] Register all fourteen workload identities and quality targets.
- [x] Set `acceptance_runs: 1` for the initial quality phase.
- [x] Provide `min`, `max`, and pro fallback execution through one CLI.
- [x] Pin or fail closed on every model, dataset, evaluator, and source input.
- [x] Emit JSON, CSV, an HTML dashboard, and provenance for each run.
- [x] Generate every dashboard by default and open it only with an explicit
  `--open-report` request.
- [x] Provide one `mlperf health` command that checks all fourteen min paths,
  verifies their manifests, and writes the suite report without opening it.
- [x] Preserve pro subrun configuration and artifacts in the aggregate report.
- [x] Generate a benchmark page for every workload from the registry.
- [x] Pass desktop and mobile layout checks for all 40 website pages.
- [x] Produce and verify the complete current BFCL artifact.
- [x] Produce and verify the complete current three-trial EDM artifact.
- [x] Publish machine-readable DLRM and MiniGo environment handoffs.
- [ ] Implement and parity-test an out-of-core local DLRM backend.
- [ ] Implement and parity-test a native CPU or MPS MiniGo backend.
- [ ] Complete a clean-cache asset audit for all fourteen workloads.
- [ ] Execute the DLRM quality contract on the supported local machine.
- [ ] Execute the MiniGo quality contract on the supported local machine.
- [ ] Populate measured working-set, arithmetic-intensity, and dispatch evidence
  for research interpretation after the authoritative max contracts settle.
- [x] Complete desktop and narrow-viewport visual review of the standalone run
  dashboard.
- [ ] Resolve any target gap without lowering a target to fit a local result.

## Production Readiness

Production publication is not required for the local-execution milestone. The
current product remains an experimental source-checkout preview.
A production release would additionally require the following work.

- [ ] Adopt an authoritative component license and package-index versioning
  policy.
- [ ] Close MLCommons review of the name, scope, governance, and result wording.
- [ ] Close dataset redistribution and fetch-only decisions.
- [x] Document the generated-code, trusted-pickle, and legacy-runtime security
  boundaries in [SECURITY_REVIEW.md](../../SECURITY_REVIEW.md), and harden the local
  EvalPlus and EDM preview paths.
- [ ] Replace executable EDM pickle inputs with reviewed safe derived
  artifacts and qualify independently built, signed DLRM and MiniGo runtimes.
- [ ] Produce signed release artifacts and authenticated provenance when
  producer identity matters.
- [ ] Reproduce the suite on independent CPU, Apple Silicon, and CUDA systems.
- [ ] Complete the later stability campaign and promote only evidence that
  meets its timing-variation contract.

The public-contract audit intentionally remains nonzero while every workload is
experimental. That policy result does not mean the CLI or an individual run is
broken.
