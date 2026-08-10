# MLPerf EDU v0.1 Release Checklist

## Release Definition

The review candidate registers fourteen workloads. The current quality-evidence
scope covers nine workloads and twelve evidence cases. Evidence is bound to clean source revision
`163d42ee3df54ab122543469ccf2b6b3bd119455`. A checked box requires direct
command output or a committed content-addressed artifact. Intent, partial runs,
and narrow smoke checks do not prove a broad release claim.

## Portfolio and Architecture

- [x] `image-classification` inherits the MLPerf Tiny ResNet8 contract.
- [x] `keyword-spotting` inherits the MLPerf Tiny DS-CNN contract.
- [x] `anomaly-detection` inherits the MLPerf Tiny ToyCar autoencoder contract.
- [x] `visual-wake-words` inherits the MLPerf Tiny MobileNetV1 0.25 contract.
- [x] `causal-language-modeling` preserves nanoGPT training and inference under one identity.
- [x] `text-classification` uses the pinned DistilBERT SST-2 checkpoint and split.
- [x] `information-retrieval` reproduces the documented CrossEncoder NanoBEIR example.
- [x] `graph-node-classification` uses the official OGB GCN recipe and evaluator.
- [x] `time-series-forecasting` uses the official PatchTST ETTm1 recipe and split.
- [x] `code-generation` has a complete HumanEval+ runner and a target-missing 91-of-164 result.
- [x] `function-calling` has a complete 1,150-case BFCL runner; the current full artifact remains pending.
- [x] `recommendation` runs the MLPerf Training v0.5 NCF contract on MovieLens-20M locally in roughly half an hour and records a measured miss against the published 0.635 HR@10.
- [x] `image-generation` has a complete 50,000-image EDM FID runner; the current full artifact remains pending.
- [x] `reinforcement-learning` has a complete resumable MiniGo quality loop that fails closed without its reviewed legacy GPU environment.
- [x] Modes, phases, configurations, scenarios, and profiles are separate taxonomy axes.
- [x] MiniGo remains the RL reference without substituting a control task.
- [x] Functional-stage candidates retain their published targets and explicit conformance blockers in the selection ledger.

## Evidence Cases

The current importer remains scoped to the original nine quality-evidence
workloads and the following twelve evidence cases. The five quality-conformance
workloads cannot enter this closure until complete results pass their
authoritative gates and the importer is expanded deliberately.

- [x] `image-classification__max__inference` — repeated-timing
- [x] `keyword-spotting__max__inference` — repeated-timing
- [x] `anomaly-detection__max__inference` — repeated-timing
- [x] `visual-wake-words__max__inference` — repeated-timing
- [x] `causal-language-modeling__max__training` — two-run provisional
- [x] `causal-language-modeling__max__inference__full` — one-run provisional
- [x] `causal-language-modeling__max__inference__prefill` — one-run provisional
- [x] `causal-language-modeling__max__inference__decode` — one-run provisional
- [x] `text-classification__max__inference` — repeated-timing
- [x] `information-retrieval__max__inference` — repeated-timing
- [x] `graph-node-classification__max__training` — one-run provisional
- [x] `time-series-forecasting__max__training` — one-run provisional

These boxes mean that class-labeled draft evidence exists for every case. They
do not erase the evidence-class labels or promote a provisional record. The authoritative draft closure is the
twelve-entry `provisional_results/index.json`, with six repeated-timing
records and six provisional records. Full promotion still requires a complete
twelve-entry `reference_results/index.json` whose summaries all pass
acceptance, repeatability, digest, source-lock, provenance, and lineage
verification.

## Current-Code Diagnostic Audit

The following single-run audit was completed on 2026-07-15 against Git HEAD
`8fd1032fc918938d9acc0a8094b10f8cef492250` plus the recorded working-tree
patch. All twelve cases passed and every generated provenance manifest verified
before this checklist update. These runs establish current-code functional and
quality continuity. They are not promotion evidence because they did not start
from a clean release commit and the long runs were executed on battery or
crossed a battery-to-AC transition.

| Evidence case | Device | Observed result | Canonical gate | Quality margin | Provenance | Promotion state |
|---|---|---:|---:|---:|---|---|
| `image-classification__max__inference` | CPU | 0.870000 top-1 | >= 0.850000 | +0.020000 | verified | Clean current-source repeated-timing sweep retained; promotion import awaits portfolio closure |
| `keyword-spotting__max__inference` | MPS | 0.902000 top-1 | >= 0.900000 | +0.002000 | verified | Adapter is quality-preserving but nonidentical; promotion is blocked |
| `anomaly-detection__max__inference` | MPS | 0.902910 ROC AUC | >= 0.850000 | +0.052910 | verified | Clean current-source repeated-timing sweep retained; strongest converted-model metric reproduction |
| `visual-wake-words__max__inference` | MPS | 0.851000 top-1 | >= 0.800000 | +0.051000 | verified | Clean current-source repeated-timing sweep retained; exact top-1 parity passes all three audited LiteRT resolvers |
| `causal-language-modeling__max__training` | MPS | 1.458786 loss | <= 1.469700 | +0.010914 | verified | Three diagnostic runs now pass; clean repeated timing campaign required |
| `causal-language-modeling__max__inference__full` | CPU | 884.48 output tokens/s and 64 decode steps | functional gate | pass | verified with training lineage | Clean repeated-timing campaign required |
| `causal-language-modeling__max__inference__prefill` | CPU | 30,280.26 prefill tokens/s | functional gate | pass | verified with training lineage | Clean repeated-timing campaign required |
| `causal-language-modeling__max__inference__decode` | CPU | 879.88 output tokens/s and 64 decode steps | functional gate | pass | verified with training lineage | Clean repeated-timing campaign required |
| `text-classification__max__inference` | MPS | 0.910550475 accuracy | >= 0.910550459 | +0.000000016 | verified | Clean current-source repeated-timing sweep retained; exact pinned-checkpoint conformance gate |
| `information-retrieval__max__inference` | MPS | 0.607168410 nDCG@10 | >= 0.607168410 | 0.000000000 | verified | Clean current-source repeated-timing sweep retained; exact published-example conformance gate |
| `graph-node-classification__max__training` | MPS | 0.722342 accuracy | >= 0.717400 | +0.004942 before tolerance | verified | Clean repeated-timing accuracy and timing distribution required |
| `time-series-forecasting__max__training` | MPS | 0.292393 MSE | <= 0.290000 | -0.002393 | verified | Historical artifact no longer passes after removing the unsupported derived gate; investigate the quality gap before rerunning |

Diagnostic artifacts are under
`/tmp/mlperf-edu-current-audit-20260715`. The strict two-runtime adapter audit
is `tflite-adapter-parity.json` with SHA-256
`9385d716ebf4c826064f0a19c7904ef158dc7020b60d5d5487e6060792066b56`.
The long-run timings are intentionally
excluded from baseline claims. PatchTST took 1,959.45 seconds on battery,
graph training took 1,628.46 seconds on battery, and nanoGPT took 2,426.35
seconds across a battery-to-AC transition.

### Clean Current-Source Campaign

Five short cases completed methodology-valid promotion-candidate sweeps on
2026-07-17 against clean Git revision
`f9e0c61da296d3b92a4705503d4bc6988c16c01c`. Each case ran in repeated fresh
processes on AC power with Low Power Mode disabled, the canonical seed of 42,
and the registry-defined inter-execution cooldown. Every run passed its quality
gate, and every timing coefficient of variation stayed below the 5% limit.

| Evidence case | Device | Quality median | Timing CV | Summary SHA-256 |
|---|---|---:|---:|---|
| `image-classification__max__inference` | CPU | 0.870000000 top-1 | 2.402% | `6550866586b395d3b506dacac5774c4901ab39dbed3b08763adfaa19d8bbdc80` |
| `anomaly-detection__max__inference` | MPS | 0.902910053 ROC AUC | 4.308% | `b2de2e91a6490f18588622cadbbedf4b611f2dce3acd2cac4e80003566ec1a16` |
| `visual-wake-words__max__inference` | MPS | 0.851000011 top-1 | 0.864% | `bd90c6ba9b60728cf05c32b940b544567e01c3144a8aeed56860b775113e7e3d` |
| `text-classification__max__inference` | MPS | 0.910550475 accuracy | 2.227% | `e96d897ab4c29f2b126348bf2bf044fea89b360a29063ead96c20bd49ef9114f` |
| `information-retrieval__max__inference` | MPS | 0.607168410 nDCG@10 | 1.899% | `65e9f3ffdda10d0590a8ad0d9f97ddd15a5a982265fdc7f2cf185f4351760950` |

The raw attempts are retained outside the repository under the local review
handoff path `f9e0c61-20260717/promotion-evidence/`. This partial campaign does
not satisfy the strict twelve-case promotion import and does not expose a
public baseline. Keyword spotting was intentionally omitted because its
nonidentical adapter blocks promotion until the execution-path issue is
resolved.

### Accuracy and Runtime Risks

- [x] All twelve current-code cases pass one canonical measurement.
- [x] The six fragile or converted fast workloads reproduce their draft quality values exactly.
- [x] Current nanoGPT training produces a quality-approved checkpoint that all three inference phases verify and consume.
- [x] Current graph training passes the nominal published target without using its tolerance.
- [x] Current time-series training records a reproducible 0.292393 MSE gap against the published 0.290 point.
- [x] Independently compare full-set PyTorch and pinned TFLite outputs for ResNet8, DS-CNN, and MobileNetV1 under LiteRT 2.1.6 XNNPACK and builtin kernels.
- [x] ResNet8 and MobileNetV1 produce identical top-1 predictions on every official sample under both audited LiteRT resolvers and reproduce the same 87.0% and 85.1% accuracy.
- [x] The KWS divergence is measured rather than hidden: PyTorch is 90.2%; LiteRT XNNPACK is 90.0% with 7/1,000 prediction disagreements; LiteRT builtin is 90.5% with 5/1,000 disagreements. Every path passes the inherited 90% gate.
- [x] Resolve the KWS promotion choice. Retain and disclose the quality-preserving PyTorch adaptation for education, classify it as nonidentical, and block promotion until exact-source execution or authoritative unquantized weights establish exact parity. Do not invent a disagreement tolerance.
- [ ] Isolate PatchTST data-order RNG from data-loader worker lifecycle before changing worker persistence or count.
- [ ] Obtain five clean externally powered runs for graph, time-series, nanoGPT training, and all nanoGPT inference phases.
- [ ] Have domain reviewers approve the graph mean-plus-tolerance rule and decide whether PatchTST needs a predeclared conformance tolerance.
- [ ] Decide whether exact fixed-model scores should remain the sole conformance gate or be paired with a separately labeled task-quality floor.
- [ ] Commit methodology-valid system-characterization sidecars for all fourteen workloads; working set, arithmetic intensity, and dispatch remain explicitly unmeasured today.

### MLPerf Tiny Adapter Audit

The strict command intentionally returns status 1 because the KWS adapter is
classified as quality-preserving but nonidentical. The committed audit is
`conformance_results/tflite-adapter-parity-20260717.json` with SHA-256
`524d183cc858cb3c7911e55f9187df40d67bd5bc546a7972ef131fc76192adb7`.
It covers XNNPACK, builtin, and builtin-reference execution from clean source
commit `b408be80350e248eed25c11966c4844bfccc15d8`. A new KWS timing campaign is
not warranted until the conformance blocker changes.

```bash
uv run --extra parity python tools/audit_tflite_adapter_parity.py \
  --resolver auto --resolver builtin \
  --output /tmp/mlperf-edu-tflite-adapter-parity.json
```

## Reference Campaign Command

Run one complete case per attempt. Use AC power, disable Low Power Mode, prevent
sleep, and avoid concurrent heavy workloads.

```bash
uv run python tools/run_reference_sweep.py \
  --workload WORKLOAD --profile max --mode MODE [--phase PHASE] \
  --runs 5 --device DEVICE --evidence-tier promotion-candidate \
  --output-dir /tmp/mlperf-edu-promotion
```

Use the registry-defined inter-execution cooldown. Do not shorten it for a
promotion-candidate campaign.

For causal inference, add the verified package selecting the median-quality
training run.

```bash
--nanogpt-lineage-package /path/to/causal-training-package.zip
```

## Evidence Import and Baseline Synchronization

```bash
uv run python tools/import_provisional_reference_results.py --check \
  --promotion-evidence-root /path/to/promotion-evidence \
  --provisional-evidence-root /path/to/provisional-evidence \
  --causal-training-attempt-root /path/to/causal-training-attempt \
  --causal-training-package /path/to/causal-training-package.zip \
  --source-git-sha 163d42ee3df54ab122543469ccf2b6b3bd119455
```

- [x] Draft importer independently verifies all retained reports, manifests,
  source closure, evidence classes, and the causal package.
- [x] Strict importer rejects missing, duplicate, stale-source, interrupted, or
  high-CV promotion cases.
- [x] All three draft causal phases share one verified provisional training lineage.
- [x] Draft source and wheel-resource mirrors match exactly for all twelve cases.
- [x] Raw checkpoints and dataset-derived bytes remain outside Git.
- [ ] A complete twelve-case promotion import exists.
- [ ] Promoted native registry baselines bind every case ID and evidence digest.

## CLI and Profile Coverage

```bash
uv run mlperf doctor
uv run mlperf audit --policy public  # expected status 1 while the portfolio is experimental
uv run mlperf validate smoke --output-dir /tmp/mlperf-edu-smoke
uv run mlperf validate coverage --output-dir /tmp/mlperf-edu-coverage
uv run mlperf validate pro --output-dir /tmp/mlperf-edu-pro
```

The audit's nonzero status is an expected policy block until promotion; it is
not an execution failure. The workflow records and checks that distinction.

- [x] Every registered `min` path passes from the public CLI.
- [x] Every canonical `max` path resolves to its authoritative runner through the public CLI.
- [x] Environment-gated `max` paths fail closed with actionable asset, terms, runtime, and hardware requirements.
- [x] Every applicable `pro` path passes and retains workload identity with one canonical measurement by default.
- [x] Explicit mode and phase selection works and invalid combinations fail early.
- [x] Fetch runs before measurement and verifies pinned assets.
- [x] JSON, CSV, an HTML dashboard, and provenance files are emitted for every completed run.

## Platform Coverage

- [x] Clean CPU install and functional execution pass.
- [x] Apple Silicon MPS execution passes where supported.
- [x] Unsupported backends fail with an actionable message.
- [x] Every report records `device_requested`, `device_executed`, and the
  executed backend.
- [x] `run` and `validate` expose `--device auto|cpu|cuda|mps`, and every
  PyTorch max runner shares the same CUDA-then-MPS-then-CPU auto policy.
- [x] Source checkouts preserve their local asset layout, installed wheels use
  a stable per-user cache, and `MLPERF_EDU_DATA_DIR` overrides both.
- [x] Power source and Low Power Mode are disclosed for reference campaigns.
- [x] Sleep or power-state changes invalidate affected attempts.

## Automated Validation

```bash
uv run pytest
uv run python tools/export_flat_registry.py --check
uv run python tools/sync_verified_baselines.py --check
uv run python tools/check_taxonomy.py
uv run python tools/check_reference_claims.py --check
uv run python tools/generate_docs.py --check
uv run make -C paper clean all check
quarto render site
uv run python tools/check_site_layout.py --build-dir site/_build --report-dir site-layout-report
uv build
```

- [x] Full Python test suite passes.
- [x] The maintainer audit exposes draft evidence class, run count, evidence
  hash integrity, quality margin, timing CV, and review eligibility.
- [x] Provenance tamper and clean-extraction package tests pass.
- [x] Registry, evidence, documentation, and paper drift checks pass.
- [x] All MLPerf EDU GitHub workflows contain no retired workload assertions.
- [x] Wheel includes registry, dataset, and twelve-case draft evidence resources.
- [x] Wheel installs and runs outside the checkout with unlocked dependencies
  under every declared Python version: 3.10, 3.11, and 3.12.
- [x] CI repeats the clean-wheel list, evidence-audit, stable-cache, and CPU
  workload checks across Python 3.10, 3.11, and 3.12.
- [x] Installed CLI can list, audit, run a smoke path, report, and verify a package.
- [x] Quarto site renders and internal links pass.
- [x] Paper builds without missing references, placeholders, or layout overflows.

## Documentation and Cleanup

- [x] Generated benchmark pages expose exactly fourteen workload pages and seven suite indexes.
- [x] Retired generated workload pages are removed.
- [x] README, specification, proposal, public rules, target review, dataset review, and paper agree.
- [x] Website and paper display exact committed evidence values from generated sources.
- [x] Review-packet generation is registry-filtered and currently emits no packets while every workload remains experimental.
- [x] Obsolete generated workload pages and retired public assertions are removed.
- [x] No cache, build output, checkpoint, dataset-derived artifact, or AI configuration is committed.

Cleanup must preserve evidence. Rejected raw attempts may be moved into a
local review archive with a reason and timestamp. They must not be deleted to
make the release appear cleaner.

## External Decisions

These items cannot be closed by repository tests:

- [ ] MLCommons reviews the name, scope, result wording, and governance path.
- [ ] Domain reviewers accept or revise each quality and measurement contract.
- [ ] Dataset redistribution and fetch-only policies receive the required review.
- [ ] The project adopts an authoritative component license.
- [ ] A package-index publication and versioning policy is approved.
- [ ] Independent hardware reproduction confirms the reference packets.

The repository may be technically ready for review while these boxes remain
open. Public release must state that distinction explicitly.

## Final Release Audit

The handoff report must include:

1. Exact source and release commit SHAs
2. Twelve case IDs, evidence classes, run counts, digests, values, aggregates,
   and applicable CVs
3. Quality and functional gate outcomes
4. Power, device, operating system, Python, PyTorch, and hardware disclosures
5. Exact validation commands and exit status
6. Clean-install and package-verification evidence
7. Website and paper build evidence
8. Functional-stage, deferred, and rejected workloads with reasons
9. Remaining external decisions

Do not mark v0.1 complete until every implementation box has direct evidence
and no public claim exceeds that evidence.
