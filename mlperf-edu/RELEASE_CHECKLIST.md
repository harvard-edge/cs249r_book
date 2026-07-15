# MLPerf EDU v0.1 Release Checklist

## Release Definition

The review candidate contains nine workloads and twelve evidence cases. Evidence
is bound to clean source revision
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
- [x] Modes, phases, configurations, scenarios, and profiles are separate taxonomy axes.
- [x] MiniGo is the RL reference, and RL is deferred without a substitute.
- [x] Rejected and deferred tasks have explicit reasons in the selection ledger.

## Evidence Cases

- [x] `image-classification__max__inference` — five-run verified
- [x] `keyword-spotting__max__inference` — five-run verified
- [x] `anomaly-detection__max__inference` — five-run verified
- [x] `visual-wake-words__max__inference` — five-run verified
- [x] `causal-language-modeling__max__training` — two-run provisional
- [x] `causal-language-modeling__max__inference__full` — one-run provisional
- [x] `causal-language-modeling__max__inference__prefill` — one-run provisional
- [x] `causal-language-modeling__max__inference__decode` — one-run provisional
- [x] `text-classification__max__inference` — five-run verified
- [x] `information-retrieval__max__inference` — five-run verified
- [x] `graph-node-classification__max__training` — one-run provisional
- [x] `time-series-forecasting__max__training` — one-run provisional

These boxes mean that class-labeled draft evidence exists for every case. They
do not erase the evidence-class labels or promote a provisional record. The authoritative draft closure is the
twelve-entry `provisional_results/index.json`, with six five-run verified
records and six provisional records. Full promotion still requires a complete
twelve-entry `reference_results/index.json` whose summaries all pass
acceptance, repeatability, digest, source-lock, provenance, and lineage
verification.

## Reference Campaign Command

Run one complete case per attempt. Use AC power, disable Low Power Mode, prevent
sleep, and avoid concurrent heavy workloads.

```bash
uv run python tools/run_reference_sweep.py \
  --workload WORKLOAD --profile max --mode MODE [--phase PHASE] \
  --runs 5 --device DEVICE --evidence-tier promotion-candidate \
  --inter-execution-cooldown-seconds 30 \
  --output-dir /tmp/mlperf-edu-promotion
```

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
- [x] Every canonical `max` path passes through the public CLI.
- [x] Every applicable `pro` path passes and retains workload identity with one canonical measurement by default.
- [x] Explicit mode and phase selection works and invalid combinations fail early.
- [x] Fetch runs before measurement and verifies pinned assets.
- [x] JSON, CSV, HTML, and provenance files are emitted for every run.

## Platform Coverage

- [x] Clean CPU install and functional execution pass.
- [x] Apple Silicon MPS execution passes where supported.
- [x] Unsupported backends fail with an actionable message.
- [x] Every report records `device_requested`, `device_executed`, and the
  executed backend.
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
- [x] Provenance tamper and clean-extraction package tests pass.
- [x] Registry, evidence, documentation, and paper drift checks pass.
- [x] All MLPerf EDU GitHub workflows contain no retired workload assertions.
- [x] Wheel includes registry, dataset, and twelve-case draft evidence resources.
- [x] Wheel installs in a clean Python 3.12 environment outside the checkout.
- [x] Installed CLI can list, audit, run a smoke path, report, and verify a package.
- [x] Quarto site renders and internal links pass.
- [x] Paper builds without missing references, placeholders, or layout overflows.

## Documentation and Cleanup

- [x] Generated benchmark pages expose exactly nine workload pages and five suite indexes.
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
8. Deferred and rejected workloads with reasons
9. Remaining external decisions

Do not mark v0.1 complete until every implementation box has direct evidence
and no public claim exceeds that evidence.
