# MLPerf EDU v0.1 Release Checklist

## Release Definition

The review candidate contains seven workloads and ten evidence cases. Evidence
is bound to clean source revision
`3cc071737454494d6a14d58fb5dc74d190d6cf7a`. A checked box requires direct
command output or a committed content-addressed artifact. Intent, partial runs,
and narrow smoke checks do not prove a broad release claim.

## Portfolio and Architecture

- [x] `image-classification` inherits the MLPerf Tiny ResNet8 contract.
- [x] `keyword-spotting` inherits the MLPerf Tiny DS-CNN contract.
- [x] `causal-language-modeling` preserves nanoGPT training and inference under one identity.
- [x] `text-classification` uses the pinned DistilBERT SST-2 checkpoint and split.
- [x] `information-retrieval` reproduces the documented CrossEncoder NanoBEIR example.
- [x] `graph-node-classification` uses the official OGB GCN recipe and evaluator.
- [x] `time-series-forecasting` uses the official PatchTST ETTm1 recipe and split.
- [x] Modes, phases, configurations, scenarios, and profiles are separate taxonomy axes.
- [x] MiniGo is the RL reference, and RL is deferred without a substitute.
- [x] Rejected and deferred tasks have explicit reasons in the selection ledger.

## Evidence Cases

- [ ] `image-classification__max__inference`
- [ ] `keyword-spotting__max__inference`
- [ ] `causal-language-modeling__max__training`
- [ ] `causal-language-modeling__max__inference__full`
- [ ] `causal-language-modeling__max__inference__prefill`
- [ ] `causal-language-modeling__max__inference__decode`
- [ ] `text-classification__max__inference`
- [ ] `information-retrieval__max__inference`
- [ ] `graph-node-classification__max__training`
- [ ] `time-series-forecasting__max__training`

These boxes are generated as review state, not manually asserted as proof.
The authoritative completion test is a ten-entry
`reference_results/index.json` whose summaries all pass import, acceptance,
repeatability, digest, source-lock, and provenance verification.

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
uv run python tools/import_reference_evidence.py \
  --evidence-root /tmp/mlperf-edu-promotion \
  --source-git-sha 3cc071737454494d6a14d58fb5dc74d190d6cf7a
uv run python tools/sync_verified_baselines.py
uv run python tools/export_flat_registry.py
uv run python tools/check_taxonomy.py
uv run python tools/check_reference_claims.py --check
```

- [ ] Importer independently verifies all retained reports and manifests.
- [ ] Importer rejects missing, duplicate, stale-source, interrupted, or high-CV cases.
- [ ] All three causal phases share one median-quality training lineage.
- [ ] Native registry baselines bind every case ID and evidence digest.
- [ ] Flat registry and packaged mirrors match native registry sources.
- [ ] Raw checkpoints and dataset-derived bytes remain outside Git.

## CLI and Profile Coverage

```bash
uv run mlperf doctor
uv run mlperf audit
uv run mlperf audit --policy public
uv run mlperf validate smoke --output-dir /tmp/mlperf-edu-smoke
uv run mlperf validate coverage --output-dir /tmp/mlperf-edu-coverage
uv run mlperf validate pro --output-dir /tmp/mlperf-edu-pro
```

- [ ] Every registered `min` path passes from the public CLI.
- [ ] Every canonical `max` path passes through the public CLI.
- [ ] Every applicable `pro` path passes and retains workload identity.
- [ ] Explicit mode and phase selection works and invalid combinations fail early.
- [ ] Fetch runs before measurement and verifies pinned assets.
- [ ] JSON, CSV, HTML, and provenance files are emitted for every run.

## Platform Coverage

- [ ] Clean CPU install and functional execution pass.
- [ ] Apple Silicon MPS execution passes where supported.
- [ ] Unsupported backends fail with an actionable message.
- [ ] Every report records `device_requested`, `device_executed`, and the
  executed backend.
- [ ] Power source and Low Power Mode are disclosed for reference campaigns.
- [ ] Sleep or power-state changes invalidate affected attempts.

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

- [ ] Full Python test suite passes.
- [ ] Provenance tamper and clean-extraction package tests pass.
- [ ] Registry, evidence, documentation, and paper drift checks pass.
- [ ] All MLPerf EDU GitHub workflows contain no retired workload assertions.
- [ ] Wheel includes registry, dataset, and ten-case evidence resources.
- [ ] Wheel installs in a clean Python 3.12 environment outside the checkout.
- [ ] Installed CLI can list, audit, run a smoke path, report, and verify a package.
- [ ] Quarto site renders and internal links pass.
- [ ] Paper builds without missing references, placeholders, or layout overflows.

## Documentation and Cleanup

- [x] Generated benchmark pages expose exactly seven workload pages and five suite indexes.
- [x] Retired generated workload pages are removed.
- [ ] README, specification, proposal, public rules, target review, dataset review, and paper agree.
- [ ] Website and paper display exact committed evidence values from generated sources.
- [ ] Review packets expose only admitted workloads and case-level evidence.
- [ ] Obsolete non-generated implementation files are removed or moved to an explicitly non-public archive.
- [ ] No cache, build output, checkpoint, dataset-derived artifact, or AI configuration is committed.

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
2. Ten case IDs, evidence IDs, digests, five values, aggregates, and CVs
3. Quality and functional gate outcomes
4. Power, device, operating system, Python, PyTorch, and hardware disclosures
5. Exact validation commands and exit status
6. Clean-install and package-verification evidence
7. Website and paper build evidence
8. Deferred and rejected workloads with reasons
9. Remaining external decisions

Do not mark v0.1 complete until every implementation box has direct evidence
and no public claim exceeds that evidence.
