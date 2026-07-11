# MLPerf EDU Review Packet: `micro-dlrm-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | micro-dlrm-train |
| Run selector | micro-dlrm-train |
| Suite | recommender |
| Public status | score-bearing |
| Scenario | training |
| Model | micro-dlrm-1m |
| Dataset | movielens-100k |

## Reviewer Commands

```bash
OUTPUT_DIR="submissions/review-micro-dlrm-train"
mlperf fetch --workload micro-dlrm-train --profile max
mlperf run --workload micro-dlrm-train --profile max --output-dir "$OUTPUT_DIR"
for manifest in "$OUTPUT_DIR"/*.provd.json; do mlperf verify "$manifest"; done
mlperf grade "$OUTPUT_DIR" --output "$OUTPUT_DIR/grade.json"
```

## Quality Contract

| Field | Value |
|---|---|
| Metric | roc_auc |
| Target | 0.76 |
| Direction | higher |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | median fixed-final-epoch evaluation ROC AUC must be >= target |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched MovieLens-100K with the official u1.base training and u1.test validation split, demographics and item-genre features only, and no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median fixed-final-epoch evaluation ROC AUC across five independent reference runs; repeatability_metric=sample coefficient of variation of train_and_eval_seconds across the five reference runs; repeatability_limit=0.05; repeatability_action=withhold the performance reference and rerun the complete protocol when the coefficient of variation exceeds 5%; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Measurement protocol | primary_metric=train_and_eval_seconds; scenario=training; timing_scope=fixed canonical training loop plus scheduled validation; included_phases=optimizer steps, scheduled validation, learning-rate scheduler steps when configured; excluded_phases=dataset or asset fetching, model construction, checkpoint serialization, report and provenance serialization; device_synchronization=synchronize immediately before and after the measured train-and-evaluation region; outer_reference_runs=5 |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=micro-dlrm-train_max_20260711T205839.712863Z; evidence_file=reference_results/micro-dlrm-train/micro-dlrm-train_max_20260711T205839.712863Z.json; evidence_sha256=946bc787b41515fbd1779e2ba75d68c0e755a15fee37bea7d4d681a529d53ab5; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=86738e4654d8f77ef1cec4698b30e0ebd20dd2b3; profile=max; device_requested=default; data_mode=real; execution_backend=pytorch-cpu; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=train_and_eval_seconds; metric_values_by_seed=2.0923957909999444, 1.999684124999476, 1.9475044169994362, 2.063613083000746, 2.0393749579998257; train_and_eval_seconds=2.0393749579998257; median=2.0393749579998257; min=1.9475044169994362; max=2.0923957909999444; mean=2.0285144747998856; sample_stdev=0.05662266217441502; wall_seconds_median=3.937821707999319; wall_seconds_min=3.8539375410000503; wall_seconds_max=4.395172499998807; wall_seconds_mean=4.011631808199672; wall_seconds_sample_stdev=0.22066340999068385; accepted_runs=5; quality_metric=roc_auc; quality_values_by_seed=0.7671171627454848, 0.7696129592578403, 0.7677678928436241, 0.7681811204950709, 0.7675421294438094; roc_auc=0.7677678928436241; quality_median=0.7677678928436241; quality_min=0.7671171627454848; quality_max=0.7696129592578403; quality_mean=0.7680442529571658; quality_sample_stdev=0.0009575452177736948; baseline_note=Clean five-run project reference from exact source commit 86738e46. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every seed passed the declared quality gate. Raw attempts are retained on the source machine. Portable packaging is blocked by the current MovieLens redistribution policy, and no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline record role | current-review-evidence |
| Baseline disclosure | Project reference evidence; not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/micro-dlrm-train/micro-dlrm-train_max_20260711T205839.712863Z.json |
| Reference package availability | local-handoff |
| External publication status | pending |
| External publication URL | not declared |

## Taxonomy Evidence

| Axis | Claim and evidence |
|---|---|
| working_set | value=unmeasured; evidence=none; sha256=none; note=No committed working-set measurement and reference-platform capacity record exists; classification is withheld. |
| arithmetic_intensity | value=unmeasured; evidence=none; sha256=none; note=No committed FLOP/byte measurement and reference-platform ridge-point record exists; classification is withheld. |
| dispatch | value=unmeasured; evidence=none; sha256=none; note=No committed synchronized utilization trace and reference-platform peak record exists; classification is withheld. |

## Assets

| Field | Value |
|---|---|
| Dataset asset | movielens-100k |
| Dataset source | https://files.grouplens.org/datasets/movielens/ml-100k.zip |
| Dataset license status | noncommercial-research-education |
| Dataset release status | restricted-needs-approval |
| Dataset release next step | Ask MLCommons reviewers whether MovieLens remains score-bearing with an official fetch-only policy, or move this workload to systems-only until permission is recorded. |
| Dataset citation | Harper and Konstan, The MovieLens Datasets, ACM TiiS 2015. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- dataset public release status is restricted-needs-approval: Ask MLCommons reviewers whether MovieLens remains score-bearing with an official fetch-only policy, or move this workload to systems-only until permission is recorded.
- external-publication blocker: registry declares local-handoff reference evidence, but no published package URL is recorded

## Source Provenance

- Registry provenance: Naumov et al. 2019 (DLRM); maps to MLPerf Training DLRM
- Runner min: mlperf.runners.dlrm:run_min
- Runner max: mlperf.runners.dlrm:run_max
