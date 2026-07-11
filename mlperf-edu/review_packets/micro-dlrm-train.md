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
mlperf fetch --workload micro-dlrm-train --profile max --dry-run
mlperf run --workload micro-dlrm-train --profile max
```

## Quality Contract

| Field | Value |
|---|---|
| Metric | accuracy |
| Target | 0.7 |
| Direction | higher |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | median best validation accuracy must be >= target |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched MovieLens-100K with fixed split, preprocessing, and seed; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median best validation accuracy across five independent reference runs; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=micro-dlrm-train_max_20260711T061336.028964Z; evidence_file=reference_results/micro-dlrm-train/micro-dlrm-train_max_20260711T061336.028964Z.json; evidence_sha256=8e9c26c9a34c6f0eab4d6f99229ac2167bee7f1b74d8daec5a63ecdca351cd74; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=cpu; data_mode=real; execution_backend=pytorch-cpu; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=accuracy; metric_values_by_seed=0.7018729967948718, 0.7093850160256411, 0.7040765224358975, 0.7019230769230769, 0.7048277243589743; accuracy=0.7040765224358975; median=0.7040765224358975; min=0.7018729967948718; max=0.7093850160256411; mean=0.7044170673076924; sample_stdev=0.003068281575263232; wall_seconds_median=3.448899875045754; wall_seconds_min=3.394848500029184; wall_seconds_max=3.6336147079709917; wall_seconds_mean=3.4715355915948747; wall_seconds_sample_stdev=0.09326881082314792; accepted_runs=5; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run passed. The content-addressed raw package is retained for local reviewer handoff but does not yet have a public URL. This is not an MLCommons-verified result. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol |  |
| Measurement protocol |  |
| Checkpoint contract |  |
| Task-quality evaluation |  |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=micro-dlrm-train_max_20260711T061336.028964Z; evidence_file=reference_results/micro-dlrm-train/micro-dlrm-train_max_20260711T061336.028964Z.json; evidence_sha256=8e9c26c9a34c6f0eab4d6f99229ac2167bee7f1b74d8daec5a63ecdca351cd74; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=cpu; data_mode=real; execution_backend=pytorch-cpu; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=accuracy; metric_values_by_seed=0.7018729967948718, 0.7093850160256411, 0.7040765224358975, 0.7019230769230769, 0.7048277243589743; accuracy=0.7040765224358975; median=0.7040765224358975; min=0.7018729967948718; max=0.7093850160256411; mean=0.7044170673076924; sample_stdev=0.003068281575263232; wall_seconds_median=3.448899875045754; wall_seconds_min=3.394848500029184; wall_seconds_max=3.6336147079709917; wall_seconds_mean=3.4715355915948747; wall_seconds_sample_stdev=0.09326881082314792; accepted_runs=5; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run passed. The content-addressed raw package is retained for local reviewer handoff but does not yet have a public URL. This is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/micro-dlrm-train/micro-dlrm-train_max_20260711T061336.028964Z.json |
| Reference package availability | local-handoff |
| External publication status | pending |
| External publication URL | not declared |
| Calibration observation |  |

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
- external-publication blocker: reference evidence package is retained for local handoff but is not yet publicly retrievable

## Source Provenance

- Registry provenance: Naumov et al. 2019 (DLRM); maps to MLPerf Training DLRM
- Runner min: mlperf.runners.dlrm:run_min
- Runner max: mlperf.runners.dlrm:run_max
