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
| Baseline record | evidence_status=pending-clean-public-candidate-reference-summary; review_eligible=False; calibration_tier=development; development_summary_id=micro-dlrm-train_max_20260711T044139.485814Z; development_summary_sha256=52e306dcc03090eddece33ebb76fd65e3328bd7cedcc2b98b0df0df9546d6601; development_summary_availability=local-handoff; seeds=0, 1, 2, 3, 4; accuracy_by_seed=0.7018729967948718, 0.7093850160256411, 0.7040765224358975, 0.7019230769230769, 0.7048277243589743; accuracy=0.7040765224358975; five_seed_best_accuracy_median=0.7040765224358975; five_seed_best_accuracy_min=0.7018729967948718; five_seed_best_accuracy_max=0.7093850160256411; five_seed_best_accuracy_stdev=0.003068281575263232; duration_seconds_median=4.93660695804283; duration_seconds_min=4.807407416054048; duration_seconds_max=5.537989708012901; note=Fresh real-data CPU development calibration across seeds 0-4. The values are not review eligible until a clean public-candidate sweep commits a complete artifact index and retains its content-addressed raw package for review. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol |  |
| Measurement protocol |  |
| Checkpoint contract |  |
| Task-quality evaluation |  |
| Baseline evidence status | pending-clean-public-candidate-reference-summary |
| Baseline review eligible | False |
| Baseline evidence file | not declared |
| Reference package availability | not declared |
| External publication status | not declared |
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
- score-bearing baseline is not backed by a committed reference summary; evidence status is pending-clean-public-candidate-reference-summary

## Source Provenance

- Registry provenance: Naumov et al. 2019 (DLRM); maps to MLPerf Training DLRM
- Runner min: mlperf.runners.dlrm:run_min
- Runner max: mlperf.runners.dlrm:run_max
