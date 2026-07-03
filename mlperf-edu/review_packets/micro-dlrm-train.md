# MLPerf EDU Review Packet: `micro-dlrm-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | micro-dlrm-train |
| Run selector | micro-dlrm-train |
| Suite | recommender |
| Public status | score-bearing |
| Scenario | server |
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
| Verified baseline | train_loss=0.535; val_loss=0.573; accuracy=0.705; five_seed_best_accuracy_median=0.704; five_seed_best_accuracy_min=0.702; five_seed_best_accuracy_max=0.709; epochs=21; time_seconds=3; note=23K params; embedding tables hold 21,168 fp32 values, about 83 KiB, and remain cache-resident on laptop CPUs |

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

## Source Provenance

- Registry provenance: Naumov et al. 2019 (DLRM); maps to MLPerf Training DLRM
- Runner min: mlperf.runners.dlrm:run_min
- Runner max: mlperf.runners.dlrm:run_max
