# MLPerf EDU Review Packet: `anomaly-ae-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | anomaly-ae-train |
| Run selector | anomaly-ae-train |
| Suite | tiny |
| Public status | score-bearing |
| Scenario | training |
| Model | anomaly-ae |
| Dataset | mnist |

## Reviewer Commands

```bash
mlperf fetch --workload anomaly-ae-train --profile max --dry-run
mlperf run --workload anomaly-ae-train --profile max
```

## Quality Contract

| Field | Value |
|---|---|
| Metric | anomaly_auroc |
| Target | 0.95 |
| Direction | higher |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | median anomaly AUROC must be >= target |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched MNIST with fixed normal/anomaly digit split, preprocessing, and seed; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median anomaly AUROC across five independent reference runs; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=anomaly-ae-train_max_20260711T061301.162532Z; evidence_file=reference_results/anomaly-ae-train/anomaly-ae-train_max_20260711T061301.162532Z.json; evidence_sha256=634b6dd63a22e3013a13210e114daeec68b9266ae78fd01a43c7e670690783e9; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=cpu; data_mode=real; execution_backend=pytorch-cpu; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=anomaly_auroc; metric_values_by_seed=0.9666612742658038, 0.9701133535454093, 0.9665642110502738, 0.9645168333408751, 0.9658918955608851; anomaly_auroc=0.9665642110502738; median=0.9665642110502738; min=0.9645168333408751; max=0.9701133535454093; mean=0.9667495135526494; sample_stdev=0.0020662715355119; wall_seconds_median=4.573835415998474; wall_seconds_min=4.530495124985464; wall_seconds_max=4.715450500021689; wall_seconds_mean=4.608597349771299; wall_seconds_sample_stdev=0.07633000623238935; accepted_runs=5; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run passed. The content-addressed raw package is retained for local reviewer handoff but does not yet have a public URL. This is not an MLCommons-verified result. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol |  |
| Measurement protocol |  |
| Checkpoint contract |  |
| Task-quality evaluation |  |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=anomaly-ae-train_max_20260711T061301.162532Z; evidence_file=reference_results/anomaly-ae-train/anomaly-ae-train_max_20260711T061301.162532Z.json; evidence_sha256=634b6dd63a22e3013a13210e114daeec68b9266ae78fd01a43c7e670690783e9; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=cpu; data_mode=real; execution_backend=pytorch-cpu; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=anomaly_auroc; metric_values_by_seed=0.9666612742658038, 0.9701133535454093, 0.9665642110502738, 0.9645168333408751, 0.9658918955608851; anomaly_auroc=0.9665642110502738; median=0.9665642110502738; min=0.9645168333408751; max=0.9701133535454093; mean=0.9667495135526494; sample_stdev=0.0020662715355119; wall_seconds_median=4.573835415998474; wall_seconds_min=4.530495124985464; wall_seconds_max=4.715450500021689; wall_seconds_mean=4.608597349771299; wall_seconds_sample_stdev=0.07633000623238935; accepted_runs=5; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run passed. The content-addressed raw package is retained for local reviewer handoff but does not yet have a public URL. This is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/anomaly-ae-train/anomaly-ae-train_max_20260711T061301.162532Z.json |
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
| Dataset asset | mnist |
| Dataset source | torchvision://MNIST |
| Dataset license status | cc-by-sa-3.0 |
| Dataset release status | public-ok-with-attribution |
| Dataset release next step | Keep attribution and license metadata in report, CSV, HTML, and package artifacts. |
| Dataset citation | LeCun et al., gradient-based learning applied to document recognition, 1998. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- external-publication blocker: reference evidence package is retained for local handoff but is not yet publicly retrievable

## Source Provenance

- Registry provenance: MLPerf Tiny AD benchmark; Koizumi et al. 2019 (ToyADMOS architecture)
- Runner min: mlperf.runners.tiny:run_anomaly_ae_min
- Runner max: mlperf.runners.tiny:run_anomaly_ae_max
