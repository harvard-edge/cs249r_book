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
| Baseline record | evidence_status=pending-clean-public-candidate-reference-summary; review_eligible=False; calibration_tier=development; development_summary_id=anomaly-ae-train_max_20260711T044139.487599Z; development_summary_sha256=4c82157b2fff4e91505cf0e3eaa8d23e8db946e85e098b09bdd303001a40e2f2; development_summary_availability=local-handoff; seeds=0, 1, 2, 3, 4; anomaly_auroc_by_seed=0.9666612742658038, 0.9701133535454093, 0.9665642110502738, 0.9645168333408751, 0.9658918955608851; anomaly_auroc=0.9665642110502738; five_seed_anomaly_auroc_median=0.9665642110502738; five_seed_anomaly_auroc_min=0.9645168333408751; five_seed_anomaly_auroc_max=0.9701133535454093; five_seed_anomaly_auroc_stdev=0.0020662715355119; duration_seconds_median=6.911759666982107; duration_seconds_min=6.631815499975346; duration_seconds_max=7.671591209014878; note=Fresh real-data CPU development calibration across seeds 0-4. The values are not review eligible until a clean public-candidate sweep commits a complete artifact index and retains its content-addressed raw package for review. |

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
| Dataset asset | mnist |
| Dataset source | torchvision://MNIST |
| Dataset license status | cc-by-sa-3.0 |
| Dataset release status | public-ok-with-attribution |
| Dataset release next step | Keep attribution and license metadata in report, CSV, HTML, and package artifacts. |
| Dataset citation | LeCun et al., gradient-based learning applied to document recognition, 1998. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- score-bearing baseline is not backed by a committed reference summary; evidence status is pending-clean-public-candidate-reference-summary

## Source Provenance

- Registry provenance: MLPerf Tiny AD benchmark; Koizumi et al. 2019 (ToyADMOS architecture)
- Runner min: mlperf.runners.tiny:run_anomaly_ae_min
- Runner max: mlperf.runners.tiny:run_anomaly_ae_max
