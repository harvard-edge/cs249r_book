# MLPerf EDU Review Packet: `anomaly-ae-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | anomaly-ae-train |
| Run selector | anomaly-ae-train |
| Suite | tiny |
| Public status | score-bearing |
| Scenario | offline |
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
| Metric | reconstruction_mse |
| Target | 0.04 |
| Direction | lower |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | median reconstruction MSE must be <= target |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched MNIST with fixed normal/anomaly digit split, preprocessing, and seed; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median reconstruction MSE across five independent reference runs; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |
| Verified baseline | reconstruction_mse=0.034; train_loss=0.034; val_loss=0.065; epochs=20; time_seconds=5; note=Val loss intentionally higher — anomalous digits have high recon error |

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

- No public-release warning from the current structured audit.

## Source Provenance

- Registry provenance: MLPerf Tiny AD benchmark; Koizumi et al. 2019 (ToyADMOS architecture)
- Runner min: mlperf.runners.tiny:run_anomaly_ae_min
- Runner max: mlperf.runners.tiny:run_anomaly_ae_max
