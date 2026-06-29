# MLPerf EDU Review Packet: `mobilenetv2-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | mobilenetv2-train |
| Run selector | mobilenetv2-train |
| Suite | vision |
| Public status | score-bearing |
| Scenario | single_stream |
| Model | mobilenetv2 |
| Dataset | fashion-mnist |

## Reviewer Commands

```bash
mlperf fetch --workload mobilenetv2-train --profile max --dry-run
mlperf run --workload mobilenetv2-train --profile max
```

## Quality Contract

| Field | Value |
|---|---|
| Metric | top1_accuracy |
| Target | 0.7 |
| Direction | higher |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | median top-1 accuracy must be >= target |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched Fashion-MNIST with fixed torchvision transforms, split, and seed; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median top-1 accuracy across five independent reference runs; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |
| Verified baseline | train_loss=0.805; val_loss=0.738; accuracy=0.727; epochs=5; time_seconds=24.2; note=Single-seed Apple MPS calibration with 50 train batches and 50 validation batches per epoch at learning rate 1e-4; CPU fallback is currently much slower and needs runner optimization before endorsement. |

## Assets

| Field | Value |
|---|---|
| Dataset asset | fashion-mnist |
| Dataset source | https://github.com/zalandoresearch/fashion-mnist |
| Dataset license status | mit |
| Dataset release status | public-ok-with-attribution |
| Dataset release next step | Keep source, citation, and license metadata in report, CSV, HTML, and package artifacts. |
| Dataset citation | Xiao, Rasul, and Vollgraf, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, 2017. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- No public-release warning from the current structured audit.

## Source Provenance

- Registry provenance: Sandler et al. 2018 (MobileNetV2); FULLY LOCAL — no torchvision.models dependency
- Runner min: mlperf.runners.vision:run_mobilenetv2_min
- Runner max: mlperf.runners.vision:run_mobilenetv2_max
