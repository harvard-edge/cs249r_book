# MLPerf EDU Review Packet: `resnet18-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | resnet18-train |
| Run selector | resnet18-train |
| Suite | vision |
| Public status | score-bearing |
| Scenario | single_stream |
| Model | resnet18 |
| Dataset | fashion-mnist |

## Reviewer Commands

```bash
mlperf fetch --workload resnet18-train --profile max --dry-run
mlperf run --workload resnet18-train --profile max
```

## Quality Contract

| Field | Value |
|---|---|
| Metric | top1_accuracy |
| Target | 0.75 |
| Direction | higher |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | median top-1 accuracy must be >= target |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched Fashion-MNIST with fixed torchvision transforms, split, and seed; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median top-1 accuracy across five independent reference runs; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |
| Verified baseline | train_loss=0.638; val_loss=0.584; accuracy=0.773; epochs=3; time_seconds=7.4; note=Single-seed Apple MPS default run with 20 train batches and 20 validation batches per epoch; target remains provisional until the five-seed reference sweep. |

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

- Registry provenance: He et al. 2016 (ResNet); FULLY LOCAL implementation — no torchvision.models dependency
- Runner min: mlperf.runners.vision:run_resnet18_min
- Runner max: mlperf.runners.vision:run_resnet18_max
