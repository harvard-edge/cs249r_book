# MLPerf EDU Review Packet: `resnet18-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | resnet18-train |
| Run selector | resnet18-train |
| Suite | vision |
| Public status | score-bearing |
| Scenario | training |
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
| Target | 0.85 |
| Direction | higher |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | median top-1 accuracy must be >= target |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched Fashion-MNIST with fixed torchvision transforms, five epochs, 100 training batches per epoch, and the complete 10,000-example test split; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median top-1 accuracy across five independent reference runs; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |
| Baseline record | evidence_status=pending-clean-public-candidate-reference-summary; review_eligible=False; calibration_tier=development; development_summary_id=resnet18-train_max_20260711T044245.159697Z; development_summary_sha256=c49fdf409f4181d0003e6256bdb9cac830d7ccd8d8425583de9818ec42baa15c; development_summary_availability=local-handoff; seeds=0, 1, 2, 3, 4; top1_accuracy_by_seed=0.8781, 0.8755, 0.8673, 0.863, 0.875; top1_accuracy=0.875; accuracy=0.875; five_seed_accuracy_median=0.875; five_seed_accuracy_min=0.863; five_seed_accuracy_max=0.8781; five_seed_accuracy_stdev=0.006350354320823367; epochs=5; duration_seconds_median=51.4034499169793; duration_seconds_min=50.63850987504702; duration_seconds_max=62.62304970808327; note=Fresh real-data Apple MPS development calibration across seeds 0-4. The values are not review eligible until a clean public-candidate sweep commits a complete artifact index and retains its content-addressed raw package for review. |

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
| Dataset asset | fashion-mnist |
| Dataset source | https://github.com/zalandoresearch/fashion-mnist |
| Dataset license status | mit |
| Dataset release status | public-ok-with-attribution |
| Dataset release next step | Keep source, citation, and license metadata in report, CSV, HTML, and package artifacts. |
| Dataset citation | Xiao, Rasul, and Vollgraf, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, 2017. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- score-bearing baseline is not backed by a committed reference summary; evidence status is pending-clean-public-candidate-reference-summary

## Source Provenance

- Registry provenance: He et al. 2016 (ResNet); local ResNet-18 implementation (no torchvision.models dependency); torchvision datasets/transforms are used only to fetch and preprocess Fashion-MNIST
- Runner min: mlperf.runners.vision:run_resnet18_min
- Runner max: mlperf.runners.vision:run_resnet18_max
