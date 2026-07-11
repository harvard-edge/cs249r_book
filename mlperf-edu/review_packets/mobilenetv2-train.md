# MLPerf EDU Review Packet: `mobilenetv2-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | mobilenetv2-train |
| Run selector | mobilenetv2-train |
| Suite | vision |
| Public status | score-bearing |
| Scenario | training |
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
| Target | 0.78 |
| Direction | higher |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | median top-1 accuracy must be >= target |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched Fashion-MNIST with fixed torchvision transforms, eight epochs, 100 training batches per epoch, and the complete 10,000-example test split; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median top-1 accuracy across five independent reference runs; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |
| Baseline record | evidence_status=pending-clean-public-candidate-reference-summary; review_eligible=False; calibration_tier=development; development_summary_id=mobilenetv2-train_max_20260711T044736.880449Z; development_summary_sha256=675621b4c853b9ff250af8d80ae96903336d6d4a5ee9756671e848440e4b0eea; development_summary_availability=local-handoff; seeds=0, 1, 2, 3, 4; top1_accuracy_by_seed=0.8089, 0.8139, 0.797, 0.8238, 0.8008; top1_accuracy=0.8089; accuracy=0.8089; five_seed_accuracy_median=0.8089; five_seed_accuracy_min=0.797; five_seed_accuracy_max=0.8238; five_seed_accuracy_stdev=0.010656312683100081; epochs=8; duration_seconds_median=96.70496737502981; duration_seconds_min=87.97821220802143; duration_seconds_max=107.52852020796854; note=Fresh real-data Apple MPS development calibration across seeds 0-4. The values are not review eligible until a clean committed reference summary indexes every raw artifact in the retained reference package. |

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

- Registry provenance: Sandler et al. 2018 (MobileNetV2); FULLY LOCAL — no torchvision.models dependency
- Runner min: mlperf.runners.vision:run_mobilenetv2_min
- Runner max: mlperf.runners.vision:run_mobilenetv2_max
