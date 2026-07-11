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
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=mobilenetv2-train_max_20260711T062054.539361Z; evidence_file=reference_results/mobilenetv2-train/mobilenetv2-train_max_20260711T062054.539361Z.json; evidence_sha256=9532063b214c92e954531dc2ec4252b6e6c2c86e10a6a6c00ed340b7b2d50c62; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=top1_accuracy; metric_values_by_seed=0.8089, 0.8139, 0.797, 0.8238, 0.8008; top1_accuracy=0.8089; median=0.8089; min=0.797; max=0.8238; mean=0.8088799999999999; sample_stdev=0.010656312683100081; wall_seconds_median=40.44746512500569; wall_seconds_min=38.940279499976896; wall_seconds_max=41.26678062498104; wall_seconds_mean=40.3596491002012; wall_seconds_sample_stdev=0.9089845594524596; accepted_runs=5; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run passed. Throughput and time fields are machine observations, not portable targets. The raw package is retained for local reviewer handoff but has no public URL. This is not an MLCommons-verified result. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol |  |
| Measurement protocol |  |
| Checkpoint contract |  |
| Task-quality evaluation |  |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=mobilenetv2-train_max_20260711T062054.539361Z; evidence_file=reference_results/mobilenetv2-train/mobilenetv2-train_max_20260711T062054.539361Z.json; evidence_sha256=9532063b214c92e954531dc2ec4252b6e6c2c86e10a6a6c00ed340b7b2d50c62; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=top1_accuracy; metric_values_by_seed=0.8089, 0.8139, 0.797, 0.8238, 0.8008; top1_accuracy=0.8089; median=0.8089; min=0.797; max=0.8238; mean=0.8088799999999999; sample_stdev=0.010656312683100081; wall_seconds_median=40.44746512500569; wall_seconds_min=38.940279499976896; wall_seconds_max=41.26678062498104; wall_seconds_mean=40.3596491002012; wall_seconds_sample_stdev=0.9089845594524596; accepted_runs=5; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run passed. Throughput and time fields are machine observations, not portable targets. The raw package is retained for local reviewer handoff but has no public URL. This is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/mobilenetv2-train/mobilenetv2-train_max_20260711T062054.539361Z.json |
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
| Dataset asset | fashion-mnist |
| Dataset source | https://github.com/zalandoresearch/fashion-mnist |
| Dataset license status | mit |
| Dataset release status | public-ok-with-attribution |
| Dataset release next step | Keep source, citation, and license metadata in report, CSV, HTML, and package artifacts. |
| Dataset citation | Xiao, Rasul, and Vollgraf, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, 2017. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- external-publication blocker: reference evidence package is retained for local handoff but is not yet publicly retrievable

## Source Provenance

- Registry provenance: Sandler et al. 2018 (MobileNetV2); FULLY LOCAL — no torchvision.models dependency
- Runner min: mlperf.runners.vision:run_mobilenetv2_min
- Runner max: mlperf.runners.vision:run_mobilenetv2_max
