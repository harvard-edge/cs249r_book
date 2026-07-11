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
OUTPUT_DIR="submissions/review-mobilenetv2-train"
mlperf fetch --workload mobilenetv2-train --profile max
mlperf run --workload mobilenetv2-train --profile max --output-dir "$OUTPUT_DIR"
for manifest in "$OUTPUT_DIR"/*.provd.json; do mlperf verify "$manifest"; done
mlperf grade "$OUTPUT_DIR" --output "$OUTPUT_DIR/grade.json"
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
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched Fashion-MNIST with fixed torchvision transforms, eight epochs, 100 training batches per epoch, and the complete 10,000-example test split; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median top-1 accuracy across five independent reference runs; repeatability_metric=sample coefficient of variation of train_and_eval_seconds across the five reference runs; repeatability_limit=0.05; repeatability_action=withhold the performance reference and rerun the complete protocol when the coefficient of variation exceeds 5%; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Measurement protocol | primary_metric=train_and_eval_seconds; scenario=training; timing_scope=fixed canonical training loop plus scheduled validation; included_phases=optimizer steps, scheduled validation, learning-rate scheduler steps when configured; excluded_phases=dataset or asset fetching, model construction, checkpoint serialization, report and provenance serialization; device_synchronization=synchronize immediately before and after the measured train-and-evaluation region; outer_reference_runs=5 |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=mobilenetv2-train_max_20260711T190411.719846Z; evidence_file=reference_results/mobilenetv2-train/mobilenetv2-train_max_20260711T190411.719846Z.json; evidence_sha256=89ab833b133570f89144bb88644ed235e6f33e8d24875739789575f09fee6fb8; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=b4366b7614f0bb8ba0a1d6224832d4caea64e68a; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=train_and_eval_seconds; metric_values_by_seed=57.539257958000235, 57.726385082998604, 58.1860116250009, 57.760918082998614, 57.58099479200064; train_and_eval_seconds=57.726385082998604; median=57.726385082998604; min=57.539257958000235; max=58.1860116250009; mean=57.7587135081998; sample_stdev=0.2566020013798961; wall_seconds_median=59.49713895899913; wall_seconds_min=59.3031687090006; wall_seconds_max=59.95719016700059; wall_seconds_mean=59.53181118360008; wall_seconds_sample_stdev=0.2624156465457438; accepted_runs=5; quality_metric=top1_accuracy; quality_values_by_seed=0.8089, 0.8139, 0.797, 0.8238, 0.8008; top1_accuracy=0.8089; quality_median=0.8089; quality_min=0.797; quality_max=0.8238; quality_mean=0.8088799999999999; quality_sample_stdev=0.010656312683100081; baseline_note=Clean five-run project reference from exact source commit b4366b76. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every seed passed the declared quality gate. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline record role | current-review-evidence |
| Baseline disclosure | Project reference evidence; not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/mobilenetv2-train/mobilenetv2-train_max_20260711T190411.719846Z.json |
| Reference package availability | local-handoff |
| External publication status | pending |
| External publication URL | not declared |

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

- external-publication blocker: registry declares local-handoff reference evidence, but no published package URL is recorded

## Source Provenance

- Registry provenance: Sandler et al. 2018 (MobileNetV2); FULLY LOCAL — no torchvision.models dependency
- Runner min: mlperf.runners.vision:run_mobilenetv2_min
- Runner max: mlperf.runners.vision:run_mobilenetv2_max
