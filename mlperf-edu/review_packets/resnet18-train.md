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
OUTPUT_DIR="submissions/review-resnet18-train"
mlperf fetch --workload resnet18-train --profile max
mlperf run --workload resnet18-train --profile max --output-dir "$OUTPUT_DIR"
for manifest in "$OUTPUT_DIR"/*.provd.json; do mlperf verify "$manifest"; done
mlperf grade "$OUTPUT_DIR" --output "$OUTPUT_DIR/grade.json"
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
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched Fashion-MNIST with fixed torchvision transforms, five epochs, 100 training batches per epoch, and the complete 10,000-example test split; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median top-1 accuracy across five independent reference runs; repeatability_metric=sample coefficient of variation of train_and_eval_seconds across the five reference runs; repeatability_limit=0.05; repeatability_action=withhold the performance reference and rerun the complete protocol when the coefficient of variation exceeds 5%; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Measurement protocol | primary_metric=train_and_eval_seconds; scenario=training; timing_scope=fixed canonical training loop plus scheduled validation; included_phases=optimizer steps, scheduled validation, learning-rate scheduler steps when configured; excluded_phases=dataset or asset fetching, model construction, checkpoint serialization, report and provenance serialization; device_synchronization=synchronize immediately before and after the measured train-and-evaluation region; outer_reference_runs=5 |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=resnet18-train_max_20260711T204117.913822Z; evidence_file=reference_results/resnet18-train/resnet18-train_max_20260711T204117.913822Z.json; evidence_sha256=ad18347f04f4b2557e16428e0c5cbc741c0ae84d9c1259163aee4dec76d05f7d; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=86738e4654d8f77ef1cec4698b30e0ebd20dd2b3; profile=max; device_requested=default; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=train_and_eval_seconds; metric_values_by_seed=58.41915458299991, 59.5122965419996, 59.46243120799954, 58.41892304099929, 57.31812245900073; train_and_eval_seconds=58.41915458299991; median=58.41915458299991; min=57.31812245900073; max=59.5122965419996; mean=58.626185566599816; sample_stdev=0.9057245869515783; wall_seconds_median=60.83007133299907; wall_seconds_min=59.53187337499912; wall_seconds_max=62.5073435409995; wall_seconds_mean=61.13990226639944; wall_seconds_sample_stdev=1.1833647597067491; accepted_runs=5; quality_metric=top1_accuracy; quality_values_by_seed=0.8781, 0.8755, 0.8673, 0.863, 0.875; top1_accuracy=0.875; quality_median=0.875; quality_min=0.863; quality_max=0.8781; quality_mean=0.87178; quality_sample_stdev=0.006350354320823367; baseline_note=Clean five-run project reference from exact source commit 86738e46. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every seed passed the declared quality gate. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline record role | current-review-evidence |
| Baseline disclosure | Project reference evidence; not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/resnet18-train/resnet18-train_max_20260711T204117.913822Z.json |
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

- Registry provenance: He et al. 2016 (ResNet); local ResNet-18 implementation (no torchvision.models dependency); torchvision datasets/transforms are used only to fetch and preprocess Fashion-MNIST
- Runner min: mlperf.runners.vision:run_resnet18_min
- Runner max: mlperf.runners.vision:run_resnet18_max
