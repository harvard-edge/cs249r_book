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
OUTPUT_DIR="submissions/review-anomaly-ae-train"
mlperf fetch --workload anomaly-ae-train --profile max
mlperf run --workload anomaly-ae-train --profile max --output-dir "$OUTPUT_DIR"
for manifest in "$OUTPUT_DIR"/*.provd.json; do mlperf verify "$manifest"; done
mlperf grade "$OUTPUT_DIR" --output "$OUTPUT_DIR/grade.json"
```

## Quality Contract

| Field | Value |
|---|---|
| Metric | anomaly_auroc |
| Target | 0.93 |
| Direction | higher |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | Every run must pass the macro-AUROC target, worst-class gate, and learned-control gate. The reported five-seed median must also be >= target. |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched MNIST hard-curve-v1 with digit 5 as normal, digits 3/8/9 as the fixed anomaly set, original test labels preserved for classwise scoring, and no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=Median macro anomaly AUROC across five independent reference runs. Every run must pass the macro target, worst-class AUROC, and learned-control-margin gates.; repeatability_metric=sample coefficient of variation of train_and_eval_seconds across the five reference runs; repeatability_limit=0.05; repeatability_action=withhold the performance reference and rerun the complete protocol when the coefficient of variation exceeds 5%; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Measurement protocol | primary_metric=train_and_eval_seconds; scenario=training; timing_scope=fixed canonical training loop plus scheduled validation; included_phases=optimizer steps, scheduled validation, learning-rate scheduler steps when configured; excluded_phases=dataset or asset fetching, model construction, checkpoint serialization, report and provenance serialization; device_synchronization=synchronize immediately before and after the measured train-and-evaluation region; outer_reference_runs=5 |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=anomaly-ae-train_max_20260711T185950.479514Z; evidence_file=reference_results/anomaly-ae-train/anomaly-ae-train_max_20260711T185950.479514Z.json; evidence_sha256=036fb7f1f6cbef38f5c229cb08f16555a8086cd9af61b944925032b9aa6f22c7; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=b4366b7614f0bb8ba0a1d6224832d4caea64e68a; profile=max; device_requested=cpu; data_mode=real; execution_backend=pytorch-cpu; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=train_and_eval_seconds; metric_values_by_seed=3.1892096660003517, 3.215892415999406, 3.118504832999861, 3.077835999999479, 3.101356291999764; train_and_eval_seconds=3.118504832999861; median=3.118504832999861; min=3.077835999999479; max=3.215892415999406; mean=3.140559841399772; sample_stdev=0.05915958119900082; wall_seconds_median=5.876781749999282; wall_seconds_min=5.837836957999571; wall_seconds_max=5.9451311669999996; wall_seconds_mean=5.887903841599472; wall_seconds_sample_stdev=0.04899795873321173; accepted_runs=5; quality_metric=anomaly_auroc; quality_values_by_seed=0.9402503904617956, 0.9422428382049718, 0.9372022985286352, 0.9369790049994332, 0.9388614489261359; anomaly_auroc=0.9388614489261359; quality_median=0.9388614489261359; quality_min=0.9369790049994332; quality_max=0.9422428382049718; quality_mean=0.9391071962241944; quality_sample_stdev=0.002199850480754223; baseline_note=Clean five-run project reference from exact source commit b4366b76. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every seed passed the declared quality gate. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline record role | current-review-evidence |
| Baseline disclosure | Project reference evidence; not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/anomaly-ae-train/anomaly-ae-train_max_20260711T185950.479514Z.json |
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
| Dataset asset | mnist |
| Dataset source | torchvision://MNIST |
| Dataset license status | cc-by-sa-3.0 |
| Dataset release status | public-ok-with-attribution |
| Dataset release next step | Keep attribution and license metadata in report, CSV, HTML, and package artifacts. |
| Dataset citation | LeCun et al., gradient-based learning applied to document recognition, 1998. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- external-publication blocker: registry declares local-handoff reference evidence, but no published package URL is recorded

## Source Provenance

- Registry provenance: MLPerf Tiny AD benchmark; Koizumi et al. 2019 (ToyADMOS architecture)
- Runner min: mlperf.runners.tiny:run_anomaly_ae_min
- Runner max: mlperf.runners.tiny:run_anomaly_ae_max
