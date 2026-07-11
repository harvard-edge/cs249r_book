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
| Baseline record | evidence_status=committed-reference-summary; review_eligible=False; protocol_compatibility=superseded; replacement_required=True; superseded_reason=The current mnist-hard-curve-v1 contract adds classwise gates and no-training controls; this packet used the former zero-versus-all-digits protocol.; evidence_tier=public-candidate; evidence_id=anomaly-ae-train_max_20260711T085421.359195Z; evidence_file=reference_results/anomaly-ae-train/anomaly-ae-train_max_20260711T085421.359195Z.json; evidence_sha256=a3393e127285bbb9dcba5af692a7cbd0105df1f6a25577c1b51fd9d491c27803; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=0ec4d3e1c415944227d0754d170edb0addc1d925; profile=max; device_requested=cpu; data_mode=real; execution_backend=pytorch-cpu; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=anomaly_auroc; metric_values_by_seed=0.9666612742658038, 0.9701133535454093, 0.9665642110502738, 0.9645168333408751, 0.9658918955608851; anomaly_auroc=0.9665642110502738; median=0.9665642110502738; min=0.9645168333408751; max=0.9701133535454093; mean=0.9667495135526494; sample_stdev=0.0020662715355119; wall_seconds_median=6.636674875044264; wall_seconds_min=6.580142458085902; wall_seconds_max=6.698525084066205; wall_seconds_mean=6.642767050047405; wall_seconds_sample_stdev=0.04565513013724323; accepted_runs=5; baseline_note=This packet documents the superseded zero-versus-all-digits protocol and is retained only for historical traceability. It is not eligible evidence for mnist-hard-curve-v1. A clean five-seed replacement packet is required before promotion. |
| Baseline record role | historical-protocol-superseded |
| Baseline disclosure | Retained for historical traceability only; it does not validate the current contract and is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | False |
| Baseline evidence file | reference_results/anomaly-ae-train/anomaly-ae-train_max_20260711T085421.359195Z.json |
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
- replacement blocker: the committed packet is historical and uses a protocol superseded by the current benchmark contract; a clean reference sweep is required before promotion. Reason: The current mnist-hard-curve-v1 contract adds classwise gates and no-training controls; this packet used the former zero-versus-all-digits protocol.

## Source Provenance

- Registry provenance: MLPerf Tiny AD benchmark; Koizumi et al. 2019 (ToyADMOS architecture)
- Runner min: mlperf.runners.tiny:run_anomaly_ae_min
- Runner max: mlperf.runners.tiny:run_anomaly_ae_max
