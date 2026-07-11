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
| Baseline record | evidence_status=committed-reference-summary; review_eligible=False; protocol_compatibility=superseded; replacement_required=True; superseded_reason=The current contract defines a synchronized canonical train-and-evaluation timing boundary; this packet predates that measurement protocol.; evidence_tier=public-candidate; evidence_id=resnet18-train_max_20260711T084315.135227Z; evidence_file=reference_results/resnet18-train/resnet18-train_max_20260711T084315.135227Z.json; evidence_sha256=fa08884c733c616157cac879c46c9833bb1f5c9ade1dfb45648658d05c491aa8; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=0ec4d3e1c415944227d0754d170edb0addc1d925; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=top1_accuracy; metric_values_by_seed=0.8781, 0.8755, 0.8673, 0.863, 0.875; top1_accuracy=0.875; median=0.875; min=0.863; max=0.8781; mean=0.87178; sample_stdev=0.006350354320823367; wall_seconds_median=44.01704033301212; wall_seconds_min=43.64901420800015; wall_seconds_max=45.53280479100067; wall_seconds_mean=44.20926372481044; wall_seconds_sample_stdev=0.7716991522189321; accepted_runs=5; baseline_note=This packet predates the synchronized canonical train-and-evaluation timing boundary. It is retained only for historical traceability and cannot support the current contract. A clean five-seed replacement packet is required. |
| Baseline record role | historical-protocol-superseded |
| Baseline disclosure | Retained for historical traceability only; it does not validate the current contract and is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | False |
| Baseline evidence file | reference_results/resnet18-train/resnet18-train_max_20260711T084315.135227Z.json |
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
- replacement blocker: the committed packet is historical and uses a protocol superseded by the current benchmark contract; a clean reference sweep is required before promotion. Reason: The current contract defines a synchronized canonical train-and-evaluation timing boundary; this packet predates that measurement protocol.

## Source Provenance

- Registry provenance: He et al. 2016 (ResNet); local ResNet-18 implementation (no torchvision.models dependency); torchvision datasets/transforms are used only to fetch and preprocess Fashion-MNIST
- Runner min: mlperf.runners.vision:run_resnet18_min
- Runner max: mlperf.runners.vision:run_resnet18_max
