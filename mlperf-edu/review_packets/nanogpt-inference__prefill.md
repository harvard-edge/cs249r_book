# MLPerf EDU Review Packet: `nanogpt-inference --variant prefill`

## Summary

| Field | Value |
|---|---|
| Internal ID | nanogpt-prefill |
| Run selector | nanogpt-inference --variant prefill |
| Suite | language |
| Public status | performance-bearing |
| Scenario | offline |
| Model | nanogpt-12m |
| Dataset | prompt-suite-local |
| Canonical workload | nanogpt-inference |
| Variant | prefill |

## Reviewer Commands

```bash
OUTPUT_DIR="submissions/review-nanogpt-inference__prefill"
mlperf fetch --workload nanogpt-train --profile max
mlperf run --workload nanogpt-train --profile max --output-dir "$OUTPUT_DIR"
mlperf fetch --workload nanogpt-inference --variant prefill --profile max
mlperf run --workload nanogpt-inference --variant prefill --profile max --output-dir "$OUTPUT_DIR"
for manifest in "$OUTPUT_DIR"/*.provd.json; do mlperf verify "$manifest"; done
mlperf grade "$OUTPUT_DIR" --output "$OUTPUT_DIR/grade.json"
```

## Functional Contract

| Field | Value |
|---|---|
| Functional metric | prefill_tokens_per_sec |
| Condition | a checkpoint-backed prefill run completes with a quality-approved NanoGPT training report, matching checkpoint SHA-256 provenance, and positive measured throughput |
| Independent reference runs | 5 |
| Reviewer notes | Quality is inherited from the nanogpt-train checkpoint; this workload reports serving prefill performance.; Review artifacts must retain the training report, training provenance manifest, checkpoint, checkpoint digest, per-run latency samples, and inference provenance manifest together. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol | profile=max; reference_runs=5; backend=PyTorch CPU or accelerator path recorded with the complete execution and hardware fingerprint; machine_class=laptop-class CPU or laptop-class accelerator; dataset_mode=checkpoint-backed fixed-shape prompt batch using a quality-approved NanoGPT training checkpoint; seeds=0, 1, 2, 3, 4; aggregation=median prefill_tokens_per_sec across five independent seeded runs; repeatability_metric=sample coefficient of variation of prefill_tokens_per_sec across the five reference runs; repeatability_limit=0.05; repeatability_action=withhold the performance reference and rerun the complete protocol when the coefficient of variation exceeds 5%; functional_acceptance=every run must pass the checkpoint, provenance, and positive-throughput functional gate; artifact_policy=create a new immutable attempt directory and SHA-256 index every report, provenance manifest, checkpoint, export, and runner-declared artifact; rerun_policy=if any run fails or times out, create a new attempt and rerun all five seeds; never replace one seed in an existing attempt |
| Measurement protocol | primary_metric=prefill_tokens_per_sec; warmup_runs=3; measured_runs=10; latency_statistics=median, p90, p99; raw_sample_metric=prefill_latency_samples_s; timing_scope=synchronized checkpoint-backed forward passes over one fixed-shape prompt batch |
| Checkpoint contract | source_workload=nanogpt-train; digest=sha256; required_artifacts=training report, training provenance manifest, checkpoint, inference report, inference provenance manifest |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-prefill_max_20260711T084140.159367Z; evidence_file=reference_results/nanogpt-prefill/nanogpt-prefill_max_20260711T084140.159367Z.json; evidence_sha256=bc3e8f01c279d1d2bbf0f8a24b15e85270584a5d35e16883a9751b4b5a04b68b; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=0ec4d3e1c415944227d0754d170edb0addc1d925; profile=max; device_requested=mps; data_mode=checkpoint-backed; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; source_training_evidence_id=nanogpt-train_max_20260711T083347.154092Z; source_training_evidence_sha256=6f58270368d1e75445a7c7bcc8c20ca710bb9994090aa4705440525ef8cc0638; source_training_seed=4; source_training_checkpoint_sha256=a0d2f31a747355d47d11c6aa77eb09faf2232f84cb519accb286a78159fb2d8a; source_training_package_sha256=0b0173d78e2c3315c4687b6319beb8a2826c98bce7f52710542f4b496edadd20; seeds=0, 1, 2, 3, 4; primary_metric=prefill_tokens_per_sec; metric_values_by_seed=115665.7588935296, 125183.0113402264, 114052.80426627811, 124578.1801076891, 125062.33989775984; prefill_tokens_per_sec=124578.1801076891; median=124578.1801076891; min=114052.80426627811; max=125183.0113402264; mean=120908.41890109661; sample_stdev=5556.061632502751; wall_seconds_median=1.6132285830099136; wall_seconds_min=1.5671893749386072; wall_seconds_max=1.6838348750025034; wall_seconds_mean=1.6172479081898927; wall_seconds_sample_stdev=0.04975564510833179; accepted_runs=5; functional_passes=5; coefficient_of_variation=0.04595264484475332; baseline_note=Clean five-run project reference from exact source commit 0ec4d3e1. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every run passed its functional gate. The primary performance metric has 4.60% sample coefficient of variation across the five runs, within the 5% promotion limit. The speed is a machine observation, not a portable target. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/nanogpt-prefill/nanogpt-prefill_max_20260711T084140.159367Z.json |
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
| Dataset asset | prompt-suite-local |
| Dataset source | mlperf-edu://bundled/prompts |
| Dataset license status | bundled-project-asset |
| Dataset release status | public-ok-bundled |
| Dataset citation | Bundled deterministic prompts maintained by MLPerf EDU. |

## Checkpoint Lineage

| Field | Value |
|---|---|
| Shared checkpoint | nanogpt-train |
| Quality dependency | nanogpt-train |
| Source run selector | nanogpt-train |
| Source quality | cross_entropy_loss lower 2.3 basis=reference_runs |
| Source baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-train_max_20260711T083347.154092Z; evidence_file=reference_results/nanogpt-train/nanogpt-train_max_20260711T083347.154092Z.json; evidence_sha256=6f58270368d1e75445a7c7bcc8c20ca710bb9994090aa4705440525ef8cc0638; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=0ec4d3e1c415944227d0754d170edb0addc1d925; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=cross_entropy_loss; metric_values_by_seed=2.1648468375205994, 2.0190127730369567, 2.088406854867935, 1.9744068205356597, 2.0788655877113342; cross_entropy_loss=2.0788655877113342; median=2.0788655877113342; min=1.9744068205356597; max=2.1648468375205994; mean=2.0651077747344972; sample_stdev=0.07251106376356625; wall_seconds_median=92.20449249993544; wall_seconds_min=91.68097287497949; wall_seconds_max=92.48971437499858; wall_seconds_mean=92.16539395838045; wall_seconds_sample_stdev=0.30237275010338804; accepted_runs=5; baseline_note=Clean five-run project reference from exact source commit 0ec4d3e1. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every seed passed the declared quality gate. The median-quality seed supplies the content-addressed checkpoint lineage used by the two NanoGPT performance references. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Policy | Preserve the source training report and .provd.json alongside checkpoint-backed inference results. |

## Public Review Notes

- external-publication blocker: registry declares local-handoff reference evidence, but no published package URL is recorded
- external-publication blocker: the raw reference package for shared checkpoint source nanogpt-train is not yet publicly retrievable

## Source Provenance

- Registry provenance: GPT-2 prefill regime; corresponds to MLPerf Inference 'prompt processing' phase
- Runner min: mlperf.runners.nanogpt:run_prefill_min
- Runner max: mlperf.runners.nanogpt:run_prefill_max
