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
mlperf fetch --workload nanogpt-inference --variant prefill --profile max --dry-run
mlperf run --workload nanogpt-inference --variant prefill --profile max
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
| Reference protocol | profile=max; reference_runs=5; backend=PyTorch CPU or accelerator path recorded with the complete execution and hardware fingerprint; machine_class=laptop-class CPU or laptop-class accelerator; dataset_mode=checkpoint-backed fixed-shape prompt batch using a quality-approved NanoGPT training checkpoint; seeds=0, 1, 2, 3, 4; aggregation=median prefill_tokens_per_sec across five independent seeded runs; functional_acceptance=every run must pass the checkpoint, provenance, and positive-throughput functional gate; artifact_policy=create a new immutable attempt directory and SHA-256 index every report, provenance manifest, checkpoint, export, and runner-declared artifact; rerun_policy=if any run fails or times out, create a new attempt and rerun all five seeds; never replace one seed in an existing attempt |
| Measurement protocol | primary_metric=prefill_tokens_per_sec; warmup_runs=3; measured_runs=10; latency_statistics=median, p90, p99; raw_sample_metric=prefill_latency_samples_s; timing_scope=synchronized checkpoint-backed forward passes over one fixed-shape prompt batch |
| Checkpoint contract | source_workload=nanogpt-train; digest=sha256; required_artifacts=training report, training provenance manifest, checkpoint, inference report, inference provenance manifest |
| Task-quality evaluation |  |
| Baseline evidence status | not declared |
| Baseline review eligible | not declared |
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
| Dataset asset | prompt-suite-local |
| Dataset source | mlperf-edu://bundled/prompts |
| Dataset license status | bundled-project-asset |
| Dataset release status | public-ok-bundled |
| Dataset release next step |  |
| Dataset citation | Bundled deterministic prompts maintained by MLPerf EDU. |

## Checkpoint Lineage

| Field | Value |
|---|---|
| Shared checkpoint | nanogpt-train |
| Quality dependency | nanogpt-train |
| Source run selector | nanogpt-train |
| Source quality | cross_entropy_loss lower 2.3 basis=reference_runs |
| Source baseline record | evidence_status=pending-clean-public-candidate-reference-summary; review_eligible=False; calibration_tier=development; development_summary_id=nanogpt-train_max_20260711T045626.172816Z; development_summary_sha256=ca74f0561f563f5b478038d555ad2e367ff155a9965e0e478d7a0f7b8be9ddd5; development_summary_availability=local-handoff; seeds=0, 1, 2, 3, 4; cross_entropy_loss_by_seed=2.1344807863235475, 2.056773912906647, 2.1026881873607635, 1.9816058337688447, 2.087835317850113; cross_entropy_loss=2.087835317850113; val_loss=2.087835317850113; five_seed_cross_entropy_median=2.087835317850113; five_seed_cross_entropy_min=1.9816058337688447; five_seed_cross_entropy_max=2.1344807863235475; five_seed_cross_entropy_stdev=0.05808833675616628; epochs=25; duration_seconds_median=127.42291408299934; duration_seconds_min=121.23956066707615; duration_seconds_max=130.2600357090123; baseline_note=Fresh real-data Apple MPS development calibration across seeds 0-4. The values are not review eligible until a clean public-candidate sweep commits a complete artifact index and retains its content-addressed raw package for review. |
| Policy | Preserve the source training report and .provd.json alongside checkpoint-backed inference results. |

## Public Review Notes

- shared checkpoint source nanogpt-train is not backed by a committed reference summary; evidence status is pending-clean-public-candidate-reference-summary

## Source Provenance

- Registry provenance: GPT-2 prefill regime; corresponds to MLPerf Inference 'prompt processing' phase
- Runner min: mlperf.runners.nanogpt:run_prefill_min
- Runner max: mlperf.runners.nanogpt:run_prefill_max
