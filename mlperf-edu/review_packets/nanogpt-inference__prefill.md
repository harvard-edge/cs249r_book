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
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-prefill_max_20260711T062700.039263Z; evidence_file=reference_results/nanogpt-prefill/nanogpt-prefill_max_20260711T062700.039263Z.json; evidence_sha256=dac0ec14b806b33a96349d4f4635c0b02b72ee665203589c95015ad33b019dd4; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=mps; data_mode=checkpoint-backed; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; source_training_evidence_id=nanogpt-train_max_20260711T061237.491822Z; source_training_evidence_sha256=3b748a64fdc7a942ad2abf20e3e13ce5af914b7ce987d8d810d0d051b1ab1807; source_training_package_sha256=1403c78341e7598b9cc4c0a10e67d54886edb58996c7622a0c3f2ef9f880bfa3; seeds=0, 1, 2, 3, 4; primary_metric=prefill_tokens_per_sec; metric_values_by_seed=117386.00003438497, 124632.3280060506, 117797.21723768071, 127046.58375707333, 116623.64244940683; prefill_tokens_per_sec=117797.21723768071; median=117797.21723768071; min=116623.64244940683; max=127046.58375707333; mean=120697.15429691928; sample_stdev=4789.7735557168835; wall_seconds_median=1.6129625419853255; wall_seconds_min=1.547860375023447; wall_seconds_max=1.6937594580231234; wall_seconds_mean=1.6276146333897485; wall_seconds_sample_stdev=0.05737648489803208; accepted_runs=5; functional_passes=5; baseline_note=Clean checkpoint-backed five-seed project reference from exact source commit 318cd842. Every run passed checkpoint, provenance, quality-dependency, and positive-throughput gates. The throughput is an Apple M5 Max observation, not a portable speed target. Raw packages are retained for local reviewer handoff but have no public URL. This is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/nanogpt-prefill/nanogpt-prefill_max_20260711T062700.039263Z.json |
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
| Source baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-train_max_20260711T061237.491822Z; evidence_file=reference_results/nanogpt-train/nanogpt-train_max_20260711T061237.491822Z.json; evidence_sha256=3b748a64fdc7a942ad2abf20e3e13ce5af914b7ce987d8d810d0d051b1ab1807; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=cross_entropy_loss; metric_values_by_seed=2.1939003109931945, 2.024900829792023, 2.0567784726619722, 1.9997187912464143, 2.102234035730362; cross_entropy_loss=2.0567784726619722; median=2.0567784726619722; min=1.9997187912464143; max=2.1939003109931945; mean=2.0755064880847933; sample_stdev=0.07646388904411043; wall_seconds_median=62.0149393749889; wall_seconds_min=61.55349308392033; wall_seconds_max=65.91061908297706; wall_seconds_mean=63.10957170857582; wall_seconds_sample_stdev=1.8439178904723237; accepted_runs=5; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run passed. The median-seed checkpoint package is content-addressed for downstream prefill and decode lineage. Raw packages are retained for local reviewer handoff but have no public URL. This is not an MLCommons-verified result. |
| Policy | Preserve the source training report and .provd.json alongside checkpoint-backed inference results. |

## Public Review Notes

- external-publication blocker: reference evidence package is retained for local handoff but is not yet publicly retrievable
- external-publication blocker: the raw reference package for shared checkpoint source nanogpt-train is not yet publicly retrievable

## Source Provenance

- Registry provenance: GPT-2 prefill regime; corresponds to MLPerf Inference 'prompt processing' phase
- Runner min: mlperf.runners.nanogpt:run_prefill_min
- Runner max: mlperf.runners.nanogpt:run_prefill_max
