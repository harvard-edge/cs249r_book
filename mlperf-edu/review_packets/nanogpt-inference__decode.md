# MLPerf EDU Review Packet: `nanogpt-inference --variant decode`

## Summary

| Field | Value |
|---|---|
| Internal ID | nanogpt-decode |
| Run selector | nanogpt-inference --variant decode |
| Suite | language |
| Public status | performance-bearing |
| Scenario | single_stream |
| Model | nanogpt-12m |
| Dataset | prompt-suite-local |
| Canonical workload | nanogpt-inference |
| Variant | decode |

## Reviewer Commands

```bash
OUTPUT_DIR="submissions/review-nanogpt-inference__decode"
mlperf fetch --workload nanogpt-train --profile max
mlperf run --workload nanogpt-train --profile max --output-dir "$OUTPUT_DIR"
mlperf fetch --workload nanogpt-inference --variant decode --profile max
mlperf run --workload nanogpt-inference --variant decode --profile max --output-dir "$OUTPUT_DIR"
for manifest in "$OUTPUT_DIR"/*.provd.json; do mlperf verify "$manifest"; done
mlperf grade "$OUTPUT_DIR" --output "$OUTPUT_DIR/grade.json"
```

## Functional Contract

| Field | Value |
|---|---|
| Functional metric | decode_steps |
| Condition | a checkpoint-backed decode run completes 64 configured steps with a quality-approved NanoGPT training report, matching checkpoint SHA-256 provenance, and positive measured output throughput |
| Independent reference runs | 5 |
| Reviewer notes | Quality is inherited from the nanogpt-train checkpoint; this workload reports decode latency, KV-cache behavior, and throughput.; Review artifacts must retain request-level TTFT and per-token ITL samples together with the training and inference provenance chain. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol | profile=max; reference_runs=5; backend=PyTorch CPU or accelerator path recorded with the complete execution and hardware fingerprint; machine_class=laptop-class CPU or laptop-class accelerator; dataset_mode=checkpoint-backed requests using a quality-approved NanoGPT training checkpoint; seeds=0, 1, 2, 3, 4; aggregation=median output_tokens_per_sec across five independent seeded runs; repeatability_metric=sample coefficient of variation of output_tokens_per_sec across the five reference runs; repeatability_limit=0.05; repeatability_action=withhold the performance reference and rerun the complete protocol when the coefficient of variation exceeds 5%; functional_acceptance=every run must complete all configured decode steps and pass checkpoint, provenance, and positive-throughput gates; artifact_policy=create a new immutable attempt directory and SHA-256 index every report, provenance manifest, checkpoint, export, and runner-declared artifact; rerun_policy=if any run fails or times out, create a new attempt and rerun all five seeds; never replace one seed in an existing attempt |
| Measurement protocol | primary_metric=output_tokens_per_sec; warmup_runs=3; measured_runs=20; decode_steps_per_request=64; latency_statistics=median, p90, p99; raw_sample_metrics=request_ttft_samples_s, first_decode_latency_samples_s, itl_samples_s, request_end_to_end_samples_s; timing_scope=synchronized sequential microbenchmark requests; request TTFT spans prompt prefill through first-token selection, every ITL sample measures one subsequent cached-token step, and the first ITL is retained separately as first-decode latency; warmups are untimed |
| Checkpoint contract | source_workload=nanogpt-train; digest=sha256; required_artifacts=training report, training provenance manifest, checkpoint, inference report, inference provenance manifest |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-decode_max_20260711T191026.069877Z; evidence_file=reference_results/nanogpt-decode/nanogpt-decode_max_20260711T191026.069877Z.json; evidence_sha256=0179a12d1e45afc11a9584f31cffc80e2104cbd638533fd82ab12a2a30b5b391; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=b4366b7614f0bb8ba0a1d6224832d4caea64e68a; profile=max; device_requested=mps; data_mode=checkpoint-backed; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; source_training_evidence_id=nanogpt-train_max_20260711T185153.818986Z; source_training_evidence_sha256=9dee70ef74c70bae260679d1920395cd504c32ed2bfe33cdbe392d5bf772c13d; source_training_seed=2; source_training_checkpoint_sha256=fa28d9fde8df8f82530bce3733962c4422e52ed45dbb192cddfbe780b4282e5d; source_training_package_sha256=3ff66f1846c1aadf35aedaa17101a0ca5a96aeb23a258297d3fbb66c45f0bed0; seeds=0, 1, 2, 3, 4; primary_metric=output_tokens_per_sec; metric_values_by_seed=124.55659409255762, 125.08664594927815, 123.49364000721026, 124.60541358064455, 124.63162784700309; output_tokens_per_sec=124.60541358064455; median=124.60541358064455; min=123.49364000721026; max=125.08664594927815; mean=124.47478429533874; sample_stdev=0.5885098874462094; wall_seconds_median=13.909272208000402; wall_seconds_min=13.838977041999897; wall_seconds_max=13.949946541999452; wall_seconds_mean=13.904206675000022; wall_seconds_sample_stdev=0.04313666472037722; accepted_runs=5; functional_passes=5; coefficient_of_variation=0.004727944625715231; baseline_note=Clean five-run project reference from exact source commit b4366b76. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every run passed its functional gate. The primary performance metric has 0.47% sample coefficient of variation across the five runs, within the 5% promotion limit. The speed is a machine observation, not a portable target. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline record role | current-review-evidence |
| Baseline disclosure | Project reference evidence; not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/nanogpt-decode/nanogpt-decode_max_20260711T191026.069877Z.json |
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
| Dataset citation | Versioned deterministic prompts maintained by MLPerf EDU contributors. |

## Checkpoint Lineage

| Field | Value |
|---|---|
| Shared checkpoint | nanogpt-train |
| Quality dependency | nanogpt-train |
| Source run selector | nanogpt-train |
| Source quality | cross_entropy_loss lower 2.3 basis=reference_runs |
| Source baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-train_max_20260711T185153.818986Z; evidence_file=reference_results/nanogpt-train/nanogpt-train_max_20260711T185153.818986Z.json; evidence_sha256=9dee70ef74c70bae260679d1920395cd504c32ed2bfe33cdbe392d5bf772c13d; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=b4366b7614f0bb8ba0a1d6224832d4caea64e68a; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=train_and_eval_seconds; metric_values_by_seed=70.8777947919998, 72.20951520799917, 73.11595533299987, 73.01703029100008, 73.54263020800045; train_and_eval_seconds=73.01703029100008; median=73.01703029100008; min=70.8777947919998; max=73.54263020800045; mean=72.55258516639988; sample_stdev=1.0530793176819444; wall_seconds_median=74.87967737500003; wall_seconds_min=72.76015512499998; wall_seconds_max=75.39698062499974; wall_seconds_mean=74.43470659159993; wall_seconds_sample_stdev=1.0453078186344293; accepted_runs=5; quality_metric=cross_entropy_loss; quality_values_by_seed=2.1900793765846593, 2.0199239628215775, 2.0787073644367147, 1.9981003942243338, 2.1139374042686123; cross_entropy_loss=2.0787073644367147; quality_median=2.0787073644367147; quality_min=1.9981003942243338; quality_max=2.1900793765846593; quality_mean=2.0801497004671794; quality_sample_stdev=0.07679103313958817; baseline_note=Clean five-run project reference from exact source commit b4366b76. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every seed passed the declared quality gate. The median-quality seed supplies the content-addressed checkpoint lineage used by the two NanoGPT performance references. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Policy | Preserve the source training report and .provd.json alongside checkpoint-backed inference results. |

## Public Review Notes

- external-publication blocker: registry declares local-handoff reference evidence, but no published package URL is recorded
- external-publication blocker: the raw reference package for shared checkpoint source nanogpt-train is not yet publicly retrievable

## Source Provenance

- Registry provenance: GPT-2 autoregressive decode with a KV cache, representing the decode phase implemented by serving systems such as vLLM, TensorRT-LLM, and TGI
- Runner min: mlperf.runners.nanogpt:run_decode_min
- Runner max: mlperf.runners.nanogpt:run_decode_max
