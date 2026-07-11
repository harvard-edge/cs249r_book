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
| Baseline record | evidence_status=committed-reference-summary; review_eligible=False; protocol_compatibility=superseded; replacement_required=True; superseded_reason=The current single-stream contract uses a content-addressed fixed prompt, defines TTFT through first-token selection, treats the first cached-token step as the first ITL, and uses the revised checkpoint-training contract; this packet predates those semantics.; evidence_tier=public-candidate; evidence_id=nanogpt-decode_max_20260711T084155.577249Z; evidence_file=reference_results/nanogpt-decode/nanogpt-decode_max_20260711T084155.577249Z.json; evidence_sha256=bae464def8db558afcd377d506f9a25098b58d3ad934e03b474fa8b534beddf7; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=0ec4d3e1c415944227d0754d170edb0addc1d925; profile=max; device_requested=mps; data_mode=checkpoint-backed; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; source_training_evidence_id=nanogpt-train_max_20260711T083347.154092Z; source_training_evidence_sha256=6f58270368d1e75445a7c7bcc8c20ca710bb9994090aa4705440525ef8cc0638; source_training_seed=4; source_training_checkpoint_sha256=a0d2f31a747355d47d11c6aa77eb09faf2232f84cb519accb286a78159fb2d8a; source_training_package_sha256=0b0173d78e2c3315c4687b6319beb8a2826c98bce7f52710542f4b496edadd20; seeds=0, 1, 2, 3, 4; primary_metric=output_tokens_per_sec; metric_values_by_seed=127.1280762592526, 133.2852094860287, 131.83444650516844, 133.72933280518927, 129.7121922302817; output_tokens_per_sec=131.83444650516844; median=131.83444650516844; min=127.1280762592526; max=133.72933280518927; mean=131.13785145718413; sample_stdev=2.735094518987277; wall_seconds_median=14.503981499932706; wall_seconds_min=14.158116249949671; wall_seconds_max=14.96386087499559; wall_seconds_mean=14.52762437479105; wall_seconds_sample_stdev=0.2882922350491197; accepted_runs=5; functional_passes=5; coefficient_of_variation=0.020856636650633795; baseline_note=This packet predates the corrected single-stream TTFT definition and revised NanoGPT training lineage. It is retained only for historical traceability and cannot support the current contract. A clean five-run replacement packet is required. |
| Baseline record role | historical-protocol-superseded |
| Baseline disclosure | Retained for historical traceability only; it does not validate the current contract and is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | False |
| Baseline evidence file | reference_results/nanogpt-decode/nanogpt-decode_max_20260711T084155.577249Z.json |
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
| Source baseline record | evidence_status=committed-reference-summary; review_eligible=False; protocol_compatibility=superseded; replacement_required=True; superseded_reason=The current contract evaluates the complete nonoverlapping held-out validation split and measures the canonical train-and-evaluation region; this packet predates both protections.; evidence_tier=public-candidate; evidence_id=nanogpt-train_max_20260711T083347.154092Z; evidence_file=reference_results/nanogpt-train/nanogpt-train_max_20260711T083347.154092Z.json; evidence_sha256=6f58270368d1e75445a7c7bcc8c20ca710bb9994090aa4705440525ef8cc0638; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=0ec4d3e1c415944227d0754d170edb0addc1d925; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=cross_entropy_loss; metric_values_by_seed=2.1648468375205994, 2.0190127730369567, 2.088406854867935, 1.9744068205356597, 2.0788655877113342; cross_entropy_loss=2.0788655877113342; median=2.0788655877113342; min=1.9744068205356597; max=2.1648468375205994; mean=2.0651077747344972; sample_stdev=0.07251106376356625; wall_seconds_median=92.20449249993544; wall_seconds_min=91.68097287497949; wall_seconds_max=92.48971437499858; wall_seconds_mean=92.16539395838045; wall_seconds_sample_stdev=0.30237275010338804; accepted_runs=5; baseline_note=This packet predates complete held-out validation and the canonical timed train-and-evaluation boundary. It is retained only for historical traceability and cannot support the current contract. A clean five-seed replacement packet is required. |
| Policy | Preserve the source training report and .provd.json alongside checkpoint-backed inference results. |

## Public Review Notes

- external-publication blocker: registry declares local-handoff reference evidence, but no published package URL is recorded
- replacement blocker: the committed packet is historical and uses a protocol superseded by the current benchmark contract; a clean reference sweep is required before promotion. Reason: The current single-stream contract uses a content-addressed fixed prompt, defines TTFT through first-token selection, treats the first cached-token step as the first ITL, and uses the revised checkpoint-training contract; this packet predates those semantics.
- replacement blocker: shared checkpoint source nanogpt-train has only protocol-superseded historical evidence
- external-publication blocker: the raw reference package for shared checkpoint source nanogpt-train is not yet publicly retrievable

## Source Provenance

- Registry provenance: GPT-2 autoregressive decode with a KV cache, representing the decode phase implemented by serving systems such as vLLM, TensorRT-LLM, and TGI
- Runner min: mlperf.runners.nanogpt:run_decode_min
- Runner max: mlperf.runners.nanogpt:run_decode_max
