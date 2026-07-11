# MLPerf EDU Review Packet: `smollm2-chat-inference --variant baseline`

## Summary

| Field | Value |
|---|---|
| Internal ID | slm-decode |
| Run selector | smollm2-chat-inference --variant baseline |
| Suite | slm |
| Public status | performance-bearing |
| Scenario | single_stream |
| Model | SmolLM2-135M-Instruct |
| Dataset | prompt-suite-local |
| Canonical workload | smollm2-chat-inference |
| Variant | baseline |

## Reviewer Commands

```bash
OUTPUT_DIR="submissions/review-smollm2-chat-inference__baseline"
mlperf fetch --workload smollm2-chat-inference --variant baseline --profile max
mlperf run --workload smollm2-chat-inference --variant baseline --profile max --output-dir "$OUTPUT_DIR"
for manifest in "$OUTPUT_DIR"/*.provd.json; do mlperf verify "$manifest"; done
mlperf grade "$OUTPUT_DIR" --output "$OUTPUT_DIR/grade.json"
```

## Functional Contract

| Field | Value |
|---|---|
| Functional metric | generated_tokens |
| Condition | default max run requests 16 decode tokens; generated_tokens must be >= 8 and continuation perplexity must be <= 10 on the bundled deterministic quality suite |
| Independent reference runs | 5 |
| Reviewer notes | The pinned model must pass both output-length and continuation-perplexity gates before latency, throughput, memory, or energy results are eligible for review. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol | profile=max; reference_runs=5; backend=Transformers and PyTorch CPU or accelerator path recorded with the complete execution and hardware fingerprint; machine_class=laptop-class CPU or laptop-class accelerator; dataset_mode=local-prompt inference with the pinned model revision and bundled deterministic quality suite; seeds=0, 1, 2, 3, 4; aggregation=median output_tokens_per_sec across five independent seeded runs; repeatability_metric=sample coefficient of variation of output_tokens_per_sec across the five reference runs; repeatability_limit=0.05; repeatability_action=withhold the performance reference and rerun the complete protocol when the coefficient of variation exceeds 5%; functional_acceptance=every run must generate at least eight tokens and pass continuation perplexity at or below 10; artifact_policy=create a new immutable attempt directory and SHA-256 index every report, provenance manifest, model metadata file, export, and runner-declared artifact; rerun_policy=if any run fails or times out, create a new attempt and rerun all five seeds; never replace one seed in an existing attempt |
| Measurement protocol | primary_metric=output_tokens_per_sec; warmup_runs=3; measured_runs=20; latency_statistics=median, p90, p99; timed_phases=prefill, greedy generation; timing_scope=synchronized timings over one fixed prompt and 16 requested decode tokens |
| Task-quality evaluation | suite=mlperf-edu-slm-quality/0.1; asset=src/mlperf_edu/slm_quality_prompts.json; asset_sha256=5fa25872d0b7dc986b12137256b16fd6329267d1640f03e4e04f1dc4e8c8ed5f; cases=4; method=mean continuation-only negative log likelihood over the deterministic suite; metric=continuation_perplexity; maximum=10.0 |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=slm-decode_max_20260711T085533.624561Z; evidence_file=reference_results/slm-decode/slm-decode_max_20260711T085533.624561Z.json; evidence_sha256=c13f7b7afb626cd4f3cdcb9620693a95ce8d46881d1e8c6f18ba0234442f1185; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=0ec4d3e1c415944227d0754d170edb0addc1d925; profile=max; device_requested=mps; data_mode=local-prompt; execution_backend=transformers-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=output_tokens_per_sec; metric_values_by_seed=101.38529541609195, 102.48414654372193, 102.2509860916646, 100.94140984139794, 100.4041265980697; output_tokens_per_sec=101.38529541609195; median=101.38529541609195; min=100.4041265980697; max=102.48414654372193; mean=101.49319289818922; sample_stdev=0.8744157286630143; wall_seconds_median=7.074083874933422; wall_seconds_min=6.775917749968357; wall_seconds_max=7.900906833005138; wall_seconds_mean=7.1484462999971585; wall_seconds_sample_stdev=0.4419313057434206; accepted_runs=5; functional_passes=5; coefficient_of_variation=0.00861551108693729; baseline_note=Clean five-run project reference from exact source commit 0ec4d3e1. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every run passed its functional gate. The primary performance metric has 0.86% sample coefficient of variation across the five runs, within the 5% promotion limit. The speed is a machine observation, not a portable target. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/slm-decode/slm-decode_max_20260711T085533.624561Z.json |
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
| Model source | https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct |
| Model license | Apache-2.0 |
| Model rationale | SmolLM2-135M-Instruct is the default because it is Apache-2.0, public on Hugging Face, small enough for laptop CPU/MPS setup runs, and large enough to exercise Transformer decode, KV-cache, quantization, batching, and long-context behavior. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- external-publication blocker: registry declares local-handoff reference evidence, but no published package URL is recorded

## Source Provenance

- Registry provenance: Off-the-shelf small language model decode path for local serving, quantization, LoRA, KV-cache, and backend comparison studies.
- Runner min: mlperf.runners.slm:run_decode_min
- Runner max: mlperf.runners.slm:run_decode_max
