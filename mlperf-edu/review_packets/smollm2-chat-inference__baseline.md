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
mlperf fetch --workload smollm2-chat-inference --variant baseline --profile max --dry-run
mlperf run --workload smollm2-chat-inference --variant baseline --profile max
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
| Reference protocol | profile=max; reference_runs=5; backend=Transformers and PyTorch CPU or accelerator path recorded with the complete execution and hardware fingerprint; machine_class=laptop-class CPU or laptop-class accelerator; dataset_mode=local-prompt inference with the pinned model revision and bundled deterministic quality suite; seeds=0, 1, 2, 3, 4; aggregation=median output_tokens_per_sec across five independent seeded runs; functional_acceptance=every run must generate at least eight tokens and pass continuation perplexity at or below 10; artifact_policy=create a new immutable attempt directory and SHA-256 index every report, provenance manifest, model metadata file, export, and runner-declared artifact; rerun_policy=if any run fails or times out, create a new attempt and rerun all five seeds; never replace one seed in an existing attempt |
| Measurement protocol | primary_metric=output_tokens_per_sec; warmup_runs=1; measured_runs=5; latency_statistics=median, p90, p99; timed_phases=prefill, greedy generation; timing_scope=synchronized timings over one fixed prompt and 16 requested decode tokens |
| Checkpoint contract |  |
| Task-quality evaluation | suite=mlperf-edu-slm-quality/0.1; asset=src/mlperf_edu/slm_quality_prompts.json; asset_sha256=5fa25872d0b7dc986b12137256b16fd6329267d1640f03e4e04f1dc4e8c8ed5f; cases=4; method=mean continuation-only negative log likelihood over the deterministic suite; metric=continuation_perplexity; maximum=10.0 |
| Baseline evidence status | not declared |
| Baseline review eligible | not declared |
| Baseline evidence file | not declared |
| Reference package availability | not declared |
| External publication status | not declared |
| External publication URL | not declared |
| Calibration observation | evidence_status=local-calibration-awaiting-committed-review-artifact; review_eligible=False; evidence_id=slm-decode_max_20260711T051714.597457Z; evidence_sha256=19d6cf1b3d87810abf3dee02f56c79da5007ecc66b5a270e6371a37610fd4f58; seeds=0, 1, 2, 3, 4; primary_metric=output_tokens_per_sec; output_tokens_per_second_values=87.1195555372376, 73.84679313530995, 74.040410374211, 73.66910712629223, 78.26006664912396; output_tokens_per_second_median=74.040410374211; output_tokens_per_second_min=73.66910712629223; output_tokens_per_second_max=87.1195555372376; output_tokens_per_second_stdev=5.767155639346823; wall_seconds_median=4.034003582899459; wall_seconds_min=4.009392875013873; wall_seconds_max=4.6925652080681175; functional_passes=5; generated_tokens=16; quality_perplexity=7.600481673911702; quality_mean_nll=2.028211623430252; baseline_note=Five independent development executions of the pinned revision; each execution used one warmup and five measured requests and passed the output-length and continuation-perplexity gates. This content-addressed local summary is not a review baseline until a clean-commit public-candidate packet is retained. |

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
| Model source | https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct |
| Model license | Apache-2.0 |
| Model rationale | SmolLM2-135M-Instruct is the default because it is Apache-2.0, public on Hugging Face, small enough for laptop CPU/MPS setup runs, and large enough to exercise Transformer decode, KV-cache, quantization, batching, and long-context behavior. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- calibration values are informational and are not a review baseline; evidence status is local-calibration-awaiting-committed-review-artifact

## Source Provenance

- Registry provenance: Off-the-shelf small language model decode path for local serving, quantization, LoRA, KV-cache, and backend comparison studies.
- Runner min: mlperf.runners.slm:run_decode_min
- Runner max: mlperf.runners.slm:run_decode_max
