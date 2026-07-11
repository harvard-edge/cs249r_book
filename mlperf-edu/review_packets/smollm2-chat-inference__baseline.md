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
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=slm-decode_max_20260711T062558.103544Z; evidence_file=reference_results/slm-decode/slm-decode_max_20260711T062558.103544Z.json; evidence_sha256=e8289a8b809c02c37f22a238fd08b0108f08be596fbf5c5c54400040c6633bb2; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=mps; data_mode=local-prompt; execution_backend=transformers-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=output_tokens_per_sec; metric_values_by_seed=127.92388533412783, 137.26889899463168, 90.8441621235267, 135.3306353387672, 101.83907035935408; output_tokens_per_sec=127.92388533412783; median=127.92388533412783; min=90.8441621235267; max=137.26889899463168; mean=118.64133043008151; sample_stdev=21.015967404904472; wall_seconds_median=3.1219681249931455; wall_seconds_min=2.8841313329758123; wall_seconds_max=3.6776643749326468; wall_seconds_mean=3.2093715915689245; wall_seconds_sample_stdev=0.32529526741613135; accepted_runs=5; functional_passes=5; generated_tokens=16; quality_perplexity=7.600481673911702; quality_mean_nll=2.028211623430252; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run used the pinned model revision, passed the output-length and continuation-perplexity gates, and recorded repeated synchronized timings. The 17.7% cross-seed throughput coefficient of variation is disclosed as a machine observation, not a speed target. Raw packages are retained for local reviewer handoff but have no public URL. This is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/slm-decode/slm-decode_max_20260711T062558.103544Z.json |
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
| Model source | https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct |
| Model license | Apache-2.0 |
| Model rationale | SmolLM2-135M-Instruct is the default because it is Apache-2.0, public on Hugging Face, small enough for laptop CPU/MPS setup runs, and large enough to exercise Transformer decode, KV-cache, quantization, batching, and long-context behavior. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- external-publication blocker: reference evidence package is retained for local handoff but is not yet publicly retrievable

## Source Provenance

- Registry provenance: Off-the-shelf small language model decode path for local serving, quantization, LoRA, KV-cache, and backend comparison studies.
- Runner min: mlperf.runners.slm:run_decode_min
- Runner max: mlperf.runners.slm:run_decode_max
