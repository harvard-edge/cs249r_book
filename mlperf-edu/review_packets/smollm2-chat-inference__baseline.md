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
| Condition | default max run requests 16 decode tokens; generated_tokens must be >= 8 and token-weighted continuation perplexity must be <= 7 overall and <= 24 in the weakest category on the bundled deterministic quality suite |
| Independent reference runs | 5 |
| Reviewer notes | The pinned model must pass both output-length and continuation-perplexity gates before latency, throughput, memory, or energy results are eligible for review. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol | profile=max; reference_runs=5; backend=Transformers and PyTorch CPU or accelerator path recorded with the complete execution and hardware fingerprint; machine_class=laptop-class CPU or laptop-class accelerator; dataset_mode=local-prompt inference with the pinned model revision and bundled deterministic quality suite; seeds=0, 1, 2, 3, 4; aggregation=median output_tokens_per_sec across five independent seeded runs; repeatability_metric=sample coefficient of variation of output_tokens_per_sec across the five reference runs; repeatability_limit=0.05; repeatability_action=withhold the performance reference and rerun the complete protocol when the coefficient of variation exceeds 5%; functional_acceptance=every run must generate at least eight tokens, pass token-weighted continuation perplexity at or below 7, and keep every category at or below 24 perplexity; artifact_policy=create a new immutable attempt directory and SHA-256 index every report, provenance manifest, model metadata file, export, and runner-declared artifact; rerun_policy=if any run fails or times out, create a new attempt and rerun all five seeds; never replace one seed in an existing attempt |
| Measurement protocol | primary_metric=output_tokens_per_sec; warmup_runs=3; measured_runs=20; latency_statistics=median, p90, p99; timed_phases=prefill, greedy generation; raw_sample_metrics=prefill_latency_samples_s, request_ttft_samples_s, itl_samples_s, request_end_to_end_samples_s; timing_scope=one cache-reusing greedy request path per fixed prompt batch; TTFT spans prompt prefill through the first output token, ITL samples time only subsequent cached-token steps, and end-to-end latency spans the complete request |
| Task-quality evaluation | suite=mlperf-edu-slm-quality/0.2; fixture_version=2.0.0; asset=src/mlperf_edu/slm_quality_prompts.json; asset_sha256=3d6d06b99dd92f1cf86fcde10f77b4db060397003bf654cc52c3148087ede556; attribution=MLPerf EDU contributors; project-authored fixture released as CC0-1.0; cases=28; categories=7; aggregation=token-weighted-continuation-nll; category_guard=maximum-category-perplexity; method=teacher-forced continuation-only negative log likelihood; prompt and continuation tokens are encoded separately, prompt prefill creates one KV cache, every subsequent continuation token reuses that cache, and token losses are weighted by continuation-token count globally and within each category; every case ID, category, and continuation-token count is bound by the versioned fixture; metric=continuation_perplexity; maximum=7.0; worst_category_maximum=24.0; calibration={'model_id': 'HuggingFaceTB/SmolLM2-135M-Instruct', 'model_revision': '12fd25f77366fa6b3b4b768ec3050bf629380bac', 'device': 'mps', 'observed_perplexity': 5.227185511982561, 'observed_worst_category': 'benchmarking', 'observed_worst_category_perplexity': 18.94030982965928, 'observed_continuation_tokens': 75, 'evaluation_seconds': 2.3826990419765934, 'threshold_rationale': 'The overall and weakest-category ceilings leave 34% and 27% perplexity headroom, respectively, for backend numeric variation while remaining tight enough to reject material degradation. This bounded run calibrates the protocol but is not a promoted reference packet.'} |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=slm-decode_max_20260711T210317.517476Z; evidence_file=reference_results/slm-decode/slm-decode_max_20260711T210317.517476Z.json; evidence_sha256=5c939ce65d2d35b680460d6e46c071063803797ca1cc3d29c11754c46bdc1524; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=86738e4654d8f77ef1cec4698b30e0ebd20dd2b3; profile=max; device_requested=cpu; data_mode=local-prompt; execution_backend=transformers-cpu; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=output_tokens_per_sec; metric_values_by_seed=61.604884873546986, 60.6295646645449, 60.40828388113117, 61.57422195326553, 61.970347187173935; output_tokens_per_sec=61.57422195326553; median=61.57422195326553; min=60.40828388113117; max=61.970347187173935; mean=61.2374605119325; sample_stdev=0.6787125781509229; wall_seconds_median=25.966474417000427; wall_seconds_min=25.48668333400019; wall_seconds_max=26.531702792000942; wall_seconds_mean=26.002557341800276; wall_seconds_sample_stdev=0.40249357097571553; accepted_runs=5; functional_passes=5; coefficient_of_variation=0.011083290725595513; baseline_note=Clean five-run project reference from exact source commit 86738e46. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every run passed its functional gate. The primary performance metric has 1.11% sample coefficient of variation across the five runs, within the 5% promotion limit. The speed is a machine observation, not a portable target. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline record role | current-review-evidence |
| Baseline disclosure | Project reference evidence; not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/slm-decode/slm-decode_max_20260711T210317.517476Z.json |
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
