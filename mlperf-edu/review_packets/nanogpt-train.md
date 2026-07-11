# MLPerf EDU Review Packet: `nanogpt-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | nanogpt-train |
| Run selector | nanogpt-train |
| Suite | language |
| Public status | score-bearing |
| Scenario | training |
| Model | nanogpt-12m |
| Dataset | tinyshakespeare |

## Reviewer Commands

```bash
OUTPUT_DIR="submissions/review-nanogpt-train"
mlperf fetch --workload nanogpt-train --profile max
mlperf run --workload nanogpt-train --profile max --output-dir "$OUTPUT_DIR"
for manifest in "$OUTPUT_DIR"/*.provd.json; do mlperf verify "$manifest"; done
mlperf grade "$OUTPUT_DIR" --output "$OUTPUT_DIR/grade.json"
```

## Quality Contract

| Field | Value |
|---|---|
| Metric | cross_entropy_loss |
| Target | 2.3 |
| Direction | lower |
| Target basis | reference_runs |
| Reference runs | 5 |
| Acceptance rule | median validation loss must be <= target |
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched Project Gutenberg Shakespeare source with deterministic TinyShakespeare excerpt recipe, fixed tokenizer, split, and seed; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median validation loss across five independent reference runs; repeatability_metric=sample coefficient of variation of train_and_eval_seconds across the five reference runs; repeatability_limit=0.05; repeatability_action=withhold the performance reference and rerun the complete protocol when the coefficient of variation exceeds 5%; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Measurement protocol | primary_metric=train_and_eval_seconds; scenario=training; timing_scope=fixed canonical training loop plus scheduled validation; included_phases=optimizer steps, scheduled validation, learning-rate scheduler steps when configured; excluded_phases=dataset or asset fetching, model construction, checkpoint serialization, report and provenance serialization; device_synchronization=synchronize immediately before and after the measured train-and-evaluation region; outer_reference_runs=5 |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-train_max_20260711T185153.818986Z; evidence_file=reference_results/nanogpt-train/nanogpt-train_max_20260711T185153.818986Z.json; evidence_sha256=9dee70ef74c70bae260679d1920395cd504c32ed2bfe33cdbe392d5bf772c13d; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=b4366b7614f0bb8ba0a1d6224832d4caea64e68a; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=train_and_eval_seconds; metric_values_by_seed=70.8777947919998, 72.20951520799917, 73.11595533299987, 73.01703029100008, 73.54263020800045; train_and_eval_seconds=73.01703029100008; median=73.01703029100008; min=70.8777947919998; max=73.54263020800045; mean=72.55258516639988; sample_stdev=1.0530793176819444; wall_seconds_median=74.87967737500003; wall_seconds_min=72.76015512499998; wall_seconds_max=75.39698062499974; wall_seconds_mean=74.43470659159993; wall_seconds_sample_stdev=1.0453078186344293; accepted_runs=5; quality_metric=cross_entropy_loss; quality_values_by_seed=2.1900793765846593, 2.0199239628215775, 2.0787073644367147, 1.9981003942243338, 2.1139374042686123; cross_entropy_loss=2.0787073644367147; quality_median=2.0787073644367147; quality_min=1.9981003942243338; quality_max=2.1900793765846593; quality_mean=2.0801497004671794; quality_sample_stdev=0.07679103313958817; baseline_note=Clean five-run project reference from exact source commit b4366b76. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every seed passed the declared quality gate. The median-quality seed supplies the content-addressed checkpoint lineage used by the two NanoGPT performance references. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline record role | current-review-evidence |
| Baseline disclosure | Project reference evidence; not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/nanogpt-train/nanogpt-train_max_20260711T185153.818986Z.json |
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
| Dataset asset | tinyshakespeare |
| Dataset source | https://www.gutenberg.org/files/100/100-0.txt |
| Dataset license status | public-domain-us |
| Dataset release status | public-ok-fetch-only |
| Dataset release next step | Keep generated-corpus recipe, source URL, and hashes in public artifacts. |
| Dataset citation | Project Gutenberg eBook 100: The Complete Works of William Shakespeare. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- external-publication blocker: registry declares local-handoff reference evidence, but no published package URL is recorded

## Source Provenance

- Registry provenance: Vaswani et al. 2017 (Transformer); maps to MLPerf Training GPT-3/LLaMA
- Runner min: mlperf.runners.nanogpt:run_min
- Runner max: mlperf.runners.nanogpt:run_max
