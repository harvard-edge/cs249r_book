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
| Reference protocol | profile=max; backend=pytorch-cpu reference path unless the report declares a different backend; machine_class=laptop-class CPU or laptop-class accelerator with full hardware fingerprint; dataset_mode=fetched Project Gutenberg Shakespeare source with deterministic TinyShakespeare excerpt recipe, fixed tokenizer, split, and seed; no synthetic fallback; seeds=0, 1, 2, 3, 4; aggregation=median validation loss across five independent reference runs; artifact_policy=preserve JSON, HTML, CSV, .provd.json, run fingerprint, dataset asset metadata, and raw metric values for each run; rerun_policy=rerun all five references when model code, dataset preprocessing, optimizer schedule, PyTorch major version, or target hardware class changes |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-train_max_20260711T083347.154092Z; evidence_file=reference_results/nanogpt-train/nanogpt-train_max_20260711T083347.154092Z.json; evidence_sha256=6f58270368d1e75445a7c7bcc8c20ca710bb9994090aa4705440525ef8cc0638; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=0ec4d3e1c415944227d0754d170edb0addc1d925; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=cross_entropy_loss; metric_values_by_seed=2.1648468375205994, 2.0190127730369567, 2.088406854867935, 1.9744068205356597, 2.0788655877113342; cross_entropy_loss=2.0788655877113342; median=2.0788655877113342; min=1.9744068205356597; max=2.1648468375205994; mean=2.0651077747344972; sample_stdev=0.07251106376356625; wall_seconds_median=92.20449249993544; wall_seconds_min=91.68097287497949; wall_seconds_max=92.48971437499858; wall_seconds_mean=92.16539395838045; wall_seconds_sample_stdev=0.30237275010338804; accepted_runs=5; baseline_note=Clean five-run project reference from exact source commit 0ec4d3e1. Evidence semantics were recomputed from the raw reports and manifests during promotion. Every seed passed the declared quality gate. The median-quality seed supplies the content-addressed checkpoint lineage used by the two NanoGPT performance references. Content-addressed portable run packages are retained for local review, but no public package URL is recorded. This is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/nanogpt-train/nanogpt-train_max_20260711T083347.154092Z.json |
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
