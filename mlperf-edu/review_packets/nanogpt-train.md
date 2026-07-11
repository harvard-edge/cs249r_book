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
mlperf fetch --workload nanogpt-train --profile max --dry-run
mlperf run --workload nanogpt-train --profile max
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
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-train_max_20260711T061237.491822Z; evidence_file=reference_results/nanogpt-train/nanogpt-train_max_20260711T061237.491822Z.json; evidence_sha256=3b748a64fdc7a942ad2abf20e3e13ce5af914b7ce987d8d810d0d051b1ab1807; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=cross_entropy_loss; metric_values_by_seed=2.1939003109931945, 2.024900829792023, 2.0567784726619722, 1.9997187912464143, 2.102234035730362; cross_entropy_loss=2.0567784726619722; median=2.0567784726619722; min=1.9997187912464143; max=2.1939003109931945; mean=2.0755064880847933; sample_stdev=0.07646388904411043; wall_seconds_median=62.0149393749889; wall_seconds_min=61.55349308392033; wall_seconds_max=65.91061908297706; wall_seconds_mean=63.10957170857582; wall_seconds_sample_stdev=1.8439178904723237; accepted_runs=5; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run passed. The median-seed checkpoint package is content-addressed for downstream prefill and decode lineage. Raw packages are retained for local reviewer handoff but have no public URL. This is not an MLCommons-verified result. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol |  |
| Measurement protocol |  |
| Checkpoint contract |  |
| Task-quality evaluation |  |
| Baseline record | evidence_status=committed-reference-summary; review_eligible=True; evidence_tier=public-candidate; evidence_id=nanogpt-train_max_20260711T061237.491822Z; evidence_file=reference_results/nanogpt-train/nanogpt-train_max_20260711T061237.491822Z.json; evidence_sha256=3b748a64fdc7a942ad2abf20e3e13ce5af914b7ce987d8d810d0d051b1ab1807; reference_package_availability=local-handoff; external_publication_status=pending; source_git_sha=318cd842efe3b90cbf56a109797d2bed4ad3dc09; profile=max; device_requested=mps; data_mode=real; execution_backend=pytorch-mps; hardware_chip=Apple M5 Max; seeds=0, 1, 2, 3, 4; primary_metric=cross_entropy_loss; metric_values_by_seed=2.1939003109931945, 2.024900829792023, 2.0567784726619722, 1.9997187912464143, 2.102234035730362; cross_entropy_loss=2.0567784726619722; median=2.0567784726619722; min=1.9997187912464143; max=2.1939003109931945; mean=2.0755064880847933; sample_stdev=0.07646388904411043; wall_seconds_median=62.0149393749889; wall_seconds_min=61.55349308392033; wall_seconds_max=65.91061908297706; wall_seconds_mean=63.10957170857582; wall_seconds_sample_stdev=1.8439178904723237; accepted_runs=5; baseline_note=Clean five-seed project reference from exact source commit 318cd842. Every run passed. The median-seed checkpoint package is content-addressed for downstream prefill and decode lineage. Raw packages are retained for local reviewer handoff but have no public URL. This is not an MLCommons-verified result. |
| Baseline evidence status | committed-reference-summary |
| Baseline review eligible | True |
| Baseline evidence file | reference_results/nanogpt-train/nanogpt-train_max_20260711T061237.491822Z.json |
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
| Dataset asset | tinyshakespeare |
| Dataset source | https://www.gutenberg.org/files/100/100-0.txt |
| Dataset license status | public-domain-us |
| Dataset release status | public-ok-fetch-only |
| Dataset release next step | Keep generated-corpus recipe, source URL, and hashes in public artifacts. |
| Dataset citation | Project Gutenberg eBook 100: The Complete Works of William Shakespeare. |

## Checkpoint Lineage

- No shared checkpoint dependency declared.

## Public Review Notes

- external-publication blocker: reference evidence package is retained for local handoff but is not yet publicly retrievable

## Source Provenance

- Registry provenance: Vaswani et al. 2017 (Transformer); maps to MLPerf Training GPT-3/LLaMA
- Runner min: mlperf.runners.nanogpt:run_min
- Runner max: mlperf.runners.nanogpt:run_max
