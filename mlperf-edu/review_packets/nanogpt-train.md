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
| Baseline record | evidence_status=pending-clean-public-candidate-reference-summary; review_eligible=False; calibration_tier=development; development_summary_id=nanogpt-train_max_20260711T045626.172816Z; development_summary_sha256=ca74f0561f563f5b478038d555ad2e367ff155a9965e0e478d7a0f7b8be9ddd5; development_summary_availability=local-handoff; seeds=0, 1, 2, 3, 4; cross_entropy_loss_by_seed=2.1344807863235475, 2.056773912906647, 2.1026881873607635, 1.9816058337688447, 2.087835317850113; cross_entropy_loss=2.087835317850113; val_loss=2.087835317850113; five_seed_cross_entropy_median=2.087835317850113; five_seed_cross_entropy_min=1.9816058337688447; five_seed_cross_entropy_max=2.1344807863235475; five_seed_cross_entropy_stdev=0.05808833675616628; epochs=25; duration_seconds_median=127.42291408299934; duration_seconds_min=121.23956066707615; duration_seconds_max=130.2600357090123; baseline_note=Fresh real-data Apple MPS development calibration across seeds 0-4. The values are not review eligible until a clean public-candidate sweep commits a complete artifact index and retains its content-addressed raw package for review. |

## Measurement and Evidence Contract

| Field | Value |
|---|---|
| Reference protocol |  |
| Measurement protocol |  |
| Checkpoint contract |  |
| Task-quality evaluation |  |
| Baseline evidence status | pending-clean-public-candidate-reference-summary |
| Baseline review eligible | False |
| Baseline evidence file | not declared |
| Reference package availability | not declared |
| External publication status | not declared |
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

- score-bearing baseline is not backed by a committed reference summary; evidence status is pending-clean-public-candidate-reference-summary

## Source Provenance

- Registry provenance: Vaswani et al. 2017 (Transformer); maps to MLPerf Training GPT-3/LLaMA
- Runner min: mlperf.runners.nanogpt:run_min
- Runner max: mlperf.runners.nanogpt:run_max
