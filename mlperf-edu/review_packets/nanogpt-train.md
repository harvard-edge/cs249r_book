# MLPerf EDU Review Packet: `nanogpt-train`

## Summary

| Field | Value |
|---|---|
| Internal ID | nanogpt-train |
| Run selector | nanogpt-train |
| Suite | language |
| Public status | score-bearing |
| Scenario | single_stream |
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
| Verified baseline | train_loss=1.958; val_loss=2.045; epochs=25; time_seconds=86.4; baseline_note=Single-seed Apple MPS default run on the Project Gutenberg generated tiny excerpt; target remains provisional until the five-seed reference sweep. |

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

- No public-release warning from the current structured audit.

## Source Provenance

- Registry provenance: Vaswani et al. 2017 (Transformer); maps to MLPerf Training GPT-3/LLaMA
- Runner min: mlperf.runners.nanogpt:run_min
- Runner max: mlperf.runners.nanogpt:run_max
