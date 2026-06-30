# MLPerf EDU Review Packet: `nanogpt-inference --variant decode`

## Summary

| Field | Value |
|---|---|
| Internal ID | nanogpt-decode |
| Run selector | nanogpt-inference --variant decode |
| Suite | language |
| Public status | performance-bearing |
| Scenario | server |
| Model | nanogpt-12m |
| Dataset | prompt-suite-local |
| Canonical workload | nanogpt-inference |
| Variant | decode |

## Reviewer Commands

```bash
mlperf fetch --workload nanogpt-inference --variant decode --profile max --dry-run
mlperf run --workload nanogpt-inference --variant decode --profile max
```

## Functional Contract

| Field | Value |
|---|---|
| Functional metric | decode_steps |
| Condition | checkpoint-backed decode completes the configured number of steps and records positive output throughput |
| Reference runs | 5 |
| Reviewer notes | Quality is inherited from the nanogpt-train checkpoint; this workload reports decode latency, KV-cache behavior, and throughput. |

## Assets

| Field | Value |
|---|---|
| Dataset asset | prompt-suite-local |
| Dataset source | mlperf-edu://bundled/prompts |
| Dataset license status | bundled-project-asset |
| Dataset release status | public-ok-bundled |
| Dataset release next step |  |
| Dataset citation | Bundled deterministic prompts maintained by MLPerf EDU. |

## Checkpoint Lineage

| Field | Value |
|---|---|
| Shared checkpoint | nanogpt-train |
| Quality dependency | nanogpt-train |
| Source run selector | nanogpt-train |
| Source quality | cross_entropy_loss lower 2.3 basis=reference_runs |
| Source verified baseline | train_loss=1.958; val_loss=2.045; epochs=25; time_seconds=86.4; baseline_note=Single-seed Apple MPS default run on the Project Gutenberg generated tiny excerpt; target remains provisional until the five-seed reference sweep. |
| Policy | Preserve the source training report and .provd.json alongside checkpoint-backed inference results. |

## Public Review Notes

- No public-release warning from the current structured audit.

## Source Provenance

- Registry provenance: GPT-2 autoregressive decode; the regime that dominates LLM serving cost in production (vLLM, TensorRT-LLM, TGI all built around this)
- Runner min: mlperf.runners.nanogpt:run_decode_min
- Runner max: mlperf.runners.nanogpt:run_decode_max
