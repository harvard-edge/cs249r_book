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
| Condition | default max run requests 16 decode tokens; generated_tokens must be >= 8 on the deterministic prompt set |
| Reference runs | 5 |
| Reviewer notes | Functional generation check is intentionally small; public comparisons use latency, throughput, memory footprint, and energy. |

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

- No public-release warning from the current structured audit.

## Source Provenance

- Registry provenance: Off-the-shelf small language model decode path for local serving, quantization, LoRA, KV-cache, and backend comparison studies.
- Runner min: mlperf.runners.slm:run_decode_min
- Runner max: mlperf.runners.slm:run_decode_max
