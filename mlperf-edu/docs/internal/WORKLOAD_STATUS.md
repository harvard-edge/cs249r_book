<!-- GENERATED FILE - do not edit by hand.
     Regenerate with: python3 tools/workload_status.py --write -->

# Workload Status

Every number here is read from the registry and the retained evidence.
Quality decisions are recomputed against the live registry contract, so a
record graded under a superseded gate cannot report its own result.

## Summary

| Dimension | Result |
|:---|:---|
| Workloads registered | 14 |
| Quality contract passed | 8 |
| Target missed, recorded | 6 |
| Blocked on a local backend | 0 |
| Configuration defects | 0 |
| Cases with repeated timing | 1 |

Quality is decided on the registry's `acceptance_runs: 1`, so one complete
run accepts or rejects a result. Timing repeatability uses
`outer_reference_runs: 5` and belongs to the later promotion phase; it never
gates a quality decision.

## Quality

| Workload | Config | Quality | Observed | Target | Timing | Needs |
|:---|:---|:---|---:|---:|:---|:---|
| `anomaly-detection` | ok | **PASS** | 0.9029 | ≥ 0.8500 | 5 run(s) | none |
| `causal-language-modeling` | ok | **PASS** | 1.4590 | ≤ 1.4697 | 2 run(s) | none |
| `code-generation` | ok | **MISS*** | 0.5549 | ≥ 0.5730 | n/a | import result into evidence index |
| `function-calling` | ok | **MISS*** | 0.7852 | ≥ 0.8292 | n/a | import result into evidence index |
| `graph-node-classification` | ok | **PASS** | 0.7210 | ≥ 0.7174 | 1 run(s) | none |
| `image-classification` | ok | **PASS** | 0.8700 | ≥ 0.8500 | 5 run(s) | none |
| `image-generation` | ok | **MISS*** | 1.8016 | ≤ 1.7900 | n/a | import result into evidence index |
| `information-retrieval` | ok | **PASS** | 0.6072 | ≥ 0.6072 | 5 run(s) | none |
| `keyword-spotting` | ok | **PASS** | 0.9020 | ≥ 0.9000 | 5 run(s) | none |
| `recommendation` | ok | **MISS*** | 0.6232 | ≥ 0.6350 | n/a | import result into evidence index |
| `reinforcement-learning` | ok | **MISS*** | 0.0276 | ≥ 0.4000 | n/a | import result into evidence index |
| `text-classification` | ok | **PASS** | 0.9106 | ≥ 0.9106 | 5 run(s) | none |
| `time-series-forecasting` | ok | **MISS** | 0.2924 ⚠ | ≤ 0.2900 | 1 run(s) | target gap, investigated |
| `visual-wake-words` | ok | **PASS** | 0.8510 | ≥ 0.8000 | 5 run(s) | none |

`MISS*` means the authoritative contract ran and missed its target, and the
result is recorded in the registry but not imported into the evidence index.
⚠ marks a retained record still carrying a superseded gate.

## Measured Runtime

Training and inference are separate cases under one workload identity.
These are the runtimes actually recorded, not estimates.

| Workload | Case | Metric | Measured | Runs | Source |
|:---|:---|:---|---:|---:|:---|
| `anomaly-detection` | inference | `inference_seconds` | 0.3 s | 5 | evidence index |
| `causal-language-modeling` | inference / decode | `output_tokens_per_sec` | 767.3 /s | 1 | evidence index |
| `causal-language-modeling` | inference / full | `output_tokens_per_sec` | 695.5 /s | 1 | evidence index |
| `causal-language-modeling` | inference / prefill | `prefill_tokens_per_sec` | 24,568.4 /s | 1 | evidence index |
| `causal-language-modeling` | training | `train_and_eval_seconds` | 35.1 min | 2 | evidence index |
| `code-generation` | inference | `generation_seconds` | 14.9 min | 1 | registry audit record |
| `function-calling` | inference | `generation_seconds` | 3.48 h | 1 | registry audit record |
| `graph-node-classification` | training | `train_and_eval_seconds` | 12.0 min | 1 | evidence index |
| `image-classification` | inference | `inference_and_evaluation_seconds` | 7.8 s | 5 | evidence index |
| `image-generation` | inference | `final_trial_generation_seconds` | 3.10 h | 3 | registry audit record |
| `information-retrieval` | inference | `inference_and_evaluation_seconds` | 18.0 s | 5 | evidence index |
| `keyword-spotting` | inference | `inference_seconds` | 8.0 s | 5 | evidence index |
| `recommendation` | training | `train_and_eval_seconds` | 1.64 h | 1 | registry audit record |
| `reinforcement-learning` | training | `self_play_and_training_seconds` | 7.7 min | 1 | registry audit record |
| `text-classification` | inference | `inference_seconds` | 4.4 s | 5 | evidence index |
| `time-series-forecasting` | training | `train_and_eval_seconds` | 14.2 min | 1 | evidence index |
| `visual-wake-words` | inference | `inference_seconds` | 0.7 s | 5 | evidence index |

Rows marked `/s` are throughput, not elapsed time. The causal inference
phases report tokens per second, so they are excluded from the total below.

One pass through every timed case in the evidence index is about 62.0 min of compute.

## What Is Missing

- 5 audited misses are recorded in the registry but not imported into the evidence index, so they carry digests and runtime without appearing as cases.
