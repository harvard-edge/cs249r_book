# MLPerf EDU Course-Image Budgets

## Measurement Scope

This initial budget covers one functional `min` run for every workload on the
July 18, 2026 Apple Silicon course image. It measures setup and artifact
generation, not authoritative `max` quality execution or stability. The exact
machine record and all 28 observations are in
[`course-budgets-apple-m5-max-20260718.json`](../../conformance_results/course-budgets-apple-m5-max-20260718.json).

The source checkout was clean at `d94c7649851d9caa8c378dc8329d8af26812336c`.
The machine was a 64 GiB Apple M5 Max MacBook Pro running macOS 26.4, Python
3.12.13, and PyTorch 2.13.0. Each workload ran in a fresh process. The collector
recorded process-tree peak resident memory, wall time including CLI startup,
generated artifact bytes, and the device that actually executed.

## Planning Ceilings

The measured maxima support simple functional-lab planning ceilings with room
for normal process noise.

| **Path** | **Rounded Per-Workload Ceiling** | **Measured Suite Total** | **Interpretation** |
|:---|:---|:---|:---|
| CPU `min` | 5 seconds, 512 MiB peak RSS, 100 KiB artifacts | 32.04 seconds and 0.92 MiB artifacts across 14 fresh processes | All 14 executed on CPU and passed. |
| MPS-requested `min` | 6 seconds, 1 GiB peak RSS, 100 KiB artifacts | 32.03 seconds and 0.92 MiB artifacts across 14 fresh processes | Five executed on MPS. Nine intentionally used CPU implementations. |
| Network and downloads | 0 bytes | 0 bytes | Functional paths use bundled deterministic inputs. `max` assets are separate. |

These are course-planning ceilings, not benchmark performance thresholds. A
student exceeding one should inspect the environment, but the result does not
fail a quality contract because of this table.

## Per-Workload Observations

| **Workload** | **Requested** | **Executed** | **Wall Time (s)** | **Peak RSS (MiB)** | **Artifacts (KiB)** |
|:---|:---|:---|---:|---:|---:|
| `anomaly-detection` | CPU | CPU | 1.59 | 224.4 | 64.3 |
| `causal-language-modeling` | CPU | CPU | 2.07 | 299.0 | 64.8 |
| `code-generation` | CPU | CPU | 4.11 | 459.9 | 72.9 |
| `function-calling` | CPU | CPU | 3.46 | 458.0 | 75.6 |
| `graph-node-classification` | CPU | CPU | 2.80 | 424.1 | 64.1 |
| `image-classification` | CPU | CPU | 1.42 | 225.9 | 64.4 |
| `image-generation` | CPU | CPU | 1.48 | 224.8 | 65.6 |
| `information-retrieval` | CPU | CPU | 3.44 | 462.9 | 70.5 |
| `keyword-spotting` | CPU | CPU | 1.48 | 223.1 | 65.1 |
| `recommendation` | CPU | CPU | 1.45 | 223.1 | 64.9 |
| `reinforcement-learning` | CPU | CPU | 1.85 | 298.3 | 65.7 |
| `text-classification` | CPU | CPU | 3.38 | 459.3 | 69.0 |
| `time-series-forecasting` | CPU | CPU | 2.03 | 349.4 | 65.6 |
| `visual-wake-words` | CPU | CPU | 1.49 | 225.5 | 64.1 |
| `anomaly-detection` | MPS | CPU | 1.42 | 224.3 | 64.4 |
| `causal-language-modeling` | MPS | CPU | 1.78 | 298.1 | 64.8 |
| `code-generation` | MPS | MPS | 3.91 | 603.5 | 72.9 |
| `function-calling` | MPS | MPS | 4.29 | 727.6 | 75.6 |
| `graph-node-classification` | MPS | CPU | 2.89 | 423.6 | 64.1 |
| `image-classification` | MPS | CPU | 1.35 | 225.9 | 64.4 |
| `image-generation` | MPS | MPS | 1.53 | 320.5 | 65.6 |
| `information-retrieval` | MPS | CPU | 3.39 | 466.7 | 70.5 |
| `keyword-spotting` | MPS | CPU | 1.35 | 221.4 | 65.1 |
| `recommendation` | MPS | MPS | 1.46 | 319.6 | 64.9 |
| `reinforcement-learning` | MPS | MPS | 1.96 | 408.1 | 65.7 |
| `text-classification` | MPS | CPU | 3.28 | 459.3 | 69.0 |
| `time-series-forecasting` | MPS | CPU | 2.06 | 348.8 | 65.6 |
| `visual-wake-words` | MPS | CPU | 1.36 | 227.1 | 64.1 |

## Remaining Budget Work

Authoritative `max` budgets still need measurement on the actual course images
chosen by an instructor. That pass should cover the selected assignment
workloads. Recommendation and MiniGo need measured budgets of their own:
both now execute locally, but local execution does not imply either fits a
short class period. Recommendation measures at roughly half an hour under its
seven-epoch budget, and MiniGo's cost scales with the generation budget. The budget should separate first-fetch bytes from steady-state
disk use and record accelerator memory where the runtime exposes a trustworthy
value. Five-process timing stability remains a later promotion activity.
