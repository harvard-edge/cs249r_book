# V1-12 Concept Module Packet: Performance Benchmarking

## Chapter Invariant

Measurement changes decisions. A benchmark is only valid when the workload,
warmup, variance, tail behavior, and comparison rules match the deployment
question being answered.

The lab uses one shared Part A/B/C/D concept sequence for every track. Tracks do
not introduce different concepts; they change the persona, operating envelope,
thresholds, evidence emphasis, failure mode, and report framing.

## Reading Map

| Lab module | Chapter anchor | Student-facing claim |
|---|---|---|
| Opening | Purpose and ML Benchmarking Framework | Benchmarking validates whether Data, Algorithm, and Machine hold together under representative conditions. |
| Part A | Benchmarks as proxies; Benchmarking Granularity; MLPerf execution scenarios | Validity depends on matching the production workload and the metric that drives the decision. |
| Part B | Micro-benchmarking rules; Statistical and methodological issues; Result interpretation | Warmup, variance, and sample size determine whether a benchmark result is believable. |
| Part C | Inference metrics; Latency and tail latency; Fallacies and Pitfalls | Means can pass while tail percentiles or guardrails fail. |
| Part D | Benchmark Components; Run rules; System specifications; MLPerf synthesis and benchmark gaming | Fair comparison requires controlled conditions, equivalent run rules, and reportable evidence. |
| Synthesis | Production Considerations; Summary | A useful benchmark report names the claim scope, confidence, tail evidence, guardrails, and rejected comparison. |

## Accepted Concepts

1. Benchmark-production gap: a benchmark result is a proxy with boundaries, not
   a universal truth.
2. Workload and metric alignment: SingleStream, duty-cycle, MultiStream,
   Server, and Offline-style workloads answer different deployment questions.
3. Warmup and variance: first-run artifacts, DVFS/cache effects, and run-to-run
   noise determine confidence.
4. Tail and guardrail reasoning: p95/p99/p999, energy, memory, and cost can
   reject a configuration whose mean or headline metric looks good.
5. Fair comparison: results require controlled hardware/software conditions,
   equivalent workload scope, repeated measurements, and explicit run rules.

## Rejected Concepts

| Rejected concept | Reason for rejection |
|---|---|
| Full MLPerf submission workflow | Too operationally broad for a short lab; retained as a reading connection for Part D. |
| Detailed roofline diagnosis | Covered in V1-11; here it appears only as a source of unrealistic peak claims. |
| Training time-to-accuracy benchmarking | Important chapter topic, but this lab focuses on deployment-facing inference benchmarks. |
| Benchmark history survey | Useful context but weak as an interactive consequence. |
| Model/data benchmark contamination | Important for model evaluation, but would dilute the measurement-methodology sequence. |

## Track Narratives

| Track | Persona | Production question | Binding amounts | Natural failure | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile release manager | Can the feature run locally through a sustained device session? | Sustained run length, ambient temperature, p95 UX latency, battery percent/hour, thermal headroom | Cold-run latency passes while sustained p95 and battery/thermal guardrails fail | "Do not ship until the benchmark includes a 10-minute sustained run, thermal soak, battery drain, and p95 UX evidence." |
| Oura Ring | Wearable validation lead | Can always-on sensing run through the night inside memory and energy windows? | Wake windows/day, duty cycle, SRAM/flash bytes, uJ/window, battery days | Isolated inference passes while SRAM/flash or energy/window fails | "Approve only a duty-cycle replay with memory fit, OTA payload, and energy/window evidence." |
| RoboTaxi | Safety validation lead | Can perception meet deadline and recall under synchronized sensor bursts and rare events? | p99/p999 perception latency, frame deadline, burst multiplier, rare-event replay count, recall floor | Average FPS passes while p99/p999 misses the deadline or rare-event recall falls below floor | "Reject average-frame benchmarks; require MultiStream replay, rare-event evidence, and p999 deadline margin." |
| Cloud Fleet | Cloud SRE lead | Can serving satisfy load, SLA, utilization, and cost/request under production demand? | QPS, p99 latency, utilization, replicas, dollars/request, quality floor | Peak Offline throughput passes while Server load breaches p99 or cost/request | "Accept only load/SLA benchmarks with p99, utilization, quality, and cost/request evidence." |

## Shared Concept Modules

### Part A - Concept Module: Valid Benchmarks Match Workload And Metric

Chapter claim:
- Benchmarks are proxies. A measurement is meaningful only for the workload,
  granularity, metric, and deployment scenario it actually measured.

Student prior:
- "The best headline benchmark number should drive the decision."

Productive failure:
- Students initially accept an easy benchmark. The same system fails when the
  workload shape and metric are switched to the production track.

Activity beats:
1. Scenario: the selected track's stakeholder must decide whether a benchmark
   claim is valid for deployment.
2. Prediction: choose whether the easy benchmark or production-like benchmark
   should decide launch.
3. Manipulation: adjust workload match and selected metric emphasis.
4. Evidence: compare easy vs. production-like metrics in a table and bar chart.
5. Consequence: name the hidden failure metric and rejected benchmark scope.
6. Math/source: validity score is a weighted overlap across workload,
   metric, duration, and guardrail coverage; reading link to benchmarks as
   proxies and MLPerf scenarios.
7. Checkpoint: decide which benchmark can support the production claim.

Track-specific amount reasoning:
- iPhone: sustained p95 UX latency and battery/thermal evidence outrank cold
  latency.
- Oura Ring: energy/window and SRAM/flash fit outrank isolated latency.
- RoboTaxi: p99/p999 deadline margin and rare-event replay outrank average FPS.
- Cloud Fleet: load p99, utilization, and cost/request outrank peak throughput.

Mechanics:
- Prediction radio, workload-match slider, metric-emphasis dropdown, easy vs.
  production evidence bars, failure callout, checkpoint radio.

Ledger output:
- `validity_decision`, `validity_score`, `selected_metric`,
  `rejected_benchmark_scope`, `hidden_failure_metric`.

### Part B - Concept Module: Warmup, Variance, And Sample Size Set Confidence

Chapter claim:
- First-run artifacts, thermal or cache state, and run-to-run variation mean a
  benchmark needs warmup, repeated samples, and confidence reporting.

Student prior:
- "One clean run is enough if the number looks precise."

Productive failure:
- A short sample appears to pass. Adding warmup discard and more samples reveals
  variance, confidence width, or sustained-state failure.

Activity beats:
1. Scenario: the stakeholder must decide whether the measured improvement is
   strong enough for a release note.
2. Prediction: choose whether a single run, 5 runs, or 20 runs is enough.
3. Manipulation: adjust warmup iterations, measured sample count, and
   environment jitter.
4. Evidence: confidence interval, coefficient of variation, and warmup artifact
   table.
5. Consequence: mark the claim as confident, noisy, or underpowered.
6. Math/source: CV = sigma / mu; CI half-width = 1.96 sigma / sqrt(n); reading
   link to micro-benchmark rules and statistical confidence.
7. Checkpoint: choose a minimum run rule for the final report.

Track-specific amount reasoning:
- iPhone: sustained device run must discard cold/cache warmup and account for
  thermal drift.
- Oura Ring: many duty-cycle windows are needed because each window is small and
  energy budgets are tight.
- RoboTaxi: rare-event replay needs enough samples to make p999/deadline claims
  credible.
- Cloud Fleet: load tests need enough request samples to estimate p99 and
  utilization with useful confidence.

Mechanics:
- Prediction radio, warmup slider, sample-count slider, jitter slider,
  warmup-vs-measured line chart, confidence table, failure callout.

Ledger output:
- `warmup_discard`, `sample_count`, `cv_pct`, `ci_half_width`,
  `confidence_verdict`.

### Part C - Concept Module: Averages Hide Tail And Guardrail Failures

Chapter claim:
- Production SLOs and constraints are often percentile or guardrail based. Mean
  latency or a single average metric can pass while the actual system fails.

Student prior:
- "If the average is under budget, the benchmark passed."

Productive failure:
- Students see a passing mean, then increase tail heaviness or guardrail stress
  and watch p95/p99/p999, energy, memory, or cost fail.

Activity beats:
1. Scenario: a launch candidate has a healthy mean but reported incidents.
2. Prediction: decide whether the mean is sufficient evidence.
3. Manipulation: adjust tail heaviness and guardrail stress.
4. Evidence: histogram/CDF plus exact mean, p95, p99, p999, and guardrail table.
5. Consequence: quantify failures per second/window/frame/day.
6. Math/source: log-normal percentile model; p_q = base * exp(z_q sigma);
   reading link to tail latency and single-metric fallacies.
7. Checkpoint: choose the guardrail that must block release.

Track-specific amount reasoning:
- iPhone: p95 UX latency and battery/thermal guardrails define ship/no-ship.
- Oura Ring: energy/window and SRAM/flash fit define whether the feature lasts
  overnight and can update.
- RoboTaxi: p99/p999 deadline and rare-event recall define safety margin.
- Cloud Fleet: p99 SLA, utilization, and cost/request define operating viability.

Mechanics:
- Prediction radio, tail-sigma slider, guardrail-stress slider,
  histogram/CDF, metric badges, danger state, checkpoint radio.

Ledger output:
- `mean_metric`, `tail_metric`, `guardrail_metric`, `guardrail_verdict`,
  `failure_rate`.

### Part D - Concept Module: Fair Comparison Requires Controlled Conditions

Chapter claim:
- Benchmark comparisons are fair only when the workload, run rules, hardware,
  software, warmup, sample count, and guardrails are controlled and reported.

Student prior:
- "The faster reported number wins."

Productive failure:
- Candidate B looks faster, but a fairness audit exposes mismatched run rules or
  missing guardrail evidence; the apparent winner is rejected.

Activity beats:
1. Scenario: two benchmark submissions compete for approval.
2. Prediction: pick the winner from headline results.
3. Manipulation: toggle control of workload, warmup, sample count, hardware
   state, and guardrail reporting.
4. Evidence: fairness ledger and accepted/rejected comparison table.
5. Consequence: name the rejected comparison and the missing evidence.
6. Math/source: fair comparison index = controlled conditions / required
   conditions, with penalty for missing guardrail; reading link to benchmark
   components, run rules, and MLPerf reference-vs-submission validation.
7. Checkpoint: choose which comparison can appear in the final benchmark report.

Track-specific amount reasoning:
- iPhone: compare only runs with same device class, ambient, run length, battery
  state, and p95 UX evidence.
- Oura Ring: compare only duty-cycle replays with identical sensing windows,
  SRAM/flash accounting, OTA assumptions, and energy boundary.
- RoboTaxi: compare only sensor-burst replays with identical frame mix,
  deadline, rare-event count, recall floor, and p99/p999 reporting.
- Cloud Fleet: compare only load tests with same demand trace, replica budget,
  SLA, utilization target, quality gate, and cost model.

Mechanics:
- Prediction radio, fairness checkboxes, comparison table, evidence matrix,
  rejected-comparison callout, checkpoint radio.

Ledger output:
- `fair_comparison_index`, `accepted_comparison`, `rejected_comparison`,
  `missing_evidence`, `reportable_evidence`.

## Synthesis - Benchmark Report

Student task:
- Produce a track-specific benchmark report that includes:
  1. selected metric and why it matches the production question;
  2. confidence evidence from warmup, variance, and sample size;
  3. tail or guardrail evidence with a concrete failure boundary;
  4. one rejected comparison and why it was unfair;
  5. final benchmark claim scope.

Report fields:
- `track_id`
- `production_question`
- `selected_metric`
- `validity_score`
- `confidence_verdict`
- `tail_guardrail_verdict`
- `fair_comparison_index`
- `rejected_comparison`
- `final_claim_scope`

## Mechanics, Evidence, And Ledger Plan

| Module | Primary mechanic | Evidence artifact | Failure boundary | Ledger use |
|---|---|---|---|---|
| A | Workload-match and metric-emphasis controls | Easy vs. production metric chart and exact table | Validity score below report threshold or hidden metric fails | Carries selected metric and rejected benchmark into report |
| B | Warmup/sample/jitter controls | CI/CV table and warmup trend chart | CV above threshold or CI too wide for claimed improvement | Carries confidence rule and sample count into report |
| C | Tail and guardrail controls | Histogram/CDF and guardrail table | p95/p99/p999, energy, memory, or cost exceeds track limit | Carries tail/guardrail failure into report |
| D | Fairness checklist controls | Fair comparison ledger | Missing run rule or guardrail evidence rejects comparison | Carries accepted/rejected comparison into report |
| Synthesis | Report card and export panel | Benchmark protocol memo | Incomplete predictions or missing evidence | Saves design ledger snapshot for downstream labs |

## Source And Amount Policy

- Hardware and model refs remain sourced through `resolve_mlsysim_ref` and
  `benchmark_track_profile`.
- Track identity remains sourced through the track selector and track profile
  registry.
- Shared calculations continue using existing `mlsysbook_labs` helpers where
  available: Amdahl, sustained benchmark, metric gate, and tail latency.
- Notebook-local helpers are allowed only for lab-specific protocol scoring and
  must be prefixed `v1_12_`.
- Scenario thresholds not already in shared metadata are expressed as
  track-specific pedagogical assumptions in notebook-local helper tables and
  surfaced in Math Peek/source notes.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Shared variants do not expose every needed threshold | Use notebook-local, prefixed helpers for protocol scoring and source notes; do not edit shared variant metadata in this wave. |
| Existing helper names imply old Amdahl/Thermal part structure | Keep imports but reposition them as supporting evidence inside the new concept modules. |
| Track-specific widgets could accidentally become different concepts | Use the same widget sequence and data schema for all tracks; change labels, limits, and report interpretation only. |
| Report may overstate simulated numbers | Label local calculations as teaching estimates and report the source/assumption boundary. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 2 | Pass |

Depth gates:
- Every concept module has at least five student-facing activity beats.
- Every concept module has a structured prediction.
- Every concept module has a manipulation control.
- Every concept module has a consequence or reversible failure boundary.
- Every concept module includes Math Peek/source-model reasoning.
- Synthesis ties the selected track back to the chapter invariant.
