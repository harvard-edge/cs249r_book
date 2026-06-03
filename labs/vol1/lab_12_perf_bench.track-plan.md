# V1-12 Track Plan: Benchmarking Trap

## Purpose

This lab teaches that benchmarks are claims about workloads. Peak, warm, component, and deployment-like numbers can disagree, and single-metric optimization hides failures.

## Shared Pedagogy

- Students predict whether a benchmark claim will hold in production.
- They compare metrics: throughput, p99 latency, energy, accuracy, cost, and tails.
- They design a benchmark protocol that catches the selected track's real failure.

## Lab Flow

### Opening - Benchmark Claim Brief

Common narrative:
- A team reports a strong benchmark number, but deployment behavior is worse.
- The student must determine what the benchmark failed to measure.

Track realization:
- iPhone: benchmark hides sustained thermal throttling and battery drain.
- Oura Ring: benchmark hides battery life, sampling cadence, and memory fit.
- RoboTaxi: benchmark hides p99 latency and rare safety cases.
- Cloud Fleet: benchmark hides load, batching, cost, and tail latency.

### Part A - Benchmark Illusion

Common pattern:
- Compare easy benchmark to deployment-like workload.
- Show which metric changes and why.

Track realization:
- iPhone compares cold short-run latency to sustained on-device run.
- Oura Ring compares isolated inference to always-on operation.
- RoboTaxi compares average frame latency to p99 under sensor bursts.
- Cloud Fleet compares peak throughput to SLO/cost under real demand.

### Part B - Multi-Metric Trade-Off

Common pattern:
- Optimize one metric and watch others move.
- Students see why "best" depends on guardrails.

Track realization:
- iPhone optimizes latency but may worsen battery/heat.
- Oura Ring optimizes accuracy but may fail energy/storage.
- RoboTaxi optimizes average latency but may fail p99/recall.
- Cloud Fleet optimizes throughput but may worsen tail latency or cost.

### Part C - Benchmark Protocol

Common pattern:
- Student selects workload, duration, warmup, metrics, and guardrails.

Track realization:
- iPhone protocol includes sustained run, battery, and thermal guardrails.
- Oura Ring protocol includes duty cycle, OTA size, and days-of-life estimate.
- RoboTaxi protocol includes p99/p999 and rare-event replay.
- Cloud Fleet protocol includes burst load, cost/request, and SLO compliance.

## Implementation Requirements

- Track variants need benchmark duration, workload shape, and success gates.
- Benchmark protocol should be structured enough for report export.
- Avoid presenting any single metric as sufficient.

## Ledger And Report

Save:
- benchmark claim tested
- hidden failure metric
- selected benchmark protocol
- guardrail thresholds
- production claim the benchmark supports

Report target:
- A benchmark protocol memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- benchmark validity, multi-metric guardrails, tails, and methodology.

Minimum classroom demo:
- compare easy benchmark and deployment-like benchmark for iPhone sustained run or RoboTaxi p99 replay.

Completion path:
- predict benchmark claim validity, compare deployment metrics, design benchmark protocol.

## Instructor Assignment Modes

Default mode:
- Individual choice. Students use the canonical track selected in Lab 00 and submit one report for that track.

Alternative modes:
- Assigned track teams. Instructor assigns tracks to teams and compares how the same pedagogy changes across systems.
- Lecture demo. Instructor demonstrates two contrasting tracks, then students complete their own track asynchronously.
- Capstone mode. Students must keep the same track across the volume so ledger decisions accumulate coherently.

Track lock:
- Implementation should eventually allow instructor-locked tracks through URL/query/config, while defaulting to the ledger-selected track.

## Expected Track Outcomes

| Track | Expected outcome |
|---|---|
| iPhone | Requires sustained thermal/battery benchmark, not short peak latency. |
| Oura Ring | Requires duty-cycle, battery-life, memory, and OTA-aware benchmark. |
| RoboTaxi | Requires p99/p999 and rare-event replay benchmark. |
| Cloud Fleet | Requires load, SLA, cost/request, and quality benchmark. |

## Common Misconceptions

- One benchmark number is enough.
- Average latency implies p99 safety.
- Peak throughput is production throughput.
- Benchmark workload and deployment workload can differ without consequence.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `benchmark_workload`
- `duration`
- `warmup`
- `metrics`
- `guardrails`

Needed outputs:
- `benchmark_comparison`
- `hidden_failure_metric`
- `protocol`
- `production_claim`

Preferred result objects:
- A typed result object for the main computation.
- `ConstraintBudget` or equivalent bottleneck report.
- A report snapshot object that can be serialized into the Design Ledger.

## Single Source Of Truth Requirements

- Hardware facts must come from MLSysIM hardware registries.
- Model facts must come from MLSysIM model registries.
- Reused equations and solvers must live in MLSysIM physics/solver APIs.
- Track identity must come from the `mlsysbook_labs` track profile registry.
- Scenario thresholds, stakeholder text, and guardrails must live in typed lab variant metadata, not scattered notebook constants.
- Any new needed device, model, workload, infrastructure, or solver fact should be added to MLSysIM first and referenced by the lab.

## Accessibility And Fallback Requirements

- Every plot that drives a decision must have a table fallback with exact values.
- Color cannot be the only indicator of feasibility, failure, or dominance.
- Failure boundaries must state value, limit, unit, and mitigation in text.
- Controls required for completion must be keyboard usable and visible without opening advanced drawers.
- The exported report must contain the decision evidence even if the visual is not inspected.

## Rubric Sketch

- Protocol matches deployment claim.
- At least one guardrail metric is included.
- Tail or sustained behavior is addressed.
- Claim scope is not overstated.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
