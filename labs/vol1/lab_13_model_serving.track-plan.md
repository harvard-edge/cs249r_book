# V1-13 Track Plan: Tail Latency Trap

## Purpose

This lab teaches serving systems: queueing, utilization, batching, autoscaling, caching, cold starts, p99 latency, and capacity planning.

## Shared Pedagogy

- Students predict when average latency stops being meaningful.
- They sweep arrival rate, utilization, batching, and replicas.
- They choose a serving configuration and protected failure mode.

## Lab Flow

### Opening - Serving Brief

Common narrative:
- The model works, but the serving path fails under realistic demand.
- The student must design for tails, not averages.

Track realization:
- iPhone: local serving must stay responsive and energy-aware.
- Oura Ring: local inference is duty-cycled; latency is tied to sensing cadence and battery.
- RoboTaxi: serving is a safety-critical local loop with hard p99/p999 requirements.
- Cloud Fleet: API serving must meet SLO under bursty demand and cost constraints.

### Part A - Queueing Failure

Common pattern:
- Arrival/service sliders produce p50/p95/p99 curves.
- Students see utilization-driven tail explosion.

Track realization:
- iPhone queueing may come from app concurrency and background work.
- Oura Ring queueing may come from duty cycle and sensor windows.
- RoboTaxi queueing may come from sensor bursts and perception pipeline stages.
- Cloud Fleet queueing may come from multi-tenant traffic and batch scheduling.

### Part B - Serving Knobs

Common pattern:
- Sweep batching, autoscaling, cache, replicas, cold-start policy, or fallback.
- Plot latency/cost/throughput frontier.

Track realization:
- iPhone uses local batching/fallback sparingly due to user latency.
- Oura Ring uses duty cycle, summarization, and deferred upload.
- RoboTaxi uses bounded queues, priority paths, and local fallback.
- Cloud Fleet uses dynamic batching, autoscaling, cache, and routing.

### Part C - Capacity Plan

Common pattern:
- Student chooses serving configuration and states the failure it protects.

Track realization:
- iPhone plan protects user-perceived latency and battery.
- Oura Ring plan protects battery life and sensing reliability.
- RoboTaxi plan protects p99 safety latency.
- Cloud Fleet plan protects SLA and cost per request.

## Implementation Requirements

- Track variants define demand model, SLO, and guardrail metric.
- Queueing model should expose utilization and tail estimates.
- Report export must include p99, not only average latency.

## Ledger And Report

Save:
- predicted tail behavior
- serving policy selected
- p99/SLO result
- cost/energy implication
- protected failure mode

Report target:
- A serving capacity plan for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- queueing, p99, batching, autoscaling, caching, and capacity planning.

Minimum classroom demo:
- increase utilization until p99 explodes for RoboTaxi or Cloud Fleet.

Completion path:
- predict tail behavior, sweep serving knobs, choose capacity/serving policy.

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
| iPhone | Protects local responsiveness and battery with limited batching/offload. |
| Oura Ring | Uses duty cycle, deferred sync, or summaries to preserve battery and sensing. |
| RoboTaxi | Uses bounded queues and priority paths to protect safety p99/p999. |
| Cloud Fleet | Uses batching/autoscaling/cache/routing to balance SLA and cost. |

## Common Misconceptions

- Average latency is enough.
- High utilization is always good.
- Batching always improves user experience.
- Local serving has no queueing problem.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `arrival_rate`
- `service_time`
- `batch_policy`
- `replicas`
- `cache_policy`
- `slo`

Needed outputs:
- `p50_p95_p99`
- `utilization`
- `serving_frontier`
- `capacity_plan`

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

- Uses p99/SLO evidence.
- Serving policy matches track demand.
- Cost/energy trade-off is stated.
- Protected failure mode is explicit.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
