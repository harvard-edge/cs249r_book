# V2-08 Track Plan: The Scheduling Trap

## Purpose

This lab teaches scheduling objectives, queueing, utilization paradox, bin packing, fragmentation, preemption, priority, and heterogeneous fleets.

## Shared Pedagogy

- Students predict when high utilization hurts responsiveness.
- They compare scheduler policies under heterogeneous constraints.
- They choose admission, priority, preemption, or packing policy.

## Lab Flow

### Opening - Scheduling Brief

Common narrative:
- Keeping resources busy is not the same as meeting system goals.
- The selected track changes what is being scheduled.

Track realization:
- iPhone: schedule on-device inference around user interaction and battery state.
- Oura Ring: schedule sensing/inference/sync around battery and duty cycle.
- RoboTaxi: schedule perception/control tasks with hard priorities and deadlines.
- Cloud Fleet: schedule jobs/requests across heterogeneous accelerators and services.

### Part A - Queue/Utilization Wall

Common pattern:
- Arrival and service sliders show queue time, SLA failures, and utilization.

Track realization:
- iPhone shows app responsiveness under background ML.
- Oura Ring shows duty-cycle contention between sensing, inference, and sync.
- RoboTaxi shows deadline misses under bursty sensor workload.
- Cloud Fleet shows queueing under high cluster utilization.

### Part B - Fragmentation And Preemption Frontier

Common pattern:
- Simulate packing, fragmentation, preemption, and heterogeneity.

Track realization:
- iPhone compares priority scheduling for foreground versus background tasks.
- Oura Ring compares duty-cycle windows and deferred sync.
- RoboTaxi compares real-time priority and degraded-mode scheduling.
- Cloud Fleet compares bin packing, preemption, and heterogeneous accelerator placement.

### Part C - Fleet Policy

Common pattern:
- Student records scheduler, admission, preemption, and fairness trade-off.

Track realization:
- iPhone policy protects foreground experience.
- Oura Ring policy protects battery and sensing continuity.
- RoboTaxi policy protects safety-critical deadline.
- Cloud Fleet policy protects utilization, wait time, and tenant fairness.

## Implementation Requirements

- Track variants need schedulable resources, task classes, deadlines, and priorities.
- Scheduler policy catalog should be reusable.
- Avoid implying the same utilization target is appropriate across tracks.

## Ledger And Report

Save:
- predicted scheduling failure
- selected scheduler policy
- utilization/latency result
- preemption/admission rule
- fairness or responsiveness risk

Report target:
- A scheduling policy memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- queueing, utilization, fragmentation, preemption, heterogeneity, and fleet policy.

Minimum classroom demo:
- increase utilization and show responsiveness/SLA collapse for Cloud Fleet and RoboTaxi.

Completion path:
- predict scheduling failure, inspect queue/utilization wall, compare policies, choose fleet policy.

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
| iPhone | Schedules foreground/background ML to protect UX and battery. |
| Oura Ring | Schedules sensing/inference/sync around duty cycle and battery. |
| RoboTaxi | Schedules safety-critical tasks with hard deadlines and fallback. |
| Cloud Fleet | Schedules jobs/requests across heterogeneous accelerators with utilization and fairness trade-offs. |

## Common Misconceptions

- High utilization is always optimal.
- Preemption is free.
- Fairness and responsiveness align automatically.
- All jobs/tasks have equal priority.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `arrival_rate`
- `task_classes`
- `resource_requirements`
- `scheduler_policy`
- `preemption_policy`

Needed outputs:
- `queue_metrics`
- `fragmentation`
- `sla_failures`
- `scheduling_policy`

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

- Policy protects correct priority/guardrail.
- Utilization trade-off is understood.
- Heterogeneity/fragmentation is addressed.
- Fairness or responsiveness risk is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
