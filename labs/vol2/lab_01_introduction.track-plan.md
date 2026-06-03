# V2-01 Track Plan: The Scale Illusion

## Purpose

This lab introduces scale as a systems change, not a bigger version of single-device ML. Students see reliability collapse, coordination tax, and nonlinear cost as systems grow.

## Shared Pedagogy

- Students predict whether scaling a system linearly will preserve efficiency.
- They sweep number of devices, users, accelerators, or services.
- They choose whether to scale, shard, specialize, simplify, or refuse the scale-up.

## Lab Flow

### Opening - Scale Brief

Common narrative:
- The system is successful enough that demand grows.
- The student must decide what breaks when the selected track scales.

Track realization:
- iPhone: scale means many heterogeneous devices, OS versions, and local privacy constraints.
- Oura Ring: scale means many wearables with battery, firmware, and data-quality variation.
- RoboTaxi: scale means fleet growth, geography diversity, failures, and operational safety.
- Cloud Fleet: scale means accelerator count, services, requests, and coordination overhead.

### Part A - Scaling Illusion

Common pattern:
- Sweep fleet/cluster/user count and show reliability/cost/latency collapse.

Track realization:
- iPhone sweeps device population and support matrix.
- Oura Ring sweeps device fleet and firmware/battery variation.
- RoboTaxi sweeps vehicle count and daily operational exposure.
- Cloud Fleet sweeps accelerators, replicas, and request volume.

### Part B - Coordination Tax

Common pattern:
- Sliders add synchronization, monitoring, retries, heterogeneity, and release overhead.
- Plot useful work versus total work.

Track realization:
- iPhone coordination tax is app rollout, privacy-safe telemetry, and device heterogeneity.
- Oura Ring coordination tax is OTA rollout, sensor drift, and intermittent connectivity.
- RoboTaxi coordination tax is safety validation, map/model rollout, and incident review.
- Cloud Fleet coordination tax is distributed execution, monitoring, retries, and orchestration.

### Part C - Scale Readiness

Common pattern:
- Student chooses scale plan and first mitigation.

Track realization:
- iPhone plan may specialize by device tier or degrade gracefully.
- Oura Ring plan may stage firmware/model rollout and simplify local inference.
- RoboTaxi plan may limit rollout by geography and safety evidence.
- Cloud Fleet plan may shard, batch, cache, or redesign infrastructure.

## Implementation Requirements

- Track variants define what "scale" means for the selected system.
- The same scaling visual should work for device fleet, vehicle fleet, and accelerator fleet.
- The report should record the scale unit and mitigation.

## Ledger And Report

Save:
- scale unit
- predicted collapse point
- observed bottleneck
- selected scale plan
- first mitigation

Report target:
- A scale-readiness memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- scale collapse, coordination tax, and scale-readiness decision.

Minimum classroom demo:
- sweep fleet size for RoboTaxi and accelerator count for Cloud Fleet to show different collapse modes.

Completion path:
- predict collapse point, inspect coordination tax, choose scale plan and first mitigation.

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
| iPhone | Treats scale as heterogeneous devices, OS versions, rollout cohorts, and privacy-safe telemetry. |
| Oura Ring | Treats scale as wearable fleet health, firmware, battery variation, and intermittent sync. |
| RoboTaxi | Treats scale as vehicle/geography exposure, safety validation, and operational incidents. |
| Cloud Fleet | Treats scale as accelerators, replicas, requests, coordination, and reliability collapse. |

## Common Misconceptions

- Scaling is linear.
- A bigger fleet only adds capacity.
- Coordination is negligible.
- Reliability per unit is enough to reason about system reliability.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `scale_unit`
- `fleet_size`
- `failure_rate`
- `coordination_overheads`
- `demand_growth`

Needed outputs:
- `scale_curve`
- `coordination_tax`
- `collapse_point`
- `scale_readiness_plan`

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

- Scale unit is track-specific.
- Collapse is explained quantitatively.
- Mitigation matches bottleneck.
- First risk is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
