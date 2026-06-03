# V1-02 Track Plan: Physics of Deployment

## Purpose

This lab teaches that deployment is constrained by physical limits: memory, compute, bandwidth, power, energy, distance, and latency. Track selection changes which physical wall appears first.

## Shared Pedagogy

- Students predict the first wall before running the solver.
- They sweep a workload knob until feasibility breaks.
- They choose a placement or mitigation strategy based on the binding physical limit.

## Lab Flow

### Opening - Physical Constraint Brief

Common narrative:
- A model that seems reasonable in isolation must be deployed into a physical system.
- The student must identify the first non-negotiable wall.

Track realization:
- iPhone: sustained thermal power and battery are as important as raw NPU TOPS.
- Oura Ring: SRAM/flash and energy per inference dominate.
- RoboTaxi: p99 perception-to-decision latency and local reliability dominate.
- Cloud Fleet: power, cost, memory bandwidth, and service latency dominate.

### Part A - First Wall

Common pattern:
- Evaluate the current scenario against track hardware.
- Display headroom for memory, compute, bandwidth, power, latency, and energy.

Track realization:
- iPhone first wall may be thermal or battery drain.
- Oura Ring first wall may be SRAM, flash, or energy.
- RoboTaxi first wall may be p99 latency or sensor-to-compute bandwidth.
- Cloud Fleet first wall may be memory bandwidth, cost, or p99 under load.

### Part B - Physics Curve

Common pattern:
- Sweep the expensive variable and plot threshold crossings.
- Use the same visual grammar across tracks.

Track realization:
- iPhone sweeps frame rate, model size, or sustained inference duration.
- Oura Ring sweeps sampling cadence, model size, or OTA payload.
- RoboTaxi sweeps frame resolution, sensor count, or perception deadline.
- Cloud Fleet sweeps request rate, batch size, or sequence length.

### Part C - Deployment Choice

Common pattern:
- Choose local, edge, cloud, hybrid, or simplification.
- Record the avoided wall and new risk.

Track realization:
- iPhone may stay on-device for privacy but simplify the model.
- Oura Ring may do tiny inference locally and defer heavy analysis.
- RoboTaxi must keep safety-critical inference local and offload noncritical tasks.
- Cloud Fleet may shard, batch, cache, or use regional placement.

## Implementation Requirements

- Hardware facts must come from MLSysIM hardware registry.
- Scenario thresholds live in track variants, not notebook constants.
- The solver output should name the first violated wall and the mitigation candidates.

## Ledger And Report

Save:
- predicted first wall
- actual first wall
- sweep knob and threshold value
- placement/mitigation decision
- residual risk

Report target:
- A physics-of-deployment memo explaining why the selected track changes the feasible system design.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- physical walls and placement feasibility.

Minimum classroom demo:
- sweep one workload knob and show first wall differences for Oura Ring versus RoboTaxi.

Completion path:
- predict first wall, run the physics sweep, select placement/mitigation, save first violated constraint.

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
| iPhone | Likely identifies thermal power, battery, or memory as the deployment wall. |
| Oura Ring | Likely identifies SRAM, flash, sampling energy, or OTA payload as the first wall. |
| RoboTaxi | Likely identifies p99 deadline, sensor bandwidth, or local power/reliability as the first wall. |
| Cloud Fleet | Likely identifies memory bandwidth, p99 under load, power/cost, or utilization as the first wall. |

## Common Misconceptions

- Deployment limits are software preferences.
- Average latency is enough.
- Distance/network can be ignored.
- Power and energy are the same constraint.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `hardware_ref`
- `workload_knob`
- `placement_choice`
- `constraint_thresholds`

Needed outputs:
- `first_wall`
- `threshold_crossing`
- `placement_feasibility`
- `mitigation_candidates`

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

- Prediction names a physical resource.
- Sweep evidence supports wall identification.
- Placement choice avoids one wall but names new risk.
- Values are sourced from MLSysIM/profile metadata.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
