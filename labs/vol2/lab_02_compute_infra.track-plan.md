# V2-02 Track Plan: The Compute Infrastructure Wall

## Purpose

This lab teaches hardware feasibility and infrastructure trade-offs across memory, bandwidth, compute, power, cost, and utilization.

## Shared Pedagogy

- Students predict which hardware resource binds.
- They evaluate workload fit on candidate hardware.
- They choose a hardware tier or placement plan with a risk trigger.

## Lab Flow

### Opening - Compute Infrastructure Brief

Common narrative:
- Compute infrastructure is a stack of constraints, not a peak-FLOPs number.
- The selected track changes what "enough compute" means.

Track realization:
- iPhone: infrastructure is the user's device and NPU under sustained thermal limits.
- Oura Ring: infrastructure is a tiny wearable MCU/storage/battery envelope.
- RoboTaxi: infrastructure is vehicle-local edge compute with safety margin.
- Cloud Fleet: infrastructure is accelerator nodes, racks, power, cooling, and TCO.

### Part A - Node Feasibility

Common pattern:
- Run workload against hardware profile and report memory, bandwidth, compute, and power headroom.

Track realization:
- iPhone checks NPU support, memory, battery, and sustained power.
- Oura Ring checks SRAM/flash, MCU compute, and battery.
- RoboTaxi checks local accelerator memory, throughput, p99, and power.
- Cloud Fleet checks HBM, FLOPs, bandwidth, power, and cost.

### Part B - Infrastructure Frontier

Common pattern:
- Sweep hardware tier, node count, utilization, or power budget.
- Plot TCO, latency, throughput, and energy.

Track realization:
- iPhone frontier compares on-device versus offload and device-tier fallback.
- Oura Ring frontier compares tiny local inference versus phone/cloud assist.
- RoboTaxi frontier compares edge accelerator headroom versus power and reliability.
- Cloud Fleet frontier compares accelerator tiers and utilization targets.

### Part C - Procurement Or Placement

Common pattern:
- Student chooses hardware/placement and invalidation assumption.

Track realization:
- iPhone decision names minimum supported device tier.
- Oura Ring decision names MCU/storage/battery envelope.
- RoboTaxi decision names vehicle-local compute and safety margin.
- Cloud Fleet decision names accelerator tier, node count, and TCO risk.

## Implementation Requirements

- Hardware facts come from MLSysIM registry.
- Track profile maps canonical track to primary hardware, plus optional comparison hardware.
- Fleet/rack assumptions should be explicit for Cloud Fleet.

## Ledger And Report

Save:
- predicted binding resource
- selected hardware/placement
- feasible headroom
- TCO or energy implication
- invalidation trigger

Report target:
- A compute infrastructure recommendation for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- node feasibility, hardware frontier, and infrastructure placement/procurement.

Minimum classroom demo:
- run the same workload on track hardware and show active wall differences.

Completion path:
- predict binding resource, inspect node feasibility, sweep infrastructure frontier, choose hardware/placement plan.

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
| iPhone | Defines minimum supported device/runtime tier and offload boundary. |
| Oura Ring | Defines MCU/storage/battery envelope and what must move off-device. |
| RoboTaxi | Defines vehicle-local compute headroom and safety margin. |
| Cloud Fleet | Defines accelerator tier, node count, utilization, power, and TCO plan. |

## Common Misconceptions

- Peak compute defines infrastructure.
- One device profile is enough for fleet planning.
- Power/cooling is after the hardware choice.
- A GPU is the same as a fleet.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `hardware_ref`
- `workload_ref`
- `utilization_target`
- `power_budget`
- `cost_model`

Needed outputs:
- `feasibility_report`
- `infrastructure_frontier`
- `hardware_plan`
- `risk_trigger`

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

- Binding resource is supported by evidence.
- Plan distinguishes device/node/fleet.
- Power/cost is considered.
- Invalidation trigger is explicit.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
