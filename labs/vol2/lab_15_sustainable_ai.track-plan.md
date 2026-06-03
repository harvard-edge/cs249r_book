# V2-15 Track Plan: The Carbon Budget

## Purpose

This lab teaches sustainability: operational energy, carbon geography, embodied carbon, lifecycle analysis, placement, time shifting, utilization, rebound effects, and carbon-aware policy.

## Shared Pedagogy

- Students predict whether efficiency or clean energy alone solves the carbon problem.
- They compare operational and lifecycle carbon.
- They choose carbon-aware scheduling, placement, compression, caps, or demand policy.

## Lab Flow

### Opening - Sustainability Brief

Common narrative:
- Energy and carbon are systems properties, not just hardware properties.
- The selected track changes the dominant sustainability term.

Track realization:
- iPhone: battery energy and device fleet scale matter; privacy can prevent centralization.
- Oura Ring: tiny per-device energy matters because always-on operation and fleet scale accumulate.
- RoboTaxi: vehicle compute power and fleet operation matter, but safety constraints limit savings.
- Cloud Fleet: datacenter energy, grid carbon, utilization, and embodied carbon dominate.

### Part A - Energy/Carbon Measurement

Common pattern:
- Stack operational energy and carbon for workload, hardware, utilization, and region.

Track realization:
- iPhone computes per-session and fleet battery/energy implications.
- Oura Ring computes duty-cycle energy and battery-life carbon proxy.
- RoboTaxi computes vehicle-local compute energy over operating hours.
- Cloud Fleet computes datacenter energy and carbon by region/PUE.

### Part B - Placement/Lifecycle Frontier

Common pattern:
- Compare geography, time shifting, utilization, embodied carbon, latency, and reliability.

Track realization:
- iPhone compares local inference versus cloud offload.
- Oura Ring compares local tiny inference versus phone/cloud handoff.
- RoboTaxi compares local safety compute versus cloud/fleet processing for noncritical tasks.
- Cloud Fleet compares regions, utilization, hardware refresh, and carbon-aware scheduling.

### Part C - Carbon-Aware Policy

Common pattern:
- Student chooses policy and accepted trade-off.

Track realization:
- iPhone policy balances battery, privacy, and offload.
- Oura Ring policy balances battery life and update/sync cadence.
- RoboTaxi policy balances safety latency and noncritical workload deferral.
- Cloud Fleet policy balances carbon, cost, latency, and demand rebound.

## Implementation Requirements

- Track variants need energy unit and carbon-relevant scenario.
- Device energy values must come from hardware/profile data or sourced estimates.
- Cloud Fleet needs grid-region provenance.

## Ledger And Report

Save:
- predicted dominant carbon/energy term
- selected carbon policy
- operational/lifecycle result
- trade-off accepted
- rebound or hidden-cost risk

Report target:
- A sustainability policy memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- energy, carbon, lifecycle, placement, time shifting, and rebound policy.

Minimum classroom demo:
- compare Cloud Fleet regional carbon with iPhone/Oura battery-scale energy framing.

Completion path:
- predict dominant carbon/energy term, inspect lifecycle/placement frontier, choose carbon-aware policy.

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
| iPhone | Balances local battery energy, privacy, and offload carbon/latency. |
| Oura Ring | Balances always-on duty cycle, battery life, and sync/update cadence. |
| RoboTaxi | Balances local safety compute with noncritical workload deferral and fleet operations. |
| Cloud Fleet | Balances region, PUE, utilization, embodied carbon, cost, and SLA. |

## Common Misconceptions

- Efficiency always lowers total energy.
- Green energy means zero carbon.
- Embodied carbon is negligible.
- Carbon policy is independent of latency/reliability.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `energy_per_workload`
- `utilization`
- `region`
- `pue`
- `embodied_carbon`
- `demand_elasticity`

Needed outputs:
- `energy_carbon_stack`
- `placement_frontier`
- `lifecycle_result`
- `carbon_policy`

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

- Operational and lifecycle terms are distinguished.
- Policy includes trade-off.
- Rebound/hidden cost is named.
- Assumptions are sourced.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
