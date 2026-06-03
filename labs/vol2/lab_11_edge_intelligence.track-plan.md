# V2-11 Track Plan: The Edge Thermodynamics Lab

## Purpose

This lab teaches device-edge-cloud placement, on-device energy, adaptation, federation, privacy, and communication trade-offs.

## Shared Pedagogy

- Students predict where intelligence should run.
- They compare local, edge, cloud, and hybrid placement under latency, privacy, energy, and quality.
- They choose edge architecture and adaptation policy.

## Lab Flow

### Opening - Edge Placement Brief

Common narrative:
- Intelligence can move between device, edge, and cloud, but each move changes constraints.

Track realization:
- iPhone: on-device NPU enables privacy-preserving inference but has battery/thermal cost.
- Oura Ring: tiny local inference saves communication but is severely memory/energy bound.
- RoboTaxi: safety-critical perception must remain local; fleet learning can use cloud.
- Cloud Fleet: edge comparison clarifies why centralization is not always enough.

### Part A - Placement Feasibility

Common pattern:
- Pipeline split diagram with latency, energy, privacy, and quality annotations.

Track realization:
- iPhone compares local NPU, phone-edge, and cloud.
- Oura Ring compares ring-only, ring-phone, and ring-cloud.
- RoboTaxi compares vehicle-local, roadside/depot, and cloud.
- Cloud Fleet compares centralized serving with edge caching/offload.

### Part B - Adaptation/Federation Frontier

Common pattern:
- Compare local adaptation, centralized retrain, and federated learning.
- Plot quality versus communication, privacy, and energy.

Track realization:
- iPhone explores personalization and federated update cost.
- Oura Ring explores tiny calibration and phone-mediated updates.
- RoboTaxi explores fleet learning from rare events.
- Cloud Fleet explores centralized retraining and edge feedback loops.

### Part C - Edge Architecture

Common pattern:
- Student records placement, adaptation, and privacy assumption.

Track realization:
- iPhone plan defends what stays on-device.
- Oura Ring plan defends tiny local model plus handoff.
- RoboTaxi plan defends vehicle-local safety path and fleet learning.
- Cloud Fleet plan defends central/edge split for cost and latency.

## Implementation Requirements

- This is a priority pilot for canonical tracks because the differences are naturally visible.
- Oura Ring and RoboTaxi hardware entries are required before full implementation.
- Placement solver should consume profile hardware and scenario thresholds.

## Ledger And Report

Save:
- predicted best placement
- selected placement
- adaptation/federation policy
- energy/latency/privacy result
- residual risk

Report target:
- An edge architecture memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- device-edge-cloud placement, adaptation, federation, energy, and privacy.

Minimum classroom demo:
- compare ring-only, phone-assisted, vehicle-local, and cloud placement for one workload.

Completion path:
- predict placement, compare local/edge/cloud/hybrid, choose adaptation and privacy policy.

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
| iPhone | Keeps privacy-sensitive inference on-device when battery/latency supports it. |
| Oura Ring | Uses tiny local inference plus phone/cloud handoff for heavier work. |
| RoboTaxi | Keeps safety-critical perception local and uses cloud for fleet learning. |
| Cloud Fleet | Uses edge comparison to decide when central serving is insufficient. |

## Common Misconceptions

- Edge always means faster.
- Cloud is always more accurate.
- Federation is free privacy.
- On-device adaptation has no battery cost.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `placement`
- `latency_budget`
- `privacy_requirement`
- `energy_budget`
- `adaptation_strategy`

Needed outputs:
- `placement_feasibility`
- `energy_latency_privacy_tradeoff`
- `adaptation_frontier`
- `edge_architecture`

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

- Placement matches hard constraints.
- Energy/privacy/quality trade-off is explicit.
- Adaptation cost is considered.
- Residual assumption is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
