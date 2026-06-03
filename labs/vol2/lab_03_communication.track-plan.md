# V2-03 Track Plan: Network Fabric Design

## Purpose

This lab should focus on network fabric choices: link bandwidth, latency, topology, bisection bandwidth, oversubscription, congestion, and placement. Collective algorithm details should stay lighter now that V2-06 exists.

## Shared Pedagogy

- Students predict whether bandwidth, latency, topology, or placement binds.
- They compare fabric/topology choices under the selected track.
- They choose a fabric or placement strategy and state the assumption it depends on.

## Lab Flow

### Opening - Fabric Brief

Common narrative:
- Communication is part of the ML system, even when the model is correct.
- The selected track changes the meaning of a network bottleneck.

Track realization:
- iPhone: communication is device-to-edge/cloud uplink, privacy-safe telemetry, and update delivery.
- Oura Ring: communication is intermittent BLE/phone sync and OTA payload movement.
- RoboTaxi: communication is vehicle-local sensor fabric plus fleet upload of events.
- Cloud Fleet: communication is NVLink, InfiniBand/Ethernet, topology, and cross-node bandwidth.

### Part A - Fabric Budget

Common pattern:
- Compare link bandwidth, latency, message size, and update cadence.

Track realization:
- iPhone checks model update, telemetry, and offload feasibility.
- Oura Ring checks BLE sync, OTA payload, and intermittent connection.
- RoboTaxi checks sensor-to-compute and event-upload budget.
- Cloud Fleet checks HBM/NVLink/IB/Ethernet hierarchy.

### Part B - Topology Frontier

Common pattern:
- Sweep node/device count, oversubscription, hop count, and bisection bandwidth.

Track realization:
- iPhone compares local, edge, and cloud paths.
- Oura Ring compares ring-to-phone-to-cloud pipeline.
- RoboTaxi compares sensor bus, in-vehicle compute, and depot/cloud upload.
- Cloud Fleet compares fat tree, torus, dragonfly, Ethernet, InfiniBand, and hierarchy.

### Part C - Fabric Decision

Common pattern:
- Student records topology, bandwidth assumption, placement constraint, and risk.

Track realization:
- iPhone decision defends what stays local and what can cross network.
- Oura Ring decision defends sync/update policy under intermittent connectivity.
- RoboTaxi decision defends local sensor pipeline and upload triage.
- Cloud Fleet decision defends fabric choice and oversubscription risk.

## Implementation Requirements

- Move collective-specific teaching content to V2-06 where possible.
- Track variants need network path definitions.
- Fabric result should name active link and mitigation.

## Ledger And Report

Save:
- predicted communication bottleneck
- selected path/topology
- active link
- bandwidth/latency assumption
- residual congestion or connectivity risk

Report target:
- A network fabric or communication-placement decision for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- network path, fabric, topology, bisection, and placement assumptions.

Minimum classroom demo:
- contrast Oura Ring intermittent sync with Cloud Fleet fabric hierarchy.

Completion path:
- predict communication bottleneck, inspect fabric/path budget, choose topology or placement policy.

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
| iPhone | Defines local versus networked path for telemetry/update/offload. |
| Oura Ring | Defines ring-phone-cloud sync and OTA payload policy. |
| RoboTaxi | Defines sensor fabric, vehicle-local path, and fleet upload triage. |
| Cloud Fleet | Defines NVLink/IB/Ethernet topology and oversubscription risk. |

## Common Misconceptions

- Communication only matters for distributed training.
- Bandwidth alone defines a network.
- Intermittent links can be treated like datacenter links.
- Placement is independent of fabric.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `message_size`
- `link_bandwidth`
- `link_latency`
- `topology`
- `placement`

Needed outputs:
- `fabric_budget`
- `active_link`
- `topology_frontier`
- `fabric_decision`

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

- Network path is track-specific.
- Bandwidth and latency are both addressed.
- Topology/placement assumption is explicit.
- Residual congestion/connectivity risk is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
