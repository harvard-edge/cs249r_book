# V2-06 Track Plan: Collective Communication

## Purpose

This lab teaches collective operations: AllReduce, AllGather, ReduceScatter, Broadcast, alpha-beta models, ring/tree/hierarchical algorithms, overlap, compression, and staleness.

## Shared Pedagogy

- Students predict which collective algorithm wins and why.
- They compare latency and bandwidth terms across topology and message size.
- They choose a communication optimization and residual risk.

## Lab Flow

### Opening - Collective Brief

Common narrative:
- Communication algorithm choice is a systems decision, not a library detail.
- Cloud Fleet is the natural primary path, but other tracks can see analogous coordination costs.

Track realization:
- iPhone: collective-like coordination appears in federated or cohort aggregation.
- Oura Ring: aggregation is constrained by intermittent device sync and tiny payloads.
- RoboTaxi: fleet learning aggregates logs/updates across vehicles and depots.
- Cloud Fleet: GPU collectives dominate training and model-parallel execution.

### Part A - Operation Anatomy

Common pattern:
- Show bytes, rounds, participants, and dependency structure.

Track realization:
- iPhone shows federated update aggregation in small cohorts.
- Oura Ring shows tiny summaries or model-update acknowledgments.
- RoboTaxi shows fleet event/update aggregation.
- Cloud Fleet shows AllReduce/AllGather/ReduceScatter across accelerators.

### Part B - Algorithm/Topology Frontier

Common pattern:
- Compare ring, tree, and hierarchical algorithms under message size and topology.

Track realization:
- iPhone emphasizes uplink, cohort size, and privacy protocol overhead.
- Oura Ring emphasizes tiny payload and intermittent connection.
- RoboTaxi emphasizes depot/cloud hierarchy and upload windows.
- Cloud Fleet emphasizes NVLink/IB/Ethernet topology and alpha-beta crossover.

### Part C - Overlap/Compression Decision

Common pattern:
- Student chooses compression, overlap, hierarchy, or simpler algorithm.
- Decision includes staleness or quality risk.

Track realization:
- iPhone decision balances privacy, bandwidth, and battery.
- Oura Ring decision balances payload size and battery.
- RoboTaxi decision balances event fidelity and fleet update latency.
- Cloud Fleet decision balances throughput, staleness, and implementation risk.

## Implementation Requirements

- Existing pilot already uses `mlsysbook_labs` wrapper; extend it with canonical track variants.
- Track variants should keep Cloud Fleet as default but support constrained analogies.
- Result schema should include algorithm, binding term, topology assumption, and residual risk.

## Ledger And Report

Save:
- predicted best collective/coordination strategy
- selected algorithm/topology
- binding latency or bandwidth term
- compression/overlap decision
- staleness or quality risk

Report target:
- A communication design review for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- collective operation anatomy, algorithm/topology frontier, and overlap/compression decision.

Minimum classroom demo:
- compare ring/tree/hierarchy for Cloud Fleet and a constrained aggregation analogy for RoboTaxi.

Completion path:
- predict best collective/coordination strategy, inspect frontier, choose algorithm/topology/optimization.

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
| iPhone | Uses federated/cohort aggregation analogy with privacy and uplink constraints. |
| Oura Ring | Uses tiny intermittent payload aggregation and sync limits. |
| RoboTaxi | Uses depot/cloud hierarchy for fleet event/update aggregation. |
| Cloud Fleet | Uses GPU collectives with alpha-beta/topology reasoning. |

## Common Misconceptions

- Ring is always best.
- Bandwidth dominates every message size.
- Hierarchy is always overhead.
- Compression has no staleness or quality cost.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `operation`
- `message_size`
- `participants`
- `topology`
- `algorithm`
- `overlap`
- `compression`

Needed outputs:
- `collective_costs`
- `binding_term`
- `algorithm_frontier`
- `communication_decision`

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

- Algorithm choice is tied to message/topology.
- Binding alpha/beta term is explained.
- Optimization risk is stated.
- Track analogy is not overclaimed.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
