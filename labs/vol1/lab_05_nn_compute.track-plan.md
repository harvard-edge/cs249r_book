# V1-05 Track Plan: Activation Tax

## Purpose

This lab teaches operator-level cost: parameters, activations, MACs, bytes moved, and how activations can dominate deployment memory.

## Shared Pedagogy

- Students predict which tensor or memory component dominates.
- They sweep shape, batch, resolution, sequence length, precision, or tiling.
- They choose an operator configuration that fits the selected track.

## Lab Flow

### Opening - Operation Ledger Brief

Common narrative:
- A layer that looks small in parameter count can still fail because activations and memory movement dominate.
- The student must reason from tensor shapes to deployment feasibility.

Track realization:
- iPhone: activations stress DRAM bandwidth, NPU memory, and thermal power.
- Oura Ring: activations must fit tiny SRAM or require impossible buffering.
- RoboTaxi: activation memory and bandwidth affect worst-case perception latency.
- Cloud Fleet: activation and KV-like state affect batch throughput and memory pressure.

### Part A - Operation Ledger

Common pattern:
- Pick a layer or block shape.
- Display weights, activations, ops, bytes moved, and arithmetic intensity.

Track realization:
- iPhone uses mobile vision/audio block defaults.
- Oura Ring uses low-rate biosignal or keyword-style block defaults.
- RoboTaxi uses perception block defaults.
- Cloud Fleet uses transformer or service-model block defaults.

### Part B - Memory Cliff

Common pattern:
- Sweep the expensive shape variable until memory or bandwidth breaks.
- Show SRAM/DRAM/HBM cliffs where applicable.

Track realization:
- iPhone sweeps resolution or batch under thermal constraints.
- Oura Ring sweeps window length or channels against SRAM/flash.
- RoboTaxi sweeps camera resolution or parallel sensor streams.
- Cloud Fleet sweeps batch or sequence length.

### Part C - Layer Design

Common pattern:
- Student chooses shape, precision, tiling, or operator replacement.
- The decision records what accuracy or flexibility was sacrificed.

Track realization:
- iPhone may select INT8 and fused mobile kernels.
- Oura Ring may select a tiny CNN/DS-CNN and strict buffering.
- RoboTaxi may select bounded-latency architecture and tiling.
- Cloud Fleet may select batching, attention/kernel optimization, or precision changes.

## Implementation Requirements

- Operator calculations should move into MLSysIM result schemas.
- Track variants should provide default tensor shapes and memory hierarchy labels.
- Every visual needs a table fallback for exact byte counts.

## Ledger And Report

Save:
- predicted dominant cost
- actual dominant tensor/resource
- selected layer/operator design
- memory and latency headroom
- residual quality risk

Report target:
- An operator budget note explaining why the selected design fits the track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- operator cost, activations, bytes moved, and memory cliffs.

Minimum classroom demo:
- sweep shape/resolution until Oura Ring hits SRAM or iPhone hits thermal/memory pressure.

Completion path:
- predict dominant tensor/resource, inspect operation ledger, sweep memory cliff, choose layer/operator design.

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
| iPhone | Chooses shape/precision/fusion that preserves mobile memory and thermal headroom. |
| Oura Ring | Chooses tiny streaming or tiled operator design that fits SRAM and flash. |
| RoboTaxi | Chooses bounded-latency operator shape for sensor workload. |
| Cloud Fleet | Chooses batching/precision/reuse that improves throughput without memory pressure. |

## Common Misconceptions

- Parameter count is the whole memory story.
- FLOPs alone predict latency.
- Batching always helps.
- Activations are temporary so they do not matter.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `layer_shape`
- `batch_or_window`
- `precision`
- `tiling_strategy`

Needed outputs:
- `operation_ledger`
- `activation_memory`
- `bytes_moved`
- `memory_cliff`
- `operator_feasibility`

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

- Dominant cost is predicted and checked.
- Tensor dimensions explain result.
- Design fits track hardware.
- Sacrifice or residual quality risk is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
