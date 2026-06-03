# V1-11 Track Plan: Hardware Roofline

## Purpose

This lab teaches roofline reasoning: compute-bound, bandwidth-bound, memory-bound, arithmetic intensity, tiling, precision, and accelerator choice.

## Shared Pedagogy

- Students predict whether a workload is compute-bound or memory-bound.
- They move a workload point using reuse, batching, precision, or tiling.
- They choose the accelerator path that leaves the least dangerous residual limitation.

## Lab Flow

### Opening - Accelerator Brief

Common narrative:
- The hardware has impressive peak numbers, but the workload only benefits if it can reach the relevant roof.

Track realization:
- iPhone: NPU peak TOPS only matter if ops are supported and memory traffic is controlled.
- Oura Ring: MCU or tiny accelerator limits make memory movement and SRAM central.
- RoboTaxi: edge accelerator must meet worst-case latency, not just average throughput.
- Cloud Fleet: GPU roofline and memory bandwidth determine utilization and cost.

### Part A - Roofline Diagnosis

Common pattern:
- Plot workload point against hardware roof.
- Label active regime and ridge point.

Track realization:
- iPhone plots mobile NPU/DRAM envelope.
- Oura Ring plots MCU/SRAM envelope.
- RoboTaxi plots vehicle-local edge accelerator envelope.
- Cloud Fleet plots H100 or fleet accelerator envelope.

### Part B - Move The Point

Common pattern:
- Apply tiling, fusion, batching, precision, or reuse.
- Show before/after roofline position and latency/energy delta.

Track realization:
- iPhone emphasizes operator fusion and supported precision.
- Oura Ring emphasizes tiling into SRAM and tiny kernels.
- RoboTaxi emphasizes deterministic latency and sensor pipeline overlap.
- Cloud Fleet emphasizes batching, tensor cores, and memory traffic reduction.

### Part C - Accelerator Decision

Common pattern:
- Student selects accelerator/runtime path and remaining limitation.

Track realization:
- iPhone decision names NPU path and CPU/GPU fallback risk.
- Oura Ring decision names MCU/tiny accelerator fit and firmware constraints.
- RoboTaxi decision names local accelerator and safety margin.
- Cloud Fleet decision names accelerator tier and utilization/cost risk.

## Implementation Requirements

- All hardware values must come from `mlsysim.Hardware`.
- Roofline APIs should be reusable across labs.
- Track variants should define default workload point and acceptable roofline headroom.

## Ledger And Report

Save:
- predicted regime
- actual regime
- transform applied
- accelerator decision
- residual bottleneck

Report target:
- A hardware acceleration diagnosis for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- roofline diagnosis and accelerator decision.

Minimum classroom demo:
- plot one workload point on Oura Ring/tiny and Cloud Fleet/H100 roofs to show different regimes.

Completion path:
- predict regime, move workload point, choose accelerator/runtime path and remaining bottleneck.

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
| iPhone | Identifies whether NPU path is compute, memory, or unsupported-op limited. |
| Oura Ring | Identifies SRAM/MCU roof and uses tiling/precision to fit. |
| RoboTaxi | Identifies edge accelerator headroom under p99 deadline. |
| Cloud Fleet | Identifies HBM/FLOP roof and utilization/cost consequences. |

## Common Misconceptions

- Peak TOPS predicts performance.
- All accelerators benefit equally from the same model.
- Arithmetic intensity is abstract and not actionable.
- Moving the roofline point has no trade-off.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `hardware_ref`
- `workload_point`
- `precision`
- `tiling`
- `fusion`
- `batching`

Needed outputs:
- `roofline_regime`
- `ridge_point`
- `before_after_point`
- `accelerator_decision`

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

- Regime is predicted and verified.
- Transform changes arithmetic intensity or bandwidth demand.
- Accelerator choice is track-specific.
- Remaining limitation is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
