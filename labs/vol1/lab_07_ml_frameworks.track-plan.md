# V1-07 Track Plan: Framework Tax

## Purpose

This lab teaches that frameworks and runtimes introduce dispatch, synchronization, portability, fusion, and compilation trade-offs. The right runtime depends on the selected track.

## Shared Pedagogy

- Students predict when overhead dominates useful compute.
- They compare eager execution, fused execution, compiled graphs, and deployment runtimes.
- They choose a runtime path with an explicit portability/operator risk.

## Lab Flow

### Opening - Runtime Brief

Common narrative:
- The model is mathematically correct, but the execution stack can erase the expected performance.
- The student must choose a framework/runtime path for the deployment context.

Track realization:
- iPhone: Core ML or mobile runtime delegate must use NPU-supported ops.
- Oura Ring: runtime overhead may be unacceptable; code generation or fixed kernels matter.
- RoboTaxi: deterministic execution and predictable scheduling matter more than flexibility.
- Cloud Fleet: graph capture, batching, and compiler amortization matter.

### Part A - Dispatch Tax

Common pattern:
- Compare many small ops versus fewer fused ops.
- Display latency stack: dispatch, transfer, compute, sync, memory.

Track realization:
- iPhone highlights mobile delegate overhead and unsupported op fallback.
- Oura Ring highlights fixed-function embedded kernels and call overhead.
- RoboTaxi highlights synchronization jitter and pipeline deadlines.
- Cloud Fleet highlights kernel launch, host-device sync, and batching.

### Part B - Fusion And Compile Break-Even

Common pattern:
- Sweep inference count, shape dynamism, and fusion depth.
- Plot when compile/fusion cost pays back.

Track realization:
- iPhone break-even depends on repeated on-device inference and stable shapes.
- Oura Ring favors ahead-of-time fixed kernels.
- RoboTaxi favors predictable compiled pipelines over dynamic flexibility.
- Cloud Fleet favors graph capture for high-volume stable workloads.

### Part C - Runtime Choice

Common pattern:
- Student chooses runtime path and records unsupported-op risk.

Track realization:
- iPhone decision covers delegate selection and fallback risk.
- Oura Ring decision covers fixed kernels and firmware update constraints.
- RoboTaxi decision covers deterministic runtime and certification risk.
- Cloud Fleet decision covers compiler/batching policy and rollback risk.

## Implementation Requirements

- Add runtime catalog in MLSysIM or `mlsysbook_labs`.
- Track variants should provide supported runtime choices and unsupported-op consequences.
- Runtime overhead should be separated from hardware dispatch tax.

## Ledger And Report

Save:
- predicted overhead source
- chosen runtime path
- break-even point
- unsupported-op or portability risk
- validation test

Report target:
- A runtime deployment recommendation for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- runtime dispatch, fusion, compilation, delegates, and portability.

Minimum classroom demo:
- compare many-small-ops eager execution to fused/compiled execution on iPhone and Cloud Fleet.

Completion path:
- predict overhead source, inspect dispatch stack, find compile/fusion break-even, choose runtime path.

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
| iPhone | Chooses mobile runtime/delegate path and tests unsupported-op fallback. |
| Oura Ring | Chooses fixed kernels or generated code rather than dynamic runtime overhead. |
| RoboTaxi | Chooses deterministic runtime with predictable scheduling and validation path. |
| Cloud Fleet | Chooses graph capture/compiled/batched runtime where reuse amortizes compilation. |

## Common Misconceptions

- Framework overhead is negligible for all workloads.
- Compilation always helps.
- Portability is free.
- Unsupported ops are rare enough to ignore.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `runtime_strategy`
- `op_count`
- `shape_dynamism`
- `reuse_count`
- `delegate_support`

Needed outputs:
- `latency_stack`
- `compile_break_even`
- `unsupported_op_warning`
- `runtime_decision`

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

- Overhead source is correctly diagnosed.
- Break-even is used as evidence.
- Runtime choice matches track.
- Portability/operator risk is explicit.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
