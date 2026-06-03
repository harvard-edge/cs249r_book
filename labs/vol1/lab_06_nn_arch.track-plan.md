# V1-06 Track Plan: Architecture Tax

## Purpose

This lab teaches that architecture families have different scaling shapes, inductive biases, and resource signatures. The selected track determines which architecture is defensible.

## Shared Pedagogy

- Students compare CNN, MLP, RNN, transformer, and efficient variants.
- They predict which architecture fails first under the selected workload.
- They choose an architecture and name the next scaling failure.

## Lab Flow

### Opening - Architecture Choice Brief

Common narrative:
- The team has several model families available, but deployment constraints make "best accuracy" insufficient.
- The student must choose architecture based on workload and system limits.

Track realization:
- iPhone: mobile architecture must use supported NPU kernels and stay thermal-safe.
- Oura Ring: architecture must be tiny, streamable, and storage-aware.
- RoboTaxi: architecture must support perception reliability and bounded latency.
- Cloud Fleet: architecture must scale under throughput, memory, and cost constraints.

### Part A - Architecture Signature

Common pattern:
- Show a cost table across architecture families.
- Include parameters, ops, activation memory, latency, and quality proxy.

Track realization:
- iPhone highlights NPU kernel support and sustained latency.
- Oura Ring highlights SRAM/flash fit and streaming.
- RoboTaxi highlights p99 latency and rare-object guardrails.
- Cloud Fleet highlights throughput, batch efficiency, and memory bandwidth.

### Part B - Scaling Shape

Common pattern:
- Sweep the variable that each architecture handles poorly.
- Plot linear, quadratic, width-squared, or resolution-driven growth.

Track realization:
- iPhone sweeps image resolution or sequence length under thermal limits.
- Oura Ring sweeps window size and sensor channels.
- RoboTaxi sweeps sensor resolution and object classes.
- Cloud Fleet sweeps context length, batch, or model width.

### Part C - Architecture Choice

Common pattern:
- Student selects architecture, constraint, and expected next failure.
- Decision should not be "highest accuracy"; it must be track-defensible.

Track realization:
- iPhone may choose a mobile CNN/efficient transformer variant.
- Oura Ring may choose a compact temporal model or tiny CNN.
- RoboTaxi may choose a perception architecture with deterministic latency envelope.
- Cloud Fleet may choose transformer serving/training architecture with capacity plan.

## Implementation Requirements

- Track variants need default workload, candidate architectures, and success metrics.
- MLSysIM should expose architecture cost functions instead of notebook formulas.
- Kernel support should be represented separately from theoretical compute.

## Ledger And Report

Save:
- predicted architecture bottleneck
- selected architecture
- track-specific guardrail metric
- next scaling failure
- validation requirement

Report target:
- An architecture recommendation memo tied to the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- architecture signatures and scaling shapes.

Minimum classroom demo:
- compare CNN/RNN/Transformer costs under Oura Ring and RoboTaxi constraints.

Completion path:
- compare architecture signatures, sweep expensive variable, choose architecture and next scaling failure.

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
| iPhone | Likely chooses mobile-efficient architecture with supported kernels and sustained latency. |
| Oura Ring | Likely chooses tiny temporal/CNN architecture with strict memory budget. |
| RoboTaxi | Likely chooses perception architecture with deterministic latency and safety guardrail. |
| Cloud Fleet | Likely chooses architecture based on throughput, context/batch scaling, and cost. |

## Common Misconceptions

- Architecture choice is only about accuracy.
- Transformers are always best.
- A model that scales in cloud scales to devices.
- Inductive bias is only a modeling issue.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `architecture_family`
- `workload_shape`
- `scaling_variable`
- `guardrail_metric`

Needed outputs:
- `architecture_cost_table`
- `scaling_curve`
- `feasibility_status`
- `next_failure`

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

- Architecture choice matches workload and constraint.
- Scaling shape is explained.
- Guardrail is named.
- Next failure is plausible.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
