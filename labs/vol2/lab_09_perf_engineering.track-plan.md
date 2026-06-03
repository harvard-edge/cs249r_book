# V2-09 Track Plan: The Optimization Trap

## Purpose

This lab teaches performance engineering as bottleneck-driven work: profile first, optimize the active constraint, watch the bottleneck move, and stop when marginal gain is not worth cost.

## Shared Pedagogy

- Students predict which optimization helps most.
- They inspect a profile and identify the active bottleneck.
- They apply an optimization ladder and choose a stop rule.

## Lab Flow

### Opening - Optimization Brief

Common narrative:
- Optimization without diagnosis wastes effort and may move the system into a worse regime.

Track realization:
- iPhone: optimize for sustained on-device latency and battery, not just peak speed.
- Oura Ring: optimize memory/energy first; large-runtime complexity can be worse.
- RoboTaxi: optimize p99 and determinism; average speedups are insufficient.
- Cloud Fleet: optimize throughput/cost by active bottleneck across compute, memory, and communication.

### Part A - Bottleneck Diagnosis

Common pattern:
- Profile view shows time breakdown and active bottleneck.

Track realization:
- iPhone profile includes dispatch, memory, NPU/CPU fallback, and thermal state.
- Oura Ring profile includes SRAM misses, flash reads, and duty-cycle energy.
- RoboTaxi profile includes sensor ingest, preprocessing, inference, and queueing tail.
- Cloud Fleet profile includes kernel, memory, network, and host overhead.

### Part B - Optimization Ladder

Common pattern:
- Apply fusion, layout, precision, attention kernels, batching, overlap, or caching.
- Waterfall chart shows new bottleneck.

Track realization:
- iPhone applies mobile delegate/fusion/precision.
- Oura Ring applies fixed kernels, quantization, and buffering.
- RoboTaxi applies pipeline overlap, bounded queues, and deterministic kernels.
- Cloud Fleet applies fusion, FlashAttention-like kernels, precision, and communication overlap.

### Part C - Stop Rule

Common pattern:
- Student records optimization order and when to stop.

Track realization:
- iPhone stop rule considers battery and device support.
- Oura Ring stop rule considers firmware complexity and battery.
- RoboTaxi stop rule considers validation/certification cost.
- Cloud Fleet stop rule considers engineering time, cost, and regression risk.

## Implementation Requirements

- Track variants need profile components and valid optimization actions.
- Result schema should include old bottleneck, new bottleneck, gain, and risk.
- Decision card should force "why not optimize further?"

## Ledger And Report

Save:
- predicted best optimization
- actual bottleneck
- ordered optimization list
- final stop rule
- regression risk

Report target:
- A bottleneck-driven optimization plan for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- profile-driven optimization, bottleneck movement, and stop rule.

Minimum classroom demo:
- apply one optimization and show the bottleneck move for iPhone or Cloud Fleet.

Completion path:
- predict best optimization, diagnose profile, apply optimization ladder, choose stop rule.

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
| iPhone | Optimizes supported mobile path and stops before battery/regression risk dominates. |
| Oura Ring | Optimizes memory/energy and avoids runtime complexity that cannot fit. |
| RoboTaxi | Optimizes p99 determinism, not average speed. |
| Cloud Fleet | Optimizes active compute/memory/network bottleneck for cost and throughput. |

## Common Misconceptions

- Optimization should start before profiling.
- The biggest code section is always the best target.
- More optimizations always help.
- Average speedup proves success.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `profile_components`
- `optimization_sequence`
- `precision`
- `layout`
- `fusion`
- `overlap`

Needed outputs:
- `active_bottleneck`
- `waterfall`
- `new_bottleneck`
- `stop_rule`

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

- Optimization follows diagnosis.
- Before/after evidence is used.
- New bottleneck is identified.
- Stop rule includes risk/cost.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
