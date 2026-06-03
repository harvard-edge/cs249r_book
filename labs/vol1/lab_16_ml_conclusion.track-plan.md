# V1-16 Track Plan: The Architect's Audit

## Purpose

This capstone synthesizes Volume 1. Students replay their ledger, inspect how earlier decisions imply an end-to-end architecture, perturb assumptions, and write a final architecture memo.

## Shared Pedagogy

- Students learn that architecture is the accumulated result of prior constraints.
- They audit sensitivity rather than merely summarize choices.
- They revise one decision and state a durable ML systems principle.

## Lab Flow

### Opening - Architecture Audit Brief

Common narrative:
- The student is now the system architect for the track they selected in Lab 00.
- The task is to defend the architecture implied by the ledger.

Track realization:
- iPhone: architecture memo defends on-device privacy, thermal limits, and mobile UX.
- Oura Ring: architecture memo defends tiny memory, battery life, sensing, and OTA updates.
- RoboTaxi: architecture memo defends safety-critical latency, reliability, and rare-event evidence.
- Cloud Fleet: architecture memo defends SLA, cost, utilization, and operational scale.

### Part A - Ledger Replay

Common pattern:
- Load actual or preset ledger.
- Render architecture map from prior decisions.

Track realization:
- iPhone map includes model, runtime, device constraints, privacy, and monitoring.
- Oura Ring map includes sensing, tiny inference, OTA, battery, and cloud handoff.
- RoboTaxi map includes sensors, local inference, fallback, validation, and fleet learning.
- Cloud Fleet map includes model, serving, data, infrastructure, ops, and cost.

### Part B - Sensitivity Audit

Common pattern:
- Perturb workload, model, hardware, or constraint assumptions.
- Show fragility heatmap.

Track realization:
- iPhone perturb battery age, device thermal state, and model size.
- Oura Ring perturb sampling cadence, flash headroom, and battery target.
- RoboTaxi perturb sensor count, weather mix, and p99 deadline.
- Cloud Fleet perturb demand, sequence length, cost, and carbon intensity.

### Part C - Architecture Memo

Common pattern:
- Student revises one decision and writes final top risk and mitigation.

Track realization:
- iPhone memo focuses on sustained on-device deployment.
- Oura Ring memo focuses on reliable tiny wearable operation.
- RoboTaxi memo focuses on safety case and latency margin.
- Cloud Fleet memo focuses on scalable production economics.

## Implementation Requirements

- Capstone must read real ledger entries or load track-specific presets.
- Track profile should drive architecture map labels and sensitivity knobs.
- Report export should produce an architecture memo, not a generic lab summary.

## Ledger And Report

Save:
- final architecture map
- most fragile assumption
- revised decision
- durable principle
- final top risk and mitigation

Report target:
- A Volume 1 architecture memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- ledger replay, sensitivity audit, and architecture memo.

Minimum classroom demo:
- load a preset track ledger, perturb one assumption, and revise one architecture decision.

Completion path:
- replay ledger, run sensitivity audit, revise one decision, export architecture memo.

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
| iPhone | Produces mobile architecture memo around privacy, thermal/battery, runtime, and monitoring. |
| Oura Ring | Produces tiny wearable memo around sensing, memory, battery, OTA, and sync. |
| RoboTaxi | Produces safety-critical edge memo around latency, reliability, validation, and rare events. |
| Cloud Fleet | Produces production architecture memo around SLA, cost, utilization, ops, and carbon. |

## Common Misconceptions

- Capstone is a summary rather than an audit.
- Earlier decisions do not constrain later architecture.
- A design is robust if it works for one workload.
- Residual risk is optional.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `ledger_entries`
- `sensitivity_knobs`
- `architecture_decisions`
- `risk_thresholds`

Needed outputs:
- `architecture_map`
- `fragility_heatmap`
- `revised_decision`
- `architecture_memo`

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

- Ledger evidence is used.
- Sensitivity result changes or validates a decision.
- Top risk is concrete.
- Memo is track-specific and defensible.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
