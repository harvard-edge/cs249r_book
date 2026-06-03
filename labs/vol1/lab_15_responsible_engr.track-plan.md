# V1-15 Track Plan: No Free Fairness

## Purpose

This lab teaches that responsible engineering constraints are system constraints. Fairness, privacy, explainability, robustness, and carbon introduce measurable overheads and trade-offs.

## Shared Pedagogy

- Students predict whether one responsibility metric can be improved without cost.
- They compare metric conflict and overhead stacks.
- They choose an obligation, system change, evidence, and residual harm.

## Lab Flow

### Opening - Responsibility Brief

Common narrative:
- The system affects people, so technical decisions create responsibility obligations.
- The student must choose which obligation is central for the selected track.

Track realization:
- iPhone: privacy, user consent, accessibility, and on-device explainability matter.
- Oura Ring: health-adjacent inference, false alarms, privacy, and battery trade-offs matter.
- RoboTaxi: safety, accountability, rare-event failure, and explainability matter.
- Cloud Fleet: fairness, privacy, carbon, abuse prevention, and governance at scale matter.

### Part A - Metric Conflict

Common pattern:
- Threshold/policy sliders show competing metrics and subgroup outcomes.

Track realization:
- iPhone subgroups may include users, contexts, and accessibility conditions.
- Oura Ring subgroups may include physiology, activity, and sensor-contact variation.
- RoboTaxi subgroups may include road users, weather, and environment types.
- Cloud Fleet subgroups may include regions, tenants, languages, or populations.

### Part B - Responsibility Budget

Common pattern:
- Add constraints: privacy, explainability, robustness, carbon, monitoring.
- Show overhead in quality, latency, energy, cost, or fairness.

Track realization:
- iPhone overhead appears as latency, battery, and model complexity.
- Oura Ring overhead appears as energy, memory, and firmware/OTA size.
- RoboTaxi overhead appears as latency, fallback complexity, and validation burden.
- Cloud Fleet overhead appears as cost, capacity, carbon, and governance delay.

### Part C - Responsible Decision

Common pattern:
- Student selects obligation, system change, audit signal, and residual harm.

Track realization:
- iPhone decision defends privacy/user impact.
- Oura Ring decision defends health-risk communication and privacy.
- RoboTaxi decision defends safety evidence and failure accountability.
- Cloud Fleet decision defends governance and population-scale trade-offs.

## Implementation Requirements

- Track variants need stakeholder, harmed party, obligation, and audit signal.
- Responsible overhead should connect to hardware/scenario metrics where possible.
- Report schema should capture "who is harmed if this fails."

## Ledger And Report

Save:
- selected responsibility obligation
- metric conflict
- system change
- audit evidence
- residual harm and owner

Report target:
- A responsible engineering decision memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- responsible constraints as measurable systems trade-offs.

Minimum classroom demo:
- add privacy/explainability/robustness controls and show overhead stack for Oura Ring and Cloud Fleet.

Completion path:
- predict metric conflict, build responsibility budget, choose obligation/system change/evidence.

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
| iPhone | Defends local privacy, accessibility, and user impact with latency/battery overhead visible. |
| Oura Ring | Defends health-adjacent privacy, false-alarm handling, and battery/memory limits. |
| RoboTaxi | Defends safety/accountability/rare-event harm with validation overhead. |
| Cloud Fleet | Defends population-scale fairness, privacy, carbon, and governance trade-offs. |

## Common Misconceptions

- Fairness is a post-processing checkbox.
- Responsible constraints are nontechnical.
- One metric can satisfy every stakeholder.
- Governance overhead is separate from system design.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `subgroups`
- `metric_choice`
- `responsibility_controls`
- `overhead_model`
- `audit_signal`

Needed outputs:
- `metric_conflict`
- `overhead_stack`
- `responsible_decision`
- `residual_harm`

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

- Obligation is track-specific.
- Metric conflict is explained.
- Overhead is quantified.
- Harmed party and residual risk are named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
