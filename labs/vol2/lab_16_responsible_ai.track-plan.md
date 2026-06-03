# V2-16 Track Plan: The Fairness Budget

## Purpose

This lab teaches responsible AI at scale: metric conflict, feedback loops, governance overhead, audit depth, approval gates, red-team review, incident response, and unresolved conflict.

## Shared Pedagogy

- Students predict whether responsible governance can satisfy all metrics simultaneously.
- They compare fairness/privacy/transparency controls against latency, cost, and operational overhead.
- They choose a responsible AI pipeline and unresolved conflict.

## Lab Flow

### Opening - Governance Brief

Common narrative:
- Responsible AI is an operating system for decisions, not a final checklist.
- The selected track changes who is affected and what governance must prove.

Track realization:
- iPhone: governance focuses on user consent, privacy, accessibility, and local decisions.
- Oura Ring: governance focuses on health-adjacent inference, false alarms, and data sensitivity.
- RoboTaxi: governance focuses on safety, accountability, incident review, and public risk.
- Cloud Fleet: governance focuses on population-scale harms, tenant policy, red-team review, and audit.

### Part A - Metric Conflict And Feedback

Common pattern:
- Sliders/policies show fairness impossibility, feedback loops, and subgroup outcomes.

Track realization:
- iPhone tracks local user-context subgroups and feedback from app behavior.
- Oura Ring tracks physiological/context subgroups and health signal feedback.
- RoboTaxi tracks environment/road-user subgroups and safety feedback.
- Cloud Fleet tracks population/tenant/language/region subgroups.

### Part B - Governance Overhead

Common pattern:
- Add audit depth, approval gates, monitoring, explainability, red-team review, and incident response.
- Plot risk reduction versus overhead.

Track realization:
- iPhone overhead appears as latency, app release friction, and telemetry limits.
- Oura Ring overhead appears as battery, firmware cadence, and health communication burden.
- RoboTaxi overhead appears as validation time, safety case complexity, and rollout delay.
- Cloud Fleet overhead appears as cost, capacity, and governance latency.

### Part C - Responsible AI Pipeline

Common pattern:
- Student records metric, pipeline, overhead, unresolved conflict, and owner.

Track realization:
- iPhone pipeline defends local privacy and accessibility.
- Oura Ring pipeline defends sensitive-health communication and monitoring.
- RoboTaxi pipeline defends safety governance and public accountability.
- Cloud Fleet pipeline defends auditability and scaled policy enforcement.

## Implementation Requirements

- Track variants need affected stakeholders, governance artifact, and unresolved conflict.
- Report should include both structured choices and short rationale.
- Governance overhead should connect to system metrics where possible.

## Ledger And Report

Save:
- selected responsible AI metric/obligation
- governance controls
- overhead
- unresolved conflict
- accountable owner

Report target:
- A responsible AI pipeline memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- metric conflict, feedback loops, governance overhead, and responsible AI pipeline.

Minimum classroom demo:
- show governance overhead frontier for RoboTaxi safety governance and Cloud Fleet audit pipeline.

Completion path:
- predict metric/governance conflict, inspect overhead frontier, choose pipeline and unresolved conflict.

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
| iPhone | Defends consent, privacy, accessibility, and local decision governance. |
| Oura Ring | Defends health-adjacent false alarm handling, privacy, and communication obligations. |
| RoboTaxi | Defends safety governance, incident review, public accountability, and rollout gates. |
| Cloud Fleet | Defends population-scale governance, auditability, red-team review, and policy enforcement. |

## Common Misconceptions

- Responsible AI is a checklist.
- One fairness metric solves conflict.
- Governance has no latency/cost impact.
- Scale makes governance easier.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `stakeholders`
- `metric_choice`
- `governance_controls`
- `audit_depth`
- `approval_gates`

Needed outputs:
- `metric_conflict`
- `feedback_loop`
- `governance_overhead`
- `responsible_pipeline`

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

- Stakeholders are track-specific.
- Metric conflict is explained.
- Governance overhead is quantified.
- Unresolved conflict and owner are named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
