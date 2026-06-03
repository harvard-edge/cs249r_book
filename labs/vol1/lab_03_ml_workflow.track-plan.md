# V1-03 Track Plan: Constraint Tax

## Purpose

This lab teaches that deployment constraints propagate backward into data, training, validation, release, and monitoring. Late discovery of a constraint creates rework and risk.

## Shared Pedagogy

- Students trace how one deployment constraint changes the whole workflow.
- They compare iteration speed against confidence and risk.
- They choose a workflow policy for the selected track.

## Lab Flow

### Opening - Constraint Propagation Brief

Common narrative:
- A team discovers a deployment constraint after the model is already trained.
- The student must redesign the workflow so the constraint is tested earlier.

Track realization:
- iPhone: thermal and battery tests must happen before feature freeze.
- Oura Ring: SRAM/flash and battery tests must happen before data collection assumptions harden.
- RoboTaxi: p99 and rare-event validation must happen before road-test expansion.
- Cloud Fleet: cost, load, and utilization tests must happen before launch.

### Part A - Constraint Propagation

Common pattern:
- Show workflow stages and highlight where the selected track's constraint first appears.
- Let students move validation earlier or later and see rework cost.

Track realization:
- iPhone highlights device profiling, thermal soak tests, and privacy review.
- Oura Ring highlights memory budgeting, OTA packaging, and battery simulation.
- RoboTaxi highlights scenario coverage, latency gates, and safety signoff.
- Cloud Fleet highlights load testing, capacity planning, and cost review.

### Part B - Iteration Frontier

Common pattern:
- Sliders control validation depth, automation, data size, and hardware realism.
- Plot iteration time versus residual deployment risk.

Track realization:
- iPhone balances fast simulator iteration against physical-device confidence.
- Oura Ring balances tiny-device realism against slow embedded test cycles.
- RoboTaxi balances scenario replay breadth against safety-critical validation time.
- Cloud Fleet balances staging load fidelity against cost and launch speed.

### Part C - Workflow Policy

Common pattern:
- Student chooses release gates, validation cadence, retraining triggers, and rollback rules.
- The decision card records cost and residual blind spot.

Track realization:
- iPhone policy includes thermal/battery gate and app privacy checks.
- Oura Ring policy includes memory/OTA gate and battery-life regression tests.
- RoboTaxi policy includes p99, rare-event, and safety-case gates.
- Cloud Fleet policy includes load, cost, and SLO gates.

## Implementation Requirements

- Add per-track workflow stage labels and failure costs.
- Store workflow policy as structured ledger data.
- Avoid hardcoded hardware thresholds in the notebook; pull device facts from the selected profile.

## Ledger And Report

Save:
- constraint discovered
- stage where it was tested
- chosen validation/release policy
- iteration cost and residual risk

Report target:
- A workflow policy memo showing how the selected track changes the cost of late discovery.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- workflow cost of discovering constraints late.

Minimum classroom demo:
- move the deployment gate earlier and show reduced rework for iPhone thermal testing or RoboTaxi safety latency.

Completion path:
- trace one constraint through workflow stages, compare iteration/risk frontier, choose validation and release policy.

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
| iPhone | Adds early device profiling, thermal/battery gate, and privacy review to the workflow. |
| Oura Ring | Adds memory/OTA/battery checks before model and data assumptions harden. |
| RoboTaxi | Adds rare-event validation, p99 replay, and safety signoff before field rollout. |
| Cloud Fleet | Adds load/cost/SLO gates before launch and retraining decisions. |

## Common Misconceptions

- Workflow is project management rather than system design.
- Validation can wait until the end.
- Fast iteration is always better.
- Release policy is independent of deployment constraints.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `constraint_type`
- `validation_stage`
- `automation_level`
- `release_policy`

Needed outputs:
- `rework_cost`
- `iteration_time`
- `residual_deployment_risk`
- `workflow_policy`

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

- Workflow identifies where constraint should be tested.
- Policy balances speed and confidence.
- Release gate is track-specific.
- Residual blind spot is explicit.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
