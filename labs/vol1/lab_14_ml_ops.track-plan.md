# V1-14 Track Plan: Silent Degradation

## Purpose

This lab teaches operations after deployment: drift, monitoring delay, retraining cadence, rollback, escalation, technical debt, and silent failures.

## Shared Pedagogy

- Students predict when drift becomes visible.
- They compare under-monitoring, over-monitoring, under-retraining, and over-retraining.
- They choose an ops policy with a residual blind spot.

## Lab Flow

### Opening - Operations Brief

Common narrative:
- The model does not fail loudly; quality degrades before the team can see it.
- The student must design monitoring and response for the selected system.

Track realization:
- iPhone: drift may be user behavior, OS/runtime change, or device population shift.
- Oura Ring: drift may be sensor fit, firmware, physiology, or activity changes.
- RoboTaxi: drift may be geography, weather, construction, or sensor aging.
- Cloud Fleet: drift may be user mix, traffic, model dependency, or serving platform changes.

### Part A - Drift Visibility

Common pattern:
- Timeline shows true quality, observed signal, alert, and accumulated damage.

Track realization:
- iPhone observed signal may be opt-in telemetry or on-device proxy metric.
- Oura Ring observed signal may be delayed health labels or sensor-quality indicators.
- RoboTaxi observed signal may be safety disengagements, near misses, or simulation replay.
- Cloud Fleet observed signal may be online metrics, logs, and delayed labels.

### Part B - Retraining Cadence

Common pattern:
- Sweep retraining frequency and monitoring thresholds.
- Plot total cost/risk curve.

Track realization:
- iPhone balances privacy, battery, and update friction.
- Oura Ring balances battery, firmware/OTA, and false health alerts.
- RoboTaxi balances safety validation cost and field risk.
- Cloud Fleet balances compute cost, regressions, and stale model risk.

### Part C - Ops Policy

Common pattern:
- Student chooses monitoring, retraining, rollback, and escalation.

Track realization:
- iPhone policy includes local telemetry limits and app rollout.
- Oura Ring policy includes firmware/OTA rollback and sensor-quality monitors.
- RoboTaxi policy includes safety fallback and conservative rollout.
- Cloud Fleet policy includes canary, rollback, alert ownership, and cost guardrails.

## Implementation Requirements

- Track variants need drift sources, monitoring signal, label delay, and rollback mechanism.
- Ledger should store ops policy for capstone replay.
- Monitoring cost should include latency, energy, or staff cost as appropriate.

## Ledger And Report

Save:
- drift scenario
- monitoring signal
- retraining cadence
- rollback/escalation policy
- residual blind spot

Report target:
- A compact incident-prevention operations policy for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- drift visibility, monitoring, retraining, rollback, and operations policy.

Minimum classroom demo:
- show true quality degrading before observed signal triggers alert for Oura Ring or Cloud Fleet.

Completion path:
- predict drift visibility, choose monitoring/retraining cadence, set rollback/escalation policy.

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
| iPhone | Uses privacy-safe telemetry and staged app/model rollout. |
| Oura Ring | Uses sensor-quality, battery, firmware, and delayed-label monitors. |
| RoboTaxi | Uses simulation replay, safety monitors, and conservative rollout/rollback. |
| Cloud Fleet | Uses online metrics, canaries, retraining triggers, and SLO/cost alerts. |

## Common Misconceptions

- Models fail loudly.
- Labels arrive in time.
- Monitoring is free.
- Retrain more often is always safer.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `drift_rate`
- `label_delay`
- `monitoring_signal`
- `threshold`
- `retraining_cadence`
- `rollback_policy`

Needed outputs:
- `drift_timeline`
- `detection_delay`
- `damage_cost`
- `ops_policy`

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

- Monitoring signal matches track.
- Label delay is considered.
- Policy balances cost and risk.
- Residual blind spot is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
