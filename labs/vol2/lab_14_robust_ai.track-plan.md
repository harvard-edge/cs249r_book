# V2-14 Track Plan: The Robustness Budget

## Purpose

This lab teaches robustness as a budget: distribution shift, silent errors, robust training, augmentation, ensembling, abstention, monitoring, fallback, and defense stack costs.

## Shared Pedagogy

- Students predict which robustness defense gives the best value.
- They compare clean accuracy, robust accuracy, latency, cost, and monitoring burden.
- They choose a defense stack and expected tax.

## Lab Flow

### Opening - Robustness Brief

Common narrative:
- Robustness failures may be silent until damage accumulates.
- The selected track changes the failure mode and acceptable fallback.

Track realization:
- iPhone: robustness covers device context, lighting/audio variation, and user behavior.
- Oura Ring: robustness covers sensor contact, physiology, activity, and firmware variation.
- RoboTaxi: robustness covers weather, rare objects, adversarial conditions, and safety fallback.
- Cloud Fleet: robustness covers distribution shift, abuse, tenant variation, and model updates.

### Part A - Robustness Tax

Common pattern:
- Add robust training, augmentation, ensembling, abstention, or fallback.
- Show clean/robust accuracy, latency, energy, and cost.

Track realization:
- iPhone robustness tax appears as battery/latency.
- Oura Ring robustness tax appears as memory/energy.
- RoboTaxi robustness tax appears as p99 latency and validation burden.
- Cloud Fleet robustness tax appears as cost, capacity, and throughput.

### Part B - Drift/Silent Error Timeline

Common pattern:
- Timeline shows shift onset, detection, accumulated harm, and response.

Track realization:
- iPhone drift appears through user context and telemetry.
- Oura Ring drift appears through sensor quality and delayed health labels.
- RoboTaxi drift appears through geography/weather and near misses.
- Cloud Fleet drift appears through online metrics, abuse, and delayed labels.

### Part C - Defense Stack

Common pattern:
- Student chooses defenses, fallback, and monitoring signal.

Track realization:
- iPhone stack includes local fallback and context monitors.
- Oura Ring stack includes sensor-quality gates and safe summaries.
- RoboTaxi stack includes abstention/degraded mode and safety monitors.
- Cloud Fleet stack includes guardrails, monitoring, and rollback.

## Implementation Requirements

- Track variants need failure taxonomy and fallback mechanism.
- Robustness result should show tax and new bottleneck.
- Report should include monitoring assumption.

## Ledger And Report

Save:
- predicted robustness defense
- selected defense stack
- robustness tax
- fallback policy
- monitoring assumption

Report target:
- A robustness defense plan for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- robustness tax, drift timeline, defense stack, fallback, and monitoring assumption.

Minimum classroom demo:
- add robustness defenses and show latency/cost tax for RoboTaxi and Cloud Fleet.

Completion path:
- predict best defense, inspect robustness tax, analyze drift timeline, choose defense stack.

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
| iPhone | Chooses context monitors/fallbacks that protect quality without draining battery. |
| Oura Ring | Chooses sensor-quality gates and safe summaries under memory/energy limits. |
| RoboTaxi | Chooses defense/fallback for weather/rare-object shifts with p99 and safety tax visible. |
| Cloud Fleet | Chooses monitoring/guardrails/rollback for distribution shift and abuse at scale. |

## Common Misconceptions

- Robustness improves without cost.
- Clean accuracy predicts robust accuracy.
- Silent errors are immediately visible.
- One defense handles every shift.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `shift_type`
- `defense_stack`
- `fallback_policy`
- `monitoring_signal`
- `robustness_budget`

Needed outputs:
- `robustness_tax`
- `clean_vs_robust_metric`
- `drift_timeline`
- `defense_plan`

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

- Defense maps to failure mode.
- Tax is quantified.
- Fallback and monitor are explicit.
- Monitoring assumption is realistic.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
