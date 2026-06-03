# V2-17 Track Plan: The Fleet Synthesis

## Purpose

This capstone synthesizes Volume 2. Students replay fleet-scale decisions, inspect interactions across compute/network/storage/ops/responsibility, perturb assumptions, and produce a final design review.

## Shared Pedagogy

- Students learn that distributed ML systems are coupled across many constraints.
- They audit interactions, not isolated optimizations.
- They defend a final architecture and top risk.

## Lab Flow

### Opening - Fleet Design Review Brief

Common narrative:
- The student is presenting a final design review for the selected track.
- The design must hold up under interaction effects, not just single-lab metrics.

Track realization:
- iPhone: fleet review covers device population, local models, rollout, privacy, and support matrix.
- Oura Ring: fleet review covers wearable fleet, firmware/model updates, battery, sensing, and sync.
- RoboTaxi: fleet review covers vehicle fleet, safety, data feedback, local compute, and operations.
- Cloud Fleet: fleet review covers accelerator fleet, serving/training, network, storage, ops, cost, and responsibility.

### Part A - Fleet Ledger Replay

Common pattern:
- Load ledger or preset and render architecture map.

Track realization:
- iPhone map connects device tiers, runtime, rollout, privacy, and telemetry.
- Oura Ring map connects sensor, MCU, model, OTA, phone/cloud sync, and ops.
- RoboTaxi map connects sensors, local compute, validation, fleet learning, and safety ops.
- Cloud Fleet map connects compute, network, storage, serving, ops, governance, and carbon.

### Part B - Interaction Map

Common pattern:
- Perturb demand, failures, privacy, carbon, model size, or latency target.
- Show constraint interactions.

Track realization:
- iPhone perturb device heterogeneity, battery age, OS changes, and demand.
- Oura Ring perturb battery target, sampling cadence, firmware size, and connectivity.
- RoboTaxi perturb weather/geography, sensor failures, p99 target, and rollout scale.
- Cloud Fleet perturb demand, accelerator failures, privacy controls, carbon cap, and cost.

### Part C - Final Design Review

Common pattern:
- Student records final blueprint, top risk, mitigation, and evidence.

Track realization:
- iPhone review defends scalable mobile ML deployment.
- Oura Ring review defends reliable tiny wearable ML deployment.
- RoboTaxi review defends safety-critical edge AI fleet deployment.
- Cloud Fleet review defends production distributed ML fleet deployment.

## Implementation Requirements

- Capstone should load real ledger entries or canonical track presets.
- Track profile should drive the architecture vocabulary.
- Report export should be titled "Fleet Design Review" for Volume 2.

## Ledger And Report

Save:
- final fleet architecture
- most important interaction effect
- top risk
- mitigation
- evidence from earlier labs

Report target:
- A Volume 2 fleet design review for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- fleet ledger replay, interaction map, and final design review.

Minimum classroom demo:
- load a preset Cloud Fleet or RoboTaxi ledger and perturb demand/failure/privacy/carbon.

Completion path:
- replay ledger, inspect interaction map, identify top risk, export fleet design review.

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
| iPhone | Produces device-fleet review around rollout, privacy, local models, and support matrix. |
| Oura Ring | Produces wearable-fleet review around battery, firmware/model updates, sensing, and sync. |
| RoboTaxi | Produces safety-critical fleet review around local compute, validation, data feedback, and operations. |
| Cloud Fleet | Produces distributed ML fleet review around compute/network/storage/ops/cost/responsibility. |

## Common Misconceptions

- Capstone is a list of previous answers.
- Optimizing each subsystem independently yields a good fleet.
- Top risk can be generic.
- Evidence from earlier labs is optional.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `ledger_entries`
- `fleet_architecture`
- `interaction_knobs`
- `risk_thresholds`
- `mitigations`

Needed outputs:
- `fleet_architecture_map`
- `interaction_map`
- `top_risk`
- `final_design_review`

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

- Design uses ledger evidence.
- Interaction effect is explained.
- Top risk is specific and mitigated.
- Final review is track-specific and defensible.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
