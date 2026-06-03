# V1-09 Track Plan: Selection Paradox

## Purpose

This lab teaches that data selection is a systems decision. More data, better data, cheaper data, and representative data pull the system in different directions.

## Shared Pedagogy

- Students predict whether quantity or quality produces more value.
- They explore utility-cost and coverage-risk frontiers.
- They choose a data policy and record accepted bias or coverage risk.

## Lab Flow

### Opening - Data Selection Brief

Common narrative:
- The team cannot collect or label everything.
- The student must choose what data matters most for the selected track.

Track realization:
- iPhone: user data is private, local, and behaviorally diverse.
- Oura Ring: continuous signals are cheap per sample but expensive in battery/storage.
- RoboTaxi: rare events are the most valuable but hardest to collect.
- Cloud Fleet: scale makes cheap noisy data tempting but can hide subgroup failure.

### Part A - Quality Versus Quantity

Common pattern:
- Sliders alter dataset size, noise, label quality, and acquisition cost.
- Plot utility, compute cost, and deployment risk.

Track realization:
- iPhone emphasizes personalization and privacy-preserving collection.
- Oura Ring emphasizes signal quality and sampling cadence.
- RoboTaxi emphasizes rare-event coverage and long-tail safety cases.
- Cloud Fleet emphasizes large-scale data quality and training cost.

### Part B - Coverage And Inequality

Common pattern:
- Selection policy changes subgroup heatmap and global metric.
- Students see that average accuracy can improve while vulnerable regions degrade.

Track realization:
- iPhone subgroups may be usage contexts, lighting, accents, or device posture.
- Oura Ring subgroups may be physiology, sleep states, skin/contact quality, or activity level.
- RoboTaxi subgroups may be weather, road users, lighting, and unusual objects.
- Cloud Fleet subgroups may be tenants, languages, regions, or demand classes.

### Part C - Data Policy

Common pattern:
- Student selects acquisition, curation, filtering, and rare-event policy.
- Decision must say what data they would collect next.

Track realization:
- iPhone policy balances privacy and local coverage.
- Oura Ring policy balances battery/storage against medically meaningful coverage.
- RoboTaxi policy prioritizes rare events and safety evidence.
- Cloud Fleet policy balances scale, cost, and subgroup reliability.

## Implementation Requirements

- Add track-specific dataset and subgroup metadata.
- Utility model should separate quality, coverage, and compute cost.
- Ledger schema should carry data policy forward into responsible/robustness labs.

## Ledger And Report

Save:
- selected data policy
- coverage risk
- quality/quantity trade-off
- next data to collect
- accepted bias or blind spot

Report target:
- A data selection policy memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- data quality, quantity, coverage, subgroup risk, and collection policy.

Minimum classroom demo:
- show how rare-event coverage changes RoboTaxi outcome while Oura Ring is constrained by sampling/storage.

Completion path:
- predict quality/quantity effect, inspect coverage heatmap, choose acquisition/curation/rare-event policy.

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
| iPhone | Balances private local data, personalization, and coverage of user contexts. |
| Oura Ring | Chooses sampling/label policy that preserves battery and physiological coverage. |
| RoboTaxi | Prioritizes rare-event and long-tail safety coverage over raw volume. |
| Cloud Fleet | Balances cheap scale with subgroup quality and compute cost. |

## Common Misconceptions

- More data always improves deployment.
- Global accuracy captures subgroup failure.
- Rare events can be ignored if average is high.
- Data selection is separate from system cost.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `dataset_size`
- `label_quality`
- `noise_rate`
- `coverage_policy`
- `rare_event_weight`

Needed outputs:
- `utility_cost_frontier`
- `coverage_heatmap`
- `subgroup_risk`
- `data_policy`

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

- Policy addresses quality and coverage.
- Subgroup/rare-event risk is named.
- Cost or battery/storage implication is included.
- Next data to collect is explicit.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
