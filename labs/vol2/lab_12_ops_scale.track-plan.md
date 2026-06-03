# V2-12 Track Plan: The Silent Fleet

## Purpose

This lab teaches operations at scale: model count, site count, dependencies, platform ROI, canaries, alert fatigue, automation, and silent failure risk.

## Shared Pedagogy

- Students predict when manual operations fail.
- They compare canary duration, monitoring thresholds, automation investment, and alert strategies.
- They choose an ops architecture.

## Lab Flow

### Opening - Fleet Ops Brief

Common narrative:
- Operational load grows faster than the team expects.
- The selected track changes what the fleet is.

Track realization:
- iPhone: fleet is app/device versions and local model variants.
- Oura Ring: fleet is wearable firmware/model versions and sync states.
- RoboTaxi: fleet is vehicles, geographies, sensors, and safety rollouts.
- Cloud Fleet: fleet is models, services, accelerators, tenants, and regions.

### Part A - Complexity Growth

Common pattern:
- Sweep model/site/service count and show operational load.

Track realization:
- iPhone sweeps device tiers, OS versions, and app rollout cohorts.
- Oura Ring sweeps firmware cohorts and sensor quality states.
- RoboTaxi sweeps vehicle count, regions, and safety dependencies.
- Cloud Fleet sweeps model/service count and platform dependencies.

### Part B - Canary/Automation Frontier

Common pattern:
- Sweep canary duration, alert threshold, automation level, and platform investment.
- Plot detection speed, false alerts, cost, and damage.

Track realization:
- iPhone canary is app/model rollout to device cohorts.
- Oura Ring canary is firmware/model rollout with battery/health signal guardrails.
- RoboTaxi canary is geography/vehicle-restricted safety rollout.
- Cloud Fleet canary is service traffic rollout and alert tuning.

### Part C - Ops Architecture

Common pattern:
- Student records monitoring, rollout, alerting, and automation investment.

Track realization:
- iPhone policy protects user experience and privacy-safe telemetry.
- Oura Ring policy protects sensing/battery continuity.
- RoboTaxi policy protects safety and fleet halt/rollback.
- Cloud Fleet policy protects SLA, cost, and platform scalability.

## Implementation Requirements

- Track variants need fleet unit, rollout unit, monitoring signal, and automation cost.
- Ops policy should carry forward into capstone.
- Incident discussion prompts should be instructor-ready.

## Ledger And Report

Save:
- predicted ops overload point
- canary policy
- monitoring/alert strategy
- automation investment
- silent-failure risk

Report target:
- An operations architecture decision for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- ops complexity growth, canaries, automation, alert fatigue, and platform ROI.

Minimum classroom demo:
- show operational load crossing team capacity for Cloud Fleet and rollout burden for RoboTaxi.

Completion path:
- predict ops overload, inspect complexity growth, choose canary/automation/monitoring architecture.

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
| iPhone | Manages app/model cohorts, device versions, and privacy-safe telemetry. |
| Oura Ring | Manages firmware/model cohorts, battery regressions, and sync states. |
| RoboTaxi | Manages vehicle/geography rollout, safety monitors, and fleet halt/rollback. |
| Cloud Fleet | Manages models/services/regions with platform automation and alert strategy. |

## Common Misconceptions

- Operations scale linearly with model count.
- More alerts mean safer operations.
- Canaries are always cheap.
- Hiring alone solves platform complexity.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `fleet_unit_count`
- `model_or_version_count`
- `canary_duration`
- `alert_threshold`
- `automation_level`

Needed outputs:
- `ops_load_curve`
- `alert_rate`
- `detection_damage_frontier`
- `ops_architecture`

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

- Ops load is quantified.
- Canary/alert policy balances detection and noise.
- Automation ROI is justified.
- Silent-failure risk is explicit.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
