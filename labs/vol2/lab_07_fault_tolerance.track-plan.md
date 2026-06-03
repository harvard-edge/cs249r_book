# V2-07 Track Plan: When Failure Is Routine

## Purpose

This lab teaches failure probability, MTBF, fleet size, job duration, checkpointing, recovery overhead, retries, replication, and serving reliability.

## Shared Pedagogy

- Students predict probability of clean completion or uninterrupted service.
- They sweep fleet size, job duration, MTBF, checkpoint interval, and recovery policy.
- They choose a resilience policy and name uncovered failures.

## Lab Flow

### Opening - Failure Budget Brief

Common narrative:
- At scale, rare failures become routine.
- The selected track changes what counts as failure and recovery.

Track realization:
- iPhone: failures are app crashes, OS/device variability, battery, and rollout regressions.
- Oura Ring: failures are battery depletion, sensor dropout, firmware rollback, and sync gaps.
- RoboTaxi: failures are safety-critical perception, sensor degradation, and vehicle downtime.
- Cloud Fleet: failures are accelerator/node/job/service failures and checkpoint storms.

### Part A - Failure Exposure

Common pattern:
- Sweep device/fleet/job size and duration.
- Show probability of clean completion or failure-free operation.

Track realization:
- iPhone sweeps deployed device population and update duration.
- Oura Ring sweeps wearable fleet, battery state, and firmware rollout.
- RoboTaxi sweeps vehicle count, hours driven, and sensor/compute failures.
- Cloud Fleet sweeps accelerators, training duration, and service replicas.

### Part B - Recovery Frontier

Common pattern:
- Compare checkpointing, retries, replication, fallback, and rollback.
- Plot useful work versus recovery overhead.

Track realization:
- iPhone uses staged rollout, rollback, and local fallback.
- Oura Ring uses firmware rollback, sync retry, and safe-mode inference.
- RoboTaxi uses redundant sensors/compute, degraded mode, and fleet halt rules.
- Cloud Fleet uses checkpoint interval, async checkpointing, replication, and retries.

### Part C - Resilience Policy

Common pattern:
- Student records covered and uncovered failures.

Track realization:
- iPhone policy protects user experience and data loss.
- Oura Ring policy protects battery/sensing continuity.
- RoboTaxi policy protects safety and controlled degradation.
- Cloud Fleet policy protects job/service availability and cost.

## Implementation Requirements

- Track variants need failure taxonomy and MTBF/incident assumptions.
- Recovery result should distinguish retry, rollback, checkpoint, replication, and fallback.
- Reports should require uncovered failure mode.

## Ledger And Report

Save:
- predicted failure exposure
- selected recovery policy
- overhead/cost
- covered failures
- uncovered failures

Report target:
- A resilience playbook for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- failure exposure, recovery frontier, and resilience policy.

Minimum classroom demo:
- sweep fleet size/job duration and compare vehicle fleet versus accelerator fleet exposure.

Completion path:
- predict failure exposure, compare recovery options, choose resilience policy and uncovered failures.

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
| iPhone | Uses staged rollout, rollback, and local fallback for app/model failures. |
| Oura Ring | Uses firmware rollback, safe mode, sync retry, and battery/sensor failure handling. |
| RoboTaxi | Uses redundancy, degraded mode, halt rules, and safety incident response. |
| Cloud Fleet | Uses checkpointing, retries, replication, and job/service recovery. |

## Common Misconceptions

- Rare failures stay rare at scale.
- Retries are free.
- Checkpointing only helps.
- Covered failure means all failure risk is gone.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `fleet_size`
- `duration`
- `mtbf`
- `checkpoint_interval`
- `recovery_strategy`

Needed outputs:
- `failure_probability`
- `recovery_frontier`
- `useful_work`
- `resilience_policy`

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

- Exposure is quantified.
- Recovery overhead is included.
- Policy matches failure taxonomy.
- Uncovered failure is explicit.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
