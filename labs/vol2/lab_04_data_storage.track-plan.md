# V2-04 Track Plan: The Data Pipeline Wall

## Purpose

This lab teaches storage-compute mismatch: storage bandwidth, preprocessing, sharding, caching, contention, checkpointing, and accelerator starvation.

## Shared Pedagogy

- Students predict whether storage/preprocessing can feed compute.
- They sweep cache, shard, worker, and locality strategies.
- They choose a storage architecture or checkpoint policy.

## Lab Flow

### Opening - Storage Pipeline Brief

Common narrative:
- Accelerators are idle if the data path cannot feed them.
- The selected track changes which storage problem matters.

Track realization:
- iPhone: local storage, privacy, and upload policy shape data movement.
- Oura Ring: flash capacity, retention, and phone sync shape what can be stored.
- RoboTaxi: high-rate sensor logs, rare-event retention, and depot upload dominate.
- Cloud Fleet: object storage, preprocessing workers, sharding, and checkpoint bandwidth dominate.

### Part A - Storage-Compute Gap

Common pattern:
- Pipeline chart compares storage, preprocessing, transfer, and compute demand.

Track realization:
- iPhone shows local preprocessing and upload bottlenecks.
- Oura Ring shows flash retention and sync limitations.
- RoboTaxi shows sensor log volume and event triage.
- Cloud Fleet shows GPU starvation from storage/preprocess limits.

### Part B - Sharding And Cache Frontier

Common pattern:
- Sweep shard size, cache rate, prefetch, workers, and locality.
- Plot stall rate versus cost or energy.

Track realization:
- iPhone compares cache/local summary versus cloud upload.
- Oura Ring compares summaries, event snippets, and full signal retention.
- RoboTaxi compares local triage, depot upload, and fleet data lake ingestion.
- Cloud Fleet compares shard/cache/prefetch architectures.

### Part C - Storage Architecture

Common pattern:
- Student chooses cache, shard, retention, and checkpoint strategy.

Track realization:
- iPhone plan defends what stays on-device.
- Oura Ring plan defends flash and OTA/sync constraints.
- RoboTaxi plan defends rare-event retention and upload priority.
- Cloud Fleet plan defends storage tier and checkpoint architecture.

## Implementation Requirements

- Track variants need data volume, retention policy, and storage capacity source.
- Device storage belongs in hardware registry; retention policy belongs in scenario.
- Cloud Fleet may require infrastructure profile beyond one GPU.

## Ledger And Report

Save:
- predicted data wall
- active pipeline bottleneck
- selected storage/cache policy
- retention/checkpoint decision
- residual data loss or starvation risk

Report target:
- A data storage architecture memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- storage-compute gap, sharding/cache frontier, and storage architecture.

Minimum classroom demo:
- show accelerator starvation in Cloud Fleet and flash/sync pressure in Oura Ring.

Completion path:
- predict data wall, inspect pipeline chart, choose sharding/cache/retention/checkpoint policy.

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
| iPhone | Chooses local summaries/cache and privacy-aware upload policy. |
| Oura Ring | Chooses flash-aware retention and phone/cloud sync strategy. |
| RoboTaxi | Chooses local event triage, rare-event retention, and depot/cloud upload. |
| Cloud Fleet | Chooses storage tier, sharding, cache, prefetch, and checkpoint policy. |

## Common Misconceptions

- Fast compute means fast pipeline.
- Storage capacity is the same as storage bandwidth.
- Checkpoint cost is secondary.
- Rare-event retention is free.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `data_volume`
- `storage_bandwidth`
- `preprocess_rate`
- `cache_policy`
- `checkpoint_policy`

Needed outputs:
- `storage_compute_gap`
- `stall_rate`
- `cache_frontier`
- `storage_architecture`

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

- Pipeline bottleneck is identified.
- Architecture matches track storage path.
- Checkpoint/retention trade-off is named.
- Starvation or data-loss risk is explicit.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
