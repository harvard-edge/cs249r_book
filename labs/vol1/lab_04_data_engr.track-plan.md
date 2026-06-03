# V1-04 Track Plan: Data Gravity

## Purpose

This lab teaches that data movement, preprocessing, storage, and retention are system design choices. The same dataset can be cheap or impossible depending on where the selected track generates and consumes data.

## Shared Pedagogy

- Students diagnose where the data pipeline starves the model or deployment.
- They compare moving data, moving compute, sampling less, caching, or compressing.
- They choose a pipeline architecture with a clear residual failure mode.

## Lab Flow

### Opening - Data System Brief

Common narrative:
- The model is only as good as the data path feeding it.
- The student must decide where data is collected, preprocessed, stored, and filtered.

Track realization:
- iPhone: user data should often stay on-device; bandwidth and privacy shape collection.
- Oura Ring: low-rate biosignals, storage limits, and battery make sampling policy central.
- RoboTaxi: high-bandwidth sensor streams and rare-event retention dominate.
- Cloud Fleet: object storage, preprocessing throughput, and accelerator starvation dominate.

### Part A - Feed The Model

Common pattern:
- Render a pipeline diagram with stage capacities and utilization.
- Show bottleneck stage and stalled compute.

Track realization:
- iPhone bottleneck may be local preprocessing or uplink constraints.
- Oura Ring bottleneck may be sampling cadence, flash storage, or BLE transfer.
- RoboTaxi bottleneck may be sensor ingest and event triage.
- Cloud Fleet bottleneck may be storage read bandwidth or preprocessing workers.

### Part B - Data Movement Frontier

Common pattern:
- Compare strategies: preprocess local, send raw data, cache, compress, sample, or move compute.
- Plot latency, cost, privacy, and quality trade-offs.

Track realization:
- iPhone compares on-device feature extraction versus cloud upload.
- Oura Ring compares raw signal retention versus summaries and rare-event snippets.
- RoboTaxi compares local event mining versus fleet upload.
- Cloud Fleet compares caching, sharding, prefetch, and regional placement.

### Part C - Pipeline Architecture

Common pattern:
- Student chooses preprocessing location, cache policy, and retention policy.
- Decision must state the accepted bias or data loss.

Track realization:
- iPhone report emphasizes privacy-preserving collection.
- Oura Ring report emphasizes battery/storage-aware sampling.
- RoboTaxi report emphasizes rare-event capture and safety evidence.
- Cloud Fleet report emphasizes throughput and cost control.

## Implementation Requirements

- Track variants need default data source, data rate, retention goal, and privacy stance.
- MLSysIM should own data-rate and pipeline result schemas.
- Plans should distinguish hardware storage capacity from scenario retention policy.

## Ledger And Report

Save:
- data source and rate assumption
- bottleneck stage
- chosen movement/preprocessing strategy
- retention policy
- accepted data risk

Report target:
- A data pipeline architecture memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- data movement, preprocessing, storage, and retention architecture.

Minimum classroom demo:
- show the pipeline bottleneck and compare local preprocessing versus movement for RoboTaxi and Cloud Fleet.

Completion path:
- identify pipeline bottleneck, compare movement strategies, choose preprocessing/cache/retention policy.

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
| iPhone | Keeps sensitive data local where possible and uses summaries or selective upload. |
| Oura Ring | Chooses battery/storage-aware sampling and retention, often summaries over raw signals. |
| RoboTaxi | Prioritizes local triage and rare-event retention under high sensor volume. |
| Cloud Fleet | Optimizes storage, prefetch, sharding, and preprocessing to avoid accelerator starvation. |

## Common Misconceptions

- Data is free once collected.
- Moving compute and moving data are equivalent.
- Average data rate captures rare bursts.
- Retention policy has no model-quality consequence.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `data_rate`
- `preprocessing_location`
- `cache_policy`
- `retention_policy`

Needed outputs:
- `pipeline_bottleneck`
- `compute_starvation`
- `movement_frontier`
- `pipeline_architecture`

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

- Bottleneck stage is identified.
- Pipeline choice matches track constraints.
- Retention trade-off is stated.
- Privacy or rare-event risk is addressed.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
