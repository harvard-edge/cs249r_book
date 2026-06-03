# V2-05 Track Plan: The Parallelism Puzzle

## Purpose

This lab teaches distributed training strategy: memory fit, communication cost, data/tensor/pipeline parallelism, ZeRO/FSDP, and hybrid plans.

## Shared Pedagogy

- Students predict whether memory or communication drives the parallelism choice.
- They sweep parallel degrees and memory-saving strategies.
- They choose a training architecture and name the new bottleneck.

## Lab Flow

### Opening - Parallelism Brief

Common narrative:
- Parallelism solves one problem by creating another.
- For non-cloud tracks, the lesson should often be why distributed training is not the device-side answer.

Track realization:
- iPhone: distributed training is a backend/federated or adaptation question, not local full training.
- Oura Ring: device training is infeasible; the question is cloud/offline training plus tiny deployment.
- RoboTaxi: fleet data enables centralized training; edge devices validate/deploy the result.
- Cloud Fleet: large-model training directly needs parallelism strategy.

### Part A - Memory Fit

Common pattern:
- Show weights, gradients, optimizer, activations, and partitioning.

Track realization:
- iPhone shows why local training is constrained and what adaptation can fit.
- Oura Ring shows why only tiny calibration/update can fit.
- RoboTaxi shows edge deployment versus central training memory.
- Cloud Fleet shows model-state sharding and activation pressure.

### Part B - Parallelism Frontier

Common pattern:
- Sweep data, tensor, pipeline, and ZeRO/FSDP choices.
- Plot memory, throughput, and communication.

Track realization:
- iPhone compares central fine-tune, federated aggregation, and on-device adaptation.
- Oura Ring compares cloud training and tiny OTA model update.
- RoboTaxi compares fleet data central training and edge validation.
- Cloud Fleet compares full distributed training strategies.

### Part C - Training Architecture

Common pattern:
- Student records parallelism plan, scaling limit, and communication assumption.

Track realization:
- iPhone plan defends where personalization happens.
- Oura Ring plan defends off-device training and update packaging.
- RoboTaxi plan defends fleet learning pipeline and local deployment validation.
- Cloud Fleet plan defends 3D/ZeRO/FSDP strategy and bottleneck.

## Implementation Requirements

- Do not force fake distributed training onto tiny/mobile tracks.
- Track variants should decide whether the device is training target, deployment target, or data source.
- Collective math should be referenced but detailed in V2-06.

## Ledger And Report

Save:
- predicted training bottleneck
- selected training location
- parallelism or adaptation plan
- communication assumption
- new bottleneck

Report target:
- A training architecture plan for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- training memory, parallelism frontier, and training architecture.

Minimum classroom demo:
- show Cloud Fleet parallelism frontier and explain why Oura Ring is deployment target, not training target.

Completion path:
- predict training bottleneck, inspect memory stack, compare parallel/adaptation strategies, choose plan.

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
| iPhone | Selects central/federated/adaptation plan rather than local full training. |
| Oura Ring | Selects off-device training plus tiny model update/calibration. |
| RoboTaxi | Selects fleet-data training plus edge validation/deployment pipeline. |
| Cloud Fleet | Selects data/tensor/pipeline/FSDP/ZeRO strategy with communication risk. |

## Common Misconceptions

- Every track should do distributed training.
- Sharding only solves memory.
- Communication is a minor detail.
- On-device training is always desirable.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `model_state_size`
- `activation_size`
- `parallelism_strategy`
- `batch_size`
- `communication_bandwidth`

Needed outputs:
- `memory_fit`
- `parallelism_frontier`
- `training_architecture`
- `new_bottleneck`

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

- Training target versus deployment target is clear.
- Parallelism/adaptation choice uses evidence.
- Communication assumption is stated.
- New bottleneck is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
