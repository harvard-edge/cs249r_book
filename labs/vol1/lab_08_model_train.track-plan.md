# V1-08 Track Plan: Training Gauntlet

## Purpose

This lab teaches training memory and throughput: weights, gradients, optimizer state, activations, data pipeline, mixed precision, checkpointing, and accumulation.

## Shared Pedagogy

- Students predict which memory component dominates training.
- They sweep feasibility knobs to trade speed, memory, and convergence risk.
- They choose a training, fine-tuning, or adaptation plan appropriate to the selected track.

## Lab Flow

### Opening - Training Feasibility Brief

Common narrative:
- The question is not only "can we train?" but where and how training or adaptation should happen.
- Some tracks should discover that full training is the wrong activity on the device.

Track realization:
- iPhone: on-device full training is usually infeasible; personalization/adaptation may be possible.
- Oura Ring: device training is usually infeasible; tiny calibration or threshold updates may be possible.
- RoboTaxi: edge retraining is limited; fleet data collection plus controlled retraining is likely.
- Cloud Fleet: large-scale training and fine-tuning are central but resource-bound.

### Part A - Training Memory Budget

Common pattern:
- Show memory stack for weights, activations, gradients, optimizer, and data.
- Identify infeasible pieces.

Track realization:
- iPhone highlights adaptation memory versus app/runtime footprint.
- Oura Ring highlights why training state cannot fit.
- RoboTaxi highlights local adaptation limits and safety validation cost.
- Cloud Fleet highlights optimizer state, activation checkpointing, and accelerator memory.

### Part B - Feasibility Knobs

Common pattern:
- Sweep batch, precision, checkpointing, accumulation, optimizer, and frozen layers.
- Plot memory versus throughput/convergence risk.

Track realization:
- iPhone uses LoRA, frozen backbone, or small-head personalization.
- Oura Ring uses threshold calibration or cloud-trained model updates.
- RoboTaxi uses fleet-curated retraining with edge validation.
- Cloud Fleet uses mixed precision, checkpointing, and distributed strategy.

### Part C - Training Plan

Common pattern:
- Student chooses where training happens and what risk remains.

Track realization:
- iPhone plan defends on-device adaptation only if battery/privacy constraints hold.
- Oura Ring plan defends cloud/offline training plus tiny OTA update.
- RoboTaxi plan defends centralized retraining with local safety regression tests.
- Cloud Fleet plan defends full training/fine-tuning capacity and hidden communication cost.

## Implementation Requirements

- Track variants must distinguish training, fine-tuning, adaptation, and calibration.
- Device hardware should not imply training support automatically.
- Result schemas should include infeasible components and recommended mitigation.

## Ledger And Report

Save:
- predicted dominant training memory component
- actual memory stack
- selected training/adaptation plan
- hidden cost or convergence risk
- where validation happens

Report target:
- A training feasibility plan for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- training memory, throughput, adaptation, and feasibility.

Minimum classroom demo:
- show why full training is infeasible on Oura Ring while Cloud Fleet uses sharding/checkpointing.

Completion path:
- predict dominant training memory component, sweep feasibility knobs, choose training/adaptation plan.

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
| iPhone | Likely selects on-device adaptation or personalization, not full training. |
| Oura Ring | Likely selects off-device training plus tiny calibration or OTA update. |
| RoboTaxi | Likely selects centralized retraining from fleet data plus local validation. |
| Cloud Fleet | Likely selects mixed precision/checkpointing/distributed training plan. |

## Common Misconceptions

- If inference fits, training fits.
- Fine-tuning is always cheap.
- Checkpointing is free.
- All tracks should train locally.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `model_ref`
- `batch_size`
- `precision`
- `checkpointing`
- `optimizer`
- `adaptation_strategy`

Needed outputs:
- `training_memory_stack`
- `feasibility_frontier`
- `training_plan`
- `hidden_cost`

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

- Training location is justified.
- Memory stack evidence is used.
- Adaptation versus full training is distinguished.
- Convergence/validation risk is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
