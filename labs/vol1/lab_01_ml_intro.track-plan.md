# V1-01 Track Plan: The AI Triad

## Purpose

This lab teaches Data, Algorithm, and Machine coupling. The same model quality problem can be caused by data coverage, architecture choice, or deployment hardware, and the correct intervention depends on the selected track.

## Shared Pedagogy

- Students diagnose which part of the D-A-M triad is actually binding.
- They predict whether data, model, or machine investment gives the best first improvement.
- They compare interventions under a fixed engineering budget.
- They save a defensible first fix, including why the other two fixes are weaker.

## Lab Flow

### Opening - Track-Specific System Brief

Common narrative:
- The team has a model that works in a demo but fails in the target deployment.
- The student must identify whether the failure is mainly data, algorithm, or machine.

Track realization:
- iPhone: demo model works plugged in but drains battery and heats the phone in normal use.
- Oura Ring: model looks accurate offline but cannot fit or run at the needed sampling cadence.
- RoboTaxi: perception model has strong average accuracy but weak rare-hazard behavior under latency pressure.
- Cloud Fleet: model quality is acceptable but serving cost and utilization make the deployment unsustainable.

### Part A - Diagnose Data, Algorithm, Machine

Common pattern:
- Sliders alter data quality, model size/architecture, and hardware budget.
- The simulator reports active bottleneck and whether the current configuration is feasible.

Track realization:
- iPhone emphasizes thermal throttling and NPU use.
- Oura Ring emphasizes memory footprint and sensing cadence.
- RoboTaxi emphasizes p99 latency and rare-event recall.
- Cloud Fleet emphasizes cost per request and accelerator utilization.

### Part B - Intervention Frontier

Common pattern:
- Students allocate a fixed budget across data collection, model change, and hardware upgrade.
- The plot shows accuracy, latency, cost, and feasibility movement.

Track realization:
- iPhone compares better on-device data, smaller model, and NPU-aware runtime.
- Oura Ring compares better signal processing, compression, and MCU/flash assumptions.
- RoboTaxi compares rare-event data, model specialization, and edge accelerator headroom.
- Cloud Fleet compares data quality, batching/runtime work, and more accelerators.

### Part C - Defensible Fix

Common pattern:
- Student chooses one first intervention and records evidence.
- The decision card must state what would invalidate the choice.

Track realization:
- iPhone decision should defend user experience under battery and thermal constraints.
- Oura Ring decision should defend days-long operation and OTA feasibility.
- RoboTaxi decision should defend safety-critical tail behavior.
- Cloud Fleet decision should defend production economics and SLA compliance.

## Implementation Requirements

- Replace generic deployment labels with canonical track profiles.
- The D-A-M result object should include `binding_axis`, `primary_metric`, and `guardrail_metric`.
- Track variants should provide stakeholder text, default workload, and acceptable thresholds.

## Ledger And Report

Save:
- selected track profile
- predicted binding D-A-M axis
- chosen intervention
- result snapshot after intervention
- rejected alternatives and residual risk

Report target:
- A one-page triad diagnosis memo explaining the first fix the student would defend.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- Data, Algorithm, Machine coupling and first-intervention choice.

Minimum classroom demo:
- compare the D-A-M diagnosis for Oura Ring and Cloud Fleet, then show how the best intervention changes.

Completion path:
- make a D-A-M bottleneck prediction, run the intervention frontier, choose one intervention, record rejected alternatives.

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
| iPhone | Likely defends a model/runtime or data-quality intervention that protects battery, privacy, and UX. |
| Oura Ring | Likely discovers that machine limits force a smaller model or better signal preprocessing before more data helps. |
| RoboTaxi | Likely prioritizes rare-event data and latency-aware architecture rather than generic accuracy improvement. |
| Cloud Fleet | Likely balances data/model improvements against serving cost and utilization. |

## Common Misconceptions

- Accuracy failure is always a model problem.
- Hardware upgrades fix data coverage.
- More data always beats better deployment design.
- The triad axes can be optimized independently.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `hardware_ref`
- `data_quality`
- `model_scale`
- `hardware_budget`
- `intervention_budget`

Needed outputs:
- `binding_axis`
- `intervention_frontier`
- `selected_intervention`
- `rejected_alternatives`

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

- Diagnosis names the binding D-A-M axis.
- Intervention is tied to evidence.
- Rejected alternatives are explained.
- Residual risk is track-specific.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
