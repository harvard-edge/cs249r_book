# V2-10 Track Plan: The Inference Economy

## Purpose

This lab teaches inference cost inversion, stateful serving, KV/cache memory, continuous batching, replicas, routing, capacity, and cost curves.

## Shared Pedagogy

- Students predict when inference dominates lifetime cost.
- They sweep demand, sequence/state, batching, cache, and replicas.
- They choose a serving fleet or local serving architecture.

## Lab Flow

### Opening - Inference Economy Brief

Common narrative:
- Training is visible, but inference can dominate the long-run system cost and constraints.

Track realization:
- iPhone: inference cost is battery, heat, responsiveness, and privacy.
- Oura Ring: inference cost is battery life, duty cycle, memory, and OTA maintenance.
- RoboTaxi: inference cost is safety-critical latency, power, and fleet maintenance.
- Cloud Fleet: inference cost is accelerators, utilization, memory state, SLO, and demand.

### Part A - Cost Inversion

Common pattern:
- Cumulative training/build cost versus inference/operation cost curve.

Track realization:
- iPhone shows per-user battery/thermal cost over lifetime.
- Oura Ring shows always-on duty-cycle energy over days/months.
- RoboTaxi shows per-mile/per-hour compute and maintenance exposure.
- Cloud Fleet shows per-request cost at scale.

### Part B - State And Batching Frontier

Common pattern:
- Sweep sequence/state, concurrency, cache, batch policy, replicas, or routing.

Track realization:
- iPhone balances responsiveness and batching/fallback.
- Oura Ring balances duty cycle and deferred inference.
- RoboTaxi balances sensor pipeline state and hard latency.
- Cloud Fleet balances KV cache, continuous batching, and routing.

### Part C - Serving Fleet

Common pattern:
- Student records local/edge/cloud/fleet serving architecture.

Track realization:
- iPhone decision defends on-device serving versus selective offload.
- Oura Ring decision defends tiny local inference and phone/cloud handoff.
- RoboTaxi decision defends local real-time serving and fleet update path.
- Cloud Fleet decision defends batching/cache/capacity policy.

## Implementation Requirements

- Track variants define "cost" in track-appropriate units.
- Cloud Fleet needs capacity/cost model; device tracks need energy and latency models.
- Report must include both primary cost and guardrail metric.

## Ledger And Report

Save:
- predicted cost inversion point
- serving policy
- state/cache/batching setting
- primary cost metric
- guardrail metric

Report target:
- An inference capacity or local-serving plan for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- inference cost inversion, state, batching, replicas, routing, and serving fleet.

Minimum classroom demo:
- show cumulative inference cost overtaking setup/training cost for Cloud Fleet and battery cost for iPhone.

Completion path:
- predict cost inversion, inspect state/batching frontier, choose serving architecture.

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
| iPhone | Balances local inference battery/latency/privacy against selective offload. |
| Oura Ring | Balances tiny local inference, duty cycle, and phone/cloud handoff. |
| RoboTaxi | Keeps real-time safety inference local while fleet learning/offline work moves elsewhere. |
| Cloud Fleet | Uses batching/cache/replicas/routing to balance cost/request and SLA. |

## Common Misconceptions

- Training cost dominates lifetime cost.
- Batching is always good.
- State/cache memory is secondary.
- Local and cloud inference have the same economics.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `demand`
- `state_size`
- `batch_policy`
- `replicas`
- `routing_policy`
- `cost_model`

Needed outputs:
- `cost_inversion_curve`
- `state_frontier`
- `capacity_policy`
- `serving_architecture`

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

- Primary cost unit is track-specific.
- Guardrail metric is preserved.
- Serving architecture uses evidence.
- Lifetime cost or energy is considered.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
