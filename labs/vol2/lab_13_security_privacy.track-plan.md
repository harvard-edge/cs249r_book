# V2-13 Track Plan: The Price of Privacy

## Purpose

This lab teaches threat modeling, privacy budget, security controls, defense overhead, quality/latency/cost trade-offs, and residual risk.

## Shared Pedagogy

- Students predict whether adding privacy/security controls is free.
- They choose a threat model and add defenses.
- They choose a security/privacy policy with residual attack or leakage risk.

## Lab Flow

### Opening - Threat And Privacy Brief

Common narrative:
- Privacy and security protections change the system's performance and failure modes.

Track realization:
- iPhone: local data, permissions, on-device processing, and telemetry minimization dominate.
- Oura Ring: health-adjacent biosignals, BLE sync, and cloud handoff dominate.
- RoboTaxi: safety data, sensor logs, fleet upload, and adversarial inputs dominate.
- Cloud Fleet: multi-tenant isolation, logging, DP, encryption, abuse, and guardrails dominate.

### Part A - Threat/Privacy Budget

Common pattern:
- Student selects threat model, privacy budget, logging, and sensitive data path.

Track realization:
- iPhone maps user data and local processing.
- Oura Ring maps biometric signal and sync path.
- RoboTaxi maps sensor data, location, and attack surface.
- Cloud Fleet maps tenant data, prompts/logs, models, and infrastructure.

### Part B - Defense Overhead Frontier

Common pattern:
- Add DP, encryption, secure aggregation, filtering, guardrails, isolation, or logging controls.
- Plot protection versus quality, latency, energy, or cost.

Track realization:
- iPhone overhead appears in latency, battery, and telemetry utility.
- Oura Ring overhead appears in energy, payload size, and sync time.
- RoboTaxi overhead appears in latency and validation complexity.
- Cloud Fleet overhead appears in throughput, cost, and utility.

### Part C - Security/Privacy Policy

Common pattern:
- Student records controls, cost, and residual risk.

Track realization:
- iPhone policy defends local-first privacy.
- Oura Ring policy defends health-data minimization and secure sync.
- RoboTaxi policy defends safety logs and adversarial resilience.
- Cloud Fleet policy defends tenant isolation and governance.

## Implementation Requirements

- Add explicit threat model selector.
- Track variants define sensitive data, attacker, and defended asset.
- Defense overhead should use track-relevant cost units.

## Ledger And Report

Save:
- threat model
- selected controls
- privacy/security strength
- overhead
- residual attack/leakage risk

Report target:
- A security and privacy policy memo for the selected track.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- threat model, privacy budget, defense overhead, and residual risk.

Minimum classroom demo:
- add defenses and show overhead for Oura Ring health data and Cloud Fleet tenant isolation.

Completion path:
- choose threat model, build defense stack, inspect overhead frontier, choose policy.

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
| iPhone | Uses local-first processing and telemetry minimization with battery/latency costs visible. |
| Oura Ring | Protects biosignal data and sync path with energy/payload overhead visible. |
| RoboTaxi | Protects sensor/location/safety data and adversarial-input path with latency risk visible. |
| Cloud Fleet | Protects tenant data, logs, prompts, model access, and infrastructure with throughput/cost overhead. |

## Common Misconceptions

- Privacy/security controls are free.
- Encryption solves every privacy problem.
- Threat model can be generic.
- Logging more always improves safety.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `threat_model`
- `privacy_budget`
- `defense_controls`
- `logging_policy`
- `sensitive_data_path`

Needed outputs:
- `threat_map`
- `defense_frontier`
- `overhead_stack`
- `security_privacy_policy`

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

- Threat model is track-specific.
- Controls map to assets/risks.
- Overhead is quantified.
- Residual attack/leakage risk is named.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
