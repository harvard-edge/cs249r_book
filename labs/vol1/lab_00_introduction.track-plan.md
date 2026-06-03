# V1-00 Track Plan: The Architect's Portal

## Purpose

This orientation lab teaches the course ritual: choose a canonical system track, make predictions before seeing simulator output, inspect evidence, save decisions to the ledger, and carry that identity forward. It is the only lab whose primary job is to explain the track mechanism itself.

## Shared Pedagogy

- Students learn that the same ML idea means different engineering work in different deployment contexts.
- The lab should make the track choice feel consequential, not decorative.
- The selected track becomes the default for later labs through the Design Ledger.
- Every track should expose hardware, workload, constraints, and one expected bottleneck before any detailed chapter content begins.

## Canonical Tracks

| Track | Category | Hardware source | Primary constraints |
|---|---|---|---|
| iPhone | Mobile ML | `Hardware.Mobile.iPhone15Pro` | Battery, thermal envelope, memory, on-device latency, privacy |
| Oura Ring | TinyML / wearable | `Hardware.Tiny.OuraRing` | SRAM/flash, battery life, sampling cadence, OTA payload size |
| RoboTaxi | Edge AI | `Hardware.Edge.RoboTaxi` | Safety-critical p99 latency, local compute, power, reliability |
| Cloud Fleet | Cloud/Fleet | Cloud fleet profile backed initially by `Hardware.Cloud.H100` | Throughput, p99 latency, cost, utilization, carbon |

## Lab Flow

### Opening - Pick A System Identity

Common pattern:
- Present the four canonical tracks as the only student-facing choices.
- Show that each track maps to a real hardware profile and a default scenario.
- Save `track_id`, `category`, `hardware_ref`, and `scenario_id` to the ledger.

Track realization:
- iPhone: a privacy-preserving mobile app that must run useful inference without heating the device.
- Oura Ring: an always-on wearable sensor that must last for days and accept small OTA updates.
- RoboTaxi: an autonomous vehicle perception loop where tail latency is a safety requirement.
- Cloud Fleet: a production service where scale, cost, utilization, and carbon dominate.

### Part A - Constraint Portrait

Common pattern:
- Compare memory, compute, power, latency, privacy, reliability, and cost headroom.
- Use the same visual grammar for all tracks so students can compare regimes.

Track realization:
- iPhone highlights battery and thermal headroom.
- Oura Ring highlights SRAM/flash and energy budget.
- RoboTaxi highlights p99 latency and reliability.
- Cloud Fleet highlights cost, throughput, and carbon.

### Part B - Same Model, Different World

Common pattern:
- Run one reference model or workload against all tracks.
- Show feasible, marginal, and infeasible outcomes.
- The learning moment is that feasibility is contextual.

Track realization:
- iPhone may fit the model but fail sustained thermal limits.
- Oura Ring likely fails memory or OTA payload size without compression.
- RoboTaxi may fit compute but fail p99 or rare-event guardrails.
- Cloud Fleet fits technically but exposes cost and utilization trade-offs.

### Part C - Engineering Lens

Common pattern:
- Student commits to one track and predicts the first bottleneck they expect to revisit across the course.
- The ledger stores the commitment and the initial bottleneck hypothesis.

Track realization:
- iPhone commitment: defend battery and privacy.
- Oura Ring commitment: defend memory and battery life.
- RoboTaxi commitment: defend p99 latency and reliability.
- Cloud Fleet commitment: defend throughput, cost, and carbon.

## Implementation Requirements

- Add a canonical `TrackProfile` registry before refactoring the notebook.
- The selector should return the canonical profile, not free-form hardware.
- The lab must render hardware facts from MLSysIM and narrative facts from the track profile.
- Reports should include the selected track and a first "why this track changes the answer" statement.

## Ledger And Report

Save:
- `track_id`
- `category`
- `hardware_ref`
- `scenario_id`
- initial bottleneck prediction
- one-sentence rationale for the chosen track

Report target:
- A short orientation memo explaining the selected track and the constraint the student expects to manage throughout Volume 1.

## Detailed Planning Addendum

This addendum upgrades the coverage plan into an implementation-ready plan following the V1-10 pilot format.

### Planning Focus

Primary concept:
- track selection, lab ritual, prediction discipline, ledger identity.

Minimum classroom demo:
- show all four track cards, run the same-model comparison, and save one track to the ledger.

Completion path:
- choose one canonical track, inspect constraint portrait, complete same-model comparison, save bottleneck hypothesis.

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
| iPhone | Recognizes Mobile ML as a privacy/battery/thermal system rather than just a smaller cloud model. |
| Oura Ring | Recognizes TinyML/wearable ML as a memory, battery, sampling, and OTA-constrained system. |
| RoboTaxi | Recognizes Edge AI as local, safety-critical, p99/reliability-constrained inference. |
| Cloud Fleet | Recognizes Cloud/Fleet as a throughput, cost, utilization, carbon, and operations system. |

## Common Misconceptions

- Tracks are cosmetic labels.
- The same model has the same feasibility everywhere.
- Cloud is always the default best answer.
- A student can switch tracks casually without changing later reports.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `hardware_ref`
- `model_ref`
- `constraint_summary`
- `scenario_id`

Needed outputs:
- `selected_track_profile`
- `constraint_portrait`
- `feasibility_matrix`
- `initial_bottleneck_hypothesis`

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

- Track choice is saved and justified.
- Student can explain why the track changes constraints.
- Prediction is made before reveal.
- Ledger fields are complete.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
