# V1-16 Track Plan: The Architect's Audit

## Chapter Invariant

Single-machine ML system design requires diagnosing the binding D-A-M/system
amount, optimizing the layer that actually relieves that amount, deploying only
inside the measured operating envelope, and defending the residual risk that
remains.

This is the Volume I capstone. The selected track changes persona, thresholds,
evidence emphasis, failure mode, and report framing, but the Part A/B/C/D
concept sequence is shared across every track.

## Reading Map

| Lab module | Conclusion anchor | Chapter claim used in the lab |
|---|---|---|
| Opening | Purpose; Synthesizing ML Systems | The system is the model; D-A-M decisions propagate across the stack. |
| Part A | Lighthouse models: Constraint propagation; Thirteen Quantitative Invariants | Evidence from earlier chapters identifies the amount that currently binds. |
| Part B | Optimize invariants; Integrated framework; Fallacies and Pitfalls | Local optimization can move complexity downstream rather than remove it. |
| Part C | Deploy invariants; Production reality; Robust AI systems | A design is valid only inside a latency, memory, energy, quality, and evidence envelope. |
| Part D | Journey Forward; Engineering responsibility; Fallacies and Pitfalls | A final design report must include rejected alternatives, source trace, and residual risk. |
| Synthesis | Summary; Horizon note from node to fleet | Volume I's single-machine design handoff names the scale question for Volume II. |

## Concept Inventory

### Accepted Concepts

| Concept | Reason for inclusion | Module |
|---|---|---|
| Binding D-A-M/system amount diagnosis | Makes the capstone more than a summary; students use accumulated evidence to name what is actually scarce. | Part A |
| Right-lever optimization | Connects iron law, Pareto frontier, Amdahl, and conservation of complexity to a concrete design lever. | Part B |
| Operating envelope validation | Forces students to treat deployment as a bounded claim with validation evidence. | Part C |
| Defensible design report | Produces the final Volume I artifact with rejected alternatives, source trace, and residual risk. | Part D |
| Carry-forward to scale | Connects Volume I single-machine reasoning to Volume II resource-boundary changes. | Synthesis |

### Rejected Or Deferred Concepts

| Concept | Disposition |
|---|---|
| Full distributed-systems reliability design | Deferred to Volume II; this lab only names the carry-forward question. |
| Re-teaching all thirteen invariants separately | Rejected; the lab uses selected invariants as source models inside concept modules. |
| Track-specific concept paths | Rejected by design rule; tracks instantiate the same concepts with different envelopes. |
| Free-form architecture essay as the main activity | Rejected; structured predictions, controls, evidence, and checkpoints drive the report. |
| Adding shared MLSysIM or lab-helper APIs | Rejected for Wave 4 ownership; new support remains notebook-local with `v1_16_` prefixes. |

## Shared Concept Modules

### Part A - Concept Module: Diagnose The Binding Amount

Chapter claim:
- The system is the model, and constraint propagation across data, algorithms,
  machines, and operations determines what can be deployed.

Student prior:
- The capstone should summarize prior answers, or the model choice alone is the
  architecture.

Storyline:
1. Scenario: the track stakeholder asks for a final ship/no-ship architecture
   audit using the student's ledger plus labeled track presets.
2. Prediction: student predicts whether Data, Algorithm, Machine, or System
   amount is binding.
3. Manipulation: student raises or lowers the evidence floor that controls how
   strongly missing ledger coverage counts as system evidence debt.
4. Evidence: a bar chart and table score all four amounts using track pressure,
   ledger constraint hits, and preset evidence debt.
5. Consequence: if ledger coverage is below the selected floor, the notebook
   marks an evidence-gap boundary that must appear in the report.
6. Math Peek/source model: binding score = track pressure + ledger hits +
   evidence-gap penalty; source trace points to D-A-M, Lighthouse propagation,
   and the local `v1_16_amount_scores` helper.
7. Checkpoint: student records whether the first action should attack the
   binding amount, patch the easiest subsystem, ignore presets, or choose a
   model-only fix.

Ledger output:
- `binding_amount_prediction`
- `evidence_floor_pct`
- `binding_amount_actual`
- `binding_amount_score_pct`
- `part_a_checkpoint`

### Part B - Concept Module: Optimize The Right Lever

Chapter claim:
- Complexity is conserved. Pulling the wrong lever creates downstream debt even
  when a local metric improves.

Student prior:
- Any local improvement is progress, especially if it improves a visible model
  or hardware metric.

Storyline:
1. Scenario: the stakeholder proposes one optimization lever before the report
   is finalized.
2. Prediction: student predicts that the defensible lever must relieve the
   binding amount and check downstream debt.
3. Manipulation: student chooses a track-specific lever and varies intervention
   intensity.
4. Evidence: a relief-vs-debt chart and table show binding relief, downstream
   debt, debt boundary, and status.
5. Consequence: a reversible failure appears when debt exceeds the track
   boundary or the lever targets the wrong amount.
6. Math Peek/source model: net margin = binding relief - downstream debt,
   connected to Pareto frontier, Amdahl's Law, and conservation of complexity.
7. Checkpoint: student records the lever policy that goes into the design memo.

Ledger output:
- `optimization_prediction`
- `optimization_lever`
- `optimization_intensity_pct`
- `optimization_status`
- `downstream_debt_pct`
- `part_b_checkpoint`

### Part C - Concept Module: Deploy Inside The Operating Envelope

Chapter claim:
- Deployment claims are statistical and operational: validation evidence,
  latency budgets, drift/skew, and guardrails define the valid envelope.

Student prior:
- A design that works at one nominal setting can be deployed broadly.

Storyline:
1. Scenario: the stakeholder asks whether the chosen design can ship under the
   track's real operating constraints.
2. Prediction: student predicts whether fragility is universal or
   track-specific.
3. Manipulation: student perturbs workload, model growth, guardrail tightening,
   and evidence confidence.
4. Evidence: a sensitivity chart and exact table show value, limit, risk,
   status, and mitigation for each axis.
5. Consequence: the notebook marks an envelope failure when any axis crosses its
   track-specific limit.
6. Math Peek/source model: deployable iff every stress axis passes and
   validation confidence is above the floor; source trace points to the
   existing `sensitivity_audit` helper and conclusion deployment invariants.
7. Checkpoint: student records whether to ship, revise, add validation, or
   reject the release.

Ledger output:
- `sensitivity_prediction`
- `workload_multiplier`
- `model_growth_pct`
- `guardrail_tightening_pct`
- `evidence_confidence_pct`
- `operating_envelope_status`
- `most_fragile_axis`
- `part_c_checkpoint`

### Part D - Concept Module: Defend The Complete Design Report

Chapter claim:
- A systems design is not defensible unless it names the decision, rejected
  alternatives, evidence source trace, validation plan, and residual risk.

Student prior:
- The final memo should simply state the preferred design and declare success.

Storyline:
1. Scenario: the architecture review board asks for the final Volume I memo.
2. Prediction: student predicts that the memo must revise a decision, reject an
   alternative, and name residual risk.
3. Manipulation: student chooses the revised decision, rejected alternative,
   top residual risk, and mitigation evidence.
4. Evidence: a completeness table scores decision, rejected alternative, source
   trace, ledger replay, operating envelope, validation, and residual risk.
5. Consequence: the report fails if the rejected alternative is missing, the
   residual risk is absent, or the operating envelope is invalid.
6. Math Peek/source model: defensible report = decision + rejected alternative +
   source trace + residual risk + validation evidence.
7. Checkpoint: student records whether the report is defensible, needs more
   validation, needs a different lever, or should not ship.

Ledger output:
- `memo_prediction`
- `revised_decision`
- `rejected_alternative`
- `top_residual_risk`
- `mitigation_evidence`
- `report_completeness_pct`
- `part_d_checkpoint`

### Synthesis - Volume I Final Report And Volume II Handoff

Student experience:
1. Replay the binding amount from Part A.
2. Replay the optimization lever and downstream debt from Part B.
3. Replay the operating envelope and most fragile axis from Part C.
4. Replay the report defense from Part D.
5. Export the final Volume I architecture memo.
6. Name the carry-forward question for Volume II: what changes when the binding
   quantity is no longer bounded by one local machine?

Ledger output:
- `volume_i_final_decision`
- `volume_i_residual_risk`
- `volume_ii_carry_forward_question`

## Track Narratives

The tracks instantiate the same modules with different persona, constraints,
thresholds, evidence emphasis, failure mode, and report framing.

| Track | Persona | Amount emphasis | Operating envelope | Natural failure | Report frame |
|---|---|---|---|---|---|
| iPhone | Mobile systems architect shipping a local feature | Machine/System amounts: battery, thermal, NPU coverage, privacy-safe evidence, UX p99 | Battery, thermal, privacy, accessibility, p99 interaction latency | Thermal throttle, CPU/GPU fallback, privacy-safe telemetry gaps, accessibility regression | Local feature design memo |
| Oura Ring | Wearable firmware/TinyML architect | Machine amount dominates: SRAM, flash, duty cycle, radio wake, firmware growth | SRAM/flash fit, OTA payload, battery regression, sensor-contact cohorts, privacy | SRAM/flash overflow, duty-cycle miss, OTA payload overrun, false alert drift | Firmware/TinyML design memo |
| RoboTaxi | Safety/perception architect | System/Data amounts: rare-event replay, p99/p999 latency, fallback validation, safety trace | Rare-event recall, p99/p999 deadline, power margin, fallback reliability | Safety margin miss, rare-event recall regression, invalid fallback, tail-latency breach | Safety/perception case |
| Cloud Fleet | Platform/SRE architect | System/Machine amounts: SLA, utilization, KV cache, cost/request, carbon, monitoring | p99 SLA, cost/request, utilization headroom, quality canary, carbon budget | Queue/SLA breach, negative economics, carbon/cost overrun, silent quality drift | Production service design memo |

## Mechanics And Evidence Plan

| Module | Controls | Evidence | Failure or boundary | Why this mechanic is used |
|---|---|---|---|---|
| Part A | Binding prediction radio, evidence-floor slider, checkpoint radio | Binding amount bar chart and ledger evidence table | Ledger coverage below evidence floor creates evidence-gap boundary | Diagnoses the bottleneck from accumulated evidence. |
| Part B | Optimization prediction radio, lever dropdown, intensity slider, checkpoint radio | Relief-vs-debt chart and lever audit table | Downstream debt exceeds boundary or lever misses binding amount | Shows local gains can create system debt. |
| Part C | Sensitivity prediction radio, four stress sliders, checkpoint radio | Risk bar chart and exact sensitivity table | Any axis exceeds limit or evidence confidence falls below floor | Forces boundary finding inside the operating envelope. |
| Part D | Memo prediction radio, revision/rejection/risk/mitigation dropdowns, checkpoint radio | Report completeness table and memo summary | Missing rejected alternative, source trace, residual risk, or invalid envelope | Converts observations into a defensible design report. |
| Synthesis | No new controls; replays prior module outputs | Final report card and export panel | Incomplete predictions/checkpoints prevent completed ledger status | Produces carry-forward evidence. |

Every decision-driving visual has a table fallback. Color is never the only
status indicator: every pass/fail row includes label, value, limit, unit or
evidence, and mitigation.

## Source And Ledger Plan

Existing helpers/modules:
- `capstone_track_profile`
- `replay_ledger`
- `sensitivity_audit`
- `architecture_memo`
- `build_lab_report`
- `report_export_panel`
- `source_trace`
- `track_selector`

Notebook-local helpers:
- `v1_16_track_lens`
- `v1_16_amount_scores`
- `v1_16_binding_amount`
- `v1_16_lever_catalog`
- `v1_16_lever_options`
- `v1_16_default_lever`
- `v1_16_lever_audit`
- `v1_16_report_audit`
- `v1_16_volume_ii_question`

Design Ledger fields:
- selected track, scenario, hardware/model refs
- binding amount prediction, actual binding amount, evidence floor, score
- optimization lever, intensity, downstream debt, optimization status
- operating envelope values, most fragile axis, feasibility status
- revised decision, rejected alternative, residual risk, mitigation evidence
- Part A/B/C/D checkpoints
- Volume II carry-forward question

Report fields:
- learning objectives tied to concept modules
- predictions and checkpoints
- knob settings and selected alternatives
- evidence summary from ledger replay, amount diagnosis, lever audit,
  sensitivity audit, report audit, and memo
- final decision and residual risk
- source trace with helper names and registry refs

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Shared helper changes would affect other workers | Do not edit shared helpers; keep support local to the notebook. |
| Capstone becomes track-specific concept branching | Keep a single A/B/C/D sequence and make only persona, thresholds, levers, and evidence track-specific. |
| Ledger gaps make student evidence sparse | Label presets, expose coverage, and treat missing coverage as report evidence debt. |
| Too many controls dilute the concept modules | Keep each module to one prediction, one primary manipulation set, and one checkpoint. |
| Report export omits new evidence | Update the report snapshot, evidence summary, and incomplete-field checks. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, manipulation, binding evidence, boundary, Math Peek, checkpoint. |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass: lever choice, intensity manipulation, debt failure, Math Peek, checkpoint. |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Pass: four stress controls, envelope boundary, source-backed sensitivity helper, checkpoint. |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Pass: report decision controls, completeness evidence, failure boundary, source trace, checkpoint. |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 3 | Pass: replays evidence and names Volume II handoff. |

Minimum acceptance:
- No dimension below 2.
- At least three dimensions at 3 in every module.
- Reversible failure states exist in Parts A, B, C, and D.
- Synthesis ties the final report to the chapter invariant and Volume II scale.
