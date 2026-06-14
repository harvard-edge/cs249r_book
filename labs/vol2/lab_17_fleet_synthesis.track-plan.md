# V2-17 Track Plan: The Fleet Synthesis

## Chapter Invariant

At fleet scale, the unit of engineering is the coupled fleet rather than an
isolated model, accelerator, scheduler, or policy. A defensible deployment
review follows the binding C3 term across infrastructure, communication,
coordination, serving, operations, and governance until the displaced cost is
visible and explicitly accepted.

This is the Volume II capstone. The selected track changes persona, thresholds,
evidence emphasis, natural failure mode, and report frame, but every track uses
the same concept sequence.

## Reading Map

| Lab module | Conclusion anchor | Chapter claim used in the lab |
|---|---|---|
| Opening | Purpose; Synthesizing Distributed ML Systems | The fleet stack is one working discipline, not a checklist of disconnected layers. |
| Part A | Six Principles; Complete Production System | Earlier Volume II decisions form an architecture ledger, and any one layer can bind the system. |
| Part B | Complete Production System; Closing diagnostic | Capacity, communication, reliability, security/privacy, robustness, carbon, and governance must pass together. |
| Part C | Fallacies and Pitfalls | Optimizing one layer often displaces overhead into another layer; revision must name what is relaxed, monitored, deferred, or redesigned. |
| Part D | Competencies Mastered; The Fleet Stack as Discipline | A final deployment review must choose a plan, reject an alternative, cite validation evidence, and name residual risks. |
| Synthesis | Systems that scale, endure, and serve | The Volume II memo closes the selected track narrative by following the active constraint across the fleet. |

## Concept Inventory

### Accepted Concepts

| Concept | Reason for inclusion | Module |
|---|---|---|
| Architecture ledger replay | Makes the capstone cumulative; students turn earlier labs into a fleet architecture rather than restating them. | Part A |
| Binding amount diagnosis | Uses the chapter's C3/fleet-stack diagnostic to identify what currently rules the design. | Part A |
| Multi-constraint launch review | Forces simultaneous reasoning across technical, operational, and governance guardrails. | Part B |
| Trade-off revision under conflict | Makes students choose how to handle guardrail conflict instead of averaging it away. | Part C |
| Deployment review board packet | Produces the final defensible Volume II artifact with selected plan, rejected alternative, validation, and risk. | Part D |
| Track narrative closure | Connects the selected track's Volume II journey to the final memo. | Synthesis |

### Rejected Or Deferred Concepts

| Concept | Disposition |
|---|---|
| Re-teaching every Volume II lab in detail | Rejected; the capstone references prior evidence and focuses on synthesis. |
| Track-specific concept paths | Rejected by design rule; tracks instantiate the same concepts with different constraints and evidence emphasis. |
| A free-form essay as the main activity | Rejected; structured predictions, controls, evidence, failure states, and checkpoints drive the memo. |
| Full production-grade fleet simulator | Deferred; notebook-local teaching models expose the coupled constraints while source traces name assumptions. |
| Modifying shared MLSysIM or lab helpers | Rejected for this wave; new support remains notebook-local with `v2_17_` prefixes. |

## Shared Concept Modules

### Part A - Concept Module: Assemble The Fleet Architecture Ledger

Chapter claim:
- A production fleet is assembled from coupled infrastructure, protocol,
  serving, operations, and governance choices. Any one layer can bind the whole
  system.

Student prior:
- The capstone is a list of previous answers, or the final plan is determined
  by whichever subsystem seems most important.

Storyline:
1. Scenario: the selected track stakeholder asks for a final deployment review
   using the student's Volume II ledger plus canonical track presets for gaps.
2. Prediction: student predicts which fleet amount will bind the design.
3. Manipulation: student sets the evidence floor and the weight assigned to
   preset evidence when earlier ledger entries are missing.
4. Evidence: a ledger replay table and binding-score chart show each Volume II
   layer, source, confidence, track realization, and constraint amount.
5. Consequence: if ledger coverage falls below the evidence floor, the notebook
   creates an evidence-debt boundary that must be reported.
6. Math Peek/source model: binding score = track pressure + ledger evidence +
   preset evidence + evidence-debt penalty, tied to the chapter's C3 diagnostic.
7. Checkpoint: student records whether the architecture review should attack
   the binding amount, fill ledger gaps, accept track presets, or use a
   model-only shortcut.

Ledger output:
- `part_a_binding_prediction`
- `evidence_floor_pct`
- `preset_weight_pct`
- `ledger_coverage_pct`
- `binding_fleet_amount`
- `part_a_checkpoint`

### Part B - Concept Module: Run A Multi-Constraint Fleet Review

Chapter claim:
- A technically feasible component is not a deployable fleet unless capacity,
  communication, reliability, privacy/security, robustness, carbon, and
  governance guardrails pass together.

Student prior:
- If the main capacity or latency metric passes, the fleet is launchable.

Storyline:
1. Scenario: the review board asks the student to stress a candidate
   architecture under growth, fanout, failures, security depth, carbon cap, and
   governance depth.
2. Prediction: student predicts which guardrail will fail first.
3. Manipulation: student selects a candidate plan and changes demand,
   communication fanout, failure rate, privacy/security depth, carbon cap, and
   governance depth.
4. Evidence: a guardrail-ratio chart and exact table show seven constraints
   against launch thresholds.
5. Consequence: a reversible failure appears when any ratio exceeds 1.0, naming
   the value, limit, binding guardrail, and mitigation.
6. Math Peek/source model: launchable iff every guardrail predicate is true;
   the fleet law and responsible-fleet constraints explain why the predicates
   are conjunctive rather than averaged.
7. Checkpoint: student records whether to approve, revise, narrow scope, or
   reject the candidate plan.

Ledger output:
- `part_b_guardrail_prediction`
- `candidate_plan`
- `demand_multiplier`
- `communication_fanout`
- `failure_multiplier`
- `privacy_security_depth`
- `carbon_cap_pct`
- `governance_depth`
- `binding_guardrail`
- `part_b_checkpoint`

### Part C - Concept Module: Revise When Guardrails Conflict

Chapter claim:
- The displacement of overhead means trade-offs cannot be eliminated; they can
  only be relocated, monitored, deferred, relaxed, or redesigned deliberately.

Student prior:
- A guardrail conflict can be solved by picking the cheapest local patch or by
  averaging the constraints into one score.

Storyline:
1. Scenario: the board refuses a simple pass/fail answer and demands a revision
   policy for the active guardrail conflict.
2. Prediction: student predicts which revision mode is defensible: relax a
   noncritical target, monitor and gate, defer scope, or redesign architecture.
3. Manipulation: student chooses the target guardrail, revision action, and
   revision intensity.
4. Evidence: a before/after chart and table show where risk moved after the
   revision.
5. Consequence: the notebook marks a failed revision when the selected action
   leaves a guardrail over threshold or creates unacceptable residual risk.
6. Math Peek/source model: revised margin = old ratio - relief + displaced
   overhead; source trace ties this to the conclusion's pitfall about hidden
   constraints.
7. Checkpoint: student records the revision rule that will appear in the final
   deployment memo.

Ledger output:
- `part_c_revision_prediction`
- `revision_axis`
- `revision_action`
- `revision_intensity_pct`
- `revision_status`
- `residual_risk_pct`
- `part_c_checkpoint`

### Part D - Concept Module: Defend The Deployment Review Board Decision

Chapter claim:
- Engineering judgment under constraint requires a selected plan, rejected
  alternative, validation evidence, source trace, and named residual risks.

Student prior:
- The final memo should only state the preferred architecture and declare that
  the key metrics pass.

Storyline:
1. Scenario: the deployment review board asks for the final go/no-go packet.
2. Prediction: student predicts the board's minimum evidence bar.
3. Manipulation: student chooses the selected plan, rejected alternative,
   validation package, and top residual risk.
4. Evidence: a board-readiness table scores ledger replay, guardrail review,
   revision status, rejected alternative, validation coverage, source trace,
   and residual risk.
5. Consequence: the report is locked or marked provisional when required
   evidence is missing, the alternative is not rejected, validation does not
   cover the binding guardrail, or the revised design is infeasible.
6. Math Peek/source model: defensible decision = selected plan + rejected
   alternative + validation evidence + residual risk + source trace.
7. Checkpoint: student records the final board decision.

Ledger output:
- `part_d_board_prediction`
- `selected_plan`
- `rejected_alternative`
- `validation_package`
- `top_residual_risk`
- `board_outcome`
- `part_d_checkpoint`

### Synthesis - Final Volume II Fleet Synthesis Memo

Student experience:
1. Replay the binding fleet amount from Part A.
2. Replay the multi-constraint binding guardrail from Part B.
3. Replay the trade-off revision from Part C.
4. Replay the board decision from Part D.
5. Export the final Volume II fleet synthesis memo.
6. Close the selected track narrative by stating what the fleet can ship, what
   was rejected, what must be validated, and what residual risk remains.

Ledger output:
- `volume_ii_fleet_decision`
- `binding_fleet_amount`
- `binding_guardrail`
- `selected_plan`
- `rejected_alternative`
- `validation_package`
- `residual_risk`
- `final_track_memo`

## Track Narratives

The tracks instantiate the same modules with different persona, constraints,
thresholds, evidence emphasis, failure mode, and report framing.

| Track | Persona | Architecture emphasis | Natural failure | Evidence emphasis | Report frame |
|---|---|---|---|---|---|
| iPhone | Mobile product engineer | Device support matrix, local runtime, privacy-safe telemetry, app rollout, model update channel | Thermal/battery regression, unsupported device cohort, privacy-safe evidence gap | Battery/thermal headroom, local latency, privacy evidence, staged rollout | Mobile ML fleet release review |
| Oura Ring | Wearable firmware engineer | Sensor/MCU runtime, firmware/model package, duty cycle, phone/cloud sync, OTA update path | SRAM/flash overflow, duty-cycle miss, OTA payload overrun, health-adjacent false alert | SRAM/flash, battery, sensing quality, OTA validation, privacy | Wearable firmware fleet review |
| RoboTaxi | Autonomous vehicle platform engineer | Sensor stack, local accelerator, fallback path, scenario replay, safety operations, fleet learning gate | Tail-latency breach, rare-event recall miss, invalid fallback, unsafe rollout scope | p99/p999 latency, safety margin, rare-event replay, accountable operations | Safety-critical fleet deployment case |
| Cloud Fleet | Fleet service owner | Accelerator pools, fabric, checkpoint/object storage, training/serving scheduler, observability, governance/carbon controls | Queue/SLA breach, communication bottleneck, failure storm, carbon/cost overrun, audit gap | Throughput, p99 SLA, utilization, cost/request, carbon, security and governance evidence | Distributed ML fleet architecture review |

## Mechanics And Evidence Plan

| Module | Controls | Evidence | Failure or boundary | Why this mechanic is used |
|---|---|---|---|---|
| Part A | Binding prediction radio, evidence-floor slider, preset-weight slider, checkpoint radio | Ledger replay table and binding-score chart | Ledger coverage below evidence floor creates evidence-debt boundary | Converts prior lab decisions into architecture evidence. |
| Part B | Candidate plan dropdown plus demand, fanout, failure, security, carbon, and governance sliders | Guardrail-ratio chart and exact constraint table | Any guardrail ratio over 1.0 creates a launch failure | Makes simultaneous constraints visible and reversible. |
| Part C | Revision prediction radio, target guardrail dropdown, action dropdown, intensity slider, checkpoint radio | Before/after guardrail table and risk displacement chart | Revision fails when risk remains above threshold or residual risk becomes unacceptable | Forces explicit trade-off revision instead of hidden averaging. |
| Part D | Board prediction radio, selected/rejected plan dropdowns, validation package dropdown, risk dropdown, checkpoint radio | Board-readiness table and decision summary | Report locks or becomes provisional when evidence is incomplete | Converts evidence into a deployment review board decision. |
| Synthesis | Student ID and final memo note | Report export panel and ledger snapshot | Incomplete required activities lock the report | Produces the final Volume II artifact. |

Every decision-driving visual has an exact table fallback. Color is not the only
indicator: every failure row includes a status label, value, limit, and
mitigation.

## Source And Ledger Plan

Existing helpers/modules:
- `get_lab_metadata`
- `get_lab_track_variant`
- `get_track_profile`
- `resolve_mlsysim_ref`
- `track_selector`
- `track_context`
- `track_arc_context`
- `source_trace`
- `build_lab_report`
- `report_export_panel`
- `DesignLedger`
- `COLORS`, `LAB_CSS`, `ACADEMIC_LAB_CSS`, `apply_plotly_theme`

Notebook-local helpers:
- `v2_17_track_packet`
- `v2_17_volume_layer_specs`
- `v2_17_ledger_replay`
- `v2_17_constraint_review`
- `v2_17_revision_review`
- `v2_17_board_review`
- `v2_17_score_fig`
- `v2_17_review_fig`
- `v2_17_revision_fig`
- `v2_17_board_fig`
- `v2_17_markdown_table`

Design Ledger fields:
- selected track, scenario, hardware/model refs
- architecture ledger coverage and binding amount
- Part A/B/C/D predictions and checkpoints
- guardrail review controls and binding guardrail
- revision action, target, intensity, status, and residual risk
- selected plan, rejected alternative, validation package, board outcome
- final Volume II fleet synthesis memo

Report fields:
- learning objectives tied to the chapter invariant
- prediction summary for Parts A-D
- knob settings and track-specific assumptions
- evidence summary with binding amount, guardrail, revision, and board outcome
- final decision with selected plan, rejected alternative, validation, and risk
- source trace with book anchor, registry refs, shared helper APIs, and local helper names

## Implementation Risks

| Risk | Mitigation |
|---|---|
| V2-17 shared variant metadata is still generic system-design metadata. | Use variant defaults for track refs, plan labels, thresholds, and validation tests; keep capstone-specific teaching constants notebook-local and source-traced. |
| Ledger history may be incomplete or from a different track. | Replay real entries when available, visibly label other-track entries, and fall back to canonical track presets with lower confidence. |
| Multi-constraint model is a teaching approximation, not a production simulator. | Expose formulas in Math Peek/source trace and make all scenario constants visible in the report snapshot. |
| Adjacent labs are being edited by other workers. | Edit only `lab_17_fleet_synthesis.py` and this track-plan file; do not modify shared helpers/tests. |
| Report save can repeat when Marimo reruns cells. | Match existing labs: save only when required widgets are complete and include an idempotent design snapshot. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A - Architecture ledger | 3 | 3 | 3 | 3 | 3 | 3 | Pass: scenario, prediction, evidence-floor manipulation, ledger table, binding chart, evidence boundary, Math Peek, checkpoint. |
| Part B - Multi-constraint review | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, stress controls, guardrail chart/table, reversible failure, Math Peek, checkpoint. |
| Part C - Trade-off revision | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, target/action/intensity controls, before/after evidence, residual-risk failure, Math Peek, checkpoint. |
| Part D - Deployment board | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, plan/rejection/validation/risk controls, readiness table, report lock/provisional outcome, Math Peek, checkpoint. |
| Synthesis - Final memo | 3 | 3 | 3 | 3 | 3 | 3 | Pass: replays all evidence, closes track narrative, exports report, saves ledger snapshot. |

Acceptance summary:
- Every concept module has at least five student-facing activity beats.
- The lab has reversible failure states in Parts A, B, C, and D.
- Tracks share the same concepts but change persona, constraints, thresholds,
  evidence emphasis, failure mode, and report framing.
- The final artifact is a Volume II fleet synthesis memo with selected plan,
  rejected alternative, validation evidence, and residual risks.
