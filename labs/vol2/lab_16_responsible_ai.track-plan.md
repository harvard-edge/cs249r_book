# V2-16 Track Plan: Responsible Fleet Governance

## Chapter Invariant

Responsible AI at fleet scale is an amount system. Stakeholder harms become measurable obligations, fairness and accountability evidence consume capacity, explanation and governance add latency/cost/storage, audit coverage determines what harms can be seen, and residual obligation remains even after a policy passes technical gates.

## Reading Map

Primary source:
- `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd`

Supporting framing:
- `book/quarto/contents/vol2/parts/responsible_fleet_principles.qmd`

Chapter anchors used by the lab:
- Purpose and Governance Imperative: responsibility is a deployment gate, not a post-hoc checklist.
- Core Principles and ML Lifecycle: fairness, explainability, transparency, privacy, accountability, and safety become control-plane invariants.
- Quantitative Fairness Measurement and Threshold Trade-offs: fairness metrics conflict, thresholds encode values, and statistical evidence matters.
- Responsible AI Across Deployment Environments: deployment context changes what can be measured, explained, audited, and governed.
- Computational Overhead of Responsible AI Techniques: fairness monitoring, DP, SHAP, red-team review, and governance add measurable system cost.
- Bias Detection and Fairness Monitoring: production monitoring requires subgroup counters, labels, alert thresholds, and feedback loops.
- Sociotechnical Dynamics: deployed models reshape future data; static validation cannot bound feedback-loop harm.
- Transparency, Contestability, and Institutional Embedding: explanations and dashboards only matter when paired with recourse, owner, escalation, and remediation authority.
- Fallacies and Pitfalls: one fairness metric is insufficient; monitoring without ownership is only a log.

## Concept Inventory

Accepted concepts:
- Stakeholder harm must be translated into a measurable obligation before a fairness metric is meaningful.
- Fairness, accountability, explanation, and monitoring controls reduce some harms while adding overhead to latency, cost, energy, evidence delay, or release cadence.
- Audit coverage is a resource allocation problem: sampling, label availability, explanation coverage, and review capacity determine blind spots.
- Residual harm remains after mitigation; responsible policies must bind it to an owner, escalation path, and future monitoring obligation.
- Policy choice is a design decision with rejected alternatives, not a global optimum.

Rejected or compressed concepts:
- Full legal survey by jurisdiction: too broad for a lab; represented only as governance guardrails and documentation obligations.
- Detailed SHAP/LIME derivations: covered by explanation overhead amounts rather than algorithm mechanics.
- Generative-era RLHF internals: important chapter content, but the lab focuses on fleet-scale governance amounts that apply across tracks.
- Comprehensive privacy/unlearning mechanics: already emphasized in V2-13; here privacy appears as an audit-visibility and data-minimization constraint.

## Track Plan

Tracks use one shared concept sequence. They do not introduce different concepts; they change stakeholder, constraints, threshold values, evidence emphasis, failure mode, and report framing.

| Track | Persona | Stakeholder Harm | Binding Amount | Natural Failure | Report Framing |
|---|---|---|---|---|---|
| iPhone | Mobile responsibility lead | Accessibility/context cohorts receive poorer local outcomes or no contestable feedback. | Local latency, privacy-safe sample coverage, battery/thermal headroom, explanation coverage. | Privacy-safe telemetry misses small cohorts or online explanations exceed the mobile UX budget. | Mobile responsible release memo with privacy-safe cohort evidence and local recourse. |
| Oura Ring | Wearable firmware responsibility lead | Wearers receive false alarms, missed health-adjacent signals, or unclear risk communication. | Duty-cycle energy, false-alert coverage, sensor-contact cohorts, tiny storage/OTA headroom. | Battery-safe monitoring under-samples rare physiology and cannot support enough audit evidence. | Wearable health-adjacent governance memo with conservative residual obligation. |
| RoboTaxi | Autonomous fleet safety accountability lead | Road users face rare-event perception failures and opaque safety decisions. | Rare-event replay coverage, p99 explanation/replay overhead, escalation latency, safety evidence floor. | Replay coverage leaves blind spots or escalation cannot execute inside the safety-review window. | Safety-case governance memo with public accountability, fallback, and residual long-tail risk. |
| Cloud Fleet | Responsible AI platform owner | Underserved tenants, languages, or regions are hidden by aggregate service metrics. | Cohort coverage, audit/storage cost, p99/SLA overhead, review capacity, appeal latency. | Aggregate monitoring passes while intersectional cohorts or delayed labels remain uncovered. | Population-scale responsible AI memo with tenant/language audit and V2-17 guardrail. |

## Concept Modules

### Part A: Concept Module - Harm Becomes a Measurable Obligation

Chapter claim:
- Responsible AI translates fairness, safety, privacy, transparency, and accountability into verifiable system properties.

Student prior:
- "Responsible AI starts by choosing a fairness metric."

Activity beats:
1. Scenario: the selected track receives a release-review request from the named stakeholder.
2. Prediction: students predict which missing element will block deployment first: stakeholder, metric, evidence, owner, or policy.
3. Manipulation: students choose the stakeholder group, harm mode, and obligation threshold.
4. Evidence: a stakeholder obligation table converts the harm into required coverage, maximum disparity, and expected affected units.
5. Consequence: a failure banner names the track-specific harm if the obligation is too vague or too weak.
6. Math Peek/source model: `affected_units = events_per_day * residual_gap_pct * exposure_share`; fairness metrics require subgroup labels and stakeholder thresholds.
7. Checkpoint: students select the binding obligation to carry into Parts B-D.

Ledger output:
- `stakeholder_group`, `harm_mode`, `obligation_metric`, `obligation_threshold`, `binding_amount`.

### Part B: Concept Module - Responsible Evidence Consumes Capacity

Chapter claim:
- Fairness monitoring, explanations, privacy controls, and review gates add latency, compute, cost, energy, and release-delay overhead.

Student prior:
- "Adding governance controls can only make the system safer."

Activity beats:
1. Scenario: the same obligation must be enforced in the selected deployment environment.
2. Prediction: students predict which knob will become the binding overhead: monitoring, explanation, privacy, or human review.
3. Manipulation: students tune monitoring intensity, explanation coverage, privacy strictness, and human-review share.
4. Evidence: an overhead frontier chart shows risk reduction against latency/cost/energy/release delay.
5. Consequence: a reversible failure state appears when a technical guardrail or governance guardrail is violated.
6. Math Peek/source model: overhead combines chapter ranges for fairness monitoring, SHAP-style explanations, privacy controls, and HITL review.
7. Checkpoint: students identify the control that creates the best feasible evidence improvement.

Ledger output:
- `monitoring_intensity`, `explanation_coverage`, `privacy_level`, `human_review_share`, `overhead_binding_constraint`.

### Part C: Concept Module - Audit Coverage Determines Blind Spots

Chapter claim:
- Dashboards and audits only matter when they cover relevant cohorts, have labels, expose blind spots, and bind to an escalation path.

Student prior:
- "A monitoring dashboard creates accountability."

Activity beats:
1. Scenario: a postdeployment audit must prove whether the Part A obligation is actually observable.
2. Prediction: students predict whether the blind spot comes from label delay, sample coverage, intersectional slices, or owner/escalation failure.
3. Manipulation: students adjust audit sample rate, label availability, intersectional depth, and escalation speed.
4. Evidence: an audit coverage chart/table reports covered units, blind units, confidence score, and escalation readiness.
5. Consequence: a failure state appears when blind units exceed the track's residual-harm cap or escalation misses its deadline.
6. Math Peek/source model: coverage score combines sample rate, label availability, slice depth, and escalation readiness; blind units scale with fleet volume.
7. Checkpoint: students name the blind spot and the escalation path that remains accountable for it.

Ledger output:
- `audit_sample_rate`, `label_availability`, `slice_depth`, `escalation_path`, `blind_spot`, `residual_harm_units`.

### Part D: Concept Module - Policy Is a Guardrailed Design Decision

Chapter claim:
- Responsible AI policies must satisfy simultaneous technical and governance guardrails; one fairness metric or dashboard is not enough.

Student prior:
- "The most aggressive mitigation is the most responsible policy."

Activity beats:
1. Scenario: a release board asks for one policy, a rejected alternative, and an owner.
2. Prediction: students predict which candidate policy will satisfy both technical and governance guardrails.
3. Manipulation: students choose a policy candidate and rejected alternative.
4. Evidence: a policy scorecard compares utility, harm reduction, overhead, audit coverage, escalation readiness, and residual obligation.
5. Consequence: failure appears when the selected policy passes model quality but violates governance, or vice versa.
6. Math Peek/source model: deployment-ready policy is a conjunction of `technical_ok`, `governance_ok`, `audit_ok`, and `residual_owner_assigned`.
7. Checkpoint: students commit to the selected policy and rejected alternative.

Ledger output:
- `selected_policy`, `rejected_alternative`, `policy_pass`, `governance_guardrail`, `residual_owner`.

### Synthesis: Responsible Fleet Memo

Chapter invariant:
- The responsible fleet memo must carry the selected policy, binding amount, residual obligation, rejected alternative, source trace, and implication for V2-17.

Activity beats:
1. Assemble the selected track, stakeholder harm, obligation, overhead, audit blind spot, selected policy, and residual owner.
2. Show a report completeness table and lock report export until required predictions and decisions are made.
3. Save a structured `DesignLedger` entry for V2-17 fleet synthesis.
4. Export a responsible fleet memo using `build_lab_report` and `report_export_panel`.
5. State the V2-17 implication: the fleet synthesis must treat the responsible-AI policy as a hard guardrail, not a side note.

Ledger output:
- `responsible_ai_policy`, `binding_amount`, `residual_obligation`, `audit_blind_spot`, `v2_17_synthesis_implication`.

## Mechanics Plan

Opening:
- Shared academic header, track selector, track context, track arc context, chapter recap, source trace, and lab map.

Part A:
- Prediction radio, stakeholder dropdown, harm dropdown, obligation threshold slider.
- Stakeholder obligation table and affected-units bar.
- Math Peek accordion for fairness metrics and affected-unit conversion.

Part B:
- Prediction radio, sliders for monitoring, explanation, privacy, and human review.
- Risk-reduction versus overhead chart and exact table fallback.
- Reversible failure banner for latency/cost/energy/release-delay violation.

Part C:
- Prediction radio, audit sample-rate slider, label-availability slider, slice-depth dropdown, escalation-speed slider.
- Audit coverage bars and blind-spot table fallback.
- Failure banner for uncovered residual harm or missed escalation path.

Part D:
- Prediction radio, selected policy dropdown, rejected-alternative dropdown, residual-owner dropdown.
- Policy scorecard table and stacked guardrail bars.
- Failure banner when policy fails technical, governance, audit, or residual-owner gates.

Synthesis:
- Evidence summary, big takeaways, ledger HUD, report export panel.

## Evidence Plan

Every decision-driving visual has a table fallback:
- Part A: stakeholder harm, metric, threshold, exposure, affected units.
- Part B: control settings, risk reduction, latency/cost/energy/release delay, violations.
- Part C: audit sample, label coverage, slice multiplier, blind units, confidence, escalation status.
- Part D: selected and rejected policy rows, pass/fail guardrails, residual obligation.
- Synthesis: final memo fields and source trace.

Report fields:
- selected track and scenario
- predictions from Parts A-D
- knob settings and selected policy
- binding constraints and guardrail violations
- evidence summary
- final responsible fleet decision
- big takeaways
- residual risk and V2-17 implication
- source trace naming shared helpers and notebook-local `v2_16_` teaching models

## Single Source and Support Strategy

Use existing support:
- `track_selector`, `track_context`, `track_arc_context`, `source_trace`
- `get_lab_metadata`, `get_lab_track_variant`, `get_track_profile`
- `resolve_mlsysim_ref`
- `responsibility_track_profile`, `metric_conflict`, `responsibility_budget`, `explanation_overhead`
- `DesignLedger`
- `build_lab_report`, `report_export_panel`

Notebook-local helpers:
- Prefix all new support with `v2_16_`.
- Local helpers may encode teaching estimates for fleet audit coverage, policy scoring, and residual obligation because current shared helpers do not model V2-specific audit blind spots or policy selection.
- Shared helpers/tests are not modified.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| V2-16 track variants currently come from the generic system-design variant path. | Use the generic variant for catalog/track/source context, then define V2-specific responsible-AI teaching profiles notebook-local. |
| Audit coverage and residual-harm units are scenario estimates, not MLSysIM registry physics. | Label them in Math Peek and Source Trace as notebook-local teaching models tied to chapter formulas and variant profile assumptions. |
| Full legal/regulatory compliance can be overfit or become stale. | Keep policy language at engineering guardrail level: traceability, owner, escalation, documentation, and residual obligation. |
| Other workers may edit adjacent labs. | Edit only `lab_16_responsible_ai.py` and this track-plan file; do not touch shared helpers. |

## Depth Audit

| Module | Concept Clarity | Activity Depth | Track Specificity | Mechanics Fit | Evidence Quality | Traceability | Pass Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Stakeholder harm becomes an amount with prediction, controls, evidence, Math Peek, and checkpoint. |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Controls create reversible overhead and guardrail failures; evidence supports feasible governance choice. |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Audit coverage, blind spots, escalation, and residual harm are quantified by track. |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Policy selection requires technical and governance gates plus rejected alternative and owner. |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 | Memo carries selected policy, binding amount, residual obligation, and V2-17 implication. |

Acceptance gates:
- Every module has at least five student-facing activity beats.
- At least one reversible failure state exists in Parts B-D.
- Tracks change stakeholder, constraints, failure threshold, evidence emphasis, and report framing.
- Synthesis ties every module back to the chapter invariant.
