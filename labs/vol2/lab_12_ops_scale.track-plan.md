# V2-12 Track Plan: Operations at Scale as Control Loops

## Chapter Invariant

Operations at scale are control loops. SLOs, canaries, rollouts, incidents, and blast radius all spend an error budget over time, so a valid operating policy must choose what amount is allowed to be at risk, how quickly evidence is collected, and when the system rolls back or escalates.

Tracks are lenses, not different concepts. Every student works through the same Part A/B/C/D sequence:

1. Reliability becomes an amount system through SLO and error budget.
2. Canary and rollout size trade learning speed against blast radius.
3. Incident response budgets recovery time and lost work.
4. Operations policy must satisfy SLO, blast radius, cost, and governance guardrails.

## Reading Map

| Lab module | Chapter anchors | Claim used in the lab |
|---|---|---|
| Opening | `#sec-ops-scale`, `#sec-ml-operations-scale-singlemodel-platform-operations-db8e` | At fleet scale, manual model operations become a control-plane problem. |
| Part A | `#sec-ml-operations-scale-freshness-slos-6912`, `#sec-ml-operations-scale-oncall-practices-ml-teams-26e4`, SRE footnote on SLO/error budget | Reliability is a measured allowance over time, not a moral target. |
| Part B | `#sec-ml-operations-scale-staged-rollout-strategies-2d1f`, `#eq-canary-duration`, `#sec-ml-operations-scale-canary-deployment-3d8b` | Smaller canaries reduce exposure but slow statistical learning. |
| Part C | `#sec-ml-operations-scale-production-debugging-incident-response-9449`, `#tbl-runbook-diagnostic-flow`, `#tbl-pir-control-questions` | Incident response converts detection, diagnosis, mitigation, and recovery time into lost work. |
| Part D | `#tbl-self-service-deployment-invariants`, `#sec-ml-operations-scale-resource-management-5550`, `#sec-ml-operations-scale-multitenancy-isolation-c3ff`, `#sec-ml-operations-scale-fallacies-pitfalls-fe00` | A policy is acceptable only if simultaneous guardrails pass. |
| Synthesis | `#sec-ml-operations-scale-summary-4d70`, `#sec-ml-operations-scale-fallacies-pitfalls-fe00` | The fleet, not the single model, is the unit that is watched, paid for, and governed. |

## Concept Inventory

### Accepted Concepts

| Concept | Why accepted | Module |
|---|---|---|
| SLO/error budget as amount system | Turns reliability into a calculable budget students can overspend and recover. | Part A |
| Canary sample duration and blast radius | Directly exposes the learning-speed versus exposure trade-off. | Part B |
| Incident response as recovery-time/lost-work budget | Makes incidents physical by accumulating impact while control loops run. | Part C |
| Multi-guardrail operations policy | Forces simultaneous SLO, blast radius, cost, and governance reasoning. | Part D |
| Carry-forward security implication | Connects Ops controls to V2-13: telemetry, rollout, and incident evidence become security evidence. | Synthesis |

### Rejected Or Deferred Concepts

| Concept | Reason deferred or rejected |
|---|---|
| Platform ROI breakeven | Important chapter concept, but it would pull the lab toward platform investment economics instead of operations control loops. |
| N-models complexity curve | Good opening context but too generic for the required amount-system reasoning. |
| Feature-store architecture internals | Freshness SLOs inform Part A, but feature-store design is not the main V2-12 lab target. |
| A/B testing and interleaving math | Related to rollout evidence, but Part B uses canary duration/exposure to keep the module focused. |
| Distributed NCCL debugging | Incident response section covers it, but this lab targets ML operations policy rather than training-debugging mechanics. |
| Full FinOps/TCO optimization | Cost is a guardrail in Part D, not the primary concept. |

## Track Narratives

The shared concepts stay fixed. Track selection changes persona, constraints, thresholds, evidence emphasis, failure mode, and report framing.

| Track | Persona | Track-specific operations unit | Binding constraints | Natural failure | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile release operations lead | App/model rollout cohorts across device and OS versions | Crash-free sessions, privacy-safe telemetry, app responsiveness, battery | Silent quality regression reaches broad app rollout before opt-in telemetry has enough evidence | Mobile rollout policy memo |
| Oura Ring | Wearable firmware operations lead | Firmware/model OTA cohorts and sensor-quality populations | Battery, sensing continuity, slow delayed labels, health-adjacent false alerts | Firmware rollout damages sensing/battery and delayed labels arrive too late | Wearable OTA operations memo |
| RoboTaxi | Autonomous fleet safety operations lead | Vehicles, geofences, replay cohorts, and safety release windows | Rare-event recall, safety margin, geofenced blast radius, board approval | Expansion learns quickly but exposes too many live miles before safety evidence is sufficient | Safety rollout control memo |
| Cloud Fleet | Platform SRE and ML service owner | Service traffic, regions, tenants, and model registry releases | Availability/latency SLO, cost/request, tenant isolation, governance | Fast rollout spends too much error budget and cost before rollback completes | Platform operations policy memo |

## Concept Modules

### Part A: Concept Module - SLO/Error Budget Turns Reliability Into An Amount System

**Chapter claim:** SLOs and freshness/quality targets create quantitative reliability contracts; classical SRE error budgets must be extended to ML quality and drift-aware signals.

**Student prior:** "A high SLO is just a stricter target." Productive failure: students can choose a target and then discover the monthly failure minutes, quality-loss allowance, or incident count has already been spent.

**Activity beats:**

1. Scenario: the selected track's operations owner has to authorize a release window.
2. Prediction: choose which amount will become binding first: downtime minutes, quality/drift budget, or incident budget.
3. Manipulation: set availability SLO, quality SLO, monthly incidents, detection delay, and impact duration.
4. Evidence: budget table and timeline compare allowed budget against observed spend.
5. Consequence: reversible failure state names the overspent budget and how much must be recovered.
6. Math Peek/source model: error-budget minutes = period minutes x (1 - SLO), quality error budget = baseline quality - quality floor, incident spend = count x impact minutes.
7. Checkpoint/report decision: record the binding ops amount and mitigation.

**Mechanics:** Structured radio prediction, sliders, stacked budget bar, exact table fallback, failure callout, source accordion.

**Ledger output:** `partA_binding_amount`, `partA_error_budget_minutes`, `partA_spend_minutes`, `partA_quality_budget_pp`, `partA_quality_spend_pp`.

### Part B: Concept Module - Canary And Rollout Size Trade Learning Speed Against Blast Radius

**Chapter claim:** Canary duration is `T_stage = n_samples_needed / (request_rate * p_stage)`, while exposure is proportional to traffic percentage and detection time.

**Student prior:** "Smaller canary is always safer." Productive failure: a very small canary protects exposure but may take longer than the release window, leaving the system blind.

**Activity beats:**

1. Scenario: the same operations owner must choose a canary percentage and stage time.
2. Prediction: choose whether tiny, moderate, or aggressive canarying gives the safest policy.
3. Manipulation: adjust traffic percentage, request rate, sample requirement, and stage hours.
4. Evidence: curve/table show statistical samples collected, required duration, exposed work/users, and blast-radius spend.
5. Consequence: boundary state distinguishes "too blind" from "too exposed."
6. Math Peek/source model: chapter canary-duration equation and exposure = traffic x detection window x traffic amount.
7. Checkpoint/report decision: choose rollout policy and rejected alternative.

**Mechanics:** Structured radio prediction, sliders, line chart with threshold band, exposure cards, table fallback.

**Ledger output:** `partB_canary_pct`, `partB_required_hours`, `partB_stage_hours`, `partB_blast_radius_units`, `partB_rollout_decision`, `partB_rejected_alternative`.

### Part C: Concept Module - Incident Response Is Recovery-Time And Lost-Work Budgeting

**Chapter claim:** Incident response needs diagnostic order: classify, attribute, mitigate blast radius, recover, and convert the gap into a stronger platform control.

**Student prior:** "Incidents are fixed by the first plausible repair." Productive failure: skipping diagnosis can reduce one visible metric while increasing lost work, user impact, or rollback exposure.

**Activity beats:**

1. Scenario: an ML incident has normal infrastructure health but degraded product/safety/quality.
2. Prediction: choose the first responder action: restart, rollback, inspect feature/data signals, or escalate.
3. Manipulation: change MTTD, diagnosis minutes, mitigation fraction, recovery minutes, and runbook maturity.
4. Evidence: incident timeline and lost-work table decompose detection, diagnosis, mitigation, recovery, and residual blast radius.
5. Consequence: failure state names whether lost work exceeds the response budget or whether diagnostic order is violated.
6. Math Peek/source model: lost work = affected units per minute x impact fraction x minutes; recovered work is reduced only after mitigation.
7. Checkpoint/report decision: select incident policy and missing control to add.

**Mechanics:** Radio prediction, sliders, timeline stacked bar, loss ledger table, runbook-choice checkpoint.

**Ledger output:** `partC_mttd_min`, `partC_recovery_min`, `partC_lost_work_units`, `partC_response_budget_units`, `partC_missing_control`.

### Part D: Concept Module - Operations Policy Must Satisfy SLO, Blast Radius, Cost, And Governance Guardrails

**Chapter claim:** Self-service deployment is safe only when platform invariants capture artifact version, resource envelope, traffic policy, quality gates, telemetry, and approval boundaries.

**Student prior:** "The best policy is the lowest expected cost or fastest rollout." Productive failure: the lowest-cost policy can fail SLO, blast radius, or governance even if its expected dollar cost is attractive.

**Activity beats:**

1. Scenario: choose a policy for the selected track's next production rollout.
2. Prediction: choose which guardrail will reject the naive fast policy.
3. Manipulation: set rollout aggressiveness, automation level, telemetry depth, and governance review level.
4. Evidence: policy comparison table with guardrail badges and amount-system metrics.
5. Consequence: explicit pass/fail callout names every violated guardrail and the binding ops amount.
6. Math Peek/source model: policy score is not a single objective; feasibility is the conjunction of SLO, blast-radius, cost, and governance predicates.
7. Checkpoint/report decision: choose final policy and record rejected alternative.

**Mechanics:** Structured prediction, sliders/dropdowns, policy score table, guardrail badges, failure state, final decision radio.

**Ledger output:** `partD_selected_policy`, `partD_binding_guardrail`, `partD_policy_feasible`, `partD_cost_index`, `partD_governance_status`.

### Synthesis: Operations-at-Scale Memo

**Student task:** Produce a concise operations memo containing:

1. Selected rollout/incident policy.
2. Binding ops amount.
3. Rejected alternative and why.
4. Evidence from Parts A-D.
5. V2-13 security implication.

**Required carry-forward:** The V2-13 implication must connect operations evidence to security/privacy. Examples: telemetry minimization affects detection; rollout blast radius bounds exploit exposure; governance review creates an audit trail; incident evidence becomes a security escalation surface.

## Mechanics Plan

| Belt | Mechanics | Why used |
|---|---|---|
| Opening | Header, track selector, track context, reading map | Frames invariant and keeps track as a lens. |
| Prediction | `mo.ui.radio` for Parts A-D | Forces a prior before evidence. |
| Manipulation | Sliders and dropdowns, 1-4 controls per module | Lets students search budgets and boundaries. |
| Evidence | Plotly budget bars, duration/exposure curves, incident timeline, policy table | Shows amount-system consequences. |
| Failure | `mo.callout` danger/success with value, limit, unit, mitigation | Makes boundary reversible and non-color-only. |
| Source | Math Peek-style markdown cards and source trace | Ties formulas to chapter anchors. |
| Decision | Checkpoint radios and final memo controls | Converts evidence into policy. |
| Ledger | `DesignLedger.save`, HUD, report export panel | Carries selected operations policy into V2-13. |

## Evidence And Ledger Plan

Every plot has an exact table fallback. The final report and ledger snapshot record:

- selected track and scenario lens
- Part A predicted and actual binding amount
- Part B canary percentage, required duration, actual duration, and blast-radius amount
- Part C recovery-time budget, lost-work amount, and missing control
- Part D selected policy, rejected alternative, pass/fail guardrails, and binding guardrail
- synthesis memo fields: selected rollout/incident policy, binding ops amount, rejected alternative, V2-13 security implication

## Source And Amount Model

Notebook-local formulas are acceptable because no shared solver currently models this exact concept sequence. Every helper must use the `v2_12_` prefix and remain in `lab_12_ops_scale.py`.

Formula anchors:

- Error budget minutes: `period_minutes * (1 - availability_slo)`.
- Quality error budget: `baseline_quality_pct - quality_floor_pct`.
- Incident spend: `incident_count * impact_minutes`.
- Canary duration: `samples_needed / (request_rate_per_hour * traffic_fraction)`.
- Blast radius: `traffic_fraction * stage_hours * traffic_amount_per_hour`.
- Lost work: `affected_units_per_min * impact_fraction * minutes`, adjusted by mitigation after the mitigation step.
- Guardrail feasibility: `slo_ok and blast_ok and cost_ok and governance_ok`.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Existing generic renderer hides the required concept sequence | Replace renderer with explicit notebook-local cells while preserving bootstrap, track selector, and ledger patterns. |
| Track differences become different concepts | Keep one formula set and one Part A/B/C/D flow; track packet only changes units, thresholds, stakeholder, failure wording, and report frame. |
| Numbers look like production facts | Label constants as teaching scenario assumptions and source chapter claims/formulas where possible. |
| Browser/WASM import drift | Preserve current WASM bootstrap and wheel paths. |
| Shared-file collisions | Edit only `lab_12_ops_scale.py` and this track plan; do not touch shared helpers or tests. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 | Yes |

Acceptance checks:

- Every module has at least five substantive activity beats.
- Every module has structured prediction, manipulation, evidence, consequence, Math Peek/source model, and checkpoint/report decision.
- At least one reversible failure state exists in every module.
- The synthesis ties all modules back to the chapter invariant.
- Track lenses change persona, constraint thresholds, evidence emphasis, failure wording, and memo framing without changing the concept sequence.
