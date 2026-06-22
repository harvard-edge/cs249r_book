# V1-14 Track Plan: ML Operations Control Loop

## Chapter Invariant

Production ML is a control loop. Monitoring, drift thresholds, rollout, rollback,
alert ownership, and retraining policy allocate two scarce resources:
operational attention and error budget. A green service is not enough; the loop
must measure deployed behavior, decide when drift deserves attention, limit the
blast radius of releases, and spend error budget deliberately under the selected
track constraints.

## Reading Map

| Module | Chapter anchor | Claim used in the lab |
|---|---|---|
| Opening | `MLOps Overview`; `Production-Monitoring Interface` | MLOps makes silent failures visible and closes the feedback loop between data reality and deployed model behavior. |
| Part A | `Observable degradation`; `Model and infrastructure monitoring`; `Data quality monitoring` | Infrastructure health and offline model metrics are incomplete; production monitoring must include statistical telemetry, deployed proxy behavior, and delayed labels. |
| Part B | `Feature distribution monitoring`; `Layered monitoring and drift quantification`; `Monitoring cost model` | Drift thresholds trade false alarms, missed degradation, label delay, and monitoring/alert cost. PSI > 0.2 is a starting point, not a universal truth. |
| Part C | `Model deployment`; `Rollback strategies and safety mechanisms`; `Incident response for ML systems` | Canary, blue-green, shadow, fallback, and rollback policies control blast radius and recovery time. Rollbacks must be tested. |
| Part D | `Cost-aware automation`; `Quantitative retraining economics`; `On-call practices`; `MLOps investment economics` | Operational policy is an error-budget spending decision across monitoring, retraining, rollout, rollback, and ownership. |
| Synthesis | `Summary`; `Fallacies and Pitfalls` | The durable artifact is a runbook memo: alert threshold, rollback rule, residual blind spot, and carry-forward responsibility risk. |

## Concept Inventory

Accepted concepts:

| Concept | Reason accepted |
|---|---|
| Production ML as a feedback control loop | This is the chapter invariant and ties monitoring, drift, deployment, rollback, and ownership together. |
| Deployed behavior vs. model-only metrics | Corrects the common prior that a validated model and green uptime prove production health. |
| Threshold false-alarm/miss trade-off | Gives students a manipulable policy boundary with visible operational cost. |
| Rollout and rollback as blast-radius controls | Turns deployment strategy into a measurable safety and recovery decision. |
| Error-budget allocation under track constraints | Forces synthesis across cost, risk, attention, and domain-specific guardrails. |

Rejected or deferred concepts:

| Concept | Reason rejected for this lab |
|---|---|
| Tool catalog of MLflow, Prometheus, Grafana, feature stores | Too taxonomic; tools do not create the core decision consequence. |
| Full training-serving skew debugging | Important, but better handled by a separate data/feature-store lab. Here it appears only as a monitored failure source. |
| Detailed A/B sample-size derivation | Too much math for the control-loop sequence; canary evidence is represented through blast-radius and recovery quantities. |
| Broad technical debt taxonomy | Kept as residual responsibility risk in synthesis rather than a separate Part D concept. |
| ClinAIOps governance case | Useful comparison, but this lab owns the four volume tracks listed below. |

## Shared Concept Module Sequence

All tracks use the same A/B/C/D concept sequence. The selected track changes the
persona, constraints, threshold amounts, evidence emphasis, failure mode, and
runbook framing; it does not create different concepts.

| UI Part | Concept module | Student prior | Track amount-system reasoning | Evidence produced |
|---|---|---|---|---|
| Part A | Monitoring must measure deployed behavior, not only model metrics. | If uptime and offline metrics are green, the model is healthy. | Compare deployed proxy behavior, delayed labels, and track guardrails such as battery, duty cycle, rare-event safety, or SLO/cost. | Timeline plus signal table showing quality breach before an actionable alert. |
| Part B | Drift thresholds trade false alarms against missed degradation. | Lower thresholds are always safer, or higher thresholds reduce noise without consequence. | Tune PSI threshold against alert-review cost, false-alarm attention, missed-damage cost, and label delay for the track. | Threshold sweep, false-alarm/missed-degradation table, chosen threshold checkpoint. |
| Part C | Rollout and rollback policy controls blast radius and recovery time. | A canary is enough if aggregate metrics look good. | Choose canary exposure and rollback window under track-specific fallback: kill switch, OTA rollback, geofence fallback, or registry pin. | Exposure timeline, blast-radius/recovery metrics, failure boundary when exposure exceeds allowed budget. |
| Part D | Operational policy must spend error budget deliberately under track constraints. | The lowest-cost policy wins, or automation can own the whole loop. | Allocate a fixed error budget across detection delay, stale-model exposure, rollback exposure, and residual blind spot. | Error-budget ledger with feasible/failing state and report-ready runbook decision. |
| Synthesis | Operations runbook memo. | Completion is a set of charts. | Convert evidence into an owned runbook with threshold, rollback rule, blind spot, and carry-forward risk. | Design Ledger save and downloadable report. |

## Concept Modules

### Part A - Concept Module: Deployed Behavior Is The Monitor

Chapter claim:
- ML systems fail silently because uptime can stay green while accuracy erodes.
- Observable degradation requires statistical telemetry, deployed proxy behavior,
  delayed labels, and guardrail metrics.

Student storyline:
1. Scenario: the selected track owner receives a green infrastructure dashboard
   while the track drift source is accumulating.
2. Prediction: choose whether uptime/offline metrics/deployed proxy/delayed labels
   will expose the problem first.
3. Manipulation: move days since deployment, PSI/day drift rate, and alert threshold.
4. Evidence: inspect true quality, observed quality, PSI, alert day, and deployed
   signal table.
5. Consequence: see the silent degradation window and its track-specific cost or
   safety implication.
6. Math Peek/source: `Accuracy ~= A0 - lambda * PSI`; detection delay is
   `alert_day - quality_breach_day`.
7. Checkpoint: decide which deployed signal must be in the runbook.

Mechanics:
- Structured radio prediction.
- Three sliders.
- Quality timeline, metric cards, exact fallback table, Math Peek callout.

Depth gate:
- Activity count: 7.
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: quality breach before alert.
- Math/source: yes.
- Track-specific consequence: yes.

Ledger fields:
- `part_a_monitor_signal`, `part_a_detection_delay_days`,
  `part_a_accumulated_damage_cost`.

### Part B - Concept Module: Thresholds Spend Attention

Chapter claim:
- PSI, KS, and JS thresholds are starting points that must be calibrated.
- Higher thresholds reduce alert noise but increase missed degradation.
- Lower thresholds catch drift earlier but spend operational attention.

Student storyline:
1. Scenario: the on-call rotation is receiving noisy drift pages, but the team is
   worried about missing real degradation.
2. Prediction: choose whether tight, default, or loose thresholds minimize total
   operational cost.
3. Manipulation: adjust PSI threshold, alert-review cost, and false-alarm rate.
4. Evidence: inspect false-alarm attention cost, missed-damage cost, detection
   day, and total threshold cost.
5. Consequence: identify when alert fatigue or silent damage dominates.
6. Math Peek/source: `threshold / drift_rate + label_delay` gives expected
   detection day; total threshold cost combines false alarms and missed damage.
7. Checkpoint: select the threshold to carry into the runbook.

Mechanics:
- Structured radio prediction.
- Threshold, false-alarm, and review-cost sliders.
- Cost curve, threshold comparison table, reversible failure callout.

Depth gate:
- Activity count: 7.
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: threshold too loose or alert burden too high.
- Math/source: yes.
- Track-specific consequence: yes.

Ledger fields:
- `part_b_threshold_psi`, `part_b_false_alarm_cost`,
  `part_b_missed_damage_cost`, `part_b_threshold_failure_mode`.

### Part C - Concept Module: Rollback Limits Blast Radius

Chapter claim:
- Staged deployment strategies validate candidate models against production
  traffic before full rollout.
- Immediate, rapid, and delayed rollback tiers determine recovery time and state
  handling.

Student storyline:
1. Scenario: a candidate model has passed validation and is entering production.
2. Prediction: choose whether rollout size, rollback speed, fallback, or aggregate
   canary metrics dominate the release risk.
3. Manipulation: adjust canary traffic, rollback exposure hours, and fallback
   coverage.
4. Evidence: inspect exposed traffic, recovery time, rollback tier, and
   blast-radius cost.
5. Consequence: cross the failure boundary when track-specific exposure exceeds
   the allowed blast-radius budget.
6. Math Peek/source: blast radius is exposure share multiplied by rollback window
   and impact rate; rollback tiers map to under 1 minute, under 15 minutes, and
   under 4 hours.
7. Checkpoint: write the rollback rule for the runbook.

Mechanics:
- Structured radio prediction.
- Canary, rollback, and fallback sliders.
- Exposure chart, tier table, failure banner.

Depth gate:
- Activity count: 7.
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: rollout exposure exceeds track budget.
- Math/source: yes.
- Track-specific consequence: yes.

Ledger fields:
- `part_c_canary_pct`, `part_c_rollback_hours`, `part_c_blast_radius_units`,
  `part_c_rollback_rule`.

### Part D - Concept Module: Error Budget Is A Policy Choice

Chapter claim:
- Cost-aware automation uses retraining economics and deployment risk to decide
  when intervention is worth it.
- On-call practices and ownership convert alerts into action.

Student storyline:
1. Scenario: leadership asks for one defensible runbook rather than separate
   monitoring, retraining, and rollback knobs.
2. Prediction: choose whether the binding risk is threshold looseness, stale
   model exposure, rollback exposure, or residual blind spot.
3. Manipulation: tune policy threshold, cadence, canary, and rollback window.
4. Evidence: inspect error-budget spending across detection, staleness, rollback,
   and residual blind-spot responsibility.
5. Consequence: see PASS/FAIL when the policy overspends the track budget or
   violates alert ownership.
6. Math Peek/source: `T* = sqrt(2C / C_drift)` and monitoring cost
   `C_ingest + C_storage + C_compute + C_alert` support the operating policy.
7. Checkpoint: choose the final policy and name the carry-forward risk.

Mechanics:
- Structured radio prediction.
- Four policy sliders.
- Error-budget stacked bar, cost cards, policy feasibility table.

Depth gate:
- Activity count: 7.
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: overspent error budget or policy violations.
- Math/source: yes.
- Track-specific consequence: yes.

Ledger fields:
- `part_d_error_budget_days`, `part_d_policy_feasible`,
  `part_d_binding_risk`, `part_d_carry_forward_risk`.

## Track Narratives

| Track | Persona | Constraint emphasis | Threshold evidence | Failure mode | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile ML operations lead | Privacy-safe telemetry, battery, thermals, app responsiveness | Opt-in proxy quality, crash-free sessions, thermal/battery regressions | Battery or thermal regression hidden by aggregate model quality | App/model rollout runbook with remote kill switch and privacy-safe blind spot. |
| Oura Ring | Wearable firmware ML lead | Firmware telemetry, duty cycle, weak ground truth, OTA size/risk | Sensor-quality indicators, battery anomalies, delayed health labels | Duty-cycle or battery regression after OTA update | Firmware/model OTA runbook with staged holdout and delayed-label review. |
| RoboTaxi | Autonomous fleet safety operations lead | Rare-event safety, near misses, simulation replay, geofence fallback | Near-miss telemetry, disengagements, replay regressions, sensor-health monitors | Rare-event recall miss before aggregate safety metric moves | Geofenced rollout runbook with safety fallback and review-board owner. |
| Cloud Fleet | Cloud ML platform owner | SLO/error budget, canary, cost/request, utilization alerts | Online metrics, logs, delayed labels, p99 latency, cost/request | SLO or cost/request regression during canary | SRE/ML owner runbook with registry pin, canary rollback, and cost guardrail. |

## Mechanics, Evidence, And Ledger Plan

Mechanics:
- Opening belt: track selector, chapter invariant, reading map, source trace.
- Prediction belt: one structured prediction for each part before instruments.
- Control belt: 1-4 sliders per module, all mapped to operational quantities.
- Evidence belt: Plotly charts plus exact table fallbacks.
- Failure belt: reversible failure banners for missed alert, threshold over/under
  spending, blast-radius overrun, and policy infeasibility.
- Source belt: Math Peek callouts tied to chapter formulas or tables.
- Decision belt: checkpoint/report decision in each module.
- Ledger belt: save selected track, thresholds, rollback rule, policy feasibility,
  and runbook evidence.

Evidence and report output:
- Part A: deployed-behavior signal and detection delay.
- Part B: chosen threshold, false-alarm cost, missed-damage cost, failure mode.
- Part C: canary percentage, rollback exposure, fallback coverage, blast radius.
- Part D: error-budget allocation, feasible policy, binding risk, carry-forward
  responsibility risk.
- Synthesis: operations runbook memo with alert threshold, rollback rule,
  residual blind spot, and named owner/risk.

## Implementation Plan

Notebook-local support:
- Add `v1_14_track_amounts` for track-specific units, budgets, fallback labels,
  and blind-spot text derived from the track id and existing `OpsTrackProfile`.
- Add `v1_14_threshold_economics` for false-alarm vs. missed-degradation costs.
- Add `v1_14_rollout_risk` for blast-radius and rollback-tier calculations.
- Add `v1_14_error_budget` for policy budget allocation and binding-risk naming.
- Keep shared calculations from `mlsysbook_labs.ops`: `drift_visibility`,
  `retraining_cadence`, and `ops_policy`.

Owned files:
- `labs/vol1/lab_14_ml_ops.py`
- `labs/vol1/lab_14_ml_ops.track-plan.md`

Out of scope:
- Shared helper edits.
- Test edits.
- Variant registry edits.
- Commit creation.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Shared helpers do not expose threshold and blast-radius economics. | Implement small notebook-local helpers prefixed with `v1_14_`. |
| Track variants lack explicit error-budget units. | Derive units and budgets from track id and existing profile quantities inside notebook-local metadata. |
| Existing report schema may expect generic dictionaries. | Keep `build_lab_report` use unchanged and pass serializable evidence summaries. |
| Multiple workers may edit other labs. | Restrict edits to owned files only and do not run broad formatting. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | yes |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | yes |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | yes |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | yes |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 | yes |

Acceptance notes:
- Every module has at least five substantive student-facing activity beats.
- Every module includes prediction, manipulation, evidence, Math Peek/source
  model, and checkpoint/report decision.
- Reversible failure states exist in every part.
- Synthesis ties the parts back to the chapter invariant and Design Ledger.
