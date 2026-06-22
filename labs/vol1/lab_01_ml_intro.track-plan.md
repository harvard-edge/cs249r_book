# V1-01 Track Plan: ML Introduction / AI Triad

## Chapter Invariant

ML systems differ from traditional software because behavior emerges from coupled
Data, Algorithm, and Machine quantities over a lifecycle. Code can stay fixed
while quality changes, and the correct intervention depends on which axis binds
inside the selected deployment track.

## Reading Map

| Lab module | Chapter anchor | Chapter claim or formula |
|---|---|---|
| Opening | Purpose; AI Moment | ML systems manage statistical uncertainty and physical constraints at the same time. |
| Part A | Data-Centric Paradigm Shift; ML vs. Traditional Software | Software 2.0 degrades silently; data-defined behavior can change without code changes. |
| Part B | Defining ML Systems; D-A-M taxonomy | Data, Algorithm, and Machine are interdependent; the binding axis determines the first useful intervention. |
| Part C | ML vs. Traditional Software; Iron Law; Deployment Case Studies | Training is throughput-optimized, inference is latency-optimized, and they require different evidence. |
| Part D | ML System Lifecycle; Deployment Spectrum; Five-Pillar Framework | Lifecycle decisions must respond to monitoring evidence under track-specific constraints. |
| Synthesis | Summary; Fallacies and Pitfalls | D-A-M bottlenecks migrate, learned behavior decays silently, and engineering is continuous co-design. |

## Concept Inventory

Accepted concepts:

- Software 2.0 / Data as Source Code: behavior is learned, not explicitly coded.
- Silent degradation and the degradation equation: quality can decay without a code diff.
- D-A-M taxonomy: the same symptom can bind on Data, Algorithm, or Machine.
- Moving bottleneck: optimizing one axis can expose another.
- Training-serving divide: training and inference have different amount systems and evidence.
- Lifecycle feedback loop: monitoring evidence sends the system back to data, training, deployment, or operations.

Rejected or deferred concepts:

- AI history and AI winters: useful context, but weaker direct lab consequence.
- Detailed bitter-lesson chronology: saved for later scale and architecture labs.
- Full iron-law derivation: referenced as Math Peek only; deeper arithmetic belongs to later foundations labs.
- Five-pillar organization chart: used as decision vocabulary, not taught as a standalone taxonomy here.

## Track Narratives

Tracks are lenses, not skins. The selected track changes controls, thresholds,
evidence, failure language, and report wording.

| Track | Stakeholder | Binding constraints | Primary evidence | Natural failure |
|---|---|---|---|---|
| iPhone | Mobile product engineer | Battery, thermal, local privacy, UX responsiveness | sustained latency, energy/use, privacy-safe cohort coverage | thermal throttle or visible UX lag |
| Oura Ring | Wearable firmware engineer | SRAM/flash, sensing cadence, duty cycle, battery | SRAM/flash fit, wake time/window, duty-cycle percentage | firmware or duty-cycle violation |
| RoboTaxi | Safety/perception engineer | rare-hazard recall, p99/p999 latency, safety margin | rare-event replay, tail latency, recall floor | safety margin miss |
| Cloud Fleet | Fleet service owner | cost/request, utilization, p99 SLA, carbon | load test, cost/request, utilization, SLO | SLO breach or negative unit economics |

Comparison rule:

- Part B compares the selected track with a contrasting track using the same
  D-A-M readiness values. iPhone compares against Oura Ring, Oura Ring and
  RoboTaxi compare against Cloud Fleet, and Cloud Fleet compares against Oura
  Ring. The comparison demonstrates that the same raw scores can produce a
  different binding axis because the deployment envelope changes.

## Concept Modules

### Part A: Concept Module - Model behavior is not normal software behavior

Chapter claim:

- ML systems can silently degrade as the production distribution moves away
  from the training distribution even when code, infrastructure, and serving
  health remain unchanged.

Student prior:

- "If code and dashboards are green, behavior is unchanged."

Storyline:

1. Scenario: the track stakeholder receives a quality complaint while crash logs
   and infrastructure health remain clean.
2. Prediction: the student predicts whether quality can fall without a code
   change and what signal should trigger action.
3. Manipulation: the student changes drift pressure, months in production, and
   monitoring cadence.
4. Evidence: a quality-vs-threshold chart and table show silent quality loss
   while code health remains fixed.
5. Consequence: the module names the track-specific violated guardrail.
6. Math Peek: Accuracy(t) ~= Accuracy_0 - lambda * D(P_t || P_0).
7. Checkpoint: the student chooses the first response, such as drift monitoring,
   cohort audit, or retraining trigger.

Mechanics:

- Structured prediction radio.
- Drift pressure slider, months slider, monitoring cadence dropdown.
- Threshold line chart plus table fallback.
- Reversible failure banner when quality falls below the track floor.

Ledger output:

- silent_degradation_prediction
- drift_pressure_pct
- months_in_production
- monitoring_cadence
- observed_quality_pct
- quality_floor_pct
- silent_failure_response

### Part B: Concept Module - The binding axis changes the correct intervention

Chapter claim:

- The D-A-M taxonomy is a diagnostic framework. Data determines learned
  behavior, Algorithm determines representational and computational demand, and
  Machine determines the feasible execution envelope.

Student prior:

- "Accuracy failure is a model problem" or "hardware upgrades fix everything."

Storyline:

1. Scenario: the selected track has a demo that works but fails in deployment.
2. Prediction: the student predicts which D-A-M axis will bind.
3. Manipulation: the student changes data, algorithm, and machine readiness.
4. Evidence: a readiness-vs-threshold chart and table identify the binding axis.
5. Consequence: a track comparison table shows why the same scores bind
   differently in a second track.
6. Math Peek: Cost is proportional to Model Size times Dataset Size divided by
   Hardware Efficiency, so axes couple rather than optimize independently.
7. Checkpoint: the student records a final binding-axis diagnosis.

Mechanics:

- Prediction radio.
- Three readiness sliders.
- Binding-axis chart, threshold markers, violation table.
- Selected-track versus comparison-track table.

Ledger output:

- predicted_binding_axis
- final_binding_axis
- comparison_track
- primary_metric
- guardrail_metric
- data_algorithm_machine_scores

### Part C: Concept Module - Training and inference produce different evidence

Chapter claim:

- Training is throughput-optimized, while inference is latency- and envelope-
  optimized. A training success is not deployment evidence.

Student prior:

- "If the training run converged, deployment is proven."

Storyline:

1. Scenario: the training report looks acceptable, but the deployment owner asks
   for runtime evidence.
2. Prediction: the student predicts which evidence should authorize shipment.
3. Manipulation: the student changes model scale and operating pressure.
4. Evidence: side-by-side training and inference amount systems show different
   units, limits, and pass/fail results.
5. Consequence: the module names the selected track's deployment failure.
6. Math Peek: T ~= D_vol/BW + O/(R_peak * eta_hw) + L_lat; training spends
   throughput budgets, inference spends request/window budgets.
7. Checkpoint: the student chooses the evidence packet to attach to the memo.

Mechanics:

- Structured prediction radio.
- Model-scale slider and pressure slider.
- Stacked bar/threshold chart for training versus inference quantities.
- Evidence table with units and decision language.

Ledger output:

- training_inference_prediction
- model_scale_pct
- operating_pressure_pct
- training_amount
- inference_amount
- selected_evidence_packet

### Part D: Concept Module - Lifecycle decisions choose the first defensible fix

Chapter claim:

- The lifecycle loops back from monitoring to data collection, training,
  deployment, or operations. The first fix must satisfy the selected track's
  constraints rather than maximize a generic metric.

Student prior:

- "Spend evenly" or "pick the cheapest/highest-accuracy fix."

Storyline:

1. Scenario: the stakeholder has one engineering budget for the next lifecycle
   loop.
2. Prediction: the student predicts the budget strategy.
3. Manipulation: the student allocates budget across Data, Algorithm, and
   Machine and selects the intervention to defend.
4. Evidence: the frontier chart shows post-intervention scores, remaining
   binding axis, rejected alternatives, and failure state.
5. Consequence: the module names the track-specific risk of choosing the wrong
   first fix.
6. Math Peek: lifecycle evidence chooses the constraint to relieve first; local
   optimization can expose a new bottleneck.
7. Checkpoint: the student chooses the validation test that would invalidate the
   decision.

Mechanics:

- Budget-strategy prediction radio.
- Three budget sliders plus selected intervention dropdown.
- Post-intervention frontier chart and table.
- Reversible failure banner when the selected plan leaves a threshold violated.
- Validation evidence dropdown.

Ledger output:

- budget_strategy_prediction
- intervention_budget_split
- selected_intervention
- best_intervention
- rejected_alternatives
- validation_evidence
- residual_binding_axis

### Synthesis: Concept Module - Triad diagnosis memo and carry-forward risk

Chapter invariant:

- A machine learning system does what its data, arithmetic, and hardware permit,
  not what its source code intends.

Storyline:

1. The student reviews silent degradation, binding-axis diagnosis, training vs.
   inference evidence, and lifecycle intervention evidence.
2. The notebook generates a track-specific triad diagnosis memo.
3. The student selects a carry-forward risk for future labs.
4. The Design Ledger saves the selected track, diagnosis, evidence packet, first
   fix, rejected alternatives, and risk.

## Mechanics And Evidence Plan

| Module | Controls | Graphs/tables | Failure or boundary | Evidence saved |
|---|---|---|---|---|
| Part A | prediction, drift, months, cadence, response | degradation line chart, table | quality below track floor | observed quality and response |
| Part B | prediction, D/A/M readiness, diagnosis | threshold bar chart, comparison table | readiness below threshold | binding axis and comparison |
| Part C | prediction, model scale, pressure, evidence packet | amount-system bar chart, table | inference limit miss despite training pass | selected evidence packet |
| Part D | prediction, D/A/M budgets, selected fix, validation | frontier bar chart, table | remaining binding axis | first fix and validation |
| Synthesis | carry-forward risk | memo card, report export | incomplete report lock | ledger/report artifact |

Source and trace policy:

- Existing `mlsysbook_labs.triad` helpers remain the source for profile,
  diagnosis, and frontier logic.
- Existing lab variant metadata remains the source for track stakeholder,
  model, hardware, primary metric, guardrail metric, thresholds, default scores,
  interventions, and validation tests.
- New notebook-local teaching models are prefixed with `v1_01_` and source-
  traced to the Introduction chapter formulas and anchors.
- No shared helpers, tests, implementation notes, or other labs are edited.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 3 | Pass |

Acceptance checks:

- No dimension is below 2.
- Each Part has scenario, prediction, manipulation, evidence, consequence,
  Math Peek/source model, and checkpoint/report decision.
- At least one reversible failure exists in Part A and Part D.
- The selected track changes controls, thresholds, evidence, and report language.
- Two-track comparison appears in Part B.
- Synthesis ties all modules back to the chapter invariant and saves a carry-
  forward risk.
