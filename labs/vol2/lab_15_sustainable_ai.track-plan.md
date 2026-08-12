# V2-15 Track Plan: The Carbon Budget

## Chapter Invariant

Sustainable AI is an amount system: a fleet is valid only when energy, carbon
intensity, utilization, embodied carbon, quality, latency, cost, reliability,
and governance all fit inside the operating envelope.

The lab should make students stop treating sustainability as a reporting
afterthought. They must measure an energy/carbon stack, find the binding
sustainability amount, then choose a carbon-aware policy that does not break
service guardrails.

## Reading Map

| Lab part | Chapter source | Chapter claim used |
|---|---|---|
| Opening | `The Energy Ceiling` | Power, cooling, water, carbon, and hardware lifetime are hard design constraints, not public-relations labels. |
| Part A | `Carbon footprint analysis`, `Lifecycle carbon estimation` | Workload carbon is energy times grid intensity, with facility PUE and embodied carbon included in the lifecycle boundary. |
| Part B | `Geographic and temporal optimization`, `Facility-level power metrics` | Utilization and placement change useful energy per job, while region and time change carbon intensity. |
| Part C | `Why optimization techniques save energy`, `Case study: Google's framework`, `Engineering guidelines` | Mitigation works only when it changes the dominant term and survives quality, latency, cost, reliability, and governance checks. |
| Part D | `Policy, Regulation, and the Path Forward`, `Fallacies and Pitfalls` | Carbon-aware policy changes the objective function; efficiency alone can be defeated by rebound or lifecycle omissions. |
| Synthesis | `Summary` | A sustainable architecture accounts for training, serving, embodied carbon, grid mix, hardware lifetime, and demand growth together. |

## Concept Inventory

Accepted concepts:

- Energy is a physical amount measured in joules or kWh; carbon is energy
  multiplied by grid intensity and facility overhead.
- PUE is necessary but insufficient because it omits grid carbon intensity,
  embodied carbon, water, and workload-level utilization.
- Utilization has two sides: low utilization wastes idle infrastructure, while
  excessive utilization can break p99 latency, freshness, or reliability.
- Embodied and operational carbon can trade places as the dominant lifecycle
  term depending on grid mix, hardware lifetime, fleet size, and workload duty
  cycle.
- Mitigation strategies are not interchangeable. Model efficiency, scheduling,
  utilization consolidation, hardware lifetime, and demand governance each move
  different amounts and can create different regressions.
- Carbon-aware policy is a guardrail bundle, not a green preference.

Rejected or synthesis-only concepts:

- Detailed CMOS derivations beyond the power proportionality needed for source
  trace. The lab focuses on system-level amounts rather than circuit design.
- Full water-usage accounting. WUE is named as a risk, but the student-facing
  simulator keeps the quantitative model to energy, carbon, utilization, and
  embodied carbon.
- Regulatory taxonomy depth. Policy appears as the final objective function and
  governance gate, not a compliance survey.
- Offsets. They are rejected as a policy alternative in narrative, but not made
  a quantitative candidate because the chapter's preferred action is direct
  reduction.

## Track Plan

Tracks keep the same concept sequence but change persona, constraints, binding
amount, failure mode, evidence emphasis, and memo framing.

| Track | Stakeholder | Sustainability lens | Likely binding amount | Failure mode | Memo framing |
|---|---|---|---|---|---|
| iPhone | Mobile product engineer | Device battery, thermal headroom, local/cloud split, privacy-safe measurement | device energy or embodied fleet carbon | battery/thermal budget miss or cloud offload privacy risk | carbon-aware mobile feature memo with battery and privacy guardrails |
| Oura Ring | Wearable firmware engineer | Tiny duty cycle, radio/sync cadence, battery replacement pressure, manufacturing share | battery duty-cycle energy or embodied fleet carbon | battery-life miss, sampling cadence miss, or sync policy failure | wearable sustainability memo with sensing quality and user-comfort guardrails |
| RoboTaxi | Autonomous vehicle platform engineer | Vehicle-local safety compute, noncritical deferral, fleet operating hours, safety reliability | operational power under safety guardrails | p99/safety margin miss if compute is deferred too aggressively | safety-critical carbon memo with rejected deferral/offload alternative |
| Cloud Fleet | Fleet service owner | Datacenter energy, PUE, utilization, region carbon intensity, carbon price, rebound | operational carbon intensity and utilization | carbon budget miss, p99 SLO breach, or rebound through uncapped demand | carbon-aware fleet operating policy with cost/SLA/quality guardrails |

## Concept Modules

### Part A: Concept Module - Carbon Is An Amount Stack

Chapter claim:
- Sustainable AI starts with workload-level accounting: operational energy,
  PUE, grid intensity, and embodied carbon are measured before optimization.

Student prior:
- "The greenest option is whichever uses less electricity per inference."

Activity beats:
1. Scenario: the selected track stakeholder must approve a workload before
   expansion.
2. Prediction: choose which amount will bind first: device/facility energy,
   carbon intensity, embodied carbon, or service guardrail.
3. Manipulation: adjust workload scale and utilization assumptions.
4. Evidence: inspect a stacked operational/embodied carbon chart and exact
   table.
5. Consequence: a reversible budget banner names the binding amount and the
   first recovery lever.
6. Math Peek: `facility_energy = IT_energy x PUE` and
   `operational_carbon = facility_energy x grid_intensity`.
7. Checkpoint: record the binding sustainability amount for downstream policy.

Ledger fields:
- `partA_binding_amount`
- `partA_operational_kg`
- `partA_embodied_kg`
- `partA_energy_kwh`

### Part B: Concept Module - Placement And Utilization Change The Carbon Bill

Chapter claim:
- Region and time can change carbon emissions by order-of-magnitude factors,
  but high utilization and deferred scheduling can break service guardrails.

Student prior:
- "Higher utilization and cleaner regions always improve sustainability."

Activity beats:
1. Scenario: operations asks whether to consolidate work, defer it to a clean
   window, or move it to a cleaner region.
2. Prediction: identify which amount will become limiting when utilization and
   placement are changed.
3. Manipulation: choose region, scheduling policy, and utilization target.
4. Evidence: compare region matrix, carbon bar chart, p99/freshness guardrail,
   and exact numbers.
5. Consequence: a carbon-budget or service-level failure appears and can be
   recovered by moving controls.
6. Math Peek: useful-energy overhead scales with utilization and
   `carbon = kWh x gCO2/kWh`.
7. Checkpoint: choose whether placement, schedule, or utilization is the next
   lever.

Ledger fields:
- `partB_region`
- `partB_schedule`
- `partB_utilization`
- `partB_carbon_kg`
- `partB_service_ok`

### Part C: Concept Module - Mitigation Must Preserve Guardrails

Chapter claim:
- The 4 Ms and engineering guidelines reduce different terms; a mitigation is
  sustainable only if quality, latency, cost, reliability, and governance still
  pass.

Student prior:
- "Efficiency is always the best mitigation."

Activity beats:
1. Scenario: a review board asks for one mitigation strategy, not a wish list.
2. Prediction: identify the guardrail most likely to reject an aggressive
   mitigation.
3. Manipulation: select mitigation strategy, intensity, and whether governance
   review is attached.
4. Evidence: inspect strategy table and selected-strategy guardrail badges.
5. Consequence: selected mitigation can fail quality, latency, cost,
   reliability, governance, or carbon.
6. Math Peek: mitigation changes multipliers on energy, carbon, embodied
   carbon, and service metrics, then applies a conjunction predicate.
7. Checkpoint: choose the mitigation and rejected alternative to carry forward.

Ledger fields:
- `partC_strategy`
- `partC_binding_guardrail`
- `partC_quality_pct`
- `partC_latency_ms`
- `partC_cost`
- `partC_reliability_pct`

### Part D: Concept Module - Carbon-Aware Policy Is A Guardrail Bundle

Chapter claim:
- Policy, carbon pricing, reporting, and demand governance change the
  optimization objective; efficiency without usage governance can rebound.

Student prior:
- "Pick the lowest-carbon point and declare the system sustainable."

Activity beats:
1. Scenario: launch review requires one policy, a rejected alternative, and a
   responsibility handoff.
2. Prediction: choose which policy will pass all guardrails.
3. Manipulation: select a policy and carbon price.
4. Evidence: inspect a launchability table and selected-policy summary.
5. Consequence: a nonlaunchable policy names failed guardrails and the safer
   alternative.
6. Math Peek: launchability is a conjunction over carbon, quality, latency,
   cost, reliability, and governance.
7. Checkpoint: save policy, binding amount, rejected alternative, residual
   risk, and V2-16 implication.

Ledger fields:
- `selected_policy`
- `binding_amount`
- `rejected_alternative`
- `residual_risk`
- `v2_16_responsible_ai_implication`

## Synthesis

Students export a carbon-aware engineering memo with:

1. Selected track and scenario.
2. Binding sustainability amount from Part A.
3. Region, utilization, and schedule evidence from Part B.
4. Mitigation selected in Part C and guardrail evidence.
5. Final policy from Part D, rejected alternative, residual risk, and V2-16
   responsible-AI implication.

## Mechanics Plan

| Need | Mechanic | Evidence |
|---|---|---|
| Commit prior | `mo.ui.radio` prediction in each part | Prediction-vs-actual reveal |
| Boundary finding | sliders for workload, utilization, mitigation intensity, carbon price | Failure/recovery callout with value, limit, and mitigation |
| Placement comparison | region dropdown plus region matrix | Carbon intensity and PUE table |
| Guardrail conjunction | strategy/policy tables with pass/fail labels | Named failed guardrails |
| Source trace | `source_trace` and Math Peek accordions | Formula, registry, and chapter source map |
| Synthesis | `DesignLedger.save`, `build_lab_report`, `report_export_panel` | Downloadable memo and JSON snapshot |

## Evidence Plan

Every part produces exact table values in addition to charts:

- Part A: operational kWh, operational kg CO2e, embodied kg CO2e, lifecycle kg
  CO2e, budget ratios, binding amount.
- Part B: chosen region, grid intensity, PUE, utilization, schedule delay, p99
  or freshness, carbon, service status.
- Part C: strategy-level carbon, quality, latency, cost, reliability,
  governance status, binding guardrail.
- Part D: policy launchability, failed guardrails, selected policy, rejected
  alternative, residual risk.

## Implementation Notes

- Use existing track selector, canonical track profiles, generic
  `v2_15_carbon_budget` variant, `DesignLedger`, `build_lab_report`,
  `report_export_panel`, `track_context`, `track_arc_context`, and
  `source_trace`.
- Add notebook-local helpers only, all prefixed `v2_15_`.
- Do not modify shared helpers or tests in this wave.
- Hardware power, battery, embodied H100 carbon, model refs, grid carbon
  intensity, and PUE should come from MLSysIM where available. Track-specific
  fleet sizes, local embodied estimates for non-H100 devices, region electricity
  prices, and service budgets are notebook-local teaching assumptions and must
  be exposed in the source trace.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 2 | Pass |

Acceptance notes:
- Each part has at least five student-facing beats.
- Part B and Part D include reversible failure states.
- Track differences change persona, assumptions, guardrails, failure, and memo
  framing.
- The main residual risk is that a reusable sustainability solver does not yet
  exist in `mlsysbook_labs`; notebook-local helpers should be promoted only
  after the wave stabilizes the API shape.
