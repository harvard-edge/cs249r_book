# V1-02 Track Plan: Physics of Deployment

Status: Wave 1 concept-module audit packet. This plan preserves the current
pilot concept-module implementation and patches gaps instead of redesigning the
lab structure.

Owned notebook: `labs/vol1/lab_02_ml_systems.py`

## Chapter Invariant

ML systems are simultaneous data, algorithm, and machine systems. Changing one
physical amount changes which constraint binds.

For this chapter, the durable lesson is that deployment is not a late packaging
choice. Memory, compute, bandwidth, energy, power, latency, and cost form an
operating envelope. The same model and data become a different system when that
envelope changes.

## Reading Map

| Lab module | Chapter anchor | Claim or formula used |
|---|---|---|
| Opening | `Deployment Paradigm Framework` and chapter purpose | Physical constraints determine where an ML model can run; D-A-M axes jointly determine what is possible. |
| Part A | `Physical Constraints: Why Paradigms Exist` | Physical laws create hard feasibility boundaries; amounts have units and budgets. |
| Part B | `Analyzing Workloads` and `The bottleneck principle` | Iron Law: `T = D / BW + O / R + L`; optimizing a non-binding term gives limited speedup. |
| Part C | `Paradigm Selection` and quantitative trade-off analysis | Each paradigm is a distinct operating envelope, so the same workload fails differently by context. |
| Part D | `Decision framework` and `Hybrid Architectures` | A valid deployment satisfies all constraints together and may need hybrid placement to mitigate the binding wall. |
| Synthesis | `Summary`, `Fallacies and Pitfalls`, and Lab 03 bridge | No single paradigm solves all ML problems; the selected design and binding constraint become workflow requirements. |

## Concept Inventory

Accepted concepts:

| Concept | Module | Why accepted |
|---|---|---|
| Amounts have units and budgets | Part A | Students must compare actual resource demand with physical limits before expressing preference. |
| The Iron Law decomposes latency | Part B | Students must identify whether compute, memory/bandwidth, network, or fixed overhead is worth optimizing. |
| Context changes the binding axis | Part C | Students must see iPhone, Oura Ring, RoboTaxi, and Cloud Fleet produce different first walls. |
| A valid design lives inside an operating envelope | Part D | Students must choose a placement plus mitigation that satisfies all constraints at once. |
| Binding constraint carries forward | Synthesis | Lab 03 uses the selected design and first wall as workflow inputs. |

Rejected or deferred concepts:

| Concept | Reason |
|---|---|
| Pareto frontier optimization | Deferred to later optimization/compression labs; this lab is about feasibility before preference. |
| Roofline analysis | Deferred to hardware acceleration; adding roofline here would distract from deployment envelopes. |
| P99/P999 latency histograms | Deferred to benchmarking and serving labs; V1-02 uses latency budgets and first-wall diagnosis. |
| Detailed fleet scheduling | Deferred to Volume II; Cloud Fleet appears here as a deployment envelope, not a scheduler. |
| Model compression recipes | Deferred to Lab 10; mitigation may mention simplification or quantization but does not teach compression mechanics. |

## Concept Modules

### Part A: Amounts Have Units And Budgets

Chapter claim: physical constraints decide which deployments are possible before
preference or model ambition matters.

Track lens:

- Primary lens: selected student track.
- Stakeholder: track-specific profile stakeholder.
- Decision: decide whether the default placement fits the selected track's
  memory, flash/OTA, latency, energy, power, bandwidth, and cost limits.

Student prior:

- Expected belief: cloud, faster hardware, or preference can usually solve the
  deployment problem.
- Productive failure: a preferred placement is infeasible because one unitful
  budget is already over the limit.

Storyline beats:

1. Scenario: stakeholder and release request are shown from the track variant.
2. Prediction: radio locks the predicted first wall.
3. Manipulation: workload slider plus placement selector.
4. Evidence: headroom table with value, limit, unit, feasibility, and first wall.
5. Consequence: reversible failure card names the violated amount and recovery
   action when any constraint fails.
6. Math Peek/source: `max(value_i / limit_i) <= 1` and
   `evaluate_deployment_envelope()` / `sweep_deployment_knob()` trace.
7. Checkpoint: saves prediction, measured wall, workload value, and placement.

Mechanics:

- Controls: `mo.ui.radio`, workload slider, placement dropdown.
- Evidence: constraint headroom table, sweep crossing, failure card.
- Failure state: workload or placement can push the envelope into a visible wall,
  then recover by reducing workload or changing placement.

Ledger output:

- `partA_predicted_wall`
- `partA_actual_wall`
- `partA_workload_value`
- `partA_placement_id`

### Part B: The Iron Law Decomposes Latency

Chapter claim: end-to-end latency is the sum of data movement, computation, and
fixed overhead; the useful optimization is the one that touches the active term.

Track lens:

- Primary lens: selected student track.
- Decision: decide whether a compute upgrade or offload change actually reduces
  total system latency.

Student prior:

- Expected belief: a 2x compute upgrade gives nearly 2x lower latency.
- Productive failure: total latency barely moves because memory/bandwidth,
  network, or fixed overhead remains binding.

Storyline beats:

1. Scenario: stakeholder proposes faster compute or offload.
2. Prediction: radio locks the expected system-level speedup class.
3. Manipulation: compute multiplier, memory/bandwidth multiplier, placement.
4. Evidence: Plotly waterfall decomposes compute, memory/bandwidth,
   placement/network, and overhead terms.
5. Consequence: reveal card compares baseline and upgraded latency, actual
   speedup, and active term.
6. Math Peek/source: `T = D / BW + O / R + L`; source trace marks the waterfall
   as a chapter-model approximation built from the envelope result.
7. Checkpoint: saves predicted speedup class, actual speedup, and active term.

Mechanics:

- Controls: `mo.ui.radio`, two sliders, placement dropdown.
- Evidence: latency waterfall and term ledger.
- Failure/boundary: compute-only upgrades can fail pedagogically by leaving a
  non-compute term binding.

Ledger output:

- `partB_predicted_speedup`
- `partB_actual_speedup`
- `partB_active_term`

### Part C: Context Changes The Binding Axis

Chapter claim: each deployment paradigm is a distinct operating envelope, so the
same feature has different first walls across contexts.

Track lens:

- Primary lens: all four tracks compared side by side.
- Decision: identify which track has the least normalized headroom for the same
  normalized stress and placement strategy.

Student prior:

- Expected belief: the best model or placement is universal.
- Productive failure: a strategy that survives one track fails on another
  because the binding axis moved.

Storyline beats:

1. Scenario: release review compares the same feature across iPhone, Oura Ring,
   RoboTaxi, and Cloud Fleet.
2. Prediction: radio locks the expected tightest track.
3. Manipulation: normalized stress slider and comparable placement strategy.
4. Evidence: all-track envelope table plus active-track sweep plot.
5. Consequence: reveal card names the tightest track and first walls by track.
6. Math Peek/source: `headroom_i = (limit_i - value_i) / limit_i` with helper
   source trace.
7. Checkpoint: saves tightest track, first walls by track, and worst headroom by
   track.

Mechanics:

- Controls: `mo.ui.radio`, stress slider, placement-strategy dropdown.
- Evidence: comparison table, threshold plot, first-wall dictionary.
- Boundary: active-track sweep marks the crossing where the selected track fails.

Ledger output:

- `partC_tightest_track`
- `partC_first_walls_by_track`
- `partC_worst_headroom_by_track`

### Part D: A Valid Design Lives Inside An Operating Envelope

Chapter claim: a shippable design must satisfy all constraints simultaneously
and carry a mitigation for the binding wall.

Track lens:

- Primary lens: selected student track.
- Decision: pick one placement and mitigation under stress, then explain the
  residual risk.

Student prior:

- Expected belief: once the wall is found, choose the fastest placement.
- Productive failure: the fastest path can still violate another constraint or
  introduce a new operational risk.

Storyline beats:

1. Scenario: stakeholder asks for one shippable deployment choice.
2. Prediction: radio locks the placement expected to survive stress.
3. Manipulation: stress workload, design placement, mitigation selector.
4. Evidence: placement review table and before/after mitigation table.
5. Consequence: reveal and failure cards show whether the design is shippable
   and name residual risk.
6. Math Peek/source: conjunction of all constraint checks plus
   `deployment_mitigation()` trace.
7. Checkpoint: saves placement, binding constraint, mitigation, and residual
   risk.

Mechanics:

- Controls: `mo.ui.radio`, stress slider, placement dropdown, mitigation dropdown.
- Evidence: feasibility table before and after mitigation.
- Failure state: stress setting can create a reversible constraint violation;
  mitigation and placement change the review result without hiding residual risk.

Ledger output:

- `partD_placement_id`
- `partD_binding_constraint`
- `partD_mitigation`
- `partD_recommended_mitigation`
- `residual_risk`

### Synthesis: Selected Design Carries To Lab 03

Chapter invariant restated: changing a physical amount changes which constraint
binds, and the binding constraint determines the deployment design.

Student product:

1. Three takeaways tied to the chapter invariant.
2. Prediction-vs-measurement comparison from Part A.
3. Lab 03 pointer: workflow design must expose the selected design's binding
   constraint before release review.
4. Local report export using `build_lab_report()` and `report_export_panel()`.

Future ledger use:

- Lab 03 can read `track_id`, `scenario_id`, selected placement, binding
  constraint, mitigation, residual risk, and first-wall evidence.

## Track Narratives And Required Differences

All four tracks remain selectable because context shift is the point of V1-02.
They differ in persona, constraints, thresholds, consequences, and report
framing through `get_track_profile()` plus
`get_lab_track_variant("v1_02_physics_of_deployment", track_id)`.

| Track | Persona | Constraint emphasis | Threshold/workload | Consequence | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile product engineer | Thermal power, battery, on-device latency, unified memory, privacy | Sustained frame rate in FPS; 33 ms latency, 4.5 W power, 0.85 mJ energy, 512 MB memory budget | Thermal soak or battery drain forces frame-rate, quality, or local-fast-path fallback | Local UX and privacy memo: defend whether on-device, cloud, or prefilter/cloud fallback preserves responsiveness. |
| Oura Ring | Wearable firmware engineer | SRAM, flash/OTA, duty-cycle energy, BLE/phone availability | Classification windows per minute; 0.512 MB memory, 1.2 MB flash/OTA, 0.12 mJ energy, 0.018 W power | Firmware or energy wall blocks always-on sensing unless cadence, window, or OTA payload is reduced | Firmware envelope memo: defend ring-only, phone handoff, or summary upload without breaking always-on sensing. |
| RoboTaxi | Autonomous vehicle platform engineer | Vehicle-local p99 latency, sensor bandwidth, power, safety margin, reliability | Active sensor streams; 35 ms latency, 180 GB/s bandwidth, 60 W power, 8192 MB memory | Safety-critical loop cannot rely on cloud; latency or sensor bandwidth miss becomes a safety-case failure | Safety-path design memo: defend local, roadside assist, or fleet cloud placement while keeping control local. |
| Cloud Fleet | Fleet service owner | SLA, memory bandwidth, utilization, cost/request, power/carbon | Requests per second per GPU; 120 ms latency, 2500 GB/s bandwidth, 650 W power, $0.040 per 1K requests | Cost/SLA/bandwidth wall forces batching, caching, regional placement, or tiering | Fleet service memo: defend central GPU, regional cache, or batch queue under SLA and cost evidence. |

## Mechanics Plan

Opening belt:

- Track selector seeded from Design Ledger.
- Header, learning objectives, reading connection, track context, and track arc.

Prediction belt:

- One structured radio prediction per concept module.
- Evidence is gated behind the prediction value in each part.

Control belt:

- Part A: workload and placement.
- Part B: compute multiplier, memory/bandwidth multiplier, placement.
- Part C: normalized stress and comparable placement strategy.
- Part D: stress workload, placement, mitigation.

Evidence belt:

- Part A: constraint headroom table and first crossing.
- Part B: latency waterfall and term ledger.
- Part C: all-track table plus active-track sweep.
- Part D: placement feasibility table plus mitigation review.

Failure belt:

- `v1_02_failure_card()` names value, limit, unit, and recovery action.
- Part A and Part D expose reversible failure states through workload and
  placement controls.

Source belt:

- Math Peek appears in every part.
- `source_trace()` names helper API, profile, hardware/model references, and
  whether a calculation is a helper result or a notebook-local approximation.

Decision and ledger belt:

- Each part writes checkpoint fields.
- Final HUD saves only after all four predictions are complete.
- Report export serializes predictions, knobs, evidence, decision, residual
  risk, and source trace.

## Evidence And Ledger Plan

Evidence required from the notebook:

| Evidence | Current implementation |
|---|---|
| Prediction-vs-actual overlay | Reveal cards in Parts A-D. |
| Boundary or failure state | Part A and Part D failure cards; Part C threshold plot. |
| Value, limit, unit, and threshold | Constraint tables and failure cards. |
| Chapter formula/source connection | Math Peek plus `source_trace()` per part. |
| Design decision | Part D placement/mitigation checkpoint and synthesis. |
| Future-lab handoff | Ledger design and report snapshot. |

Ledger schema:

```json
{
  "track_id": "...",
  "scenario_id": "...",
  "partA_predicted_wall": "...",
  "partA_actual_wall": "...",
  "partA_workload_value": 0.0,
  "partA_placement_id": "...",
  "partB_predicted_speedup": "...",
  "partB_actual_speedup": 0.0,
  "partB_active_term": "...",
  "partC_tightest_track": "...",
  "partC_first_walls_by_track": {},
  "partC_worst_headroom_by_track": {},
  "partD_placement_id": "...",
  "partD_binding_constraint": "...",
  "partD_mitigation": "...",
  "partD_recommended_mitigation": "...",
  "residual_risk": "...",
  "completed": true
}
```

## Notebook Depth Audit

Depth-gate result after reading the current notebook:

| Module | Scenario | Prediction | Manipulation | Evidence | Consequence/failure | Math/source | Checkpoint | Result |
|---|---|---|---|---|---|---|---|---|
| Part A | Pass | Pass | Pass | Pass | Pass: reversible envelope failure card | Pass | Pass | Meets concept-module depth gate. |
| Part B | Pass | Pass | Pass | Pass | Pass: non-bottleneck speedup reveal | Pass | Pass | Meets concept-module depth gate. |
| Part C | Pass | Pass | Pass | Pass | Pass: tightest-track reveal plus threshold plot | Pass | Pass | Meets concept-module depth gate; wording should keep consequence explicit. |
| Part D | Pass | Pass | Pass | Pass | Pass: shippability and residual-risk review | Pass | Pass | Meets concept-module depth gate. |
| Synthesis | Pass | n/a | n/a | Pass | Pass: carries binding constraint to Lab 03 | n/a | Ledger/report | Meets handoff requirement. |

Rubric score:

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability |
|---|---:|---:|---:|---:|---:|---:|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 |
| Part B | 3 | 3 | 2 | 3 | 3 | 3 |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 |

Acceptance notes:

- No module has a score below 2.
- At least one reversible failure state is present.
- Every part has structured prediction, manipulation, Math Peek/source model,
  evidence, consequence, and checkpoint.
- Part B track specificity is lower than the other modules because the Iron Law
  mechanics are intentionally shared; the placement terms and selected track
  profile still change the numbers and active term.

## Implementation Risks And Guardrails

Preserve:

- Existing WASM bootstrap and local wheel paths.
- Local helper prefix `v1_02_`.
- Shared helper contracts in `mlsysbook_labs.deployment`, `tracks`, `variants`,
  and `reports`.
- `mo.ui.tabs` structure and Design Ledger/report export.

Risks:

- `LatencyWaterfall` is not used directly; the notebook uses a Plotly waterfall
  because the Part B model is an envelope-derived chapter approximation rather
  than an `Engine.solve()` performance profile. The source trace must keep that
  label visible.
- Track thresholds live in variant metadata. Do not move them into notebook
  constants.
- Other workers may edit other lab files in parallel; keep V1-02 edits scoped to
  this owned notebook and plan.

Minimum verification:

```bash
python3 -m py_compile labs/vol1/lab_02_ml_systems.py
```
