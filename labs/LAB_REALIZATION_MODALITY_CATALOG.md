# Lab Realization Modality Catalog

This catalog defines the reusable modalities used to turn a lab plan into an actual interactive lab. The track plan says what the lab teaches. This catalog says how the lab can realize that teaching through controls, plots, evidence, and report artifacts.

The goal is a small set of repeatable building blocks. A lab should not invent a new interaction pattern when an existing modality communicates the idea clearly.

## Design Rule

Every lab should be built from three layers:

1. **Pedagogy kernel** - the invariant concept, prediction, evidence, reflection, and decision.
2. **Track realization** - iPhone, Oura Ring, RoboTaxi, and Cloud Fleet narratives, defaults, hardware, metrics, and thresholds.
3. **Modality stack** - the controls, plots, tables, cards, and report artifacts that make the concept visible.

Hardware facts come from MLSysIM. Scenario facts come from track/lab variants. Modalities render the result; they do not own constants.

## Canonical Track Profiles

| Track | Category | Hardware source | Typical primary metrics |
|---|---|---|---|
| iPhone | Mobile ML | `Hardware.Mobile.iPhone15Pro` | Battery drain, thermal headroom, on-device latency, memory |
| Oura Ring | TinyML / wearable | `Hardware.Tiny.OuraRing` | SRAM/flash fit, battery life, OTA payload, sampling cadence |
| RoboTaxi | Edge AI | `Hardware.Edge.RoboTaxi` | p99/p999 latency, rare-event recall, power, reliability |
| Cloud Fleet | Cloud/Fleet | Cloud fleet profile backed initially by `Hardware.Cloud.H100` | Throughput, p99 latency, cost/request, utilization, carbon |

## Opening Modalities

The student-facing opening headers are fixed by `LAB_STRUCTURE_AND_REPORT_CONTRACT.md`: `Learning Objectives`, `Chapter Recap`, `Your Track`, `Scenario Brief`, and `Lab Map`.

### Track Selector

Use for:
- Lab 00 and the top of every track-aware lab.
- Selecting the canonical system identity.

Required behavior:
- Four choices only: iPhone, Oura Ring, RoboTaxi, Cloud Fleet.
- Shows category, hardware source, and dominant constraints.
- Reads prior selection from the Design Ledger when available.
- Saves selection back to the ledger.

Output contract:
- `track_id`
- `category`
- `hardware_ref`
- `scenario_id`
- `primary_constraints`

### Chapter Recap

Use for:
- Every lab opening.

Required behavior:
- One compact recap of the chapter idea.
- One systems translation: why the idea matters in deployed ML.
- One common trap.

Output contract:
- Display-only, but should be backed by `ChapterRecap`.

### Scenario Strip

Use for:
- Every part/nugget.

Required behavior:
- Track, stakeholder, hardware, workload, model, objective, primary metric, guardrail metric.
- Short enough to scan before interacting.

Output contract:
- `track_id`
- `stakeholder`
- `hardware_ref`
- `model_ref`
- `workload_summary`
- `objective`
- `primary_metric`
- `guardrail_metric`

## Prediction Modalities

### Multiple-Choice Prediction Lock

Use when:
- The student must choose one conceptual outcome before seeing evidence.

Examples:
- Which compression method wins?
- Which resource binds first?
- Which deployment placement is feasible?

Required behavior:
- 3-5 options.
- Locked before reveal.
- Reveals actual outcome and explains the misconception.

Output contract:
- `prediction_id`
- `selected_option`
- `correct_option`
- `reason`

### Numeric Prediction Lock

Use when:
- Magnitude matters.

Examples:
- Memory footprint.
- p99 latency.
- Battery drain.
- OTA payload size.

Required behavior:
- Unit-bearing input.
- Shows actual value and error factor.
- Explains why the order of magnitude was high or low.

Output contract:
- `prediction_id`
- `predicted_value`
- `unit`
- `actual_value`
- `error_factor`

## Control Modalities

### Primary Knob Slider

Use when:
- One continuous variable reveals a threshold or cliff.

Examples:
- Bit width.
- Sparsity.
- Batch size.
- Sampling cadence.
- Request rate.

Required behavior:
- Unit, bounds, default, step.
- Default comes from track variant.
- Shows linked outputs live.

Output contract:
- `knob_id`
- `mlsysim_param`
- `value`
- `unit`
- `watched_outputs`

### Strategy Selector

Use when:
- Student chooses one engineering strategy.

Examples:
- Quantization, pruning, distillation.
- Local, edge, cloud, hybrid.
- Ring, tree, hierarchy.

Required behavior:
- 2-5 options.
- Each option includes a short trade-off label.

Output contract:
- `strategy_id`
- `selected_strategy`
- `available_strategies`
- `unsupported_warnings`

### Stack Builder

Use when:
- Student composes several interventions.

Examples:
- Compression recipe.
- Defense stack.
- Monitoring controls.
- Optimization passes.

Required behavior:
- Cumulative benefit and cumulative cost.
- Compatibility warnings.
- Diminishing return or interaction notes.

Output contract:
- `stack_id`
- `selected_items`
- `cumulative_costs`
- `cumulative_benefits`
- `interaction_warnings`

## Visual Modalities

### Constraint Budget

Use when:
- The student must see which resource is close to violation.

Best for:
- Memory, flash, SRAM, latency, battery, power, cost, carbon, reliability.

Required behavior:
- Current value, limit, headroom, status, binding constraint.
- Units visible.
- Color is not the only signal.

Output contract:
- `ConstraintBudget`
- `BottleneckReport`

### Pareto Frontier

Use when:
- There is no single best choice.

Best for:
- Accuracy versus latency.
- Quality versus memory.
- Cost versus p99.
- Carbon versus latency.

Required behavior:
- Dominated points visible or filterable.
- Selected point highlighted.
- Guardrail violations marked.
- Table fallback.

Output contract:
- `ParetoFrontier`
- `selected_candidate`
- `dominated`
- `violations`

### Phase Diagram / Heatmap

Use when:
- Two knobs jointly determine feasibility or bottleneck regime.

Best for:
- Bit width x sparsity.
- Batch size x sequence length.
- Request rate x replicas.
- Fleet size x MTBF.

Required behavior:
- Feasible/infeasible regions.
- Current operating point.
- Regime labels.
- Table fallback.

Output contract:
- `SensitivityGrid`
- `regime_labels`
- `current_point`

### Budget Stack

Use when:
- A total is made of meaningful parts.

Best for:
- Memory: weights, activations, optimizer, runtime.
- Latency: compute, dispatch, queueing, transfer.
- Energy: compute, radio, idle, sensing.
- Cost: compute, storage, network, operations.

Required behavior:
- Parts sum to total.
- Before/after comparison where relevant.

Output contract:
- `BudgetStack`
- `components`
- `total`

### Before/After Comparison

Use when:
- A student applies one intervention.

Best for:
- Compression before/after.
- Fusion before/after.
- Monitoring policy before/after.

Required behavior:
- Baseline marker.
- New value.
- Delta.
- New bottleneck.

Output contract:
- `baseline`
- `candidate`
- `deltas`
- `new_bottleneck`

### Timeline

Use when:
- Behavior unfolds over time.

Best for:
- Drift.
- Battery drain.
- OTA rollout.
- Canary.
- Checkpoint/recovery.

Required behavior:
- Events, decision points, observed versus true state when relevant.

Output contract:
- `TimelineEvent`
- `observed_metric`
- `true_metric`
- `decision_point`

### Topology / Pipeline Diagram

Use when:
- Structure matters more than a scalar.

Best for:
- Device-edge-cloud placement.
- Network fabric.
- Sensor-to-compute path.
- Data pipeline.

Required behavior:
- Nodes, links/stages, capacities, active bottleneck.

Output contract:
- `nodes`
- `edges`
- `stage_capacities`
- `active_bottleneck`

## Evidence Modalities

### Source Trace / Math Peek

Use when:
- The student needs to trust the numbers or connect to the book.

Required behavior:
- Equation or model name.
- Key constants.
- MLSysIM API or source.
- Track-specific assumption.

Output contract:
- `equation_id`
- `mlsysim_api`
- `hardware_ref`
- `scenario_assumptions`

### Failure Boundary Callout

Use when:
- The selected configuration crosses a hard limit.

Required behavior:
- Names the failed constraint.
- Shows value versus limit.
- Provides first mitigation to try.

Output contract:
- `failed_constraint`
- `value`
- `limit`
- `mitigation_hint`

## Reflection And Decision Modalities

### Reflection Card

Use for:
- Every nugget or part.

Required prompts:
- What improved?
- What worsened?
- What assumption would invalidate the decision?

Output contract:
- `diagnosis`
- `tradeoff`
- `residual_risk`

### Decision Card

Use for:
- End of a nugget or lab.

Required fields:
- Selected configuration or policy.
- Primary metric.
- Guardrail metric.
- Binding constraint.
- Rationale.
- Residual risk.
- Save-to-ledger action.

Output contract:
- `decision_id`
- `selected_configuration`
- `primary_metric`
- `guardrail_metric`
- `binding_constraint`
- `rationale`
- `residual_risk`

### Report Export

Use for:
- Every lab.

Required fields:
- Lab ID and version.
- Track ID.
- Scenario ID.
- Hardware ref.
- Predictions.
- Results.
- Decision.
- Residual risk.
- Timestamp.

Output contract:
- Markdown report.
- Optional JSON snapshot.

## Standard Part Recipe

Each lab part should usually follow this order:

1. `Part <Letter> - <Concept>` header with systems question.
2. `What You Need To Know` micro-brief.
3. `Scenario Slice`.
4. `Your Prediction`.
5. `Try It` controls.
6. `Evidence` visual/table.
7. `Constraint Check`.
8. `Source Trace`.
9. `Reflection`.
10. `Checkpoint` or `Decision`.

Not every part needs every element, but every lab should include all major elements at least once.

The formal structure and local report requirements live in `LAB_STRUCTURE_AND_REPORT_CONTRACT.md`.

## Modality Selection Matrix

| Teaching goal | Recommended modalities |
|---|---|
| Show a threshold | Numeric prediction, primary knob slider, line curve, failure boundary |
| Show multi-objective trade-off | Strategy selector, Pareto frontier, constraint budget, decision card |
| Show cumulative overhead | Stack builder, budget stack, before/after comparison |
| Show scale collapse | Slider, log curve, phase diagram, failure boundary |
| Show structure | Scenario strip, topology/pipeline diagram, source trace |
| Show drift over time | Timeline, monitoring controls, reflection card |
| Build final judgment | Ledger timeline, sensitivity heatmap, decision card, report export |

## Compression Lab Modality Recipe

Use for V1-10 and related compression pilots.

| Part | Common pedagogy | Primary modalities |
|---|---|---|
| Opening | Smaller is not automatically better | Track selector, chapter recap, scenario strip |
| Part A | Compression feasibility depends on target hardware | Multiple-choice prediction, strategy selector, constraint budget, before/after table |
| Part B | Compression is a frontier, not a single knob | Sliders for bit width/sparsity/student size, Pareto frontier, table fallback |
| Part C | A deployable recipe must include validation | Stack builder, guardrail checklist, decision card |
| Synthesis | Defend the recipe and residual risk | Report export, source trace, reflection card |

Track-specific metric mapping:

| Track | Primary metric | Guardrail metric | Visual emphasis |
|---|---|---|---|
| iPhone | Battery/thermal headroom | Quality and on-device latency | Battery/latency before-after plus Pareto frontier |
| Oura Ring | Flash/SRAM fit and OTA payload | Battery life and signal quality | Memory/OTA constraint budget plus phase diagram |
| RoboTaxi | p99 latency | Rare-event recall | p99/recall Pareto frontier plus failure boundary |
| Cloud Fleet | Cost/request or throughput | SLA and quality | Cost/SLA frontier plus utilization budget |
