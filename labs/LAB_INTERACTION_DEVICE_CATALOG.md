# Lab Interaction Device Catalog

This catalog defines the interaction tools the MLSysBook labs should use. The goal is not to add widgets randomly. Every device should serve a pedagogical purpose: help students reason like ML systems engineers.

The labs should feel rich and responsive, but the interaction design should be disciplined. A control or visual belongs in a lab only if it helps answer one of these questions:

- What knob am I moving?
- Which constraint changes?
- Which metric improves?
- Which metric gets worse?
- What decision would I defend?
- What assumption would invalidate that decision?

## Local Marimo Capability Check

The current local environment has Marimo `0.23.1`. The following primitives are available and appropriate for the lab system:

| Capability | Available | Main use |
|---|---:|---|
| `mo.ui.slider` | yes | Continuous numeric knob |
| `mo.ui.range_slider` | yes | Interval, budget range, threshold range |
| `mo.ui.number` | yes | Numeric prediction or exact value |
| `mo.ui.radio` | yes | Exclusive prediction or strategy |
| `mo.ui.dropdown` | yes | Hardware/model/region/runtime selection |
| `mo.ui.multiselect` | yes | Choose several defenses, metrics, or workloads |
| `mo.ui.checkbox` | yes | Include/exclude one option |
| `mo.ui.switch` | yes | Enable/disable a policy |
| `mo.ui.tabs` | yes | Switch between nuggets or views |
| `mo.ui.form` | yes | Commit a grouped decision |
| `mo.ui.dataframe` | yes | Inspect tabular sweep results |
| `mo.ui.table` | yes | Compact static or computed tables |
| `mo.ui.plotly` | yes | Interactive charts, maps, frontiers, rooflines |
| `mo.ui.altair_chart` | yes | Declarative charts where useful |
| `mo.ui.text_area` | yes | Short rationale or reflection |
| `mo.ui.button` | yes | Save snapshot, reset, compare, export |
| `mo.ui.file` | yes | Optional import of a ledger or scenario |
| `mo.ui.array` / `mo.ui.batch` | yes | Repeated or grouped controls |

Marimo also supports Markdown/HTML composition, `mo.callout`, `mo.accordion`, `mo.hstack`, and `mo.vstack`, which are useful for explanation, source trace, and layout.

## Device Card Template

Every reusable lab device should have a card in `mlsysbook_labs` with this metadata:

| Field | Purpose |
|---|---|
| Device name | Human-readable component name |
| Pedagogical use | Why this device exists |
| Best for | The type of concept or trade-off it teaches |
| Avoid when | Misuse cases |
| Student prompt | How the lab frames the interaction |
| Required labels | Units, bounds, default, and meaning |
| MLSysIM inputs | Which solver/scenario/workload parameters it changes |
| MLSysIM outputs | Which result fields it displays |
| Reflection hook | The prompt that turns interaction into reasoning |
| Accessibility notes | Keyboard, contrast, color-independent state, mobile behavior |

## Critical Thinking Lens

Each nugget should pair one or more devices with a critical thinking lens:

| Lens | Student behavior | Good devices |
|---|---|---|
| Bottleneck diagnosis | Identify what fails first | Constraint budget, heatmap, roofline, budget stack |
| Trade-off reasoning | Explain what improves and what worsens | Slider + curve, Pareto scatter, stacked bars |
| Systems comparison | Compare track, hardware, model, runtime, or policy choices | Strategy selector, comparison table, grouped bars |
| Scaling intuition | See nonlinear growth or collapse | Log-scale line chart, queueing chart, reliability curve |
| Policy judgment | Choose controls under conflicting goals | Checklists, switches, decision card, frontier |
| Geographic/placement reasoning | Understand location, grid, network, or latency effects | World map, region selector, cost/carbon table |
| Architecture reasoning | Understand dataflow, topology, pipelines, and communication | Network diagram, Sankey/pipeline, topology map |
| Reflection | Convert observed behavior into explanation | Reflection card, short rationale, residual-risk field |

## Control Devices

### Slider Card

Use when:

- A single continuous knob changes system behavior.
- Students need to feel a threshold, cliff, or nonlinear transition.

Examples:

- Batch size.
- Sequence length.
- Model width.
- Utilization.
- Arrival rate.
- Compression ratio.
- Carbon cap.

Required elements:

- Clear label with units.
- Default value selected from the track scenario.
- Bounds that represent meaningful engineering limits.
- Tick labels or helper text for important thresholds.
- Live output preview of the value and the linked metric.
- Reflection prompt when the value crosses a constraint boundary.

Avoid when:

- The values are categorical.
- The range is arbitrary.
- More than three sliders are needed at once.

MLSysIM contract:

- Input: one numeric scenario/workload/model parameter.
- Output: updated `BottleneckReport`, `ConstraintBudget`, or `TradeoffSweepResult`.

Pedagogical prompt:

> Move the batch size until the active constraint changes. What changed first: compute, memory, queueing, or latency?

### Range Slider Card

Use when:

- Students need to compare feasible intervals.
- A budget or threshold has lower and upper bounds.

Examples:

- Acceptable latency range.
- Carbon cap interval.
- Privacy epsilon range.
- Model size search range.

Required elements:

- Explain both endpoints.
- Show feasible and infeasible regions.
- Avoid hidden units.

MLSysIM contract:

- Input: two numeric bounds.
- Output: feasible interval, violated constraint, frontier subset.

### Number Prediction Card

Use when:

- The student should estimate magnitude before seeing the result.

Examples:

- How many GPUs are needed?
- How many seconds will a checkpoint take?
- What is the p99 latency?
- How much memory does the KV cache require?

Required elements:

- Unit.
- Expected order of magnitude.
- Reveal after compute with actual value and error factor.

MLSysIM contract:

- Input: prediction value.
- Output: comparison to computed result.

Pedagogical prompt:

> Estimate the memory required before running the solver. After the reveal, explain why your estimate was high or low.

### Radio / Segmented Strategy Selector

Use when:

- Students must choose one strategy from a small set.

Examples:

- Local, edge, or cloud deployment.
- Ring or tree all-reduce.
- Eager, graph, or compiled runtime.
- Quantization, pruning, or distillation.

Required elements:

- Two to five choices.
- Each choice should represent a real engineering option.
- Include a short trade-off label, not just a name.

MLSysIM contract:

- Input: categorical strategy.
- Output: scenario variant or solver branch.

### Dropdown Selector

Use when:

- The option set is larger or secondary.

Examples:

- Hardware target.
- Model family.
- Datacenter region.
- Network fabric.
- Runtime backend.

Required elements:

- Group related options if possible.
- Use defaults from the selected track.
- Include concise descriptions in adjacent helper text.

Avoid when:

- The choice is the central decision. Use radio/segmented control instead.

### Checkbox / Multiselect Stack Builder

Use when:

- Students can combine multiple interventions.

Examples:

- Defense stack.
- Compression recipe.
- Monitoring signals.
- Optimization passes.
- Responsible AI controls.

Required elements:

- Show cumulative cost as choices are added.
- Mark incompatible choices.
- Show diminishing returns or interaction effects.

MLSysIM contract:

- Input: set of enabled interventions.
- Output: stacked overhead, new bottleneck, interaction warning.

Pedagogical prompt:

> Add controls until the risk target is met. Which control contributed the most overhead, and which one had the least effect?

### Switch

Use when:

- A binary policy changes the system path.

Examples:

- Enable carbon cap.
- Enable activation checkpointing.
- Enable autoscaling.
- Enable secure aggregation.

Required elements:

- State what changes when enabled.
- Show before/after metric change.

Avoid when:

- There are more than two meaningful states.

### Form / Commit Panel

Use when:

- A set of controls should be treated as one decision.

Examples:

- Final serving policy.
- Compression recipe.
- Benchmark protocol.
- Governance pipeline.

Required elements:

- Summary of selected knobs.
- Primary metric.
- Guardrail metric.
- Residual risk field.

MLSysIM contract:

- Input: grouped configuration.
- Output: saved ledger snapshot and result hash/summary.

## Visual Devices

### Constraint Budget

Use when:

- Students need to see which resources are close to violation.

Best for:

- Memory, latency, energy, cost, carbon, reliability, privacy budget.

Required elements:

- Current value.
- Limit.
- Headroom.
- Status.
- Active constraint.

MLSysIM contract:

- Consumes `ConstraintBudget` and `BottleneckReport`.

### Stacked Bar / Budget Stack

Use when:

- A total is composed of meaningful parts.

Examples:

- Training memory: weights, activations, gradients, optimizer state.
- Latency: compute, memory, dispatch, queueing.
- Cost: capex, energy, maintenance, carbon.

Pedagogical use:

- Shows where the budget is actually going.

### Line Curve

Use when:

- One knob is swept and the student needs to see growth, threshold, or collapse.

Examples:

- Sequence length versus memory.
- Utilization versus p99 latency.
- Fleet size versus reliability.
- Batch size versus throughput.

Required elements:

- Mark the current operating point.
- Mark constraint boundaries.
- Use log scale when growth is multiplicative or spans orders of magnitude.

### Pareto Frontier Scatter

Use when:

- Students must choose among trade-offs with no single best answer.

Examples:

- Accuracy versus latency.
- Cost versus p99 latency.
- Carbon versus cost.
- Compression versus quality.

Required elements:

- Highlight dominated and non-dominated configurations.
- Let the student select a point.
- Show why the selected point is defensible or fragile.

MLSysIM contract:

- Consumes `ParetoFrontier`.

### Heatmap / Phase Diagram

Use when:

- Two knobs jointly determine feasibility or bottleneck regime.

Examples:

- Batch size x sequence length.
- GPUs x network bandwidth.
- Model size x precision.
- Arrival rate x replica count.

Required elements:

- Color by bottleneck or feasible/infeasible state.
- Add contour lines for major thresholds.
- Avoid using color alone; labels or tooltips must identify regime.

### Roofline Chart

Use when:

- The lab is about compute versus memory bandwidth.

Examples:

- Hardware acceleration.
- Performance engineering.
- Framework fusion.
- Operator design.

Required elements:

- Hardware roof.
- Workload point.
- Ridge point.
- Before/after optimization point.
- Bottleneck label.

MLSysIM contract:

- Consumes `PerformanceProfile`, arithmetic intensity, peak flops, peak bandwidth.

### Queueing Chart

Use when:

- Utilization, arrival rate, burstiness, or concurrency drives tail latency.

Examples:

- Serving.
- Benchmarking.
- Orchestration.

Required elements:

- p50/p95/p99 latency.
- Utilization.
- Stability boundary.
- SLO line.

### Timeline

Use when:

- The key behavior unfolds over time.

Examples:

- Drift and monitoring.
- Thermal throttling.
- Retraining cadence.
- Checkpoint/recovery.
- Canary rollout.

Required elements:

- Event markers.
- Observed versus true state when relevant.
- Decision point.

### World Map / Region Map

Use when:

- Geography matters.

Examples:

- Carbon intensity.
- Datacenter placement.
- Latency to user population.
- Data residency/privacy.

Required elements:

- Region selector.
- Metric legend.
- Table fallback for exact values.
- Explain that map values are scenario inputs, not universal truths.

Implementation:

- Use Plotly `choropleth`, `scatter_geo`, or a simplified regional map.

### Network / Topology Diagram

Use when:

- Students need to understand communication structure.

Examples:

- Ring versus tree all-reduce.
- Fat tree versus torus.
- Device-edge-cloud placement.
- Pipeline stages.

Required elements:

- Nodes.
- Links.
- Bandwidth/latency labels.
- Active bottleneck link.
- Message volume.

Implementation:

- Use Plotly scatter/line, SVG-in-HTML, or a small custom component.

### Sankey / Pipeline Diagram

Use when:

- Bytes, tokens, requests, or work flow through stages.

Examples:

- Data pipeline.
- Serving request path.
- Training pipeline.
- Edge/cloud split.

Required elements:

- Stage capacity.
- Flow amount.
- Bottleneck stage.
- Stall or queue indicator.

### Data Table / Sweep Table

Use when:

- Exact comparison matters.

Examples:

- Hardware comparison.
- Benchmark protocol results.
- Top Pareto candidates.
- Scenario presets.

Required elements:

- Sortable key metric.
- Feasible status.
- Selected row.
- Unit-aware formatting.

## Pedagogical Devices

### Prediction Lock

Use when:

- Students should commit to an intuition before seeing simulator output.

Required elements:

- Structured radio or numeric prediction.
- Reveal actual result after prediction.
- Show error factor or wrong assumption.

Avoid:

- Do not block every minor chart; prediction locks should mark meaningful learning moments.

### Source Trace / Math Peek

Use when:

- Students need to connect the lab to the book and MLSysIM.

Required elements:

- Equation or model name.
- Key constants.
- MLSysIM function or result field.
- Chapter principle.

This should be a compact expandable section, not a wall of derivation.

### Failure Boundary Callout

Use when:

- The student crosses a meaningful constraint boundary.

Examples:

- OOM.
- SLA violation.
- Unstable queue.
- Carbon cap exceeded.
- Privacy budget depleted.

Style:

- Professional warning, not a dramatic animation.
- Explain the failure and the first mitigation to try.

### Reflection Card

Use when:

- The student has observed a result and needs to interpret it.

Required elements:

- One diagnosis question.
- One trade-off question.
- One residual-risk question.

Example:

> Which metric improved? Which metric got worse? What assumption would make your decision fail?

### Decision Card

Use when:

- The nugget or lab asks the student to choose a configuration.

Required elements:

- Selected configuration.
- Primary metric.
- Guardrail metric.
- Binding constraint.
- Rationale.
- Residual risk.
- Save-to-ledger action.

### Before / After Comparison

Use when:

- Students apply an optimization or intervention.

Examples:

- Before/after fusion.
- Before/after compression.
- Before/after autoscaling.
- Before/after monitoring policy.

Required elements:

- Baseline ghost marker.
- New value.
- Delta.
- New bottleneck.

### Nugget Navigator

Use when:

- A lab has multiple nuggets.

Required elements:

- Current nugget.
- Completed nuggets.
- Chapter idea label.
- Final synthesis state.

Implementation:

- `mo.ui.tabs` can be used for nugget navigation, but the visual label should be "Nugget 1", "Nugget 2", etc., with concept names.

## Device Selection Matrix

| Teaching goal | Preferred device combination |
|---|---|
| Show a threshold or cliff | Slider + line curve + constraint boundary |
| Show bottleneck regime | Constraint budget + heatmap or roofline |
| Show multi-objective trade-off | Strategy selector + Pareto frontier + decision card |
| Show cumulative overhead | Checklist + stacked bar + reflection card |
| Show scale collapse | Slider + log curve + failure boundary callout |
| Show geographic placement | Region dropdown + world map + cost/carbon table |
| Show communication structure | Topology diagram + line/bar comparison + source trace |
| Show policy design | Switches/checklist + budget stack + decision card |
| Show exact comparison | Dropdowns + sweep table + selected row summary |
| Build synthesis | Ledger timeline + sensitivity heatmap + final decision card |

## Standard Nugget Layout

Each nugget should use a consistent screen structure:

1. Nugget header: concept, book chapter key idea, systems question.
2. Scenario strip: track, model, hardware, workload, constraints.
3. Prediction card.
4. Control panel.
5. Main trade-off poster.
6. Constraint budget and bottleneck explanation.
7. Source trace / math peek.
8. Reflection and decision card.

Recommended desktop layout:

- Left: scenario and controls.
- Center: main trade-off poster.
- Right: budget, explanation, reflection, decision.

Recommended mobile layout:

- Scenario.
- Prediction.
- Controls.
- Poster.
- Budget/explanation.
- Reflection/decision.

## Device Data Contract

Every interactive device should be backed by structured metadata:

```python
DeviceSpec(
    device_id="v1_05_activation_batch_slider",
    nugget_id="activation_memory_cliff",
    device_type="slider",
    label="Batch size",
    unit="samples",
    default=8,
    min=1,
    max=128,
    step=1,
    mlsysim_param="workload.batch_size",
    pedagogical_intent="Reveal when activations exceed memory headroom.",
    watched_outputs=["memory_utilization", "binding_constraint", "latency"],
    reflection_prompt="What changed first as batch size increased?"
)
```

This enables:

- Consistent rendering.
- Automatic tests.
- Better instructor guides.
- Easier migration from notebook-local controls to MLSysIM-backed controls.

## Components To Build In `mlsysbook_labs`

| Component | Purpose |
|---|---|
| `lab_header()` | Professional volume/chapter header |
| `chapter_anchor()` | Suggested reading and key chapter ideas |
| `chapter_recap()` | Self-contained mini recap of chapter emphasis and systems translation |
| `nugget_shell()` | Standard nugget layout |
| `nugget_navigator()` | Shows progress through multiple nuggets |
| `scenario_strip()` | Track, model, hardware, workload, constraints |
| `slider_card()` | Pedagogical slider with units, bounds, and threshold text |
| `strategy_selector()` | Radio/segmented strategy choice |
| `stack_builder()` | Multiselect/checklist with cumulative overhead |
| `constraint_budget()` | Unified budget view |
| `tradeoff_poster()` | Wrapper for frontier/curve/heatmap/roofline/map |
| `source_trace()` | Equation, constants, MLSysIM provenance |
| `reflection_card()` | Structured diagnosis/trade-off/risk reflection |
| `decision_card()` | Final choice and ledger save |
| `report_export()` | Downloadable Markdown report plus optional JSON snapshot |
| `advanced_knob_drawer()` | Collapsible optional controls for deeper experiments |
| `instructor_metadata()` | Teaching goals, misconceptions, prompts, and rubric hints |
| `adoption_pack()` | Assignment text, report expectations, adoption modes, and setup notes |
| `compare_snapshot()` | Before/after comparison |
| `ledger_timeline()` | Shows decisions carried forward |

## Tests And Guardrails

Add tests that enforce:

- Every nugget has a prediction, trade-off visual, reflection, and decision or synthesis output.
- Every slider has units, bounds, default, and linked MLSysIM parameter.
- Every chart consumes a typed MLSysIM result or sweep result.
- Every map has a table fallback.
- Every failure state has a mitigation hint.
- Every decision card saves track, scenario, selected configuration, binding constraint, rationale, and residual risk.
- Every lab can generate a downloadable report from the ledger and typed MLSysIM result snapshot.
- Every lab has a chapter recap that can orient a student who has not just read the chapter.
- Every lab has instructor adoption metadata: why assign it, where it fits, what to collect, and how to grade it.
- Advanced knobs are hidden by default and are not required to complete the report.
- No new lab uses inline dark gradient headers.
- No notebook-local formula duplicates an MLSysIM API without an explicit reason.

## Recommended First Device Pilots

Build these first:

1. `lab_header()`
2. `chapter_anchor()`
3. `nugget_shell()`
4. `slider_card()`
5. `constraint_budget()`
6. `tradeoff_poster()`
7. `reflection_card()`
8. `decision_card()`

Then pilot them in:

- V1-01 for triad diagnosis and intervention frontier.
- V1-05 for operator/activation memory cliffs.
- V2 communication or inference for topology, queueing, and scale trade-offs.
