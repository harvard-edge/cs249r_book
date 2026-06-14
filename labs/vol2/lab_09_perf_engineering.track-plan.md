# V2-09 Track Plan: The Optimization Trap

## Chapter Invariant

Optimization is measurement-driven. A performance engineer first localizes the
binding bottleneck, then proves whether a change is detectable against baseline
variance, converts the measured limit into capacity and cost headroom, and
reports the final optimization as a defended trade-off with a rejected
alternative.

The selected track changes the persona, operating thresholds, evidence emphasis,
failure mode, and memo framing. It does not change the Part A/B/C/D concept
sequence.

## Reading Map

| Lab module | Chapter anchors | Concept carried into the lab |
|---|---|---|
| Opening | Purpose; The iron law of ML performance; The efficiency frontier | Performance work is useful only when it targets the exposed time term and the stakeholder's operating envelope. |
| Part A | The iron law of ML performance; System Profiling; Using the roofline for diagnosis; Common bottleneck patterns | Localize the active bottleneck before optimizing. |
| Part B | The profiling feedback loop; Profiling at scale; Detecting scaling regressions; Fallacies and Pitfalls | A regression needs baseline, variance, and detectability evidence, not one before/after number. |
| Part C | Measurement at Scale; Fleet efficiency metric; Benchmark vs. reality; Optimization Playbook diagnostic sequence | Capacity planning converts measured service rate into demand headroom, cost, and launch risk. |
| Part D | Efficiency frontier; Combining techniques; Case study lessons; Pitfalls | An optimization report must defend a trade-off and name the rejected alternative. |
| Synthesis | Summary; From optimization to serving | The V2-10 inference plan inherits measured capacity, p99/tail risk, and residual bottleneck evidence. |

## Concept Inventory

### Accepted Concepts

| Concept | Why it belongs in this lab |
|---|---|
| Bottleneck localization | It is the calibration concept: students must stop optimizing the most visible code path and identify the exposed term first. |
| Baseline and variance for regressions | It turns "the patch changed performance" into a measurable production decision under noise and nonstationarity. |
| Fleet capacity headroom from measured throughput | It transfers local profiling into amount-system planning: demand, serving units, utilization, p99, headroom, and daily cost. |
| Trade-off report with rejected alternative | It is the design concept: optimization is accepted only when the memo defends why the chosen lever beats an attractive but weaker alternative. |
| V2-10 carry-forward implication | It connects low-level performance evidence to serving-system policy, batching, and tail-latency work in the next lab. |

### Rejected Or Deferred Concepts

| Concept | Reason rejected for this lab packet |
|---|---|
| Full roofline construction from hardware counters | Important, but too deep for this lab's time budget. Part A uses iron-law term evidence and references roofline diagnosis. |
| FlashAttention implementation details | Covered by chapter reading; the lab only uses "reduce data movement" as a bottleneck-targeted lever. |
| PTQ/QAT calibration workflow | Belongs in model compression and precision-focused labs; here it appears only as quality risk in trade-off reporting. |
| CUDA Graphs or Triton coding | Would turn the lab into a tool exercise. The concept target is measurement and decision quality. |
| MoE routing and speculative decoding math | Deferred to inference and serving policy. Part D treats algorithmic change as a candidate with conditional value and risk. |

## Concept Module Packet

### Opening: Optimization Brief

- Scenario beat: the selected track owner receives a production escalation:
  performance is below target, and a stakeholder wants a quick optimization.
- Invariant beat: the notebook names the shared rule: measure, localize,
  detect, plan, and defend.
- Track beat: the selected track changes thresholds and stakeholder language:
  battery and thermal for iPhone, SRAM/flash/duty cycle for Oura Ring, p99/p999
  safety margin for RoboTaxi, and SLA/cost/headroom for Cloud Fleet.
- Reading beat: students see the chapter anchors that support each module.
- Ledger beat: final evidence will be saved as a performance engineering memo
  for V2-10 inference.

### Part A: Concept Module - Localize The Bottleneck Before Optimizing

```yaml
concept_module:
  part_label: "Part A"
  concept_name: "Localize the bottleneck before optimizing"
  chapter_claim: "The iron-law decomposition and profiler hierarchy determine which optimization can move wall-clock time."
  reading_connection:
    chapter_section: "The iron law of ML performance; System Profiling; Common bottleneck patterns"
    claim_or_formula: "T ~= max(data/BW, compute/R_peak) + overhead; optimize the exposed term."
  track_lens:
    primary_track: "selected canonical track"
    optional_comparison_tracks: []
    stakeholder: "track-specific performance owner"
    system_decision: "Which optimization experiment is justified first?"
  student_prior:
    expected_belief: "The largest or most familiar code path should be optimized first."
    productive_failure: "A non-binding optimization gives little p99 gain and may still violate the track guardrail."
  storyline:
    beat_1_scenario: "A stakeholder asks for a quick optimization before launch."
    beat_2_prediction: "Student predicts the active bottleneck from a structured radio choice."
    beat_3_controls: "Student selects workload pressure and a candidate optimization lever."
    beat_4_evidence: "Grouped iron-law chart and table show data, compute, overhead, p99, speedup, and active bottleneck."
    beat_5_consequence: "Notebook flags wasted optimization effort when the selected lever misses the active term."
    beat_6_math_peek: "Iron-law source model explains exposed term and Amdahl-style speedup bound."
    beat_7_checkpoint: "Student chooses the first optimization experiment to authorize."
  mechanics:
    controls: ["prediction radio", "workload-pressure slider", "optimization-lever radio"]
    graphs: ["before/after grouped term chart", "exact evidence table"]
    failure_state: "SLO/deadline remains missed because the selected lever did not target the bottleneck."
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["predicted_bottleneck", "actual_bottleneck", "selected_first_lever", "part_a_p99_ms", "part_a_speedup"]
    downstream_use: "Feeds the final memo and the capacity model's selected local performance profile."
```

### Part B: Concept Module - Regressions Need Baseline, Variance, And Detectability

```yaml
concept_module:
  part_label: "Part B"
  concept_name: "Regression evidence requires baseline, variance, and detectability"
  chapter_claim: "Scaling regressions and profiling nonstationarity require repeated measurements and baseline comparison."
  reading_connection:
    chapter_section: "Profiling feedback loop; Profiling at scale; Detecting scaling regressions"
    claim_or_formula: "Minimum detectable effect ~= 1.96 * sqrt(2) * CV / sqrt(n)."
  track_lens:
    primary_track: "selected canonical track"
    optional_comparison_tracks: []
    stakeholder: "release owner deciding whether a patch can proceed"
    system_decision: "Ship, hold for more samples, or block a performance change."
  student_prior:
    expected_belief: "One faster or slower run proves the patch outcome."
    productive_failure: "High variance hides a small but policy-relevant regression."
  storyline:
    beat_1_scenario: "A candidate patch changes p99 and the release owner asks whether it is a regression."
    beat_2_prediction: "Student predicts whether the evidence is shippable, hidden, or blocking."
    beat_3_controls: "Student changes sample count, run-to-run CV, and candidate delta."
    beat_4_evidence: "Error-bar chart and detectability table show baseline mean, candidate mean, CI width, MDE, and guardrail status."
    beat_5_consequence: "Notebook names false confidence when the regression budget is exceeded but not statistically detectable."
    beat_6_math_peek: "Source model ties CV, n, and confidence interval to regression detection."
    beat_7_checkpoint: "Student selects a canary decision."
  mechanics:
    controls: ["prediction radio", "sample-count slider", "variance slider", "candidate-delta slider"]
    graphs: ["baseline vs candidate error-bar chart", "detectability table"]
    failure_state: "Undetectable regression or p99 guardrail breach."
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["regression_prediction", "sample_count", "cv_pct", "candidate_delta_pct", "mde_pct", "regression_decision"]
    downstream_use: "Justifies whether the final optimization can be trusted."
```

### Part C: Concept Module - Capacity Planning Converts Bottlenecks Into Headroom And Cost

```yaml
concept_module:
  part_label: "Part C"
  concept_name: "Capacity planning converts measured bottlenecks into headroom and cost decisions"
  chapter_claim: "Fleet-scale performance must translate measured service rate into utilization, headroom, p99, and cost."
  reading_connection:
    chapter_section: "Measurement at Scale; Fleet efficiency metric; Benchmark vs. reality; Diagnostic sequence"
    claim_or_formula: "capacity = serving_units * measured_service_rate; headroom = capacity / demand - 1."
  track_lens:
    primary_track: "selected canonical track"
    optional_comparison_tracks: []
    stakeholder: "capacity planner for the selected deployment"
    system_decision: "How many serving units are needed and whether optimization or scale is the cheaper way to buy headroom."
  student_prior:
    expected_belief: "Average throughput or a hero benchmark is enough for capacity planning."
    productive_failure: "The plan has apparent throughput but fails p99 or headroom after the reality tax."
  storyline:
    beat_1_scenario: "Demand forecast arrives for the selected track."
    beat_2_prediction: "Student predicts whether the current plan has enough headroom."
    beat_3_controls: "Student changes demand multiplier, serving units, and local optimization profile."
    beat_4_evidence: "Capacity curve/table show demand, capacity, utilization, p99, headroom, and daily cost."
    beat_5_consequence: "Notebook flags under-capacity, SLO breach, or overbuying cost."
    beat_6_math_peek: "Source model ties Little's-law-style utilization pressure to tail latency and headroom."
    beat_7_checkpoint: "Student chooses add units, optimize first, or reduce scope."
  mechanics:
    controls: ["prediction radio", "demand multiplier slider", "serving-unit slider", "optimization-profile radio"]
    graphs: ["capacity-vs-demand line", "amount-system evidence table"]
    failure_state: "headroom below target or p99 over track SLO"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["capacity_prediction", "demand_amount", "serving_units", "utilization_pct", "headroom_pct", "daily_cost", "capacity_decision"]
    downstream_use: "Provides the amount-system plan carried into V2-10 serving policy."
```

### Part D: Concept Module - Optimization Reports Must Defend A Trade-Off

```yaml
concept_module:
  part_label: "Part D"
  concept_name: "Optimization report must defend a trade-off and rejected alternative"
  chapter_claim: "The efficiency frontier makes optimization multi-objective; throughput, p99, cost, memory, and quality interact."
  reading_connection:
    chapter_section: "Efficiency frontier; Combining techniques; Case study lessons; Fallacies and Pitfalls"
    claim_or_formula: "A feasible point must satisfy p99, quality/risk, headroom, and cost constraints; dominated alternatives are rejected."
  track_lens:
    primary_track: "selected canonical track"
    optional_comparison_tracks: []
    stakeholder: "engineering lead signing the launch memo"
    system_decision: "Which optimization ships, and which tempting alternative is rejected?"
  student_prior:
    expected_belief: "The fastest or cheapest option is the correct optimization."
    productive_failure: "An attractive option violates quality, safety, energy, cost, or headroom."
  storyline:
    beat_1_scenario: "The lead asks for a one-page memo, not a benchmark screenshot."
    beat_2_prediction: "Student predicts which candidate will survive guardrails."
    beat_3_controls: "Student adjusts quality/risk budget and cost ceiling."
    beat_4_evidence: "Pareto chart/table compare p99, headroom, daily cost, quality risk, and feasibility."
    beat_5_consequence: "Notebook names the rejected alternative and why it fails or is dominated."
    beat_6_math_peek: "Source model defines the feasibility conjunction and dominance score."
    beat_7_checkpoint: "Student selects the memo decision and rejected alternative."
  mechanics:
    controls: ["prediction radio", "risk-budget slider", "cost-ceiling slider", "report-decision radio"]
    graphs: ["cost-vs-p99 frontier scatter", "candidate evidence table"]
    failure_state: "candidate violates at least one guardrail or is dominated"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["tradeoff_prediction", "selected_candidate", "rejected_candidate", "tradeoff_reason", "quality_risk_pct", "cost_ceiling"]
    downstream_use: "Becomes the final performance memo's defended decision."
```

### Synthesis: Performance Engineering Memo

The synthesis assembles a report with:

1. Bottleneck localized in Part A.
2. Regression evidence and canary decision from Part B.
3. Capacity and cost plan from Part C.
4. Chosen optimization and rejected alternative from Part D.
5. V2-10 inference implication: the next lab must use measured p99, headroom,
   residual bottleneck, and capacity units rather than peak or average
   throughput.

The ledger writes a compact JSON snapshot keyed by chapter 9 and track ID.

## Track Narratives

| Track | Persona | Constraints and thresholds | Evidence emphasis | Natural failure | Report frame |
|---|---|---|---|---|---|
| iPhone | Mobile product engineer | Sustained p99 responsiveness, thermal headroom, battery drain, unified memory | p99 latency, battery/thermal cost per session, sustained-run variance | Thermal or battery budget violation after a seemingly fast cold run | Mobile performance release memo |
| Oura Ring | Wearable firmware engineer | Wake-window latency, SRAM/flash limits, duty-cycle energy, OTA size | wake-window time, sample budget, energy per window, detectability under low-power sampling noise | Duty-cycle or memory budget miss | Firmware performance acceptance memo |
| RoboTaxi | Autonomous vehicle platform engineer | p99/p999 deadline, safety recall, sensor bandwidth, power envelope | tail latency, deadline miss, deterministic variance, safety risk | Safety margin or tail-latency miss | Safety-critical performance memo |
| Cloud Fleet | Fleet service owner | SLA p99, throughput, cost/request, utilization, capacity headroom, carbon/cost | MFU/MBU-style useful capacity, cost/day, p99, headroom | SLO breach, queueing pressure, or negative ROI | Fleet capacity and optimization memo |

Every track runs the same module sequence:

1. Part A localizes a bottleneck.
2. Part B tests detectability.
3. Part C turns measured service rate into headroom and cost.
4. Part D defends a trade-off.
5. Synthesis writes a performance memo.

## Mechanics Plan

| Module | Controls | Graph/table | Failure/boundary | Why this mechanic fits |
|---|---|---|---|---|
| Opening | track selector | reading map and track mission | none | Establishes shared invariant and track-specific lens. |
| Part A | prediction radio, workload pressure, optimization lever | grouped term chart and bottleneck table | wrong lever leaves p99 over SLO or gives weak speedup | Bottleneck diagnosis requires prediction, manipulation, and measured evidence. |
| Part B | prediction radio, n, CV, candidate delta | error-bar chart and detectability table | hidden regression or guardrail breach | Regression evidence is a function of baseline, variance, and sample size. |
| Part C | prediction radio, demand multiplier, serving units, optimization profile | capacity curve and amount-system table | insufficient headroom or p99 breach | Capacity planning is demand/capacity/headroom/cost arithmetic. |
| Part D | prediction radio, risk budget, cost ceiling, memo decision | Pareto scatter and candidate table | quality/cost/headroom/SLO violation or dominated option | Reports must justify a trade-off and reject a tempting alternative. |
| Synthesis | student ID and memo implication | report panel plus ledger HUD | incomplete if decisions missing | Saves durable evidence for V2-10. |

## Evidence And Ledger Plan

Ledger design fields:

- `track_id`, `scenario_id`, `stakeholder`, and `report_frame`
- Part A: predicted bottleneck, actual bottleneck, selected first lever, p99,
  speedup, failure state
- Part B: sample count, CV, candidate delta, minimum detectable effect,
  detectability, canary decision
- Part C: demand amount, serving units, measured service rate, utilization,
  p99, headroom, daily cost, capacity decision
- Part D: selected candidate, rejected candidate, feasibility reason,
  quality/risk budget, cost ceiling
- Synthesis: final memo summary and V2-10 inference implication

The report contains exact evidence numbers even if the student does not inspect
the chart. Tables duplicate all chart-critical quantities.

## Amount-System Reasoning

Part C is the explicit amount-system module, and the other parts feed it:

- Part A measures exposed milliseconds per unit of work.
- Part B decides whether the measured change is trustworthy under variance.
- Part C converts measured service rate into fleet amounts:
  demand amount, serving units, capacity, utilization, headroom, p99, and daily
  cost.
- Part D chooses the optimization only if that amount system satisfies the
  track guardrails.

## Implementation Notes

- Use existing bootstrap, track selector, track context, report export, and
  Design Ledger helpers.
- Keep all V2-09 calculations notebook-local and prefix helpers with `v2_09_`.
- Do not add shared MLSysIM solvers or shared lab abstractions in this worker
  pass.
- Use Plotly charts plus exact HTML tables for accessibility fallback.
- Use structured radio, slider, and number controls; no free-text prediction
  gates.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Track-specific constants are teaching estimates rather than MLSysIM registry facts | Label them in Math Peek/source model blocks and keep them notebook-local. |
| The selected track may have a different active bottleneck | Keep the concept invariant identical while allowing the track packet to change thresholds and term magnitudes. |
| Students may treat Part A speedup as production proof | Part B immediately requires variance and detectability evidence before launch. |
| Capacity plan may look like a simple replica calculator | Include p99 and headroom failure states so the amount system is not only throughput. |
| Candidate scoring could hide the trade-off | Expose every candidate in a table and require a rejected alternative in the report. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 2 | Pass |

Minimum acceptance checks:

- No dimension is below 2.
- Each concept module has at least five substantive beats.
- Each module has prediction, manipulation, evidence, consequence/boundary,
  Math Peek/source model, and checkpoint/report decision.
- At least one reversible failure state exists in every module.
- Synthesis ties all modules back to the chapter invariant and carries evidence
  into V2-10 inference.
