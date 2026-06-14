# V2-10 Track Plan: The Inference Economy

Status: Wave 7 concept-module audit packet. This plan preserves the current
pilot concept-module notebook and patches gaps instead of redesigning the lab
from scratch.

Owned notebook: `labs/vol2/lab_10_inference.py`

## Chapter Invariant

Serving large models couples prefill/decode phase behavior, queues, live
KV/cache memory, tail latency, and recurring cost. Mean throughput is not a
shipping criterion; a serving policy is valid only when latency, memory,
quality, and cost guardrails pass together.

For this chapter, the durable lesson is that inference is a live operating
system, not a one-time model run. The request path contains different amount
systems: prompt/input work creates time-to-first-token pressure, decode or
streaming work creates bandwidth and state pressure, queues convert utilization
into p99 latency, and every recurring event compounds into cost.

## Reading Map

| Lab module | Chapter anchor | Claim or formula used |
|---|---|---|
| Opening | `The Economics and Architecture of Inference`, `Serving Architecture Dimensions` | Distributed inference must preserve latency bounds while managing memory, routing, utilization, and recurring cost. |
| Part A | `TTFT vs. TPOT`, `Prefill and Decode Phases`, `Serving cost can dominate training cost` | Prefill is compute/input shaped; decode is bandwidth/output-loop shaped; recurring events compound. |
| Part B | `The KV cache wall: Memory-bound capacity`, `KV-cache capacity estimator`, `PagedAttention` | `M_KV = 2 * layers * hidden_dim * sequence * bytes * batch`; live state turns context and concurrency into memory pressure. |
| Part C | `Batching Strategies at Scale`, `Little's Law`, `Continuous batching throughput analysis`, `Fallacies and Pitfalls` | Batching and queues trade throughput against p99; continuous batching helps only when workload variance and volume justify overhead. |
| Part D | `Load Balancing and Request Routing`, `Quantization selection guidelines`, `Circuit breakers and backpressure` | A release policy must satisfy `memory_ok and slo_ok and quality_ok and cost_ok`; routing and fallback must respect the binding guardrail. |
| Synthesis | `Summary` and `From data center to edge` | The final product is an inference deployment memo: selected policy, binding amount, rejected alternative, residual risk, and V2-11 edge implication. |

Matching concept YAML anchors:

- Primary concepts: Inference at Scale, Serving Cost Dominance, Workload-Specific
  Batching, Continuous Batching, KV Cache, Prefill, Decode, Disaggregated
  Serving, Load Balancing, Autoscaling, Weight Quantization for Serving.
- Secondary concepts: TTFT, TPOT, Little's Law, M/G/c/K queue model, KV Cache
  Wall, Prefix Caching, KV Cache Quantization, Chunked Prefill, Power of Two
  Choices, Backpressure, Deadline Awareness.
- Formulas: total serving cost, batching throughput, Little's Law, waste ratio,
  KV-cache memory, two-choice routing, decode throughput proportional to memory
  bandwidth over model bytes.

## Concept Inventory

Accepted concepts:

| Concept | Module | Why accepted |
|---|---|---|
| Prefill and decode are different amount systems | Part A | Students must stop treating "inference latency" as one scalar and separate input/prompt work from output-token or streaming work. |
| KV/cache memory is a capacity and concurrency wall | Part B | Weights fitting is not enough; state grows per active request and can become the first physical limit. |
| Batching and queueing shift throughput, p99, and cost | Part C | Continuous batching is conditional; queues and deadline pressure decide whether utilization gains are usable. |
| Inference policy must satisfy simultaneous guardrails | Part D | Students must combine precision, scheduling, serving units, routing, quality, memory, SLO, and cost into one release decision. |
| Binding amount carries into edge placement | Synthesis | V2-11 changes the same inference problem by moving memory, power, connectivity, and privacy constraints outward to devices. |

Rejected or deferred concepts:

| Concept | Reason |
|---|---|
| Full autoscaler design | Deferred to operations-scale labs; V2-10 uses warm capacity and routing only as guardrails. |
| Full router implementation | Deferred to fleet orchestration; V2-10 includes routing/handoff as a policy axis, not a scheduler implementation. |
| Complete PagedAttention allocator | Referenced through state/KV evidence and Math Peek; implementing an allocator would distract from the capacity concept. |
| Speculative decoding tuning | Mentioned as residual risk and source context only; its acceptance-rate policy is a separate optimization lesson. |
| Edge/federated inference architecture | Deferred to V2-11; synthesis names the implication without adding a new concept path. |

## Concept Modules

### Part A: Prefill And Decode Are Different Amount Systems

Chapter claim: a large-model request is not one homogeneous inference amount.
Prefill/input processing controls TTFT and compute pressure; decode or streaming
output controls TPOT, bandwidth, live state, and recurring cost.

Track lens:

- Primary lens: selected student track.
- Stakeholder: track-specific owner from `v2_10_inference_economy`.
- Decision: identify whether the selected workload is bound by input/prefill
  work, output/decode work, or recurring event cost before changing demand.

Student prior:

- Expected belief: lower average latency or lower cost/event is the only amount
  to track.
- Productive failure: a design improves the wrong phase, so TTFT, TPOT, or
  recurring cost still violates the stakeholder's guardrail.

Storyline beats:

1. Scenario: the track stakeholder asks whether the always-on serving loop is
   dominated by setup cost, input processing, or output/streaming work.
2. Prediction: structured radios lock both the crossover window and the expected
   binding phase amount.
3. Manipulation: student changes demand, cost/event, optimization percent,
   planning horizon, prefill/input tokens, and decode/output tokens.
4. Evidence: cumulative setup-vs-serving chart plus phase-amount cards for
   prefill/input latency proxy and decode/output bandwidth proxy.
5. Consequence: reveal compares prediction with actual crossover and binding
   phase; recurring cost and phase imbalance are named as separate failure
   modes.
6. Math Peek/source: `C_total = C_setup + C_event * QPS * seconds * horizon`,
   `TTFT ~= f(prompt/input work)`, and
   `TPOT ~= model_bytes / memory_bandwidth`.
7. Checkpoint: saves predicted crossover, actual crossover, predicted phase,
   binding phase, prompt/input amount, decode/output amount, cost/event, and
   optimization percent.

Mechanics:

- Controls: two prediction radios, number inputs for demand and cost/event,
  sliders for horizon, optimization, prefill/input tokens, and decode/output
  tokens.
- Evidence: cost crossover chart, phase decomposition bars/cards, reveal card.
- Failure/boundary: a low recurring-cost setting can still expose a decode or
  input amount as binding; changing token/window controls can reverse the
  binding phase.

Ledger output:

- `partA_predicted_crossover`
- `partA_actual_crossover_days`
- `partA_phase_prediction`
- `partA_binding_phase`
- `partA_prefill_tokens`
- `partA_decode_tokens`
- `partA_prefill_ms`
- `partA_decode_ms`

### Part B: KV/Cache Memory Becomes Capacity And Concurrency Limit

Chapter claim: model weights are the fixed memory tax, but live state/KV cache
is request-private and grows with sequence/window length and concurrency.

Track lens:

- Primary lens: selected student track.
- Decision: determine how many concurrent sessions fit before live state or KV
  cache consumes the available memory budget.

Student prior:

- Expected belief: if the model weights fit, serving fits.
- Productive failure: reducing only compute or adding requests causes an OOM
  because state/cache, not arithmetic, is binding.

Storyline beats:

1. Scenario: the platform owner increases concurrency or context/state window.
2. Prediction: radio locks the expected concurrency bucket.
3. Manipulation: precision, context/state length, and devices per serving unit.
4. Evidence: stacked memory chart for weights plus state/KV against available
   memory.
5. Consequence: reversible OOM card names value, limit, unit, and recovery path.
6. Math Peek/source: `state_capacity()` plus the chapter KV formula or fixed
   state-per-request variant metadata for non-transformer tracks.
7. Checkpoint: saves max concurrency, context/state window, precision bytes, and
   OOM/binding status.

Mechanics:

- Controls: structured prediction radio, precision dropdown, context slider,
  devices-per-serving-unit slider, checkpoint radio.
- Evidence: stacked memory bar, capacity metric cards, prediction reveal.
- Failure state: `weight_gb + state_gb > total_memory_gb` or
  `max_concurrent < 1`; student can recover by reducing context, lowering
  precision, or adding devices per serving unit.

Ledger output:

- `partB_max_concurrent`
- `partB_context_tokens`
- `partB_precision_bytes`
- `partB_oom`

### Part C: Batching And Queueing Shift Throughput, P99, And Cost

Chapter claim: batching is a workload-specific policy. It can recover idle
capacity, but queues, scheduler overhead, padding waste, and p99 deadlines decide
whether that capacity is usable.

Track lens:

- Primary lens: selected student track.
- Decision: choose no batching, static, dynamic, or continuous scheduling for
  the current variance and deadline envelope.

Student prior:

- Expected belief: continuous batching is always the production answer.
- Productive failure: continuous batching loses when variance is low, volume is
  too low to refill slots, scheduler overhead violates the deadline, or the track
  is a streaming/duty-cycle service.

Storyline beats:

1. Scenario: the serving scheduler owner receives mixed-length or bursty work.
2. Prediction: radio locks the policy expected to win after SLO risk is counted.
3. Manipulation: average length, maximum length, batch/live slots, fill factor,
   and SLO/deadline.
4. Evidence: throughput curve, policy table, padding waste, tail estimate, and
   score winner.
5. Consequence: reversible failure cards show scheduler-overhead SLO violation
   and static-padding waste.
6. Math Peek/source: `waste = 1 - avg_len / max_len`,
   `TP_continuous = batch_size * (max_len / avg_len) * fill_factor`,
   `Q = lambda * T` as the queueing background.
7. Checkpoint: saves selected scheduling policy, speedup, padding waste, and SLO
   risk.

Mechanics:

- Controls: structured prediction radio and five scheduling sliders.
- Evidence: Plotly throughput curve, table fallback, failure cards, reveal.
- Failure state: continuous scheduler overhead or static padding waste can be
  entered and recovered by changing deadline, batch size, variance, or policy.

Ledger output:

- `partC_scheduling_policy`
- `partC_speedup`
- `partC_padding_waste_pct`
- `partC_tail_ms`
- `partC_slo_risk_pct`

### Part D: Inference Policy Must Satisfy Latency, Memory, Quality, And Cost

Chapter claim: a serving design ships only if all guardrails pass together.
Cheapest, fastest, and highest-utilization configurations can each be invalid.

Track lens:

- Primary lens: selected student track.
- Decision: choose one release policy across target demand, precision,
  scheduling policy, devices per serving unit, and routing/handoff.

Student prior:

- Expected belief: choose the cheapest or fastest configuration.
- Productive failure: a policy is blocked by memory, p99/deadline, quality, or
  recurring cost even though one headline metric looks good.

Storyline beats:

1. Scenario: release review needs one policy for the selected track.
2. Prediction: radio locks which policy candidate survives all guardrails.
3. Manipulation: target events/s, precision, scheduling policy, devices per
   serving unit, and routing/handoff.
4. Evidence: serving-plan candidate table, cost chart, replicas/devices, p99
   estimate, and binding-constraint card.
5. Consequence: failure cards expose OOM, SLA/deadline, and cost/power failures.
6. Math Peek/source: `serving_plan()` and feasibility conjunction
   `memory_ok and slo_ok and quality_ok and cost_ok`.
7. Checkpoint: saves selected policy, precision, replicas, cost/day, SLO margin,
   binding constraint, rejected alternative, and residual risk.

Mechanics:

- Controls: structured prediction radio, target events/s number input,
  precision dropdown, scheduling dropdown, devices slider, routing dropdown.
- Evidence: candidate table, cost bar, metric cards, failure cards.
- Failure states: OOM, deadline/SLA violation, and cost/power violation can be
  entered and recovered through precision, policy, devices, and demand.

Ledger output:

- `partD_selected_precision`
- `partD_selected_policy`
- `partD_selected_routing`
- `partD_replicas_needed`
- `partD_cost_per_day`
- `partD_slo_margin_ms`
- `partD_binding_constraint`
- `partD_rejected_alternative`
- `residual_risk`

### Synthesis: Inference Deployment Memo

Chapter invariant restated: production inference couples phase amounts,
state/cache memory, queues, tail latency, and cost; the binding amount determines
the policy that can ship.

Student product:

1. Selected policy with precision, scheduling, serving-unit count, and routing.
2. Binding amount: prefill/input, decode/output, memory/state, SLO/deadline,
   quality, or recurring cost.
3. Rejected alternative and why it failed the guardrail review.
4. V2-11 edge implication: what changes when the same policy moves toward
   devices with smaller memory, power, and connectivity envelopes.
5. Local report export using `build_lab_report()` and `report_export_panel()`.

Future ledger use:

- V2-11 can read `track_id`, `scenario_id`, selected policy, binding amount,
  memory/concurrency evidence, scheduling choice, residual risk, and edge
  implication.

## Track Narratives And Required Differences

The lab has one shared Part A/B/C/D concept sequence. Tracks do not create
different concepts. The selected track changes persona, constraints, thresholds,
evidence emphasis, failure mode, and memo framing through
`get_lab_track_variant("v2_10_inference_economy", track_id)`.

| Track | Persona | Same concepts realized as | Constraint emphasis | Natural failure | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile product owner | Local interactive inference with optional offload; input pass and output/decision loop are battery/thermal amounts | Battery energy, heat, local memory, responsiveness, privacy | Thermal or battery budget miss; offload privacy/latency risk | Defend local versus selective offload under p99, battery, and privacy evidence. |
| Oura Ring | Wearable firmware lead | Duty-cycle sensing where input windows and recurring decisions consume SRAM and mWh/day | SRAM/flash, duty cycle, mWh/day, sensing quality, phone handoff | SRAM overflow or duty-cycle/battery miss | Defend ring-local inference cadence and phone/cloud handoff under memory and battery evidence. |
| RoboTaxi | Autonomous fleet operations lead | Vehicle-local streaming inference where input sensor bursts and output decisions have hard deadlines | P99/P999 latency, safety margin, power, sensor bursts | Deadline/safety margin miss or power envelope violation | Defend a local safety path with reserve headroom and explicit fallback risk. |
| Cloud Fleet | Inference platform owner | LLM serving where prefill, decode, KV cache, continuous batching, replicas, and routing set cost/request | KV cache, p99 SLO, cost/request, utilization, quality | OOM, queue explosion, SLO breach, negative cost efficiency | Defend a continuous/dynamic batching fleet policy under SLA, memory, quality, and cost evidence. |

## Mechanics Plan

Opening belt:

- Track selector seeded from the Design Ledger.
- Header, learning objectives, reading connection, track context, and track arc.
- Explicit note that every track follows the same concept sequence.

Prediction belt:

- One or more structured controls before evidence in every part.
- Reveals compare the student prediction with computed evidence instead of only
  marking correctness.

Control belt:

- Part A: demand, cost/event, optimization, horizon, input/prefill amount, and
  output/decode amount.
- Part B: precision, context/state length, devices per serving unit.
- Part C: average length, maximum length, batch/live slots, fill factor, SLO.
- Part D: target events/s, precision, scheduling, devices, routing/handoff.

Evidence belt:

- Part A: cumulative cost curve plus phase amount cards/bars.
- Part B: stacked memory chart and exact capacity metrics.
- Part C: throughput curve, table fallback, and tail-risk status.
- Part D: candidate policy table, cost chart, and guardrail metric cards.
- Synthesis: deployment memo and downloadable report.

Failure belt:

- `v2_10_failure_card()` names the violated constraint and recovery action.
- Part B exposes reversible memory/OOM failure.
- Part C exposes reversible scheduler-overhead and static-padding failures.
- Part D exposes reversible memory, SLO/deadline, and cost/power failures.

Source belt:

- Math Peek appears in every part.
- Source models are named: `cost_crossover()`, `state_capacity()`,
  `batching_result()`, `serving_plan()`, plus notebook-local teaching estimates
  for phase split, tail estimate, and policy guardrail review.

Decision and ledger belt:

- Each part ends with a checkpoint control.
- Final HUD saves only after all required predictions and checkpoints are
  complete.
- Report export serializes predictions, knobs, evidence, final decision,
  rejected alternative, residual risk, and source trace.

## Evidence And Ledger Plan

Evidence required from the notebook:

| Evidence | Notebook implementation target |
|---|---|
| Prediction-vs-actual overlay | Reveal cards in Parts A-D. |
| Manipulation | Numeric inputs/sliders/dropdowns in every part. |
| Boundary or failure state | Part B OOM, Part C SLO/waste, Part D OOM/SLO/cost. |
| Value, limit, unit, and threshold | Metric cards, tables, and failure cards. |
| Chapter formula/source connection | Math Peek/source model in every part. |
| Design decision | Part D release policy plus synthesis memo. |
| Future-lab handoff | Ledger fields and report snapshot for V2-11. |

Ledger schema:

```json
{
  "track_id": "...",
  "scenario_id": "...",
  "partA_predicted_crossover": "...",
  "partA_actual_crossover_days": 0.0,
  "partA_phase_prediction": "...",
  "partA_binding_phase": "...",
  "partA_prefill_tokens": 0,
  "partA_decode_tokens": 0,
  "partA_prefill_ms": 0.0,
  "partA_decode_ms": 0.0,
  "partB_max_concurrent": 0,
  "partB_context_tokens": 0,
  "partB_precision_bytes": 0.0,
  "partB_oom": false,
  "partC_scheduling_policy": "...",
  "partC_speedup": 0.0,
  "partC_padding_waste_pct": 0.0,
  "partC_tail_ms": 0.0,
  "partC_slo_risk_pct": 0.0,
  "partD_selected_precision": "...",
  "partD_selected_policy": "...",
  "partD_selected_routing": "...",
  "partD_replicas_needed": 0,
  "partD_cost_per_day": 0.0,
  "partD_slo_margin_ms": 0.0,
  "partD_binding_constraint": "...",
  "partD_rejected_alternative": "...",
  "v2_11_edge_implication": "...",
  "residual_risk": "..."
}
```

## Notebook Depth Audit

Depth-gate result after reading the current notebook and planned patch:

| Module | Scenario | Prediction | Manipulation | Evidence | Consequence/failure | Math/source | Checkpoint | Result |
|---|---|---|---|---|---|---|---|---|
| Part A | Pass | Pass: crossover plus phase prediction | Pass | Pass: cost curve plus phase metrics | Pass: phase/cost reveal | Pass | Pass | Meets concept-module depth gate after phase-split patch. |
| Part B | Pass | Pass | Pass | Pass | Pass: reversible OOM boundary | Pass | Pass | Meets concept-module depth gate. |
| Part C | Pass | Pass | Pass | Pass | Pass: reversible scheduler/SLO and padding-waste failures | Pass | Pass | Meets concept-module depth gate. |
| Part D | Pass | Pass | Pass | Pass | Pass: memory, SLO/deadline, and cost/power guardrails | Pass | Pass | Meets concept-module depth gate. |
| Synthesis | Pass | n/a | n/a | Pass | Pass: rejected alternative and residual risk | n/a | Ledger/report | Meets deployment-memo handoff requirement after memo patch. |

Rubric score:

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability |
|---|---:|---:|---:|---:|---:|---:|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 2 |

Acceptance notes:

- No dimension scores below 2.
- At least one reversible failure state is present; Parts B, C, and D expose
  reversible failures.
- Every concept module has scenario, structured prediction, manipulation,
  evidence, consequence or failure, Math Peek/source model, and checkpoint.
- Traceability is 2 rather than 3 in Parts A/C/D because local teaching
  estimates are used for phase split, tail estimate, and release-review status;
  the shared helper APIs and chapter formulas remain named.
- Synthesis ties the modules back to the chapter invariant and carries a
  future-usable V2-11 edge implication.

## Implementation Risks And Guardrails

Preserve:

- Existing WASM bootstrap and local wheel paths.
- Local helper prefix `v2_10_`.
- Existing verified behavior and the current Marimo scoping pattern where every
  helper used by a later cell is returned from its defining cell.
- Shared helper APIs and typed track variants; no edits to helpers, tests, or
  other labs in this wave.

Risks:

- Part A phase split uses a notebook-local teaching proxy because the shared
  helper surface exposes cost, state, batching, and serving-plan calculations
  but not a full prefill/decode profiler.
- Device tracks are not literal LLM decode services; they realize the same
  concept as input/window work versus output/decision-loop work, with the report
  framing making the constrained interpretation explicit.
- Cost and p99 estimates remain classroom estimates. The report must preserve
  residual risk that production traces, thermal/power replay, token
  distributions, quality canaries, routing behavior, and current pricing are
  required before deployment.
