# V1-13 Track Plan: Model Serving As A Capacity System

## Chapter Invariant

Serving is a latency distribution plus a capacity system: batching, queueing,
replicas, live state, cold starts, and cost interact, so a feasible launch policy
must satisfy percentile latency, capacity, and track-specific guardrails at the
same time.

## Reading Map

| Lab module | Chapter anchor | Claim/formula carried into the lab |
|---|---|---|
| Part A - Batching changes throughput and latency | Traffic-Aware Batching Strategy; The batching tax | `W_total ~= (B - 1) / (2 * lambda) + T_inf(B)`; larger batches can increase throughput while adding formation delay. |
| Part B - Utilization creates queueing tails | Queuing Theory for Capacity Planning; Tail Latency and Headroom | `rho = lambda / (c * mu)` and M/M/c tail estimates; p99 fails near the queueing knee even when the mean looks acceptable. |
| Part C - Replicas/autoscaling trade cost and p99 evidence | Multi-server considerations; Tail-tolerant techniques; Capacity planning | More replicas lower utilization and p99 but raise recurring cost/request and cold-start exposure. |
| Part D - Serving policy is a constrained launch decision | Serving cost per inference; batching strategy; cold start/model loading | A policy is valid only if p99/SLA, capacity/state, cost/energy, and track guardrails all pass together. |
| Synthesis - Launch memo | Chapter summary and fallacies | The report must state selected policy, binding constraint, rejected alternative, evidence, and carry-forward ops risk. |

## Concept Inventory

Accepted concepts:

- Serving inversion: the optimization target shifts from training throughput to
  per-request percentile latency under live traffic.
- Batching tax: batching can improve device utilization and throughput while
  spending user-visible latency waiting for requests to form the batch.
- Queueing knee: arrival rate and utilization produce nonlinear p95/p99/p999
  failure modes.
- Replicas and autoscaling: scale-out buys headroom and lower tail latency at
  recurring cost and with warm-pool/cold-start constraints.
- Capacity and state: live request buffers or KV cache limit concurrency; a
  policy that passes latency but fails memory is not launchable.
- Serving economics: cost/request or energy/request is a first-class guardrail,
  not a later business note.

Rejected or deferred concepts:

| Concept | Reason for rejecting as a main module |
|---|---|
| Training-serving skew | Important, but V1-14 ML Operations owns drift, rollout, and monitoring failures. V1-13 can mention this as residual ops risk only. |
| Interface protocol and serialization tax | Useful latency budget detail, but it would dilute the required batching/queueing/replica capacity sequence. |
| LLM KV-cache internals | Appears through the Cloud Fleet state capacity calculation; the detailed LLM serving memory wall belongs to inference-scale labs. |
| Runtime and precision optimization | Appears as source-model context for service time and cost; full runtime selection is covered earlier. |
| Cold start as an independent part | Still used in Part C/D evidence, but the concept is scale policy feasibility rather than a standalone cold-start lecture. |

## Concept Module Table

| UI Part | Concept module | Student prior to challenge | Evidence produced | Ledger/report output |
|---|---|---|---|---|
| Part A | Batching changes throughput and latency at the same time | "Batching helps serving because it increases throughput." | Batch-size sweep showing formation delay, batched service time, total p99, throughput gain, and SLO status. | `batch_size`, `batch_total_p99_ms`, `batch_throughput_gain`, `batch_binding_term`. |
| Part B | Arrival rate and utilization create queueing/tail failures | "If average latency is below SLO, the service is healthy." | Utilization sweep with mean, p95, p99/p999 and a failure boundary against the track SLO. | `arrival_qps`, `utilization`, `queue_p99_ms`, `queue_slo_ok`, `tail_failure_mode`. |
| Part C | Replicas/autoscaling trade cost, utilization, and p99/SLA evidence | "More replicas are always the safe answer." | Replica frontier table and chart with p99, utilization, cost/request, warm-pool gap, and cold-start exposure. | `replicas`, `autoscale_buffer_pct`, `cost_per_request`, `replica_binding_constraint`. |
| Part D | Serving policy must satisfy SLO, capacity, and cost guardrails | "Pick the fastest policy and launch it." | Candidate policy comparison with latency, capacity/state, cost/energy, warm-pool, and track guardrail checks. | `selected_policy`, `binding_constraint`, `rejected_alternative`, `carry_forward_ops_risk`. |
| Synthesis | Serving launch memo | "Completion means all charts were inspected." | Memo card that binds the policy to numeric evidence and names the next operational risk. | Saved Design Ledger snapshot and downloadable report. |

## Module Packets

### Part A: Concept Module - Batching Is A Throughput/Latency Exchange

Chapter claim:
- Dynamic/static batching follows traffic, not habit; batch formation delay is
  paid by the request even if device throughput improves.

Track lens:
- iPhone: local/edge requests are sparse and user-visible, so batching is small
  and fallback is a responsiveness/battery choice.
- Oura Ring: batching usually means periodic windows or deferred updates; the
  amount is duty-cycle wake time and data freshness, not interactive QPS.
- RoboTaxi: batching sensor frames risks perception-deadline misses; p99/p999
  and safety margin dominate.
- Cloud Fleet: batching improves accelerator economics but must respect p99 SLA
  and cost/request.

Activity beats:
1. Scenario: the track stakeholder asks whether the default batch policy is
   worth launching.
2. Prediction: structured radio asks what term becomes binding first.
3. Manipulation: batch size, arrival rate/QPS or cadence, and SLO sliders.
4. Evidence: stacked latency chart plus exact table with formation delay,
   batched service, queue p99, total p99, and throughput gain.
5. Consequence: failure callout when total p99 exceeds the track SLO or when
   formation delay spends too much of the budget.
6. Math Peek/source model: `W_total ~= (B - 1)/(2 lambda) + T_inf(B) + W_queue`;
   source helper is `mlsysbook_labs.batching_tax`.
7. Checkpoint: student chooses whether to keep, reduce, or reject batching for
   the selected track.

Depth gate:
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: yes.
- Math/source: yes.
- Track consequence: yes.

### Part B: Concept Module - Utilization Turns Arrival Rate Into Tail Failure

Chapter claim:
- Queueing theory makes capacity planning nonlinear; p99 rises sharply as
  utilization approaches saturation.

Track lens:
- iPhone: bursts from app concurrency/background work create UI stalls.
- Oura Ring: the sparse queue still fails if radio/update windows are missed.
- RoboTaxi: sensor bursts and service-time variance create p99/p999 deadline
  risk.
- Cloud Fleet: multi-tenant QPS pushes API replicas through the queueing knee.

Activity beats:
1. Scenario: the stakeholder sees mean latency within budget and asks whether
   launch can proceed.
2. Prediction: structured radio asks which metric reveals failure.
3. Manipulation: arrival-rate slider, service-time slider, and optional
   replica count held at the track default.
4. Evidence: line chart of mean, p95, p99, and p999 versus arrival rate plus
   an exact table of selected values.
5. Consequence: reversible failure state names value, SLO limit, utilization,
   and mitigation.
6. Math Peek/source model: `rho = lambda/(c*mu)` and M/M/c percentile tail;
   source helper is `mlsysbook_labs.queueing_latency`.
7. Checkpoint: student identifies the queueing boundary or headroom target to
   carry into Part C.

Depth gate:
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: yes.
- Math/source: yes.
- Track consequence: yes.

### Part C: Concept Module - Replicas Buy Tail Headroom At A Price

Chapter claim:
- Autoscaling and replicas are tail-tolerant techniques, but capacity planning
  trades utilization, cost/request, p99/SLA evidence, and warm-start exposure.

Track lens:
- iPhone: "replicas" are local execution lanes or cloud fallback paths; extra
  capacity costs energy, thermal headroom, or privacy/cloud fallback.
- Oura Ring: "replicas" are duty-cycle windows or phone-mediated upload lanes;
  extra capacity costs wake time and battery.
- RoboTaxi: redundant perception lanes or priority paths buy safety deadline
  margin but cost power and warm spare capacity.
- Cloud Fleet: more API replicas reduce p99 and utilization but increase
  recurring cost/request and warm-pool requirements.

Activity beats:
1. Scenario: the operations lead asks whether adding replicas is enough.
2. Prediction: structured radio asks what will bind after scale-out.
3. Manipulation: replica count, autoscale buffer, and warm-pool coverage.
4. Evidence: replica frontier chart/table with utilization, p99, p999 where
   relevant, cost/request, warm-pool gap, and exposed cold-start latency.
5. Consequence: failure banner when p99/SLA, cost/energy, or warm-pool
   guardrail fails.
6. Math Peek/source model: `capacity = replicas * 1000/service_ms`,
   `cost/request ~= replica_cost_per_hour / (3600 * arrival_qps)`, and
   `cold_start = weight_read + deserialize + init + warmup`.
7. Checkpoint: student chooses the minimum acceptable replica/autoscale policy
   and names the binding constraint.

Depth gate:
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: yes.
- Math/source: yes.
- Track consequence: yes.

### Part D: Concept Module - A Serving Policy Is A Constrained Launch Gate

Chapter claim:
- Serving design is not a single knob. A valid launch policy satisfies latency
  percentiles, capacity/state, cold-start behavior, and cost/track guardrails
  under the same workload.

Track lens:
- iPhone: launch policy must balance local responsiveness, battery/thermal
  budget, and cloud fallback.
- Oura Ring: launch policy must balance phone-mediated serving/update cadence,
  duty cycle, SRAM/flash/state fit, and data freshness.
- RoboTaxi: launch policy must balance perception deadline p99/p999, safety
  fallback margin, warm spare capacity, and power.
- Cloud Fleet: launch policy must balance QPS, batching, replicas, p99 SLA,
  cost/request, and capacity headroom.

Activity beats:
1. Scenario: the stakeholder asks for a launch/no-launch policy memo.
2. Prediction: structured radio asks what makes a policy valid.
3. Manipulation: policy candidate selector plus SLO/cost/guardrail strictness.
4. Evidence: candidate table comparing latency, capacity/state, replicas,
   warm-pool, and cost/energy guardrails.
5. Consequence: boundary card names the binding constraint and first mitigation.
6. Math Peek/source model: `feasible = p99_ok and capacity_ok and cost_ok and
   track_guardrail_ok`; source helpers are queueing, batching, capacity, and
   cold start, with notebook-local policy scoring.
7. Checkpoint: student selects the policy, rejected alternative, and ops risk
   for the synthesis memo.

Depth gate:
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: yes.
- Math/source: yes.
- Track consequence: yes.

### Synthesis: Concept Module - Serving Launch Memo

The synthesis writes the chapter invariant in operational form:

1. Selected policy.
2. Binding constraint.
3. Numeric evidence: p99/SLO, utilization, batch p99, capacity/state, and
   cost/energy/request.
4. Rejected alternative and why it fails.
5. Carry-forward operations risk for Lab 14.

## Track Narratives And Amount-System Reasoning

| Track | Persona | Amount system | Failure mode | Report framing |
|---|---|---|---|---|
| iPhone | Mobile performance lead | Local requests/sec, app p99, mJ/request, battery/thermal headroom, cloud fallback share. | UI responsiveness failure or thermal/battery budget breach. | "Launch local serving only if p99 stays responsive and fallback is bounded." |
| Oura Ring | Wearable firmware lead | Sensor windows/hour, wake duty cycle, SRAM/state bytes, freshness lag, battery/day. | Missed cadence, stale update, or duty-cycle battery violation. | "Launch periodic serving only if windows stay fresh without draining battery." |
| RoboTaxi | Autonomous vehicle safety lead | Frames/sec, perception p99/p999, warm spare margin, power, deadline slack. | Perception deadline miss or fallback safety-margin violation. | "Launch only if p99/p999 and fallback evidence protect safety margin." |
| Cloud Fleet | Cloud SRE lead | QPS, batch size, replicas, utilization, p99 SLA, cost/request, warm pool. | SLA breach, queue explosion, or uneconomic overprovisioning. | "Launch minimum feasible fleet policy with explicit cost/request evidence." |

## Mechanics, Evidence, And Ledger Plan

Mechanics:
- Opening belt: track selector, chapter invariant, reading map, track mission.
- Prediction belt: one structured prediction per part before instruments render.
- Control belt: 1-3 controls per part; values are physical amounts such as QPS,
  batch size, replicas, warm-pool coverage, and cost/guardrail strictness.
- Evidence belt: Plotly charts plus HTML table fallbacks with exact values.
- Failure belt: danger/warn callouts name value, limit, unit, and mitigation.
- Source belt: Math Peek/source-model accordion in every module.
- Decision belt: checkpoint radio/dropdown for each part and synthesis memo.
- Ledger belt: save selected track, predictions, manipulated settings, evidence
  numbers, policy, rejected alternative, binding constraint, and ops risk.

Evidence:
- Part A: batching latency decomposition and throughput-gain table.
- Part B: queueing tail chart and utilization boundary table.
- Part C: replica/autoscaling frontier and cost/request table.
- Part D: policy feasibility table with guardrail status.
- Synthesis: launch memo with selected policy and carry-forward risk.

Ledger fields:
- `track_id`, `scenario_id`, `hardware_ref`, `model_ref`.
- `part_a_batch_decision`, `batch_size`, `batch_total_p99_ms`,
  `batch_throughput_gain`.
- `part_b_queue_decision`, `arrival_qps`, `utilization`, `queue_p99_ms`,
  `queue_slo_ok`.
- `part_c_replica_decision`, `replicas`, `autoscale_buffer_pct`,
  `cost_per_request`, `replica_binding_constraint`.
- `part_d_policy_decision`, `selected_policy`, `binding_constraint`,
  `rejected_alternative`, `carry_forward_ops_risk`.

## Source And Number Ownership

- Track identity, stakeholder, primary metric, guardrail metric, hardware refs,
  model refs, and default serving amounts come from the V1-13 lab variant and
  canonical track profile registry.
- Hardware and model facts are resolved from MLSysIM refs in the notebook shell.
- Queueing, batching, state capacity, and cold-start estimates use existing
  `mlsysbook_labs.serving` helpers.
- Notebook-local helpers are allowed only for V1-13 teaching calculations:
  track-specific narrative copy, cost/request approximations, policy candidate
  composition, feasibility labels, and table formatting. They must be prefixed
  `v1_13_`.
- Scenario cost/energy coefficients are teaching assumptions surfaced in Math
  Peek/source notes and report output, not hidden production claims.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Track-specific "replicas" can feel artificial for device tracks. | Frame them as local lanes, duty-cycle windows, fallback/warm spare capacity, or fleet replicas depending on track. Keep the concept invariant identical. |
| Cost/request is not available as a shared helper. | Use notebook-local V1-13 coefficients documented as scenario assumptions and expose them in Math Peek/source trace. |
| State capacity is not the core concept target but still matters for Part D. | Keep capacity as a guardrail inside policy feasibility rather than a standalone concept part. |
| Long policy table could overwhelm students. | Use a small fixed set of policy candidates with one selected policy and one rejected alternative. |
| Visual-only evidence would fail accessibility. | Include exact HTML tables after each chart and make failure text explicit. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A - Batching | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, batch/arrival/SLO controls, decomposition chart, table, Math Peek, checkpoint. |
| Part B - Queueing | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, arrival/service controls, queueing curve, failure boundary, Math Peek, checkpoint. |
| Part C - Replicas/autoscaling | 3 | 3 | 3 | 3 | 3 | 2 | Pass: shared queue/cold-start helpers plus documented local cost assumptions. |
| Part D - Policy gate | 3 | 3 | 3 | 3 | 3 | 2 | Pass: candidate table, guardrail conjunction, Math Peek, report decision; local scoring assumptions documented. |
| Synthesis - Launch memo | 3 | 3 | 3 | 3 | 3 | 3 | Pass: memo binds all evidence to the chapter invariant and Design Ledger fields. |

Minimum acceptance:
- No dimension below 2.
- Every module has at least five student-facing beats.
- At least one reversible failure state exists in Parts A, B, and C.
- Synthesis names selected policy, binding constraint, rejected alternative, and
  carry-forward operational risk.
