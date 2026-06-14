# V1-06 Track Plan: Neural Network Architectures

## Chapter Invariant

Architecture choices create different resource shapes. Inductive bias and scaling laws determine which amount grows first: operations, activation memory, state, latency, energy, quality risk, or fleet cost.

This lab treats architecture as a systems contract. A student does not choose the model with the largest leaderboard score; they recommend a family that fits the selected deployment envelope and can explain which alternative was rejected, why it failed, and what residual risk remains.

There is one shared concept sequence for every student: Part A topology and locality, Part B scaling memory wall, Part C inductive-bias trade-off, Part D constrained recommendation, then synthesis memo. The selected track realizes those same concepts differently by changing persona, constraints, thresholds, evidence emphasis, failure mode, and report framing.

## Reading Map

| Module | Chapter connection | Claim used in lab |
|---|---|---|
| Opening | Purpose; Architectural Principles | Architecture is the algorithm axis of D-A-M and fixes operation count and data movement. |
| Part A | CNNs: algorithmic structure; RNNs: system implications; Computational Primitives | Topology changes operation locality, memory reuse, and the dominant bottleneck. |
| Part B | Attention: dynamic processing; Transformers: system implications; KV cache sizing | Attention and sequence scaling can make memory or state the hidden wall even when weights fit. |
| Part C | Learnability gap; No Free Lunch; Inductive bias hierarchy | Bias trades quality, data need, parameter efficiency, and deployment fit. |
| Part D | Architecture Selection Framework; Fallacies and Pitfalls | Selection is a constrained deployment recommendation, not a leaderboard choice. |
| Synthesis | Summary: Architecture is infrastructure | The memo records the recommendation, rejected alternatives, measured evidence, and residual risk. |

## Concept Inventory

Accepted concepts:

- Architecture topology changes operation and memory locality.
- Attention and sequence scaling create non-obvious memory/state walls.
- Inductive bias is a quality/data/deployability trade-off, not a pure modeling preference.
- Architecture selection is a constrained deployment recommendation.
- Track-specific amount systems determine which constraint is meaningful.

Rejected concepts:

- Historical architecture chronology. It does not force a deployment decision.
- Pure taxonomy of MLP/CNN/RNN/Transformer/DLRM. Taxonomy alone has no consequence.
- Detailed derivation of every layer formula. The lab needs formulas only where they explain observed failure.
- Shared building block catalog as a separate part. Skip connections, normalization, and gating are referenced as validation risks but are not the central student decision.
- DLRM capacity wall as a full module. It is important in the chapter, but V1-06 uses the four course tracks and keeps the main comparison to local, sequential, perception, and service architectures.

## Track Narratives

| Track | Stakeholder | Amount system | Natural failure |
|---|---|---|---|
| iPhone | Mobile product engineer | Local vision/audio model fit, latency, activation memory, supported kernels, sustained energy | NPU fallback, thermal throttle, or user-visible latency. |
| Oura Ring | Wearable firmware engineer | Tiny sequence/signal model, SRAM, flash, duty cycle, uJ/window, OTA payload | SRAM overflow or duty-cycle/battery miss. |
| RoboTaxi | Autonomous vehicle platform engineer | Perception architecture, rare-event recall, p99/p999 deadline, sensor burst power | Safety margin miss or p99 deadline violation. |
| Cloud Fleet | Fleet service owner | Transformer/service architecture, cost/request, memory, utilization, p99 SLA | HBM/KV-cache pressure, queue/SLA breach, or negative economics. |

Track-specific behavior changes at least four surfaces: scenario persona, default workload scale, metric priorities, failure threshold, checkpoint prompt, and ledger fields.

The track never changes the concept assigned to Part A, B, C, or D. It changes the amount system through which the shared concept becomes concrete.

## Concept Modules

### Part A - Concept Module: Topology Changes Locality

Chapter claim:
- CNNs exploit local filters and weight sharing; RNNs trade state locality for sequential dependency; transformers use all-to-all gather/reduce patterns.

Student prior:
- "If two models have similar quality, the cheaper FLOP count wins."

Activity beats:
1. Scenario: the selected stakeholder must pick an architecture family for the track workload.
2. Prediction: the student predicts which topology risk will bind first.
3. Manipulation: the student changes the track workload scale and sees the resource signature update.
4. Evidence: grouped bars and a table compare parameters, GMAC, activation memory, latency, power, kernel support, and dominant constraint.
5. Consequence: the module names the violated locality or dispatch constraint.
6. Math/source: convolution locality, RNN hidden state, and attention gather/reduce are tied back to chapter anchors.
7. Checkpoint: the student records which topology best matches the amount system.

Track-specific amount reasoning:
- iPhone: kernel support and sustained latency decide whether local vision/audio stays private and responsive.
- Oura Ring: SRAM-resident temporal state and duty cycle dominate more than nominal quality.
- RoboTaxi: p99 perception latency and rare-event replay guardrails dominate average quality.
- Cloud Fleet: memory, batching, utilization, and cost/request dominate local-device concerns.

### Part B - Concept Module: Sequence Scaling Hides The Memory Wall

Chapter claim:
- Attention training has quadratic score memory, while autoregressive serving adds KV/state memory that grows with sequence length and concurrency.

Student prior:
- "If model weights fit, the deployment fits."

Activity beats:
1. Scenario: the team increases context, sensor resolution, or signal window size.
2. Prediction: the student predicts whether latency, activation memory, power, or quality/kernels fails first.
3. Manipulation: the workload-scale slider is moved across the reachable range.
4. Evidence: latency and activation curves show current point, budget lines, and first infeasible scale.
5. Consequence: a reversible failure state names value, limit, unit, and mitigation.
6. Math/source: attention score memory `O(S^2)` and KV/state memory `O(B x layers x heads x S x d_head)` explain the curve.
7. Checkpoint: the student chooses the mitigation: shorten scale/context, choose a local architecture, or require memory-aware kernels.

Track-specific amount reasoning:
- iPhone: larger image/audio windows turn attention into activation memory and thermal pressure.
- Oura Ring: sequence windows fit only when state stays in SRAM and wake time remains short.
- RoboTaxi: higher sensor resolution increases token/feature-map pressure and p99 latency.
- Cloud Fleet: longer context reduces batching efficiency and increases memory per request.

### Part C - Concept Module: Inductive Bias Trades Quality, Data Need, And Deployability

Chapter claim:
- No Free Lunch means a bias helps matching data and hurts mismatched data. Learnability and deployment efficiency are part of the same choice.

Student prior:
- "The architecture with the highest quality proxy is the safest choice."

Activity beats:
1. Scenario: the stakeholder must defend a bias under the selected track's data regime.
2. Prediction: the student predicts whether local bias, flexible attention, or leaderboard quality will win.
3. Manipulation: the student changes data/coverage pressure.
4. Evidence: a table compares quality floor, data-need index, deployability score, and feasibility.
5. Consequence: the module names the quality, data, or deployability failure.
6. Math/source: sample complexity and No Free Lunch explain why stronger bias can reduce data need but narrow applicability.
7. Checkpoint: the student records the bias they would defend and what validation would disprove it.

Track-specific amount reasoning:
- iPhone: local vision/audio bias must preserve privacy, latency, memory, and energy.
- Oura Ring: tiny sequence/signal bias must preserve SRAM, duty cycle, and sensing quality.
- RoboTaxi: perception bias must preserve rare-event recall and p99 deadline.
- Cloud Fleet: transformer/service bias must preserve quality, utilization, and cost/request.

### Part D - Concept Module: Selection Is A Deployment Recommendation

Chapter claim:
- The architecture selection framework matches data characteristics, computational complexity, hardware mapping, and production constraints.

Student prior:
- "Architecture review should approve the most accurate feasible model."

Activity beats:
1. Scenario: the student sits in an engineering review and must make a recommendation.
2. Prediction: the student predicts which review rule will approve the architecture.
3. Manipulation: the student selects an architecture recommendation.
4. Evidence: decision cards compare selected architecture, leaderboard alternative, feasibility, guardrail, next failure, and validation requirement.
5. Consequence: the notebook labels reject, approve with mitigation, or approve.
6. Math/source: feasibility is the conjunction of memory, latency, power, quality, and kernel support constraints.
7. Checkpoint: the student records approve/reject/mitigate before writing the memo.

Track-specific amount reasoning:
- iPhone: ship only if local fit, latency, memory, energy, and NPU support hold.
- Oura Ring: ship only if SRAM, duty cycle, OTA/flash, and signal quality hold.
- RoboTaxi: ship only if rare-event safety evidence and p99/p999 deadline hold.
- Cloud Fleet: ship only if memory, utilization, p99 SLA, and cost/request hold.

### Synthesis - Concept Module: Architecture Recommendation Memo

Student output:
- Recommendation.
- Rejected alternatives.
- Measured binding constraint.
- Evidence number.
- Residual risk.
- Validation requirement.

Ledger output:
- `track_id`
- `scenario_id`
- `selected_architecture`
- `failure_prediction`
- `attention_memory_prediction`
- `bias_prediction`
- `review_prediction`
- `dominant_constraint`
- `next_failure`
- `quality_pct`
- `kernel_support_pct`
- `residual_risk`
- `validation_requirement`

Downstream use:
- V1-07 can ask whether the runtime/framework keeps this architecture on the fast path.
- V1-10 can ask whether compression reduces the binding resource.
- V1-16 can replay the architecture memo in the final Volume I audit.

## Mechanics Plan

| Module | Controls | Graphs/tables | Failure state |
|---|---|---|---|
| Opening | Track selector | Reading map and workflow source trace | None |
| Part A | Topology prediction; workload scale slider | Signature grouped bar; topology/resource table | Dominant constraint and feasibility badge |
| Part B | Memory-wall prediction; workload scale slider | Latency curve; activation-memory curve; first-failure table | Memory/latency/power budget callout with mitigation |
| Part C | Bias prediction; data pressure slider | Bias/deployability frontier table and chart | Quality floor or deployability/data-need warning |
| Part D | Review prediction; architecture dropdown | Decision cards; rejected alternatives; validation tests | Reject/mitigate/approve callout |
| Synthesis | Memo reflection | Design record and report export | Incomplete report fields until predictions and memo exist |

## Evidence Plan

Every plot that drives a decision has a table fallback with exact values:

- Part A: signature table with parameters, GMAC, activation memory, latency, power, quality, kernels, feasibility, and dominant constraint.
- Part B: first infeasible scale table plus activation/latency budget values.
- Part C: quality, data-need index, deployability score, and feasibility table.
- Part D: selected-vs-leaderboard comparison and rejected alternatives.
- Synthesis: memo-ready design record with residual risk and validation requirement.

## Implementation Notes And Risks

- Use existing `architecture_track_profile`, `architecture_signature`, `architecture_scaling_curve`, and `architecture_decision`.
- Keep added calculations notebook-local and prefix helper names with `v1_06_`.
- Do not edit MLSysIM, `mlsysbook_labs`, tests, or other lab files.
- The shared helper currently models latency, activation memory, power, quality, and kernel support but not full p99 distributions or actual KV cache bytes. The notebook will state this as a source-model abstraction and keep the calculations scenario-scaled.
- If a future shared helper adds typed attention/KV-cache or bias frontier objects, this notebook-local code should migrate there.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 3 | Yes |

Minimum acceptance check:

- No dimension below 2.
- At least three dimensions at 3 for every module.
- Reversible failure states exist through the scale slider and architecture dropdown.
- Synthesis ties all parts back to the invariant that architecture choices create different resource shapes.
