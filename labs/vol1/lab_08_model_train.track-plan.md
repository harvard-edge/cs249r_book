# V1-08 Track Plan: Model Training Concept Module Packet

## Chapter Invariant

Training is budgeted optimization: batch, precision, optimizer state, memory,
throughput, convergence cost, and deployment evidence interact. A training plan
is valid only when the binding resource and the downstream deployment
implication are both named.

## Shared Concept Sequence

The lab has one shared concept sequence for every student:

1. Part A: batch size changes throughput and convergence, not just speed.
2. Part B: optimizer state and activations create a memory budget.
3. Part C: precision changes memory/throughput and can change stability/evidence.
4. Part D: training plan selection must satisfy cost, time, memory, and
   deployment evidence constraints.

The selected track does not create different concepts. It changes the persona,
constraints, thresholds, evidence emphasis, failure mode, and report framing used
to realize the same four concepts.

## Reading Map

| Lab module | Chapter anchor | Claim used in the lab |
|---|---|---|
| Opening | `training.qmd` purpose, Training Systems Fundamentals | Training multiplies inference cost through backward pass, optimizer state, activations, data movement, and repeated iteration. |
| Part A | Iron Law of Training Performance; Fallacies and Pitfalls | Batch size changes utilization and convergence behavior, so "bigger batch" is not merely a speed knob. |
| Part B | Optimization Algorithm System Implications; Activation Memory Requirements; Memory-Computation Trade-offs | Training memory is weights + gradients + optimizer state + activations + batch data. Optimizer state and activations often bind before compute. |
| Part C | Mixed-Precision Training; Mixed-Precision Hardware Support; mixed precision pitfall | Precision can reduce memory and improve throughput, but reduced precision requires stability evidence, loss-scaling policy, or BF16/FP8 validation. |
| Part D | GPT-2 Optimization Walkthrough; When to Scale; Fallacies and Pitfalls | The right plan is selected by simultaneous cost, time, memory, and validation constraints, not by the fastest single metric. |
| Synthesis | Summary and "Why training costs millions" takeaways | The durable skill is reading the current bottleneck and spending only against that bottleneck. |

## Concept Inventory

Accepted concepts:

1. Batch size is a coupled throughput, memory, and convergence lever.
2. Optimizer state and activations turn training into a memory-budget problem.
3. Precision changes memory and throughput while adding numerical evidence
   requirements.
4. A training plan must satisfy resource, time/cost, and deployment-evidence
   constraints at the same time.

Rejected or deferred concepts:

1. FlashAttention internals: important, but too kernel-specific for the
   deployment-track training-plan memo.
2. Detailed distributed parallelism tax: held for Volume II distributed
   training; this lab names communication and checkpoint hidden costs only.
3. Full optimizer derivations: the lab uses optimizer state amount reasoning,
   not a derivation of Adam updates.
4. Data-loader prefetch mechanics: referenced as throughput context but not
   made a primary control because V1-04 already owns data movement.
5. Training energy/carbon derivation: included as Cloud Fleet consequence and
   memo field, not a full carbon calculator.

## Track Narrative Plan

Tracks are lenses, not skins. Each track changes stakeholder, amount system,
failure threshold, evidence, and memo prompt.

| Track | Stakeholder | Amount system | Natural failure | Evidence emphasis |
|---|---|---|---|---|
| iPhone | Mobile product engineer | App/runtime memory, thermal and battery headroom, privacy-local data, adapter storage | Full local training exceeds app memory or creates thermal/battery regressions | On-device replay, local adapter validation, privacy-safe telemetry, rollback |
| Oura Ring | Wearable firmware engineer | SRAM/flash, duty cycle, OTA payload, scarce biosignal labels | Full training state cannot fit firmware or always-on sensing budget | SRAM trace, OTA payload, battery regression, biosignal replay |
| RoboTaxi | Autonomous vehicle platform engineer | Rare-event coverage, p99/p999 replay time, safety compute, route evidence | Local retraining competes with safety compute or invalidates certification | Fleet/simulation retraining plus vehicle-local rare-event and fallback replay |
| Cloud Fleet | Fleet service owner | Accelerator memory, throughput, job time, cost, carbon, evaluation coverage | OOM, slow iteration, low utilization, communication/checkpoint overhead | Memory profiler, throughput run, quality regression, serving canary, cost/time/carbon memo |

## Concept Modules

### Part A: Concept Module - Batch Size Changes Throughput And Convergence

Chapter claim:
- The iron law makes utilization a training-time lever.
- The fallacies section warns that memory and computation cannot be optimized independently and that large-batch hyperparameters require validation.

Student prior:
- "Increase batch size and training just gets faster."

Track lens:
- iPhone and Oura: the local batch is limited by data collection and device
  budgets; full training is usually upstream.
- RoboTaxi: larger curated batches may improve throughput, but rare-event
  evidence can be diluted or delayed.
- Cloud Fleet: batch size is an accelerator-utilization and job-cost lever, but
  large effective batches must preserve convergence.

Activity beats:
1. Scenario: the stakeholder must choose the batch size for the track's training
   or adaptation loop.
2. Prediction: student predicts whether the biggest feasible batch is always the
   best training decision.
3. Manipulation: student moves batch size and watches throughput, memory, and
   convergence/evidence caution.
4. Evidence: batch-frontier chart and table show throughput, memory, feasibility,
   and caution state.
5. Consequence: the lab names the first violated constraint or the hidden
   convergence/evidence cost.
6. Math/source: Math Peek ties the behavior to
   `T_train = O / (R_peak * eta_hw)` and effective-batch reasoning.
7. Checkpoint: student states which batch choice is defensible for the track.

Mechanics:
- Structured radio prediction.
- Batch-size slider.
- Frontier line chart with failure-colored points.
- Table fallback with exact memory, throughput, and convergence/evidence state.
- Track-specific consequence callout.

Ledger output:
- `batch_prediction`
- `batch_size`
- `batch_feasible`
- `batch_consequence`

### Part B: Concept Module - Optimizer State And Activations Create A Memory Budget

Chapter claim:
- Total training memory equals weights + gradients + optimizer state +
  activations + batch data.
- Adam's state and activation storage are first-order training constraints.

Student prior:
- "If inference weights fit, training should fit."

Track lens:
- iPhone/Oura: trainable fraction is the only way local adaptation survives the
  amount system.
- RoboTaxi: validation can happen locally, but full local optimizer and
  activation state is not a safety-compute neighbor.
- Cloud Fleet: even where full training is central, optimizer state and
  activations decide whether a job fits HBM.

Activity beats:
1. Scenario: stakeholder must explain why the naive full-training option fails
   or survives.
2. Prediction: student predicts the dominant memory component.
3. Manipulation: student changes batch size and strategy, then compares the
   selected stack against the strategy table.
4. Evidence: stacked memory bar and exact table show weights, gradients,
   optimizer state, activations, data batch, total, and budget.
5. Consequence: OOM/throughput violation is shown with numbers and mitigation.
6. Math/source: Math Peek uses the total-memory equation and activation
   per-batch equation.
7. Checkpoint: student selects the mitigation that addresses the binding term.

Mechanics:
- Dominant-component radio prediction.
- Shared strategy selector.
- Stacked bar plus memory budget hline.
- Strategy comparison table.
- Reversible OOM/throughput failure state.

Ledger output:
- `memory_prediction`
- `dominant_component`
- `total_memory_mb`
- `training_memory_budget_mb`
- `binding_resource`

### Part C: Concept Module - Precision Changes Memory, Throughput, And Stability Evidence

Chapter claim:
- Mixed precision reduces memory and can increase throughput, but FP16, BF16,
  and FP8 differ in stability and validation burden.
- Treating mixed precision as a simple toggle is a production pitfall.

Student prior:
- "Lower precision is just cheaper and faster."

Track lens:
- iPhone/Oura: precision choices usually happen upstream or in adapter/calibration
  code, then must survive deployment conversion and local replay.
- RoboTaxi: reduced precision must pass rare-event and route replay before it
  can be considered a safety artifact.
- Cloud Fleet: BF16/FP16/FP8 choices trade HBM, throughput, cost, carbon, and
  numerical-risk evidence.

Activity beats:
1. Scenario: stakeholder considers lowering precision to recover memory or time.
2. Prediction: student predicts which precision policy creates the main risk.
3. Manipulation: student changes precision policy for the selected strategy.
4. Evidence: precision table and bar chart show memory, throughput multiplier,
   stability risk, and required evidence.
5. Consequence: the lab names whether the policy is blocked by memory,
   throughput, hardware/track fit, or missing stability evidence.
6. Math/source: Math Peek ties precision bytes to memory and Tensor Core
   throughput while naming FP16/BF16/FP8 stability caveats.
7. Checkpoint: student chooses what evidence would make the reduced-precision
   plan acceptable.

Mechanics:
- Structured precision prediction.
- Precision-policy dropdown/radio.
- Notebook-local precision evidence helper prefixed `v1_08_`.
- Exact table fallback and current-policy metric cards.

Ledger output:
- `precision_prediction`
- `precision_policy`
- `precision_total_memory_mb`
- `precision_evidence_required`

### Part D: Concept Module - Training Plan Selection Is A Multi-Constraint Decision

Chapter claim:
- The physical ceiling appears when memory, wall-clock time, or dataset scale
  exceeds local optimization.
- Scaling, checkpointing, precision, and adaptation are valid only when they
  answer the measured bottleneck.

Student prior:
- "Pick the feasible plan with the best throughput."

Track lens:
- iPhone/Oura: the plan is usually central training plus local adaptation or
  calibration because deployment and data collection reality constrain training.
- RoboTaxi: the plan must produce safety evidence, not just model weights.
- Cloud Fleet: the plan must fit accelerator memory while controlling job time,
  cost, carbon, and serving handoff risk.

Activity beats:
1. Scenario: stakeholder must write a decision memo for the selected track.
2. Prediction: student predicts which constraint will rule out the naive plan.
3. Manipulation: student selects a training/adaptation strategy and batch size.
4. Evidence: plan table compares selected plan, feasibility, memory, max feasible
   batch, training location, validation location, hidden cost, and residual risk.
5. Consequence: rejected alternatives are shown with the binding reason.
6. Math/source: Math Peek references the iron-law budget and physical-ceiling
   limits.
7. Checkpoint: student records the final plan decision and report reflection.

Mechanics:
- Structured constraint prediction.
- Strategy dropdown plus shared batch slider.
- Plan evidence cards and rejected-alternative table.
- Final reflection text area used only for report/memo synthesis.

Ledger output:
- `selected_training_plan`
- `training_location`
- `validation_location`
- `hidden_cost`
- `residual_risk`
- `deployment_handoff`

### Synthesis: Training Plan Memo

Student output:
- Decision.
- Binding resource.
- Evidence number.
- Reading connection.
- Carry-forward deployment implication.

Memo invariant:
- "We chose this training plan because the binding training resource was X, the
  measured evidence was Y, and the next deployment lab must carry Z."

Ledger output:
- `memo_binding_resource`
- `memo_evidence_number`
- `carry_forward_deployment_implication`
- `report_artifact`

## Mechanics And Evidence Plan

| Need | Mechanic | Evidence |
|---|---|---|
| Productive failure | Radio predictions before each instrument | Prediction-vs-actual text and exact values |
| Boundary finding | Batch slider and frontier chart | Max feasible batch, first infeasible batch, violation text |
| Bottleneck diagnosis | Stacked memory bar and strategy table | Dominant memory component and budget utilization |
| Context transfer | Track selector and track-specific narrative | Different stakeholder, amount system, failure, report prompt |
| Trade-off reasoning | Precision policy selector and plan selector | Memory/throughput/stability and selected plan table |
| Synthesis | Report panel and ledger save | Training memo with binding resource and deployment implication |

Accessibility:
- Every plot has an exact table fallback.
- Failure states include value, limit, unit, and mitigation in text.
- Color is not the only indicator; tables include `yes/no`, `blocked`, or
  named consequence.

## Implementation Plan

Owned files:
- `labs/vol1/lab_08_model_train.py`
- `labs/vol1/lab_08_model_train.track-plan.md`

Do not edit:
- shared helpers
- tests
- other labs
- permanent checkout `/Users/VJ/GitHub/MLSysBook`

Notebook-local helpers:
- Prefix with `v1_08_`.
- Derive evidence from existing `training_memory_stack`,
  `training_frontier`, and `training_plan` outputs.
- Keep helper outputs simple dictionaries/tuples suitable for Marimo HTML,
  Plotly, and ledger serialization.

Existing helper contracts reused:
- `training_track_profile`
- `training_memory_stack`
- `training_frontier`
- `training_plan`
- `build_lab_report`
- `DesignLedger`
- `track_selector`
- `track_context`
- `track_arc_context`
- `report_export_panel`

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Precision-policy controls could imply unsupported device training. | Text and evidence table explicitly separate upstream training from deployment/adaptation on iPhone/Oura. |
| More widgets could break Marimo dataflow if not returned. | Return each widget from its own cell and consume values in downstream cells. |
| Track-specific reasoning could become generic. | Use per-track stakeholder, amount system, failure, and evidence strings in every module. |
| Shared helpers do not expose precision sweeps. | Keep precision sweep notebook-local and clearly scenario-derived from selected stack values. |
| Parallel workers may edit other labs. | Only touch owned files and do not run broad formatting over the tree. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 | Pass |

Rubric notes:
- No dimension is below 2.
- Every module includes scenario, prediction, manipulation, evidence,
  consequence, source/math tie, and checkpoint/report decision.
- Part B includes the reversible memory/throughput failure state.
- Synthesis ties all modules back to the invariant that training is budgeted
  optimization.
