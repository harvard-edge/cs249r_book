# V2-07 Track Plan: When Failure Is Routine

## Chapter Invariant

At fleet scale, failures become routine; MTBF, checkpoint interval, lost work,
recovery time, and redundancy are design amounts rather than after-the-fact
incident notes.

The lab uses one shared Part A/B/C/D concept sequence for every track. Tracks do
not introduce different concepts. The selected track changes persona,
constraints, thresholds, evidence emphasis, failure mode, and memo framing.

## Reading Map

| Lab module | Chapter anchor | Claim or formula used |
|---|---|---|
| Opening | Purpose and Failure Analysis at Scale | Scale transforms component failure from rare event to routine fleet condition. |
| Part A | The mathematics of inevitable failure | `R_system(t) = exp(-N lambda t)` and `MTBF_system = MTBF_component / N`. |
| Part B | The Young-Daly law and Checkpoint interval from failure analysis | `tau_opt = sqrt(2 * T_write * MTBF_system)`; too-frequent and too-rare checkpoints both waste work. |
| Part C | Checkpointing, Recovery procedures, Warm restart vs cold restart | Lost work, recovery time, storage history, and write bandwidth trade against one another. |
| Part D | Serving fault tolerance, Redundancy and replication, Graceful degradation | Replication improves availability only when cost, failover latency, and guardrails remain feasible. |
| Synthesis | Summary and From resilience to resource management | Checkpointing, elasticity, and redundancy create the reliability plan that orchestration must schedule in V2-08. |

## Concept Inventory

Accepted concepts:

- Fleet-level MTBF collapses as independent components are added.
- Clean-completion probability is an amount produced by fleet size, component
  MTBF, and mission duration.
- Checkpoint interval has an optimum, not a maximum or minimum.
- Checkpoint policy must budget save overhead, expected rework, and restart
  overhead separately.
- Recovery policy trades state history, write bandwidth, storage footprint,
  recovery time, and failure coverage.
- Redundancy helps only when recovery objectives, cost budgets, performance
  budgets, and guardrail metrics all pass together.
- The synthesis memo must name the binding failure amount and reject a tempting
  alternative.

Rejected or deferred concepts:

- Full hardware fault taxonomy: used as failure-mode vocabulary, not a separate
  activity.
- Detailed SDC detection algorithms: mentioned in policy coverage and Math Peek,
  deferred to deeper reliability/debugging work.
- Full distributed checkpoint protocols such as two-phase commit: referenced in
  source notes, not simulated.
- Fault injection tooling details: appears as validation requirement, not a
  separate tool tutorial.
- Complete serving architecture design: narrowed to redundancy, state loss, and
  graceful degradation guardrails.

## Concept Modules

### Part A: Concept Module - Aggregate MTBF Falls as Fleet Size Grows

Chapter claim:
- Individual component reliability does not compose by intuition. For independent
  components, aggregate failure rate grows with fleet size, so
  `MTBF_system = MTBF_component / N`.

Student prior:
- "A reliable component stays reliable when deployed in a large fleet."

Track lens:
- iPhone: mobile product engineer watches a staged rollout across many phones.
- Oura Ring: wearable firmware engineer watches sensing or sync gaps across a
  tiny battery fleet.
- RoboTaxi: safety platform engineer watches vehicle-hours accumulate across a
  live fleet.
- Cloud Fleet: SRE watches accelerator-hours accumulate during a training or
  service window.

Activity beats:
1. Scenario: a named track stakeholder must decide whether the current fleet can
   complete a mission window without a visible failure.
2. Prediction: structured radio asks how MTBF changes when fleet size grows 10x.
3. Manipulation: fleet-size and duration sliders change aggregate exposure.
4. Evidence: log-scale MTBF curve plus exact table show current MTBF, expected
   failures, and clean-run probability.
5. Consequence: failure banner appears when expected failures exceed one or clean
   probability misses the track target.
6. Math Peek/source model: exponential reliability and inverse MTBF scaling.
7. Checkpoint: student chooses the planning amount that should drive recovery
   design.

Evidence saved:
- fleet size, duration, component MTBF, system MTBF, expected failures,
  clean-run probability, binding exposure amount.

### Part B: Concept Module - Checkpoint Interval Has an Optimum

Chapter claim:
- The Young-Daly interval balances checkpoint write overhead against expected
  lost work. Making checkpoints infinitely frequent or rare both loses capacity.

Student prior:
- "More checkpointing is always safer" or "the longest possible interval
  maximizes useful work."

Track lens:
- All tracks use the same Young-Daly mechanism. Track defaults change checkpoint
  write time, state size, overhead threshold, and what "wasted work" means.

Activity beats:
1. Scenario: the stakeholder must set a checkpoint/rollback cadence before
   production failure pressure rises.
2. Prediction: structured radio asks which interval policy is safest.
3. Manipulation: checkpoint write time and chosen interval sliders expose the
   U-shaped tax.
4. Evidence: Young-Daly curve decomposes save overhead, rework overhead, and
   total waste.
5. Consequence: warning banner names whether the current interval is too
   frequent, too rare, or near optimum.
6. Math Peek/source model: `tau_opt = sqrt(2 * T_write * MTBF_system)` and
   `waste = T_write/tau + tau/(2*MTBF)`.
7. Checkpoint: student chooses the operational fix if the checkpoint tax exceeds
   the track threshold.

Evidence saved:
- write time, chosen interval, optimal interval, total waste percent, dominant
  side of the U-curve.

### Part C: Concept Module - Lost Work and Recovery Policy Trade Storage, Write Bandwidth, Time, and Resilience

Chapter claim:
- A recovery plan is not just "restore from checkpoint." The policy must account
  for lost work since the last checkpoint, detection, restart, loading, warmup,
  checkpoint history, and failure-mode coverage.

Student prior:
- "Once checkpointing exists, every failure has the same recovery path."

Track lens:
- iPhone: staged rollout rollback and local fallback must cap user-visible bad
  sessions without excessive device or CDN storage.
- Oura Ring: firmware rollback and safe mode must protect overnight sensing
  while respecting flash/radio energy.
- RoboTaxi: degraded mode and redundant state must preserve safety margin during
  sensor or compute faults.
- Cloud Fleet: checkpoint, warm restart, and replicated state must keep training
  or service goodput inside RTO and storage budgets.

Activity beats:
1. Scenario: an incident arrives; the stakeholder must choose the least costly
   recovery path that still covers the failure.
2. Prediction: structured radio asks which amount usually dominates the failure
   cost.
3. Manipulation: recovery-policy, checkpoint-history, and write-bandwidth
   controls expose storage/write/recovery trade-offs.
4. Evidence: policy table compares lost work, recovery time, storage footprint,
   write bandwidth, coverage, and feasibility.
5. Consequence: boundary callout names the violated objective if storage,
   bandwidth, or recovery objective fails.
6. Math Peek/source model:
   `T_failure = lost_work + T_detect + T_restart + T_load + T_warmup`; history
   and replication change storage and coverage.
7. Checkpoint: student chooses which failure type remains uncovered and what
   validation drill is required.

Evidence saved:
- selected policy, lost-work minutes, recovery minutes, storage GB, write GB/s,
  covered failure classes, uncovered failure class, policy feasibility.

### Part D: Concept Module - A Fault-Tolerance Plan Must Satisfy Recovery Objective, Cost, and Performance Guardrails

Chapter claim:
- Redundancy and graceful degradation are only valid if they meet recovery
  objectives without violating cost, latency/performance, quality, or safety
  guardrails.

Student prior:
- "More replicas or the strongest safety path is automatically the best plan."

Track lens:
- iPhone: responsiveness, quality, battery/thermal headroom, and staged rollback.
- Oura Ring: signal quality, flash/radio budget, and offline safe mode.
- RoboTaxi: safety margin, tail latency, and fail-closed degraded operation.
- Cloud Fleet: SLA, cost, utilization, and capacity headroom.

Activity beats:
1. Scenario: a design review demands one plan that passes all guardrails.
2. Prediction: structured radio asks which guardrail most often rejects the
   naive plan.
3. Manipulation: plan and replica count controls let the student test best-effort,
   graceful degradation, and redundant safety paths.
4. Evidence: guardrail table and availability chart show recovery, cost,
   latency/performance, quality, and guardrail pass/fail state.
5. Consequence: red failure state appears until all guardrails pass; the binding
   amount is named.
6. Math Peek/source model: `A_system = 1 - (1 - A_single)^k` plus
   `feasible = recovery_ok and cost_ok and performance_ok and guardrail_ok`.
7. Checkpoint: student selects the final plan and rejected alternative.

Evidence saved:
- final plan, replica count, availability, failover/recovery time, cost, latency,
  quality, guardrail, binding failure amount, rejected alternative.

## Synthesis: Reliability Memo

Student output:
- A reliability memo with checkpoint/replication policy, binding failure amount,
  rejected alternative, and V2-08 orchestration implication.

Required memo fields:
- selected track and stakeholder
- checkpoint interval and recovery policy
- replication or graceful-degradation plan
- binding failure amount from Parts A-D
- rejected alternative and why it failed
- validation drill
- carry-forward orchestration implication for V2-08

Ledger output:
- `track_id`
- `scenario_id`
- `fleet_size`
- `system_mtbf_h`
- `checkpoint_interval_min`
- `young_daly_interval_min`
- `checkpoint_tax_pct`
- `recovery_policy`
- `lost_work_min`
- `recovery_time_min`
- `replication_plan`
- `replica_count`
- `binding_failure_amount`
- `rejected_alternative`
- `v2_08_orchestration_implication`

## Track Narratives

| Track | Persona | Constraints | Failure mode | Evidence emphasis | Report frame |
|---|---|---|---|---|---|
| iPhone | Mobile product engineer | battery, thermal, responsiveness, quality, privacy | bad rollout, app crash, offline fallback miss | user-visible failure probability, rollback latency, quality guardrail | mobile reliability memo for staged rollout and local fallback |
| Oura Ring | Wearable firmware engineer | SRAM/flash, battery, radio duty cycle, signal quality | firmware rollback, sensor dropout, sync gap | overnight sensing continuity, flash/storage footprint, recovery under radio budget | wearable safe-mode and rollback memo |
| RoboTaxi | Autonomous vehicle platform engineer | p99/p999 latency, safety margin, sensor/compute redundancy | perception compute fault, sensor degradation, degraded-mode entry | safety margin, failover time, redundancy guardrail | safety recovery memo with fail-closed alternative |
| Cloud Fleet | Fleet service owner | SLA, throughput, utilization, cost, capacity headroom | accelerator/node failure, checkpoint storm, replica failure | aggregate MTBF, checkpoint tax, RTO, cost/SLA pass state | SRE reliability memo for training/service fleet |

## Mechanics Plan

| Belt | Concrete mechanics | Why used |
|---|---|---|
| Opening | track selector, track context, reading map, invariant card | Frames the same concept sequence through the selected track. |
| Prediction | `mo.ui.radio` in each module | Forces prior commitment before evidence is shown. |
| Control | fleet, duration, checkpoint write, interval, policy, history, bandwidth, plan, replicas | Manipulates the design amounts named in the invariant. |
| Evidence | Plotly MTBF curve, Young-Daly curve, policy tables, availability/guardrail charts | Makes the decision boundary visible and auditable. |
| Failure | callouts with explicit value, limit, unit, and mitigation | Creates reversible failures students can fix with controls. |
| Source | Math Peek accordions and source-model text | Ties each module to chapter formulas and assumptions. |
| Decision | checkpoint radio controls and final memo fields | Converts evidence into a track-specific decision. |
| Ledger | `DesignLedger.save` plus report export panel | Carries reliability amounts into V2-08 orchestration. |

## Evidence and Ledger Plan

Every chart that drives a decision has a table fallback with exact values. Color
is not the only pass/fail signal; callouts state the violated amount, limit,
unit, and mitigation. The final report contains all selected values even if the
student does not inspect the visual.

The binding amount is selected by severity from:
- expected failures during the mission window
- checkpoint tax percent
- recovery objective miss
- storage or write-bandwidth miss
- cost miss
- latency/performance miss
- quality or guardrail miss

The synthesis stores that binding amount in the Design Ledger so V2-08 can
interpret it as a scheduling/orchestration requirement.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Existing shared renderer is too generic for concept modules | Replace only this notebook with explicit local helpers; preserve bootstrap, track selector, report, and ledger patterns. |
| Fault-tolerance solvers are not exposed as shared MLSysIM APIs for every track | Use notebook-local `v2_07_` helpers with formulas tied directly to chapter anchors and track registry defaults. |
| Scenario thresholds are partly local because shared variant metadata is generic | Keep thresholds notebook-local, prefix helpers, and expose assumptions in Math Peek/source text. |
| Parallel workers may edit other labs | Restrict edits to `lab_07_fault_tolerance.py` and this track plan only. |
| Checkpointing is training-centric while tracks include devices and serving | Keep the concept sequence identical but interpret checkpoint as rollback/snapshot/state preservation for each track. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A - Aggregate MTBF | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, fleet/duration manipulation, MTBF chart/table, boundary, Math Peek, checkpoint. |
| Part B - Checkpoint optimum | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, write/interval manipulation, Young-Daly curve, consequence, Math Peek, checkpoint. |
| Part C - Recovery policy | 3 | 3 | 3 | 3 | 3 | 2 | Pass: prediction, policy/history/bandwidth manipulation, policy table, failure boundary, Math Peek, checkpoint; local thresholds documented. |
| Part D - Guardrail plan | 3 | 3 | 3 | 3 | 3 | 2 | Pass: prediction, plan/replica manipulation, availability/guardrail evidence, reversible failure, Math Peek, final plan. |
| Synthesis - Reliability memo | 3 | 3 | 3 | 3 | 3 | 3 | Pass: memo binds evidence to invariant, rejected alternative, and V2-08 orchestration implication. |

Minimum acceptance check:
- No dimension below 2.
- Every module includes prediction, manipulation, evidence, consequence,
  Math Peek/source model, and checkpoint/report decision.
- Reversible failure states appear in Parts A-D.
- Synthesis ties checkpointing, recovery policy, and redundancy back to the
  chapter invariant and V2-08 orchestration.
