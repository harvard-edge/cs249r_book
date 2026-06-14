# V2-04 Track Plan: Data Storage Concept Modules

## Chapter Invariant

Storage feeds both training and serving. Throughput, locality, consistency,
checkpoint load, and lifecycle policy shape usable system capacity because the
consumer only sees data that arrives at the right place, time, and evidence
quality.

The lab has one shared Part A/B/C/D concept sequence. Tracks do not introduce
separate concepts; the selected track changes the persona, constraints,
thresholds, evidence emphasis, failure mode, and report framing.

## Reading Map

| Module | Chapter anchor | Claim used in the lab |
|---|---|---|
| Opening | Purpose; The Fuel Line | Storage is an active fuel line for training data, model weights, checkpoints, and serving state. |
| Part A | How ML Workloads Invert Storage Assumptions; The Data Pipeline Equation | Required bandwidth scales with consumer count, target utilization, data volume per batch, and iteration time. If supply is lower than demand, backlog and accelerator starvation grow. |
| Part B | The ML Storage Hierarchy; Data locality and placement | Placing data closer to the consumer changes latency, available bandwidth, egress cost, and staleness risk. |
| Part C | Checkpoint Storage; Distributed checkpoint coordination | Checkpoint and consistency choices trade pause time, write storms, restore evidence, and recovery complexity. |
| Part D | Storage Economics; Tiering strategies; Fallacies and Pitfalls; Synthetic Fuel Line | Lifecycle policy must jointly satisfy freshness, retention, cost, durability, provenance, and reliability guardrails. |
| Synthesis | Summary | Storage architecture must name the binding storage amount and carry that constraint into distributed training. |

## Concept Inventory

### Accepted Concepts

| Concept | Why accepted | Module |
|---|---|---|
| Storage throughput must match consumer demand | Produces a visible backlog and starvation boundary. | Part A |
| Locality and placement change latency and bandwidth cost | Same data amount behaves differently depending on distance and cache placement. | Part B |
| Consistency and checkpointing trade recovery evidence against write storms | Makes checkpointing a systems design choice, not a checkbox. | Part C |
| Lifecycle policy must satisfy simultaneous guardrails | Forces a design choice across freshness, retention, cost, and reliability. | Part D |
| Storage architecture memo | Converts measurements into a carry-forward decision for distributed training. | Synthesis |

### Rejected Or Deferred Concepts

| Candidate | Reason rejected for this lab |
|---|---|
| Full taxonomy of storage tiers | Useful background, but a taxonomy without a decision does not create enough student consequence. |
| GPUDirect Storage internals | Important chapter topic, but it is a mechanism detail; here it appears as a source-model note under locality rather than a separate module. |
| Vector index retrieval design | Strong serving counterexample, but it would create a second concept sequence instead of reinforcing the shared storage invariant. |
| Synthetic data provenance in depth | Retained as a lifecycle/report guardrail; full provenance modeling belongs in a later governance lab. |
| Detailed distributed filesystem implementation | Deferred to systems readings; the lab uses capacity, latency, and write-storm amounts rather than filesystem internals. |

## Track Narratives

| Track | Persona | Constraint emphasis | Failure mode | Evidence emphasis | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile product engineer | local cache, privacy, uplink, battery-visible delay | consented evidence upload backs up while local serving stays responsive | upload backlog, local-vs-cloud latency, short raw retention | privacy-aware storage memo for a mobile feature |
| Oura Ring | Wearable firmware engineer | flash, BLE sync, battery duty cycle, health-signal evidence | summaries fit but raw windows exceed flash/sync envelope | retained MB, sync delay, missed waveform evidence | flash-aware retention and phone-sync memo |
| RoboTaxi | Autonomous vehicle platform engineer | sensor ingest, local triage, depot upload, rare-event evidence | raw fleet logs cannot be moved fast enough for safety replay | GB/s sensor flow, rare-event retention, restore evidence | safety-evidence storage and depot-upload memo |
| Cloud Fleet | Platform/SRE lead | object store, preprocessing, sharding, cache, checkpoint bursts | accelerators starve or checkpoint writes consume shared bandwidth | GB/s fuel line, cache locality, checkpoint storm, lifecycle cost | storage architecture memo for distributed training |

## Concept Modules

### Part A: Concept Module - Throughput Must Match Consumer Demand

- Chapter claim: `BW_required = N_consumer x target_utilization x data_per_step / iteration_time`.
- Student prior: storage capacity sounds sufficient, so students may ignore bandwidth.
- Scenario: the track stakeholder must feed a training, serving, or evidence consumer without building a backlog.
- Prediction: identify which amount will bind first: storage/read throughput, preprocessing, network/upload, or no backlog.
- Manipulation: sweep data demand pressure.
- Evidence: grouped bar chart and table compare stage demand, stage capacity, utilization, backlog GB/hour, and consumer starvation.
- Consequence/failure: if a stage has utilization above 100 percent, backlog grows and the consumer is starved.
- Math Peek/source model: pipeline bandwidth equation plus backlog growth `max(0, demand - capacity) x time`.
- Checkpoint/report decision: choose whether to add throughput, reduce demand, move placement closer, or accept a documented backlog.
- Ledger fields: `part_a_prediction`, `demand_multiplier`, `throughput_bottleneck`, `backlog_gb_per_hour`, `starvation_pct`, `part_a_checkpoint`.

### Part B: Concept Module - Locality And Placement Change Latency And Bandwidth Cost

- Chapter claim: hierarchy and locality decide whether bytes arrive with acceptable latency, cost, and staleness.
- Student prior: a cache is just a capacity optimization.
- Scenario: the same data must be placed remotely, regionally, locally, or in a summarized tier.
- Prediction: choose which placement class can satisfy the track guardrails.
- Manipulation: choose a placement policy and sweep working-set pressure.
- Evidence: placement frontier chart plus table with request latency, daily bytes moved, egress or movement cost, freshness lag, and feasibility.
- Consequence/failure: remote placement may have enough capacity but violate latency, cost, privacy, or freshness.
- Math Peek/source model: `latency = base + tail + request_MB x 8 / bandwidth_Gbps`; `movement_cost = GB_moved x price_per_GB`.
- Checkpoint/report decision: record the placement policy and rejected alternative.
- Ledger fields: `part_b_prediction`, `placement_policy`, `working_set_pressure`, `placement_latency_ms`, `placement_cost_per_day`, `part_b_checkpoint`.

### Part C: Concept Module - Consistency And Checkpointing Trade Recovery Evidence Against Write Storms

- Chapter claim: checkpoint design minimizes exposed pause while proving that the system can restore a consistent point in time.
- Student prior: more frequent checkpointing is always safer.
- Scenario: the stakeholder must pick a checkpoint/evidence consistency policy before failures or audits.
- Prediction: choose whether synchronous durable writes, async local staging, incremental verified checkpoints, or fast unverified writes survive.
- Manipulation: choose checkpoint policy and checkpoint interval.
- Evidence: pause/write-storm chart and table show pause seconds, durable delay, write GB/hour, restore evidence, and lost-work exposure.
- Consequence/failure: short intervals reduce lost work but can create write storms; fast writes without verification do not count as recovery evidence.
- Math Peek/source model: `T_write = checkpoint_size / write_bandwidth`; write storm `= checkpoint_size x writes_per_hour`.
- Checkpoint/report decision: record the recovery policy and the recovery evidence it buys.
- Ledger fields: `part_c_prediction`, `checkpoint_policy`, `checkpoint_interval_min`, `checkpoint_pause_s`, `write_storm_gb_per_hour`, `restore_evidence_pct`, `part_c_checkpoint`.

### Part D: Concept Module - Lifecycle Policy Must Satisfy Freshness, Retention, Cost, And Reliability Guardrails

- Chapter claim: tiering is only valid when freshness, retention, cost, reliability, and provenance guardrails are all inside the operating envelope.
- Student prior: lifecycle policy is an after-the-fact cost cleanup task.
- Scenario: the stakeholder must decide what stays hot, what becomes warm or cold, what is summarized, and what is discarded.
- Prediction: choose which lifecycle class satisfies all guardrails.
- Manipulation: choose lifecycle policy and freshness target.
- Evidence: stacked cost chart and policy table show hot/warm/cold footprint, monthly cost, freshness lag, retention days, durability, and feasibility.
- Consequence/failure: the cheapest policy can fail freshness or evidence retention; the freshest policy can fail cost.
- Math Peek/source model: `retained_GB = ingest_GB_per_day x retention_days x reduction_factor`; monthly cost is footprint by tier times tier price.
- Checkpoint/report decision: record the lifecycle policy and the guardrail that made alternatives invalid.
- Ledger fields: `part_d_prediction`, `lifecycle_policy`, `freshness_target_min`, `monthly_cost`, `retention_days`, `durability_pct`, `part_d_checkpoint`.

### Synthesis: Storage Architecture Memo

- Required artifact: storage architecture memo for the selected track.
- Memo must include:
  - selected placement policy,
  - selected lifecycle policy,
  - binding storage amount,
  - rejected alternative,
  - checkpoint/recovery evidence,
  - carry-forward implication for distributed training.
- Ledger fields: `selected_track`, `selected_placement_policy`, `selected_lifecycle_policy`, `binding_storage_amount`, `rejected_alternative`, `distributed_training_implication`, `architecture_memo`.

## Mechanics And Evidence Plan

| Belt | Mechanic | Evidence |
|---|---|---|
| Opening | track selector, track mission, chapter invariant panel | selected track and track-specific storage mission |
| Prediction | `mo.ui.radio` predictions in every part | prediction-vs-actual callouts |
| Control | demand slider, placement dropdown, working-set slider, checkpoint dropdown, interval slider, lifecycle dropdown, freshness slider | reversible boundary exploration |
| Evidence | Plotly grouped bars, frontier scatter, checkpoint bars, lifecycle stacked bars, HTML table fallbacks | exact values for every chart |
| Failure | red failure callouts when utilization, latency, pause, write storm, cost, freshness, retention, or reliability fails | concrete value, limit, unit, and mitigation |
| Source | Math Peek accordions plus `source_trace` blocks | chapter formulas, MLSysIM hardware/model refs, notebook-local scenario assumptions |
| Decision | checkpoint radio in every part plus synthesis decision and memo | design choices with evidence |
| Ledger | `DesignLedger.save(chapter=4, design=...)` | carry-forward storage architecture for V2-05 distributed training |

## Implementation Notes

- Existing shared helpers used: `track_selector`, `track_context`, `track_arc_context`, `source_trace`, `build_lab_report`, `report_export_panel`, `DesignLedger`, `COLORS`, `LAB_CSS`, and `apply_plotly_theme`.
- New support remains notebook-local in `lab_04_data_storage.py` and helper functions use the `v2_04_` prefix.
- Hardware references come from canonical track profiles and MLSysIM hardware/model refs. Scenario thresholds that do not exist in MLSysIM are documented in source traces as notebook-local lab assumptions.
- Every plot has a table fallback with exact values.
- No shared helper, shared test, implementation-note, or other lab file is edited.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A - Throughput/backlog | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, demand sweep, backlog failure, chart/table, Math Peek, checkpoint. |
| Part B - Locality/placement | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, placement/pressure manipulation, frontier evidence, boundary callout, Math Peek, checkpoint. |
| Part C - Consistency/checkpointing | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, interval/policy manipulation, write-storm failure, recovery evidence, Math Peek, checkpoint. |
| Part D - Lifecycle policy | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, policy/freshness manipulation, guardrail table, cost chart, Math Peek, checkpoint. |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 | Pass: memo requires selected policies, binding amount, rejected alternative, and distributed-training implication. |

Minimum acceptance is met: no dimension below 2, every module has at least five student-facing beats, reversible failure states exist in Parts A-D, and synthesis ties the modules back to the chapter invariant.
