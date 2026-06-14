# V2-05 Track Plan: Distributed Training

## Chapter Invariant

Distributed training trades compute for communication, memory, synchronization,
and convergence cost. No parallelism strategy removes the work; each strategy
moves the binding amount to a different part of the fleet.

The lab has one shared Part A/B/C/D concept sequence. Tracks do not create
different concepts. The selected track changes persona, constraints, thresholds,
evidence emphasis, failure mode, and report framing.

## Required Reading Map

| Lab module | Chapter anchor | Claim or source model used in the lab |
|---|---|---|
| Opening | `#sec-distributed-training-systems-systems-multimachine-scaling-fundamentals-ff96`, `#sec-distributed-training-systems-physics-cluster-a42d` | The fleet step-time law decomposes distributed training into compute, communication, synchronization, and overlap terms. |
| Part A | `#sec-distributed-training-systems-systems-engineering-tradeoffs-selecting-parallelism-strategy-b344`, `#sec-distributed-training-systems-systems-data-parallelism-6132`, `#sec-distributed-training-systems-systems-pipeline-parallelism-8748`, `#sec-distributed-training-systems-systems-tensor-parallelism-d76e` | Data, tensor, and pipeline parallelism solve different capacity problems and shift the binding amount to AllReduce, intra-layer collectives, activation transfer, or pipeline bubbles. |
| Part B | `#sec-distributed-training-systems-systems-distributed-training-efficiency-metrics-9488`, `#sec-distributed-training-systems-systems-physics-scaling-amdahls-law-communication-4d7f`, `#sec-distributed-training-systems-systems-pipeline-parallelism-8748` | Scaling efficiency falls when exposed communication or pipeline bubble time dominates useful compute. |
| Part C | `#sec-distributed-training-systems-systems-memoryefficient-data-parallelism-zero-fsdp-0e69`, `#sec-distributed-training-systems-systems-critical-batch-size-parallelism-hurt-4961`, `#sec-distributed-training-systems-systems-learning-rate-scaling-rules-fa26` | Mixed-precision Adam state, ZeRO/FSDP sharding, global batch, gradient accumulation, and critical batch size jointly set memory, convergence, and step time. |
| Part D | `#sec-distributed-training-systems-systems-parallelism-strategy-comparison-d92a`, `#sec-distributed-training-systems-systems-framework-integration-cf71`, `#sec-distributed-training-systems-summary` | A distributed training plan is valid only when time, memory, communication, and evidence guardrails all pass together. |
| Synthesis | `#sec-distributed-training-systems-summary`, `#sec-collective-communication` forward reference | The memo names the selected parallelism, binding bottleneck, rejected alternative, and collective-communication implication for V2-06. |

Matching concept YAML anchors:

- Primary concepts: Distributed Training, Data Parallelism, Tensor
  Parallelism, Pipeline Parallelism, Hybrid Parallelism, ZeRO, FSDP, Scaling
  Efficiency, Communication-Computation Ratio, Critical Batch Size.
- Secondary concepts: Distributed Step-Time Law, Communication Wall,
  Coordination Tax, Memory Wall, Gradient Synchronization, Global Batch Size,
  Gradient Accumulation, Optimizer State Sharding, Activation Checkpointing,
  AllReduce, AllGather, ReduceScatter, Pipeline Bubble, 1F1B Scheduling,
  Bandwidth Matching.
- Methodologies: distributed step-time modeling, communication-computation
  ratio diagnosis, parallelism strategy selection, ZeRO stage selection,
  scaling efficiency analysis, critical batch size estimation, tensor and
  pipeline partitioning, MFU and bottleneck attribution.

## Concept Inventory

### Accepted Concepts

| Concept | Why accepted | Lab role |
|---|---|---|
| Parallelism shifts the binding amount | Directly supports the chapter invariant and creates a measurable strategy choice. | Part A calibration concept. |
| Scaling efficiency and pipeline bubbles | Shows why adding devices can reduce useful work when communication or idle stages dominate. | Part B mechanism concept. |
| Batch size, optimizer state, and sharding | Connects memory fit, convergence, and step time instead of treating them as independent knobs. | Part C transfer concept. |
| Multi-constraint training plan | Forces a decision across time, memory, communication, and evidence. | Part D design concept. |
| Collective implication | Carries logical traffic patterns forward without implementing V2-06 collective algorithms. | Synthesis. |

### Rejected Or Deferred Concepts

| Concept | Reason rejected for this lab | Destination |
|---|---|---|
| Ring/tree/hierarchical collective algorithm derivations | V2-06 owns collective algorithm mechanics; V2-05 should only name which collective pattern the plan creates. | V2-06 Collective Communication. |
| Full RLHF/PPO/DPO fleet orchestration | Important chapter material but would add a second multi-model scheduling lab concept. | Reading map and future alignment labs. |
| Expert parallelism and MoE routing | Too specialized for the required A-D sequence and All-to-All details belong near collectives. | Synthesis note or V2-06 extension. |
| Fault-tolerance checkpoint optimization | Already gets a dedicated reliability treatment elsewhere; here checkpointing appears only as evidence risk. | V2-07 Fault Tolerance. |
| Real framework API walkthroughs | The lab teaches amount-system reasoning rather than PyTorch or Megatron API usage. | Source notes only. |

## Shared Concept-Module Sequence

### Part A: Concept Module - Parallelism Moves The Binding Amount

```yaml
concept_module:
  part_label: "Part A"
  concept_name: "Data, tensor, and pipeline parallelism shift the binding amount"
  chapter_claim: "The 3D parallelism cube maps logical splits to memory, communication, and idle-time costs."
  reading_connection:
    chapter_section: "#sec-distributed-training-systems-systems-engineering-tradeoffs-selecting-parallelism-strategy-b344"
    claim_or_formula: "N_total=d*p*t; each axis changes memory footprint and traffic pattern"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "track-specific owner of the training/deployment decision"
    system_decision: "choose the first parallelism/adaptation strategy to relieve the current bottleneck"
  student_prior:
    expected_belief: "The best strategy is the one that uses the most devices."
    productive_failure: "The chosen strategy relieves memory or time but creates a larger communication, bubble, or convergence bottleneck."
  storyline:
    beat_1_scenario: "A stakeholder has a model/update workload that does not fit the current training envelope."
    beat_2_prediction: "Student predicts whether data, tensor, pipeline, or adaptation/off-device training becomes binding."
    beat_3_controls: "Student changes strategy and worker count or pipeline depth."
    beat_4_evidence: "Strategy comparison chart/table shows memory per device, communication time, bubble time, and binding amount."
    beat_5_consequence: "Failure callout names the shifted cost and mitigation."
    beat_6_math_peek: "Math Peek ties N_total=d*p*t and strategy traffic patterns to the chapter decision tree."
    beat_7_checkpoint: "Student chooses the strategy to carry forward."
  mechanics:
    controls: ["prediction radio", "strategy dropdown", "worker/stage slider", "checkpoint radio"]
    graphs: ["strategy amount stacked bar", "exact comparison table"]
    failure_state: "selected strategy violates memory, communication, or bubble threshold"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partA_prediction", "partA_strategy", "partA_scale", "partA_binding_amount", "partA_checkpoint"]
    downstream_use: "Part D uses the selected strategy as the baseline plan."
```

### Part B: Concept Module - Scaling Efficiency Falls When Overhead Dominates

```yaml
concept_module:
  part_label: "Part B"
  concept_name: "Scaling efficiency falls when communication or bubbles dominate"
  chapter_claim: "The fleet step-time law makes useful speedup depend on exposed communication, synchronization, and idle time."
  reading_connection:
    chapter_section: "#sec-distributed-training-systems-systems-physics-scaling-amdahls-law-communication-4d7f"
    claim_or_formula: "eta_scaling=T_compute/(N*T_step(N)); bubble=(p-1)/(m+p-1)"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "same selected-track owner"
    system_decision: "decide whether scaling out still buys useful work"
  student_prior:
    expected_belief: "Doubling devices should nearly double training speed."
    productive_failure: "At the selected scale, exposed communication or bubble time consumes the expected speedup."
  storyline:
    beat_1_scenario: "The team asks whether the next scale step is worth buying or scheduling."
    beat_2_prediction: "Student predicts which overhead term will first push efficiency below the track threshold."
    beat_3_controls: "Student changes accelerator count, interconnect tier, and microbatch count."
    beat_4_evidence: "Scaling curve and step-time table show useful speedup, efficiency, communication share, and bubble share."
    beat_5_consequence: "Failure callout shows the first point where the plan becomes inefficient."
    beat_6_math_peek: "Math Peek shows the fleet step-time law and pipeline bubble equation."
    beat_7_checkpoint: "Student chooses whether to scale out, stay local with accumulation, or change parallelism."
  mechanics:
    controls: ["prediction radio", "GPU-count slider", "network dropdown", "microbatch slider", "checkpoint radio"]
    graphs: ["scaling efficiency line chart", "step-time exact table"]
    failure_state: "scaling efficiency falls below the track threshold or bubble share exceeds threshold"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partB_prediction", "partB_gpus", "partB_network", "partB_microbatches", "partB_efficiency", "partB_checkpoint"]
    downstream_use: "Part C and Part D use efficiency as the time/evidence guardrail."
```

### Part C: Concept Module - Batch And Optimizer State Couple Memory, Convergence, And Step Time

```yaml
concept_module:
  part_label: "Part C"
  concept_name: "Batch size and optimizer state change memory, convergence, and step time"
  chapter_claim: "Sharding buys memory with messages, while global batch size changes the optimization regime."
  reading_connection:
    chapter_section: "#sec-distributed-training-systems-systems-memoryefficient-data-parallelism-zero-fsdp-0e69"
    claim_or_formula: "Adam state ~= 16 bytes/parameter; B_global=N*b*accum; B* ~= tr(Sigma)/||grad L||^2"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "same selected-track owner"
    system_decision: "choose batch, accumulation, and sharding policy that fits memory without wasting convergence"
  student_prior:
    expected_belief: "Bigger batches and deeper sharding are always better once memory fits."
    productive_failure: "FSDP can make memory pass but add step-time overhead, and oversized global batch can waste samples."
  storyline:
    beat_1_scenario: "The team tries to turn the Part A/B strategy into a trainable optimizer setup."
    beat_2_prediction: "Student predicts whether memory, convergence, or step time will bind after sharding."
    beat_3_controls: "Student changes per-device batch, accumulation, and ZeRO/FSDP stage."
    beat_4_evidence: "Memory ledger and convergence table show per-device state, global batch, critical-batch ratio, and exposed step time."
    beat_5_consequence: "Failure callout names OOM, convergence waste, or communication overhead."
    beat_6_math_peek: "Math Peek shows Adam bytes/parameter, ZeRO division, and critical batch formula."
    beat_7_checkpoint: "Student chooses the optimizer/batch policy for the final plan."
  mechanics:
    controls: ["prediction radio", "batch slider", "accumulation slider", "sharding dropdown", "checkpoint radio"]
    graphs: ["memory stacked bar", "global-batch vs critical-batch table"]
    failure_state: "OOM, critical-batch violation, or step-time threshold miss"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partC_prediction", "partC_batch", "partC_accumulation", "partC_sharding", "partC_memory_gb", "partC_batch_ratio", "partC_checkpoint"]
    downstream_use: "Part D evaluates final plan feasibility with memory and convergence guardrails."
```

### Part D: Concept Module - A Training Plan Must Pass Simultaneous Guardrails

```yaml
concept_module:
  part_label: "Part D"
  concept_name: "Distributed training plan must satisfy time, memory, communication, and evidence constraints"
  chapter_claim: "Parallelism strategy selection is a constraint satisfaction problem, not a single metric ranking."
  reading_connection:
    chapter_section: "#sec-distributed-training-systems-systems-parallelism-strategy-comparison-d92a"
    claim_or_formula: "valid plan = time_ok and memory_ok and communication_ok and evidence_ok"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "same selected-track owner"
    system_decision: "approve or revise the training architecture plan"
  student_prior:
    expected_belief: "The strategy with the best throughput is the strategy to approve."
    productive_failure: "A throughput winner can fail memory, communication, or evidence constraints."
  storyline:
    beat_1_scenario: "The stakeholder must sign a distributed-training memo for the selected track."
    beat_2_prediction: "Student predicts which guardrail rejects the naive plan."
    beat_3_controls: "Student changes candidate strategy, evidence strictness, and communication overlap."
    beat_4_evidence: "Guardrail matrix compares selected and rejected plans."
    beat_5_consequence: "Failure callout states which constraint invalidates the plan and how to recover."
    beat_6_math_peek: "Math Peek shows simultaneous feasibility inequalities and the V2-06 collective implication."
    beat_7_checkpoint: "Student chooses final approval/revision decision."
  mechanics:
    controls: ["prediction radio", "candidate dropdown", "evidence threshold dropdown", "overlap slider", "final decision radio"]
    graphs: ["guardrail matrix", "candidate comparison bar"]
    failure_state: "any guardrail fails"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partD_prediction", "partD_candidate", "partD_overlap", "partD_time_ok", "partD_memory_ok", "partD_comm_ok", "partD_evidence_ok", "partD_final_decision"]
    downstream_use: "Synthesis saves the distributed training memo and V2-06 implication."
```

### Synthesis: Distributed Training Memo

The synthesis memo must include:

1. Selected parallelism or off-device/adaptation plan.
2. Binding bottleneck amount.
3. Rejected alternative.
4. Evidence number from the guardrail matrix.
5. V2-06 collective-communication implication.

Ledger fields:

- `track_id`
- `scenario_id`
- `selected_parallelism`
- `training_location`
- `binding_bottleneck`
- `memory_per_device_gb`
- `scaling_efficiency`
- `critical_batch_ratio`
- `rejected_alternative`
- `collective_implication`
- `completed`

## Track Narratives

| Track | Persona | Same concepts realized as | Constraint emphasis | Natural failure | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile ML product lead | Personalization/update plan where the phone is a deployment and evidence source, not a full distributed trainer | local memory, privacy, battery/radio evidence, backend fine-tune budget | attempting full local training or unbounded raw telemetry upload | "Where does personalization happen, and what communication/evidence does it create?" |
| Oura Ring | TinyML firmware lead | Off-device training plus tiny calibration/update packaging for a wearable | SRAM/flash, duty cycle, intermittent sync, OTA size | pretending the ring can run full training or ship oversized updates | "What is trained off-device, what is calibrated locally, and what update package fits?" |
| RoboTaxi | Safety/perception platform lead | Fleet-data central training plus vehicle-local validation/deployment evidence | safety validation, p99 evidence, depot/cloud upload, training turnaround | missing validation evidence or central training deadline | "How does fleet evidence become a trainable model without weakening safety validation?" |
| Cloud Fleet | Training platform owner | Large-model distributed training using data/tensor/pipeline/FSDP choices | wall-clock target, HBM, NVLink/InfiniBand, scaling efficiency, MFU evidence | idle accelerators from communication wall or pipeline bubble | "Which 3D/sharded strategy should be scheduled, and what bottleneck remains?" |

Track deltas required in implementation:

- Persona and scenario copy change by track.
- Default strategy, worker counts, memory cap, time budget, efficiency threshold,
  critical-batch threshold, and evidence threshold change by track.
- Failure wording changes by track.
- Evidence emphasis changes by track: privacy/evidence for iPhone, SRAM/OTA for
  Oura, safety validation for RoboTaxi, cluster efficiency for Cloud Fleet.
- Final report framing changes by track.

## Mechanics, Evidence, And Ledger Plan

| Module | Mechanics | Evidence produced | Ledger/checkpoint output |
|---|---|---|---|
| Opening | Track selector, track mission, invariant card, reading map | Selected track and scenario | `track_id`, `scenario_id` |
| Part A | Prediction radio, strategy dropdown, scale slider, stacked amount chart/table, Math Peek | Memory, communication, bubble, convergence score, binding amount | `partA_*` prediction, strategy, scale, binding, checkpoint |
| Part B | Prediction radio, GPU-count slider, network dropdown, microbatch slider, scaling curve/table, Math Peek | Scaling efficiency, useful speedup, communication share, bubble share | `partB_*` prediction, scale, network, efficiency, checkpoint |
| Part C | Prediction radio, batch slider, accumulation slider, sharding dropdown, memory ledger/table, Math Peek | Per-device memory, global batch, critical-batch ratio, step overhead | `partC_*` prediction, batch, accumulation, sharding, memory and batch ratio |
| Part D | Prediction radio, candidate dropdown, overlap slider, evidence threshold dropdown, guardrail matrix, Math Peek | Time/memory/communication/evidence pass-fail and rejected alternative | `partD_*` prediction, candidate, guardrails, final decision |
| Synthesis | Memo builder, report export panel, Design Ledger save | Distributed training memo and V2-06 implication | `completed`, selected plan, binding amount, rejected alternative |

Accessibility and fallback requirements:

- Each chart has an adjacent exact-value HTML table.
- Feasibility is shown with PASS/FAIL labels, not color alone.
- Each failure state states value, limit, unit, and mitigation.
- Required controls are visible in the active module.
- The exported report contains the same evidence as the visuals.

## Implementation Notes

- Owned files only: `labs/vol2/lab_05_dist_train.py` and this track plan.
- No shared helper, test, implementation-note, or registry edits.
- Use existing helpers where possible: `track_selector`, `get_track_profile`,
  `get_lab_metadata`, `get_lab_track_variant`, `MathPeek`, `source_trace`,
  `build_lab_report`, `report_export_panel`, `DesignLedger`, `COLORS`,
  `LAB_CSS`, and `apply_plotly_theme`.
- Keep new support notebook-local with `v2_05_` prefixes.
- Scenario thresholds that are not direct MLSysIM facts are local pedagogical
  assumptions and must be surfaced in Math Peek/source trace.
- Preserve WASM bootstrap, relative wheel paths, canonical track selector, and
  Design Ledger save pattern.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Full distributed-training solver support may be broader than needed. | Use notebook-local amount models with formulas sourced from the chapter and MLSysIM hardware/model constants where available. |
| Mobile/wearable tracks can feel artificial if forced into cluster training. | Treat device as deployment/evidence/adaptation endpoint while keeping the same Part A-D concept sequence. |
| V2-06 collective details can leak into this lab. | Only name collective implication; do not derive ring/tree/hierarchical algorithms. |
| Scenario constants may appear unsourced. | Surface them as local teaching assumptions in Math Peek and source trace. |
| Dirty shared worktree from other lab workers. | Edit only owned V2-05 files and do not revert unrelated changes. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 2 | Pass |

Reversible failure states:

- Part A: selected strategy violates memory, communication, or bubble threshold;
  student can recover by changing strategy or scale.
- Part B: scaling efficiency falls below threshold; student can recover by
  changing GPU count, network tier, microbatches, or strategy.
- Part C: OOM, critical-batch violation, or step-time overhead miss; student can
  recover by changing batch, accumulation, or sharding.
- Part D: any guardrail fails; student can recover by choosing a different
  candidate, overlap level, or evidence threshold.

Synthesis gate:

- The memo is complete only when all four predictions are made and the final
  plan names selected parallelism, binding bottleneck, rejected alternative,
  evidence number, and V2-06 collective implication.
