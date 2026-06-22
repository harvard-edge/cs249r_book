# V2-06 Track Plan: Collective Communication

## Chapter Invariant

Collectives are algorithms whose latency and bandwidth terms depend on
topology, payload, overlap, and compression. The durable design question is not
"which backend is fastest?" but "which collective plan satisfies the selected
fleet envelope once alpha, beta, topology, exposed step time, and residual
training or safety risk are all accounted for?"

The lab has one shared Part A/B/C/D concept sequence. The selected track changes
persona, constraints, thresholds, evidence emphasis, failure mode, and report
framing. Tracks do not create different concepts.

## Required Reading Map

| Lab module | Chapter anchor | Claim or source model used in the lab |
|---|---|---|
| Opening | `#sec-collective-communication`, `#sec-communication-collective-operations-collective-operations-communication-fundamentals-44eb` | Communication is the fleet instruction set; gradient synchronization and analogous aggregation are constrained by latency, bandwidth, topology, and energy. |
| Part A | `#sec-communication-collective-operations-collective-operations-alphabeta-model-f9b4`, `#sec-communication-collective-operations-collective-operations-ring-allreduce-ffce`, `#sec-communication-collective-operations-tree-allreduce`, `#sec-communication-collective-operations-algorithm-crossover` | `T_ring = 2(N-1)alpha + 2((N-1)/N)M/beta`; tree-like collectives trade lower alpha depth for worse bandwidth terms; message size determines the crossover. |
| Part B | `#sec-communication-collective-operations-collective-operations-mapping-collectives-topology-3214`, `#sec-communication-collective-operations-collective-operations-hierarchical-allreduce-1338`, `#sec-communication-collective-operations-collective-operations-topology-detection-selection-2dc7`, `#sec-collective-communication-railoptimized-routing-nvidia-dgx-7081` | Hierarchical and topology-aware collectives exploit fast local links, scarce inter-node links, and rail/rank mapping; a topology can make a collective dominant, invalid, or misleading. |
| Part C | `#sec-communication-collective-operations-collective-operations-gradient-compression-7a5c`, `#sec-communication-collective-operations-error-feedback`, `#sec-communication-overlap`, `#sec-communication-overlap-limits` | Compression reduces payload but can harm convergence or evidence quality; overlap hides only the communication that has concurrent useful work and leaves an exposed residual. |
| Part D | `#sec-communication-collective-operations-collective-operations-fallacies-pitfalls-9cd0`, `#sec-communication-collective-operations-summary` | A communication plan is valid only when exposed time, topology assumptions, and track-specific quality/reliability guardrails pass together. |
| Synthesis | `#sec-communication-collective-operations-summary`, V2-07 forward connection to `#sec-fault-tolerance-reliability` | Communication choices carry reliability implications: topology mismatch, silent corruption, validation gaps, and recovery cost become V2-07 concerns. |

Matching concept YAML anchors:

- Primary concepts: Collective Communication, AllReduce, Ring AllReduce, Tree
  AllReduce, Hierarchical AllReduce, Alpha-Beta Model,
  Communication-Computation Overlap, Topology-Aware Routing, Gradient
  Compression, Error Feedback.
- Secondary concepts: Latency-Bound Communication, Bandwidth-Bound
  Communication, Critical Message Size, Algorithm Crossover Point, Bandwidth
  Hierarchy, Rail-Optimized Routing, NCCL, Gradient Bucket, Bucket Fusion,
  Nonoverlappable Cost, Silent Data Corruption.
- Methodologies: Alpha-Beta Communication Modeling, Ring-vs-Tree Crossover
  Analysis, Hierarchical AllReduce Decomposition, Three-Level Hierarchical
  Bandwidth Budgeting, Topology Detection and Path Selection, Compression
  Payback Analysis, Compression-Aware Optimization, Layer-by-Layer Overlap,
  Bucket Fusion Tuning, Collective Benchmarking.

## Concept Inventory

### Accepted Concepts

| Concept | Why accepted | Lab role |
|---|---|---|
| Ring/tree alpha-beta cost decomposition | Converts "collective overhead" into separable latency and bandwidth terms students can calculate and manipulate. | Part A calibration concept. |
| Topology-dependent feasibility and dominance | Explains why the same collective formula changes under flat, hierarchical, or constrained track topologies. | Part B mechanism concept. |
| Overlap and compression as conditional optimizations | Shows that hiding or shrinking communication changes exposed time but can create convergence, fidelity, scheduling, or memory risk. | Part C transfer concept. |
| Simultaneous communication guardrails | Forces students to approve a plan only when exposed step time and topology assumptions both survive track constraints. | Part D design concept. |
| Communication design review with reliability implication | Carries selected algorithm/topology/optimization into the next chapter's fault-tolerance framing. | Synthesis. |

### Rejected Or Deferred Concepts

| Concept | Reason rejected for this lab | Destination |
|---|---|---|
| Full collective primitive taxonomy | The lab needs AllReduce-style amount-system reasoning, not a catalog of Broadcast, Reduce, AllGather, ReduceScatter, AllToAll, and Send/Recv. | Reading map and synthesis notes. |
| Recursive halving-doubling and double binary tree implementation details | Useful chapter concepts, but adding them as separate algorithms would dilute the ring/tree/topology concept sequence. | Mentioned as rejected alternative or library nuance. |
| Communication library selection workflow | NCCL/MPI/Gloo selection is operationally important but would become a backend catalog instead of a concept module. | Source trace and future implementation/ops labs. |
| SHARP/in-network reduction resource limits | Too specialized for the shared track sequence; topology and hierarchy are enough for Part B. | Reading/source note. |
| Full convergence proof for error feedback | Part C needs the consequence of compression risk, not a proof-heavy optimizer lab. | Math Peek/source model only. |
| Silent data corruption instrumentation | Important V2-07 bridge but not the main V2-06 manipulation. | Synthesis reliability implication. |

## Shared Concept-Module Sequence

### Part A: Concept Module - Ring And Tree Costs Bind Different Alpha/Beta Terms

```yaml
concept_module:
  part_label: "Part A"
  concept_name: "Ring and tree costs bind different alpha/beta terms"
  chapter_claim: "Ring AllReduce is bandwidth-optimal but pays O(N) latency; Tree AllReduce reduces latency depth but can pay a bandwidth penalty."
  reading_connection:
    chapter_section: "#sec-communication-collective-operations-algorithm-crossover"
    claim_or_formula: "T_ring = 2(N-1)alpha + 2((N-1)/N)M/beta; T_tree ~= 2log2(N)alpha + 2log2(N)M/beta"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "track-specific communication owner"
    system_decision: "decide whether the current payload should favor latency depth or bandwidth efficiency"
  student_prior:
    expected_belief: "Ring is always the best AllReduce because it is bandwidth-optimal."
    productive_failure: "Small payloads and large participant counts can make the alpha term dominate and favor tree-like schedules."
  storyline:
    beat_1_scenario: "A stakeholder must select a first-pass collective/aggregation schedule for the selected track."
    beat_2_prediction: "Student predicts whether ring, tree, or message-size dependence wins."
    beat_3_controls: "Student changes participant count, payload size, and fabric/link analogy."
    beat_4_evidence: "Stacked alpha/beta chart and table show ring/tree total time and binding term."
    beat_5_consequence: "Failure/boundary callout names the algorithm penalty and whether the step-time budget is already exposed."
    beat_6_math_peek: "Math Peek shows the ring and tree alpha/beta terms and the crossover intuition."
    beat_7_checkpoint: "Student chooses the algorithm family to carry into topology analysis."
  mechanics:
    controls: ["prediction radio", "participants slider", "payload slider", "fabric dropdown", "checkpoint radio"]
    graphs: ["stacked ring/tree alpha-beta chart", "exact evidence table"]
    failure_state: "selected algorithm exceeds track step-time budget before topology mitigation"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partA_prediction", "partA_checkpoint", "participants", "payload_gb", "fabric", "ring_ms", "tree_ms", "partA_binding_term"]
    downstream_use: "Part B tests whether topology changes the Part A algorithm decision."
```

### Part B: Concept Module - Topology Changes Which Collective Is Feasible Or Dominant

```yaml
concept_module:
  part_label: "Part B"
  concept_name: "Topology changes which collective is feasible or dominant"
  chapter_claim: "Hierarchical and topology-aware collectives are valid only when the physical fabric actually provides faster local tiers and aligned cross-node paths."
  reading_connection:
    chapter_section: "#sec-communication-collective-operations-collective-operations-hierarchical-allreduce-1338"
    claim_or_formula: "hierarchical time = local reduce-scatter + inter-node shard allreduce + local allgather; inter-node payload becomes M/G"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "same selected-track communication owner"
    system_decision: "choose a topology assumption and reject collectives that the track cannot support"
  student_prior:
    expected_belief: "Hierarchy always helps, or topology is only a display label."
    productive_failure: "When local groups collapse to one participant, or the track lacks a fast local tier, hierarchy cannot deliver its advertised multiplier."
  storyline:
    beat_1_scenario: "The same collective must run over the selected track's real or analogous topology."
    beat_2_prediction: "Student predicts whether flat ring, tree, or hierarchy is feasible/dominant."
    beat_3_controls: "Student changes local group size and topology assumption."
    beat_4_evidence: "Topology frontier and table compare flat, tree, and hierarchical candidates with feasible flags."
    beat_5_consequence: "Failure/boundary callout names topology mismatch or missing local/global tier."
    beat_6_math_peek: "Math Peek shows the M/G inter-node payload reduction and rank/rail assumption."
    beat_7_checkpoint: "Student chooses the topology assumption to carry into optimization."
  mechanics:
    controls: ["prediction radio", "participants slider", "local group slider", "fabric dropdown", "topology checkpoint radio"]
    graphs: ["message-size frontier", "topology candidate table"]
    failure_state: "hierarchical candidate is infeasible or fails topology guardrail"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partB_prediction", "partB_checkpoint", "local_group_size", "topology_guardrail", "dominant_candidate", "topology_rejected_reason"]
    downstream_use: "Part C uses the selected feasible topology when overlap and compression are tested."
```

### Part C: Concept Module - Overlap And Compression Hide Communication With Risk

```yaml
concept_module:
  part_label: "Part C"
  concept_name: "Overlap and compression can hide communication but create convergence or scheduling risk"
  chapter_claim: "Compression reduces the bandwidth term but can bias updates or remove evidence; overlap hides only communication that has useful concurrent work and leaves exposed residual time."
  reading_connection:
    chapter_section: "#sec-communication-overlap-limits and #sec-communication-collective-operations-error-feedback"
    claim_or_formula: "exposed = max(0, comm_ms - overlap_window_ms); e_{t+1} = (g_t + e_t) - v_t"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "same selected-track communication owner"
    system_decision: "choose an optimization level that reduces exposed time without violating quality, fidelity, battery, or convergence guardrails"
  student_prior:
    expected_belief: "Compression and async overlap are free speedups."
    productive_failure: "A high-compression or high-overlap setting can pass the time target while failing quality/fidelity or schedulability."
  storyline:
    beat_1_scenario: "The stakeholder tries to hide or shrink the selected feasible collective."
    beat_2_prediction: "Student predicts which residual risk remains after optimization."
    beat_3_controls: "Student changes compression ratio and overlap percent."
    beat_4_evidence: "Optimization table shows raw time, compressed time, exposed time, quality proxy, and schedule risk."
    beat_5_consequence: "Failure callout names the first optimization guardrail that fails."
    beat_6_math_peek: "Math Peek ties compression to payload M and overlap to exposed residual communication."
    beat_7_checkpoint: "Student chooses the optimization policy and validation test to carry into Part D."
  mechanics:
    controls: ["prediction radio", "compression slider", "overlap slider", "optimization checkpoint radio"]
    graphs: ["exposed-time bar", "optimization guardrail table"]
    failure_state: "quality/fidelity/schedule guardrail fails despite lower exposed time"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partC_prediction", "partC_checkpoint", "compression_ratio", "overlap_pct", "exposed_ms", "quality_proxy", "optimization_risk"]
    downstream_use: "Part D validates the full communication plan against simultaneous guardrails."
```

### Part D: Concept Module - Communication Plan Must Satisfy Exposed Step-Time And Topology Guardrails

```yaml
concept_module:
  part_label: "Part D"
  concept_name: "Communication plan must satisfy exposed step-time and topology guardrails"
  chapter_claim: "A collective design is only valid inside an operating envelope; wrong topology assumptions, excessive exposed time, or unsafe optimization risk invalidate the plan."
  reading_connection:
    chapter_section: "#sec-communication-collective-operations-collective-operations-fallacies-pitfalls-9cd0"
    claim_or_formula: "valid = exposed_ms <= budget_ms and topology_guardrail == true and optimization_guardrail == true"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "same selected-track communication owner"
    system_decision: "approve, revise, or reject the collective communication plan"
  student_prior:
    expected_belief: "The fastest modeled option should be approved."
    productive_failure: "The fastest option can depend on a false topology assumption or unsafe compression/overlap claim."
  storyline:
    beat_1_scenario: "The stakeholder must sign a communication design review."
    beat_2_prediction: "Student predicts which guardrail will reject the naive fastest plan."
    beat_3_controls: "Student selects final algorithm family, topology assumption, compression, and overlap."
    beat_4_evidence: "Guardrail matrix compares selected plan against a rejected alternative."
    beat_5_consequence: "Failure callout states which guardrail failed and how to recover."
    beat_6_math_peek: "Math Peek shows simultaneous feasibility inequalities and the exposed-time equation."
    beat_7_checkpoint: "Student chooses approve/revise/reject and names the rejected alternative."
  mechanics:
    controls: ["prediction radio", "algorithm checkpoint", "topology checkpoint", "compression slider", "overlap slider", "final decision radio"]
    graphs: ["candidate comparison chart", "simultaneous guardrail table"]
    failure_state: "any one of exposed time, topology, or optimization risk guardrails fails"
  depth_gate:
    activity_count: 7
    has_prediction: true
    has_manipulation: true
    has_failure_or_boundary: true
    has_math_peek: true
    has_track_specific_consequence: true
  ledger_output:
    fields: ["partD_prediction", "final_decision", "selected_algorithm", "selected_topology", "binding_guardrail", "rejected_alternative", "valid_plan"]
    downstream_use: "Synthesis saves a communication design review and V2-07 reliability implication."
```

### Synthesis: Collective Communication Design Review

The synthesis report must include:

1. Selected algorithm/topology/optimization.
2. Binding alpha, beta, topology, exposed-time, or optimization-risk term.
3. Rejected alternative and why it failed.
4. Evidence number from the chart/table.
5. Track-specific validation test.
6. V2-07 reliability implication.

Ledger fields:

- `track_id`
- `scenario_id`
- `selected_algorithm`
- `selected_topology`
- `selected_optimization`
- `binding_term`
- `exposed_ms`
- `budget_ms`
- `topology_guardrail`
- `optimization_guardrail`
- `rejected_alternative`
- `v2_07_reliability_implication`
- `completed`

## Track Narratives

| Track | Persona | Same concept sequence realized as | Constraint emphasis | Natural failure | Evidence emphasis | Report framing |
|---|---|---|---|---|---|---|
| iPhone | Mobile federated learning engineer | Cohort update aggregation with secure aggregation and uplink analogy | Battery/radio time, privacy protocol overhead, small payload latency | Secure aggregation or battery overhead dominates byte reduction | Completion time plus privacy/battery risk | "Which cohort aggregation plan reduces upload exposure without violating privacy or battery guardrails?" |
| Oura Ring | Wearable systems engineer | Intermittent ring-phone-cloud summary aggregation | Tiny payloads, sync window, wakeups, reliability | Intermittent connectivity dominates modeled communication time | Sync-window pass/fail and tiny-payload alpha cost | "Which sync aggregation policy fits the phone-nearby window and preserves battery?" |
| RoboTaxi | Autonomous fleet data platform lead | Vehicle/depot/cloud event and update aggregation | Event fidelity, safety review completeness, depot hierarchy | Compression removes rare-event detail or depot upload misses the window | Fidelity proxy plus fleet update latency | "Which depot hierarchy and compression policy keeps safety evidence usable?" |
| Cloud Fleet | Distributed training performance lead | GPU AllReduce across NVLink nodes and InfiniBand | Exposed step time, topology mapping, convergence risk, throughput | Wrong NCCL/rank topology or unsafe compression erases training gains | Alpha/beta decomposition, frontier, exposed-time guardrail | "Which collective plan should the training team approve before the reliability review?" |

Track deltas required in implementation:

- Persona and scenario copy change by track.
- Participants, payload defaults, local group size, fabric, step-time budget,
  topology guardrail, optimization-risk threshold, and validation tests change
  by track.
- Failure wording changes by track.
- Evidence emphasis changes by track: privacy/battery for iPhone, sync window
  for Oura Ring, rare-event fidelity for RoboTaxi, throughput/convergence for
  Cloud Fleet.
- Final report framing changes by track while preserving the same concept
  sequence.

## Mechanics, Evidence, And Ledger Plan

| Module | Mechanics | Evidence produced | Ledger/checkpoint output |
|---|---|---|---|
| Opening | Track selector, track context, invariant card, reading map | Selected track, scenario, defaults, source trace | `track_id`, `scenario_id` |
| Part A | Prediction radio, participants slider, payload slider, fabric dropdown, alpha/beta stacked chart, exact table, Math Peek | Ring/tree time, alpha term, beta term, binding term, budget status | `partA_prediction`, `partA_checkpoint`, `ring_ms`, `tree_ms`, `binding_term` |
| Part B | Prediction radio, local group slider, fabric/topology controls, message-size frontier, topology table, Math Peek | Flat/tree/hierarchy times, feasible flags, topology mismatch reason | `partB_prediction`, `partB_checkpoint`, `dominant_candidate`, `topology_guardrail` |
| Part C | Prediction radio, compression slider, overlap slider, exposed-time chart, optimization guardrail table, Math Peek | Raw time, compressed time, exposed time, quality/fidelity/schedule status | `partC_prediction`, `partC_checkpoint`, `compression_ratio`, `overlap_pct`, `optimization_risk` |
| Part D | Prediction radio, final algorithm/topology/optimization controls, candidate comparison chart, simultaneous guardrail table, final decision radio | Selected vs rejected plan, binding guardrail, valid plan flag | `partD_prediction`, `final_decision`, `binding_guardrail`, `rejected_alternative`, `valid_plan` |
| Synthesis | Report panel, memo note, Design Ledger save | Communication design review with V2-07 reliability implication | `completed`, selected plan, binding term, rejected alternative, reliability implication |

Accessibility and fallback requirements:

- Each decision chart has an adjacent exact-value HTML table.
- Feasibility is shown with text labels, not color alone.
- Each failure state states value, limit, unit, and mitigation.
- Required controls are visible in the active module.
- The exported report contains the same evidence as the visuals.

## Implementation Notes

- Owned files only: `labs/vol2/lab_06_collective_communication.py` and this
  track plan.
- No shared helper, test, implementation-note, or registry edits.
- Use existing `mlsysim.physics` collective functions where available:
  `calc_ring_allreduce_time`, `calc_tree_allreduce_time`, and
  `calc_hierarchical_allreduce_time`.
- Keep new support notebook-local with `v2_06_` prefixes.
- Track thresholds not already in MLSysIM or `LabTrackVariant.defaults` are
  notebook-local pedagogical assumptions and must be listed in the report source
  trace.
- Preserve WASM bootstrap, canonical track selector, track context, source trace
  and report export, and Design Ledger save patterns.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 2 | Pass |

Reversible failure states:

- Part A: selected algorithm exceeds the track step-time budget before topology
  mitigation; student can recover by reducing payload, changing participants,
  or choosing the other algorithm family.
- Part B: hierarchy/topology is infeasible or slower because the local/global
  tier assumption is false; student can recover by changing local group size,
  fabric, or topology checkpoint.
- Part C: exposed time improves but optimization guardrail fails; student can
  recover by reducing compression, reducing overlap assumptions, or choosing a
  validation test.
- Part D: any one of exposed time, topology guardrail, or optimization risk
  fails; student can recover by revising algorithm/topology/optimization.

Synthesis gate:

- The design review is complete only when all four predictions, all four
  checkpoints, and the final approval/revision decision are recorded, and the
  report names selected algorithm/topology/optimization, binding term, rejected
  alternative, evidence number, validation test, and V2-07 reliability
  implication.
