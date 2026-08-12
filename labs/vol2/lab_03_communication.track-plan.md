# V2-03 Track Plan: Communication / Network Fabrics

## Chapter Invariant

Network shape governs distributed work. Bandwidth, latency, bisection,
topology, placement, and congestion turn communication into a binding fleet
amount. Link speed is only one input; the useful amount is the communication
the selected topology can deliver before the system deadline.

## Required Reading Map

| Lab module | Chapter anchor | Claim or source model used in the lab |
|---|---|---|
| Opening | `#sec-network-fabrics`, `#sec-network-fabrics-introduction` | The fabric is the synchronization backbone; at scale the network becomes the gradient bus and can dominate useful compute. |
| Part A | `#sec-network-fabrics-performance-model` | Point-to-point communication follows `T(n)=alpha+n/beta`; `n*=alpha*beta` separates latency-bound and bandwidth-bound regimes. |
| Part B | `#sec-network-fabrics-topology`, `#sec-network-fabrics-fat-tree`, `#sec-network-fabrics-rail-optimized` | Bisection bandwidth, oversubscription, and hop count decide which parallel work remains feasible. |
| Part C | `#sec-network-fabrics-behavior`, `#sec-network-fabrics-pfc`, `#sec-network-fabrics-congestion-control` | BSP turns congestion and placement into fleet-wide tail-latency bottlenecks. |
| Part D | `#sec-network-fabrics-monitoring`, `#sec-network-fabrics-summary`, `#sec-network-fabrics-fallacies` | A communication plan is valid only if step-time/SLO, utilization, and topology guardrails are all inside the operating envelope. |
| Synthesis | `#sec-network-fabrics-summary` and V2-06 forward references | The V2-03 output is a network communication memo that carries forward an implication for collective-communication choices. |

Matching concept YAML anchors:

- Primary concepts: Network Fabrics, Topology, Bisection Bandwidth,
  Lossless Fabric, Congestion Control.
- Secondary concepts: Bandwidth Hierarchy, Alpha-Beta Model,
  Communication-Computation Overlap, Fat-Tree, Rail-Optimized Topology,
  Dragonfly Topology, Oversubscription, Bulk Synchronous Parallel,
  Tail Latency, Priority Flow Control, Adaptive Routing, Incast.
- Methodologies: Alpha-Beta Communication Modeling, Bisection Bandwidth
  Analysis, Topology-Aware Placement, Non-Blocking Fabric Design,
  Rail-Optimized Fabric Design, Oversubscription Analysis, Proactive
  Congestion Control, Link-Level Telemetry Monitoring.

## Concept Inventory

### Accepted Concepts

| Concept | Why accepted | Lab role |
|---|---|---|
| Alpha/beta latency-bandwidth model | Converts "network is slow" into separable startup and per-byte terms. | Part A calibration concept. |
| Bisection bandwidth and topology | Explains why advertised link speed does not equal useful global fabric speed. | Part B mechanism concept. |
| Congestion and topology-aware placement | Makes local scheduling and routing choices produce fleet-wide stragglers. | Part C transfer concept. |
| Step-time/SLO, utilization, and topology guardrails | Forces a design decision across simultaneous constraints. | Part D design concept. |
| Carry-forward collective implication | Keeps V2-03 focused on fabric physics while preparing V2-06 collective algorithm reasoning. | Synthesis. |

### Rejected Or Deferred Concepts

| Concept | Reason rejected for this lab | Destination |
|---|---|---|
| Detailed ring/tree collective algorithms | V2-06 owns collective algorithm details; V2-03 should only carry the implication. | V2-06 Collective Communication. |
| Full RDMA/Verbs API mechanics | Too implementation-specific for the concept chain; only the effect on alpha/losslessness is used. | Reading/source trace only. |
| PAM4/FEC physical-layer derivation | Important chapter material but too low-level for the A-D decision chain. | Math/source notes and reading map. |
| SR-IOV and network virtualization | Valuable but would add a second sharing/isolation lab concept. | Synthesis risk note or future ops lab. |
| Full monitoring workflow | Monitoring supports validation but is not the main concept module. | Part D validation test and report evidence. |

## Shared Concept-Module Sequence

The lab has one conceptual sequence across tracks. The selected track changes
persona, constraints, thresholds, evidence emphasis, failure mode, and report
framing. It does not create a different set of concepts.

### Part A: Concept Module - Alpha/Beta Terms Predict Communication Cost

```yaml
concept_module:
  part_label: "Part A"
  concept_name: "Alpha/beta latency-bandwidth terms predict communication cost"
  chapter_claim: "The alpha/beta model separates startup latency from per-byte transfer time."
  reading_connection:
    chapter_section: "#sec-network-fabrics-performance-model"
    claim_or_formula: "T(n)=alpha+n/beta and n*=alpha*beta"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "track-specific engineer"
    system_decision: "decide whether the next communication pressure is latency-bound or bandwidth-bound"
  student_prior:
    expected_belief: "More bandwidth is always the fix."
    productive_failure: "For small payloads, increasing beta barely changes transfer time because alpha dominates."
  storyline:
    beat_1_scenario: "A stakeholder must move a payload through the track's communication path."
    beat_2_prediction: "Student predicts whether alpha, beta, or topology is the binding amount."
    beat_3_controls: "Student changes payload size and active link/path."
    beat_4_evidence: "Stacked alpha-vs-beta chart plus exact table show predicted vs actual binding term."
    beat_5_consequence: "Failure callout states the budget miss and the mitigation lever."
    beat_6_math_peek: "Source model shows T(n)=alpha+n/beta and crossover n*=alpha*beta."
    beat_7_checkpoint: "Student chooses whether to reduce hops, increase bandwidth, or reduce payload."
  mechanics:
    controls: ["prediction radio", "payload slider", "link/path dropdown", "checkpoint radio"]
    graphs: ["stacked alpha/beta bar", "exact evidence table"]
    failure_state: "transfer time exceeds the track communication budget"
  ledger_output:
    fields: ["partA_prediction", "active_link", "payload_mb", "binding_term", "transfer_ms", "alpha_beta_checkpoint"]
    downstream_use: "Part D reuses the binding term when judging final communication policy."
```

### Part B: Concept Module - Topology And Bisection Change Feasible Parallel Work

```yaml
concept_module:
  part_label: "Part B"
  concept_name: "Topology and bisection bandwidth change feasible parallel work"
  chapter_claim: "A global communication pattern is limited by the narrowest bisection cut, not by edge-link speed alone."
  reading_connection:
    chapter_section: "#sec-network-fabrics-topology"
    claim_or_formula: "BW_bisect=(N/2)*beta/oversubscription"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "same selected-track engineer"
    system_decision: "choose the topology shape that keeps parallel work feasible"
  student_prior:
    expected_belief: "If each endpoint has a fast link, the whole fleet can communicate in parallel."
    productive_failure: "Oversubscription makes the global step fail even though every endpoint link still looks fast."
  storyline:
    beat_1_scenario: "The stakeholder scales from one path to many devices/nodes."
    beat_2_prediction: "Student predicts which topology preserves feasible parallel work."
    beat_3_controls: "Student changes endpoint count and topology."
    beat_4_evidence: "Bisection/time chart and topology table compare non-blocking, oversubscribed, aligned, and grouped fabrics."
    beat_5_consequence: "Failure callout names the bisection bottleneck and idle-work consequence."
    beat_6_math_peek: "Source model shows bisection bandwidth and oversubscription slowdown."
    beat_7_checkpoint: "Student selects the topology assumption to carry forward."
  mechanics:
    controls: ["prediction radio", "endpoint slider", "topology dropdown", "checkpoint radio"]
    graphs: ["topology comparison bar", "exact table with bisection, sync time, feasible flag"]
    failure_state: "sync time exceeds the track step/SLO budget"
  ledger_output:
    fields: ["partB_prediction", "participants", "topology", "bisection_gbps", "sync_ms", "topology_checkpoint"]
    downstream_use: "Part C uses selected topology when placement and congestion are evaluated."
```

### Part C: Concept Module - Congestion And Placement Make Local Choices Fleet-Wide Bottlenecks

```yaml
concept_module:
  part_label: "Part C"
  concept_name: "Congestion and placement make local communication choices fleet-wide bottlenecks"
  chapter_claim: "Under BSP, the slowest congested path paces the whole fleet."
  reading_connection:
    chapter_section: "#sec-network-fabrics-behavior"
    claim_or_formula: "rho=offered_load/capacity; tail multiplier rises as rho approaches 1"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "same selected-track engineer"
    system_decision: "choose placement that avoids turning local convenience into global congestion"
  student_prior:
    expected_belief: "Placement is operational bookkeeping and adaptive routing will handle the network."
    productive_failure: "A spread/noisy placement can violate utilization even when the topology looked feasible in Part B."
  storyline:
    beat_1_scenario: "A scheduler or product policy places communication across the fabric."
    beat_2_prediction: "Student predicts whether locality, background traffic, or link speed becomes the bottleneck."
    beat_3_controls: "Student changes placement policy and burst/background pressure."
    beat_4_evidence: "Utilization and tail-time chart/table show congestion amplification."
    beat_5_consequence: "Failure callout names queue growth, PFC/tail risk, or connectivity miss."
    beat_6_math_peek: "Source model connects utilization to tail multiplier and BSP straggler cost."
    beat_7_checkpoint: "Student chooses placement mitigation for the final memo."
  mechanics:
    controls: ["prediction radio", "placement dropdown", "burst pressure slider", "checkpoint radio"]
    graphs: ["utilization/tail bar", "placement evidence table"]
    failure_state: "utilization exceeds track guardrail or topology placement guardrail is false"
  ledger_output:
    fields: ["partC_prediction", "placement_policy", "burst_multiplier", "utilization", "tail_ms", "placement_checkpoint"]
    downstream_use: "Part D must pass both utilization and topology guardrails."
```

### Part D: Concept Module - Communication Plan Must Satisfy Step-Time/SLO, Utilization, And Topology Guardrails

```yaml
concept_module:
  part_label: "Part D"
  concept_name: "A communication plan must satisfy step-time/SLO, utilization, and topology guardrails"
  chapter_claim: "Network fabric decisions are valid only inside an operating envelope."
  reading_connection:
    chapter_section: "#sec-network-fabrics-summary"
    claim_or_formula: "valid plan = exposed_time <= SLO and utilization <= limit and topology_guardrail == true"
  track_lens:
    primary_track: "selected canonical track"
    stakeholder: "same selected-track engineer"
    system_decision: "approve or revise a topology/placement/payload plan"
  student_prior:
    expected_belief: "One winning metric is enough to approve the network plan."
    productive_failure: "A plan can pass step time but fail utilization or topology guardrails."
  storyline:
    beat_1_scenario: "The stakeholder must sign a communication plan for the selected track."
    beat_2_prediction: "Student predicts which guardrail will reject the naive plan."
    beat_3_controls: "Student changes topology, placement, payload reduction, and overlap."
    beat_4_evidence: "Guardrail matrix shows step-time/SLO, utilization, and topology status."
    beat_5_consequence: "Failure callout names the rejected alternative and why it fails."
    beat_6_math_peek: "Source model shows exposed time and simultaneous feasibility inequalities."
    beat_7_checkpoint: "Student chooses the final approval/revision decision."
  mechanics:
    controls: ["prediction radio", "topology dropdown", "placement dropdown", "payload reduction slider", "overlap slider", "final decision radio"]
    graphs: ["guardrail table", "candidate comparison bar"]
    failure_state: "any one of the three guardrails fails"
  ledger_output:
    fields: ["partD_prediction", "final_topology", "final_placement", "payload_reduction_pct", "overlap_pct", "step_ok", "utilization_ok", "topology_ok", "final_decision"]
    downstream_use: "Synthesis saves the memo and V2-06 collective implication."
```

### Synthesis: Network Communication Memo

The synthesis memo must include:

1. Selected topology/placement policy.
2. Binding network amount.
3. Rejected alternative.
4. Evidence number.
5. Carry-forward collective-communication implication for V2-06.

Ledger fields:

- `track_id`
- `scenario_id`
- `selected_topology`
- `selected_placement`
- `active_link`
- `payload_mb`
- `binding_network_amount`
- `step_time_ms`
- `utilization`
- `rejected_alternative`
- `collective_communication_implication`
- `completed`

## Track Narratives

| Track | Persona | Same concept sequence realized as | Constraint emphasis | Natural failure | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile product engineer | Device-to-edge/cloud telemetry, offload, update delivery | Interactive latency, privacy-safe payload, battery/radio time | Responsiveness miss or privacy-hostile cloud routing | "What stays local, what can cross the network, and what payload must be staged?" |
| Oura Ring | Wearable firmware engineer | Intermittent BLE ring-phone-cloud sync and OTA movement | Connection window, tiny radio budget, payload size | Sync window/OTA budget miss | "Which sync/update policy survives intermittent connectivity?" |
| RoboTaxi | Autonomous vehicle platform engineer | Vehicle-local sensor fabric plus triaged fleet upload | P99/P999 latency, safety margin, upload triage | Safety-latency miss or event-upload backlog | "Which communication stays vehicle-local and which evidence is delayed?" |
| Cloud Fleet | Fleet service owner | NVLink/InfiniBand/Ethernet hierarchy and rack/pod placement | Step time, utilization, bisection, oversubscription | SLO breach, queue growth, idle accelerators | "Which topology and placement policy makes distributed work feasible?" |

Track deltas required in implementation:

- Persona and scenario copy change by track.
- Payload defaults, SLO/step budget, utilization limit, and link choices change
  by track.
- Failure wording changes by track.
- Evidence emphasis changes by track: edge latency for iPhone/RoboTaxi,
  intermittent sync for Oura Ring, bisection/utilization for Cloud Fleet.
- Final report framing changes by track.

## Mechanics, Evidence, And Ledger Plan

| Module | Mechanics | Evidence produced | Ledger/checkpoint output |
|---|---|---|---|
| Opening | Track selector, track mission, reading map, invariant card | Selected track and scenario | `track_id`, `scenario_id` |
| Part A | Prediction radio, payload slider, link/path dropdown, alpha/beta chart, exact table, Math Peek | Actual binding term, transfer time, crossover size | `partA_*` prediction, payload, link, binding term, checkpoint |
| Part B | Prediction radio, endpoint slider, topology dropdown, bisection comparison chart/table | Bisection bandwidth, topology sync time, feasible flag | `partB_*` prediction, participant count, topology, sync result |
| Part C | Prediction radio, placement dropdown, burst slider, utilization/tail chart/table | Offered load, utilization, tail multiplier, failure state | `partC_*` prediction, placement, burst, utilization, tail result |
| Part D | Prediction radio, final topology/placement controls, payload reduction, overlap, guardrail table | SLO, utilization, topology guardrail pass/fail | `partD_*` prediction, final policy, guardrail matrix |
| Synthesis | Memo builder, report export panel, Design Ledger save | Network communication memo and carry-forward implication | `completed`, selected policy, binding amount, rejected alternative |

Accessibility and fallback requirements:

- Each chart has an adjacent exact-value HTML table.
- Feasibility is shown with text labels, not color alone.
- Each failure state states value, limit, unit, and mitigation.
- Required controls are visible in the active module; no required control is
  hidden in an advanced drawer.
- The downloaded report contains the same evidence as the visuals.

## Implementation Notes

- Owned files only: `labs/vol2/lab_03_communication.py` and this track plan.
- No shared helper, test, implementation-note, or registry edits.
- Use existing `mlsysim.physics` functions where available:
  `calc_point_to_point_time`, `calc_alpha_beta_crossover`,
  `calc_bisection_bandwidth`, and `calc_oversubscription_effect`.
- Keep new communication support notebook-local with `v2_03_` prefixes.
- Scenario thresholds that are not existing MLSysIM facts are local pedagogical
  assumptions and must be listed in the notebook source trace.
- Preserve WASM bootstrap, canonical track selector, source trace/report export,
  and Design Ledger save patterns.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 2 | Pass |

Reversible failure states:

- Part A: transfer time exceeds track communication budget; student can recover
  by reducing payload or choosing a lower-alpha/higher-beta path.
- Part B: topology sync time exceeds step/SLO budget; student can recover by
  changing topology or participant count.
- Part C: utilization exceeds the guardrail; student can recover by choosing a
  topology-aware placement or lowering burst pressure.
- Part D: any one guardrail fails; student can recover by changing topology,
  placement, payload reduction, or overlap.

Synthesis gate:

- The memo is complete only when all four predictions are made and the final
  plan names selected topology/placement, binding amount, rejected alternative,
  evidence number, and collective-communication implication.
