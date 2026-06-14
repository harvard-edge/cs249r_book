# V2-01 Track Plan: Scale Changes the Unit

## Chapter Invariant

Scale changes the unit of analysis from a single machine to a fleet. Once the
system spans many devices, coordination, communication, and capacity become
first-order amounts rather than background details.

This invariant is shared by every track. The selected track does not create a
different concept path; it changes the stakeholder lens, operating envelope,
failure threshold, evidence priority, and memo framing for the same sequence of
concept modules.

## Reading Map

| Lab module | Chapter anchors | Claim carried into the lab |
|---|---|---|
| Part A - Fleet is the unit | Volume II introduction: Scale Moment; Machine Learning Fleet definition; Fleet Stack overview | The system is no longer one accelerator or one model instance. Capacity, failure opportunity, and coordination surface all scale with fleet size. |
| Part B - Single-node limits force distribution | Scale Moment; single-node to distributed fleet transition; Fleet Stack infrastructure/distribution layers | Single-node memory, throughput, energy, or latency ceilings force a distributed capacity choice before model quality can matter. |
| Part C - C3 trade-off at scale | C3 taxonomy; fleet law; communication intensity; appendix C3 diagnostic summary | Step time is a budget split across compute, communication, coordination, and overlap. Optimizing the wrong term wastes money. |
| Part D - Rare events become routine | Reliability gap; failure becomes routine; appendix C3 coordination case | Per-component reliability does not compose linearly. Fleet size converts rare failures into routine operational load. |
| Synthesis - Fleet scale memo | Summary; Fleet Stack; next chapter connection to compute infrastructure | A defensible scale decision names the operating envelope, binding amount, mitigation, and carry-forward infrastructure question. |

## Concept Inventory

### Accepted Concepts

1. Machine Learning Fleet as the unit of analysis.
   - Reason: it is the first durable conceptual shift of Volume II.
   - Consequence: students must stop reporting a single-device metric as if it
     described the system.

2. Single-node resource ceilings force distributed capacity choices.
   - Reason: the introduction frames scale as crossing memory, network, and
     energy walls.
   - Consequence: students compute the first point where "run it on one box"
     is infeasible and choose sharding, replication, specialization, or refusal.

3. C3/fleet law decomposition.
   - Reason: Compute, Communication, and Coordination are the diagnostic lens
     used throughout Volume II.
   - Consequence: students identify whether the next dollar should buy compute,
     interconnect/overlap, or recovery/orchestration work.

4. Reliability gap and routine failure.
   - Reason: the introduction uses 10K-25K GPU examples to show that failure is
     steady state.
   - Consequence: students calculate failure cadence and reject manual recovery
     when the cadence is inside an operating shift.

5. Fleet-stack memo discipline.
   - Reason: the chapter ends by connecting requirements to compute
     infrastructure.
   - Consequence: students leave with a selected operating envelope and a
     question for the compute-infrastructure chapter.

### Rejected Concepts

| Candidate | Rejection reason | Where it may appear |
|---|---|---|
| Detailed AI scaling laws and Chinchilla allocation | Valuable, but it would create a separate model-resource planning lab and distract from the fleet unit shift. | Synthesis note only. |
| CAP theorem and edge consistency trade-offs | Important but too deep for the opening lab; it belongs after students can already name C3 and failure cadence. | Optional track consequence text. |
| Governance as control plane | Central to Volume II, but not the strongest hands-on amount system for V2-01. | Memo risk language. |
| Framework Rosetta Stone | Taxonomy/reference content without enough immediate consequence. | Not implemented. |
| Full energy-scale invariant | Useful later for sustainable AI and infrastructure; V2-01 only names energy as a track-specific capacity amount. | Track thresholds and carry-forward question. |

## Shared Concept Sequence

| UI part | Concept module | Productive prior | Student consequence |
|---|---|---|---|
| Part A | Fleet is the unit, not one accelerator or one model instance. | "If one unit works, many units are just more capacity." | Capacity rises linearly, but coordination surface and failure opportunity rise too; the report must name the fleet unit. |
| Part B | Single-node limits force distributed capacity choices. | "Use the best single machine until optimization is done." | The workload crosses memory/throughput/energy limits and needs sharding, replication, specialization, or scope reduction. |
| Part C | C3 trade-offs determine useful fleet work. | "More devices or faster devices are the primary scale lever." | The fleet law shows compute can shrink while communication or coordination dominates the step. |
| Part D | Emergent bottlenecks and failure rates make rare events routine. | "A low component failure rate is enough." | System MTBF collapses with N; manual recovery becomes nonviable. |
| Synthesis | Fleet scale memo with selected operating envelope and carry-forward question. | "The lab is complete when charts are inspected." | A decision is saved with evidence, a binding amount, mitigation, and next infrastructure question. |

## Concept Modules

### Part A - Concept Module: Fleet Is the Unit

- Chapter claim: the Machine Learning Fleet is a distributed system that must
  operate as one coherent computer.
- Reading connection: Scale Moment; Machine Learning Fleet definition; Scale
  Discontinuity figure.
- Track lens:
  - iPhone: mobile release lead reasoning over installed devices, telemetry
    coverage, privacy-safe rollout cohorts, and user-visible regression risk.
  - Oura Ring: wearable firmware lead reasoning over devices with intermittent
    sync, battery variance, and firmware cohort risk.
  - RoboTaxi: safety platform lead reasoning over vehicle-hours, geography
    exposure, and incident review capacity.
  - Cloud Fleet: platform/SRE lead reasoning over accelerator count, jobs,
    requests, coordination surface, and partial failure.
- Expected prior: one healthy unit implies a healthy scaled system.
- Activities:
  1. Scenario: stakeholder asks whether the system can be described by a
     single-unit metric.
  2. Prediction: student chooses which amount grows fastest with scale:
     capacity, coordination surface, or failure opportunity.
  3. Manipulation: student sweeps fleet size for the selected track.
  4. Evidence: chart compares normalized capacity, coordination surface, and
     failure opportunity; table reports exact values.
  5. Consequence: failure/coordination boundary appears when fleet health or
     coordination burden crosses the track threshold.
  6. Math Peek/source model: capacity = N times unit capacity; coordination
     surface roughly N log2 N; healthy-fleet probability = p_unit^N.
  7. Checkpoint: student chooses the first fleet-level metric to put in the
     memo.
- Ledger fields: track_id, scale_unit, fleet_size, first_order_amount,
  fleet_health_pct, coordination_index.

### Part B - Concept Module: Single-Node Limits Force Distributed Capacity

- Chapter claim: the boundary left open by Volume I appears when the local
  machine is no longer the system.
- Reading connection: Scale Moment; single-node versus distributed fleet stack;
  network wall and capacity walls.
- Track lens:
  - iPhone: one device cannot represent the installed base; local model size,
    thermal window, and rollout cohorting force tiering.
  - Oura Ring: SRAM/flash/duty-cycle ceilings force model simplification and
    staged firmware cohorts.
  - RoboTaxi: one vehicle computer may meet nominal latency, but fleet demand
    and redundancy force regional capacity planning.
  - Cloud Fleet: one accelerator cannot hold or serve the model; HBM and
    throughput ceilings force model parallelism, data parallelism, or serving
    replication.
- Expected prior: the best single unit should be optimized before distribution.
- Activities:
  1. Scenario: stakeholder asks whether one unit can carry the workload.
  2. Prediction: student predicts the first single-node limit: memory/capacity,
     throughput/latency, or operational headroom.
  3. Manipulation: student changes model/workload amount and state multiplier.
  4. Evidence: memory/capacity chart shows required amount versus single-unit
     budget and minimum distributed units.
  5. Consequence: failure callout names the violated amount and the smallest
     feasible distributed choice.
  6. Math Peek/source model: required units =
     max(ceil(state / unit_budget), ceil(demand / unit_capacity)).
  7. Checkpoint: student chooses shard, replicate, specialize/tier, or refuse.
- Ledger fields: model_amount, state_multiplier, single_unit_budget,
  required_units, capacity_choice, binding_limit.

### Part C - Concept Module: C3 Trade-Off at Scale

- Chapter claim: the fleet law decomposes every distributed step into compute,
  communication, coordination, and overlap.
- Reading connection: C3 taxonomy; fleet law; C3 traffic light and diagnostic
  table in Appendix C3.
- Track lens:
  - iPhone: compute is local, communication is privacy-safe telemetry or
    federated update traffic, coordination is rollout and cohort control.
  - Oura Ring: compute is local sensing inference, communication is intermittent
    sync/OTA payload, coordination is firmware staging and sensor drift control.
  - RoboTaxi: compute is perception/planning, communication is map/model update
    flow, coordination is safety gating and incident review.
  - Cloud Fleet: compute is accelerator math, communication is gradient or KV
    movement, coordination is barriers, checkpoints, scheduler churn, and
    recovery.
- Expected prior: adding more devices or buying faster devices is the main
  scale response.
- Activities:
  1. Scenario: stakeholder asks where to spend the next infrastructure budget.
  2. Prediction: student chooses the likely C3 bottleneck before seeing the
     decomposition.
  3. Manipulation: student changes fleet width, communication reduction, and
     visible coordination overhead.
  4. Evidence: stacked bar/table decomposes step time into compute,
     communication, coordination, and hidden overlap.
  5. Consequence: red state if communication share exceeds 40 percent or
     goodput falls below 75 percent.
  6. Math Peek/source model: T_step(N) = T_compute/N + T_comm(N) +
     T_sync(N) - T_overlap; compute fraction flags idle silicon.
  7. Checkpoint: student chooses compute, communication, or coordination
     mitigation and rejects the wrong lever.
- Ledger fields: fleet_width, communication_reduction, coordination_pct,
  dominant_c3_axis, compute_fraction_pct, mitigation_choice.

### Part D - Concept Module: Routine Failure

- Chapter claim: hardware failure becomes a routine event at fleet scale.
- Reading connection: reliability gap; failure becomes routine; appendix C3
  coordination tax case.
- Track lens:
  - iPhone: rare per-device regression becomes many daily support events across
    a large install base.
  - Oura Ring: rare firmware or battery/sensor issue becomes a nightly cohort
    operations problem.
  - RoboTaxi: rare vehicle incident becomes routine over fleet-hours and must
    be handled by safety operations.
  - Cloud Fleet: rare GPU/node failure becomes frequent cluster interruption
    and checkpoint/restart cost.
- Expected prior: a high per-unit MTBF or reliability percentage is sufficient.
- Activities:
  1. Scenario: stakeholder asks if manual recovery is acceptable.
  2. Prediction: student estimates whether interruptions remain rare or become
     routine.
  3. Manipulation: student changes fleet size, per-unit MTBF, and recovery
     time.
  4. Evidence: reliability curve and exact table report system MTBF, failures
     per day, lost time, and goodput.
  5. Consequence: danger callout when failures/day exceeds the track routine
     threshold or goodput falls below the operating envelope.
  6. Math Peek/source model: MTBF_system = MTBF_component / N and
     P(failure before t) = 1 - exp(-t / MTBF_system).
  7. Checkpoint: student chooses manual restart, checkpointing, elastic
     recovery, or smaller failure domains.
- Ledger fields: component_mtbf_hours, system_mtbf_hours, failures_per_day,
  recovery_minutes, goodput_pct, recovery_policy.

### Synthesis - Fleet Scale Memo

- Chapter claim: the scale mindset is a decision discipline, not a vocabulary
  list.
- Activities:
  1. Student selects an operating envelope: conservative, balanced, or
     aggressive.
  2. Notebook composes a memo with evidence from Parts A-D.
  3. Memo names the fleet unit, binding amount, selected capacity plan, C3
     bottleneck, routine-failure mitigation, and carry-forward question.
  4. Design Ledger saves the memo quantities for Lab V2-02 Compute
     Infrastructure.
- Ledger fields: operating_envelope, selected_capacity_plan, binding_amount,
  c3_axis, failure_policy, carry_forward_question.

## Track Narrative Packet

| Track | Persona | Same concept realized as... | Primary threshold | Evidence emphasis | Natural failure | Report framing |
|---|---|---|---|---|---|---|
| iPhone | Mobile product/release lead | A fleet of heterogeneous phones and OS versions | rollout health, memory/thermal headroom, privacy-safe telemetry coverage | install-base amount, support-event cadence, tiering decision | hot devices, regression blast radius, privacy-limited observability | mobile rollout envelope memo |
| Oura Ring | Wearable firmware lead | A wearable fleet with tiny resource envelopes and intermittent sync | SRAM/flash/duty-cycle fit and nightly sync health | firmware cohort size, battery and sync budget, OTA payload | flash overflow, battery miss, sensor drift cohort | wearable firmware scale memo |
| RoboTaxi | Safety/perception platform lead | A vehicle fleet accumulating safety exposure and map/model coordination | p99/p999 latency, incident cadence, redundancy | vehicle-hours, safety margin, regional rollout capacity | tail-latency miss, incident review overload | safety operations scale memo |
| Cloud Fleet | Platform/SRE lead | A training/serving fleet of accelerators and jobs | communication share, goodput, SLO/capacity headroom | accelerator count, step-time decomposition, failure cadence | SLO breach, queue/step-time explosion, cluster restart churn | fleet operating envelope memo |

## Mechanics And Evidence Plan

| Module | Controls | Chart/table | Failure state | Evidence saved |
|---|---|---|---|---|
| Part A | prediction radio; fleet-size slider; metric checkpoint radio | normalized amount chart; amount table | fleet health or coordination boundary | first-order amount and fleet unit |
| Part B | prediction radio; model/workload slider; state multiplier dropdown; capacity choice radio | single-unit budget bar; distributed units table | single-unit budget violation | required units and capacity plan |
| Part C | prediction radio; fleet width slider; communication reduction slider; coordination overhead slider; mitigation radio | C3 stacked bar; C3 table | communication > 40 percent or goodput < 75 percent | dominant C3 axis and mitigation |
| Part D | prediction radio; fleet size slider; component MTBF slider; recovery-time slider; recovery policy radio | failures/day curve; reliability table | failures/day routine or goodput miss | failure cadence and recovery policy |
| Synthesis | envelope radio; carry-forward question radio | memo card and ledger HUD | incomplete or internally inconsistent memo | operating envelope and next question |

Evidence forms include prediction-vs-actual text, chart annotations at
thresholds, exact fallback tables, reversible red failure states, Math
Peek/source model accordions, and a Design Ledger memo.

## Data And Source Policy

- Existing helpers are retained for bootstrap, track selection, track profile
  context, styling, Plotly theme, and Design Ledger.
- New calculations are notebook-local because V2-01 is an introductory
  conceptual simulator rather than a reusable solver.
- Notebook-local helpers are prefixed `v2_01_`.
- Scenario thresholds are pedagogical operating envelopes, not new MLSysIM
  registry facts. The plan labels them as track lenses.
- Chapter source models used directly in the notebook:
  - Capacity = N times per-unit capacity.
  - Coordination surface approximately N log2 N for the introductory
    comparison.
  - Required distributed units =
    max(ceil(state / unit budget), ceil(demand / unit capacity)).
  - Fleet law:
    T_step(N) = T_compute/N + T_comm(N) + T_sync(N) - T_overlap.
  - Reliability:
    MTBF_system = MTBF_component / N and
    P(failure before t) = 1 - exp(-t / MTBF_system).
- Appendix C3 traffic-light thresholds are used as instructional thresholds:
  communication fraction above 40 percent is red; goodput below 75 percent is
  red.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| The existing lab is a thin shared-renderer wrapper. | Replace only this notebook with local concept modules while keeping the bootstrap, selector, context, and ledger pattern. |
| Track-specific values can look like different concepts. | Keep the same calculations and module sequence for every track; only scenario labels, thresholds, default amounts, and report prompts vary. |
| Overloading Part A with reliability and coordination could duplicate Part D. | Part A uses reliability only to prove the fleet is the unit; Part D isolates failure cadence and recovery policy. |
| Browser/WASM fragility from new dependencies. | Use only installed runtime packages already present in the lab bootstrap: marimo, plotly, pandas availability not required, mlsysim, and mlsysbook_labs. |
| Other workers may edit adjacent labs. | Edit only `labs/vol2/lab_01_introduction.py` and this track-plan. Do not commit. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 2 | Pass |

Minimum acceptance checks:
- No module dimension is below 2.
- Every Part A-D has at least five student-facing beats.
- Every Part has a structured prediction, manipulation, evidence, boundary or
  failure consequence, Math Peek/source model, and checkpoint decision.
- At least two reversible failure states are reachable: Part B single-unit
  capacity violation, Part C C3 red state, and Part D routine-failure/goodput
  miss.
- Synthesis ties all modules back to the chapter invariant and saves a
  fleet-scale memo.
