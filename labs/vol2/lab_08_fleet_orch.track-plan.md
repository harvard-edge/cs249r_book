# V2-08 Track Plan: The Scheduling Trap

## Concept-Module Packet

This packet applies the concept-module methodology to Volume II, Chapter 8,
Fleet Orchestration. The lab uses one shared Part A/B/C/D concept sequence for
every track. The selected track changes persona, constraints, thresholds,
evidence emphasis, failure mode, and report framing; it does not introduce
different concepts.

## Chapter Invariant

Schedulers allocate scarce resources. Queueing, priorities, placement/bin
packing, and fairness/utilization interact, so a good orchestration policy is
the conjunction of utilization, fairness, SLO, topology, and starvation
guardrails rather than the maximization of one dashboard metric.

## Reading Map

| Module | Chapter anchors | Claim carried into lab |
|---|---|---|
| Opening | `#sec-fleet-orchestration-introduction`, `#sec-fleet-orchestration-objectives` | Orchestration decides who gets what, when, and where under throughput, fairness, latency, and cost conflicts. |
| Part A | `#sec-fleet-orchestration-objectives`, `#nbk-fleet-orchestration-queuing-theory-gpu-clusters`, fallacy "Treating high utilization as proof of cluster health" | Queue wait rises nonlinearly with utilization and service-time variability. |
| Part B | `#sec-fleet-orchestration-burst-capacity`, `#sec-fleet-orchestration-preemption-cascades`, custom scheduler sections for fair/priority signals | Priority and preemption reduce urgent latency by imposing recovery tax and starvation risk elsewhere. |
| Part C | `#sec-fleet-orchestration-bin-packing`, `#sec-fleet-orchestration-placement-algorithms`, `#sec-fleet-orchestration-topology-aware` | Global free capacity is not enough; the location of free capacity changes feasibility and communication cost. |
| Part D | `#sec-fleet-orchestration-fair-share-multi`, `#sec-fleet-orchestration-resource-accounting`, `#sec-fleet-orchestration-fallacies`, `#sec-fleet-orchestration-summary` | Launchable policy passes several guardrails together and must evolve with infrastructure and workload mix. |
| Synthesis | `#sec-fleet-orchestration-summary`, "From orchestration to optimization" | The orchestration decision becomes a performance-engineering precondition for V2-09. |

## Accepted Concepts

1. Queueing pressure emerges from job mix and arrival rate.
2. Priority/preemption trades latency for starvation risk.
3. Placement/bin packing changes utilization and topology costs.
4. Orchestration policy must satisfy utilization, fairness, SLO, and
   starvation guardrails at the same time.
5. Synthesis converts evidence into an orchestration policy memo with the
   selected scheduler policy, binding resource, rejected alternative, and V2-09
   performance implication.

## Rejected Concepts

| Rejected concept | Reason |
|---|---|
| Slurm versus Kubernetes feature taxonomy | Important chapter material, but a taxonomy alone does not create a short consequence chain. The lab references paradigms through policy framing only. |
| CAP theorem scheduler consistency | Conceptually important, but the requested sequence is queueing, priority, placement, and guardrail policy. CAP belongs in a deeper distributed-control lab. |
| Spot instance economics | Strong cost concept, but would dilute the priority/preemption and placement focus. |
| Elastic training framework contract | Referenced as a mitigation, but not the core module because elasticity belongs with fault tolerance and performance handoff. |
| Security namespace isolation | Important multi-tenancy material, but not part of the requested orchestration policy memo. |

## Track Narrative Matrix

| Track | Persona | Binding constraints | Evidence emphasis | Failure mode | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile product lead | foreground responsiveness, thermal headroom, battery, local memory | milliseconds of UI queue delay and background work share | foreground task misses responsiveness because background ML fills the scheduler | App scheduler memo that protects foreground UX and defers background work. |
| Oura Ring | Wearable firmware lead | duty cycle, battery/day, SRAM/flash, sensing cadence | wake windows, backlog windows, and signal freshness | deferred sensing/sync work misses cadence or burns battery | Duty-cycle scheduler memo that preserves sensing continuity. |
| RoboTaxi | Safety/perception lead | p99/p999 deadline, warm safety lane, power, recall margin | deadline miss rate, safety task latency, and starvation of noncritical tasks | perception/control deadlines are protected by preempting lower priority work too often | Safety scheduler memo with bounded preemption and fallback mode. |
| Cloud Fleet | Platform/SRE lead | SLA, throughput, utilization, cost, tenant fairness | queue depth, GPU slots, fair-share, topology penalty | high utilization creates queue explosion, SLO breach, or tenant starvation | Fleet policy memo for accelerator scheduling across tenants. |

Each track changes at least five fields: persona, thresholds, control defaults,
failure state, evidence priority, and synthesis/report language. The same
concept modules and formulas remain visible across all tracks.

## Concept Modules

### Part A: Concept Module - Queueing Pressure Emerges From Job Mix And Arrival Rate

`chapter_claim`: ML workloads are heavy-tailed; near the utilization knee,
queue wait becomes a large multiple of service time.

`student_prior`: High utilization means the fleet is healthy and efficient.

`storyline`:
1. Scenario: the track stakeholder receives a mixed workload queue and must set
   an admission target before the next burst.
2. Prediction: student chooses which amount will hit first: arrival pressure,
   job mix variability, raw capacity, or no problem.
3. Manipulation: student changes arrival multiplier, heavy-job mix, and service
   variability.
4. Evidence: queueing curve plus exact table show utilization, wait multiplier,
   queue depth, p95/SLO, and failure boundary.
5. Consequence: reversible SLO or responsiveness violation appears when
   utilization and variability cross the track guardrail.
6. Math Peek/source model: Pollaczek-Khinchine style
   `Wq / E[S] = rho/(1-rho) * (1 + Cs^2)/2`, with track-local scenario
   constants documented.
7. Checkpoint: student chooses the admission/headroom rule carried into Part B.

`mechanics`: prediction radio, three sliders, queueing line chart, exact table,
failure callout, Math Peek accordion, checkpoint radio.

`ledger_output`: `part_a_prediction`, `arrival_multiplier`, `heavy_mix_pct`,
`service_variability`, `rho`, `queue_wait_multiplier`, `p95_latency_ms`,
`queue_failure`, `part_a_checkpoint`.

### Part B: Concept Module - Priority/Preemption Trades Latency For Starvation Risk

`chapter_claim`: Priorities and preemption are useful but create recovery tax,
preemption cascades, and starvation risk.

`student_prior`: Higher priority and preemption simply make urgent work faster.

`storyline`:
1. Scenario: the track stakeholder must protect urgent work without making the
   lower-priority backlog permanently unschedulable.
2. Prediction: student identifies whether preemption is free, only affects
   urgent latency, or transfers risk into starvation and recovery.
3. Manipulation: student adjusts preemption aggressiveness, urgent share, and
   checkpoint interval.
4. Evidence: trade-off chart compares urgent latency, batch wait, preemption
   tax, and starvation guardrail.
5. Consequence: failure state shows either missed urgent SLO or starvation tax.
6. Math Peek/source model: preemption tax equals lost work plus reload/warmup,
   and effective priority ages with unmet fair-share.
7. Checkpoint: student chooses the bounded priority rule to carry into placement.

`mechanics`: prediction radio, sliders, trade-off line chart, exact table,
preemption tax callout, Math Peek accordion, checkpoint radio.

`ledger_output`: `part_b_prediction`, `preemption_aggressiveness`,
`urgent_share_pct`, `checkpoint_interval_min`, `urgent_latency_ms`,
`starvation_wait_h`, `preemption_tax_gpu_min`, `part_b_checkpoint`.

### Part C: Concept Module - Placement/Bin Packing Changes Utilization And Topology Costs

`chapter_claim`: Placement is performance. A fleet can have enough free GPUs
globally while no valid contiguous/topology-aware placement exists.

`student_prior`: If total free resources exceed the request, the job can run.

`storyline`:
1. Scenario: a pending gang or high-priority job arrives while the fleet has
   fragmented free slots.
2. Prediction: student predicts whether global free capacity, best-fit packing,
   or topology-aware placement determines the result.
3. Manipulation: student changes job size, placement policy, and topology
   sensitivity.
4. Evidence: node-slot heatmap and exact table show raw free GPUs, contiguous
   fit, locality score, topology penalty, and utilization.
5. Consequence: failure state distinguishes stranded capacity from a feasible
   but slow scattered placement.
6. Math Peek/source model: locality score
   `Cost_locality = sum_{i<j} w(d(g_i,g_j))` plus fragmentation index.
7. Checkpoint: student chooses a placement policy and names the topology cost.

`mechanics`: prediction radio, job-size slider, policy dropdown, topology
sensitivity slider, heatmap, cost/utilization table, Math Peek accordion,
checkpoint radio.

`ledger_output`: `part_c_prediction`, `job_size_gpus`, `placement_policy`,
`fragmentation_index`, `locality_cost`, `topology_penalty_pct`,
`placement_failure`, `part_c_checkpoint`.

### Part D: Concept Module - Policy Must Pass Utilization, Fairness, SLO, And Starvation Guardrails

`chapter_claim`: A scheduler policy is an operating envelope, not a single
score. Utilization, fair-share, p95/p99 SLO, topology, and starvation must pass
together.

`student_prior`: Pick the policy with the best single headline metric.

`storyline`:
1. Scenario: the stakeholder must approve a scheduler policy for the selected
   track.
2. Prediction: student predicts whether utilization, SLO, fairness, or all
   guardrails will decide launchability.
3. Manipulation: student chooses policy, utilization target, fairness floor, and
   starvation guardrail.
4. Evidence: candidate gate table and normalized guardrail chart compare every
   policy under the same track thresholds.
5. Consequence: launch gate passes only if all guardrails pass; otherwise the
   binding guardrail is named.
6. Math Peek/source model: `feasible = utilization_ok and fairness_ok and
   slo_ok and starvation_ok and topology_ok`.
7. Checkpoint: student records selected policy, rejected alternative, binding
   guardrail, and V2-09 performance implication.

`mechanics`: structured policy controls, candidate table, normalized guardrail
bar chart, failure callout, Math Peek accordion, synthesis memo.

`ledger_output`: `selected_scheduler_policy`, `rejected_policy`,
`binding_guardrail`, `utilization_target_pct`, `fairness_floor_pct`,
`starvation_guard_h`, `policy_feasible`, `v2_09_implication`.

## Synthesis Module

Students submit an orchestration policy memo:

1. Selected scheduler policy.
2. Binding resource or guardrail.
3. Rejected alternative and why.
4. Evidence numbers from Parts A-D.
5. V2-09 performance implication: the next lab can optimize kernels and
   execution only if orchestration hands it stable, topology-valid, SLO-safe
   capacity.

The Design Ledger saves the track, scenario, selected policy, rejected
alternative, binding guardrail, queueing evidence, preemption evidence,
placement evidence, and V2-09 implication.

## Mechanics And Evidence Plan

| Module | Controls | Evidence | Failure/boundary | Ledger use |
|---|---|---|---|---|
| Opening | track selector | reading map and track mission | none | selected track |
| Part A | prediction radio, arrival slider, mix slider, variability slider | queueing curve, exact table, metric cards | SLO/responsiveness breach or utilization `rho >= 1` | headroom rule and queue evidence |
| Part B | prediction radio, preemption slider, urgent share slider, checkpoint interval slider | priority trade-off curve, exact table | urgent SLO miss or starvation wait above guardrail | bounded preemption rule |
| Part C | prediction radio, job size slider, placement policy dropdown, topology sensitivity slider | placement heatmap, locality/fragmentation table | stranded free capacity or topology penalty | placement policy and topology penalty |
| Part D | prediction radio, selected policy dropdown, rejected policy dropdown, utilization target, fairness floor, starvation guard | policy gate chart and candidate table | guardrail conjunction fails | final policy memo |
| Synthesis | student id, final memo controls | report export and ledger HUD | incomplete report lock until predictions made | future V2-09 handoff |

All decision-driving plots include exact tables. Color is not the only signal:
tables and callouts state value, limit, unit, and mitigation.

## Implementation Risks

| Risk | Handling |
|---|---|
| No shared MLSysIM scheduler solver for queueing, priority, and placement. | Keep the teaching model notebook-local with `v2_08_` helper names, and document formulas and scenario constants in Math Peek/source blocks. |
| Existing lab is a generic renderer wrapper. | Preserve bootstrap, track selector, and ledger patterns, but replace the body with direct concept modules. |
| Track thresholds differ in kind. | Use a typed local track packet derived from `get_track_profile` and `get_lab_track_variant`; expose thresholds in the opening and tables. |
| Placement model could imply exact production scheduling. | Label it as a teaching model with chapter-local formulas and scenario assumptions. |
| Parallel workers may edit other labs. | Edit only `labs/vol2/lab_08_fleet_orch.py` and this track plan; do not touch shared helpers or tests. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A - queueing pressure | 3 | 3 | 3 | 3 | 3 | 3 | Pass: scenario, prediction, manipulation, queue chart/table, failure, Math Peek, checkpoint. |
| Part B - priority/preemption | 3 | 3 | 3 | 3 | 3 | 2 | Pass: chapter formula plus local preemption-tax assumptions documented. |
| Part C - placement/bin packing | 3 | 3 | 3 | 3 | 3 | 3 | Pass: heatmap, locality score, fragmentation boundary, Math Peek, checkpoint. |
| Part D - policy gate | 3 | 3 | 3 | 3 | 3 | 2 | Pass: guardrail conjunction and track thresholds; local scoring assumptions documented. |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 | Pass: memo ties every module to the invariant and V2-09 handoff. |

Acceptance checks:

- No module scores below 2.
- Every module has at least five student-facing beats.
- Every module includes structured prediction, manipulation, evidence, a
  consequence/boundary, Math Peek/source model, and checkpoint/report decision.
- Part A, Part B, and Part D have reversible failure states.
- Synthesis ties all modules to the chapter invariant.
