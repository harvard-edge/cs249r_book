# Lab Part Storyline Depth Blueprint - 2026-06-13

This document extends the lab depth audit with a more detailed teaching model. The key idea is that a part is not a single question. A part is a short systems storyline that moves the student through a concept, a prediction, a controlled experiment, evidence, a source model, and a checkpoint.

The tracks can remain student-selectable. The depth comes from the work inside each concept, not from forcing every student through every track. The same Part A concept should feel different on iPhone, Oura Ring, RoboTaxi, and Cloud Fleet because the binding constraint, stakeholder pressure, and acceptable evidence are different.

## Core Analysis

The current plans usually name three concepts per lab:

- Part A names the first chapter idea.
- Part B names the second chapter idea.
- Part C names the final choice.

That is necessary but not sufficient. A deep lab needs a within-part narrative:

1. The student sees a concrete system situation.
2. The student predicts the failure mode or trade-off before evidence.
3. The student changes multiple controls, not one isolated knob.
4. The lab shows evidence as a plot, table, boundary, or incident trace.
5. The lab explains the source model or math behind the evidence.
6. The student saves a checkpoint that is reused later.
7. The synthesis asks for a decision that depends on multiple checkpoints.

The target shape should be "concept modules," not "question pages." Each part should take roughly 8-15 minutes and should have 4-6 small beats.

## Track Lenses

Every part should instantiate the same concept through the selected track.

| Track | Deep lab framing | What evidence should usually look like |
|---|---|---|
| iPhone | A product engineer protects sustained UX under battery, thermal, privacy, memory, and responsiveness limits. | Battery drain, thermal headroom, on-device latency, memory pressure, supported runtime path. |
| Oura Ring | A firmware engineer fits sensing and inference into SRAM, flash, radio, sampling, and battery budgets. | SRAM/flash fit, duty cycle, OTA payload, BLE/upload budget, battery-life regression. |
| RoboTaxi | A platform engineer protects safety-critical p99/p999 latency, rare-event recall, power, and fallback behavior. | Tail latency, sensor bandwidth, power envelope, safety margin, rare-event replay, fallback drill. |
| Cloud Fleet | A fleet owner manages SLA, throughput, utilization, cost/request, capacity headroom, and carbon. | p99 latency, throughput, cost/request, utilization, queueing/capacity, carbon, rollback/canary evidence. |

The same control can have different meaning by track:

- A "scale" knob is users/devices on iPhone, wearables and firmware cohorts on Oura, vehicles/geographies on RoboTaxi, and replicas/accelerators/requests on Cloud Fleet.
- A "latency" guardrail is responsiveness on iPhone, duty-cycle wake window on Oura, safety deadline on RoboTaxi, and SLA on Cloud Fleet.
- A "cost" guardrail is battery and thermal wear on device tracks, operational exposure on RoboTaxi, and dollars plus utilization on Cloud Fleet.

## New Support Needed

The current code has `LabTrackVariant`, `NuggetSpec`, and the generic `render_system_design_lab()` helper. That is not enough to represent deep parts. We need a typed storyline layer that is visible both to renderers and tests.

Recommended new structures:

```python
@dataclass(frozen=True)
class LabStoryline:
    lab_id: str
    title: str
    chapter_anchor: str
    throughline: str
    parts: tuple[PartStory, ...]
    synthesis: SynthesisStory

@dataclass(frozen=True)
class PartStory:
    part_id: str
    label: str
    concept: str
    storyline: str
    estimated_minutes: int
    beats: tuple[PartBeat, ...]
    track_lenses: Mapping[str, TrackPartLens]
    evidence_contract: EvidenceContract
    report_fields: tuple[str, ...]

@dataclass(frozen=True)
class PartBeat:
    beat_id: str
    role: str  # scenario, prediction, control, evidence, math, reflection, checkpoint
    prompt: str
    required: bool = True
```

Recommended renderer support:

- `storyline_part_tabs(storyline, controls, evidence)` should render Part A-D plus Synthesis.
- `part_beat_panel(part, beat)` should render consistent beat labels without making every notebook hand-code HTML.
- `math_peek(spec)` should expose the formula/source model behind every part.
- `checkpoint_writer(part_id, fields)` should save local ledger evidence.
- `synthesis_from_checkpoints(storyline, ledger)` should require cross-part evidence.
- `track_lens_panel(part, track)` should adapt scenario, metrics, defaults, and failure boundaries by track.

Recommended test support:

- Static tests should accept typed storyline metadata for shared-renderer labs instead of looking only for `def build_part_a()` in 109-line wrappers.
- Protocol tests should verify each part has required beat roles.
- Report tests should verify each required part checkpoint appears in the downloaded report.
- Browser smoke should click through tabs and check that the visible part contains at least four beat categories.

## Part Depth Contract

Every non-orientation part should satisfy this contract.

| Beat | Student move | Implementation requirement |
|---|---|---|
| Scenario | Understand the local system problem. | Track-specific stakeholder message and concrete constraint. |
| Prediction | Commit before seeing the answer. | Radio/dropdown/number input, not free text. |
| Experiment | Change the system. | At least one primary control and one secondary/advanced control when appropriate. |
| Evidence | Read the result. | Plot, table, budget card, trace, or phase diagram with fallback text. |
| Source model | Explain why the result happened. | Math Peek, solver/API name, MLSysIM refs, assumptions. |
| Reflection | Interpret the trade-off. | Structured reflection prompt tied to the concept. |
| Checkpoint | Save work for synthesis. | Ledger/report field with prediction, evidence, decision, residual risk. |

The synthesis should combine at least two part checkpoints. A synthesis that only repeats the last part's decision is too shallow.

## Volume I Storyline Blueprint

### V1-00 - The Architect's Portal

Purpose:
- Orient students to tracks and establish that the same ML idea changes under system constraints.

Part A - Constraint Portrait:
- A1: Select or inspect a track and stakeholder.
- A2: Predict the primary binding constraint for that track.
- A3: Compare the track's hardware, metric, and guardrail cards.
- A4: Identify what evidence would convince the stakeholder.
- A5: Save the selected track and first constraint hypothesis.

Part B - Same Model, Different World:
- B1: Hold the model idea constant across tracks.
- B2: Predict which track changes the feasibility most.
- B3: Inspect side-by-side device/fleet budgets.
- B4: Explain why "same model" is not "same system."
- B5: Save one cross-track contrast for future labs.

Part C - Engineering Lens:
- C1: Read the repeated lab workflow.
- C2: Practice prediction before reveal.
- C3: Identify controls, evidence, source trace, and report fields.
- C4: Decide how the track should shape future lab reports.
- C5: Export orientation memo.

### V1-01 - The AI Triad

Purpose:
- Teach that data, algorithm, and machine fail together, but one often binds first.

Part A - Diagnose Data, Algorithm, Machine:
- A1: Read a track-specific incident: bad UX, missed sensing, rare safety miss, or SLA regression.
- A2: Predict whether data, algorithm, or machine is the first binding cause.
- A3: Adjust data quality, model capacity, and hardware/runtime budget.
- A4: Inspect triad contribution scores and failure explanation.
- A5: Read Math Peek: weighted triad score and binding-cause selection.
- A6: Save first diagnosis.

Part B - Intervention Frontier:
- B1: Choose an intervention family: improve data, change algorithm, or change machine/runtime.
- B2: Predict which intervention gives the largest improvement per unit cost.
- B3: Sweep intervention budget and inspect diminishing returns.
- B4: Compare primary metric improvement against guardrail regression.
- B5: Save selected intervention and rejected alternatives.

Part C - Defensible Fix:
- C1: Combine diagnosis and intervention evidence.
- C2: Choose the fix that survives the track guardrail.
- C3: Name the validation test: cohort audit, SRAM/flash fit, rare-event replay, or load/SLO test.
- C4: Write residual risk.
- C5: Export triad diagnosis memo.

Depth addition:
- Add Part D only if needed: "Validation Review," where students test whether their fix merely moves the bottleneck from one triad vertex to another.

### V1-02 - Physics of Deployment

Purpose:
- Teach that deployment is constrained by memory, compute, bandwidth, energy, and latency before model ambition.

Part A - First Wall:
- A1: Read deployment request for selected track.
- A2: Predict first wall: memory, compute, bandwidth, energy, latency, cost.
- A3: Try candidate placements: local, edge/offload, cloud/fleet.
- A4: Inspect feasibility cards and first violated constraint.
- A5: Read Math Peek: feasibility as max of normalized constraint ratios.
- A6: Save first wall.

Part B - Physics Curve:
- B1: Move workload size, request rate, or sampling cadence.
- B2: Predict where the curve bends.
- B3: Sweep the pressure knob and inspect the first failure point.
- B4: Compare device tracks against Cloud Fleet.
- B5: Save collapse point and responsible constraint.

Part C - Deployment Choice:
- C1: Choose placement and mitigation.
- C2: Inspect trade-off table for latency, privacy, cost, and reliability.
- C3: Name what must be measured before rollout.
- C4: Save deployment decision.

Part D - Deployment Review:
- D1: Simulate stakeholder pushback: "why not just make the model bigger or move it?"
- D2: Re-evaluate the chosen placement under a stress scenario.
- D3: Confirm whether the mitigation still protects the guardrail.
- D4: Write final deployment review with residual risk.

### V1-03 - Constraint Tax

Purpose:
- Teach that late discovery of system constraints adds iteration cost.

Part A - Constraint Propagation:
- A1: Start with an ML workflow plan that omits a system constraint.
- A2: Predict when the hidden constraint appears.
- A3: Move discovery timing earlier or later.
- A4: Inspect rework cost, lost time, and guardrail damage.
- A5: Read Math Peek: rework tax as discovery delay times dependency breadth.
- A6: Save discovered constraint.

Part B - Iteration Frontier:
- B1: Allocate time among data, model, validation, and systems instrumentation.
- B2: Predict the best allocation under the track.
- B3: Sweep allocation and inspect quality versus delivery risk.
- B4: Identify the point where more ML iteration stops helping.
- B5: Save iteration policy.

Part C - Workflow Policy:
- C1: Choose gating tests and review cadence.
- C2: Compare lightweight and heavy process options.
- C3: Name what evidence must be collected before shipping.
- C4: Save workflow policy.

Part D - Workflow Budget Review:
- D1: Introduce a deadline or incident.
- D2: Re-run the chosen policy under time pressure.
- D3: Decide what to cut and what cannot be cut.
- D4: Export workflow policy memo.

### V1-04 - Data Gravity

Purpose:
- Teach that data movement, freshness, storage, and privacy shape ML systems.

Part A - Feed The Model:
- A1: Read where data is generated on the selected track.
- A2: Predict whether compute waits for data or data waits for compute.
- A3: Change data volume, sampling rate, freshness, or retention.
- A4: Inspect pipeline time and bottleneck stage.
- A5: Read Math Peek: stage throughput and bottleneck latency.
- A6: Save pipeline bottleneck.

Part B - Data Movement Frontier:
- B1: Choose local processing, upload, caching, or summarization.
- B2: Predict which strategy protects the primary metric.
- B3: Sweep bandwidth/freshness/retention pressure.
- B4: Inspect privacy, storage, freshness, and cost trade-offs.
- B5: Save frontier candidate.

Part C - Pipeline Architecture:
- C1: Build a track-specific data path.
- C2: Add one guardrail: privacy, safety, quality, SLA, or battery.
- C3: Inspect architecture feasibility and failure mode.
- C4: Save pipeline design.

Part D - Pipeline Incident Review:
- D1: Introduce stale data, upload failure, storage surge, or privacy restriction.
- D2: Test whether the pipeline degrades gracefully.
- D3: Choose a retention and fallback policy.
- D4: Export pipeline architecture memo.

### V1-05 - Activation Tax

Purpose:
- Teach that neural computation is not just parameters and FLOPs; activations and memory movement matter.

Part A - Operation Ledger:
- A1: Pick an operator or layer pattern.
- A2: Predict whether compute or memory dominates.
- A3: Change batch, tensor shape, precision, or sequence/window size.
- A4: Inspect operation, activation, and movement ledger.
- A5: Read Math Peek: operations, bytes, arithmetic intensity.
- A6: Save dominant operator wall.

Part B - Memory Cliff:
- B1: Increase working set until it crosses a cache/SRAM/memory boundary.
- B2: Predict the cliff location.
- B3: Sweep shape or sequence length.
- B4: Inspect latency and energy jump at the boundary.
- B5: Save cliff and mitigation.

Part C - Layer Design:
- C1: Compare candidate layer/block designs.
- C2: Predict which one survives the selected track.
- C3: Inspect accuracy proxy, latency, memory, and guardrail.
- C4: Save layer design.

Part D - Layer Budget Review:
- D1: Reassemble operator, memory, and design evidence.
- D2: Apply a track-specific stress case.
- D3: Choose fusion, tiling, quantization, or architecture simplification.
- D4: Export operator budget note.

### V1-06 - Architecture Tax

Purpose:
- Teach that architecture choice creates deployment obligations.

Part A - Architecture Signature:
- A1: Compare candidate architectures by compute, memory, dataflow, and kernel support.
- A2: Predict the architecture that fits the track.
- A3: Change input size, width/depth, or operator mix.
- A4: Inspect signature cards and constraint ratios.
- A5: Read Math Peek: architecture signature vector.
- A6: Save first architecture candidate.

Part B - Scaling Shape:
- B1: Scale model size or input complexity.
- B2: Predict which architecture degrades fastest.
- B3: Sweep scale and inspect non-linear pressure.
- B4: Name the first guardrail to fail.
- B5: Save scaling evidence.

Part C - Architecture Choice:
- C1: Choose architecture under track constraints.
- C2: Inspect validation and operational risks.
- C3: State rejected alternatives.
- C4: Save architecture recommendation.

Part D - Architecture Rollout Review:
- D1: Introduce deployment rollout and maintenance pressure.
- D2: Test the chosen architecture against runtime support and update cadence.
- D3: Decide whether to specialize, simplify, or accept complexity.
- D4: Export architecture recommendation memo.

### V1-07 - Framework Tax

Purpose:
- Teach that frameworks, compilers, runtimes, and kernels alter system behavior.

Part A - Dispatch Tax:
- A1: Run a workload through eager, compiled, and specialized paths.
- A2: Predict whether dispatch overhead matters.
- A3: Change batch, op count, and call frequency.
- A4: Inspect overhead share and latency budget.
- A5: Read Math Peek: dispatch overhead fraction.
- A6: Save dispatch diagnosis.

Part B - Fusion And Compile Break-Even:
- B1: Turn fusion/compilation on and off.
- B2: Predict the break-even point.
- B3: Sweep repetitions, compile cost, and kernel speedup.
- B4: Inspect amortization curve.
- B5: Save break-even threshold.

Part C - Runtime Choice:
- C1: Choose runtime/delegate/backend.
- C2: Compare portability, supported ops, performance, and debugging cost.
- C3: Name a fallback path.
- C4: Save runtime decision.

Part D - Runtime Migration Review:
- D1: Introduce a model or OS/runtime update.
- D2: Test whether the chosen framework path still works.
- D3: Decide whether to pin, migrate, or add fallback.
- D4: Export runtime migration memo.

### V1-08 - Training Gauntlet

Purpose:
- Teach that training feasibility is a systems budget across memory, precision, batch, checkpoints, and time.

Part A - Training Memory Budget:
- A1: Break training memory into weights, gradients, optimizer state, activations, and buffers.
- A2: Predict the largest term.
- A3: Change precision, batch, checkpointing, or model size.
- A4: Inspect memory stack and fit status.
- A5: Read Math Peek: training memory accounting.
- A6: Save memory bottleneck.

Part B - Feasibility Knobs:
- B1: Try mitigation knobs: smaller batch, checkpointing, mixed precision, offload.
- B2: Predict which knob buys most feasibility per cost.
- B3: Sweep knobs and inspect throughput/quality/training-time effects.
- B4: Save feasible configuration.

Part C - Training Plan:
- C1: Choose a training plan under track or fleet constraints.
- C2: Add validation and recovery requirements.
- C3: Name what can fail after the run begins.
- C4: Save training plan.

Part D - Training Run Review:
- D1: Introduce interrupted training, memory pressure, or cost spike.
- D2: Test whether the plan can recover.
- D3: Choose checkpoint/retry/scale-down policy.
- D4: Export training run memo.

### V1-09 - Selection Paradox

Purpose:
- Teach that more data is not automatically better; selection changes coverage, bias, cost, and freshness.

Part A - Quality Versus Quantity:
- A1: Compare data quantity, quality, and labeling cost.
- A2: Predict whether more data improves the selected track.
- A3: Sweep data size and quality.
- A4: Inspect utility, cost, and guardrail.
- A5: Read Math Peek: diminishing returns and selection utility.
- A6: Save selection pressure point.

Part B - Coverage And Inequality:
- B1: Inspect subgroup, geography, sensor, or request-slice coverage.
- B2: Predict the under-covered slice.
- B3: Adjust sampling or retention policy.
- B4: Inspect coverage matrix and risk.
- B5: Save coverage risk.

Part C - Data Policy:
- C1: Choose collection, retention, and labeling policy.
- C2: Compare privacy/safety/cost/quality impacts.
- C3: State rejected policy.
- C4: Save data policy.

Part D - Data Governance Review:
- D1: Introduce a new consent, safety, fairness, or SLA constraint.
- D2: Re-test the policy.
- D3: Decide what data is worth keeping.
- D4: Export data policy memo.

### V1-10 - Compression Paradox

Existing deep shape should remain. Preserve the current five-part structure:

- Part A: Quantization feasibility.
- Part B: Pruning trap.
- Part C: Pareto frontier.
- Part D: Energy dividend.
- Part E: Distillation.

Depth focus:
- Ensure every part checkpoint contributes to the final compression recipe.
- Continue moving reusable constants into MLSysIM or typed metadata.

### V1-11 - Hardware Roofline

Existing deep shape should remain. Preserve the current five-part structure:

- Part A: Roofline diagnosis.
- Part B: Move the point through fusion or data movement.
- Part C: Balance shift.
- Part D: Energy roofline.
- Part E: Tiling dividend.

Depth focus:
- Keep the math/source model visible and ensure track-specific hardware refs are explicit.

### V1-12 - Benchmarking Trap

Existing deep shape should remain. Preserve the current four-part structure:

- Part A: Amdahl ceiling.
- Part B: Thermal cliff.
- Part C: Multi-metric trap.
- Part D: Tail latency.

Depth focus:
- Ensure synthesis asks students to reject at least one misleading benchmark.

### V1-13 - Tail Latency Trap

Purpose:
- Teach that serving quality is dominated by queues, tails, batching, state, and cold paths.

Part A - Queueing Failure:
- A1: Read an SLO incident.
- A2: Predict whether service time or arrival rate drives the p99 breach.
- A3: Change QPS, service time, replicas, and utilization.
- A4: Inspect p50/p95/p99 and queue wait.
- A5: Read Math Peek: utilization and queueing tail.
- A6: Save queueing diagnosis.

Part B - Serving Knobs:
- B1: Try batching, cache, replicas, routing, or admission control.
- B2: Predict which knob improves p99 without hurting the guardrail.
- B3: Sweep knob settings.
- B4: Inspect throughput, p99, memory, and cost.
- B5: Save serving knob choice.

Part C - Capacity Plan:
- C1: Choose capacity or local-serving plan.
- C2: Inspect risk under burst traffic.
- C3: Save capacity plan and rollback condition.

Part D - Tail Incident Review:
- D1: Introduce cold start, cache miss, or burst.
- D2: Test whether the capacity plan survives.
- D3: Choose fallback or admission policy.
- D4: Export serving capacity memo.

### V1-14 - Silent Degradation

Purpose:
- Teach that deployed ML systems degrade when monitoring, retraining, and ownership are weak.

Part A - Drift Visibility:
- A1: Read a hidden drift incident.
- A2: Predict which slice drifts first.
- A3: Change monitoring coverage, sampling, and alert threshold.
- A4: Inspect detection delay and false alert rate.
- A5: Read Math Peek: detection power and threshold trade-off.
- A6: Save visibility diagnosis.

Part B - Retraining Cadence:
- B1: Choose retraining trigger: schedule, drift, performance, or incident.
- B2: Predict best cadence.
- B3: Sweep cadence and inspect freshness, cost, and risk.
- B4: Save retraining policy.

Part C - Ops Policy:
- C1: Choose canary, rollback, ownership, and escalation path.
- C2: Compare lightweight and heavy policies.
- C3: Save ops policy.

Part D - Debt Cascade:
- D1: Introduce accumulated tech debt or alert fatigue.
- D2: Inspect cascading failure path.
- D3: Choose what to automate and what to keep human-reviewed.
- D4: Export ops memo.

### V1-15 - No Free Fairness

Purpose:
- Teach that responsible engineering is a systems budget, not a post-hoc slogan.

Part A - Metric Conflict:
- A1: Read a track-specific harm or stakeholder conflict.
- A2: Predict which metric conflict binds.
- A3: Adjust threshold, subgroup weighting, or operating point.
- A4: Inspect utility, fairness, latency, cost, and quality.
- A5: Read Math Peek: subgroup metric conflict.
- A6: Save metric conflict.

Part B - Responsibility Budget:
- B1: Add mitigation strength, review load, or explanation requirement.
- B2: Predict overhead and residual harm.
- B3: Sweep mitigation budget.
- B4: Save responsibility budget.

Part C - Explainability Tax:
- C1: Choose explanation/audit depth.
- C2: Inspect latency, storage, cost, and governance effects.
- C3: Save explanation policy.

Part D - Carbon Ledger:
- D1: Add energy/carbon accountability.
- D2: Compare deployment choices under carbon and quality guardrails.
- D3: Decide what trade-off is defensible.
- D4: Export responsible engineering memo.

### V1-16 - The Architect's Audit

Purpose:
- Teach students to synthesize earlier evidence into a systems architecture memo.

Part A - Ledger Replay:
- A1: Read prior local ledger decisions.
- A2: Predict which earlier decision is most fragile.
- A3: Replay key evidence.
- A4: Save fragile assumption.

Part B - Architecture Map:
- B1: Map data, model, runtime, deployment, monitoring, and governance.
- B2: Predict where the architecture will fail first.
- B3: Inspect dependency and risk map.
- B4: Save architecture map.

Part C - Sensitivity Audit:
- C1: Change one assumption at a time.
- C2: Predict which assumption matters most.
- C3: Inspect sensitivity ranking.
- C4: Read Math Peek: weighted risk score.
- C5: Save sensitivity result.

Part D - Architecture Memo:
- D1: Choose final architecture.
- D2: State rejected alternatives.
- D3: State validation plan and residual risk.
- D4: Export final Volume I memo.

## Volume II Storyline Blueprint

### V2-01 - The Scale Illusion

Part A - Scaling Illusion:
- A1: Define what scale means for the selected track.
- A2: Predict the first non-linear collapse point.
- A3: Sweep fleet/users/requests/accelerators.
- A4: Inspect reliability, cost, and latency collapse.
- A5: Read Math Peek: compounded reliability and coordination overhead.
- A6: Save collapse point.

Part B - Coordination Tax:
- B1: Add rollout, monitoring, retries, synchronization, or heterogeneity.
- B2: Predict useful work versus overhead.
- B3: Sweep coordination burden.
- B4: Inspect useful work fraction.
- B5: Save coordination tax.

Part C - Scale Readiness:
- C1: Choose scale strategy: shard, specialize, stage, simplify, or refuse.
- C2: Inspect mitigation evidence.
- C3: Save readiness decision.

Part D - Rollout Gate:
- D1: Introduce a track-specific rollout incident.
- D2: Decide whether to continue, pause, or redesign.
- D3: Export scale failure-mode memo.

### V2-02 - The Compute Infrastructure Wall

Part A - Node Feasibility:
- A1: Compare workload demand with hardware envelope.
- A2: Predict whether compute, memory, bandwidth, power, or cost binds.
- A3: Change demand, precision, memory, and accelerator profile.
- A4: Inspect roofline and utilization.
- A5: Read Math Peek: compute/memory roofline.
- A6: Save first infrastructure wall.

Part B - Infrastructure Frontier:
- B1: Compare raw compute, balanced node, and demand reduction.
- B2: Predict best infrastructure move.
- B3: Sweep demand and capacity.
- B4: Inspect latency, cost, utilization, and quality.
- B5: Save frontier result.

Part C - Procurement Or Placement:
- C1: Choose buy/build/offload/local placement.
- C2: Inspect TCO, power, cooling, and SLA risk.
- C3: Save placement decision.

Part D - Capacity Review:
- D1: Add growth, supply constraint, or thermal limit.
- D2: Re-test infrastructure plan.
- D3: Export compute infrastructure memo.

### V2-03 - Network Fabric Design

Part A - Fabric Budget:
- A1: Identify message type and network path.
- A2: Predict whether bandwidth, latency, retries, or topology binds.
- A3: Change payload, frequency, fanout, and link capacity.
- A4: Inspect transfer time and deadline miss.
- A5: Read Math Peek: alpha-beta communication model.
- A6: Save fabric budget.

Part B - Topology Frontier:
- B1: Compare local, staged, tree, ring, mesh, or hierarchical paths.
- B2: Predict best topology under track constraints.
- B3: Sweep payload and participants.
- B4: Inspect frontier.
- B5: Save topology choice.

Part C - Fabric Decision:
- C1: Choose communication policy.
- C2: Add compression, delay tolerance, or reliability controls.
- C3: Save communication strategy.

Part D - Network Incident Review:
- D1: Inject congestion, intermittent connectivity, or synchronization delay.
- D2: Test policy under failure.
- D3: Export communication strategy memo.

### V2-04 - The Data Pipeline Wall

Part A - Storage-Compute Gap:
- A1: Compare storage bandwidth with compute demand.
- A2: Predict whether storage starves compute.
- A3: Change shard size, cache hit rate, freshness, and bandwidth.
- A4: Inspect stall time and utilization.
- A5: Read Math Peek: pipeline throughput minimum.
- A6: Save storage-compute gap.

Part B - Sharding And Cache Frontier:
- B1: Choose sharding, caching, prefetch, or summarization.
- B2: Predict best frontier point.
- B3: Sweep freshness and data volume.
- B4: Inspect cost, latency, and retention risk.
- B5: Save data movement policy.

Part C - Storage Architecture:
- C1: Build retention and movement architecture.
- C2: Add privacy/safety/SLA guardrail.
- C3: Save architecture.

Part D - Freshness Incident:
- D1: Introduce stale data, failed upload, or checkpoint storm.
- D2: Test architecture.
- D3: Export data pipeline memo.

### V2-05 - The Parallelism Puzzle

Part A - Memory Fit:
- A1: Break training memory by parallelism strategy.
- A2: Predict whether data, tensor, pipeline, or expert parallelism helps.
- A3: Change model size, batch, devices, and precision.
- A4: Inspect fit and utilization.
- A5: Read Math Peek: memory partitioning.
- A6: Save memory-fit result.

Part B - Parallelism Frontier:
- B1: Sweep parallelism configuration.
- B2: Predict scaling efficiency.
- B3: Inspect communication, bubble, and throughput.
- B4: Save frontier point.

Part C - Training Architecture:
- C1: Choose training strategy.
- C2: Add checkpoint, recovery, and deployment handoff.
- C3: Save architecture.

Part D - Distributed Run Review:
- D1: Inject straggler, failure, or communication bottleneck.
- D2: Test whether architecture survives.
- D3: Export parallelism strategy memo.

### V2-06 - Collective Communication

Keep the existing four-part shape, but add explicit math/source depth:

- Part A: Operation anatomy with alpha-beta cost and collective semantics.
- Part B: Algorithm/topology frontier with ring/tree/hierarchical comparison.
- Part C: Hierarchy as systems decision with interconnect boundaries.
- Part D: Overlap and compression with compute overlap and error feedback.
- Synthesis: Communication design review.

### V2-07 - When Failure Is Routine

Part A - Failure Exposure:
- A1: Define failure unit for the track.
- A2: Predict expected failure exposure.
- A3: Change fleet size, duty cycle, MTBF, and criticality.
- A4: Inspect expected incidents and impact.
- A5: Read Math Peek: failure probability over fleet-time.
- A6: Save exposure.

Part B - Recovery Frontier:
- B1: Compare retry, checkpoint, redundancy, degradation, and rollback.
- B2: Predict best recovery option.
- B3: Sweep recovery interval and overhead.
- B4: Inspect lost work, user/safety impact, and cost.
- B5: Save recovery policy.

Part C - Resilience Policy:
- C1: Choose resilience stack.
- C2: Add validation drill.
- C3: Save policy.

Part D - Failure Drill:
- D1: Inject outage, battery failure, sensor failure, node failure, or region failure.
- D2: Test response.
- D3: Export failure recovery memo.

### V2-08 - The Scheduling Trap

Part A - Queue/Utilization Wall:
- A1: Define jobs/requests/tasks for the track.
- A2: Predict whether utilization or latency binds.
- A3: Change arrival rate, service time, priority, and capacity.
- A4: Inspect queues, SLA, and utilization.
- A5: Read Math Peek: utilization and queue growth.
- A6: Save queue wall.

Part B - Fragmentation And Preemption Frontier:
- B1: Add heterogeneity, preemption, priority, or rollout cohorts.
- B2: Predict scheduler side effect.
- B3: Sweep policy and inspect fairness, SLA, and utilization.
- B4: Save frontier point.

Part C - Fleet Policy:
- C1: Choose scheduler policy.
- C2: Add guardrail and rollback.
- C3: Save policy.

Part D - Scheduler Incident:
- D1: Inject burst, urgent job, rollout, or safety event.
- D2: Test policy.
- D3: Export scheduling memo.

### V2-09 - The Optimization Trap

Part A - Bottleneck Diagnosis:
- A1: Identify the metric being optimized.
- A2: Predict local bottleneck.
- A3: Change compute, memory, batching, data path, or policy.
- A4: Inspect before/after bottleneck.
- A5: Read Math Peek: local speedup versus system speedup.
- A6: Save diagnosis.

Part B - Optimization Ladder:
- B1: Apply successive optimizations.
- B2: Predict when returns diminish.
- B3: Inspect side effects across guardrails.
- B4: Save ladder result.

Part C - Stop Rule:
- C1: Choose when to stop optimizing.
- C2: State what must not regress.
- C3: Save stop rule.

Part D - Side-Effect Audit:
- D1: Introduce a hidden guardrail: energy, safety, cost, quality, privacy, or carbon.
- D2: Test optimized system.
- D3: Export optimization side-effect memo.

### V2-10 - The Inference Economy

Existing deep shape should remain:

- Part A: Serving cost inversion.
- Part B: KV cache wall.
- Part C: Continuous batching.
- Part D: Fleet design challenge.
- Synthesis: Serving plan.

Depth focus:
- Ensure track-specific definitions of cost are explicit: battery on devices, safety exposure for RoboTaxi, cost/request and SLA for Cloud Fleet.

### V2-11 - The Edge Thermodynamics Lab

Existing deep shape should remain:

- Part A: Memory amplification tax.
- Part B: Adaptation strategy selector.
- Part C: Battery drain reality.
- Part D: Federation paradox.
- Synthesis: Edge architecture.

Depth focus:
- Preserve edge-specific thermodynamics and local update constraints.

### V2-12 - The Silent Fleet

Part A - Complexity Growth:
- A1: Define fleet slices and observability signals.
- A2: Predict which slice goes silent.
- A3: Change telemetry rate, slice coverage, and threshold.
- A4: Inspect blind spots and alert load.
- A5: Read Math Peek: coverage versus alert volume.
- A6: Save observability gap.

Part B - Canary/Automation Frontier:
- B1: Compare manual review, canary, automation, and rollback.
- B2: Predict best control point.
- B3: Sweep automation strength and canary size.
- B4: Inspect detection delay, false positives, and blast radius.
- B5: Save frontier.

Part C - Ops Architecture:
- C1: Choose monitoring and action policy.
- C2: State owner and escalation.
- C3: Save architecture.

Part D - Silent Failure Review:
- D1: Inject hidden drift or telemetry loss.
- D2: Test policy.
- D3: Export fleet monitoring memo.

### V2-13 - The Price of Privacy

Part A - Threat/Privacy Budget:
- A1: Identify data, model, and system assets.
- A2: Predict highest-risk exposure.
- A3: Change data access, retention, encryption, or privacy strength.
- A4: Inspect privacy risk and utility.
- A5: Read Math Peek: privacy/utility or overhead model.
- A6: Save privacy budget.

Part B - Defense Overhead Frontier:
- B1: Compare isolation, DP, secure aggregation, enclave, or access control.
- B2: Predict overhead and utility loss.
- B3: Sweep defense strength.
- B4: Inspect latency, utility, cost, and governance.
- B5: Save defense choice.

Part C - Security/Privacy Policy:
- C1: Choose policy.
- C2: Add deletion lineage, consent, audit, or incident response.
- C3: Save policy.

Part D - Incident Response:
- D1: Inject leakage, model theft, consent issue, or supply-chain threat.
- D2: Test policy.
- D3: Export privacy control memo.

### V2-14 - The Robustness Budget

Part A - Robustness Tax:
- A1: Define stress or shift scenario.
- A2: Predict failure mode.
- A3: Change stress exposure and mitigation strength.
- A4: Inspect utility, robustness, and cost.
- A5: Read Math Peek: robustness budget or shift distance.
- A6: Save robustness tax.

Part B - Drift/Silent Error Timeline:
- B1: Simulate drift or rare silent error.
- B2: Predict detection time.
- B3: Sweep monitoring and fallback.
- B4: Inspect timeline and residual risk.
- B5: Save drift response.

Part C - Defense Stack:
- C1: Choose hardening, monitoring, fallback, and governance.
- C2: Compare options.
- C3: Save defense stack.

Part D - Stress Review:
- D1: Inject out-of-distribution or adversarial condition.
- D2: Test defense stack.
- D3: Export robustness budget memo.

### V2-15 - The Carbon Budget

Part A - Energy/Carbon Measurement:
- A1: Define energy boundary: device, vehicle, datacenter, or lifecycle.
- A2: Predict dominant energy/carbon source.
- A3: Change demand, utilization, region, duty cycle, or cooling.
- A4: Inspect operational and embodied carbon.
- A5: Read Math Peek: energy times carbon intensity plus lifecycle allocation.
- A6: Save carbon source.

Part B - Placement/Lifecycle Frontier:
- B1: Compare local, edge, region, schedule, or model choice.
- B2: Predict best carbon-aware move.
- B3: Sweep placement and demand.
- B4: Inspect quality, latency, cost, and carbon.
- B5: Save frontier.

Part C - Carbon-Aware Policy:
- C1: Choose energy/carbon policy.
- C2: Add quality and SLA guardrails.
- C3: Save policy.

Part D - Sustainability Review:
- D1: Introduce grid constraint, cooling failure, or demand spike.
- D2: Test policy.
- D3: Export carbon budget memo.

### V2-16 - The Fairness Budget

Part A - Metric Conflict And Feedback:
- A1: Define affected groups or slices.
- A2: Predict metric conflict.
- A3: Change threshold, mitigation, or data policy.
- A4: Inspect fairness, utility, latency, and cost.
- A5: Read Math Peek: fairness metric conflict and feedback loop.
- A6: Save conflict.

Part B - Governance Overhead:
- B1: Add review depth, explanation, audit, or appeal path.
- B2: Predict overhead.
- B3: Sweep governance intensity.
- B4: Inspect utility, workload, latency, and risk.
- B5: Save governance budget.

Part C - Responsible AI Pipeline:
- C1: Choose pipeline controls.
- C2: Add monitoring and escalation.
- C3: Save pipeline.

Part D - Harm Review:
- D1: Inject subgroup regression or stakeholder complaint.
- D2: Test pipeline response.
- D3: Export fairness budget memo.

### V2-17 - The Fleet Synthesis

Part A - Fleet Ledger Replay:
- A1: Read prior Volume II decisions.
- A2: Predict which earlier decision creates the biggest interaction risk.
- A3: Replay selected decisions and risks.
- A4: Save ledger risk.

Part B - Interaction Map:
- B1: Map interactions among scale, infrastructure, network, data, training, failure, ops, security, robustness, carbon, and fairness.
- B2: Predict the strongest coupling.
- B3: Inspect interaction graph.
- B4: Save interaction map.

Part C - Final Design Review:
- C1: Choose final integrated architecture.
- C2: State rejected patchwork alternatives.
- C3: Save design review.

Part D - Risk Register:
- D1: Build residual risk register.
- D2: Assign validation tests and owners.
- D3: Decide launch/no-launch criteria.
- D4: Export final fleet architecture report.

## Implementation Consequences

The main implementation gap is not a missing chart type. It is a missing representation of the internal narrative of each part.

Highest-leverage work:

1. Add a typed `LabStoryline` registry, starting with the 14 shared Volume II shell labs.
2. Teach `render_system_design_lab()` to render actual Part A-D tabs from that registry.
3. Add `MathPeekSpec` and `EvidenceContract` so every part has source-model depth.
4. Add a ledger checkpoint contract per part.
5. Add static tests that verify part beats, not only literal notebook function names.
6. Convert V1-02 through V1-09 to the same story contract instead of hand-maintaining inline panels.

The deep exemplar labs should guide implementation. The goal is not to make every lab long. The goal is to make every lab coherent: each concept should have a sequence of work that teaches students how to reason through a system, and the final report should prove that reasoning happened.
