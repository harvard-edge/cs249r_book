# V1-07 Track Plan: ML Frameworks Concept Module Packet

## Chapter Invariant

Framework abstractions carry runtime consequences: graph shape, dispatch, portability, and kernel support change the deployed system even when the model math is unchanged.

## Shared Sequence Rule

This lab has one shared concept sequence for every student:

1. Part A: eager and graph execution pay different overheads depending on reuse and dynamism.
2. Part B: kernel fusion/runtime support can remove dispatch and memory traffic, but only for supported shapes.
3. Part C: portability is an amount-system trade where compatibility can cost performance or capability.
4. Part D: framework selection must satisfy deployment constraints and validation evidence.
5. Synthesis: the student records a runtime deployment recommendation with source-traced assumptions.

Tracks do not create different concepts. The selected track changes persona, constraints, thresholds, evidence emphasis, failure mode, and report framing for the same shared concept modules.

## Reading Map

| Lab module | Chapter anchor | Claim used in the lab |
|---|---|---|
| Part A - Execution overhead depends on reuse and dynamism | `Execution Problem`, `Three execution strategies`, `The dispatch tax` | Eager execution exposes each operation to host/runtime dispatch; graph execution can reduce repeated overhead only when the graph is stable enough to reuse. |
| Part B - Fusion removes dispatch and memory traffic only for supported shapes | `Why execution strategy matters: The memory wall`, `Kernel fusion`, `Hybrid approaches: JIT compilation` | Fusion cuts intermediate memory traffic and dispatch count, but unsupported operations, graph breaks, and shape guards return work to slower paths. |
| Part C - Portability is an amount-system trade | `Deployment Targets`, `ONNX`, `Framework Selection` | Compatibility reduces migration and target lock-in but can cost latency, supported operators, memory footprint, or accelerator capability. |
| Part D - Framework selection must satisfy deployment constraints and validation evidence | `Framework Selection`, `Fallacies and Pitfalls`, `Summary` | The runtime choice is a constrained optimization over execution model, operator support, target hardware, validation burden, and operational risk. |
| Synthesis - Runtime deployment recommendation | `Purpose`, `ML compiler`, `Framework selection trade-off space` | The recommendation must state assumptions, measured evidence, constraint boundaries, and source trace. |

## Concept Inventory

### Accepted Concepts

1. Frameworks as compiler-like translation layers.
   - Reason: this is the chapter-level invariant and connects every activity to algorithm-machine co-design.
2. Dispatch tax and execution model trade-off.
   - Reason: dispatch overhead becomes visible in a small, manipulable latency stack.
3. Kernel fusion and graph support.
   - Reason: fusion gives students a mechanism for why graph execution can win and why unsupported shapes can erase the win.
4. Deployment runtimes and portability.
   - Reason: the same model must travel through Core ML, TFLite Micro, TensorRT, ONNX, or graph capture paths with different supported amounts.
5. Framework selection as constrained optimization.
   - Reason: the final decision must satisfy track constraints and validation evidence, not just minimize median latency.

### Rejected Or Deferred Concepts

1. Full automatic differentiation internals.
   - Reason: important to the chapter, but it would dilute this lab's deployment-runtime storyline; training-memory consequences belong in later labs.
2. Detailed `nn.Module` mechanics and serialization.
   - Reason: useful for framework fluency but not the strongest deployment consequence for V1-07.
3. Framework history and abstraction ladder.
   - Reason: provides context but does not create a short interactive design decision.
4. Low-level custom-kernel authoring.
   - Reason: the lab can reason about kernel support and fusion without asking students to implement kernels.
5. Full benchmark methodology.
   - Reason: V1-12 owns benchmarking depth; this lab uses simplified source-traced scenario calculations.

## Track Narratives And Amount Systems

| Track | Stakeholder | Amount system | Binding constraints | Natural failure | Report emphasis |
|---|---|---|---|---|---|
| iPhone | Mobile product engineer | Local runtime with Core ML or TFLite-style delegates | Operator support, latency, memory footprint, thermal/battery headroom | Unsupported op falls back off delegate and turns an interactive feature into a hot, battery-heavy path | Recommend local runtime and delegate coverage test. |
| Oura Ring | Wearable firmware engineer | TFLite Micro-like fixed memory arena and tiny firmware image | SRAM/flash footprint, operator resolver, wake time, duty-cycle energy | Runtime or custom op set exceeds memory arena or OTA payload | Recommend fixed kernels or micro runtime with memory trace. |
| RoboTaxi | Autonomous vehicle platform engineer | Deterministic runtime with safety validation evidence | p99/p999 latency, accelerator support, plugin certification, fallback determinism | Portable fallback path injects synchronization jitter into the safety loop | Recommend deterministic runtime and replay/plugin audit. |
| Cloud Fleet | Fleet service owner | Graph compiler/reuse economics for high-volume serving | Throughput, shape stability, graph breaks, p99 latency, cost/request | Compile cost or dynamic shapes prevent amortization and raise cost/SLA risk | Recommend graph capture or compiled serving path with load canary. |

Track amount-system rule: every module expresses the same framework concept in the track's quantities. iPhone reports latency, delegate coverage, footprint, and battery/thermal validation. Oura Ring reports memory arena, kernel count, OTA/runtime footprint, and duty-cycle risk. RoboTaxi reports deterministic p99/p999 evidence, plugin coverage, and safety validation. Cloud Fleet reports reuse count, throughput/cost amortization, graph-break risk, and rollback canary.

## Concept Modules

### Part A: Concept Module - Eager And Graph Execution Pay Different Overheads Depending On Reuse And Dynamism

Chapter claim:
- Eager execution is debuggable because it executes immediately, but the framework cannot see enough of the graph to remove dispatch overhead. Graph execution can reduce overhead when the graph is stable and reused.

Student prior:
- Expected belief: framework overhead is a small constant, so the faster hardware or lower FLOP count should dominate.
- Productive failure: a dynamic/eager path can lose even when useful compute is small because dispatch is paid for every operation.

Storyline:
1. Scenario: the selected track's stakeholder must decide whether the current runtime can ship.
2. Prediction: the student chooses which overhead source will bind first.
3. Manipulation: the student adjusts operation count and compares eager, delegate, compiled, and portable runtime paths.
4. Evidence: stacked latency chart and table separate useful compute, runtime dispatch, hardware dispatch, transfer, sync, and memory.
5. Consequence: a failure callout names the first violated track constraint and the recovery lever.
6. Math/source: overhead ratio uses `N_ops * t_dispatch / (T_compute + T_memory)` and the dispatch tax chapter anchor.
7. Checkpoint: the student chooses whether the workload is dispatch-bound, memory-bound, or support-bound for the selected track.

Mechanics:
- Structured radio prediction.
- Operation-count slider.
- Stacked latency chart with latency budget line.
- Feasibility table with violations.
- Prediction-vs-actual feedback card.

Ledger fields:
- `part_a_prediction`, `part_a_actual_dominant_overhead`, `op_count`, `part_a_checkpoint`.

### Part B: Concept Module - Kernel Fusion And Runtime Support Remove Dispatch And Memory Traffic Only For Supported Shapes

Chapter claim:
- Fusion reduces data movement and launch count by combining visible, supported operations, but graph breaks, dynamic shapes, and unsupported operators shorten compiled regions.

Student prior:
- Expected belief: compilation or fusion always helps once enabled.
- Productive failure: a runtime with a high fusion factor can still fail if support is below the track floor or if shape dynamism reduces reuse.

Storyline:
1. Scenario: the stakeholder needs to know if the runtime's compile/delegate cost pays back for the expected operating volume.
2. Prediction: the student predicts whether the selected compiled path pays back before the expected reuse count.
3. Manipulation: the student adjusts reuse count and shape dynamism.
4. Evidence: break-even chart/table compares compile cost, per-inference savings, selected reuse, and shape-adjusted support.
5. Consequence: a boundary callout names no-payback, support-floor miss, or shape-guard failure.
6. Math/source: `N_breakeven = T_compile / (T_eager - T_compiled)` plus support-adjusted reuse.
7. Checkpoint: the student records whether to compile, bucket/pad shapes, or stay eager/portable.

Mechanics:
- Reuse-count slider.
- Shape-dynamism slider.
- Break-even bar chart with expected reuse line.
- Support-adjusted runtime table.
- Failure state for no-payback or unsupported-shape boundary.

Ledger fields:
- `part_b_prediction`, `reuse_count`, `shape_dynamism_pct`, `selected_break_even_inferences`, `part_b_checkpoint`.

### Part C: Concept Module - Portability Is An Amount-System Trade

Chapter claim:
- Interchange formats and portable runtimes reduce fragmentation, but compatibility can lose target-specific performance, supported operators, or capabilities.

Student prior:
- Expected belief: a portable runtime is the safest default because it keeps options open.
- Productive failure: the portable path may be feasible in one amount system and unacceptable in another because compatibility consumes latency, memory, or safety/cost budget.

Storyline:
1. Scenario: the stakeholder asks whether to optimize for native target performance or portable deployment.
2. Prediction: the student predicts which amount compatibility will cost most on the selected track.
3. Manipulation: the student chooses a runtime path and compares portability risk across all candidates.
4. Evidence: portability table shows footprint, kernel support, latency headroom, support headroom, and risk.
5. Consequence: a failure/boundary callout explains the trade: compatibility can cost performance or capability.
6. Math/source: normalized compatibility score combines latency headroom, support headroom, and footprint headroom.
7. Checkpoint: the student chooses native/delegate, portable interchange, generated code, or rollback baseline.

Mechanics:
- Runtime dropdown.
- Portability-risk prediction radio.
- Runtime feasibility table with headroom amounts.
- Selected-runtime evidence card.
- Source trace to deployment targets and framework selection anchors.

Ledger fields:
- `part_c_prediction`, `selected_runtime`, `portability_risk`, `unsupported_op_warning`, `part_c_checkpoint`.

### Part D: Concept Module - Framework Selection Requires Deployment Constraints And Validation Evidence

Chapter claim:
- Framework selection is constrained optimization across execution visibility, hardware abstraction, operator support, team workflow, and validation evidence.

Student prior:
- Expected belief: choose the fastest runtime from the chart.
- Productive failure: the fastest runtime is not enough if validation evidence, fallback behavior, or rollback path does not satisfy the selected deployment context.

Storyline:
1. Scenario: the stakeholder asks for a release decision with validation evidence.
2. Prediction: the student chooses which validation item is non-negotiable for the track.
3. Manipulation: the student selects a release posture and runtime, then inspects rejected alternatives.
4. Evidence: release-readiness table combines feasibility, break-even, support, validation requirement, and residual risk.
5. Consequence: a checkpoint warns when the selected runtime is infeasible, does not pay back, or lacks validation discipline.
6. Math/source: amount-system decision rule requires latency <= budget, footprint <= budget, support >= floor, and reuse >= break-even where applicable.
7. Checkpoint: the student records a go/no-go/rework recommendation.

Mechanics:
- Validation radio.
- Release posture radio.
- Runtime decision table and rejected alternatives.
- Failure/recovery callout.
- Final recommendation text area.

Ledger fields:
- `part_d_validation_focus`, `part_d_release_posture`, `runtime_feasible`, `validation_requirement`, `residual_risk`, `final_recommendation`.

### Synthesis: Concept Module - Runtime Deployment Recommendation With Source-Traced Assumptions

Chapter claim:
- Frameworks are part of the system architecture; the final recommendation must make the runtime assumptions auditable.

Student output:
1. Decision: selected runtime and release posture.
2. Constraint: the binding overhead or violated amount.
3. Evidence: latency, break-even, support, and footprint numbers.
4. Source trace: chapter anchors, MLSysIM refs, shared helper APIs, and track profile.
5. Carry-forward: risk that future training, compression, serving, or monitoring labs should preserve.

Ledger fields:
- `track_id`, `scenario_id`, `hardware_ref`, `model_ref`, `selected_runtime`, `dominant_overhead`, `break_even_inferences`, `total_latency_ms`, `kernel_support_pct`, `runtime_feasible`, `validation_requirement`, `residual_risk`, `final_recommendation`.

## Mechanics And Evidence Plan

| Need | Mechanic | Evidence |
|---|---|---|
| Productive failure | Structured prediction radios for Parts A-D | Prediction-vs-actual cards with actual values. |
| Boundary finding | Operation count, reuse count, and shape dynamism sliders | Feasibility state and boundary text. |
| Bottleneck diagnosis | Stacked latency chart and dominant-overhead table | Named overhead source and violated constraint. |
| Context transfer | Track selector and track-specific profile values | Same runtime concept rendered in different amount systems. |
| Trade-off reasoning | Runtime dropdown and portability table | Selected runtime with rejected alternatives. |
| Source model | Math Peek accordions and source trace blocks | Formulas and source refs tied to chapter anchors. |
| Synthesis | Report export panel and Design Ledger save | Runtime deployment recommendation with assumptions. |

## Data And Solver Contracts

Existing helpers:
- `framework_track_profile`
- `dispatch_stack`
- `compile_break_even`
- `runtime_decision`
- `track_selector`, `track_context`, `track_arc_context`
- `build_lab_report`, `report_export_panel`, `source_trace`

Notebook-local helpers:
- Use only `v1_07_`-prefixed helpers for formatting, prediction feedback, support-adjusted calculations, and tables.
- Do not create broad shared abstractions for this wave.

Needed inputs:
- `track_id`
- `runtime_strategy`
- `op_count`
- `shape_dynamism`
- `reuse_count`
- `delegate_support`
- validation/release posture choices

Needed outputs:
- dispatch stack
- compile break-even
- support-adjusted feasibility
- portability risk
- release recommendation
- report snapshot object serialized into the Design Ledger

## Accessibility And Fallback Plan

- Every plot has a table fallback with exact values.
- Failure states state value, limit, unit, and mitigation in text.
- Color is not the only indicator: tables include `Feasible`, `Pays back`, `Violation`, and `Decision` text.
- Required predictions use structured controls, not free text.
- The report contains the same decision evidence as the visual notebook.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Existing shared helper does not model shape-dynamism directly | Add notebook-local support-adjusted rows rather than editing shared helper APIs. |
| Runtime catalog numbers live in variant metadata and are simplified | Source trace makes the registry/helper source explicit and labels the calculations as lab scenario models. |
| Parallel workers may edit other labs | Restrict edits to `lab_07_ml_frameworks.py` and this track plan only. |
| Marimo dataflow can become brittle after a large rewrite | Keep helper functions in one local cell, avoid broad shared state, and run `py_compile`. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Pass gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, manipulation, boundary, evidence, Math Peek, checkpoint. |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass: reuse and shape controls reveal compile/fusion boundary. |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Pass: runtime choice exposes portability amount trade. |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Pass: release decision requires validation evidence. |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 | Pass: source-traced runtime deployment recommendation saved to ledger/report. |

Minimum acceptance check:
- No dimension below 2.
- Each module has 5+ activity beats.
- At least one reversible failure state is reachable through operation count, reuse count, shape dynamism, or runtime choice.
- Synthesis ties all modules back to the chapter invariant.
