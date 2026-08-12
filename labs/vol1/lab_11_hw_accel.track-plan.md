# V1-11 Concept-Module Track Plan: Hardware Acceleration

## Chapter Invariant

Accelerators expose bottlenecks; speedup comes from matching workload arithmetic intensity, precision, memory hierarchy, and hardware capability to the deployment envelope.

This lab has one shared concept sequence for all tracks. The selected track changes persona, limits, units, thresholds, evidence emphasis, failure mode, and memo framing; it does not change the underlying concepts.

## Reading Map

| Lab module | Chapter anchor | Claim carried into the notebook |
|---|---|---|
| Opening | Purpose and learning objectives | Hardware selection is not peak-FLOP comparison; data movement and hardware fit determine realized speedup. |
| Part A | Roofline Model, Hardware ridge points | Arithmetic intensity compared with ridge point separates memory-bound from compute-bound regimes. |
| Part B | AI Memory Systems, Memory hierarchy, Host-accelerator communication, Kernel fusion | Data movement through slow memory tiers can dominate accelerator performance and energy even when compute is abundant. |
| Part C | Tensor Cores, Numerics in AI acceleration, Tensor Core contract | Tensor cores and reduced precision accelerate only supported shapes, supported formats, and numerically tolerable paths. |
| Part D | Heterogeneous SoC Design, Power and thermal management, Automotive systems, Hardware Sustainability, Fallacies and Pitfalls | Accelerator fit is a deployment recommendation constrained by latency, cost, power, validation, and residual risk. |
| Synthesis | Summary, Feasibility assessment | A hardware acceleration memo must name the bottleneck, selected accelerator and precision, rejected alternatives, and remaining validation risk. |

## Concept Inventory

Accepted concepts:

- Roofline diagnosis: attainable performance is `min(peak, bandwidth * arithmetic_intensity)`.
- Hardware balance and ridge point: the same workload can move between regimes across targets.
- Memory hierarchy and data movement: HBM, LPDDR, SRAM, cache, host transfer, fusion, and tiling determine practical latency and energy.
- Tensor core and precision contract: low precision helps only when the hardware supports the format, shape alignment, and quality envelope.
- Deployment fit: recommendation requires simultaneous checks for speed, memory, power, cost, and validation evidence.

Rejected or deferred concepts:

- Multi-chip collective scaling: important but belongs to distributed training and systems-scale labs.
- Full compiler pipeline internals: referenced through fusion and kernel selection, but not a separate concept module.
- Wafer-scale and ASIC history: useful context, not a student decision in this lab.
- Vendor portability strategy: kept as residual risk in synthesis, not a standalone module.
- Detailed cycle-accurate microarchitecture: outside MLSysIM first-order estimation scope.

## Module Table

| UI part | Concept module | Student experience | Evidence artifact |
|---|---|---|---|
| Part A | Roofline separates compute-bound from memory-bound regimes. | Predict why peak is not reached, manipulate matrix size and precision, find the ridge boundary. | Roofline chart, prediction-vs-actual regime, boundary dimension, checkpoint action. |
| Part B | Memory hierarchy and data movement can dominate accelerator performance. | Predict whether fusion/tiling helps, manipulate execution mode and local workspace, observe traffic and spill. | Memory traffic chart, local-buffer table, reversible spill/failure callout, movement decision. |
| Part C | Tensor cores and precision accelerate only supported shapes and tolerable numeric formats. | Predict whether lower precision always wins, manipulate precision and shape alignment, inspect fast path and quality gate. | Precision-path comparison table, alignment/quality failure state, tensor-contract decision. |
| Part D | Accelerator fit is a deployment recommendation under cost/power/validation constraints. | Predict whether highest peak is best, choose a candidate accelerator path and validation level, compare constraints. | Deployment candidate table, pass/fail recommendation, rejected alternatives, residual risk. |
| Synthesis | The chapter invariant across concepts. | Generate a memo that carries Part A-D evidence into a hardware acceleration recommendation. | Design Ledger entry and downloadable report. |

## Concept Modules

### Part A: Concept Module - Roofline Separates Regimes

Chapter claim:
- The roofline model exposes whether a workload is bounded by compute throughput or memory bandwidth.

Track lens:
- The selected track supplies the stakeholder, hardware profile, default GEMM size, precision, and amount-system threshold.
- iPhone asks whether the Neural Engine is actually useful under local latency and thermal limits.
- Oura Ring asks whether the tiny MCU/DSP-like path is compute-starved or bandwidth-constrained inside a duty-cycle budget.
- RoboTaxi asks whether the edge accelerator has deterministic p99 headroom.
- Cloud Fleet asks whether H100 utilization and cost/request justify the chosen path.

Student prior:
- Peak TOPS or TFLOP/s predicts runtime.

Activity beats:

1. Scenario: the stakeholder reports low utilization on the selected accelerator.
2. Prediction: student chooses compute-bound, memory-bound, unsupported path, or thermal/power as the likely first explanation.
3. Manipulation: matrix dimension and precision controls move the point along the roofline.
4. Evidence: roofline chart and exact table report AI, ridge point, attainable throughput, MFU, and regime.
5. Consequence: a boundary callout states the first matrix size/precision that crosses the ridge and the track-specific amount at risk.
6. Math/source: Math Peek shows `AI = FLOPs / bytes`, `ridge = peak / bandwidth`, and `R_attainable = min(peak, BW * AI)` using `mlsysbook_labs.roofline` and MLSysIM hardware refs.
7. Checkpoint: student chooses the first engineering action: increase reuse/batch, reduce bytes, switch precision, or reject the accelerator path.

Depth gate:
- Activity count: 7.
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: ridge crossing.
- Math Peek: yes.
- Track-specific consequence: yes.

Ledger fields:
- `part_a_prediction`, `workload_ai`, `ridge_flop_per_byte`, `roofline_regime`, `mfu_pct`, `ridge_boundary_dimension`, `part_a_action`.

### Part B: Concept Module - Memory Movement Dominates

Chapter claim:
- Memory hierarchy and host/device movement can dominate performance even when arithmetic is cheap.

Track lens:
- iPhone emphasizes LPDDR/unified-memory traffic, local latency, energy, and thermal pressure.
- Oura Ring emphasizes SRAM fit, flash/firmware memory, and duty-cycle energy.
- RoboTaxi emphasizes memory traffic causing p99 jitter and safety fallback.
- Cloud Fleet emphasizes HBM bandwidth, utilization, cost, and carbon waste.

Student prior:
- Reducing FLOPs is the main route to faster inference.

Activity beats:

1. Scenario: the same model spends time in elementwise and movement-heavy kernels.
2. Prediction: student estimates whether fusion/tiling gives minor, 2x, 3-5x, or 10x speedup.
3. Manipulation: execution mode, batch/window count, and local workspace controls change traffic and spill status.
4. Evidence: bar chart compares eager versus fused traffic/latency; table reports selected bytes, movement time, local workspace, local buffer, and spill.
5. Consequence: reversible failure state appears when local workspace exceeds the track buffer or movement time exceeds the track budget.
6. Math/source: Math Peek ties fusion to eliminating HBM/LPDDR/SRAM round-trips and the chapter memory hierarchy.
7. Checkpoint: student chooses the memory tactic: fuse kernels, tile into local memory, lower precision, shrink window, or reject the path.

Depth gate:
- Activity count: 7.
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: local-buffer spill or movement-budget miss.
- Math Peek: yes.
- Track-specific consequence: yes.

Ledger fields:
- `part_b_prediction`, `memory_mode`, `memory_batch`, `local_workspace_kb`, `local_buffer_kb`, `memory_spill`, `movement_time_us`, `part_b_action`.

### Part C: Concept Module - Tensor Core And Precision Contracts

Chapter claim:
- Tensor cores and reduced precision help only when the workload presents supported numeric formats, supported shapes, and acceptable quality loss.

Track lens:
- iPhone checks NPU/GPU/CPU fast-path support, shape alignment, and local model quality.
- Oura Ring checks int8-only MCU/DSP-like kernels, SRAM shape packing, and biosignal false-alert tolerance.
- RoboTaxi checks int8/FP16 edge acceleration against rare-event recall and safety validation.
- Cloud Fleet checks H100 tensor-core precision and alignment against quality/SLA.

Student prior:
- Lower precision is always faster and always acceptable.

Activity beats:

1. Scenario: the team tries to enable a faster numeric path.
2. Prediction: student chooses why low precision may fail: unsupported format, misaligned shape, quality loss, or no failure.
3. Manipulation: precision and dimension controls change format, shape alignment, and arithmetic intensity.
4. Evidence: precision table compares FP32, FP16, and INT8 effective throughput/latency plus fast-path status.
5. Consequence: failure callout distinguishes unsupported precision, shape-padding fallback, and quality-gate rejection.
6. Math/source: Math Peek connects precision to bytes moved, tensor-core peak, alignment multiples, and validation tolerance.
7. Checkpoint: student chooses the shipped precision path and required validation note.

Depth gate:
- Activity count: 7.
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: unsupported format, unaligned shape, or quality fail.
- Math Peek: yes.
- Track-specific consequence: yes.

Ledger fields:
- `part_c_prediction`, `precision_choice`, `shape_dimension`, `shape_aligned`, `precision_supported`, `quality_delta_pct`, `quality_tolerance_pct`, `precision_fast_path`, `part_c_action`.

### Part D: Concept Module - Accelerator Fit Is A Deployment Recommendation

Chapter claim:
- Hardware acceleration is a recommendation under deployment constraints, not a benchmark number.

Track lens:
- iPhone compares NPU, GPU, and CPU fallback against local latency, energy, thermal, privacy, and validation.
- Oura Ring compares MCU/DSP-like int8, scalar MCU, and phone offload against duty cycle, memory fit, radio cost, and comfort.
- RoboTaxi compares vehicle accelerator, GPU fallback, and cloud fallback against deterministic p99, power, safety margin, and validation.
- Cloud Fleet compares H100 tensor cores, older GPU pool, and CPU fleet against utilization, memory bandwidth, cost/request, SLA, and carbon.

Student prior:
- Choose the highest peak accelerator.

Activity beats:

1. Scenario: the stakeholder asks for a deployment sign-off.
2. Prediction: student chooses the constraint likely to reject the highest-peak path.
3. Manipulation: candidate accelerator path and validation evidence level controls.
4. Evidence: candidate table compares latency, power/energy/cost/carbon, validation level, pass/fail, and rejection reason.
5. Consequence: failure state names the violated constraint and numbers.
6. Math/source: Math Peek ties feasibility to `T_process = ops / throughput`, memory fit, power/cost, and validation coverage.
7. Checkpoint: student commits the recommendation, rejected alternatives, and residual risk to the memo.

Depth gate:
- Activity count: 7.
- Prediction: yes.
- Manipulation: yes.
- Failure/boundary: deployment constraint violation.
- Math Peek: yes.
- Track-specific consequence: yes.

Ledger fields:
- `part_d_prediction`, `accelerator_choice`, `validation_level`, `deployment_pass`, `deployment_rejection_reason`, `selected_precision`, `residual_risk`.

## Track Narratives And Amount Systems

| Track | Persona | Accelerator choices | Primary amounts | Failure mode | Memo framing |
|---|---|---|---|---|---|
| iPhone | Mobile accelerator engineer | Neural Engine, GPU shaders, CPU fallback | local p99 latency, mJ/inference, sustained watts, thermal headroom | NPU unsupported-op fallback, LPDDR traffic, thermal throttle, responsiveness miss | "Ship locally only if the fast path is supported and thermal/latency headroom remains." |
| Oura Ring | Wearable firmware engineer | MCU/DSP-like int8 path, scalar MCU, phone offload | SRAM KB, flash/OTA KB, wake ms, uJ/window, duty cycle | SRAM spill, duty-cycle miss, battery regression, radio/privacy penalty | "Run on-ring only if tiles fit SRAM and energy stays inside the nightly budget." |
| RoboTaxi | Autonomous vehicle platform engineer | Vehicle accelerator, GPU fallback, cloud fallback | deterministic p99 ms, safety margin, watts, sensor-burst headroom | p99 deadline miss, safety recall risk, power envelope violation, nondeterministic network path | "Approve only with deterministic edge evidence and explicit fallback risk." |
| Cloud Fleet | GPU performance engineer | H100 tensor cores, older GPU pool, CPU fleet | MFU, HBM bandwidth, p99 SLA, cost/request, gCO2e/request | low utilization, SLO breach, negative unit economics, carbon waste | "Buy/use accelerators only when utilization and cost/carbon beat alternatives under SLA." |

## Mechanics Plan

- Opening belt: header, invariant, reading map, track selector, track context.
- Prediction belt: one structured prediction per part, gated before evidence.
- Control belt: matrix dimension, precision, execution mode, local workspace, candidate path, validation level.
- Evidence belt: roofline plot, memory-traffic bar chart, precision comparison table, deployment candidate table.
- Failure belt: ridge boundary, memory spill/movement miss, precision/shape/quality failure, deployment constraint miss.
- Source belt: Math Peek accordions with formulas and source-trace text.
- Decision belt: checkpoint radio in every part plus synthesis memo.
- Ledger belt: `DesignLedger.save` and report export.

## Evidence And Ledger Plan

Evidence recorded by the report:

- Bottleneck diagnosis: AI, ridge, attainable throughput, MFU, regime, boundary dimension.
- Memory movement: eager/fused bytes, selected movement time, local buffer fit, spill status.
- Precision contract: selected precision, supported format, shape alignment, quality delta, fast-path status.
- Deployment recommendation: chosen accelerator, validation evidence level, pass/fail constraints, rejected alternatives, residual risk.

Design Ledger save criteria:

- Save only after all four structured predictions have values.
- Include current track and MLSysIM hardware/model refs.
- Include numeric evidence from every module, not only `completed: True`.
- Preserve downstream use for Lab 12 benchmarking: expected bottleneck, expected precision path, and validation risk.

## Implementation Risks

- Some track-specific amounts are lab scenario thresholds because the existing shared V1-11 registry stores hardware facts but not all deployment budgets; keep these notebook-local and label them as scenario assumptions.
- MLSysIM roofline helpers cover GEMM and fusion traffic, but not tensor-core contract or deployment candidate scoring; implement these as notebook-local `v1_11_` helpers.
- The Oura profile has extremely small peak throughput; use memory/duty-cycle evidence to avoid turning every concept into compute-only reasoning.
- Do not modify shared helpers, tests, or variants in this wave.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A Roofline separates regimes | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Part B Memory movement dominates | 3 | 3 | 3 | 3 | 3 | 2 | Yes |
| Part C Tensor/precision contract | 3 | 3 | 3 | 3 | 3 | 2 | Yes |
| Part D Deployment recommendation | 3 | 3 | 3 | 3 | 3 | 2 | Yes |
| Synthesis chapter invariant | 3 | 3 | 3 | 2 | 3 | 2 | Yes |

Rubric notes:

- No module has a dimension below 2.
- Every module includes prediction, manipulation, evidence, consequence, Math Peek/source model, and checkpoint.
- Reversible failure states appear in Parts A, B, C, and D.
- Synthesis ties all modules to the invariant: match workload arithmetic intensity, precision, memory hierarchy, and hardware capability before recommending acceleration.
