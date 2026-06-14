# V1-10 Track Plan: Model Compression Concept Module

Status: Wave 3 audit packet for `labs/vol1/lab_10_model_compress.py`.

Owned files:
- `labs/vol1/lab_10_model_compress.py`
- `labs/vol1/lab_10_model_compress.track-plan.md`

## Chapter Invariant

Compression helps only when it reduces the resource that actually constrains
deployment, and only when quality, runtime support, and hardware behavior
survive validation.

The lab implements one shared concept sequence for every track. Tracks change
persona, constraints, thresholds, failure mode, evidence emphasis, and report
framing, but not the concepts.

## Reading Map

| Module | Chapter anchor | Claim used in lab |
|---|---|---|
| Opening and Part A | `Purpose`, `Optimization Framework`, `Deployment Context`, `Efficiency Measurement` | Compression is deployment co-design; a smaller artifact is useful only when it relieves the deployed bottleneck and maps to the runtime. |
| Part B | `Pruning`, `Unstructured pruning`, `Structured pruning`, `Pruning trade-offs`, `Sparsity exploitation`, `Fallacies and Pitfalls` | Zeros do not become latency reduction unless sparsity structure matches executable kernels. |
| Part C | `Knowledge distillation`, `Distillation mathematics`, `Efficiency gains and trade-offs` | A dense student trades repeated inference cost for teacher quality, distillation data, student capacity, and validation burden. |
| Part D | `Technique Selection`, `Decision framework`, `Optimization Strategies`, `Measuring optimization effectiveness`, `Summary` | A release recipe must satisfy simultaneous constraints and cite validation evidence, not only a compression ratio. |
| Synthesis | `Summary` and `From benchmark winner to production model` | The durable rule is to name the binding resource, rejected method, quality guardrail, residual risk, and next hardware-evidence obligation. |

## Concept Inventory

Accepted concepts:

| Concept | Module | Why accepted |
|---|---|---|
| Smaller is not automatically faster | Part A | Directly attacks the main misconception and uses solver feasibility to show unsupported or non-binding compression failures. |
| Pruning has structure | Part B | Makes the chapter's unstructured-versus-structured warning physical through a reversible sparsity failure state. |
| Distillation trades teacher quality for deployable student constraints | Part C | Separates dense-student deployment from pruning and exposes teacher/student validation risk. |
| Compression strategy depends on binding constraint and validation evidence | Part D | Forces a release recipe across size, quality, speed, hardware support, calibration, validation, and residual risk. |
| Carry-forward compression contract | Synthesis | Saves fields later labs can reuse for hardware acceleration, serving, ops, and capstone replay. |

Rejected or deferred concepts:

| Concept | Disposition | Reason |
|---|---|---|
| Full distillation training procedure | Deferred | Requires a real training loop and belongs in TinyTorch/build work; this lab only models release risk. |
| NAS and architecture search | Deferred | Chapter-relevant but too broad for the compression-release consequence chain. |
| Low-rank factorization and tensor decomposition | Deferred | Useful but lower priority than pruning, distillation, and release validation for V1-10. |
| Standalone quantization cliff module | Folded into Part A/D evidence | The pilot packet had this as Part B, but the Wave 3 target sequence assigns Part B to pruning. Precision remains visible in solver candidates and the Part D recipe builder. |
| KV-cache compression and LLM serving memory | Deferred | Belongs in Volume II inference except for cloud-fleet context notes. |

## Concept Modules

### Part A: Smaller Is Not Automatically Faster

Chapter claim:
- Compression is not a bag of tricks. It works only when it reduces the resource that constrains the deployment and maps to the target runtime.

Track lens:
- iPhone: local latency, energy, thermal headroom, memory, and supported NPU path.
- Oura Ring: flash, SRAM, duty cycle, OTA payload, and signal/battery guardrails.
- RoboTaxi: safety floor, p99/p999 deadline, and rare-hazard recall.
- Cloud Fleet: throughput, cost/request, request SLA, utilization, and quality.

Student prior:
- Pick the highest compression ratio or default to INT8 without checking whether that resource is binding.

Activity beats:
1. Scenario: selected track stakeholder must ship a model under the track envelope.
2. Prediction: radio asks which single method is most likely to win for the track.
3. Manipulation: method selector inspects no compression, INT8, structured pruning, unstructured pruning, and distillation recommendation paths.
4. Evidence: `CompressionModel.sweep()` candidate table shows size, ratio, accuracy drop, speedup, hardware support, feasibility, and binding constraint.
5. Consequence: unsupported hardware, speedup miss, quality miss, or size overflow appears as a red failure row and failure card when selected.
6. Math Peek/source: compression ratio and feasibility conjunction are tied to `CompressionModel.sweep()`.
7. Checkpoint: card records predicted method, actual best feasible method, binding resource, and required validation.

Ledger output:
- `predicted_method`, `method_inspected`, `best_candidate_label`, `binding_resource`, `quality_guardrail`.

Depth gate:
- Scenario: pass.
- Structured prediction: pass.
- Manipulation: pass.
- Evidence: pass.
- Consequence/failure: pass.
- Math Peek/source: pass.
- Checkpoint: pass.

### Part B: Pruning Has Structure

Chapter claim:
- Unstructured pruning usually saves storage only; speedup requires structured sparsity or hardware-specific N:M kernels.

Track lens:
- iPhone: unstructured sparsity can fall off the mobile fast path and fail latency/energy expectations.
- Oura Ring: storage savings matter, but flash/SRAM and runtime support still gate release.
- RoboTaxi: p99 deadline and safety recall make unsupported sparse kernels unacceptable.
- Cloud Fleet: sparse formats can lose throughput or cost/request gains when kernels and batching do not exploit them.

Student prior:
- At 90 percent sparsity, expect about 10x speedup.

Activity beats:
1. Scenario: stakeholder asks whether pruning can rescue release.
2. Prediction: radio asks expected speedup at 90 percent sparsity.
3. Manipulation: sparsity slider and sparsity structure selector switch unstructured, structured, and N:M.
4. Evidence: solver-backed pruning table compares size, compression ratio, accuracy drop, speedup, hardware support, feasibility, and binding constraint.
5. Consequence: selected sparse recipe can enter a red failure state for quality, speedup, or hardware support.
6. Math Peek/source: pruning ratio and hardware branch are tied to `CompressionModel.candidate()`.
7. Checkpoint: card records predicted speedup, actual unstructured speedup, checkpoint choice, and validation risk.

Ledger output:
- `predicted_pruning_speedup`, `selected_sparsity_type`, `selected_sparsity_pct`, `sparsity_checkpoint`, `rejected_method`.

Depth gate:
- Scenario: pass.
- Structured prediction: pass.
- Manipulation: pass.
- Evidence: pass.
- Consequence/failure: pass.
- Math Peek/source: pass.
- Checkpoint: pass.

### Part C: Distillation Trades Teacher Quality For Student Constraints

Chapter claim:
- Distillation moves cost from repeated inference to teacher-student training and validation; a student is bounded by teacher quality, distillation data, and student capacity.

Track lens:
- iPhone: dense student may protect local responsiveness and energy if teacher quality survives on-device validation.
- Oura Ring: dense student may fit flash/SRAM and OTA, but signal-quality and battery tests decide release.
- RoboTaxi: dense student must preserve rare-event recall and p99/p999 safety replay evidence.
- Cloud Fleet: dense student can improve cost/request and throughput only if quality regression and load/SLA canary pass.

Student prior:
- Distillation is lossless compression because the student imitates the teacher.

Activity beats:
1. Scenario: stakeholder asks whether a smaller dense student can replace the current model.
2. Prediction: radio asks which distillation risk most likely blocks release.
3. Manipulation: teacher-quality dropdown, student-size slider, validation test selector, and distillation checkpoint.
4. Evidence: risk cards show student size, compression ratio, estimated quality drop, latency speedup, and pass/fail against the track guardrail.
5. Consequence: weak teacher, undersized student, or mismatched validation creates a reversible failure card.
6. Math Peek/source: student compression ratio and local risk model are shown; the card explicitly says `CompressionModel` does not model trained dense students.
7. Checkpoint: card records teacher quality, student constraint, checkpoint choice, and validation test.

Notebook-local scenario logic:
- `v1_10_evaluate_distillation()` is notebook-local because `CompressionModel` supports shrink-in-place quantization and pruning candidates but does not train or evaluate a new dense student architecture.
- The local logic is intentionally limited to release-risk bookkeeping: teacher penalty plus student-capacity penalty versus the track's `max_accuracy_drop`.
- The plan requires the report and Math Peek to label this as a risk card, not solver physics.

Ledger output:
- `predicted_distillation_risk`, `distillation_teacher_quality`, `distillation_student_scale_pct`, `distillation_checkpoint`, `validation_test`.

Depth gate:
- Scenario: pass.
- Structured prediction: pass.
- Manipulation: pass.
- Evidence: pass.
- Consequence/failure: pass.
- Math Peek/source: pass.
- Checkpoint: pass.

### Part D: Compression Strategy Depends On Binding Constraint And Evidence

Chapter claim:
- A deployable compression strategy is a recipe plus validation. A leaderboard candidate cannot ship unless all track guardrails pass.

Track lens:
- iPhone: recipe must include sustained-device benchmark, NPU fast-path verification, and thermal soak.
- Oura Ring: recipe must include flash/SRAM check, OTA payload test, and battery-life regression.
- RoboTaxi: recipe must include rare-event replay, p99 burst-latency test, and safety recall regression.
- Cloud Fleet: recipe must include load/SLA test, quality regression suite, and cost/request canary.

Student prior:
- Pick the frontier winner and ship it.

Activity beats:
1. Scenario: release review asks what recipe should ship.
2. Prediction: radio asks which recipe survives, with dominated, unsupported, aggressive, and best-feasible options.
3. Manipulation: recipe builder sets quantization, pruning, dense-student fallback, calibration/QAT, validation test, and residual risk.
4. Evidence: Pareto scatter and table show frontier, dominated, feasible, and infeasible candidates.
5. Consequence: release-gate card names exact failed guardrails; controls can recover the failure.
6. Math Peek/source: multi-constraint release condition and dominance condition are tied to `CompressionModel.sweep()` and candidate checks.
7. Checkpoint: card records selected recipe, rejected method, quality guardrail, and carry-forward implication.

Ledger output:
- `selected_recipe`, `recipe_quantization`, `recipe_pruning`, `recipe_distillation`, `recipe_calibration`, `selected_precision`, `binding_constraint`, `release_ok`, `primary_metric_gain`, `residual_risk`.

Depth gate:
- Scenario: pass.
- Structured prediction: pass.
- Manipulation: pass.
- Evidence: pass.
- Consequence/failure: pass.
- Math Peek/source: pass.
- Checkpoint: pass.

### Synthesis: Compression Recipe

Required synthesis artifact:
- Binding resource.
- Rejected method and reason.
- Quality guardrail.
- Selected recipe.
- Validation test.
- Residual risk.
- Carry-forward implication for Lab 11 hardware acceleration/roofline.

Ledger fields added or audited:
- `track_id`
- `scenario_id`
- `binding_resource`
- `quality_guardrail`
- `predicted_method`
- `method_inspected`
- `best_candidate_label`
- `rejected_method`
- `rejected_method_reason`
- `selected_precision`
- `lowest_feasible_bit_width`
- `predicted_pruning_speedup`
- `selected_sparsity_type`
- `selected_sparsity_pct`
- `sparsity_checkpoint`
- `predicted_distillation_risk`
- `distillation_teacher_quality`
- `distillation_student_scale_pct`
- `distillation_checkpoint`
- `selected_recipe`
- `recipe_quantization`
- `recipe_pruning`
- `recipe_distillation`
- `recipe_calibration`
- `binding_constraint`
- `validation_test`
- `residual_risk`
- `release_ok`
- `primary_metric_gain`
- `carry_forward_implication`
- `compression_candidates`

## Track Narratives

| Track | Persona | Binding constraints | Failure mode | Evidence emphasis | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile product lead | local latency, energy, thermal, memory, supported acceleration | unsupported fast path, thermal/battery headroom miss, p99 UX miss | feasibility table, fast-path support, sustained-device validation | local feature can ship only if acceleration and sustained UX survive |
| Oura Ring | Wearable firmware lead | flash, SRAM, duty cycle, OTA payload, battery, signal quality | flash/SRAM overflow, duty-cycle or battery regression, signal-quality loss | size limit from flash, solver feasibility, OTA/battery validation | tiny firmware recipe must fit the whole envelope, not only model weights |
| RoboTaxi | Safety/perception lead | p99/p999 deadline, rare-event recall, deterministic local perception | safety recall floor or tail-latency miss | p99/safety validation, quality guardrail, unsupported sparse path | no compression ships without rare-event replay and deadline evidence |
| Cloud Fleet | Infrastructure lead | throughput, cost/request, request SLA, utilization, quality | SLA breach, quality regression, negative cost/request ROI | Pareto frontier, cost/throughput framing, load/SLA canary | infrastructure win means request economics improve without quality/SLA debt |

## Mechanics And Evidence Plan

| Belt | Notebook mechanism | Evidence produced |
|---|---|---|
| Opening | `track_selector`, `track_context`, `scenario_brief`, lab map | Track-specific stakeholder, objective, primary metric, guardrail, validation tests |
| Prediction | `mo.ui.radio` in A/B/C/D | Structured pre-reveal commitments |
| Manipulation | method dropdown, sparsity slider/type dropdown, teacher-quality dropdown, student-size slider, recipe builder | Boundary-finding and diagnosis controls |
| Evidence | candidate tables, metric cards, Pareto scatter, release gate | Solver-backed candidate rows and local distillation risk card |
| Failure | `v1_10_failure_card` | Reversible failures with numeric value versus guardrail when controls violate quality, speed, support, or validation |
| Source | Math Peek accordions | Formula/source model for compression ratio, pruning branch, distillation risk, and release feasibility |
| Decision | checkpoint cards | Part-level decisions and carry-forward facts |
| Ledger | `DesignLedger.save` | Future-usable compression recipe and validation evidence |

Reversible failure states:
- Part A: inspect unsupported INT8 or sparse path on tracks without fast-path support.
- Part B: choose unstructured 90 percent sparsity or overly aggressive structured sparsity.
- Part C: select brittle/weak teacher or undersized student under a strict quality guardrail.
- Part D: choose no compression, sub-8-bit without QAT, unsupported pruning, or missing quality validation for distillation.

Table fallback:
- Candidate tables accompany frontier and pruning evidence, so students can read exact values without relying only on color or scatter positions.

## Data And Solver Plan

Shared sources:
- Track, persona, hardware, model, defaults, validation tests: `get_lab_track_variant("v1_10_compression_paradox", track_id)`.
- Model and hardware references: `resolve_mlsysim_ref()`.
- Candidate rows, feasibility, Pareto status, source trace: `CompressionModel.candidate()` and `CompressionModel.sweep()`.

Notebook-local helpers:
- `v1_10_size_limit_for()` maps variant `size_limit_ref` to hardware memory, flash, or storage.
- `v1_10_candidate_to_row()` adapts solver candidates to display/report rows.
- `v1_10_evaluate_recipe()` composes selected quantization/pruning anchors plus calibration/validation checks.
- `v1_10_evaluate_distillation()` is local for dense-student risk only, because the solver does not train or validate a new student model.
- `v1_10_checkpoint_card()` is presentation-only and exists to satisfy explicit checkpoint beats.

This remains inside notebook ownership because shared helper or solver changes are outside this worker's allowed files.

## Depth Audit

| Module | Scenario | Prediction | Manipulation | Evidence | Consequence/failure | Math/source | Checkpoint | Status |
|---|---|---|---|---|---|---|---|---|
| A | yes | method radio | method selector | solver sweep table/cards | infeasible method failure card | ratio and feasibility formula | A checkpoint card | pass |
| B | yes | sparsity speedup radio | sparsity slider/type/checkpoint | pruning candidate table/cards | sparse recipe failure card | pruning ratio and hardware branch | B checkpoint card | pass |
| C | yes | distillation risk radio | teacher, student, validation, checkpoint controls | dense-student risk cards | dense-student failure card | local risk formula and source note | C checkpoint card | pass |
| D | yes | release recipe radio | recipe builder and validation controls | Pareto scatter/table/release gate | release gate failure card | release and dominance formulas | D checkpoint card | pass |
| Synthesis | yes | n/a | final recipe state | report and ledger | rejected method and residual risk | carry-forward source trace | ledger/report save | pass |

Rubric score:
- Concept clarity: 3 for all modules.
- Activity depth: 3 for all modules.
- Track specificity: 3 for all modules via persona, constraints, failure, report/ledger framing.
- Mechanics fit: 3 for A/B/D, 2 for C because dense-student evidence is local risk modeling rather than solver physics.
- Evidence quality: 3 for A/B/D, 2 for C with explicit source limitation.
- Traceability: 3 for A/B/D, 2 for C because the local distillation constants are scenario assumptions documented here and in Math Peek.

Minimum acceptance:
- No dimension below 2.
- At least three dimensions at 3.
- At least one reversible failure state exists; this lab has four.
- Synthesis ties back to the chapter invariant and saves future-usable ledger fields.

## Implementation Risks

| Risk | Status | Mitigation |
|---|---|---|
| `CompressionModel` does not represent trained dense students | accepted | Part C labels the local calculation as risk evidence only and requires validation. |
| Oura/RoboTaxi hardware facts may be estimates | accepted | Track variants and MLSysIM hardware provenance own those assumptions; notebook does not create new registry facts. |
| Precision cliff no longer has its own module | accepted | Precision remains in solver candidates and Part D recipe controls; Wave 3 concept sequence prioritizes pruning and distillation. |
| Other lab files changing in parallel | active | This worker edits only the two owned V1-10 files and does not touch shared helpers/tests. |

## Completion Definition

Valid student completion requires:
1. Part A method prediction.
2. Part B sparsity speedup prediction and pruning checkpoint.
3. Part C distillation risk prediction and distillation checkpoint.
4. Part D release recipe prediction and recipe controls.
5. Exported report with selected recipe, rejected method, binding resource, quality guardrail, validation test, residual risk, and carry-forward implication.
