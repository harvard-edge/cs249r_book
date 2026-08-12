# V1-05 Track Plan: Neural Computation Amounts

## Chapter Invariant

Neural computation turns tensors into bounded amounts of operations,
activations, memory traffic, and energy. The lab must make those amounts
visible before students choose an operator design.

## Shared Concept Sequence Contract

The lab has one shared concept sequence for every learner:

1. Part A: tensor shape growth changes activation and operation amounts.
2. Part B: memory and activation budgets can bind before arithmetic does.
3. Part C: batch and precision choices trade throughput, memory, latency, and energy.
4. Part D: compute-vs-memory diagnosis determines the right optimization.
5. Synthesis: the student records an operator budget note with a binding amount
   and V1-06 architecture implication.

The selected track realizes those same concepts differently. It changes the
persona, constraints, thresholds, evidence emphasis, failure mode, and report
framing; it does not introduce a different concept path.

## Reading Map

| Lab module | Chapter anchor | Claim used by the lab |
|---|---|---|
| Opening | Purpose; From Logic to Arithmetic | Matrix dimensions determine operations and bytes moved, and model math becomes a physical contract with hardware. |
| Part A | Forward pass computation; Matrix multiplication formulation | Tensor shape growth changes activation storage and operation counts before code changes are visible. |
| Part B | Memory: training vs. inference; Quick estimation for ML engineers | Activation and memory budgets can bind before arithmetic throughput does. |
| Part C | Batch Processing footnote; Precision trades against power | Batch and precision are system knobs, not only statistical or numerical choices. |
| Part D | Arithmetic intensity of matrix multiply vs. element-wise work; Memory wall | Compute-bound and memory-bound diagnoses imply different optimizations. |
| Synthesis | D-A-M taxonomy; chapter summary | The durable artifact is an operator budget note with a binding amount and an architecture implication for the next lab. |

## Concept Inventory

### Accepted Concepts

| Concept | Why it belongs | Module |
|---|---|---|
| Tensor shape growth changes amounts | It is the first place students see tensor dimensions become activation memory, MACs, and bytes moved. | Part A |
| Memory can bind before arithmetic | It corrects the common belief that FLOPs alone decide feasibility. | Part B |
| Batch and precision trade several amounts | It forces students to compare throughput, memory, latency, and energy together. | Part C |
| Bottleneck diagnosis determines optimization | It teaches that the right fix depends on the binding amount, not on a generic desire for speed. | Part D |
| Operator budget note | It turns measured amounts into a reusable design record. | Synthesis |

### Rejected Or Deferred Concepts

| Candidate | Reason deferred |
|---|---|
| Full backpropagation derivation | Important chapter concept, but it belongs in model training labs where gradients, optimizer state, and update rules can be explored directly. |
| Activation-function numerical stability | Useful, but V1-05 is the operator budget lab; stability and function choice can appear as Math Peek context and residual risk. |
| Complete architecture search | Deferred to V1-06 so this lab can stop at operator-level evidence and carry a clean architecture implication forward. |
| Framework/kernel implementation details | Deferred to framework and acceleration labs; here the student reasons from amounts before tools. |
| Production threshold routing | Useful for inference pipelines, but it would distract from operation, activation, memory-traffic, and energy amounts. |

## Track Narratives

Tracks are not skins. The same operator ledger creates different failure modes
because each track has a different bounded amount.

| Track | Stakeholder | Binding amounts | Natural failure | Report prompt |
|---|---|---|---|---|
| iPhone | Mobile product engineer | Local inference latency, activation memory, DRAM/NPU bandwidth, power and thermal headroom | Smooth local inference turns into thermal throttling or visible latency | Name the activation or bandwidth reduction that protects local responsiveness. |
| Oura Ring | Wearable firmware engineer | SRAM, flash, wake time, microjoules per sensing window | The activation window overflows SRAM or extends the wake period | Name the streaming/tiled operator budget that fits all-night sensing. |
| RoboTaxi | Autonomous vehicle platform engineer | Real-time frame deadline, sensor pipeline headroom, p99 activation bandwidth, power | Perception misses the frame deadline during bursty sensor input | Name the bounded-latency operator choice and the remaining safety margin. |
| Cloud Fleet | Fleet service owner | Throughput, accelerator memory, utilization, cost/request, p99 latency | Batch improves average throughput while activation state or p99 latency breaks the service | Name the batch/precision/kernel budget that improves utilization without hiding cost. |

Each track changes persona, input assumptions, metric priorities, failure
threshold, report prompt, and carry-forward implication.

## Concept Modules

### Part A: Concept Module - Tensor Shape Growth Changes Amounts

Chapter claim:
- The layer expression `Z = A W + b` turns tensor dimensions into activation
  shapes, weight bytes, and MAC counts.

Student prior:
- Parameter count or model name is enough to predict feasibility.

Storyline:
1. Scenario: the selected track stakeholder has to approve one operator block.
2. Prediction: student predicts whether weights, activations, operations, or
   bytes moved dominate.
3. Manipulation: student changes the shape multiplier for the track operator.
4. Evidence: the notebook shows a shape-growth chart and an operation ledger
   table for weights, activations, GMACs, bytes moved, intensity, and estimated
   latency.
5. Consequence: the largest amount is named in track language.
6. Math Peek/source model: activations scale with tensor elements and bytes per
   element; operations scale with the relevant matrix/tensor dimensions.
7. Checkpoint: student chooses which amount should be reduced first.

Mechanics:
- Controls: dominant-resource prediction, shape multiplier, checkpoint radio.
- Graphs: activation and operation growth lines; ledger bar/table.
- Failure/boundary: none required in Part A; it calibrates the amount model.
- Evidence: prediction-vs-actual dominant resource and current ledger values.

Ledger output:
- `resource_prediction`, `dominant_resource`, `shape_multiplier`,
  `activation_memory_mb`, `ops_gmac`, `bytes_moved_mb`.

### Part B: Concept Module - Memory And Activations Can Bind First

Chapter claim:
- Activations scale with batch and layer widths, and memory budgets can fail
  before arithmetic throughput is exhausted.

Student prior:
- If the accelerator has enough TOPS/FLOPs, the operator is feasible.

Storyline:
1. Scenario: the stakeholder asks how far the shape can grow before the
   deployment envelope fails.
2. Prediction: student predicts where the activation cliff appears.
3. Manipulation: student sweeps the same shape multiplier.
4. Evidence: the notebook draws the activation curve against the track memory
   budget and shows normalized activation, bandwidth, latency, and power ratios.
5. Consequence: the current design is labeled feasible or failed with the exact
   violated amount and mitigation path.
6. Math Peek/source model: `activation_MB = tensor_elements x bytes_per_value`
   and feasibility is the maximum normalized budget ratio.
7. Checkpoint: student chooses the largest defensible shape policy.

Mechanics:
- Controls: cliff prediction, shared shape slider, memory policy checkpoint.
- Graphs: threshold line chart, normalized budget bar, table fallback.
- Failure/boundary: reversible activation, bandwidth, latency, or power failure
  driven by the shape slider.
- Evidence: first threshold multiplier and current violation list.

Ledger output:
- `memory_cliff_multiplier`, `binding_budget_ratio`,
  `current_feasible`, `violations`.

### Part C: Concept Module - Batch And Precision Trade Several Amounts

Chapter claim:
- Batch size is a hardware-memory decision, and precision trades accuracy,
  energy, memory traffic, latency, and throughput.

Student prior:
- Batching always helps, and lower precision is automatically better.

Storyline:
1. Scenario: the stakeholder wants more useful work per wake, frame, request, or
   accelerator slot without breaking the track envelope.
2. Prediction: student predicts which batch/precision strategy survives.
3. Manipulation: student changes batch/window policy and precision policy.
4. Evidence: the notebook compares activation memory, latency, bandwidth,
   throughput, and energy for candidate strategies.
5. Consequence: a selected strategy either fits or exposes the amount it breaks.
6. Math Peek/source model: batch multiplies activation elements; precision
   changes bytes per value and energy per operation.
7. Checkpoint: student records the batch/precision policy to carry forward.

Mechanics:
- Controls: batch/window policy dropdown, precision policy dropdown, checkpoint
  radio.
- Graphs: latency-vs-memory scatter with feasibility status; table fallback.
- Failure/boundary: selected strategy can exceed memory, latency, bandwidth, or
  power.
- Evidence: selected strategy row, binding amount, throughput, energy/window or
  energy/request.

Ledger output:
- `batch_policy`, `precision_policy`, `throughput_per_s`,
  `energy_per_inference_j`, `batch_precision_binding`.

### Part D: Concept Module - Diagnosis Determines Optimization

Chapter claim:
- Arithmetic intensity and the memory wall determine whether compute reduction
  or memory/data movement reduction is the right optimization.

Student prior:
- The best optimization is always the one with the fastest advertised kernel or
  the smallest model.

Storyline:
1. Scenario: the stakeholder asks which operator optimization to implement.
2. Prediction: student predicts compute-bound versus memory-bound behavior.
3. Manipulation: student selects an operator design option.
4. Evidence: the notebook compares current arithmetic intensity with the
   hardware crossover and lists all design candidates.
5. Consequence: a mismatch between diagnosis and selected design is called out.
6. Math Peek/source model: `roofline_crossover = peak_ops / memory_bandwidth`;
   below it, bytes moved dominate.
7. Checkpoint: student chooses the optimization family and residual risk.

Mechanics:
- Controls: diagnosis prediction, operator design dropdown, optimization
  checkpoint radio.
- Graphs: roofline-style intensity marker and design candidate table.
- Failure/boundary: selected design may remain infeasible or target the wrong
  amount.
- Evidence: compute-vs-memory diagnosis, design feasibility, quality risk, and
  residual risk.

Ledger output:
- `diagnosis_prediction`, `diagnosed_wall`, `operator_design`,
  `optimization_family`, `quality_risk`, `residual_risk`.

### Synthesis: Concept Module - Operator Budget Note

Chapter claim:
- Neural-computation reasoning is useful only when it becomes a decision record
  that names the binding amount and the next architecture implication.

Storyline:
1. Student reviews the track, measured binding amount, selected design, and
   residual risk.
2. Student chooses a final budget decision.
3. Student writes a short operator budget note.
4. Notebook saves the decision to the Design Ledger.
5. Report export includes the amount evidence and source trace.

Mechanics:
- Controls: final decision radio and operator budget text area.
- Evidence: decision record card, ledger HUD, report export.
- Failure/boundary: incomplete report if required predictions or budget note are
  missing.

Ledger output:
- selected track, scenario, resource prediction, dominant amount, shape,
  memory cliff, batch/precision policy, diagnosis, selected design, binding
  amount, residual risk, next-lab implication.

## Mechanics And Evidence Plan

| Module | Student action | Marimo mechanics | Evidence produced |
|---|---|---|---|
| Part A | Predict and manipulate tensor shape | radio, slider, line chart, ledger table | prediction-vs-actual dominant amount and shape growth evidence |
| Part B | Find a boundary | radio, shared slider, threshold chart, normalized budget bar | first activation cliff and exact current violation or headroom |
| Part C | Compare trade-offs | dropdowns, scatter, strategy table | selected batch/precision row with memory, latency, throughput, and energy |
| Part D | Diagnose bottleneck | radio, design dropdown, roofline marker, design table | compute-vs-memory diagnosis and recommended optimization family |
| Synthesis | Save decision | radio, text area, Design Ledger, report export | operator budget note and JSON/report snapshot |

Accessibility and fallback:
- Every plot has a table with exact values.
- Feasibility is named with pass/fail text, not color alone.
- Boundary messages include value, limit, unit, and mitigation.
- Prediction widgets are structured controls; text area is only for final note.

## Data And Solver Contracts

Existing shared helpers remain the source of truth:
- `neural_compute_profile()`
- `operation_ledger()`
- `memory_cliff()`
- `operator_design()`

Notebook-local helpers are allowed only for presentation and derived diagnosis:
- table/field HTML rendering
- prediction check rendering
- track narrative copy
- normalized budget ratios
- batch/precision candidate rows built from the track profile and
  `operation_ledger()`
- roofline-style compute-vs-memory diagnosis

No shared helper, registry, test, implementation-note, or other lab file should
be edited for this wave.

## Implementation Risks

| Risk | Mitigation |
|---|---|
| Shared helper lacks explicit batch override | Use a notebook-local derived profile via standard dataclass replacement, then call `operation_ledger()` so equations stay shared. |
| Energy units differ by track | Express energy as `J` in the calculation and describe Oura as microjoules/window in text where appropriate. |
| Existing lab lacks tabbed Part D | Convert the current notebook body to `mo.ui.tabs` with Parts A-D plus Synthesis while preserving bootstrap, track selector, and Design Ledger save. |
| Other workers edit other lab files | Only edit `labs/vol1/lab_05_nn_compute.py` and this track plan. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 3 | Pass |

Minimum acceptance checks:
- No dimension is below 2.
- At least three dimensions score 3 in every module.
- Reversible failure states exist in Part B and Part C.
- Synthesis ties all modules back to the invariant and saves a carry-forward
  operator budget note.
