# V1-10 Track Plan: Compression Paradox

## Pilot Assessment

The first version of this plan had the right concept and track narrative, but it was not yet detailed enough to implement consistently across notebooks. It named "Pareto frontier" and "compression recipe" but did not specify the controls, plots, evidence objects, failure boundaries, source traces, or report fields that realize those ideas.

This file is the pilot format for the rest of the per-lab plans. It explicitly connects:

- the shared pedagogy,
- the four canonical track variants,
- the modality stack from `labs/LAB_REALIZATION_MODALITY_CATALOG.md`,
- the solver/data contracts needed from MLSysIM or `mlsysbook_labs`,
- the exact ledger/report artifacts students should produce.

## Purpose

This lab teaches that compression is not automatically better. Quantization, pruning, and distillation affect size, latency, energy, quality, robustness, and hardware support differently by track.

## Shared Pedagogy

- Students predict which compression method gives the best deployment improvement.
- They sweep compression choices and inspect Pareto frontiers.
- They choose a compression recipe and validation test.
- They learn that compression is a deployment-specific systems decision, not a generic model-size reduction trick.

## Track Profiles Used

| Track | Category | Hardware source | Compression question |
|---|---|---|---|
| iPhone | Mobile ML | `Hardware.Mobile.iPhone15Pro` | Can compression reduce sustained battery/thermal load while staying on the supported NPU path? |
| Oura Ring | TinyML / wearable | `Hardware.Tiny.OuraRing` | Can the model, runtime, and OTA payload fit inside flash/SRAM without ruining battery life? |
| RoboTaxi | Edge AI | `Hardware.Edge.RoboTaxi` | Can compression reduce p99 perception latency without losing rare-hazard recall? |
| Cloud Fleet | Cloud/Fleet | Cloud fleet profile backed initially by `Hardware.Cloud.H100` | Can compression reduce cost/request or raise throughput without violating quality and SLA? |

## Modality Stack

Use the following modalities from `LAB_REALIZATION_MODALITY_CATALOG.md`:

| Lab element | Modality | Purpose |
|---|---|---|
| Track choice | Track selector | Pick iPhone, Oura Ring, RoboTaxi, or Cloud Fleet and load defaults |
| Opening context | Chapter recap + scenario strip | Explain compression as a deployment trade-off |
| Prediction | Multiple-choice prediction lock | Commit to quantization, pruning, distillation, or no compression before evidence |
| Part A controls | Strategy selector | Apply one compression method at a time |
| Part A visual | Before/after comparison + constraint budget | Show feasibility movement and active constraint |
| Part B controls | Sliders for bit width, sparsity, student size | Explore the compression design space |
| Part B visual | Pareto frontier + table fallback | Show non-dominated choices and guardrail violations |
| Part C controls | Stack builder + guardrail checklist | Compose final recipe and validation plan |
| Evidence | Source trace / math peek | Tie size, latency, energy, and quality estimates to MLSysIM |
| Failure state | Failure boundary callout | Explain unsupported kernels, memory overflow, p99 violation, or quality cliff |
| Close | Decision card + report export | Save deployable recipe and residual risk |

## Lab Flow

### Opening - Compression Brief

Common narrative:
- The team must ship a model under deployment constraints.
- Smaller is useful only if the target hardware and validation guardrails support it.
- The same compression method can be an improvement on one track and a regression on another.

Track realization:
- iPhone: the product manager wants the feature to run continuously without heating the phone or draining battery. Unsupported ops fall back to CPU/GPU and erase the win.
- Oura Ring: the firmware team needs a model update small enough for OTA and a runtime footprint that leaves memory for sensing buffers.
- RoboTaxi: the safety lead wants lower p99 perception latency, but rare-hazard recall is the guardrail.
- Cloud Fleet: the infra lead wants lower cost/request and higher accelerator utilization, but SLA and quality cannot regress.

Opening scenario strip fields:
- `track_id`
- `hardware_ref`
- `model_ref`
- `baseline_model_size`
- `baseline_latency`
- `baseline_energy_or_cost`
- `primary_metric`
- `guardrail_metric`

### Part A - Compression Feasibility

Common pattern:
- Apply quantization, pruning, and distillation.
- Show size, latency, energy, quality, and support status.
- The prediction asks: "Which method gives the best deployment improvement on your selected track?"
- The reveal should show that the best answer depends on hardware support and guardrails.

Track realization:
- iPhone emphasizes INT8/mobile delegate support.
- Oura Ring emphasizes model plus runtime plus OTA payload.
- RoboTaxi emphasizes deterministic p99 and rare-event validation.
- Cloud Fleet emphasizes throughput and quality/cost frontier.

Required controls:
- Strategy selector: `No compression`, `INT8 quantization`, `structured pruning`, `unstructured pruning`, `distillation`.
- Optional advanced drawer: calibration set size, kernel support override, and validation depth.

Required visual:
- Before/after comparison table with columns: size, latency, energy/cost, quality, hardware support, feasibility.
- Constraint budget card showing the selected track's limiting resource.

Failure boundaries:
- iPhone: unsupported quantization path or thermal/battery headroom exhausted.
- Oura Ring: model/runtime/OTA exceeds flash or activation buffer exceeds SRAM.
- RoboTaxi: p99 exceeds safety deadline or rare-event recall drops below guardrail.
- Cloud Fleet: SLA violation, quality regression, or cost/request improvement too small.

### Part B - Compression Frontier

Common pattern:
- Sweep bit width, sparsity, and student model size.
- Highlight dominated and non-dominated configurations.
- Students should see that "smaller" creates a frontier, not a monotonic improvement.

Track realization:
- iPhone frontier axes: battery/thermal versus quality.
- Oura Ring frontier axes: flash/SRAM versus signal quality.
- RoboTaxi frontier axes: p99 latency versus rare-hazard recall.
- Cloud Fleet frontier axes: cost/request versus quality/SLO.

Required controls:
- Bit-width slider: e.g., 16, 8, 6, 4, 3, 2 bits.
- Sparsity slider: 0-95 percent, with structured/unstructured selector.
- Student-size slider or dropdown for distillation.

Required visual:
- Pareto frontier with selected candidate highlighted.
- Guardrail violation markers.
- Table fallback with exact values and feasibility status.

Recommended secondary visual:
- Phase diagram for bit width x sparsity showing feasible, quality-cliff, unsupported-kernel, and memory-fit regions.

### Part C - Compression Recipe

Common pattern:
- Student chooses validated compression strategy and names a failure test.
- The recipe must include both a selected candidate and a validation plan.

Track realization:
- iPhone recipe includes sustained-device benchmark.
- Oura Ring recipe includes memory/OTA and battery-life regression.
- RoboTaxi recipe includes rare-event safety validation.
- Cloud Fleet recipe includes load/SLA and quality regression tests.

Required controls:
- Stack builder for quantization, pruning, distillation, calibration, and fallback.
- Guardrail checklist: quality, latency, memory/storage, energy/cost, hardware support.
- Decision card fields: selected recipe, primary win, guardrail, validation test, residual risk.

Required visual:
- Budget stack or before/after comparison for the final recipe.
- Failure boundary if the recipe violates any guardrail.

### Synthesis - Deployment Recipe Review

Common pattern:
- Student exports a compression deployment recipe.
- The report should make clear why the selected recipe is track-specific.

Track realization:
- iPhone report: "This recipe is acceptable because it stays on supported mobile acceleration and protects sustained UX."
- Oura Ring report: "This recipe is acceptable because it fits flash/SRAM and preserves battery/signal guardrails."
- RoboTaxi report: "This recipe is acceptable because p99 improves without sacrificing rare-hazard recall."
- Cloud Fleet report: "This recipe is acceptable because it improves cost/throughput while preserving SLA and quality."

Required modalities:
- Source trace for the compression formulas and hardware values.
- Decision card.
- Markdown report export.

## Track Variant Detail

### iPhone

Stakeholder:
- Mobile product lead.

Primary question:
- Can INT8 or distillation reduce sustained energy/thermal pressure without CPU/GPU fallback?

Default model/workload:
- Mobile vision or multimodal on-device inference with repeated user-facing execution.

Primary metric:
- Battery or thermal headroom.

Guardrail metric:
- Quality and on-device p95/p99 latency.

Preferred visuals:
- Before/after battery/latency comparison.
- Pareto frontier: quality versus energy/latency.
- Constraint budget: memory, thermal power, battery.

### Oura Ring

Stakeholder:
- Wearable firmware lead.

Primary question:
- Can the compressed model, runtime, and OTA package fit without shortening battery life?

Default model/workload:
- Low-rate biosignal classifier or anomaly detector.

Primary metric:
- Flash/SRAM fit and OTA payload size.

Guardrail metric:
- Battery life and signal quality.

Preferred visuals:
- Memory/OTA constraint budget.
- Phase diagram: bit width x model size.
- Before/after battery-life estimate.

### RoboTaxi

Stakeholder:
- Safety/perception lead.

Primary question:
- Can compression reduce worst-case perception latency without suppressing rare hazards?

Default model/workload:
- Vehicle-local perception model under bursty sensor workload.

Primary metric:
- p99 or p999 latency.

Guardrail metric:
- Rare-event recall.

Preferred visuals:
- Pareto frontier: p99 latency versus rare-event recall.
- Failure boundary for safety deadline.
- Before/after latency stack.

### Cloud Fleet

Stakeholder:
- Infrastructure lead.

Primary question:
- Can compression improve cost/request or throughput without quality/SLA regressions?

Default model/workload:
- High-volume inference service.

Primary metric:
- Cost/request or throughput.

Guardrail metric:
- Quality and SLA.

Preferred visuals:
- Cost/SLA Pareto frontier.
- Utilization/throughput constraint budget.
- Table fallback sorted by cost/request.

## Instructor Assignment Modes

Default mode:
- Individual choice. Students use the track they selected in Lab 00 and submit one compression recipe for that track.

Alternative modes:
- Assigned team tracks. Instructor assigns teams to iPhone, Oura Ring, RoboTaxi, or Cloud Fleet and compares reports in discussion.
- Lecture demo. Instructor demonstrates Oura Ring and Cloud Fleet because they make the memory/cost contrast sharp.
- Capstone preparation. Students must preserve the same track and carry their selected compression recipe into later serving, ops, robustness, and synthesis labs.

Track lock:
- Implementation should allow instructors to lock a track through URL/query/config later, but the default should read from the ledger.

## Prerequisites And Book Anchors

Primary anchor:
- Volume 1, Chapter 10: Model Compression.

Useful prior labs:
- V1-05 Activation Tax for parameter, activation, and byte accounting.
- V1-06 Architecture Tax for architecture-specific resource signatures.
- V1-11 Hardware Roofline as a follow-up for whether compression moves the workload point.

Concepts to refresh:
- Quantization levels and calibration.
- Structured versus unstructured pruning.
- Distillation and student/teacher quality transfer.
- Hardware kernel support versus theoretical model sparsity.

## Completion Path

Required for a valid report:
- Select or load one canonical track.
- Make the Part A prediction.
- Compare at least three compression strategies.
- Select one Pareto candidate in Part B.
- Complete the Part C recipe and guardrail checklist.
- Export the report.

Optional exploration:
- Advanced calibration controls.
- Kernel support override.
- Full bit-width x sparsity phase diagram.
- Cross-track comparison after the student's own track is complete.

Minimum classroom demo:
- Show the track selector.
- Run Part A for Oura Ring and Cloud Fleet.
- Compare why the same compression method has different primary wins.
- End with one decision card.

## Expected Track Outcomes

These are teaching anchors, not hard-coded answer keys.

| Track | Likely good answer | Likely trap | Instructor discussion prompt |
|---|---|---|---|
| iPhone | INT8 or distillation if it stays on the supported mobile acceleration path | Picking unstructured pruning and assuming speedup without kernel support | "What evidence proves the model stayed on the NPU path?" |
| Oura Ring | Aggressive quantization plus small architecture/student model, validated against flash/SRAM and OTA size | Optimizing latency while ignoring flash/SRAM or battery life | "What memory component is not part of the neural network but still consumes the device budget?" |
| RoboTaxi | Structured compression or distillation only if p99 improves and rare-event recall stays above guardrail | Optimizing average latency or size while rare-hazard recall drops | "Which validation set would make this compression recipe unacceptable?" |
| Cloud Fleet | Compression that improves cost/request or throughput under SLA and quality guardrails | Assuming smaller always lowers serving cost, even with batching or kernel overhead | "What metric makes this an infrastructure win rather than just a model win?" |

## Common Misconceptions

- Smaller models always run faster.
- 90 percent sparsity means a 10x speedup.
- INT8 always preserves quality.
- Compression is only about model size.
- A compression recipe that works for Cloud Fleet should also work for Oura Ring.
- Hardware support is an implementation detail rather than a first-order systems constraint.

## Assumptions To Surface

Hardware assumptions:
- iPhone hardware facts come from `Hardware.Mobile.iPhone15Pro`.
- Oura Ring requires `Hardware.Tiny.OuraRing` before full implementation.
- RoboTaxi requires `Hardware.Edge.RoboTaxi` before full implementation.
- Cloud Fleet starts from a cloud fleet profile backed by `Hardware.Cloud.H100`, but one H100 is not the whole fleet model.

Scenario assumptions:
- Track-specific workload, SLA, battery target, OTA payload target, rare-event guardrail, and cost target should live in the lab track variant.
- Approximate or convention values must be labeled in the source trace and report.

Required source traces:
- Model size equation.
- Compression-size equation.
- Latency or energy/cost estimate.
- Hardware support warning source.

## Accessibility And Fallback Requirements

- Pareto frontier must have a table fallback with exact candidate values.
- Phase diagram must use labels or tooltips for regimes, not color alone.
- Failure boundaries must state value versus limit in text.
- Decision card and exported report must include the selected candidate and guardrail result.
- Advanced controls should be hidden by default and not required for completion.

## Single Source Of Truth Requirements

- Hardware facts must come from MLSysIM hardware registries.
- Model facts must come from MLSysIM model registries.
- Compression equations and reusable sweep/frontier logic should live in MLSysIM solvers or physics APIs.
- Track identity must come from the `mlsysbook_labs` track profile registry.
- Scenario thresholds, stakeholder text, and guardrails must live in typed lab variant metadata, not scattered notebook constants.
- Any new compression hardware support table should be a registry-backed capability source, not notebook-local conditionals.

## Data And Solver Contracts

Needed inputs:
- `track_id`
- `hardware_ref`
- `model_ref`
- `baseline_model_size`
- `baseline_latency`
- `baseline_quality`
- `compression_method`
- `bit_width`
- `sparsity`
- `student_model_scale`
- `kernel_support`
- `validation_guardrail`

Needed outputs:
- `compressed_size`
- `latency`
- `energy_or_cost`
- `quality`
- `hardware_supported`
- `binding_constraint`
- `feasible`
- `pareto_status`
- `guardrail_violations`

Preferred typed result:
- `CompressionCandidate`
- `CompressionSweepResult`
- `ParetoFrontier`
- `ConstraintBudget`
- `ValidationPlan`

## Implementation Requirements

- Hardware kernel support must be explicit in track/device metadata.
- Compression result schema should include unsupported fast-path warnings.
- Track variants should define guardrail metric and acceptable degradation.
- Notebook-local compression constants should be migrated into MLSysIM or a typed lab scenario registry.
- The current notebook uses iPhone, H100, and Jetson hardware directly; the refactor should route those through canonical track profiles and add Oura Ring and RoboTaxi once their hardware entries exist.

## Ledger And Report

Save:
- predicted best compression method
- selected Pareto point
- recipe and validation test
- guardrail metric
- residual robustness/quality risk
- hardware support warning, if any
- track-specific primary metric improvement
- final feasibility status

Report target:
- A compression deployment recipe for the selected track.

Minimum report sections:
- Track and scenario.
- Prediction and reveal.
- Candidate frontier.
- Final recipe.
- Guardrail validation.
- Residual risk.

## Rubric Sketch

| Criterion | Evidence in report |
|---|---|
| Prediction discipline | Student made a concrete pre-reveal prediction and compared it to the result |
| Evidence use | Student cites frontier, constraint budget, or before/after result |
| Track-specific reasoning | Student explains why the selected track changes the compression decision |
| Guardrail awareness | Student names quality, latency, memory, battery, safety, SLA, or cost guardrail as appropriate |
| Residual risk | Student states one assumption that could invalidate the recipe |

## Template Decision

This upgraded structure should be the model for the other `*.track-plan.md` files. The short plans are useful as coverage maps, but implementation should proceed only after each plan has the modality stack, assignment modes, expected outcomes, assumptions, accessibility requirements, and rubric sketch.

## Continuous Improvement Notes

- When implementation reveals a better modality, data contract, or track assumption, update this plan and `labs/LAB_IMPLEMENTATION_NOTES.md`.
- If any notebook-local constant is introduced during implementation, stop and decide whether it belongs in MLSysIM or typed lab variant metadata.
- If a track feels artificial for this lab, document the constrained interpretation rather than forcing fake behavior.
