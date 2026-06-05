# MLSysBook Trade-Off Lab Improvement Plan

This plan treats the Marimo labs as interactive trade-off environments for ML systems engineering. TinyTorch asks students to build, hardware kits ask students to deploy, and these labs ask students to reason through system choices. The goal is not to fit a fixed time budget. The goal is for each lab to expose the important knobs, make the trade-offs visible, and require a defensible engineering decision.

MLSysIM should remain the single source of truth for simulation, not the owner of the pedagogy. Each lab should call into MLSysIM for hardware, workload, model, scenario, solver, and trade-off computations. Lab notebooks and a separate MLSysBook lab helper package should contain presentation, controls, explanation, chapter mapping, track selection, and decision capture.

The consolidated implementation contract is in `LAB_SYSTEM_SPECIFICATION.md`. That file defines the lab wrapper, chapter recap, nugget, track, report, instructor adoption, MLSysIM, `mlsysbook_labs`, testing, and rollout requirements.

The activity-level blueprint is in `MLSYSIM_LAB_ACTIVITY_BLUEPRINT.md`. That companion file lists the core concepts, each planned nugget or activity, what it covers, how it is taught interactively, and what MLSysIM capability is needed.

The student experience, visual design, pedagogy, and MLSysIM readiness checklist are in `LAB_EXPERIENCE_AND_MLSYSIM_READINESS.md`.

The interaction-device catalog is in `LAB_INTERACTION_DEVICE_CATALOG.md`. It defines when to use sliders, selectors, graphs, maps, topology diagrams, reflection cards, decision cards, and other Marimo-supported devices.

The design-completeness checklist is in `LAB_DESIGN_COMPLETENESS_CHECKLIST.md`. It captures the remaining design dimensions: ML-to-systems translation, thematic nuggets, misconceptions, student artifacts, rubrics, instructor support, and open author decisions.

## Current Audit

All 33 labs now boot with the repaired MLSysIM infrastructure:

- Native Python tests pass for the labs package.
- Pyodide/WASM smoke tests pass.
- Representative browser smoke tests pass.
- The rebuilt MLSysIM wheel includes `mlsysim.labs`, fixing the browser import failure. This is a tactical compatibility fix, not the desired final package boundary.

The content audit shows that the labs are runnable but uneven as trade-off environments:

- Lab 00 has an actual track selector. Most later labs only read or display the selected track, or ignore it entirely.
- Many labs use `Hardware` and `Models` registries, but only a subset call `Engine.solve`.
- Several labs contain local formulas for latency, memory, energy, queueing, or cost that should move into MLSysIM or through MLSysIM helper APIs.
- Most labs are structured as four or five parts. The user-facing target should be a clearer Part A, Part B, Part C flow: baseline constraint, trade-off curve, engineering decision.
- The current index and template still describe time estimates and "Co-Labs"; those should be changed when implementation begins.
- Volume 2 chapter-to-lab mapping needs alignment because the book sequence includes both Network Fabrics and Collective Communication. The improved sequence should add a dedicated Collective Communication lab and renumber downstream labs as needed.
- Every current notebook imports `DesignLedger`, styles, and UI helpers from `mlsysim.labs`. That coupling should be migrated out of MLSysIM.

## Package Boundary

MLSysIM should not own book labs.

The clean boundary is:

- MLSysIM is a standalone ML systems simulator.
- MLSysBook labs are a pedagogical layer that uses MLSysIM.
- The design ledger, Marimo components, CSS, chapter metadata, track selector, student state, browser persistence, and instructor affordances belong to the labs layer, not to MLSysIM.

MLSysIM can still expose deployment-relevant simulation data:

- Hardware and accelerator registries.
- Model and workload registries.
- Scenario presets such as mobile device, microcontroller, edge gateway, and datacenter fleet.
- Solvers and result objects.
- Canonical formulas and bottleneck attribution.

But MLSysIM should not know about:

- Volume or chapter numbers.
- Lab parts.
- Student progress.
- The design ledger.
- Marimo.
- CSS or visual design tokens.
- "Co-Lab" or MLSysBook curriculum language.

Recommended final import boundary:

```python
from mlsysim import Engine, Hardware, Models, Scenarios
from mlsysbook_labs import (
    DesignLedger,
    LAB_CSS,
    apply_plotly_theme,
    track_selector,
    scenario_controls,
    tradeoff_poster,
    decision_card,
)
```

Recommended migration path:

- Create a local lab helper package, for example `labs/mlsysbook_labs`.
- Move `mlsysim/mlsysim/labs/state.py`, `style.py`, and `components.py` into that package.
- Update all notebooks to import from `mlsysbook_labs`.
- Keep MLSysIM focused on simulator APIs and data models.
- Remove `mlsysim.labs` from the MLSysIM wheel once all notebooks and tests are migrated.

During migration, a compatibility shim can exist only if needed to keep current deployed labs working. It should be clearly marked as deprecated and should not be expanded.

## Pedagogical Alignment To ML Systems

The labs should line up with the core habits of ML systems engineering, not with isolated app demos. Tracks are contexts that change constraints; they are not separate curricula.

Every lab should teach at least one of these ML systems ideas:

- Resource budgets: memory, compute, bandwidth, energy, storage, carbon, money, and human operational capacity.
- Bottleneck attribution: the reason a system fails should be explicit and computable.
- Trade-off frontiers: students should see dominated and non-dominated designs.
- Workload, model, data, and hardware co-design: the "right" answer depends on all four.
- Deployment constraints: local, edge, cloud, tiny, mobile, and fleet settings change the binding constraint.
- Feedback loops: data, drift, monitoring, retraining, and user behavior affect the system after deployment.
- Scale effects: behavior changes when there are many devices, many users, many models, many accelerators, or many failures.
- Responsible constraints: privacy, security, fairness, robustness, and carbon must be represented as engineering constraints, not as side notes.

For each chapter lab, the student should leave with a transferable systems principle:

- What knob did I move?
- What metric improved?
- What metric got worse?
- Which constraint became binding?
- What decision would I defend in a design review?

## Chapter Alignment Summary

This table summarizes the chapter-level alignment. It is based on the current book chapter tree under `book/quarto/contents` and the current Marimo labs under `labs/vol1` and `labs/vol2`.

### Volume 1

| Book chapter | ML systems focus | Current lab | Current state | Planned improvement |
|---|---|---|---|---|
| Orientation, not a book chapter | Choose a deployment context and establish student state | V1-00, The Architect's Portal | Useful track selector, but mostly standalone | Make it the reusable track/scenario source for all labs; move state/UI to `mlsysbook_labs` |
| Introduction | Data, algorithm/model, and machine as a coupled system | V1-01, The AI Triad | Strong fit; already uses `Engine.solve` | Make triad diagnosis track-aware; turn outputs into bottleneck attribution and intervention frontier |
| ML Systems | Physical limits, system constraints, and deployment feasibility | V1-02, The Physics of Deployment | Covers memory, power, energy, and light-speed walls with local formulas | Move wall calculations into MLSysIM; expose feasible/infeasible regions by track |
| ML Workflow | Constraint propagation through development and deployment | V1-03, The Constraint Tax | Good workflow framing; some track state but not central | Connect latency, memory, privacy, and validation constraints to iteration policy decisions |
| Data Engineering | Data movement, preprocessing, storage, and feeding accelerators | V1-04, The Data Gravity Trap | Good topic fit; formulas are notebook-local | Make data placement, cache, compression, and compute-placement trade-offs MLSysIM-backed |
| Neural Computation | Operators, activations, memory traffic, and compute scaling | V1-05, The Activation Tax | Strong concept but should be more systems-grounded | Build an operator cost ledger and activation-memory cliff driven by MLSysIM |
| Network Architectures | Architecture shapes and their scaling/resource signatures | V1-06, The Architecture Tax | Strong fit; already uses solver in places | Compare architecture signatures across tracks and make sequence/resolution/width scaling explicit |
| ML Frameworks | Runtime overhead, compilation, fusion, portability, and deployment | V1-07, The Framework Tax | Good kernel/fusion content; formulas local | Model dispatch tax, compile amortization, operator support, and runtime choice by track |
| Model Training | Training memory, pipeline bottlenecks, precision, and communication | V1-08, The Training Gauntlet | Strong memory and training pipeline themes | Convert to training budget, feasibility frontier, and training plan decisions |
| Data Selection | Data quality, coverage, curation, and selection bias | V1-09, The Data Selection Paradox | Good ICR/frontier ideas | Add subgroup coverage, rare-event selection, and track-specific data acquisition policies |
| Model Compression | Quantization, pruning, distillation, hardware support, and accuracy loss | V1-10, The Compression Paradox | Strong topic coverage; currently many parts | Reframe as compression feasibility, Pareto frontier, and validated compression recipe |
| Hardware Acceleration | Rooflines, arithmetic intensity, tiling, fusion, and energy | V1-11, The Hardware Roofline | Strong roofline alignment | Make track-specific hardware targets and reuse MLSysIM roofline/operator APIs |
| Benchmarking | Metrics, tails, thermal effects, and benchmark validity | V1-12, The Benchmarking Trap | Strong systems lesson | Make deployment-like benchmark protocols and multi-metric trade-offs explicit |
| Model Serving | Queues, batching, cold starts, KV cache, and tail latency | V1-13, The Tail Latency Trap | Good serving coverage | Use shared serving/queueing APIs; distinguish mobile fallback, edge streams, and cloud p99 |
| ML Operations | Drift, monitoring, retraining, release policy, and debt | V1-14, Silent Degradation | Strong operations theme | Tie workflow choices to drift detection, retraining cadence, and rollback policy |
| Responsible Engineering | Fairness, explainability, privacy, carbon, and system overhead | V1-15, No Free Fairness | Good topic fit; could be more tied to systems knobs | Represent responsible requirements as constraints with measurable cost and residual risk |
| Conclusion | Synthesis of Volume 1 principles and design decisions | V1-16, The Architect's Audit | Good synthesis intent | Replay ledger decisions, run sensitivity analysis, and produce a final architecture memo |

### Volume 2

| Book chapter | ML systems focus | Current lab | Current state | Planned improvement |
|---|---|---|---|---|
| Introduction | Scale changes reliability, coordination, and cost assumptions | V2-01, The Scale Illusion | Good conceptual fit | Make scale failure and coordination tax track-aware across devices, sites, and clusters |
| Compute Infrastructure | Nodes, accelerators, memory, power, utilization, and TCO | V2-02, Compute Infrastructure Wall | Strong fit; uses solver | Add procurement/placement frontier and structured TCO result objects |
| Network Fabrics | Link bandwidth, latency, topology, bisection, and placement | V2-03, Communication at Scale, partially | Current communication lab mixes fabrics and collectives | Refocus V2-03 on network fabrics and topology |
| Data Storage | Storage-compute gap, sharding, preprocessing, caching, and checkpoints | V2-04, Data Pipeline Wall | Strong fit | Reuse V1 data gravity at fleet scale; add checkpoint IO and shard-contention models |
| Distributed Training | Data/model/tensor/pipeline parallelism and memory fit | V2-05, Distributed Training | Strong topic fit | Build memory-fit diagnosis, parallelism frontier, and 3D training decision |
| Collective Communication | All-reduce/all-gather algorithms, hierarchy, compression, and overlap | No clean separate current lab; partially V2-03/V2-05 | Mapping gap | Add a dedicated Collective Communication lab |
| Fault Tolerance and Reliability | MTBF, checkpointing, recovery, redundancy, and useful work | V2-06, Fault Tolerance | Strong topic fit but chapter numbering shifts if collectives gets its own lab | Reuse storage/checkpoint models and separate training versus serving reliability decisions |
| Fleet Orchestration | Scheduling, fragmentation, preemption, heterogeneity, and utilization | V2-07, Fleet Orchestration | Strong topic fit | Make fleet mean devices, sites, or accelerators depending on track |
| Performance Engineering | Bottleneck diagnosis, fusion, precision, attention kernels, and optimization order | V2-09, Optimization Trap | Strong fit; current lab order differs from book order | Align ordering/links with book; make optimization ladder MLSysIM-backed |
| Inference at Scale | Serving cost, KV cache, batching, routing, and tail latency | Current V2-08, planned V2-10 after renumbering | Strong fit but current order differs from book order | Reuse V1 serving primitives and add fleet cost inversion by track |
| Edge Intelligence | Device-edge-cloud placement, energy, privacy, adaptation, and federation | V2-10, Edge Thermodynamics | Strong fit | Make this the cross-track placement lab for mobile, tiny, edge, and cloud |
| ML Operations at Scale | Silent failures, canaries, alert fatigue, platform ROI, and automation | V2-11, Silent Fleet | Strong fit | Distinguish model quality monitoring from systems health monitoring |
| Security & Privacy | Threat models, DP, encryption, secure aggregation, and defense overhead | V2-12, Price of Privacy | Strong fit | Make threat model and privacy budget drive measurable system costs |
| Robust AI | Robustness tax, drift, silent errors, defenses, and fallback policies | V2-13, Robustness Budget | Strong fit | Link robustness defenses to operations and monitoring signals |
| Sustainable AI | Energy, carbon, geography, lifecycle, and rebound effects | V2-14, Carbon Budget | Strong fit | Keep as trade-off lab, not calculator; add placement, lifecycle, and carbon caps |
| Responsible AI | Fairness at scale, feedback loops, governance, audits, and overhead | V2-15, Fairness Budget | Strong fit | Extend V1 responsible engineering into feedback and governance scaling |
| Conclusion | Fleet architecture synthesis across all scale constraints | V2-16, Fleet Synthesis | Strong synthesis intent | Replay Volume 2 ledger, run interaction/sensitivity map, and produce final design review |

## Universal Lab Shape

Every chapter lab should have the same conceptual skeleton, while preserving chapter-specific content:

### Opening: Track And Scenario

Every lab opens with a compact track selector:

- Mobile ML
- TinyML
- Edge AI
- Cloud/Scale

The selected track sets defaults for:

- Hardware target
- Model family
- Workload shape
- Latency, memory, power, privacy, reliability, and cost constraints
- Suggested reading link to the paired book chapter
- Scenario narrative and final decision prompt

Students can switch tracks at any point. The lab should recompute the scenario rather than simply changing text.

### Part A: Find The Binding Constraint

Part A should ask: "What breaks first?"

The student starts from a default scenario and manipulates a small number of knobs. The output should identify the active constraint: memory, compute, bandwidth, latency, energy, reliability, privacy, fairness, carbon, or cost.

### Part B: Draw The Trade-Off Curve

Part B should ask: "What changes when I move the engineering knobs?"

The lab should show a curve, frontier, heatmap, roofline, queueing plot, budget stack, or feasibility region. This is the "poster" view: the visual artifact that makes the system behavior legible.

### Part C: Make The Engineering Decision

Part C should ask: "What design would I defend?"

Students choose a configuration, record the reason, and save the decision to the design ledger. The ledger should capture enough structured data to replay decisions in synthesis labs.

## Track Model

The tracks should share concepts but differ in constraints and default stakes.

### Mobile ML

Default situation:

- User-facing application on a phone or tablet
- Tight latency, battery, thermal, app-size, and privacy constraints
- Intermittent network
- Model families such as MobileNet, EfficientNet-Lite, small transformers, audio or vision models

Typical trade-offs:

- Accuracy versus latency
- Local inference versus cloud fallback
- Batch size versus interactivity
- Compression versus quality
- On-device privacy versus update complexity

### TinyML

Default situation:

- Microcontroller or tiny NPU deployment
- Strict SRAM, flash, power, and duty-cycle constraints
- Simple sensors, wake-word, anomaly detection, gesture, or environmental monitoring
- Long battery life and low maintenance

Typical trade-offs:

- SRAM versus model capacity
- Sampling rate versus battery life
- Quantization versus accuracy
- Event-driven inference versus continuous sensing
- Simplicity versus robustness

### Edge AI

Default situation:

- Gateway, Jetson-class device, industrial PC, smart camera, or local server
- Moderate power budget
- Multiple streams or users
- Local privacy and low-latency requirements
- Occasional coordination with cloud services

Typical trade-offs:

- Throughput versus tail latency
- Local aggregation versus upstream bandwidth
- Multi-model scheduling versus specialization
- Thermal envelope versus utilization
- Fleet management complexity versus autonomy

### Cloud/Scale

Default situation:

- Datacenter training or inference
- GPU/accelerator clusters
- High throughput, cost, reliability, networking, storage, and carbon constraints
- Multi-tenant or fleet-scale operations

Typical trade-offs:

- Utilization versus latency
- Parallelism versus communication
- Checkpoint cost versus lost work
- Placement cost versus carbon
- Automation versus operational complexity

## MLSysIM Integration Requirements

Implementation should converge on a small shared lab API:

```python
from mlsysim import Engine, Hardware, Models, Scenarios
from mlsysbook_labs import lab_app, track_selector, scenario_controls, tradeoff_plot, decision_card
```

Recommended MLSysIM responsibilities:

- Deployment scenario defaults that are useful outside the book.
- Hardware, workload, and model registries.
- Canonical formulas for memory, latency, throughput, bandwidth, energy, carbon, cost, privacy, reliability, and queueing.
- Solver calls that return structured results.
- Feasibility classification and bottleneck attribution.

Recommended MLSysBook lab helper responsibilities:

- Curriculum track definitions that map to MLSysIM scenarios.
- Design ledger schema and persistence.
- Marimo components.
- Style tokens and plot themes.
- Chapter metadata and suggested reading links.
- Track-aware scenario text and decision prompts.

Recommended notebook responsibilities:

- Marimo widgets and layout.
- Chapter-specific framing.
- Track-specific wording.
- Interactive visualization wiring.
- Prompting students to make and justify decisions.

Each lab should save:

- `volume`
- `chapter`
- `lab_id`
- `track`
- `scenario_id`
- `chosen_configuration`
- `binding_constraint`
- `frontier_snapshot`
- `decision_rationale`
- `open_question`

## Detailed Lab Plans

### V1-00 - The Architect's Portal

Suggested reading:

- Volume 1 introduction

Core systems question:

- What kind of ML systems engineering world am I entering, and which constraints define it?

Track adaptation:

- Mobile ML: personal device, battery, app size, privacy, and responsiveness.
- TinyML: microcontroller, SRAM, flash, sensor rate, and battery lifetime.
- Edge AI: gateway or local accelerator, stream count, uplink, thermal budget, and autonomy.
- Cloud/Scale: shared accelerator fleet, cost, throughput, reliability, and carbon.

Part A - Constraint Portrait:

- Student selects a track and sees the default hardware, workload, model family, and operating constraints.
- Knobs: track, application domain, risk tolerance, connectivity assumption.
- MLSysIM source: `Scenarios.track_defaults`, `Hardware`, `Models`.
- Visual output: four-axis constraint radar covering memory, latency, energy/power, and cost/reliability.

Part B - Same Model, Different World:

- Student applies one model across all tracks and sees which deployment worlds reject it.
- Knobs: model size, input rate, precision, local/cloud split.
- MLSysIM source: feasibility solver over track defaults.
- Visual output: feasibility poster with green/yellow/red cells for each track.

Part C - Save The Engineering Lens:

- Student chooses the track they will use by default and writes what they expect the dominant constraint to be.
- Ledger output: selected track, scenario seed, expected bottleneck, first design hypothesis.
- Instructor use: compare how the same chapter concepts behave across different deployment regimes.

Implementation notes:

- Keep Lab 00 as the source of track state, but move track definitions into MLSysIM.
- Add a reusable track selector component that every other lab imports.
- Replace any static "course path" language with dynamic track state.

### V1-01 - Introduction To ML Systems: The AI Triad

Suggested reading:

- Volume 1, Introduction

Core systems question:

- When performance is bad, is the binding problem data, model, or machine?

Track adaptation:

- Mobile ML: image or language assistant on phone; app responsiveness and battery are binding.
- TinyML: wake-word or anomaly detector; false positives and SRAM are binding.
- Edge AI: multi-camera detection; stream throughput and local storage are binding.
- Cloud/Scale: online inference service; cost per request and tail latency are binding.

Part A - Diagnose The Triad:

- Student adjusts data quality, model complexity, and hardware capability.
- Knobs: data noise, data volume, parameter count, precision, accelerator class.
- MLSysIM source: `Engine.solve` with scenario constraints and bottleneck attribution.
- Visual output: triangular "Data, Model, Machine" balance map with active bottleneck highlighted.

Part B - Investment Frontier:

- Student allocates a fixed improvement budget among data, model, and hardware.
- Knobs: data-cleaning investment, architecture upgrade, hardware upgrade, compression.
- MLSysIM source: scenario solver plus cost model.
- Visual output: Pareto frontier of quality versus latency/cost, with dominated designs hidden or dimmed.

Part C - Defensible Fix:

- Student chooses one intervention and explains why the other two were not first.
- Ledger output: diagnosed bottleneck, chosen intervention, expected second-order consequence.
- Instructor use: demonstrate that "better model" is often not the correct first fix.

Implementation notes:

- Keep current `Engine.solve` usage and extend it to return explicit triad attribution.
- Make all track defaults feed the same solver call.
- Store a triad decision in the ledger for later replay in V1-16.

### V1-02 - ML Systems: The Physics Of Deployment

Suggested reading:

- Volume 1, ML Systems

Core systems question:

- Which physical resource limits deployment first: memory, bandwidth, compute, energy, or light-speed latency?

Track adaptation:

- Mobile ML: thermal throttling and memory bandwidth dominate sustained use.
- TinyML: SRAM and energy per inference dominate feasibility.
- Edge AI: sensor ingest, memory bandwidth, and uplink bandwidth dominate throughput.
- Cloud/Scale: accelerator memory, network distance, and power budget dominate scale.

Part A - First Wall:

- Student runs a workload through hardware targets and finds the first violated constraint.
- Knobs: batch size, input size, model size, precision, request rate.
- MLSysIM source: hardware resource model and feasibility classifier.
- Visual output: stacked resource budget showing which wall is exceeded.

Part B - Physics Curve:

- Student sweeps one major knob and sees when the system crosses from feasible to infeasible.
- Knobs: sequence length, image resolution, sampling rate, batch size, geographic distance.
- MLSysIM source: compute, memory, bandwidth, energy, and latency equations.
- Visual output: multi-line curve with threshold bands for memory wall, light barrier, power wall, and energy wall.

Part C - Deployment Choice:

- Student chooses where the model should run and what constraint they accept.
- Ledger output: deployment location, binding physical wall, avoided wall, mitigation.
- Instructor use: connect abstract constraints to non-negotiable physical limits.

Implementation notes:

- Move local wall formulas into MLSysIM and use one structured result object.
- Track-specific scenarios should differ by physical constraint, not just text.

### V1-03 - ML Workflow: The Constraint Tax

Suggested reading:

- Volume 1, ML Workflow

Core systems question:

- How do deployment constraints propagate backward through the ML development workflow?

Track adaptation:

- Mobile ML: app-release cadence, on-device QA, and privacy review slow iteration.
- TinyML: firmware flashing, sensor collection, and field debugging slow iteration.
- Edge AI: site-specific validation and fleet heterogeneity slow iteration.
- Cloud/Scale: distributed experiments, queue time, and evaluation cost slow iteration.

Part A - Constraint Propagation:

- Student starts from a deployment requirement and sees which workflow stages become constrained.
- Knobs: latency SLA, memory cap, privacy rule, accuracy target, deployment frequency.
- MLSysIM source: workflow stage model plus scenario constraints.
- Visual output: workflow dependency graph with constrained stages highlighted.

Part B - Iteration Throughput:

- Student changes data, training, validation, and deployment choices and sees iteration speed versus confidence.
- Knobs: dataset sample size, validation suite depth, hardware allocation, automated checks.
- MLSysIM source: iteration cost model and risk model.
- Visual output: curve of iteration speed versus escaped-risk probability.

Part C - Workflow Policy:

- Student chooses an iteration policy for the selected track.
- Ledger output: release policy, validation investment, accepted risk, expected bottleneck.
- Instructor use: show that workflow design is a systems decision, not process overhead.

Implementation notes:

- Preserve current constraint-propagation idea but make stage costs track-specific.
- Add a ledger replay hook so V1-14 and V1-16 can reference workflow choices.

### V1-04 - Data Engineering: The Data Gravity Trap

Suggested reading:

- Volume 1, Data Engineering

Core systems question:

- When data movement dominates, should the system move data, move compute, cache, compress, or reduce sampling?

Track adaptation:

- Mobile ML: local data privacy and upload limits shape training and personalization.
- TinyML: sensor rate, buffering, and preprocessing determine feasibility.
- Edge AI: video streams and local retention create data gravity.
- Cloud/Scale: data lake locality, preprocessing throughput, and accelerator starvation dominate cost.

Part A - Feed The Model:

- Student estimates whether the pipeline can keep the model or accelerator busy.
- Knobs: sample size, feature size, compression, preprocessing cost, storage bandwidth.
- MLSysIM source: data pipeline throughput and accelerator utilization model.
- Visual output: pipeline budget with ingest, preprocessing, transfer, and compute stages.

Part B - Data Gravity Frontier:

- Student compares moving raw data, preprocessed features, compressed data, or compute.
- Knobs: compression ratio, cache hit rate, compute placement, retention period.
- MLSysIM source: data movement, storage, and compute-placement cost model.
- Visual output: cost/latency frontier for data movement strategies.

Part C - Pipeline Architecture:

- Student selects a data architecture and identifies the failure mode it prevents.
- Ledger output: data placement, preprocessing location, cache policy, residual risk.
- Instructor use: make data engineering visible as a system bottleneck.

Implementation notes:

- Move feeding-tax and data-gravity formulas into MLSysIM.
- Track-specific defaults should use different data shapes: audio windows, sensor readings, video streams, records, or tokens.

### V1-05 - Neural Network Computation: The Activation Tax

Suggested reading:

- Volume 1, Neural Network Computation

Core systems question:

- Which neural network operations dominate memory, compute, and activation movement?

Track adaptation:

- Mobile ML: activation memory and memory bandwidth dominate real latency.
- TinyML: intermediate activations can exceed SRAM even when weights fit.
- Edge AI: batch and stream concurrency make activation buffers the bottleneck.
- Cloud/Scale: activation checkpointing and tensor core utilization shape throughput.

Part A - Operation Ledger:

- Student decomposes a layer or small network into parameters, activations, MACs, and memory traffic.
- Knobs: layer type, width, channels, resolution, sequence length, precision.
- MLSysIM source: operator cost model.
- Visual output: operation ledger showing weights, activations, MACs, and bytes moved.

Part B - Memory Cliff:

- Student sweeps width, resolution, or sequence length and sees when activations dominate weights.
- Knobs: batch size, precision, activation reuse, checkpointing, tiling.
- MLSysIM source: memory hierarchy and activation model.
- Visual output: activation versus weight curve with SRAM/DRAM/cache thresholds.

Part C - Layer Design Decision:

- Student chooses a layer configuration that fits the selected track and explains what was sacrificed.
- Ledger output: chosen operator shape, bottleneck, mitigation, sacrificed objective.
- Instructor use: counter the intuition that parameter count is the whole story.

Implementation notes:

- Replace local computation formulas with MLSysIM operator APIs.
- Use the same layer cost object later in compression, hardware acceleration, and performance labs.

### V1-06 - Neural Network Architectures: The Architecture Tax

Suggested reading:

- Volume 1, Neural Network Architectures

Core systems question:

- How does architectural structure change scaling behavior and deployment feasibility?

Track adaptation:

- Mobile ML: compact CNNs, efficient attention, and token limits matter.
- TinyML: depthwise convolutions, small MLPs, and feature extraction matter.
- Edge AI: video models, multi-stream inference, and spatial resolution matter.
- Cloud/Scale: transformer sequence length, MoE routing, and batch behavior matter.

Part A - Architecture Signature:

- Student compares CNN, MLP, RNN, transformer, and efficient variants under the same task.
- Knobs: resolution, sequence length, hidden width, depth, sparsity, precision.
- MLSysIM source: architecture cost model and `Engine.solve`.
- Visual output: signature table with compute, memory, bandwidth, latency, and accuracy proxy.

Part B - Scaling Law Shape:

- Student sweeps the variable that makes each architecture expensive.
- Knobs: sequence length for attention, image size for vision, hidden width for MLPs, stream count for edge.
- MLSysIM source: architecture scaling functions.
- Visual output: log-scale curve showing linear, quadratic, and width-squared growth.

Part C - Architecture Choice:

- Student picks an architecture for the track and identifies where it will fail as workload grows.
- Ledger output: architecture family, scaling risk, deployment constraint, mitigation.
- Instructor use: show architecture as a resource allocation decision.

Implementation notes:

- Keep existing `Engine.solve` integration.
- Add architecture signatures to MLSysIM so later labs can reuse them.

### V1-07 - ML Frameworks: The Kernel Fusion Dividend

Suggested reading:

- Volume 1, ML Frameworks

Core systems question:

- When do framework overhead, kernel launch cost, compilation, and fusion change the best implementation choice?

Track adaptation:

- Mobile ML: runtime size, operator support, and delegate availability matter.
- TinyML: generated C, operator kernels, and static memory planning matter.
- Edge AI: runtime portability and accelerator delegates matter.
- Cloud/Scale: compilation amortization, graph capture, and fused kernels matter.

Part A - Dispatch Tax:

- Student compares many small operations against fewer fused operations.
- Knobs: operator count, tensor size, batch size, runtime, accelerator.
- MLSysIM source: runtime overhead and kernel launch model.
- Visual output: latency stack split into useful compute, dispatch, transfer, and synchronization.

Part B - Fusion And Compilation Break-Even:

- Student varies request volume and model shape to find when compilation or fusion pays off.
- Knobs: compile time, inference count, fusion depth, dynamic shapes, cache hit rate.
- MLSysIM source: framework execution model.
- Visual output: break-even curve for eager, graph, compiled, and fused paths.

Part C - Runtime Choice:

- Student selects a framework/runtime path and defends it for the track.
- Ledger output: runtime, optimization level, break-even assumption, portability risk.
- Instructor use: show that framework choice is a systems trade-off, not only developer preference.

Implementation notes:

- Move dispatch and compile amortization equations into MLSysIM.
- Track-specific defaults should include realistic runtimes: TFLite/Core ML, TFLite Micro, ONNX/TensorRT, PyTorch/XLA/Triton.

### V1-08 - Model Training: The Training Memory Budget

Suggested reading:

- Volume 1, Training

Core systems question:

- What actually consumes memory and time during training, and which training knobs trade accuracy, speed, and feasibility?

Track adaptation:

- Mobile ML: personalization or fine-tuning is constrained by device memory and privacy.
- TinyML: training is usually off-device, but data and adaptation constraints matter.
- Edge AI: local fine-tuning and site adaptation compete with serving.
- Cloud/Scale: optimizer state, activation memory, and communication dominate.

Part A - Training Budget:

- Student accounts for weights, activations, gradients, optimizer state, and data pipeline.
- Knobs: batch size, precision, optimizer, sequence length, checkpointing.
- MLSysIM source: training memory and throughput model.
- Visual output: memory budget stack with infeasible components marked.

Part B - Feasibility Knobs:

- Student trades batch size, precision, checkpointing, accumulation, and optimizer choice.
- Knobs: mixed precision, gradient checkpointing, gradient accumulation, optimizer state format.
- MLSysIM source: training solver with memory and throughput constraints.
- Visual output: speed versus memory frontier.

Part C - Training Plan:

- Student chooses a training or adaptation plan and explains its hidden cost.
- Ledger output: batch plan, precision, optimizer, memory mitigation, expected convergence risk.
- Instructor use: reveal why training feasibility is often memory-bound before compute-bound.

Implementation notes:

- Connect this lab to distributed training later through a shared training-state model.
- Track-specific scenarios should distinguish full training, fine-tuning, personalization, and simulation-only exploration.

### V1-09 - Data Selection: The Selection Paradox

Suggested reading:

- Volume 1, Data Selection

Core systems question:

- Which data should the system collect, label, filter, keep, or discard under limited budget?

Track adaptation:

- Mobile ML: privacy and representativeness constrain personalization data.
- TinyML: rare events and sensor conditions constrain coverage.
- Edge AI: site-specific data drift and labeling cost dominate.
- Cloud/Scale: massive datasets create curation, deduplication, and subgroup coverage trade-offs.

Part A - Quality Versus Quantity:

- Student compares adding more data with improving selected data quality.
- Knobs: dataset size, label quality, duplication, class imbalance, noise.
- MLSysIM source: data utility and training cost model.
- Visual output: information-cost frontier.

Part B - Coverage And Inequality:

- Student sees how global accuracy can improve while subgroup performance worsens.
- Knobs: selection policy, subgroup prevalence, rare-event sampling, active labeling.
- MLSysIM source: coverage and subgroup risk model.
- Visual output: subgroup performance heatmap plus overall metric.

Part C - Data Policy:

- Student chooses a data acquisition and filtering policy.
- Ledger output: selection policy, target subgroup or condition, accepted bias risk, mitigation.
- Instructor use: show that data selection is an engineering control surface.

Implementation notes:

- Keep ICR/frontier concepts but expose more track-specific scenarios.
- Save subgroup and coverage choices for responsible engineering labs.

### V1-10 - Model Compression: The Compression Paradox

Suggested reading:

- Volume 1, Model Compression

Core systems question:

- When does making a model smaller improve the system, and when does it create new costs or failures?

Track adaptation:

- Mobile ML: quantization and pruning target latency, memory, app size, and battery.
- TinyML: compression is required for SRAM/flash feasibility.
- Edge AI: compression enables more streams but can reduce robustness.
- Cloud/Scale: compression trades serving cost against quality and engineering complexity.

Part A - Compression Feasibility:

- Student applies quantization, pruning, distillation, or low-rank methods and checks constraints.
- Knobs: precision, sparsity, distillation strength, target hardware, batch size.
- MLSysIM source: compression model plus hardware kernel support.
- Visual output: size, latency, energy, and quality budget before/after compression.

Part B - Compression Frontier:

- Student sweeps compression strength and sees quality, latency, memory, and energy together.
- Knobs: bit width, structured versus unstructured pruning, calibration data size, student model size.
- MLSysIM source: compression-quality and runtime support model.
- Visual output: Pareto frontier with unsupported or accuracy-collapsed regions.

Part C - Compression Recipe:

- Student chooses a recipe and identifies the validation test needed before deployment.
- Ledger output: compression method, target metric, quality risk, hardware dependency, validation plan.
- Instructor use: show that smaller is not automatically faster or safer.

Implementation notes:

- Keep five existing topics but reorganize into A/B/C.
- Add explicit hardware support checks for sparse and low-bit operators.

### V1-11 - Hardware Acceleration: The Hardware Roofline

Suggested reading:

- Volume 1, Hardware Acceleration

Core systems question:

- Is the workload compute-bound, bandwidth-bound, memory-bound, or limited by data movement overhead?

Track adaptation:

- Mobile ML: NPU/GPU/CPU delegate selection and thermal throttling matter.
- TinyML: MCU, DSP, and tiny NPU choices depend on memory and operator support.
- Edge AI: accelerator choice depends on stream mix, batching, and thermal envelope.
- Cloud/Scale: GPU/TPU selection depends on arithmetic intensity, memory capacity, and interconnect.

Part A - Roofline Diagnosis:

- Student places the workload on a hardware roofline.
- Knobs: operator shape, precision, batch size, memory bandwidth, peak compute.
- MLSysIM source: roofline model.
- Visual output: roofline chart with active regime and headroom.

Part B - Move The Point:

- Student applies tiling, fusion, batching, precision changes, and layout changes.
- Knobs: tile size, fusion depth, precision, batch size, data reuse.
- MLSysIM source: arithmetic intensity and memory traffic model.
- Visual output: before/after roofline position with latency and energy deltas.

Part C - Accelerator Decision:

- Student chooses hardware and optimization strategy for the track.
- Ledger output: accelerator, bottleneck regime, optimization, remaining limitation.
- Instructor use: teach roofline as a design tool rather than a static chart.

Implementation notes:

- Fix existing SyntaxWarning in this lab during implementation.
- Keep current roofline content but make the track change hardware defaults and operator mix.

### V1-12 - Performance Benchmarking: The Benchmarking Trap

Suggested reading:

- Volume 1, Performance Benchmarking

Core systems question:

- What benchmark would reveal the real deployment bottleneck instead of hiding it?

Track adaptation:

- Mobile ML: cold start, thermal steady state, and battery impact matter.
- TinyML: duty cycle, sensor timing, and power measurement matter.
- Edge AI: multi-stream tail latency and sustained thermals matter.
- Cloud/Scale: p50, p95, p99, cost per request, and availability matter.

Part A - Benchmark Illusion:

- Student runs the same system under an easy benchmark and a deployment-like benchmark.
- Knobs: warmup, batch size, concurrency, input distribution, duration, metric.
- MLSysIM source: benchmark workload and measurement model.
- Visual output: metric comparison showing how the conclusion changes.

Part B - Multi-Metric Trade-Off:

- Student chooses which metric to optimize and sees what gets worse.
- Knobs: throughput target, tail latency target, energy cap, accuracy threshold, cost cap.
- MLSysIM source: metric aggregation and queueing/thermal model.
- Visual output: dashboard where optimizing one metric moves others.

Part C - Benchmark Protocol:

- Student writes a benchmark protocol that would catch the dominant failure.
- Ledger output: benchmark design, primary metric, guardrail metrics, known blind spot.
- Instructor use: expose why benchmarking is a hypothesis about deployment.

Implementation notes:

- Move benchmark distributions and metric calculations into MLSysIM.
- Track-specific scenarios should change the benchmark protocol, not only the target numbers.

### V1-13 - Model Serving: The Tail Latency Trap

Suggested reading:

- Volume 1, Model Serving

Core systems question:

- How do batching, queues, cold starts, memory state, and request variability shape serving behavior?

Track adaptation:

- Mobile ML: local inference and cloud fallback create availability and privacy choices.
- TinyML: serving is event-triggered and constrained by duty cycle.
- Edge AI: many local streams compete for accelerator time.
- Cloud/Scale: request bursts, KV cache, autoscaling, and p99 latency dominate cost.

Part A - Queueing Failure:

- Student increases arrival rate or variability and watches tail latency explode.
- Knobs: arrival rate, service time, concurrency, batch size, burstiness.
- MLSysIM source: queueing model and serving solver.
- Visual output: p50/p95/p99 latency curve with utilization.

Part B - Serving Knobs:

- Student tests batching, autoscaling, cold-start mitigation, cache size, and model choice.
- Knobs: max batch size, batch timeout, replicas, cache policy, quantization.
- MLSysIM source: serving cost and latency model.
- Visual output: throughput/latency/cost frontier.

Part C - Capacity Plan:

- Student chooses serving configuration and states the failure mode it protects against.
- Ledger output: capacity plan, batching policy, autoscaling policy, tail-latency risk.
- Instructor use: make p99 latency and utilization trade-offs concrete.

Implementation notes:

- Reuse serving APIs in V2-08.
- Add track-specific request distributions.

### V1-14 - MLOps: The Silent Degradation Problem

Suggested reading:

- Volume 1, MLOps

Core systems question:

- How much monitoring, retraining, and release machinery is needed to control degradation without overspending?

Track adaptation:

- Mobile ML: app updates, privacy restrictions, and user telemetry limits matter.
- TinyML: field failures are hard to inspect and update.
- Edge AI: site drift and fleet heterogeneity create uneven degradation.
- Cloud/Scale: model fleets require monitoring automation and retraining pipelines.

Part A - Drift Visibility:

- Student simulates drift and sees whether chosen monitors detect it.
- Knobs: drift type, drift speed, monitor frequency, label delay, alert threshold.
- MLSysIM source: drift, monitoring, and degradation model.
- Visual output: timeline showing true quality, observed signal, alert time, and damage.

Part B - Retraining Cadence:

- Student trades retraining cost against degradation cost.
- Knobs: retrain frequency, data refresh size, validation depth, deployment risk.
- MLSysIM source: ops cost and risk model.
- Visual output: total cost curve with undertraining and overtraining regions.

Part C - Operations Policy:

- Student chooses monitoring, retraining, rollback, and escalation policies.
- Ledger output: monitoring plan, retrain trigger, release policy, residual blind spot.
- Instructor use: show operational design as part of the ML system, not afterthought.

Implementation notes:

- Link this lab to workflow choices from V1-03.
- Move degradation and retraining formulas into MLSysIM.

### V1-15 - Responsible Engineering: No Free Fairness

Suggested reading:

- Volume 1, Responsible Engineering

Core systems question:

- Which responsible engineering requirement changes the system design, and what cost or constraint does it introduce?

Track adaptation:

- Mobile ML: privacy, battery, and user control shape telemetry and personalization.
- TinyML: simple models make explainability easier but robustness harder.
- Edge AI: site-level bias and surveillance concerns shape deployment policy.
- Cloud/Scale: fairness audits, privacy controls, explainability, and carbon accounting add platform overhead.

Part A - Metric Conflict:

- Student adjusts thresholds and sees fairness metrics disagree.
- Knobs: subgroup distribution, threshold, calibration, target metric.
- MLSysIM source: fairness metric model.
- Visual output: metric conflict panel showing accuracy, equality, calibration, and subgroup error.

Part B - Responsibility Budget:

- Student adds privacy, explainability, robustness, or carbon constraints and sees engineering overhead.
- Knobs: DP noise, explanation depth, audit sample size, logging retention, carbon cap.
- MLSysIM source: responsible-system overhead model.
- Visual output: quality/cost/latency/fairness budget stack.

Part C - Responsible Design Decision:

- Student chooses a responsible engineering policy and states what it costs.
- Ledger output: selected obligation, system change, trade-off accepted, audit evidence.
- Instructor use: show responsible engineering as a design constraint with measurable effects.

Implementation notes:

- Reuse subgroup choices from V1-09.
- Connect this lab to Volume 2 responsible AI and security/privacy labs.

### V1-16 - Conclusion: The Architect's Audit

Suggested reading:

- Volume 1 conclusion

Core systems question:

- Across the full Volume 1 design ledger, which decisions were robust, which were fragile, and which constraints moved?

Track adaptation:

- Mobile ML: audit battery, privacy, latency, and app deployment decisions.
- TinyML: audit SRAM, flash, duty cycle, update, and field validation decisions.
- Edge AI: audit stream scaling, thermal, data movement, and site drift decisions.
- Cloud/Scale: audit cost, utilization, serving, operations, and responsible engineering decisions.

Part A - Ledger Replay:

- Student loads prior lab decisions and sees the implied end-to-end architecture.
- Knobs: include/exclude decisions, track switch, risk tolerance.
- Source: `mlsysbook_labs` design ledger plus MLSysIM scenario solver.
- Visual output: architecture map annotated with saved constraints and decisions.

Part B - Sensitivity Audit:

- Student perturbs workload, hardware, data, or target metric and sees which decisions break.
- Knobs: request rate, data drift, model size, power cap, latency SLA, cost cap.
- MLSysIM source: sensitivity solver and feasibility classifier.
- Visual output: fragility heatmap across prior decisions.

Part C - Final Architecture Memo:

- Student revises one decision and writes the principle behind it.
- Ledger output: final Volume 1 architecture, revised decision, dominant principle, open risk.
- Instructor use: make the course cumulative without becoming a build project.

Implementation notes:

- This lab depends on consistent ledger schemas from prior labs.
- Add a fallback mode for students who skipped earlier labs: generate a synthetic ledger by track.

### V2-01 - Introduction To ML Systems At Scale: The Scale Illusion

Suggested reading:

- Volume 2 introduction

Core systems question:

- Which assumptions fail when an ML system moves from a single deployment to a fleet or cluster?

Track adaptation:

- Mobile ML: many devices create update, telemetry, privacy, and version skew issues.
- TinyML: many deployed sensors create maintenance, battery, and silent failure issues.
- Edge AI: many sites create heterogeneity and orchestration issues.
- Cloud/Scale: many accelerators create coordination, reliability, and cost issues.

Part A - Scaling Illusion:

- Student scales device count, model count, users, or accelerators and sees reliability/cost collapse.
- Knobs: fleet size, per-node failure rate, coordination overhead, utilization.
- MLSysIM source: scale reliability and coordination model.
- Visual output: scale curve showing where linear assumptions fail.

Part B - Coordination Tax:

- Student introduces coordination, monitoring, communication, or orchestration and sees useful work decline.
- Knobs: synchronization frequency, monitoring overhead, retry policy, heterogeneity.
- MLSysIM source: coordination and effective-throughput model.
- Visual output: useful work versus overhead poster.

Part C - Scale Readiness Decision:

- Student chooses whether to scale, shard, specialize, or simplify the system.
- Ledger output: scale plan, coordination bottleneck, reliability risk, first mitigation.
- Instructor use: set up Volume 2 as a study of fleet and cluster trade-offs.

Implementation notes:

- Use this lab to introduce Volume 2 ledger fields for fleet size, coordination, and failure budget.

### V2-02 - Compute Infrastructure: The Compute Infrastructure Wall

Suggested reading:

- Volume 2, Compute Infrastructure

Core systems question:

- Which compute infrastructure choice best fits the workload once memory, bandwidth, power, cost, and utilization are included?

Track adaptation:

- Mobile ML: device SoC choices and accelerators vary by user hardware.
- TinyML: MCU class, tiny accelerator, and memory hierarchy dominate.
- Edge AI: local GPU/NPU/server choices determine stream density.
- Cloud/Scale: GPU/TPU nodes, interconnect, rack power, and procurement dominate.

Part A - Node Feasibility:

- Student maps workload to candidate hardware and finds the active wall.
- Knobs: model, precision, batch size, input shape, memory capacity, peak compute.
- MLSysIM source: `Engine.solve`, hardware registry, roofline and memory model.
- Visual output: hardware comparison table plus active bottleneck.

Part B - Infrastructure Frontier:

- Student sweeps hardware tier or fleet size and sees cost, latency, throughput, and utilization.
- Knobs: accelerator type, node count, utilization target, power cap, memory tier.
- MLSysIM source: infrastructure cost and throughput model.
- Visual output: TCO/performance frontier with infeasible regions.

Part C - Procurement Or Placement Decision:

- Student chooses infrastructure and states the assumption that would invalidate it.
- Ledger output: hardware tier, capacity plan, cost assumption, invalidation trigger.
- Instructor use: tie hardware catalogs to engineering constraints.

Implementation notes:

- Preserve current `Engine.solve` usage and add structured TCO output.
- Reuse roofline components from V1-11.

### V2-03 - Network Fabrics And Communication: Communication At Scale

Suggested reading:

- Volume 2, Network Fabrics

Core systems question:

- When does network fabric topology, bandwidth, latency, and bisection bandwidth become the limiting resource?

Track adaptation:

- Mobile ML: communication means cloud fallback, telemetry, model updates, and bandwidth caps.
- TinyML: communication means low-rate radio, duty cycle, and intermittent connectivity.
- Edge AI: communication means uplink, site aggregation, and cross-device coordination.
- Cloud/Scale: communication means all-reduce, parameter exchange, network topology, and bisection bandwidth.

Part A - Communication Budget:

- Student estimates communication time relative to compute time.
- Knobs: message size, link bandwidth, latency, node count, synchronization frequency.
- MLSysIM source: network and communication time model.
- Visual output: communication versus compute budget stack.

Part B - Topology Frontier:

- Student compares fabric and topology choices under scale.
- Knobs: node count, topology, oversubscription, bisection bandwidth, hop count.
- MLSysIM source: network fabric and topology model.
- Visual output: frontier showing feasible regions by topology and scale.

Part C - Fabric Design:

- Student chooses fabric/topology and states the placement assumption it requires.
- Ledger output: fabric choice, topology assumption, placement constraint, residual risk.
- Instructor use: show network fabric as a first-class systems design dimension.

Implementation notes:

- Keep this lab focused on network fabrics. Move collective-operation content into the dedicated Collective Communication lab.
- Move topology and fabric formulas into MLSysIM.

### V2-04 - Data Storage: The Data Pipeline Wall

Suggested reading:

- Volume 2, Data Storage

Core systems question:

- How should storage, sharding, caching, preprocessing, and checkpointing be designed so compute does not starve?

Track adaptation:

- Mobile ML: local cache, sync, and privacy-preserving upload matter.
- TinyML: storage is flash, logs, and compressed sensor traces.
- Edge AI: local retention, stream buffering, and site bandwidth dominate.
- Cloud/Scale: object storage, parallel file systems, sharding, and checkpoint IO dominate.

Part A - Storage-Compute Gap:

- Student compares data delivery throughput with compute consumption.
- Knobs: sample size, dataset size, storage bandwidth, preprocessing cost, accelerator count.
- MLSysIM source: storage pipeline and compute demand model.
- Visual output: storage-to-compute gap chart.

Part B - Sharding And Cache Frontier:

- Student changes sharding, prefetching, cache size, and preprocessing placement.
- Knobs: shard count, cache hit rate, prefetch depth, worker count, locality.
- MLSysIM source: storage contention and cache model.
- Visual output: stall rate versus storage cost frontier.

Part C - Storage Architecture:

- Student chooses storage and checkpoint design.
- Ledger output: sharding policy, cache policy, checkpoint location, stall risk.
- Instructor use: reveal why fast accelerators need carefully designed storage systems.

Implementation notes:

- Reuse data-gravity concepts from V1-04 at larger scale.
- Add checkpoint IO hooks that V2-06 can reuse.

### V2-05 - Distributed Training: Parallelism Design

Suggested reading:

- Volume 2, Distributed Training

Core systems question:

- Which parallelism strategy fits the model, hardware, memory budget, and communication fabric?

Track adaptation:

- Mobile ML: distributed training appears as federated or split learning, with client heterogeneity.
- TinyML: distributed training appears as federated sensor learning or cloud training for tiny deployments.
- Edge AI: site-level training and federated aggregation trade locality against coordination.
- Cloud/Scale: data, model, tensor, pipeline, and expert parallelism dominate training feasibility.

Part A - Memory Fit:

- Student checks whether model states fit on one device or require sharding.
- Knobs: parameter count, optimizer, precision, sequence length, batch size, device memory.
- MLSysIM source: distributed training memory model.
- Visual output: memory stack across weights, gradients, optimizer, activations, and KV/sequence state.

Part B - Parallelism Frontier:

- Student compares data parallelism, ZeRO/FSDP, tensor parallelism, pipeline parallelism, and hybrid 3D plans.
- Knobs: device count, microbatch size, pipeline stages, tensor parallel degree, sharding level.
- MLSysIM source: training throughput, memory, and communication solver.
- Visual output: throughput versus memory versus communication frontier.

Part C - Training Architecture:

- Student chooses a parallelism plan and describes the bottleneck it moves to.
- Ledger output: parallelism plan, memory mitigation, communication assumption, scaling limit.
- Instructor use: connect model scale to hardware and network topology.

Implementation notes:

- Current lab already has communication wall and ZeRO content. Reorganize into A/B/C and drive all calculations through MLSysIM.
- Cross-link this lab to V2-03 Network Fabrics and V2-06 Collective Communication rather than duplicating formulas.

### V2-06 - Collective Communication: The Communication Lab

Suggested reading:

- Volume 2, Collective Communication

Core systems question:

- When do collective communication algorithms dominate useful work, and which algorithm or mitigation changes the result?

Track adaptation:

- Mobile ML: communication means model updates, telemetry, and privacy-preserving aggregation.
- TinyML: communication means low-rate radio, duty cycle, and intermittent federation.
- Edge AI: communication means local aggregation, site coordination, and cross-device synchronization.
- Cloud/Scale: communication means all-reduce, all-gather, reduce-scatter, all-to-all, hierarchy, and overlap.

Part A - Collective Operation Anatomy:

- Student inspects AllReduce, AllGather, ReduceScatter, Broadcast, and All-to-All as data movement patterns.
- Knobs: operation type, tensor size, GPU count, topology, precision.
- MLSysIM source: collective operation primitives.
- Visual output: operation diagram with bytes moved, rounds, and critical links.

Part B - Algorithm/Topology Frontier:

- Student compares ring, tree, hierarchical, and topology-aware collectives.
- Knobs: node count, message size, bandwidth, latency, hierarchy, oversubscription.
- MLSysIM source: alpha-beta collective and topology model.
- Visual output: crossover chart showing when each collective wins.

Part C - Communication Optimization Decision:

- Student chooses compression, overlap, hierarchy, or algorithm change and names the trade-off.
- Ledger output: collective strategy, optimization, residual risk, scaling assumption.
- Instructor use: connect distributed training performance to communication algorithms instead of treating communication as one opaque overhead term.

Implementation notes:

- This should be a dedicated Volume 2 lab. Downstream lab numbering should be adjusted to match the book sequence.
- Reuse fabric/topology results from V2-03 and distributed training context from V2-05.

### V2-07 - Fault Tolerance: Failure Budget Engineering

Suggested reading:

- Volume 2, Fault Tolerance

Core systems question:

- How much redundancy, checkpointing, and recovery machinery is worth paying for under a given failure budget?

Track adaptation:

- Mobile ML: failures are offline devices, stale models, interrupted updates, and privacy-safe recovery.
- TinyML: failures are battery depletion, sensor faults, and unreachable deployments.
- Edge AI: failures are site outages, thermal shutdown, disk wear, and partial fleet loss.
- Cloud/Scale: failures are node loss, job interruption, checkpoint storms, and serving outages.

Part A - Failure Exposure:

- Student scales runtime or fleet size and sees expected failures.
- Knobs: fleet size, job duration, MTBF, update frequency, service redundancy.
- MLSysIM source: reliability and failure probability model.
- Visual output: probability of clean completion or service survival.

Part B - Recovery Frontier:

- Student compares checkpoint interval, async checkpointing, replication, retries, and graceful degradation.
- Knobs: checkpoint interval, checkpoint size, storage bandwidth, redundancy factor, retry limit.
- MLSysIM source: checkpoint economics and serving reliability model.
- Visual output: useful work versus recovery overhead curve.

Part C - Resilience Policy:

- Student chooses a resilience strategy and names the failure it does not cover.
- Ledger output: failure budget, checkpoint/redundancy policy, recovery time target, blind spot.
- Instructor use: show reliability as a costed design choice.

Implementation notes:

- Reuse checkpoint IO APIs from V2-04.
- Add separate training and serving resilience modes.

### V2-08 - Fleet Orchestration: Scheduling Under Constraint

Suggested reading:

- Volume 2, Fleet Orchestration

Core systems question:

- How should a fleet be scheduled when utilization, latency, fragmentation, preemption, and heterogeneity conflict?

Track adaptation:

- Mobile ML: orchestration is rollout, version skew, and feature gating across devices.
- TinyML: orchestration is staged updates and battery-aware duty scheduling.
- Edge AI: orchestration is site placement, stream admission, and heterogeneous devices.
- Cloud/Scale: orchestration is accelerator scheduling, bin packing, quotas, and preemption.

Part A - Queueing And Utilization:

- Student changes demand and sees queue length, wait time, and utilization.
- Knobs: arrival rate, job size distribution, device count, priority class.
- MLSysIM source: scheduler and queueing model.
- Visual output: queue/utilization chart with SLA violation region.

Part B - Fragmentation And Preemption:

- Student compares packing and scheduling policies.
- Knobs: placement policy, GPU slice size, memory reservation, preemption rate, heterogeneity.
- MLSysIM source: bin-packing, fragmentation, and preemption cost model.
- Visual output: utilization versus rejected-work frontier.

Part C - Fleet Policy:

- Student chooses scheduling policy for the track.
- Ledger output: scheduler policy, admission rule, preemption rule, fairness or priority trade-off.
- Instructor use: show why high utilization can harm responsiveness and fairness.

Implementation notes:

- Move current queuing and fragmentation formulas into MLSysIM.
- Track-specific scenarios should make "fleet" mean devices, sites, or accelerators depending on track.

### V2-09 - Performance Engineering: The Optimization Trap

Suggested reading:

- Volume 2, Performance Engineering

Core systems question:

- Which optimization is worth doing, and when does it only move the bottleneck?

Track adaptation:

- Mobile ML: optimize for sustained latency and battery, not just peak speed.
- TinyML: optimize memory layout, operator selection, and duty cycle.
- Edge AI: optimize stream density and thermal stability.
- Cloud/Scale: optimize MFU, kernel fusion, attention kernels, precision, and communication overlap.

Part A - Bottleneck Diagnosis:

- Student profiles a workload and identifies bottleneck category.
- Knobs: model, hardware, batch, precision, sequence length, stream count.
- MLSysIM source: roofline, memory, and execution model via `Engine.solve`.
- Visual output: bottleneck report with roofline and time breakdown.

Part B - Optimization Ladder:

- Student applies one optimization at a time and sees speedup, cost, and new bottleneck.
- Knobs: fusion, layout, precision, FlashAttention-style kernels, overlap, tiling.
- MLSysIM source: optimization transformation model.
- Visual output: waterfall chart of improvement and bottleneck movement.

Part C - Optimization Plan:

- Student chooses an optimization sequence and stops when marginal gains fall below cost.
- Ledger output: first optimization, expected speedup, new bottleneck, stop criterion.
- Instructor use: teach disciplined optimization instead of random tuning.

Implementation notes:

- Keep current optimization topics but structure them as diagnosis, frontier, decision.
- Share roofline components with V1-11 and compute infrastructure.

### V2-10 - Inference At Scale: Serving Cost Inversion

Suggested reading:

- Volume 2, Inference

Core systems question:

- When does inference cost, memory state, and tail latency dominate the ML system?

Track adaptation:

- Mobile ML: decide local inference, cloud fallback, or hybrid serving.
- TinyML: decide event frequency, duty cycle, and on-device thresholding.
- Edge AI: decide stream admission, batching, and local acceleration.
- Cloud/Scale: decide batching, KV cache, autoscaling, and model routing.

Part A - Cost Inversion:

- Student compares training cost with recurring inference cost under demand.
- Knobs: request rate, model size, hardware, precision, cache size, SLA.
- MLSysIM source: inference cost and serving solver.
- Visual output: cumulative training versus inference cost curve.

Part B - State And Batching Frontier:

- Student explores KV cache, continuous batching, batch timeout, and model parallel serving.
- Knobs: sequence length, concurrent sessions, batch policy, cache precision, replicas.
- MLSysIM source: serving memory, queueing, and `Engine.solve`.
- Visual output: latency/cost/memory frontier.

Part C - Serving Fleet Design:

- Student chooses serving architecture and capacity policy.
- Ledger output: local/cloud/fleet design, batching policy, cache policy, cost risk.
- Instructor use: connect Volume 1 serving concepts to fleet-scale economics.

Implementation notes:

- Reuse V1-13 serving primitives.
- Make track-specific serving scenarios substantially different.

### V2-11 - Edge Intelligence: The Edge Thermodynamics Lab

Suggested reading:

- Volume 2, Edge Intelligence

Core systems question:

- What should run on the device, at the edge, or in the cloud when latency, privacy, energy, adaptation, and reliability conflict?

Track adaptation:

- Mobile ML: phone local inference versus cloud fallback.
- TinyML: sensor thresholding versus gateway inference versus cloud analytics.
- Edge AI: camera/gateway/server split for multi-stream workloads.
- Cloud/Scale: cloud fleet serving with edge caches or regional placement.

Part A - Placement Feasibility:

- Student places model stages across device, edge, and cloud.
- Knobs: split point, input size, compression, network bandwidth, privacy constraint.
- MLSysIM source: placement, memory, latency, and energy model.
- Visual output: device-edge-cloud pipeline with cost and latency annotations.

Part B - Adaptation And Federation Frontier:

- Student compares local adaptation, centralized retraining, federated learning, and no adaptation.
- Knobs: update frequency, client count, privacy constraint, communication budget, drift rate.
- MLSysIM source: adaptation and federated communication model.
- Visual output: adaptation quality versus communication/energy frontier.

Part C - Edge Architecture:

- Student chooses local, edge, cloud, or hybrid architecture.
- Ledger output: placement, adaptation strategy, privacy assumption, energy/latency trade-off.
- Instructor use: show edge intelligence as a placement and coordination problem.

Implementation notes:

- This lab should become the canonical cross-track demonstration because all tracks map naturally to it.
- Move battery drain and federation formulas into MLSysIM.

### V2-12 - Operations At Scale: The Silent Fleet

Suggested reading:

- Volume 2, Operations At Scale

Core systems question:

- How do monitoring, alerting, canaries, automation, and platform investment scale with fleet complexity?

Track adaptation:

- Mobile ML: telemetry limits, staged rollout, and app versions complicate operations.
- TinyML: sparse telemetry and physical maintenance make failures silent.
- Edge AI: site heterogeneity and partial outages dominate operations.
- Cloud/Scale: many models and services create alert fatigue and platform ROI questions.

Part A - Complexity Growth:

- Student scales models, sites, devices, or services and watches manual operations break.
- Knobs: fleet size, model count, telemetry rate, incident rate, operator capacity.
- MLSysIM source: operations complexity model.
- Visual output: operational load versus human/platform capacity chart.

Part B - Automation And Canary Frontier:

- Student adjusts canary size, duration, alert threshold, and platform investment.
- Knobs: canary traffic, detection threshold, false positive tolerance, automation level.
- MLSysIM source: canary detection and platform ROI model.
- Visual output: detection speed versus false alerts versus cost.

Part C - Ops Architecture:

- Student chooses monitoring, rollout, and platform investment policy.
- Ledger output: canary policy, alert policy, automation investment, residual silent-failure risk.
- Instructor use: show why ML systems need operational design before scale arrives.

Implementation notes:

- Reuse V1-14 drift and release concepts, but scale them to fleet operations.
- Add a clear distinction between model quality monitors and systems health monitors.

### V2-13 - Security And Privacy: The Price Of Privacy

Suggested reading:

- Volume 2, Security And Privacy

Core systems question:

- Which security or privacy control is required, and what does it cost in accuracy, latency, compute, memory, or usability?

Track adaptation:

- Mobile ML: local data, secure enclave, telemetry minimization, and user consent matter.
- TinyML: physical access, limited crypto budget, and update integrity matter.
- Edge AI: site privacy, secure aggregation, and local attack surfaces matter.
- Cloud/Scale: DP, access control, audit logs, red-team defenses, and serving protections matter.

Part A - Threat And Privacy Budget:

- Student chooses a threat model or privacy requirement and sees affected system components.
- Knobs: attacker capability, data sensitivity, epsilon target, logging retention, encryption.
- MLSysIM source: privacy/security requirement model.
- Visual output: threat surface map and privacy budget meter.

Part B - Defense Overhead Frontier:

- Student adds DP, encryption, secure aggregation, filtering, rate limits, or monitoring.
- Knobs: epsilon, noise scale, encryption mode, audit depth, guardrail strictness.
- MLSysIM source: defense overhead and quality model.
- Visual output: privacy/security strength versus accuracy/latency/cost frontier.

Part C - Security/Privacy Policy:

- Student chooses controls and states what attack or leakage remains.
- Ledger output: threat model, controls, cost paid, residual risk, audit signal.
- Instructor use: make privacy and security measurable engineering constraints.

Implementation notes:

- Reuse responsible engineering concepts but keep threat modeling explicit.
- Track-specific defaults should make TinyML and Mobile cases distinct, not just smaller cloud cases.

### V2-14 - Robust AI: The Robustness Budget

Suggested reading:

- Volume 2, Robust AI

Core systems question:

- How much robustness should the system buy, and what performance, cost, or complexity does that introduce?

Track adaptation:

- Mobile ML: device conditions, user behavior, and network fallback affect robustness.
- TinyML: sensor noise, physical environment, and rare events dominate robustness.
- Edge AI: weather, lighting, site differences, and hardware degradation create drift.
- Cloud/Scale: adversarial inputs, distribution shift, and cascading failures dominate.

Part A - Robustness Tax:

- Student adds robust training, augmentation, ensembling, abstention, or filtering.
- Knobs: perturbation level, augmentation strength, ensemble size, abstention threshold.
- MLSysIM source: robustness-quality-cost model.
- Visual output: clean accuracy, robust accuracy, latency, and cost trade-off chart.

Part B - Drift And Silent Error:

- Student simulates distribution shift and sees how errors accumulate before detection.
- Knobs: shift speed, monitor sensitivity, retraining delay, subgroup exposure.
- MLSysIM source: drift and robustness model.
- Visual output: timeline of true performance, observed signal, and accumulated harm/cost.

Part C - Defense Stack:

- Student chooses robustness defenses and identifies operational signals needed.
- Ledger output: defense stack, expected tax, monitored failure, fallback policy.
- Instructor use: show robustness as a budgeted design choice rather than a slogan.

Implementation notes:

- Link to V1-14 drift and V2-11 operations.
- Track-specific scenarios should expose different failure modes.

### V2-15 - Sustainable AI: The Carbon Budget

Suggested reading:

- Volume 2, Sustainable AI

Core systems question:

- How should energy, carbon, geography, hardware lifecycle, and rebound effects shape ML systems design?

Track adaptation:

- Mobile ML: battery impact and device replacement pressure matter.
- TinyML: lifetime energy and maintenance trips matter.
- Edge AI: local power, thermal design, and site energy mix matter.
- Cloud/Scale: datacenter placement, utilization, hardware lifecycle, and grid carbon dominate.

Part A - Energy And Carbon Measurement:

- Student estimates operational energy and carbon for the selected scenario.
- Knobs: hardware, utilization, runtime, grid carbon intensity, precision, batch size.
- MLSysIM source: energy and carbon model.
- Visual output: energy/carbon budget stack.

Part B - Placement And Lifecycle Frontier:

- Student compares geography, hardware refresh, utilization, and workload shifting.
- Knobs: region, time shifting, hardware efficiency, embodied carbon, utilization target.
- MLSysIM source: lifecycle carbon and placement model.
- Visual output: carbon versus cost/latency frontier.

Part C - Carbon-Aware Policy:

- Student chooses carbon-aware scheduling, compression, placement, or model-size policy.
- Ledger output: carbon target, chosen policy, cost/latency trade-off, rebound risk.
- Instructor use: connect sustainability to concrete systems knobs.

Implementation notes:

- Avoid one-dimensional "carbon calculator" behavior. The key is trade-off with latency, cost, and reliability.
- Add region and time-dependent carbon data as scenario inputs in MLSysIM.

### V2-16 - Responsible AI At Scale: The Fairness Budget

Suggested reading:

- Volume 2, Responsible AI

Core systems question:

- How do fairness, governance, feedback loops, audits, and platform overhead change at scale?

Track adaptation:

- Mobile ML: personalized experiences create uneven outcomes and consent constraints.
- TinyML: limited observability makes fairness and failure auditing difficult.
- Edge AI: site-specific populations create local fairness issues.
- Cloud/Scale: large-scale decisions create feedback loops, governance load, and audit complexity.

Part A - Impossibility And Feedback:

- Student adjusts thresholds or policies and sees metric conflicts plus feedback effects.
- Knobs: subgroup prevalence, threshold, intervention policy, feedback strength.
- MLSysIM source: fairness and feedback-loop model.
- Visual output: fairness metric conflict chart plus feedback timeline.

Part B - Governance Overhead:

- Student adds audit frequency, documentation, approval gates, red-team review, and incident response.
- Knobs: audit depth, approval latency, monitoring coverage, escalation threshold.
- MLSysIM source: governance cost and risk model.
- Visual output: risk reduction versus operational overhead frontier.

Part C - Responsible AI Pipeline:

- Student chooses an audit and governance pipeline for the track.
- Ledger output: fairness metric, audit pipeline, governance overhead, unresolved conflict.
- Instructor use: distinguish responsible engineering at product scale from single-model evaluation.

Implementation notes:

- Reuse V1-15 fairness mechanics but add feedback loops and governance scaling.
- Track-specific scenarios should change who is affected and how harms are observed.

### V2-17 - Conclusion: The Fleet Synthesis

Suggested reading:

- Volume 2 conclusion

Core systems question:

- Given all prior decisions, what fleet architecture survives scale, failures, operations, security, responsibility, and sustainability constraints?

Track adaptation:

- Mobile ML: synthesize app rollout, local/cloud inference, privacy, battery, and telemetry.
- TinyML: synthesize sensor fleet, battery, update, robustness, and maintenance policy.
- Edge AI: synthesize site fleet, orchestration, drift, stream density, and local reliability.
- Cloud/Scale: synthesize compute, network, storage, training, inference, ops, and governance.

Part A - Fleet Ledger Replay:

- Student loads Volume 2 decisions and sees the implied fleet architecture.
- Knobs: track, fleet size, demand growth, failure budget, carbon target.
- Source: `mlsysbook_labs` design ledger plus MLSysIM fleet-scale solver.
- Visual output: system architecture map with constraints and saved decisions.

Part B - Interaction Map:

- Student perturbs one constraint and sees which decisions interact.
- Knobs: request growth, region shift, hardware shortage, failure rate, privacy rule, carbon cap.
- MLSysIM source: sensitivity and interaction model.
- Visual output: principle interaction map showing reinforcing and conflicting choices.

Part C - Final Design Review:

- Student creates a final architecture blueprint and names the top risk.
- Ledger output: final fleet architecture, top risk, mitigation, principle learned.
- Instructor use: make Volume 2 cumulative through engineering judgment, not a large project.

Implementation notes:

- Requires consistent Volume 2 ledger schemas.
- Provide generated ledger presets for students who only complete this lab independently.

## Chapter Mapping And Naming Cleanup

Recommended cleanup before implementation:

- Rename "Co-Labs" references in indexes to "Interactive Trade-Off Labs" or "MLSysIM Labs".
- Remove time estimates from index, template, and any lab body text.
- Update `labs/index.qmd` to describe the learning-nugget model.
- Update `labs/PROTOCOL.md` and `labs/TEMPLATE.md` to match the new lab contract.
- Replace current `mlsysim.labs` imports with a separate lab helper package such as `mlsysbook_labs`.
- Confirm Volume 2 lab mapping against the current book order:
  - Add a dedicated Collective Communication lab.
  - Refocus the current communication lab on Network Fabrics.
  - Current V2-08 and V2-09 ordering should be checked against book chapter order because the book lists Performance Engineering before Inference in one inventory, while current labs list Inference before Performance Engineering.
  - Update lab IDs, index links, and chapter references together.

## Implementation Roadmap

### Phase 1 - Package Boundary

- Create `mlsysbook_labs` under the labs tree or as a separate package if distribution needs require it.
- Move design ledger, Marimo UI components, lab CSS, plot themes, chapter metadata, and track selector code out of `mlsysim.labs`.
- Update every notebook import from `mlsysim.labs.*` to `mlsysbook_labs.*`.
- Keep MLSysIM free of Marimo, CSS, student state, chapter metadata, and book-specific language.
- Update tests so the browser wheel no longer requires `mlsysim.labs`; instead the labs bundle should include `mlsysbook_labs`.

### Phase 2 - Shared Lab Infrastructure

- Add MLSysIM deployment scenario definitions.
- Add shared Marimo components in `mlsysbook_labs` for chapter recap, track selector, scenario header, knob panel, poster view, decision card, advanced knob drawer, and report export.
- Add typed ledger schema and validation.
- Add report schema and Markdown export built from ledger entries plus serializable MLSysIM result snapshots.
- Add instructor/adoption metadata schema: why assign, where it fits, report expectation, rubric, misconceptions, discussion prompts, extensions, and setup notes.
- Move duplicated formulas from notebooks into MLSysIM helpers.
- Update lab template, protocol, and index language.

### Phase 3 - Pilot Labs

Pilot three labs that exercise different parts of the system:

- V1-01: triad diagnosis and intervention frontier.
- V1-05: operator/activation cost and memory cliff.
- V2-10: inference serving, KV cache, batching, and fleet cost.

The pilot should prove:

- Track switching changes scenario values and computed results.
- The learning-nugget flow works in Marimo.
- The opening recap orients students who have not just read the chapter.
- MLSysIM produces the trade-off data.
- Ledger entries are saved and reloadable.
- A student can download a simple completion report from the saved ledger and MLSysIM snapshot.
- Instructor-facing assignment text and rubric exist for the pilot.
- Browser smoke tests cover the new component path.

### Phase 4 - Volume 1 Conversion

- Convert V1-00 through V1-16 to the shared track and nugget model.
- Preserve the strongest existing visuals, but connect them to MLSysIM result objects.
- Add chapter links and independent-study framing.
- Ensure V1-16 can synthesize prior ledger decisions or generate a preset ledger.

### Phase 5 - Volume 2 Conversion

- Resolve chapter mapping first, including the dedicated Collective Communication lab.
- Convert V2-01 through V2-17 to the shared track and nugget model.
- Reuse Volume 1 primitives wherever possible: roofline, serving, ops, fairness, data movement, compression, and ledger replay.
- Ensure V2-17 can synthesize prior ledger decisions or generate a preset ledger.

### Phase 6 - Verification And Instructor Support

- Add a smoke test that imports every lab, renders track controls, and runs at least one MLSysIM-backed computation.
- Add a static test that rejects time-estimate language and old "Co-Lab" references.
- Add a ledger schema test for every lab.
- Add a report-export test for every lab.
- Add an instructor guide listing, for each lab, why to assign it, where it fits, the key knobs, expected misconceptions, report expectations, rubric, and suggested discussion prompts.

## Immediate Next Work

The next implementation step is to split the lab helper code from MLSysIM, then update shared lab infrastructure and pilot the new pattern in three labs:

- V1-01 because it already uses `Engine.solve` and teaches the basic systems framing.
- V1-05 because it should force MLSysIM to own operator and activation formulas.
- V2-10 because serving and inference expose track differences clearly and reuse Volume 1 concepts.

After the pilots are working, the remaining labs can be converted chapter by chapter with a stable contract instead of each notebook inventing its own structure.
