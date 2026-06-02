---
title: "Labs Moonshot Review Matrix"
subtitle: "Chapter coverage, simulated adoption feedback, and what still has to be added for a signature MLSysBook labs release."
---

# Executive Read

The current draft is technically strong: all 34 labs render, export, and boot in the browser. The pedagogical system is not yet fully complete. The difference matters.

The release can become the signature ML systems lab system if we finish one more systematic pass:

1. Migrate every lab to the same opening pattern: chapter recap, systems translation, track choice, scenario brief, nugget map, report target.
2. Add report export and version metadata visibly to every lab, not only the new collective communication lab.
3. Separate Volume II Chapter 3 and Chapter 6 cleanly: network fabrics should teach topology, bisection, latency, bandwidth, and physical fabric choices; collective communication should teach AllReduce/AllGather/ReduceScatter algorithms, hierarchy, overlap, compression, and residual risk.
4. Add instructor-facing assignment metadata: why assign it, what students submit, rubric, discussion prompts, and common misconceptions.
5. Move recurring formulas, sweeps, frontiers, and result schemas into MLSysIM so the textbook and labs share one engine.

## Simulated Review Feedback

| Reviewer | What They Like | What They Need Before Global Adoption |
|---|---|---|
| First-time student | Labs open in browser; prediction locks and sliders make trade-offs visible; light style is more professional. | A consistent mini recap at the start of every lab, clear track selection, and a downloadable report button so they know what to submit. |
| Strong systems student | The knobs expose real constraints: memory, bandwidth, p99, failures, carbon, fairness, queueing, topology. | More explicit assumptions, advanced knobs behind disclosures, and a clearer connection between each knob and the binding resource. |
| Instructor adopting one lab | The labs are self-contained enough to assign without local setup. | A short assignment card, grading rubric, expected student report, and common wrong answers for every lab. |
| Instructor adopting the sequence | The chapter-to-lab structure is coherent and the capstones can become portfolio artifacts. | Stable version IDs, old-to-new lab numbering map, and confidence that lab content tracks the current book chapters. |
| Maintainer | The new `mlsysbook_labs` package gives the right separation from MLSysIM. | More MLSysIM typed APIs for scenario defaults, sweeps, bottleneck attribution, report snapshots, and reusable frontier computations. |

## Moonshot Readiness Key

| Rating | Meaning |
|---|---|
| A | Concept aligned and technically working; needs only wrapper/report migration. |
| B | Concept aligned, but needs stronger narrative, tracks, or instructor metadata. |
| C | Technically working but pedagogically needs a focused redesign before we call it signature. |

# Chapter-To-Lab Matrix

| Chapter / Lab | Book Core Concepts | Current Lab Coverage | What To Add For The Moonshot Release | Readiness |
|---|---|---|---|---|
| V1-00 Orientation / The Architect's Portal | Lab ritual, prediction before simulation, design ledger, systems identity. | Introduces the lab workflow and prediction discipline. | Add a report-export demo, version badge explanation, and track-selection preview so students understand the whole course pattern before Lab 01. | B |
| V1-01 Introduction / The AI Triad | AI paradigm shift, D-A-M coupling, infrastructure bottleneck, model as part of a system. | Strong coverage through Data-Algorithm-Machine diagnosis, Iron Law surprise, deployment spectrum. | Add consistent chapter recap, track selector, report export, and explicit "AI systems engineer" reflection prompt. | A |
| V1-02 ML Systems / Physics of Deployment | Deployment paradigms, physical constraints, memory wall, light barrier, power wall, energy wall. | Strong coverage of memory, light, power, and energy walls. | Track-specific constraints: mobile battery, TinyML SRAM, edge uplink, cloud power/cost. Add binding-resource summary card. | A |
| V1-03 ML Workflow / Constraint Tax | Lifecycle stages, iteration velocity, constraint propagation, feedback loops, late discovery cost. | Strong workflow/system-cost framing across constraints and feedback loops. | Add roles/stakeholders by track and a final workflow policy decision report. | A |
| V1-04 Data Engineering / Data Gravity | Dataset compilation, data movement, data gravity, entropy, feeding tax, data pipelines. | Strong coverage of feeding tax, data gravity, cascades, false positives. | Add MLSysIM data-source and data-pipeline scenario schemas; add tracks for on-device data, edge streams, cloud object stores. | A |
| V1-05 Neural Computation / Activation Tax | Operators, activations, arithmetic, memory movement, forward/backward cost. | Covers transistor tax, memory cliff, width-squared behavior, forward/backward cost. | Move operator-level calculations into MLSysIM result schemas; add table fallback for each operator budget. | A |
| V1-06 Network Architectures / Architecture Tax | Inductive bias, arithmetic intensity, workload signatures, MLP/CNN/Transformer trade-offs. | Covers structure vs no structure, quadratic wall, depth/width, workload signatures. | Add track-aware architecture defaults and a "why this architecture on this target" report decision. | A |
| V1-07 ML Frameworks / Framework Tax | Abstraction ladder, computational graph, execution strategies, dispatch, fusion, compilation. | Covers dispatch tax, fusion, compilation break-even, deployment spectrum. | Add runtime catalog in MLSysIM or `mlsysbook_labs`: eager, graph, compiled, mobile runtime, browser runtime. | A |
| V1-08 Model Training / Training Gauntlet | Training pipeline, accelerator bubbles, memory budget, mixed precision, communication tax. | Covers memory budget, pipeline bottleneck, mixed precision, communication. | Link explicitly to V2 distributed training and collectives; add "single-node vs distributed transition" reflection. | A |
| V1-09 Data Selection / Selection Paradox | Information-compute ratio, selection frontier, static pruning, preprocessing cost, scaling asymmetry. | Covers ICR frontier, selection inequality, preprocessing tax, scaling frontier. | Add dataset quality/drift scenarios and report field for "what data would I collect next?" | B |
| V1-10 Model Compression / Compression Paradox | Quantization, pruning, distillation, deployment context, accuracy/latency/memory trade-off. | Covers quantization, pruning, Pareto frontier, energy, distillation. | Add hardware kernel support and calibration/robustness risks; add cross-track model viability chart. | A |
| V1-11 Hardware Acceleration / Hardware Roofline | Hardware specialization, roofline, memory hierarchy, tensor cores, tiling, energy. | Strong roofline, fusion, balance shift, energy roofline, tiling. | Move repeated roofline/tiling calculations fully into MLSysIM APIs; add accelerator comparison catalog. | A |
| V1-12 Benchmarking / Benchmarking Trap | Benchmark design, peak vs sustained performance, energy benchmarks, tail behavior, workload validity. | Covers Amdahl ceiling, thermal cliff, multi-metric trap, tail latency. | Add benchmark methodology checklist and "benchmark claim vs production claim" report section. | A |
| V1-13 Model Serving / Tail Latency Trap | Serving architectures, load balancing, batching, p99, cold start, memory wall. | Covers p99 explosion, batching, LLM memory wall, cold start. | Add track-specific serving contexts: mobile on-device, edge gateway, cloud API. Add SLA report export. | A |
| V1-14 ML Operations / Silent Degradation | MLOps, technical debt, correction cascades, monitoring, drift, retraining. | Covers silent drift, retraining cadence, deployment cost asymmetry, debt cascade. | Add monitoring policy card and a compact incident postmortem report. | A |
| V1-15 Responsible Engineering / No Free Fairness | Responsibility as systems engineering, fairness, explainability, carbon, silent failures. | Covers fairness illusion, fairness cost, explainability tax, carbon ledger. | Add governance/stakeholder dimensions and report fields for "who is harmed by this trade-off?" | B |
| V1-16 Conclusion / Architect's Audit | Constraint propagation, quantitative invariants, synthesis across foundations/build/optimize/deploy. | Strong capstone coverage: cost/token, conservation of complexity, design ledger, Amdahl, cascades. | Connect to real design ledger import/export; generate final Volume I architecture memo. | A |
| V2-01 Introduction / Scale Illusion | Scale moment, fleet stack, communication dominance, routine failure, coordination tax. | Covers reliability collapse, coordination tax, scaling law budget, Iron Law at scale, C-cubed. | Add a track selector that makes cloud/fleet the default but lets edge/mobile compare against scale. | A |
| V2-02 Compute Infrastructure / Compute Wall | Accelerator spectrum, HBM, roofline, tensor cores, memory wall, node/rack economics. | Covers memory wall, roofline diagnostic, bandwidth staircase, node memory, TCO. | Add rack-level power/cooling and procurement assumptions through MLSysIM infrastructure schemas. | A |
| V2-03 Network Fabrics / Network Fabric Design | Wire/link, topology, bisection, latency, bandwidth, oversubscription, fabric hierarchy. | Current lab is technically working but still teaches a lot of collective/allreduce content. | Redesign to focus on fabric choices: Ethernet vs InfiniBand vs NVLink, topology, bisection bandwidth, oversubscription, tail congestion. Move collective algorithm details to V2-06. | C |
| V2-04 Data Storage / Data Pipeline Wall | Storage hierarchy, GPU starvation, data tiers, shard contention, checkpoint economics. | Covers storage-compute chasm, pipeline equation, shard contention, stall diagnostic, checkpoints. | Add real trace-style workloads and storage-tier catalog in MLSysIM. | A |
| V2-05 Distributed Training / Parallelism Puzzle | Data/tensor/pipeline parallelism, ZeRO, cluster physics, parallelism strategy selection. | Covers communication wall, ZeRO memory trap, 3D parallelism, hardware tier comparison. | Keep collective math lighter now that V2-06 exists; add explicit transition: parallelism creates communication requirements. | B |
| V2-06 Collective Communication / Collective Communication | AllReduce, AllGather, ReduceScatter, alpha-beta/LogP, ring/tree/hierarchy, overlap, compression. | New lab covers operation anatomy, algorithm frontier, hierarchy, overlap/compression; uses MLSysIM communication physics. | Promote from preview to stable after content review; add track variants and richer report export across all parts. | A |
| V2-07 Fault Tolerance / When Failure Is Routine | Failure probability, Young-Daly checkpointing, checkpoint storms, recovery policies, serving reliability. | Covers Young-Daly, checkpoint storms, async checkpointing, serving fault tolerance. | Add failure taxonomy selector and a final recovery playbook report. | A |
| V2-08 Fleet Orchestration / Scheduling Trap | Scheduling objectives, bin packing, utilization paradox, preemption, heterogeneous fleets. | Covers queuing wall, fragmentation, preemption cost, heterogeneous fleet. | Add scheduler policy catalog and instructor prompts about utilization vs responsiveness. | A |
| V2-09 Performance Engineering / Optimization Trap | Efficiency frontier, roofline, memory hierarchy, fusion, FlashAttention, precision. | Covers roofline, fusion, FlashAttention, precision engineering, optimization playbook. | Add "optimize only the active bottleneck" decision rubric and MLSysIM bottleneck attribution output. | A |
| V2-10 Inference at Scale / Inference Economy | Training/inference cost inversion, serving tax, KV cache, continuous batching, fleet design. | Covers serving cost inversion, KV cache wall, continuous batching, fleet design. | Add separate student report for capacity plan, SLA, and cost curve. | A |
| V2-11 Edge Intelligence / Edge Thermodynamics | On-device/edge learning, adaptation, federation, battery, memory amplification. | Covers memory amplification, adaptation strategy, battery drain, federation paradox. | Add stronger track split: Mobile ML vs TinyML vs Edge AI should visibly change scenario and limits. | B |
| V2-12 Ops at Scale / Silent Fleet | N-model problem, platform economics, platform ROI, canaries, alert fatigue. | Covers complexity explosion, silent failure tax, platform ROI, canary duration, alert fatigue. | Add instructor-ready incident discussion and platform build-vs-buy decision card. | A |
| V2-13 Security & Privacy / Price of Privacy | Expanded attack surface, privacy/security definitions, trade-offs, defense overhead, privacy budget. | Covers privacy scaling, privacy-accuracy frontier, defense overhead, privacy budget depletion. | Add explicit threat model selector and privacy/security interaction matrix. | B |
| V2-14 Robust AI / Robustness Budget | Silent failure, distribution drift, defense stack, robustness across cloud/edge/embedded systems. | Covers robustness tax, silent errors, drift timeline, defense stack, compression-robustness collision. | Add failure-mode taxonomy and report field for monitoring assumption. | A |
| V2-15 Sustainable AI / Carbon Budget | Energy ceiling, carbon geography, lifecycle carbon, Jevons paradox, carbon-aware systems. | Covers energy wall, geography of carbon, lifecycle shift, Jevons trap, carbon-aware fleet design. | Add world/carbon map instrument and grid-region provenance through MLSysIM. | A |
| V2-16 Responsible AI / Fairness Budget | Governance, lifecycle responsibility, transparency, fairness, privacy, responsible operations. | Covers impossibility wall, fairness tax, feedback loop, responsible overhead, audit pipeline. | Add governance artifact and instructor rubric for defensible responsible-AI trade-offs. | B |
| V2-17 Conclusion / Fleet Synthesis | Six principles of distributed ML systems, complete production system, competencies. | Strong capstone: sensitivity, failure budget, principle interaction map, fleet blueprint. | Generate a final Volume II fleet design review from the ledger and report schemas. | A |

# Coverage By Durable Systems Principle

| Principle | Coverage Now | Gap To Close |
|---|---|---|
| D-A-M coupling | Strong in V1-01, V1-03, V1-16; implicit elsewhere. | Add explicit D-A-M diagnosis card to more labs so students keep using the lens. |
| Iron Law / physical constraints | Strong across V1-02, V1-11, V1-12, V2-01, V2-09. | Centralize formulas and result schemas in MLSysIM. |
| Data movement dominates | Strong in V1-04, V1-08, V2-03, V2-04, V2-06. | Separate network fabric vs collective algorithm content cleanly. |
| Scale changes behavior | Strong in Volume II. | Add more cross-links from late Volume I to Volume II so students see the transition. |
| Tails and failures matter | Strong in V1-12, V1-13, V2-07, V2-08, V2-12. | Add report prompts that force students to cite p99/failure probability, not averages. |
| Responsible constraints are systems constraints | Strong in V1-15, V2-13, V2-14, V2-15, V2-16. | Add stakeholder and governance report fields. |
| Engineering judgment | Present in prediction/reveal flows and capstones. | Standardize decision cards, residual risk, and report export across every lab. |

# Required MLSysIM Capabilities

| Capability | Why It Matters | Status / Action |
|---|---|---|
| Scenario catalog with track defaults | Tracks should change constraints and defaults, not only labels. | Add Mobile ML, TinyML, Edge AI, Cloud/Fleet scenario defaults. |
| Typed result schemas | Reports need stable evidence snapshots. | Add or consolidate schemas for memory, serving, communication, storage, reliability, carbon, privacy, fairness. |
| Sweep/frontier APIs | Labs should explore trade-off surfaces without notebook-local formulas. | Add `sweep_*`, `frontier_*`, and `binding_constraint` helpers. |
| Bottleneck attribution | Students need to see why the constraint changed. | Add engine-level attribution output. |
| Device/fabric/workload registries | Tracks and chapters need credible defaults. | Extend registries for fabrics, storage tiers, mobile/TinyML devices, edge gateways, grid regions. |
| Report snapshot serialization | Instructor adoption depends on submit-ready artifacts. | MLSysIM should serialize result snapshots; `mlsysbook_labs` formats student reports. |
| Provenance/assumption display | Students and instructors need trust in constants. | Expose assumptions in Math Peek and report metadata. |

# Implementation Priority

| Priority | Work | Why First |
|---|---|---|
| P0 | Refactor V2-03 into a true Network Fabric Design lab and keep V2-06 as Collective Communication. | This is the main concept-drift issue after the book update. |
| P0 | Add visible report export/version/track wrapper to every lab. | This is the signature adoption requirement: "do the lab, download report, submit." |
| P1 | Add instructor metadata to every lab. | Makes the labs assignable worldwide without custom course prep. |
| P1 | Move repeated formulas/sweeps into MLSysIM APIs. | Keeps the textbook and labs aligned to one source of truth. |
| P2 | Add richer interaction devices: frontier plots, heatmaps, carbon/world maps, workload timelines, topology diagrams. | Makes the labs feel immersive while still teaching systems trade-offs. |

# Bottom Line

If we finish the P0 and P1 items above, this becomes more than a set of notebooks. It becomes a coherent lab pedagogy for ML systems engineering: students repeatedly learn to predict, measure, diagnose, decide, and defend trade-offs under real system constraints.
