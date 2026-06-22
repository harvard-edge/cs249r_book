# MLSysIM Lab Activity Blueprint

This document turns the Volume 1 and Volume 2 chapters into an activity-level lab plan. The labs are not mini build projects and they are not deployment kits. They are interactive ML systems trade-off environments. The student should change knobs, see constraints move, and make an engineering decision that would hold up in a design review.

MLSysIM is the single source of truth for calculations, constants, scenarios, and solvers. It is also used by the textbook, so the engine should expose reusable system models that work for book callouts, figures, notebooks, and labs. The labs should not create a second formula layer. The labs layer should provide Marimo UI, track state, chapter mapping, student ledger, and pedagogy.

The implementation contract for the full lab system is in `LAB_SYSTEM_SPECIFICATION.md`. This blueprint provides the chapter/activity mapping that feeds that contract.

## Core Concepts Covered By The Material

### Cross-Cutting ML Systems Concepts

| Concept | What students need to learn | MLSysIM responsibility |
|---|---|---|
| Data, algorithm/model, machine coupling | ML system behavior depends on all three, not any one layer alone | Unified scenario/result object with data, model, workload, and hardware fields |
| Resource budgets | Memory, compute, bandwidth, storage, energy, carbon, cost, reliability, and human ops capacity are finite | Canonical budget calculators and feasibility checks |
| Bottleneck attribution | The system should explain why it fails or slows down | Solver output should name active constraint and next constraint |
| Trade-off frontiers | A good design is usually on a frontier, not at a single optimum | Sweep APIs returning Pareto points and dominated points |
| Deployment context | Mobile, TinyML, edge, and cloud/fleet settings change which knob matters | Scenario registry with track-compatible defaults |
| Workload shape | Request rate, sequence length, image resolution, stream count, and burstiness change feasibility | Workload models and distribution parameters |
| Data movement | Moving bytes often dominates arithmetic | Data transfer, storage, cache, network, and memory traffic models |
| Memory hierarchy | SRAM, DRAM, HBM, NVMe, network, and cache create cliffs | Hardware memory hierarchy models |
| Queueing and tails | Utilization and variability create nonlinear latency | Queueing models with p50/p95/p99 outputs |
| Training scale | Optimizer state, activations, sharding, and communication dominate training | Training memory and throughput solvers |
| Serving scale | Batching, caches, cold starts, routing, and autoscaling dominate inference systems | Serving solver with latency, cost, memory, and utilization outputs |
| Reliability | More nodes, devices, and time increase failure exposure | Reliability, checkpoint, recovery, and useful-work models |
| Operations | Drift, monitoring, canaries, alert fatigue, and retraining shape system quality after deployment | Monitoring, drift, rollout, and platform-cost models |
| Responsible constraints | Privacy, security, fairness, robustness, and sustainability are engineering constraints | Constraint models with measurable quality/cost/latency effects |

### Volume 1 Core Arc

Volume 1 teaches single-system ML engineering: how a model becomes a deployed system and how constraints propagate through data, model, machine, workflow, optimization, serving, operations, and responsibility.

The lab arc should be:

1. Choose a deployment track and constraint lens.
2. Diagnose bottlenecks using the data/model/machine frame.
3. Understand physical walls and deployment feasibility.
4. See how workflow and data pipelines change system behavior.
5. Account for neural computation, architecture, frameworks, training, compression, and acceleration.
6. Validate performance with realistic benchmarks and serving models.
7. Operate and govern the system.
8. Synthesize the full design ledger.

### Volume 2 Core Arc

Volume 2 teaches distributed and fleet-scale ML systems: how the same principles change under scale, coordination, heterogeneity, failure, operations, security, and sustainability.

The lab arc should be:

1. See why scale changes the assumptions.
2. Choose compute, fabric, storage, and parallelism under resource limits.
3. Understand communication and reliability as first-class constraints.
4. Schedule and operate fleets.
5. Optimize and serve at scale.
6. Place intelligence across device, edge, and cloud.
7. Handle security, robustness, sustainability, and responsible AI at scale.
8. Synthesize the fleet architecture.

## Lab Design Contract

Every lab should follow the same recognizable pattern, but the repeated unit should be a learning nugget. A nugget is one focused chapter idea explored through MLSysIM. Some chapters may need one nugget; others may need two or three.

Every lab opens with a wrapper:

| Wrapper element | Purpose |
|---|---|
| Chapter anchor | Names the volume, chapter, and key ideas students should read in the book |
| Track and scenario | Sets Mobile ML, TinyML, Edge AI, or Cloud/Scale context |
| Nugget map | Shows the chapter ideas the lab will explore |
| Synthesis target | States the final engineering decision students will make |

Then each nugget follows the same methodical cycle:

| Activity | Student question | Output |
|---|---|---|
| Chapter idea | Which concept from the chapter am I testing? | Named concept and key principle |
| System setup | What deployment situation am I analyzing? | Track-specific hardware, model, workload, and constraints |
| Prediction | What do I think will happen before seeing results? | Structured prior belief |
| Binding constraint | What breaks first? | Bottleneck attribution and budget view |
| Trade-off surface | What changes when I move the knobs? | Frontier, curve, heatmap, roofline, queueing plot, or feasibility region |
| Engineering decision | What design would I defend? | Configuration or policy choice |
| Reflection | What does this result mean? | Explanation of improved metric, worsened metric, residual risk |
| Ledger update | What should carry forward? | Structured decision, rationale, and result snapshot |

The activity tables below use Part A/B/C labels because that is how many current labs are organized. In the target design, these rows should be interpreted as nugget stages or nugget topics, not as a rigid requirement that every chapter has exactly three top-level parts.

## Volume 1 Activity Plan

| Chapter/Lab | Activity | What will be covered | How it will be covered | MLSysIM needs |
|---|---|---|---|---|
| V1-00 Orientation, Architect's Portal | Track and scenario | Deployment tracks: Mobile ML, TinyML, Edge AI, Cloud/Scale | Student selects a track and sees default model, hardware, workload, and constraints | Scenario registry, track-compatible defaults, hardware/model/workload IDs |
| V1-00 Orientation, Architect's Portal | Part A: Constraint portrait | How each track changes the dominant resource budget | Radar or budget chart comparing memory, latency, energy, cost, reliability, privacy | Track resource budget summaries |
| V1-00 Orientation, Architect's Portal | Part B: Same model, different world | Same model can be feasible in one context and impossible in another | Sweep one model across all tracks and show feasible/infeasible cells | Feasibility solver over multiple scenarios |
| V1-00 Orientation, Architect's Portal | Part C: Engineering lens | Student commits to a track and initial bottleneck hypothesis | Save track, scenario seed, and expected constraint in ledger | Lab layer ledger; MLSysIM scenario ID |
| V1-01 Introduction, The AI Triad | Part A: Diagnose data/model/machine | DAM coupling and bottleneck attribution | Sliders for data quality, model size, and hardware; solver names active bottleneck | DAM scenario result with bottleneck field |
| V1-01 Introduction, The AI Triad | Part B: Intervention frontier | Data investment versus model change versus hardware upgrade | Fixed budget allocation produces accuracy, latency, and cost frontier | Intervention sweep API and cost model |
| V1-01 Introduction, The AI Triad | Part C: Defensible fix | Choosing the first system intervention | Student selects intervention and records why alternatives are worse | Ledger entry references MLSysIM result snapshot |
| V1-02 ML Systems, Physics of Deployment | Part A: First wall | Memory, compute, bandwidth, power, energy, and distance limits | Run scenario against track hardware and show first violated wall | Physical feasibility and wall attribution model |
| V1-02 ML Systems, Physics of Deployment | Part B: Physics curve | Workload scaling against physical limits | Sweep batch, sequence length, image size, or distance; plot threshold crossings | Compute/memory/bandwidth/energy/latency equations |
| V1-02 ML Systems, Physics of Deployment | Part C: Deployment choice | Local, edge, or cloud placement under hard constraints | Student chooses placement and mitigation for avoided wall | Placement feasibility model |
| V1-03 ML Workflow, Constraint Tax | Part A: Constraint propagation | Deployment constraints flow backward into data, training, validation, release | Scenario graph highlights constrained workflow stages | Workflow dependency and stage-cost model |
| V1-03 ML Workflow, Constraint Tax | Part B: Iteration frontier | Iteration speed versus confidence/risk | Sliders for validation depth, data size, automation, hardware; plot iteration/risk curve | Iteration cost and risk model |
| V1-03 ML Workflow, Constraint Tax | Part C: Workflow policy | Release and validation strategy as system design | Student chooses validation, retraining, and release policy | Ledger schema for workflow decisions |
| V1-04 Data Engineering, Data Gravity | Part A: Feed the model | Ingest, preprocessing, transfer, and compute starvation | Pipeline diagram with throughput at each stage and accelerator utilization | Data pipeline throughput model |
| V1-04 Data Engineering, Data Gravity | Part B: Data movement frontier | Move data, move compute, compress, cache, or sample less | Compare placement and cache strategies on cost/latency frontier | Data movement, storage, cache, and placement APIs |
| V1-04 Data Engineering, Data Gravity | Part C: Pipeline architecture | Data architecture decision and residual failure mode | Student chooses preprocessing location, cache policy, and retention | Pipeline configuration result object |
| V1-05 Neural Computation, Activation Tax | Part A: Operation ledger | Parameters, activations, MACs, and bytes moved | Pick layer/network shape and see weights, activations, ops, memory traffic | Operator cost model |
| V1-05 Neural Computation, Activation Tax | Part B: Memory cliff | Activations can dominate weights and exceed memory | Sweep width, resolution, sequence length, batch; show SRAM/DRAM/HBM cliffs | Activation and memory hierarchy model |
| V1-05 Neural Computation, Activation Tax | Part C: Layer design | Fit an operator to a deployment constraint | Student chooses shape/precision/tiling and explains sacrifice | Operator feasibility solver |
| V1-06 Network Architectures, Architecture Tax | Part A: Architecture signature | CNN, MLP, RNN, transformer, efficient variants have different resource signatures | Side-by-side cost table by architecture and track | Architecture cost model |
| V1-06 Network Architectures, Architecture Tax | Part B: Scaling shape | Linear, quadratic, width-squared, and resolution scaling | Sweep the expensive variable and plot growth curves | Architecture scaling functions |
| V1-06 Network Architectures, Architecture Tax | Part C: Architecture choice | Choose architecture based on workload and constraint | Student selects architecture and predicts next scaling failure | Architecture feasibility and risk fields |
| V1-07 ML Frameworks, Framework Tax | Part A: Dispatch tax | Eager execution, kernel launch, synchronization, and transfer overhead | Many-small-ops versus fused-ops latency stack | Runtime overhead model |
| V1-07 ML Frameworks, Framework Tax | Part B: Fusion and compile break-even | Compilation/fusion pays off only after enough reuse | Sweep inference count, shape dynamism, fusion depth; plot break-even | Compile amortization and fusion model |
| V1-07 ML Frameworks, Framework Tax | Part C: Runtime choice | Framework/runtime/delegate selection under deployment constraints | Student chooses runtime path and notes portability/operator risk | Runtime support registry |
| V1-08 Model Training, Training Gauntlet | Part A: Training memory budget | Weights, gradients, optimizer state, activations, and data pipeline | Stack chart shows memory components and infeasible pieces | Training memory model |
| V1-08 Model Training, Training Gauntlet | Part B: Feasibility knobs | Batch, precision, checkpointing, accumulation, optimizer | Sweep knobs to produce speed versus memory frontier | Training throughput/memory solver |
| V1-08 Model Training, Training Gauntlet | Part C: Training plan | Choose fine-tuning/training/adaptation strategy | Student records plan, convergence risk, and hidden cost | Training plan result schema |
| V1-09 Data Selection, Selection Paradox | Part A: Quality versus quantity | More data versus better data versus cheaper data | Dataset size/noise/label quality sliders produce utility-cost frontier | Data utility and training-cost model |
| V1-09 Data Selection, Selection Paradox | Part B: Coverage and inequality | Overall accuracy can hide subgroup failure | Selection policy changes subgroup heatmap and global metric | Coverage/subgroup risk model |
| V1-09 Data Selection, Selection Paradox | Part C: Data policy | Acquisition, curation, filtering, and rare-event policy | Student chooses data policy and accepted bias/coverage risk | Data policy schema for later responsible labs |
| V1-10 Model Compression, Compression Paradox | Part A: Compression feasibility | Quantization, pruning, distillation, and hardware support | Apply methods and see size, latency, energy, and quality deltas | Compression model and kernel support registry |
| V1-10 Model Compression, Compression Paradox | Part B: Compression frontier | Smaller can be slower or less robust if unsupported | Sweep bit width/sparsity/student size; show Pareto frontier | Compression-quality-runtime sweep |
| V1-10 Model Compression, Compression Paradox | Part C: Compression recipe | Choose validated compression strategy | Student selects recipe and validation test before deployment | Compression result and validation schema |
| V1-11 Hardware Acceleration, Hardware Roofline | Part A: Roofline diagnosis | Compute-bound, bandwidth-bound, and memory-bound regimes | Plot workload point on roofline with active regime | Roofline model |
| V1-11 Hardware Acceleration, Hardware Roofline | Part B: Move the point | Tiling, fusion, batching, precision, and reuse | Before/after roofline position plus latency/energy delta | Arithmetic intensity and memory traffic transforms |
| V1-11 Hardware Acceleration, Hardware Roofline | Part C: Accelerator decision | Choose CPU/GPU/NPU/TPU/MCU accelerator | Student selects hardware and remaining limitation | Hardware comparison and feasibility solver |
| V1-12 Benchmarking, Benchmarking Trap | Part A: Benchmark illusion | Peak, component, warm, and deployment-like benchmarks disagree | Compare easy benchmark to deployment workload | Benchmark workload model |
| V1-12 Benchmarking, Benchmarking Trap | Part B: Multi-metric trade-off | Throughput, latency, energy, accuracy, cost, and tails conflict | Optimize one metric and watch others move | Metric aggregation and queueing/thermal model |
| V1-12 Benchmarking, Benchmarking Trap | Part C: Benchmark protocol | Design a benchmark that catches the real failure | Student selects workload, duration, warmup, metrics, guardrails | Benchmark protocol schema |
| V1-13 Model Serving, Tail Latency Trap | Part A: Queueing failure | Utilization and burstiness explode tail latency | Arrival/service sliders produce p50/p95/p99 curve | Queueing model |
| V1-13 Model Serving, Tail Latency Trap | Part B: Serving knobs | Batching, autoscaling, cache, replicas, cold starts | Sweep policies to plot latency/cost/throughput frontier | Serving solver |
| V1-13 Model Serving, Tail Latency Trap | Part C: Capacity plan | Choose serving configuration and protected failure mode | Student records batching, autoscaling, cache, and risk | Serving configuration schema |
| V1-14 ML Operations, Silent Degradation | Part A: Drift visibility | Drift may happen before labels or monitors notice it | Timeline of true quality, observed signal, alert, damage | Drift and monitoring model |
| V1-14 ML Operations, Silent Degradation | Part B: Retraining cadence | Undertraining and overtraining both cost money/risk | Retrain frequency sweep gives total cost curve | Retraining cost/risk model |
| V1-14 ML Operations, Silent Degradation | Part C: Ops policy | Monitoring, retraining, rollback, escalation | Student chooses operational policy and residual blind spot | Ops policy schema |
| V1-15 Responsible Engineering, No Free Fairness | Part A: Metric conflict | Fairness metrics can be mutually incompatible | Threshold sliders show accuracy, calibration, equality, subgroup error | Fairness metric model |
| V1-15 Responsible Engineering, No Free Fairness | Part B: Responsibility budget | Privacy, explainability, robustness, carbon add system overhead | Add constraints and see quality/cost/latency/fairness stack | Responsible constraint overhead models |
| V1-15 Responsible Engineering, No Free Fairness | Part C: Responsible decision | Pick obligation, system change, cost, and evidence | Student records responsible policy and audit signal | Responsible decision schema |
| V1-16 Conclusion, Architect's Audit | Part A: Ledger replay | Prior decisions imply an end-to-end architecture | Load ledger or preset and render architecture map | Lab ledger plus MLSysIM scenario solver |
| V1-16 Conclusion, Architect's Audit | Part B: Sensitivity audit | Decisions break under workload, model, or constraint changes | Perturb knobs and show fragility heatmap | Sensitivity solver |
| V1-16 Conclusion, Architect's Audit | Part C: Architecture memo | Revise one decision and state the principle | Student writes final architecture decision and top risk | Result snapshot export |

## Volume 2 Activity Plan

| Chapter/Lab | Activity | What will be covered | How it will be covered | MLSysIM needs |
|---|---|---|---|---|
| V2-01 Introduction, Scale Illusion | Part A: Scaling illusion | Linear assumptions fail as devices/users/accelerators grow | Sweep fleet/cluster size and show reliability/cost collapse | Scale reliability and cost model |
| V2-01 Introduction, Scale Illusion | Part B: Coordination tax | Coordination reduces useful work | Sliders for sync, monitoring, retries, heterogeneity; useful-work plot | Coordination overhead model |
| V2-01 Introduction, Scale Illusion | Part C: Scale readiness | Decide whether to scale, shard, specialize, or simplify | Student chooses scale plan and first mitigation | Scale decision schema |
| V2-02 Compute Infrastructure, Compute Wall | Part A: Node feasibility | Hardware fit across memory, bandwidth, compute, power | Run workload on candidate hardware and show active wall | Hardware feasibility solver |
| V2-02 Compute Infrastructure, Compute Wall | Part B: Infrastructure frontier | TCO, latency, throughput, utilization, power | Sweep accelerator tier/node count/utilization and plot frontier | Infrastructure TCO and capacity model |
| V2-02 Compute Infrastructure, Compute Wall | Part C: Procurement/placement | Choose hardware tier and invalidation assumption | Student records hardware plan and risk trigger | Hardware plan schema |
| V2-03 Network Fabrics, Fabric Design | Part A: Fabric budget | Link bandwidth, latency, topology, bisection cliff | Compare HBM/NVLink/InfiniBand/Ethernet budgets | Network fabric model |
| V2-03 Network Fabrics, Fabric Design | Part B: Topology frontier | Fat tree, torus, dragonfly, Ethernet, InfiniBand, and hierarchy trade-offs | Sweep node count, bisection bandwidth, hop count, and oversubscription | Topology and network fabric model |
| V2-03 Network Fabrics, Fabric Design | Part C: Fabric decision | Choose network fabric and placement assumption | Student records topology, bandwidth assumption, placement constraint, and risk | Fabric strategy schema |
| V2-04 Data Storage, Data Pipeline Wall | Part A: Storage-compute gap | Storage bandwidth can starve accelerators | Pipeline chart comparing storage/preprocess/compute demand | Storage pipeline model |
| V2-04 Data Storage, Data Pipeline Wall | Part B: Sharding/cache frontier | Shards, prefetch, cache, workers, locality | Sweep storage strategy; plot stall rate versus cost | Sharding/cache/contention model |
| V2-04 Data Storage, Data Pipeline Wall | Part C: Storage architecture | Storage and checkpoint architecture decision | Student chooses shard/cache/checkpoint policy | Storage architecture schema |
| V2-05 Distributed Training, Parallelism Design | Part A: Memory fit | Model states and activations force sharding | Memory stack across weights, gradients, optimizer, activations | Distributed training memory model |
| V2-05 Distributed Training, Parallelism Design | Part B: Parallelism frontier | Data, tensor, pipeline, ZeRO/FSDP, hybrid 3D | Sweep parallel degrees; plot memory/throughput/communication frontier | Distributed training solver |
| V2-05 Distributed Training, Parallelism Design | Part C: Training architecture | Pick parallelism plan and new bottleneck | Student records plan, scaling limit, communication assumption | Parallelism plan schema |
| V2-06 Collective Communication, Communication Lab | Part A: Collective operation anatomy | AllReduce, AllGather, ReduceScatter, Broadcast | Interactive operation diagram with bytes and rounds | Collective operation primitives |
| V2-06 Collective Communication, Communication Lab | Part B: Algorithm/topology frontier | Ring/tree/hierarchical algorithms under real topology | Crossover plot with latency and bandwidth terms | Alpha-beta topology model |
| V2-06 Collective Communication, Communication Lab | Part C: Overlap/compression decision | Compression, overlap, hierarchy, and staleness trade-offs | Student chooses communication optimization and residual risk | Communication optimization schema |
| V2-07 Fault Tolerance, Failure Budget Engineering | Part A: Failure exposure | Fleet size and job duration raise failure probability | Sweep MTBF/fleet/job duration; probability of clean completion | Reliability model |
| V2-07 Fault Tolerance, Failure Budget Engineering | Part B: Recovery frontier | Checkpoint interval, async checkpoints, replication, retries | Useful work versus recovery overhead curve | Checkpoint/recovery model |
| V2-07 Fault Tolerance, Failure Budget Engineering | Part C: Resilience policy | Choose failure budget and recovery strategy | Student records covered and uncovered failures | Resilience policy schema |
| V2-08 Fleet Orchestration, Scheduling Under Constraint | Part A: Queue/utilization wall | High utilization increases wait and SLA failures | Job arrival/size sliders produce queue and utilization chart | Scheduler queueing model |
| V2-08 Fleet Orchestration, Scheduling Under Constraint | Part B: Fragmentation/preemption frontier | Packing, fragmentation, preemption, heterogeneity conflict | Simulate scheduler policies and rejected work | Bin-packing and preemption model |
| V2-08 Fleet Orchestration, Scheduling Under Constraint | Part C: Fleet policy | Choose scheduler, admission, preemption, priority | Student records scheduling policy and fairness trade-off | Scheduling policy schema |
| V2-09 Performance Engineering, Optimization Trap | Part A: Bottleneck diagnosis | Roofline, memory wall, kernel overhead, communication | Profile view with active bottleneck and time breakdown | Performance diagnostic solver |
| V2-09 Performance Engineering, Optimization Trap | Part B: Optimization ladder | Fusion, layout, precision, attention kernels, overlap | Apply optimizations in sequence; waterfall and new bottleneck | Optimization transform model |
| V2-09 Performance Engineering, Optimization Trap | Part C: Stop rule | Stop when marginal gain is not worth cost/risk | Student records optimization order and stop criterion | Optimization decision schema |
| V2-10 Inference at Scale, Serving Cost Inversion | Part A: Cost inversion | Inference can dominate training cost over demand | Cumulative training versus inference cost curve | Inference cost model |
| V2-10 Inference at Scale, Serving Cost Inversion | Part B: State/batching frontier | KV cache, continuous batching, replicas, routing | Sweep sequence/concurrency/cache/batch policy | Inference serving solver |
| V2-10 Inference at Scale, Serving Cost Inversion | Part C: Serving fleet | Choose local/edge/cloud/fleet serving architecture | Student records batching/cache/capacity policy | Serving fleet schema |
| V2-11 Edge Intelligence, Edge Thermodynamics | Part A: Placement feasibility | Device-edge-cloud split under latency, privacy, energy | Pipeline split diagram with latency and energy annotations | Placement solver |
| V2-11 Edge Intelligence, Edge Thermodynamics | Part B: Adaptation/federation frontier | Local adaptation, centralized retrain, federated learning | Compare quality versus communication/energy/privacy | Adaptation and federated communication model |
| V2-11 Edge Intelligence, Edge Thermodynamics | Part C: Edge architecture | Choose local, edge, cloud, or hybrid architecture | Student records placement, adaptation, and privacy assumption | Edge architecture schema |
| V2-12 Ops at Scale, Silent Fleet | Part A: Complexity growth | Models/sites/services outgrow manual operations | Scale model count/site count; operational load chart | Ops complexity model |
| V2-12 Ops at Scale, Silent Fleet | Part B: Canary/automation frontier | Canary duration, false alerts, detection speed, platform ROI | Sweep canary and alert thresholds; plot detection versus false alerts/cost | Canary, alert, platform ROI models |
| V2-12 Ops at Scale, Silent Fleet | Part C: Ops architecture | Monitoring, rollout, alerting, automation investment | Student records ops policy and silent-failure risk | Ops architecture schema |
| V2-13 Security & Privacy, Price of Privacy | Part A: Threat/privacy budget | Attack surface and privacy budget across ML lifecycle | Choose threat model/epsilon/logging; threat map updates | Threat and privacy model |
| V2-13 Security & Privacy, Price of Privacy | Part B: Defense overhead frontier | DP, encryption, secure aggregation, filtering, guardrails | Add defenses; plot security/privacy strength versus quality/latency/cost | Defense overhead model |
| V2-13 Security & Privacy, Price of Privacy | Part C: Security/privacy policy | Pick controls and residual risk | Student records controls, cost, residual attack/leakage | Security/privacy schema |
| V2-14 Robust AI, Robustness Budget | Part A: Robustness tax | Robust training, augmentation, ensembling, abstention cost | Add defenses and see clean/robust accuracy, latency, cost | Robustness cost model |
| V2-14 Robust AI, Robustness Budget | Part B: Drift/silent error timeline | Distribution shift and silent errors before detection | Drift timeline with monitors and accumulated harm/cost | Drift/robustness model |
| V2-14 Robust AI, Robustness Budget | Part C: Defense stack | Choose defenses, fallback, and monitoring signal | Student records defense stack and expected tax | Robust defense schema |
| V2-15 Sustainable AI, Carbon Budget | Part A: Energy/carbon measurement | Operational energy and carbon by hardware/workload/region | Energy/carbon budget stack | Energy and carbon model |
| V2-15 Sustainable AI, Carbon Budget | Part B: Placement/lifecycle frontier | Geography, time shifting, utilization, embodied carbon | Carbon versus cost/latency/reliability frontier | Lifecycle carbon and placement model |
| V2-15 Sustainable AI, Carbon Budget | Part C: Carbon-aware policy | Scheduling, placement, compression, caps, rebound risk | Student records carbon policy and accepted trade-off | Carbon policy schema |
| V2-16 Responsible AI, Fairness Budget | Part A: Metric conflict and feedback | Fairness impossibility and feedback loops at scale | Threshold/policy sliders show metric conflict and feedback timeline | Fairness/feedback model |
| V2-16 Responsible AI, Fairness Budget | Part B: Governance overhead | Audit depth, approval gates, red-team review, incident response | Risk reduction versus operational overhead frontier | Governance cost/risk model |
| V2-16 Responsible AI, Fairness Budget | Part C: Responsible AI pipeline | Choose audit/governance pipeline and unresolved conflict | Student records metric, pipeline, overhead, conflict | Governance pipeline schema |
| V2-17 Conclusion, Fleet Synthesis | Part A: Fleet ledger replay | Prior decisions imply fleet architecture | Load ledger/preset and render architecture map | Lab ledger plus fleet solver |
| V2-17 Conclusion, Fleet Synthesis | Part B: Interaction map | Constraints interact across compute/network/storage/ops/responsibility | Perturb demand, failure, privacy, carbon; show interaction map | Sensitivity and interaction solver |
| V2-17 Conclusion, Fleet Synthesis | Part C: Final design review | Defensible fleet architecture and top risk | Student records final blueprint, top risk, mitigation | Result export schema |

## Numbering Recommendation

The current Volume 2 labs do not cleanly match the book chapter sequence because the book has both Network Fabrics and Collective Communication. The clean pedagogical sequence is:

1. V2-01 Introduction
2. V2-02 Compute Infrastructure
3. V2-03 Network Fabrics
4. V2-04 Data Storage
5. V2-05 Distributed Training
6. V2-06 Collective Communication
7. V2-07 Fault Tolerance
8. V2-08 Fleet Orchestration
9. V2-09 Performance Engineering
10. V2-10 Inference at Scale
11. V2-11 Edge Intelligence
12. V2-12 Operations at Scale
13. V2-13 Security & Privacy
14. V2-14 Robust AI
15. V2-15 Sustainable AI
16. V2-16 Responsible AI
17. V2-17 Fleet Synthesis

Decision: add the dedicated Collective Communication lab and renumber downstream labs as needed. The previous mapping drifted as the book changed, and the improved labs should follow the book structure.

## MLSysIM Implementation Plan

### Engine APIs To Add Or Consolidate

| API area | Purpose | Used by |
|---|---|---|
| `Scenarios` | Track-compatible deployment scenarios, independent of curriculum wording | All labs and book examples |
| `Workloads` | Request, stream, training, storage, and data distributions | Serving, training, storage, benchmarking, ops |
| `Engine.solve()` | Common feasibility, bottleneck, latency, memory, energy, cost result | Most labs |
| `Engine.sweep()` | Produce trade-off curves/frontiers from parameter ranges | Every Part B |
| `BottleneckReport` | Name active constraint, next constraint, and violated budgets | Every Part A |
| `ParetoFrontier` | Mark dominated/non-dominated configurations | Every Part B |
| `SensitivityReport` | Show which design choices break under perturbation | Synthesis labs |
| `TrainingSystemModel` | Memory, throughput, parallelism, optimizer, checkpointing | V1 training, V2 distributed training |
| `ServingSystemModel` | Queueing, batching, cache, replicas, cold starts, cost | V1 serving, V2 inference |
| `CommunicationModel` | Fabric bandwidth/latency, collective algorithms, hierarchy | V2 network, collectives, training |
| `StoragePipelineModel` | Shards, cache, prefetch, checkpoint IO, accelerator starvation | V1 data, V2 storage |
| `ReliabilityModel` | MTBF, failure probability, recovery, useful work | V2 scale, fault tolerance, conclusion |
| `OpsModel` | Drift, monitoring, canary, alert fatigue, platform ROI | V1 MLOps, V2 Ops |
| `ResponsibleSystemsModel` | Fairness, privacy, security, robustness, carbon, governance overhead | Responsible/safety/sustainability labs |

### Package Boundary

MLSysIM should contain:

- Constants and registries used by the book and labs.
- Domain models and formulas.
- Solvers, sweeps, and result schemas.
- Book-friendly helper functions where needed for reproducible figures and callouts.

The lab helper package should contain:

- Marimo UI components.
- Track selector and curriculum mapping.
- Design ledger and browser persistence.
- CSS, Plotly themes, and student-facing layout.
- Chapter/lab metadata and instructions.

The immediate migration target is:

```python
from mlsysim import Engine, Hardware, Models, Scenarios
from mlsysbook_labs import (
    DesignLedger,
    track_selector,
    tradeoff_poster,
    decision_card,
    report_export,
)
```

### Suggested Build Order

1. Split `mlsysim.labs` into `mlsysbook_labs` so MLSysIM stays standalone.
2. Add `Scenario`, `Workload`, `BottleneckReport`, `ParetoFrontier`, and `SensitivityReport` result schemas.
3. Pilot V1-01, V1-05, and V2-10 Inference because they exercise solver, operator, and serving APIs.
4. Add the dedicated V2-06 Collective Communication lab and refocus V2-03 on network fabrics.
5. Add report export from ledger entries plus typed MLSysIM result snapshots.
6. Convert labs chapter-by-chapter using the activity matrix above.
7. Add tests that reject notebook-local duplicate formulas when an MLSysIM API exists.
8. Add smoke tests that run one nugget computation per lab in native Python and browser/WASM.
