# Lab Track Coherence Audit

Date: 2026-06-04

Scope:
- All 34 Volume I and Volume II labs.
- Four canonical student tracks: iPhone, Oura Ring, RoboTaxi, Cloud Fleet.
- Student-facing flow after the track-arc pass: learning objectives, chapter recap, track mission, where-this-fits arc, scenario/case, lab map or part flow, evidence, decision, reflection, report.

## Current Balance

| Track | Category | Device family | Model family | Primary narrative |
|---|---|---|---|---|
| iPhone | Mobile ML | Phone-class mobile SoC | MobileNet-class local models | Private local feature under battery, thermal, memory, and interactive-latency limits. |
| Oura Ring | TinyML / wearable | Ring / MCU-class wearable | DS-CNN and compact anomaly/time-series models | Always-on sensing under SRAM, flash, OTA, radio, and battery limits. |
| RoboTaxi | Edge AI | Vehicle-local autonomy compute | YOLOv8 Nano-class perception and replay workloads | Safety-adjacent edge perception under p99/p999 latency, sensor bandwidth, power, and fallback constraints. |
| Cloud Fleet | Cloud / fleet | H100-backed service cluster | BERT/GPT/Llama-class service workloads | Service operation under throughput, p99 latency, cost/request, utilization, SLA, and carbon constraints. |

The balance is intentionally one track per category. Adding more student-facing tracks would increase narrative surface area faster than it improves learning. Additional devices can still exist in registries as comparison points, but the course spine should stay at four tracks.

## Volume Arc

| Volume | Level | Throughline | Student artifact |
|---|---|---|---|
| Volume I | Senior undergraduate | Can this ML idea survive the physical and operational constraints of the selected deployment context? | A track-specific engineering memo for one chapter decision. |
| Volume II | Senior/Master's/Ph.D. | What breaks when the same selected deployment context becomes a fleet or scaled system? | A system design memo with capacity, reliability, cost, operations, validation, and residual risk. |

## Lab Roles

| Lab | Concept role in the arc | Carry-forward |
|---|---|---|
| V1-00 Architect's Portal | Track selection | Carry the selected track into the Volume I journey. |
| V1-01 AI Triad | Data/algorithm/machine diagnosis | Use the triad to explain later failures. |
| V1-02 Physics of Deployment | Physical deployment envelope | Treat feasibility as a physical budget. |
| V1-03 Constraint Tax | Workflow gate | Move validation earlier in the lifecycle. |
| V1-04 Data Gravity | Data movement and retention | Carry data movement costs into training and operations. |
| V1-05 Activation Tax | Tensor and activation budget | Use tensor cost as the bridge to architecture. |
| V1-06 Architecture Tax | Architecture choice | Carry architecture assumptions into framework/runtime decisions. |
| V1-07 Framework Tax | Runtime and framework support | Use runtime support as a guardrail before training and compression. |
| V1-08 Training Gauntlet | Training plan | Carry validation and handoff requirements into data selection. |
| V1-09 Selection Paradox | Data selection policy | Carry coverage gaps into compression, serving, and responsible engineering. |
| V1-10 Compression Paradox | Compression recipe | Carry the selected recipe into hardware acceleration. |
| V1-11 Hardware Roofline | Hardware roofline | Use bottleneck evidence before benchmarking. |
| V1-12 Benchmarking Trap | Benchmark design | Carry benchmark evidence into serving decisions. |
| V1-13 Tail Latency Trap | Serving policy | Carry serving risk into operations. |
| V1-14 Silent Degradation | Operations and drift | Carry operational risk into responsibility review. |
| V1-15 No Free Fairness | Responsible engineering | Carry unresolved risks into the final audit. |
| V1-16 Architect's Audit | Volume I synthesis | Use the audit as the handoff into Volume II scale. |
| V2-01 Scale Illusion | Scale transition | Carry scale assumptions into infrastructure. |
| V2-02 Compute Wall | Compute infrastructure | Carry the infrastructure wall into communication design. |
| V2-03 Network Fabric Design | Communication fabric | Carry payload pressure into storage and data movement. |
| V2-04 Data Pipeline Wall | Storage and freshness | Carry pipeline limits into distributed training. |
| V2-05 Parallelism Design | Distributed training | Carry training communication into collectives. |
| V2-06 Collective Communication | Collective communication | Carry communication evidence into reliability. |
| V2-07 Failure Budget Engineering | Failure budget | Carry failure modes into orchestration. |
| V2-08 Fleet Orchestration | Scheduling and orchestration | Carry scheduling side effects into optimization. |
| V2-09 Optimization Trap | System optimization | Carry optimization evidence into inference economics. |
| V2-10 Inference Economy | Inference economics | Carry serving economics into edge placement. |
| V2-11 Edge Thermodynamics | Edge placement | Carry placement tradeoffs into fleet monitoring. |
| V2-12 Silent Fleet | Fleet observability | Carry observability into privacy/security decisions. |
| V2-13 Price of Privacy | Privacy and security | Carry security constraints into robustness. |
| V2-14 Robustness Budget | Robustness budget | Carry robustness gaps into sustainability. |
| V2-15 Carbon Budget | Sustainability budget | Carry sustainability tradeoffs into responsibility. |
| V2-16 Fairness Budget | Fleet responsibility | Carry residual risks into the final synthesis. |
| V2-17 Fleet Synthesis | Volume II synthesis | Close the narrative with a deployment review. |

## Simulated Reviewer Feedback

| Reviewer | Positive signal | Concern | Implemented response |
|---|---|---|---|
| Volume I instructor | The four tracks make abstract systems ideas concrete for senior undergraduates. | The opening screen should not feel like a provenance audit. | Added `track_arc_context()` and removed visible source-trace panels from the launch flow. |
| Volume II instructor | The shared renderer gives graduate labs a consistent grammar. | Shared V2 labs need a clearer long-form journey so they do not feel generic. | Added Volume I/II arc text per track and rendered it in shared and direct pages. |
| TA / grader | Reports can still preserve track, scenario, evidence, decision, and caveats. | Visible implementation refs are not useful for grading. | Kept provenance in report internals and tests, not as primary learner copy. |
| Mobile expert | iPhone track uses the right first-order constraints. | Do not imply exact proprietary hardware internals beyond the registered profile. | Dashboard and arc text now speak in device/model families. |
| TinyML / wearable expert | Oura Ring track correctly foregrounds SRAM, flash, OTA, duty cycle, and battery. | Students need repeated reminders that accuracy is not first-order when memory and energy fail. | Oura arc carry-forward explicitly says memory and energy come before accuracy optimization. |
| Edge/autonomy expert | RoboTaxi gives edge systems a concrete, memorable story. | Generic edge and safety-critical autonomy should not be treated as interchangeable. | RoboTaxi arc names safety-adjacent validation and explicitly separates autonomy from generic edge. |
| Cloud/fleet expert | Cloud Fleet covers SLA, utilization, cost, and carbon. | Students may confuse single-accelerator performance with service operation. | Cloud arc carry-forward distinguishes a single accelerator result from service-level operating decisions. |
| Student proxy | Track choice is easier to remember when it recurs in every lab. | Too many boxes in the opening can overwhelm the first read. | The launch path is now track mission, where-this-fits, scenario, then lab map/parts. |

## Release Gates

- Every catalog lab has exactly four canonical track variants.
- Every track variant uses a hardware and model family allowed by its track arc.
- Student-facing launch panels do not display `Source Trace`, `Model source`, `Hardware source`, or `System source`.
- The dashboard renders 4 track cards, 4 arc cards, 34 coverage rows, 14 concept rows, and 103 activity rows.
- Browser smoke must scroll, click track controls where possible, and catch page errors before a lab pass is considered done.

## Current Example

Lab 10 is the most complete visible exemplar:
- The student selects a track at the top.
- Learning objectives use bullets.
- The header tags are split between metadata and chapter concepts; `Track-aware` was removed from the visible tags.
- The student sees the track mission, where-this-fits arc, scenario brief, lab map, five parts, synthesis, big takeaways, and report export.
- The launch flow no longer shows source trace or implementation refs, while the report still records source trace internally.
