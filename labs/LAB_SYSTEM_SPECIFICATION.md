# MLSysBook Signature Lab System Specification

Status: implementation-ready design plan.

This document consolidates the lab redesign into one concrete contract. The goal is to make the MLSysBook labs globally adoptable as signature interactive ML systems labs while keeping MLSysIM as the single source of truth for simulation, formulas, constants, result schemas, sweeps, and reproducible textbook examples.

The labs should be chapter-connected but self-contained. Students should be encouraged to read the relevant chapter, but a lab must not fail pedagogically if a student opens it independently. Every lab should provide a compact recap, focus attention on the relevant ML systems concept, let students explore real trade-offs, and produce a downloadable report that an instructor can collect.

## 1. Core Commitment

The lab system should teach ML systems engineering judgment.

This is more than a set of good examples. The labs should establish a signature pedagogy for AI systems engineering: students learn to see an ML model as part of a machine system, reason from constraints, use evidence from a simulator, and defend engineering trade-offs. The target identity is not "someone who ran a notebook." The target identity is an AI systems engineer: someone who understands data, models, hardware, infrastructure, operations, and responsible constraints as one coupled system.

The recurring student experience should be:

1. Read a compact chapter recap.
2. Choose a deployment track.
3. Understand the scenario and constraints.
4. Predict what will happen.
5. Move a small number of meaningful systems knobs.
6. Observe the binding constraint and trade-off surface.
7. Make an engineering decision.
8. Reflect on what improved, what worsened, and what assumption could break the decision.
9. Save the decision to a design ledger.
10. Download a simple report for submission.

The recurring intellectual pattern should be:

1. Frame the ML idea as a systems question.
2. Identify the resource or constraint that could bind.
3. Predict system behavior before observing output.
4. Use simulation evidence to test the prediction.
5. Inspect the frontier instead of chasing a single optimum.
6. Make a defensible decision under constraints.
7. Name the residual risk and assumption that could change the answer.

The recurring instructor experience should be:

1. See why the lab is worth assigning.
2. Know where it fits in a course.
3. Copy assignment text if needed.
4. Assign the lab without custom infrastructure.
5. Collect a simple student report.
6. Grade with a lightweight rubric focused on systems reasoning.
7. Use discussion prompts to connect the lab back to the chapter.

The recurring maintainer experience should be:

1. Update MLSysIM as the engine.
2. Keep UI and pedagogy in `mlsysbook_labs`.
3. Keep lab behavior testable in Python and browser/WASM.
4. Avoid notebook-local formulas when an MLSysIM API exists.
5. Preserve stable lab IDs, report schemas, and result schemas.

## 2. Non-Negotiable Design Principles

| Principle | Meaning | Design consequence |
|---|---|---|
| MLSysIM is the source of truth | Simulations, formulas, constants, sweeps, and result schemas live in MLSysIM | Labs call typed engine APIs instead of duplicating equations |
| Labs are trade-off explorations | Labs are not TinyTorch build projects and not hardware deployment kits | Interactions emphasize knobs, constraints, frontiers, decisions, and reflection |
| Chapter-connected but self-contained | Students may not have just read the chapter | Every lab opens with a mini recap and chapter-to-systems translation |
| Consistent rhythm | Students should learn the lab pattern once and reuse it | Every lab uses the same wrapper and nugget cycle |
| Track-aware | Mobile, TinyML, edge, and cloud/fleet scenarios change the answer | Track selection changes scenario defaults, constraints, and report context |
| Submit-ready | Instructors need evidence of completion | Each lab exports a lightweight Markdown report, optional JSON snapshot |
| Instructor-adoptable | The labs should be easy to use in courses worldwide | Each lab has assignment text, rubric, misconceptions, discussion prompts, and setup notes |
| Professional academic presentation | The labs should feel like part of the Machine Learning Systems book | Use clean light styling, volume accents, compact headers, and readable visual hierarchy |
| Versioned and stable | Instructors need predictable assignments | Lab IDs, schemas, engine versions, and report versions are explicit |
| Formation-oriented | The goal is AI systems engineering judgment, not widget completion | Each lab teaches a way of thinking: constraints, evidence, trade-offs, and defensible decisions |

## 3. System Boundary

### MLSysIM Owns

MLSysIM should be a standalone engine used by the textbook and labs.

It should contain:

- Hardware, model, dataset, workload, system, and infrastructure registries.
- Scenario definitions that are not book-specific.
- Constants and assumptions with provenance.
- Physics models and solvers.
- Result schemas for performance, memory, serving, communication, training, data, reliability, sustainability, economics, and responsible engineering.
- Sweep APIs and Pareto/frontier utilities.
- Bottleneck attribution and constraint-budget outputs.
- Serialization support for result snapshots.
- Reproducible examples usable in textbook figures and lab outputs.

MLSysIM should not contain:

- Marimo UI components.
- CSS, visual layout, or lab theme.
- Student state or browser persistence.
- Instructor guide text.
- Assignment prompts.
- Book-specific lab numbering or pedagogy.
- Download report formatting.

### `mlsysbook_labs` Owns

The lab helper package should live outside MLSysIM. It can be packaged under the labs tree or as a separate local package depending on distribution needs.

It should contain:

- Marimo UI components.
- Lab header and visual system.
- Chapter recap rendering.
- Track selector and curriculum mapping.
- Scenario brief rendering.
- Nugget shell and progress navigation.
- Prediction, reflection, and decision cards.
- Device catalog wrappers for sliders, frontiers, heatmaps, maps, rooflines, tables, and timelines.
- Design ledger and browser persistence.
- Report export.
- Instructor/adoption metadata.
- Tests for lab structure and UI contracts.

Target import pattern:

```python
from mlsysim import Engine, Hardware, Models, Scenarios
from mlsysbook_labs import (
    DesignLedger,
    chapter_recap,
    track_selector,
    tradeoff_poster,
    decision_card,
    report_export,
)
```

## 4. Lab Wrapper Contract

Every lab should open with the same structure.

| Element | Purpose | Required content |
|---|---|---|
| Lab header | Establish identity and book alignment | Volume, chapter, lab ID, title, track, suggested reading |
| Mini chapter recap | Make the lab self-contained | Chapter emphasis, key terms, ML-to-systems translation, common trap |
| Systems question | Focus the lab | One sentence naming the engineering decision |
| Track selector | Adapt context | Mobile ML, TinyML, Edge AI, Cloud/Fleet, or other approved track |
| Scenario brief | Ground the exercise | Stakeholder, workload, hardware, constraints, objective |
| Nugget map | Preview activities | One to three focused learning nuggets |
| Completion target | Explain what students submit | Report contents and final decision requirement |

The lab opening should not feel like a marketing page or a decorative dashboard. It should read like a polished academic lab handout with interactive instruments.

## 5. Chapter Recap Contract

The chapter recap is not a replacement for reading the book. It is the minimum orientation needed for a student or instructor to use the lab productively.

Every recap should include:

| Field | Description |
|---|---|
| Chapter emphasis | What the chapter is trying to teach |
| Key terms | Three to five terms that appear in the lab |
| ML concept | The ML idea being translated |
| Systems translation | How the idea becomes a resource, deployment, scale, reliability, cost, or operations question |
| What to watch | The constraint or frontier students should pay attention to |
| Common trap | A misconception the lab is designed to correct |
| Suggested reading | Volume and chapter, with key ideas rather than fragile section links |

Example pattern:

```text
Volume I, Chapter 5: Neural Computation
The chapter emphasizes that neural networks consume resources through operators, weights, activations, and memory movement. In this lab, that becomes a systems question: when does the active constraint shift from arithmetic to memory capacity or memory bandwidth? Watch activation memory, arithmetic intensity, and latency as you change batch size, precision, and operator shape. Common trap: assuming parameter count is the whole system cost.
```

## 6. Learning Nugget Contract

The repeated unit is the learning nugget. Current labs may still label activities as Part A, Part B, and Part C, but the target design should not force exactly three top-level parts.

Each nugget should contain:

| Stage | Student action | System output |
|---|---|---|
| Chapter idea | Read the focused recap for this nugget | Key concept and vocabulary |
| Scenario setup | Inspect track-specific workload and constraints | Scenario strip |
| Prediction | State what they expect before seeing results | Locked prediction entry |
| Binding constraint | See what breaks first | Bottleneck report |
| Trade-off surface | Move knobs and compare options | Frontier, curve, heatmap, budget stack, roofline, map, or timeline |
| Engineering decision | Choose a configuration or policy | Decision card |
| Reflection | Explain what improved, what worsened, and what assumption matters | Reflection entry |
| Ledger update | Save evidence and rationale | Typed ledger record |

Default nugget rules:

- Use no more than three primary knobs by default.
- Show units for every knob.
- Show the relevant constraint budget.
- Include one main visual artifact.
- Provide a table fallback for complex charts or maps.
- Include one short reflection prompt.
- Save the selected track, scenario, knob values, results, rationale, and residual risk.
- Hide advanced knobs by default.

## 7. Track Model

Tracks make the same chapter concept behave differently.

Recommended initial tracks:

| Track | Typical context | Constraints that usually matter |
|---|---|---|
| Mobile ML | Phone or app deployment | Latency, battery, memory, privacy, thermal limits |
| TinyML | Microcontroller or sensor deployment | SRAM/flash, energy, sampling rate, model size, intermittent connectivity |
| Edge AI | Gateway, local server, vehicle, or near-user system | Network link, local compute, privacy, reliability, synchronization |
| Cloud/Fleet | Datacenter or global service | Throughput, p99 latency, cost, utilization, reliability, carbon, orchestration |

Track behavior:

- A saved track can be the default for later labs.
- Students should be allowed to switch tracks unless a specific assignment locks the track.
- Track selection should modify scenario defaults, not merely change labels.
- Reports must record the selected track.
- Instructor metadata should explain how the core lesson differs by track.

## 8. Student Report Contract

Each lab should produce a lightweight report that can be downloaded and submitted.

The report is evidence of completion, not a full grading platform.

Recommended primary format:

- Markdown report.

Recommended optional format:

- JSON ledger/result snapshot for automated checking or future tooling.

Required report fields:

| Field | Purpose |
|---|---|
| Student identifier | Allows submission without accounts |
| Lab ID and title | Identifies the assignment |
| Book volume/chapter | Connects to the text |
| Lab version | Supports stable adoption |
| MLSysIM version | Shows engine provenance |
| Track | Records deployment context |
| Scenario summary | Shows the system analyzed |
| Nugget completion summary | Shows the path through the lab |
| Predictions | Shows prior reasoning |
| Binding constraints | Shows simulator evidence |
| Knob settings | Shows what changed |
| Trade-off interpretation | Shows what improved and worsened |
| Selected configuration/policy | Shows the engineering decision |
| Residual risk | Shows judgment and assumptions |
| Result snapshot | Captures typed MLSysIM output |
| Timestamp | Helps instructors interpret reports |

Report generation boundary:

- `mlsysbook_labs` builds the Markdown and optional JSON.
- MLSysIM provides serializable result snapshots.
- The lab notebook wires the current ledger state to the report export component.

## 9. Instructor Adoption Contract

Each lab should have instructor-facing metadata. This should not clutter the student lab, but it should be available in a consolidated instructor guide and optionally in local lab metadata.

Required instructor fields:

| Field | Purpose |
|---|---|
| Why assign this lab | Explains the pedagogical value |
| Where it fits | Connects to book chapter and course module |
| Assignment prompt | Copy-ready instructions |
| Expected report | What students should submit |
| Rubric | Lightweight evaluation dimensions |
| Common misconceptions | What the lab is designed to correct |
| Discussion prompts | How to connect lab results to class discussion |
| Extension ideas | Advanced knobs or optional deeper experiments |
| Setup notes | Browser/local/WASM expectations and known limitations |

Adoption modes:

- Standalone homework: assign one lab and collect reports.
- Lecture companion: run one nugget live and assign the rest.
- Module sequence: assign related labs around serving, data, distributed training, edge, or responsibility.
- Course spine: use Volume 1 and Volume 2 labs as recurring trade-off exercises.
- Capstone preparation: use the design ledger to build an architecture memo or fleet design review.

Instructor value proposition:

These labs let instructors teach ML systems engineering judgment without building a custom simulator, maintaining a cluster, owning specialized hardware, or designing all assignments from scratch.

## 10. Assessment Rubric

A simple rubric should be available for every lab.

Suggested dimensions:

| Dimension | Good answer shows |
|---|---|
| Chapter concept | Uses the chapter idea accurately |
| Bottleneck diagnosis | Identifies the active constraint with evidence |
| Trade-off reasoning | Names what improves and what worsens |
| Systems vocabulary | Uses terms such as latency, throughput, memory, bandwidth, utilization, reliability, carbon, privacy, or tail behavior correctly |
| Decision quality | Selects a defensible configuration or policy |
| Track awareness | Explains why the answer depends on deployment context |
| Residual risk | Names an assumption that could invalidate the result |
| MLSysIM interpretation | Treats simulator output as evidence, not magic |

The rubric should be compact enough for instructors to use quickly.

## 11. Interaction Device Rules

Widgets should be chosen pedagogically.

| Device | Use when students need to understand |
|---|---|
| Slider | A continuous or ordered knob with visible thresholds |
| Range slider | Feasible operating windows |
| Number prediction | Quantitative intuition before simulation |
| Radio/segmented selector | Mutually exclusive strategy choices |
| Dropdown | Scenario, hardware, model, or policy selection |
| Checkbox/multiselect | Additive system features or overheads |
| Constraint budget | Current feasibility against one or more limits |
| Line curve | How one metric changes across a sweep |
| Pareto frontier | Non-dominated engineering options |
| Heatmap | Interaction between two knobs |
| Roofline | Compute versus memory bottlenecks |
| Queueing chart | Utilization, waiting, and tail latency |
| Timeline | Warmup, training, recovery, rollout, or failure evolution |
| World map | Geography, carbon, latency region, or jurisdiction |
| Topology diagram | Cluster/network/collective structure |
| Table | Exact values, comparisons, and accessible fallback |
| Reflection card | Reasoning and residual risk |
| Decision card | Final choice and saved evidence |

Guardrails:

- No random sliders.
- Every visible knob must map to an MLSysIM parameter.
- Every chart should consume a typed MLSysIM result or sweep.
- Every visual should have a short interpretation.
- Every chart with color should also encode state with labels or tables.
- Advanced controls should not be required for report completion.

## 12. MLSysIM Capability Plan

The engine needs capabilities that support both textbook examples and labs.

Priority capability families:

| Capability | Purpose | Lab pressure |
|---|---|---|
| Scenario registry | Track-compatible defaults | All labs |
| Workload profiles | Burstiness, stream count, sequence length, request rate, job size | Serving, edge, benchmarking, orchestration |
| Bottleneck reports | Explain what binds first | All labs |
| Constraint budgets | Memory, latency, energy, cost, reliability, carbon, privacy | All labs |
| Trade-off sweeps | Structured data for curves and frontiers | All labs |
| Pareto frontiers | Distinguish dominated and non-dominated choices | Compression, serving, hardware, training |
| Sensitivity reports | Show fragility of decisions | Synthesis labs |
| Operator cost model | Params, activations, MACs, bytes moved | Neural computation, hardware acceleration |
| Architecture cost model | CNNs, transformers, SSMs, MoE, diffusion, retrieval | Architectures and serving |
| Runtime execution model | Kernel launch, fusion, dispatch, compilation, delegates | Frameworks and performance engineering |
| Benchmark protocol model | Warm/cold, tail, thermal, steady-state, energy | Benchmarking |
| Data selection model | Quality, coverage, label cost, rare events, noise | Data and responsible systems |
| Communication model | Fabrics, collectives, hierarchy, overlap, compression | V2 network, collectives, distributed training |
| Storage pipeline model | Sharding, cache, prefetch, checkpoint IO, stalls | Data storage and training |
| Ops model | Drift, canaries, monitors, retraining, alert fatigue | MLOps |
| Responsible systems model | Fairness, privacy, security, robustness, governance | Responsible AI and robust AI |
| Sustainability model | Energy, carbon, placement, PUE, time shifting | Sustainable AI |
| Serialization/provenance | Reports and textbook traceability | All labs |

## 13. Volume 1 Lab Plan

Volume 1 should teach the foundations of ML systems thinking: the data-algorithm-machine coupling, resource budgets, model cost, training, serving, deployment, monitoring, and responsible constraints.

| Lab | Recap focus | Main systems question | Primary trade-off |
|---|---|---|---|
| V1-00 Architect's Portal | Tracks and scenarios | Which deployment context changes the meaning of "good"? | Track constraints and system goals |
| V1-01 Introduction | Data, algorithm, machine coupling | What intervention fixes the actual bottleneck? | Improve data, model, or hardware under budget |
| V1-02 Lifecycle | Workflow and iteration | Where should engineering effort go in an ML lifecycle? | Speed, confidence, cost, and risk |
| V1-03 Evaluation | Metrics under deployment | Which metric matters for the system objective? | Accuracy, latency, fairness, cost, and tail behavior |
| V1-04 Data Engineering | Data gravity | When does data movement dominate learning? | Storage, preprocessing, transfer, and freshness |
| V1-05 Neural Computation | Operators and memory movement | When do activations, weights, arithmetic, or bandwidth bind? | Batch, precision, operator shape, memory |
| V1-06 Architectures | Model family as resource allocation | Which architecture fits the deployment constraint? | Quality, memory, compute, latency, state |
| V1-07 Frameworks | Runtime execution | When do framework and compiler decisions matter? | Flexibility, fusion, portability, overhead |
| V1-08 Training | Training feasibility | What limits training before accuracy does? | Batch, memory, optimizer state, time, cost |
| V1-09 Data Selection | Data as a systems knob | Which data should be collected or labeled? | Coverage, rare events, noise, cost, quality |
| V1-10 Compression | Smaller is not automatically better | Which compression method preserves system value? | Accuracy, latency, memory, calibration risk |
| V1-11 Hardware Acceleration | Roofline thinking | Is the workload compute-bound or memory-bound? | Throughput, memory bandwidth, utilization |
| V1-12 Benchmarking | Measurement as hypothesis | What benchmark actually predicts deployment? | Component/end-to-end, warm/cold, tail, energy |
| V1-13 On-Device Learning | Adaptation under constraints | When is local learning worth it? | Freshness, energy, privacy, drift, memory |
| V1-14 MLOps | Operating the model | What monitoring and retraining policy is defensible? | Drift, alert burden, retraining cost, risk |
| V1-15 Responsible AI | Responsibility as engineering constraint | How do fairness/privacy/robustness change system design? | Performance, cost, governance, user harm |
| V1-16 Conclusion | Architecture synthesis | What system design follows from prior decisions? | Combined constraints and residual risk |

## 14. Volume 2 Lab Plan

Volume 2 should teach scale: infrastructure, communication, distributed training, reliability, orchestration, performance engineering, inference at scale, edge intelligence, operations, security, robustness, sustainability, and responsible fleet design.

| Lab | Recap focus | Main systems question | Primary trade-off |
|---|---|---|---|
| V2-01 Introduction | Scale changes assumptions | When does scale create new failure modes? | Coordination, utilization, cost, reliability |
| V2-02 Compute Infrastructure | Nodes, accelerators, memory, power | What infrastructure choice is defensible? | TCO, utilization, power, memory, capacity |
| V2-03 Network Fabrics | Fabric topology and placement | When does the network become the system? | Bandwidth, latency, bisection, topology |
| V2-04 Data Storage | Storage-compute gap | How should data be placed and fed to accelerators? | Sharding, cache, checkpoint IO, stalls |
| V2-05 Distributed Training | Parallelism and memory fit | Which parallelism strategy actually scales? | Memory, communication, throughput, convergence risk |
| V2-06 Collective Communication | Collectives as algorithms | Which collective strategy fits topology and workload? | Ring/tree/hierarchical, overlap, compression |
| V2-07 Fault Tolerance | Failure is normal at scale | How much redundancy/checkpointing is worth it? | Useful work, recovery time, cost, reliability |
| V2-08 Fleet Orchestration | Scheduling and fragmentation | How should shared resources be allocated? | Utilization, fairness, preemption, latency |
| V2-09 Performance Engineering | Optimization order | What should be optimized first? | Kernel, memory, batching, fusion, bottleneck migration |
| V2-10 Inference at Scale | Serving cost inversion | When does inference dominate total system cost? | KV cache, batching, replicas, routing, p99 |
| V2-11 Edge Intelligence | Placement and coordination | What should run on device, edge, or cloud? | Latency, privacy, energy, bandwidth, consistency |
| V2-12 ML Operations at Scale | Silent fleet failures | What operations policy catches problems without overload? | Monitoring, canaries, alert fatigue, retraining |
| V2-13 Security and Privacy | Threat and privacy budgets | Which defense is worth its system cost? | Privacy, latency, accuracy, infrastructure overhead |
| V2-14 Robust AI | Robustness tax | How much robustness is operationally justified? | Robustness, latency, compute, fallback policy |
| V2-15 Sustainable AI | Carbon-aware systems | Where and when should workloads run? | Carbon, energy, latency, cost, availability |
| V2-16 Responsible AI | Governance at product scale | How do responsible constraints affect fleet design? | Fairness, auditability, privacy, monitoring burden |
| V2-17 Conclusion | Fleet synthesis | What fleet architecture can be defended? | Combined infrastructure, reliability, cost, risk |

## 15. Versioning And Release Contract

The labs should be versioned as instructional infrastructure. Instructors need to know when a lab changed, students need reports that identify what they completed, and maintainers need to evolve MLSysIM without silently changing assignments.

### Version Fields

Every lab should expose these fields:

| Field | Example | Purpose |
|---|---|---|
| `lab_id` | `v1_05_neural_computation` | Stable identifier for assignments and reports |
| `lab_version` | `1.0.0` | Version of the lab content and activity flow |
| `report_schema_version` | `1.0` | Version of the downloadable report structure |
| `ledger_schema_version` | `1.0` | Version of saved student decision records |
| `mlsysim_version` | `0.2.0` | Engine version that produced results |
| `mlsysbook_labs_version` | `0.1.0` | UI/helper package version |
| `book_anchor` | `Volume I, Chapter 5` | Book chapter pairing |
| `book_revision` | `2026-06` or commit hash | Optional book/content provenance |
| `updated_at` | `2026-06-02` | Human-readable update date |

### Visibility Rules

Version information should be visible, but not distracting.

- The lab header should show lab version and updated date in a compact metadata row or info drawer.
- The report must include lab version, MLSysIM version, report schema version, and timestamp.
- The instructor guide should list the current version of each lab.
- The dashboard should show the active release or version family.
- Debug/source trace views can show commit hashes and detailed provenance.

### Change Semantics

Recommended lab version semantics:

| Change type | Version bump | Examples |
|---|---|---|
| Major | `2.0.0` | Learning objective changes, report schema breaks, nugget sequence changes substantially, assignment no longer comparable |
| Minor | `1.1.0` | Adds a nugget, improves recap, adds track support, adds instructor metadata, adds optional advanced knobs |
| Patch | `1.0.1` | Fixes typo, fixes visual bug, fixes calculation display without changing MLSysIM result semantics |

MLSysIM should maintain its own version. A lab patch may depend on an MLSysIM minor or patch release. Reports should record both so instructors can interpret older submissions.

### Release Channels

Recommended release channels:

| Channel | Purpose |
|---|---|
| Stable | What instructors should assign in courses |
| Preview | New labs or redesigned labs available for testing |
| Dev | Active worktree or unreleased implementation |

Stable releases should have:

- Passing native tests.
- Passing browser/WASM smoke tests.
- Frozen lab IDs.
- Published changelog.
- Instructor guide snapshot.
- Known limitations.

### Changelog Requirements

Every release should include:

- New labs.
- Renamed or renumbered labs.
- Changes to learning objectives.
- Changes to MLSysIM assumptions or models.
- Report schema changes.
- Instructor-facing adoption notes.
- Migration notes for assignments already in use.

### Compatibility Rules

- Lab IDs should be stable once published.
- If a lab is renamed, keep an alias or redirect in metadata.
- If Volume 2 numbering changes, preserve the old-to-new mapping in the changelog.
- Old reports should remain readable when possible.
- If a report schema changes incompatibly, the version should make that obvious.
- A lab should not silently change the meaning of a student submission.

## 16. Implementation Roadmap

### Phase 0 - Lock The Contract

Exit criteria:

- This specification is accepted as the implementation contract.
- Volume 2 mapping includes a dedicated Collective Communication lab.
- Existing docs link back to this spec.
- Lab naming, IDs, and report versioning rules are decided.
- Versioning fields, release channels, and changelog rules are decided.

### Phase 1 - Package Boundary

Tasks:

- Create or move toward `mlsysbook_labs`.
- Move `mlsysim.labs` UI/state/style helpers into the lab helper package.
- Keep MLSysIM importable without lab UI dependencies.
- Update notebook imports.
- Update wheel/browser tests so the correct packages are bundled.

Exit criteria:

- MLSysIM has no Marimo/CSS/student-state dependency.
- Labs import shared UI from `mlsysbook_labs`.
- Existing lab smoke tests still pass.

### Phase 2 - Core Schemas

Tasks:

- Define lab metadata schema.
- Define chapter recap schema.
- Define nugget schema.
- Define track/scenario mapping schema.
- Define ledger schema.
- Define report schema.
- Define instructor/adoption metadata schema.

Exit criteria:

- Every lab can declare metadata without ad hoc dictionaries.
- Static tests can verify required fields.
- Report export can be generated from typed state.

### Phase 3 - Shared UI Components

Tasks:

- Implement `lab_header()`.
- Implement `chapter_recap()`.
- Implement `track_selector()`.
- Implement `scenario_brief()`.
- Implement `nugget_shell()`.
- Implement `prediction_card()`.
- Implement `constraint_budget()`.
- Implement `tradeoff_poster()`.
- Implement `reflection_card()`.
- Implement `decision_card()`.
- Implement `advanced_knob_drawer()`.
- Implement `report_export()`.

Exit criteria:

- A pilot lab can be assembled mostly from shared components.
- Dark banner styles are retired.
- Components use the new academic visual system.

### Phase 4 - MLSysIM Result Readiness

Tasks:

- Standardize `BottleneckReport`.
- Standardize `ConstraintBudget`.
- Standardize `TradeoffSweepResult`.
- Standardize `ParetoFrontier`.
- Standardize serializable result snapshots.
- Add or tighten operator, serving, communication, storage, ops, responsibility, and sustainability models needed by pilots.

Exit criteria:

- Pilot labs do not duplicate formulas locally.
- Result objects include units, labels, and provenance.
- Reports can embed result snapshots.

### Phase 5 - Pilot Labs

Pilot labs:

- V1-01 Introduction: core ML systems framing.
- V1-05 Neural Computation: operator, activation, memory movement.
- V2-10 Inference at Scale: serving, KV cache, batching, fleet cost.

Each pilot must prove:

- Chapter recap orients students without assuming prior reading.
- Track switching changes scenario values and results.
- At least one nugget runs end-to-end.
- MLSysIM produces the core result.
- The student saves a decision.
- The student downloads a report.
- Instructor adoption metadata exists.
- Browser/WASM smoke path still works.

### Phase 6 - Volume 1 Conversion

Tasks:

- Convert V1-00 through V1-16 to the wrapper and nugget contract.
- Preserve strong existing visuals when they still serve the learning goal.
- Replace notebook-local formulas with MLSysIM APIs.
- Add reports and adoption metadata for every lab.
- Add tests for every lab.

Exit criteria:

- Every Volume 1 lab is self-contained, track-aware, report-exporting, and MLSysIM-backed.
- V1-16 can synthesize prior ledger decisions or load a preset ledger.

### Phase 7 - Volume 2 Conversion

Tasks:

- Refocus V2-03 on network fabrics.
- Add V2-06 Collective Communication.
- Renumber downstream Volume 2 labs consistently.
- Convert V2-01 through V2-17 to the wrapper and nugget contract.
- Add reports and adoption metadata for every lab.
- Add tests for every lab.

Exit criteria:

- Every Volume 2 lab is self-contained, track-aware, report-exporting, and MLSysIM-backed.
- V2-17 can synthesize prior ledger decisions or load a preset fleet design.

### Phase 8 - Instructor Guide And Adoption Kit

Tasks:

- Generate a consolidated instructor guide.
- Include lab-by-lab assignment prompts.
- Include report expectations.
- Include rubrics.
- Include misconceptions.
- Include discussion prompts.
- Include setup instructions.
- Include suggested adoption modes.

Exit criteria:

- An instructor can adopt one lab, a module, or the full sequence without reverse-engineering notebook internals.

### Phase 9 - Verification And Release

Tasks:

- Native Python tests for schemas and MLSysIM-backed computations.
- Browser/WASM smoke tests.
- Static tests for required lab metadata.
- Static tests rejecting old "Co-Lab" language and time-estimate language.
- Tests rejecting notebook-local formulas where MLSysIM APIs exist.
- Report export tests.
- Accessibility checks for table fallbacks, color use, keyboard navigation, and readable layouts.
- Version/changelog update.

Exit criteria:

- Labs are ready to publish as a stable adoption target.

## 17. Testing Contract

Required tests:

- Every lab imports.
- Every lab declares metadata.
- Every lab declares chapter recap.
- Every lab declares track compatibility.
- Every lab declares at least one nugget.
- Every nugget declares prediction, result, visual, reflection, and decision.
- Every visible knob maps to an MLSysIM parameter.
- Every chart consumes a typed MLSysIM result or sweep.
- Every lab can generate a report.
- Every report includes lab ID, lab version, MLSysIM version, track, scenario, decision, result snapshot, and timestamp.
- Every lab declares report schema version, ledger schema version, helper package version, and updated date.
- Every lab has instructor adoption metadata.
- No lab uses old dark gradient header styles.
- No lab uses old "Co-Lab" references unless intentionally preserved in historical notes.

## 18. Open Decisions To Close Before Implementation

These are the remaining author decisions that affect implementation:

1. Should the downloaded report include free-form reflection text, structured choices plus short rationale, or both?
2. Should the report artifact be called "lab report" everywhere, or should Volume 1 use "architecture memo" and Volume 2 use "fleet design review" for synthesis labs?
3. Should the selected track persist by default across all labs?
4. Should instructors be able to lock a track for an assignment?
5. Should the JSON report snapshot be generated by default or only behind an instructor/debug option?
6. Should the consolidated instructor guide be generated from lab metadata or written as a separate hand-authored document?
7. Should labs include explicit "related TinyTorch" and "related hardware kit" callouts, or should those remain separate pathways?
8. Should the lab dashboard become the public instructor landing page, or remain an internal planning view?
9. What should the release cadence be: semester-based, book-release-based, or continuous with stable tags?
10. Should stable lab releases get their own DOI/archive snapshot for instructors who need citation-stable assignments?

## 19. Immediate Next Work

The next concrete implementation move is to build the contract around the pilots:

1. Add this spec to the dashboard and existing plan documents.
2. Create the `mlsysbook_labs` package boundary.
3. Define typed schemas for lab metadata, recap, nugget, ledger, report, and instructor metadata.
4. Define versioning metadata and changelog format.
5. Implement the shared academic shell and chapter recap components.
6. Convert V1-01 as the first pilot.
7. Convert V1-05 as the operator/memory pilot.
8. Convert V2-10 as the serving/fleet pilot.
9. Add the report export and instructor adoption metadata to each pilot.
10. Run native and browser/WASM verification.
11. Use pilot results to convert the remaining labs systematically.

Once the pilots pass, conversion should become mechanical: each lab maps chapter ideas to nuggets, nuggets map knobs to MLSysIM parameters, and reports/adoption metadata follow the shared schema.
