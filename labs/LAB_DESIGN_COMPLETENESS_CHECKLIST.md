# Lab Design Completeness Checklist

This document captures the remaining design dimensions we should settle before implementation. The goal is to make the labs feel inevitable: every chapter has a clear ML systems idea, every nugget exercises a real trade-off, and every interaction is backed by MLSysIM.

## What Else The Design Should Capture

## Decisions Captured So Far

These author decisions should be treated as design constraints:

- Volume 2 should have a dedicated Collective Communication lab. The current lab sequence drifted as the book changed; the improved sequence should realign labs to the book rather than preserve outdated numbering.
- Labs should encourage chapter reading, but they should not assume the student read the chapter immediately before opening the lab.
- Every lab needs a mini chapter recap that names the chapter emphasis, key terms, systems translation, and common trap.
- Students should be able to complete a lab, download a simple report, and submit it as evidence of completion.
- The report should be lightweight: enough to show the student did the lab and made a reasoned decision, not a complicated grading platform.
- Labs should be versioned so students and instructors can tell when content, schemas, or MLSysIM assumptions changed.
- Advanced knobs should not clutter the default student flow. They should be optional controls for deeper exploration, hidden behind an "Advanced" disclosure when useful.
- Instructor support should exist, but it should not make the student lab feel heavier.
- Instructor adoption should be designed explicitly: assignment text, grading expectations, discussion prompts, and setup instructions should be easy to reuse.

### 1. The Chapter-To-Systems Translation

Many chapters start from a machine learning concept, but the lab has to translate that concept into an ML systems question.

Each chapter/lab should explicitly capture:

| Field | Question |
|---|---|
| ML concept | What ML idea does the chapter introduce? |
| Systems translation | What does that idea become when deployed, scaled, operated, or constrained? |
| System knob | What can the engineer actually change? |
| Binding resource | What resource or constraint is most likely to bind? |
| Failure mode | What breaks when the student pushes too far? |
| Decision | What engineering decision should the student be able to defend? |

Example:

| ML concept | Systems translation |
|---|---|
| Transformer sequence length | KV cache, memory footprint, batching pressure, tail latency, serving cost |
| Quantization | Hardware kernel support, memory bandwidth, accuracy loss, calibration risk, robustness shift |
| Fairness metric | Threshold policy, subgroup error, monitoring burden, governance cost |

This translation should be visible in every lab, because it is the bridge from "knowing ML" to "engineering ML systems."

### 2. The Big Thematic Nuggets

The labs should repeatedly reinforce a small set of durable ML systems ideas. These are the conceptual anchors students should remember after the details fade.

| Thematic nugget | System principle | Where it should recur |
|---|---|---|
| Data-Algorithm-Machine coupling | System behavior is a coupled product of data, model, and hardware | V1 intro, workflow, data, architecture, conclusion |
| The Iron Law | Time/cost/energy decompose into work, rate, movement, and overhead | Performance, training, serving, distributed systems |
| Resource budgets | Feasibility is determined by finite memory, compute, bandwidth, energy, cost, reliability, and carbon | Every lab |
| Bottleneck migration | Optimizing one constraint often exposes the next constraint | Compression, frameworks, acceleration, performance engineering |
| Data movement dominates | Moving data is often more expensive than computing on it | Data engineering, storage, communication, edge |
| Tails matter | Median behavior can hide production failure | Benchmarking, serving, orchestration, reliability |
| Scale changes the system | Fleet behavior is not single-node behavior multiplied by N | Volume 2 introduction, training, ops, conclusion |
| Coordination tax | Synchronization, scheduling, monitoring, and communication reduce useful work | Distributed training, collectives, orchestration |
| Reliability is probabilistic | More components make failure normal, not exceptional | Fault tolerance, fleet synthesis |
| Responsible constraints are system constraints | Privacy, fairness, security, robustness, and carbon change architecture and operations | Responsible engineering, security, robust AI, sustainable AI |
| Design is a frontier | There is rarely one optimum; there are defensible points on a frontier | Every Part B/trade-off surface |
| Engineering judgment | A good design names its assumption and residual risk | Every decision card |

### 3. Misconceptions To Correct

Each nugget should name the misconception it is designed to correct.

Examples:

| Misconception | Corrective systems idea |
|---|---|
| Smaller model always means faster model | Runtime support, sparsity structure, memory movement, and accuracy validation matter |
| Higher utilization is always better | High utilization can destroy tail latency and scheduling responsiveness |
| More GPUs means linear speedup | Communication, synchronization, stragglers, failures, and pipeline bubbles dominate |
| Accuracy is the only model metric | Latency, memory, energy, reliability, fairness, and maintainability can bind first |
| Fairness/privacy/security are external concerns | They impose measurable compute, data, latency, governance, and monitoring costs |
| Benchmarks tell the truth | Benchmarks are hypotheses; they can hide thermal, tail, workload, and end-to-end failures |
| Data is just input | Data movement, curation, drift, storage, and labeling are system design surfaces |
| The model is the system | The system includes data, runtime, serving, monitoring, users, infrastructure, and policy |

This is important because good labs do not just show outputs; they change a student's mental model.

### 4. Nugget Inventory Per Lab

The design should include a nugget inventory for every chapter:

| Field | Meaning |
|---|---|
| Nugget name | Short reusable name |
| Book anchor | Chapter and key ideas |
| Student misconception | What prior belief this nugget challenges |
| MLSysIM model | Which engine capability powers it |
| Interaction devices | Slider, frontier, map, roofline, queueing chart, etc. |
| Student artifact | Prediction, decision, reflection, ledger entry |
| Instructor discussion hook | What an instructor can ask after the lab |

This is the level of planning that will make implementation straightforward.

### 5. Track-Specific Narrative Rules

Tracks should not just swap labels. Each track should change the story, constraints, and stakes.

| Track | Narrative emphasis | Typical binding constraints |
|---|---|---|
| Mobile ML | User experience, battery, privacy, app size, thermal behavior | Latency, energy, memory, privacy, update path |
| TinyML | Duty cycle, sensor rate, SRAM/flash, field reliability | SRAM, flash, power, robustness, maintenance |
| Edge AI | Streams, local autonomy, uplink, site heterogeneity | Throughput, thermal envelope, uplink, local storage, orchestration |
| Cloud/Scale | Fleet economics, utilization, reliability, network, carbon | Cost, p99 latency, accelerator memory, communication, carbon, failures |

For every nugget, the design should say what changes by track:

- Default hardware.
- Workload shape.
- Constraint limits.
- Failure mode.
- Final decision framing.

### 6. Student Artifacts

The labs should produce reusable student artifacts, not just transient slider output.

Each lab should capture:

- Prediction.
- Bottleneck diagnosis.
- Trade-off interpretation.
- Selected configuration.
- Residual risk.
- Book principle referenced.
- Result snapshot.
- Track/scenario.

Across a volume, those artifacts become:

- A Volume 1 architecture memo.
- A Volume 2 fleet design review.
- A design ledger that shows how the student's reasoning evolved.

### 6a. Downloadable Student Report

Each lab should support a simple downloadable report. The goal is not to build a full LMS. The goal is to let an instructor say, "Do this lab and submit the report," and let the student produce a clean artifact.

Recommended report fields:

| Field | Purpose |
|---|---|
| Student-entered name or identifier | Lets the report be submitted without requiring accounts |
| Lab ID and title | Identifies the assignment |
| Volume and chapter | Connects the report to the book |
| Selected track | Shows the deployment context |
| Scenario summary | Shows what system was analyzed |
| Nugget completion summary | Shows which nuggets were completed |
| Predictions | Shows prior reasoning |
| Binding constraints | Shows key simulator result |
| Selected configuration or policy | Shows engineering decision |
| Trade-off interpretation | Shows what improved and what worsened |
| Residual risk | Shows judgment rather than button-clicking |
| MLSysIM result snapshot | Gives evidence from the engine |
| Timestamp and version | Helps instructors interpret results |

Recommended export formats:

- Markdown report first, because it is easy to generate, inspect, and convert.
- JSON ledger snapshot alongside it, if useful for automated checking.
- Optional HTML report later, if we want a polished student-facing artifact.

Suggested report implementation:

```python
report = build_lab_report(ledger_entry, scenario, mlsysim_results)
download_report(report, formats=["markdown", "json"])
```

This should live in `mlsysbook_labs`, while MLSysIM should provide serializable result snapshots.

### 6b. Versioning And Provenance

Each lab should carry explicit version metadata. This is important for global adoption: instructors need stable assignments, students need report provenance, and maintainers need to change labs without silently changing what a submission means.

Recommended required metadata:

| Field | Purpose |
|---|---|
| Lab ID | Stable assignment identifier |
| Lab version | Indicates content/activity changes |
| Report schema version | Indicates report structure |
| Ledger schema version | Indicates saved decision structure |
| MLSysIM version | Indicates engine/model assumptions |
| Lab helper version | Indicates UI/component behavior |
| Book anchor | Connects to volume/chapter |
| Updated date | Helps humans see recency |

Recommended release channels:

- Stable: assignable in courses.
- Preview: available for testing redesigned labs.
- Dev: current implementation work.

Recommended changelog entries:

- Lab additions, removals, renames, and renumbering.
- Learning objective changes.
- MLSysIM model or assumption changes.
- Report/ledger schema changes.
- Instructor adoption notes.
- Old-to-new mapping for any renamed labs.

### 7. Assessment And Rubric

The labs should have a lightweight rubric so students and instructors know what good reasoning looks like.

Suggested rubric dimensions:

| Dimension | Good answer shows |
|---|---|
| Bottleneck diagnosis | Identifies the active constraint and cites evidence |
| Trade-off reasoning | Names what improved and what worsened |
| Systems vocabulary | Uses the chapter concept accurately |
| Decision quality | Chooses a defensible configuration, not just the max/min |
| Residual risk | Names an assumption that could invalidate the decision |
| Track awareness | Explains why the answer changes across deployment context |
| MLSysIM interpretation | Uses simulator output as evidence, not as magic |

### 8. Instructor Support

For each lab, we should capture:

- Teaching goal.
- Expected misconception.
- Key MLSysIM output.
- Discussion questions.
- Optional extension.
- Common student failure mode.
- What to emphasize in class.

This does not need to be visible to students in the lab, but it should exist for instructors.

Recommended structure:

- One consolidated instructor guide for the whole lab sequence, organized by volume and chapter.
- Optional short instructor notes beside each lab for local maintenance.
- The consolidated guide should include the nugget inventory, expected misconceptions, discussion prompts, and rubric.

This avoids scattering instructor guidance across many notebooks while still letting each lab carry lightweight metadata.

### 8a. Instructor Adoption Package

If these labs are meant to become widely adopted signature ML systems labs, each lab needs a small adoption package. The instructor should not have to reverse-engineer the purpose of the lab from the notebook.

Recommended instructor-facing fields:

| Field | Purpose |
|---|---|
| Why assign this lab | The ML systems concept students will understand better after doing it |
| Where it fits | Suggested book chapter, course module, and prerequisite ideas |
| Assignment prompt | Copy-ready text an instructor can paste into a syllabus or LMS |
| Expected student artifact | What the downloaded report should contain |
| Rubric | Lightweight grading dimensions |
| Common misconceptions | What students are likely to get wrong |
| Discussion questions | Questions for class, recitation, or office hours |
| Extension ideas | Optional advanced-knob explorations |
| Setup notes | Browser/local/WASM expectations and known limitations |

Recommended adoption modes:

- Standalone homework: assign one lab and collect the downloaded report.
- Lecture companion: use one nugget live in class, then assign the rest.
- Module sequence: assign a cluster of labs around serving, data, edge, distributed training, or responsibility.
- Capstone preparation: use the ledger artifacts to build a final architecture memo or fleet design review.

Design implication: the student lab should be self-contained, but the instructor guide should make the adoption story explicit. The value proposition is that students practice ML systems engineering judgment without needing a custom cluster, hardware kit, or long build project.

### 9. MLSysIM Capability Coverage Matrix

Before implementation, every nugget should map to an MLSysIM capability:

| Nugget | Required MLSysIM capability | Exists now? | Needed work |
|---|---|---:|---|
| Activation memory cliff | Operator/activation memory model | Partial | Add lab-ready operator result |
| Serving tail latency | Queueing and serving model | Yes | Standardize result and frontier sweep |
| Collective crossover | Communication topology model | Partial | Add all-reduce/all-gather primitives and frontier output |
| Fairness tax | Responsible systems model | Partial | Add subgroup and governance result schema |
| Carbon placement | Sustainability and placement model | Yes | Add lab-ready map/frontier output |

This prevents lab design from outrunning the engine.

### 10. Provenance And Book Connection

Because MLSysIM also powers the textbook, every lab nugget should include:

- Chapter name.
- Key ideas from the chapter.
- MLSysIM function/model used.
- Formula or concept label.
- Constants or assumptions, with provenance when available.

The student-facing version should be compact. The instructor/debug version can be more detailed.

### 11. Accessibility And Cognitive Load

The labs should be rich, but not overwhelming.

Design rules:

- No more than three primary knobs visible in a nugget by default.
- Advanced knobs should be collapsible.
- Every visual must have a text interpretation.
- Color should not be the only signal.
- Units must always be visible.
- The student should always know which question they are answering.
- Reflection prompts should be short and targeted.

### 12. Lab Completion Definition

Each lab should define what "done" means.

Suggested completion requirements:

- Student selected a track.
- Student completed each nugget prediction.
- Student inspected at least one computed result per nugget.
- Student saved one final synthesis decision.
- Ledger contains decision, rationale, binding constraint, and residual risk.
- Student can download a report containing completion evidence and the final decision.

This lets us add progress indicators and tests without turning the lab into a rigid grading system.

## Advanced Knobs Definition

An advanced knob is a control that is useful for deeper exploration but not required for the main learning objective.

Examples:

| Main knob | Advanced knobs |
|---|---|
| Batch size | Service-time variance, request burstiness, queue discipline |
| Sequence length | KV precision, page size, prefix-cache hit rate |
| Model size | Layer count, hidden dimension, attention heads |
| Hardware target | Effective utilization, memory bandwidth override, interconnect selection |
| Compression method | Calibration set size, sparsity pattern, outlier handling |
| Carbon region | PUE, embodied carbon amortization, time-shifting window |

Design rule:

- Default view should have the minimum knobs needed for the nugget.
- Advanced controls should be hidden behind a disclosure.
- Advanced controls should never be required to complete the lab report.
- Instructor guide can suggest advanced controls for extensions.

## Proposed Additions To The Design Documents

We should add or expand these sections before implementation:

1. Nugget inventory for every lab.
2. Track-specific scenario table for every nugget.
3. Misconception and corrective principle for every nugget.
4. MLSysIM capability coverage matrix.
5. Student artifact and ledger schema.
6. Instructor guide template.
7. Assessment rubric.
8. Visual/device selection for every nugget.
9. Completion criteria.
10. Accessibility and cognitive-load guardrails.

## Open Questions For The Author

These are the decisions worth confirming before implementation:

1. Should tracks be persistent for an entire student journey, or should every lab default to the saved track but allow switching?
2. Should the final ledger artifact be framed as an "architecture memo," a "design review," a "lab report," or different names depending on volume?
3. Should each nugget save a mini-decision plus one final synthesis decision, or should only the final lab decision be required for submission?
4. Should the downloadable report include free-form reflection text, or only structured choices plus a short final rationale?
5. How explicitly should the labs reference TinyTorch and hardware kits: as related pathways, or mostly separate experiences?
