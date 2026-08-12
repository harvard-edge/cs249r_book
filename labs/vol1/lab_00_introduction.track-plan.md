# V1-00 Track Plan: The Architect's Portal

## Chapter Invariant

MLSys labs teach deployed behavior under operating envelopes; track choice makes
those constraints real. A model is not treated as finished when it scores well in
isolation. It becomes an engineering object only after a stakeholder, workload,
machine, metric, guardrail, failure mode, and report obligation are attached.

Lab 00 is a special-case orientation. It avoids advanced physics instruments, but
it still asks students to practice the lab ritual: commit to a prior, inspect a
track-specific envelope, connect evidence to a decision, and save a ledger entry
that future labs can replay.

## Reading Map

| Lab module | Reading anchor | Claim used in the lab |
|---|---|---|
| Opening | `book/quarto/contents/vol1/index.qmd` | Volume I establishes AI engineering as co-design of Data, Algorithm, and Machine under statistical and computational constraints. |
| Part A | `book/quarto/contents/vol1/introduction/introduction.qmd`, Purpose and AI Moment | ML systems are data-shaped behavior running under physical constraint; deployment is not a late detail. |
| Part B | Introduction, Defining ML Systems, D-A-M taxonomy, deployment spectrum | A deployment context changes which axis binds and which evidence a stakeholder accepts. |
| Part C | Introduction, ML System Lifecycle and fallacies | Later labs repeat a cycle: case, prediction, manipulation, evidence, decision, report. |
| Part D | Introduction, deployment shapes lifecycle, summary takeaways | Decisions persist because deployed systems are maintained through monitoring, updates, and accountable reports. |
| Synthesis | Introduction, chapter connection to ML Systems | Physical laws determine where a model can run; operating envelopes turn abstract reasoning into design constraints. |

## Concept Inventory

Accepted concepts:

- ML systems labs are about deployed behavior, not isolated model metrics.
- A track is an operating envelope with stakeholder, constraints, metrics,
  guardrails, failure modes, and report framing.
- The repeated lab workflow is case -> prediction -> manipulation -> evidence ->
  decision -> report.
- Ledger/report continuity makes each lab decision available to later labs.
- The selected track changes amount-system reasoning by changing which quantity
  is scarce first: battery, SRAM, p99 latency, cost, utilization, carbon, or
  another operating limit.

Rejected or deferred concepts:

- Detailed iron-law calculations. Deferred to later Volume I labs because Lab 00
  is orientation.
- Advanced physics instruments such as roofline, queueing curves, CDFs, or power
  simulators. Deferred until students have a technical chapter target.
- Free-form career exploration. Rejected because tracks are learning lenses, not
  career identities.
- Full hardware registry refactoring. Rejected for this wave because shared
  helpers and registries are outside ownership.
- Track switching policy and instructor-locked URL modes. Deferred because the
  current task owns only the Lab 00 notebook and plan.

## Concept Modules

### Part A: Concept Module - Deployed Behavior Is The Object

- Chapter claim: ML systems differ from model-only ML because learned behavior
  must satisfy physical and operational constraints after deployment.
- Reading connection: Purpose, AI Moment, Defining ML Systems.
- Student prior: "If the model is accurate, the main work is done."
- Productive failure: The student initially treats the lab as model tuning or
  device memorization, then sees that the correct answer is evidence-backed
  systems decision-making.
- Track lens: all tracks appear only as examples; no track choice yet.

Storyline beats:

1. Scenario: a student joins an MLSys lab sequence and is asked what work the
   notebook will require.
2. Prediction: choose what the labs are mainly practicing.
3. Manipulation: compare isolated model thinking against deployed-system
   thinking in a guided contrast card.
4. Evidence: feedback explains which parts of a real deployment sit outside the
   model score.
5. Consequence: a model that works in a notebook can still fail a stakeholder's
   operating envelope.
6. Source/math peek: the notebook points back to the chapter definition that
   behavior depends on data quality, algorithm choice, and machine capacity.
7. Checkpoint: students must identify evidence-backed systems decisions as the
   goal before Part B unlocks.

Mechanics:

- Structured radio prediction.
- Contrast cards for isolated model vs deployed system.
- Feedback callout that names the misconception.
- Reading/source accordion in the notebook text.

Evidence and ledger:

- Evidence: selected orientation answer and correctness.
- Ledger field: `orientation_goal_answer`, `orientation_goal_correct`.
- Downstream use: establishes that later reports should argue from deployed
  evidence, not from a model score alone.

Depth gate:

- Activity count: 5+.
- Has prediction: yes.
- Has manipulation: yes, through guided comparison and check feedback.
- Has failure/boundary: yes, conceptual boundary between model-only and
  deployed-system answers.
- Has source connection: yes.
- Track-specific consequence: introduced but not yet selected.

### Part B: Concept Module - A Track Is An Operating Envelope

- Chapter claim: The deployment spectrum changes constraints, metrics, update
  paths, monitoring, and acceptable evidence.
- Reading connection: D-A-M taxonomy, deployment spectrum, efficiency priorities
  by context, deployment shapes lifecycle.
- Student prior: "Tracks are cosmetic labels or device themes."
- Productive failure: The student selects only narrative/device fields or also
  selects the chapter objective as track-specific, then feedback separates the
  shared concept from the track-specific envelope.
- Track lens: iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Storyline beats:

1. Scenario: four students must defend the same MLSys idea to different
   stakeholders.
2. Prediction: select which parts of later labs should change with the track.
3. Manipulation: inspect four track cards with different stakeholder, metrics,
   guardrails, dominant constraints, likely failure, and report frame.
4. Evidence: a correctness table shows which selected fields belong to the
   operating envelope and which must remain shared.
5. Consequence: the same model idea becomes a battery decision, memory decision,
   safety decision, or fleet economics decision depending on track.
6. Source/math peek: D-A-M and deployment spectrum anchors explain why the
   machine and mission change the feasible answer.
7. Checkpoint: students must identify story, hardware assumptions, metrics,
   guardrails, and report framing as track-specific before Part C.

Track narratives:

| Track | Stakeholder | Constraint emphasis | Failure mode | Report framing |
|---|---|---|---|---|
| iPhone | Mobile product engineer / UX director | Battery, thermal envelope, unified memory, privacy, interactive latency | Thermal throttle, battery drain, or sluggish local UX | Local-device readiness memo: prove responsiveness, privacy, and sustained comfort. |
| Oura Ring | Wearable firmware engineer / hardware lead | SRAM, flash, duty cycle, sampling cadence, OTA payload, battery | SRAM or flash overflow, duty-cycle violation, or radio wakeup budget miss | Firmware fit memo: prove the model, sensing window, update package, and battery budget fit together. |
| RoboTaxi | Autonomous vehicle platform engineer / safety director | p99/p999 latency, rare-event recall, redundancy, sensor bandwidth, power | Deadline miss, safety-margin miss, or unsupported fallback path | Safety evidence memo: prove worst-case latency, rare-event behavior, and fallback before deployment. |
| Cloud Fleet | Fleet service owner / CTO | Throughput, p99 SLA, utilization, cost/request, capacity, carbon | SLO breach, queue explosion, negative ROI, or carbon budget miss | Fleet operations memo: prove SLA, economics, utilization, and sustainability at production scale. |

Mechanics:

- Multi-select orientation check.
- Track cards populated from canonical profiles plus local orientation story
  fields.
- Table-like feedback for correct, missing, and wrong selections.
- Track-specific consequence callout.

Evidence and ledger:

- Evidence: selected track-change fields, correctness feedback, inspected cards.
- Ledger field: `track_change_selections`.
- Downstream use: later labs read `track_id`, hardware refs, metrics, guardrails,
  dominant constraints, expected failure mode, and report frame.

Depth gate:

- Activity count: 6+.
- Has prediction: yes.
- Has manipulation: yes, inspect/compare cards and revise checkbox choices.
- Has failure/boundary: yes, failure modes differ by operating envelope.
- Has source connection: yes.
- Track-specific consequence: yes for all four tracks.

### Part C: Concept Module - Labs Repeat Prediction, Manipulation, Evidence, Decision

- Chapter claim: ML systems are maintained through lifecycle loops, monitoring,
  and evidence rather than one-time deployment.
- Reading connection: ML system lifecycle, fallacies and pitfalls.
- Student prior: "Interactive labs are mainly sliders and charts."
- Productive failure: The student chooses to tune first or download a report
  before decisions, then feedback reinforces the workflow order.
- Track lens: selected later; Part C shows the common ritual.

Storyline beats:

1. Scenario: a later lab opens with a stakeholder case and asks the student to
   diagnose a failure.
2. Prediction: choose the repeated workflow order.
3. Manipulation: move a simple orientation slider that changes evidence strength
   and decision readiness without simulating advanced physics.
4. Evidence: live preview and component tour show which notebook elements record
   priors, change evidence, reveal optional details, and support reports.
5. Consequence: tuning without a case or report without evidence produces an
   indefensible memo.
6. Source/math peek: optional details preview explains how later labs attach
   formulas, definitions, or source claims.
7. Checkpoint: students must identify case -> prediction -> manipulation ->
   evidence -> decision -> report as the repeated order.

Mechanics:

- Structured radio workflow check.
- `mo.ui.tabs` component tour.
- Simple slider with live qualitative status.
- Optional detail accordion.
- Completion banner naming the repeated loop.

Evidence and ledger:

- Evidence: workflow answer and correctness; live slider status is visible but
  not an advanced measurement.
- Ledger field: `lab_workflow_answer`, `lab_workflow_correct`.
- Downstream use: prepares students for later prediction locks and evidence
  tables.

Depth gate:

- Activity count: 6+.
- Has prediction: yes.
- Has manipulation: yes, live slider in the component tour.
- Has failure/boundary: yes, workflow order fails if report precedes evidence.
- Has source connection: yes.
- Track-specific consequence: deferred to Part D.

### Part D: Concept Module - Ledger Continuity Makes Decisions Carry Forward

- Chapter claim: ML systems are lifecycle objects; deployment decisions must be
  recorded so future evaluation, monitoring, and redesign can reuse them.
- Reading connection: ML system lifecycle, deployment shapes lifecycle, summary
  takeaways.
- Student prior: "A report is just the end of the lab."
- Productive failure: The student may treat the track choice as a page state;
  the notebook shows it becomes a ledger entry future labs read.
- Track lens: selected track.

Storyline beats:

1. Scenario: a future lab needs a default operating envelope before it can pick
   scenario thresholds.
2. Prediction: the selected track implies a first bottleneck and stakeholder
   evidence standard.
3. Manipulation: choose a canonical track and write a brief decision rationale.
4. Evidence: track context, stakeholder message, course arc, and ledger HUD show
   what was saved.
5. Consequence: changing tracks later changes prior ledger context and report
   continuity.
6. Source/math peek: source trace records hardware and system refs from MLSysIM
   or canonical lab metadata.
7. Checkpoint: the report unlocks only after orientation checks and track
   selection complete.

Mechanics:

- Canonical track radio selector.
- DecisionLog text area for rationale.
- Track context and arc cards.
- Ledger save and HUD.
- Report export panel.

Evidence and ledger:

- Evidence: selected track profile, hardware ref, system ref, dominant
  constraints, stakeholder quote, first bottleneck, rationale, report snapshot.
- Ledger fields: existing `track_id`, `track_label`, `track_category`,
  `hardware_ref`, `system_ref`, `primary_metrics`, `guardrail_metrics`,
  `dominant_constraints`, plus Lab 00 continuity fields for likely failure,
  report frame, first bottleneck, and rationale.
- Downstream use: Lab 01 and beyond can default to the selected track and frame
  future reports around the saved operating envelope.

Depth gate:

- Activity count: 6+.
- Has prediction: yes, first bottleneck and report frame are made explicit.
- Has manipulation: yes, track choice and rationale.
- Has failure/boundary: yes, selected likely failure mode.
- Has source connection: yes, hardware/system refs and report source trace.
- Track-specific consequence: yes.

### Synthesis: The Selected Track Shapes Amount-System Reasoning

- Chapter invariant restated: MLSys reasoning is amount-system reasoning under
  an operating envelope. Every future quantity has a different meaning depending
  on the selected track.
- Student output: explain which track was chosen, what constraint will keep
  returning, why notebook success is not deployment success, and how the later
  workflow produces evidence.
- Report output: a local orientation memo with track, scenario, predictions,
  evidence summary, final decision, takeaways, residual risk, and source trace.
- Ledger output: durable track identity and Lab 00 completion context.

## Mechanics Plan

| Need | Mechanic | Why it fits Lab 00 |
|---|---|---|
| Productive failure | Radio checks for lab purpose and workflow | Students commit before seeing feedback without advanced computation. |
| Context transfer | Four canonical track cards | Same course invariant appears under four operating envelopes. |
| Track specificity | Local orientation story map plus canonical track profiles | Keeps stakeholder, constraints, failure, and report frame distinct. |
| Manipulation | Guided checkbox set and simple live slider | Students revise choices and see evidence status change without physics instruments. |
| Evidence | Feedback table, track cards, report preview, HUD | Evidence is textual and structured because Lab 00 is orientation. |
| Source trace | Accordion/source language and report source trace | Connects chapter claims and registry refs without exposing shared internals. |
| Ledger continuity | `DesignLedger.save` plus HUD and report export | Makes the selected envelope available to future labs. |

## Evidence Plan

Student-visible evidence:

- Correctness feedback for the goal of MLSys labs.
- Correctness feedback for which track fields should change.
- Track cards with stakeholder, metrics, guardrails, constraints, failure mode,
  and report frame.
- Workflow check feedback.
- Live orientation evidence preview in the interface tour.
- Track-specific stakeholder message and course arc after selection.
- Downloadable report and ledger HUD.

Ledger/report evidence:

- Selected track and canonical refs.
- Orientation predictions and correctness.
- Track-specific fields selected by the student.
- Recurring workflow answer and correctness.
- First bottleneck / likely failure mode for the chosen operating envelope.
- Rationale from the decision log when provided.
- Source trace for track id, hardware ref, system ref, profile source policy, and
  report builder.

## Implementation Risks

- Shared registry limitation: `TrackProfile` does not currently include explicit
  failure mode or report frame fields. Mitigation: keep a Lab 00-local helper
  with `v1_00_` prefix and do not edit shared schemas.
- Orientation depth vs. overload: adding too many technical claims would turn
  Lab 00 into Lab 01. Mitigation: use guided choices, cards, preview states, and
  report/ledger continuity rather than physics solvers.
- Parallel worker risk: other workers may edit other labs. Mitigation: edit only
  `labs/vol1/lab_00_introduction.py` and this plan.
- WASM behavior risk: preserve existing bootstrap and ledger save shape.
- Report contract risk: keep `build_lab_report` schema keywords intact.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Pass? |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 2 | 2 | 2 | 2 | Yes |
| Part B | 3 | 3 | 3 | 3 | 3 | 2 | Yes |
| Part C | 3 | 3 | 2 | 3 | 2 | 2 | Yes |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Yes |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 | Yes |

Rubric notes:

- No dimension is below 2.
- At least three dimensions score 3 in Parts B, D, and Synthesis.
- Reversible failure state is conceptual in Lab 00: wrong workflow or track-field
  assumptions can be corrected before the report unlocks. Advanced physical
  failure states begin in later labs.
- Synthesis ties all modules back to the invariant that deployed behavior is
  governed by the operating envelope.
