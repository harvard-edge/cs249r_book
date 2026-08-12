# Lab Structure And Report Contract

This contract defines the required structure for every track-aware MLSysBook lab and every internal lab part. The structure should be predictable enough that students learn the rhythm once and instructors can grade reports consistently.

## Design Goal

Every lab follows the same student rhythm:

1. Understand the chapter idea.
2. Confirm or choose the track.
3. Read the scenario.
4. Make a prediction.
5. Change a small number of controls.
6. Inspect evidence.
7. Reflect on the trade-off.
8. Make a defensible decision.
9. Save evidence locally.
10. Download a report.

The track changes narrative, defaults, hardware, constraints, and metrics. The pedagogy rhythm does not change.

## Student-Facing Header Standard

Use these headers consistently in the rendered lab. Details can vary by lab and track, but the labels should not drift without a deliberate design-system change.

Header names are semantic requirements. They do not all need to render as large visual blocks. Inside a part, compact panels, inline subheaders, accordions, and side-by-side layouts are acceptable as long as the labels and required content are present.

### Lab-Level Headers

| Order | Required header | Purpose |
|---:|---|---|
| 1 | `Learning Objectives` | State what students should be able to do by the end of the lab. |
| 2 | `Chapter Recap` | Give the compact book-to-systems bridge. |
| 3 | `Your Track` | Show the selected iPhone, Oura Ring, RoboTaxi, or Cloud Fleet profile. |
| 4 | `Scenario Brief` | Explain the stakeholder, workload, constraints, and decision. |
| 5 | `Lab Map` | Preview the parts and completion status. |
| 6 | `Part A - <Concept>` | First focused learning nugget. |
| 7 | `Part B - <Concept>` | Second focused learning nugget. |
| 8 | `Part C - <Concept>` | Third focused learning nugget or decision stage. |
| 9 | `Synthesis` | Combine evidence into one engineering decision. |
| 10 | `Big Takeaways` | Name what should carry forward. |
| 11 | `Download Report` | Produce the local submission artifact. |

If a lab has more than three parts, continue with `Part D - <Concept>` and `Part E - <Concept>`. Do not rename parts as clever section titles without the part label.

### Part-Level Headers

Every part should use these labels in this order:

| Order | Required header | Purpose |
|---:|---|---|
| 1 | `Part <Letter> - <Concept>` | Anchor the part and name the idea. |
| 2 | `What You Need To Know` | Give the minimal knowledge needed before interacting. |
| 3 | `Scenario Slice` | Explain the part-specific track situation. |
| 4 | `Your Prediction` | Capture the pre-reveal commitment. |
| 5 | `Try It` | Hold the controls students manipulate. |
| 6 | `Evidence` | Show the main result visual/table. |
| 7 | `Constraint Check` | Explain feasibility, bottleneck, or failure boundary. |
| 8 | `Source Trace` | Show MLSysIM APIs, registry values, equations, and assumptions. |
| 9 | `Reflection` | Ask what improved, worsened, and could break. |
| 10 | `Checkpoint` | Save part evidence for the final synthesis. |

Use `Decision` instead of `Checkpoint` only when the part itself requires a final policy/configuration choice.

`Checkpoint` means "save evidence for later synthesis." `Decision` means "commit to a configuration or policy."

### Synthesis Headers

The synthesis section should use:

1. `Evidence Summary`
2. `Final Decision`
3. `Big Takeaways`
4. `Residual Risk`
5. `Download Report`

### Report Headers

The downloaded Markdown report should use:

1. `Lab`
2. `Track And Scenario`
3. `Learning Objectives`
4. `Predictions`
5. `Evidence Summary`
6. `Final Decision`
7. `Big Takeaways`
8. `Reflections`
9. `Residual Risk`
10. `Source Trace`

If required fields are missing, the report should include an `Incomplete Fields` section rather than silently omitting them.

## Local-First Requirement

All student work must be local-first.

- Predictions, knob settings, results, decisions, and reflections are stored in the browser/local Design Ledger.
- Report generation happens locally in the browser or local notebook process.
- The downloadable report must not require a hosted backend.
- If local storage fails or is unavailable, students should still be able to generate the current session report.
- If students lose or overwrite local work, the system does not need to recover it remotely; the local report is the submission artifact.

## Lab-Level Structure

Each lab must have the following sections in order.

### 1. Lab Header

Purpose:
- Establish identity, chapter, volume, lab ID, version, and release status.

Required fields:
- Volume.
- Lab number.
- Lab title.
- Chapter anchor.
- Lab version.
- MLSysIM version.
- Report schema version.

Student-facing header:
- The visible hero/header can use the lab title, but the first content block after it should be `Learning Objectives`.

### 1a. Learning Objectives

Purpose:
- State the measurable student outcomes for the lab.

Required content:
- 3-5 objectives.
- Each objective starts with an action verb such as diagnose, quantify, compare, choose, defend, or explain.
- Objectives should name the track-sensitive system behavior when relevant.

Avoid:
- Vague goals like "understand compression".
- Objectives that cannot be observed in the report.

### 2. Chapter Recap

Purpose:
- Make the lab self-contained without replacing the book.

Required fields:
- Chapter idea.
- Key terms.
- Systems translation.
- Common trap.
- Suggested reading.

Student-facing header:
- `Chapter Recap`

### 3. Track Context

Purpose:
- Show which system the student is working in.

Required fields:
- Track name.
- Category.
- Hardware reference.
- Stakeholder.
- Primary metric.
- Guardrail metric.
- Dominant constraints.

Behavior:
- In Lab 00, students choose the track.
- In later labs, the lab reads the track from the Design Ledger by default.
- Students may switch only if the assignment mode allows it.

Student-facing header:
- `Your Track`

### 4. Scenario Brief

Purpose:
- Ground the lab in a specific engineering situation.

Required fields:
- Stakeholder message.
- Workload.
- Model.
- Hardware.
- Constraints.
- Objective.
- What the student must decide.

Student-facing header:
- `Scenario Brief`

### 5. Part/Nugget Navigator

Purpose:
- Show the structure of the lab and completion state.

Required behavior:
- Labels should identify the part and concept.
- Completion state should be visible after a prediction/result/decision is saved.
- Completion state should distinguish: not started, prediction saved, evidence viewed, checkpoint saved, and decision complete.

Student-facing header:
- `Lab Map`

### 6. Parts

Purpose:
- Each part teaches one focused nugget.

Required behavior:
- Every part follows the part-level structure below.
- Parts should usually be A, B, C, then Synthesis unless the lab already requires more.

Student-facing headers:
- `Part A - <Concept>`
- `Part B - <Concept>`
- `Part C - <Concept>`

### 7. Synthesis

Purpose:
- Pull the parts into a single engineering judgment.

Required fields:
- Final selected configuration or policy.
- Primary evidence.
- Guardrail evidence.
- Residual risk.
- What would invalidate the decision.
- Link to next lab or capstone consequence.

Student-facing header:
- `Synthesis`

### 8. Download Report

Purpose:
- Produce the artifact students submit.

Required behavior:
- Export Markdown report locally.
- Optional JSON snapshot for instructors/debugging.
- Report includes all required fields listed below.

Student-facing header:
- `Download Report`

## Part-Level Structure

Each part should use the following internal structure.

### 1. Part Header

Required fields:
- Part label.
- Concept name.
- Systems question.
- Track-specific framing sentence.

Example:
- "Part B - Compression Frontier"
- "How does bit width change the feasible frontier for your selected track?"

Student-facing header:
- `Part <Letter> - <Concept>`

### 2. What You Need To Know

Purpose:
- Give just enough knowledge to interact intelligently.

Required content:
- 2-4 short bullets.
- One key equation or relationship if needed.
- Track-specific interpretation.
- One thing to watch for.
- Keep this section short enough to fit above the fold on a laptop when possible.

Avoid:
- Long textbook replacement.
- Multiple unrelated concepts.

Student-facing header:
- `What You Need To Know`

### 3. Scenario Slice

Purpose:
- Restate the part-specific situation.

Required fields:
- Stakeholder pressure.
- Workload slice.
- Active constraint.
- Primary metric.
- Guardrail metric.

Student-facing header:
- `Scenario Slice`

### 4. Prediction

Purpose:
- Force a pre-reveal commitment.

Required behavior:
- Use multiple-choice or numeric prediction.
- Prediction must be saved to the ledger.
- Result should not be revealed before prediction unless instructor/demo mode explicitly allows it.

Student-facing header:
- `Your Prediction`

### 5. Controls

Purpose:
- Let students manipulate one small system slice.

Required behavior:
- Use no more than three primary controls by default.
- Controls have labels, units, bounds, defaults, and source.
- Advanced controls are hidden by default.

Student-facing header:
- `Try It`

### 6. Evidence

Purpose:
- Show what the system did.

Required behavior:
- One primary visual.
- One exact table fallback if the visual is decision-critical.
- Constraint budget or bottleneck explanation.
- Failure boundary if a constraint is violated.

Student-facing header:
- `Evidence`

### 6a. Constraint Check

Purpose:
- State whether the selected configuration is feasible and what binds first.

Required behavior:
- Show value, limit, unit, and status for the active constraint.
- If a configuration fails, show the first mitigation to try.

Student-facing header:
- `Constraint Check`

### 7. Source Trace

Purpose:
- Show where numbers and equations came from.

Required fields:
- MLSysIM API or registry source.
- Hardware/model refs.
- Scenario assumptions.
- Equation or solver name.
- Provenance label for estimates/conventions.

Rendering:
- Source Trace can be collapsed by default.
- A one-line visible source summary should remain visible when collapsed.

Student-facing header:
- `Source Trace`

### 8. Reflection

Purpose:
- Convert interaction into reasoning.

Required prompts:
- What improved?
- What worsened?
- What assumption could break your conclusion?

Behavior:
- Reflection should be short and structured.
- Free-form text is allowed, but report generation should not depend only on long prose.
- Part reflections should be quick. Save longer prose for Synthesis when instructors want it.

Student-facing header:
- `Reflection`

### 9. Part Decision Or Checkpoint

Purpose:
- Save evidence for the final synthesis.

Required fields:
- Selected option or observed result.
- Binding constraint.
- Primary metric value.
- Guardrail metric value.
- Residual risk.

Student-facing header:
- `Checkpoint` or `Decision`

## Synthesis Structure

Every synthesis section should include:

1. `Evidence Summary`
2. `Final Decision`
3. Primary metric and guardrail metric.
4. `Big Takeaways`
5. `Residual Risk`
6. `Download Report` button.

## Big Takeaways

Every lab should end with 3-5 takeaways:

- One chapter concept takeaway.
- One systems engineering takeaway.
- One track-specific takeaway.
- One misconception corrected.
- One carry-forward note for a later lab or capstone.

Avoid:
- Generic takeaways such as "trade-offs matter" without naming the specific trade-off and track.

## Reflection Contract

Reflections should be small but mandatory.

Required structured fields:
- `diagnosis`: what happened.
- `tradeoff`: what improved and what worsened.
- `residual_risk`: what assumption could break the decision.

Optional field:
- `short_rationale`: one free-form paragraph for instructors who want prose.

Reflections should be saved locally and included in the report.

## Design Ledger Contract

Each part should save:

- `lab_id`
- `part_id`
- `track_id`
- `scenario_id`
- `hardware_ref`
- `prediction`
- `knob_settings`
- `result_snapshot`
- `binding_constraint`
- `primary_metric`
- `guardrail_metric`
- `reflection`
- `decision_or_checkpoint`
- `timestamp`

The synthesis should save:

- `final_decision`
- `evidence_summary`
- `big_takeaways`
- `residual_risk`
- `report_snapshot`

## Downloadable Report Contract

The Markdown report must include:

- Student-entered name or identifier, if provided.
- Lab ID, title, version, and timestamp.
- Track and hardware reference.
- Scenario summary.
- Prediction summary.
- Results/evidence summary.
- Final decision.
- Primary metric result.
- Guardrail metric result.
- Big takeaways.
- Reflections.
- Residual risk.
- Source/assumption trace.

Optional JSON snapshot:
- Same fields in machine-readable form.
- Include schema versions.

## Failure And Recovery

Local-first means failures should be understandable.

If ledger load fails:
- Show a warning.
- Let the student continue with the current session.
- Allow report download from current state.

If report export fails:
- Show the report content in a visible text area as fallback.

If a required part is incomplete:
- Report can still download, but it must mark missing fields in an `Incomplete Fields` section.

## Compact Rendering Guidance

The structure should feel like a guided engineering worksheet, not a compliance form.

- `Learning Objectives`, `Chapter Recap`, `Your Track`, and `Scenario Brief` should be concise.
- `Source Trace`, table fallbacks, and advanced controls may be collapsed by default.
- `Evidence` and `Constraint Check` should be visually adjacent.
- `Reflection` should use structured fields and short prompts.
- `Checkpoint` should be compact unless it is a final `Decision`.
- `Big Takeaways` should be prominent in Synthesis.

## Implementation Checklist

Before a lab is complete:

- Lab has header, recap, track context, scenario brief, parts, synthesis, and report export.
- Every part has "What You Need To Know", prediction, controls, evidence, source trace, reflection, and checkpoint.
- All hardware/model facts come from MLSysIM.
- All scenario thresholds come from typed lab variant metadata.
- Ledger entries include part-level evidence.
- Report downloads locally.
- Report includes big takeaways and residual risk.
- Visuals have table/text fallbacks.
