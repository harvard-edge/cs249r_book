# Lab Structure Feedback Simulation

This document simulates feedback on `LAB_STRUCTURE_AND_REPORT_CONTRACT.md`, especially the fixed student-facing headers and local-first report flow.

The goal is to pressure-test whether the structure is clear enough for instructors, predictable enough for students, and practical enough for implementation.

## Reviewed Draft

- `labs/LAB_STRUCTURE_AND_REPORT_CONTRACT.md`

## Overall Assessment

The structure is strong. It gives labs a repeatable rhythm and makes report generation straightforward. The main risk is not conceptual; it is density. If every required header is rendered as a large visual section in every part, a lab could feel bureaucratic. The contract should stay strict, but implementation should use compact UI treatments where appropriate.

Recommended stance:

- Keep the header names.
- Keep the local-first report requirement.
- Make `Source Trace`, advanced controls, and table fallbacks collapsible by default.
- Let `Checkpoint` be compact unless it is the final `Decision`.

## Instructor Feedback Simulation

### 1. Course Instructor: "This is finally gradeable."

Likely positive feedback:
- The report headers map cleanly to grading.
- `Learning Objectives`, `Evidence Summary`, `Final Decision`, and `Residual Risk` make the submission concrete.
- Fixed headers reduce grading friction across 30+ labs.

Concern:
- Instructors will want a direct rubric mapping for every report section.

Recommended improvement:
- Add a standard report rubric:
  - Objectives addressed.
  - Prediction made before reveal.
  - Evidence used.
  - Track-specific decision.
  - Residual risk.
  - Source trace/assumptions.

### 2. Standalone Adopter: "Can I assign just one lab?"

Likely positive feedback:
- `Chapter Recap`, `Scenario Brief`, and `What You Need To Know` make a standalone lab plausible.

Concern:
- The opening sequence could be too much if assigning one 45-minute lab.

Recommended improvement:
- Add an "Instructor Compact Mode" guidance:
  - Header and objectives stay visible.
  - Chapter recap can be concise.
  - Source trace can be collapsed.
  - Lab map can be compact.

### 3. Teaching Assistant: "I need to know what counts as complete."

Likely positive feedback:
- `Checkpoint` and `Download Report` make completion visible.
- Part-level saved fields should help debug missing reports.

Concern:
- Students may skip parts and still download a report.

Recommended improvement:
- Report should mark missing required parts explicitly.
- UI should show completion state per part in `Lab Map`.
- Report should include an `Incomplete Fields` section if needed.

### 4. Systems Instructor: "The headers are good, but don't over-explain."

Likely positive feedback:
- `What You Need To Know` is the right idea if it is short.

Concern:
- The micro-brief could become a mini textbook and slow down active learning.

Recommended improvement:
- Enforce 2-4 bullets.
- One equation maximum unless the part is specifically about derivation.
- Keep "one thing to watch for" as the last bullet.

### 5. Accessibility-Minded Instructor: "The table fallback requirement is important."

Likely positive feedback:
- Visuals are backed by exact text/table results.
- Failure boundary must state value, limit, unit, and mitigation.

Concern:
- `Evidence` and `Constraint Check` may duplicate information if poorly implemented.

Recommended improvement:
- `Evidence` answers "what happened?"
- `Constraint Check` answers "is this acceptable for the selected track?"
- Keep them visually adjacent.

### 6. Practitioner Guest Lecturer: "The real win is the decision artifact."

Likely positive feedback:
- `Final Decision`, `Residual Risk`, and `Source Trace` look like a design-review artifact.

Concern:
- `Big Takeaways` can become generic if not track-specific.

Recommended improvement:
- Require one track-specific takeaway and one misconception-correction takeaway.
- Avoid vague takeaways such as "trade-offs matter."

## Student Reaction Simulation

### What Students Will Like

- The repeated structure reduces uncertainty.
- `Your Prediction` makes it clear what to do before manipulating controls.
- `Try It` is an approachable label for interaction.
- `Checkpoint` makes progress visible.
- `Download Report` gives a concrete finish line.

### Where Students May Struggle

1. **Too many sections per part**
   - If every header is visually heavy, students may feel like they are reading forms instead of exploring.
   - Fix: use compact section treatments inside parts.

2. **Source Trace may feel optional or intimidating**
   - Students may ignore it unless it is clearly tied to trust and reports.
   - Fix: collapse by default, but include a one-line visible source summary.

3. **Reflection fatigue**
   - Repeating three questions in every part can produce shallow answers.
   - Fix: use structured quick reflections and reserve longer prose for synthesis.

4. **Checkpoint versus Decision**
   - Students may not understand the difference.
   - Fix: `Checkpoint` means "save evidence for later"; `Decision` means "commit to a configuration or policy."

5. **Learning Objectives may be skipped**
   - Students often skip objectives if they are generic.
   - Fix: objectives should be action-oriented and report-aligned.

## Recommended Contract Adjustments

### Keep As-Is

- Fixed lab-level headers.
- Fixed part-level headers.
- Local-first report requirement.
- Required `Big Takeaways`.
- Required `Residual Risk`.
- Report headers.

### Add Or Clarify

1. Add **Compact Rendering Guidance**:
   - Headers are required semantically, but not every section must be a large block.
   - Source Trace, table fallback, and advanced controls can be collapsed.

2. Add **Completion State Rules**:
   - Lab Map should show incomplete, predicted, evidence viewed, checkpoint saved, and decision complete states.

3. Add **Report Completeness Rules**:
   - Download can always happen locally.
   - Missing required fields are marked, not silently omitted.

4. Add **Reflection Scope Rules**:
   - Part reflection is structured and short.
   - Synthesis reflection can include one longer rationale.

5. Add **Takeaway Quality Rules**:
   - At least one track-specific takeaway.
   - At least one misconception corrected.
   - At least one carry-forward note.

## Applied Contract Updates

These feedback items have been folded into `LAB_STRUCTURE_AND_REPORT_CONTRACT.md`:

- Header labels remain fixed, but compact rendering is allowed.
- `Source Trace`, table fallbacks, and advanced controls may be collapsed by default.
- `Lab Map` must distinguish not started, prediction saved, evidence viewed, checkpoint saved, and decision complete states.
- Reports can always download locally, but missing required fields must appear under `Incomplete Fields`.
- Part reflections should be short and structured, with longer prose reserved for Synthesis when needed.
- `Big Takeaways` must include a track-specific takeaway, a misconception correction, and a carry-forward note.
- `Checkpoint` and `Decision` now have distinct meanings.

## Second-Pass Feedback After Adjustments

### Instructor View

The structure is ready to use as the standard for implementation planning. It is now specific enough to support grading, TA review, and reusable report export without forcing every lab to look visually identical.

Remaining instructor asks:
- Provide one completed exemplar report for a pilot lab.
- Add a reusable rubric component that maps directly to the downloaded report headers.
- Define expected time-on-task for default, compact, and extended assignment modes.

### Student View

The structure should make sense if Lab 00 introduces the rhythm explicitly and later labs repeat it consistently. The student experience depends on implementation restraint: each part should feel like prediction, interaction, evidence, and checkpoint, not like a long form.

Remaining student risks:
- Too many visible panels could make short labs feel slow.
- `Source Trace` may be skipped unless its visible one-line summary is useful.
- Students need clear visual progress in `Lab Map` so the report does not feel mysterious.

## Proposed Header-Level Refinement

The header names are good. Do not rename them now.

Potential future rename only if student testing shows confusion:

| Current | Possible alternative | Recommendation |
|---|---|---|
| `Scenario Slice` | `This Part's Scenario` | Keep current for now |
| `Try It` | `Controls` | Keep `Try It`; more student-friendly |
| `Constraint Check` | `Feasibility Check` | Consider if constraints are not always hard limits |
| `Checkpoint` | `Save Checkpoint` | Consider if students miss the save action |

## Implementation Implications

The UI component library should provide:

- `learning_objectives()`
- `track_context()`
- `lab_map()`
- `part_header()`
- `micro_brief()`
- `scenario_slice()`
- `prediction_lock()`
- `try_it_panel()`
- `evidence_panel()`
- `constraint_check()`
- `source_trace()`
- `reflection_card()`
- `checkpoint_card()`
- `synthesis_panel()`
- `big_takeaways()`
- `download_report()`
- `incomplete_report_notice()`

## Decision

The structure is good enough to become the standard. The main caveat remains implementation discipline: labs should feel like guided engineering worksheets, not long compliance forms. The next useful feedback artifact should be an exemplar completed report from one pilot lab, followed by a rubric pass.
