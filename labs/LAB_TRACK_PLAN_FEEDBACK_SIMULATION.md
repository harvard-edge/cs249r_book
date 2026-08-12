# Lab Track Plan Feedback Simulation

This document simulates feedback on the V1-10 Compression pilot plan and the lab realization modality catalog. The goal is not to replace real user testing. The goal is to pressure-test whether the plan is adoption-ready before scaling the format across all labs.

The highest-value feedback should come from instructors because this planning layer is primarily an adoption and implementation artifact. Student feedback matters later, after the notebook experience exists, but students are less likely to comment usefully on schemas, modality catalogs, report contracts, and instructor packaging.

## Reviewed Drafts

- `labs/LAB_REALIZATION_MODALITY_CATALOG.md`
- `labs/vol1/lab_10_model_compress.track-plan.md`

## Simulated Review Personas

| Persona | What they care about |
|---|---|
| Course instructor using the full sequence | Consistency, grading load, progression across labs, track persistence |
| Instructor adopting one lab standalone | Setup time, assignment prompt, expected report, rubric |
| Teaching assistant | Clear expected outputs, common wrong answers, how to help stuck students |
| Systems instructor at another institution | Whether the lab maps to their syllabus and can run in one class session |
| ML systems practitioner guest lecturer | Whether scenarios sound realistic and decisions are defensible |
| Accessibility-minded instructor | Whether students can use the interactions and reports without visual-only cues |
| Student experience proxy | Whether the task feels coherent and not like arbitrary widgets |

## Instructor Feedback Simulation

### 1. The Track Model Is Strong, But Needs An Assignment-Level Contract

Likely instructor reaction:
- "I understand why iPhone, Oura Ring, RoboTaxi, and Cloud Fleet are the four tracks. But for assigning this lab, I need to know whether students all choose one track, whether I assign tracks to groups, or whether the lab expects them to compare all tracks."

Improvement:
- Each track plan should include an **Instructor Assignment Modes** section:
  - Individual choice: student picks one track and follows it across the course.
  - Assigned track: instructor assigns tracks to teams for comparison.
  - Lecture demo: instructor demonstrates two contrasting tracks.
  - Capstone mode: students must keep the same track across a volume.

Plan impact:
- Add an instructor-facing line to every plan: "Default adoption mode: individual choice, ledger-persistent track."
- Add optional "track lock" support in implementation.

### 2. The Modality Catalog Is Useful, But Instructors Need A Rubric Mapping

Likely instructor reaction:
- "The catalog tells me what UI pieces exist, but I still need to grade the report. What counts as a good answer?"

Improvement:
- Add rubric categories to each track plan:
  - Prediction discipline.
  - Evidence use.
  - Trade-off reasoning.
  - Track-specific constraint awareness.
  - Residual risk.

Plan impact:
- Every lab plan should include a small **Rubric Sketch** with 4-5 grading criteria.
- The report export should align to those criteria.

### 3. The Pilot Needs Expected Outputs For Each Track

Likely instructor reaction:
- "If students submit four different track reports, I need to know what a reasonable answer looks like for each one."

Improvement:
- Each plan should include **Expected Track Outcomes**:
  - iPhone: likely recipe, likely guardrail, likely misconception.
  - Oura Ring: likely recipe, likely guardrail, likely misconception.
  - RoboTaxi: likely recipe, likely guardrail, likely misconception.
  - Cloud Fleet: likely recipe, likely guardrail, likely misconception.

Plan impact:
- V1-10 should say, for example, that unstructured pruning may look attractive but should fail if hardware support is absent.
- These outcomes can be approximate; they are teaching anchors, not answer keys.

### 4. Timing And Scope Need To Be Explicit

Likely instructor reaction:
- "The plan is detailed, but how long does this take? Can I run part of it live?"

Improvement:
- Add a **Timebox** section:
  - Opening: 5 minutes.
  - Part A: 10-12 minutes.
  - Part B: 15 minutes.
  - Part C: 10 minutes.
  - Synthesis/report: 8 minutes.
  - Short lecture demo path: Opening + Part A + one track reveal.

Plan impact:
- Every plan should include a full-lab time estimate and a "minimum viable classroom demo."

### 5. Instructors Want Misconception Notes

Likely instructor reaction:
- "The compression lab should explicitly tell me what wrong idea it is designed to break."

Improvement:
- Add **Common Misconceptions**:
  - Smaller model always runs faster.
  - 90 percent sparsity means 10x speedup.
  - INT8 always preserves accuracy.
  - Compression is only about model size.
  - Cloud and device compression goals are the same.

Plan impact:
- Each prediction reveal should map to one misconception.
- Instructor adoption pack should list discussion prompts for those misconceptions.

### 6. Track Narratives Are Good, But Need Stable Stakeholder Prompts

Likely instructor reaction:
- "The track narratives are compelling, but I need a prompt I can put directly in an assignment."

Improvement:
- Add one reusable stakeholder message per track per lab.

Plan impact:
- For V1-10, each track should have a 2-3 sentence stakeholder prompt:
  - Mobile product lead.
  - Wearable firmware lead.
  - Safety/perception lead.
  - Infrastructure lead.

### 7. The Plan Needs Prerequisites And Book Anchors

Likely instructor reaction:
- "I need to know what students should read and what concepts I should refresh before assigning this."

Improvement:
- Add **Prerequisites And Anchors**:
  - Primary chapter.
  - Prior labs.
  - Concepts to refresh.

Plan impact:
- This can be generated from the existing catalog and chapter metadata.

### 8. Provenance And Assumptions Should Be Instructor-Visible

Likely instructor reaction:
- "If a student asks where Oura Ring battery assumptions came from, I need a clean answer."

Improvement:
- Add assumption/provenance requirements:
  - Hardware facts from MLSysIM hardware registry.
  - Scenario thresholds from lab track variant.
  - Approximate or convention values labeled clearly.

Plan impact:
- Every plan should include an **Assumptions To Surface** section.
- The source trace modality should be mandatory whenever a lab displays hardware-derived numbers.

### 9. Accessibility Needs To Be Planned, Not Retrofitted

Likely instructor reaction:
- "If the main evidence is a colored Pareto scatter or heatmap, some students will need a table or label fallback."

Improvement:
- Require:
  - Table fallback for Pareto frontiers and heatmaps.
  - Color-independent feasibility labels.
  - Keyboard-usable controls.
  - Text report that contains the actual result.

Plan impact:
- The modality catalog already mentions table fallback; per-lab plans should name the fallback where the visual is required.

### 10. The Pilot Should Distinguish Required Versus Optional Interactions

Likely instructor reaction:
- "The plan has many modalities. Which are required to complete the assignment?"

Improvement:
- Mark each modality as:
  - Required for report.
  - Optional exploration.
  - Instructor/demo only.

Plan impact:
- Every lab plan should include a **Completion Path**: the shortest path through the lab that produces a valid report.

## Student Reaction Simulation

Student feedback is simulated as a usability proxy. Real student feedback should happen after Lab 00 and one fully implemented pilot exist.

### What Students Will Probably Like

- A single track identity makes the sequence feel coherent.
- Concrete devices make constraints feel less abstract.
- The same lab structure across tracks reduces cognitive load.
- Prediction before reveal gives a clear task before charts appear.
- Reports give students a tangible artifact instead of "I clicked through a notebook."

### Where Students May Get Confused

1. **Track versus device**
   - Students may ask whether "iPhone" means all mobile systems or exactly one phone model.
   - Fix: label iPhone as the canonical Mobile ML profile, not the universe of mobile devices.

2. **Why their track sees different metrics**
   - Students may wonder why Oura Ring focuses on flash while Cloud Fleet focuses on cost.
   - Fix: scenario strip must explain primary and guardrail metrics every part.

3. **Too many visual artifacts**
   - If a part has a Pareto plot, table, budget stack, source trace, and decision card all visible at once, students may not know where to look.
   - Fix: one main visual per part; table/source trace can be collapsible.

4. **Report fatigue**
   - If every part asks for free-form reflection, students may write shallow text.
   - Fix: use structured choices plus one short rationale and one residual-risk field.

5. **Advanced knobs**
   - Students may overfit to optional controls and lose the core concept.
   - Fix: advanced knobs hidden by default and excluded from required report path.

### Student-Facing Improvements

- Use a persistent "You are working as the [track role]" line at the top.
- Keep each part to one primary question.
- Use one primary chart per part.
- Make the final decision card feel like the real assignment target.
- Show "what changed because of your track" explicitly after each reveal.

## Recommended Template Upgrades

Every `*.track-plan.md` should eventually include these sections:

1. Purpose.
2. Shared pedagogy.
3. Track profiles used.
4. Instructor assignment modes.
5. Prerequisites and book anchors.
6. Modality stack.
7. Lab flow with track realization.
8. Required versus optional modalities.
9. Expected track outcomes.
10. Common misconceptions.
11. Data and solver contracts.
12. Assumptions to surface.
13. Accessibility and fallback requirements.
14. Ledger and report.
15. Rubric sketch.

## V1-10 Pilot Improvement Backlog

Priority order:

1. Add instructor assignment modes.
2. Add expected track outcomes.
3. Add rubric sketch.
4. Add common misconceptions.
5. Add completion path.
6. Add assumptions/provenance section.
7. Add accessibility/fallback section.
8. Add direct stakeholder prompts for each track.

## Implementation Implications

The feedback suggests the implementation should include:

- `TrackProfile` registry.
- `LabTrackVariant` registry.
- `InstructorPack` metadata.
- `RubricCriterion` metadata.
- `ExpectedOutcome` metadata per lab and track.
- Report generator aligned to rubric fields.
- Track lock or assignment-mode setting.
- Table fallback APIs for every nontrivial visual.

## Decision

Do not scale the current short per-lab plan format directly. Use V1-10 as the pilot and upgrade it once with instructor-facing and student-usability sections. After that, apply the upgraded template mechanically across the other 33 plans.
