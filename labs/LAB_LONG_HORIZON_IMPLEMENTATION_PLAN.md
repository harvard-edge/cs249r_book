# Lab Track Migration Long-Horizon Plan

This plan breaks the track-aware lab migration into small reviewable commits. Work happens in the `/Users/VJ/GitHub/MLSysBook-labs` worktree on branch `codex/labs`. The permanent `/Users/VJ/GitHub/MLSysBook` checkout remains the local `dev` reference.

## Working Rules

- Commit one coherent change at a time.
- Keep notebook migrations separate from shared infrastructure unless the notebook is the only consumer being proven.
- Keep generated browser wheels in the same commit as the source change that requires them.
- Update `LAB_IMPLEMENTATION_NOTES.md` after each implementation pass.
- Do not add notebook-local hardware/model/system constants when MLSysIM can own them.
- Every migrated lab must support local-first report generation and incomplete-field reporting.

## Commit Sequence

### 1. Planning Baseline

Purpose:
- Land the per-lab track plans and planning governance docs.

Scope:
- `labs/*.md` planning docs.
- `labs/vol1/*.track-plan.md`.
- `labs/vol2/*.track-plan.md`.
- `labs/lab-modality-catalog.html`.

Checks:
- Count 34 notebooks and 34 track plans.
- Verify all plans mention iPhone, Oura Ring, RoboTaxi, Cloud Fleet, and MLSysIM.
- ASCII scan planning docs.

### 2. Track Foundation

Purpose:
- Establish the single source of truth for canonical tracks.

Scope:
- `Hardware.Tiny.OuraRing`.
- `Hardware.Edge.RoboTaxi`.
- `mlsysbook_labs.tracks`.
- Track profile schema.
- Track selector and track context UI.
- Contract-aligned report builder and report fallback.
- Lab 00 ledger save of canonical `track_id`, `hardware_ref`, and `system_ref`.
- Rebuilt browser wheels.

Checks:
- `python3 -m pytest labs/tests/test_track_profiles.py -q`
- `python3 -m pytest mlsysim/tests/test_hardware.py mlsysim/tests/test_provenance_audit.py -q`
- `python3 -m pytest labs/tests/test_static.py::TestWheelConsistency -q`
- Lab 00 AST parse.

### 3. Lab 00 Structure Pass

Purpose:
- Make Lab 00 visibly teach the new track rhythm.

Scope:
- Use contract headers: `Learning Objectives`, `Chapter Recap`, `Your Track`, `Scenario Brief`, `Lab Map`, `Synthesis`, `Big Takeaways`, `Download Report`.
- Add `track_context()` display.
- Add report export for the track-selection artifact.
- Preserve existing concept checks unless replacement is necessary.

Checks:
- Lab 00 AST parse.
- Lab 00 static tests.
- Report builder unit tests.

### 4. Lab Variant Registry

Purpose:
- Add typed scenario variants before migrating content-heavy labs.

Scope:
- `LabTrackVariant` registry or module.
- Pilot variants for Lab 00, V1-10, and V2-11.
- Defaults for track-specific workload, model, metrics, guardrails, and assumptions.

Checks:
- Unit test that every canonical track has a variant for each pilot lab.
- Unit test that every variant references a valid track profile.
- Unit test that hardware/system refs resolve.

### 5. Shared Modalities Implementation

Purpose:
- Convert catalog concepts into reusable helper components.

Scope:
- Learning objectives component.
- Lab map/progress component.
- Prediction lock wrapper.
- Constraint check component.
- Source trace component.
- Structured reflection component.
- Big takeaways component.
- Report export panel.

Checks:
- Unit tests for HTML/text rendering where practical.
- No direct hardware constants in helper components.

### 6. V1-10 Compression Pilot

Purpose:
- Prove the full track-aware pedagogy on the most mature Volume I pilot.

Scope:
- Track selector read from ledger.
- Track-specific narrative for iPhone, Oura Ring, RoboTaxi, Cloud Fleet.
- Compression candidate data from typed variants and MLSysIM solver/result objects.
- Contract headers across opening, parts, synthesis, and report.
- Source trace for hardware/model/solver assumptions.

Checks:
- AST parse.
- Static lab tests.
- Focused report export test.
- Manual source-truth audit for hardware/model constants.

### 7. V1-10 Exemplar Report And Rubric

Purpose:
- Produce instructor-facing evidence before broad migration.

Scope:
- One completed exemplar report.
- Rubric mapping to downloaded report headers.
- Compact/default/extended time-on-task guidance.

Checks:
- Report has all required headers.
- No incomplete fields in exemplar.
- Rubric maps every required report section.

### 8. V2-11 Edge Intelligence Pilot

Purpose:
- Stress RoboTaxi and cloud/edge narrative differences in Volume II.

Scope:
- Track variants for edge intelligence.
- Tail latency and power/source trace emphasis.
- RoboTaxi-first narrative with iPhone/Oura/Cloud realizations.

Checks:
- AST parse.
- Static lab tests.
- Source-truth audit for latency/power constants.

### 9. Pilot Feedback Pass

Purpose:
- Feed implementation lessons back into plans before broad migration.

Scope:
- Update modality catalog.
- Update structure contract if needed.
- Update all track plans only where a reusable lesson applies.
- Update implementation notes.

Checks:
- Planning-doc section completeness script.
- ASCII scan.

### 10. Volume I Batch Migration

Purpose:
- Migrate remaining Volume I labs in small chapter-group commits.

Commit slices:
- V1-01 to V1-04 foundations.
- V1-05 to V1-08 model/compute/build.
- V1-09 to V1-12 optimization/hardware/benchmark.
- V1-13 to V1-16 serving/ops/responsible/conclusion.

Checks per slice:
- AST parse touched labs.
- Static tests touched labs.
- Track variants exist for touched labs.
- Report export path exists.

### 11. Volume II Batch Migration

Purpose:
- Migrate remaining Volume II labs in small chapter-group commits.

Commit slices:
- V2-01 to V2-04 infrastructure/data.
- V2-05 to V2-08 distributed/fleet.
- V2-09 to V2-12 performance/inference/edge/ops.
- V2-13 to V2-17 security/robust/sustainable/responsible/synthesis.

Checks per slice:
- AST parse touched labs.
- Static tests touched labs.
- Track variants exist for touched labs.
- Source trace covers fleet/system facts.

### 12. End-To-End Browser Pass

Purpose:
- Validate the student path across local browser execution.

Scope:
- Lab 00 track selection.
- V1-10 pilot report download.
- V2-11 pilot report download.
- Wheel integrity.
- Browser fallback text area.

Checks:
- Existing browser smoke where available.
- Manual screenshot pass if browser tooling is installed.
- Verify local ledger continuity.

### 13. Cleanup And PR Prep

Purpose:
- Prepare a reviewable branch.

Scope:
- Remove stale temporary artifacts.
- Ensure wheels match source package versions.
- Final implementation notes update.
- Final status and test summary.

Checks:
- Focused tests.
- Full feasible labs test subset.
- Git status review.

## Current Next Task

After the foundation commits, the next task is Commit 3: Lab 00 Structure Pass. It should be a notebook-only pass plus any tiny helper adjustment needed to render the track context and report artifact cleanly.
