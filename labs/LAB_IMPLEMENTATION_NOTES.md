# Lab Implementation Notes

This is the living implementation log for the track-aware lab migration. Update it whenever a lab is implemented or improved.

The purpose is to prevent one-off improvements from staying isolated. If a better component, track assumption, solver API, report field, or instructor workflow emerges while improving one lab, record it here and propagate it to the rest of the plans/labs.

## Current Planning Baseline

- Canonical tracks: iPhone, Oura Ring, RoboTaxi, Cloud Fleet.
- Hardware and model facts must come from MLSysIM.
- Track identity and lab scenarios should live in typed `mlsysbook_labs` metadata.
- V1-10 Compression is the pilot for the detailed track-plan template.
- All other track plans should be upgraded to the V1-10 structure before notebook implementation.

## Entry Template

```text
Date:
Lab:
Track(s):
Files touched:
What changed:
MLSysIM facts/APIs needed:
Notebook-local constants removed:
Reusable component or modality improved:
Plan updates needed in other labs:
Tests or checks run:
Follow-up:
```

## Notes

### 2026-06-03 - Planning Baseline

Lab:
- All labs, planning layer only.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/*.md`
- `labs/vol1/*.track-plan.md`
- `labs/vol2/*.track-plan.md`

What changed:
- Added per-lab track plans and a modality catalog.
- Upgraded V1-10 Compression into the pilot detailed plan format.
- Added this implementation notes log and the single-source-of-truth policy.

MLSysIM facts/APIs needed:
- `Hardware.Tiny.OuraRing`
- `Hardware.Edge.RoboTaxi`
- Cloud Fleet profile or system registry entry beyond a single H100.
- Compression candidate/sweep/Pareto result schema.

Notebook-local constants removed:
- None yet. No notebook code edited in this pass.

Reusable component or modality improved:
- Defined reusable modality catalog for track selector, scenario strip, prediction locks, sliders, strategy selectors, stack builders, constraint budgets, frontiers, source traces, failure boundaries, decision cards, and reports.

Plan updates needed in other labs:
- All plans should include assignment modes, completion path, expected track outcomes, misconceptions, assumptions, accessibility/fallback requirements, data contracts, and rubric sketch.

Tests or checks run:
- Counted 34 lab notebooks and 34 track plans.
- Checked all track plans mention iPhone, Oura Ring, RoboTaxi, and Cloud Fleet.
- ASCII scan clean for new detailed planning docs.

Follow-up:
- Upgrade the remaining track plans to the detailed template.
- Implement track profile registry.
- Add Oura Ring and RoboTaxi hardware registry entries.

### 2026-06-03 - Detailed Plan Propagation

Lab:
- All labs, planning layer only.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/vol1/*.track-plan.md`
- `labs/vol2/*.track-plan.md`
- `labs/LAB_SINGLE_SOURCE_OF_TRUTH_POLICY.md`
- `labs/LAB_IMPLEMENTATION_NOTES.md`

What changed:
- Propagated the V1-10 detailed planning structure across every per-lab track plan.
- Every plan now includes instructor assignment modes, expected track outcomes, common misconceptions, data/solver contracts, single-source-of-truth requirements, accessibility/fallback requirements, a rubric sketch, and continuous-improvement notes.

MLSysIM facts/APIs needed:
- Same baseline needs remain: `Hardware.Tiny.OuraRing`, `Hardware.Edge.RoboTaxi`, a Cloud Fleet profile/system abstraction, and typed solver/result APIs for track-aware lab computations.

Notebook-local constants removed:
- None yet. No notebook code edited in this pass.

Reusable component or modality improved:
- Planning now treats every lab as pedagogy kernel + track realization + modality stack + typed result/report contract.

Plan updates needed in other labs:
- No missing detailed sections remain in the track-plan files.
- Future refinements should be driven by actual implementation discoveries and recorded here.

Tests or checks run:
- Verified all 34 plans include the required detailed sections.
- Verified all 34 plans mention iPhone, Oura Ring, RoboTaxi, Cloud Fleet, and MLSysIM.
- ASCII scan clean across the planning docs.

Follow-up:
- Start implementation by adding MLSysIM registry entries for missing canonical hardware.
- Add `TrackProfile` and `LabTrackVariant` registries before notebook refactors.
- Migrate Lab 00 first, then the V1-10 pilot, then one Volume 2 pilot such as V2-11 Edge Thermodynamics.

### 2026-06-03 - Lab Structure And Local Report Contract

Lab:
- All labs, implementation contract only.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/LAB_STRUCTURE_AND_REPORT_CONTRACT.md`
- `labs/LAB_REALIZATION_MODALITY_CATALOG.md`
- `labs/LAB_SINGLE_SOURCE_OF_TRUTH_POLICY.md`
- `labs/LAB_IMPLEMENTATION_NOTES.md`

What changed:
- Added the formal lab-level and part-level structure contract.
- Required every part to include a part header, "What You Need To Know", scenario slice, prediction, controls, evidence, source trace, reflection, and checkpoint/decision.
- Required every lab to end with synthesis, big takeaways, and a local-first downloadable report.
- Added explicit student-facing header labels for lab sections, part sections, synthesis, and downloaded reports.

MLSysIM facts/APIs needed:
- No new hardware/model facts from this contract.
- Report snapshots need typed result objects from MLSysIM/lab variants so source traces and evidence can be serialized cleanly.

Notebook-local constants removed:
- None. No notebook code edited in this pass.

Reusable component or modality improved:
- Standard part recipe now includes the micro-brief, reflection, checkpoint, and report artifact requirements.
- Header names are now part of the contract: `Learning Objectives`, `Chapter Recap`, `Your Track`, `Scenario Brief`, `Lab Map`, `Your Prediction`, `Try It`, `Evidence`, `Constraint Check`, `Source Trace`, `Reflection`, `Checkpoint`, `Synthesis`, `Big Takeaways`, and `Download Report`.
- Simulated structure feedback added compact rendering, completion-state, and incomplete-report guidance.

Plan updates needed in other labs:
- During implementation, every lab plan should be checked against `LAB_STRUCTURE_AND_REPORT_CONTRACT.md`.
- If one lab develops a better report field, reflection prompt, or source-trace layout, update the contract and propagate it.

Tests or checks run:
- Documentation-only update.

Follow-up:
- Add or update `mlsysbook_labs` helpers for local report export, fallback report text area, part checkpoints, and structured reflections.
- Add tests that verify report export can be produced locally without a backend.

### 2026-06-03 - Structure Feedback Pass

Lab:
- All labs, structure feedback only.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/LAB_STRUCTURE_FEEDBACK_SIMULATION.md`
- `labs/LAB_IMPLEMENTATION_NOTES.md`

What changed:
- Recorded simulated instructor and student feedback on the lab header, part header, synthesis, and local report structure.
- Marked which feedback items were folded into the structure contract.
- Added a second-pass verdict that the structure is ready to become the standard, with implementation discipline around compact rendering.

MLSysIM facts/APIs needed:
- None from this feedback pass.

Notebook-local constants removed:
- None. No notebook code edited in this pass.

Reusable component or modality improved:
- Confirmed the need for reusable components for progress state, report completeness, source trace summaries, structured reflections, and rubric-aligned report sections.

Plan updates needed in other labs:
- Pilot implementation should produce one completed exemplar report and a rubric mapping before broad notebook migration.

Tests or checks run:
- Documentation-only update.

Follow-up:
- Build one pilot report artifact from Lab 00 or V1-10.
- Add a reusable rubric component that maps to the downloaded report headers.
- Define expected time-on-task for compact, default, and extended assignment modes.

### 2026-06-03 - Belts And Knobs Catalog Page

Lab:
- All labs, planning/UI reference only.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/lab-modality-catalog.html`
- `labs/LAB_IMPLEMENTATION_NOTES.md`

What changed:
- Added a static HTML companion page for the modality catalog.
- The page shows reusable belts for lab flow and reusable knobs for controls, visuals, evidence, decisions, and reports.
- Added a track-aware demo showing how the same controls and evidence panels change narrative by track.

MLSysIM facts/APIs needed:
- None added by this page. Demo values are explicitly illustrative placeholders.
- Real implementation should replace demo values with MLSysIM registry facts, solver outputs, and typed lab variants.

Notebook-local constants removed:
- None. No notebook code edited in this pass.

Reusable component or modality improved:
- Made the catalog easier to inspect visually before implementing notebook components.
- Reinforced the distinction between belts as repeatable lab flows and knobs as reusable interactive/rendering devices.

Plan updates needed in other labs:
- Use this catalog as a reference when choosing modality stacks for each lab plan.
- When implementing a new reusable device, add it to the Markdown catalog and keep the HTML companion aligned.

Tests or checks run:
- Static HTML validation pending after this note.

Follow-up:
- After the first pilot implementation, update the page with actual component names from `mlsysbook_labs`.
- Add an exemplar report link once a pilot report artifact exists.

### 2026-06-03 - Track Foundation Implementation Pass

Lab:
- Lab 00, plus shared MLSysIM and `mlsysbook_labs` foundation.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `mlsysim/mlsysim/hardware/data/tiny/OuraRing.yaml`
- `mlsysim/mlsysim/hardware/data/edge/RoboTaxi.yaml`
- `labs/mlsysbook_labs/schemas.py`
- `labs/mlsysbook_labs/tracks.py`
- `labs/mlsysbook_labs/ui.py`
- `labs/mlsysbook_labs/reports.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/vol1/lab_00_introduction.py`
- `labs/tests/test_track_profiles.py`
- `mlsysim/tests/test_hardware.py`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`
- `wheels/mlsysim-0.1.2-py3-none-any.whl`

What changed:
- Added MLSysIM hardware registry entries for `Hardware.Tiny.OuraRing` and `Hardware.Edge.RoboTaxi`.
- Added canonical lab track profiles for iPhone, Oura Ring, RoboTaxi, and Cloud Fleet.
- Mapped legacy category inputs (`mobile`, `tinyml`, `edge`, `cloud`) to canonical track IDs.
- Updated the shared track selector to show the four student-facing tracks.
- Added a reusable `track_context()` renderer for source-traced track cards.
- Updated `build_lab_report()` to emit the contract headers: Lab, Track And Scenario, Learning Objectives, Predictions, Evidence Summary, Final Decision, Big Takeaways, Reflections, Residual Risk, and Source Trace.
- Added `Incomplete Fields` reporting for missing required report sections.
- Added `report_text_fallback()` and `report_export_panel()` so labs can expose a local Markdown fallback if browser download fails.
- Updated Lab 00 so the selected regime saves a canonical track ID and MLSysIM source references to the Design Ledger.
- Rebuilt the browser wheels so Lab 00's WASM path can import the new track helpers and hardware registry entries.

MLSysIM facts/APIs needed:
- `Hardware.Tiny.OuraRing` now exists as an estimate-backed wearable reference profile.
- `Hardware.Edge.RoboTaxi` now exists as an estimate-backed DRIVE AGX Orin-class reference profile.
- Cloud Fleet uses existing `Hardware.Cloud.H100` and `Systems.Clusters.Lab_64_H100`.

Notebook-local constants removed:
- None removed yet. Lab 00 still contains legacy explanatory copy and demo constants; it now records canonical source references for downstream labs.

Reusable component or modality improved:
- Track selection, track context display, report completeness, and report fallback are now reusable helper APIs.

Plan updates needed in other labs:
- Future lab migrations should read the selected canonical track from the Design Ledger and call `get_track_profile()` before choosing scenario variants.
- Pilot labs should pass learning objectives, big takeaways, source trace, and evidence summary into `build_lab_report()`.
- Compression pilot should use `Hardware.Tiny.OuraRing`, `Hardware.Edge.RoboTaxi`, `Hardware.Mobile.iPhone15Pro`, and Cloud Fleet refs rather than notebook-local hardware constants.

Tests or checks run:
- `python3 -m pytest labs/tests/test_track_profiles.py -q`
- `python3 -m pytest mlsysim/tests/test_hardware.py -q`
- `python3 -m pytest mlsysim/tests/test_provenance_audit.py mlsysim/tests/test_system_registry.py -q`
- `python3 -m pytest labs/tests/test_static.py::TestLabCatalog -q`
- `python3 -m pytest labs/tests/test_static.py::TestWheelConsistency::test_mlsysbook_labs_wheel_present_when_imported labs/tests/test_static.py::TestRequiredImports::test_imports_design_ledger --tb=short -q`
- `python3 -m pytest labs/tests/test_static.py::TestWheelConsistency -q`
- `python3` AST parse check for `labs/vol1/lab_00_introduction.py`
- Verified `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` contains `mlsysbook_labs/tracks.py`.
- Verified `wheels/mlsysim-0.1.2-py3-none-any.whl` contains the Oura Ring and RoboTaxi hardware YAML entries.

Follow-up:
- Migrate the Lab 00 visible structure to the full header contract in a separate pass.
- Build the V1-10 Compression pilot on the new profile/report APIs.
- Add typed `LabTrackVariant` entries for each pilot lab before broad notebook migration.

### 2026-06-03 - Lab 00 Structure Pass

Lab:
- V1-00 The Architect's Portal.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/vol1/lab_00_introduction.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`

What changed:
- Added explicit Lab 00 metadata and learning-objective tuples.
- Added visible contract headers for `Learning Objectives`, `Chapter Recap`, `Scenario Brief`, `Your Track`, `Lab Map`, `Big Takeaways`, and `Download Report`.
- Added the reusable `track_context()` panel after track selection so Lab 00 shows the selected canonical track and MLSysIM source references.
- Added local report generation with `build_lab_report()` and `report_export_panel()`.
- The downloaded report records the selected track, hardware reference, optional system reference, check answers, big takeaways, source trace, and residual risk.

MLSysIM facts/APIs needed:
- No new MLSysIM facts. Lab 00 now consumes canonical track profiles and source references created in the foundation pass.

Notebook-local constants removed:
- None. This was a structure/report pass; the remaining Lab 00 explanatory constants will be handled only if they become active solver inputs.

Reusable component or modality improved:
- Proved that Lab 00 can use `track_context()` and the local report export panel without rewriting the existing concept checks.

Plan updates needed in other labs:
- V1-10 should follow the same local report pattern but with solver-backed evidence instead of orientation-only evidence.
- The report should be complete only when predictions, evidence summary, final decision, big takeaways, reflections, residual risk, and source trace are present.

Tests or checks run:
- `python3` AST parse check for `labs/vol1/lab_00_introduction.py`
- `python3 -m pytest labs/tests/test_track_profiles.py -q`
- `python3 -m pytest labs/tests/test_static.py::TestWheelConsistency labs/tests/test_static.py::TestRequiredImports::test_imports_design_ledger --tb=short -q`
- `rg` check for required Lab 00 contract headers and helper calls.

Follow-up:
- Start the Lab Variant Registry pass.
- Then migrate V1-10 Compression as the first content-heavy pilot.

### 2026-06-03 - Pilot Lab Variant Registry

Lab:
- V1-00 The Architect's Portal.
- V1-10 Compression Paradox.
- V2-11 Edge Thermodynamics.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/variants.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/tests/test_lab_variants.py`
- `labs/tests/test_static.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added typed `LabTrackVariant` entries for the first three pilot labs.
- Each pilot lab now has one scenario variant per canonical track.
- Variants carry stakeholder, workload summary, objective, primary metric, guardrail metric, hardware ref, optional system ref, model ref, defaults, and assumptions.
- Added lookup helpers: `list_lab_variants()`, `get_lab_track_variant()`, `variant_coverage()`, and `canonical_track_ids()`.
- Exported the variant registry from `mlsysbook_labs`.
- Updated the wheel contract test so browser wheels must include `tracks.py` and `variants.py`.
- Rebuilt the `mlsysbook_labs` browser wheel.

MLSysIM facts/APIs needed:
- No new hardware or model facts. Variants reference existing MLSysIM model paths:
  - `Models.Vision.MobileNetV2`
  - `Models.Tiny.DS_CNN`
  - `Models.Tiny.AnomalyDetector`
  - `Models.Vision.YOLOv8_Nano`
  - `Models.Language.BERT_Base`

Notebook-local constants removed:
- None. This was a metadata foundation pass before notebook migration.

Reusable component or modality improved:
- Scenario defaults and assumptions now have a typed home before V1-10 and V2-11 implementation.

Plan updates needed in other labs:
- After V1-10 and V2-11 prove the registry shape, extend `variants.py` or split it into per-volume modules for the remaining labs.
- Add source-trace rendering from `LabTrackVariant` during notebook migration.

Tests or checks run:
- `python3 -m pytest labs/tests/test_lab_variants.py -q`
- `python3 -m pytest labs/tests/test_track_profiles.py -q`
- `python3 -m pytest labs/tests/test_static.py::TestWheelConsistency -q`
- Verified `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` contains `mlsysbook_labs/tracks.py` and `mlsysbook_labs/variants.py`.

Follow-up:
- Implement shared modality helpers needed by V1-10.
- Begin V1-10 Compression pilot migration using `get_lab_track_variant()`.

### 2026-06-03 - Shared Modality Helper Layer

Lab:
- All labs, shared `mlsysbook_labs` UI layer.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/ui.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/tests/test_ui_helpers.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added reusable helpers for the contract sections: `learning_objectives()`, `lab_map()`, `part_header()`, `what_you_need_to_know()`, `scenario_slice()`, `constraint_check()`, `source_trace()`, `evidence_summary()`, `checkpoint_card()`, and `big_takeaways()`.
- Added `COMPLETION_STATES` with the required lab-map order: not started, prediction saved, evidence viewed, checkpoint saved, and decision complete.
- Added compact shared CSS for lists, status pills, part titles, and collapsed source traces.
- Exported the helpers from `mlsysbook_labs` so browser notebooks can import the same structure vocabulary.

MLSysIM facts/APIs needed:
- No new MLSysIM facts. The helpers are renderers; they expect hardware, model, system, solver, and scenario values to come from MLSysIM registries or typed lab variant metadata.

Notebook-local constants removed:
- None. This pass creates the shared target APIs before migrating content-heavy notebooks.

Reusable component or modality improved:
- Lab-level structure, part-level structure, source trace, constraint check, evidence summary, checkpoint, big takeaways, and completion-state rendering now have one package-level implementation.

Plan updates needed in other labs:
- V1-10 should consume these helpers directly instead of writing notebook-local section HTML.
- Future pilots should add missing helper APIs here first if a new modality is genuinely reusable.

Tests or checks run:
- `python3 -m py_compile labs/mlsysbook_labs/ui.py labs/mlsysbook_labs/__init__.py labs/tests/test_ui_helpers.py`
- `python3 -m pytest labs/tests/test_ui_helpers.py -q`
- `python3 -m pytest labs/tests/test_ui_helpers.py labs/tests/test_track_profiles.py labs/tests/test_lab_variants.py -q`
- `python3 -m pytest labs/tests/test_static.py::TestWheelConsistency -q`
- Verified `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` contains the shared helper API in `mlsysbook_labs/ui.py`.

Follow-up:
- Begin V1-10 Compression pilot migration using the shared helper layer.

### 2026-06-03 - V1-10 Track-Aware Outer Contract Pass

Lab:
- V1-10 The Compression Paradox.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/vol1/lab_10_model_compress.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`

What changed:
- Added the `mlsysbook_labs` browser wheel install to the V1-10 WASM bootstrap.
- Imported the shared lab metadata, track, structure, source-trace, and report helpers.
- Added V1-10 `LabMetadata`, `ChapterRecap`, learning objectives, and big takeaways.
- Added a canonical track selector that defaults from the local Design Ledger when Lab 00 has already stored a track.
- Loaded the V1-10 `LabTrackVariant` for the selected canonical track.
- Replaced the old opening hero/briefing with the shared academic header, `Learning Objectives`, `Chapter Recap`, `Your Track`, `Scenario Brief`, `Lab Map`, and `Source Trace` blocks.
- Added a local `Download Report` section that records selected track, scenario variant, source trace, and prediction values while marking solver-backed evidence and the final recipe as incomplete.
- Updated the ledger save payload to include canonical `track_id`, `scenario_id`, `hardware_ref`, `system_ref`, `model_ref`, primary metric, guardrail metric, and completion state.

MLSysIM facts/APIs needed:
- No new MLSysIM facts. This pass consumes existing track profiles and pilot variants.
- The remaining V1-10 calculation code still needs a source-traced compression candidate/sweep API before notebook-local constants can be removed safely.

Notebook-local constants removed:
- None yet. H100, iPhone, Jetson, ResNet-50, MobileNetV2, and Llama constants are still used by the legacy Parts A-E calculations.

Reusable component or modality improved:
- V1-10 now proves the shared helper layer can drive a content-heavy lab opening and a local report skeleton.

Plan updates needed in other labs:
- Later migrations should copy this outer-contract pattern before changing part internals.
- The compression solver pass should determine whether reusable compression result types belong in MLSysIM or `mlsysbook_labs` metadata before moving formulas out of the notebook.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_10_model_compress.py`
- `python3 -m pytest labs/tests/test_static.py::TestSyntax::test_ast_parse labs/tests/test_static.py::TestWheelConsistency::test_mlsysbook_labs_wheel_present_when_imported labs/tests/test_static.py::TestWheelConsistency::test_micropip_url_matches_pyproject_version --tb=short -q labs/vol1/lab_10_model_compress.py`
- `rg` check for `mlsysbook_labs`, `Scenario Brief`, `Lab Map`, `Source Trace`, `Download Report`, `build_lab_report`, and `ledger.save`.

Follow-up:
- Implement a shared compression candidate/frontier result layer before changing Parts A-E.
- Migrate Parts A-C first because they are the core compression pedagogy and report evidence path.
