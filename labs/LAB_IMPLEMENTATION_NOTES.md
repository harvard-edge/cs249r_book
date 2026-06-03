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

### 2026-06-03 - MLSysIM Compression Candidate Layer

Lab:
- V1-10 The Compression Paradox, shared MLSysIM solver layer.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `mlsysim/mlsysim/engine/results.py`
- `mlsysim/mlsysim/engine/solvers/compression.py`
- `mlsysim/tests/test_compression_candidates.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysim-0.1.2-py3-none-any.whl`

What changed:
- Added typed `CompressionCandidate` and `CompressionSweepResult` result models.
- Extended `CompressionModel` with `candidate()` for one source-traced compression configuration and `sweep()` for a candidate list with Pareto status.
- Candidate evaluation now records method, bit width, sparsity, compressed size, compression ratio, estimated accuracy delta, memory savings, speedup, hardware-support status, feasibility, binding constraint, guardrail violations, and source trace.
- Sweep evaluation now marks candidates as `frontier` or `dominated` and selects a best feasible frontier candidate for the default objective.
- Rebuilt the MLSysIM browser wheel so lab notebooks can use the new methods in WASM.

MLSysIM facts/APIs needed:
- This pass adds the reusable solver API needed by V1-10 Parts A-C.
- Track-specific guardrail thresholds still need to be supplied by typed lab variants or a dedicated scenario contract.
- Hardware support currently uses explicit precision entries and sparsity conventions from the hardware object; if a device needs INT8/NPU support represented, it must be added to the hardware registry rather than a notebook.

Notebook-local constants removed:
- None yet. This pass creates the MLSysIM API that will replace notebook-local compression candidate calculations in the next V1-10 pass.

Reusable component or modality improved:
- Compression candidate, feasibility, source trace, and Pareto metadata now have a single typed source in MLSysIM.

Plan updates needed in other labs:
- V1-10 should migrate Parts A-C to call `CompressionModel.candidate()` and `CompressionModel.sweep()`.
- Any future compression, serving, or hardware lab that needs candidate/frontier reporting should consume these typed results instead of rebuilding the logic.

Tests or checks run:
- `python3 -m py_compile mlsysim/mlsysim/engine/results.py mlsysim/mlsysim/engine/solvers/compression.py mlsysim/tests/test_compression_candidates.py`
- `python3 -m pytest mlsysim/tests/test_compression_candidates.py -q`
- `python3 -m pytest mlsysim/tests/test_solver_suite.py::TestCompressionModel -q`
- `python3 -m pytest mlsysim/tests/test_compression_candidates.py mlsysim/tests/test_solver_suite.py::TestCompressionModel mlsysim/tests/test_solver_module_exports.py -q`
- `python3 -m pytest labs/tests/test_static.py::TestWheelConsistency::test_micropip_url_matches_pyproject_version -q`
- Verified `wheels/mlsysim-0.1.2-py3-none-any.whl` contains `CompressionModel.candidate()`, `CompressionModel.sweep()`, `CompressionCandidate`, and `CompressionSweepResult`.

Follow-up:
- Migrate V1-10 Parts A-C to use the new MLSysIM compression candidate and sweep APIs.
- Decide whether iPhone INT8 fast-path support should be encoded in `Hardware.Mobile.iPhone15Pro` before making iPhone-specific V1-10 feasibility claims.

### 2026-06-03 - MLSysIM Registry Reference Resolver

Lab:
- All labs, shared `mlsysbook_labs` helper layer.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/registry_refs.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/tests/test_registry_refs.py`
- `labs/tests/test_static.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added `resolve_mlsysim_ref()` to resolve canonical strings such as `Hardware.Tiny.OuraRing`, `Models.Tiny.DS_CNN`, and `Systems.Clusters.Lab_64_H100`.
- Exported the resolver from `mlsysbook_labs`.
- Added tests for hardware, model, system, unsupported-root, and missing-path behavior.
- Updated the browser-wheel contract so `registry_refs.py` must ship in `mlsysbook_labs`.
- Rebuilt the `mlsysbook_labs` browser wheel.

MLSysIM facts/APIs needed:
- No new facts. This helper resolves existing MLSysIM registry objects from typed lab variant references.

Notebook-local constants removed:
- None yet. The next V1-10 notebook pass can use this resolver instead of local maps from reference strings to objects.

Reusable component or modality improved:
- Track variants can now carry canonical registry strings while notebooks resolve the actual MLSysIM object through one shared helper.

Plan updates needed in other labs:
- Future lab migrations should call `resolve_mlsysim_ref()` when turning typed variant refs into MLSysIM objects.

Tests or checks run:
- `python3 -m py_compile labs/mlsysbook_labs/registry_refs.py labs/mlsysbook_labs/__init__.py labs/tests/test_registry_refs.py`
- `python3 -m pytest labs/tests/test_registry_refs.py -q`
- `python3 -m pytest labs/tests/test_registry_refs.py labs/tests/test_lab_variants.py -q`
- `python3 -m pytest labs/tests/test_registry_refs.py labs/tests/test_static.py::TestWheelConsistency -q`
- Verified `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` contains `mlsysbook_labs/registry_refs.py` and exports `resolve_mlsysim_ref`.

Follow-up:
- Use `resolve_mlsysim_ref()` in V1-10 when building compression candidates from the selected track variant.

### 2026-06-03 - V1-10 MLSysIM Candidate Evidence Pass

Lab:
- V1-10 The Compression Paradox.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/vol1/lab_10_model_compress.py`
- `labs/mlsysbook_labs/variants.py`
- `labs/tests/test_lab_variants.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added V1-10 compression guardrail defaults to each track variant: size-limit source, maximum accuracy drop, minimum speedup, and hardware-support requirement.
- Updated V1-10 to resolve the selected variant's `model_ref` and `hardware_ref` with `resolve_mlsysim_ref()`.
- Added a reactive `CompressionModel.sweep()` cell that evaluates quantization and pruning candidates for the selected track.
- Added a visible MLSysIM-backed `Evidence Summary` with best feasible frontier candidate, frontier candidates, dominated candidates, feasible count, size limit, and guardrail source.
- Updated the local report export to include compression candidate rows, best candidate, frontier labels, dominated labels, and solver source trace.
- Updated the Design Ledger payload to save the same plain candidate snapshot.
- Rebuilt the `mlsysbook_labs` browser wheel so the updated variant defaults are available in WASM.

MLSysIM facts/APIs needed:
- This pass consumes `CompressionModel.sweep()` and `resolve_mlsysim_ref()`.
- iPhone fast-path support remains conservative because `Hardware.Mobile.iPhone15Pro` has no explicit `int8` precision entry. The lab records that through hardware-support feasibility rather than adding a notebook override.

Notebook-local constants removed:
- The report and opening evidence path no longer depend on notebook-local compression candidate calculations.
- Legacy Parts A-E still contain their original calculation snippets and will be migrated part-by-part.

Reusable component or modality improved:
- V1-10 now demonstrates the intended flow: typed track variant -> MLSysIM registry objects -> MLSysIM compression sweep -> evidence summary -> ledger/report snapshot.

Plan updates needed in other labs:
- Other labs with typed variants should follow this ref-resolution pattern when turning `hardware_ref`, `model_ref`, or `system_ref` into solver inputs.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_10_model_compress.py labs/mlsysbook_labs/variants.py labs/tests/test_lab_variants.py`
- `python3 -m pytest labs/tests/test_lab_variants.py labs/tests/test_registry_refs.py -q`
- `python3 -m pytest labs/tests/test_static.py::TestSyntax::test_ast_parse labs/tests/test_static.py::TestWheelConsistency::test_mlsysbook_labs_wheel_present_when_imported labs/tests/test_static.py::TestWheelConsistency::test_micropip_url_matches_pyproject_version --tb=short -q labs/vol1/lab_10_model_compress.py`
- Runtime sweep check across all four V1-10 variants with `PYTHONPATH=mlsysim:labs`.
- `python3 -m pytest labs/tests/test_lab_variants.py labs/tests/test_registry_refs.py labs/tests/test_static.py::TestWheelConsistency -q`
- `python3 -m pytest mlsysim/tests/test_compression_candidates.py -q`
- Verified `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` contains the V1-10 guardrail defaults.

Follow-up:
- Migrate the internals of Parts A-C so the interactive controls and visuals read from the MLSysIM candidate/sweep outputs instead of legacy notebook formulas.
- Add structured reflection/checkpoint widgets for the final recipe so the report can become complete.

### 2026-06-03 - V1-10 Synthesis And Complete Report Pass

Lab:
- V1-10 The Compression Paradox.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/vol1/lab_10_model_compress.py`
- `labs/mlsysbook_labs/variants.py`
- `labs/tests/test_lab_variants.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added track-specific validation-test options to the V1-10 variant defaults.
- Added a structured `Synthesis` block with final recipe choice, validation-test choice, diagnosis reflection, tradeoff reflection, and residual-risk field.
- Updated report generation so `Final Decision`, `Reflections`, and `Residual Risk` come from local widgets.
- Updated the ledger payload to save final decision, validation test, reflections, and residual risk alongside the compression candidate snapshot.
- Rebuilt the `mlsysbook_labs` browser wheel so validation-test defaults ship to WASM notebooks.

MLSysIM facts/APIs needed:
- No new facts. This pass uses existing candidate evidence and typed variant defaults.

Notebook-local constants removed:
- None. This pass completes the report contract around the solver-backed evidence.
- Legacy Part A-C internals still need migration from local formulas to `CompressionModel.candidate()` and `CompressionModel.sweep()`.

Reusable component or modality improved:
- V1-10 now has the local-first report loop: prediction values, solver-backed evidence, final decision, structured reflection, residual risk, and downloadable report.

Plan updates needed in other labs:
- Future lab migrations should add structured synthesis widgets before claiming the report is complete.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_10_model_compress.py labs/mlsysbook_labs/variants.py labs/tests/test_lab_variants.py`
- `python3 -m pytest labs/tests/test_lab_variants.py labs/tests/test_static.py::TestSyntax::test_ast_parse labs/tests/test_static.py::TestWheelConsistency::test_mlsysbook_labs_wheel_present_when_imported --tb=short -q labs/vol1/lab_10_model_compress.py`
- `python3 -m pytest labs/tests/test_lab_variants.py labs/tests/test_registry_refs.py labs/tests/test_static.py::TestWheelConsistency -q`
- `python3 -m pytest mlsysim/tests/test_compression_candidates.py -q`
- Verified `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` contains the V1-10 validation-test defaults.
- Note: a combined labs-plus-MLSysIM pytest invocation was split into separate commands because both suites expose a top-level `tests.conftest`.

Follow-up:
- Migrate Part A, Part B, and Part C internals to render directly from MLSysIM candidate/sweep outputs.
- Consider adding a reusable structured synthesis widget helper if V2-11 repeats this pattern.

### 2026-06-03 - All-Lab Baseline Variant Coverage

Lab:
- All 34 cataloged labs.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/variants.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/tests/test_lab_variants.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added `ALL_LAB_IDS` from the lab catalog.
- Kept hand-authored variants for Lab 00, V1-10, and V2-11.
- Added generated baseline variants for every non-pilot catalog lab and every canonical track.
- Baseline variants use canonical track profiles for stakeholder, hardware/system refs, primary metrics, guardrails, dominant constraints, and source policy.
- Baseline variants use stable model refs by track: MobileNetV2 for iPhone, DS-CNN for Oura Ring, YOLOv8-Nano for RoboTaxi, and BERT-Base for Cloud Fleet.
- Updated variant coverage tests to require all 34 labs to expose all four canonical track variants.
- Rebuilt the `mlsysbook_labs` browser wheel.

MLSysIM facts/APIs needed:
- No new facts. Baseline variants point to existing MLSysIM model, hardware, and system registries.
- Lab-specific solver thresholds still need hand-authored defaults when a notebook migration needs more than the baseline variant.

Notebook-local constants removed:
- None. This pass creates source-of-truth track/scenario coverage before batch notebook edits.

Reusable component or modality improved:
- Every lab now has a typed track variant entry that can drive track context, scenario brief, source trace, ledger metadata, and report exports.

Plan updates needed in other labs:
- Batch notebook migrations can now depend on `get_lab_track_variant()` for every catalog lab.
- Replace baseline variants with hand-authored variants as individual labs receive deeper solver-backed migrations.

Tests or checks run:
- `python3 -m py_compile labs/mlsysbook_labs/variants.py labs/mlsysbook_labs/__init__.py labs/tests/test_lab_variants.py`
- `python3 -m pytest labs/tests/test_lab_variants.py labs/tests/test_registry_refs.py -q`
- `python3 -m pytest labs/tests/test_lab_variants.py labs/tests/test_registry_refs.py labs/tests/test_static.py::TestWheelConsistency -q`
- Verified `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` contains `ALL_LAB_IDS`, `_baseline_variants`, and `baseline_variant_pending_notebook_migration`.

Follow-up:
- Add a compact shared migration shell for legacy notebooks so batch migration does not duplicate outer-contract/report boilerplate.

### 2026-06-03 - Legacy Migration Shell Helper

Lab:
- All legacy notebooks pending batch migration.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/migration.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/tests/test_migration_shell.py`
- `labs/tests/test_static.py`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added `baseline_learning_objectives()` and `baseline_big_takeaways()` for not-yet-deep-migrated labs.
- Added `variant_source_trace()` for serializable track/source metadata.
- Added `build_migration_report()` to create a local report from metadata, track profile, and lab variant.
- Added `legacy_migration_panel()` to render learning objectives, track context, scenario brief, source trace, big takeaways, and local report export from one shared helper.
- Exported the migration helpers from `mlsysbook_labs`.
- Added tests for baseline copy, source trace, and incomplete-field behavior.
- Updated the wheel contract and rebuilt the browser wheel.

MLSysIM facts/APIs needed:
- No new facts. The helper consumes `LabMetadata`, `TrackProfile`, and `LabTrackVariant` source-of-truth values.

Notebook-local constants removed:
- None. The next batch pass can import this helper instead of copy-pasting track/report boilerplate into every notebook.

Reusable component or modality improved:
- Legacy notebooks now have a compact path to become track-aware and report-capable before deeper part-level solver migration.

Plan updates needed in other labs:
- Batch migrations should call `legacy_migration_panel()` as the baseline outer-contract layer.
- Deeper lab-specific migrations can replace the baseline report fields with solver-backed evidence and hand-authored reflections.

Tests or checks run:
- `python3 -m py_compile labs/mlsysbook_labs/migration.py labs/mlsysbook_labs/__init__.py labs/tests/test_migration_shell.py`
- `python3 -m pytest labs/tests/test_migration_shell.py labs/tests/test_lab_variants.py labs/tests/test_ui_helpers.py -q`
- `python3 -m pytest labs/tests/test_migration_shell.py labs/tests/test_lab_variants.py labs/tests/test_static.py::TestWheelConsistency -q`
- Verified `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` contains `mlsysbook_labs/migration.py` and exports `legacy_migration_panel`.

Follow-up:
- Batch-apply the migration shell to the remaining Volume I notebooks first.

### 2026-06-03 - Volume I Baseline Migration Panels

Lab:
- Volume I labs 01-09 and 11-16.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/vol1/lab_01_ml_intro.py`
- `labs/vol1/lab_02_ml_systems.py`
- `labs/vol1/lab_03_ml_workflow.py`
- `labs/vol1/lab_04_data_engr.py`
- `labs/vol1/lab_05_nn_compute.py`
- `labs/vol1/lab_06_nn_arch.py`
- `labs/vol1/lab_07_ml_frameworks.py`
- `labs/vol1/lab_08_model_train.py`
- `labs/vol1/lab_09_data_selection.py`
- `labs/vol1/lab_11_hw_accel.py`
- `labs/vol1/lab_12_perf_bench.py`
- `labs/vol1/lab_13_model_serving.py`
- `labs/vol1/lab_14_ml_ops.py`
- `labs/vol1/lab_15_responsible_engr.py`
- `labs/vol1/lab_16_ml_conclusion.py`

What changed:
- Installed the `mlsysbook_labs` browser wheel in each remaining Volume I notebook.
- Added a track-aware migration panel to each remaining Volume I notebook.
- Each panel resolves lab metadata from the catalog, reads the saved ledger track, falls back to iPhone when no track is set, resolves the canonical track profile and lab variant, and renders the shared legacy migration panel.
- The panels add learning objectives, track context, scenario brief, source trace, big takeaways, and local report export without duplicating constants in each notebook.

MLSysIM facts/APIs needed:
- No new facts. The notebooks now consume the shared `mlsysbook_labs` catalog, track profiles, variants, and migration helper.

Notebook-local constants removed:
- None in this slice. Existing lab mechanics were preserved while adding the shared outer contract.

Reusable component or modality improved:
- The Volume I notebooks now exercise `legacy_migration_panel()` at notebook level, proving the shared shell can be applied across heterogeneous lab structures.

Plan updates needed in other labs:
- Apply the same baseline migration shell to Volume II labs that are not already deeply migrated.
- Replace baseline panels with deeper part-level track variants as solver-backed evidence is added to each lab.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_01_ml_intro.py labs/vol1/lab_02_ml_systems.py labs/vol1/lab_03_ml_workflow.py labs/vol1/lab_04_data_engr.py labs/vol1/lab_05_nn_compute.py labs/vol1/lab_06_nn_arch.py labs/vol1/lab_07_ml_frameworks.py labs/vol1/lab_08_model_train.py labs/vol1/lab_09_data_selection.py labs/vol1/lab_11_hw_accel.py labs/vol1/lab_12_perf_bench.py labs/vol1/lab_13_model_serving.py labs/vol1/lab_14_ml_ops.py labs/vol1/lab_15_responsible_engr.py labs/vol1/lab_16_ml_conclusion.py`
- `python3 -m pytest labs/tests/test_static.py::TestWheelConsistency -q`
- `python3 -m pytest labs/tests/test_static.py -q`

Follow-up:
- Batch-apply the migration shell to the remaining Volume II notebooks.

### 2026-06-03 - Volume II Baseline Migration Panels

Lab:
- Volume II labs 01-17.

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/vol2/lab_01_introduction.py`
- `labs/vol2/lab_02_compute_infra.py`
- `labs/vol2/lab_03_communication.py`
- `labs/vol2/lab_04_data_storage.py`
- `labs/vol2/lab_05_dist_train.py`
- `labs/vol2/lab_06_collective_communication.py`
- `labs/vol2/lab_07_fault_tolerance.py`
- `labs/vol2/lab_08_fleet_orch.py`
- `labs/vol2/lab_09_perf_engineering.py`
- `labs/vol2/lab_10_inference.py`
- `labs/vol2/lab_11_edge_intelligence.py`
- `labs/vol2/lab_12_ops_scale.py`
- `labs/vol2/lab_13_security_privacy.py`
- `labs/vol2/lab_14_robust_ai.py`
- `labs/vol2/lab_15_sustainable_ai.py`
- `labs/vol2/lab_16_responsible_ai.py`
- `labs/vol2/lab_17_fleet_synthesis.py`

What changed:
- Installed the `mlsysbook_labs` browser wheel in Volume II notebooks that did not already include it.
- Added the shared track-aware migration panel to every Volume II catalog notebook.
- V2-06 keeps its existing richer `mlsysbook_labs` wrapper and now also has the common track/report panel.
- Each panel resolves catalog metadata, ledger track, canonical track profile, and lab variant from shared helpers.
- The added panels expose learning objectives, track context, scenario brief, source trace, big takeaways, and local report export from one shared implementation.

MLSysIM facts/APIs needed:
- No new facts. This slice consumes the existing shared track profile, variant, catalog, and migration helper APIs.

Notebook-local constants removed:
- None in this slice. Existing Volume II lab interactions and ledger saves were preserved.

Reusable component or modality improved:
- The full lab catalog now has a consistent notebook-level track/report surface.

Plan updates needed in other labs:
- The next implementation pass should start replacing baseline panels with deeper part-level track realization where a lab needs solver-backed evidence, specialized plots, or device-specific computed constraints.
- Any new hardware/model/system facts discovered during those deeper migrations must be added to MLSysIM or `mlsysbook_labs`, not embedded as notebook-local constants.

Tests or checks run:
- `python3 -m py_compile labs/vol2/lab_01_introduction.py labs/vol2/lab_02_compute_infra.py labs/vol2/lab_03_communication.py labs/vol2/lab_04_data_storage.py labs/vol2/lab_05_dist_train.py labs/vol2/lab_06_collective_communication.py labs/vol2/lab_07_fault_tolerance.py labs/vol2/lab_08_fleet_orch.py labs/vol2/lab_09_perf_engineering.py labs/vol2/lab_10_inference.py labs/vol2/lab_11_edge_intelligence.py labs/vol2/lab_12_ops_scale.py labs/vol2/lab_13_security_privacy.py labs/vol2/lab_14_robust_ai.py labs/vol2/lab_15_sustainable_ai.py labs/vol2/lab_16_responsible_ai.py labs/vol2/lab_17_fleet_synthesis.py`
- `python3 -m pytest labs/tests/test_static.py::TestWheelConsistency -q`
- `python3 -m pytest labs/tests/test_static.py -q`

Follow-up:
- Run catalog-wide checks that every `.py` lab has the common track/report surface, then decide which labs should receive deeper part-level migrations after V1-10.

### 2026-06-03 - V2-11 Edge Thermodynamics Deep Track Migration

Lab:
- `labs/vol2/lab_11_edge_intelligence.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/edge.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/mlsysbook_labs/ui.py`
- `labs/tests/test_edge_helpers.py`
- `labs/tests/test_ui_helpers.py`
- `labs/tests/test_static.py`
- `labs/vol2/lab_11_edge_intelligence.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added shared edge helper dataclasses and calculations for edge device profiles, training memory amplification, adaptation storage, energy budget use, and federated communication.
- Built V2-11 edge profiles from canonical track profiles, hand-authored lab variants, and MLSysIM hardware references.
- Replaced V2-11 notebook-local smartphone constants with shared `mlsysbook_labs.edge` calculations.
- Added a track selector, track context, source trace, track-aware widget defaults, track-specific memory/energy/communication plots, and a local report export.
- Replaced the baseline migration panel in V2-11 with a deep report that records predictions, knob settings, evidence, final decision, reflections, residual risk, and source trace.
- Fixed a shared `track_selector()` bug where Marimo dictionary radio options require the selected display label as the initial value while returning the canonical track ID.
- Corrected stale V2-10 labels in the V2-11 notebook.
- Marked V2-11 complete in the deep migration checklist.

MLSysIM facts/APIs needed:
- No new MLSysIM hardware facts were required. Existing refs supply memory, TDP, and battery where available:
  - `Hardware.Mobile.iPhone15Pro`
  - `Hardware.Tiny.OuraRing`
  - `Hardware.Edge.RoboTaxi`
  - `Hardware.Cloud.H100`
- Shared edge teaching thresholds, such as active memory budget and per-session energy budget, live in `mlsysbook_labs.edge`.

Notebook-local constants removed:
- Phone-only memory, battery, CPU/NPU power, NPU speedup, LoRA fraction, bias fraction, and FedAvg payload constants were removed from the V2-11 notebook path.

Reusable component or modality improved:
- `track_selector()` is now safe for Marimo radio dictionaries and can be reused by later deep-migrated labs.
- `mlsysbook_labs.edge` is the reusable edge-device calculation layer for V2-11 and likely future V2-10/V1-11 work.

Plan updates needed in other labs:
- Use V2-11 as the reference pattern for track selector plus shared helper plus report export.
- V1-11 Hardware Roofline is the next recommended deep migration target because it should reuse the same hardware-source discipline.

Tests or checks run:
- `python3 -m py_compile labs/vol2/lab_11_edge_intelligence.py`
- `python3 -m pytest labs/tests/test_edge_helpers.py labs/tests/test_static.py::TestSyntax labs/tests/test_static.py::TestWheelConsistency labs/tests/test_static.py::TestLabCatalog -q`
- `python3 -m pytest labs/tests/test_static.py -q`
- `python3 -m pytest labs/tests/test_ui_helpers.py labs/tests/test_edge_helpers.py labs/tests/test_static.py::TestWheelConsistency -q`
- `python3 -m pytest labs/tests/test_engine.py -k "vol2/lab_11 or lab_11_edge" -q`

Follow-up:
- Deep-migrate V1-11 Hardware Roofline next, reusing the source-traced hardware profile pattern.

### 2026-06-03 - V1-11 Hardware Roofline Deep Track Migration

Lab:
- `labs/vol1/lab_11_hw_accel.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/vol1/lab_11_hw_accel.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`

What changed:
- Replaced the baseline migration shell in V1-11 with a full track selector, track context, source trace, and local report export.
- Wired the selected track into the lab narrative, widget defaults, roofline plot, fusion calculation, hardware comparison, energy discussion, tiling estimate, synthesis, ledger payload, and downloadable report.
- Removed stale notebook-local H100/Jetson/iPhone roofline constants from the setup path.
- Report export now records predictions, knob settings, selected hardware ref, model ref, ridge point, workload arithmetic intensity, MFU, fusion speedup, comparison regime, final decision, reflections, residual risk, and source trace.
- V1-11 uses the previously added `mlsysbook_labs.roofline` helpers and hand-authored `v1_11_hardware_roofline` track variants.

MLSysIM facts/APIs needed:
- No new MLSysIM facts were required for this slice. The lab resolves:
  - `Hardware.Mobile.iPhone15Pro`
  - `Hardware.Tiny.OuraRing`
  - `Hardware.Edge.RoboTaxi`
  - `Hardware.Cloud.H100`
- Shared teaching calculations live in `mlsysbook_labs.roofline`: hardware roofline profiles, GEMM workload arithmetic intensity, roofline point evaluation, and fusion traffic.

Notebook-local constants removed:
- H100, Jetson, and iPhone peak/BW/TDP/ridge constants were removed from the V1-11 notebook setup. Displayed roofline facts now come from the selected track variant and MLSysIM hardware ref.

Reusable component or modality improved:
- `mlsysbook_labs.roofline` is now exercised by a real notebook and should be reused for benchmarking, inference economy, and serving labs when they need source-traced hardware ceilings.

Plan updates needed in other labs:
- V2-10 should reuse roofline and edge helper patterns for deployment-target economics instead of introducing notebook-local latency/cost constants.
- V1-12 and V1-13 should follow the same pattern: add shared solver/helper APIs first, then wire part-level plots and report evidence to those helpers.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_11_hw_accel.py`
- `python3 -m pytest labs/tests/test_engine.py -k "vol1/lab_11 or lab_11_hw" -q`
- `python3 -m pytest labs/tests/test_static.py::TestSyntax labs/tests/test_static.py::TestLabCatalog -q`

### 2026-06-03 - V2-10 Inference Economy Deep Track Migration

Lab:
- `labs/vol2/lab_10_inference.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/inference.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/mlsysbook_labs/variants.py`
- `labs/tests/test_inference_helpers.py`
- `labs/tests/test_lab_variants.py`
- `labs/tests/test_static.py`
- `labs/vol2/lab_10_inference.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added `mlsysbook_labs.inference`, a shared helper layer for cost crossover, state/cache capacity, batching speedup, and serving-plan sizing.
- Added hand-authored V2-10 track variants with track-specific cost units, demand, setup budgets, state/cache assumptions, SLOs, batching defaults, serving policies, and validation tests.
- Replaced the V2-10 baseline migration shell with a track selector, track context, source trace, track-aware widget defaults, track-neutral part narrative, ledger payload, and local report export.
- Converted the old cloud-only H100/70B notebook calculations into selected-track calculations using MLSysIM hardware/model refs plus `mlsysbook_labs.inference`.
- Rebuilt the browser helper wheel and extended the wheel contract to include `mlsysbook_labs/inference.py`.

MLSysIM facts/APIs needed:
- No new MLSysIM hardware or model facts were required. The lab resolves:
  - `Hardware.Mobile.iPhone15Pro` + `Models.Vision.MobileNetV2`
  - `Hardware.Tiny.OuraRing` + `Models.Tiny.DS_CNN`
  - `Hardware.Edge.RoboTaxi` + `Models.Vision.YOLOv8_Nano`
  - `Hardware.Cloud.H100` + `Models.Language.Llama2_70B`
- Scenario-specific cost units, demand, state/cache teaching assumptions, SLOs, and validation tests live in typed V2-10 variants.

Notebook-local constants removed:
- H100 RAM/cost, T4, Jetson, `TRAINING_COST_2M`, `LLAMA2_70B`, `calc_kv_cache_size`, and `Engine.solve` notebook-local paths were removed from V2-10 setup and tab calculations.

Reusable component or modality improved:
- `mlsysbook_labs.inference` is reusable for V1-12 benchmarking, V1-13 serving, V2-09 performance engineering, and later fleet synthesis labs whenever they need cost, state/cache, batching, or serving-plan evidence.

Plan updates needed in other labs:
- V1-12 should reuse the cost/state/batching result objects when constructing benchmark traps.
- V1-13 should reuse `state_capacity` and `serving_plan` for tail-latency serving choices instead of rebuilding concurrency math locally.

Tests or checks run:
- `python3 -m py_compile labs/mlsysbook_labs/inference.py`
- `python3 -m py_compile labs/vol2/lab_10_inference.py`
- `python3 -m pytest labs/tests/test_inference_helpers.py labs/tests/test_lab_variants.py -q`
- `python3 -m pytest labs/tests/test_engine.py -k "vol2/lab_10 or lab_10_inference" -q`
- `python3 -m pytest labs/tests/test_inference_helpers.py labs/tests/test_lab_variants.py labs/tests/test_static.py::TestWheelConsistency -q`

### 2026-06-03 - V1-12 Benchmarking Trap Deep Track Migration

Lab:
- `labs/vol1/lab_12_perf_bench.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/benchmarking.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/mlsysbook_labs/variants.py`
- `labs/tests/test_benchmarking_helpers.py`
- `labs/tests/test_lab_variants.py`
- `labs/tests/test_static.py`
- `labs/vol1/lab_12_perf_bench.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added `mlsysbook_labs.benchmarking` with source-traced helpers for Amdahl speedup, sustained/burst benchmark behavior, multi-metric SLO gates, and log-normal tail latency.
- Added hand-authored V1-12 variants with track-specific bad benchmark claims, hidden failure metrics, benchmark durations, cooling/ambient defaults, SLO gates, tail-latency defaults, and validation tests.
- Replaced the baseline migration shell with track selector, track context, source trace, selected-track widget defaults, ledger metadata, and local report export.
- Removed notebook-local H100/A100/Jetson/ResNet benchmark constants from setup and routed displayed evidence through selected MLSysIM hardware/model refs plus `mlsysbook_labs.benchmarking`.
- Rebuilt the browser helper wheel and extended the wheel contract to include `mlsysbook_labs/benchmarking.py`.

MLSysIM facts/APIs needed:
- No new MLSysIM facts were required. The lab resolves the same canonical track hardware refs and model refs from the V1-12 variants.
- Track-specific benchmark protocol assumptions live in typed variant defaults, not notebook-local constants.

Notebook-local constants removed:
- H100/A100 peak and bandwidth constants, Jetson TDP/TFLOPS, and ResNet-50 FLOP/parameter constants were removed from the V1-12 notebook setup path.

Reusable component or modality improved:
- `mlsysbook_labs.benchmarking` can be reused by V1-13 tail latency, V2-09 performance engineering, and any lab that needs benchmark claims, guardrail gates, or tail-distribution evidence.

Plan updates needed in other labs:
- V1-13 should reuse `tail_latency` and add queueing-specific helpers rather than duplicating tail distribution code.
- V2-09 should reuse `amdahl_speedup` and `metric_gate` for optimization-trap evidence.

Tests or checks run:
- `python3 -m py_compile labs/mlsysbook_labs/benchmarking.py`
- `python3 -m py_compile labs/vol1/lab_12_perf_bench.py`
- `python3 -m pytest labs/tests/test_benchmarking_helpers.py labs/tests/test_lab_variants.py -q`
- `python3 -m pytest labs/tests/test_engine.py -k "vol1/lab_12 or lab_12_perf" -q`

### 2026-06-03 - V1-13 Tail Latency Trap Deep Track Migration

Lab:
- `labs/vol1/lab_13_model_serving.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/serving.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/mlsysbook_labs/variants.py`
- `labs/tests/test_serving_helpers.py`
- `labs/tests/test_lab_variants.py`
- `labs/tests/test_static.py`
- `labs/vol1/lab_13_model_serving.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added `mlsysbook_labs.serving`, a shared helper layer for queueing p50/p95/p99/p999, static batching formation delay, state/cache capacity, and cold-start exposure.
- Added hand-authored V1-13 variants with track-specific arrival rates, service times, replicas, SLOs, service variability, batching policy, state/cache assumptions, cold-start settings, serving policy, and validation tests.
- Replaced the old cloud-only notebook constants with a track selector, track context, source trace, selected-track widget defaults, helper-backed plots, ledger metadata, and local report export.
- Converted Part A-D into track-aware narratives: queueing explosion, batching tax, state/cache wall, and cold-start scale-out.
- Rebuilt the browser helper wheel and extended the wheel contract to include `mlsysbook_labs/serving.py`.

MLSysIM facts/APIs needed:
- No new MLSysIM facts were required. The lab resolves:
  - `Hardware.Mobile.iPhone15Pro` + `Models.Vision.MobileNetV2`
  - `Hardware.Tiny.OuraRing` + `Models.Tiny.DS_CNN`
  - `Hardware.Edge.RoboTaxi` + `Models.Vision.YOLOv8_Nano`
  - `Hardware.Cloud.H100` + `Models.Language.Llama2_70B`
- Scenario-specific request patterns, SLOs, batching defaults, state/cache teaching assumptions, and warm-pool policy live in typed V1-13 variants.

Notebook-local constants removed:
- H100/A100/Jetson hardware constants, ResNet/Llama constants, PCIe/NVMe/network storage constants, and notebook-local KV/cache formulas were removed from V1-13.

Reusable component or modality improved:
- `mlsysbook_labs.serving` is now reusable by V2-09 performance engineering, V2-10 inference economy, V2-12 operations at scale, and any future lab that needs source-traced serving/SLA evidence.

Plan updates needed in other labs:
- V1-14 should reuse the report and source-trace structure when adding drift/degradation evidence.
- V2-06 and V2-12 can reuse the queueing helper for collective/fleet overload narratives when p99 or backpressure is a teaching target.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_13_model_serving.py labs/mlsysbook_labs/serving.py`
- `python3 -m pytest labs/tests/test_engine.py -k "lab_13_model_serving" -q`
- `python3 -m pytest labs/tests/test_serving_helpers.py labs/tests/test_lab_variants.py labs/tests/test_static.py -q`
- `git diff --check`

### 2026-06-03 - V1-14 Silent Degradation Deep Track Migration

Lab:
- `labs/vol1/lab_14_ml_ops.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/ops.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/mlsysbook_labs/variants.py`
- `labs/tests/test_ops_helpers.py`
- `labs/tests/test_lab_variants.py`
- `labs/tests/test_static.py`
- `labs/vol1/lab_14_ml_ops.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added `mlsysbook_labs.ops`, a shared helper layer for drift visibility with delayed labels, retraining cadence, operations policy scoring, and technical debt cascade.
- Added hand-authored V1-14 variants with track-specific drift sources, monitoring signals, label delays, quality floors, retraining economics, rollback/escalation policies, downstream dependency counts, and validation tests.
- Replaced the old fraud-only notebook constants with a track selector, track context, source trace, selected-track defaults, helper-backed plots, ledger metadata, and local report export.
- Converted Part A-D into track-aware narratives: drift visibility, retraining cadence, ops policy, and debt cascade.
- Rebuilt the browser helper wheel and extended the wheel contract to include `mlsysbook_labs/ops.py`.

MLSysIM facts/APIs needed:
- No new MLSysIM facts were required. The lab resolves:
  - `Hardware.Mobile.iPhone15Pro` + `Models.Vision.MobileNetV2`
  - `Hardware.Tiny.OuraRing` + `Models.Tiny.DS_CNN`
  - `Hardware.Edge.RoboTaxi` + `Models.Vision.YOLOv8_Nano`
  - `Hardware.Cloud.H100` + `Models.Language.BERT_Base`
- Scenario-specific drift, monitoring, cadence, rollback, and escalation assumptions live in typed V1-14 variants.

Notebook-local constants removed:
- H100/Jetson retraining hardware constants, PayGuard-only drift values, hard-coded retraining cost examples, and notebook-local debt formulas were removed from V1-14.

Reusable component or modality improved:
- `mlsysbook_labs.ops` can be reused by V2-07 fault tolerance, V2-08 fleet orchestration, V2-12 operations at scale, V2-14 robust AI, and V2-16 responsible AI whenever labs need drift, delayed labels, rollback, or debt-cascade evidence.

Plan updates needed in other labs:
- V1-15 should reuse the operations-policy report structure when defining responsibility/fairness mitigation and residual risk.
- V2-12 should reuse the ops helper for fleet-scale monitoring, escalation, and rollback policies.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_14_ml_ops.py labs/mlsysbook_labs/ops.py`
- `python3 -m pytest labs/tests/test_engine.py -k "lab_14_ml_ops" -q`
- `python3 -m pytest labs/tests/test_ops_helpers.py labs/tests/test_lab_variants.py labs/tests/test_static.py -q`
- `git diff --check`

### 2026-06-03 - V2-06 Collective Communication Deep Track Migration

Lab:
- `labs/vol2/lab_06_collective_communication.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/variants.py`
- `labs/tests/test_lab_variants.py`
- `labs/vol2/lab_06_collective_communication.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added hand-authored V2-06 variants with track-specific operations, payload sizes, participants, fabric/link analogies, topology assumptions, overlap, compression, residual risk, and validation tests.
- Replaced the previous cloud-only wrapper with a canonical track selector, track context, source trace, selected-track widget defaults, ledger payload, and local report export.
- Kept the core timing calculations in MLSysIM physics (`calc_ring_allreduce_time`, `calc_tree_allreduce_time`, `calc_hierarchical_allreduce_time`) instead of duplicating collective formulas in `mlsysbook_labs`.
- Converted the lab narrative from GPU-only collectives to a track-aware communication design review: mobile federated aggregation, wearable intermittent sync, RoboTaxi depot hierarchy, and Cloud Fleet GPU collectives.
- Rebuilt the browser helper wheel so the new V2-06 variants are available in WASM.

MLSysIM facts/APIs needed:
- No new MLSysIM facts were required. The lab resolves canonical track hardware/model refs through variants and uses MLSysIM collective physics plus existing fabric entries:
  - `Systems.Fabrics.InfiniBand_NDR`
  - `Systems.Fabrics.Ethernet_100G`
  - `Hardware.Cloud.H100.nvlink`

Notebook-local constants removed:
- Manual H100/Jetson/NVLink/IB/Ethernet constants and hard-coded Cloud Fleet report fields were removed from the V2-06 setup and report path. Track payload/topology defaults now live in typed variants.

Reusable component or modality improved:
- V2-06 demonstrates the "use MLSysIM physics directly when it already owns the equations" pattern. `mlsysbook_labs` owns track variants, source trace, report structure, and UI composition.

Plan updates needed in other labs:
- V2-07 and V2-08 should follow the same pattern when MLSysIM already has the underlying solver/fact surface: add variants first, then keep notebook code as composition and reporting.

Tests or checks run:
- `python3 -m py_compile labs/vol2/lab_06_collective_communication.py labs/mlsysbook_labs/variants.py`
- `python3 -m pytest labs/tests/test_engine.py -k "lab_06_collective_communication" -q`
- `python3 -m pytest labs/tests/test_lab_variants.py labs/tests/test_static.py -q`
- `git diff --check`

### 2026-06-03 - V1-15 Responsible Engineering Deep Track Migration

Lab:
- `labs/vol1/lab_15_responsible_engr.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/responsibility.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/mlsysbook_labs/variants.py`
- `labs/tests/test_responsibility_helpers.py`
- `labs/tests/test_lab_variants.py`
- `labs/tests/test_static.py`
- `labs/vol1/lab_15_responsible_engr.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added `mlsysbook_labs.responsibility`, a shared helper layer for metric conflict, responsibility budget scoring, explanation overhead, and carbon accounting.
- Added hand-authored V1-15 variants with track-specific harmed parties, obligations, audit signals, subgroup/context labels, metric gap targets, explanation defaults, latency SLOs, retraining energy, carbon intensity, governance delay, validation tests, and residual harm.
- Replaced the old loan-only notebook with a track selector, track context, source trace, selected-track defaults, helper-backed plots/tables, ledger metadata, and local report export.
- Converted Part A-D into track-aware narratives: metric conflict, responsibility budget, explainability tax, and carbon ledger.
- Rebuilt the browser helper wheel and extended the wheel contract to include `mlsysbook_labs/responsibility.py`.

MLSysIM facts/APIs needed:
- No new MLSysIM facts were required. The lab resolves:
  - `Hardware.Mobile.iPhone15Pro` + `Models.Vision.MobileNetV2`
  - `Hardware.Tiny.OuraRing` + `Models.Tiny.DS_CNN`
  - `Hardware.Edge.RoboTaxi` + `Models.Vision.YOLOv8_Nano`
  - `Hardware.Cloud.H100` + `Models.Language.BERT_Base`
- Scenario-specific harmed-party text, subgroup/context assumptions, audit signals, and responsibility thresholds live in typed V1-15 variants.

Notebook-local constants removed:
- Loan-only fairness assumptions, H100/Jetson carbon constants, grid constants, SHAP formulas, and notebook-local responsibility/carbon formulas were removed from V1-15.

Reusable component or modality improved:
- `mlsysbook_labs.responsibility` can be reused by V2-14 robust AI, V2-16 responsible AI, V2-17 future systems, and any lab that needs source-traced harmed-party, audit, explanation, or carbon evidence.

Plan updates needed in other labs:
- V2-16 should reuse the V1-15 helper rather than reintroducing local fairness/responsibility formulas.
- V1-16 can include V1-15's responsible decision memo as one of the capstone constraints in the ledger synthesis.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_15_responsible_engr.py labs/mlsysbook_labs/responsibility.py`
- `python3 -m pytest labs/tests/test_engine.py -k "lab_15_responsible_engr" -q`
- `python3 -m pytest labs/tests/test_responsibility_helpers.py labs/tests/test_lab_variants.py labs/tests/test_static.py -q`
- `git diff --check`

### 2026-06-03 - V1-16 Architect's Audit Deep Track Migration

Lab:
- `labs/vol1/lab_16_ml_conclusion.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/mlsysbook_labs/capstone.py`
- `labs/mlsysbook_labs/__init__.py`
- `labs/mlsysbook_labs/variants.py`
- `labs/tests/test_capstone_helpers.py`
- `labs/tests/test_lab_variants.py`
- `labs/tests/test_static.py`
- `labs/vol1/lab_16_ml_conclusion.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`
- `labs/LAB_IMPLEMENTATION_NOTES.md`
- `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`

What changed:
- Added `mlsysbook_labs.capstone`, a shared helper layer for ledger replay with track presets, sensitivity audits, and architecture memo packaging.
- Added hand-authored V1-16 variants with track-specific architecture components, prior-decision presets, sensitivity thresholds, revision options, top risks, durable principles, and validation tests.
- Replaced the old H100/Llama-only capstone with a selected-track architecture audit: ledger replay, architecture map, sensitivity audit, final revision memo, ledger metadata, and local report export.
- Missing ledger entries now degrade gracefully by using labeled track presets instead of pretending the evidence exists.
- Rebuilt the browser helper wheel and extended the wheel contract to include `mlsysbook_labs/capstone.py`.

MLSysIM facts/APIs needed:
- No new MLSysIM facts were required. The lab resolves:
  - `Hardware.Mobile.iPhone15Pro` + `Models.Vision.MobileNetV2`
  - `Hardware.Tiny.OuraRing` + `Models.Tiny.DS_CNN`
  - `Hardware.Edge.RoboTaxi` + `Models.Vision.YOLOv8_Nano`
  - `Hardware.Cloud.H100` + `Models.Language.Llama2_70B`
- Track-specific prior decisions, capstone sensitivity thresholds, and memo risks live in typed V1-16 variants.

Notebook-local constants removed:
- H100/Jetson/Llama capstone constants, class-median prediction profiles, notebook-local constraint cascade formulas, and the legacy migration shell were removed from V1-16.

Reusable component or modality improved:
- `mlsysbook_labs.capstone` can be reused by Volume II capstones or any lab that needs ledger replay, preset fallback labeling, sensitivity scoring, or memo packaging.

Plan updates needed in other labs:
- Volume II labs should keep ledger/report fields structured enough for future capstones to replay them.
- V2-17 can reuse the capstone helper rather than creating a separate ledger summarizer.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_16_ml_conclusion.py labs/mlsysbook_labs/capstone.py`
- `python3 -m pytest labs/tests/test_engine.py -k "lab_16_ml_conclusion" -q`
- `python3 -m pytest labs/tests/test_capstone_helpers.py labs/tests/test_lab_variants.py labs/tests/test_static.py -q`
- `git diff --check`

### 2026-06-03 - V1-00 Orientation Track Contract Cleanup

Lab:
- `labs/vol1/lab_00_introduction.py`

Track(s):
- iPhone, Oura Ring, RoboTaxi, Cloud Fleet.

Files touched:
- `labs/vol1/lab_00_introduction.py`
- `labs/vol1/lab_00_introduction.track-plan.md`
- `labs/tests/test_track_profiles.py`
- `labs/LAB_DEEP_MIGRATION_CHECKLIST.md`
- `labs/LAB_IMPLEMENTATION_NOTES.md`

What changed:
- Made the Lab 00 track selector return canonical track IDs directly: `iphone`, `oura_ring`, `robotaxi`, and `cloud_fleet`.
- Replaced the old physical-regime display cards with profile-derived canonical track cards using `CANONICAL_TRACKS`.
- Changed the track reveal copy to pull label, stakeholder, narrative, constraints, hardware ref, system ref, and source policy from the resolved `TrackProfile`.
- Confirmed the ledger save writes the canonical track ID and profile-derived refs that later labs consume.
- Added a regression test that Lab 00 exposes all four canonical track IDs and that its report/source trace include track, hardware, system, and source policy fields.

MLSysIM facts/APIs needed:
- No new MLSysIM facts were required. Lab 00 references track profiles; later labs resolve the hardware/system refs through MLSysIM.

Notebook-local constants removed:
- Old selector aliases and several hard-coded track reveal facts were replaced with canonical track profile fields.

Reusable component or modality improved:
- Lab 00 is now explicitly documented as the canonical track-selection surface for the course.

Plan updates needed in other labs:
- Later labs should continue to use `ledger.get_track()` plus `get_track_profile()`/`get_lab_track_variant()` rather than their own selector aliases.

Tests or checks run:
- `python3 -m py_compile labs/vol1/lab_00_introduction.py`
- `python3 -m pytest labs/tests/test_engine.py -k "lab_00_introduction" -q`
- `python3 -m pytest labs/tests/test_track_profiles.py labs/tests/test_ui_helpers.py labs/tests/test_static.py -q`
- `git diff --check`
