# Lab Depth Audit And Expansion Plan - 2026-06-13

This audit covers the `codex/labs` worktree after merging local `dev` into the branch.

Current branch state:

- Worktree: `/Users/VJ/GitHub/MLSysBook-labs`
- Branch: `codex/labs`
- Merge commit: `5b698b5de0` (`Merge branch 'dev' into codex/labs`)
- Local `dev` state after merge: 0 commits ahead of this branch

## What The Labs Are Trying To Be

The labs are not standalone demos. They are chapter-aligned, track-aware teaching cases for Volume I and Volume II. The common student rhythm should be stable:

1. Read the chapter idea and the track context.
2. Enter a concrete stakeholder scenario.
3. Make a prediction before seeing evidence.
4. Change controls and inspect evidence.
5. Read the math or system model behind the evidence.
6. Make a checkpoint decision.
7. Synthesize across parts into a defensible engineering recommendation.
8. Export a local-first report with source trace, residual risk, and incomplete-field reporting.

The canonical tracks are iPhone, Oura Ring, RoboTaxi, and Cloud Fleet. MLSysIM should own system facts and solver equations. The lab layer should own story, controls, student flow, report prompts, and track-specific framing.

## Post-Merge Runtime Evidence

Focused checks passed:

- `PYTHONPATH=mlsysim python3 -m pytest mlsysim/tests/test_hardware.py mlsysim/tests/test_compression_candidates.py mlsysim/tests/test_solver_module_exports.py -q`
- `PYTHONPATH=labs:mlsysim python3 -m pytest labs/tests/test_track_profiles.py labs/tests/test_lab_variants.py labs/tests/test_report_contract.py labs/tests/test_ui_helpers.py labs/tests/test_track_arcs.py -q`
- `python3 -m py_compile` across `labs/mlsysbook_labs`, `labs/vol1`, `labs/vol2`, `labs/tools`, and `mlsysim/mlsysim`

Representative browser smoke passed:

- V1-10 Compression Paradox
- V2-01 Scale Illusion
- V2-10 Inference Economy
- V2-17 Fleet Synthesis
- `lab-plan-dashboard.html`
- `lab-modality-catalog.html`

Full lab suite result:

- `121 failed, 1372 passed, 99 skipped, 180 xfailed`

The failures cluster around protocol depth, not basic runtime breakage:

- Missing explicit `build_synthesis()` functions.
- Missing visible Synthesis tabs in several partial migrations.
- Shared Volume II shell wrappers have only one generic runtime flow from the static tests' perspective.
- Several labs lack Math Peek/formula/source-model sections.
- Many shared shells have only three controls.
- Volume II shared shells lack a ledger HUD in static structure tests.
- Prediction reveal and `mo.stop` style gates are uneven.

## Current Lab Depth Classification

This classification is based on static structure, test output, and representative browser smoke.

### Deep Exemplars

These already look close to the desired model: multiple parts, synthesis, substantial controls, evidence, and math/source sections.

| Lab | Current assessment | Keep or improve |
|---|---|---|
| V1-10 Compression Paradox | Strong 5-part pilot with candidate frontier, Math Peek sections, and report path. | Keep as exemplar. Polish source-truth constants over time. |
| V1-11 Hardware Roofline | Strong 5-part roofline progression. | Keep as exemplar. |
| V1-12 Benchmarking Trap | Strong 4-part lab with math and multi-metric evidence. | Keep as exemplar. |
| V2-10 Inference Economy | Strong 4-part inference-at-scale storyline. | Keep as Volume II exemplar. |
| V2-11 Edge Thermodynamics | Strong 4-part edge storyline. | Keep as Volume II exemplar. |

### Bespoke Labs That Need Depth Polish

These have useful chapter-specific content, but miss formula/source-model sections or some protocol pieces.

| Lab | Main gap |
|---|---|
| V1-01 AI Triad | Has 3 parts and synthesis, but no Math Peek/source-model section. Needs a deeper Part D or richer per-part steps. |
| V1-13 Tail Latency Trap | Good 4-part shape, but no Math Peek markers. |
| V1-14 Silent Degradation | Good 4-part shape, but no Math Peek markers. |
| V1-15 No Free Fairness | Good 4-part shape, but no Math Peek markers. |
| V1-16 Architect's Audit | Good 4-part capstone shape, but no Math Peek markers. |
| V2-06 Collective Communication | Good 4-part shape, but no Math Peek markers and one widget-state failure in the full suite. |

### Inline Partial Volume I Migrations

These have three visible parts and some synthesis text, but no explicit builder/synthesis functions. Several also have only three controls and no math sections.

| Lab | Current shape | Needed upgrade |
|---|---|---|
| V1-02 Physics of Deployment | Inline A/B/C plus synthesis text. | Convert to explicit Part A-D plus synthesis. |
| V1-03 Constraint Tax | Inline A/B/C plus synthesis text. | Convert to explicit Part A-D plus synthesis. |
| V1-04 Data Gravity | Inline A/B/C plus synthesis text. | Convert to explicit Part A-D plus synthesis. |
| V1-05 Activation Tax | Inline A/B/C plus synthesis text. | Convert to explicit Part A-D plus synthesis. |
| V1-06 Architecture Tax | Inline A/B/C tabs, no synthesis tab. | Add Part D and Synthesis tab. |
| V1-07 Framework Tax | Inline A/B/C tabs, no synthesis tab. | Add Part D and Synthesis tab. |
| V1-08 Training Gauntlet | Inline A/B/C tabs, no synthesis tab. | Add Part D and Synthesis tab. |
| V1-09 Selection Paradox | Inline A/B/C tabs, no synthesis tab. | Add Part D and Synthesis tab. |

### Shared Volume II Shells

These render and smoke-test, but they are too generic. They are all 109-line wrappers around `render_system_design_lab`, which currently gives a generic three-part decision flow rather than chapter-specific storylines.

| Lab | Needed upgrade |
|---|---|
| V2-01 Scale Illusion | Chapter-specific scale failure story with coordination, saturation, and readiness parts. |
| V2-02 Compute Infrastructure Wall | Compute, memory, power/cooling, and procurement/TCO parts. |
| V2-03 Network Fabric Design | Bandwidth/latency, topology, synchronization, and fabric decision parts. |
| V2-04 Data Pipeline Wall | Storage growth, freshness, movement, and retention architecture parts. |
| V2-05 Parallelism Puzzle | Memory, parallel strategy, communication overhead, and deployment handoff parts. |
| V2-07 Failure Budget Engineering | Failure exposure, checkpoint/recovery, redundancy, and policy parts. |
| V2-08 Fleet Orchestration | Queueing, fragmentation, preemption/rollout, and scheduler policy parts. |
| V2-09 Optimization Trap | Bottleneck diagnosis, optimization ladder, side-effect audit, and stop rule parts. |
| V2-12 Silent Fleet | Telemetry coverage, slice drift, alert thresholds, and action policy parts. |
| V2-13 Price of Privacy | Threat/data access, privacy mechanism, overhead/utility, and governance decision parts. |
| V2-14 Robustness Budget | Shift diagnosis, stress coverage, robustness spend, and fallback policy parts. |
| V2-15 Carbon Budget | Measurement, placement, lifecycle, and carbon-aware policy parts. |
| V2-16 Fairness Budget | Metric conflict, subgroup trade-off, governance overhead, and responsible pipeline parts. |
| V2-17 Fleet Synthesis | Ledger replay, interaction map, risk register, and final review parts. |

## Target Depth Standard

Every non-orientation lab should feel like a 45-60 minute case, not a single knob demo.

Minimum target:

- 4 parts plus synthesis for most labs.
- 3 parts only when each part has 3-4 explicit steps and the synthesis is substantial.
- Each part should have a local storyline with a named tension, not just a metric.
- Each part should include prediction, controls, evidence, math/source trace, reflection, and checkpoint/decision.
- Each part should take roughly 8-15 minutes.
- The synthesis should require evidence from at least two parts.

Part contract:

| Step | Required student move | Required implementation |
|---|---|---|
| A1/B1/C1/D1 | Scenario slice | Stakeholder message or concrete system state. |
| A2/B2/C2/D2 | Prediction | Radio, dropdown, or numeric prediction before evidence. |
| A3/B3/C3/D3 | Experiment | One or more controls that change the system state. |
| A4/B4/C4/D4 | Evidence | Plot/table/fallback text generated from MLSysIM or typed lab metadata. |
| A5/B5/C5/D5 | Math/source model | Math Peek, source trace, solver name, registry refs, and assumptions. |
| A6/B6/C6/D6 | Decision | Saved checkpoint or policy choice with residual risk. |

## Implementation Strategy

The fastest durable path is not to hand-edit 34 notebooks independently. Use the strong labs as exemplars and lift reusable structure into shared helpers.

### Phase 1 - Stabilize Shared Depth Primitives

Add shared data structures:

- `LabStoryline`
- `LabPartStory`
- `PartStep`
- `MathPeekSpec`
- `EvidenceSpec`
- `CheckpointSpec`

Add shared rendering helpers:

- `storyline_lab_map()`
- `storyline_part_tabs()`
- `storyline_part_panel()`
- `math_peek()`
- `prediction_gate()`
- `ledger_hud()`
- `synthesis_panel()`

Tests:

- Unit tests for the rendered headers and required semantic labels.
- Static tests should inspect `LabStoryline` metadata for shared-renderer labs instead of treating 109-line wrappers as empty notebooks.

### Phase 2 - Upgrade The Shared Volume II Renderer

Replace the generic three-part `render_system_design_lab()` flow with a four-part, chapter-specific renderer:

- Part A: Diagnose the local system wall.
- Part B: Sweep the scaling or budget frontier.
- Part C: Introduce the chapter-specific intervention.
- Part D: Choose and validate the operating policy.
- Synthesis: Export evidence-backed memo with residual risk.

Each Volume II shared shell should get a typed storyline in `variants.py` or a new `system_storylines.py` module. The renderer should produce actual tabs at runtime and should expose metadata for tests.

This phase upgrades 14 labs at once:

- V2-01 to V2-05
- V2-07 to V2-09
- V2-12 to V2-17

### Phase 3 - Convert Inline Partial Volume I Labs

Convert V1-02 through V1-09 from inline panels into explicit `build_part_*()` and `build_synthesis()` structure. Add a fourth integrative part where the current three-part plan is too thin.

Recommended Part D additions:

| Lab | New Part D |
|---|---|
| V1-02 Physics of Deployment | Deployment Review: choose placement and name the first physics wall. |
| V1-03 Constraint Tax | Workflow Budget Review: decide where to add instrumentation or iteration time. |
| V1-04 Data Gravity | Pipeline Incident Review: handle freshness, movement, and storage under a deadline. |
| V1-05 Activation Tax | Layer Budget Review: trade activation memory against compute and model quality. |
| V1-06 Architecture Tax | Architecture Rollout Review: select architecture under memory, latency, and validation risk. |
| V1-07 Framework Tax | Runtime Migration Review: decide when compile/fusion overhead is worth the operational cost. |
| V1-08 Training Gauntlet | Training Run Review: choose batch/precision/checkpointing and state failure mode. |
| V1-09 Selection Paradox | Data Governance Review: choose data policy under coverage, cost, and bias risk. |

### Phase 4 - Add Math/Source Depth To Existing Bespoke Labs

Add Math Peek/source-model sections without rewriting the whole lab:

| Lab | Math/source addition |
|---|---|
| V1-01 | Triad score decomposition and intervention utility formula. |
| V1-13 | Queueing/tail formula, batching tax, cache capacity, cold start budget. |
| V1-14 | Drift visibility, retraining cadence, debt cascade, alert precision/recall. |
| V1-15 | Fairness metric conflict, budget overhead, explanation latency, carbon budget. |
| V1-16 | Ledger sensitivity math, architecture scoring, residual risk aggregation. |
| V2-06 | Alpha-beta communication model, ring/tree comparison, hierarchy model, overlap/compression equation. |

### Phase 5 - Keep Exemplar Labs As Standards

Use V1-10, V1-11, V1-12, V2-10, and V2-11 as design baselines.

Do not rewrite them broadly. Only:

- Remove remaining notebook-local constants when MLSysIM can own them.
- Normalize report and source-trace fields.
- Add missing accessibility fallback text if smoke finds gaps.
- Ensure each part contributes to the final report.

### Phase 6 - QA And Release Gates

Required gates after each migration slice:

- `python3 -m py_compile` for touched labs and shared helpers.
- Focused unit tests for touched helpers.
- `PYTHONPATH=labs:mlsysim python3 -m pytest labs/tests/test_track_profiles.py labs/tests/test_lab_variants.py labs/tests/test_report_contract.py labs/tests/test_ui_helpers.py labs/tests/test_track_arcs.py -q`
- Relevant protocol/widget tests for touched labs.
- Browser smoke on at least one touched lab per slice.
- Full `labs/tests -q` after each phase.

## Proposed Work Order

1. Upgrade shared renderer metadata and tests first. This addresses the largest failure cluster and prevents false negatives on shared labs.
2. Add chapter-specific Volume II storylines for the 14 shared shells.
3. Convert V1-02 through V1-05, because these are early-course foundations and set student expectations.
4. Convert V1-06 through V1-09, because they bridge model design and optimization.
5. Add Math Peek/source sections to V1-01, V1-13 through V1-16, and V2-06.
6. Run full lab tests and all-lab browser smoke in chunks.
7. Update `LAB_IMPLEMENTATION_NOTES.md` with each completed slice.

## Definition Of Done

A lab is done when:

- It has 4 parts plus synthesis, or a justified 3-part structure with multi-step depth.
- Every part has prediction, controls, evidence, math/source trace, reflection, and checkpoint/decision.
- The lab uses track-specific defaults and narrative.
- Student work is captured in a local-first report.
- Hardware/model/fleet facts resolve through MLSysIM or typed `mlsysbook_labs` metadata.
- Static tests and browser smoke agree that it renders, scrolls, interacts, and exports without runtime errors.
