# Overnight Lab Feedback Audit

Date: 2026-06-04

Scope:
- Render and browser-smoke the lab experience instead of relying only on static and engine tests.
- Simulate instructor, TA, and student feedback without waiting for user iteration.
- Implement high-confidence fixes immediately.
- Preserve the single-source-of-truth rule: device, model, system, and scenario facts remain in MLSysIM or `mlsysbook_labs`.

## Rendered Set

Representative first pass:
- `labs/vol1/lab_01_ml_intro.py`
- `labs/vol1/lab_09_data_selection.py`
- `labs/vol2/lab_01_introduction.py`
- `labs/vol2/lab_17_fleet_synthesis.py`

Full catalog pass:
- All 34 labs in `labs/vol1` and `labs/vol2`.

Browser checks:
- Marimo app loads in Chromium.
- No visible traceback/import/runtime markers.
- Track selector appears at most once.
- All non-orientation labs switch across iPhone, Oura Ring, RoboTaxi, and Cloud Fleet.
- Track-specific rendered text changes across all four tracks.
- `.mlsysbook-field` cards do not overflow horizontally.

Result:
- Full catalog browser smoke: 34 passed, 0 failed.
- Stronger full-text track switching check: every non-orientation lab produced four distinct rendered track states.
- Interaction browser pilot after workflow fixes: 34 passed, 0 failed, with chunked runs covering all Volume I and Volume II labs, all four canonical track clicks, scroll checks, visible part/tab checks, answer clicks where safe, dashboard checks, and catalog checks.

Screenshots:
- Representative screenshots are written under `/tmp/mlsysbook-render-smoke`.
- Full-catalog screenshots are written under `/tmp/mlsysbook-render-smoke-all-v2`.
- Latest interaction screenshots are written under:
  - `/tmp/mlsysbook-interaction-smoke-v1-head-final2`
  - `/tmp/mlsysbook-interaction-smoke-v1-tail-final2`
  - `/tmp/mlsysbook-interaction-smoke-v2-01-05-final`
  - `/tmp/mlsysbook-interaction-smoke-v2-06-11-final`
  - `/tmp/mlsysbook-interaction-smoke-v2-12-17-final`

## Simulated Instructor Feedback

### Volume I Instructor: Senior Undergraduate ML Systems

What works:
- The repeated track card gives students a stable mental model: stakeholder, device family, model family, metrics, guardrails, and dominant constraints.
- Volume I labs now feel like engineering worksheets rather than isolated model exercises.
- V1-01 and V1-09 make the core level appropriate for senior undergraduates: students can reason from constraints without needing distributed-systems depth.

Concerns:
- Students need a clean first screen. Repeated widgets or clipped registry names make the lab feel less polished and distract from the learning objective.
- The same track selector should not appear twice.
- Implementation provenance should not dominate the opening screen; the learner path should read as a case, then preserve provenance in reports and tests.

Implemented:
- Removed duplicate track selector rendering from shared Volume II wrappers.
- Added shared wrapping rules for `.mlsysbook-field` and inline `code` values.
- Added a reusable `part_workflow()` scaffold after the second browser pass showed that V1-03, V1-04, V1-05, and V1-11 had enough content but did not make the learner loop explicit enough in the visible flow.

### Volume II Instructor: Senior/Master's/Ph.D. Systems

What works:
- V2-01 and V2-17 now demonstrate that the same concept changes with track selection.
- The shared system-design renderer gives Volume II a common grammar: frontier, scaling boundary, decision, validation, residual risk.
- V2-17 correctly behaves like synthesis rather than a standalone isolated chapter because it can surface prior Volume II ledger decisions.

Concerns:
- The generic shared renderer is good enough to make every lab usable, but some V2 labs should eventually get richer lab-specific helpers where the concept demands it.
- The first grad-level refinement target should be adding stronger evidence modalities to the most conceptually important V2 labs, not changing the track registry.

Deferred:
- Lab-specific V2 refinements should be separate future commits. The shared renderer is now a working baseline with track arcs, decisions, validation, and residual risk.

### TA/Grading Feedback

What works:
- Local report export exists in rendered labs.
- Reports preserve track, scenario, hardware/model refs, helper outputs, and caveats for grading without making those refs the main student-facing copy.
- The common report fields are suitable for rubric-based grading.

Concerns:
- Report schema should eventually become a formal test across all deep labs.
- The browser smoke should be easy to rerun before releases.

Implemented:
- Added `labs/tools/render_lab_smoke.py` for repeatable rendered-browser checks.

### Student Feedback Simulation

Likely positive reactions:
- The track choice visibly changes the story, stakeholder, model family, device family, and narrative.
- The same track names recur, so students understand that they are carrying a perspective through the course.
- The where-this-fits arc helps explain how one lab connects to the larger Volume I or Volume II journey.

Likely confusion:
- Lab 00 is orientation and does not use the same "Your Track" rendered selector pattern as later labs. This is acceptable but the smoke tool must treat it differently.
- Volume II shared-renderer labs may feel similar in mechanics. That consistency is useful, but later refinements should make the evidence feel more chapter-specific.
- Long Marimo pages can hide the "what do I do next?" structure below the first viewport. Students need a visible workflow bridge that names prediction, controls, evidence, decision, and reflection before they enter the parts.

Implemented:
- `render_lab_smoke.py` treats Lab 00 as load-only orientation and track-clicks the other labs.
- `interaction_lab_smoke.py` now scrolls Marimo's actual scroll containers and accumulates visible text across the page, so the smoke check better approximates a student who scrolls through the lab.
- V1-03, V1-04, V1-05, and V1-11 now use the shared workflow scaffold to make the learner motion explicit.

## High-Confidence Fixes Made

- Removed duplicate track selector display from the shared-renderer Volume II notebook shells.
- Added shared CSS wrapping for long registry refs in `.mlsysbook-field` cards.
- Rebuilt `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`.
- Added `labs/tools/render_lab_smoke.py`.
- Verified representative and full-catalog browser smoke passes.
- Added `part_workflow()` as a shared UI component and exported it from `mlsysbook_labs`.
- Improved `labs/tools/interaction_lab_smoke.py` for scroll-aware interaction checks.
- Verified all 34 Volume I/II labs with the interaction smoke in chunks.

## Commands Run

```bash
python3 -m py_compile labs/tools/render_lab_smoke.py
python3 -m py_compile labs/mlsysbook_labs/ui.py
python3 -m py_compile labs/vol2/lab_01_introduction.py labs/vol2/lab_02_compute_infra.py labs/vol2/lab_03_communication.py labs/vol2/lab_04_data_storage.py labs/vol2/lab_05_dist_train.py labs/vol2/lab_07_fault_tolerance.py labs/vol2/lab_08_fleet_orch.py labs/vol2/lab_09_perf_engineering.py labs/vol2/lab_12_ops_scale.py labs/vol2/lab_13_security_privacy.py labs/vol2/lab_14_robust_ai.py labs/vol2/lab_15_sustainable_ai.py labs/vol2/lab_16_responsible_ai.py labs/vol2/lab_17_fleet_synthesis.py
python3 -m pytest labs/tests/test_engine.py -q -k "lab_01_introduction or lab_17_fleet_synthesis"
python3 -m pytest labs/tests/test_static.py -q
python3 -m build --wheel labs
cp labs/dist/mlsysbook_labs-0.1.0-py3-none-any.whl wheels/mlsysbook_labs-0.1.0-py3-none-any.whl
python3 labs/tools/render_lab_smoke.py --labs labs/vol1/lab_01_ml_intro.py labs/vol1/lab_09_data_selection.py labs/vol2/lab_01_introduction.py labs/vol2/lab_17_fleet_synthesis.py --output-dir /tmp/mlsysbook-render-smoke
python3 labs/tools/render_lab_smoke.py --labs $(find labs/vol1 labs/vol2 -maxdepth 1 -name 'lab_*.py' | sort) --port-start 29700 --output-dir /tmp/mlsysbook-render-smoke-all-v2 > /tmp/mlsysbook-render-smoke-all-v2.json
python3 labs/tools/interaction_lab_smoke.py --labs labs/vol1/lab_00_introduction.py labs/vol1/lab_01_ml_intro.py labs/vol1/lab_02_ml_systems.py labs/vol1/lab_03_ml_workflow.py labs/vol1/lab_04_data_engr.py labs/vol1/lab_05_nn_compute.py labs/vol1/lab_06_nn_arch.py labs/vol1/lab_07_ml_frameworks.py labs/vol1/lab_08_model_train.py labs/vol1/lab_09_data_selection.py labs/vol1/lab_10_model_compress.py labs/vol1/lab_11_hw_accel.py --html-pages labs/lab-plan-dashboard.html labs/lab-modality-catalog.html --output-dir /tmp/mlsysbook-interaction-smoke-v1-head-final2 --port-start 30800
python3 labs/tools/interaction_lab_smoke.py --labs labs/vol1/lab_12_perf_bench.py labs/vol1/lab_13_model_serving.py labs/vol1/lab_14_ml_ops.py labs/vol1/lab_15_responsible_engr.py labs/vol1/lab_16_ml_conclusion.py --html-pages labs/lab-plan-dashboard.html labs/lab-modality-catalog.html --output-dir /tmp/mlsysbook-interaction-smoke-v1-tail-final2 --port-start 30820
python3 labs/tools/interaction_lab_smoke.py --labs labs/vol2/lab_01_introduction.py labs/vol2/lab_02_compute_infra.py labs/vol2/lab_03_communication.py labs/vol2/lab_04_data_storage.py labs/vol2/lab_05_dist_train.py --html-pages labs/lab-plan-dashboard.html labs/lab-modality-catalog.html --output-dir /tmp/mlsysbook-interaction-smoke-v2-01-05-final --port-start 30840
python3 labs/tools/interaction_lab_smoke.py --labs labs/vol2/lab_06_collective_communication.py labs/vol2/lab_07_fault_tolerance.py labs/vol2/lab_08_fleet_orch.py labs/vol2/lab_09_perf_engineering.py labs/vol2/lab_10_inference.py labs/vol2/lab_11_edge_intelligence.py --html-pages labs/lab-plan-dashboard.html labs/lab-modality-catalog.html --output-dir /tmp/mlsysbook-interaction-smoke-v2-06-11-final --port-start 30860
python3 labs/tools/interaction_lab_smoke.py --labs labs/vol2/lab_12_ops_scale.py labs/vol2/lab_13_security_privacy.py labs/vol2/lab_14_robust_ai.py labs/vol2/lab_15_sustainable_ai.py labs/vol2/lab_16_responsible_ai.py labs/vol2/lab_17_fleet_synthesis.py --html-pages labs/lab-plan-dashboard.html labs/lab-modality-catalog.html --output-dir /tmp/mlsysbook-interaction-smoke-v2-12-17-final --port-start 30880
git diff --check
```

## Next Autonomous Hardening Targets

No user loop is required for these; they can be handled as future engineering passes:

1. Add report schema validation across all deep labs.
2. Add a release-mode browser smoke wrapper that calls `render_lab_smoke.py` on a selected canary set.
3. Specialize the highest-value Volume II shared-renderer labs with richer evidence helpers:
   - V2-02 Compute Infrastructure Wall
   - V2-03 Network Fabric Design
   - V2-05 Parallelism Puzzle
   - V2-13 Price of Privacy
   - V2-17 Fleet Synthesis
4. Add a compact instructor rubric exemplar for one Volume I lab and one Volume II lab.
