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

Screenshots:
- Representative screenshots are written under `/tmp/mlsysbook-render-smoke`.
- Full-catalog screenshots are written under `/tmp/mlsysbook-render-smoke-all-v2`.

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

Implemented:
- `render_lab_smoke.py` treats Lab 00 as load-only orientation and track-clicks the other labs.

## High-Confidence Fixes Made

- Removed duplicate track selector display from the shared-renderer Volume II notebook shells.
- Added shared CSS wrapping for long registry refs in `.mlsysbook-field` cards.
- Rebuilt `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`.
- Added `labs/tools/render_lab_smoke.py`.
- Verified representative and full-catalog browser smoke passes.

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
