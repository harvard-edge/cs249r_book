# Lab User Activity Feedback Pass - 2026-06-04

This is the first executed pass of `LAB_USER_ACTIVITY_FEEDBACK_LOOP.md`.

The goal was to simulate realistic lab use, route the results through the
student, TA, instructor, domain-expert, and maintainer lenses, and work
backward to concrete requirements.

## Scope

Selected cluster:

| Role in feedback loop | Lab |
|---|---|
| Orientation and track selection | `labs/vol1/lab_00_introduction.py` |
| Introductory Mobile ML pass | `labs/vol1/lab_01_ml_intro.py` |
| TinyML / wearable data pass | `labs/vol1/lab_09_data_selection.py` |
| RoboTaxi / autonomy compression pass | `labs/vol1/lab_10_model_compress.py` |
| Cloud/fleet inference pass | `labs/vol2/lab_10_inference.py` |
| Edge intelligence pass | `labs/vol2/lab_11_edge_intelligence.py` |
| Privacy/responsible systems pass | `labs/vol2/lab_13_security_privacy.py` |
| Sustainability and region/carbon pass | `labs/vol2/lab_15_sustainable_ai.py` |
| Capstone/fleet synthesis pass | `labs/vol2/lab_17_fleet_synthesis.py` |

Expert lenses exercised:

| Expert lens | Covered by |
|---|---|
| Mobile ML expert | iPhone activity in V1-01 and shared track cue review. |
| TinyML / wearable expert | Oura Ring activity in V1-09 and V2-11. |
| Edge systems expert | V2-11 plus cross-check against RoboTaxi-specific claims. |
| RoboTaxi / autonomy expert | RoboTaxi activity in V1-10. |
| Cloud / fleet expert | Cloud Fleet activity in V2-10, V2-15, and V2-17. |
| Instructor | Chapter-to-lab pedagogy across the cluster. |
| TA / grader | Report export and report-marker inspection. |
| Student proxy | Track switching, visible source trace, and report path. |
| Maintainer | Source-of-truth and shared-component review. |

## Browser Evidence

Pre-fix cluster render smoke:

```bash
python3 labs/tools/render_lab_smoke.py \
  --labs labs/vol1/lab_00_introduction.py labs/vol1/lab_01_ml_intro.py \
    labs/vol1/lab_09_data_selection.py labs/vol1/lab_10_model_compress.py \
    labs/vol2/lab_10_inference.py labs/vol2/lab_11_edge_intelligence.py \
    labs/vol2/lab_13_security_privacy.py labs/vol2/lab_15_sustainable_ai.py \
    labs/vol2/lab_17_fleet_synthesis.py \
  --port-start 29900 \
  --output-dir /tmp/mlsysbook-feedback-pass-20260604 \
  > /tmp/mlsysbook-feedback-pass-20260604/results.json
```

Result:

| Check | Result |
|---|---|
| Rendered labs | 9 |
| Passed | 9 |
| Failed | 0 |
| Minimum distinct track states, non-orientation labs | 4 |

Post-fix cluster render smoke:

```bash
python3 labs/tools/render_lab_smoke.py \
  --labs labs/vol1/lab_00_introduction.py labs/vol1/lab_01_ml_intro.py \
    labs/vol1/lab_09_data_selection.py labs/vol1/lab_10_model_compress.py \
    labs/vol2/lab_10_inference.py labs/vol2/lab_11_edge_intelligence.py \
    labs/vol2/lab_13_security_privacy.py labs/vol2/lab_15_sustainable_ai.py \
    labs/vol2/lab_17_fleet_synthesis.py \
  --port-start 30200 \
  --output-dir /tmp/mlsysbook-feedback-pass-20260604-after-cue \
  > /tmp/mlsysbook-feedback-pass-20260604-after-cue/results.json
```

Result:

| Check | Result |
|---|---|
| Rendered labs | 9 |
| Passed | 9 |
| Failed | 0 |
| Minimum distinct track states, non-orientation labs | 4 |
| Maximum overflowing `.mlsysbook-field` count | 0 |

Screenshots:

- `/tmp/mlsysbook-feedback-pass-20260604`
- `/tmp/mlsysbook-feedback-pass-20260604-after-cue`

## Activity Marker Evidence

A deeper Playwright activity-marker pass inspected rendered body text for report
and source-trace markers after selecting tracks.

| Lab | Track states checked | Source Trace | Download Report | Residual Risk visible | Incomplete Fields visible |
|---|---:|---:|---:|---:|---:|
| V1-00 Orientation | 4 after gated checks | Report source trace generated, not separately visible in marker pass | 4/4 | Report generated | 0/4 |
| V1-01 AI Triad | 4 | 4/4 | 4/4 | In report | 0/4 |
| V1-09 Selection Paradox | 4 | 4/4 | 4/4 | In report | 0/4 |
| V1-10 Compression Paradox | 4 | 4/4 | 4/4 | Input/report path | 4/4 |
| V2-10 Inference Economy | 4 | 4/4 | 4/4 | In report | 0/4 |
| V2-11 Edge Intelligence | 4 | 4/4 | 4/4 | In report | 0/4 |
| V2-13 Price of Privacy | 4 | 4/4 | 4/4 | 4/4 | 0/4 |
| V2-15 Carbon Budget | 4 | 4/4 | 4/4 | 4/4 | 0/4 |
| V2-17 Fleet Synthesis | 4 | 4/4 | 4/4 | 4/4 | 0/4 |

Lab 00 required a separate gated activity script because students must complete
the orientation checks before the track selector and report export unlock.
That is pedagogically reasonable. Release smoke should eventually include a
Lab 00-specific activity path.

## High-Confidence Fix Applied

Feedback signal:

- Students need an explicit cue for what changes because of the selected track.

Fix:

- Updated `mlsysbook_labs.ui.track_context()` to render a shared field:
  `What changed because of your track`.
- The cue is generated from the canonical `TrackProfile`:
  first primary metric, first guardrail metric, and first dominant constraint.
- No notebook-local facts were added.
- Rebuilt `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl`.

Representative rendered text from V1-01 after the fix:

```text
WHAT CHANGED BECAUSE OF YOUR TRACK
Watch battery drain first, protect quality, and test battery before treating the design as feasible.
```

## Feedback Packets

### Mobile ML Expert - iPhone In V1-01

Activity:
- Select iPhone in V1-01.
- Inspect D-A-M diagnosis, intervention frontier, source trace, and report path.

What works:
- The track card names the iPhone hardware source and foregrounds battery,
  thermal headroom, privacy, local latency, and memory.
- The lab asks for the first engineering fix rather than treating model quality
  as the only axis.

Expert correction:
- The mobile narrative should keep app UX and update friction visible. Students
  should not leave thinking the phone is just a smaller accelerator.

Requirement generated:
- P1: Keep the new shared "What changed because of your track" cue visible in
  every track context.
- P1: When report schema tests are added, assert that V1-01 reports include
  selected D-A-M bottleneck and rejected alternative.

### TinyML / Wearable Expert - Oura Ring In V1-09

Activity:
- Select Oura Ring in V1-09.
- Inspect data policy, coverage, storage/cost evidence, and source trace.

What works:
- The Oura Ring track makes radio, OTA, storage, battery, and signal quality
  feel like first-class constraints.
- The selection helper gives the lab a reusable policy surface instead of a
  notebook-local table.

Expert correction:
- Any non-public internals must stay visibly labeled as MLSysIM estimates.
- Students need exact table fallback values for coverage and budget, not only a
  plot-level story.

Requirement generated:
- P1: Add explicit report assertions for next-data recommendation and residual
  sensor-quality risk.
- P1: Keep table fallbacks visible for coverage and budget plots.

### RoboTaxi / Autonomy Expert - RoboTaxi In V1-10

Activity:
- Select RoboTaxi in V1-10.
- Inspect compression options, p99/rare-event framing, source trace, and report
  path.

What works:
- The track correctly treats p99/p999 latency, safety margin, and rare-event
  recall as guardrails.
- V1-10 visibly marks incomplete fields when the residual-risk text area is
  empty. That is good TA-facing behavior.

Expert correction:
- The lab should not imply that generic edge inference and safety-critical
  autonomy are interchangeable.
- Safety claims should remain scenario logic for pedagogy, not claims about a
  real operator's internal system.

Requirement generated:
- P1: Add an autonomy/safety caveat to the RoboTaxi source-trace language where
  the lab discusses validation.
- P1: Ensure final reports include compression recipe, rejected alternative,
  hardware-support caveat, guardrail metric, and validation risk.

### Edge Systems Expert - V2-11 And Cross-Edge Review

Activity:
- Select each track in V2-11.
- Compare the edge-intelligence story against the RoboTaxi track.

What works:
- V2-11 has concrete memory and energy budget evidence.
- The track split is useful because iPhone, Oura Ring, RoboTaxi, and Cloud Fleet
  produce different rendered states.

Expert correction:
- RoboTaxi is the canonical Edge AI student track, but edge systems are broader
  than autonomous vehicles.
- General edge concepts should appear as comparison examples, not additional
  canonical tracks.

Requirement generated:
- P1: In V2-03 and V2-11 refinement passes, distinguish general edge patterns
  from the RoboTaxi safety-critical track.
- P1: Use topology/pipeline visuals in network and edge-placement labs where
  structure is the actual concept.

### Cloud / Fleet Expert - V2-10, V2-15, V2-17

Activity:
- Select Cloud Fleet in V2-10, V2-15, and V2-17.
- Inspect cost, latency, utilization, carbon, source trace, and report path.

What works:
- Cloud Fleet consistently foregrounds throughput, p99 latency, cost/request,
  utilization, and carbon.
- V2-17 uses the shared renderer to provide a consistent decision/report
  grammar.

Expert correction:
- Cloud economics should expose assumptions and avoid fake precision.
- Sustainability labs need region or grid provenance, ideally as a region table
  or map.

Requirement generated:
- P1: Add capacity-plan report fields for V2-10: SLA, traffic assumption,
  cost curve, utilization, and residual capacity risk.
- P1: Add region/carbon evidence to V2-15 as a table first, map later if useful.
- P1: V2-17 should eventually generate a final fleet design review from ledger
  decisions.

### Privacy / Responsible Systems Expert - V2-13

Activity:
- Select each track in V2-13.
- Inspect privacy/security source trace, decision, and residual risk.

What works:
- The shared renderer exposes residual risk visibly for all four tracks.
- The track switch changes the privacy/security story by deployment context.

Expert correction:
- Privacy/security needs a threat-model or control-stack modality; a generic
  frontier is a baseline, not the final expression of the concept.

Requirement generated:
- P1: Add a threat-model selector and privacy/security control-stack evidence to
  V2-13 in a later specialization pass.

### Sustainability Reviewer - V2-15

Activity:
- Select all tracks in V2-15 and compare device/fleet energy narratives.

What works:
- Carbon and residual risk appear in the shared renderer.
- Track switching keeps device energy distinct from cloud/fleet carbon.

Expert correction:
- Region/grid assumptions need a stronger evidence modality.

Requirement generated:
- P1: Add region/carbon table fallback before adding a world/region map.
- P2: Add a map only if it clarifies placement or grid mix without turning into
  decorative geography.

### TA / Grader

Activity:
- Inspect report export markers and incomplete-field behavior.

What works:
- Report export appears across the selected cluster after required activity.
- V1-10 correctly marks incomplete fields when a required residual-risk field is
  empty.
- Reports are local and do not require accounts or LMS state.

Correction:
- The grading story needs formal schema tests, not only rendered buttons.

Requirement generated:
- P0: Add report schema/generation tests for all deep labs.
- P0: Assert required sections: selected track, scenario, predictions, evidence
  summary, final decision, residual risk, source trace, and incomplete fields.

### Instructor

Activity:
- Review selected labs as assignable standalone activities.

What works:
- The four-track system gives instructors clear assignment modes: individual
  choice, assigned track, lecture demo, or capstone continuity.
- The repeated source-trace and report-export pattern makes the labs assignable.

Correction:
- Instructors still need rubric exemplars and expected outcomes.

Requirement generated:
- P1: Add one complete rubric exemplar for V1-10.
- P1: Add one complete rubric exemplar for V2-17.
- P1: Keep instructor metadata outside the default student flow.

### Maintainer

Activity:
- Check whether this pass added source facts or notebook-local constants.

What works:
- The applied UI fix consumes existing `TrackProfile` fields only.
- No new hardware, model, system, cost, memory, latency, energy, or fleet facts
  were added.
- Render smoke and static tests remained clean.

Correction:
- Variant ref resolution should be tested directly rather than inferred from
  render success.

Requirement generated:
- P1: Add tests that every lab variant resolves referenced hardware, model,
  system, and infrastructure refs.

## Priority Requirements After This Pass

P0:

| Requirement | Reason |
|---|---|
| Add report schema/generation tests across all deep labs | TA and instructor adoption depends on gradeable reports, not only visible buttons. |
| Verify stable labs mark incomplete required fields explicitly | V1-10 shows the right behavior; it needs to be a contract. |

P1:

| Requirement | Reason |
|---|---|
| Add Lab 00-specific release activity smoke | Lab 00 has gated checks and cannot be validated fully by load-only smoke. |
| Add variant ref resolution tests | Maintainer confidence should not depend on rendered text inspection. |
| Add rubric exemplars for V1-10 and V2-17 | Instructors need copy-ready grading anchors. |
| Specialize V2-03 with network/topology/fabric visuals | Current review matrix already flags this as concept drift. |
| Add V2-13 threat-model/control-stack modality | Privacy/security needs a better concept-specific interaction. |
| Add V2-15 region/carbon table, then possibly map | Sustainability needs stronger provenance and evidence. |
| Add V2-17 final fleet design review report | Capstone should become the portfolio artifact. |

P2:

| Requirement | Reason |
|---|---|
| Add more numeric prediction moments | Magnitude intuition is under-used. |
| Add multiselect stack builders for defenses, monitoring, and governance | The catalog has these modalities but implementation is sparse. |
| Add heatmaps only where two knobs define a regime | Avoid adding visual variety without pedagogy. |

## Scoring Snapshot

| Dimension | Current cluster score | Notes |
|---|---:|---|
| Track differentiation | 2 | All non-orientation labs rendered four distinct track states. |
| Chapter pedagogy | 1-2 | Bespoke labs are stronger; shared V2 renderer is consistent but sometimes generic. |
| Prediction discipline | 1-2 | Radios/sliders are widespread; numeric prediction remains sparse. |
| Evidence modality fit | 1-2 | Frontiers and budget bars are strong; maps/topology/heatmaps are still gaps. |
| Source trace | 2 | Source trace is visible in selected non-orientation labs. |
| Report usefulness | 1 | Report export exists, but schema/generation testing is still missing. |
| Domain realism | 1-2 | Track narratives are strong; domain experts request caveats and richer modalities. |
| Accessibility fallback | 1 | Table fallback exists in places, but it is not yet a release-tested contract. |
| Maintainability | 2 for this pass | No new source facts or notebook constants were added. |

## Commands Run

```bash
python3 labs/tools/render_lab_smoke.py --labs labs/vol1/lab_00_introduction.py labs/vol1/lab_01_ml_intro.py labs/vol1/lab_09_data_selection.py labs/vol1/lab_10_model_compress.py labs/vol2/lab_10_inference.py labs/vol2/lab_11_edge_intelligence.py labs/vol2/lab_13_security_privacy.py labs/vol2/lab_15_sustainable_ai.py labs/vol2/lab_17_fleet_synthesis.py --port-start 29900 --output-dir /tmp/mlsysbook-feedback-pass-20260604 > /tmp/mlsysbook-feedback-pass-20260604/results.json
python3 -m py_compile labs/mlsysbook_labs/ui.py
python3 -m pytest labs/tests/test_static.py -q
python3 -m build --wheel labs
cp labs/dist/mlsysbook_labs-0.1.0-py3-none-any.whl wheels/mlsysbook_labs-0.1.0-py3-none-any.whl
python3 labs/tools/render_lab_smoke.py --labs labs/vol1/lab_01_ml_intro.py labs/vol2/lab_17_fleet_synthesis.py --port-start 30100 --output-dir /tmp/mlsysbook-feedback-cue-check > /tmp/mlsysbook-feedback-cue-check/results.json
python3 labs/tools/render_lab_smoke.py --labs labs/vol1/lab_00_introduction.py labs/vol1/lab_01_ml_intro.py labs/vol1/lab_09_data_selection.py labs/vol1/lab_10_model_compress.py labs/vol2/lab_10_inference.py labs/vol2/lab_11_edge_intelligence.py labs/vol2/lab_13_security_privacy.py labs/vol2/lab_15_sustainable_ai.py labs/vol2/lab_17_fleet_synthesis.py --port-start 30200 --output-dir /tmp/mlsysbook-feedback-pass-20260604-after-cue > /tmp/mlsysbook-feedback-pass-20260604-after-cue/results.json
```

## Decision

The first execution pass confirms the overall direction:

- The four canonical tracks are the right balance.
- The expert council should remain explicit: Cloud/Fleet, Edge systems,
  RoboTaxi/autonomy, Mobile ML, and TinyML/wearable each catch different issues.
- The shared track card was the right place to add the first student-facing
  feedback improvement.
- The next engineering pass should focus on report schema/generation tests,
  because the simulated TA and instructor feedback both converge there.
