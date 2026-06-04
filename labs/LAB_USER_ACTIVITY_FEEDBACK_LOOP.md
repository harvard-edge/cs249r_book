# Lab User Activity Feedback Loop

Date: 2026-06-04

This document defines the simulated adoption loop for the MLSysBook labs. It is
not a replacement for real classroom testing. It is the engineering feedback
loop we can run before real deployment: simulate student use, TA grading,
instructor adoption, domain-expert review, and maintainer review, then work
backward to identify what the labs still need.

The loop should stay autonomous. The goal is not to ask the user for repeated
decisions. The goal is to generate lab artifacts, review them through the right
expert lenses, produce concrete requirements, and improve the labs while
preserving the single-source-of-truth rule.

## Current Balance Snapshot

The lab system currently has 34 labs:

| Volume | Labs | Current status |
|---|---:|---|
| Volume I | 17 | Track-aware student labs with richer bespoke helper coverage in several labs. |
| Volume II | 17 | Track-aware labs, with many remaining labs using the shared system-design renderer. |

Rendered-browser validation:

| Check | Result |
|---|---|
| Full catalog browser smoke | 34 passed, 0 failed |
| Track switching | Every non-orientation lab produced four distinct rendered track states |
| Track selector duplication | Checked by browser smoke |
| Long source-ref overflow | Checked by browser smoke |

The four canonical student tracks are the right spine. Do not add more student
tracks unless the course structure changes.

| Category | Canonical track | Source of truth | Narrative purpose |
|---|---|---|---|
| Mobile ML | iPhone | `Hardware.Mobile.iPhone15Pro` | User experience, battery, thermal behavior, privacy, local latency. |
| TinyML / wearable | Oura Ring | `Hardware.Tiny.OuraRing` | SRAM, flash, duty cycle, battery, OTA payload, sensor quality. |
| Edge AI | RoboTaxi | `Hardware.Edge.RoboTaxi` | p99/p999 latency, safety, sensor bandwidth, local autonomy. |
| Cloud/Fleet | Cloud Fleet | `Hardware.Cloud.H100` and `Systems.Clusters.Lab_64_H100` | Throughput, SLA, cost/request, utilization, reliability, carbon. |

Domain feedback should be broader than the four student tracks. RoboTaxi is the
canonical Edge track, but it needs both a general edge-systems review and an
autonomy/safety review.

## Expert Review Council

Every serious feedback pass should include these expert lenses.

| Expert lens | Reviews | Questions they should answer |
|---|---|---|
| Mobile ML expert | iPhone track and any mobile runtime or on-device inference claims | Do the constraints feel like a real phone deployment? Are thermal, battery, privacy, UX, and update-path issues represented correctly? |
| TinyML / wearable expert | Oura Ring track and MCU/wearable claims | Are SRAM, flash, duty cycle, radio, battery, OTA, and sensor-quality limits treated as first-class constraints? |
| Edge systems expert | Edge deployment patterns beyond one vehicle | Are local compute, uplink, site heterogeneity, update orchestration, reliability, and disconnected operation represented clearly? |
| RoboTaxi / autonomy expert | RoboTaxi track | Are rare-event recall, p99/p999 latency, safety validation, sensor bandwidth, and fallback behavior handled with the right seriousness? |
| Cloud / fleet expert | Cloud Fleet track | Are throughput, utilization, SLA, cost, carbon, failures, observability, and multi-tenant assumptions realistic enough for teaching? |
| ML systems instructor | Chapter-to-lab pedagogy | Does the lab teach the chapter concept, or just expose controls? |
| TA / grader | Report and rubric | Can this be graded from the downloaded report without reverse-engineering the notebook? |
| First-time student proxy | Usability | Does the student know what to do, what changed, and what to submit? |
| Strong systems student proxy | Depth | Are advanced assumptions visible enough without overwhelming the main flow? |
| Maintainer | Source-of-truth and implementation quality | Are facts in MLSysIM or `mlsysbook_labs`, not floating in notebooks? Can tests guard the behavior? |

## Feedback Packet

Every simulated reviewer should produce the same feedback packet so findings can
be compared across labs and tracks.

| Field | Meaning |
|---|---|
| Reviewer role | Student, TA, instructor, domain expert, maintainer. |
| Track under review | iPhone, Oura Ring, RoboTaxi, Cloud Fleet, or cross-track. |
| Lab and part | Exact lab and part or shared component being reviewed. |
| Activity performed | What the reviewer clicked, changed, read, or graded. |
| Expected learning | What the reviewer thinks the lab is trying to teach. |
| Observed learning | What the rendered lab actually made visible. |
| Strongest evidence | Plot, table, report field, decision card, or scenario text that best supported learning. |
| Confusion or defect | Missing context, weak narrative, bad default, unclear graph, grading problem, or source-truth issue. |
| Expert correction | Domain-specific correction or missing assumption. |
| Requirement generated | Concrete implementation requirement. |
| Priority | P0 release blocker, P1 adoption polish, P2 enrichment. |

## User Activity Scripts

These scripts simulate the whole motion of using and reviewing the labs. Each
script should produce a feedback packet.

### Activity 1 - New Student Chooses A Track In Lab 00

Persona:
- Senior undergraduate student with one prior ML course and one systems course.

Steps:
- Open `labs/vol1/lab_00_introduction.py`.
- Read the opening structure.
- Select one track.
- Inspect the track profile and where the track fits across the course.
- Download or preview the first local report artifact if available.

Feedback to collect:
- Does the student understand that the track persists?
- Does the student understand that iPhone, Oura Ring, RoboTaxi, and Cloud Fleet
  are canonical profiles, not exhaustive device lists?
- Does the student know what later labs will change because of this choice?

Backward requirements:
- Lab 00 must explicitly preview the rhythm: prediction, evidence, decision,
  reflection, and report.
- Lab 00 should show the four tracks as course identities and explain why there
  are not many more tracks.
- If report export is demonstrated in Lab 00, the report should include track,
  scenario, predictions, evidence, decision, reflection, and residual risk.

### Activity 2 - Student Completes V1-01 On iPhone

Persona:
- First-time student using the Mobile ML track.

Steps:
- Open `labs/vol1/lab_01_ml_intro.py`.
- Select iPhone.
- Make the Data-Algorithm-Machine prediction.
- Inspect the intervention frontier.
- Choose the first fix and download a report.

Expected behavior:
- The student should see the iPhone story as local UX under battery, thermal,
  privacy, memory, and latency constraints.

Expert feedback required:
- Mobile ML expert checks whether the narrative overstates exact iPhone hardware
  details or underplays app UX and update friction.
- Instructor checks whether D-A-M coupling is visible enough.
- TA checks whether the report explains why the selected intervention wins.

Backward requirements:
- The final report should include the selected D-A-M bottleneck.
- The lab should include a concise "what changed because of iPhone" statement.
- Hardware-derived values must resolve through shared registry refs.

Current example in place:
- V1-01 variants already differentiate iPhone, Oura Ring, RoboTaxi, and Cloud
  Fleet objectives in `mlsysbook_labs/variants.py`.

### Activity 3 - Student Completes V1-09 On Oura Ring

Persona:
- Student using TinyML / wearable track.

Steps:
- Open `labs/vol1/lab_09_data_selection.py`.
- Select Oura Ring.
- Predict which data policy improves coverage without blowing the budget.
- Inspect coverage, storage, cost, and guardrail evidence.
- Choose the next-data policy and export the report.

Expected behavior:
- The student should feel that "more data" is not free because radio, storage,
  OTA payload, battery, and signal quality bind.

Expert feedback required:
- TinyML/wearable expert checks whether SRAM, flash, duty cycle, radio, and OTA
  assumptions are surfaced with the right caveats.
- TA checks whether the selected policy and rejected policy are visible in the
  report.
- Student proxy checks whether "coverage" and "data quality" are concrete.

Backward requirements:
- Oura Ring reports should include the next-data recommendation and residual
  sensor-quality risk.
- Visual evidence should include table fallback with exact coverage and budget
  values.
- Any non-public device assumption must be labeled as an MLSysIM estimate.

Current example in place:
- V1-09 has a dedicated `mlsysbook_labs.selection` helper and track-specific data
  policy variants.

### Activity 4 - Student Completes V1-10 On RoboTaxi

Persona:
- Strong student using the RoboTaxi track.

Steps:
- Open `labs/vol1/lab_10_model_compress.py`.
- Select RoboTaxi.
- Predict which compression approach wins.
- Compare quantization, pruning, and distillation evidence.
- Choose a compression recipe and residual validation risk.

Expected behavior:
- The student should learn that smaller is not automatically safer or faster.
  Rare-event recall and p99/p999 latency can dominate.

Expert feedback required:
- RoboTaxi/autonomy expert checks safety framing, rare-event recall language,
  latency-tail treatment, fallback assumptions, and validation seriousness.
- Edge systems expert checks whether sensor-to-compute and local deployment
  constraints are realistic beyond the vehicle story.
- Instructor checks whether compression is taught as deployment engineering, not
  only model-size reduction.

Backward requirements:
- The report should include the compression recipe, rejected alternative,
  hardware support caveat, guardrail metric, and validation risk.
- The lab should avoid implying that generic edge deployment and safety-critical
  autonomous driving are interchangeable.
- Any safety claim should be framed as pedagogical scenario logic, not an
  assertion about a real operator's internal system.

Current example in place:
- V1-10 uses `CompressionModel` plus track variants; the RoboTaxi track points to
  `Hardware.Edge.RoboTaxi` and `Models.Vision.YOLOv8_Nano`.

### Activity 5 - TA Grades A V1 Report

Persona:
- TA grading 30 reports from mixed tracks.

Steps:
- Open a downloaded Markdown report.
- Identify selected track, lab ID, predictions, evidence, decision, residual
  risk, and any caveats.
- Grade without opening the notebook.

Feedback to collect:
- Can the TA distinguish "student clicked through" from "student reasoned"?
- Are incomplete fields visible?
- Does the report map to a stable rubric?
- Can mixed-track reports be graded fairly?

Backward requirements:
- Add formal report schema tests across all deep labs.
- Add rubric fields that match report fields.
- Reports should mark missing required fields rather than silently omitting them.
- The report should preserve enough internal provenance for domain assumptions,
  without requiring the notebook page to show implementation refs.

Current gap:
- Rendered report export exists, but exhaustive report-generation and schema
  validation remain a separate hardening pass.

### Activity 6 - Instructor Assigns One Standalone Lab

Persona:
- Instructor adopting one lab outside the full MLSysBook sequence.

Steps:
- Pick one lab, such as V1-10 Compression or V2-13 Privacy.
- Read the track plan and feedback docs.
- Decide whether to assign all tracks, assigned tracks, or one default track.
- Use the downloaded report as the submission artifact.

Feedback to collect:
- Is the assignment prompt copy-ready?
- Is the expected report clear?
- Are common misconceptions named?
- Is the timebox realistic?

Backward requirements:
- Every lab plan should include assignment modes, expected track outcomes,
  common misconceptions, completion path, assumptions, and rubric sketch.
- Student-facing notebooks should remain compact; instructor metadata should not
  make the lab feel heavier.

Current examples in place:
- `LAB_TRACK_PLAN_FEEDBACK_SIMULATION.md` already recommends assignment modes,
  expected outcomes, timebox, misconceptions, and rubric sketches.

### Activity 7 - Cloud Expert Reviews V2-10 Or V2-17

Persona:
- Cloud/fleet systems expert.

Steps:
- Select Cloud Fleet in V2-10 Inference Economy or V2-17 Fleet Synthesis.
- Inspect capacity, latency, cost, utilization, and residual risk.
- Review the final report artifact.

Feedback to collect:
- Are cost/request, p99 latency, utilization, and carbon treated as coupled?
- Does the lab avoid fake precision in cloud economics?
- Does the fleet story include observability and failure modes?

Backward requirements:
- Cloud-heavy labs should include capacity-plan report fields: SLA, traffic
  assumption, cost curve, utilization, residual capacity risk.
- Cloud economics should expose scenario assumptions and avoid hidden constants.
- V2-17 should produce a final fleet design review from ledger decisions.

Current examples in place:
- Shared Volume II renderer provides frontier, scaling curve, decision memo,
  validation tests, residual risk, ledger save, and report export.

### Activity 8 - Edge Expert Reviews V2-03, V2-06, And V2-11

Persona:
- Edge systems expert.

Steps:
- Review V2-03 communication/fabric, V2-06 collective communication, and V2-11
  edge intelligence.
- Compare how edge constraints are represented across network, collectives, and
  local adaptation.

Feedback to collect:
- Are fabric, collective, and edge-device concepts separated cleanly?
- Are local compute, uplink, heterogeneity, updates, and disconnected operation
  present where appropriate?
- Does the RoboTaxi track dominate edge thinking too much?

Backward requirements:
- V2-03 should specialize into real network fabric/topology design.
- V2-06 should remain the collective communication lab.
- V2-11 should keep device constraints explicit, especially memory and energy.
- If general edge scenarios are discussed, they should be comparison examples,
  not new canonical tracks.

Current gap:
- The current review matrix already flags V2-03 as the main concept-drift issue:
  network fabrics and collective communication need a cleaner separation.

### Activity 9 - Sustainability/Region Reviewer Checks V2-15

Persona:
- Cloud/fleet expert with sustainability focus.

Steps:
- Select Cloud Fleet and compare against iPhone, Oura Ring, and RoboTaxi.
- Inspect energy/carbon evidence and final decision.

Feedback to collect:
- Does the lab distinguish device energy from datacenter/fleet carbon?
- Are region/grid assumptions visible?
- Is the carbon visual appropriate?

Backward requirements:
- V2-15 should eventually include a world/region map or region table modality.
- Region values must be scenario inputs or MLSysIM registry facts with provenance.
- Carbon reports should include residual risk around grid mix and demand rebound.

Current gap:
- World/region maps are in the catalog but are not yet meaningfully realized in
  the implemented modality mix.

### Activity 10 - Maintainer Runs Source-Of-Truth Audit

Persona:
- Maintainer responsible for MLSysIM and lab integrity.

Steps:
- Inspect changed notebooks and helpers.
- Verify hardware/model/system facts resolve from MLSysIM or `mlsysbook_labs`.
- Check whether plots and reports consume typed helper outputs.
- Run static and browser smoke tests.

Feedback to collect:
- Are there notebook-local constants that should be registry facts?
- Are scenario values typed variants rather than scattered literals?
- Are reusable plots/helpers shared instead of copied?

Backward requirements:
- Add tests that every variant resolves hardware, model, system, and
  infrastructure refs.
- Add report schema tests.
- Keep `render_lab_smoke.py` as release smoke, not just one-off validation.

## Modality Balance Review

The current implementation uses the common controls well, but does not yet
exercise the full interaction catalog.

| Modality | Current balance | Feedback interpretation |
|---|---|---|
| Track selector | Strong | Keep stable; four canonical tracks only. |
| Radio prediction | Strong | Good for prediction-before-reveal; avoid using it for trivial checks. |
| Sliders | Strong | Useful, but many labs may feel similar if every lesson is slider-driven. |
| Dropdown strategy selectors | Strong | Appropriate for secondary options and report choices. |
| Tabs | Strong | Good for compact rendering; verify students do not miss required tabs. |
| Scatter/frontier plots | Strong | Core ML systems trade-off modality. |
| Bar/budget stacks | Strong | Good for memory, latency, energy, cost. |
| Text reflection | Moderate | Needs more consistent report alignment, but avoid reflection fatigue. |
| Numeric prediction | Sparse | Add where magnitude is the lesson: memory, p99, energy, cost, checkpoint time. |
| Stack builder / multiselect | Sparse | Add for defenses, monitoring signals, compression recipes, governance controls. |
| Heatmap / phase diagram | Gap | Add where two knobs define regimes: batch x sequence, bit-width x sparsity, replicas x arrival rate. |
| Topology / network diagram | Gap | Needed for V2-03 and V2-06. |
| Sankey / pipeline diagram | Gap | Needed for data pipeline, serving path, training pipeline, storage. |
| World / region map | Gap | Needed for V2-15 carbon and possibly cloud placement/privacy. |
| Timeline | Partial | Useful for drift, canary, fault tolerance, retraining, thermal behavior. |
| Report schema validation | Gap | Needed for TA/instructor adoption. |

The feedback loop should not force every lab to use every modality. The target
is one primary visual per part, selected because it teaches the chapter concept.

## Balance Targets By Track

Each track should repeatedly teach different system instincts.

| Track | Good recurring evidence | Avoid |
|---|---|---|
| iPhone | Battery/thermal budget, local latency, memory pressure, privacy boundary, app/update friction. | Treating the phone as a tiny cloud server. |
| Oura Ring | SRAM/flash fit, duty cycle, OTA payload, radio wakeups, sensor quality, battery life. | Using generic mobile assumptions or hiding estimated internals. |
| RoboTaxi | p99/p999 latency, sensor bandwidth, rare-event recall, safety validation, fallback and thermal headroom. | Framing safety-critical autonomy as ordinary edge inference. |
| Cloud Fleet | Throughput, cost/request, p99 latency, utilization, failures, observability, carbon. | Treating cloud as unlimited hardware or using fake economic precision. |

## Balance Targets By Volume

| Volume | Student level | Feedback target |
|---|---|---|
| Volume I | Senior undergraduate | Make one core ML systems concept visible through a concrete track-specific decision. Reports should show prediction, evidence, bottleneck, decision, and residual risk. |
| Volume II | Senior/Master's/Ph.D. | Make fleet/system design trade-offs visible. Reports should include capacity, reliability, cost, operational risk, validation caveats, and defensible system policy. |

Volume I should be concrete and approachable. Volume II can be more systems-rich,
but it still needs clear student artifacts.

## Scoring Rubric For Each Lab

Use a 0-2 score for each dimension.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Track differentiation | Track only changes labels. | Track changes some narrative or defaults. | Track changes constraints, evidence, failure mode, and final decision. |
| Chapter pedagogy | Concept unclear. | Concept present but not central. | Lab clearly teaches one chapter systems idea. |
| Prediction discipline | No prediction. | Prediction exists but is not used. | Prediction is compared to computed evidence and misconception. |
| Evidence modality fit | Visual feels generic. | Visual partly supports the lesson. | Visual is the right representation for the concept. |
| Source-of-truth discipline | Facts are hidden or notebook-local. | Some refs resolve centrally but leak into learner copy. | Facts resolve centrally while learner copy stays scenario-focused and reports preserve provenance. |
| Report usefulness | Report is missing or vague. | Report captures outputs but weak reasoning. | Report can be graded without opening the notebook. |
| Domain realism | Expert finds major mismatch. | Acceptable with caveats. | Expert agrees the scenario is pedagogically realistic and caveated. |
| Accessibility fallback | Visual-only. | Some text/table support. | Table/text fallback contains exact values and status labels. |
| Maintainability | Notebook owns facts or formulas. | Mixed ownership. | Facts and reusable computation live in MLSysIM or `mlsysbook_labs`. |

Release target:
- No P0 issue in any dimension.
- Score 2 on track differentiation, chapter pedagogy, source-of-truth discipline, and report
  usefulness for every stable lab.
- Score 2 on domain realism for the primary track narrative in each lab.

## Backward Requirements Matrix

| Feedback signal | Requirement |
|---|---|
| Student cannot tell what changed by track | Add a compact "What changed because of your track" callout after the main reveal. |
| Student clicks controls but cannot name the lesson | Strengthen chapter recap, systems translation, and misconception reveal. |
| TA cannot grade the report | Add report schema tests and rubric-aligned fields. |
| Instructor cannot assign the lab standalone | Add assignment mode, expected report, timebox, common misconception, and rubric sketch to the track plan or instructor guide. |
| Domain expert rejects assumptions | Move the assumption to MLSysIM or typed variants, label it as an estimate, and update report provenance. |
| Expert says track is too generic | Add track-specific failure mode and final decision framing. |
| Volume II labs feel too mechanically similar | Specialize high-value shared-renderer labs with concept-specific modalities. |
| Plot is visually interesting but not gradeable | Add table fallback and report snapshot. |
| Heatmap/map/topology is needed but absent | Add the modality only in the labs where it teaches the chapter concept. |
| Notebook contains source facts | Move facts to MLSysIM registry or `mlsysbook_labs` variant/helper first. |

## Priority Findings From The Current State

P0:
- Add report schema validation across all deep labs.
- Ensure every stable lab report can be generated locally and marks incomplete
  fields explicitly.

P1:
- Add instructor rubric exemplars for one Volume I lab and one Volume II lab.
- Add per-lab assignment/adoption metadata where it is currently only planned.
- Add formal variant-ref resolution tests for hardware, model, system, and
  infrastructure refs.

P1 targeted modality enrichment:
- V2-03 should use network/topology/fabric visuals.
- V2-05 should use parallelism or memory/communication regime visuals.
- V2-13 should use privacy/security control stack and trade-off frontier.
- V2-15 should use region/carbon map or region table.
- V2-17 should produce a final fleet design review from ledger decisions.

P2:
- Add richer stack builders for monitoring, defenses, compression recipes, and
  governance controls.
- Add numeric prediction moments where order-of-magnitude intuition matters.
- Add compact student progress state in the lab map.

## Current Examples To Reuse

| Lab | What it demonstrates |
|---|---|
| V1-01 AI Triad | Track-specific Data-Algorithm-Machine diagnosis. |
| V1-09 Selection Paradox | Data policy and subgroup/coverage reasoning tied to track constraints. |
| V1-10 Compression Paradox | Compression decisions depend on hardware support and guardrail metrics. |
| V1-12 Benchmarking Trap | Benchmark visuals expose sustained behavior, tails, and misleading metrics. |
| V2-11 Edge Intelligence | Memory and energy budgets become concrete for local adaptation. |
| V2-17 Fleet Synthesis | Capstone behavior can summarize prior ledger decisions. |
| Shared Volume II renderer | Gives a common grammar: track context, where-this-fits arc, frontier, scaling curve, decision, validation, residual risk, report export. |

## Operating Loop

Run this loop for each release pass.

1. Pick a lab cluster:
   - one introductory lab,
   - one device-constrained lab,
   - one cloud/fleet lab,
   - one responsible/sustainability lab,
   - one capstone or synthesis lab.
2. Render the labs with `labs/tools/render_lab_smoke.py`.
3. Simulate student completion for every canonical track.
4. Generate or inspect report artifacts.
5. Route feedback to the expert review council.
6. Convert feedback packets into requirements.
7. Classify each requirement as P0, P1, or P2.
8. Implement only high-confidence P0/P1 fixes in the current pass.
9. Update implementation notes and checklist.
10. Commit the pass as a coherent slice.

## Decision

The track balance is right. The next quality gains should come from expert
feedback and modality discipline, not from adding more canonical devices.

The review council must include Cloud/Fleet, Edge systems, RoboTaxi/autonomy,
Mobile ML, and TinyML/wearable expertise. Those perspectives should shape the
requirements for each lab because the same pedagogical lesson must feel
different when the deployment constraints are different.
