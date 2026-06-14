# V1-15 Track Plan: Responsible Engineering Release Gate

## Chapter Invariant

Responsible ML constraints include people and policy. Fairness, privacy, safety,
auditability, and blast radius are measurable system guardrails, not separate
review slogans.

The lab uses one shared Part A/B/C/D concept sequence for every track. Tracks
change the stakeholder, operating envelope, evidence emphasis, thresholds,
failure mode, and memo framing; they do not introduce different concepts.

## Reading Map

| Lab module | Chapter anchor | Source claim used in the notebook |
|---|---|---|
| Opening | Responsibility as Systems Engineering | A system can satisfy conventional metrics while amplifying harm if the specification omits people and policy. |
| Part A | Testing across populations; worked fairness example | Thresholds encode values because demographic parity, equal opportunity, equalized odds, calibration, FPR, and FNR cannot all be optimized at once under different base rates. |
| Part B | Technical privacy protection methods; architecting for regulatory compliance | Differential privacy, data minimization, consent, retention, and erasure change what evidence can be collected and whether deployment is allowed. |
| Part C | Silent failure modes; predeployment assessment; monitoring and incident response | Release safety depends on rare-event thresholds, rollback, human review, and blast-radius limits, not on average accuracy alone. |
| Part D | Building data lineage infrastructure; audit infrastructure and accountability; regulatory landscape | Lineage, immutable audit logs, decision context, and ownership convert technical measurements into accountable release decisions. |
| Synthesis | Summary | Abstract obligations become concrete, testable engineering constraints carried into the capstone release audit. |

## Concept Inventory

### Accepted Concepts

- Metric thresholds encode values and trade off errors across groups and
  stakeholders.
- Privacy budget and data minimization change the available model evidence and
  the deployability of the release.
- Safety thresholds and blast radius determine whether a release is acceptable
  even when the main model metric is strong.
- Audit trail and governance convert measurements, policy choices, and residual
  risk into accountable decisions.
- The synthesis memo carries one responsible release constraint into the final
  Volume I capstone.

### Rejected Or Deferred Concepts

- Carbon/TCO as the main Part D concept. It belongs to the chapter, but this lab
  needs the release-governance arc requested for V1-15; carbon can appear as a
  secondary track concern, especially for Cloud Fleet.
- Explainability latency as a standalone concept module. Explanation remains a
  governance artifact, but privacy, safety, and auditability carry the stronger
  shared release sequence here.
- A pure legal-compliance checklist. The chapter frames compliance as systems
  engineering; the lab must make students manipulate measurable constraints.
- Separate track-specific concepts. Track differences are realizations of the
  same four concepts, not separate curricula.

## Concept Modules

### Part A: Concept Module - Thresholds Encode Values

- Chapter claim: fairness metrics and subgroup thresholds are engineering
  choices because aggregate quality hides harmed stakeholders.
- Student prior: one high aggregate score or one threshold should be enough.
- Scenario beat: the selected track's stakeholder must decide whether a shared
  decision threshold can ship.
- Prediction beat: students predict which risk blocks release.
- Manipulation beat: students adjust subgroup/context base rates and the shared
  threshold.
- Evidence beat: grouped metric chart plus exact table for accuracy, FPR, FNR,
  PPV, approval/share, and threshold status.
- Consequence beat: the notebook names which stakeholder absorbs false-positive
  or false-negative harm when the selected threshold crosses the target gap.
- Math/source beat: Math Peek ties the result to TPR, FPR, PPV, demographic
  parity/equal opportunity/equalized odds tension, and the chapter threshold
  example.
- Checkpoint beat: students choose the release threshold policy for the report.
- Ledger output: threshold, FPR gap, target gap, harmed stakeholder, selected
  threshold policy.

### Part B: Concept Module - Privacy Budget Changes Evidence

- Chapter claim: privacy-preserving techniques and data minimization are
  deployability constraints; they limit what individual evidence can be retained
  while reducing membership-inference and retention risk.
- Student prior: more data always improves responsible evidence, or stronger
  privacy only changes the policy text.
- Scenario beat: the same stakeholder must collect enough audit evidence without
  violating consent, retention, epsilon, or local-processing constraints.
- Prediction beat: students predict how privacy/data minimization changes the
  release decision.
- Manipulation beat: students set epsilon, retained evidence days, raw-data
  collection percentage, and local/federated processing level.
- Evidence beat: privacy/evidence chart plus table showing epsilon use, records
  retained, effective audit evidence, evidence confidence, model-evidence
  penalty, membership risk, retention status, and deployability.
- Consequence beat: the lab names when a release fails because privacy budget is
  exhausted or because minimization leaves too little evidence for the audit.
- Math/source beat: Math Peek cites the epsilon-DP bound, data minimization, TTL,
  membership inference, and the chapter's privacy/governance section.
- Checkpoint beat: students choose deploy, revise evidence collection, or hold
  for a privacy/data-card review.
- Ledger output: epsilon, retention days, raw collection, local processing,
  evidence confidence, privacy deployability decision.

### Part C: Concept Module - Safety Thresholds Bound Blast Radius

- Chapter claim: silent failures and incident response require safety thresholds,
  rollback, and blast-radius limits before deployment.
- Student prior: if average model quality is high, the release can ship.
- Scenario beat: the selected track plans a staged release and must bound harm if
  the rare-event slice fails.
- Prediction beat: students predict which release gate will block first.
- Manipulation beat: students set safety threshold, canary exposure, rollback
  readiness, and human-review/fallback coverage.
- Evidence beat: release-gate chart and exact table for rare-event risk, affected
  units, p99/latency or operational guardrail, rollback minutes, human review,
  blast radius, and pass/fail status.
- Consequence beat: the notebook names the failure boundary: safety threshold
  miss, blast-radius cap exceeded, rollback too slow, or human review below the
  track minimum.
- Math/source beat: Math Peek links risk above threshold, exposure size,
  detection time, and rollback to the chapter's silent failure and incident
  response framework.
- Checkpoint beat: students choose canary, hold, or expand release, with a named
  failure owner.
- Ledger output: safety threshold, canary share, blast radius, release status,
  residual safety risk.

### Part D: Concept Module - Governance Makes Evidence Accountable

- Chapter claim: lineage, immutable audit logs, and prediction-time context are
  required to reconstruct who used which data/model/threshold and why.
- Student prior: a model card or dashboard is enough accountability.
- Scenario beat: the release board asks whether the evidence from Parts A-C is
  auditable and assigned to accountable owners.
- Prediction beat: students predict which governance gap prevents sign-off.
- Manipulation beat: students set lineage coverage, immutable log retention,
  decision-context logging, access review cadence, and owner sign-off.
- Evidence beat: governance readiness chart and table showing audit completeness,
  decision reconstruction, retention obligation, missing evidence, and sign-off.
- Consequence beat: the lab names when release evidence cannot support
  contestability, erasure, adverse-action, safety-case, or incident review.
- Math/source beat: Math Peek ties audit volume to decision volume and retention,
  and ties lineage to erasure/deletion propagation.
- Checkpoint beat: students choose a governance decision and memo owner.
- Ledger output: audit readiness score, missing governance control, accountable
  owner, release memo status.

### Synthesis: Responsible Release Memo

- Students assemble one memo with the selected threshold/policy, privacy
  evidence policy, safety/blast-radius release gate, governance decision, harmed
  stakeholder or residual risk, and one carry-forward capstone constraint.
- The synthesis ties every module back to the invariant: responsible engineering
  is constrained optimization over people, policy, evidence, and machine limits.

## Track Narratives

| Track | Persona | Same concept sequence realized as | Track-specific failure mode | Report framing |
|---|---|---|---|---|
| iPhone | Mobile responsibility lead | On-device privacy, accessibility cohorts, battery/UX harm, local explanation and telemetry limits | Accessibility/context cohort fails or privacy-safe telemetry misses a small harmed group | Local release memo defending privacy-preserving threshold and accessibility evidence. |
| Oura Ring | Wearable responsibility lead | Sensitive biosignals, consent, low-power sensing, false reassurance, short retention | Privacy budget passes but evidence is too thin, or false reassurance risk exceeds the safety gate | Health-adjacent release memo with consent, retention, and residual false reassurance risk. |
| RoboTaxi | Autonomous safety accountability lead | Safety-critical rare events, p99 deadlines, canary size, fallback and replay evidence | Rare-event recall or blast radius fails despite strong aggregate perception quality | Safety-case memo with canary limit, rollback path, replay evidence, and accountable owner. |
| Cloud Fleet | Responsible AI platform owner | Cohort fairness, tenant/language evidence, privacy/security review, policy moderation, blast radius | Population-scale subgroup harm or policy/moderation incident exceeds canary cap | Platform governance memo with cohort policy, appeal path, audit evidence, and blast-radius cap. |

Each track changes at least persona, constraints, metric priorities, failure
threshold, report prompt, and ledger evidence. The underlying concepts remain
identical.

## Mechanics Plan

| Module | Controls | Evidence mechanics | Failure state |
|---|---|---|---|
| Part A | Prediction radio; subgroup/base-rate sliders; threshold slider | Grouped bar chart; metric cards; exact metric table | FPR gap exceeds track target. |
| Part B | Prediction radio; epsilon slider; retention-days slider; raw collection slider; local/federated slider | Budget/evidence bar chart; deployability cards; exact privacy/evidence table | Privacy budget exhausted or audit evidence below minimum. |
| Part C | Prediction radio; safety threshold slider; canary slider; rollback slider; human review slider | Gate-ratio chart; blast-radius cards; release gate table | Safety gate miss, blast-radius cap exceeded, rollback too slow, or review below minimum. |
| Part D | Prediction radio; lineage slider; audit retention slider; decision-context slider; access review slider; owner sign-off radio | Governance readiness chart; sign-off cards; audit table | Audit readiness below gate or missing accountable owner. |
| Synthesis | Memo decision cards and report export | Release memo and Design Ledger snapshot | Memo incomplete until all module predictions have values. |

All decision-driving plots must have exact table fallbacks. Color is paired with
textual pass/fail labels and violated numbers.

## Evidence And Ledger Plan

The report and Design Ledger should capture:

- selected track, scenario, hardware ref, model ref, stakeholder, harmed party
- Part A threshold, target gap, FPR gap, threshold policy
- Part B epsilon, retention, raw collection, local processing, evidence
  confidence, privacy deployability
- Part C safety threshold, canary share, blast radius estimate, rollback time,
  release gate status
- Part D audit readiness, lineage coverage, decision-context logging, log
  retention, accountable owner, governance decision
- synthesis decision, residual risk, carry-forward capstone constraint

## Source And Number Ownership

- Track identity, stakeholder text, harmed party, obligation, audit signal, base
  quality, subgroup labels, latency SLOs, and target gaps come from
  `get_lab_track_variant`, `get_track_profile`, and
  `responsibility_track_profile`.
- Part A uses `mlsysbook_labs.metric_conflict`.
- Part B-D use notebook-local teaching models prefixed `v1_15_` because the
  requested privacy, blast-radius, and audit release gates are lab-specific
  pedagogy and must not create shared abstractions in this wave.
- All scenario thresholds are tied to chapter concepts and track metadata in the
  notebook comments/tables, not to external services.

## Implementation Risks

- The shared helper layer does not yet expose dedicated privacy-budget,
  blast-radius, or audit-readiness solvers. The notebook therefore uses
  local `v1_15_` teaching models with explicit Math Peek/source-model text.
- Track metadata was originally written for the "No Free Fairness" pilot. The
  revised lab must reinterpret existing fields carefully rather than editing
  shared variant data.
- Browser rendering is not required for this wave, but syntax and diff checks
  must pass from `/Users/VJ/GitHub/MLSysBook-labs`.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 2 | Pass |

Minimum gate:

- No dimension below 2.
- At least three dimensions at 3 in every module.
- Reversible failure states exist in Parts A-D.
- Synthesis returns to the chapter invariant and records a carry-forward
  capstone constraint.
