# V1-03 Track Plan: Constraint Tax

## Chapter Invariant

Deployment constraints propagate backward through the ML workflow. If a team
discovers the deployment constraint late, the fix does not stay at deployment:
it reopens data assumptions, model choices, validation evidence, release
readiness, and monitoring thresholds. Late discovery therefore creates rework,
risk, and weaker evidence.

## Reading Map

| Lab module | Chapter anchor | Claim used in the lab |
|---|---|---|
| Opening | ML Lifecycle; Lifecycle Stages | The workflow is a loop of Data-Algorithm-Machine decisions, not a linear checklist. |
| Part A | Six core lifecycle stages; Stage Interface Specification | Each stage has contracts whose outputs harden assumptions for later stages. |
| Part B | The iteration tax; Constraint Propagation Principle | Late discoveries compound because each downstream stage built on the wrong assumption. |
| Part C | Evaluation and Validation; Offline and online evaluation; Deployment readiness | Validation gates trade slower iteration for stronger evidence under production conditions. |
| Part D | Systems Thinking; Fallacies and Pitfalls | A release policy is a design decision about when evidence can block the system. |
| Synthesis | Summary and Takeaways | Workflow carries constraints through time and must name the residual blind spot. |

## Concept Inventory

Accepted concepts:

| Concept | Reason accepted | Module |
|---|---|---|
| Constraint propagation across lifecycle stages | It is the chapter invariant and creates immediate consequence for every track. | A |
| Iteration tax from late discovery | It turns the propagation principle into a measurable cost. | B |
| Evaluation gate confidence-speed trade-off | It explains why teams cannot only optimize for fast iteration. | C |
| Workflow policy as system design | It turns observations into a durable release and rollback decision. | D |
| Residual blind spot | It prevents the policy from pretending that evidence eliminates all risk. | Synthesis |

Rejected or deferred concepts:

| Concept | Reason rejected for this lab |
|---|---|
| Full CRISP-DM history | Important context but not a decision with a manipulable consequence here. |
| Data scientist time-allocation survey | Useful motivation, but it does not drive the selected track policy. |
| Complete DR screening case study | The lab uses the chapter's principle but keeps the track storyline aligned to the student's selected system. |
| Data versioning tool catalog | Deferred to Data Engineering and ML Ops labs. |
| Detailed regulatory validation | Included only as a gate/blind-spot idea where the selected track needs it. |
| Full iron-law derivation | Used as a Math Peek source model, not expanded into a separate module. |

## Concept Modules

### Part A - Concept Module: Constraints Propagate Through The Workflow

- Chapter claim: deployment constraints reshape data, model, validation,
  release, and monitoring decisions.
- Student prior: "We can validate the deployment constraint near release."
- Productive failure: the selected track's constraint is moved late and the
  table shows every hardened stage that now has to be revisited.
- Scenario beat: the track stakeholder must decide where to first test the
  deployment constraint.
- Prediction beat: predict whether the constraint should be tested before data
  and model assumptions harden, during model design, during release, or after
  launch.
- Manipulation beat: move the discovery stage across the six track-specific
  workflow stages.
- Evidence beat: stage table shows which assumptions harden and which gate
  should block progress.
- Consequence beat: a late stage marks a reversible failure state because the
  workflow has built artifacts on an invalid assumption.
- Math/source beat: Math Peek ties the stage map to the iron law of workflow:
  deployment limits on latency, hardware, or cost reshape data volume, model
  operations, and efficiency.
- Checkpoint beat: choose the earliest gate that should block the track.
- Ledger output: selected track, constraint name, predicted gate timing,
  discovery stage.

### Part B - Concept Module: Late Discovery Creates A Measurable Iteration Tax

- Chapter claim: correction cost grows roughly as `base * 2^(stage - 1)` when a
  constraint is found later in the lifecycle.
- Student prior: "Late validation is annoying but mostly a schedule slip."
- Productive failure: the bar chart shows stage 5 or stage 6 discovery costing
  many times the recommended gate.
- Scenario beat: the team asks whether it can absorb the late discovery or must
  change workflow.
- Prediction beat: predict the cost shape: constant, linear, exponential, or
  mostly documentation overhead.
- Manipulation beat: slide the discovery stage and compare current rework to
  the recommended gate.
- Evidence beat: rework chart and table show multiplier, person-days, avoidable
  rework, and artifacts to rebuild.
- Consequence beat: if discovery is after the recommended gate, the failure
  banner names the avoidable rework and affected artifacts.
- Math/source beat: Math Peek uses the constraint propagation cost formula and
  the chapter's stage-5 16x / stage-6 32x framing.
- Checkpoint beat: decide whether to pay the tax, move the gate earlier, or cut
  scope until the constraint is known.
- Ledger output: cost multiplier, avoidable rework days, artifacts to rebuild.

### Part C - Concept Module: Evaluation Gates Trade Speed For Confidence

- Chapter claim: validation must test the full deployment constraint surface,
  but realism, data scale, and gate depth make each iteration slower.
- Student prior: "The fastest workflow is best as long as model metrics look
  good."
- Productive failure: shallow gates keep iteration time low but leave residual
  deployment risk above the derived risk budget.
- Scenario beat: the stakeholder must set a validation mix before feature
  freeze, OTA packaging, road-test expansion, or launch.
- Prediction beat: predict the current bottleneck among validation depth,
  automation, hardware realism, and data scale.
- Manipulation beat: tune validation depth, automation, hardware realism, and
  data-scale coverage.
- Evidence beat: frontier chart plots iteration days against residual risk,
  with a current point and a budget boundary.
- Consequence beat: the notebook distinguishes "fast but blind" from "slow but
  confident" and names the bottleneck dimension.
- Math/source beat: Math Peek exposes the source model used by
  `iteration_frontier`: confidence rises with validation depth, realism, data
  scale, and automation; residual risk falls toward a floor.
- Checkpoint beat: choose the evaluation stance: ship faster, add realism, add
  automation, or reduce scope.
- Ledger output: knob settings, iteration days, confidence, residual risk,
  bottleneck.

### Part D - Concept Module: Workflow Policy Is System Design

- Chapter claim: a workflow policy is the system-level rule for when evidence
  can block release; it is not project paperwork.
- Student prior: "The policy records what the team already decided."
- Productive failure: the gate comparison table shows that a late gate saves
  early effort but leaves the highest rework and weakest evidence.
- Scenario beat: the stakeholder must write the release rule the team will use
  for the selected track.
- Prediction beat: predict whether the non-negotiable policy gate should be the
  early contract gate, a mid-workflow validation gate, or the release gate.
- Manipulation beat: select the workflow gate, release policy, and rollback
  rule.
- Evidence beat: policy table compares gate stage, rework at gate, validation
  focus, residual risk, release policy, and rollback rule.
- Consequence beat: selecting a late gate triggers a boundary warning because
  the policy lets weak evidence survive too long.
- Math/source beat: Math Peek treats policy as a system tuple:
  `gate timing + evidence requirement + rollout + rollback + blind spot`.
- Checkpoint beat: decide the policy and name the residual blind spot.
- Ledger output: selected gate, release policy, rollback rule, policy summary,
  blind spot.

### Synthesis - Concept Module: Release Memo With Residual Blind Spot

- Chapter claim: workflow carries constraints through time.
- Student prior: "A passed gate means the system is safe."
- Student action: record a release memo for the selected track that includes the
  release policy, evidence numbers, and the residual blind spot.
- Evidence: final report and Design Ledger snapshot include track, constraint,
  discovery stage, avoidable rework, iteration days, residual risk, policy, and
  blind spot.
- Carry-forward: later labs can read the selected track and workflow gate as
  assumptions for data, compression, serving, and operations decisions.

## Track Narratives

| Track | Stakeholder | Coherent storyline | Expected gate |
|---|---|---|---|
| iPhone | Mobile product engineer | A mobile feature clears simulator tests, then fails thermal soak, battery drain, and privacy review after feature freeze. | Thermal/battery profiling and privacy checks before feature freeze. |
| Oura Ring | Wearable firmware engineer | A classifier works offline, but the signal window, firmware image, OTA payload, and battery duty cycle conflict after assumptions harden. | SRAM, flash, OTA, and battery checks before data/model assumptions harden. |
| RoboTaxi | Autonomous vehicle platform engineer | Average perception metrics pass, but p99 latency and rare construction-zone scenarios fail before road-test expansion. | p99, rare-event replay, and safety signoff before road-test expansion. |
| Cloud Fleet | Fleet service owner | Offline quality is ready, but staging exposes p99 SLA misses, cost/request overrun, and bad utilization after launch commitments. | Load, cost, utilization, and SLO gates before launch. |

Track specificity requirements:

- Persona changes in every track.
- Stage names come from the V1-03 track profile.
- Constraint name, failure story, gate options, release policies, rollback
  rules, and blind spots come from the V1-03 variant metadata.
- Metrics and evidence names are track-specific through `primary_metric`,
  `guardrail_metric`, and workflow profile fields.

## Mechanics Plan

| Module | Student action | Mechanics | Why this mechanic fits |
|---|---|---|---|
| Opening | Select track and read the workflow brief | Track selector, context cards, workflow map | Establishes persona and selected deployment constraint. |
| A | Predict and move where the constraint appears | Radio prediction, discovery-stage slider, stage table | Makes propagation visible across lifecycle stages. |
| B | Measure late-discovery tax | Radio prediction, discovery-stage slider, bar chart, artifacts table | Forces boundary finding around the recommended gate. |
| C | Tune confidence versus speed | Radio prediction, four sliders, frontier chart, risk budget badge | Makes evaluation realism an explicit trade-off. |
| D | Choose policy | Radio prediction, gate/release/rollback dropdowns, comparison table | Turns evidence into a system rule. |
| Synthesis | Save memo | Text area, report card, Design Ledger save, export panel | Produces carry-forward evidence. |

Failure states:

- A: discovery after the recommended gate warns that assumptions already hardened.
- B: late discovery shows avoidable rework and artifacts to rebuild.
- C: residual risk above the derived budget marks the gate as fast but blind.
- D: selected gate after the recommended stage marks the policy as late-evidence
  debt.

## Evidence And Ledger Plan

Evidence produced:

- Prediction-vs-actual feedback for gate timing, cost shape, validation
  bottleneck, and policy gate.
- Stage table showing propagation through data, model, validation, release, and
  monitoring.
- Rework chart/table showing multiplier and avoidable person-days.
- Frontier chart/table showing iteration days, confidence, residual risk, and
  bottleneck.
- Gate policy table showing validation focus and blind spot for each option.
- Final release memo with residual blind spot.

Ledger fields:

- `track_id`, `scenario_id`, `hardware_ref`, `model_ref`
- `gate_prediction`, `tax_prediction`, `frontier_prediction`,
  `policy_prediction`
- `constraint_name`, `discovery_stage`, `selected_gate_id`
- `avoidable_rework_days`, `iteration_days`, `confidence_pct`,
  `residual_risk_pct`, `risk_budget_pct`
- `release_policy`, `rollback_rule`, `policy_summary`, `blind_spot`

## Implementation Notes

- Use existing helpers: `workflow_track_profile`, `constraint_tax`,
  `iteration_frontier`, and `workflow_policy`.
- Keep new support notebook-local and prefix helper names with `v1_03_`.
- Preserve WASM bootstrap, track selector, source trace, report export, and
  Design Ledger save patterns.
- Do not add shared MLSysIM or `mlsysbook_labs` abstractions in this wave.
- Use profile/variant metadata for track facts. Notebook-local constants are
  limited to presentation logic and derived risk-budget display.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 2 | 3 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Synthesis | 3 | 2 | 3 | 2 | 3 | 3 | Pass |

Depth gate checklist:

- Each A-D module has scenario, structured prediction, manipulation, evidence,
  consequence, Math Peek/source model, and checkpoint.
- At least one reversible failure state exists in every module.
- Synthesis ties all modules back to the chapter invariant.
- No dimension falls below 2.
- At least three dimensions score 3 for every A-D module.
