# V1-09 Track Plan: Data Selection

## Chapter Invariant

Data quantity is not data value. A data selection decision is defensible only when
it explains marginal utility, coverage, label cost, and residual bias/quality
risk in the same amount system.

The selected track changes the persona, budgets, thresholds, evidence emphasis,
failure mode, and report framing, but it does not change the concept sequence.

## Reading Map

| Module | Chapter anchor | Claim used by the lab |
|---|---|---|
| Part A | Data Selection Fundamentals; Information-Compute Ratio; ICR Frontier | Marginal information per FLOP decays as datasets become redundant. |
| Part B | Static Pruning; Coreset Selection; Data Pruning by Quality | Coverage guardrails must protect deployment-relevant slices before average utility is trusted. |
| Part C | Dynamic Selection; Active Learning; Cost Modeling; Selection Inequality | Labels, curation, storage, and processing create a budgeted frontier, not a free quality knob. |
| Part D | Measurement Framework; Fallacies and Pitfalls | Deployment success requires stratified risk evidence, not only aggregate validation score. |
| Synthesis | Summary; Curate, do not accumulate | A data selection memo must name the selected cohort, binding budget, rejected alternatives, and carry-forward risk. |

## Concept Inventory

### Accepted Concepts

- Marginal data value saturates: extra examples can add linear cost with little
  additional learning signal.
- Coverage and diversity can dominate raw dataset size: a smaller cohort can be
  safer when it protects high-risk slices.
- Label cost and label quality create a budgeted frontier: active selection
  saves labels only if selection overhead, review cost, and quality floors hold.
- Residual bias and downstream risk must be defended: validation score is not a
  release argument unless subgroup gaps and operational harms are named.
- Data selection is upstream D-A-M co-design: it changes the work that later
  models and machines must perform.

### Rejected Or Deferred Concepts

- Full EL2N, GraNd, k-Center, or MinHash implementation: used as source-model
  references only; the lab focuses on decision logic rather than algorithm
  implementation.
- Synthetic data generation and model collapse: referenced in risk framing, but
  deferred because the required task is selection under real data budgets.
- Distributed sharding, random-access I/O, and data echoing: important chapter
  concepts, but they would create a separate systems-engineering lab.
- Chinchilla token/parameter diagnostics: summarized as a compute-optimal
  frontier idea, but not made a separate student activity.

## Shared Module Packet

| Part | Concept module | Student prior | Consequence | Evidence |
|---|---|---|---|---|
| A | Marginal data value saturates | More examples are the best next spend. | The marginal signal-per-cost curve falls while cost continues rising. | Saturation chart, current multiplier marker, policy table, checkpoint. |
| B | Coverage and diversity can dominate size | The largest cohort is the safest cohort. | The largest cohort can leave the highest-risk slice below floor. | Per-policy coverage table, subgroup bar chart, worst-slice failure callout. |
| C | Label cost/quality creates a budgeted frontier | Better labels are always worth buying. | Label, review, processing, storage, or quality floors become binding. | Budget stacked bars, frontier table, binding-budget checkpoint. |
| D | Residual bias and downstream risk must be defended | A strong validation score is enough. | Release is blocked when residual risk exceeds the track tolerance. | Risk register chart/table, validation gate, residual-risk checkpoint. |
| Synthesis | Data selection memo | A policy is complete once selected. | The memo is incomplete unless it states rejected alternatives and carry-forward risk. | Design Ledger save and downloadable report. |

## Concept Modules

### Part A: Concept Module - Marginal Data Value Saturates

- Scenario beat: the track stakeholder has one more collection/labeling window
  and asks whether to increase the selected cohort.
- Prediction beat: student predicts whether quantity, quality, coverage, or
  cost will dominate the next spend.
- Manipulation beat: student moves the dataset fraction multiplier and observes
  the same selected policy at lower and higher retained volume.
- Evidence beat: a saturation chart plots information proxy, marginal gain, and
  current cost; a table lists exact examples, cost, utility, and ICR proxy.
- Consequence beat: the notebook names when the next increment adds little
  signal or crosses a track budget.
- Math/source beat: Math Peek uses `ICR = Delta I / Delta FLOPs` and the
  chapter's `ICR(D) ~= 1 / (O_sample * D)` decay.
- Checkpoint beat: student chooses whether the next spend should expand volume
  or redirect toward a higher-value cohort.

Ledger fields: `value_prediction`, `fraction_multiplier`,
`marginal_signal_per_cost`, `selected_examples_k`.

### Part B: Concept Module - Coverage Can Beat Raw Size

- Scenario beat: the stakeholder must defend the selected cohort to a product,
  safety, health, or platform review.
- Prediction beat: student predicts which policy will best protect the
  under-covered cohort.
- Manipulation beat: student switches data policy and sees subgroup coverage
  change under the same chapter concept.
- Evidence beat: a coverage chart and policy table compare selected examples,
  coverage, rare-event score, worst subgroup, and weighted gap.
- Consequence beat: the notebook calls out the worst subgroup and whether the
  largest cohort is a false winner.
- Math/source beat: Math Peek uses the coreset coverage idea: retain examples
  that preserve deployment-relevant slices, not just average distribution mass.
- Checkpoint beat: student chooses the defensible coverage policy for the memo.

Ledger fields: `coverage_prediction`, `selected_policy`, `worst_subgroup`,
`worst_risk_score`.

### Part C: Concept Module - Label Cost And Quality Create A Frontier

- Scenario beat: the stakeholder receives a fixed labeling/curation budget and
  must choose which data is worth expert review.
- Prediction beat: student predicts the binding budget: label spend, review
  throughput, quality floor, coverage floor, rare-event floor, compute, or
  storage.
- Manipulation beat: student changes the label-budget multiplier while keeping
  the same policy candidates.
- Evidence beat: stacked bars show label/review/process cost against budget;
  a frontier table names feasibility and binding budget for every policy.
- Consequence beat: infeasible choices show the violated amount and mitigation.
- Math/source beat: Math Peek uses total data cost
  `C_total = C_acquire + C_label + C_store + C_process` and the Selection
  Inequality as the overhead gate.
- Checkpoint beat: student chooses the policy that best sits on the budgeted
  frontier.

Ledger fields: `label_prediction`, `label_budget_multiplier`,
`binding_budget`, `label_frontier_policy`.

### Part D: Concept Module - Residual Risk Must Be Defended

- Scenario beat: the validation score looks acceptable and the team wants to
  ship, but a reviewer asks what risk remains.
- Prediction beat: student predicts whether aggregate validation score,
  subgroup risk, rare-event evidence, or governance/SLA review will gate release.
- Manipulation beat: student changes residual-risk tolerance and validation
  evidence focus.
- Evidence beat: a risk register chart and table show weighted subgroup gaps,
  current tolerance, status, and mitigation.
- Consequence beat: release is blocked if residual risk exceeds tolerance even
  when aggregate utility looks strong.
- Math/source beat: Math Peek expresses the gate as
  `release_ok = aggregate_ok AND subgroup_risk_ok AND evidence_ok`.
- Checkpoint beat: student chooses ship, collect more data, or reject/redo.

Ledger fields: `validation_prediction`, `risk_tolerance`,
`validation_focus`, `release_gate`, `carry_forward_risk`.

### Synthesis: Data Selection Memo

The final artifact is a data selection memo, not a completion checkbox. It must
record:

- selected cohort or policy
- binding budget or guardrail
- rejected alternatives and why they were rejected
- worst residual subgroup or downstream risk
- next data to collect
- validation requirement carried into future labs

## Track Narratives

| Track | Persona | Amount system | Failure mode | Evidence emphasis | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile product engineer | privacy-safe on-device cohorts, UX edge cases, consent review, app storage, collection cost | broad collection violates trust/storage while missing private local contexts | private-context coverage, consented hard examples, storage audit | privacy-safe cohort memo |
| Oura Ring | Wearable firmware engineer | biosignal windows, sensor-contact quality, night/activity cohorts, battery, OTA/storage, scarce labels | continuous windows create noisy labels and burn storage/battery while physiology remains thin | contact-quality coverage, sleep/activity balance, battery regression | biosignal selection memo |
| RoboTaxi | Safety/perception engineer | rare-event clips, scenario coverage, redaction, expert labels, replay storage, fallback validation | random miles improve averages while rare hazards remain uncovered | rare-event recall, construction/weather/road-user coverage, redaction and replay | safety evidence memo |
| Cloud Fleet | Fleet service owner | traffic/query cohorts, freshness, labeling throughput, cost/request, tenants/languages/regions, SLA impact | cheap scale hides tenant or language failures and increases compute cost | subgroup reliability, freshness, quality regression, cost/request canary | platform reliability memo |

## Mechanics And Evidence Plan

| Need | Mechanic | Evidence |
|---|---|---|
| Commit prior | `mo.ui.radio` prediction in every part | prediction-vs-actual reveal card |
| Manipulate system | sliders/dropdowns for fraction, policy, label budget, risk tolerance | chart marker and table rows update through Marimo dataflow |
| Show saturation | Plotly line chart plus exact table | marginal signal, ICR proxy, cost, feasibility |
| Show coverage | Plotly subgroup bar chart plus policy table | worst subgroup, weighted risk, largest-policy comparison |
| Show frontier | Plotly stacked budget bars plus exact table | label/review/process cost, budget, binding budget |
| Show residual risk | Plotly risk register bars plus table | release gate and mitigation |
| Source model | Math Peek accordions plus `source_trace()` | chapter anchor and helper provenance |
| Ledger/report | `DesignLedger.save`, HUD, `build_lab_report`, export panel | memo fields carried forward |

## Source And Helper Plan

- Existing shared helpers remain the source of truth for profiles, variants,
  frontiers, utility, coverage, data policy decisions, track selector, source
  trace, report export, and ledger.
- Notebook-local helpers are allowed only for V1-09 concept instrumentation and
  are prefixed `v1_09_`.
- No shared helpers, tests, implementation notes, or other labs are changed.

## Implementation Risks

- The shared helper utility function is intentionally simple. The notebook adds
  a local ICR-style saturation proxy for Part A rather than changing the shared
  helper.
- Label-frontier costs are notebook-local scenario instrumentation. They are
  derived from profile costs and track-specific factors, then source-labeled as
  lab instrumentation rather than MLSysIM facts.
- Risk tolerance is a teaching control. It does not claim to be a validated
  production release threshold.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, slider, saturation chart, consequence, Math Peek, checkpoint. |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, policy manipulation, coverage chart, failure state, source, checkpoint. |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, budget slider, frontier bars, binding budget, Math Peek, checkpoint. |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Pass: prediction, risk tolerance, risk register, release gate, Math Peek, checkpoint. |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 | Pass: memo, ledger, report, rejected alternatives, carry-forward risk. |

Required reversible failure state: Part C can become infeasible by tightening
label budget, and Part D can be recovered by lowering residual risk through a
different policy or by raising the tolerance during exploration. The synthesis
ties all modules back to the invariant that data quantity is not data value.
