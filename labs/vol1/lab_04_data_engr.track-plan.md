# V1-04 Concept-Module Packet: Data Engineering

## Chapter Invariant

Data is infrastructure. Quality, lineage, throughput, and contracts are amount
systems that shape model behavior just as directly as model size or hardware
capacity. A data pipeline decision is valid only when its quality budget,
split boundary, flow capacity, and governance contract are all explicit.

## Reading Map

| Module | Chapter anchors | Claim used in the lab |
|---|---|---|
| Opening | Purpose; Dataset Compilation | Data is source code for ML systems, and every pipeline choice propagates into model behavior. |
| Part A | Quality through validation and monitoring; Data quality as code; Data drift detection and response; Quality debt remediation | Data quality must be measured as a budget of remaining defects, coverage loss, drift, and review capacity. |
| Part B | Dataset Compilation; Data versioning for ML reproducibility; Feature stores | Split boundaries and point-in-time correctness determine whether evaluation evidence is valid. |
| Part C | Physics of Data; The feeding problem; Data ingestion; Batch vs. streaming ingestion; Storage performance and cost | Throughput, backlog, and freshness are physical flow constraints, not implementation details. |
| Part D | Data cascades; Reliability through graceful degradation; Transformation lineage; Data debt; Remediation strategies | Contracts, lineage, and governance prevent small upstream changes from becoming downstream system debt. |
| Synthesis | Summary; Fallacies and Pitfalls | A final pipeline memo must name the binding data constraint and the residual risk that remains. |

## Concept Inventory

Accepted concepts:

- Data quality is a measurable budget, not a vague property.
- Leakage and split integrity can invalidate evidence even when metrics look good.
- Pipeline throughput and backlog are physical system constraints.
- Data contracts, lineage, and governance prevent downstream data debt.
- Data pipeline decisions must record the binding constraint and residual risk.

Rejected or deferred concepts:

- Storage taxonomy as a standalone catalog. It supports Part C but does not by itself create a decision.
- Labeling method taxonomy. Label quality appears in Part A, but the lab does not teach the full labeling chapter section.
- Full data acquisition strategy selection. Coverage and provenance are included as budget terms, not as a separate source-selection lab.
- Distributed processing framework selection. Amdahl and coordination tax support throughput reasoning, but framework choice is too broad for this lab.
- Data lakehouse and data mesh architecture survey. Governance appears through contracts and lineage rather than architecture branding.

## Track Narratives

| Track | Stakeholder | What data means | Binding constraints | Natural failure |
|---|---|---|---|---|
| iPhone | Mobile product engineer | Private local sensor and app-context events for a latency-sensitive feature pipeline. | Local privacy, radio energy, feature freshness, battery impact of collection. | Upload-heavy or over-collecting design drains battery and expands privacy scope. |
| Oura Ring | Wearable firmware engineer | PPG, temperature, accelerometer, and derived health windows gathered overnight. | Sensor quality, duty cycle, tiny flash/SRAM, BLE transfer, OTA/schema compatibility. | Raw retention or excessive validation wake time breaks nighttime battery and storage budgets. |
| RoboTaxi | Safety/perception data lead | Camera, lidar, radar, telemetry, rare-event labels, and scenario evidence. | Rare-event coverage, scenario labels, safety validation data, local triage, upload capacity. | Random sampling looks good but erases long-tail safety evidence. |
| Cloud Fleet | Platform/SRE lead | Object-store shards, feature logs, feedback events, and production features. | Ingestion throughput, feature freshness, storage/cost, contract enforcement, regional policy. | Accelerators or serving features starve while stale or contract-breaking data looks superficially healthy. |

Track deltas that must be visible in the notebook:

- Persona and system decision change by track.
- Part A quality metric changes: privacy/coverage for iPhone, sensor reliability for Oura Ring, rare-event label quality for RoboTaxi, freshness/null/duplicate budget for Cloud Fleet.
- Part B split boundary changes: device/user grouping, night/session grouping, route/scenario/time grouping, point-in-time feature retrieval.
- Part C throughput pressure changes: radio/local preprocessing, BLE/storage, sensor upload/triage, object-store/preprocessing.
- Part D governance contract changes: consent/deletion, firmware/schema compatibility, safety label lineage, contract enforcement and feature freshness.
- The final report records track-specific binding constraint and residual risk.

## Concept Modules

### Part A: Concept Module - Data Quality Is A Budget

Chapter claim:
- Robust systems validate mechanical and semantic quality. Quality debt requires allocated remediation capacity, not informal inspection.

Student prior:
- "If the schema is valid, the data is good enough."

Storyline beats:
1. Scenario: the selected stakeholder must choose how much validation and review to spend before training or serving.
2. Prediction: the student predicts which budget term will fail first: residual defects, coverage loss, review load, or no failure.
3. Manipulation: sliders adjust validation strictness and review/audit sample rate.
4. Evidence: chart and table show caught defects, residual defects, dropped records, coverage retained, and target budget.
5. Consequence: reversible failure state names the violated budget and mitigation.
6. Math Peek/source model: residual defects = base defects x (1 - detection rate); pass requires residual <= target and coverage >= floor.
7. Checkpoint: student chooses a quality gate action for the final memo.

Mechanics:
- Structured radio prediction, two sliders, budget bar chart, evidence table, failure callout, MathPeek, source trace, checkpoint radio.

Ledger output:
- quality prediction, residual defects per 10K, quality budget pass/fail, quality checkpoint.

### Part B: Concept Module - Leakage Makes Evidence Invalid

Chapter claim:
- Dataset partitions are trust boundaries. Leakage from duplicates, related entities, augmentation, or future-derived features turns validation/test metrics into memorization evidence.

Student prior:
- "A high validation metric is enough if the test set was not explicitly trained on."

Storyline beats:
1. Scenario: the stakeholder has a strong metric but must decide whether the split is valid.
2. Prediction: student predicts whether the reported metric is valid, inflated, or unusable.
3. Manipulation: controls adjust leakage pressure, split policy, and temporal gap.
4. Evidence: reported metric is compared with leakage-adjusted metric and the track-specific validity floor.
5. Consequence: failure state blocks the metric when effective leakage or temporal gap violates the boundary.
6. Math Peek/source model: reported metric = true metric + leakage inflation; point-in-time correctness requires features available at prediction time.
7. Checkpoint: student chooses whether to ship, redo the split, or block until point-in-time retrieval is fixed.

Mechanics:
- Structured radio prediction, leakage slider, split-policy dropdown, temporal gap slider, metric bar chart, split integrity table, failure callout, MathPeek, source trace, checkpoint radio.

Ledger output:
- split prediction, effective leakage percent, adjusted metric, split validity, split checkpoint.

### Part C: Concept Module - Throughput And Backlog Are Physical Constraints

Chapter claim:
- Training and serving speed are bounded by data supply. Backpressure appears when arrival rate exceeds service rate, and freshness SLAs turn backlog into model behavior.

Student prior:
- "More compute will fix a slow model pipeline."

Storyline beats:
1. Scenario: the stakeholder must keep the selected feature/training pipeline fed under bursts.
2. Prediction: student predicts the first flow failure: ingest, preprocessing, storage/write, movement, or no failure.
3. Manipulation: controls adjust traffic/sampling multiplier and worker/capacity count.
4. Evidence: utilization chart and backlog timeline show whether service capacity exceeds arrival rate.
5. Consequence: failure state names queue explosion, freshness miss, accelerator starvation, or battery/duty-cycle stress.
6. Math Peek/source model: backlog(t) = max(0, arrival - service) x t; freshness lag = backlog / service.
7. Checkpoint: student chooses the capacity or demand policy to carry into the memo.

Mechanics:
- Structured radio prediction, flow slider, worker slider, existing `evaluate_pipeline` stage utilization, local backlog model, backlog chart, table fallback, MathPeek, source trace, checkpoint radio.

Ledger output:
- throughput prediction, actual bottleneck, utilization, backlog after window, freshness lag, throughput checkpoint.

### Part D: Concept Module - Contracts Prevent Data Debt

Chapter claim:
- Data cascades and schema evolution failures are prevented by data contracts, lineage, versioning, freshness checks, and governance. Unmanaged assumptions compound as data debt.

Student prior:
- "Governance is documentation after the pipeline works."

Storyline beats:
1. Scenario: the stakeholder must approve a movement/retention strategy while upstream producers keep changing.
2. Prediction: student predicts which governance control prevents the next downstream failure.
3. Manipulation: controls select movement strategy, retention policy, contract enforcement level, network budget, dataset window, and upstream change pressure.
4. Evidence: movement frontier table, contract debt table, and contract-risk chart show cost, quality retained, privacy/governance exposure, caught violations, and silent debt.
5. Consequence: failure state names contract violation, lineage gap, or unmanaged freshness debt.
6. Math Peek/source model: debt_n = debt_0 x (1 + r)^n and enforcement reduces silent debt by catching incompatible changes early.
7. Checkpoint: student chooses the governance gate for the final memo.

Mechanics:
- Structured radio prediction, dropdowns/sliders, existing movement frontier and architecture helpers, local contract-debt model, evidence table, failure callout, MathPeek, source trace, checkpoint radio.

Ledger output:
- contract prediction, movement strategy, retention policy, contract policy, silent debt index, contract checkpoint.

### Synthesis: Pipeline Decision With Binding Constraint

Chapter invariant:
- Data is infrastructure. The final decision is not complete until the student records the binding data constraint and residual risk.

Student activity:
- The notebook assembles a local data pipeline memo from Parts A-D.
- The student chooses a final stance: proceed with constraint, redesign before launch, or collect more evidence.
- The student records a residual risk note.
- The Design Ledger saves track, scenario, quality budget, split validity, throughput/backlog, contract policy, movement strategy, retention policy, binding constraint, final stance, and residual risk.

## Mechanics And Evidence Plan

| Module | Controls | Chart/table evidence | Failure boundary | Report evidence |
|---|---|---|---|---|
| A | Quality prediction, validation strictness, review sample, checkpoint | Defect budget bar and quality ledger table | residual defects > target or coverage < floor | quality gate and remaining defect budget |
| B | Split prediction, leakage pressure, split policy, temporal gap, checkpoint | Reported vs adjusted metric bar and split integrity table | effective leakage > allowed or time gap < required | split policy and adjusted evidence validity |
| C | Throughput prediction, traffic multiplier, workers, checkpoint | Stage utilization chart, backlog timeline, flow table | arrival >= service or freshness lag > target | bottleneck, backlog, service headroom |
| D | Contract prediction, movement strategy, dataset window, network, retention, contract policy, change pressure, checkpoint | Movement frontier table, debt chart, governance table | contract debt > threshold, lineage absent, or freshness debt unmanaged | contract, retention, movement choice, accepted governance risk |
| Synthesis | Final stance, residual risk note | Report card and ledger HUD | missing required prediction/checkpoint/residual risk | data pipeline memo and JSON snapshot |

Every chart that drives a decision has a table fallback. Failure state text includes value, limit, unit, and mitigation. Color is never the only indicator because pass/fail text appears beside each metric.

## Source And Number Ownership

- Existing track profile, hardware/model refs, movement strategies, pipeline capacities, and retention options come from `mlsysbook_labs` V1-04 track variants and data pipeline helpers.
- New calculations for quality budget, leakage integrity, backlog, and contract debt remain notebook-local and use helper names prefixed with `v1_04_`.
- Scenario thresholds are track-specific teaching constants in the notebook because shared MLSysIM support does not yet expose quality-budget, leakage, or contract-debt solvers.
- The track-plan records this as an implementation risk rather than adding shared abstractions.

## Ledger Plan

Save under chapter 4:

- `track_id`, `scenario_id`, `hardware_ref`, `model_ref`
- `quality_prediction`, `quality_budget_pass`, `residual_defects_per_10k`, `quality_checkpoint`
- `split_prediction`, `effective_leakage_pct`, `adjusted_metric_pct`, `split_valid`, `split_checkpoint`
- `throughput_prediction`, `actual_bottleneck`, `utilization_pct`, `backlog_gb`, `freshness_lag_s`, `throughput_checkpoint`
- `movement_strategy`, `retention_policy`, `contract_prediction`, `contract_policy`, `silent_debt_index`, `contract_checkpoint`
- `binding_data_constraint`, `final_pipeline_stance`, `residual_risk`

Downstream use:
- Later labs can read the selected track, binding data constraint, movement strategy, retained evidence policy, and residual data risk when asking whether training, serving, and operations assumptions are trustworthy.

## Implementation Risks

- MLSysIM has no first-class quality-budget, leakage-integrity, backlog, or contract-debt result objects for this lab. Keep support notebook-local for now.
- The existing V1-04 shared helper is named `data_pipeline` and centers "Data Gravity"; the revised lab must reuse it for profile, movement, and architecture while adding concept-module calculations locally.
- Track-specific values must be visibly scenario-justified in the notebook until a typed lab variant schema exists for these concepts.
- Other workers may edit other labs, helpers, or tests. This worker owns only `labs/vol1/lab_04_data_engr.py` and this track-plan.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 2 | Pass |

Audit notes:

- No dimension is below 2.
- Every module has at least five student-facing beats and at least one manipulation.
- Part C and Part D include reversible failure states students can enter and recover from.
- Part A, Part B, and Part C each produce prediction-vs-actual or budget-vs-actual evidence.
- Synthesis ties all modules back to the invariant and saves a design decision with residual risk.
