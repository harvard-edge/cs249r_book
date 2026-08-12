# V2-13 Track Plan: The Price of Privacy

## Chapter Invariant

Security and privacy are amount systems. A deployment is not secure because a
control exists; it is secure enough only when the threat surface, privacy budget,
control overhead, access/deletion lineage, audit evidence, and residual risk all
fit inside the selected track's operating envelope.

Tracks are lenses, not different concepts. Every student works through the same
Part A/B/C/D sequence:

1. Threat and privacy surface become measurable budgets.
2. Stronger controls trade risk reduction against latency, utility, and
   governance overhead.
3. Access, retention, deletion, and audit lineage determine residual exposure.
4. A deployable policy must satisfy multiple guardrails and reject an attractive
   but invalid alternative.

## Reading Map

| Lab module | Chapter anchors | Claim used in the lab |
|---|---|---|
| Opening | `#sec-security-privacy-security-privacy-ml-systems-0b1e`, `#sec-security-privacy-security-vs-privacy-e0b8` | Security and privacy change the fleet's coordination, communication, and execution contracts. |
| Part A | `#sec-security-privacy-systematic-threat-analysis-risk-assessment-3ef1`, `#sec-security-privacy-attack-surface-analysis-b10b`, `#sec-security-privacy-threat-prioritization-framework-f2d5` | A useful threat model names asset, boundary, adversary, and control; the binding amount depends on exposed surface, privacy spend, and evidence gaps. |
| Part B | `#sec-security-privacy-model-extraction-defenses-8f3b`, `#nbk-security-privacy-tax-trusted-compute`, `#tbl-privacy-technique-comparison` | Controls reduce leakage and extraction value, but they spend latency, utility, compute, and governance budget. |
| Part C | `#sec-security-privacy-practical-roadmap-8f3a`, HIPAA/GDPR footnotes, access/audit/provenance discussion | Access control, retention, deletion, lineage, and audit evidence are deployment obligations, not documentation afterthoughts. |
| Part D | `#sec-security-privacy-dp-decision-framework-c4a8`, `#sec-security-privacy-fallacies-pitfalls-0c20`, `#tbl-security-privacy-maturity-model` | A local mechanism is not a system guarantee; policy feasibility is a conjunction across privacy, latency, utility, evidence, deletion, and residual-risk guardrails. |
| Synthesis | `#sec-security-privacy-summary-831c`, chapter connection to Robust AI | Security/privacy policy carries residual adversarial and information-leakage risk into V2-14 robustness. |

## Concept Inventory

### Accepted Concepts

| Concept | Why accepted | Module |
|---|---|---|
| Threat model as amount map | Turns asset, boundary, adversary, and control into measurable surface and privacy-spend ratios. | Part A |
| Privacy/control overhead frontier | Makes the price of DP, output limiting, secure aggregation, TEE, and FHE visible in latency, utility, and governance units. | Part B |
| Access/deletion lineage | Connects access control, retention, deletion SLA, audit evidence, and provenance to residual exposure. | Part C |
| Multi-guardrail deployment policy | Forces students to reject broad access or strict isolation when any guardrail fails. | Part D |
| Robustness carry-forward | Security/privacy constrains what robustness evidence can be collected and what adversarial risks remain. | Synthesis |

### Rejected Or Deferred Concepts

| Concept | Reason deferred or rejected |
|---|---|
| Full legal compliance workflow | Important, but the lab should teach system amounts rather than jurisdiction-specific legal process. |
| Detailed cryptographic protocol implementation | The chapter uses cryptography to show control cost; implementing protocols would displace the amount-system goal. |
| Model watermarking and ownership proof | Relevant to model theft, but less central than threat surface, privacy budget, and deployment policy. |
| Hardware side-channel taxonomy | Included as track context and source trace, but a full hardware security lab would need different instruments. |
| Complete machine unlearning algorithms | The lab reasons about deletion lineage and retraining obligation without implementing unlearning. |

## Track Narratives

The shared concepts stay fixed. Track selection changes persona, constraints,
thresholds, evidence emphasis, failure mode, and report framing.

| Track | Persona | Sensitive asset and boundary | Binding constraints | Natural failure | Report framing |
|---|---|---|---|---|---|
| iPhone | Mobile privacy product lead | On-device personalization data, opt-in telemetry, app/model update path | Local latency, battery, consent scope, deletion SLA, privacy-safe evidence | Verbose telemetry improves debugging but violates consent and deletion expectations | Mobile privacy release memo |
| Oura Ring | Wearable firmware/privacy lead | Biosignal windows, BLE sync, phone/cloud handoff, firmware OTA | Battery, sync payload, small cohort utility, retention, health-adjacent audit | Strong DP on a small cohort collapses utility, while broad sync keeps too much biosignal data | Wearable health-data privacy memo |
| RoboTaxi | Autonomous fleet safety/security lead | Sensor logs, location traces, safety incidents, geofence upload | P99 safety latency, rare-event evidence, board review, retention/deletion lineage | Deleting evidence too soon preserves privacy but weakens incident replay and robustness work | Safety data security memo |
| Cloud Fleet | Platform SRE/security owner | Tenant prompts, fine-tune data, logs, model registry, shared accelerators | Tenant isolation, p95 latency, cost/request, query monitoring, audit evidence | Strict isolation can pass privacy while missing latency/cost, and broad logs can leak tenants | Multi-tenant privacy/security memo |

## Concept Modules

### Part A: Concept Module - Threat Surface And Privacy Budget Become Binding Amounts

**Chapter claim:** Threat modeling must specify asset, boundary, adversary, and
control; ML systems expose data, model, interface, and infrastructure surfaces
whose risks grow with lifecycle reach and distributed communication.

**Student prior:** "Security/privacy risk is mainly about choosing the right
control." Productive failure: the same control can leave a privacy-budget breach
or audit-evidence gap as the binding amount.

**Activity beats:**

1. Scenario: the selected track's owner must approve which sensitive path can be used.
2. Prediction: choose whether threat surface, privacy budget, or evidence lineage binds first.
3. Manipulation: adjust distributed nodes, sensitive paths, privacy-consuming accesses, and logging scope.
4. Evidence: budget bar/table show surface ratio, epsilon spend, and evidence gap.
5. Consequence: reversible failure state names the amount over budget and how to recover.
6. Math Peek/source model: threat surface is lifecycle paths plus superlinear communication channels; privacy spend composes as summed epsilon; evidence gap is the audit floor minus logging evidence.
7. Checkpoint/report decision: record the binding amount and the first mitigation.

**Mechanics:** Structured prediction, sliders/dropdown, ratio bar chart, exact
table fallback, failure callout, Math Peek card.

**Ledger output:** `partA_binding_amount`, `partA_surface_index`,
`partA_privacy_epsilon_spend`, `partA_evidence_score`.

### Part B: Concept Module - Control Strength Spends Latency, Utility, And Governance Budget

**Chapter claim:** Output limiting, rate controls, DP, secure aggregation,
TEEs, and FHE reduce leakage or attacker economics, but each carries measurable
cost in latency, accuracy/utility, compute, memory, or operational evidence.

**Student prior:** "More privacy/security is always safer." Productive failure:
a stronger control can make the deployment invalid by exceeding latency,
utility-loss, or governance overhead.

**Activity beats:**

1. Scenario: the owner has to harden the service without breaking the track's product or safety envelope.
2. Prediction: choose which overhead amount binds when controls are strengthened.
3. Manipulation: tune control strength, compute isolation, output exposure, and aggregation/privacy mode.
4. Evidence: overhead frontier compares latency ratio, utility loss ratio, governance ratio, and protection score.
5. Consequence: boundary state distinguishes a viable control stack from a stack that is too slow, too inaccurate, or too hard to govern.
6. Math Peek/source model: latency adds protocol overhead; DP/noise and output limiting spend utility; governance evidence grows with control complexity.
7. Checkpoint/report decision: choose the control family worth carrying into policy.

**Mechanics:** Structured prediction, sliders/dropdowns, overhead bar chart,
table fallback, failure callout, Math Peek card.

**Ledger output:** `partB_control_stack`, `partB_latency_ms`,
`partB_utility_loss_pp`, `partB_governance_items`, `partB_binding_overhead`.

### Part C: Concept Module - Access, Retention, Deletion, And Audit Lineage Determine Residual Exposure

**Chapter claim:** Least privilege, encrypted transport, audit logging,
retention policy, reproducible control checks, and provenance evidence are the
governance boundary that lets a system prove what happened after deployment.

**Student prior:** "If access is restricted, privacy is handled." Productive
failure: broad retention, slow deletion, or weak lineage can dominate residual
exposure even with stricter access roles.

**Activity beats:**

1. Scenario: a data subject or auditor asks which sensitive records, logs, and checkpoints still carry their influence.
2. Prediction: choose whether access roles, retained record-days, deletion SLA, or audit evidence will dominate residual exposure.
3. Manipulation: set access model, retention days, deletion window, lineage coverage, and audit sampling.
4. Evidence: lineage table shows access ratio, retention ratio, deletion ratio, audit score, and residual exposure.
5. Consequence: failure state names whether exposure, deletion, or evidence is outside the track guardrail.
6. Math Peek/source model: residual exposure is a weighted amount across access breadth, retained record-days, deletion delay, and audit gap.
7. Checkpoint/report decision: choose the lineage control needed before deployment.

**Mechanics:** Structured prediction, dropdown/sliders, lineage ratio chart,
exact table fallback, failure callout, Math Peek card.

**Ledger output:** `partC_access_model`, `partC_retention_days`,
`partC_deletion_window_days`, `partC_audit_score`, `partC_residual_exposure`.

### Part D: Concept Module - Deployment Policy Is A Multi-Guardrail Conjunction

**Chapter claim:** Security/privacy maturity is cumulative. A local mechanism
does not guarantee a system; access, privacy accounting, model integrity,
adversarial monitoring, and governance evidence must jointly pass.

**Student prior:** "Choose the policy with either the strongest privacy or the
lowest overhead." Productive failure: broad access fails risk/evidence, while
strict isolation may fail latency or utility.

**Activity beats:**

1. Scenario: approve one security/privacy policy for the selected track's next deployment.
2. Prediction: choose which guardrail rejects the naive broad-access policy.
3. Manipulation: compare broad access, privacy-preserving control, strict isolation, and a custom policy from Parts A-C.
4. Evidence: policy table marks privacy budget, latency, utility, evidence, deletion, and residual-risk pass/fail states.
5. Consequence: failure callout names every violated guardrail for the selected policy.
6. Math Peek/source model: policy feasibility is a logical conjunction, not a weighted average.
7. Checkpoint/report decision: choose final policy and rejected alternative.

**Mechanics:** Structured prediction, policy dropdowns, guardrail table, grouped
bar chart, failure state, Math Peek card.

**Ledger output:** `partD_selected_policy`, `partD_rejected_policy`,
`partD_binding_guardrail`, `partD_policy_feasible`.

### Synthesis: Security/Privacy Memo

**Student task:** Produce a concise memo containing:

1. Selected security/privacy policy.
2. Binding amount from the sequence.
3. Residual risk and why it remains.
4. Rejected alternative and guardrail failure.
5. V2-14 robustness implication.

**Required carry-forward:** The V2-14 implication must connect privacy/security
to robustness. Examples: telemetry minimization reduces drift evidence; output
limiting changes attack detection; retention/deletion policy affects incident
replay; adversarial query monitoring becomes a robustness stress signal.

## Mechanics Plan

| Belt | Mechanics | Why used |
|---|---|---|
| Opening | Header, track selector, track context, reading map, source trace | Frames the invariant and keeps track as a lens. |
| Prediction | `mo.ui.radio` for Parts A-D | Forces a prior before evidence appears. |
| Manipulation | Sliders/dropdowns with 1-5 controls per module | Lets students search for boundaries and compare controls. |
| Evidence | Plotly ratio bars, policy table, exact dataframe fallbacks | Shows amount-system consequences numerically. |
| Failure | `mo.callout` danger/success with value, limit, unit, and mitigation | Makes constraints reversible and not color-only. |
| Source | Math Peek cards and shared `source_trace` | Connects formulas to chapter anchors and local teaching estimates. |
| Decision | Checkpoint radios and final memo controls | Converts observation into a deployment policy. |
| Ledger | `DesignLedger.save`, HUD, `build_lab_report`, `report_export_panel` | Carries selected policy and residual risk into V2-14. |

## Evidence And Ledger Plan

Every plot has an exact table fallback. The final report and ledger snapshot
record:

- selected track and scenario lens
- Part A predicted and actual binding amount, surface index, privacy epsilon
  spend, and evidence score
- Part B control stack, protection score, latency, utility loss, governance
  overhead, and binding overhead
- Part C access model, retention days, deletion window, audit score, and
  residual exposure
- Part D selected policy, rejected alternative, pass/fail guardrails, and
  binding guardrail
- synthesis memo fields: selected policy, binding amount, residual risk, and
  V2-14 robustness implication

## Source And Amount Model

Notebook-local formulas are acceptable because no shared MLSysIM solver currently
models this exact security/privacy sequence. Every helper uses the `v2_13_`
prefix and remains in `lab_13_security_privacy.py`.

Track identity, metadata, report export, source trace, and ledger persistence use
existing `mlsysbook_labs` and `mlsysim.labs.state` APIs. The notebook-local
amounts are teaching estimates tied to chapter claims:

- threat surface combines lifecycle-sensitive paths with distributed
  communication channels
- privacy budget composes as summed epsilon across accesses
- control overhead combines latency, utility loss, and governance evidence
- residual exposure combines access breadth, retention, deletion delay, and
  audit gaps
- deployment policy is feasible only when every guardrail predicate passes

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability |
|---|---:|---:|---:|---:|---:|---:|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 |
| Synthesis | 3 | 3 | 3 | 3 | 3 | 3 |

Minimum gates passed: every module has 5+ beats, every module has a prediction
and manipulation, Part A/B/C/D include reversible boundary/failure states, and
synthesis ties the selected policy to the chapter invariant and V2-14.
