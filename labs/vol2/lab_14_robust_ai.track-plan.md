# V2-14 Track Plan: The Robustness Budget

## Chapter Invariant

Robustness is an amount system: a deployed model buys bounded behavior under shift
by spending finite stress coverage, hardening compute, monitoring attention, and
fallback capacity. Under-spending creates silent residual failure; over-spending
consumes latency, cost, energy, clean quality, and sustainability headroom.

## Reading Map

| Lab module | Chapter source | Claim used in the lab |
|---|---|---|
| Part A - Shift exposure and failure cost | The Silent Failure Problem; Quantitative drift detection; PSI thresholds | Average-case accuracy hides distributional stress; PSI/KL/KS-style monitors translate shift into an operational response. |
| Part B - Robustness budget allocation | Drift response framework; failsafe and uncertainty footnotes; hardening strategy footnote | Coverage, retraining, monitoring, and fallback are different spending accounts; each catches different failures. |
| Part C - Robustness tax frontier | Adversarial defenses; robustness tax example; uncertainty compute footnote | Robustness improves worst-case behavior by paying latency, compute, energy, clean-quality, and regression taxes. |
| Part D - Robustness policy gate | Defense-in-depth workflow; fallacies and pitfalls; summary | A policy must name guardrails, rejected alternatives, and residual failure rather than claiming universal robustness. |
| Synthesis - Robustness budget memo | Summary; From resilience to sustainability | The final memo carries the binding amount and residual risk into V2-15 sustainability. |

## Accepted Concept Inventory

| Concept | Why it stays | Module |
|---|---|---|
| Silent failure | It makes robustness operational: uptime can be green while model behavior is wrong. | A |
| Distribution shift measurement | PSI and threshold bands create a manipulative amount model. | A |
| Failure consequence asymmetry | Same shift score has different meaning for iPhone, Oura, RoboTaxi, and Cloud Fleet. | A |
| Budgeted defense accounts | Students must allocate a fixed amount across coverage, retraining, monitoring, and fallback. | B |
| Defense tax | The chapter explicitly says robustness is bought with accuracy, compute, energy, latency, and validation cost. | C |
| Policy gate with residual risk | Robustness is bounded to a threat model and operating envelope. | D |

## Rejected Or Compressed Concepts

| Concept | Treatment |
|---|---|
| Full adversarial attack taxonomy | Compressed into threat-model fit and Math Peek because the lab focus is budgeting, not attack mechanics. |
| Certified radius derivation | Mentioned in Math Peek/source trace; not a separate module because it would become a math-only detour. |
| Data poisoning internals | Appears as one residual failure case and source-trace item; V2-13 already establishes adversarial/security boundaries. |
| Detailed KS/chi-square statistics | PSI is the primary manipulative signal; KS/KL are referenced as escalation evidence. |
| Federated adaptation mechanics | Deferred to Edge Intelligence and privacy labs; here it is an adaptation cost note. |

## Track Plan

Tracks realize the same concepts with different personas, constraints, thresholds,
failure costs, evidence emphasis, and report framing.

| Track | Persona | Likely shift | Highest failure cost | Binding robustness amount |
|---|---|---|---|---|
| iPhone | Mobile product engineer | Lighting, acoustics, device/user context | Battery drain, privacy-sensitive wrong action, visible UX regression | Latency and battery headroom for monitors/fallback |
| Oura Ring | Wearable firmware engineer | Sensor contact, physiology, activity seasonality, firmware variance | Missed health signal, false wellness summary, battery depletion | SRAM/energy and delayed labels |
| RoboTaxi | Autonomous vehicle platform engineer | Weather, occlusion, rare objects, physical adversarial artifacts | Safety recall miss, p99 deadline miss, unsafe fallback | Rare-event coverage and deterministic fallback |
| Cloud Fleet | Fleet service owner | Tenant/user mix, abuse, prompt/data distribution, model updates | SLO breach, bad decisions at scale, rollback blast radius | Monitoring coverage, capacity, cost/request, and carbon |

## Concept Modules

### Part A - Concept Module: Shift Exposure Has A Cost

Chapter claim:
- Robustness measures bounded behavior under distribution shift, adversarial perturbation, and system faults, not held-out i.i.d. accuracy.
- PSI thresholds separate negligible, minor, moderate, and major shift bands.

Student prior:
- "If the model is accurate and the service is healthy, robustness is probably fine."

Activity beats:
1. Scenario: a track-specific owner sees green latency/uptime but worries about silent failure.
2. Prediction: choose which shift will dominate and which cost matters most.
3. Manipulation: change shift type, stress exposure, and failure-cost multiplier.
4. Evidence: PSI-style cohort shift chart plus expected harm metric.
5. Consequence: the same shift score maps to different operational decisions by track.
6. Math/source beat: PSI and expected-loss model.
7. Checkpoint: choose monitor, investigate, retrain, or fallback-first response.

Ledger output:
- Track, selected shift, predicted failure mode, PSI, response tier, failure cost.

### Part B - Concept Module: Robustness Budget Is Allocated, Not Added

Chapter claim:
- Hardening, drift monitoring, retraining, uncertainty, and fallback each spend resources and catch different failure modes.

Student prior:
- "Spend more on the strongest defense and the system becomes robust."

Activity beats:
1. Scenario: the same owner has 100 robustness points to allocate.
2. Prediction: choose which account will reduce residual failure most.
3. Manipulation: allocate points across stress coverage, retraining, monitoring, and fallback.
4. Evidence: budget bar and residual-risk decomposition.
5. Consequence: over-budget or under-funded accounts expose a named residual failure.
6. Math/source beat: diminishing-return risk reduction and fixed budget.
7. Checkpoint: choose the account to protect before spending more elsewhere.

Ledger output:
- Budget allocation, over/under budget, residual risk, underfunded account, binding account.

### Part C - Concept Module: Robustness Has A Tax Frontier

Chapter claim:
- Adversarial training, certification, uncertainty sampling, ensembles, and guardrails improve worst-case behavior while taxing clean quality, latency, cost, energy, and regression risk.

Student prior:
- "The more robust option is always the better engineering option."

Activity beats:
1. Scenario: release review compares hardening strategies.
2. Prediction: choose which tax will bind first.
3. Manipulation: choose defense family, strength, and uncertainty samples.
4. Evidence: robustness-gain versus tax frontier and candidate table.
5. Consequence: a design can improve stress behavior while failing latency, energy, cost, or clean-quality guardrails.
6. Math/source beat: robust objective and robustness-tax formulas.
7. Checkpoint: choose whether to harden, monitor, fallback, or defer.

Ledger output:
- Selected defense, strength, uncertainty samples, robustness gain, tax terms, binding tax.

### Part D - Concept Module: A Robustness Policy Has Guardrails And Residual Failure

Chapter claim:
- Robustness is threat-model-bound; robust systems combine detection, defense, fallback, and regular testing while naming residual risks.

Student prior:
- "A policy can be declared robust if it passes one stress test."

Activity beats:
1. Scenario: governance review asks for one shippable policy.
2. Prediction: choose which policy candidate survives all guardrails.
3. Manipulation: select a policy, guardrail strictness, and residual failure case.
4. Evidence: policy gate table with latency, cost, quality, coverage, fallback, and residual-risk status.
5. Consequence: rejected alternatives fail for a specific amount, not a generic preference.
6. Math/source beat: feasibility as a conjunction of guardrails.
7. Checkpoint: choose the final policy objective.

Ledger output:
- Selected policy, guardrail strictness, rejected alternative, residual failure case, binding amount.

## Mechanics Plan

| Module | Controls | Evidence | Failure state |
|---|---|---|---|
| A | Shift prediction radios, shift dropdown, stress exposure slider, failure-cost slider | PSI cohort chart, source table, expected harm cards | Major shift or high expected harm warning |
| B | Budget prediction radio, four allocation sliders, checkpoint | Stacked budget chart, residual-risk decomposition table | Over-budget and underfunded account warning |
| C | Tax prediction radio, defense dropdown, strength slider, uncertainty sample slider | Frontier scatter/bar chart, candidate table | Tax guardrail violation for latency/cost/energy/quality |
| D | Policy prediction radio, policy dropdown, guardrail strictness slider, residual-case dropdown | Policy gate table, decision memo | Feasibility conjunction fails |
| Synthesis | Local decision text plus export panel | Robustness budget memo | Missing predictions/checkpoints marked incomplete |

## Evidence Plan

The report must contain:
- Selected track and scenario.
- Part A shift exposure, PSI tier, and failure-cost evidence.
- Part B budget allocation and residual-risk evidence.
- Part C tax frontier evidence and binding tax.
- Part D selected policy, guardrails, rejected alternative, and residual failure case.
- V2-15 implication: extra robustness compute/monitoring/fallback becomes an energy and carbon budget input.

## Source Trace

Notebook-local helpers are prefixed `v2_14_` because shared MLSysIM support does
not yet expose typed robustness-budget result objects. The source trace records:

- `book/quarto/contents/vol2/robust_ai/robust_ai.qmd`
- `v2_14_shift_exposure`
- `v2_14_budget_result`
- `v2_14_defense_result`
- `v2_14_policy_result`
- Track profile and lab variant refs from `mlsysbook_labs`
- Chapter formulas: PSI, robust minimax objective, feasibility conjunction

## Implementation Risks

| Risk | Mitigation |
|---|---|
| No shared robustness solver exists yet. | Use notebook-local helpers with explicit source trace and keep them prefixed `v2_14_`. |
| Track thresholds are scenario constants rather than registry values. | Tie them to existing track profiles and variant guardrails; document them in Math Peek/source trace. |
| Four budget sliders can exceed 100 points. | Make over-budget a reversible failure state and keep the report evidence explicit. |
| Robustness-tax coefficients are teaching proxies. | Label them as proxy costs anchored to chapter claims about robustness tax and UQ compute. |
| Other workers may edit adjacent labs. | Only edit `lab_14_robust_ai.py` and this track-plan. |

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Pass notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Prediction, manipulation, PSI evidence, cost consequence, Math Peek, checkpoint. |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Fixed-budget allocation with reversible over-budget and underfunded-account failures. |
| Part C | 3 | 3 | 3 | 3 | 3 | 3 | Defense frontier exposes robustness gain versus tax guardrails. |
| Part D | 3 | 3 | 3 | 3 | 3 | 3 | Policy gate requires guardrails, rejected alternative, residual failure, and report framing. |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 3 | Memo carries selected policy, binding amount, residual risk, and V2-15 implication. |

Minimum gates:
- Every module has at least five activity beats.
- At least one reversible failure exists in Parts A-D.
- The same concept sequence is shared across tracks.
- Track differences change constraints, thresholds, failure cost, and report wording.
