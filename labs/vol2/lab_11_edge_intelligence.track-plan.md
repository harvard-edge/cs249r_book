# V2-11 Edge Intelligence Concept Module Packet

## Chapter Invariant

Moving intelligence to the edge moves constraints outward: device budgets,
federated updates, privacy and energy policy, and intermittent connectivity
interact as one amount system rather than four independent checkboxes.

The lab uses one shared Part A/B/C/D concept sequence for every track. Tracks do
not introduce different concepts; the selected track changes persona,
thresholds, binding amounts, evidence emphasis, failure mode, and memo framing.

## Reading Map

| Lab module | Chapter anchor | Claim used in the lab |
|---|---|---|
| Opening | `sec-edge-intelligence-distributed-learning-paradigm-shift-883d` | Edge intelligence places inference, adaptation, and coordination near data under power, memory, privacy, and connectivity constraints. |
| Part A | `sec-edge-intelligence-design-constraints-c776`; `sec-edge-intelligence-quantifying-training-overhead-edge-devices-3e4c` | On-device learning multiplies inference resource needs through gradients, optimizer state, activations, bandwidth, and energy. |
| Part B | `sec-edge-intelligence-federated-learning-6e7e`; `sec-edge-intelligence-learning-protocols-139a`; `sec-edge-intelligence-federated-privacy-a1ed` | FedAvg trades local epochs, communication rounds, non-IID drift, update compression, and privacy mechanisms. |
| Part C | `sec-edge-intelligence-client-scheduling-f675`; `sec-edge-intelligence-largescale-device-orchestration-1360` | Intermittent eligibility, stragglers, duty cycle, and connectivity decide which evidence reaches the fleet. |
| Part D | `sec-edge-intelligence-pillar-integration-7e21`; `sec-edge-intelligence-production-integration-beb5`; `sec-edge-intelligence-resource-management-691a`; `sec-edge-intelligence-production-deployment-risk-assessment-db49` | A deployable edge policy must satisfy resource, privacy, update, and quality guardrails together. |
| Synthesis | `sec-edge-intelligence-summary-0af9`; `Sec-ops-scale` connection | The V2-12 implication is operational: sustaining a heterogeneous edge fleet requires monitoring, rollback, and lifecycle controls. |

## Concept Inventory

### Accepted Concepts

- Edge feasibility is an amount-system problem: memory, energy, latency, privacy,
  communication, and quality must all remain inside their envelopes.
- On-device limits bound what can run or learn locally; inference feasibility
  does not imply adaptation feasibility.
- Federated updates keep raw data local but trade update size, local epochs,
  staleness, privacy protection, and convergence.
- Duty cycle and connectivity turn intermittent evidence into a first-order
  design input.
- Production policy must integrate adaptation, data efficiency, federation,
  validation, rollback, and compliance guardrails.

### Rejected Concepts

- A standalone placement taxonomy. It is too static for a lab and does not force
  an amount-system decision.
- LoRA storage as a separate concept module. It remains useful evidence in Part
  D, but the chapter-level burden for this lab is broader than adapter storage.
- A pure energy-drain module. Energy is retained as a guardrail and policy
  amount, not isolated as a separate concept from duty cycle and feasibility.
- A cloud-versus-edge debate. The lab rejects alternatives through guardrails,
  not by treating cloud and edge as different chapters.
- Privacy as a slogan. Privacy is represented as a guardrail with differential
  privacy cost and secure aggregation participation requirements.

## Shared Concept Modules

| Part | Concept module | Student decision | Binding amount surfaced |
|---|---|---|---|
| A | On-device limits bound local learning | Choose whether the current model can adapt locally or must shrink/defer. | Active memory headroom and energy margin. |
| B | Federated updates trade communication, staleness, privacy, and convergence | Tune local epochs and compression, then decide if federation is still viable. | Round count, uploaded bytes, stale-update penalty, privacy overhead. |
| C | Intermittent evidence is first-order | Set duty cycle and connectivity assumptions, then judge evidence freshness. | Eligible update windows and evidence age. |
| D | Edge deployment policy is a guardrail bundle | Select a policy that satisfies memory, energy, privacy, update, and quality thresholds. | First failed guardrail or remaining slack. |
| Synthesis | Edge intelligence memo | State the selected policy, binding edge amount, rejected alternative, and V2-12 operations implication. | Ledger-ready edge/federated policy. |

## Module Details

### Part A - Concept Module: On-Device Limits Bound Local Learning

- Chapter claim: training amplifies inference resource needs by 4-12x and can
  exceed the edge device memory/energy envelope even when inference fits.
- Track lens: the selected track supplies the stakeholder, active memory budget,
  energy budget label, accelerator name, and natural failure threshold.
- Student prior: "If the model runs on the device, it can probably fine-tune
  there."
- Scenario beat: a track stakeholder asks whether local adaptation can run inside
  the active memory budget.
- Prediction beat: structured radio on expected training-memory multiplier.
- Manipulation beat: model size, batch size, and adaptation strategy sliders/dropdown.
- Evidence beat: stacked memory chart, exact component table, fit/fail badge.
- Consequence beat: OOM or headroom callout names required MB, available MB,
  and mitigation.
- Math/source beat: memory formula and `mlsysbook_labs.training_memory_breakdown`.
- Checkpoint beat: choose the adaptation path to carry into the policy memo.
- Ledger fields: track, memory prediction, adaptation strategy, active memory,
  training memory, fit boolean, binding resource.

### Part B - Concept Module: Federated Updates Trade Communication, Staleness, Privacy, and Convergence

- Chapter claim: FedAvg is a coordination protocol whose convergence depends on
  client participation, local epochs, communication rounds, heterogeneity, and
  privacy-preserving update rules.
- Track lens: each track changes the update payload, privacy framing, acceptable
  round count, and consequence of stale population learning.
- Student prior: "Federation solves privacy, so the remaining problem is only
  uploading gradients."
- Scenario beat: legal/product requires raw data to remain local, but the fleet
  still needs population learning.
- Prediction beat: structured radio asks which trade-off dominates first.
- Manipulation beat: heterogeneity beta, local epochs, compression mode.
- Evidence beat: convergence curve, total communication, bytes/round, privacy
  overhead and stale-weight table.
- Consequence beat: non-IID, compression, or privacy can push the protocol beyond
  the track's freshness window.
- Math/source beat: FedAvg convergence and communication formulas plus
  `mlsysbook_labs.federated_communication`.
- Checkpoint beat: choose a federated update policy or reject federation for the
  selected track.
- Ledger fields: federation prediction, beta, local epochs, compression,
  non-IID rounds, total communication, privacy mode, stale rounds.

### Part C - Concept Module: Duty Cycle And Connectivity Make Intermittent Evidence First-Order

- Chapter claim: client scheduling and large-scale orchestration must treat
  intermittent eligibility, stragglers, power state, and network partitions as
  normal operation.
- Track lens: iPhone emphasizes overnight charging/Wi-Fi, Oura emphasizes tiny
  duty-cycle windows, RoboTaxi emphasizes depot/roadside evidence gaps, and Cloud
  Fleet emphasizes edge-site backhaul availability.
- Student prior: "A successful local update is enough evidence to promote a
  policy."
- Scenario beat: the team must decide whether enough fresh evidence reaches the
  coordinator before the model or environment changes.
- Prediction beat: structured numeric prediction of eligible update windows or
  evidence age.
- Manipulation beat: duty-cycle percent, connectivity percent, and update window
  controls.
- Evidence beat: evidence freshness chart and exact table for eligible,
  missed, stale, and usable windows.
- Consequence beat: stale evidence failure names the missing windows and the
  operational mitigation.
- Math/source beat: eligibility product and over-selection/staleness connection
  to client scheduling.
- Checkpoint beat: choose promote, defer, or stay local based on evidence age.
- Ledger fields: duty cycle, connectivity, update windows, usable updates,
  evidence age, intermittent-evidence decision.

### Part D - Concept Module: Edge Deployment Policy Is A Guardrail Bundle

- Chapter claim: production edge learning is deployable only when adaptation,
  replay/data policy, federation, validation, rollback, and compliance guardrails
  are integrated by device tier.
- Track lens: each track receives the same policies but different thresholds and
  report framing; failure names the track-specific violated guardrail.
- Student prior: "Pick the highest-quality policy that passed the earlier charts."
- Scenario beat: launch review asks for a single edge/federated policy.
- Prediction beat: structured radio asks which guardrail will bind the chosen
  candidate.
- Manipulation beat: choose among candidate policies and inspect guardrail table.
- Evidence beat: policy guardrail table covering energy, memory, privacy,
  update freshness, and quality.
- Consequence beat: rejected alternative is named with the violated guardrail and
  amount.
- Math/source beat: feasibility predicate over guardrail inequalities and
  production-integration reading anchor.
- Checkpoint beat: final report decision selects the policy, binding amount,
  rejected alternative, and V2-12 operations implication.
- Ledger fields: selected policy, rejected alternative, binding guardrail,
  guardrail pass/fail table, ops implication.

## Track Narratives

| Track | Persona and context | Constraint emphasis | Natural failure | Report framing |
|---|---|---|---|---|
| iPhone | Mobile product lead shipping personalization | Battery, thermal, active memory, privacy, Wi-Fi eligibility | Thermal/battery miss or privacy leakage from updates | "Can we keep sensitive adaptation on device without visible drain?" |
| Oura Ring | TinyML firmware lead shipping health adaptation | SRAM/flash, duty cycle, phone handoff, energy per wake window | SRAM/flash overflow or stale nightly evidence | "Can a tiny local model learn enough before handing off?" |
| RoboTaxi | Safety/perception lead using vehicle-local autonomy and fleet learning | Real-time local path, depot connectivity, safety evidence freshness | Safety margin miss or stale rare-event updates | "What remains local, and when can the fleet learn from rare events?" |
| Cloud Fleet | Platform/SRE lead comparing central serving with edge placement | Update bandwidth, privacy boundaries, edge cache quality, SLO | SLO/update freshness breach or negative ROI | "When does centralization fail the edge evidence and privacy guardrails?" |

## Mechanics, Evidence, And Ledger Plan

| Module | Mechanics | Evidence artifact | Reversible failure |
|---|---|---|---|
| A | Prediction radio; model/batch/strategy controls; stacked memory chart | Memory table and fit/fail badge | OOM when training memory exceeds active budget; recover by LoRA/bias/model reduction. |
| B | Prediction radio; beta/local-epoch/compression controls; convergence and communication chart | Federation table with rounds, bytes, privacy overhead, stale penalty | Freshness or communication miss; recover with compression, fewer local epochs, or local-only policy. |
| C | Numeric prediction; duty/connectivity/window controls; freshness chart | Eligible-window table and evidence-age badge | Stale evidence miss; recover by lowering update requirement, improving connectivity, or deferring promotion. |
| D | Prediction radio; policy selector; guardrail table | Pass/fail guardrail matrix and selected/rejected policy card | Any guardrail fails; recover by selecting a less aggressive policy. |
| Synthesis | Report card and export panel | Edge intelligence memo and Design Ledger snapshot | Incomplete memo if required predictions are missing. |

Ledger save should include:

- `selected_edge_policy`
- `binding_edge_amount`
- `rejected_alternative`
- `v2_12_ops_implication`
- memory, federation, intermittent-evidence, and policy evidence summaries
- selected track, scenario, hardware/model refs, and source policy

## Implementation Risks

- Shared MLSysIM and `mlsysbook_labs` helpers do not expose all requested policy
  quantities, so new policy and intermittency support must remain notebook-local
  and prefixed `v2_11_`.
- Existing track variant thresholds may be broad teaching estimates. The report
  must state that production deployment requires measured device traces.
- The lab must preserve WASM bootstrap, the track selector, and Design Ledger
  patterns while editing only the owned notebook and plan files.
- Other workers may modify unrelated labs in parallel; this task must not change
  shared helpers, tests, implementation notes, or other labs.

## Depth Audit

| Module | Concept clarity | Activity depth | Track specificity | Mechanics fit | Evidence quality | Traceability | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Part A | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part B | 3 | 3 | 3 | 3 | 3 | 3 | Pass |
| Part C | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Part D | 3 | 3 | 3 | 3 | 3 | 2 | Pass |
| Synthesis | 3 | 3 | 3 | 2 | 3 | 2 | Pass |

Minimum acceptance checks:

- Each module has at least five student-facing beats.
- Each module includes a structured prediction and manipulation.
- At least one reversible failure state exists; the notebook has reversible OOM,
  stale evidence, communication/freshness miss, and guardrail failure states.
- Synthesis ties the decision back to the invariant and records the V2-12
  operations implication.
