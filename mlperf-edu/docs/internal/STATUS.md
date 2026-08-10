# MLPerf EDU Status

Where the suite stands and what remains. This replaces the four overlapping
ledgers that previously restated the same fourteen workloads and the same two
blockers in slightly different words.

## Read Generated State, Not This File

Counts, quality decisions, timings, and evidence classes are **not** recorded
here. They drift the moment a run lands. Get them from the source that
computes them:

| Question | Where to get it |
|:---|:---|
| **Where does every workload stand?** | **[WORKLOAD_STATUS.md](WORKLOAD_STATUS.md)** — generated; quality, runtime, and what each needs |
| Is every workload configured to run? | `python3 tools/workload_status.py` |
| Which cases have evidence, and of what class? | `provisional_results/index.json` |
| What is each workload's contract and gate? | `registry/suites/**` |
| What did a run actually measure? | the report that run wrote |
| Does the published material still match? | `tools/check_reference_claims.py --check` and `tools/generate_docs.py --check` |

This file records only decisions and open work, which change deliberately
rather than per run.

## Workload States

Every registered workload has a public CLI path, a contract with an upstream
basis, an asset policy, report export, and a provenance boundary. They differ
in how far their authoritative `max` contract has gotten.

| State | Meaning | Workloads |
|:---|:---|:---|
| Target met | At least one complete authoritative `max` result meets the declared target | image classification, keyword spotting, anomaly detection, visual wake words, causal language modeling, text classification, information retrieval, graph node classification |
| Target gap recorded | A complete provenance-bound measurement does not meet the unchanged published point | time-series forecasting, code generation, function calling, image generation |
| Environment gated | The authoritative contract needs an environment outside the declared laptop envelope | recommendation, reinforcement learning |

The gaps are recorded, not negotiated. No target was lowered to convert a gap
into a pass; the withdrawn time-series tolerance is the worked example, and the
registry reviewer notes explain why it was withdrawn.

Two states are worth distinguishing because documents have confused them
before. A *target gap* means the contract executed and the result missed. An
*environment gate* means the contract has not executed locally at all.

## Open Work

### Blocking a downloadable release

These are ordered in [SHIP_PLAN](SHIP_PLAN.md), which is the plan of record.

- [ ] Adopt an authoritative component license and package-index versioning
      policy.
- [ ] Close MLCommons review of the name, scope, governance, and result
      wording.
- [ ] Close dataset redistribution and fetch-only decisions.
- [ ] Publish the package so installation does not require the source
      checkout.
- [ ] Reproduce the suite on independent CPU, Apple Silicon, and CUDA systems.

### Local execution

- [x] Recommendation executes its contract locally. The identity moved to
      MLPerf Training v0.5 NCF on MovieLens-20M, which fits the envelope; the
      DLRM path it replaced never could.
- [x] MiniGo executes its contract locally through a PyTorch adapter that
      replaces only the network. Go rules, MCTS, and the professional-move
      evaluation remain the pinned reference code.
- [ ] Record a real MiniGo result. Only a smoke run exists so far.
- [ ] Complete a clean-cache asset audit for all workloads, so a first run on a
      machine with no prior state is verified rather than assumed.
- [ ] Add resumable execution for long preprocessing and training paths.

### Quality follow-up

These refine or approve interpretation. None of them justifies lowering a
target to fit a local result.

- [ ] Obtain domain approval for the one-sided OGB GCN target interpretation.
- [ ] Obtain independent approval for the nanoGPT target interpretation.
- [ ] Resolve each recorded target gap without weakening its target, or record
      the shortfall as final.

### Stability and promotion

- [ ] Complete the five-process stability campaign for cases that do not yet
      have it.
- [ ] Enforce the declared timing-variation contract on promotion candidates.
- [ ] Promote only results whose quality, provenance, compatibility, and timing
      requirements all pass.
- [ ] Populate measured working-set, arithmetic-intensity, and dispatch
      evidence, or leave those fields `unmeasured`.

### Production release

- [ ] Replace executable EDM pickle inputs with reviewed safe artifacts.
- [ ] Produce signed release artifacts and authenticated provenance where
      producer identity matters.
- [ ] Define support, vulnerability response, retention, and rollback
      procedures.
- [ ] Measure authoritative `max` budgets on each actual course image.
- [ ] Add controlled cross-hardware comparisons.

## Settled Decisions

Recorded so they are not relitigated.

- One complete authoritative run is enough to accept or reject a quality target
  at this stage. Repeated runs and timing variation belong to the later
  stability phase.
- A `min` probe establishes setup only. It never counts as quality evidence.
- Dashboards are generated for every run and opened only on explicit request.
- The keyword-spotting adapter is retained as a disclosed, quality-preserving
  educational adaptation, and is blocked from promotion until an exact-source
  execution path establishes prediction parity.
- Restricted dataset bytes are rejected from portable packages while
  redistribution decisions remain open.
- Measurements do not appear on the website or in these documents. They belong
  to the run artifact that produced them.

## Where Other Material Went

- Motivation, design commitments, admission test, and research boundary:
  [DIRECTION](DIRECTION.md).
- Release sequencing and the acceptance test for a downloadable benchmark:
  [SHIP_PLAN](SHIP_PLAN.md).
- Target authority and rationale: [QUALITY_TARGET_REVIEW](QUALITY_TARGET_REVIEW.md).
- How a student, instructor, or researcher actually uses the suite: the
  website, which owns the getting-started, running, results, and labs guides.
