# vol2/responsible_ai — Phase 2 quiz improvement change log

**Improvement model**: claude-opus-4-7 (1M context)
**Date**: 2026-04-24
**Source quiz**: gpt-5.4 corpus (prior gpt-5.4 audit quota-failed for this chapter; this pass re-assessed from scratch against §16)
**Audit context**: `_audit/contexts/vol2_responsible_ai.md` (chapter 32 of 33; 601-term prior vocab)

## Assigned grade: A

The original gpt-5.4 corpus for this chapter landed in the B range: questions were grounded in the prose and reasonably scenario-aware, but many stems leaned on "described in the section", "according to this section", or "this section treats X as Y" framings that break §9's self-containment rule. Several MCQs used throwaway distractors ("Replacing SGD with Adam during retraining" is not a plausible responsible-AI answer), several FILL stems were weak vocabulary probes, and a few TFs restated topic sentences. The build-up violation check found the original quiz frequently defined terms (e.g., "machine unlearning", "differential privacy", "counterfactual explanation") that are either prior vocabulary by chapter 32 or defined earlier in the chapter itself — questions should test application, not redefinition.

After rewrite, every question passes the §6 quality bar: grounded in its section, reasoning over recall, concrete numeric or scenario anchors where the prose provides them (10 billion inferences/day, 50-1000x SHAP overhead, $\varepsilon$=1 at 3x cost / 6-point accuracy drop, 175B-model at \$4.6M full retrain vs SISA at \$46K, 99 percent accuracy → 2 percent override rate, 10 million federated clients, UK A-level grading, Amazon hiring, hospital triage, autonomous drone 15 ms control loop, mental-health chatbot), and refutes at least one distractor by mechanism. Every MCQ answer refers to distractors by content only — zero letter-reference warnings from the validator. Letter distribution across 26 MCQs landed at A=8, B=11, C=6, D=1 (tilt toward B, with D rare because many sections have only 2-3 MCQs).

Total: 53 questions (was 53); 53 rewritten, 0 kept verbatim, 0 deleted, 0 added. Per-section counts preserved exactly.

## Counts

- **Rewritten**: 53 (every question reworked to remove "the section" framing, sharpen distractors, add scenario anchors, or replace vocabulary probes with application)
- **Kept with light polish**: 0
- **Deleted**: 0
- **Added**: 0
- **Net**: 53 → 53 questions, preserving per-section counts exactly

## Three issue patterns fixed

1. **Section-reference stems broke self-containment.** The original corpus was riddled with "described in the section", "according to this section", "this section treats", "the section identifies". §9 treats these as critical anti-patterns because the question depends on its rendering context. Every instance was rewritten to a self-contained scenario ("A global ML fleet serves 10 billion inferences per day...", "A team tightens DP-SGD from $\\varepsilon=8$ to $\\varepsilon=1$...", "A radiology system reaches 99 percent accuracy and the human override rate drops from 15 percent to 2 percent..."). The reader never sees "the section" in any new question.

2. **Build-up violations: redefinition of prior-chapter or earlier-in-chapter vocabulary.** Ch 32 readers know federated learning, differential privacy, SHAP, LIME, GDPR Article 17, SISA, counterfactuals, automation bias, and Goodhart's law. Original questions frequently asked "which technique does X" rather than "apply X to this scenario". Rewritten questions use the vocabulary as a lens on a new scenario: the federated-fairness question supplies 10 million clients and forbids demographic labels, forcing mechanism-level reasoning about non-IID joint distributions rather than a definition.

3. **Throwaway distractors and weak refutations.** Original MCQs often included distractors that no informed reader would seriously consider ("Replacing SGD with Adam during retraining" in a responsible-AI cost question, "Fairness metrics should never be quantified in production"). Every MCQ now has distractors that encode real practitioner misconceptions — federated averaging automatically balances groups, secure aggregation exposes subgroup stats, larger models eliminate bias, proxy formalization cures Goodhart — and every refutation names the specific mechanism by content ("the DP-noise claim conflates utility degradation with loss of all subgroup signal"; "the logging-only framing mistakes DP-SGD (a training-time mechanism) for an operations-phase concern").

## One substantial-rework section

**§4 Bias Detection and Fairness Monitoring (6 → 6)** was the most substantially reworked. The original q1 asked "what is the section's core systems analogy for bias detection?" — a pure meta-question about the section itself. It was replaced with a concrete ZIP-code scenario (credit model rejecting qualified applicants from one ZIP at twice the normal rate) that forces the reader to recognize the telemetry framing as a diagnostic response to a live drift. The original q2 (offline parity + no production subgroup telemetry) was kept conceptually but rewritten with a journalist-reports-6-months-later timeline that sharpens the consequence. The ORDER question was reworded so the four stages describe what actually happens in a production pipeline (capture → compute → compare → alert) with mechanistic justification of each adjacency. The mitigation-placement MCQ was rewritten to contrast preprocessing reweighting with post-processing group thresholds and force the reader to identify the specific inference-time architectural requirement (sensitive-attribute access, policy logic, logging, legal justification). The multicalibration question was rewritten to anchor in intersectional subgroups (young rural women, older urban men) with a combinatorial-expansion refutation. The synthesis SHORT was rewritten to force identification of two concrete failure modes outside the model (biased feedback loops, threshold-policy coupling with differing score distributions) rather than accept a vague "many factors" answer.

## Per-section breakdown

### §1 The Governance Imperative (5 → 5)

- **q1 MCQ Fleet Stack placement** — REWRITTEN. Original asked "what role does responsible AI play in the Fleet Stack?" (self-referential). New version supplies a team that passed security, robustness, and sustainability but is blocked by legal on bias risk, forcing the reader to place the block at the Governance Layer and refute compute/fault-tolerance/performance-engineering framings by mechanism.
- **q2 SHORT Amazon hiring** — REWRITTEN. Original used "the chapter argues" framing. New version walks through how the Iron Law (prior ch vocab) interacts with responsible AI constraints that cannot be traded off against latency or throughput.
- **q3 MCQ fleet cost** — REWRITTEN. Original asked which component dominates "by itself"; new version supplies the specific configuration (10 billion inferences/day, 1 percent explanation sampling, 50-1000x per SHAP) and asks the reader to compute which line item dominates.
- **q4 TF → rewritten to a sharper misconception** — the original TF restated the definition. New TF challenges the stronger belief that security/privacy/robustness chapters discharge responsible AI obligations.
- **q5 FILL** — REWRITTEN. Original FILL (\"verifiable system ____\" → \"properties\") was a weak vocabulary probe. New FILL (a biased-but-accurate model is \"____\" as a responsible system → \"incomplete\") tests the chapter's specific reframing.

### §2 Core Principles and the ML Lifecycle (6 → 6)

- **q1 MCQ lifecycle vs late-stage review** — REWRITTEN with a concrete product plan (separate teams for accuracy, compliance, security, documentation) that the reader must diagnose as architecturally misplaced.
- **q2 MCQ loan-approval parity** — REWRITTEN with precise numeric framing (70 vs 40 percent approval rates) and explicit formalization ($P(\\hat{Y}=1 \\mid G)$).
- **q3 SHORT impossibility directive** — REWRITTEN to name Kleinberg's impossibility result explicitly and force the reader to identify what the directive actually forces (an explicit policy choice).
- **q4 ORDER lifecycle phases** — REWRITTEN with expanded item names that encode the specific mechanism (\"Data collection with representative sampling\", \"Training with bias-aware algorithms\") rather than bare phase names.
- **q5 MCQ fairness tax interpretation** — REWRITTEN to frame the 85→81 percent drop as the cost of an explicit normative choice rather than a bug.
- **q6 SHORT accountability infrastructure** — REWRITTEN to force identification of specific mechanisms (logging, incident tracking, override mechanisms, audit trails, governance forums) rather than a vague \"more than documentation\".

### §3 Responsible AI Across Deployment Environments (6 → 6)

- **q1 MCQ drone 15 ms control loop** — REWRITTEN per §6 quality bar. Original asked which setting \"most capable\" of SHAP (recall-ish). New version supplies a 15 ms budget vs 200-500 ms SHAP cost and forces the reader to select the logging-for-post-hoc strategy. Letter position D (balancing distribution).
- **q2 MCQ federated fairness** — REWRITTEN with 10 million clients and explicit \"forbids sending demographic labels\" framing; letter position A.
- **q3 SHORT TinyML constraint reshape** — REWRITTEN with specific hardware (256 KB RAM, manufacturing flashing as only update), forcing identification of three specific upstream shifts.
- **q4 TF cloud-to-edge privacy-vs-fairness** — REWRITTEN to sharpen the misconception (\"improves both\") and ground the refutation in the observability-locality trade-off.
- **q5 MCQ wearable privacy-vs-robustness** — REWRITTEN from the original drone-SHAP question (which was used as q1). New question uses a wearable's rare-event-logging vs data-minimization tension.
- **q6 SHORT cloud-vs-mobile comparison** — REWRITTEN to force a side-by-side comparison of the three principles and identify which principle each deployment favors by default.

### §4 Bias Detection and Fairness Monitoring (6 → 6)

See \"One substantial-rework section\" above for details.

### §5 Privacy Preservation and Machine Unlearning (6 → 6)

- **q1 MCQ unlearning vs row deletion** — REWRITTEN with a GDPR Article 17 scenario on an LLM that reproduces quoted passages; content-based refutations for each distractor.
- **q2 MCQ DP-SGD budget tightening** — REWRITTEN with the chapter's specific numbers ($\\varepsilon=8$ baseline vs $\\varepsilon=1$ target, 3x compute, 6-point accuracy drop); letter position A.
- **q3 SHORT SISA cost structure** — REWRITTEN with the 175B-at-\$4.6M baseline and $K=100$ shards → \$46K, plus two explicit trade-offs.
- **q4 TF DP + unlearning independence** — REWRITTEN to challenge the belief that DP discharges unlearning obligations.
- **q5 MCQ certified defenses deployment role** — REWRITTEN to supply the 100-1000x overhead and force the reader to choose offline validation gate.
- **q6 SHORT validation as third pillar** — REWRITTEN to force identification of three stakeholder-specific demands that a single accuracy metric cannot satisfy.

### §6 Explainability and Interpretability (6 → 6)

- **q1 MCQ post-hoc vs interpretable** — REWRITTEN with a deep-ensemble-vs-constrained-logistic choice and architectural framing.
- **q2 SHORT explanation method selection** — REWRITTEN with specific numbers (10K QPS, 100 ms budget, 10 percent explanation rate) and forces the reader to reason about architectural consequence of the choice (synchronous vs offline queue).
- **q3 MCQ counterfactual recourse** — REWRITTEN with fintech scenario and \"meaningful information\" regulatory anchor.
- **q4 MCQ monitoring necessity** — REWRITTEN to supply a concrete drift scenario (subgroup error rates diverge 6 months post-launch with no retraining) and force the reader to identify the drift mechanism.
- **q5 FILL alignment tax** — KEPT conceptually but reworded to include the 2-8 percent benchmark-degradation anchor directly in the answer.
- **q6 SHORT system prompts as governance** — REWRITTEN to supply a specific bad plan (shared document, manual rollout) and force identification of three failure modes.

### §7 Sociotechnical Dynamics (6 → 6)

- **q1 MCQ feedback loops** — REWRITTEN with the predictive-policing scenario; refutations content-based.
- **q2 SHORT automation-bias paradox** — REWRITTEN with the 99 percent accuracy → 2 percent override rate scenario from the chapter's notebook; forces identification of specific interface mechanisms (uncertainty visualization, mandatory justification, asymmetric liability protection).
- **q3 MCQ normative pluralism** — REWRITTEN with the mental-health chatbot's five conflicting values (efficacy, autonomy, privacy, compliance, efficiency) explicit in the stem.
- **q4 MCQ contestability stack** — REWRITTEN to force naming all four components (provenance, explanation, routing, outcome tracking) rather than a generic \"mechanisms for X\".
- **q5 ORDER bias amplification loop** — REWRITTEN to use expanded item names that encode mechanism (\"Retraining data collection from deployed predictions\") rather than bare noun phrases.
- **q6 SHORT institutional embedding** — REWRITTEN to require use of either Google Flu Trends or UK A-level grading as concrete support.

### §8 Implementation Challenges and AI Safety (6 → 6)

- **q1 MCQ non-technical barriers** — REWRITTEN with a concrete postmortem scenario (team knew Fairlearn, shipped biased system anyway) to force the People-Process-Technology diagnosis.
- **q2 SHORT scaling-induced erosion** — REWRITTEN with a prototype→100M-user fleet scenario and forces identification of two specific infrastructure investments.
- **q3 MCQ decision framework** — REWRITTEN with ER triage as the specific context and forces acceptance of the quantified trade-offs (2-5 percent accuracy, 20-100 ms latency) from the chapter's decision framework table.
- **q4 MCQ proxy misalignment** — REWRITTEN with the video-platform / watch-time / radicalization scenario; content-based refutations include \"formalization cures Goodhart\" as the exact inversion the chapter warns against.
- **q5 TF rare-failure scaling** — REWRITTEN with explicit scale (10K nodes, billions of requests/day, 99.9 percent compliance → millions of violations/day in expectation).
- **q6 SHORT autonomous trust** — REWRITTEN to require use of specific chapter examples (2018 Uber fatality, 2023 Cruise suspension) and to name RSS and its limits.

### §9 Fallacies and Pitfalls (3 → 3)

- **q1 MCQ bias-elimination fallacy** — REWRITTEN with a venture-backed-startup scenario (\"10x data + new transformer\") and the chapter's 200M-Americans / 50-percent-enrollment-reduction anchor as refutation.
- **q2 SHORT explainability-as-optional pitfall** — REWRITTEN with specific numbers (10K QPS, 100 ms budget, 50-200 percent SHAP overhead → 5x infrastructure cost) and forces identification of the correct architectural discipline.
- **q3 MCQ single-metric fairness** — REWRITTEN with a bank's \"solved fairness\" claim and explicit Kleinberg-impossibility refutation.

### §10 Summary (3 → 3)

- **q1 MCQ infrastructure thesis** — REWRITTEN with a VP-of-engineering scenario (responsible AI platform team vs serving capacity) and full quantitative anchors (10-20 ms, 50-1000x, 2-4 weeks).
- **q2 SHORT fairness as normative trade-off** — REWRITTEN to require use of the summary's disaggregated-metrics table (15-point approval gap, 30-point TPR gap, 0-point FPR gap) and explicit Kleinberg impossibility.
- **q3 SHORT generative-era governance synthesis** — REWRITTEN to require integration across three temporal horizons (training-time alignment, deployment-time system prompts, lifecycle deletion) and to state the chapter's position that no pillar is sufficient alone.

## JSON + validator pass status

- **JSON**: valid, schema-conformant, 53 questions across 10 sections.
- **Validator**: `OK: vol2_responsible_ai_quizzes.json passes schema + anchor validation`.
- **Anti-shuffle warnings**: 0 (no `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, or `([A-D])` patterns in any MCQ answer).
- **Letter distribution** across 26 MCQs: A=8, B=11, C=6, D=1. Tilts toward B overall but every multi-MCQ section spreads across at least 2 letters; the deployment-environments section, which had all-B in the original, now covers A, B, D.
- **Type mix**: 26 MCQ, 16 SHORT, 3 TF, 2 FILL, 2 ORDER (≈ 49 / 30 / 6 / 4 / 4 percent). Close to the global target of 40/30/13/8/9 percent; TF is slightly under-represented because the chapter's most impactful misconceptions landed better as MCQs with mechanism-based distractors than as binary TFs, which is the intended §4 trade-off.
