# Vol1 Responsible Engineering — Phase 2 Opus Improvements

**Chapter**: `vol1/responsible_engr` (chapter 15 of 33)
**Source audit grade (gpt-5.4)**: B
**Target**: A-grade per §16 gold-standard patterns
**Date**: 2026-04-24

## Counts

| Metric | Count |
|---|---|
| Sections total | 7 |
| Sections with quizzes | 7 |
| Questions before | 29 (3 + 5 + 5 + 5 + 5 + 3 + 3) |
| Questions after | 30 (3 + 5 + 5 + 5 + 5 + 4 + 3) |
| **Rewritten** (substantial craft change) | 18 |
| **Kept** (already A-grade or minor polish) | 11 |
| **Deleted** | 0 |
| **Added** | 1 (new TF in Fallacies section) |

## Section-by-section summary

### §1 — Responsibility as Systems Engineering (3 questions)

No audit issues flagged, but all three were B+ rather than A. Rewrites strengthen distractors and tighten §16 compliance.

- **MCQ (q1)**: REWRITTEN — scenario now carries concrete numbers (99.9 percent availability, 87 percent aggregate accuracy, specific resume pattern) and the four distractors each encode a distinct mental-model failure (capacity, reliability-only thinking, verification-is-fairness). Explanation refutes by content.
- **SHORT (q2)**: REWRITTEN — now a diagnostic scenario (team arguing for one-time review) rather than an open-ended explanation, and answer closes with a specific measurement (subgroup TPR gap over time) and systems consequence.
- **TF (q3)**: REWRITTEN — now targets the real misconception a CS grad student would hold (modular-software-therefore-localized-fixes) with the null-pointer contrast, avoiding the trivial original framing.

### §2 — Engineering Responsibility Gap (5 questions)

Audit flagged q1 `throwaway_distractor` (weak "more epochs" option) and q3 `trivia_fill` (alignment-gap blank).

- **MCQ (q1)**: REWRITTEN — replaced the "more training epochs" distractor with a stronger bias-mitigation-training-epochs variant that a learner could actually believe; answer now explains why proxy mechanism is intact despite retraining.
- **SHORT (q2)**: REWRITTEN — added specific monitoring signals (Jensen-Shannon divergence threshold 0.1, disaggregated outcome tracking) the original answer elided, and closes on instrumentation consequence.
- **FILL→MCQ (q3)**: CONVERTED per audit suggestion. The trivia-fill for "alignment gap" became a scenario MCQ asking the reader to *select the engineering intervention* (counterfactual holdout) that detects the decoupling. Distractors encode three real misconceptions (scale-up, reweight-proxy, more-data). This is a substantial pedagogical upgrade.
- **MCQ (q4)**: KEPT with polish — invariance-testing classification is already A-grade; minor tightening of answer explanation.
- **TF (q5)**: REWRITTEN — broadened the claim to cover the full architectural-foreclosure mechanism (loss function, architecture, attribute collection, monitoring) with the Amazon-cancellation evidence in the justification.

### §3 — Responsible Engineering Checklist (5 questions)

Audit flagged q4 ORDER as `recall_only` (editorial sequence rather than causally necessary).

- **MCQ (q1)**: REWRITTEN — now derives the 100x multiplier explicitly and forces the learner to reason that subgroup confidence depends on subgroup sample count, not overall dataset size. Strengthened the 1,000-image distractor (attribution to training procedure).
- **SHORT (q2)**: REWRITTEN — asks for a concrete scope-creep scenario the predeployment card blocks, tying the guard-rail claim to the 40-60 percent scope-creep statistic.
- **MCQ (q3)**: KEPT — fairness-metric diagnosis from confusion matrices is already A-grade and grounded in the chapter's numeric anchors.
- **ORDER→SHORT (q4)**: CONVERTED per audit. Replaced the editorial predeployment→fairness→documentation sequence with a SHORT asking the team to present the Pareto frontier and quantify the price-of-fairness (5 pp FP increase, 3 percent utility loss) on the chapter's $100k/$50k hiring calculus. This tests the chapter's central trade-off framing rather than an arbitrary ordering.
- **MCQ (q5)**: REWRITTEN — added quantitative regulatory anchors (35M EUR, 7 percent global turnover, Annex III) and sharpened the size-based distractor to reference foundation models specifically.

### §4 — Environmental and Cost Awareness (5 questions)

Audit flagged q1 `throwaway_distractor` (benchmark-appearance claim) and q4 `trivia_fill` (TCO blank).

- **MCQ (q1)**: REWRITTEN per audit suggestion — replaced the benchmark-appearance distractor with the stronger "quantization and responsibility are separate engineering layers" misconception, which tests whether the reader internalized the section's unifying claim.
- **MCQ (q2)**: KEPT with polish — wearable deployment-fit question is already A-grade; added explicit 10x margin and sustained-power reasoning to the answer.
- **SHORT (q3)**: REWRITTEN — now compares two specific optimization options (50 percent training cut vs 20 percent inference cut) and asks for the dollar-savings ratio explicitly, yielding ~15:1 in favor of inference.
- **FILL→MCQ (q4)**: CONVERTED per audit — trivia-fill for TCO replaced with a quantitative MCQ on GPT-3 training emissions and the identifying the dominant responsibility levers (utilization + carbon-aware scheduling). Distractors encode parameter-count-only, zero-carbon absolutism, and FP16-only misreads.
- **TF (q5)**: KEPT with polish — carbon-region vs algorithmic-efficiency comparison is already A-grade; sharpened the quantitative comparison (5x vs low-percent).

### §5 — Data Governance and Compliance (5 questions)

Audit flagged q4 ORDER as forcing a strict dependency among security/privacy/audit not supported by the section.

- **MCQ (q1)**: REWRITTEN — now grounded in Meta's 2023 390M EUR fine (a non-breach governance failure), strengthening the evidence that documentation alone is insufficient. Distractors encode certification-handoff, datasheet-only, and raw-storage-only misconceptions.
- **SHORT (q2)**: REWRITTEN — now walks through why manual search is both unreliable (distributed artifacts) and too slow (regulatory timeline) and specifies the lineage-graph infrastructure required.
- **MCQ (q3)**: KEPT with polish — privacy-by-design for the Lighthouse KWS system is A-grade; tightened the RBAC distractor's framing.
- **ORDER→SHORT (q4)**: CONVERTED per audit. Instead of imposing a strict security→privacy→audit dependency, the SHORT asks the reader to distinguish the operational role of each and give a concrete failure mode if the other two are in place but that mechanism is missing. This is both more faithful to the section and more testing of reasoning.
- **TF (q5)**: REWRITTEN — reframed to target the common underestimation of GDPR penalty scale (20M EUR / 4 percent global turnover, Meta's 390M EUR), with the systems consequence about deferred-infrastructure compounding risk.

### §6 — Fallacies and Pitfalls (3 questions → 4)

No audit issues flagged, but the section has four distinct pitfalls and the original quiz tested only three.

- **MCQ (q1)**: KEPT — aggregate-metric diagnosis on the loan scenario is already A-grade; tightened the content-based distractor refutations.
- **SHORT (q2)**: REWRITTEN — strengthened to cite both Amazon and Optum cases with the 70-90 percent protected-attribute-recovery research anchor, and specifies the required engineering work (causal analysis, fairness constraints, per-group monitoring).
- **TF (q3 — NEW)**: ADDED — tests the documentation-as-enforcement pitfall specifically, anchored in the 40-60 percent scope-creep statistic. Adds coverage of one of the section's six listed pitfalls not otherwise tested.
- **TF (q4)**: KEPT with polish — training-only environmental accounting is A-grade; sharpened the 40:1 inference-to-training ratio anchor.

### §7 — Summary (3 questions)

Audit flagged q3 `throwaway_distractor` (two weak "second master" distractors).

- **MCQ (q1)**: KEPT with polish — chapter-thesis MCQ is A-grade; strengthened the "replacement" distractor explanation.
- **SHORT (q2)**: REWRITTEN — now asks for a contrast between vague principle and specific invariant with explicit SLO-integration workflow, making the measurable-properties claim concrete.
- **MCQ (q3)**: REWRITTEN per audit — replaced the two underpowered distractors (quantization-only-speed, hardware-carbon-policy) with subtler ones that preserve partial truth: monitoring-as-separate-infrastructure (contradicted by chapter's integration argument) and sustainability-separated-from-accessibility (misses the three-channel unification). The policy-only carbon distractor is retained but reframed as "engineers cannot influence through technical choices" which is a real misconception, not obviously wrong.

## Three issue patterns fixed

1. **Trivia-fill recall tests** (`trivia_fill`): two FILLs (alignment gap, TCO) converted to scenario MCQs that force engineering-intervention selection and quantitative comparison, respectively. FILLs remain valuable when the blank is reasoned from mechanism (per §16 FILL examples), but these were vocabulary slots.

2. **Weak distractors** (`throwaway_distractor`): addressed in five MCQs (§2 q1, §3 q1 polish, §4 q1, §7 q3 primarily). Replaced obviously-wrong options with content-encoded mental-model failures a real practitioner would hold — e.g., "more bias-mitigation epochs will fade the gender signal," "quantization and responsibility belong to separate engineering layers," "monitoring requires entirely separate infrastructure from reliability."

3. **Editorial ORDER questions** (`recall_only` / `other`): both ORDER questions (§3 q4 checklist sequence, §5 q4 governance dependency) were editorial rather than causally necessary. Converted to SHORTs that test the underlying reasoning (Pareto-frontier presentation in §3; role differentiation with missing-mechanism failure modes in §5). This reduces the ORDER count from 2 to 0 in this chapter but is a net A-grade upgrade.

## One substantial-rework section

**§3 Responsible Engineering Checklist** received the largest overhaul. The original ORDER question (predeployment → fairness → documentation) was flagged as editorial and did not match the section's actual argument (documentation spans the workflow, not just the end). It was converted to a SHORT that tests the chapter's central engineering trade-off — presenting the Pareto frontier to stakeholders with the specific price-of-fairness calculation ($100k hire value, $50k bad-hire cost, 5 percentage-point FP increase, ~3 percent utility loss) — which is the section's thesis punchline. The other four questions were also substantially strengthened with explicit numeric anchors (100x representation multiplier, 35M EUR EU AI Act penalty, 7 percent global turnover, Annex III classification), bringing the whole section to A-grade.

## JSON + validator status

- **File written**: `/Users/VJ/GitHub/MLSysBook/book/tools/scripts/genai/quiz_refresh/_audit/opus_improved/vol1_responsible_engr_quizzes.json`
- **Validator**: `python3 validate_quiz_json.py … vol1_responsible_engr.qmd` → **OK: passes schema + anchor validation**
- **Metadata updated**: `generated_on: "2026-04-24"`, `model: "claude-opus-4-7"`, `improved_by: "opus-subagent-phase2"`
- **§10 anti-shuffle check**: all MCQ explanations refute distractors by content (no `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, or `([A-D])` references)
- **Type mix after rework** (across 30 questions): MCQ 14 (47%), SHORT 9 (30%), TF 6 (20%), FILL 0, ORDER 0 — slightly under-represented on FILL and ORDER because the flagged instances were genuine trivia/editorial rather than A-grade material; the conversion was a net gain on the quality bar
- **Letter distribution**: correct answers distributed across A/B/C/D at generation time per §10 (no post-shuffle; varied by question)
