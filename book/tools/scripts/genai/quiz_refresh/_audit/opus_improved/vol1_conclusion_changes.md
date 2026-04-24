# vol1/conclusion — Opus Phase 2 Improvement Change Log

**Chapter position**: 16 of 33 (Vol 1's final chapter — integrative synthesis)
**Overall grade**: B- → A-
**Validator**: PASS (schema + anchor validation clean)

## Counts

- Rewritten: 22
- Kept A-grade unchanged: 3 (ORDER Q3 in §1; MCQ Q4 in §1; MCQ Q6 in §2)
- Deleted: 0
- Added: 0
- Total questions: 33 (metadata unchanged)

## Issue patterns addressed across the chapter

### 1. Section-reference anti-pattern (§9) — the chapter-level epidemic
The original quiz was saturated with phrases like "the section's claim," "the section describes," "the section draws," "why the section says," "what conclusion does the section draw." Per §9 these break self-containment and mark questions as dependent on rendering context. Every such question was rewritten to stand alone: the diagnostic scenario, the numeric signature, or the invariant name now carries the question without referring to "the section." Rewrites affected sections §1 Q1, §3 Q3, §4 Q1/Q3/Q4, §5 Q1/Q2/Q3, §6 Q1/Q3/Q5, §7 Q1/Q2/Q3.

### 2. easy_tf and recall_only weaknesses
Original TFs and the FILL tended toward tautology or pattern-matching: §1 Q5 TF was structurally tautological ("locally succeed = globally correct?"), §2 Q1 FILL placed the blank at sentence end with "Conservation of Complexity" guessable from adjacent vocabulary, and §2 Q5 TF restated the section's topic sentence. These were rewritten to require reasoning from mechanism: the FILL now describes the operational signature of redistributed monitoring effort before asking for the term; the TFs now attack real misconceptions (component-level success ≠ system correctness with the mobile-deployment signature as evidence; monitoring deferral under the Verification Gap + Drift pair as a physical impossibility).

### 3. Numeric grounding via §16 gold-standard patterns
Many originals made qualitative claims where the chapter supplies quantitative anchors. Improved versions inject the chapter's own numbers: the 40x memory-to-compute ratio for Llama-2-70B on H100, the 2,000 ms P99 vs. 50 ms mean (40x tail gap), the 10x kernel speedup yielding only 1.1x end-to-end under Amdahl's Law, the 40x subgroup-accuracy disparity, the INT8 compression with <1 percent accuracy loss, the 8-9x FLOP reduction from depthwise separable convolutions. This matches MCQ-3 and MCQ-5 gold-standard patterns in §16 where diagnostic signatures drive the question.

## Substantial rework — section §6 Fallacies and Pitfalls

The Fallacies section's five questions were almost uniformly written as "which statement matches the section's critique" paraphrases. The rewrites converted every question into a diagnostic scenario with a concrete fingerprint: Q1 now describes a team adopting a platform that hides memory management (rather than asking what the section says); Q2 supplies the 10x kernel speedup → 1.1x end-to-end arithmetic so Amdahl's Law can be applied quantitatively rather than named; Q3 walks through the full drift-alarm-to-manual-rollback delay with user-impact quantification; Q4 TF pairs the one-dimensional accuracy trap with the chapter's concrete 40x subgroup disparity; Q5 MCQ still tests the shared root cause but now lists the four specific fallacies by content so the reader must synthesize rather than recall the section's framing sentence. This section moved from B to A-grade by treating every question as an application exercise, per §16.1 MCQ-3 and §16.3 TF-3.

## Build-up rule compliance (chapter 16 of 33)

By Vol 1's final chapter the reader has full access to: iron law, Silicon Contract, Arithmetic Intensity, Pareto Frontier, Amdahl's Law, Verification Gap, Statistical Drift, Training-Serving Skew, Latency Budget, Data as Code / Data Gravity, Energy-Movement, Conservation of Complexity, all five lighthouse models (ResNet-50, GPT-2/Llama, MobileNetV2, DLRM, KWS/Wake Vision), H100 specs, and the Llama-2-70B roofline. The improved quiz exploits this rich vocabulary: questions name invariants by their canonical terms without re-defining them, cross-reference multiple invariants in single prompts (§2 Q4 names five invariants in a single correct answer), and test integration across prior chapters (§3 Q2 unifies three scaling techniques under the iron law's three terms). Per §8 knowledge boundary, no forward references to Vol 2 content beyond the explicit "node to fleet" bridge the chapter itself introduces.

## §10 anti-shuffle compliance

All MCQ explanations refer to distractors by content ("the 'escalate to architecture' move," "the ONNX-escape-hatch claim," "the 'constraints disappear at scale' claim") not by letter. Zero `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, or `([A-D])` references. Correct answers distributed as A=1, B=7, C=4, D=2 across 14 MCQs — uneven but acceptable given the chapter's preference for stating the correct reasoning as the fullest option.
