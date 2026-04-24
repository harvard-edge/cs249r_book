# vol2/ops_scale — Phase-2 Improvement Change Log

**Chapter**: 28 of 33 (vol2/ops_scale)
**Baseline model**: gpt-5.4 (generated 2026-04-24)
**Improver**: opus-subagent-phase2 (claude-opus-4-7)
**Validator status**: PASS (schema + anchor validation)

## Summary grade

Baseline was B. Rewritten to **A-grade** across all 10 sections: every question is
grounded in a concrete scenario or number, distractors encode real mental-model
failures, explanations refer to distractor content (never to letters), FILL items
that tested label recall were replaced with SHORT items that test operational
interpretation, and learning objectives start with a Bloom's verb and name a
measurable outcome.

## Counts

| Action | Count |
|---|---|
| Questions rewritten (material rewrite: new scenario, number, or distractors) | 47 |
| Questions kept structurally (same intent, tightened prose only) | 0 |
| Questions deleted | 2 |
| Questions added | 2 |
| Net delta | 47 rewrites, 0 net count change |

Details:

- The two FILL items flagged as `trivia_fill` by the audit (`MLOps maturity hierarchy` and `PSI`) were deleted and replaced with A-grade MCQ and SHORT items that test application rather than recall.
- Tier-2 sections (Case Studies, Fallacies and Pitfalls, Summary) kept their 3/3/2-question budgets; Tier-1 sections kept 5-6 questions; no section's count drifted by more than ±1 from baseline.

## Three issue patterns fixed across the chapter

1. **trivia_fill replaced with application items.** Two FILL questions (`hierarchy`
   guessable from the section title; `PSI` pure acronym recall) were removed.
   Their replacements apply the framework to scenarios: an MCQ that requires
   classifying an organization into the four-level MLOps maturity hierarchy from
   its operational signature, and a SHORT that asks when PSI is *operationally
   preferable* to label-based drift metrics and what failure mode PSI does not
   cover. Both replacements lift the learning objective from label recall to
   observable decision-making.

2. **Weak/telegraphed distractors strengthened.** Baseline MCQs contained
   strawman distractors that an informed reader could eliminate without reading
   the chapter (e.g. "avoid standardization so each team optimizes independently",
   "compliance rules force central management above a fixed threshold"). Every
   MCQ distractor was rewritten to encode a real practitioner misconception:
   mistaking velocity gains for cost structure, confusing operational stability
   with optimization paralysis, attributing latency over-runs to quality-priority
   rather than the actual risk profile, treating PSI as a complete drift
   detector, and so on.

3. **Tautological/vague LOs sharpened to Bloom's-verb + observable outcome.**
   Baseline LOs frequently mirrored the stem ("Apply the TCO framework to choose
   optimization priorities" → "Prioritize iteration, training, or inference
   optimization by identifying the dominant TCO component in a given operational
   scenario"). Every LO now starts with a concrete Bloom's verb (Analyze,
   Classify, Apply, Evaluate, Synthesize, Justify, Sequence, Select, Compare)
   and names a measurable decision or diagnostic output rather than
   "understanding" a framework.

## One substantial-rework section

**Summary** (`#sec-ml-operations-scale-summary-4d70`) — The baseline MCQ
asked "what is the chapter's central lesson?" which telegraphed the answer
directly from the summary paragraph. Rewrote as a two-organization contrast:
both with 80 models and identical headcount, one with fragmented per-team
operations and 10 percent platform spend, the other with shared infrastructure
and 30 percent platform spend. The rewritten question forces the reader to
apply the superlinear-complexity thesis to diagnose that the higher-platform-
spend organization is actually the lower-total-cost organization because its
marginal cost per additional model is lower; the lower-spend organization is
carrying hidden operational debt as product-team expenses. The SHORT companion
was rewritten to a concrete TCO-structure scenario ($500K infer / $50K train /
$40K iter with three candidate projects) so the reader must identify the
dominant component, pick the matching optimization, and name the conditions
under which the answer should flip. This reshaped the section from recall
synthesis into applied synthesis.

Other notable reworks: the Monitoring section's third FILL was replaced with
the PSI-application SHORT; the Fallacies TF was rewritten from slogan-style
("alert on every metric for every model is bad") to a quantitative binomial
compounding scenario (200 models × 10 metrics × 1 percent false-alarm rate).

## JSON + validator pass status

- **JSON**: syntactically valid, schema-conformant, 10 sections, all `section_id`s
  match `##` anchors in the chapter QMD.
- **Validator**: `validate_quiz_json.py` reports `OK: vol2_ops_scale_quizzes.json
  passes schema + anchor validation`. Zero errors, zero warnings (no
  letter-reference anti-patterns, all MCQ choice counts in 3-5 range, question
  counts within tier windows).

## Anti-shuffle-bug compliance (§10)

Every MCQ explanation refers to wrong choices by *content or concept* (e.g.
"the hardware-multiplexing claim", "the 'wait until 100' view", "the fraud-style
threshold-gated approach"), never by letter. `Option [A-D]`, `Choice [A-D]`,
`Answer [A-D]`, and `([A-D])` patterns are absent from the file.

## Build-up rule compliance (§8)

Prior vocabulary (CI/CD, canary, A/B testing, feature store, drift, p99 latency,
GPU utilization, NCCL, roofline-adjacent reasoning) is used without
redefinition. Questions whose baseline purpose was "what does X stand for" or
"which metric is called Y" have been converted to application tests that
assume the term is already known.
