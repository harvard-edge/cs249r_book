# vol2/performance_engineering — Opus Phase-2 Improvements

**Chapter**: 25 of 33 (Vol2)
**Base input**: `book/quarto/contents/vol2/performance_engineering/performance_engineering_quizzes.json` (gpt-5.4, B grade)
**Output**: A-grade improvements per §6 / §9 / §10 / §16 of the canonical quiz-generation spec
**Validator**: PASS (schema + anchor validation, 0 errors)

## Summary

- **Sections**: 12/12 with quizzes (unchanged)
- **Question counts**: 52 before → 52 after (unchanged; all sections preserved ±1)
- **Rewrites**: 18 substantive question rewrites
- **Kept**: 34 questions unchanged (already A-grade or only minor polish)
- **Added**: 0
- **Deleted**: 0

## High-severity fix: speculative-decoding build-up violation (audit HIGH)

Section `#sec-performance-engineering-speculative` q1 was a definition-style MCQ
for speculative decoding, which is prior vocabulary introduced in earlier
chapters. Per §16 SHORT-4 gold-standard example (which is explicitly drawn
from this chapter), replaced the MCQ with a SHORT that applies the
acceptance-rate gate:

- Scenario: 1B draft + 70B target, 4 tokens/step, 25% acceptance
- Answer covers: amortization math, 75%/25%/50% thresholds, throughput
  direction, tail-latency direction, and the "deployment gate not tuning knob"
  conclusion
- Matches §16 SHORT-4 three-move structure: main answer → quantitative
  example → system consequence

The companion TF (q2 `easy_tf`) was replaced with a sharper misconception
about speculative decoding at compute-saturated batch sizes, and q3 was
rewritten as a new MCQ anchored to a specific profile signature (8% MFU,
88% HBM BW) so the whole section tests application rather than definition.

## Issue patterns fixed

1. **Definitional/recall-only questions reframed as diagnosis or
   application** (affected sections: memory-wall, compilation FILL,
   speculative, moe, measurement-scale, fusion q5). For each, the question
   now forces the reader to reason from a profile signature, a scenario, or
   an iron-law decomposition rather than restate a term already defined in
   prior chapters. This defends against `build_up_violation` and
   `recall_only` audit flags.

2. **MCQ distractors that were secretly plausible tightened or
   rebalanced** (affected sections: memory-wall q1, summary q2, playbook q5
   added). Following §16 MCQ-3, distractors now each encode a distinct
   mental-model failure; no distractor doubles as a correct answer. In
   particular:
   - memory-wall q1: the "higher-bandwidth GPU" distractor that the audit
     flagged as co-correct was replaced with an explicit compute-only
     clock-frequency boost that leaves bandwidth untouched.
   - summary q2: all four pairings now encode specific misattributions
     (fusion→compute, speculation→bandwidth-halving, MoE→synchronization,
     precision→overhead); no throwaway distractors.

3. **Content-based distractor refutation (§10 anti-shuffle) applied
   throughout**. Every MCQ answer now refutes wrong options by naming the
   mechanism (e.g., "confuses kernel structure with model architecture",
   "inverts the mechanism") rather than "Option A is wrong". Zero
   `Option/Choice/Answer [A-D]` or bare `([A-D])` letter references appear
   anywhere in the file.

## One substantial-rework section: Mixture of Experts

`#sec-performance-engineering-moe` q1 was a textbook recall MCQ ("why can
MoE have more total parameters without paying the same cost?"). Per the
audit's suggested fix (comparing dense vs. MoE serving costs and asking
which bottleneck shifts), the question was rebuilt around a concrete
dense-vs-MoE comparison (1T dense vs. 370B-total / 37B-active) with
numerical byte-per-decode-step signatures (2 TB vs. 74 GB) drawn from the
chapter's own `MoEEconomics` LEGO cell. The correct answer now requires
iron-law reasoning about which term shrinks, and every distractor encodes
a specific iron-law misreading (hardware-independent intensity,
all-experts-loaded misconception, routing-doubles-weights inversion). The
SHORT companion (q2) was also rewritten to frame benefit and cost in
iron-law terms rather than as bare definitions, and now explicitly ties
the sweet spot (batch-1 latency-critical) to the cost regime
(communication-bound all-to-all at scale).

## Additional rewrites, by section

- **fusion**: q5 FILL (`online`) replaced with SHORT on tile-level vs.
  thread-level kernel programming, addressing chapter-25-appropriate
  systems reasoning (SRAM reuse, fusion expressivity) rather than
  remembering the term `online`.
- **compilation**: q5 FILL (`tile`) replaced with MCQ asking when Triton
  is the right level of the stack vs. a full graph compiler. Distractors
  encode specific misconceptions about Triton's automation, memory
  hierarchy, and Tensor Core usage.
- **measurement-scale**: q1 recast as a 92%-HWU-vs-30%-MFU scenario
  forcing metric selection; the right answer requires naming recomputation
  and pipeline bubbles as the cycles HWU hides but MFU excludes.
- **playbook**: added a new MCQ (q5) that tests sub-multiplicative
  speedups (INT4→3.1x, +FlashAttn→1.4x instead of 1.9x) grounded in the
  bottleneck-shift principle, replacing the weakest throwaway-distractor
  item.
- **Answer explanations throughout**: added explicit mechanism-based
  refutation of at least one distractor per MCQ, plus a "practical
  consequence" closing sentence to every SHORT, matching §16 patterns.

## Metadata changes

- `generated_on`: `2026-04-24`
- `model`: `claude-opus-4-7`
- `improved_by`: `opus-subagent-phase2`
- All other metadata (schema_version, source_file, counts) preserved.

## Validator status

```
OK: vol2_performance_engineering_quizzes.json passes schema + anchor validation
```

Zero anti-shuffle warnings (no `Option/Choice/Answer [A-D]` or bare `([A-D])`
patterns in any MCQ answer).
