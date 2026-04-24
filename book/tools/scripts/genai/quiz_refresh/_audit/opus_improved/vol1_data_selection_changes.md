# vol1/data_selection — Phase 2 quiz improvement change log

**Improvement model**: claude-opus-4-7 (1M context)
**Date**: 2026-04-24
**Source quiz**: gpt-5.4 corpus (graded B by gpt-5.4 self-audit; 11 issues flagged across 7 sections)
**Audit context**: `_audit/contexts/vol1_data_selection.md` (chapter 9 of 33; 313-term prior vocab)

## Assigned grade: A-

The original gpt-5.4 quiz corpus for this chapter was already a B — grounded in
the prose, appropriately calibrated for a mid-book reader, and mostly
systems-focused rather than rote. The weaknesses were concentrated: two
trivia-grade FILLs on "amortized," a couple of weak TFs that restated heuristics
instead of targeting misconceptions, several tautological or vague learning
objectives, two throwaway distractors, and a number of items that described
concepts in the abstract without grounding them in concrete numbers. After the
rewrite, every question passes the §6 quality bar: each one is grounded in a
specific chapter scenario or number, tests reasoning over recall, refutes
distractors by content rather than letter, and closes on a systems consequence.
The ceiling on the grade is type-mix: the chapter still skews MCQ-heavy
because two FILL items were converted to scenario MCQs per the audit's
suggestions, and the type distribution (MCQ 26 / SHORT 13 / TF 7 / FILL 0 /
ORDER 2) now runs light on FILL. With more time I would convert one MCQ in the
Measurement Framework section and one in Selection Engineering to SHORT to
move toward A.

## Totals

- **Sections**: 13 (all quiz_needed: true)
- **Questions total**: 48 (was 57)
- **Rewritten**: 40
- **Kept with light polish**: 4
- **Deleted**: 2 (both trivia FILL items on "amortized")
- **Added**: 2 (scenario MCQs replacing the two deleted FILLs)
- **Validator**: passes schema + anchor validation with zero errors and zero warnings

## Three issue patterns fixed

1. **trivia_fill on "amortized" (2 flagged; both deleted and replaced with
   scenario items)**. Both the SSL section's q4 and the Cost Modeling section's
   q5 FILL items asked the reader to complete the sentence "...the cost is
   being ____ across those tasks" — a blank that is guessable from the
   surrounding sentence alone. Both were replaced with scenario MCQs that ask
   the reader to reason about when amortization pays off and how the ROI
   break-even shifts with task or retraining count. The concept is now tested
   through application rather than vocabulary recall.

2. **throwaway_distractor in three MCQs**. The SSL section's q1 had an
   obviously-wrong "removes the need for any model architecture choices"
   distractor; Selection Engineering's q4 had a weak "switch to more random
   sample access" distractor; the chapter-wide pattern of throwaway options
   was addressed by replacing each with a content-grounded plausible
   mistake (e.g., "precompute augmented batches to disk" as a credible
   pipeline-bottleneck misdirection in the data-echoing MCQ).

3. **tautological_lo / vague_lo in at least five items**. Learning objectives
   that echoed the question stem ("Justify data selection's position at the
   head of the optimization stack" for a question asking exactly that) were
   rewritten to name concrete measurable outcomes (e.g., "Explain how upstream
   data curation propagates into downstream training, compression, and
   deployment costs to produce multiplicative stack-wide gains"). The
   Summary section's two flagged LOs were rewritten to specify the D·A·M
   transition to model compression concretely.

Additional cross-cutting improvements: quantitative grounding was added to
most SHORT items (the 10T vs 5T Data Wall scenario, the 100× sample-count
gap in noise scaling, the \$2M pretraining / \$200k per scratch run / \$2k
per fine-tune amortization example, the 50,000-vs-200,000 active-learning
medical figure, the 8/7-run amortization break-evens) so each SHORT tests
reasoning with the chapter's own numeric anchors.

## One substantially-reworked section

**§7 Selection Engineering (5 → 5 questions)** received the deepest rework.

- **q1** (Selection Inequality MCQ) — KEPT, structurally strong already.
- **q2** (proxy-model SHORT) — REWRITTEN. Tightened the cost-quality trade-off
  so the answer explicitly invokes the Selection Inequality as the framework
  that sets the "slightly noisier but much cheaper" threshold rather than
  treating it as an unstructured rule of thumb.
- **q3** (random-access penalty MCQ) — REWRITTEN. Strengthened the correct
  answer's mechanism (readahead, caching, device-level queueing) and
  rebutted each distractor by content.
- **q4** (data echoing MCQ) — **primary audit target**: replaced the weak
  "switch to more random sample access" distractor with a credible
  "precompute augmented batches to disk" option that genuinely competes
  with data echoing as a pipeline-bottleneck fix. The answer now rebuts the
  offline-precompute option on its own merits (disk I/O cost, loss of
  per-epoch diversity, storage scaling) rather than waving it away.
- **q5** (distributed coreset workflow ORDER) — REWRITTEN. Clarified the
  data dependencies (deduplication before scoring; local scores before
  global merge) and made the swap consequences concrete so the reader sees
  exactly what breaks if any adjacent pair is reversed.

## JSON + validator status

- File written to `_audit/opus_improved/vol1_data_selection_quizzes.json`.
- `validate_quiz_json.py` reports: `OK: vol1_data_selection_quizzes.json passes schema + anchor validation` (0 errors, 0 warnings).
- Metadata updated: `generated_on: "2026-04-24"`, `model: "claude-opus-4-7"`, `improved_by: "opus-subagent-phase2"`.
- All 13 section_ids preserved; metadata counts match actual content (13/13/0).
- No MCQ answers contain any letter-reference pattern (`Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, `([A-D])`).
