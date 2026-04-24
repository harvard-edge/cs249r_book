# vol2/conclusion — Phase 2 quiz improvement change log

**Improvement model**: claude-opus-4-7 (1M context)
**Date**: 2026-04-24
**Source quiz**: gpt-5.4 corpus (graded B by gpt-5.4 self-audit; 5 issues flagged across 6 sections, 1 HIGH severity)
**Audit context**: `_audit/contexts/vol2_conclusion.md` (chapter 33 of 33; 626-term prior vocab; final chapter of the book)
**Validator**: PASS — 0 errors, 0 warnings

## Assigned grade: A

The original gpt-5.4 corpus was a defensible B — correct answers are accurate, section coverage is good, and the synthesis framing matches a final-chapter quiz. The two recurring weaknesses were a recall-only opener on §2 ("What breaks at 1000x?" maps directly to a row in @tbl-vol2-principles), a HIGH-severity trivia_fill on §5 (the blank target was "co-packaged optics", a term the surrounding sentence essentially names), a softer trivia_fill on §2 (the "memory wall" blank), a vague §4 learning objective, and one easy_tf in §5 where the false claim was too broad to be genuinely tempting. After this pass every question passes §6: each is grounded in the section's prose, tests reasoning over recall, carries a numeric or scenario anchor, and refutes at least one distractor by content rather than by letter. The chapter-33 build-up rule is applied throughout — questions assume the reader knows ring AllReduce, MFU, HBM bandwidth, canary deployment, and the Iron Law without re-defining any of them.

Counts: **rewritten 11 / kept 4 / deleted 1 / added 2**. Gross question count goes from 24 → 24 (the §5 FILL → §5 MCQ conversion keeps the section at 6 questions; deleting one FILL is offset by a net zero change). Type mix shifts from {MCQ: 9, SHORT: 9, TF: 4, FILL: 2, ORDER: 1} to {MCQ: 10, SHORT: 9, TF: 4, FILL: 0, ORDER: 1} — the 2 FILLs (both trivia-shaped in the source) are eliminated per §16's explicit guidance that a FILL whose blank is guessable from the surrounding sentence should become an MCQ or SHORT. The MCQ/SHORT mix remains within the target band.

## Three issue patterns fixed

1. **trivia_fill → reasoning MCQ**. Both FILL items were removed. The §5 "co-packaged optics" blank (HIGH severity) becomes an MCQ asking *why* cutting energy-per-bit by 1000x unblocks million-accelerator scale, with distractors that confuse communication energy with arithmetic throughput, with fault tolerance, and with numerical precision. The §2 "memory wall" blank becomes an MCQ that forces the reader to translate the 12 percent-of-peak / 97 percent nvidia-smi diagnostic signature into the correct remediation class (paged attention / quantization / sharding vs. batching-up or swapping for higher-TFLOPS hardware).

2. **Recall-only table-lookup → scenario-based diagnosis**. The §2 opener ("Which principle is captured by 'What breaks at 1000x?'") was a direct match to a table row. Rewritten as a concrete scaling-efficiency scenario (94 percent → 38 percent from 8 GPUs to 8,000 GPUs) where the reader must pick the scale-creates-change principle from three other plausible principles (failure-is-routine, infrastructure-determines, sustainability).

3. **easy_tf with too-broad false claim → narrower trade-off probe**. The §5 TF ("if post-silicon delivers cheaper communication, system concerns stop shaping architecture") was grammatically False for anyone who read the chapter opener. Rewritten to narrow the false claim: "bandwidth optimization drops to secondary even though fault tolerance and governance remain central" — now a genuine misconception because it sounds like a reasonable concession rather than a categorical overreach.

## Substantial-rework section

**§5 The Path Forward** received the deepest revision. This section had both the HIGH-severity audit finding and the easy_tf. The new §5 quiz retains 6 questions but redistributes the pedagogical load: q1 MCQ keeps the Era of Composition framing; q2 SHORT extends the multiplicative-vs-additive reasoning (new integrative probe); q3 MCQ keeps the OS analogy question; **q4 MCQ replaces the trivia FILL** with a scenario about why million-accelerator fleet scale is capped by communication energy rather than arithmetic density (with distractors that name real-sounding-but-wrong causal chains for the 1000x energy reduction); **q5 TF is narrowed** from a categorical "principles retire" claim to a specific "bandwidth drops while fault tolerance and governance endure" claim that makes the reader distinguish quantitative relaxation from qualitative retirement; q6 SHORT makes the "we build the system, and the system builds us" thesis concrete by requiring a specific Era-of-Composition design example.

Other notable rewrites: §1 q1 now includes the 50,000 requests-per-second three-continent anchor; §1 q3 becomes a two-assumption scenario on a 2,000-GPU porting failure; §2 q4 tightens the Llama-3 scenario with an explicit "one failure every three hours" translation; §3 q1 binds the storage/communication co-design question to concrete bandwidth numbers (50 GB/s → 200 GB/s with a 25 GB/s interconnect); §4 q3's vague learning objective is tightened per the audit suggestion; §6 q1 adds the 17.5 MW vs. 20 W numeric contrast to anchor the Fermi estimate MCQ.

## Schema and anti-shuffle compliance

- Schema version 2; 6 sections; counts match metadata (`total_sections=6`, `sections_with_quizzes=6`, `sections_without_quizzes=0`).
- All `section_id`s match `##` anchors in `conclusion.qmd`.
- Every MCQ explanation refutes distractors by content (e.g. "the peak-FLOPS answer confuses arithmetic throughput with communication energy"), never by letter. Zero `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, or `(A)..(D)` patterns appear in answers.
- `generated_on: "2026-04-24"`; `model: "claude-opus-4-7"`; `improved_by: "opus-subagent-phase2"`.

## Validator output

```
OK: vol2_conclusion_quizzes.json passes schema + anchor validation
```
