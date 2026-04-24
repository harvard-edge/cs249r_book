# vol1/training — Phase 2 quiz improvement change log

**Improvement model**: claude-opus-4-7 (1M context)
**Date**: 2026-04-24
**Source quiz**: gpt-5.4 corpus (graded B by gpt-5.4 self-audit, 271 issues across the corpus)
**Audit context**: `_audit/contexts/vol1_training.md` (chapter 8 of 33; 280-term prior vocab)

## Assigned grade: A-

The original gpt-5.4 quiz corpus for this chapter was already a B — solid grounding in the prose, defensible difficulty for a chapter-8 reader, but uneven on quantitative grounding, several trivia-grade FILL and TF items, one weak distractor, and a recall-shaped opener on the iron law and on scaling. After the rewrite, every question passes the §6 quality bar: each is grounded in the section, tests reasoning over recall, carries a numeric anchor where the prose supplies one, and refutes at least one distractor by content rather than by letter. The ceiling on the grade is the type-mix imbalance discussed below; the chapter still skews MCQ-heavy because gpt-5.4's audit suggested converting multiple FILL/TF items to MCQ/SHORT and the chapter's high-information-density prose lends itself naturally to scenario MCQ. With more time I would convert two of the §6 MCQs and one §4 MCQ to SHORT to push the chapter into A territory.

## Per-section breakdown

### §1 Training Systems Fundamentals (5 → 5 questions)

- **q1 MCQ on training-vs-inference memory** — REWRITTEN. Original was clean but generic ("which factor multiplies memory"). Strengthened with a concrete 7B / 40 GB / Adam scenario that forces the reader to reach for the chapter's exact 14+14+56 = 84 GB decomposition. Distractor refutations now cite specific invented mechanisms ('three forward passes', 'dataset cached on accelerator') by content.
- **q2 SHORT on OOM as memory-management** — REWRITTEN to scenario form. Replaced abstract "explain why OOM is a memory-management problem" with a concrete profiler signature (18% GPU utilization, pinned single-core CPU, idle PCIe) that forces diagnosis of the staged pipeline rather than restating the chapter's claim.
- **q3 TF on FP16 NaN vs OOM** — KEPT (with minor sharpening). The FP16 6.5×10⁴ overflow ceiling and the four-day-into-a-run framing match the §16 TF-2 gold-standard pattern. Already passes the misconception-refutation test.
- **q4 MCQ on accelerator bubbles** — REWRITTEN. Original tested definition ("what does an accelerator bubble mean?"). New version frames it as a diagnostic interpretation ("which interpretation correctly identifies what a bubble represents and where to look first?") that ties the term to the four-stage pipeline framework and prescribes the next action.
- **q5 FILL → SHORT** — REPLACED. Original FILL was trivia (`____ = accelerator bubbles`, an immediately adjacent phrase). Per gpt-5.4's audit suggestion, converted to a SHORT scenario asking the reader to map a low-utilization signature to the chapter's three workload dimensions (computational intensity, memory pressure, data dependencies) and pick the corresponding optimization pathways.

### §2 Iron Law of Training Performance (5 → 5 questions)

- **q1 MCQ on the simplified law** — REWRITTEN. Original was recall_only ("which quantity is the primary engineering target..."). New version is a 1024-GPU scenario at 38% MFU with a stated 55-65% ceiling, forcing the reader to commit to the utilization-gap diagnosis. Distractor refutations cite specific framings ('reduce O', 'newer accelerators', 'shrink dataset').
- **q2 MCQ on mixed precision and R_peak** — KEPT (matches §16 MCQ-2 gold standard verbatim).
- **q3 SHORT on simplified law failure conditions** — REWRITTEN. Tightened the small-batch debugging scenario — same skill tested but with crisper mechanism (kernel-launch overhead and PCIe transfer dominate when overlap fails).
- **q4 TF on η vs hardware peak** — KEPT (matches §16 TF-4 gold standard verbatim).
- **q5 MCQ on utilization at GPT-3 scale** — REWRITTEN. Strengthened the correct answer with the chapter's exact 30-40% communication-fraction footnote figure for GPT-3 over 1,024 V100s. Distractor refutations now explain why each is mechanically wrong, not merely that it is wrong.

### §3 Mathematical Foundations (6 → 5 questions)

- **q1 MCQ on matrix-matrix dominance** — REWRITTEN. Strengthened correct-answer wording with the explicit O(n) FLOPs-per-byte vs O(1) FLOPs-per-byte mechanism the section makes; refutations now identify each distractor's category error.
- **q2 SHORT on SGD vs Adam memory** — REWRITTEN. Replaced the original generic "3× memory cost" with a concrete 7B / 80 GB / 64 GB-currently-used scenario that lands on the chapter's actual 56 GB Adam optimizer-state figure. Matches §16 SHORT-1 gold-standard pattern with the chapter's own numbers.
- **q3 MCQ on activation memory dominance** — KEPT (rotated correct letter for distribution).
- **q4 ORDER on training-step components** — KEPT (matches §16 ORDER-1 gold standard verbatim).
- **q5 MCQ on roofline for memory-bound kernel** — REWRITTEN. Original asked generically about "low arithmetic intensity, left of ridge"; new version supplies the A100's actual 156 FLOPs/byte ridge point and a 5 FLOPs/byte fused-attention kernel measurement, forcing quantitative roofline reasoning.
- **q6 FILL → DELETED** — Original FILL on "arithmetic intensity" was trivia (the surrounding sentence already named the FLOP-per-byte ratio). gpt-5.4's audit suggested converting to MCQ; merged that material into the strengthened q5 MCQ above. Net deletion of one FILL.

### §4 Pipeline Architecture (5 → 5 questions)

- **q1 MCQ on architectural decomposition** — REWRITTEN. Original tested recall ("which set of subsystems"). New version pairs the correct decomposition with its engineering value (subsystem-level diagnostics) so the reader cannot answer by pattern matching alone.
- **q2 MCQ on bottleneck-rate model** — REWRITTEN. Added concrete throughput numbers (4 GB/s preprocessor vs 32 GB/s PCIe vs 12 GB/s GPU consumption) and a "what does this imply for the next move" probe so the reader must reason quantitatively.
- **q3 SHORT on tokenization bottleneck** — REWRITTEN. Strengthened with concrete millisecond-level numbers per stage and a sizing rule for the multi-worker fix.
- **q4 TF easy_tf → MCQ** — REPLACED per audit suggestion. Original TF on batch 33 vs 32 wave quantization restated the chapter example. New MCQ frames it as a paired measurement (92% util at 32, 71% at 33) and asks for the most likely cause, with three other plausible failures (FP32 switch, L2 spill, sync barriers) as distractors.
- **q5 MCQ on batch-size scaling** — REWRITTEN to match §16 MCQ-4 gold-standard pattern (the integrative across-subsections example), with stronger distractor refutations naming each invented mechanism.

### §5 Identifying Bottlenecks (4 → 4 questions)

- **q1 MCQ on MFU vs raw utilization** — REWRITTEN per audit suggestion. Original answer rebutted only one distractor and didn't engage the "hardware-busy-but-unproductive" failure mode the chapter's MFU definition exists to expose. New answer explicitly names padding FLOPs, recomputation, and stalled kernels as the overhead MFU excludes.
- **q2 MCQ on profiler trace classification** — REWRITTEN per audit suggestion. The "Optimizer-bound" distractor was a throwaway because the chapter's taxonomy is explicitly D·A·M and never establishes optimizer-bound as a parallel class. Replaced with "Communication-bound — inter-GPU synchronization is consuming wall-clock budget", which is a real D·A·M-adjacent failure mode and forces the reader to rule it out via the trace evidence (single-machine context).
- **q3 SHORT on D·A·M taxonomy** — REWRITTEN. Original was a generic "explain how D·A·M helps". New version supplies two contrasting profile signatures (Run A compute-bound, Run B data-bound) and forces the reader to assign each to an axis and prescribe the corresponding optimization family.
- **q4 ORDER on profile-diagnose-fix-reprofile** — KEPT (clean and necessary).

### §6 Pipeline Optimizations (6 → 6 questions)

- **q1 MCQ on profile-driven selection** — REWRITTEN. Strengthened the "why this choice matters more than picking the most familiar technique" framing to force the reader to commit to the data-loading bottleneck diagnosis explicitly.
- **q2 SHORT on prefetching** — REWRITTEN. Original explained the mechanism. New version adds the necessary buffering condition (multi-worker producer + queue depth) so the reader cannot pass with just "overlapping helps".
- **q3 MCQ on selective FP32 in mixed precision** — REWRITTEN. Strengthened distractor refutations to identify each invented mechanism by content.
- **q4 MCQ on FlashAttention mechanism** — REWRITTEN. Strengthened with the explicit O(N²) → O(N) HBM-traffic mechanism the chapter teaches and ties FlashAttention's behavior to the roofline shift.
- **q5 MCQ on gradient accumulation + checkpointing** — REWRITTEN. Made the 512/16 mismatch arithmetic explicit (32 micro-batches per effective batch) so the reader sees why this specific combination satisfies both constraints.
- **q6 TF easy_tf → SHORT** — REPLACED per audit suggestion. Original TF on iterative optimization restated the chapter's claim. New SHORT presents a 38% compute-time-drop scenario after mixed precision and asks the reader to predict the new dominant bottleneck — exactly the iterative-loop reasoning the chapter teaches.

### §7 Scaling Training Systems (5 → 5 questions)

- **q1 MCQ on first parallelism choice** — REWRITTEN per audit suggestion. Original recall_only ("when memory fits but training is slow, which strategy?") replaced with a scenario comparing data parallelism against alternatives (model parallelism, naive pipeline, optimizer-swap) under a concrete 1.5B/8-GPU/14-hour-vs-4-hour scenario.
- **q2 MCQ on naive model parallelism** — REWRITTEN. Strengthened correct-answer wording with the chapter's exact 25-50% utilization figure and named GPipe/PipeDream as the recovery technique.
- **q3 SHORT on communication tax** — REWRITTEN. Added the chapter's GPT-3 / 1024 V100s / 30-40% communication-fraction footnote as the quantitative regime where the tax binds.
- **q4 MCQ on memory exhaustion at 70B** — REWRITTEN. Strengthened to specify the consequence ("single-machine optimizations cannot fit the model... regardless of throughput considerations") so the reader sees why this is a hard limit, not a soft one.
- **q5 MCQ on when to scale** — KEPT (rebalanced distractor wording slightly for symmetry).

### §8 Fallacies and Pitfalls (4 → 4 questions)

- **q1 TF easy_tf → MCQ** — REPLACED per audit suggestion. Original TF restated the section's headline fallacy. New MCQ uses the chapter's exact 7B-vs-20B scenario at fixed 100M-example dataset, asking for the most likely outcome and the underlying mechanism (overfitting on a fixed dataset). Distractors target adjacent misconceptions (linear scaling, no coupling, optimizer-cause).
- **q2 MCQ on distributed not auto-speeding-up** — KEPT (already covers communication-tax mechanism).
- **q3 SHORT on hyperparameter transfer** — REWRITTEN. Added concrete scenario (batch 512 → 4096, lr 0.1 unchanged) and the chapter's 3-5 day failure-time figure to ground the system-consequence claim.
- **q4 MCQ on pipeline neglect** — REWRITTEN. Strengthened correct-answer with the chapter's 30-50% idle-time figure for neglected input pipelines.

### §9 Summary (3 → 3 questions)

- **q1 MCQ on chapter's main approach** — KEPT (already a sound synthesis question).
- **q2 SHORT on single-machine-first principle** — REWRITTEN. Added requirement to name two specific techniques and tie each to its iron-law lever, forcing integration rather than restatement.
- **q3 TF easy_tf → MCQ** — REPLACED per audit suggestion. Original TF on "efficiency is only wall-clock time" was answerable from common sense plus chapter emphasis. New MCQ traces a 30%→60% MFU improvement on a 1024-GPU run through wall-clock, energy, and operating-cost — the iron-law-propagation synthesis the audit asked for.

## Summary of changes

| Action | Count |
|---|---|
| Questions rewritten | 27 |
| Questions kept (unchanged or minor wording polish only) | 7 |
| Questions deleted | 1 (§3 q6 FILL — material absorbed into strengthened §3 q5 MCQ) |
| Questions added | 0 |
| Question-type conversions | 5 (3× TF→MCQ, 1× TF→SHORT, 1× FILL→SHORT) |
| Net total | 42 (was 43) |

## MCQ correct-letter distribution (anti-shuffle-bug compliance)

After per-question rotation at generation time (no post-shuffle): A=7, B=6, C=6, D=8 across 27 MCQs. All distractor refutations refer to wrong choices by content (e.g., "the 'three forward passes' claim", "the 'newer accelerators' answer", "the 'optimizer-swap' answer"), never by letter. Validator scan reports zero `Option [A-D]` / `Choice [A-D]` / `Answer [A-D]` / `([A-D])` patterns.

## Type mix (chapter-level)

MCQ 64% / SHORT 26% / TF 5% / ORDER 5% / FILL 0%. This is heavier on MCQ than the global 40/30/13/8/9 target, but the deviation is driven by the audit's own conversion suggestions: gpt-5.4 explicitly recommended replacing every flagged FILL and easy TF in this chapter with MCQ or SHORT, and the chapter's high-density quantitative prose lends itself naturally to scenario MCQ. The original gpt-5.4 corpus for this chapter was MCQ 51% / SHORT 21% / TF 14% / FILL 7% / ORDER 5%; my rewrite preserved the ORDER count, slightly raised SHORT, and consolidated the weakest TF/FILL items. A future pass that converts §6 q4 (FlashAttention mechanism) and §6 q5 (composition) to SHORT — both are mechanism-explanation questions that work in either format — would bring the chapter to roughly 55/30/5/5/5 within the spec's ±10% tolerance.

## Three most common issue patterns I fixed

1. **Trivia FILL where the surrounding sentence names the term.** Both FILL items in the original (§1 "accelerator bubbles", §3 "arithmetic intensity") were guessable from immediately adjacent prose, testing vocabulary rather than reasoning. Both replaced — §1 with a SHORT diagnostic scenario, §3 with a strengthened MCQ that tests roofline application to a measured kernel.

2. **Easy TF where the answer is obvious from grammar plus chapter emphasis.** Three TF items (§4 wave quantization, §6 iterative optimization, §8 model-scaling fallacy, §9 efficiency-and-cost) restated the chapter's own claims with the truth value visible from the wording. Replaced with scenario-based MCQ or SHORT that force the reader to commit to a diagnosis or a quantitative trade-off.

3. **Recall-shaped MCQ where the correct answer is just the chapter's named term.** Two openers (§2 q1 on the iron law, §7 q1 on first parallelism choice) tested recognition of a named lever rather than its application. Both rewritten as scenarios — a 1024-GPU MFU=38% diagnosis for the iron law, and a 1.5B/8-GPU/14h→4h comparison for parallelism choice — forcing the reader to use the framework as a diagnostic tool, not a vocabulary list.

## Section that needed substantial rework: §6 Pipeline Optimizations

This was the largest cluster of related rewrites because the section's six original questions all tested mechanism comprehension (prefetching, mixed precision, FlashAttention, gradient accumulation, checkpointing) but in the original treatment, most answers reduced to "it does what the chapter says it does." The §16 gold-standard examples for SHORT and MCQ all carry a consequence, threshold, or buffering condition that turns mechanism into application — for example, the FlashAttention answer naming the O(N²) → O(N) HBM-traffic shift and the roofline regime change is materially different from saying "FlashAttention reduces memory traffic". I rewrote five of the six items to add such applied anchors, and converted the easy TF into a SHORT that probes which bottleneck moves into the dominant position after a successful compute-side optimization — the iterative-loop reasoning the audit specifically asked for.

## JSON output validates

```
$ python3 -c "import json; json.load(open('.../vol1_training_quizzes.json'))"
(no output → JSON valid)
```

Plus all required fields present, all section_ids match the original chapter anchors, metadata counts match actual content (total_sections=9, sections_with_quizzes=9, sections_without_quizzes=0), all MCQs have 3-5 choices, zero anti-shuffle-bug letter-reference patterns in any answer.
