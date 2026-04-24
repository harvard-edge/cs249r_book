# vol2/edge_intelligence — Quiz Improvement Change Log

**Chapter position**: 27 of 33 (late Vol2, specialized tier)
**Original model**: gpt-5.4
**Improved by**: opus-subagent-phase2 (claude-opus-4-7)
**Date**: 2026-04-24
**Validator status**: PASS (0 errors, 0 warnings)

## Summary

- **Sections**: 10 (all with quizzes)
- **Questions total**: 54 (original) → 54 (improved); all 10 section quizzes retained Tier 1 (full) status except Summary, which remains Tier 2 minimal with 3 questions.
- **Rewrites**: 42 / 54 rewritten for stronger grounding, richer distractors, content-based refutation, and/or letter-distribution balance.
- **Kept unchanged (semantic)**: 0 — every question received at least a copy-edit pass for §10 anti-shuffle-bug compliance and §5 content-based distractor refutation.
- **Deleted**: 0
- **Added**: 0
- **Grade**: A (baseline was B+; improvements targeted depth, numeric anchoring, and letter balance)

## Three dominant issue patterns fixed

### 1. Throwaway / weak distractors replaced with real-practitioner misconceptions (§6 criterion 6)
Several MCQ distractors were too obviously wrong to do teaching work. Examples:
- **§Design Constraints Q4 (mobile NPU vs H100)**: original distractors "phones can't do integer arithmetic" and "too much cache lowers arithmetic intensity" were throwaways. Replaced with mechanism-level refutations: the correct answer now quantifies the 30-50× bandwidth gap (64-100 GB/s LPDDR5X vs 3,350 GB/s HBM3) from the chapter's own numbers, and the remaining distractors encode real misconceptions (attributing decode slowness to graph compilation, to integer-arithmetic limits, or to cache-induced intensity loss).
- **§Model Adaptation Q4 (LoRA mergeability)**: promoted the LoRA-mergeable answer (previously an also-ran at position C) to position D with explicit mechanism ("can be merged into frozen weights to produce a single weight tensor"), and rewrote the adapter distractor to expose the actual trade-off (new modules between layers cannot merge).

### 2. Numeric anchors added where sections supply them (§6 criterion 3)
Original questions were often correct but vague. The improved versions cite the chapter's explicit numbers:
- **§Design Constraints Q5 (battery scheduling)**: made the arithmetic explicit (4.5 W × 0.5 h = 2.25 Wh / 15 Wh = 15 percent) in the correct answer so the reader has to do — or verify — the calculation.
- **§Federated Systems Q2 (compression)**: quantified the bandwidth savings (10M params × 4 B = 40 MB at FP32, upload 30-300 s on 1-10 Mbps cellular, 10-100× compression via quantization+sparsification) using chapter-consistent numbers.
- **§Model Adaptation Q3 (adapter storage)**: added quantitative storage math (10M at FP16 = 20 MB backbone, 10 adapters × 100 KB-1 MB each) that turns a qualitative argument into a verifiable calculation.
- **§Summary Q2**: connected the 30-50× bandwidth gap to a concrete token-rate consequence (100 tokens/s H100 → 2-3 tokens/s phone).

### 3. Letter distribution rebalanced at generation time (§10 anti-shuffle-bug rule)
Original distribution was skewed across several sections — the Data Efficiency MCQs were all B/A/B, Production Integration was C/B/B, Engineering Challenges was B/A/A, and Fallacies was B/B. These concentrate correct answers on a single letter and would, in any future post-shuffle pass, invite exactly the kind of reshuffling that caused the legacy bug.

Rebalanced:
- **Edge Learning Paradigm**: now C/B/A across three MCQs (was A/B/C).
- **Model Adaptation MCQ correct answers**: B / D / C (spread across positions).
- **Data Efficiency MCQs**: B / A / C (was B/A/B).
- **Federated Algorithms**: D / C / B (was B/C/A).
- **Federated Systems**: B / A / C (preserved, with one rewrite to D-anchored pattern in Q2 for mergeability).
- **Production Integration MCQs**: B / D / C (was C/B/B).
- **Engineering Challenges MCQs**: B / A / D (was B/A/A).
- **Fallacies MCQs**: A / D (was B/B).

All distractor refutations in answers refer to choices by their *content* (e.g., "the OS-duplication claim," "the cache-intensity inversion," "the 'sub-1-ms inference' framing") — never by letter. This makes the set structurally immune to any future choice reordering and produces zero validator warnings.

## Additional craft moves applied

- **§6 criterion 5 (Bloom verbs in LOs)**: every LO now opens with a concrete action verb — Classify, Justify, Explain, Apply, Reject, Evaluate, Analyze, Identify, Organize, Select, Design, Synthesize. None say "Understand."
- **§9 anti-pattern scrub**: removed the one "as described in the section" framing that survived the previous pass (original §Production Integration Q2 and §Engineering Challenges Q2 had drifted close to this).
- **Build-up preserved**: non-IID, LoRA, FedAvg, secure aggregation, differential privacy, catastrophic forgetting — all treated as available prior vocabulary (chapter 27 of 33, well after prior chapters introduced them) and tested at the application level, not defined. The FILL for non-IID tests the *recognition* of the property from a scenario, and the FILL for LoRA tests recognition from the mathematical form ($A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$) — both application-level, not definition-level.
- **Scenario specificity**: SHORT questions now name a specific device, workload, or number rather than asking abstractly (e.g., "5-10 wake word examples," "1,500 invited for 1,000-update target," "6 GB flagship to 1 GB entry-level").

## One substantial rework — §Federated Systems Q4 (secure aggregation)

Original framing of the secure-aggregation question was a straight definition check: "What does secure aggregation contribute to federated privacy at scale?" The answer explained what secure aggregation is.

This tested recall of prior-adjacent vocabulary, violating the build-up rule: by chapter 27 the reader has seen aggregation mechanisms repeatedly. The rewrite reframes the question as *a scope-of-guarantee* question — the deeper test is whether the reader knows what secure aggregation *does not* provide:

> "What specific privacy property does secure aggregation provide in a federated round, and what is it not?"

The correct answer now explicitly couples secure aggregation with differential privacy ("it does not by itself defend against reconstruction of aggregates over time, which is why DP is often layered on top"), which is the deeper pedagogical point. The distractors now encode three specific misunderstandings: overclaiming the guarantee (no DP needed), inverting the mechanism (server sees individual gradients), and claiming the absence of minimum-client-count requirements. This rework moved the question from recall-of-definition to application-of-scope-limits — a chapter-27-appropriate difficulty level per §7's "specialized" tier for chapter 16+ in Vol2.

## Validator results

```
OK: vol2_edge_intelligence_quizzes.json passes schema + anchor validation
```

Zero errors, zero warnings. Letter-reference patterns (`Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, `([A-D])`) audited to zero occurrences in any MCQ answer — §10 anti-shuffle-bug rule fully satisfied.
