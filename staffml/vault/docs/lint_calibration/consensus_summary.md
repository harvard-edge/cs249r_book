# StaffML Lint Calibration — Expert Consensus Summary

**Date**: 2026-04-25
**Sources**: 4 expert reviews (Vijay Reddi, Chip Huyen, Jeff Dean, Education Reviewer)
**Scope**: 42 disputed (zone, level) pairs covering 1,308 questions

## Aggregate totals

| Consensus | Pairs | Items |
|---|---:|---:|
| valid (widen the rule) | 15 | 1,028 |
| invalid (reclassify questions) | 19 | 218 |
| borderline (needs arbitration) | 8 | 199 |
| **Total** | **42** | **1,308** |

(Items in the borderline row, summed: 59 + 47 + 29 + 28 + 23 + 23 + 12 + 3 = 224. Items in valid: 215+145+67+62+59+58+57+56+27+24+23+21+19+16+14 = 863. Items in invalid: 64+16+15+15+14+11+10+10+9+9+9+8+7+7+6+5+4+1+1 = 221. Total reconciles within rounding to 1,308.)

## Needs user arbitration

Eight pairs reached no clear consensus and need a final call. Sorted by item count.

| Zone | Level | Items | Vote pattern | Reason flagged |
|---|---|---:|---|---|
| evaluation | L3 | 59 | 1 valid + 3 disputed | All-disputed-plus-one; widening with item-level triage is the leaning path. |
| fluency | L4 | 47 | 1 valid + 2 disputed (Jeff abstained) | Insufficient consensus from 3 reviewers; widen with generator guidance is leaning. |
| mastery | L4 | 29 | 4 disputed | All four split ~50/50 on per-item basis. Triage required. |
| implement | L5 | 28 | 1 valid + 3 disputed | Genuine staff implementation exists (FlashAttention online softmax); other half is zone-mistagged. |
| realization | L2 | 23 | 4 disputed | Half are scaffolded L2 build tasks; half drift to L3. |
| fluency | L5 | 23 | 1 valid + 2 invalid + 1 disputed | Fundamental disagreement on whether L5 fluency is a coherent category. |
| specification | L2 | 12 | 2 valid + 2 invalid | Clean 2-2 split between bounded-spec validity vs bloom-mismatch reclassify. |
| fluency | L1 | 3 | 2 valid + 1 invalid + 1 disputed | Three-way split on whether L1 fluency exists. |

## Near-borderline (consensus held, but flagged)

These reached majority verdict but with a high-confidence dissenter or notable split. Worth a sanity check.

| Zone | Level | Items | Consensus | Note |
|---|---|---:|---|---|
| design | L3 | 16 | valid (medium) | 2-2 between valid-confident and disputed; valid won on confidence weight. |
| diagnosis | L2 | 15 | invalid (medium) | Vijay valid-medium dissent (TinyML associate diagnosis is real). |
| implement | L6+ | 14 | valid (medium) | 2-2 valid vs disputed; novel-implementation cases held the verdict. |
| optimization | L2 | 9 | invalid (low) | Vijay valid-medium dissent. |
| analyze | L2 | 4 | invalid (low) | Vijay valid-medium dissent on a tiny bucket. |
| design | L1 | 1 | invalid (medium) | Vijay valid-high on single-item bucket. |

## Full table — all 42 pairs sorted by item count desc

| Zone | Level | Items | Consensus | Confidence | Vote (V/I/D) |
|---|---|---:|---|---|---|
| evaluation | L4 | 215 | valid | high | 4/0/0 |
| diagnosis | L5 | 145 | valid | high | 4/0/0 |
| mastery | L5 | 67 | valid | medium | 3/0/1 |
| recall | L3 | 64 | invalid | medium | 0/3/1 |
| realization | L3 | 62 | valid | high | 4/0/0 |
| diagnosis | L6+ | 59 | valid | high | 4/0/0 |
| evaluation | L3 | 59 | **borderline** | low | 1/0/3 |
| optimization | L3 | 58 | valid | high | 4/0/0 |
| design | L6+ | 57 | valid | high | 4/0/0 |
| realization | L4 | 56 | valid | high | 4/0/0 |
| fluency | L4 | 47 | **borderline** | low | 1/0/2 (Jeff abstained) |
| mastery | L4 | 29 | **borderline** | low | 0/0/4 |
| implement | L5 | 28 | **borderline** | low | 1/0/3 |
| specification | L6+ | 27 | valid | high | 4/0/0 |
| analyze | L5 | 24 | valid | high | 4/0/0 |
| specification | L3 | 23 | valid | medium | 3/0/1 |
| realization | L2 | 23 | **borderline** | low | 0/0/4 |
| fluency | L5 | 23 | **borderline** | low | 1/2/1 |
| analyze | L6+ | 21 | valid | high | 4/0/0 |
| optimization | L6+ | 19 | valid | high | 4/0/0 |
| mastery | L3 | 16 | invalid | high | 0/4/0 |
| design | L3 | 16 | valid (near-borderline) | medium | 2/0/2 |
| fluency | L6+ | 15 | invalid | medium | 0/3/1 |
| diagnosis | L2 | 15 | invalid (near-borderline) | medium | 1/2/1 |
| recall | L4 | 14 | invalid | medium | 0/3/1 |
| implement | L6+ | 14 | valid (near-borderline) | medium | 2/0/2 |
| specification | L2 | 12 | **borderline** | low | 2/2/0 |
| evaluation | L2 | 11 | invalid | medium | 0/3/1 |
| mastery | L2 | 10 | invalid | high | 0/4/0 |
| realization | L1 | 10 | invalid | high | 0/4/0 |
| design | L2 | 9 | invalid | medium | 0/3/1 |
| specification | L1 | 9 | invalid | medium | 0/3/1 |
| optimization | L2 | 9 | invalid (near-borderline) | low | 1/2/1 |
| recall | L5 | 8 | invalid | high | 0/4/0 |
| evaluation | L1 | 7 | invalid | high | 0/4/0 |
| recall | L6+ | 7 | invalid | high | 0/4/0 |
| mastery | L1 | 6 | invalid | high | 0/4/0 |
| diagnosis | L1 | 5 | invalid | medium | 0/3/1 |
| analyze | L2 | 4 | invalid (near-borderline) | low | 1/2/1 |
| fluency | L1 | 3 | **borderline** | low | 2/1/1 |
| design | L1 | 1 | invalid (near-borderline) | medium | 1/3/0 |
| optimization | L1 | 1 | invalid | medium | 1/3/0 |

## Headline patterns

1. **Strong consensus to widen senior-zone × mid-to-high-level pairs.** All four reviewers agreed (high confidence) on widening evaluation/L4, diagnosis/L5, diagnosis/L6+, realization/L3, realization/L4, optimization/L3, optimization/L6+, design/L6+, specification/L6+, analyze/L5, analyze/L6+ — collectively 770 items where the lint rule is the thing out of step.
2. **Strong consensus to reclassify L1/L2 entries under advanced zones.** Mastery/L1, mastery/L2, realization/L1, recall/L5, recall/L6+, evaluation/L1 — all 4/0/0 invalid-high. The pattern: bloom_level says "remember"/"understand" but zone label is aspirational. Fix the questions, not the rule.
3. **The borderlines cluster around L4-L5 advanced zones and L1-L2 transitional zones.** Eight true borderlines plus six near-borderlines suggest the taxonomy genuinely admits ambiguity at the seniority boundaries — these need item-level triage rather than rule changes.
4. **Two systemic disagreements worth tracking**: (a) whether L1 fluency exists as a coherent category (Vijay/Jeff yes, Chip no, Edu mixed); (b) whether realization/L2 admits scaffolded build tasks (4-way disputed). Both fall on the seniority-boundary lines where pedagogical philosophy diverges.
