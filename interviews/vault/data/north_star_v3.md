# StaffML North Star v3 (Post-Consensus, Definitive)

> **StaffML is the quantitative reasoning benchmark for ML systems engineering
> interviews.** It measures whether a candidate can connect hardware constraints
> to architectural decisions using napkin math, tradeoff analysis, and system
> design — not memorized API signatures or framework trivia.

## The Derivation

The corpus size follows from four principled constraints:

### 1. Topics: ~86 (79 current + 7 Tier A additions)
The minimum spanning set of Staff-level ML systems knowledge.

New topics (consensus of 12 expert reviewers):
1. software-portability (CUDA/ROCm/SYCL/Metal)
2. communication-computation-overlap
3. autograd-computational-graphs
4. chiplet-architecture
5. model-adaptation-systems (LoRA/QLoRA/RLHF infra)
6. disaggregated-serving (prefill/decode split)
7. recommendation-systems-engineering

### 2. Hardware Applicability: ~245 pairs (after corrections)
Each topic applies to a subset of 4 tracks, filtered by physics.

Corrections from reviewer consensus:
- ADD: mixed-precision→tinyml, systolic-dataflow→tinyml+mobile,
  dma-data-movement→mobile, differential-privacy→edge
- REMOVE: datacenter-efficiency→tinyml, 3d-parallelism→tinyml+mobile,
  gradient-synchronization→tinyml

### 3. Cognitive Zones: 12 (11 current + "debug")
4 fundamental skills → 6 compound zones → 1 mastery zone → 1 debug zone.
Debug = recall + implement + analyze (identified by Soumith Chintala).

### 4. Capacity: Variable by Zone × Level
Not a flat constant. Varies by cognitive complexity AND difficulty:

| Zone Type | L1-L2 | L3-L4 | L5-L6+ |
|-----------|:-----:|:-----:|:------:|
| Simple (recall, implement) | 3 | 5 | 5 |
| Complex (analyze, design, ...) | 4 | 6 | 8 |
| Mastery/Debug | 5 | 7 | 10 |

**These are HYPOTHESES until validated empirically** (Patterson's requirement).
Validation: semantic similarity study on overfilled cells.

### Derived Total
~245 pairs × ~50 avg capacity (12 zones × variable) ≈ **12,000-14,000**
Exact number determined by empirical saturation, not formula.

## Five Principles

1. **Quantitative over qualitative**: Every question answerable with numbers
2. **Constraints drive architecture**: Tracks exist because physics differs
3. **Vendor-neutral, physics-first**: Test bandwidth/compute/memory, not APIs
4. **Calibrated to real hiring bars**: L3=textbook, L5=production, L6+=systems-of-systems
5. **Distribution quality over raw count**: Balance score (σ/μ) < 0.5

## What NOT to Do

1. Don't add all 20 proposed topics — cap at 7-10 for v1.1
2. Don't split the cloud track — use metadata (phase, scale_tier)
3. Don't merge mobile and edge — physics differs
4. Don't expand without rebalancing first
5. Don't treat capacity constants as proven — they're hypotheses
6. Don't assume LLM-generated questions are correct — verify

## Execution Sequence (Dependencies Matter)

1. Fix applicability matrix → BEFORE rebalancing
2. Add new topics → BEFORE filling to minimums
3. Add metadata fields → BEFORE rubrics
4. Rebalance (cap at 150, floor at 50) → BEFORE generating more
5. Validate capacity empirically → BEFORE claiming final numbers
6. Vendor audit → AFTER adding software-portability topic

## Convergence Criteria

The north star is stable when:
- No reviewer identifies a missing topic that 3+ others agree with
- Feedback shifts from "missing X" to "improve phrasing of Y"
- Capacity model is empirically validated
- Inter-rater reliability on zones exceeds κ > 0.7

Status: **STRUCTURALLY CONVERGED** (no more structural gaps expected).
Remaining: empirical validation, rebalancing, quality improvement.
