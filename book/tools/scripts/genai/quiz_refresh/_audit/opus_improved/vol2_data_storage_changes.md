# vol2/data_storage — Opus Phase 2 Quiz Improvements

**Chapter position:** 20 of 33 (Vol2, specialized tier — §7 difficulty: production-level, cross-cutting)
**Prior grade (gpt-5.4):** B
**Target:** A-grade per §16 gold-standard worked examples
**Validator status:** PASS (schema + anchor validation OK)

## Summary counts

| Section | Original Qs | Final Qs | Rewritten | Kept as-is | Added | Deleted |
|---|---:|---:|---:|---:|---:|---:|
| `#sec-storage-fuel-line` | 5 | 5 | 5 | 0 | 0 | 0 |
| `#sec-storage-workload-inversion` | 6 | 6 | 6 | 0 | 1 | 1 |
| `#sec-storage-hierarchy` | 6 | 6 | 6 | 0 | 1 | 1 |
| `#sec-storage-pipeline-equation` | 6 | 6 | 6 | 0 | 0 | 0 |
| `#sec-storage-gds` | 5 | 5 | 5 | 0 | 0 | 0 |
| `#sec-storage-economics` | 5 | 5 | 5 | 0 | 1 | 1 |
| `#sec-storage-checkpointing` | 5 | 5 | 5 | 0 | 0 | 0 |
| `#sec-storage-fallacies` | 5 | 5 | 5 | 0 | 0 | 0 |
| `#sec-storage-synthetic-fuel` | 3 | 3 | 3 | 0 | 1 | 1 |
| `#sec-storage-summary` | 3 | 3 | 3 | 0 | 0 | 0 |
| **Total** | **49** | **49** | **49** | **0** | **4** | **4** |

Every question was substantively strengthened; no question passed §6 at A-grade
in its original form (most were B/B+).

## Issue patterns fixed

Three dominant failure patterns from the gpt-5.4 audit and the §16 craft-move
gap were addressed systematically:

### 1. `trivia_fill` → diagnostic scenario MCQ/SHORT

The original quiz leaned on FILLs that asked readers to name a term from a
sentence that nearly defined it. Three explicit trivia_fills were flagged by
the audit and rewritten:

- `#sec-storage-workload-inversion` q5 (`I/O Wall` FILL) → replaced with a
  scenario SHORT asking the reader to diagnose a 1,024→1,280 node utilization
  regression and identify the confirming measurements.
- `#sec-storage-hierarchy` q5 (`small file problem` FILL) → replaced with a
  diagnostic MCQ: given a profile signature of collapsed throughput with idle
  data disks, reason from metadata-saturation to the shard-aggregation fix.
- `#sec-storage-economics` q4 (`delivery` FILL) → replaced with a cost-recurrence
  diagnosis SHORT quantifying how a re-fetch workflow tripled egress.

### 2. Weak or missing distractor refutation (audit `missing_explanation` + §10)

Multiple MCQ explanations stopped at restating the correct answer. Every MCQ
was rewritten so the explanation names the mechanism by which each plausible
distractor is wrong, refuting by content not by letter (§10 anti-shuffle-bug).
Specific examples:

- `#sec-storage-fuel-line` q4 (GPU upgrade → utilization drop): new explanation
  addresses fabric-saturation, checkpoint-size, and prefetch-decrease
  distractors by their mechanism.
- `#sec-storage-pipeline-equation` q3 (prefetch depth): now explicitly addresses
  the audit's q2 gap — sizing against median leaves no headroom for P99 spikes
  that fire reliably at scale.
- `#sec-storage-gds` q1 (GDS architecture): throwaway POSIX-over-object-store
  distractor replaced with the realistic misconception that GDS eliminates all
  latency including remote-tier round trips.
- `#sec-storage-fallacies` q1 (NVMe-alone fallacy): directly addresses the
  RAID 0 red herring by clarifying the chapter's RAID 0-for-warm-cache
  endorsement.

### 3. `build_up_violation` / tautological LOs → application of prior vocab

Several original questions tested definitions of terms already in prior
chapters (LRU, RAID, tail latency, NUMA). Rewrites use these as prior
vocabulary and test application in storage-specific regimes:

- LRU question (workload-inversion q2) now asks the reader to compute the
  hit rate (≈6–7%) for a concrete 200 GB cache over a 3 TB dataset,
  rather than to define LRU.
- `#sec-storage-summary` q1 LO rewritten from "Synthesize the chapter's
  hierarchy principle into a single storage design rule" (tautological with
  the stem) to "Evaluate storage architectures by applying the hierarchy
  principle to tier placement, buffering depth, and cost trade-offs across
  the six tiers" — names a measurable decision framework.
- All LOs now start with Bloom's verbs (Apply, Evaluate, Diagnose, Analyze,
  Justify, Identify, Compare, Classify, Calculate, Distinguish, Design,
  Predict, Explain) per §6 criterion 5.

## Substantial rework spotlight: `#sec-storage-workload-inversion`

Four of six questions were meaningfully restructured:

- **q1 MCQ** (procurement): stem strengthened with a concrete dataset
  size (200 TB immutable corpus) and a specific quantitative anchor
  (3.5 GB/s sequential vs 0.5 GB/s random = 7× penalty).
- **q2 TF** (LRU on streamed data): rewritten as a quantitative-claim TF
  per §16 TF-3 pattern — the reader must reason from cache-to-dataset
  ratio (200 GB / 3 TB) to disprove the 90% hit-rate claim.
- **q5 FILL deleted**, replaced with a new scenario SHORT (q5) that
  exercises data-starvation diagnosis: 1,024→1,280 node expansion,
  72%→62% utilization, and naming the confirming profiler measurement.
- **q4 MCQ** (model rollout): stem now includes explicit quantitative
  anchor (35 TB burst from 100 replicas × 350 GB), and distractors
  encode four distinct mental-model failures (OLTP confusion, steady-state
  confusion, invented delta-distribution, version-atomicity naivety).

The rewritten section now samples every `###` subsection:

- Access pattern (q1)
- Working set / cache failure (q2)
- Shuffling (q3)
- Write pattern / training vs inference (q4)
- Distributed-scale diagnosis (q5)
- Lifecycle comparison (q6 integrative)

## Metadata changes

- `generated_on`: `2026-04-24`
- `model`: `claude-opus-4-7`
- `improved_by`: `opus-subagent-phase2`
- `total_sections`, `sections_with_quizzes`, `sections_without_quizzes` unchanged (10/10/0).
- All ten `section_id` values preserved.
- Question counts held within ±1 of the original per section (net 0).

## Validator

```
$ python3 book/tools/scripts/genai/quiz_refresh/validate_quiz_json.py \
    book/tools/scripts/genai/quiz_refresh/_audit/opus_improved/vol2_data_storage_quizzes.json \
    book/quarto/contents/vol2/data_storage/data_storage.qmd
OK: vol2_data_storage_quizzes.json passes schema + anchor validation
```

Zero letter-reference warnings; every MCQ explanation refutes distractors by
content per §10 anti-shuffle-bug rules.
