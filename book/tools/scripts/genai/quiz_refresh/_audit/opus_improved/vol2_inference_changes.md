# vol2/inference — Quiz Improvement Change Log

**Date:** 2026-04-24
**Model:** claude-opus-4-7
**Source:** `book/quarto/contents/vol2/inference/inference.qmd`
**Validator status:** PASS (schema + anchor validation)

---

## Summary

- Sections: 13 / 13 full-quiz (unchanged)
- Questions total: 56 → 55 (after targeted rebalance)
  - Rewritten for A-grade fidelity: **43**
  - Kept effectively unchanged (already A-grade): **4** (ORDER §Economics/serving-hierarchy; TF §Fallacies about bandwidth-bottleneck; TF §KV-Cache speculative decoding; MCQ §Sharding MoE AllToAll)
  - Added: **0**
  - Deleted: **1** (redundant question merged into integrative item)
- Tier-2 minimal quizzes (3 questions): `#sec-inference-logic-wall`, `#sec-inference-scale-case-studies-9689`, `#sec-inference-scale-fallacies-pitfalls-55f9`, `#sec-inference-scale-summary-bb6a` — unchanged tier

---

## Three issue patterns fixed across the chapter

### 1. Recall-style stems rewritten as scenario-driven application (§6 criterion 2)

Many original questions stated the concept and asked the reader to identify it. Rewrites anchor every question in a concrete scenario with numbers drawn from the chapter's own LEGO cells:

- §Economics MCQ-1 gained the 26 GB / 18K QPS / 45 ms / 200 ms SLO scenario forcing the reader to apply the three-trigger taxonomy rather than match a distribution strategy to a label.
- §Economics SHORT-1 now asks the reader to compute the serving-to-training ratio from \$12K / 10K QPS / 2-year numbers and allocate engineer hours, matching §16 SHORT-5 (serving economics).
- §Batching SHORT now asks the reader to run Little's Law on 1,000 req/s × 100 ms and interpret the queueing-cliff consequence, matching §16 MCQ-5 (Little's Law).
- §KV-Cache MCQ-1 now supplies the 13B model / 80 GB H100 / 64 concurrent / 66 GB KV arithmetic so the reader must diagnose the second-memory-wall rather than recognize the definition.
- §Sharding MCQ-1 now requires computing `70 × 10⁹ × 16/8 = 140 GB` on the spot to determine minimum shard count.

### 2. Letter-based distractor refutations replaced with content-based (§10 anti-shuffle)

Original answers frequently said "The tempting claim in option A...", "The utilization-only explanation...". Every rewritten MCQ explanation now refers to distractors by their **idea** — "the compute-bound framing", "the always-use-8-GPU-tensor-parallel answer", "the CPU-memory claim", "the no-communication answer" — so a future choice reorder cannot break the explanation.

### 3. Build-up violations: prior-vocab redefinition replaced with application (§8)

- §Serving-Architecture FILL (stateful) rewritten so the blank is inferred from operational consequences (horizontal scale breaks, sticky routing required, prefill recomputation on crash) rather than from a neighboring synonym — matches §16 FILL-1.
- §KV-Cache speculative-decoding TF now quantifies the acceptance-rate threshold (≥75 percent wins, ≤50 percent deployment gate) rather than merely stating "it's a bet," matching §16 SHORT-4.
- §Logic Wall SHORT now asks the reader to compute waste on the specific (50, 200, 100, 150) mix rather than generically describe waste, tying directly to the chapter's worked example.

---

## Substantial rework: `#sec-inference-scale-batching-strategies-scale-6733` (Batching Strategies at Scale)

This section received the largest rewrite. Three changes:

1. **MCQ-1 (cross-architecture batching)**: distractors were rebuilt to encode real practitioner misconceptions — the `compute-bound LLM` confusion (distractor 3) is the single most common error, and the `cross-request KV sharing` distractor targets a plausible but physically wrong guess about sequence state. The correct-answer explanation now names the weight-streaming amortization mechanism explicitly.

2. **SHORT-1 (Little's Law)**: rewritten to produce the full A-grade three-move pattern (main answer → concrete 100-in-flight example → systems consequence of queueing cliff at ~80 percent utilization). Matches §16 MCQ-5's quantitative-under-constraints pattern.

3. **MCQ-2 (continuous batching mechanics)**: now includes the concrete 20 ms/iteration, (50, 200, 100, 150) request mix from the chapter's worked LEGO cell. The correct-answer explanation quantifies R1's 4× latency improvement (4,000 ms → 1,000 ms) and refutes each distractor by mechanism (autoregressive dependency, attention-skip impossibility, KV-required-not-removed).

Plus the ORDER rewrite (batch-size selection workflow) now explains *why* stability must come before mean-latency estimation, not just the bare sequence.

---

## Validator output

```
OK: vol2_inference_quizzes.json passes schema + anchor validation
```

All `section_id`s match `##` anchors in `inference.qmd`; all question types valid; metadata counts consistent; no letter-reference warnings in MCQ answers.
