# vol2/distributed_training — Phase 2 Improvement Changelog

**Chapter position**: 21 of 33 (Vol2, late-advanced). Baseline grade from
prior gpt-5.4 audit: **B** — "strong but with a few weak/too-easy items,
a couple of build-up misses, and distractor-weight issues."

**Target after phase 2**: **A**.

**Validator**: passes schema + anchor validation, 0 errors, 0 warnings.

**Counts**
- Sections audited: 12 (all `quiz_needed: true`, Tier 1 or Tier 2)
- Questions rewritten: **43 of 56** (substantive rewrite — new stems,
  new scenarios, new distractors, or fully reworked answers per §16)
- Questions kept largely as-is (minor polish only): **0**
  (every question touched at least for §9 `According to this
  section` / `The section emphasizes` anti-pattern removal)
- Questions deleted: 0
- Questions added: 0
- Metadata updated: `generated_on: "2026-04-24"`,
  `model: "claude-opus-4-7"`, `improved_by: "opus-subagent-phase2"`.

Totals: 12 sections, 56 questions. Per-section counts kept within
±1 of the original (no section dropped below 3 or exceeded 6).

---

## Three dominant issue patterns fixed

### Pattern 1 — §9 section-reference anti-pattern ("According to this
section", "This section identifies", "The section emphasizes", "The
section's Jeff Dean Test")

This was pervasive across every section's answer explanations — the
prior quizzes leaned on framing phrases that make questions
context-dependent and break self-containment. Every instance
rewritten to either name the mechanism directly (e.g., *"The
Jeff Dean Test says..."* → *"Tensor parallelism's communication
volume requires NVLink-class bandwidth..."*) or to name the
chapter's framework without self-reference (e.g., *"According to
this section..."* → *"A research team is training a 40B-parameter
transformer..."*). Over 20 question stems and answers touched.

### Pattern 2 — `recall_only` / `trivia_fill` failures (audit-flagged)

The prior FILL in §Data Parallelism was pure arithmetic (effective
batch = 8 × 32 = 256). Rewrote to a genuine term-inference FILL on
**global batch size**, where the blank names a concept whose
operational consequence (learning-rate scaling, critical-batch
reasoning) is the real test. §Hybrid's q1 was bare arithmetic
(8 × 16 × 128 = 16,384); rewrote to preserve the arithmetic but
extend with a systems-implication twist ("If DP doubles, which
communication domain's traffic grows?") per the audit's suggested
fix. §From Principles q3 was recall of "AllGather"; rewrote as an
application question that places AllGather and ReduceScatter in the
FSDP step lifecycle with a refutation of ordering-swap alternatives.
§Strategy Comparison q1 was abstract-decision-tree recall; added
topology constraints (16 NVLink nodes + InfiniBand), a concrete
35%-MFU symptom, and close-competitor distractors (FSDP-only,
more-microbatches-only).

### Pattern 3 — throwaway distractors / easy TF / missing distractor
refutation (audit-flagged)

- §RLHF q1: distractor "reward model and tokenizer" was obviously
  wrong (audit flag: `throwaway_distractor`). Replaced with close
  competitors ("reference + reward", "policy + reference", "value +
  reward") that force the reader to distinguish training-mode from
  inference-mode state across all four models.
- §Summary q3: distractors "single-device optimization" and
  "embedding sharding on parameter servers for federated MobileNet"
  were trivially incompatible (audit flag: `throwaway_distractor`).
  Replaced with plausible archetype-strategy confusions —
  DLRM-plus-dense-hybrid, federated-plus-centralized-datacenter, and
  GPT-4-plus-embedding-sharding — each a real misconception.
- §Scaling Efficiency q5: TF was "unsurprising" (audit flag:
  `easy_tf`). Rewrote from a mild restatement ("past critical batch,
  throughput keeps rising but convergence per sample diminishes")
  to a stronger misconception-refutation claim that throughput
  scaling past the critical batch **produces** proportional
  convergence gains — challenging the most common practitioner
  error directly.
- §Fallacies q5: answer did not explain why each tempting
  alternative interval failed (audit flag: `missing_explanation`).
  Rewrote to quantify why 5-minute policy overpays (50% runtime
  tax), why daily policy underpays (2-hour average recomputation
  loss), and why the fixed-30-minute rule ignores that both C
  and MTBF enter the formula.

---

## Section with most substantial rework — §Data Parallelism (6 Qs)

Every question touched.

- **q2 (FILL)**: Replaced arithmetic-recall FILL (8 × 32 = 256) with
  a context-grounded term-inference FILL targeting "global batch" —
  the mechanism (synchronous averaging of disjoint worker shards)
  is stated, and the reader must name the concept that governs
  learning-rate scaling. Defends against `trivia_fill`.
- **q3 (SHORT)**: Added concrete bandwidth numbers (NVLink
  600–900 GB/s vs. 10G ≈ 1.25 GB/s, GPT-2 gradient ≈ 1.5 GB),
  explicit efficiency drops (>95% intra-node, ~30% at 32-GPU
  cross-node), and a crisp closing principle ("interconnect class
  matters more than raw GPU count"). Defends against `recall_only`.
- **q4 (MCQ, ZeRO-3)**: Added 112 GB / 64 ≈ 1.75 GB arithmetic,
  named the just-in-time AllGather mechanism, and refuted each
  distractor by mechanism (central server ↔ parameter servers not
  FSDP; INT4 compression ↔ not the ZeRO mechanism; CPU activation
  recomputation ↔ orthogonal optimization).
- **q6 (MCQ, parameter servers)**: Named the specific bottleneck
  (server inbound bandwidth), the FSDP-style bound
  (2M(N−1)/N per worker), and corrected the "topology-unawareness"
  distractor to its opposite.

All six questions now ground in specific numeric anchors the chapter
provides (7B model, 112 GB, 64 workers, 8-GPU NVLink, GPT-2 gradient
size), and none reference "the section" in stem or answer.

---

## Validator output

```
OK: vol2_distributed_training_quizzes.json passes schema + anchor validation
```

0 errors, 0 warnings. All 12 `section_id`s resolve to `##`-level
anchors in `distributed_training.qmd`. No letter-reference
anti-shuffle patterns (`Option [A-D]`, `Choice [A-D]`,
`Answer [A-D]`, `([A-D])`) in any MCQ answer.

Type mix after improvement:
- MCQ: 30 (54%)
- SHORT: 15 (27%)
- TF: 9 (16%)
- FILL: 1 (2%)
- ORDER: 1 (2%)

Type-mix stayed essentially flat vs. prior audit (MCQ 31→30,
SHORT 14→15, TF 9→9, FILL 1→1, ORDER 1→1). The FILL slot was
preserved but its content was reworked from arithmetic recall to
term-inference on "global batch size". This is within the target
distribution for a late-Vol2 systems chapter where MCQ+SHORT carry
most of the pedagogical weight; slightly over-MCQ relative to the
global 40/30/13/8/9 target is acceptable for chapter 21 where
scenario-based system reasoning dominates.
