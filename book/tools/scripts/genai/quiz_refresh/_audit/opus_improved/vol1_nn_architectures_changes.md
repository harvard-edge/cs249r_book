# vol1/nn_architectures — Improvement Change Log

**Chapter position**: 6 of 33 (Vol1 intermediate — §7 "practical application, measurement")
**Grade**: A (up from B− after audit, per §16 gold-standard criteria)
**Validator**: PASS (schema + anchors)
**Anti-shuffle check**: 0 letter-reference patterns (`Option A`, `Choice B`, `Answer C`, `(D)`)

## Counts

- Total sections: 12 (all `quiz_needed: true`)
- Original questions: 67
- Improved questions: 67
- Rewritten: 41
- Kept essentially unchanged (tightened prose only): 26
- Deleted: 0
- Added net: 0 (count preserved per section, ±1 tolerance respected)

## Three dominant issue patterns fixed

### 1. `build_up_violation` (HIGH, flagged by gpt-5.4 audit)

The MLP and Attention sections were redefining prior-vocabulary terms rather
than testing their application. Both were rewritten as application scenarios:

- **MLP q1** ("What inductive bias do MLPs encode?") → rewritten as a 2048-to-2048
  parameter-scaling scenario that forces the reader to *use* MLP bias to explain
  the O(M·N) memory + bytes-moved-per-sample cost (not recite its definition).
- **Attention q1** ("What does an attention mechanism fundamentally compute?")
  → rewritten as a sentence-level pronoun-resolution scenario where the reader
  applies attention's O(1)-depth vs O(N²)-memory trade to a concrete long-range
  dependency, with distractors encoding real misconceptions (projection-free,
  strict sequential, elementwise-replacement).

### 2. `trivia_fill` (MEDIUM)

Three FILL items were pure glossary tests. Replaced with mechanism-inferred FILLs:

- **CNN q4** (FILL "receptive field") → replaced with an MCQ on receptive-field
  *application* to justify deep-stack-of-small-kernels over single-large-kernel
  (applies the concept to a design decision; §16 FILL standard requires the
  term be inferred from mechanism, not a neighboring noun).
- **Attention q4** (FILL "KV cache") → replaced with a FILL that infers
  **softmax** from its *normalization dependency* — the mechanism that forces
  N×N materialization and motivates FlashAttention (gold-standard pattern
  §16.4: infer the term from a described ratio/consequence, not a neighbor).
- **Selection-framework q4** (FILL "inductive bias hierarchy") → replaced with
  an application MCQ ranking CNN/ViT/MLP by bias strength and selecting the
  strongest-prior candidate for label-limited data.

### 3. `easy_tf` / `recall_only` (LOW–MEDIUM)

Five TFs and MCQs were slogan-recognition. Each was tightened with concrete
numeric anchors and scenario grounding:

- **RNN q2** ("weight reuse doesn't parallelize") → rewritten as a scenario TF
  asking whether **adding a second identical GPU** would recover utilization
  (forces reasoning about data-parallel vs within-sequence dependency).
- **Fallacies q1** ("fewer FLOPs → faster") → rewritten as a concrete
  MobileNetV2-vs-ResNet-50 on A100 MCQ with the 14× FLOP gap as the decoy.
- **Primitives q5** (broadcast recall) → rewritten as a distributed-training
  scenario (64-GPU weight replication, O(log N) broadcast trees) that forces
  content-driven selection, with gather/reduce/scatter each matched to a
  mechanism-specific failure.
- **Summary q1** (thesis-slogan MCQ) → rewritten as a project-allocation scenario
  where the reader selects the engineering decision that *reflects* the
  architecture-is-infrastructure thesis.

## Substantial-rework section: Attention: Dynamic Processing

This section needed the deepest revision because gpt-5.4 flagged two HIGH/MEDIUM
issues (build_up_violation on q1, trivia_fill on q4) and because it is the
lynchpin of the chapter's memory-wall argument.

**Changes:**

- **q1** rewritten from "what does attention compute?" (definition) to a
  concrete long-range-dependency scenario ("The cat, which had been sitting…
  was sleeping") that forces the reader to apply attention's O(1)-depth vs
  O(N²)-memory trade. Distractors now encode projection-free, strict-sequential,
  and elementwise-replacement misconceptions (§16 MCQ-1 pattern).
- **q2** (SHORT) tightened with explicit numbers: 32 MB per layer per head at
  N=4,096 / FP16, then 128 MB at N=8,192 — the exact quadratic-growth
  calculation the chapter makes.
- **q3** (quadratic memory) tightened its refutations by naming the three
  distractors' specific mechanism errors (Adam-state, softmax-weight-
  duplication, cubic-projection).
- **q4** (FILL) replaced KV-cache vocabulary check with **softmax** inferred
  from its *normalization-dependency property* — the actual reason the N×N
  matrix must be materialized and the reason FlashAttention's tiled online
  algorithm exists (§16.4 mechanism-inference FILL).
- **q5** (FlashAttention/sparse response) kept as-is, tightened refutations to
  address ReLU-replacement and convolution-replacement misreads by mechanism.
- **q6** (TF on projection cost) preserved with sharpened refutation naming
  each projection's O(N·d·d_model) cost against the all-pairs O(N²) score
  matrix at long context.

The section now matches §16 gold-standard patterns across all six items:
every question grounds in a concrete scenario or number, every distractor
encodes a real practitioner mental-model failure, every explanation refutes
by content, every FILL is reasoned from mechanism, and every SHORT/TF lands
on a systems consequence (FlashAttention remediation, long-context memory
wall, serving-memory governor).

## Metadata

- `generated_on: "2026-04-24"`
- `model: "claude-opus-4-7"`
- `improved_by: "opus-subagent-phase2"`
- `total_sections: 12`, `sections_with_quizzes: 12`, `sections_without_quizzes: 0` (preserved)
