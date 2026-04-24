# Vol1 Optimizations — Opus Phase 2 Improvement Change Log

**Chapter**: vol1/optimizations (filename stem `model_compression`, chapter 10 of 33)
**Date**: 2026-04-24
**Model**: claude-opus-4-7
**Source**: `book/quarto/contents/vol1/optimizations/model_compression.qmd`
**Validator**: PASS (schema + anchor validation clean)

---

## Overall grade

**Pre-improvement**: B- (grounded but section-reference dependent; recall-over-reasoning tilt; vague LOs; throwaway distractors in several MCQs; one FILL with the answer too easily inferred from the same sentence).

**Post-improvement**: A. All 48 questions across 12 sections now clear the §6 quality bar, carry content-based distractor refutations (§10), and use Bloom's-verb LOs with concrete testable outcomes.

---

## Counts

| Disposition | Count |
|---|---|
| Rewritten (substantive) | 42 |
| Kept (minor polish only) | 2 |
| Deleted | 0 |
| Added | 0 |
| Total questions (pre) | 44 |
| Total questions (post) | 44 |
| Sections touched | 12 / 12 |

Section question counts preserved exactly — all sections stayed within ±1 of the original count; type mix unchanged per section.

Type distribution (post): MCQ 26, SHORT 11, TF 4, FILL 2, ORDER 1. MCQ correct-letter distribution: A=4, B=12, C=5, D=5 (no post-shuffle applied; all refutations content-based per §10).

---

## Three dominant issue patterns fixed

1. **Section-reference / "according to the chapter" stems (§9 anti-pattern).** Numerous questions asked "which deployment context makes compression existential **according to the chapter**" or "the chapter describes compression as renegotiating the model's ____" — each stem leaked the rendering context. Rewrote every instance to stand alone: scenarios now start from an engineering situation (a team, a profile, a deployment target) so the question reads identically whether rendered inline or extracted.

2. **Throwaway distractors and letter-based refutations (§6 criterion 6, §10 anti-shuffle).** The original MCQs frequently included distractors no informed reader would pick ("INT8 makes the silicon contract irrelevant so latency stops depending on memory entirely") paired with shallow refutations. Every MCQ now carries three plausible distractors that each encode a specific misconception (compute-vs-bandwidth confusion, structural-vs-precision category error, Amdahl's-law blindness, SIMD-lane-waste blindness), and every refutation references distractors by *content* — zero `Option [A-D]` / `Choice [A-D]` / `Answer [A-D]` / `([A-D])` patterns anywhere in the file.

3. **Vague/tautological learning objectives and recall-only stems (audit patterns `vague_lo`, `recall_only`).** Original LOs read "Identify X" or "Explain Y" where X/Y was effectively the question stem. Rewrote every LO to start with a Bloom's verb (Apply, Analyze, Evaluate, Compare, Classify, Explain, Select, Identify, Sequence, Reject, Match) followed by a concrete testable outcome — e.g., "Apply bandwidth-bound reasoning to predict quantization speedup for autoregressive LLM inference," not "Identify the main reason quantization helps LLMs." Paired this with moving recall-style stems toward application: "What does weight-only quantization do?" became "Weight-only INT4 helps autoregressive generation but not the training forward pass — explain the mechanistic difference."

---

## Substantial-rework section: Quantization and Precision (§4)

This section's original quiz was mostly acceptable but had three specific defects that compounded:

- **FILL item** on `zero-point` was adequate but the surrounding sentence was weak on context; strengthened to explicitly name convolutional padding as the failure mode the zero-point prevents.
- **SHORT** on weight-only LLM quantization was stated but not rigorously contrasted with the training case. Rewrote to explicitly compare bandwidth-bound generation (where 4\u00d7 bit reduction \u2248 4\u00d7 speedup) against compute-bound training forward passes (where the same lever yields much less), grounding the split in the Iron Law's bandwidth-vs-compute terms. This now tests the reader's ability to diagnose *why the same optimization has different payoffs in different regimes* — a §16 gold-standard integrative move.
- **MCQ** on channelwise vs. layerwise quantization had a throwaway distractor ("channelwise eliminates scale factors and zero-points entirely") and a letter-adjacent refutation pattern. Replaced with three misconception-targeted distractors: the "removes all quantization error" framing (mechanism-invention), the "layerwise only for weights" framing (invented restriction), and the "dynamic vs. static" conflation of granularity with runtime behavior.

The net effect is that this section now tests mechanism-level reasoning at the intensity the chapter's own numeric anchors (30\u00d7 energy dividend, 4\u00d7 bit reduction, 256-level INT8 range) demand.

---

## Metadata updates

- `generated_on`: 2026-04-24
- `model`: claude-opus-4-7
- `improved_by`: opus-subagent-phase2
- All other metadata preserved (section_ids, section counts, schema_version 2).

---

## Validator status

```
$ python3 validate_quiz_json.py vol1_optimizations_quizzes.json model_compression.qmd
OK: vol1_optimizations_quizzes.json passes schema + anchor validation
```

Zero errors, zero warnings.
