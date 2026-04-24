# vol2/robust_ai — Quiz Improvement Change Log (opus-subagent-phase2)

**Date:** 2026-04-24
**Chapter position:** 30 of 33 (91% through the book — Vol2, late)
**Model:** claude-opus-4-7
**Validator:** PASS (0 errors, 0 warnings)

## Overview

Chapter 30 ("Robust AI") already sat at a B+ baseline after the prior improvement pass — schema clean, no anti-shuffle violations, reasonable coverage across its 9 sections. This pass lifts it to A-grade against the §6 quality bar and §16 gold-standard patterns, with particular attention to:

1. **Concrete numeric anchoring (§6 criterion 3, §16 pattern 1).** Every scenario MCQ and SHORT now cites the chapter's own figures (ε = 8/255 perturbation with 30-60 percent accuracy drop; Meta's 1e-4/hr SDC rate at 10,000 GPUs; ResNet-50 76 → 50 percent after adversarial training; PSI 0.10/0.25 thresholds with 5 percent performance gate; randomized smoothing at ~100K samples; 60 percent pipeline-origin incident share) rather than generic "explain why" framings.
2. **Distractor design for MCQs (§6 criterion 6, §16 MCQ-3 pattern).** Every MCQ now carries at least two plausible distractors that encode real practitioner mental-model failures: compute-first bias against memory-starved kernels; conflating scalability with service-level guarantees; single-feature monitoring blind to concept drift; cloud-backup architectures on edge devices; etc. Throwaway options removed.
3. **§10 anti-shuffle compliance.** All MCQ answers refute distractors by *content* (naming the concept or claim) rather than by letter. Zero `Option [A-D]`/`Choice [A-D]`/`Answer [A-D]`/`([A-D])` patterns — validator confirms 0 warnings.

## Counts

| Section | Original Qs | Improved Qs | Rewritten | Kept as-is | Deleted | Added |
|---|---:|---:|---:|---:|---:|---:|
| Silent Failure Problem | 6 | 6 | 6 | 0 | 0 | 0 |
| Real-World Robustness Failures | 5 | 5 | 5 | 0 | 0 | 0 |
| Unified Framework | 5 | 5 | 5 | 0 | 0 | 0 |
| Environmental Shifts | 6 | 6 | 6 | 0 | 0 | 0 |
| Input-Level Attacks | 6 | 6 | 6 | 0 | 0 | 0 |
| Adversarial Defenses | 5 | 5 | 5 | 0 | 0 | 0 |
| Data Poisoning Defenses | 5 | 5 | 5 | 0 | 0 | 0 |
| Fallacies and Pitfalls (Tier 2) | 3 | 3 | 3 | 0 | 0 | 0 |
| Summary (Tier 2) | 3 | 3 | 3 | 0 | 0 | 0 |
| **Total** | **44** | **44** | **44** | **0** | **0** | **0** |

Every question was materially rewritten to raise quantitative specificity, sharpen distractors, or close on a systems consequence. Question types, counts, section IDs, and the 9-section / 9-quizzed / 0-skipped metadata layout are all preserved. Section 1 keeps its 6 questions (above-band); all other full-quiz sections sit at 5-6; Tier 2 synthesis sections (Fallacies, Summary) stay at 3. Validator accepts all section question counts.

## Three issue patterns fixed

### 1. `recall_only` → scenario-grounded quantitative application

Original Q1 of §Silent Failure asked abstractly which situation "best captures why robustness is a different systems property from ordinary i.i.d. test accuracy." The rewrite grounds the question in a specific 95-percent-accurate medical classifier and asks which *observation* would signal robustness failure — forcing the reader to differentiate a 30-60 percent drop under ε = 8/255 perturbation from a 0.2-point i.i.d. variance, a latency change, and an unconverged loss. Same move applied throughout: every MCQ now carries a scenario with numbers, not a bare conceptual contrast.

### 2. `throwaway_distractor` → every wrong choice encodes a real mental-model failure

Original framework-MCQ distractors included weak throwaways like "Training failures, validation failures, and deployment failures as fully separate pillars" that a careful reader could eliminate on grammar. The rewrite keeps the same correct answer but adds distractors corresponding to *observed* practitioner confusions: lifecycle-stage conflation (fragments the same adversarial attack across three teams), performance-axis confusion (accuracy/latency/size taxonomy), and category-scope errors (privacy and energy mis-assigned to robustness). Same treatment applied to the unified-framework SDC-scaling question (1e-8 vs 0.63 vs 1e-4 vs 1.0 — each a specific arithmetic mistake a learner actually makes) and to the adversarial-defense MCQ on certified vs empirical robustness (cost direction, threat-model scope, dropout confusion).

### 3. `build_up_violation` → test application of prior vocabulary, not its definition

Original Q on transferability simply asked the reader to name the property enabling black-box attacks. The rewrite keeps the term but frames it as a concrete scenario: "An attacker trains a public ResNet locally on ImageNet, crafts PGD perturbations against that model, and finds the resulting images fool the target API 45 percent of the time." Transferability is prior vocabulary from vol2/security_privacy — the question now tests its application to a specific black-box pipeline, not its definition. Same rewrite applied to: concept drift (applied to fraud-behavior evolution with stable P(X)); MMD (inferred from a described correlation-only shift that passes PSI/KS); graceful degradation (inferred from described bounded-capability reduction); feature squeezing (inferred from described serving-time compare-and-flag behavior); Huber loss (applied to bound 100x-normal outlier gradients).

## Substantial-rework section: §A Unified Framework for Robust AI

This section underwent the deepest rewrite because the original questions tested framework *recall* rather than framework *application*:

- **Q1 (taxonomy classification)** now asks how to divide robustness engineering headcount across teams, which forces the reader to apply the three-pillar structure to an actual organizational decision. Distractors encode three specific real-world anti-patterns (lifecycle-stage fragmentation, performance-axis confusion, scope misallocation).
- **Q2 (SDC scaling math)** was already numeric but now forces the reader to compute P(≥1) = 1 − (1 − p)^N for 10,000 GPUs at p=1e-4 and interpret the result operationally (~0.63/hr → architectural defenses required). Distractors represent four specific arithmetic or conceptual errors.
- **Q3 (detection → degradation → adaptive-response)** keeps the ORDER type but rewrites the answer to explicitly name the *swap consequence* for each pair (adaptation ahead of detection → blind oscillation; skipping degradation → collapse during adaptation window), modeled on §16.5 ORDER-1/3 patterns.
- **Q4 (quantization vs robustness margin)** moves from abstract "why efficiency hurts robustness" to a concrete 2-4x latency / 75 percent memory / 1-3 percent clean-accuracy setup with ResNet-50 anchor, forcing the reader to reason about decision-boundary slack.
- **Q5 (defense-in-depth)** keeps the correct choice but rewrites distractors to reflect real anti-patterns (single-classifier simplicity, training-only concentration, hardware-ECC-only) instead of obviously-wrong straw alternatives.

Every question closes on a systems consequence (resource commitment, operational cycle, fleet impact, pipeline coverage) rather than restating the concept.

## Validator status

```
$ python3 book/tools/scripts/genai/quiz_refresh/validate_quiz_json.py \
    book/tools/scripts/genai/quiz_refresh/_audit/opus_improved/vol2_robust_ai_quizzes.json \
    book/quarto/contents/vol2/robust_ai/robust_ai.qmd
OK: vol2_robust_ai_quizzes.json passes schema + anchor validation
```

0 errors, 0 warnings, 0 letter-reference anti-shuffle hits.
