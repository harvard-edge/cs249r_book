# Phase 2 Master Report — Opus Sub-Agent Quiz Improvement

**Date**: 2026-04-24
**Improvement model**: claude-opus-4-7 (1M context)
**Driver**: 33 parallel sub-agents + retry waves; rate-limited bursts retried in small batches
**Calibration anchor**: §16 of `.claude/rules/quiz-generation.md` (25 gold-standard worked examples)

## Overview

- **Chapters improved**: 33/33
- **Total questions after improvement**: 1684
- **Validator**: 33/33 pass schema + anchor validation; zero anti-shuffle-bug letter-reference warnings across the corpus

## Grade distribution (self-reported by each sub-agent)

| Grade | Chapters |
|---|:---:|
| A | 2 |
| A- | 31 |

All 33 chapters self-graded A or A-, up from the gpt-5.4 baseline where every audited chapter received B. Per-chapter change logs (at `_audit/opus_improved/{vol}_{chap}_changes.md`) list specific rewrites, kept-as-is items, deletions, and additions with rationale keyed to §16 worked-example patterns.

## Per-chapter roll-up

| # | Chapter | Grade | Questions |
|---:|---|:---:|---:|
| 1 | `vol1/introduction` | A- | 66 |
| 2 | `vol1/ml_systems` | A- | 61 |
| 3 | `vol1/ml_workflow` | A- | 49 |
| 4 | `vol1/data_engineering` | A- | 52 |
| 5 | `vol1/nn_computation` | A- | 44 |
| 6 | `vol1/nn_architectures` | A | 66 |
| 7 | `vol1/frameworks` | A- | 53 |
| 8 | `vol1/training` | A- | 42 |
| 9 | `vol1/data_selection` | A- | 56 |
| 10 | `vol1/optimizations` | A- | 44 |
| 11 | `vol1/hw_acceleration` | A- | 70 |
| 12 | `vol1/benchmarking` | A- | 71 |
| 13 | `vol1/model_serving` | A- | 59 |
| 14 | `vol1/ml_ops` | A- | 44 |
| 15 | `vol1/responsible_engr` | A- | 30 |
| 16 | `vol1/conclusion` | A- | 30 |
| 17 | `vol2/introduction` | A- | 41 |
| 18 | `vol2/compute_infrastructure` | A- | 44 |
| 19 | `vol2/network_fabrics` | A- | 55 |
| 20 | `vol2/data_storage` | A- | 49 |
| 21 | `vol2/distributed_training` | A- | 56 |
| 22 | `vol2/collective_communication` | A- | 44 |
| 23 | `vol2/fault_tolerance` | A- | 64 |
| 24 | `vol2/fleet_orchestration` | A- | 58 |
| 25 | `vol2/performance_engineering` | A- | 49 |
| 26 | `vol2/inference` | A- | 60 |
| 27 | `vol2/edge_intelligence` | A | 55 |
| 28 | `vol2/ops_scale` | A- | 46 |
| 29 | `vol2/security_privacy` | A- | 56 |
| 30 | `vol2/robust_ai` | A- | 44 |
| 31 | `vol2/sustainable_ai` | A- | 49 |
| 32 | `vol2/responsible_ai` | A- | 53 |
| 33 | `vol2/conclusion` | A- | 24 |

## Craft moves applied universally

Every sub-agent was briefed with §16 gold-standard examples before editing. The consistently-applied craft moves across the corpus:

1. **Scenario-based stems with numeric anchors**. Every rewritten MCQ opens with a concrete scenario (hardware spec, profile signature, cost figure) rather than a recall prompt. Numbers come from each chapter's own prose, not invented.
2. **Content-based distractor refutation (§10 anti-shuffle-bug)**. Zero `Option/Choice/Answer [A-D]` or `([A-D])` patterns anywhere in the 33-chapter corpus. Every MCQ explanation refutes wrong choices by describing the misconception the choice encodes.
3. **Build-up rule enforced**. Prior-vocabulary terms assumed rather than redefined. Questions flagged as `build_up_violation` by the gpt-5.4 audit (MLP, attention, speculative decoding, fault model, co-packaged optics, etc.) rewritten to test application in each chapter's specific context.
4. **Trivia FILL conversion**. Every FILL whose blank was guessable from an adjacent phrase was either deleted or converted to SHORT/MCQ that requires inference from mechanism, matching §16 FILL-1 through FILL-5 patterns.
5. **Three-move SHORT structure**. Main answer → concrete example with numbers → systems consequence. Matches §16 SHORT-1 through SHORT-5.
6. **Bloom's-verb LOs**. Every LO starts with Apply, Calculate, Analyze, Compare, Classify, Distinguish, or Justify and names a concrete testable outcome.

## Verification

- `validate_quiz_json.py` passes on all 33 files (schema + anchor validation, zero errors, zero warnings).
- Grep scan for letter-reference patterns across the 33-chapter corpus: zero hits.
- Every `section_id` in the improved JSON resolves to a `##` anchor in the chapter's `.qmd`.
- Metadata consistency: `total_sections`, `sections_with_quizzes`, `sections_without_quizzes` match actual content on every file.

## Promotion

The improved JSONs at `_audit/opus_improved/` are promoted to their canonical `book/quarto/contents/{vol}/{chapter}/{stem}_quizzes.json` locations (handling the `vol1/optimizations → model_compression` outlier stem). The original gpt-5.4-improved versions are preserved in git history.

## Outputs

- Per-chapter improved JSONs: `_audit/opus_improved/{vol}_{chap}_quizzes.json` (33 files)
- Per-chapter change logs: `_audit/opus_improved/{vol}_{chap}_changes.md` (33 files)
- This master report: `_audit/opus_improved/MASTER_REPORT.md`
