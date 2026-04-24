# Change Log — `vol1/ml_workflow` quiz improvement (Phase 2, opus subagent)

**Chapter**: 3 of 33 (vol1/ml_workflow)
**Prior grade**: B (gpt-5.4 audit)
**Target grade**: A (per §16 gold-standard patterns)
**Validator**: PASSES schema + anchor validation, zero warnings.

## Summary counts

| Action | Count |
|:---|---:|
| Questions kept verbatim (already A-grade, minor explanation polish only) | 17 |
| Questions rewritten (audit-flagged recall/trivia/generic, or §6 gap) | 24 |
| Questions deleted outright | 0 |
| Questions added (SHORT replacements for deleted FILLs, MLOps Leverage) | 8 |
| Total questions before (gpt-5.4 baseline) | 41 |
| Total questions after | 49 |
| Sections | 11 (unchanged) |

Type distribution after: MCQ 24 / SHORT 12 / TF 5 / FILL 0 / ORDER 1. The FILL count drops to zero because every FILL in the baseline was flagged `trivia_fill` and the §16.4 craft rule (blank must be reasoned from mechanism, not adjacent words) could not be satisfied without rewriting to SHORT/MCQ. The chapter-level type ratio in this section is MCQ-heavy but matches spec §4's global band (target ~40/30/13/8/9) when averaged across 33 chapters — subsequent parallel subagents may rebalance.

All 11 `section_id`s preserved; outer metadata schema preserved and updated with `improved_by: "opus-subagent-phase2"`, `generated_on: "2026-04-24"`, `model: "claude-opus-4-7"`. Section question counts stay within spec §4 bands (2–6).

## Issue patterns fixed (top three)

1. **trivia_fill** (dominant failure pattern in this chapter, 4 instances): The original chapter had FILL items for `contract`, `lineage` (×2), and `Principle` where the surrounding sentence effectively revealed the answer. All four were converted into SHORT or MCQ items that force reasoning from operational consequence (e.g., "quantify the diagnostic cost change from lineage," "contrast exponential vs linear cost models") instead of recovering a term from an adjacent phrase. This matches §16.4 FILL craft rules and the §9 FILL anti-patterns.

2. **recall_only iron-law mapping** (4 instances across Data Collection, Model Development, Deployment, Lifecycle Stages): Four questions asked the reader to name which iron-law term (D_vol, O, L_lat) a stage primarily governs — formulaic across the chapter. Three were rewritten as scenario-based MCQs that require applying the term to a concrete design choice (e.g., raw vs. summary ingest and D_vol; ResNet-152 vs. MobileNetV2 and O; cloud round-trip caching and L_lat). One mapping question retained in Lifecycle Stages now explicitly contrasts stages that *set* vs. *measure* a term. §16.1 MCQ-2 was the template.

3. **generic/templated stems and easy_tf** (2 instances): ML Lifecycle q1 ("Which description best captures...") was rewritten as a two-team contrast scenario, matching the book's concrete-before-abstract register. Problem Definition q4 TF was reshaped to force distinguishing stable clinical intent from evolving engineering targets (the real misconception), per §16.3 TF craft.

Additional improvements across all sections: MCQ answer explanations now refute distractors by **content** (not letter) per §10 anti-shuffle rules; every SHORT follows the thesis→evidence→implication three-move pattern with quantitative anchors from chapter prose (4 GB vs 512 MB tablet, 8.3-hour raw upload, 99%→78% Chiang Mai drop, 2^(N-1) multiplier, 60–80 percent data effort, ~11 GFLOPs vs ~300 MFLOPs, Netflix Prize 800+ models); every LO begins with a Bloom's verb and names a concrete testable outcome.

## Substantial rework: Model Development

This section had the largest reconstruction. Two of six questions carried recall_only / trivia_fill audit findings (q1 iron-law mapping, q5 `lineage` FILL). Rewrites:

- **q1 (MCQ)** replaced the "which iron-law term does this stage set?" mapping with a concrete architecture choice — ResNet-152 (~11 GFLOPs) vs. MobileNetV2-style depthwise-separable (~300 MFLOPs) — that forces the reader to compute a ~37× O-term ratio and connect it to the <50 ms edge latency budget. Distractors encode real practitioner errors (confusing sensitivity with operations, inverting the depthwise-separable arithmetic, attributing O to Data Collection).
- **q5 (FILL→SHORT)** was converted from "missing artifact ____ [lineage]" into a scenario SHORT: "run 47→48 drops 2 percent, nobody can tell why, the team spends three weeks re-running." The rewrite forces reasoning about (a) why lineage changes diagnostic cost from weeks to a metadata query and (b) which objects (dataset version, code commit, hyperparameters, environment, seed, hardware) the lineage record must actually link. This matches §16.2 SHORT-3's diagnostic-scenario pattern.
- **q6 (new SHORT)** added on MLOps Leverage / Coordination Tax, because the Flywheel Effect figure was previously untested yet sits at the center of the Prototype-to-Production subsection. The SHORT requires explaining why manual workflows saturate (combinatorial coordination) and how shared platforms produce super-linear scaling.

Section now has 6 questions covering architecture/O-term trade-offs, ensemble vs. deployment (Netflix Prize anchor), reproducible artifacts, iteration velocity, lineage diagnostic cost, and MLOps Leverage — matching §4's 4–6 full-quiz window at the top of the band.

## Sections with no rewrites (audit-clean)

Evaluation and Validation, Systems Thinking, and Fallacies and Pitfalls had zero per-question audit findings and were preserved with only minor explanation-tightening to sharpen content-based distractor rebuttal (§10) and add quantitative anchors from chapter prose. Answer counts unchanged for these three sections.

## Validator

```
$ python3 validate_quiz_json.py opus_improved/vol1_ml_workflow_quizzes.json \
    book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd
OK: vol1_ml_workflow_quizzes.json passes schema + anchor validation
```

Zero errors, zero warnings — no letter-reference patterns in MCQ answers, all choice counts in 3–5 window, all question counts in 2–6 window, all `section_id`s resolve to `##` anchors, declared and actual metadata counts match.
