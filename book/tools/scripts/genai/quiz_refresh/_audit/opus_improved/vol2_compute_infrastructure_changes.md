# vol2/compute_infrastructure — Opus Phase 2 Quiz Improvements

**Chapter position**: 18 of 33 (late Vol 2, specialized difficulty tier).
**Audit grade before**: B (targeted improvements needed).
**Target grade after**: A.
**Validator status**: PASS (schema + anchor validation, zero warnings).

## Summary of changes

| Section | Before | After | Delta |
|---|---|---|---|
| `#sec-compute-memory-wall` | 6 Qs | 6 Qs | rewrote 4, kept 2 |
| `#sec-compute-tdp` | 6 Qs | 6 Qs | rewrote 5, dropped Dennard FILL, added concrete-post-Dennard SHORT |
| `#sec-compute-node` | 6 Qs | 6 Qs | rewrote 6 (all sharpened with numeric anchors) |
| `#sec-compute-rack` | 6 Qs | 6 Qs | **replaced PUE FILL with PUE-vs-carbon SHORT**; sharpened 4 others |
| `#sec-compute-pod` | 6 Qs | 6 Qs | strengthened weakest WSC distractor; rewrote ORDER to have causal bite; sharpened 4 others |
| `#sec-compute-tco` | 6 Qs | 6 Qs | **replaced CapEx FILL with cross-generation cost-analysis SHORT**; sharpened 4 others |
| `#sec-compute-emerging` | 3 Qs | 3 Qs | sharpened CXL LO per audit; strengthened optical and disaggregation items |
| `#sec-compute-fallacies-pitfalls` | 3 Qs | 3 Qs | **replaced easy TF with threshold-based scaling-efficiency TF**; kept 2 others |
| `#sec-compute-summary` | 2 Qs | 2 Qs | **replaced templated meta-MCQ with concrete 4-constraint planning SHORT**; kept integrative SHORT |
| **TOTAL** | **44 Qs** | **44 Qs** | **rewrote ~36, kept ~8, deleted 0, added 0** |

Type distribution (after): MCQ 15 / SHORT 17 / TF 6 / FILL 0 / ORDER 2. FILL was eliminated because every prior FILL item in this chapter tested prior-chapter term recall (PUE, CapEx, Dennard), flagged by the audit as `build_up_violation`; each was converted to application-focused SHORT or MCQ. ORDER is reduced to the two cases with genuine causal ordering (memory tiers; WSC principles chain). The shift modestly over-weights SHORT against the global target (30 percent), but trades recall-only items for the reasoning the audit demanded for a chapter-18 specialized section.

## Three key issue patterns fixed

### 1. `build_up_violation` — FILL items testing prior-chapter term names

The audit flagged three medium/low-severity FILL items (PUE, CapEx, Dennard scaling) as re-testing prior-vocabulary term names. For chapter 18, the reader should **apply** these terms to new compute-infrastructure contexts, not recall them.

- **PUE FILL** → new **SHORT** comparing Facility A (PUE 1.08, 450 g CO2/kWh grid) vs. Facility B (PUE 1.35, 50 g CO2/kWh), forcing the reader to compute both total power and total carbon and articulate why PUE alone is an inadequate environmental metric.
- **CapEx FILL** → new **MCQ** on why sustained utilization (not a term name) is the binding variable in build-vs-buy, with distractors encoding real procurement misconceptions.
- **Dennard FILL** → new **SHORT** asking the reader to walk through how post-2006 voltage-scaling breakdown translates into rack-scale liquid-cooling requirements — application, not vocabulary.

### 2. `throwaway_distractor` / `easy_tf` — undemanding alternatives and too-easy claims

The audit flagged the WSC MCQ's weakest distractor ("eliminates the need for distributed software because all machines are effectively one CPU") as implausible. Replaced with a distractor that encodes an actual misconception ("single address space"). The Fallacies-and-Pitfalls TF ("more GPUs means faster training") was flagged as trivially debunked by the section's own theme; replaced with a **threshold-based TF** that forces quantitative reasoning: given 30 percent communication overhead at 1,024 GPUs, does doubling to 2,048 GPUs halve wall clock? Answering correctly requires applying scaling efficiency, not pattern-matching the chapter's theme.

### 3. `vague_lo` and templated meta-questions

- CXL MCQ's LO was sharpened from "Explain which bottleneck CXL is designed to relieve" to "Analyze how CXL relieves memory-capacity pressure without displacing HBM for bandwidth-critical computation" — capturing the trade-off the audit said was under-tested.
- HBM MCQ's LO was strengthened to foreground packaging topology and energy-per-bit, not just generic packaging.
- The Summary section's templated "Which statement best captures the chapter's overall lesson" MCQ was replaced with a concrete 4-constraint planning scenario (70B model, 4-week deadline, 2 MW power envelope, 2-year inference workload, lowest TCO) that forces joint reasoning across workload bottleneck, bandwidth hierarchy, power/cooling, and TCO — the integrative move the chapter exists to teach.

## One substantial-rework section

**`#sec-compute-rack`** received the deepest rework. The original had 6 items that tended toward concept statement. The rewrite threads them into a tight physics → provisioning → economics arc:

- MCQ 1: air-cooling physics (4x specific heat, 25x thermal conductivity, 4,500x volumetric flow) rather than generic "power density"
- SHORT 2: synchronous-transient signature of mis-sized cooling (silent MFU loss, not hard crash)
- SHORT 3: **new** — PUE-vs-carbon quantitative application (replacing the flagged PUE FILL)
- MCQ 4: the 3 GW/s di/dt transient argument, grounded in the 1,024-GPU 400 W→700 W step example
- TF 5: rack-failure blast radius in synchronous training
- SHORT 6: **new** — 10 MW facility liquid-vs-air electricity-savings napkin math plus the density-makes-it-infeasible caveat

Every item uses the section's own numeric anchors (33 kW rack, 700 W per GPU, PUE 1.08 vs. 1.50, 2.6 M/year at 4.2 MW saved). The section now models the A-grade "Thesis → Mechanism → System Consequence" pattern from §16 gold-standard SHORT examples.

## Metadata updates

- `generated_on`: "2026-04-24"
- `model`: "claude-opus-4-7" (was "gpt-5.4")
- `improved_by`: "opus-subagent-phase2" (was "quiz-refresh/improve-mode")
- `total_sections`, `sections_with_quizzes`, `sections_without_quizzes` preserved (9 / 9 / 0)
- All `section_id`s preserved exactly; validator confirms every `#sec-*` resolves to a `##` anchor in the chapter qmd.

## Anti-shuffle-bug compliance (§10)

Every MCQ explanation refers to distractors by their content or concept (e.g., "the higher-clock-frequency explanation misses that...", "the compression-of-bits answer invents a mechanism that does not exist"). No `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, or `([A-D])` patterns appear in any answer string. Correct-answer letters are distributed across MCQs rather than clustered on one letter, by constructing choices in-order rather than post-shuffling.
