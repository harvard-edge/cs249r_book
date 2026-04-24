# vol1/hw_acceleration — Quiz Improvement Change Log

**Chapter position**: 11 of 33 (late-Vol1 advanced tier per §7)
**Prior grade**: B (68 questions, minor fixes recommended)
**Target grade**: A

## Summary counts

| Action | Count |
|---|---|
| Questions rewritten (substantial) | 31 |
| Questions kept (A-grade already) | 37 |
| Questions deleted | 0 |
| Questions added | 0 |
| Total questions | 68 (unchanged) |

Section-level question counts are preserved (±0 per section).

## Three issue-pattern clusters fixed

### 1. `trivia_fill` — FILLs rewritten to require reasoning from mechanism

Audit flagged 4 FILLs as close to glossary lookups:

- **Acceleration Fundamentals #5** (arithmetic intensity): Rewrote so the student must infer the term from two profile traces (40 GB / 8 TFLOPs vs 2 GB / 6 TFLOPs) and predict which kernel benefits from added compute. Gold-standard pattern §16.4 FILL-2.
- **AI Compute Primitives #5** (fringe tax): Replaced label-recall FILL with a scenario-based MCQ quantifying 127/128 = 99% waste when 257 output channels hit a 128-wide unit; teaches the design consequence (aligned shapes), not the coined label.
- **Roofline #4** (ridge point): Rewrote to require inference from a measured 0.8 FLOP/byte vs 156 FLOP/byte gap, teaching when to cut bytes rather than just naming the threshold.
- **Summary #3** (computing): Replaced chapter-mantra quote-completion FILL with a scenario MCQ forcing the reader to select between two chips given concrete FLOPS/bandwidth ratios and a 1 FLOP/byte workload.

### 2. `easy_tf` and `throwaway_distractor` — Misconception-resistant rewrites

- **Runtime Support #4** (easy_tf): Rewrote the "fragmentation and multi-tenant interference change performance" TF into a quantitative claim (isolated 10 ms p99 → 2–5× worse under multi-tenant load via HBM contention + L2 pollution + thermal changes).
- **Multi-Chip Scaling #4** (throwaway_distractor): Replaced "stop using parallelism altogether" and "thermal disappears" distractors with quantitatively grounded alternatives; correct choice now names the aggregation mechanism (per-device MTTF constant, aggregate scales linearly with device count).
- **Acceleration Fundamentals #3** (TF on peak FLOPS): Anchored in LayerNorm at ~0.5 FLOP/byte (concrete number) rather than vague "depends on bandwidth."

### 3. `recall_only`, `vague_lo`, `build_up_violation` — Integrative rewrites

- **Acceleration Fundamentals #2** (SHORT on Amdahl): Now requires computing the 1/(0.2 + 0.8/500) ≈ 4.98× ceiling and reasoning about diminishing returns when GPU throughput doubles again; assumes Amdahl as prior vocab per §8.
- **Hardware Specialization #4** (ORDER, recall_only): Converted chronological ordering into causal ordering — each wave's lessons enable the next. Swap consequence is now substantive (mass parallelism without an offload playbook has no market path).
- **Hardware Mapping #1 LO** (vague_lo): Tightened from "Identify the components..." to "Classify placement, memory allocation, and execution order as the three joint dimensions...".
- **Hardware Mapping #4** (recall_only ORDER → scenario MCQ): Replaced the formulaic order-the-mapping-dimensions item with a scenario diagnosing a mapping failure (95% HBM saturation + 18% PE utilization → allocation, not placement).
- **Roofline #6** (forward_reference on KV caching): Removed KV-cache vocabulary; replaced with fusion/quantization/continuous-batching framing that stays within the chapter's scope.

## Substantial-rework section: Roofline Model

All six questions in this section were rewritten to exploit the chapter's own concrete ridge-point numbers (A100 = 156 FLOP/byte, V100 ≈ 139, H100 ≈ 296, LayerNorm ≈ 0.5, ReLU ≈ 0.125). Key upgrades:

- **Q1**: Now grounds classification in the actual A100 ridge point (156 FLOP/byte), not generic roofline vocabulary.
- **Q2**: SHORT now requires quantitative analysis (2 → 200 FLOP/byte at batch 1 vs 128) and names the lever explicitly.
- **Q3**: LayerNorm's "handful of operations per element" is now the diagnostic signature rather than a label.
- **Q4**: TF now uses the A100 = 156, H100 = 296 gap to show how a 200 FLOP/byte kernel flips regime across generations — the rising-ridge-point lesson §16.3 TF-5 models.
- **Q5**: FILL reasoned from 0.8 vs 156 FLOP/byte, not vocabulary.
- **Q6**: Distractor "more compute silicon" now explicitly refuted by the rising-ridge point (more FLOPS moves the kernel further from the ceiling on memory-bound workloads).

## Validator status

- **JSON parse**: OK
- **Anchor validation**: 14 section_ids match 14 `##` anchors in source QMD
- **Metadata counts**: `total_sections=14`, `sections_with_quizzes=14`, `sections_without_quizzes=0` — match actual content
- **Anti-shuffle-bug (§10)**: Zero MCQ answers contain `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, or `([A-D])` patterns; all distractor refutations are content-based
- **Question counts per section**: All sections 3–7 questions (within 4–6 tier-1 window; hardware-sustainability and summary in 2–3 tier-2 window)
- **Output**: `OK: vol1_hw_acceleration_quizzes.json passes schema + anchor validation` (no errors, no warnings)

## Metadata updates

- `generated_on`: `"2026-04-24"` ✓
- `model`: `"claude-opus-4-7"` ✓
- `improved_by`: `"opus-subagent-phase2"` ✓
