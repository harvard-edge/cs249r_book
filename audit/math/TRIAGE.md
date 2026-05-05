# MLSysBook Vol I/II Math Audit Triage

Generated from the 68 per-file reports in `audit/math/reports`.

## Status

- Canonical files audited: 68 / 68
- Vol I reports: 33
- Vol II reports: 35
- Source `.qmd` files modified during audit: none
- Audit method: direct sub-agent review, no Gemini

## Fix Order

### P0: Correct Before Any Math-Facing Release

These are high-severity issues where the displayed arithmetic, units, formulas, or conclusion are materially wrong.

| Volume | Report | Main issue class | Why it matters |
|---|---|---|---|
| Vol I | `hw_acceleration.md` | Transformer QKV FLOPs, mixed precision labels, small-batch roofline claims, carbon ROI energy totals | Several hardware-performance conclusions are numerically unsupported or use incomparable precision modes. |
| Vol I | `model_compression.md` | NAS GPU-hours vs GPU-days, FP16-to-INT8 speedup, low-rank compute direction | These are direct arithmetic or algorithmic-complexity errors. |
| Vol I | `training.md` | FlashAttention FLOP-count claim, Adam memory accounting, activation-memory example | Core training math mixes memory, optimizer state, and FLOP/IO effects. |
| Vol I | `model_serving.md` | Seconds-vs-ms batching units, Llama-3-8B KV-cache capacity and GQA | Serving capacity and latency guidance can be wrong by large factors. |
| Vol I | `appendix_assumptions.md` | GPT-3 electricity example omits 1024 accelerators; FLOPS/FLOPs misuse | The appendix is likely reused as a constants reference, so bad constants propagate. |
| Vol I | `appendix_machine.md` | DRAM energy ratio, training-memory decomposition, H100 FP8/INT8 labeling | Foundational machine arithmetic needs consistent units and precision labels. |
| Vol II | `compute_infrastructure.md` | HBM cost, INT4 175B fit, FP16/BF16/FP8 label mix, training time, rack/pod/TCO scale | This chapter has many high-impact order-of-magnitude errors. |
| Vol II | `data_storage.md` | Checkpoint size/interval, HBM-storage bandwidth ratios, cost-tier ratios | Storage sizing and economic conclusions are inconsistent with the stated model. |
| Vol II | `distributed_training.md` | GPT-2 scaling example, RLHF memory/KV cache, convergence theorem, LR scaling | Several central distributed-training examples draw wrong conclusions from the math. |
| Vol II | `collective_communication.md` | Ring AllReduce factor, Ring-vs-Tree crossover, tree bandwidth penalty, LogP overlap | Collective communication formulas need a dedicated correction pass. |
| Vol II | `fault_tolerance.md` | Reliability/checkpointing high-severity issues | Multiple reliability calculations need line-by-line reconciliation before release. |
| Vol II | `security_privacy.md` | DP-SGD noise direction, RDP optimum, moments-accountant sign, Gaussian mechanism notation | Privacy math is conceptually wrong in places, not just numerically rounded. |
| Vol II | `appendix_communication.md` | Compression latency scaling, pipeline bubble table, Ring-vs-Tree selection | Appendix formulas conflict with chapter-level communication math. |
| Vol II | `appendix_fleet.md` | AllReduce/prose contradiction, goodput model, PUE reduction | Fleet reference math has direct contradictions. |

### P1: High Value Consistency Fixes

These are mostly medium-severity issues that recur across chapters and should be fixed as batches.

| Theme | Representative reports | Suggested treatment |
|---|---|---|
| `FLOPS` vs `FLOPs` vs `OPS/TOPS` | `appendix_assumptions.md`, `benchmarking.md`, `security_privacy.md`, `compute_infrastructure.md`, `introduction.md` | Create a style rule: `FLOPs` for total work, `FLOP/s` or `FLOPS` for rate, `OPS/TOPS` only for integer/non-floating operations. |
| Decimal GB vs GiB | `appendix_assumptions.md`, `appendix_machine.md`, `appendix_algorithm.md`, `data_storage.md` | Pick decimal GB for prose examples or explicitly label GiB from hardware capacity constants. |
| Precision-mode consistency | `hw_acceleration.md`, `benchmarking.md`, `compute_infrastructure.md`, `appendix_machine.md` | Do not compare FP16, BF16, TF32, FP8, and INT8 under one unlabeled throughput column. |
| Training-state accounting | `training.md`, `appendix_machine.md`, `distributed_training.md`, `compute_infrastructure.md`, `data_storage.md` | Use explicit categories: weights, gradients, FP32 master weights, Adam first moment, Adam second moment, activations, KV cache. |
| Ring AllReduce and collective notation | `collective_communication.md`, `distributed_training.md`, `appendix_communication.md`, `appendix_fleet.md` | Standardize message size symbol and per-worker volume formula: `2(D-1)/D * M` for Ring AllReduce bandwidth volume. |
| Percent vs percentage points | `frontmatter-about.md`, `ml_workflow.md`, `benchmarking.md`, `responsible_ai.md` | Change metric deltas in accuracy/fairness/availability to percentage points where appropriate. |
| Strong vs weak scaling | `parts-fleet_principles.md`, `parts-distributed_ml_principles.md`, `distributed_training.md` | Label equations explicitly as strong-scaling, weak-scaling, throughput-scaling, or fixed-step comparisons. |

## Files That Look Clean Or Mostly Administrative

Several frontmatter/reference files contain no substantive math and have clean reports:

- Vol I: `index-vol1.md`, `frontmatter-dedication.md`, `frontmatter-foreword.md`, `backmatter-references.md`
- Vol II: `index-vol2.md`, `frontmatter-dedication.md`, `frontmatter-foreword.md`, `frontmatter-acknowledgements.md`, `backmatter-references.md`

## Recommended Next Pass

1. Fix P0 issues in source `.qmd` files, starting with shared constants/notation patterns so repeated errors do not need separate one-off edits.
2. Run a mechanical terminology pass for `FLOPS/FLOPs`, `GB/GiB`, percent/percentage-point, and precision labels.
3. Re-audit the changed files only, using the corresponding report files as checklists.
4. After fixes, render Vol I and Vol II to catch executable-code and displayed-value mismatches.

## Notes

The worktree has pre-existing modified binary/PDF files from checkout/LFS pointer behavior. They are unrelated to the math audit. The audit reports are under `audit/math/`.
