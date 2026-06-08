# Algorithm Pass Audit

Status: implementation complete; final validation in progress. Worktree: `/Users/VJ/GitHub/MLSysBook-algorithm-pass-audit`. Branch: `codex/algorithm-pass-audit`.

## Rendering Decision

Use Quarto's native theorem-style algorithm form:

```markdown
::: {#alg-example name="Example algorithm"}
...

**Systems cost:** ...
:::
```

The native `#alg-...` form rendered successfully to standalone HTML and PDF on Quarto 1.9.36. It avoids duplicate raw-LaTeX/HTML sources and preserves cross-reference labels in both formats.

Quarto renders a bare `@alg-example` as `alg. N`. For textbook prose, use Pandoc citation-prefix syntax instead:

- Mid-sentence: `[algorithm @alg-example]` renders as `algorithm N`.
- Sentence start: `[Algorithm @alg-example]` renders as `Algorithm N`.

Do not add a custom float with `key: alg`: Quarto already defines `algorithm` as a built-in theorem environment, and a custom float using the same key fails PDF compilation with `Command \algorithm already defined`.

## Selection Gate

An algorithm earns a place only when the procedure's structure creates a systems consequence: memory, bandwidth, parallelism, latency, throughput, or energy. The box must be motivated before it appears and cashed out in prose afterward. Policy frameworks, governance checklists, formulas, cost notebooks, and runnable framework idioms remain prose, equations, notebooks, or listings.

## Agent Audit Summary

### Volume 1 Core Development

| Candidate | Decision | Notes |
|---|---|---|
| Adam update, `training.qmd` | Convert | Existing listing is pseudocode. The algorithm should expose the two persistent moment buffers and the resulting parameter-plus-state memory cost. |
| Mini-batch SGD, `nn_computation.qmd` | Convert | Existing listing is already algorithm-like. The payoff must connect batch size to activation memory, GEMM utilization, and HBM pressure. |
| Backpropagation, `nn_computation.qmd` | Add | Add a compact systems-level algorithm while preserving the worked numeric trace. The cost line should motivate checkpointing and recomputation. |
| Reverse-mode autodiff, `frameworks.qmd` | Pair | Preserve concrete path-accumulation intuition while adding a graph/tape algorithm and a short real autograd snippet. |

### Volume 1 Compression

| Candidate | Decision | Notes |
|---|---|---|
| Iterative magnitude pruning | Add if budget allows | Good systems hook: prune/fine-tune cycles cost training compute, and only hardware-visible sparsity speeds inference. |
| PTQ calibration | Pair if budget allows | Good systems hook: calibration sets static ranges and trades 4x byte reduction against saturation/accuracy loss. Avoid notation collisions for clipping endpoints. |
| QAT forward pass | Secondary convert | Existing pseudocode is not runnable. Convert only if the sparse budget expands. |
| Conv-BN-ReLU fusion | Secondary convert | Clean systems algorithm, but overlaps with performance/hardware material and may exceed the sparse target. |

### Volume 2 Distributed and Inference

| Candidate | Decision | Notes |
|---|---|---|
| Ring AllReduce, `collective_communication.qmd` | Convert | Replace phase bullets with an algorithm; do not duplicate the trace notebook. Preserve per-GPU byte accounting and alpha-beta latency. |
| 1F1B pipeline schedule, `distributed_training.qmd` | Pair after correction | First fix overclaims that conflate GPipe bubble reduction with 1F1B activation-memory savings. Preserve NVLink/InfiniBand distinctions. |
| Continuous batching, `inference.qmd` | Add | Keep the algorithm tied to decode iterations, admission/eviction, and KV-cache page management. |
| Speculative decoding, `inference.qmd` | Convert | Convert the three-phase recipe while preserving rejection-sampling exactness and the 7B/70B engineering anchor. |
| Power-of-two-choices routing, `inference.qmd` | Add | Use relative load for heterogeneous fleets; keep H100/A100 capacity-weighted example adjacent. |

### Volume 2 Security and Privacy

| Candidate | Decision | Notes |
|---|---|---|
| DP-SGD, `security_privacy.qmd` | Defer | Strong systems hook, but it needs a citation/notation/accounting pass before conversion. Keep out of this implementation branch rather than add privacy math without the dedicated source check. |
| API protection notebook | Reject | Service-control policy stack, not an algorithm whose structure creates the systems cost. |
| Secure multi-tenancy / trusted compute notebooks | Reject | Arithmetic comparisons and cost notebooks, not procedures. |
| Secure aggregation | Reject for this pass | Not locally developed with protocol recipe or bandwidth/round-trip payoff. |
| Data poisoning and synthetic data loops | Reject | Would drift toward ML-security theory; current figures/prose are the right form. |
| DP decision framework and maturity tables | Reject | Governance/checklist artifacts, not algorithm floats. |

### Additional Read-Only Agent Findings

Later read-only agents found several plausible algorithm candidates in collective communication, distributed training, inference, fleet orchestration, security/privacy, responsible AI, and operations. These were recorded as future audit material, not automatic implementation work. The main accepted additions for this branch were the candidates that already sat on the initial spine and had clear systems-cost payoff: Ring AllReduce, 1F1B, continuous batching, speculative decoding, and power-of-two routing. The following families remain deferred for future task branches if a chapter-level pass needs them:

- **Distributed systems mechanics:** hierarchical AllReduce, rail-aware rank mapping, DDP bucket overlap, ZeRO/FSDP gather-reshard, tensor-parallel transformer blocks, MoE All-to-All dispatch, sharded checkpoint commit, warm restart, FlashAttention tiling, PagedAttention allocation, chunked prefill, topology-aware placement, and memory-aware model routing.
- **Security/privacy/operations mechanics:** threat-model triage, adaptive API extraction defense, runtime containment and rollback, secure boot, DP-SGD accounting, rolling-window fairness monitoring, SISA unlearning, dependency-graph deployment gates, canary controllers, multi-region rollback, shadow replay gates, point-in-time feature joins, and incident attribution.
- **Rejected for this branch:** governance matrices, maturity lists, broad policy frameworks, and trade-off prose that reads better as narrative than as an algorithm float.

## Implemented Spine

This branch keeps the algorithm core sparse across both volumes:

1. Adam update.
2. Mini-batch SGD.
3. Backpropagation.
4. Reverse-mode autodiff.
5. PTQ calibration.
6. Ring AllReduce.
7. 1F1B pipeline schedule, with the GPipe/1F1B bubble-overclaim corrected.
8. Continuous batching.
9. Speculative decoding.
10. Power-of-two-choices routing.

Conv-BN-ReLU fusion, QAT, iterative pruning, DP-SGD, FlashAttention, and PagedAttention remain useful backups or future branch candidates. They should not be added to this branch unless a dedicated chapter-level pass shows that the current prose fails pedagogically without them.

## Validation Notes

- Targeted pre-commit passed on every touched chapter after each task commit.
- HTML renders passed for representative Vol. I and Vol. II chapters after algorithm insertion.
- Vol. II `collective_communication` HTML rendered after Ring AllReduce. The fast Vol. II PDF build reached LaTeX and failed on an existing `\copyrightpage` titlepage issue, not on the algorithm block.
- The algorithm-reference rendering pass normalized all mid-sentence prose references to `[algorithm @alg-...]`; rendered HTML shows `algorithm N`, not `alg. N`, and no `?@alg-` leaks.
- Binder builds must run sequentially in this worktree because the build command rewrites shared `index.qmd` and `_quarto.yml` symlinks per volume/format.

## Checks Before Commit

- Build at least one converted Volume 1 chapter to HTML and PDF.
- Build at least one converted Volume 2 chapter to HTML and PDF.
- Grep rendered outputs for `?@alg-`, `alg.` in authored algorithm references, and leaked raw LaTeX.
- Run targeted pre-commit on touched files.
- Recheck emphasis: algorithm `**Systems cost:**` is a structural label; do not add bold for rhetorical stress.
- Recheck notation and references when adding symbols or quantitative overhead claims.
