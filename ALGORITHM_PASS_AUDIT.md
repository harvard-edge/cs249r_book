# Algorithm Pass Audit

Status: in progress. Worktree: `/Users/VJ/GitHub/MLSysBook-algorithm-pass-audit`. Branch: `codex/algorithm-pass-audit`.

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
| DP-SGD, `security_privacy.qmd` | Pair after source/notation check | Strong systems hook: per-example clipping breaks batch-level gradient reduction and creates memory/throughput tax. Overhead ranges and symbols need validation before conversion. |
| API protection notebook | Reject | Service-control policy stack, not an algorithm whose structure creates the systems cost. |
| Secure multi-tenancy / trusted compute notebooks | Reject | Arithmetic comparisons and cost notebooks, not procedures. |
| Secure aggregation | Reject for this pass | Not locally developed with protocol recipe or bandwidth/round-trip payoff. |
| Data poisoning and synthetic data loops | Reject | Would drift toward ML-security theory; current figures/prose are the right form. |
| DP decision framework and maturity tables | Reject | Governance/checklist artifacts, not algorithm floats. |

## Current Implementation Spine

The sparse core should stay near twelve algorithms across both volumes:

1. Adam update.
2. Mini-batch SGD.
3. Backpropagation.
4. Reverse-mode autodiff.
5. Iterative pruning or PTQ calibration, with preference decided in context.
6. Ring AllReduce.
7. 1F1B pipeline schedule, after correctness repair.
8. Continuous batching.
9. Speculative decoding.
10. Power-of-two-choices routing.
11. DP-SGD, after citation and notation repair.

Conv-BN-ReLU fusion and QAT are useful backups if one of the above fails the narrative or source-support gate, but they should not push the pass past the sparse target.

## Checks Before Commit

- Build at least one converted Volume 1 chapter to HTML and PDF.
- Build at least one converted Volume 2 chapter to HTML and PDF.
- Grep rendered outputs for `?@alg-`, `alg.` in authored algorithm references, and leaked raw LaTeX.
- Run targeted pre-commit on touched files.
- Recheck emphasis: algorithm `**Systems cost:**` is a structural label; do not add bold for rhetorical stress.
- Recheck notation and references when adding symbols or quantitative overhead claims.
