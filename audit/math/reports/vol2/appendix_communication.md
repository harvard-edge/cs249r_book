# Math Audit: `appendix_communication.qmd`

Source audited: `book/quarto/contents/vol2/backmatter/appendix_communication.qmd`

Method: direct reasoning only. No Gemini or external model checks.

## Findings

### 1. High: Compression example incorrectly divides the collective latency term by the compression ratio

- Lines: 120-122, 432-443
- Issue: `compressed_comm_ms = ring_time_ms / compress_ratio` scales the entire Ring AllReduce time by `1/C`. That is not the alpha-beta model for the compressed message. Compression reduces the bandwidth payload term, but it does not reduce the number of Ring startup steps `2(N-1) alpha`.
- Concrete check: For `N=256`, `M=1e9 B`, `beta=50e9 B/s`, `alpha=5 us`:
  - Uncompressed bandwidth term: `2*255/256 * 1e9/50e9 = 39.84375 ms`
  - Latency term: `2*255*5 us = 2.55 ms`
  - Uncompressed total: `42.39375 ms`
  - Correct compressed communication for `C=64`: `39.84375/64 + 2.55 = 3.1726 ms`
  - With `2.0 ms` codec overhead: `5.1726 ms`, for about `8.2x` total speedup
  The source instead computes `42.39375/64 + 2.0 = 2.6624 ms`, for about `15.9x`.
- Proposed correction: Compute compression from the bandwidth and latency terms separately, for example `compressed_comm_ms = ring_bw_ms / C + ring_latency_ms`. Update the rendered compressed communication, total, and speedup values.

### 2. High: Pipeline bubble table reuses the same P=8, M=16 value for every populated cell

- Lines: 115-118, 147-148, 395-400
- Issue: `CF.bubble_std_pct` is computed only once using `P=8, M=16`, so the table renders `30.4%` in every non-dash cell. The table claims to show variation by pipeline depth and microbatch count, but it does not.
- Correct values from `b = (P-1)/(P-1+M)`:
  - `P=4, M=8`: `3/11 = 27.3%`
  - `P=4, M=16`: `3/19 = 15.8%`
  - `P=8, M=16`: `7/23 = 30.4%`
  - `P=8, M=32`: `7/39 = 17.9%`
  - `P=8, M=64`: `7/71 = 9.9%`
- Proposed correction: Precompute per-cell values or render literal checked values. Do not reuse `CF.bubble_std_pct` except for the `P=8, M=16` cell.

### 3. High: Ring-vs-Tree algorithm selection uses point-to-point crossover instead of collective crossover

- Lines: 198-214, 346-361, 486-488
- Issue: `M_cross = alpha beta` is the point-to-point latency/bandwidth crossover. It is not the Ring-vs-Tree AllReduce decision boundary. With the appendix's own AllReduce formulas, the crossover solves:
  `2(N-1)/N * M/beta + 2(N-1)alpha = 2log2(N) * M/beta + 2log2(N)alpha`.
- Concrete check for the appendix's worked setting, `N=256`, `alpha beta = 250 KB`, `log2 N = 8`:
  `M = ((255 - 8) / (8 - 255/256)) * alpha beta ≈ 35.3 * 250 KB ≈ 8.8 MB`.
  Therefore messages at `10 * alpha beta = 2.5 MB` are still Tree-favored by the stated formulas, contradicting the table's "Large messages" row.
- Proposed correction: Keep `alpha beta` as the point-to-point latency/bandwidth crossover, but introduce a separate Ring-vs-Tree collective crossover. For AllReduce, use the equation above, or state that recursive halving-doubling/tree implementations should be benchmarked in the intermediate region.

### 4. Medium: Broadcast formula omits the per-level bandwidth cost under the appendix's atomic alpha-beta model

- Lines: 222-233, 313-317, 330
- Issue: The appendix says alpha-beta treats a message as a single atomic transfer, then gives Tree Broadcast as `T = M/beta + log2(N) alpha`. In a store-and-forward tree with an atomic `M`-byte message at each level, the critical path pays `log2(N)` full-message transmissions: `log2(N) * (alpha + M/beta)`. The stated `M/beta + log2(N)alpha` requires segmentation/pipelining across tree levels, which the preceding alpha-beta model explicitly excludes and the broadcast prose does not state.
- Proposed correction: Either change the formula/table to `log2(N) * M/beta + log2(N)alpha` for the simple atomic model, or explicitly say the formula assumes a pipelined/segmented broadcast whose steady-state bandwidth term approaches `M/beta`.

### 5. Medium: Collective summary says Ring algorithms move nearly one message, but Ring AllReduce moves nearly two

- Lines: 249-251, 323-334, 486-487
- Issue: Line 334 says bandwidth-optimal Ring algorithms share the `(N-1)/N` factor and move "nearly equal to the message size." Ring AllReduce has a `2(N-1)/N` bandwidth coefficient, so each participant sends/receives nearly `2M`, not `M`. The statement is true for Ring AllGather, ReduceScatter, and AllToAll as written, but not for Ring AllReduce.
- Proposed correction: Say "one ring pass moves about `M`; Ring AllReduce uses two passes and moves about `2M` per participant." Update the summary bullet similarly.

### 6. Medium: Hierarchical AllReduce prose implies inter-node bandwidth volume shrinks by node-local reduction

- Lines: 365-371
- Issue: Reducing the intra-node group from `N` GPUs to `N/8` nodes reduces the inter-node Ring latency steps, but the per-node result is still the full `M`-byte gradient. The inter-node AllReduce bandwidth term remains approximately `2M/beta_ib`, not `2(M/8)/beta_ib`. Line 371 says the method confines "bandwidth-hungry phases" to fast intra-node links, which overstates the bandwidth benefit.
- Proposed correction: State that hierarchy primarily reduces inter-node latency and exploits faster NVLink for the intra-node reduce/broadcast phases. If claiming inter-node bandwidth reduction, specify a sharded/reduce-scatter variant and derive that separate model.

### 7. Medium: Interleaved pipeline prose overstates the exact V-fold bubble reduction

- Lines: 404-408, 489
- Issue: The denominator in `b_interleaved = (P-1)/(P-1+M V)` does not grow by an exact factor of `V` relative to `P-1+M`; only the microbatch term does. For `P=8`, `M=16`, `V=4`, the standard bubble is `7/23 = 30.4%` and the interleaved bubble is `7/71 = 9.9%`, a `3.1x` reduction, not `4x`.
- Proposed correction: Replace "denominator grows by a factor of `V`" and "reduces bubbles by a factor of `V`" with "the microbatch term grows by `V`; the reduction approaches `V` when `M` dominates `P-1`."

### 8. Medium: LogGP extension charges only one overhead term for k messages

- Lines: 222-230
- Issue: The text defines `o` as processor overhead to initiate or complete a message, then uses `+ o` for `k` messages. If overhead is per message, the term should scale with `k` (and often separately for send/receive endpoints). As written, the model says splitting a transfer into many messages adds gaps and bandwidth time but only one processor overhead, contradicting the definition.
- Proposed correction: Use a consistent simplified form such as `T = alpha + (k-1)g + k*m/beta + k*o` if `o` is per message, or explain that `o` is a one-time aggregate overhead for the whole transfer.

### 9. Low: Pipeline rule-of-thumb uses "below 20 percent" but equality gives exactly 20 percent

- Lines: 391, 400, 472
- Issue: From `b = (P-1)/(P-1+M)`, the condition `b < 0.2` gives `M > 4(P-1)`. At `M = 4(P-1)`, the bubble fraction is exactly `20%`, not below it. The text's `M >= 4(P-1)` wording is an off-by-one/strictness issue for integer microbatch counts.
- Proposed correction: Use "at or below 20 percent" for `M >= 4(P-1)`, or use `M > 4(P-1)` for "below 20 percent." The looser `M >= 4P` practical rule is sufficient but should be described as conservative.

### 10. Low: Compression break-even table is only valid when latency is negligible or unchanged

- Lines: 424-428, 451-458
- Issue: The table computes maximum codec overhead as `T_comm(M) * (1 - 1/C)`. That is correct for a purely bandwidth-dominated `T_comm` or if `T_comm` means only the bandwidth term. For a full collective time `T_comm = A M + L`, the exact savings under the same collective schedule is `A M (1 - 1/C)` because the latency term `L` cancels.
- Proposed correction: Clarify the table header/caption to say the AllReduce times are bandwidth-term times or bandwidth-dominated approximations. For latency-significant collectives, subtract only the reduced bandwidth portion.

## Overall Assessment

The basic alpha-beta units and the main Ring AllReduce worked example are mostly sound: `400 Gbps = 50 GB/s`, `5 us * 50 GB/s = 250 KB`, and a 1 GB FP16 gradient for 500M parameters are consistent. The main risks are model-boundary errors: using point-to-point crossover for collective algorithm choice, scaling latency away during compression, and presenting one pipeline-bubble calculation as a full table. These should be fixed before relying on the appendix for algorithm-selection or compression-payback guidance.
