# Math Audit: `collective_communication.qmd`

Source audited: `book/quarto/contents/vol2/collective_communication/collective_communication.qmd`

Method: direct reasoning only. No Gemini or external model checks.

## Findings

### 1. High: Opening 70B Ring AllReduce estimate omits the Ring factor and mislabels the bottleneck

- Lines: 59, 69, 157-175
- Issue: Line 59 estimates BF16 70B gradient synchronization as `140 GB / 50 GB/s = 2.8 s`. Ring AllReduce bandwidth time is not `M / beta`; it is `2(N-1)/N * M / beta`, which approaches `2M/beta`. For 1,000 GPUs, this is `2 * 999/1000 * 140 / 50 = 5.5944 s`, before latency. The sentence also says this dominates the `L_lat` term, but the calculation is a bandwidth term, not a latency term.
- Proposed correction: Replace the estimate with `2 * (999/1000) * 140 GB / 50 GB/s ≈ 5.6 s`, and say it dominates the bandwidth/communication term rather than `L_lat`. If the intended example is a one-way transfer, do not describe it as a Ring AllReduce.

### 2. High: Ring-vs-Tree crossover formula drops a non-constant factor

- Lines: 1074-1114, 1118-1183, 2041, 2069, 2128
- Issue: The derivation correctly reaches `M_crossover ≈ N alpha beta / log2 N` on line 1101, then replaces it with `N alpha beta` on lines 1103-1104. The removed `log2 N` factor is not a constant; it is 6.6 for `N=100`, 10 for `N=1024`, etc. This materially changes algorithm choice.
- Concrete check: In the worked example, `N=64`, `alpha=10 us`, `beta=10 GB/s`. Using the chapter's own equations:
  - Ring: `1260 us + 196.875 us/MB * M`
  - Tree: `120 us + 1200 us/MB * M`
  - Exact crossover: `M = 1140 / 1003.125 ≈ 1.14 MB`
  The stated rule gives `64 * 10 us * 10 GB/s = 6.4 MB`, off by about `5.6x`.
- Proposed correction: Use the exact crossover from the stated equations:
  `M = beta * alpha * [(N-1) - log2 N] / [log2 N - (N-1)/N]`.
  For large `N`, simplify to `M ≈ N alpha beta / log2 N`, not `N alpha beta`. Update the 25 MB example, the 6.4 MB example, the fallacy section, summary, and takeaways consistently.

### 3. High: Tree bandwidth penalty is internally inconsistent

- Lines: 938-944
- Issue: The Tree formula on line 939 has bandwidth term `2 log2(N) M/beta`. Ring's term is `2(N-1)/N M/beta`, approximately `2M/beta`. Therefore the bandwidth-time ratio is roughly `log2 N`, not `log2 N / 2`. For `N=64`, the ratio is `12 / 1.96875 ≈ 6.1x`, not `3x`.
- The 175B example is also inconsistent: for `M=350 GB` and `log2(1000)≈10`, the Tree bandwidth term corresponds to about `2 * 10 * 350 GB = 7 TB` of per-critical-path transfer equivalent, while Ring is about `700 GB`. The text says `3.5 TB`, which is only half the stated formula.
- Proposed correction: Change the `64 GPUs` statement to about `6x` more bandwidth time, or explicitly define a one-direction-only byte count and keep that convention throughout. Change the 175B example to `~7 TB` under the stated formula, or revise the formula/prose together if a different tree model is intended.

### 4. High: LogP overlap example contradicts its own non-overlap statement

- Lines: 448-455, 462-492, 1901, 2049
- Issue: The prose says processor overhead `o` cannot overlap with compute, and the example says `2o = 100 us` is exposed. But the effective time is computed as `max(compute, L + 2o) = 500 us`, which hides the `2o` inside the compute interval. If `2o` is truly non-overlappable, the simple timeline should be `2o + max(compute, L) = 100 + 500 = 600 us` when compute and network latency overlap but initiation/completion overhead does not.
- Proposed correction: Either change the formula to `T = 2o + max(T_compute, L)` and the result to `600 us`, or soften the prose to say some overhead can overlap with host/GPU work depending on implementation. The current example cannot simultaneously say `2o` is exposed and conclude the total remains `500 us`.

### 5. Medium: FSDP communication-volume statement is wrong at the operation level

- Lines: 780-788
- Issue: Line 784 says FSDP issues `2L` collectives, "but each operation transfers approximately the same total bytes as the single AllReduce would." That contradicts the previous clause that each operation is smaller and layer-local. Per operation, an FSDP AllGather/ReduceScatter transfers only that layer's shard traffic; summed across layers and both phases, the total per-step bandwidth is comparable to a full-gradient AllReduce.
- Proposed correction: Replace with: "For a model with `L` layers, FSDP issues `2L` collectives per step instead of one. Each collective is layer-sized, and the sum across all layers is comparable to the single full-gradient AllReduce bandwidth volume, but spread across many smaller operations."

### 6. Medium: AllGather and ReduceScatter use inconsistent `M` notation

- Lines: 654-662
- Issue: For AllGather, the text says each worker starts with `x_i` and ends with `[x_0,...,x_{N-1}]`, then says Ring bandwidth cost is `(N-1)/N * M/beta` and "total data grows to `N x M` per worker." These are only simultaneously true under different definitions of `M`. If `M` is the final concatenated tensor size, cost is `(N-1)/N * M/beta` and the final data is `M`. If `M` is each worker's input size, final data is `N M` and the cost is `(N-1)M/beta`.
- ReduceScatter has the same ambiguity.
- Proposed correction: Define `M` once for this table. Recommended: let `M` be the total logical tensor size after concatenation/reduction. Then AllGather and ReduceScatter Ring bandwidth terms are `(N-1)/N * M/beta`, and each worker's local input/output shard is `M/N`.

### 7. Medium: Naive collective comparison understates/ambiguously states the slowdown

- Lines: 627, 798-806
- Issue: Line 627 says a naive reduce-then-broadcast costs `O(N n/beta)` and is `500x` worse for `N=1024`. A parameter-server/star root receives and sends about `2(N-1)M` bytes through the root, while Ring costs about `2M` per worker for large `N`, so the root bottleneck ratio is about `N`, i.e. roughly `1024x`, not `500x`. A `500x` ratio would correspond to comparing only one phase or using a different baseline, but that is not stated.
- Proposed correction: Change to "roughly `N x` worse at the root (`~1024x` for `N=1024`) compared with Ring's per-worker bandwidth term" or explicitly define the comparison that yields `500x`.

### 8. Medium: Figure code for Tree uses a different bandwidth model than the prose

- Lines: 987-1022, 1078-1083
- Issue: The figure code computes `t_tree` bandwidth as `(2 log2 N) * M/beta * 0.5`, i.e. `log2 N * M/beta`. The prose formula immediately afterward uses `2 log2 N * M/beta`. This shifts the plotted Ring/Tree crossover by about `2x` and makes the figure inconsistent with the derivation.
- Proposed correction: Remove the `* 0.5` in the figure code, or revise the prose formula if the intended model assumes bidirectional/tree parallelism that halves the effective bandwidth cost.

### 9. Medium: Hierarchical 70B code comment is stale relative to computed output

- Lines: 522-525, 551-586, 592-598
- Issue: The comment says the hierarchical total is approximately `399 ms`, but the code computes about `928 ms`:
  - Intra ReduceScatter: `(140 * 7/8) / 900 * 1000 ≈ 136.1 ms`
  - Inter Ring: `2 * 15/16 * 17.5 / 50 * 1000 ≈ 656.25 ms`
  - Latency: `0.09 ms`
  - Intra AllGather: `136.1 ms`
  - Total: `~928.5 ms`
  The rendered prose uses the computed value, but the source comment is mathematically stale.
- Proposed correction: Update the comment to `~929 ms vs ~5,557 ms`, or change assumptions if the intended result is `399 ms`.

### 10. Medium: SHARP byte-reduction claim overstates what in-network reduction changes

- Lines: 1514-1520
- Issue: The text says SHARP reduces bytes traversing each network link by aggregating at switches, whereas a standard Tree AllReduce "sends the full message `L` times." A software tree also aggregates partial sums at internal nodes; each tree edge carries an `M`-sized partial/result per phase. SHARP primarily removes endpoint GPU involvement, memory traffic, and store-and-forward/software latency; it does not generally reduce the asymptotic per-edge payload of a correct tree reduction by `L x`.
- Proposed correction: Reframe the benefit as reducing GPU memory traffic, host/device synchronization, and per-hop processing latency, with possible fabric-level efficiency improvements depending on topology. Avoid claiming an `L x` reduction in link bytes unless a specific non-aggregating baseline is defined.

### 11. Medium: Torus dimension-ordered reduction description mixes full AllReduce and reduce-scatter math

- Lines: 1538-1546
- Issue: The text says each dimension is itself a Ring AllReduce, then claims total bandwidth cost `2M/beta`, same as a single Ring. If full `M` is AllReduced independently along X, Y, and Z, the bandwidth cost is closer to `2M[(X-1)/X + (Y-1)/Y + (Z-1)/Z]/beta`, about `6M/beta` for large equal dimensions. A bandwidth-optimal multidimensional algorithm can approach `2M/beta`, but it must use reduce-scatter/allgather partitioning across dimensions, not full-message Ring AllReduce in every dimension.
- The sentence saying the X ring "already holds the partial sum from the `Y x Z` TPUs in its YZ plane after the first step" is also dimensionally reversed: reducing along X gives sums across the X dimension for each fixed `(Y,Z)` line, not across the entire YZ plane.
- Proposed correction: Either describe a true multidimensional reduce-scatter/allgather and derive its bandwidth, or change the bandwidth claim for per-dimension full Ring AllReduce. Correct the partial-sum description to "for each fixed Y,Z line."

### 12. Medium: Error-feedback "unbiased estimator" wording is too strong

- Lines: 1647-1663, 1737-1743, 2131
- Issue: The telescoping identity shows conservation over time if the residual remains bounded: `sum v_t = sum g_t + e_1 - e_{T+1}`. That is not the same as making each compressed gradient an unbiased estimator. Error feedback controls accumulated bias over time; it does not generally make Top-K or deterministic quantization unbiased at each step.
- Proposed correction: Replace "unbiased estimator over time" and "turns biased estimators into unbiased ones over time" with "residual-corrected over time" or "asymptotically preserves the average gradient under bounded residual assumptions."

### 13. Low: AllToAll complexity line omits time units and needs `M` definition

- Lines: 666-669
- Issue: The AllToAll line says "bandwidth cost is `(N-1)/N M` per worker" without dividing by `beta`. It is a byte volume, not a time cost. As with AllGather, `M` must be defined as either total per-worker payload or global payload.
- Proposed correction: Write "per-worker byte volume is `(N-1)/N M` if `M` is the total local send buffer; time is approximately `((N-1)/N M)/beta_eff` plus contention/latency terms."

### 14. Low: MoE AllToAll example mixes KB and MB conventions

- Lines: 706-714, 748-754
- Issue: The example computes `kb_per_token = bytes / 1024` but later converts KB to MB by dividing by `1000` and converts KB to bytes with `1e3`. This produces `1.792 MB` for `1792 KiB`, while binary conversion gives `1.75 MiB`. The difference is small but the comments call it "binary KB."
- Proposed correction: Use decimal units throughout (`4.096 KB/token`) or binary units throughout (`4 KiB/token`, `1.75 MiB total`, bandwidth conversion using `1024`).

### 15. Low: Compression threshold uses `n` instead of `n*`

- Lines: 1775, 2033, 2127
- Issue: Line 1775 says compression helps when gradients are bandwidth-bound (`M > n`) and not when latency-bound (`M < n`). The threshold variable introduced earlier is `n^* = alpha beta`; plain `n` is the message size in the alpha-beta model.
- Proposed correction: Change to `M > n^*` and `M < n^*`, or use `message size > alpha beta`.

### 16. Low: Compression-payback code comment is stale

- Lines: 2077-2084, 2090-2105, 2111-2119
- Issue: The comment says "Show: speedup ≈ 3.5x", but the code and rendered prose compute `40 / (40/8 + 2) = 5.7x`.
- Proposed correction: Update the comment to `speedup ≈ 5.7x`, or change the ratio/overhead if `3.5x` is intended.

## Overall Assessment

The chapter has many correct first-order formulas, especially the basic alpha-beta model, Ring AllReduce bandwidth term, critical message size examples, and most unit conversions. The main mathematical risk is not arithmetic but consistency: several prose explanations simplify formulas beyond validity, especially the Ring/Tree crossover and Tree bandwidth penalty. Fixing those repeated claims should be prioritized because they drive algorithm-selection guidance in the chapter summary and takeaways.
