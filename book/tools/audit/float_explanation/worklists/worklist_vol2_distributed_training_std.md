# Float Exposition Worklist — `distributed_training.qmd` (vol2)

Graded against the Float Exposition Standard. Caption, fig-alt, in-figure labels, code comments, and callout interiors do not count toward the prose's job. Only running body prose is assessed.

---

## Summary Table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:-----:|:------:|:--:|:--:|:--:|
| Algorithm | 🔴 | 1 | 1 | 0 | 0 |
| Equation | 🔴 | 1 | 0 | 1 | 0 |
| Figure | 🟠 | 18 | 13 | 5 | 0 |
| Table | 🟠 | 9 | 5 | 3 | 1 |
| **Total** | | **29** | **19** | **9** | **1** |

---

## Findings (⚠️ and 🛑 only)

---

### `eq-distributed-training-scaling-efficiency` (equation 🔴) — def L963

**Ref sentence:** No `@eq-` cross-reference anywhere in body prose; equation is presented inline as a display block with surrounding explanation but never formally cited.

**Missing move:** Citation. The prose at L961 introduces and explains the equation fluently ("Scaling efficiency is defined as...where $T_N$...An efficiency of 1.0 means perfect linear scaling"), so the Interpret move is present. What is absent is a formal citation by cross-reference label, meaning the equation is technically orphaned by the scanner and violates the "no orphans" rule.

**Where takeaway currently lives:** Body prose L961-965 and the definition callout at L967 deliver the full symbol interpretation and consequence. The prose teaches the equation adequately; it simply never cites `@eq-distributed-training-scaling-efficiency` by name.

**Rule-compliant diff rewrite (add citation to the lead-in at L961):**

> Before: "Scaling efficiency is defined as:"
>
> After: "Scaling efficiency, shown in @eq-distributed-training-scaling-efficiency, captures this gap:"

Alternatively, append the reference to L965: "...An efficiency of 0.5 means we achieve only half the expected speedup (@eq-distributed-training-scaling-efficiency)."

---

### `fig-sync-model-timeline` (figure 🟠) — def L548

**Ref sentence (L536):** "The key trade-offs across synchronization models are summarized in @tbl-sync-models, and @fig-sync-model-timeline illustrates how each strategy schedules work across workers over time."

**Missing move:** Interpret. The cite sentence points ("illustrates how each strategy schedules work") but delivers no conclusion from the timeline. The payoff at L552 says "The choice of synchronization model directly affects both system throughput and model convergence" — a generic assertion about the topic, not a statement of what the timeline *shows*. The visual insight (that BSP has a hard synchronization cliff at the barrier, SSP shows a bounded-lag band, and async shows workers fully decoupled) is never stated in body prose.

**Where takeaway currently lives:** Caption only.

**Rule-compliant diff rewrite (expand the cite sentence at L536):**

> Before: "The key trade-offs across synchronization models are summarized in @tbl-sync-models, and @fig-sync-model-timeline illustrates how each strategy schedules work across workers over time."
>
> After: "The key trade-offs across synchronization models are summarized in @tbl-sync-models. @Fig-sync-model-timeline shows what those trade-offs look like on a wall-clock timeline: BSP workers align at a hard synchronization boundary each step, SSP workers drift by at most $s$ steps before the fastest stalls, and async workers proceed entirely independently with no alignment point. The timeline makes concrete why BSP throughput is bounded by the slowest worker and why async throughput comes at the cost of no convergence guarantee."

---

### `fig-zero-memory` (figure 🟠) — def L906

**Ref sentence (L904):** "ZeRO addresses this redundancy through progressive sharding, as @fig-zero-memory illustrates and @tbl-zero-stages summarizes."

**Missing move:** Interpret. The cite sentence is a bare double pointer ("illustrates...summarizes") with no takeaway from the figure. The body prose that follows (L918 onward) explains the ZeRO stages in detail, but it is keyed to the table, not to the figure. The bar chart's core visual message — that each successive ZeRO stage takes a step-change reduction in per-GPU memory, and that ZeRO-3 achieves near-linear scaling — is never stated as a conclusion from the figure in body prose.

**Where takeaway currently lives:** Caption ("ZeRO-3 achieves linear memory scaling, enabling models with 100B+ parameters to fit on accelerators that could not hold the replicated state").

**Rule-compliant diff rewrite (add an Interpret sentence after the figure at L908/before the table):**

> Before: (table immediately follows figure at L910)
>
> After (insert between figure close and table): "The bar chart makes the progression concrete: each ZeRO stage cuts per-GPU memory by roughly half relative to the previous, and ZeRO-3 reduces per-GPU footprint to $1/N$ of the DDP baseline, making the memory scaling independent of model size and proportional only to worker count."

---

### `fig-critical-batch-size` (figure 🟠) — def L1499

**Ref sentence (L1497):** "@Fig-critical-batch-size illustrates this relationship between batch size and training efficiency."

**Missing move:** Interpret. The cite sentence is a pure float-announcer. The three-regime characterization appears in the bullet list at L1489-1495 (body prose), so the Lead-in is present, but that bullet list precedes the citation and the figure. After the figure (L1503), the prose pivots immediately to "implications for system design" without ever stating what the figure itself *shows* as a conclusion. The Interpret move (the visual message of the curve: flat efficiency below $B^{*}$, then declining above it) is never named in body prose.

**Where takeaway currently lives:** Caption.

**Rule-compliant diff rewrite (replace the bare cite sentence at L1497):**

> Before: "@Fig-critical-batch-size illustrates this relationship between batch size and training efficiency."
>
> After: "Below $B^{*}$, the curve in @Fig-critical-batch-size is flat: each additional worker reduces sample variance proportionally and training cost falls linearly with hardware count. Above $B^{*}$, the curve bends downward — throughput still increases, but the total samples required to reach the same loss increases faster, so hardware efficiency erodes. The bend is the signal that the communication and synchronization cost of more workers now outweighs the statistical benefit of a larger batch."

---

### `fig-comm-convergence-tradeoff` (figure 🟠) — def L1704

**Ref sentence (L1702):** "The fundamental trade-off in distributed training is between communication efficiency and convergence quality. @Fig-comm-convergence-tradeoff visualizes this trade-off space."

**Missing move:** Interpret. The prose states the topic ("fundamental trade-off") then points to the figure. The payoff at L1708 ("Several techniques occupy different positions on this trade-off curve") transitions directly into a catalog of techniques. No body prose identifies what makes the Pareto frontier significant, which cluster of configurations dominates, or what the practical implication of a point being off the frontier means. The positions of BSP (high convergence, low throughput) and ASP (high throughput, degraded convergence) are in the caption, not in body prose.

**Where takeaway currently lives:** Caption.

**Rule-compliant diff rewrite (expand the cite paragraph at L1702):**

> Before: "The fundamental trade-off in distributed training is between communication efficiency and convergence quality. @Fig-comm-convergence-tradeoff visualizes this trade-off space."
>
> After: "The fundamental trade-off in distributed training is between communication efficiency and convergence quality. @Fig-comm-convergence-tradeoff maps configurations onto this space: BSP anchors the upper-left corner, sacrificing throughput for full convergence guarantees, while async methods anchor the lower-right, recovering that throughput at the cost of gradient staleness. Gradient compression and SSP cluster near the Pareto frontier, the configurations where improving either metric requires giving up something in the other. The practical implication is that no configuration dominates: the right choice depends on which constraint the cluster is hitting."

---

### `fig-model-parallel-flow` (figure 🟠) — def L1799

**Ref sentence (L1797):** "Tensor parallelism, illustrated in @fig-model-parallel-flow, addresses this by partitioning individual weight matrices across devices..."

**Missing move:** Interpret (partial). The cite sentence names the mechanism at a high level (partitioning weight matrices), but the figure's specific mechanism — column-splitting, partial outputs, and AllGather recombination — is not explained at this point of introduction. The prose at L1797 continues with memory numbers but does not describe what the figure actually shows. The column/row split detail is deferred all the way to @fig-tensor-parallel-split at L2167. A reader looking at @fig-model-parallel-flow at L1799 sees the AllGather recombination and has no prose to explain it.

**Where takeaway currently lives:** Caption ("column-splits a single weight matrix across four GPUs, with each GPU computing a partial output that is recombined through an AllGather").

**Rule-compliant diff rewrite (extend the cite sentence at L1797 to add the mechanism):**

> Before: "Tensor parallelism, illustrated in @fig-model-parallel-flow, addresses this by partitioning individual weight matrices across devices..."
>
> After: "Tensor parallelism, illustrated in @fig-model-parallel-flow, addresses this by column-splitting individual weight matrices so each GPU holds a distinct vertical slice and computes a partial result; the partial results are recombined via AllGather before the output passes to the next layer. This keeps the per-GPU parameter footprint proportional to $1/t$ of the full matrix while preserving the mathematical output of the full multiply..."

---

### `fig-moe-all-to-all-routing` (figure 🟠) — def L2268

**Ref sentence (L2261):** "The training process introduces a distinct communication pattern, as @fig-moe-all-to-all-routing illustrates:"

**Missing move:** Interpret (partial). The citation uses a float-announcer colon that causes the figure to carry the explanation: the numbered list that follows (1. Gating, 2. All-to-All Dispatch, 3. Computation, 4. All-to-All Combine) is body prose, which is positive. But the cite sentence itself is a pure pointer, and no sentence after the figure states what is *distinctive or costly* about the pattern the figure demonstrates before the payoff pivot at L2272. The figure's core insight — that every GPU potentially exchanges tokens with every other GPU (all-to-all bisection stress) — is never stated as a lead-out.

**Where takeaway currently lives:** Caption and payoff L2272 (payoff names the bandwidth and load-imbalance constraint but as a separate paragraph, not as a reading of the figure).

**Rule-compliant diff rewrite (replace the colon-ending cite sentence at L2261 and add a payoff sentence after the numbered list):**

> Before: "The training process introduces a distinct communication pattern, as @fig-moe-all-to-all-routing illustrates:"
>
> After: "The training process introduces a distinct communication pattern. @Fig-moe-all-to-all-routing shows the four-phase token shuffle that MoE routing requires:"
>
> (Numbered list stays as is.)
>
> Add after the numbered list (before the figure or immediately after it closes): "The figure makes the cost visible: unlike tensor or pipeline parallelism, where communication flows between adjacent stages or within a node, the dispatch and combine phases require every GPU to potentially send tokens to and receive tokens from every other GPU. This all-to-all pattern stresses bisection bandwidth in proportion to the number of active experts, and any load imbalance — some experts receiving far more tokens than others — leaves hardware idle on the under-loaded side while the over-loaded expert becomes the critical path."

---

### `tbl-sync-models` (table 🟠) — def L544

**Ref sentence (L536):** "The key trade-offs across synchronization models are summarized in @tbl-sync-models..."

**Missing move:** Interpret. The cite says "summarized" — a bare pointer. The payoff (L546) reads: "The same trade-off becomes clearer when the schedules are placed on a timeline" — a transition sentence to the next float, not a conclusion from this table. The H&P standard requires a "the key result is" sentence. The preceding prose (L526-534) explains each model, but it precedes the table and does not extract a row-level conclusion from the table cells. No sentence in body prose states the sharpest contrast visible in the table (that BSP is fully bounded by the slowest worker while async has maximum throughput but zero consistency).

**Where takeaway currently lives:** Caption.

**Rule-compliant diff rewrite (add a payoff sentence after the table at L545/546):**

> Before: "The same trade-off becomes clearer when the schedules are placed on a timeline."
>
> After: "The table's sharpest contrast is the middle column: BSP throughput is bounded by the slowest worker at every step, while async throughput is bounded only by per-device compute. SSP occupies the gap, delivering near-BSP convergence with a bounded but nonzero staleness budget. The same trade-off becomes clearer when the schedules are placed on a timeline."

---

### `tbl-zero-stages` (table 🟠) — def L916

**Ref sentence (L904):** "ZeRO addresses this redundancy through progressive sharding, as @fig-zero-memory illustrates and @tbl-zero-stages summarizes."

**Missing move:** Interpret. The cite says "summarizes" — a bare pointer. The payoff (L918) walks through each ZeRO stage in detail, which is good, but no sentence extracts the key conclusion *from the table's structure*: that each successive stage adds one more communication primitive (ZeRO-1 adds none, ZeRO-2 adds ReduceScatter, ZeRO-3 adds AllGather) in exchange for a step-change in memory reduction, and ZeRO-3 is the only stage that achieves $N$-linear reduction.

**Where takeaway currently lives:** Caption ("ZeRO-1 is free relative to DDP; ZeRO-2 swaps AllReduce for ReduceScatter; ZeRO-3 (FSDP) adds an AllGather before every layer but unlocks near-linear memory scaling").

**Rule-compliant diff rewrite (add a lead-out sentence after the table at L916/918):**

> Before: "ZeRO-1 shards optimizer states across GPUs."
>
> After: "The table's right column reveals the communication trade the book keeps for ZeRO-3: only the final stage — full parameter sharding — achieves $N$-linear memory reduction, and it does so by adding an AllGather before every layer forward pass. ZeRO-1 is free in communication terms; ZeRO-2 and ZeRO-3 each add a collective, but the memory return grows from $4\times$ to $8\times$ to $N\times$. ZeRO-1 shards optimizer states across GPUs."

---

### `tbl-convergence-comparison` (table 🟠) — def L1425

**Ref sentence (L1417):** "@Tbl-convergence-comparison summarizes the convergence properties of each synchronization model."

**Missing move:** Interpret. The cite is a one-sentence bare pointer. The payoff (L1429) pivots directly to learning rate scaling rules — a new topic — with no conclusion extracted from the table. The table contains convergence rate formulas. The key conclusion (that BSP achieves full $1/N$ variance reduction in the dominant term, SSP adds a staleness penalty term proportional to $s^2$, and ASP loses the $1/N$ factor entirely) is never stated in body prose as a reading of the table.

**Where takeaway currently lives:** Caption and the preceding prose on staleness (L1410-1415), which explains the penalty terms but not as a reading of the table rows.

**Rule-compliant diff rewrite (expand the cite sentence at L1417 and add a payoff):**

> Before: "@Tbl-convergence-comparison summarizes the convergence properties of each synchronization model."
>
> After: "@Tbl-convergence-comparison places the three models side by side to show what staleness costs. The critical row is ASP: the $\mathcal{O}(1/\sqrt{K})$ dominant term contains no $N$ in the denominator, meaning adding workers does not reduce variance — the parallel speedup disappears. BSP retains the full $\mathcal{O}(1/\sqrt{NbK})$ rate; SSP pays an additive staleness penalty of $\mathcal{O}(s^2\eta^2 L_s^2/Nb)$ that grows with the staleness bound $s$ and the learning rate. Choosing $s$ small keeps SSP close to BSP quality while recovering throughput."

---

### `tbl-scaling-bsp-ssp` (table 🟠) — def L1686

**Grade: 🛑 Fails**

**Ref sentence:** The only reference in body prose is the bold heading **"Analysis** (@tbl-scaling-bsp-ssp)" at L1677. This is not a body prose sentence — it is a section-label bold heading. No running text cites the table with a reference and names what it shows.

**Missing move:** Citation (no body prose sentence cites the table). The payoff at L1688 delivers a genuine conclusion ("The 64-GPU BSP configuration achieves X speedup despite only Y sample efficiency because the communication overhead is offset by the massive parallelism"), which would pass as an Interpret move, but it cannot rescue a missing Citation — the float-level contract requires body prose to cite the float before or at introduction.

**Where takeaway currently lives:** Payoff prose (L1688) and the detailed analysis bullets above (L1670-1676).

**Rule-compliant diff rewrite (convert the bold heading label into a body prose sentence that cites the table):**

> Before: "**Analysis** (@tbl-scaling-bsp-ssp)"
>
> After: "@Tbl-scaling-bsp-ssp collects the key numbers from this configuration sweep: iteration count, communication overhead, wall-clock speedup, and sample efficiency for the same job at 1, 8, and 64 GPUs."

This gives the table a formal citation in body prose. The payoff at L1688 already delivers the Interpret move and requires no change.

---

## Dangling Reference Note

`@fig-fleet-stack` at L135 has no matching definition in this file. This is an orphan cross-reference that will fail at render time and should be verified against the fleet chapter or corrected to the appropriate `@sec-` reference.
