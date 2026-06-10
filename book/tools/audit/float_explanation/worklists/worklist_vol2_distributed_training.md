# Float-explanation worklist — distributed_training.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 18 | 16 | 2 | 0 |
| table | 9 | 7 | 2 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 1 | 1 | 0 | 0 |
| equation | 1 | 1 | 0 | 0 |
| **total** | **29** | **25** | **4** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

---

### ⚠️ `fig-comm-convergence-tradeoff` — def L1704  (Thin)
- **Caption:** "Communication-Convergence Trade-Off Space: Each point represents a different distributed training configuration. The Pareto frontier (dashed line) shows optimal configurations where improving one metric requires sacrificing the other. BSP sits at high convergence quality but lower throughput; ASP provides maximum throughput at convergence cost. Gradient compression and SSP occupy intermediate positions."
- **Ref(s):** L1702 `@Fig-comm-convergence-tradeoff`: "The fundamental trade-off in distributed training is between communication efficiency and convergence quality. @Fig-comm-convergence-tradeoff visualizes this trade-off space."
- **Context checked:** ref ✗ (only "visualizes this trade-off space") · prev ¶ = section heading only · next ¶ = float · payoff ¶ ✗ ("Several techniques occupy different positions on this trade-off curve" — names the curve but draws no conclusion from where the points actually land) · caption ✓ (BSP/ASP positions described)
- **What's missing:** The reader is told techniques occupy different positions, but the figure is never used to make the key design judgment: when does the Pareto frontier gap between BSP and gradient compression justify the implementation cost? The payoff paragraph jumps straight to describing individual techniques without using the figure's geometry to motivate the choice.
- **Suggested rewrite (flag-only):**
  ```diff
  - The fundamental trade-off in distributed training is between communication efficiency and
  - convergence quality. @Fig-comm-convergence-tradeoff visualizes this trade-off space.
  + The fundamental trade-off in distributed training is between communication efficiency and
  + convergence quality. @Fig-comm-convergence-tradeoff plots each strategy on that frontier:
  + BSP occupies the high-convergence, low-throughput corner; ASP occupies the opposite corner;
  + gradient compression and SSP fill the middle, where most production systems operate when
  + bandwidth cost exceeds the convergence penalty of moderate staleness.
  ```

---

### ⚠️ `fig-sync-model-timeline` — def L548  (Thin)
- **Caption:** "Distributed Synchronization Models: Timeline comparison of three synchronization strategies. (A) BSP forces all workers to wait at a global barrier every step. (B) SSP allows workers to proceed up to $s$ steps ahead of the slowest worker. (C) Asynchronous SGD eliminates barriers entirely, allowing maximum throughput but introducing gradient staleness."
- **Ref(s):** L536 `@fig-sync-model-timeline`: "The key trade-offs across synchronization models are summarized in @tbl-sync-models, and @fig-sync-model-timeline illustrates how each strategy schedules work across workers over time."
- **Context checked:** ref ✗ (only "illustrates how each strategy schedules work across workers over time" — generic restatement of the caption) · prev ¶ ✓ (BSP/SSP/Async described in prose) · payoff ¶ ✗ ("The choice of synchronization model directly affects both system throughput and model convergence" — makes no contact with the timeline geometry) · caption ✓ (panels described)
- **What's missing:** The timeline's specific pedagogical value is showing the straggler idle time in BSP (all workers wait for the slowest), the bounded-slack window in SSP, and the fully overlapping arrows in Async — the visual that makes the throughput difference legible. Neither the ref sentence nor the payoff paragraph tells the reader what to read from the diagram.
- **Suggested rewrite (flag-only):**
  ```diff
  - The key trade-offs across synchronization models are summarized in @tbl-sync-models, and
  - @fig-sync-model-timeline illustrates how each strategy schedules work across workers over time.
  + The key trade-offs across synchronization models are summarized in @tbl-sync-models.
  + @fig-sync-model-timeline makes the throughput cost of BSP visible: in the BSP panel, every
  + worker idles at the barrier until the slowest arrives, leaving blank compute lanes proportional
  + to the straggler gap; the SSP panel shows that gap closing as the slack window absorbs small
  + delays; the Async panel shows fully overlapping work with no idle lanes and no synchronization
  + point.
  ```

---

### ⚠️ `tbl-convergence-comparison` — def L1425  (Thin)
- **Caption:** "Convergence Properties by Synchronization Model: BSP provides optimal convergence guarantees at the cost of synchronization overhead. SSP offers a tunable trade-off between throughput and convergence. ASP maximizes throughput but loses the variance reduction benefit of parallelism."
- **Ref(s):** L1417 `@Tbl-convergence-comparison`: "@Tbl-convergence-comparison summarizes the convergence properties of each synchronization model."
- **Context checked:** ref ✗ (bare "summarizes" pointer) · prev ¶ ✓ (detailed staleness-penalty math for BSP/SSP/ASP) · payoff ¶ ✗ ("When increasing the effective batch size through data parallelism, the learning rate must be adjusted..." — pivots to a new topic without reading any conclusion from the table) · caption ✓ (BSP/SSP/ASP take-aways stated)
- **What's missing:** The table contains three convergence-rate formulas side by side, and the key design insight is that ASP's dominant term scales with $\bar{\tau}_{\text{stale}}^2$ while BSP achieves $\mathcal{O}(1/\sqrt{NbK})$. Neither the ref sentence nor the payoff paragraph surfaces that comparison — the reader who skips the formulas learns nothing from the ref.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Tbl-convergence-comparison summarizes the convergence properties of each synchronization model.
  + @Tbl-convergence-comparison places the three convergence rates side by side: BSP achieves
  + $\mathcal{O}(1/\sqrt{NbK})$ with full $1/N$ variance reduction; SSP adds a staleness-penalty
  + term that grows as $s^2$; ASP loses variance reduction entirely, so its dominant term scales
  + with average staleness squared rather than worker count. The practical consequence is that
  + increasing $N$ only improves ASP convergence if staleness is actively controlled.
  ```

---

### ⚠️ `tbl-scaling-bsp-ssp` — def L1686  (Thin)
- **Caption:** (none — caption text is the `**Analysis**` label in the preceding paragraph at L1677)
- **Ref(s):** L1677 `@tbl-scaling-bsp-ssp`: "**Analysis** (@tbl-scaling-bsp-ssp)"
- **Context checked:** ref ✗ (label-plus-parenthetical only — no sentence) · prev ¶ ✓ (bullet list with concrete numbers for 1-GPU baseline, 8-GPU BSP, 64-GPU BSP, 64-GPU SSP) · payoff ¶ ✓ ("The 64-GPU BSP configuration achieves ... speedup despite only ... sample efficiency because ...") · no caption
- **What's missing:** The ref is not a sentence — it is a bold heading with the ref in parentheses. This is weak pointer form even though the payoff is solid. The table also has no caption, which is a book-style violation separate from explanation quality. The explanation lands in the payoff, so this is thin rather than dead-end, but the entry form should be a sentence.
- **Suggested rewrite (flag-only):**
  ```diff
  - **Analysis** (@tbl-scaling-bsp-ssp)
  + @Tbl-scaling-bsp-ssp compares wall-clock speedup, communication overhead, and sample
  + efficiency across the four configurations, showing that 64-GPU BSP achieves the highest
  + speedup but at the cost of the lowest sample efficiency.
  ```
  *(The table also needs a proper `:{#tbl-scaling-bsp-ssp}` caption line.)*
