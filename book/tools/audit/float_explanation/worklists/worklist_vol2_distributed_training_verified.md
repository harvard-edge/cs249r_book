# Verified findings — distributed_training.qmd (vol2)
Prior findings: 4 | Survived: 2 | Refuted: 2

---

## SURVIVING findings

### ⚠️ `fig-comm-convergence-tradeoff` — def L1704
- Ref: "The fundamental trade-off in distributed training is between communication efficiency and convergence quality. @Fig-comm-convergence-tradeoff visualizes this trade-off space."
- Why it survives: The ref sentence is a pure announcer with no geometric reading of the figure. The prev context is only a section heading (L1700). The payoff paragraph (L1708) reads "Several techniques occupy different positions on this trade-off curve" and then pivots directly into describing individual techniques (gradient compression, Local SGD, decentralized SGD) without ever reading a conclusion from the Pareto geometry. The caption names the positions of BSP/ASP/SSP/gradient-compression on the frontier but does not state the practical design consequence of where the gaps between them fall. No neighborhood element tells the reader what the frontier's shape means for actual system choices.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - The fundamental trade-off in distributed training is between communication efficiency and
  - convergence quality. @Fig-comm-convergence-tradeoff visualizes this trade-off space.
  + The fundamental trade-off in distributed training is between communication efficiency and
  + convergence quality. @Fig-comm-convergence-tradeoff plots each strategy on that frontier:
  + BSP occupies the high-convergence corner at the cost of throughput; ASP maximizes
  + throughput at the cost of convergence; gradient compression and SSP sit in the middle,
  + where most production systems operate when bandwidth cost outweighs the convergence
  + penalty of moderate staleness or lossy updates.
  ```

---

### ⚠️ `tbl-convergence-comparison` — def L1425
- Ref: "@Tbl-convergence-comparison summarizes the convergence properties of each synchronization model."
- Why it survives: The preceding paragraphs (L1392–L1415) develop each convergence bound individually, but no sentence synthesizes what the side-by-side comparison reveals. The ref sentence is a bare pointer. The payoff paragraph (L1427) opens a new section on learning-rate scaling rules and makes no contact with the table's content. The caption correctly states per-model take-aways but does not surface the cross-model comparison that makes the table worth a read: BSP achieves $\mathcal{O}(1/\sqrt{NbK})$ with full $1/N$ variance reduction; SSP adds an $s^2$ staleness-penalty term; ASP loses variance reduction entirely so its dominant term scales with average staleness squared rather than worker count. No neighborhood element draws that comparison.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - @Tbl-convergence-comparison summarizes the convergence properties of each synchronization model.
  + @Tbl-convergence-comparison places the three convergence rates side by side. BSP achieves
  + $\mathcal{O}(1/\sqrt{NbK})$ with full $1/N$ variance reduction; SSP adds a staleness-penalty
  + term that grows as $s^2$; ASP loses variance reduction entirely, so its dominant term scales
  + with average staleness squared rather than worker count. The practical consequence is that
  + increasing $N$ only improves ASP convergence if staleness is actively controlled.
  ```

---

## REFUTED findings

- `fig-sync-model-timeline` — REFUTED: explanation is in the bridge sentence at L546 and the payoff at L552. The scanner's prior-pass context stopped at L536 (the ref sentence) and L534 (the prev paragraph) but missed L546, which reads: "The same trade-off becomes clearer when the schedules are placed on a timeline." That sentence explicitly states what the figure adds over the preceding table (visual clarity of the scheduling geometry) and follows a rich prev-paragraph that characterizes all three models operationally. The payoff at L552 then draws the design-use conclusion (BSP for final runs, SSP/async for hyperparameter search). Combined with the detailed caption describing panels A/B/C, the neighborhood fully tells the reader what the figure shows and why it matters. The first-pass flag was generated before the scanner captured L546.

- `tbl-scaling-bsp-ssp` — REFUTED: explanation is in the payoff paragraph at L1688. The prior-pass concern was that the ref is a bold heading with a parenthetical rather than a sentence, and that there is no formal caption. However, the refutation bar is "ANY neighborhood element tells the reader what the float shows and why it matters." L1688 reads: "The 64-GPU BSP configuration achieves [speedup] despite only [sample efficiency] because the communication overhead ([pct]) is offset by the massive parallelism. SSP provides comparable wall-clock time with lower communication overhead but requires more total samples." This directly reads the key comparison from the table. The table also has a proper caption at L1686 ("Scaling SGD across cluster sizes: Iteration count, communication overhead, wall-clock speedup, and sample efficiency..."). The ref form is weak (bold heading plus parenthetical), but weak pointer form with a solid payoff meets the refutation bar. Flagging as a surviving finding on form alone, when explanation is present, would be false positive inflation.
