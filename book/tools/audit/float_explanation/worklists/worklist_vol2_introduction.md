# Float-explanation worklist — introduction.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 14 | 11 | 3 | 0 |
| table | 4 | 4 | 0 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 5 | 4 | 1 | 0 |
| **total** | **23** | **19** | **4** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

---

### ⚠️ `eq-energy-scale-invariant` — def L1562  (Thin)
- **Caption:** (none)
- **Ref(s):** L1560 `@Eq-energy-scale-invariant`: "While the fleet law governs time, the **Energy-Scale Invariant** governs the sustainability and economic viability of the fleet. At scale, every training step is a thermodynamic event. @Eq-energy-scale-invariant defines the **fleet energy productivity** ($\rho_{\text{energy}}$), measured in FLOP/J, as the ratio of useful work to total energy consumed:"
- **Context checked:** ref ✓ (defines metric) · prev ¶ ✓ (names the concept) · next ¶ = equation body · caption ✗ (absent) · payoff ¶ L1566 ✗ ("The energy-side metric is only half of the scaling diagnosis" — pivots immediately without explaining what a good or bad $\rho_{\text{energy}}$ value implies or what engineering actions it motivates)
- **Why thin:** The equation is introduced and the numerator/denominator are clear, but the prose never tells the reader what threshold matters, what moves $\rho_{\text{energy}}$ up or down in practice, or why this particular decomposition of energy costs (compute + cooling + network) is the diagnostic insight. The payoff sentence pivots to the time-side metric before landing the energy-side implication.
- **Suggested rewrite (flag-only — payoff sentence, L1566):**
  ```diff
  - The energy-side metric is only half of the scaling diagnosis. A production team also needs a time-side efficiency scalar that says whether extra devices are shortening the training step or merely increasing coordination overhead.
  + Fleet energy productivity $\rho_{\text{energy}}$ falls when cooling and network energy grow faster than useful arithmetic output, which happens precisely when the fleet is oversized or poorly overlapped. A production team therefore needs both metrics: $\rho_{\text{energy}}$ reveals whether the thermodynamic cost is justified, and the time-side scalar that follows reveals whether adding more devices shortens the training step or merely shifts the same overhead into coordination.
  ```

---

### ⚠️ `fig-loss-vs-n-d` — def L1101  (Thin)
- **Caption:** **Loss vs. Dataset Size Across Model Scales**: Test loss curves showing how models of different sizes (393K to 708M parameters) benefit from increased training data. Larger models achieve lower loss but all curves exhibit diminishing returns at high token counts.
- **Ref(s):** L1081 `@Fig-loss-vs-n-d`: "@Fig-loss-vs-n-d shows *how* early-stopped test loss varies predictably with both dataset size and model size, confirming that learning curves across configurations align through appropriate parameterization."
- **Context checked:** ref ✗ (announces the figure, restates the caption, adds no implication) · prev ¶ ✗ (inside a callout `::: {}`, L1079) · next ¶ ✗ (L1083 opens a new subsection on resource-constrained regimes, no look-back) · caption ✓ (names the behavior) · payoff ¶ L1172 ✗ (distant; does not reference this figure; discusses the lifecycle distinction instead)
- **Why thin:** "Confirms that learning curves align through appropriate parameterization" is a methodological restatement, not a systems implication. The reader learns nothing about what this specific shape — multiple curves all plateauing — should change about how they allocate training resources. The figure's key takeaway (all model sizes plateau, larger models plateau at lower loss, so capacity does not rescue data starvation) is never stated in prose.
- **Suggested rewrite (flag-only — ref sentence, L1081):**
  ```diff
  - These predictions find strong empirical support across multiple model configurations. @Fig-loss-vs-n-d shows *how* early-stopped test loss\index{Scaling Laws!loss curves} varies predictably with both dataset size and model size, confirming that learning curves across configurations align through appropriate parameterization.
  + These predictions find strong empirical support across multiple model configurations. @Fig-loss-vs-n-d shows that all model sizes plateau as token counts grow, and that larger models plateau at lower loss rather than avoiding the plateau entirely — meaning capacity does not rescue data starvation, only lowers the floor that data volume determines.
  ```

---

### ⚠️ `fig-fleet-stack` — def L1704  (Thin)
- **Caption:** **The Fleet Stack**: The organizing framework for this book. We build from the Infrastructure Layer (compute, network, data) up through the Distribution Layer (parallelism, communication, fault tolerance) and Serving Layer (inference, performance, edge, operations) to the Governance Layer (security, robustness, sustainability, responsible engineering). Engineering decisions at the bottom constrain possibilities at the top.
- **Ref(s):** L1702 `@Fig-fleet-stack`: "@Fig-fleet-stack organizes the complexity of this book into **The Fleet Stack**, a four-layer framework where engineering decisions at the bottom constrain possibilities at the top."
- **Context checked:** ref ✗ (restates the caption's last clause verbatim; adds nothing) · prev ¶ ✓ (L1700 establishes why the stack is the spine and how the other lenses relate to it) · next ¶ ✗ (L1738: "This layered progression structures the textbook's four parts" — names the four parts but does not explain why bottom-to-top constraint flow matters) · caption ✓ (lists layers) · payoff ¶ L1738 ✗ (structural description only, no implication)
- **Why thin:** The prose around the figure names the four layers but never explains what "engineering decisions at the bottom constrain possibilities at the top" means concretely — for example, that a choice of network fabric (Infrastructure) bounds which parallelism strategies (Distribution) are feasible, which in turn bounds achievable inference latency (Serving), which in turn constrains governance SLAs (Governance). The reader is told the stack exists but not why it is the right mental model for reading the book.
- **Suggested rewrite (flag-only — replace L1738 payoff):**
  ```diff
  - This layered progression structures the textbook's four parts: the physical substrate, the logic of distribution, deployment at scale, and the responsible fleet. The detailed chapter map appears in @sec-vol2-introduction-structure; here the stack establishes the dependency order.
  + This layered progression names a dependency, not a reading order: a network-fabric choice at the Infrastructure layer bounds which parallelism strategies the Distribution layer can sustain; those parallelism constraints bound achievable serving latency at the Serving layer; and serving latency bounds which governance SLAs the Governance layer can promise. The detailed chapter map appears in @sec-vol2-introduction-structure; here the stack establishes why each layer's treatment must precede the one above it.
  ```

---

### ⚠️ `fig-vol2-ai-triad` — def L1744  (Thin)
- **Caption:** **The AI Triad at Scale**: The three interdependent components of every ML system. At production scale, each component's requirements intensify: data pipelines must handle petabytes with consistent quality; algorithms can demand $10^{25}$ FLOPs for large-model training; and infrastructure must coordinate thousands of accelerators while maintaining fault tolerance. Changes to any vertex cascade through the others, creating the multi-dimensional optimization challenge that defines ML systems engineering.
- **Ref(s):** L1742 `@Fig-vol2-ai-triad`: "@Fig-vol2-ai-triad visualizes these dependencies between data, algorithms, and infrastructure, revealing the optimization landscape that ML systems engineers must address."
- **Context checked:** ref ✗ (float-announcer sentence; "revealing the optimization landscape" is vague and adds no substance beyond the caption) · prev ¶ ✓ (L1740 explains the three components and gives a GPT-4-class compute scale example) · next ¶ ✗ (L1860 — distant payoff; discusses the Five-Pillar Framework, does not reference the triad figure) · caption ✓ (names the cascade behavior) · payoff ✗ (no payoff paragraph follows the figure closely; text moves directly to the Five-Pillar discussion)
- **Why thin:** The prev paragraph establishes the three vertices but only illustrates one direction of the triangle (algorithm demand → infrastructure requirement). The ref sentence does not state the figure's diagnostic implication: that changes in any vertex propagate as a constraint wave through the other two, meaning no single-axis optimization (scale data alone, scale model alone, or scale compute alone) is stable without rebalancing the triangle. That cascade consequence is in the caption but nowhere in the prose.
- **Suggested rewrite (flag-only — ref sentence, L1742):**
  ```diff
  - @Fig-vol2-ai-triad visualizes these dependencies between data, algorithms, and infrastructure, revealing the optimization landscape that ML systems engineers must address.
  + @Fig-vol2-ai-triad shows that the three vertices form a closed constraint loop: scaling the algorithm vertex (larger model) immediately stresses both the data vertex (more tokens required for compute-optimality) and the infrastructure vertex (more memory and bandwidth for distribution), so single-axis scaling always propagates imbalance around the triangle.
  ```
