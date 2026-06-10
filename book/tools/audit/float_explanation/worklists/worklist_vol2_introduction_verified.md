# Verified findings — introduction.qmd (vol2)
Prior findings: 4 | Survived: 1 | Refuted: 3

---

## SURVIVING findings

### ⚠️ `fig-loss-vs-n-d` — def L1101
- **Ref:** "These predictions find strong empirical support across multiple model configurations. @Fig-loss-vs-n-d shows *how* early-stopped test loss varies predictably with both dataset size and model size, confirming that learning curves across configurations align through appropriate parameterization."
- **Why it survives:** Every neighborhood element was checked. The ref sentence (L1081) is a pure float-announcer that restates the caption in methodological terms ("learning curves align through appropriate parameterization") without stating any systems implication. The prev paragraph is a callout close marker (:::). The next paragraph (L1083) opens a new subsection ("Resource-constrained scaling regimes") with no look-back at the figure. The caption names the behavior ("all curves exhibit diminishing returns at high token counts") but does not land the engineering implication. The payoff paragraph (L1172) is distant and does not reference this figure. No neighborhood element states the key takeaway: that all model sizes plateau as token volume grows, and that larger models reach a lower plateau rather than avoiding it, meaning model capacity does not rescue data starvation but only lowers the floor that data volume determines.
- **Suggested rewrite (ref sentence, L1081):**
  ```diff
  - These predictions find strong empirical support across multiple model configurations. @Fig-loss-vs-n-d shows *how* early-stopped test loss\index{Scaling Laws!loss curves} varies predictably with both dataset size and model size, confirming that learning curves across configurations align through appropriate parameterization.
  + These predictions find strong empirical support across multiple model configurations. @Fig-loss-vs-n-d shows that every model size plateaus as token volume grows, and that larger models reach a lower plateau rather than avoiding it: capacity lowers the floor that data volume determines, but does not escape the plateau entirely.
  ```

---

## REFUTED findings

- `eq-energy-scale-invariant` — REFUTED: explanation is in the where-clause paragraph immediately after the equation (L1564): "where $O_{\text{useful}}$ is useful work in FLOPs and $E_{\text{network}}$ often becomes a nonnegligible fraction of the total budget as we move terabytes across optical fabrics. Mastery of scale requires optimizing for the Pareto frontier of both laws: minimizing $T_{\text{step}}$ while maximizing $\rho_{\text{energy}}$." The first pass scored L1566 as the payoff and found it thin, but overlooked L1564 — the where-clause paragraph is the actual payoff, and it explicitly states what drives the metric (network energy growing at scale) and why the decomposition matters (it defines one axis of the Pareto frontier that fleet operators must optimize).

- `fig-fleet-stack` — REFUTED: explanation is in prev ¶ (L1700): "The fleet stack is the organizing spine for this book. The C$^3$ taxonomy... gives the local diagnostic lens... Scaling laws predict how much computation a target capability level demands... Those lenses support the stack rather than competing with it... The dependency order comes from the fleet stack." This paragraph tells the reader why the stack is the correct organizing framework (it supplies the dependency order that the other lenses do not), satisfying the refutation bar. The payoff paragraph (L1738) reinforces with "here the stack establishes the dependency order." The prior pass correctly noted the ref sentence (L1702) restates the caption, but the prev ¶ carries the full rationale.

- `fig-vol2-ai-triad` — REFUTED: explanation is in the caption: "Changes to any vertex cascade through the others, creating the multi-dimensional optimization challenge that defines ML systems engineering." The caption explicitly states the cascade behavior and its engineering consequence. The refutation bar accepts caption as a valid carrier of takeaway. The prior pass required the cascade to appear in prose rather than caption, which is a stricter standard than the refutation bar permits.
