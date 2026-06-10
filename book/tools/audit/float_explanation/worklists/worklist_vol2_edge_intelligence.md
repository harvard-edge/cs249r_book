# Float-explanation worklist — edge_intelligence.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 17 | 16 | 1 | 0 |
| table | 7 | 7 | 0 | 0 |
| listing | 4 | 4 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 0 | 0 | 0 | 0 |
| **total** | **28** | **27** | **1** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `fig-fl-communication-computation` — def L2942  (Thin)

- **Caption:** "The Communication-Computation Trade-Off in Federated Learning: As network bandwidth decreases (Fast to Slow), the optimal number of local epochs shifts rightward to amortize the high cost of communication over more computation. However, excessive local computation eventually increases total time due to model drift (requiring more global rounds to converge)."
- **Ref(s):** L2940 `@fig-fl-communication-computation`: "While @fig-fl-communication-computation illustrates the fundamental tradeoff between local computation and network bandwidth, other communication-efficient updates introduce their own tradeoffs."
- **Context checked:** ref ✗ (contrast pivot, not a delivery) · prev ¶ ✗ (transition between topics) · next ¶ ✗ (moves on to gradient compression) · caption ✓ (insight is there) · payoff ✗ (L3021 is a pivot sentence to next section, not an explanation)
- **What is missing:** The figure's core insight is that the optimal number of local epochs is bandwidth-dependent: slow networks should run more local epochs to amortize communication cost, but too many epochs triggers model drift and raises total time. This U-curve property and its design implication (engineers must tune E differently per network tier) are never stated in prose. The ref sentence uses the figure as a foil to introduce other communication techniques rather than extracting what this particular figure shows.
- **Suggested rewrite (flag-only):**
  ```diff
  - While @fig-fl-communication-computation illustrates the fundamental tradeoff between local computation and network bandwidth, other communication-efficient updates introduce their own tradeoffs.
  + @Fig-fl-communication-computation makes the bandwidth dependence concrete: on fast networks, a small number of local epochs minimizes total training time because communication is cheap; on slow networks, the optimal point shifts to more local epochs to amortize each round's cost. Beyond that optimum, however, additional local computation increases total time because model drift forces more global rounds to converge. The implication for deployment is that the right value of $E$ is not a model hyperparameter but a network-tier parameter, and systems that fix it across all clients leave efficiency on the table. Gradient compression and selective update sharing address the bandwidth side of the same constraint.
  ```
