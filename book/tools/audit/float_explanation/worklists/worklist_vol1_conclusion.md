# Float-explanation worklist — conclusion.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 1 | 1 | 0 | 0 |
| table | 2 | 2 | 0 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 0 | 0 | 0 | 0 |
| **total** | **3** | **3** | **0** | **0** |

No under-explained floats found; all references explained in-neighborhood.

### Audit notes

- `tbl-lighthouse-journey-mobilenet` (def L165): Reference at L153 explains the 7-phase structure and the constraint-propagation purpose. Payoff paragraph at L167 draws out the "every row constrains the next row" pattern and bridges to the thirteen invariants. Caption restates the cross-domain propagation thesis. Fully explained in neighborhood.

- `tbl-thirteen-principles` (def L247): First reference at L171 describes the table's organization (four Parts, five columns: name, part, equation, predictive power). Second reference at L249 ties the table to the conservation-of-complexity meta-principle. Caption at L247 states the unifying thesis. Payoff paragraph at L249 explains what the table reveals as a whole. Fully explained in neighborhood.

- `fig-invariants-cycle` (def L285): Reference at L283 walks through every structural element of the figure (four phases, central hub, constraint-flow arrows, specific Build-to-Optimize and Deploy-to-Foundations paths). Payoff paragraph at L590 names the critical insight (the Deploy-to-Foundations feedback arrow) and explains what it means for system evolution. Caption states the conservation-of-complexity framing. Fully explained in neighborhood.
