# Float-explanation worklist — nn_computation.qmd (vol1)

## Summary

| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 20 | 20 | 0 | 0 |
| table | 15 | 15 | 0 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 2 | 2 | 0 | 0 |
| equation | 27 | 27 | 0 | 0 |
| **total** | **64** | **64** | **0** | **0** |

No under-explained floats found; all references explained in-neighborhood.

## Notes on judgment calls

Several floats warranted extra scrutiny before being cleared:

- **`tbl-nn-computation-mnist-params` / `tbl-nn-computation-mnist-activations` / `tbl-nn-computation-mnist-memory-budget`**: The scanner reported "(none found)" captions because these tables use the Quarto foot-of-table caption syntax (`: **Caption**: ...`). All three have substantive captions, and each ref sentence plus the surrounding step-by-step walkthrough fully explains purpose and content.

- **`tbl-nn-computation-napkin-math-checks`**: The post-table paragraph (L3627) pivots to parameter distribution and does not explain the table. The explanation is entirely in the ref sentence (L3607), the worked example (L3609-3611), and the table's own caption — all within the callout block. Cleared on neighborhood rule.

- **`fig-double-descent`**: The payoff paragraph is deferred 100+ lines after the float (L828), separated by the float definition itself. The ref sentence (L725) names and distinguishes all three regimes, and the payoff (L828) delivers the full significance of the double-descent shape. Cleared.

- **`fig-usps-inference-pipeline`**: The ref sentence (L5127) tells the reader what to trace and why, but the payoff paragraph (L5454) is far downstream (the figure is followed by a long case-study section). The explanation accumulates across the surrounding case-study prose, which explicitly discusses the pipeline stages. Cleared on full-neighborhood read.
