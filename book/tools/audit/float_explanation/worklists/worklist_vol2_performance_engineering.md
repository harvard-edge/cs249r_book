# Float-explanation worklist — performance_engineering.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 8 | 8 | 0 | 0 |
| table | 4 | 4 | 0 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 1 | 1 | 0 | 0 |
| equation | 3 | 3 | 0 | 0 |
| **total** | **16** | **16** | **0** | **0** |

No under-explained floats found; all references explained in-neighborhood.

## Notes

All 16 floats carry their explanation in the local neighborhood (setup paragraph, reference sentence, payoff paragraph, or caption). Key highlights:

- `eq-iron-law-perf` — defined at L88 with a full explanatory paragraph at L96 naming all three terms and their optimization levers; the only `@eq-iron-law-perf` ref appears in a comprehension-check callout (L119), which is appropriate.
- `fig-roofline-model` — reference sentence (L340) is functional ("illustrates this relationship graphically") but the surrounding equations and rich caption carry the load; this meets the neighborhood standard.
- `tbl-performance-engineering-bottleneck-patterns` — reference sentence (L2109) names what the table does and how to use it; the caption adds the scoping note about when the table is most useful.
- `fig-overlap-budget` — reference sentence (L1808) quantifies the degradation curve (90% to 18%) before the figure appears, making the figure a confirmation rather than a bare pointer.

One dangling cross-ref (`@fig-fleet-stack` at L72) points to a figure defined in another chapter — this is a cross-chapter reference, not an in-chapter orphan, and is outside the scope of this audit.
