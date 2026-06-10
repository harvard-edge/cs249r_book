# Float-explanation worklist — data_selection.qmd (vol1)

## Summary

| type      | floats | ✅ | ⚠️ | 🛑 |
|-----------|--------|----|----|-----|
| figure    | 14     | 14 | 0  | 0   |
| table     | 21     | 21 | 0  | 0   |
| listing   | 1      | 1  | 0  | 0   |
| algorithm | 0      | 0  | 0  | 0   |
| equation  | 2      | 2  | 0  | 0   |
| **total** | **38** | **38** | **0** | **0** |

No under-explained floats found; all references explained in-neighborhood.

## Auditor notes

**Scanner caption false negatives.** The scanner reported "(none found)" for 18 of the 21 tables. All 18 have proper captions written in Quarto's trailing `:` syntax (e.g., `: **Title**: Explanation. {#tbl-foo ...}`), which the scanner does not parse. Every table has a substantive caption.

**Closest call — `fig-distributed-coreset-architecture` (def L3695).** The sole reference (L3669) is a bare label sentence: "**Setup**: @Fig-distributed-coreset-architecture shows the coordinator-worker topology." On its own this would be ⚠️. However, it appears inside a structured example callout where the **Mechanism** list immediately following enumerates every role shown in the figure (coordinator, workers, score flow, broadcast), satisfying the neighborhood check. Caption is also substantive. Verdict: ✅.

**All other floats** have at least one reference site where the surrounding paragraph directly states what the float shows and why it matters, meeting the full neighborhood test.
