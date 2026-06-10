# Float-explanation worklist — data_storage.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 6 | 6 | 0 | 0 |
| table | 5 | 5 | 0 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 3 | 3 | 0 | 0 |
| **total** | **14** | **14** | **0** | **0** |

No under-explained floats found; all references explained in-neighborhood.

## Notes

- `tbl-storage-hierarchy-merged` (def L475): the scanner reported "caption: none found" because this table uses Quarto's pipe-table caption syntax (`: **Title**: ...  {#id}` after the closing row), which the scanner may not parse. The caption is present at L475: "**Extended Memory Hierarchy for ML Systems**: The roughly 30× aggregate-to-aggregate bandwidth gap between HBM and object storage...". All refs (L445, L477, L479, L509) carry substantive explanation.

- `fig-data-stall-frontier` (def L1410): the scanner assigned payoff ¶ at L1556 (about the pipeline equation, not the figure). The actual pre-float paragraph L1408 carries the full explanation of the S-curve and its architectural implication. The neighborhood check confirms ✅.

- `tbl-data-formats` (def L1302): the post-table payoff sentence ("The comparison in @tbl-data-formats sets the data-volume term in the pipeline equation, turning format selection into a bandwidth-sizing problem") is lean in isolation, but five preceding paragraphs (L1280–L1292) explain every row in the table before it appears. Verdict ✅ stands on the neighborhood rule.

- Dangling ref `@fig-fleet-stack` at L132: this figure is defined in another chapter (not in data_storage.qmd). Out of scope for this audit.
