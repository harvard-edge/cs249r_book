# Float-explanation worklist — collective_communication.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 9 (+1 dangling) | 9 | 0 | 1 (dangling cross-chapter ref) |
| table | 15 | 15 | 0 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 1 | 1 | 0 | 0 |
| equation | 0 | 0 | 0 | 0 |
| **total** | **25 (+1 dangling)** | **25** | **0** | **1** |

> The chapter's float coverage is exceptionally strong overall. Every defined float has rich
> neighborhood explanation — often multi-paragraph, with quantitative payoff. The single
> finding is a dangling cross-chapter reference that points at a figure defined elsewhere
> in the book.

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### 🛑 `fig-fleet-stack` — def NONE in this chapter  (Dead-end cross-chapter ref)

- **Caption:** N/A — figure is not defined in this chapter; defined elsewhere in the book.
- **Ref(s):** L50 `@fig-fleet-stack`: "In the fleet stack shown in @fig-fleet-stack, communication algorithms sit squarely in the Distribution Layer."
- **Context checked:** ref ✗ (bare assertion) · prev ¶ — none (sentence is the paragraph opener) · next ¶ introduces parallelism strategies via `@sec-distributed-training-systems` · caption N/A · payoff ✗ — no prose in this chapter defines what the fleet stack is or explains what "Distribution Layer" means to a reader who does not have the cross-chapter figure rendered in front of them.
- **Issue:** The sentence drops an architectural claim ("communication algorithms sit squarely in the Distribution Layer") that relies entirely on the cross-chapter figure for its grounding. A reader encountering this chapter standalone, or in PDF pagination where the figure does not appear, receives a bare assertion with no local anchor. The figure's three-layer model (Infrastructure / Distribution / Serving) is never explained in the surrounding prose.
- **Suggested rewrite (flag-only):**
  ```diff
  - In the fleet stack shown in @fig-fleet-stack, communication algorithms sit squarely in the Distribution Layer.
  + Communication algorithms occupy the Distribution Layer of the fleet stack (@fig-fleet-stack): the Infrastructure Layer below provides raw bandwidth through hardware, while the Serving Layer above depends on efficient gradient synchronization to deliver trained models.
  ```
  *(The rewrite names all three layers so the positional claim is self-contained even if the figure is not visible. Adjust layer names to match whatever the source figure actually labels them.)*
