# Float-explanation worklist — frameworks.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 14 | 14 | 0 | 0 |
| table | 17 | 17 | 0 | 0 |
| listing | 32 | 31 | 1 | 0 |
| algorithm | 1 | 1 | 0 | 0 |
| equation | 4 | 3 | 1 | 0 |
| **total** | **68** | **66** | **2** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

---

### ⚠️ `lst-torchscript-ir` — def L1365  (Thin)
- **Caption:** **TorchScript IR Inspection**: The generated intermediate representation shows primitive operations and constants, useful for debugging and understanding compilation results.
- **Ref(s):** L1363 `@lst-torchscript-ir`: "To understand what the compiler produces, @lst-torchscript-ir inspects the generated intermediate representation directly."
- **Context checked:** ref ✗ (bare pointer) · prev ¶ ✓ (scripting preserves branches) · next ¶ ✗ (pivots to scripting constraints, never describes the IR) · caption ⚠️ (generic: "useful for debugging") · payoff ✗ (discusses Python-subset constraints, not what the IR shows)
- **Gap:** The listing displays a concrete IR graph — `prim::Constant`, `aten::mul`, `aten::add` nodes in SSA form — but no prose explains what the reader should observe in it, why the `aten` and `prim` namespaces are significant, or what the SSA structure enables. The payoff paragraph at L1385 moves directly to scripting constraints without connecting back to the IR. A reader who has never seen TorchScript IR leaves without knowing what they just looked at.
- **Suggested rewrite (flag-only):**
  ```diff
  - To understand what the compiler produces, @lst-torchscript-ir inspects the generated intermediate representation directly.
  + @lst-torchscript-ir shows what the compiler actually produces. The IR uses two namespaces: `aten` for core tensor operations (the same operations the eager runtime dispatches) and `prim` for primitives such as constants and control-flow nodes. Constants are extracted and hoisted — the `2` and `1` in `x * 2 + 1` become `prim::Constant` nodes computed once — and every value is assigned exactly once (static single-assignment form), which is what lets the compiler safely reorder and fuse operations without aliasing ambiguity.
  ```

---

### ⚠️ `eq-execution-continuum` — def L1768  (Thin — float-announcer colon)
- **Caption:** (none — inline equation)
- **Ref(s):** L1764 `@eq-execution-continuum`: "The execution models form a continuum from maximum flexibility to maximum optimization, visualized in @eq-execution-continuum:"
- **Context checked:** ref ⚠️ (colon announcer violates book prose rule) · prev ¶ ✓ (execution problem framing) · next ¶ ✓ (payoff at L1770 explains "each step rightward sacrifices flexibility for performance") · caption n/a · payoff ✓
- **Gap:** The explanation in the neighborhood is solid — the equation is well set up and the payoff is immediate. The only issue is the colon at the end of the reference sentence, which announces the float rather than integrating the ref into the prose. Per book prose rules, a float reference must use a period, not a colon, as the terminal punctuation before the float.
- **Suggested rewrite (flag-only):**
  ```diff
  - The execution models form a continuum from maximum flexibility to maximum optimization, visualized in @eq-execution-continuum:
  + The execution models form a continuum from maximum flexibility to maximum optimization. @Eq-execution-continuum maps the four positions on that axis, with each arrow labeling the mechanism that moves a project one step rightward.
  ```
