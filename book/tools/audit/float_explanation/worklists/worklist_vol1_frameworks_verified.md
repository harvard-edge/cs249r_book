# Verified findings — frameworks.qmd (vol1)
Prior findings: 2 | Survived: 1 | Refuted: 1

## SURVIVING findings

### ⚠️ `eq-execution-continuum` — def L1768  (punctuation-only)
- Ref: "The execution models form a continuum from maximum flexibility to maximum optimization, visualized in @eq-execution-continuum:"
- Why it survives: The explanation is fully present in the neighborhood. The preceding paragraph (L1762) frames the quantitative problem, and the payoff at L1770 states "Each step rightward sacrifices flexibility for performance" and immediately introduces @eq-compilation-benefit to operationalize the principle. The float's content and significance are clear. The surviving issue is punctuation only: the ref sentence terminates with a colon before the float block, violating the book's prose rule that float references use a period rather than a colon as terminal punctuation. No substantive rewrite is needed.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - The execution models form a continuum from maximum flexibility to maximum optimization, visualized in @eq-execution-continuum:
  + The execution models form a continuum from maximum flexibility to maximum optimization. @Eq-execution-continuum maps the four positions on that axis, with each arrow labeling the mechanism that moves a project one step rightward.
  ```

## REFUTED findings
- `lst-torchscript-ir` — REFUTED: the explanation the first pass said was absent is present at L1398, the paragraph that closes the TorchScript section: "The TorchScript IR represents operations using the `aten` namespace for core tensor operations, the `prim` namespace for primitives and control flow, static types for every value, and static single-assignment (SSA) form, where each variable is assigned exactly once to simplify compiler analysis." This directly names both namespaces and explains the SSA structure, addressing every gap the first pass identified. The first pass examined only the immediate payoff paragraph (L1385, which pivots to scripting constraints) and the caption (generic "useful for debugging"), and stopped before reaching L1398. L1398 is 15 lines after the listing closes (L1383) but is still squarely within the section-level neighborhood and explains the IR content and its purpose without ambiguity.
