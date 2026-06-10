# Float-explanation worklist — ml_workflow.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 6 | 6 | 0 | 0 |
| table | 3 | 3 | 0 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 0 | 0 | 0 | 0 |
| **total** | 9 | 9 | 0 | 0 |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

No under-explained floats found; all references explained in-neighborhood.

## Scanner false positives (verified, not findings)

The scanner's "orphan" and "dangling ref" flags here are all artifacts of trailing
punctuation in the `@ref` token, not genuine problems. Confirmed against source:

- `fig-eye-dr` (def L859) was flagged **orphan** because its sole reference at L855
  is written `@fig-eye-dr:` (trailing colon), which the scanner parsed as a dangling
  ref instead of matching the def. The reference is substantive: L855 tells the reader
  to look closely at the figure to see the hemorrhages (dark red spots) that signal
  disease progression, and the caption plus payoff ¶ (L865) carry the "lab-to-clinic"
  point. ✅ Explained.
- `@fig-ai-triad` (L52), `@fig-lifecycle-overview.` (L884), `@tbl-stage-interface.`
  (L968), and `@fig-cascades.` (L2117) appear in "Dangling refs" only because of a
  trailing `:` or `.`; each resolves to a real, well-explained float (or a forward
  ref to another chapter's float, in the case of `fig-ai-triad` and `fig-cascades`,
  which are not defined in this chapter and so are out of scope here).
- Tables show "(none found)" for caption in the bundle because the scanner does not
  parse markdown-pipe table captions. All three tables have full `**Bold Title**:
  Explanation` captions in source (L429, L845, L882).

## Notes on the strongest ✅ cases (for reference, not action)

- `fig-ml-lifecycle` (def L101): referenced six times; each site adds a distinct
  takeaway (the dual-pipeline trace at L99, iteration-cycle reading at L285, and four
  pitfall/fallacy applications at L2165–L2225). Richly explained.
- `fig-mlops-returns` (def L1652): the setup ¶ (L1650) names the coordination tax and
  flywheel effect, and the payoff ¶ (L1750) explains *why* each curve has its shape
  (combinatorial sync cost vs. infrastructure-absorbed coordination). Model example.
- `tbl-stage-interface` (def L882): four references, with the payoff ¶ (L884) tying the
  contract table back to the iteration-cycle figure and the cost of late violations.
