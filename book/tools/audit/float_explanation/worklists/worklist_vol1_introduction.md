# Float-explanation worklist — introduction.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 8 | 8 | 0 | 0 |
| table | 9 | 8 | 1 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 6 | 6 | 0 | 0 |
| **total** | 23 | 22 | 1 | 0 |

## Scanner note
The scanner flagged six floats as "orphans" (`eq-cost-scaling`, `eq-intro-iron-law`, `fig-ai-triad`, `tbl-ai-evolution-strengths`, `tbl-introduction-engineering-missions`, `tbl-efficiency-priorities`) and listed matching "dangling refs." This is a regex artifact: each of these IS referenced in prose, but the reference is immediately followed by a colon or period (for example `@eq-cost-scaling:`, `@fig-ai-triad:`), which the scanner's ref-matcher does not treat as a resolved reference. All six were verified against source and are properly referenced and explained. No true orphans exist in this chapter.

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `tbl-software-1-vs-2` — def L128  (Thin)
- **Caption:** **The Paradigm Shift from Software 1.0 to Software 2.0**: In Software 2.0, the "programmer" does not write the logic; they curate the dataset that the optimization process uses to write the logic. Debugging therefore moves upstream from code to data. The "compiler" analogy is approximate: unlike a deterministic compiler, the training process is stochastic and may produce different "executables" from the same "source code."
- **Ref(s):** L116 `@Tbl-software-1-vs-2`: "Andrej Karpathy formalized this distinction as the shift from **Software 1.0** to **Software 2.0** [@karpathy2017software], a framing that captures *why* ML systems require entirely new engineering approaches. @Tbl-software-1-vs-2 summarizes this paradigm shift."
- **Context checked:** ref (bare "summarizes") ✗ · prev ¶ (section heading) ✗ · this ¶ (names 1.0/2.0 framing but previews no row content) partial · next ¶ (Karpathy footnote) ✗ · caption (carries the real explanation, but it is the only place that does) ✓-caption-only · payoff ¶ L130 ("Google researchers quantified the resulting technical debt") moves on without unpacking the table ✗
- **Why thin, not dead-end:** the caption genuinely explains what the table shows (debugging moves upstream from code to data, the compiler analogy is approximate), so the reader is not stranded. But the ref sentence is a bare "summarizes this paradigm shift" and the body prose never tells the reader the single takeaway the table is built around: the failure-mode row (loud crash vs. silent metric degradation) is the one that motivates the entire rest of the chapter. A one-clause payoff in the ref sentence would carry the table instead of announcing it.
- **Suggested rewrite (flag-only):**
  ```diff
  - Andrej Karpathy[^fn-karpathy-sw2] formalized this distinction as the shift from **Software 1.0**\index{Software 1.0} to **Software 2.0**\index{Software 2.0} [@karpathy2017software], a framing that captures *why* ML systems require entirely new engineering approaches. @Tbl-software-1-vs-2 summarizes this paradigm shift.
  + Andrej Karpathy[^fn-karpathy-sw2] formalized this distinction as the shift from **Software 1.0**\index{Software 1.0} to **Software 2.0**\index{Software 2.0} [@karpathy2017software], a framing that captures *why* ML systems require entirely new engineering approaches. @Tbl-software-1-vs-2 maps the shift term by term, and the row that drives this entire chapter is the failure mode, which moves from a loud crash in Software 1.0 to silent metric degradation in Software 2.0.
  ```
