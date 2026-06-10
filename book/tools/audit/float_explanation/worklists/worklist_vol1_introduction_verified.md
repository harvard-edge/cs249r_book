# Verified findings — introduction.qmd (vol1)
Prior findings: 1 | Survived: 1 | Refuted: 0

## SURVIVING findings

### ⚠️ `tbl-software-1-vs-2` — def L128
- Ref: "@Tbl-software-1-vs-2 summarizes this paradigm shift."
- Why it survives: The ref sentence is a bare pointer ("summarizes this paradigm shift") with no inline takeaway. The preceding paragraph names the 1.0/2.0 framing but describes no row content. The next paragraph (L130) pivots immediately to Google's technical-debt paper without unpacking the table. The caption carries the real explanation (debugging moves upstream from code to data; the compiler analogy is stochastic), so the reader is not stranded, but the caption is the *only* place the table's significance is stated. No body-prose sentence tells the reader why the failure-mode row (loud crash vs. silent metric degradation) is the consequential distinction that motivates the entire chapter. The adversarial standard requires that explanation live in the neighborhood, not solely in the caption.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - Andrej Karpathy[^fn-karpathy-sw2] formalized this distinction as the shift from **Software 1.0**\index{Software 1.0} to **Software 2.0**\index{Software 2.0} [@karpathy2017software], a framing that captures *why* ML systems require entirely new engineering approaches. @Tbl-software-1-vs-2 summarizes this paradigm shift.
  + Andrej Karpathy[^fn-karpathy-sw2] formalized this distinction as the shift from **Software 1.0**\index{Software 1.0} to **Software 2.0**\index{Software 2.0} [@karpathy2017software], a framing that captures *why* ML systems require entirely new engineering approaches. @Tbl-software-1-vs-2 maps the shift term by term. The row that drives the rest of this chapter is the failure mode: Software 1.0 fails loudly with a crash, while Software 2.0 fails silently through metric degradation, making the failure invisible until a monitoring system catches it.
  ```

## REFUTED findings
(none)
