# Vol1 Formatting Status {#sec-vol1-formatting-status}

## Scope {#sec-vol1-formatting-status-scope-8c26}
- Vol1 QMD files: `book/quarto/contents/vol1/**/*.qmd`
- Standard: PIPO calc blocks + mlsys inline usage rules
- Inline rule: `{python} *_str` or `{python} *_math` only

## Last completed {#sec-vol1-formatting-status-last-completed-f439}
- Inserted a blank line before every `# INPUT` header in vol1 QMDs.
- Added explicit `PURPOSE` sections in `introduction.qmd` figure blocks.
- Added explicit `PURPOSE` sections in `ops.qmd` calc blocks.
- Fixed duplicate `INPUT` header in `introduction.qmd` (`amdahls-pitfall`).
- Shifted calc output formatting to `mlsys.formatting` helpers (reduced inline f-strings).
- Normalized `_str` formatting in vol1 calc blocks to use `fmt`.

## Verified {#sec-vol1-formatting-status-verified-a9bb}
- No direct inline `{python}` formatting violations found in vol1.
- PIPO structure now consistent for reviewed blocks.

## Next suggested actions {#sec-vol1-formatting-status-next-suggested-actions-78b7}
- Optional: add `md_math` / `md_frac` where prose embeds LaTeX-style fractions.
- Add a small lint script to enforce PIPO headers + blank-line rule.
- Apply the figure/plot PURPOSE template for consistency.

## Notes {#sec-vol1-formatting-status-notes-084c}
- Canonical guidance lives in `book/quarto/mlsys/README.md` and `book/quarto/mlsys/calc.qmd`.
