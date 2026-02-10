# Vol1 Formatting Status

## Scope
- Vol1 QMD files: `book/quarto/contents/vol1/**/*.qmd`
- Standard: PIPO calc blocks + mlsys inline usage rules
- Inline rule: `{python} *_str` or `{python} *_math` only

## Last completed
- Inserted a blank line before every `# INPUT` header in vol1 QMDs.
- Added explicit `PURPOSE` sections in `introduction.qmd` figure blocks.
- Added explicit `PURPOSE` sections in `ops.qmd` calc blocks.
- Fixed duplicate `INPUT` header in `introduction.qmd` (`amdahls-pitfall`).
- Shifted calc output formatting to `mlsys.formatting` helpers (reduced inline f-strings).
- Normalized `_str` formatting in vol1 calc blocks to use `fmt`.

## Verified
- No direct inline `{python}` formatting violations found in vol1.
- PIPO structure now consistent for reviewed blocks.

## Next suggested actions
- Optional: add `md_math` / `md_frac` where prose embeds LaTeX-style fractions.
- Add a small lint script to enforce PIPO headers + blank-line rule.
- Apply the figure/plot PURPOSE template for consistency.

## Notes
- Canonical guidance lives in `book/quarto/mlsys/README.md` and `book/quarto/mlsys/calc.qmd`.
