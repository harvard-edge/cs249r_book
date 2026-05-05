# Math Audit: Volume 1 Frontmatter About

Source: `book/quarto/contents/vol1/frontmatter/about.qmd`

## Scope

Audited the about frontmatter for math, numeric, unit, complexity, and prose-equation consistency issues using direct reasoning only. Source `.qmd` files were not modified.

## Findings

### 1. Accuracy loss is expressed as "percent" where "percentage points" is likely intended

- **Line 32:** `reducing parameters by 4$\times$ decreases inference latency by 2.3$\times$ on this hardware, at a cost of 1.2 percent accuracy`

The speedup-style quantities, `4$\times$` and `2.3$\times$`, are dimensionless ratios and are internally plausible as an illustrative measurement. The issue is the accuracy loss wording. Accuracy is usually reported as a percentage, so an absolute drop should be stated in percentage points. "1.2 percent accuracy" is ambiguous: it could mean a relative 1.2% change in the accuracy value, not a 1.2-point drop.

**Proposed correction:** Change `at a cost of 1.2 percent accuracy` to `at a cost of 1.2 percentage points of accuracy` or `with a 1.2 percentage-point accuracy drop`.

### 2. Responsible-engineering data-bar prose overstates the numeric intensity

- **Lines 80-81:** `Responsible Engr.` uses `\mlsysstack{0}{0}{10}{10}{20}{40}{90}{15}`
- **Line 86:** `the top layers dominate and the data bar carries a moderate tint`

The stack macro takes arguments as `{hw}{fw}{models}{train}{serve}{ops}{apps}{data}`, so the data-bar intensity is the eighth argument. For Responsible Engineering, the data-bar value is `15` on a 0-100 scale. That is faint, and it is lower than the Training example's data value of `20`, which the same sentence describes as "moderately lit." The prose therefore overstates the visible data intensity for the responsible-engineering example.

**Proposed correction:** Either change the prose to `the top layers dominate and the data bar carries a faint tint`, or raise the eighth macro argument if the intended visual is genuinely moderate, for example `\mlsysstack{0}{0}{10}{10}{20}{40}{90}{30}`.

## Notes

- **Line 36:** `one to eight accelerators` is a clear scope statement for a single-node system and does not conflict with the surrounding prose.
- **Lines 56-60:** `\mlsysstackfull{60}{60}{60}{60}{60}{60}{60}{60}` is internally consistent: all seven layer intensities and the data-bar intensity are equal.
- **Lines 68-81:** The three example stack macros use values in the expected 0-100 range, and the dominant layers match their captions: Hardware Acceleration peaks at hardware (`90`), Training peaks at training (`90`), and Responsible Engineering peaks at applications (`90`).
- No algorithmic complexity claims or formal equations appear in this file.
