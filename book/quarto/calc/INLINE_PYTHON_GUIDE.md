# Inline Python Rendering Guide

This guide documents the correct patterns for using inline Python (`{python} var`) in Quarto QMD files, and how to avoid common rendering issues.

## The Problem

Quarto's inline Python feature (`{python} var`) has a known limitation: **output inside LaTeX math mode gets escaped**, causing decimal points to be stripped.

Example of the bug:
```markdown
$x = `{python} "5.9"`$   →  renders as:  x = 59  (WRONG!)
```

This affects:
- Inline Python inside `$...$` math delimiters
- Inline Python adjacent to LaTeX symbols like `$\times$`
- Inline Python in grid tables

## Quick Reference

### ✅ CORRECT Patterns

```markdown
<!-- Python value OUTSIDE math, symbol as Unicode -->
The speedup is $\frac{1}{0.05}$ = `{python} speedup_str`×

<!-- Separate math and computed value -->
Ridge point: $R_{peak}/BW$ = `{python} ridge_point` FLOP/byte

<!-- Pipe tables with inline Python -->
| GPU | TFLOPS |
|-----|--------|
| H100 | `{python} h100_tflops` |
```

### ❌ INCORRECT Patterns

```markdown
<!-- Python INSIDE math - will strip decimals! -->
$x = `{python} "5.9"`$

<!-- Python adjacent to LaTeX symbol -->
`{python} val`$\times$

<!-- Grid tables with inline Python -->
+------+--------+
| GPU  | TFLOPS |
+======+========+
| H100 | `{python} h100_tflops` |
+------+--------+
```

## Migration Checklist

### Step 1: Identify Issues

Run the pattern validator:
```bash
cd book/quarto
python3 calc/validate_inline_refs.py --check-patterns
```

This reports three issue types:
- `LATEX_MATH`: Python inside `$...$`
- `LATEX_ADJACENT`: Python next to `$\times$` etc.
- `GRID_TABLE`: Grid tables with Python

### Step 2: Fix LATEX_MATH Issues

Move Python values outside the dollar signs:

**Before:**
```markdown
$\text{Time} = `{python} time_str` \text{ seconds}$
```

**After:**
```markdown
Time = `{python} time_str` seconds
<!-- OR if you need the equation format: -->
$\text{Time} =$ `{python} time_str` seconds
```

### Step 3: Fix LATEX_ADJACENT Issues

Replace LaTeX symbols with Unicode equivalents:

| LaTeX | Unicode | Usage |
|-------|---------|-------|
| `$\times$` | `×` | Multiplication |
| `$\approx$` | `≈` | Approximately |
| `$\ll$` | `≪` | Much less than |
| `$\gg$` | `≫` | Much greater than |
| `$\mu$` | `μ` | Micro (μs, μm) |
| `$10^9$` | `10⁹` | Powers of 10 |
| `$10^6$` | `10⁶` | Powers of 10 |

**Before:**
```markdown
`{python} speedup_str`$\times$ faster
```

**After:**
```markdown
`{python} speedup_str`× faster
```

### Step 4: Convert Grid Tables to Pipe Tables

Grid tables with `+---+` separators don't work well with inline Python.

**Before (Grid Table):**
```markdown
+------+--------+
| GPU  | TFLOPS |
+======+========+
| H100 | `{python} h100_tflops` |
+------+--------+
```

**After (Pipe Table):**
```markdown
| GPU | TFLOPS |
|-----|--------|
| H100 | `{python} h100_tflops` |
```

### Step 5: Verify with HTML Render

Always verify changes render correctly:

```bash
# Render single file to HTML
quarto render contents/vol1/chapter/file.qmd --to html --output-dir /tmp/test

# Open and inspect
open /tmp/test/file.html
```

Check that:
- No raw `` `{python}` `` appears in output
- Decimal points are preserved (5.9 not 59)
- Tables render with all columns visible

## Best Practices for New Content

### 1. Use the Calc Module Pattern

Define all computed values in a chapter module:

```python
# calc/ch_myfile.py
class C:
    ridge_point = f"{_ridge:.0f}"        # "153"
    speedup_str = f"{_speedup:.1f}"      # "5.9"
```

Then import once in the QMD:
```python
#| echo: false
from ch_myfile import C
```

### 2. Pre-format Values as Strings

Do formatting in Python, not inline:
```python
# Good: format in the module
speedup_str = f"{speedup:.1f}"   # "5.9"

# Then use directly
`{python} C.speedup_str`
```

### 3. Keep Math and Values Separate

Write equations in pure LaTeX, show computed results separately:

```markdown
The equation is:
$$\text{Speedup} = \frac{1}{s + (1-s)/n}$$

For our example ($s = 0.05$, $n = 8$): Speedup = `{python} C.speedup_str`×
```

### 4. Use Unicode for Adjacent Symbols

When a computed value is followed by a symbol, use Unicode:
```markdown
`{python} C.value`× faster
`{python} C.time`μs latency
≈ `{python} C.approx_value` GB
```

## Validation Tools

### Pre-Render Validation
```bash
# Check for undefined variables AND problematic patterns
python3 calc/validate_inline_refs.py --check-patterns --verbose
```

### Post-Render Verification
```bash
# Render and manually verify
quarto render FILE.qmd --to html --output-dir /tmp/verify
open /tmp/verify/FILE.html
```

## Reference: Issue Types

| Issue | Cause | Effect | Fix |
|-------|-------|--------|-----|
| `LATEX_MATH` | Python inside `$...$` | Decimals stripped | Move Python outside |
| `LATEX_ADJACENT` | Python next to `$\times$` | Decimals stripped | Use Unicode × |
| `GRID_TABLE` | Grid table with Python | Columns collapse | Use pipe table |

## Migration Progress

Work through files in book order. For each file:
1. Run validator: `python3 calc/validate_inline_refs.py --check-patterns`
2. Fix issues found
3. Render to HTML: `quarto render FILE.qmd --to html --output-dir /tmp/test`
4. Visually verify computed values
5. Mark complete below
6. Document any new lessons learned

### Vol1 Progress

| # | Chapter | File | Status | Issues Found | Notes |
|---|---------|------|--------|--------------|-------|
| 0 | Foreword | `foreword.qmd` | ⬜ | | |
| 1 | Introduction | `introduction.qmd` | ⬜ | | |
| 2 | ML Systems | `ml_systems.qmd` | ⬜ | | |
| 3 | DL Primer | `dl_primer.qmd` | ⬜ | | |
| 4 | Workflows | `workflow.qmd` | ⬜ | | |
| 5 | Data Engineering | `data_engineering.qmd` | ⬜ | | |
| 6 | Data Selection | `data_selection.qmd` | ⬜ | | |
| 7 | DNN Architectures | `dnn_architectures.qmd` | ⬜ | | |
| 8 | Frameworks | `frameworks.qmd` | ⬜ | | |
| 9 | Training | `training.qmd` | ⬜ | | |
| 10 | HW Acceleration | `hw_acceleration.qmd` | ⬜ | | |
| 11 | Model Compression | `model_compression.qmd` | ⬜ | | |
| 12 | Serving | `serving.qmd` | ⬜ | | |
| 13 | Benchmarking | `benchmarking.qmd` | ⬜ | | |
| 14 | Ops | `ops.qmd` | ⬜ | | |
| 15 | Responsible Engr | `responsible_engr.qmd` | ⬜ | | |
| 16 | Conclusion | `conclusion.qmd` | ⬜ | | |
| A | Appendix: DAM | `appendix_dam.qmd` | ✅ | Grid tables, LaTeX math | Converted to pipe tables, Unicode symbols |
| B | Appendix: Data | `appendix_data.qmd` | ✅ | Grid tables | Converted to pipe tables |
| C | Appendix: Algorithm | `appendix_algorithm.qmd` | ✅ | Grid tables | Converted to pipe tables |
| D | Appendix: Machine | `appendix_machine.qmd` | ✅ | Grid tables, LaTeX adjacent | Pipe tables, Unicode ×≈μ |

Legend: ⬜ Not started | 🔄 In progress | ✅ Complete | ⏭️ No inline Python

---

## Lessons Learned

Document new issues and fixes as you encounter them.

### Lesson 1: Decimal Points in LaTeX (2026-02-01)
**Issue:** `$value = `{python} "5.9"`$` renders as `59`
**Cause:** Quarto escapes markdown chars (`.`) inside math mode
**Fix:** Move Python outside: `$value =$ `{python} "5.9"``

### Lesson 2: Grid Tables Break with Python (2026-02-01)
**Issue:** Grid tables with inline Python have collapsed columns
**Cause:** Grid table parser conflicts with inline code
**Fix:** Convert to pipe tables (`|---|` format)

### Lesson 3: LaTeX Symbols Adjacent to Python (2026-02-01)
**Issue:** `` `{python} val`$\times$`` strips decimals
**Cause:** Same escaping issue at boundary
**Fix:** Use Unicode: `` `{python} val`×``

### Lesson 4: Bullet Lists Need Blank Line Before Start (2026-02-01)
**Issue:** Bullet points render inline in PDF (all on one line)
**Cause:** PDF/LaTeX needs a blank line before the bullet list starts
**Fix:** Add blank line between intro text and first bullet

```markdown
<!-- WRONG - no blank line before list -->
Where:
*   Item 1
*   Item 2

<!-- CORRECT - blank line before list -->
Where:

*   Item 1
*   Item 2
*   Item 3
```

Note: Bullets can be consecutive - only need blank line BEFORE the list starts.

### Lesson 5: (Add new lessons here)
**Issue:** 
**Cause:** 
**Fix:** 

---

## Workflow Commands

```bash
# 1. Check a specific file for issues
python3 calc/validate_inline_refs.py --check-patterns | grep "filename"

# 2. Render single file to HTML for testing
quarto render contents/vol1/chapter/file.qmd --to html --output-dir /tmp/test

# 3. Open rendered HTML
open /tmp/test/file.html

# 4. After fixing, verify no issues remain
python3 calc/validate_inline_refs.py --check-patterns | grep "filename"
```

---

*Last updated: 2026-02-01*
*See also: [Quarto Issue #12546](https://github.com/quarto-dev/quarto-cli/discussions/12546)*
