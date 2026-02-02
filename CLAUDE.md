# Claude Instructions for mlsysbook-vols

Project-wide conventions for AI assistance.

## QMD Inline Python: Two Approaches

When working with `.qmd` files, there are two approaches for inline Python:

### Approach 1: Pre-Formatted Strings (Simple Values)

For simple values outside of LaTeX math blocks, use pre-formatted `_str` variables:

```python
# ✅ GOOD - In Python block, format everything as strings
cloud_compute_ms_str = f"{cloud_stats['compute_ms']:.3f}"
ratio_str = f"{ratio:.0f}"
```

```markdown
<!-- ✅ GOOD - Reference pre-formatted strings -->
Compute time: `{python} cloud_compute_ms_str` ms
```

### Approach 2: Markdown() for LaTeX (Nice Fractions & Math)

For LaTeX math with dynamic values (fractions, scientific notation), use `Markdown()` from IPython:

```python
from IPython.display import Markdown
from calc.formulas import md_frac, md_sci, md_math

# Create Markdown objects that preserve LaTeX
frac_result = md_frac("4.10 × 10⁹", "3.12 × 10¹⁴", "0.013", "ms")
sci_result = md_sci(RESNET50_FLOPs)
math_eq = md_math(f"T_{{comp}} = {value}")
```

```markdown
<!-- ✅ GOOD - Markdown() preserves LaTeX -->
Compute time: `{python} frac_result`
```

**Helper functions in `calc/formulas.py`:**
- `md(latex_str)` - Wrap any LaTeX string
- `md_frac(num, denom, result, unit)` - Create fraction with optional result
- `md_sci(val, precision)` - Scientific notation wrapped in Markdown
- `md_math(expr)` - Wrap math expression in $...$
- `sci(val)` - Unicode scientific notation for plain text (e.g., "4.10 × 10⁹")
- `sci_latex(val)` - LaTeX scientific notation for use inside fractions (e.g., "4.10 \\times 10^{9}")

### What NOT To Do

```markdown
<!-- ❌ BAD - Raw values inside $...$ math blocks -->
$T = `{python} value`$ ms  <!-- Decimals get stripped! -->

<!-- ❌ BAD - Inline f-string formatting -->
`{python} f"{value:.2f}"`

<!-- ❌ BAD - Function calls inline (except Markdown helpers) -->
`{python} fmt(value, "ms", 3)`
```

### Why This Matters

Quarto automatically escapes inline code output, which breaks:
1. **LaTeX backslashes**: `\times` → `times`, `\frac` → `frac`
2. **Decimal points** inside `$...$` math blocks
3. **Superscripts**: `^{9}` → `{9}`

Using `Markdown()` tells Quarto to render the output as-is, preserving LaTeX.

### Validation

Run validation to check for violations:
```bash
python3 book/quarto/calc/validate_inline_refs.py --check-patterns
```
