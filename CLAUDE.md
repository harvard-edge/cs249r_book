# Claude Instructions for mlsysbook-vols

Project-wide conventions for AI assistance.

## QMD Inline Python: The Just-in-Time Pattern

When working with `.qmd` files, Python compute cells should follow the **just-in-time pattern**: small, focused cells placed immediately before the prose that uses their values.

### Cell Documentation Format

Every compute cell should have a documentation header box:

```python
```{python}
#| echo: false
#| label: descriptive-name
# ┌─────────────────────────────────────────────────────────────────────────────
# │ CELL NAME IN CAPS
# ├─────────────────────────────────────────────────────────────────────────────
# │ Context: Which section/callout/figure uses these values
# │
# │ Why: 2-3 sentences explaining the pedagogical purpose of this calculation
# │
# │ Imports: physx.constants (LIST_CONSTANTS), physx.formatting (fmt)
# │ Exports: var1_str, var2_str
# └─────────────────────────────────────────────────────────────────────────────
from physx.constants import CONSTANT1, CONSTANT2
from physx.formatting import fmt

# --- Inputs (description of source) ---
input_var = 100                          # description

# --- Outputs (formatted strings for prose) ---
output_value = input_var * 2
output_str = fmt(output_value, precision=0)  # e.g. "200" units
` ` `
```

### Inline Python with LaTeX Math

When combining Python values with mathematical notation, use **simple mixing**:

```markdown
<!-- ✅ GOOD - Mix inline Python with LaTeX $...$ -->
`{python} params_b_str` $\times 10^9$ $\times$ `{python} bytes_str` bytes = `{python} result_str` GB
```

**Key rules:**
- Python variables hold simple formatted strings (no LaTeX)
- LaTeX `$...$` wraps math symbols (`×`, `10^9`, `\times`)
- Use Unicode `×` in tables (tables don't process LaTeX)

### What NOT To Do

```markdown
<!-- ❌ BAD - Raw values inside $...$ math blocks -->
$T = `{python} value`$ ms  <!-- Decimals get stripped! -->

<!-- ❌ BAD - Inline f-string formatting -->
`{python} f"{value:.2f}"`

<!-- ❌ BAD - Function calls inline -->
`{python} fmt(value, "ms", 3)`

<!-- ❌ BAD - Plain text exponents -->
`{python} value` × 10^9 × 2  <!-- 10^9 won't render as superscript! -->
```

### LaTeX Formatting Rules

| Context | Correct Format | Wrong Format |
|---------|----------------|--------------|
| Prose | `$10^{12}$` | `10^12` |
| Tables | `×` (Unicode) | `\times` or `$\times$` |
| Python strings | `f"{value}"` | `f"$10^{value}$"` |

### Validation

Run validation to check all inline references resolve:
```bash
python3 book/quarto/physx/validate_inline_refs.py --verbose
```
