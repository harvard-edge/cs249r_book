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

## QMD Section Headers and Bold Text Style Guide

Consistent formatting of section headers and bold text across all `.qmd` chapters.

### Rule 1: Headers for Section Divisions — Never Substitute Bold Text

If a bold-start paragraph introduces a new topic that gets its own paragraph(s) of discussion, it **must** be a proper header (`####`, `#####`) at the appropriate level. Bold text must never act as a pseudo-header.

**The test:** "Does this introduce a distinct topic with its own paragraph(s) of discussion?" If yes, it needs a header.

```markdown
<!-- ❌ BAD — bold text acting as a section header -->
**Tensor Core Architecture.** Modern GPUs include dedicated...

<!-- ✅ GOOD — proper header -->
#### Tensor Core Architecture

Modern GPUs include dedicated...
```

```markdown
<!-- ❌ BAD — bold paragraph lead as sub-section -->
**Data Quality and Heterogeneity** present the first hurdle...

<!-- ✅ GOOD — proper header -->
##### Data Quality and Heterogeneity

Data quality and heterogeneity present the first hurdle...
```

Note: When converting `**Bold Title.**` or `**Bold Title**` to a header, remove the bold markers and any trailing period. The first sentence of the following paragraph should NOT repeat the header text — rewrite to flow naturally from the header.

### Rule 2: Bold Leads ARE Allowed Inside Callouts

Inside `.callout-*` boxes (notebooks, examples, perspectives), bold labels provide internal structure. These are contained pedagogical units, not chapter sections:

```markdown
<!-- ✅ GOOD — internal callout structure -->
::: {.callout-notebook title="Training GPT-3"}
**The Variables**: ...
**The Calculation**: ...
**The Systems Conclusion**: ...
:::
```

### Rule 3: Bold Leads ARE Allowed in Parallel Definition-Style Lists

When introducing 3+ parallel items in quick succession (each 1-2 sentences), bold leads work as a lightweight definition list:

```markdown
<!-- ✅ GOOD — parallel definition items, each 1-2 sentences -->
**Cloud ML** deploys models on datacenter GPUs with virtually unlimited compute.
**Edge ML** runs inference on local hardware near the data source.
**Mobile ML** targets smartphones with strict power and thermal budgets.
**TinyML** targets microcontrollers with kilobytes of memory.
```

**The test:** Each item is 1-2 sentences max, and they form a clearly parallel structure. If any item expands to a full paragraph or more, convert ALL items to headers.

### Rule 4: Fallacy/Pitfall Format Is Standard

The Fallacies and Pitfalls section at the end of every chapter uses this established format. Keep it:

```markdown
**Fallacy:** *One deployment paradigm solves all ML problems.*

Explanation paragraph...

**Pitfall:** *Minimizing computational resources minimizes total cost.*

Explanation paragraph...
```

### Rule 5: Bold Terms in List Items Are Fine

Standard list formatting with a bold lead term:

```markdown
<!-- ✅ GOOD — bold term in list item -->
1. **Checkpointing**: saves model state periodically...
- **Forward activations**: stored during the forward pass...
```

### Rule 6: No Bold-Start Paragraphs in Flowing Body Text

Outside of callouts, definition lists, Fallacy/Pitfall sections, and list items, do NOT start a paragraph with bold text. Options:

```markdown
<!-- ❌ BAD — bold lead in body text -->
**An important caveat.** The Iron Law assumes...

<!-- ✅ Option A — just write it as prose -->
An important caveat: the Iron Law assumes...

<!-- ✅ Option B — if it's truly important, use a callout -->
::: {.callout-important}
The Iron Law assumes...
:::

<!-- ✅ Option C — if it introduces a new topic, make it a header -->
#### Important Caveat
The Iron Law assumes...
```

### Rule 7: Header Hierarchy Must Be Strict

Never skip heading levels:

- `##` — Major chapter sections (appear in TOC)
- `###` — Sub-sections within a major section
- `####` — Sub-sub-sections
- `#####` — Fine-grained topics (use sparingly)
- `######` — Avoid; restructure instead

### Rule 8: Deciding Header Level for Conversions

When converting a bold-start paragraph to a header, choose the level that fits the local hierarchy:

- If inside a `##` section with no `###` siblings, use `###`
- If inside a `###` section, use `####`
- If inside a `####` section, use `#####`
- Never introduce a header level that skips over its parent

### Summary Decision Tree

```
Is the bold text inside a .callout-* box?
  → YES: Keep as bold label (Rule 2)
  → NO: Continue...

Is it a Fallacy/Pitfall label?
  → YES: Keep as **Fallacy:**/**Pitfall:** (Rule 4)
  → NO: Continue...

Is it a bold term in a numbered/bulleted list item?
  → YES: Keep as bold list item (Rule 5)
  → NO: Continue...

Is it one of 3+ parallel items, each 1-2 sentences?
  → YES: Keep as definition-style list (Rule 3)
  → NO: Continue...

Does it introduce a topic with its own paragraph(s)?
  → YES: Convert to header at appropriate level (Rule 1, 8)
  → NO: Remove bold or rewrite as plain prose (Rule 6)
```
