# The MLSys Physics Engine: Owner's Manual

This directory (`book/quarto/mlsys/`) contains the MLSys Physics Engine — the "source of truth"
for all quantitative claims in the textbook. Instead of hardcoding numbers (which rot), we
calculate them from first principles with unit safety and reusable formulas.

## 🏗 Architecture

1.  **`constants.py` (The Database)**
    *   Definitions of hardware specs (H100 FLOPS), physical limits (Speed of Light), and economic assumptions (Electricity cost).
    *   **Rule:** All values must be `pint` Quantities (e.g., `10 * ureg.ms`).
    *   **Rule:** Never define derived values here. Only base constants.

2.  **`formulas.py` (The Logic)**
    *   Canonical equations for TCO, Bottlenecks, and Latency.
    *   **Rule:** Physics only—return raw numbers or Pint Quantities (no formatting).

3.  **`formatting.py` (The Presentation)**
    *   Markdown/LaTeX helpers (`fmt`, `sci`, `md_math`, `md_frac`, ...).
    *   **Rule:** Use these for display output only.

4.  **`viz.py` (The Style)**
    *   Global `matplotlib` configuration.
    *   **Rule:** Import this before plotting to ensure consistent fonts/colors across volumes.

5.  **`export.py` (The Bridge)**
    *   Exports constants to `constants.json` for Web/OJS interactivity.
    *   **Rule:** Run this script after updating constants.

6.  **`_legacy_ch_ml_systems.py`**
    *   Deprecated code. Keep for reference but do not import.

## 📦 Core Modules (Calculator Engine)

**Foundations**
- `constants.py` — base constants + unit registry
- `formatting.py` — output helpers (value/str/math/frac)

**Core formulas**
- `formulas.py` — atomic equations (latency, speedup, bottleneck)

**System helpers**
- `roofline.py` — arithmetic intensity, ridge point, roofline bound
- `latency.py` — pipeline latency helpers
- `throughput.py` — QPS/tokens per second
- `speedup.py` — Amdahl/efficiency
- `energy.py` — energy move/compute

**Catalogs**
- `archetypes.py` — workload archetypes
- `workloads.py` — lighthouse models
- `systems.py` — system profiles (single-node, mobile, tiny)
- `tiers.py` — deployment/service tiers

**Execution trace**
- `registry.py` — chapter tape (`start_chapter`, `record`, `dump_tape`)

## 🚀 How to Add Content

## ✅ Calculation Conventions (for QMD)

These rules make calculations auditable and keep prose consistent across HTML/EPUB/PDF.

### 1) Compute anything derived
If a value is the result of combining inputs (ratio, conversion, multiplication, rounding), compute it in a Python block and reference the display string in the text or table.

### 2) Leave raw facts as literals
Dates, names, event years, and citations stay in prose. Do not over-encode non-derived facts.

### 3) Every table cell must be either:
- A computed display variable (preferred), or
- A literal fact (explicitly non-computed)

### 4) Tie calculations to their targets
Add a short comment in the calc block:
```python
# Used in: Table "Latency Numbers" (rows: compute + network)
```
This creates a traceable link from computation → narrative.

### 5) Use a consistent block structure (PIPO+)
**PIPO+** = **Purpose → Input → Process → Output**, with optional **Context** and **Checks** for clarity and auditability.
````markdown
```{python}
#| label: <calc-id>
#| echo: false

# =============================================================================
# PURPOSE
# =============================================================================
# Purpose: One-line description of the calculation.
# Used in: Section/Table/Figure reference.
# Context: (Optional) One sentence on why this matters.

# =============================================================================
# INPUT (SOURCES)
# =============================================================================
# from mlsys.constants import ...

# =============================================================================
# INPUT (ASSUMPTIONS)
# =============================================================================
# a_value = 1.6  # x/year
# b_value = 1.2  # x/year

# =============================================================================
# PROCESS
# =============================================================================
# ratio_value = a_value / b_value  # 1.6 / 1.2 = 1.33...

# =============================================================================
# CHECKS
# =============================================================================
# assert a_value > 0
# assert 0 < ratio_value < 10

# =============================================================================
# OUTPUT
# =============================================================================
# ratio_value = ratio_value  # keep raw numeric for auditability
# a_str = f"{a_value:.1f}"
# b_str = f"{b_value:.1f}"
# ratio_str = f"{ratio_value:.2f}"

# Optional: LaTeX output goes here (still OUTPUT)
# from mlsys.formatting import md_math
# ratio_math = md_math(rf"\frac{{{a_str}}}{{{b_str}}} \approx {ratio_str}")
```
````

**Formatting note:** The separator line should be followed immediately by the header (no blank line between `# =============================================================================` and the next header).

### 6) Use display strings in text blocks
Inline prose should use `{python} <name>_str` or `{python} <name>_math`.
Do not hardcode derived numbers in prose.
Prefer `mlsys.formatting` helpers (`fmt`, `display_value`, `md_math`, `md_frac`) over inline f-strings.

### 6a) Use `Markdown()` for LaTeX output
If the output contains LaTeX (fractions, `\times`, exponents), return a `Markdown()` object (via `md_math`, `md_frac`, `md_sci`) and reference it inline:
```python
from mlsys.formatting import md_math
ratio_math = md_math(r"\frac{4.1 \times 10^{9}}{3.1 \times 10^{14}}")
```
Then in prose:
```
`{python} ratio_math`
```

### 7) Name by meaning, not formatting
Use explicit raw values (`latency_ms_value`) alongside display strings (`latency_ms_str`) for auditability.

### 7a) Make representation explicit in Markdown
Only reference explicit representation variables in prose/tables:
- `*_value` — raw numeric (keep for auditability; avoid inline use)
- `*_str` — plain text / formatted numbers
- `*_math` or `*_frac` — LaTeX via `Markdown()`
This makes the output type obvious at a glance.

### 8) Prefer one block per narrative chunk
Avoid mixing unrelated calculations into a single Python block.

### 9) Figures/Plots (PIPO + SETUP/DATA/PLOT)
Figure blocks can use `SETUP/DATA/PLOT`, but should still start with a PURPOSE header.
````markdown
```{python}
#| label: fig-<name>
#| echo: false
#| fig-cap: "<caption>"
#| fig-alt: "<alt>"

# =============================================================================
# PURPOSE
# =============================================================================
# Purpose: One-line description of the figure.
# Used in: Figure "<caption short name>".

# =============================================================================
# SETUP
# =============================================================================
# imports, viz setup

# =============================================================================
# DATA
# =============================================================================
# data tables or arrays

# =============================================================================
# PLOT
# =============================================================================
# plotting code
```
````

## Checklist

- Use the PIPO+ template exactly (Purpose, Input, Process, Output; optional Context/Checks).
- Keep one calc block per narrative chunk.
- Name outputs with explicit suffixes: `*_value`, `*_str`, `*_math`.
- Use `mlsys.formatting` helpers instead of inline f-strings.
- Inline prose should reference only `*_str` or `*_math`.

## 📌 Where should the code live?

**Default (recommended):** Keep calculations inline in the QMD for context.
Use `mlsys/` helpers for shared math, units, and formatting.

**Why:** It keeps prose clean, enables reuse, and makes tests easy.

### A) Chapter calculator modules (preferred)
Create a chapter module:
```python
# book/quarto/mlsys/ch_introduction.py
def calc_intro_setup():
    return {"google_search_b": "8.5"}
```

In the QMD:
```python
```{python}
#| label: intro-setup
#| echo: false
from mlsys.ch_introduction import calc_intro_setup
_intro = calc_intro_setup()
google_search_b = _intro["google_search_b"]
```
```

### B) Inline QMD blocks (only for tiny one-offs)
Inline blocks are fine for **small, local** calculations that are not reused and don’t warrant a chapter module.
If a value appears in more than one place, move it into `mlsys/`.

## 🧭 Naming & structure conventions

- **Files:** `ch_<chapter>.py` (e.g., `ch_introduction.py`)
- **Functions:** `calc_<section>()` (e.g., `calc_gpt3_training()`)
- **Variables:** raw values use `_value` suffix; display strings use `_str`; LaTeX objects use `_math`
- **Provenance:** Add a comment describing where the values are used:
  ```python
  # Used in: Table "Latency Numbers" (rows: compute + network)
  ```

### 1. Adding a New Hardware Accelerator
1.  Open `constants.py`.
2.  Define the spec using `pint`:
    ```python
    BLACKWELL_FLOPS_FP16 = 20000 * TFLOPs / second
    ```

### 2. Writing a Formula in Markdown
Don't write `$4.1 \times 10^9$`.
Write:
```python
from mlsys.formatting import sci
{python} sci(RESNET50_FLOPs)
```
This renders as $4.10 \times 10^{9}$ and auto-updates if the model changes.

### 3. Debugging Narrative Logic
If the text says "Edge is Cheaper" but the math says "Cloud is Cheaper", the `test_narrative_invariants.py` will fail.
1.  Run tests: `pytest book/tests/test_narrative_invariants.py`
2.  If it fails, **rewrite the prose**, don't just change the test. The test protects the truth.

## 📦 Dependencies
*   `pint`: For unit safety.
*   `matplotlib`: For charts.
*   `pandas`: For data tables.

## ⚠️ Common Pitfalls
*   **Dimensionality Errors:** If `pint` yells about "Cannot convert second to meter", you likely divided Distance by Time incorrectly.
*   **Format Strings:** Do not use `f"{constant}"` directly. Use `f"{constant.magnitude}"` or `fmt(constant)`.
