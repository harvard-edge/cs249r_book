# The MLSys Physics Engine: Owner's Manual

This directory (`book/quarto/calc/`) contains the "source of truth" for the entire textbook.
Instead of hardcoding numbers (which rot), we calculate them from first principles.

## 🏗 Architecture

1.  **`constants.py` (The Database)**
    *   Definitions of hardware specs (H100 FLOPS), physical limits (Speed of Light), and economic assumptions (Electricity cost).
    *   **Rule:** All values must be `pint` Quantities (e.g., `10 * ureg.ms`).
    *   **Rule:** Never define derived values here. Only base constants.

2.  **`formulas.py` (The Logic)**
    *   Canonical equations for TCO, Bottlenecks, and Latency.
    *   **Rule:** Functions accept raw numbers or Quantities, but always return clean floats for Markdown.
    *   **Rule:** Use `sci()` helper to format numbers for LaTeX (e.g., `3.12 \times 10^{14}`).

3.  **`viz.py` (The Style)**
    *   Global `matplotlib` configuration.
    *   **Rule:** Import this before plotting to ensure consistent fonts/colors across volumes.

4.  **`export.py` (The Bridge)**
    *   Exports constants to `constants.json` for Web/OJS interactivity.
    *   **Rule:** Run this script after updating constants.

5.  **`_legacy_ch_ml_systems.py`**
    *   Deprecated code. Keep for reference but do not import.

## 🚀 How to Add Content

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
