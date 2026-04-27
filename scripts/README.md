# Figure Audit Automation

This directory contains `figure_audit.py`, a script designed to automate the visual auditing of figures within the ML Systems textbook.

## What it does

The script orchestrates a multimodal audit of every figure across Volume 1 and Volume 2 of the textbook. It ensures that the prose, the captions (`fig-cap`), and the alt-text (`fig-alt`) precisely match the content of the fully rendered visual images.

1.  **Discovery:** It scans the `book/quarto/contents/` directory to identify all `.qmd` chapters containing figures.
2.  **Visual Extraction:** It resolves the corresponding published HTML URL for each chapter, parses the HTML, and downloads the exact rendered `<img src="...">` and inline `<svg>` visual assets locally.
3.  **Auditing:** It dispatches parallel worker tasks via the `gemini` CLI. The CLI is given explicit instructions to load the local images visually, compare them directly against the `.qmd` source text, and evaluate them based on the `figure-audit-brief.md` rubric.
4.  **Reporting:** It generates strict, granular YAML output files in `.claude/_reviews/Figure Audit/`, detailing any misalignments (e.g., the text claims $10^4$ but the chart shows $10^3$) along with surgically precise `.qmd` fix recommendations.

## How to use it

Run the script from the repository root:

```bash
python3 scripts/figure_audit.py
```

### Pre-requisites

*   You must have `gemini` CLI installed and authenticated on your local machine.
*   The script assumes the rendered HTML book is available at `https://harvard-edge.github.io/cs249r_book_dev/...` (used purely to scrape the final image variants).

### Applying the fixes

Once `figure_audit.py` finishes running, your `.claude/_reviews/Figure Audit/` directory will be populated with `.yml` files containing `proposed_fix` entries.

These fixes are written as precise, minimal adjustments targeting the `.qmd` source files. They can either be applied manually by a human reviewing the YAML reports, or parsed programmatically/agentically to apply the diffs across the workspace.