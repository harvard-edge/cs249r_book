# Math-rendering audit tools

Three scripts for catching LaTeX leakage in rendered HTML and PDF output.
Built during the April 2026 math-rendering audit (see
`.claude/rules/book-prose.md` for the underlying conventions these tools
enforce).

All scripts are designed to run from the **repo root**.

## What each script does

| Script | Purpose |
| --- | --- |
| `audit_math_rendering.py` | Builds each chapter's HTML via `binder` and scans the rendered output for raw LaTeX leakage outside MathJax/code zones. |
| `audit_math_pdf.py` | Builds per-chapter PDFs, extracts text via `pdftotext`, renders pages to PNG via `pdftoppm` for visual spot-checking, and applies the same leak detector to the extracted text. |
| `audit_pdf_spot_check.py` | Scans the PDFs produced by `audit_math_pdf.py` for known fix sites (regex map maintained in the script) and emits a markdown manifest pointing to the exact page numbers / PNGs to inspect. |

## Quick start

```bash
# HTML audit across the whole book (~10 minutes)
python3 tools/audit/audit_math_rendering.py

# Targeted audit
python3 tools/audit/audit_math_rendering.py vol1/introduction vol2/inference

# Just re-scan an existing build without rebuilding
python3 tools/audit/audit_math_rendering.py --skip-build

# PDF audit (slower; needs LaTeX toolchain + poppler-utils)
python3 tools/audit/audit_math_pdf.py vol1/introduction
python3 tools/audit/audit_math_pdf.py --fixed   # only chapters from the April 2026 fix set

# Generate visual spot-check map for the saved PDFs
python3 tools/audit/audit_pdf_spot_check.py
```

## Outputs

Both audits write reports to the **repo root** by default:

- `audit-math-report.json` / `audit-math-report.md` — HTML audit results
- `audit-pdf-report.json` / `audit-pdf-report.md` — PDF audit results
- `audit-pdf-spot-check.md` — visual spot-check manifest
- `audit-pdf-output/<vol>/<chap>/{chap.pdf,pages/page-NNN.png}` — saved PDFs and page images

These paths are **gitignored** (see top-level `.gitignore`); they are local
artifacts intended for inspection, not commits.

## Concurrency warning

Do **not** run multiple `binder build` invocations in parallel. They all
mutate the shared `book/quarto/_quarto.yml` and will corrupt each other's
state. The HTML and PDF auditors are both serial internally; just don't run
them in two terminals at once.

## Dependencies

- Standard `binder` build dependencies (Quarto + project venv)
- `pdftotext` and `pdftoppm` from `poppler-utils` (PDF audit only)

## Known false-positive: code blocks in PDFs

`pdftotext` extracts code blocks verbatim, so any chapter that contains
LaTeX-style pseudocode in a `code` block will produce "leaks" in the PDF
text scan that are not actual rendering bugs. Treat the PDF text scan as a
soft signal; the rendered PNGs are the source of truth for PDF output.
