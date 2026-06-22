# Fmt / notation audit (`book/tools/audit/fmt`)

Reusable pipeline for catching **spurious `.0`** in `{python}` narrative numbers
(e.g. `153.0 FLOP/byte`, `739,726.0 USD`) across MLSysBook QMD chapters.

This is separate from the MIT Press editorial audit loop documented in
[`../README.md`](../README.md). It pairs with runtime guards in `mlsysim.fmt` (`fmt()`,
`fmt_int()`).

## Three-layer defense

| Layer | What | Tool / code |
|-------|------|-------------|
| **Source** | Explicit author intent | `fmt(x, precision=N)`, `fmt_int(x)` in QMD cells |
| **Runtime** | Fail on bad precision at exec | `mlsysim/mlsysim/fmt.py` `_check_fmt_precision` |
| **Static + preview + HTML** | Catch what still slips through | Scripts in this directory |

## When to use what in QMD

| Situation | Pattern |
|-----------|---------|
| Spec is already a whole number (312 TFLOP/s) | `fmt(x, precision=0)` |
| Meaningful fraction (8.5 µs, 13.1 percent) | `fmt(x, precision=1)` or higher |
| Computed value, display as integer | `fmt_int(x)` or `fmt_int(round(x))` |

## Per-chapter workflow

Run from **repo root**. Set `PYTHONPATH=mlsysim` for any script that execs
chapter cells.

```bash
CH=book/quarto/contents/vol1/training/training.qmd   # example

# 1. Auto-fix fmt() guard failures (traceback line numbers)
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/fix_precision.py "$CH"

# 2. Fix assignments that still render as *.0 in prose preview
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/fix_spurious_prose.py "$CH"

# 3. Prose preview (fast; no Quarto build)
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose.py "$CH" --flagged-only

# 4. Static fmt-family checks (suffix discipline, prefer_fmt_int, etc.)
python3 book/tools/audit/audit_math_canonical.py "$CH"
# or: ./book/binder check math --scope canonical --path "$CH"

# 5. HTML render + final scan
./book/binder build html --vol1 vol1/training --skip-hygiene --skip-validate
python3 book/tools/audit/fmt/audit_html.py \
  book/quarto/_build/html-vol1/contents/vol1/training/training.html
```

Exit code **0** = clean; **1** = findings (or exec failure for prose preview).

## Batch HTML render + audit

Fast single-chapter builds overwrite `_build/html-vol1/`; the render script
**archives** each chapter HTML before moving to the next:

```bash
./book/tools/audit/fmt/render_html.sh vol1
./book/tools/audit/fmt/render_html.sh vol2   # extend chapter list in script
```

Archived output: `book/quarto/_build/html-audit/<vol>/<chapter>.html`

Legacy wrappers at `book/tools/audit/render_vol1_audit.sh` and
`render_html_audit.sh` forward to this script.

## Tool reference

| Script | Purpose |
|--------|---------|
| `fix_precision.py` | Apply fixes when `fmt()` raises `Formatting Precision Error` |
| `fix_spurious_prose.py` | Rewrite `*_str = fmt(...)` lines whose output ends in `.0` |
| `audit_prose.py` | Exec cells, substitute `{python}` refs, flag spurious `.0` |
| `audit_math_canonical.py` | Thin CLI wrapper → `cli/checks/math_canonical.py` |
| `audit_html.py` | Scan rendered HTML narrative for spurious `.0` |
| `spurious_zero.py` | Shared false-positive filters (prose + HTML) |
| `render_html.sh` | Sequential build, archive, audit all chapters in a volume |

## Shared filters

`spurious_zero.py` centralizes false-positive rules (PCIe 4.0, NVLink 3.0,
LaTeX walkthrough literals, `1.0 KFLOPs`, DOI strings, etc.). Update filters
there when HTML audit and prose preview disagree.

## Dependencies

- **Python:** `mlsysim` on `PYTHONPATH` for cell exec
- **HTML audit:** `beautifulsoup4` (`pip install beautifulsoup4`)
- **Build:** `./book/binder` from repo root

## Not part of this toolkit

Exploratory one-off scripts under `scripts/` (`debug_precision.py`,
`converge_precision.py`, …) were used during development and should **not** be
committed. Use the tools above for repeatable audits.

## Related

- `mlsysim/mlsysim/fmt.py` — `fmt()`, `fmt_int()`, precision guards
- [`book/docs/LEGO_CELLS.md`](../../../docs/LEGO_CELLS.md) — inline `{python}` cell placement and review contract
- Pre-commit: `book-check-math` (includes `canonical` scope), `book-check-code` (echo, LEGO)

## Backward compatibility

Thin wrappers remain at the old top-level paths (`fix_fmt_precision.py`,
`audit_prose_preview.py`, etc.) and delegate here. Prefer the `fmt/` paths for
new scripts and docs.
