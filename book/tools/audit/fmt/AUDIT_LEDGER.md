# FMT audit ledger

This ledger records formatter migration validation. Keep it current while
working. The goal is to make each touched LEGO cell and final inline prose audit
reviewable without rediscovery.

## Ledger rules

- Add an entry for every LEGO cell touched on this branch.
- Validate every exported `_str`, `_math`, `_eq`, and `_frac` from that cell.
- Record whether output is byte-identical or intentionally changed.
- Read the rendered sentence after inline substitution.
- Record exceptions instead of leaving them implicit.

## Entry template

```text
Chapter:
File:
Cell anchor / line:
Change type:
Exports checked:
Before values:
After values:
Equivalence:
Rendered prose checked:
Spacing/unit/glyph result:
Manual prose read:
HTML evidence:
PDF/tex evidence:
Notes / exceptions:
Status:
```

## Current session notes

- Plan of record added before continuing implementation.
- Existing branch history already includes multiple WS4/A1/A2/render
  verifications summarized in `NIGHT_RESUME.md`.
- Structured formatter API batch added:
  - `fmt_usd(scale=..., per=...)`
  - `fmt_count(label=..., plural_label=...)`
  - `fmt_rate(...)`
  - `fmt_time(..., style="symbol"|"word")`
  - `fmt_qty_range`, `fmt_time_range`, `fmt_count_range`, `fmt_usd_range`
- No QMD LEGO cells were touched in this API batch, so there are no per-cell
  rendered prose entries yet.
- API batch verification:
  - `python3 -m py_compile mlsysim/mlsysim/fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS
  - `git diff --check` PASS
  - `PYTHONPATH=mlsysim python3 -m pytest mlsysim/tests/test_fmt.py book/tests/test_codemod_fmt.py book/tests/test_fmt_prose_contract.py book/tests/test_audit_prose_semantics.py book/tests/test_visible_text.py -q -o addopts=''` PASS, 157 tests
  - `PYTHONPATH=mlsysim python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents` PASS, 0 violations
  - `PYTHONPATH=mlsysim python3 book/tools/audit/fmt/codemod_fmt.py queue --root book/quarto/contents` PASS, `by kind: {}`
  - `PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py --root book/quarto/contents` PASS, 0 findings
