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

## 2026-05-31 — Currency denominator relocation

Change type: byte-identical formatter relocation. Replaced 91
`fmt_usd(..., suffix="/...")` chapter call sites with structured
`fmt_usd(..., per="...")`. This keeps the rendered denominator text unchanged
while moving denominator validation into `fmt_usd`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/data_engineering/data_engineering.qmd` | 26 | 203 | 99 | identical values + prose |
| `vol1/hw_acceleration/hw_acceleration.qmd` | 5 | 255 | 140 | identical values + prose |
| `vol1/ml_ops/ml_ops.qmd` | 8 | 132 | 75 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 15 | 296 | 143 | identical values + prose |
| `vol1/ml_workflow/ml_workflow.qmd` | 8 | 77 | 36 | identical values + prose |
| `vol1/responsible_engr/responsible_engr.qmd` | 3 | 168 | 79 | identical values + prose |
| `vol1/training/training.qmd` | 6 | 453 | 214 | identical values + prose |
| `vol2/distributed_training/distributed_training.qmd` | 1 | 261 | 142 | identical values + prose |
| `vol2/ops_scale/ops_scale.qmd` | 16 | 195 | 95 | identical values + prose |
| `vol2/sustainable_ai/sustainable_ai.qmd` | 3 | 197 | 120 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_currency_per_assess`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` confirms the `rate_denominator` `suffix=` bucket is now
  gone; only the pre-existing single `extra_suffix=` denominator remains.
- Render evidence: deferred to the chapter render sweep; this batch is a pure
  byte-identical relocation with executed value/prose proof.
