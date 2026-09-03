# LEGO closed-export ↔ prose audit (2026-06-06)

Explicit manual + script review of whether prose correctly uses closed LEGO
`*_str` exports without repeating units (FLOP/byte, TFLOP/s, g/kWh, GB, …).

## Checks reviewed

| Check | Scope | Gap found | Action |
|-------|--------|-----------|--------|
| `book_check_lego_prose_units.py` | Domain units immediately after `` `{python} *_str` `` | **Missing** `FLOP/byte`, `g/kWh`, `km/h`, `GFLOP/s` in token regex | **Fixed** — expanded `PROSE_UNIT_AFTER_REF` + `_FMT_UNITS` |
| `fmt/fmt_prose_contract.py` | `%`, `$`, scale, `×` after typed exports | Correct for its lane; **does not** cover domain formatters | Documented complement; keep both |
| `lint_lego_units.py` | Closed-name / open-`fmt()` mismatches | Separate concern (naming), not prose duplication | No change |
| Immediate-token grep (ad hoc) | All `_str` + next token | ~374 hits; mostly `$` LaTeX delimiters or **open** exports | Used for triage only |

## Explicit corpus pass (AST, independent of regex checker)

Built formatter map from LEGO cells (`fmt_prose_contract.build_formatter_records`),
classified closed vs open exports, scanned 48 chars after each `` `{python} *_str` `` ref.

**Result:** 1 real violation — `sustainable_ai.qmd` L1614 duplicate `g/kWh` after
`fmt_carbon_intensity` export. **Fixed.**

**Open-export FLOP/byte (valid):** ~20 hits in `hw_acceleration.qmd` and
`appendix_machine.qmd` where `*_ai_str` / `ridge_*_str` use bare `fmt()` — prose
*must* supply `FLOP/byte`. Regex checker correctly skips these (not in closed map).

**Closed-export FLOP/byte (invalid):** None remaining after P6 migration.

## Script design notes (for maintainers)

1. **Closed map** comes from OUTPUT assignments only — open `fmt()` scalars never enter the map.
2. **Immediate adjacency** — checker flags `ref` + token with only whitespace between.
   Duplicates inside the same sentence but separated by words (e.g. “… `{python} x_str` … later g/kWh”) are **not** caught; extend window if that pattern appears.
3. **`fmt_prose_contract`** owns percent/USD/mult; **`lego-prose-units`** owns SI/domain glyphs.
4. Run both before push:
   ```bash
   python3 book/tools/audit/book_check_lego_prose_units.py
   python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents
   pytest book/tests/test_lego_prose_units.py book/tests/test_fmt_prose_contract.py -q
   ```

## Verdict

| Gate | Status |
|------|--------|
| `lego-prose-units` (enhanced) | **PASS** (80 QMD) |
| `fmt_prose_contract` | **PASS** (80 QMD) |
| Explicit AST spot-check | **PASS** after sustainable_ai fix |
