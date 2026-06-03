# PLAN — Design B (formatter owns `×`) + Rate `_per_s` normalization

**Status:** draft for execution. Two independent migrations, run sequentially, each
proven *render-identical* before moving on. Author: fmt-convention pass.

## Guiding principles (locked by user)

1. **Convention-correctness beats convenience.** If a value's kind changes, its
   suffix changes — no fudged `_str`.
2. **100% coverage.** Every export and every prose ref, both volumes. No sampling.
3. **Zero reader-visible change.** Rendered HTML/PDF output must be visually
   equivalent before vs after (these are *refactors*, not content edits). Raw
   HTML need not be byte-identical when moving a prose glyph from MathJax markup
   to a literal formatter-owned character.
4. **Render is the truth.** Static gates are necessary but not sufficient. Every
   step ends with at least a spot HTML render; the migration is not "done" until
   the FULL HTML render is clean and the FULL PDF (final gatekeeper) is clean.

---

## Locked decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| **D1** | `fmt_multiple` / `fmt_multiple_range` **emit `N×`** (a `MarkdownStr` with a literal rendered glyph, not a `$\\times$` math fragment). | `×` is a constant glyph that belongs to the formatter (like `$`→`fmt_usd`, `%`→`fmt_percent`). Render audit showed Quarto inline substitutions do not reliably re-parse math delimiters returned from Python, so the formatter must return the visible glyph directly. |
| **D2** | Their exports rename to semantic **`*_mult_str`** names. | This matches the book's existing convention: the final suffix `_str` means "safe inline rendered string," while the token before `_str` declares what the string contains (`_usd_str`, `_pct_str`, `_pp_str`, `_ms_str`, now `_mult_str`). We do **not** introduce a new bare `_mult` export family. |
| **D3** | `fmt_multiple` is only for prose multipliers. Any value substituted into display math, or any genuinely bare ratio, moves to a number-only helper such as `fmt_ratio` and a name such as `*_value_str` / `*_ratio_str`. | Manual review showed the earlier "10 bare-ratio" list was overbroad: most were legitimate prose multipliers. The real invariant is context based: `fmt_multiple` may render `N×` only where that whole string belongs in prose. |
| **D4** | Checker **inverts**: delete `mult_missing_glyph` + `mult_literal_x`; add `mult_double_glyph` (flag a `$\times$` *following* a `_mult_str` ref) + `mult_suffix` (a `fmt_multiple` export must be named `*_mult_str`, not generic `*_str`). | After B the bug class flips from "forgot ×" to "added × twice." |
| **D5** | Rate exports use **`<unit>_per_s`**: `gb_per_s`, `tb_per_s`, `mb_per_s`, `gbit_per_s` (bits), `tflop_per_s`, `flop_per_s`, `tokens_per_s`. | One unambiguous form; dodges the bytes(`gb_s`)/bits(`gbps`) blur. |
| **D6 (OPEN — confirm)** | Acronym rates `qps` / `rps` / `tps`: **keep as standard field acronyms** (recommended) vs convert to `queries_per_s` etc. | QPS/RPS/TPS are unambiguous, field-standard, and reader-facing output is owned by `fmt_rate`. Recommend keep; flag for user. |

---

## Verification ladder (run at the marked points)

| Lvl | What | Command | When |
|-----|------|---------|------|
| **L0** | static gates | `./book/binder check math --scope canonical` · `--scope suffix-consistency` · `--scope suffix-semantics` · `--scope multiplier-style` ; `fmt_prose_contract.py` ; `pytest test_fmt_prose_contract.py` | after every edit batch |
| **L1** | render-identical (no build) | `assess_equiv.py baseline --qmd <f> --ref HEAD --out /tmp/<f>.before` → edit → `assess_equiv.py snapshot --qmd <f> --json … --prose …` → `assess_equiv.py diff --before … --after …` → **visible-prose diff MUST be empty** | per touched chapter, each phase |
| **L2** | spot HTML render | `./book/binder build html <2–3 rep chapters>` then scan rendered HTML (below) | end of each phase |
| **L3** | FULL HTML render (ultimate) | `./book/binder build html --all` + `audit_lego_html.py` + scans | end of each migration |
| **L4** | FULL PDF (final gatekeeper) | `./book/binder build pdf --all` (runs pdftotext cross-ref scan) | once, after both migrations |

**Rendered-HTML scan (L2/L3)** — every one of these must return ZERO on the built HTML:
```bash
grep -rn '{python}'  <built-html>      # literal unrendered inline-python
grep -rn 'Traceback\|NameError\|raise ' <built-html>
grep -rn '××\|\$\\times\$'  <built-html>   # double-glyph (B bug) or leaked literal $\times$
python3 book/tools/audit/fmt/audit_lego_html.py --report /tmp/lego_html.json
```
For direct `quarto render` spot checks, set `PYTHONPATH` with absolute worktree
paths, not relative paths. Jupyter may execute from the chapter directory, so a
relative `PYTHONPATH=../..:../../mlsysim` can silently import a stale installed
`mlsysim` and make the HTML audit lie. Use:
```bash
env PYTHONPATH=/Users/VJ/GitHub/MLSysBook-fmt-audit:/Users/VJ/GitHub/MLSysBook-fmt-audit/mlsysim \
  MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/mlsysbook-mplconfig \
  quarto render <chapter.qmd> --to html --output-dir /private/tmp/<out> \
  --execute --execute-daemon-restart --no-cache
```
`audit_lego_html.py` requires the repository's archived `html-audit/` tree; when
that tree is absent, `NO_HTML` is a precondition failure and the applicable L2
gate is the fresh render plus raw HTML scans above.
**Representative chapters** (highest multiplier + rate density): `training` (multipliers),
`network_fabrics` + `compute_infrastructure` (rates `gb_s`/`gbs`/`tbs`), `inference`,
`benchmarking`. Use `training` + `network_fabrics` for L2 spot renders.

---

## PHASE 0 — Baseline & inventory (no edits)

0.1 Freeze the exact work-lists to files (so coverage is provable):
```bash
python3 book/tools/audit/fmt/inventory_design_b_rates.py \
  --root book/quarto/contents \
  --json book/tools/audit/artifacts/fmt_design_b_inventory.json
```
Current frozen scope after removing equation-only ROI values and arithmetic
coefficient uses from the multiplier lane: 433 multiplier exports, 503
multiplier refs, 0 multiplier refs in math context, and 470 rate-name
candidates (308 byte/s, 24 bit/s, 100 compute/s, 9 tokens/s, 29 QPS/RPS/TPS
acronym rates).
0.2 Capture render baselines for the rep chapters (the "before" truth):
```bash
for c in training network_fabrics compute_infrastructure benchmarking inference; do
  assess_equiv.py baseline --qmd <path/$c.qmd> --ref HEAD --out /tmp/base_$c
done
```
0.3 L0 must be green NOW (it is). Record current gate state.

**Exit 0:** inventory files written; baselines captured; L0 green.

---

## PHASE 1 — Design B (`fmt_multiple` owns `×`, multiplier exports become `*_mult_str`)

**1.1 Edge cases first (manual, judgement).** Reassign values that are not prose
multipliers before the formatter starts owning `×`:
- Bare ratio uses stay `fmt_ratio` and `_ratio_str` where prose renders `N:1`
  or another bare quotient rather than `N×`.
- Equation-only multiplier values use `fmt_ratio` and an explicit
  `*_value_str` name, leaving `\times` inside the equation. This precondition is
  important: `fmt_multiple()` will emit `N×` for prose after Design B,
  which must never be substituted inside display math.
- Verify the frozen inventory reports `mult_refs_in_math_context == 0` before
  changing `fmt_multiple()`. Do not treat every variable named `ratio_str` as a
  bare ratio: read the prose. If the prose says "N times", "N× faster",
  "N× reduction", "multiplier", "speedup", or "factor", it belongs in the
  multiplier lane and should become `*_mult_str`.
→ L0 + L1 on touched files.

**1.2 Formatter change (`mlsysim/mlsysim/fmt.py`).** `fmt_multiple` and
`fmt_multiple_range` return `MarkdownStr(f"{number}×")` (range: `f"{lo}–{hi}×"`).
Add/extend unit tests in `mlsysim` asserting the visible `×` is present and the number is
correct. Run `mlsysim` test suite.

**1.3 Codemod (build `codemod_b.py`, dry-run first).** One pass over all chapter `.qmd`:
- **export rename:** `NAME_str = fmt_multiple(…)` → `NAME_mult_str = fmt_multiple(…)`; range names become `*_range_mult_str`.
- **ref transform:** `` `{python} CLASS.NAME_str`$\times$ `` → `` `{python} CLASS.NAME_mult_str` `` (rename + strip the now-duplicate `$\times$`, incl. optional space variants `` `…`$\times$ ``, `` `…` $\times$ ``).
- **only** for names that are fmt_multiple exports (from the 1.0 inventory) — never touch `fmt_int`/etc. refs.
- Dry-run prints counts for export rows, unique source names, duplicate export
  rows, and stripped prose refs. Require all 501 old prose glyphs to be stripped. Export
  rows may exceed source names because a file can assign the same output name in
  more than one branch or cell; after write, the inventory must report
  `mult_exports_without_mult_token == 0`.

**1.4 Checker flip (`fmt_prose_contract.py` + `math_multiplier_style.py` + tests).**
- delete `mult_missing_glyph`, `mult_literal_x`.
- add `mult_double_glyph`: a `_mult_str` ref immediately followed by `$\times$`/`×`/`x` → error.
- add `mult_suffix`: an export built by `fmt_multiple`/`_range` whose name does not include `_mult` before `_str` → error (must be semantic `*_mult_str`).
- update `test_fmt_prose_contract.py`.

**1.5 Docs.** `fmt.md` (§3 catalog row, §6.1 suffix table → add `_mult_str`, §7 prose contract →
"`fmt_multiple` is now closed; prose adds nothing"), `math.md` §1–2 (keep `_str` as the
approved final suffix; document `_mult_str` as the semantic multiplier token),
`numbers-and-math-in-prose.md` (multiplier subsection: drop the "prose adds
`$\times$`" rule, replace with "`{python} x_mult_str`"), `lego-units.md`.

**1.6 VERIFY.**
- L0 (gates) green.
- L1: `assess_equiv diff` per touched chapter → **visible-prose identical** (the rendered
  "4×" is unchanged; only source moved the glyph). Value snapshot will differ (export now
  holds "4×") — that's expected; the *visible* diff is the gate.
- **L2 spot render:** `./book/binder build html training network_fabrics` → run the
  rendered-HTML scan. Confirm e.g. "achieving only 4–6× speedup" renders, no "4–6××", no
  literal `{python}`, no leaked `$\times$`.

**Exit 1:** L0 green; L1 visible-identical on all touched chapters; L2 spot render clean.

---

## PHASE 2 — Rate `_per_s` normalization

**2.1 Codemod (build `codemod_rates.py`, dry-run).** Rename export + all refs:
`_gb_s_str`/`_gbs_str` → `_gb_per_s_str`; `_tb_s_str`/`_tbs_str` → `_tb_per_s_str`;
`_mb_s_str`/`_mbs_str` → `_mb_per_s_str`; `_tflops_str`/`_tflop_s_str` → `_tflop_per_s_str`;
`_tokens_s_str` → `_tokens_per_s_str`. **Leave `_gbps_/_mbps_` (bits) untouched** unless
they should be `_gbit_per_s_` (decide per-site — bits stay bits). Per D6, leave qps/rps/tps.
Dry-run counts must match the frozen inventory after applying the chosen exclusions
(470 total candidates; with QPS/RPS/TPS retained, 441 non-acronym candidates).

**2.2 Output is unchanged by construction** (the formatter owns "GB/s"); still:
- L0 gates green.
- L1: `assess_equiv diff` per touched chapter → **both value AND visible-prose identical**
  (only variable names moved).
- **L2 spot render:** `./book/binder build html network_fabrics compute_infrastructure` →
  rendered-HTML scan; confirm bandwidths still read "3.35 TB/s" etc.

**2.3 Docs.** `fmt.md §6.1` rate-token row already states `_per_s`; confirm + add the
bits/bytes note and the qps/rps/tps carve-out.

**Exit 2:** L0 green; L1 fully identical; L2 spot render clean.

---

## PHASE 3 — Ultimate verification

3.1 **L3 — FULL HTML render, both volumes:**
```bash
./book/binder build html --all
python3 book/tools/audit/fmt/audit_lego_html.py --report /tmp/lego_html_all.json
# + the rendered-HTML scan greps over the entire built site (zero hits required)
```
Inspect: every multiplier renders "N×" (no `××`, no bare "N" where × expected), every
rate renders its unit, zero literal `{python}`, zero tracebacks, zero leaked `$\times$`.

3.2 **L4 — FULL PDF render, both volumes (final gatekeeper):**
```bash
./book/binder build pdf --all        # runs pdftotext cross-ref scan after render
```
Scan the PDF text for `××`, literal `\times`, `{python}`, missing-ref markers.

**Exit 3 (DONE):** full HTML clean + full PDF clean → migration complete.

---

## Rollback / safety

- Each phase is a separate, self-contained set of edits; if L1/L2 fails, revert that
  phase's files (`git checkout -- <files>`) and re-run the codemod after fixing it.
- Codemods are **dry-run-first**; never `--write` until the dry-run count == inventory.
- Nothing is committed until the user reviews; the whole thing lives in the uncommitted
  worktree.

## Effort estimate

- Phase 1: formatter + 10 edge cases + ~939 codemod sites + checker flip + 5 doc files.
- Phase 2: ~430 codemod sites + doc.
- Full HTML build: minutes per volume; full PDF: longer (the slow gatekeeper).
- Bulk is codemod-driven + assess_equiv-gated, so wall-time is dominated by the L3/L4 builds.

## Open item for user

- **D6:** keep `qps`/`rps`/`tps` acronyms, or convert to `queries_per_s` etc.? (Recommend keep.)
