# PLAN — Design B (formatter owns `×`) + Rate `_per_s` normalization

**Status:** draft for execution. Two independent migrations, run sequentially, each
proven *render-identical* before moving on. Author: fmt-convention pass.

## Guiding principles (locked by user)

1. **Convention-correctness beats convenience.** If a value's kind changes, its
   suffix changes — no fudged `_str`.
2. **100% coverage.** Every export and every prose ref, both volumes. No sampling.
3. **Zero reader-visible change.** Rendered HTML/PDF output is byte-identical
   before vs after (these are *refactors*, not content edits).
4. **Render is the truth.** Static gates are necessary but not sufficient. Every
   step ends with at least a spot HTML render; the migration is not "done" until
   the FULL HTML render is clean and the FULL PDF (final gatekeeper) is clean.

---

## Locked decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| **D1** | `fmt_multiple` / `fmt_multiple_range` **emit `N$\times$`** (a `MarkdownStr`). | `×` is a constant glyph that belongs to the formatter (like `$`→`fmt_usd`, `%`→`fmt_percent`). |
| **D2** | Their exports rename to semantic **`*_mult_str`** names. | This matches the book's existing convention: the final suffix `_str` means "safe inline rendered string," while the token before `_str` declares what the string contains (`_usd_str`, `_pct_str`, `_pp_str`, `_ms_str`, now `_mult_str`). We do **not** introduce a new bare `_mult` export family. |
| **D3** | The **10 bare-ratio mis-uses** of `fmt_multiple` move to **`fmt_ratio`** (stay `_str`). | They render "5:1 ratio" / bare — they are ratios, not multipliers. Prerequisite so `fmt_multiple` genuinely always means "×". |
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
**Representative chapters** (highest multiplier + rate density): `training` (multipliers),
`network_fabrics` + `compute_infrastructure` (rates `gb_s`/`gbs`/`tbs`), `inference`,
`benchmarking`. Use `training` + `network_fabrics` for L2 spot renders.

---

## PHASE 0 — Baseline & inventory (no edits)

0.1 Freeze the exact work-lists to files (so coverage is provable):
```bash
python3 /tmp/scope_b.py > book/tools/audit/fmt/_b_inventory.txt   # 436 exports, 503 refs, 10 edge, 444 rates
```
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

**1.1 Edge cases first (manual, judgement).** Reassign the 10 bare-ratio uses to
`fmt_ratio` (rename export to `_ratio_str` where the name still says `ratio`, else keep
name + swap helper). Verify each in prose context. Sites:
`responsible_engr:1762`, `ops_scale:1381`, `fleet_orchestration:1491`,
`data_selection:4622`, `data_engineering:2992`, `appendix_assumptions:448`,
`collective_communication:191`, `training:2649`, `ml_ops:2882`, `ml_ops:3543`.
→ L0 + L1 on these files.

**1.2 Formatter change (`mlsysim/mlsysim/fmt.py`).** `fmt_multiple` and
`fmt_multiple_range` return `MarkdownStr(f"{number}$\\times$")` (range: `f"{lo}–{hi}$\\times$"`).
Add/extend unit tests in `mlsysim` asserting the `$\times$` is present and the number is
correct. Run `mlsysim` test suite.

**1.3 Codemod (build `codemod_b.py`, dry-run first).** One pass over all chapter `.qmd`:
- **export rename:** `NAME_str = fmt_multiple(…)` → `NAME_mult_str = fmt_multiple(…)` (and range names become semantic `*_mult_range_str` / `*_range_mult_str` where that reads better).
- **ref transform:** `` `{python} CLASS.NAME_str`$\times$ `` → `` `{python} CLASS.NAME_mult_str` `` (rename + strip the now-duplicate `$\times$`, incl. optional space variants `` `…`$\times$ ``, `` `…` $\times$ ``).
- **only** for names that are fmt_multiple exports (from the 1.0 inventory) — never touch `fmt_int`/etc. refs.
- Dry-run prints a diff + counts; require counts == inventory (436 / 503) before `--write`.

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
  holds "4$\times$") — that's expected; the *visible* diff is the gate.
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
Dry-run counts must == inventory (444 minus the qps/rps/tps + bits exclusions).

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
