# FMT migration — overnight session resume checkpoint

> **Purpose.** A clean pick-up point after every commit. If a session is
> interrupted, read this top-to-bottom + `MIGRATION.md` and you can resume with
> zero re-discovery. Update the "NOW / NEXT" block **before each commit**.

Worktree: `/Users/VJ/GitHub/MLSysBook-fmt-fix`  ·  branch off `dev`
Always run fmt tooling with `PYTHONPATH=mlsysim` from the repo root.

## Mission (from the user, this session)
1. Migrate dangerous value-kinds into typed/guarded formatters (DONE for
   multiplier + percent + scale-division; see MIGRATION.md).
2. **Verify every migrated LEGO output renders correctly in prose — not just
   glyph-wise but SEMANTICALLY.** (e.g. the robust_ai "26 percent percentage
   points" bug: byte-identical migration preserved a real pre-existing error.)
3. Land on something **consistent across Volume 1 AND Volume 2**.
4. Commit at every milestone; keep this note current so resume is seamless.

## Invariants (never break)
- Every source edit must keep the chapter executing headlessly
  (`assess_equiv.snapshot_file`) and keep `fmt_prose_contract` at 0 violations.
- Value-changing edits (semantic fixes) are allowed but must be deliberate,
  noted, and verified (re-render the value, read the surrounding sentence).
- Pure formatter relocations must stay byte-identical (gate-verified).
- Don't `git commit --amend`, don't force-push, let pre-commit run. The
  `prettify-tables` hook may re-touch tables after staging → `git add -u` and
  re-commit (this is normal, not an error).

## Re-verify in one shot (repo root, PYTHONPATH=mlsysim)
```
python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents   # expect 0
python3 book/tools/audit/fmt/codemod_fmt.py queue --root book/quarto/contents     # remaining dangerous
python3 -m pytest mlsysim/tests/test_fmt.py book/tests/test_codemod_fmt.py book/tests/test_fmt_prose_contract.py book/tests/test_visible_text.py -q -o addopts=''
```

---

## NOW / NEXT  (update before every commit)

**Last commit:** _(this commit)_ semantic scanner + unit-dup prose fixes.

**State:** multiplier + percent 100% migrated; scale-division done (41 sites);
81/81 chapters exec clean; contract 0 violations; semantic scanner 0 findings.

**NEW TOOL:** `audit_prose_semantics.py` — executes each chapter, substitutes
LEGO values into prose, normalizes LaTeX→visible, flags duplicated glyph/unit,
abbr+spelled-word dup, unresolved refs, leaked glyph-commands. Run:
`PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py --root book/quarto/contents`
This is now the third gate alongside fmt_prose_contract and codemod queue.

**NOW done:** Pass 2 — scanner extended with numeric-aware checks
(mult_direction: "0.5× faster" contradiction; currency_as_percent: "$5 percent").
Corpus CLEAN for all checks. Regression test `book/tests/test_audit_prose_semantics.py`
(7 cases) locks each pattern to fire-on-bad / quiet-on-good.

**Pass 3 progress:** corpus grep for §10 anti-patterns found:
- 4 DANGEROUS glyph-in-suffix stragglers the exact-match lanes missed because
  the suffix was *compound* — NOW FIXED (visible-identical, gates clean):
  * ml_systems `mem_bw_growth_pct_str` `"% annually"` → fmt_percent symbol + prose "annually"
  * sustainable_ai `compute_annual_growth_str` `"×/year"` → fmt_multiple + prose `$\times$/year`
  * appendix_reliability ×2 `row.append(fmt(p*100, suffix="%"))` → fmt_percent symbol
- ~17 `suffix=" percentage point(s)"` / `"percentage-point"` sites: belong to
  fmt_pp per the rule, BUT fmt_pp only emits plural-prose ("N percentage points")
  or " pp" — it has NO singular / attributive(hyphen) mode. Migrating blindly
  would change grammar ("5 percentage-point gap" → "5 percentage points gap").
  DECISION PENDING (see NEXT). Low correctness risk (word spelled out, no 100x).

**NEXT:**
1. fmt_pp consistency: inspect each pp-site's prose context; migrate the clean
   plural-noun ones byte-identically; for singular/attributive, either extend
   fmt_pp with a grammatical mode or leave + document. (id: pp-consistency)
2. Pass 4 scale adjudication queue (44 items). 3. Pass 5 render verify (HTML).

Gates to keep green (run all three):
- fmt_prose_contract.py --root book/quarto/contents  → 0
- audit_prose_semantics.py --root book/quarto/contents → 0 findings
- codemod_fmt.py queue --root book/quarto/contents → only known-deferred

## Editorial decisions left for the user (rendered TEXT would change)
These are latent grammar issues in pre-existing prose. I fixed the isolated,
unambiguous ones (model_serving fs_acc_drop_str, fc_acc_loss_str → hyphenated
attributive "N percentage-point loss"). The two below are entangled and need a
human editorial call — left as `fmt(..., suffix=" percentage point")` for now:

1. `benchmarking.qmd` `mv2_acc_drop_str` (renders "0.9 percentage point",
   value 0.9). BOTH prose uses are attributive: "(0.9 percentage point drop)"
   and "(0.9 percentage point drop; below 1 percentage point threshold)".
   RECOMMEND: `fmt_pp(acc_drop, precision=1, attributive=True)` → "0.9
   percentage-point". NOTE: the *hardcoded* sibling "1 percentage point
   threshold" in the same parenthetical should then also be hyphenated to
   "1 percentage-point threshold" for in-sentence consistency.

2. `benchmarking.qmd` `mv2_edge_drop_str` (renders "6.8 percentage point",
   value 6.8). CONFLICT — used as a noun ("edge-case accuracy dropped 6.8
   percentage point[s]") AND attributively in a table cell ("(6.8 percentage
   point drop)"). One `_str` cannot be both. RECOMMEND: make the export the
   plural NOUN `fmt_pp(edge_drop, precision=1)` → "6.8 percentage points" (fixes
   the prose), and reword the table cell from "(… drop)" to "(drop of …)" so the
   noun form reads correctly there too; or split into two exports.

## Session commit log (newest first)
- Pass3d: fixed 2 isolated attributive pp grammar bugs (model_serving fs/fc:
  "N percentage point loss" → "N percentage-point loss", hyphenated). Verified.
- Pass3c: migrated 14 percentage-point sites to fmt_pp BYTE-IDENTICALLY (10 noun,
  4 attributive across 6 chapters: benchmarking, introduction, ml_workflow,
  responsible_engr, model_serving, data_selection). All 14 rendered values proven
  unchanged; contract + semantic + suffix grep clean. responsible_engr got fmt_pp
  added to its 3 using-cells (it uses per-cell selective imports, no star).
  4 pp sites DEFERRED (would change rendered grammar) — see "Editorial decisions".
- Pass3b: extended fmt_pp (singular/plural agreement + attributive hyphen mode) + 6 tests.
- Pass3a: fixed 4 dangerous glyph-in-suffix stragglers (compound suffixes missed
  by exact-match lanes): "% annually", "×/year", 2× row.append("%"). Visible-identical.
- Pass2: numeric semantic checks (mult-direction, currency-as-percent) + 7 regression
  tests; corpus clean.
- semantic scanner tool + fixed "7.6 PB PB"/"PB petabytes" (data_storage ×5) and
  "2.56 MW megawatts" (network_fabrics ×2) unit duplications; scanner CLEAN.
