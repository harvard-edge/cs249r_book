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

**STATUS: Codex WS4 unit-lane batch complete and verified. Safe to continue.**

**State:** multiplier + percent 100% migrated; scale-division done (41 sites);
pp → typed fmt_pp (14 sites + grammar fixes); 4 dangerous glyph stragglers killed;
NEW semantic scanner gate (+ unit-dup bug fixes). 81/81 chapters exec clean;
prose-contract 0; semantic scanner 0; 124 tests pass. WS4 unit suffixes are now
started: `run_unit_lane.py` migrates clean `fmt(q.m_as(UNIT), suffix=" UNIT")`
sites through the same byte-identical gate as the percent/scale lanes. First
batch migrated 26 Quantity-backed unit sites to `fmt_qty` across:
`vol1/benchmarking` (1), `vol1/conclusion` (5), `vol1/responsible_engr` (17),
`vol2/backmatter/appendix_reliability` (1),
`vol2/collective_communication` (1), and `vol2/introduction` (1).
All six newly changed chapters are HTML-render-verified. Remaining: scale queue
(44, deferred — style call), 4 pp editorial sites (documented below), and the
rest of WS4/WS3. Nothing is half-done or broken.

**If continuing:** A1/A2 still need user decisions. Otherwise continue WS4 with
`PYTHONPATH=mlsysim python3 book/tools/audit/fmt/run_unit_lane.py --write <qmd>`
one chapter at a time. Current dry-run reports 235 remaining clean unit candidates
across 20 chapters; many more suffix sites are plain floats and should stay queued
unless the source is refactored to carry a Pint Quantity.

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

**NOW done:** WS4 first pass — `run_unit_lane.py` added; tests added to
`book/tests/test_codemod_fmt.py`; 26 clean unit sites migrated byte-identically.
`audit_fmt_usage.py` moved `fmt_qty` calls 35 → 61, physical-unit suffixes
1491 → 1476, and time-unit suffixes 659 → 648.

**NEXT:**
1. Continue WS4 with `run_unit_lane.py` chapter-sized batches. Highest remaining
   clean counts: `vol2/compute_infrastructure` (64), `vol1/ml_systems` (28),
   `vol2/distributed_training` (20), `vol2/backmatter/appendix_fleet` (14),
   `vol2/ops_scale` (14).
2. User-decision items: scale house-style (44 queued) and the 4 benchmarking pp
   editorial sites remain deferred.
3. Render-verify any new WS4-changed chapters before Phase 3B/PDF sign-off.

Gates to keep green (run all three):
- fmt_prose_contract.py --root book/quarto/contents  → 0
- audit_prose_semantics.py --root book/quarto/contents → 0 findings
- codemod_fmt.py queue --root book/quarto/contents → only known-deferred

## Render verification (Phase 3A HTML) — DONE for all changed chapters
Built each changed chapter with `./book/binder build html --volN volN/<ch>
--skip-hygiene --skip-validate` (~10-20s each) and grepped the rendered HTML for
the migrated value. ALL PASS:
- ml_systems "improved only ~20% annually"; model_serving "3.2/5 percentage-point
  loss" (hyphen) + "0.33 percentage points"; responsible_engr "15/30/5
  percentage-point" (attributive); benchmarking "±1 percentage point" + "0.9
  percentage points"; introduction "4.8 percentage points"; ml_workflow "0.1/0.15
  percentage points"; data_selection "0.5/4.5/5/4 percentage points".
- data_storage "7.6 PB" x8, ZERO "PB PB"/"PB petabytes"; network_fabrics
  "consume 2.56 MW just to move light"; appendix_reliability % cells render.

Codex WS4 unit batch HTML verification is DONE for the six newly changed
chapters. Built the changed chapters together by volume with Quarto/Deno caches
under `/private/tmp` and grepped rendered HTML:
- `benchmarking`: "700 W" H100 TDP claim.
- `conclusion`: "140 GB", "3.35 TB/s", "989 TFLOP/s", "41.8 ms", "0.14 ms".
- `responsible_engr`: edge deployment power/latency values including "5 W",
  "100 ms", "100 mW", "1.2 W", "50 mW", "200 ms", plus "400 W" and "10 ms".
- `appendix_reliability`: "~7.1 min" recovery replay table cell.
- `collective_communication`: "50 GB/s" InfiniBand bandwidth.
- `introduction` (vol2): "700 GB" GPT-3 synchronization size.

FINDING (sustainable_ai fig-cap): the visible <figcaption> renders `$\times$`
correctly as a math span ("6.2×/year"), but Quarto copies the caption into the
figure `title=` hover-tooltip WITHOUT processing math, so the tooltip shows raw
"6.2\times/year". This is PRE-EXISTING book-wide behavior — the same HTML already
shows "350,000\times" in another figure's title from an untouched caption ref.
Not a regression from this migration; visible output is correct. (If the user
wants tooltips clean, that's a separate book-wide caption-math decision.)

## Scale queue (44 sites) — DEFERRED to a house-style decision (rationale)
The remaining `codemod_fmt queue` items are all `kind=scale`: `fmt(x, suffix="K"/
"M"/"B")` where `x` is ALREADY pre-scaled (e.g. `model_params_b = 7` → "7B";
`_anomaly_k = …m_as(Kparam)` → "70 K"). The auto-migratable form `fmt(x/MILLION,
suffix="M")` was already done (41 sites, prior commit). These 44 are NOT cleanly
migratable because:
  1. The raw count isn't at the call site — `fmt_count(raw, scale=…)` would need
     an ugly `x * BILLION` reconstruction or pint `.m_as("param")` retracing.
  2. ~most carry a SPACE ("70 K", "1.2 M"); `fmt_count` emits no space ("70K",
     "1.2M"), so migrating CHANGES rendered text.
  3. ZERO correctness risk: a scale glyph can't cause a silent 100x error the way
     a percent/multiplier can. This is pure typing/consistency.
DECISION (experienced-eng call): do NOT churn 44 sites + change rendered spacing
overnight for a marginal consistency gain. This needs ONE house-style ruling from
the user: **"scaled counts render as `<n><glyph>` no-space (70B / 5.3M / 12K)"?**
If yes, a follow-up lane migrates all 44 to fmt_count (raw via `.m_as("param")`
or `* FACTOR`) and normalizes the spacing in one reviewable pass. Until then they
remain `fmt(..., suffix=…)` — harmless and rendering correctly today.

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
- Codex WS4 batch: added `run_unit_lane.py` + tests; migrated 26 clean
  Quantity-backed unit suffixes to `fmt_qty` across 6 chapters; contract 0,
  semantic 0, queue still only `{'scale': 44}`, targeted suite 124 passing;
  all 6 changed chapters HTML-render-verified.
- Final: corpus gates green (contract 0, semantic 0, 81/81 exec, 119 tests); all
  changed chapters HTML-render-verified; MIGRATION.md + NIGHT_RESUME updated.
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
