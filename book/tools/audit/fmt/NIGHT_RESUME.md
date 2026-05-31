# FMT migration — overnight session resume checkpoint

> **Purpose.** A clean pick-up point after every commit. If a session is
> interrupted, read this top-to-bottom + `MIGRATION.md` and you can resume with
> zero re-discovery. Update the "NOW / NEXT" block **before each commit**.

Worktree: `/Users/VJ/GitHub/MLSysBook-fmt-fix`  ·  branch off `dev`
Always run fmt tooling with `PYTHONPATH=mlsysim` from the repo root.

## Mission (from the user, this session)
1. Migrate dangerous value-kinds into typed/guarded formatters (DONE for
   multiplier + percent + scale; see MIGRATION.md).
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
python3 book/tools/audit/fmt/codemod_fmt.py queue --root book/quarto/contents     # expect empty
python3 -m pytest mlsysim/tests/test_fmt.py book/tests/test_codemod_fmt.py book/tests/test_fmt_prose_contract.py book/tests/test_audit_prose_semantics.py book/tests/test_visible_text.py -q -o addopts=''
```

---

## NOW / NEXT  (update before every commit)

**STATUS: Currency formatter suffix migration complete and verified. Safe to continue with remaining migration lanes.**

**State:** multiplier + percent + scale 100% migrated; pp → typed fmt_pp (14
byte-identical sites + grammar fixes, plus the user-approved A2 benchmarking
edits); 4 dangerous glyph stragglers killed; NEW semantic scanner gate (+ unit-dup
bug fixes). 81/81 chapters exec clean; prose-contract 0; semantic scanner 0;
codemod queue empty; 157 focused tests pass. User ruled for no-space scaled
counts, so `run_scale_style_lane.py` migrated the 44 queued scale sites to
`fmt_count` and one manual `fmt(...) + "B"` blind spot. User also approved A2, so
`benchmarking` now renders `0.9 percentage-point drop`, `below 1
percentage-point threshold`, and `drop of 6.8 percentage points`. WS4 unit
suffixes are also started: `run_unit_lane.py` migrates clean
`fmt(q.m_as(UNIT), suffix=" UNIT")` sites through the same byte-identical gate as
the percent/scale lanes. First batch migrated 26 Quantity-backed unit sites to
`fmt_qty` across:
`vol1/benchmarking` (1), `vol1/conclusion` (5), `vol1/responsible_engr` (17),
`vol2/backmatter/appendix_reliability` (1),
`vol2/collective_communication` (1), and `vol2/introduction` (1).
Second batch migrated 60 clean Quantity-backed unit sites in
`vol2/compute_infrastructure`; 4 candidates were correctly queued because
canonical Pint labels would change visible text (`MB`→`megabyte`, `GB`→`GiB`).
`fmt_qty` now rejects plain numbers, so callers must keep Pint Quantity type
information until the formatter can dimension-check. Raw `prefix=` use in QMD
formatter calls is now eliminated: `fmt`/`fmt_int`/`fmt_qty`/`fmt_count` accept
named `approx=True` and `lower_bound=True` marker flags, and the 16 remaining
chapter-level prefix sites were migrated byte-identically. All WS4-changed
chapters are HTML-render-verified. WS4 continued with `vol1/ml_systems`, adding
28 more byte-identical `fmt_qty` migrations and moving corpus `fmt_qty` calls
to 149 while reducing physical-unit suffix calls to 1,385. WS4 then continued
with `vol2/distributed_training`, migrating 15 of 20 clean candidates
byte-identically; 5 were correctly queued because canonical Pint labels would
change visible text (`80 GB`→`80 GiB`). Corpus `fmt_qty` calls are now 164 and
physical-unit suffix calls are now 1,370. The A1 scale-style pass is
also HTML-render-verified for all 13 source-changed chapters. A2 is
HTML-render-verified for `benchmarking`. Remaining: the rest of WS4/WS3 and later
PDF/lock phases. Nothing is half-done or broken.

**If continuing:** Continue with semantic lanes in `PLAN_OF_RECORD.md`. Good next
targets are currency scale/per oddities, count labels/rates, then time values.
For physical Quantity-backed sites, continue WS4 with
`PYTHONPATH=mlsysim python3 book/tools/audit/fmt/run_unit_lane.py --write <qmd>`
one chapter at a time. Current dry-run reports 132 remaining clean unit candidates
across 19 chapters; many more suffix sites are plain floats and should stay queued
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
- `fmt_pp` now has singular/plural and attributive-hyphen modes. The user
  approved the remaining benchmarking grammar sites, so no pp editorial decision
  remains open.

**NOW done:** WS4 first pass — `run_unit_lane.py` added; tests added to
`book/tests/test_codemod_fmt.py`; 26 clean unit sites migrated byte-identically.
`audit_fmt_usage.py` moved `fmt_qty` calls 35 → 61, physical-unit suffixes
1491 → 1476, and time-unit suffixes 659 → 648.

**NOW done:** WS4 compute-infrastructure batch — migrated 60 clean
Quantity-backed unit sites byte-identically. `audit_fmt_usage.py` moved
`fmt_qty` calls 61 → 121 and physical-unit suffixes 1473 → 1413. Four
`compute_infrastructure` candidates remain queued for visible unit-label drift:
`50 MB` would become `50 megabyte`, and `80/640 GB` would become `80/640 GiB`.

**NOW done:** marker-prefix cleanup — added named `approx=True` and
`lower_bound=True` display markers to `fmt`, `fmt_int`, `fmt_qty`, and
`fmt_count`; migrated all 16 QMD formatter calls that used raw `prefix="~"` or
`prefix="> "`; verified no QMD `prefix=` call sites remain. This was a
byte-identical API cleanup: `assess_equiv` reported identical values and
identical inline prose for all 8 changed chapters, and HTML builds were grepped
for the marker-bearing values.

**NOW done:** WS4 `vol1/ml_systems` batch — migrated 28 clean Quantity-backed
unit suffix sites to `fmt_qty`, byte-identical by `run_unit_lane.py`. Corpus
audit moved `fmt_qty` calls 121 → 149 and physical-unit suffixes 1,413 → 1,385.
HTML build for `ml_systems` succeeded and representative migrated values were
grepped in the output.

**NOW done:** WS4 `vol2/distributed_training` partial batch — migrated 15 clean
Quantity-backed unit suffix sites to `fmt_qty`, byte-identical by
`run_unit_lane.py`. Five memory-capacity sites remain queued for visible
unit-label drift (`80 GB` would become `80 GiB`). During HTML verification,
fixed a pre-existing rendered prose bug, `600 GB/s+ GB/s NVLink domain`, to
`NVLink domain (600 GB/s or higher)`, and added a semantic-scanner regression
for this `unit+ unit` shape. Corpus audit moved `fmt_qty` calls 149 → 164 and
physical-unit suffixes 1,385 → 1,370.

**NOW done:** Plan-of-record + structured formatter API batch — added
`PLAN_OF_RECORD.md` and `AUDIT_LEDGER.md` so future work records both per-cell
validation and the final whole-book inline render/prose audit. Finished the
structured API layer: `fmt_usd(scale=..., per=...)`, `fmt_count(label=...,
plural_label=...)`, `fmt_rate`, `fmt_time(style="symbol"|"word")`, and typed
range helpers. No QMD LEGO cells were touched in this batch. Verification:
py_compile PASS; `git diff --check` PASS; focused pytest suite PASS
(157 tests); `fmt_prose_contract` 0; `codemod_fmt queue` empty; substituted
prose semantic audit CLEAN across 81 files.

**NOW done:** Currency denominator relocation — migrated 91 chapter call sites
from `fmt_usd(..., suffix="/...")` to structured `fmt_usd(..., per="...")`
across 10 chapters. This removed the `rate_denominator` `suffix=` bucket from
`audit_fmt_usage.py` while keeping rendered values and substituted prose
byte-identical by `assess_equiv` for every touched chapter. Verification:
`git diff --check` PASS; focused pytest suite PASS (157 tests);
`fmt_prose_contract` 0; `codemod_fmt queue` empty; substituted prose semantic
audit CLEAN across 81 files.

**NOW done:** Currency scale/range relocation — migrated the remaining 78 QMD
`fmt_usd(..., suffix=...)` call sites to structured `scale=`, `per=`,
checked `marker="*"`, or `fmt_usd_range(...)`. QMD now has zero
`fmt_usd(..., suffix=...)` calls. `assess_equiv` proved every scale/marker
relocation byte-identical; the only intentional diff is the H100 table price
range normalizing `~\$25,000-30,000` to `~\$25,000–30,000` through
`fmt_usd_range(..., repeat_symbol=False)`. Verification: py_compile PASS;
`git diff --check` PASS; `./book/binder check math` PASS; focused pytest suite
PASS (161 tests); `fmt_prose_contract` 0; `codemod_fmt queue` empty;
substituted prose semantic audit CLEAN across 81 files.

**NEXT:**
1. Continue the semantic non-Quantity lanes: count labels, non-physical rates,
   and time values. Add entries to `AUDIT_LEDGER.md` for every touched LEGO
   cell.
2. Continue WS4 with `run_unit_lane.py` chapter-sized batches. Highest remaining
   clean counts: `vol2/backmatter/appendix_fleet` (14), `vol2/ops_scale` (14),
   `vol1/hw_acceleration` (11), `vol1/introduction` (11),
   `vol2/network_fabrics` (11), `vol2/performance_engineering` (11).
3. Render-verify any newly changed chapters before Phase 3B/PDF sign-off.

Gates to keep green (run all three):
- fmt_prose_contract.py --root book/quarto/contents  → 0
- audit_prose_semantics.py --root book/quarto/contents → 0 findings
- codemod_fmt.py queue --root book/quarto/contents → by kind: {}

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

Codex A1 scale-style HTML verification is DONE for all 13 source-changed
chapters. Built the changed chapters by volume with Quarto/Deno caches under
`/private/tmp` and grepped representative no-space scale strings:
- Vol1: appendix_assumptions `7B`; benchmarking `270K`, `3.5M`, `25.6M`, `7B`;
  data_selection `1M`, `100K`, `10K`, `500K`; frameworks `25.6M`;
  ml_systems `612K`; nn_architectures `20M`, `421.4K`; nn_computation `100M`;
  responsible_engr `3.5M`, `5.3M`, `25.6M`, `270K`; training `7B`, `1.5B`,
  `100B`, `175B`, `70B`, `20B`.
- Vol2: appendix_assumptions `70B`; appendix_fleet `7B`, `70B`, `175B`;
  distributed_training `10K`; fleet_orchestration `7B`, `70B`, `175B`.

Codex A2 pp editorial HTML verification is DONE for `benchmarking`. Built with
Quarto/Deno caches under `/private/tmp` and grepped:
- `0.9 percentage-point drop`
- `below 1 percentage-point threshold`
- `drop of 6.8 percentage points`

Codex WS4 compute-infrastructure HTML verification is DONE. Built
`compute_infrastructure` with Quarto/Deno caches under `/private/tmp` and grepped
representative migrated values:
- `125 TFLOP/s`, `300 W`, `300 GB/s`, `900 GB/s`
- `3.35 TB/s`, `700 W`, `1000 W`

Codex marker-prefix cleanup HTML verification is DONE for all 8 source-changed
chapters. Built the changed chapters by volume and grepped representative marker
strings:
- Vol1: appendix_algorithm `~39.1 GB` and `~120.8 GB`;
  appendix_assumptions `~25 wall-clock days`; appendix_data `~100 MB/s`,
  `~300 MB/s`, and `> 1,000 MB/s`; model_compression `~107 GB`, `~8 GB`, and
  `~524.3 KB`; model_serving `~1.5 s` and `~35 s`;
  responsible_engr `~99 tons`.
- Vol2: appendix_assumptions `~3.9 failures per day` and `~48 failures per day`;
  data_storage build succeeded and semantic/equivalence gates covered the
  migrated `archive_lineage_tb_str`.

Codex WS4 `ml_systems` HTML verification is DONE. Built `ml_systems` with
Quarto/Deno caches under `/private/tmp` and grepped representative migrated
values:
- `1 MB`, `100 mJ/MB`, `0.1 mJ/inference`, `102.4 MB`, `51.2 MB`, `25.6 MB`
- `312 TFLOP/s`, `2.04 TB/s`, `35 TOPS`, `100 GB/s`
- `131 TB`, `128 GB`, `4 MB`, `186.6 MB/s`, `18.7 GB/s`, `1.25 GB/s`, `1 KB`
- `100 TOPS`, `15 W`, `1 TOPS`, `2 W`, `15 Wh`

Codex WS4 `distributed_training` HTML verification is DONE. Built
`distributed_training` with Quarto/Deno caches under `/private/tmp` and grepped
representative migrated values:
- `600 GB/s`, `3 GB`, `25 GB/s`, `900 GB/s`, `50 GB/s`, `100 GB/s`
- `1979 TFLOP/s`
- Confirmed the previous `600 GB/s+ GB/s` rendered string is gone and the
  corrected prose renders as `NVLink domain (600 GB/s or higher)`.

FINDING (sustainable_ai fig-cap): the visible <figcaption> renders `$\times$`
correctly as a math span ("6.2×/year"), but Quarto copies the caption into the
figure `title=` hover-tooltip WITHOUT processing math, so the tooltip shows raw
"6.2\times/year". This is PRE-EXISTING book-wide behavior — the same HTML already
shows "350,000\times" in another figure's title from an untouched caption ref.
Not a regression from this migration; visible output is correct. (If the user
wants tooltips clean, that's a separate book-wide caption-math decision.)

## Scale queue (44 sites) — RESOLVED
The user ruled for no-space scaled counts: `<n><glyph>` (`70B`, `5.3M`, `12K`).
`run_scale_style_lane.py` migrated all 44 queued scale sites to `fmt_count(raw,
scale=...)`, reconstructing raw counts with Pint `.m_as("param")`, `* THOUSAND`,
`* MILLION`, or `* BILLION` as appropriate. One manual blind spot,
`fmt(...) + "B"` in `fleet_orchestration`, was also migrated to `fmt_count`.

This pass intentionally changes spacing/case only: examples include `70 B`→`70B`,
`270 K`→`270K`, and `100k`→`100K`. The `codemod_fmt.py queue` gate is now empty.

## Editorial decisions — RESOLVED
The user approved A2. `benchmarking.qmd` now uses
`fmt_pp(acc_drop, precision=1, attributive=True)` for the attributive top-1
accuracy drop and noun-form `fmt_pp(edge_drop, precision=1)` for the edge-case
drop. The adjacent prose/table text was updated to `1 percentage-point threshold`
and `(drop of 6.8 percentage points)`. No user-decision items remain open.

## Session commit log (newest first)
- Codex WS4 `distributed_training` partial batch: migrated 15 clean
  Quantity-backed unit suffix sites to `fmt_qty`; 5 `80 GB`→`80 GiB` candidates
  left queued; fixed pre-existing `600 GB/s+ GB/s` prose; extended semantic
  scanner to catch `unit+ unit`; contract 0, semantic 0, queue empty, targeted
  suite 134 passing; `distributed_training` HTML-render-verified.
- Codex WS4 `ml_systems` batch: migrated 28 clean Quantity-backed unit suffix
  sites to `fmt_qty`; `run_unit_lane.py` proved output byte-identical; contract
  0, semantic 0, queue empty, targeted suite 134 passing; `ml_systems`
  HTML-render-verified.
- Codex marker-prefix cleanup: added named marker flags (`approx=True`,
  `lower_bound=True`) to the formatter core and quantity/count wrappers; migrated
  the 16 remaining QMD raw-prefix formatter calls byte-identically; no QMD
  `prefix=` call sites remain; `assess_equiv` values/prose identical for all 8
  changed chapters; HTML-render-verified marker values; contract 0, semantic 0,
  queue empty, targeted suite 134 passing.
- Codex fmt_qty guard + WS4 compute-infrastructure batch: `fmt_qty` now requires
  Pint Quantity input; migrated 60 clean `compute_infrastructure` unit suffixes
  to `fmt_qty`; contract 0, semantic 0, queue empty, targeted suite 130 passing;
  compute_infrastructure HTML-render-verified. Four local unit candidates remain
  queued for intentional visible unit-label drift (`MB`/`GB` vs Pint labels).
- Codex A2 pp editorial pass: user approved the benchmarking wording; migrated
  `mv2_acc_drop_str` and `mv2_edge_drop_str` to `fmt_pp`, hyphenated the hardcoded
  threshold, reworded the edge-case table cell, and HTML-render-verified
  `benchmarking`; contract 0, semantic 0, queue empty, targeted suite 129 passing.
- Codex A1 scale-style pass: user chose no-space scaled counts; added
  `run_scale_style_lane.py` + tests; migrated 44 queued scale suffix sites and
  one `fmt(...) + "B"` blind spot to `fmt_count`; contract 0, semantic 0, queue
  empty, targeted suite 129 passing; all 13 source-changed chapters
  HTML-render-verified.
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
