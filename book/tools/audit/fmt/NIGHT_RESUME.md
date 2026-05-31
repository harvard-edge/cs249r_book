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

**STATUS: Compound-scale suffix migration complete; remaining suffix work is WS4 physical units, MarkdownStr review, render/PDF audit, API cleanup, and final lock.**

**State:** multiplier + percent + scale 100% migrated; pp → typed fmt_pp (14
byte-identical sites + grammar fixes, plus the user-approved A2 benchmarking
edits); 4 dangerous glyph stragglers killed; NEW semantic scanner gate (+ unit-dup
bug fixes). 81/81 chapters exec clean; prose-contract 0; semantic scanner 0;
codemod queue empty; 190 focused tests pass. User ruled for no-space scaled
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
HTML-render-verified for `benchmarking`. The time-unit lane is now complete:
`run_time_lane.py` migrated the remaining 522 exact `time_unit` suffix sites to
`fmt_time(...)`, QMD `fmt_time` calls now use full unit-name strings in source,
and `audit_fmt_usage.py` reports no remaining `time_unit` suffix bucket.
The exact scale-word lane is also complete, and the later compound-scale lane
resolved the API spelling: compact scales use `scale="B"` / `scale="M"` while
word scales use `scale="billion"` / `scale="million"`. The older
`scale_style="word"` form remains compatible but should not be added in new
QMD edits.
Residual plain count labels are also reduced: 17 suffixes such as `errors`,
`steps`, `photos`, `requests`, `servers`, `workers`, `stages`, `V100s`, and
`link tiers` now use `fmt_count(..., label=...)`, byte-identical across 9
chapters.
The time codemod now also recognizes word-form `microseconds` and
`milliseconds`; the three remaining exact word-form time suffixes moved to
`fmt_time(..., style="word")` byte-identically.
`audit_fmt_usage.py` split the remaining suffix inventory into actionable
sub-buckets: physical units (1,126), resource-time labels (19), unit
rates/denominators (16), compound scale (14), operation counts (12), and
time compounds (8). One more plain count label (`epochs`) was moved to
`fmt_count(..., plural_label="epochs")`.
Four denominator-style time compounds (`s/hr`, `hours/day`, `μs/op`, `ms/step`)
now use `fmt_time(..., per=...)` byte-identically, leaving only four
prose-adjective/lower-bound time compounds (`ms latency`, `ms round-trip`,
`ms+`) for manual wording/API decisions.
Straightforward resource-time labels are now also migrated: `PFLOP-days`,
`TPUv4-hours`, `person-years`, `instance-seconds`, `GPU-hours`, and `GPU-hr`
use `fmt_count(..., label=...)` byte-identically across 7 chapters. The
hyphenated `-hour`/`-minute` forms now use `fmt_time(..., style="word",
attributive=True)` byte-identically, so the `resource_time` suffix bucket is
gone.
Exact FLOP-count suffixes are also migrated: 12 `GFLOPs`/`MFLOPs`/`KFLOPs`/
`PFLOPs` sites now use `fmt_qty(...)` with Pint FLOP units, byte-identical
across 5 chapters. No separate `fmt_ops` wrapper was added because Pint already
provides the unit check; word-scale FLOP phrases (`billion FLOPs`, `trillion
FLOPs`) now use `fmt_qty(..., unit_label=...)` so visible wording stays intact.
The four `time_compound` suffixes are gone too: `ms latency` / `ms round-trip`
now keep the unit in `fmt_time(...)`, and `ms+` uses checked
`fmt_time(..., marker="+")`. The 16 `unit_rate_or_denominator` suffixes are now
also migrated through `fmt_qty(...)`, using checked `unit_label=` where Pint's
compact label did not match visible house style. Compound-scale suffixes are now
gone too: direct word scales, checked scaled rates, attributive count modifiers,
and `fmt_qty(unit_label=...)` cleared all 14 sites. Intentional visible fixes:
`200 K parameters`→`200K parameters` and floored `32K tokens`→`32.8K tokens`
for the exact 32,768-token batch.
The appendix bandwidth denominator lane then moved 16 split `GB`/`TB` + prose
`/s` values to full `fmt_qty(..., GB/second|TB/second)` strings while keeping
substituted prose byte-identical. The network `Gb/s` lane then migrated all 20
`suffix=" Gb/s"` sites to `fmt_qty(..., Gbps, unit_label="Gb/s")`
byte-identically across seven chapters. The GiB-backed memory-capacity lane then
migrated the remaining obvious `m_as(GiB)` + `suffix=" GB"` specs to
`fmt_qty(..., GiB, unit_label="GB")` across seven chapters. The direct Quantity
physical-unit lane then added `fmt_qty_int(...)` for intentionally rounded
Quantity displays and migrated 29 more sites byte-identically. The
`vol1/ml_systems` lane then migrated all 30 remaining physical-unit suffixes in
that chapter to typed quantity formatters, byte-identically. The
`vol2/backmatter/appendix_fleet` chapter lane then migrated all 28 remaining
physical-unit suffixes in that chapter, byte-identically, leaving
`appendix_fleet` with zero `suffix=` calls. The
`vol2/collective_communication` chapter lane then migrated all 26 remaining
physical-unit suffixes in that chapter, byte-identically, leaving
`collective_communication` with zero `suffix=` calls. The
`vol2/fault_tolerance` chapter lane then migrated all 10 remaining
physical-unit suffixes in that chapter, byte-identically, leaving
`fault_tolerance` with zero `suffix=` calls. The
`vol2/backmatter/appendix_communication` chapter lane then migrated all 9
remaining physical-unit suffixes in that chapter, byte-identically, leaving
`appendix_communication` with zero `suffix=` calls. The tiny Vol2 physical-unit
tail then migrated 7 sites across `appendix_c3`, `robust_ai`, `conclusion`, and
`appendix_reliability`, byte-identically, leaving all four files with zero
`suffix=` calls. The small chapter physical-unit cleanup then migrated 17 sites
across `ml_ops`, `appendix_data`, and `security_privacy`, byte-identically,
leaving all three files with zero `suffix=` calls. The assumptions appendix
physical-unit cleanup then migrated 10 sites across the Vol1 and Vol2
`appendix_assumptions` files, byte-identically, leaving both files with zero
`suffix=` calls. The Vol1 introduction/data-selection physical-unit cleanup then
migrated 8 sites across `introduction` and `data_selection`, byte-identically,
leaving both files with zero `suffix=` calls. The `ml_workflow` physical-unit
cleanup then migrated all 8 remaining sites in that chapter, byte-identically,
leaving the file with zero `suffix=` calls. The `vol2/introduction`
physical-unit cleanup then migrated all 8 remaining sites in that chapter,
byte-identically, leaving the file with zero `suffix=` calls. Remaining suffix
bucket is only 881 `physical_unit` suffixes. The `appendix_algorithm`
physical-unit cleanup then migrated all 11 remaining sites in that appendix,
byte-identically, leaving the file with zero `suffix=` calls. Remaining suffix
bucket is only 870 `physical_unit` suffixes. Remaining: the rest of WS4/WS3 and
later PDF/lock phases. Nothing is half-done or broken.

**New TODOs from user discussion:** add a prose-bound output contract/gate so
computed OUTPUT values consumed by inline prose are typed formatter results or
intentional `MarkdownStr`; add a stock unit display-label registry so common
units like `Gbps` render as `Gb/s` without repeating `unit_label="Gb/s"` at every
call site; design hardware/model display accessors so hardware specs can print
their own canonical display units for memory capacity, bandwidth, TDP, and peak
FLOP/s without QMD call sites guessing the stored unit; handle split-rate prose
contracts such as `100 GB` plus a literal `/s`, where the value can be checked
as a rate today but the rendered unit is still shared between formatter output
and prose.

**If continuing:** Continue with semantic lanes in `PLAN_OF_RECORD.md`. Good next
targets are physical-unit suffixes, MarkdownStr sites, and the API-cleanup pass.
For physical Quantity-backed sites, continue WS4 with
`PYTHONPATH=mlsysim python3 book/tools/audit/fmt/run_unit_lane.py --write <qmd>`
one chapter at a time. The latest all-chapter run migrated the remaining
byte-identical clean candidates; the follow-up write run confirmed the remaining
20 Quantity-backed candidates all fail the byte-identical gate because they
would visibly change output (`GB`→`GiB`, `TB`→`TB/s`, `GB`→`GB/s`).
Many more suffix sites are plain floats and should stay queued unless the source
is refactored to carry a Pint Quantity.

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

**NOW done:** QPS rate relocation — migrated 26 `suffix=" QPS"` sites to
`fmt_rate(..., "QPS")`. `fmt_rate` now defaults to `commas=True`, preserving old
`fmt`/`fmt_int` output unless the site explicitly requests `commas=False`.
`assess_equiv` proved byte-identical values and substituted prose across all 5
touched chapters. Verification: py_compile PASS; `git diff --check` PASS;
`./book/binder check math` PASS; focused pytest suite PASS (161 tests);
`fmt_prose_contract` 0; `codemod_fmt queue` empty; substituted prose semantic
audit CLEAN across 81 files.

**NOW done:** GPU count labels — migrated 77 `suffix=" GPUs"` sites to
`fmt_count(..., label="GPU")`. The one singular `suffix=" GPU"` site remains
intentionally because it renders `GPU-days`, not a standalone count noun.
`assess_equiv` proved byte-identical values and substituted prose across all 20
touched chapters. `audit_fmt_usage.py` now reports `fmt_count` calls at 163 and
the `count_label` suffix bucket down to 41. Verification: py_compile PASS;
`git diff --check` PASS; `./book/binder check math` PASS; focused pytest suite
PASS (161 tests); `fmt_prose_contract` 0; `codemod_fmt queue` empty;
substituted prose semantic audit CLEAN across 81 files.

**NOW done:** Remaining direct count labels — migrated 40
`tokens`/`nodes`/`layers`/`queries`/`images` suffix sites to
`fmt_count(..., label=...)`, byte-identical across all 15 touched chapters. Old
`fmt_int` query sites now round explicitly before `fmt_count`, preserving output
while making the integer boundary visible. The only remaining `count_label`
suffix is the documented `GPU-days` compound. Verification: py_compile PASS;
`git diff --check` PASS; `./book/binder check math` PASS; focused pytest suite
PASS (161 tests); `fmt_prose_contract` 0; `codemod_fmt queue` empty;
substituted prose semantic audit CLEAN across 81 files.

**NOW done:** Benchmarking millisecond time suffixes — migrated 25
`suffix=" ms"` sites in `vol1/benchmarking/benchmarking.qmd` to
`fmt_time(..., "ms")`, byte-identical by `assess_equiv` across 240 value exports
and 102 substituted prose lines. Explicit `commas=False` was preserved for this
migration; redundant comma arguments are deferred to the formatter-default
cleanup pass. `audit_fmt_usage.py` now reports 25 `fmt_time` calls and the
`time_unit` suffix bucket down to 623. Verification: py_compile PASS;
`git diff --check` PASS; `./book/binder check math` PASS; focused pytest suite
PASS (161 tests); `fmt_prose_contract` 0; `codemod_fmt queue` empty;
substituted prose semantic audit CLEAN across 81 files.

**NOW done:** Benchmarking remaining time suffixes — migrated the final 7
time-label suffix sites in `vol1/benchmarking/benchmarking.qmd` to
`fmt_time(...)`, finishing that chapter's `time_unit` suffix bucket.
`style="word"` now owns pluralization for `hours`, `seconds`, and `minutes`;
compact seconds use symbol style. `assess_equiv` stayed byte-identical across
240 value exports and 102 substituted prose lines. `audit_fmt_usage.py` now
reports 32 `fmt_time` calls and the `time_unit` suffix bucket down to 616.
Verification: py_compile PASS; `git diff --check` PASS; `./book/binder check
math` PASS; focused pytest suite PASS (161 tests); `fmt_prose_contract` 0;
`codemod_fmt queue` empty; substituted prose semantic audit CLEAN across 81
files.

**NOW done:** `vol1/ml_systems` time suffixes — migrated all 24 `time_unit`
suffix sites to `fmt_time(...)`, finishing that chapter's time suffix bucket.
This also hardened `fmt_time(style="word")` for Pint's canonical `year` alias
(`a`), used `allow_negative=True` for the negative latency-headroom string, and
kept an explicit `commas=True` override for the one grouped millisecond value.
`assess_equiv` stayed byte-identical across 296 value exports and 143
substituted prose lines. `audit_fmt_usage.py` now reports 56 `fmt_time` calls
and the `time_unit` suffix bucket down to 592. Verification: py_compile PASS;
`git diff --check` PASS; `./book/binder check math` PASS; focused pytest suite
PASS (161 tests); `fmt_prose_contract` 0; `codemod_fmt queue` empty;
substituted prose semantic audit CLEAN across 81 files.

**NOW done:** `vol1/introduction` time suffixes — migrated all 15 `time_unit`
suffix sites to `fmt_time(...)`, finishing that chapter's time suffix bucket.
Old `fmt_int` duration sites now round explicitly before `fmt_time`, preserving
the rendered integer wording while keeping formatter precision checks intact.
`assess_equiv` stayed byte-identical across 91 value exports and 45 substituted
prose lines. `audit_fmt_usage.py` now reports 71 `fmt_time` calls and the
`time_unit` suffix bucket down to 577. Verification: py_compile PASS; `git
diff --check` PASS; `./book/binder check math` PASS; focused pytest suite PASS
(161 tests); `fmt_prose_contract` 0; `codemod_fmt queue` empty; substituted
prose semantic audit CLEAN across 81 files.

**NOW done:** `vol1/hw_acceleration` millisecond suffixes — migrated all 6
`time_unit` suffix sites to `fmt_time(..., "ms")`, finishing that chapter's
time suffix bucket. `assess_equiv` stayed byte-identical across 255 value
exports and 140 substituted prose lines. `audit_fmt_usage.py` now reports 77
`fmt_time` calls and the `time_unit` suffix bucket down to 571. Verification:
py_compile PASS; `git diff --check` PASS; `./book/binder check math` PASS;
focused pytest suite PASS (161 tests); `fmt_prose_contract` 0;
`codemod_fmt queue` empty; substituted prose semantic audit CLEAN across 81
files.

**NOW done:** structured quantity denominator cleanup — migrated the only QMD
`fmt_qty(..., extra_suffix="/inference")` call to `per="inference"` in
`vol1/ml_systems/ml_systems.qmd`. `assess_equiv` stayed byte-identical across
296 value exports and 143 substituted prose lines, and `audit_fmt_usage.py` now
reports no `extra_suffix=` calls in QMD.

**NOW done:** `vol1/ml_ops` pp suffix gap — migrated 3 surviving
`suffix=" pp"` sites to `fmt_pp(..., style="symbol")`, byte-identical across
132 value exports and 75 substituted prose lines. The semantic suffix checker
now treats `pp` as a percentage-point suffix and has a regression test for
`suffix=" pp"`; `audit_fmt_usage.py` classifies percentage-point suffixes
separately from percent-share suffixes.

**AUDIT note:** exact `scale_word` suffixes and compound scale-word suffixes
remain a separate design lane, not a blind migration. Current examples include
`"million"`, `"billion"`, `"million parameters"`, `"million queries"`,
`"million tokens/hour"`, and `"billion FLOPs"`. Recommended path: add
formatter-owned word-scale support (`fmt_count(..., scale_style="word")`),
structured counted-rate support, and a decision on FLOP prose (`fmt_qty` vs.
thin `fmt_ops`), then extend the suffix checker/audit tests to catch compound
scale suffixes.

**NOW done:** `vol1/ml_ops` time suffixes — migrated all 19 `time_unit` suffix
sites to `fmt_time(...)`, finishing that chapter's time suffix bucket. Symbol
units use compact style; prose labels use `style="word"`. `assess_equiv` stayed
byte-identical across 132 value exports and 75 substituted prose lines.
`audit_fmt_usage.py` now reports 96 `fmt_time` calls and the `time_unit` suffix
bucket down to 552. Verification: py_compile PASS; `git diff --check` PASS;
`./book/binder check math` PASS; focused pytest suite PASS (167 tests);
`fmt_prose_contract` 0; `codemod_fmt queue` empty; substituted prose semantic
audit CLEAN across 81 files.

**NOW done:** `vol1/data_engineering` time suffixes — migrated all 21
`time_unit` suffix sites to `fmt_time(...)`, finishing that chapter's exact
time suffix bucket. `assess_equiv` stayed byte-identical across 203 value
exports and 99 substituted prose lines. `audit_fmt_usage.py` now reports 116
`fmt_time` calls and the `time_unit` suffix bucket down to 532. Verification:
py_compile PASS; `git diff --check` PASS; `./book/binder check math` PASS;
focused pytest suite PASS (167 tests); `fmt_prose_contract` 0;
`codemod_fmt queue` empty; substituted prose semantic audit CLEAN across 81
files.

**NOW done:** `vol1/ml_workflow` time suffixes — migrated all 10 `time_unit`
suffix sites to `fmt_time(...)`, finishing that chapter's exact time suffix
bucket. `assess_equiv` stayed byte-identical across 77 value exports and 36
substituted prose lines. `audit_fmt_usage.py` now reports 126 `fmt_time` calls
and the `time_unit` suffix bucket down to 522. Verification: py_compile PASS;
`git diff --check` PASS; `./book/binder check math` PASS; focused pytest suite
PASS (167 tests); `fmt_prose_contract` 0; `codemod_fmt queue` empty;
substituted prose semantic audit CLEAN across 81 files.

**NOW done:** corpus time-unit suffix lane — added `run_time_lane.py` and
migrated the remaining 522 exact `time_unit` suffix sites across 34 chapters to
`fmt_time(...)`. The lane keeps full unit names in source (`"millisecond"`,
`"second"`, `"hour"`, etc.) while `style` controls symbol vs word rendering.
The only auto-queue was the `μs` vs `µs` glyph distinction; `fmt_time` now owns
the book's microsecond output as `μs`, and those sites then passed the
byte-identical gate. A follow-up source-only pass normalized earlier
`fmt_time(..., "ms"/"s"/"h")` calls to full unit-name strings. Final audit:
`fmt_time` calls 650; `time_unit` suffix bucket gone. Verification: `git diff
--check` PASS; py_compile PASS; focused pytest suite PASS (171 tests);
`fmt_prose_contract` 0; `codemod_fmt queue` empty; `./book/binder check math`
PASS; substituted prose semantic audit CLEAN across 81 files.

**NOW done:** WS4 clean Quantity-backed unit batch 2 — migrated 99 additional
`fmt(q.m_as(UNIT), suffix=" UNIT")` sites to `fmt_qty(q, UNIT)`, byte-identical
by `run_unit_lane.py`. `fmt_qty` calls now stand at 263 and the `physical_unit`
suffix bucket is down to 1,258. The lane now emits `per="token"` rather than
legacy `extra_suffix="/token"`, and QMD has zero `extra_suffix=` calls. Twenty
Quantity-backed candidates remain because accepting them would visibly change
units (`GB`→`GiB` or missing `/s` denominators); treat those as correctness
decisions. Verification: `git diff --check` PASS; py_compile PASS; focused
pytest suite PASS (171 tests); `fmt_prose_contract` 0; `codemod_fmt queue`
empty; `./book/binder check math` PASS; substituted prose semantic audit CLEAN
across 81 files.

**NOW done:** service-rate suffix lane — added `run_rate_lane.py` and migrated
41 exact counted service-rate suffix sites (`tokens/s`, `img/s`, `images/s`,
`req/s`, `samples/s`, `FPS`) to `fmt_rate(...)`, byte-identical across 9
chapters. `fmt_rate` calls now stand at 67 and the `physical_unit` suffix
bucket is down to 1,217. `fmt_semantic_suffix` now flags these as
`rate_in_suffix`, so future service-rate labels cannot hide in generic
`suffix=`. Verification: `git diff --check` PASS; py_compile PASS; focused
pytest suite PASS (174 tests); `fmt_prose_contract` 0; `codemod_fmt queue`
empty; `./book/binder check math` PASS; substituted prose semantic audit CLEAN
across 81 files.

**NOW done:** compound GPU-day count — migrated the last `count_label` suffix
site to `fmt_count(..., label="GPU-day", plural_label="GPU-days")`, with the
footnote prose adjusted so visible output stays identical. The matching
GPU-hour compound also uses `fmt_count`. `audit_fmt_usage.py` now has no
`count_label` bucket. Verification after the service-rate and GPU-day count
batch: `git diff --check` PASS; py_compile PASS; focused pytest suite PASS (174
tests); `fmt_prose_contract` 0; `codemod_fmt queue` empty; `./book/binder check
math` PASS; substituted prose semantic audit CLEAN across 81 files.

**NEXT:**
1. Add a formatter-default cleanup pass after the semantic lanes: each
   formatter should own comma defaults by value kind, and explicit `commas=`
   in QMD should remain only when it is an intentional override. During
   byte-identical migration, preserve explicit `commas=` first; remove
   redundant arguments only after equivalence is proven.
2. Add a keyword-unit API ergonomics pass after suffix migration stabilizes:
   support and migrate toward `unit=` on unit-bearing helpers
   (`fmt_time`, `fmt_qty`, `fmt_rate`, and range variants), then lint against
   new positional unit arguments. Add tests for keyword compatibility,
   missing/both-unit errors, time-unit validation, rate allowlists, and range
   endpoint checks through keyword units.
3. Continue the semantic non-Quantity lanes: any remaining non-physical rates
   and time values. Add entries to `AUDIT_LEDGER.md` for every touched LEGO
   cell.
4. Continue WS4 with `run_unit_lane.py` chapter-sized batches. Highest remaining
   clean counts: `vol2/backmatter/appendix_fleet` (14), `vol2/ops_scale` (14),
   `vol1/hw_acceleration` (11), `vol1/introduction` (11),
   `vol2/network_fabrics` (11), `vol2/performance_engineering` (11).
5. Render-verify any newly changed chapters before Phase 3B/PDF sign-off.

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
- Codex `vol1/backmatter/appendix_algorithm` physical-unit cleanup: migrated
  all 11 remaining physical-unit suffix sites in the appendix to typed quantity
  formatters, byte-identically. `appendix_algorithm` now has 0 `suffix=` calls;
  physical-unit suffixes dropped 881 -> 870.
- Codex `vol2/introduction` physical-unit cleanup: migrated all 8 remaining
  physical-unit suffix sites in the chapter to typed quantity formatters,
  byte-identically. `vol2/introduction` now has 0 `suffix=` calls;
  physical-unit suffixes dropped 889 -> 881.
- Codex `vol1/ml_workflow` physical-unit cleanup: migrated all 8 remaining
  physical-unit suffix sites in the chapter to typed quantity formatters,
  byte-identically. `ml_workflow` now has 0 `suffix=` calls; physical-unit
  suffixes dropped 897 -> 889.
- Codex Vol1 introduction/data-selection physical-unit cleanup: migrated 8
  physical-unit suffix sites across `introduction` and `data_selection` to
  typed quantity formatters, byte-identically. Both touched files now have 0
  `suffix=` calls; physical-unit suffixes dropped 905 -> 897.
- Codex assumptions appendix physical-unit cleanup: migrated 10 physical-unit
  suffix sites across the Vol1 and Vol2 `appendix_assumptions` files to typed
  quantity formatters, byte-identically. Both touched files now have 0
  `suffix=` calls; physical-unit suffixes dropped 915 -> 905.
- Codex small chapter physical-unit cleanup: migrated 17 physical-unit suffix
  sites across `ml_ops`, `appendix_data`, and `security_privacy` to typed
  quantity formatters, byte-identically. All three touched files now have 0
  `suffix=` calls; physical-unit suffixes dropped 932 -> 915.
- Codex tiny Vol2 physical-unit tail: migrated 7 physical-unit suffix sites
  across `appendix_c3`, `robust_ai`, `conclusion`, and `appendix_reliability`
  to typed quantity formatters, byte-identically. All four touched files now
  have 0 `suffix=` calls; physical-unit suffixes dropped 939 -> 932.
- Codex `vol2/backmatter/appendix_communication` physical-unit lane: migrated
  all 9 remaining physical-unit suffix sites in the chapter to typed quantity
  formatters, byte-identically. `appendix_communication` now has 0 `suffix=`
  calls; physical-unit suffixes dropped 948 → 939.
- Codex `vol2/fault_tolerance` physical-unit lane: migrated all 10 remaining
  physical-unit suffix sites in the chapter to typed quantity formatters,
  byte-identically. `fault_tolerance` now has 0 `suffix=` calls; physical-unit
  suffixes dropped 958 → 948.
- Codex `vol2/collective_communication` physical-unit lane: migrated all 26
  remaining physical-unit suffix sites in the chapter to typed quantity
  formatters, byte-identically. `collective_communication` now has 0 `suffix=`
  calls; physical-unit suffixes dropped 984 → 958.
- Codex `vol2/backmatter/appendix_fleet` physical-unit lane: migrated all 28
  remaining physical-unit suffix sites in the chapter to typed quantity
  formatters, byte-identically. `appendix_fleet` now has 0 `suffix=` calls;
  physical-unit suffixes dropped 1,012 → 984.
- Codex `vol1/ml_systems` physical-unit lane: migrated all 30 remaining
  physical-unit suffix sites in the chapter to typed quantity formatters,
  byte-identically. `ml_systems` now has 0 physical-unit suffix sites;
  physical-unit suffixes dropped 1,042 → 1,012.
- Codex direct Quantity physical-unit lane: added `fmt_qty_int(...)` for
  checked Pint Quantities that intentionally render as rounded integers and
  migrated 29 direct Quantity-backed physical-unit suffix sites
  byte-identically across 8 chapters. `fmt_qty_int` now has 20 call sites;
  physical-unit suffixes dropped 1,071 → 1,042.
- Codex GiB-backed memory-capacity lane: migrated remaining obvious
  `m_as(GiB)` + `suffix=" GB"` memory specs to
  `fmt_qty(..., GiB, unit_label="GB")`, byte-identical across 7 chapters.
  Physical-unit suffixes dropped 1,090 → 1,071. Follow-up TODO: hardware/model
  display accessors should own canonical display units for memory capacity,
  bandwidth, TDP, and peak FLOP/s.
- Codex network `Gb/s` lane: migrated all 20 `suffix=" Gb/s"` sites to
  `fmt_qty(..., Gbps, unit_label="Gb/s")`, byte-identical across 7 chapters.
  No `Gb/s` suffix sites remain; physical-unit suffixes dropped 1,110 → 1,090.
  Follow-up TODO: central stock unit label registry so the label is not
  repeated at every call site.
- Codex appendix bandwidth denominator lane: migrated 16 split bandwidth values
  in `appendix_assumptions` and `appendix_fleet` to
  `fmt_qty(..., GB/second|TB/second)` and removed external prose/table `/s`.
  Substituted prose stayed byte-identical; physical-unit suffixes dropped
  1,126 → 1,110; contract 0, semantic 0, queue empty, targeted suite 182
  passing.
- Codex compound-scale lane: cleared all 14 `compound_scale` suffix sites by
  adding direct word-scale support (`scale="million"` / `"billion"`),
  checked scaled `fmt_rate(..., scale=...)`, and `fmt_count(...,
  attributive=True)` for `7-billion` modifiers; word-scale FLOP phrases use
  `fmt_qty(..., unit_label=...)`. Intentional adjudicated changes:
  `200 K parameters`→`200K parameters`, `32K tokens`→`32.8K tokens`, and
  `cost_saving_str` now owns `\$` through `fmt_usd` while substituted prose is
  unchanged. No `compound_scale` suffix bucket remains; remaining suffix bucket
  is only `physical_unit` 1,126; contract 0, semantic 0, queue empty, targeted
  suite 182 passing.
- Codex unit-rate/denominator lane: added checked `fmt_qty(unit_label=...)` and
  migrated all 16 `TFLOP/s per W`, `kg/kWh`, `MWh/household-year`,
  `FLOP/byte`, `GB/day`, `GB per day`, `MB/photo`, `KB/patient`, `MWh/year`,
  and `kJ per hour` suffix sites byte-identically across 7 chapters; no
  `unit_rate_or_denominator` suffix bucket remains; contract 0, semantic 0,
  queue empty, targeted suite 181 passing.
- Codex time-compound lane: cleared the 4 remaining `ms latency`,
  `ms round-trip`, and `ms+` suffixes using `fmt_time(...)` plus checked
  `marker="+"`; byte-identical across 3 chapters; no `time_compound` suffix
  bucket remains; contract 0, semantic 0, queue empty, targeted suite 179
  passing.
- Codex attributive time lane: added `fmt_time(..., style="word",
  attributive=True)` for `1-hour`/`24-hour`/`5-minute` noun modifiers and
  migrated the 4 remaining resource-time suffix sites byte-identically across 3
  chapters; no `resource_time` suffix bucket remains; contract 0, semantic 0,
  queue empty, targeted suite 178 passing.
- Codex exact FLOP-count lane: migrated 12 `GFLOPs`/`MFLOPs`/`KFLOPs`/`PFLOPs`
  suffix sites to `fmt_qty(...)` with Pint FLOP units, byte-identical across 5
  chapters; no `op_count` suffix bucket remains; contract 0, semantic 0, queue
  empty, targeted suite 176 passing.
- Codex resource-time count-label lane: migrated 15 straightforward
  resource-time suffixes (`PFLOP-days`, `TPUv4-hours`, `person-years`,
  `instance-seconds`, `GPU-hours`, `GPU-hr`) to `fmt_count(label=...)`
  byte-identically across 7 chapters; resource_time suffixes are down to 4
  hyphenated attributive decision sites; contract 0, semantic 0, queue empty,
  targeted suite 176 passing.
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
