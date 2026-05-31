# FMT migration — handoff plan for Codex (finish the remaining work)

> **You are continuing a corpus-wide migration to a typed `fmt_*` formatter
> family.** The dangerous, error-prone part (percent / multiplier / scale-division
> / percentage-points) is **already done and verified**. Your job is the remaining
> lower-risk lanes + the later render/lock phases.
> Work the same way the previous agent did: small commits, keep the gates green,
> never guess on the user's prose.

Worktree: `/Users/VJ/GitHub/MLSysBook-fmt-fix` · branch `fmt-fix` (off `dev`).
Run ALL fmt tooling from the repo root with `PYTHONPATH=mlsysim`.

---

## 0. Read these first (orientation, in order)
1. `.claude/rules/fmt.md` — **the contract**: which formatter for which value-kind,
   the OUTPUT-block recipe, the prose rules, the per-kind guards. This is the law.
2. `book/tools/audit/fmt/PLAN_OF_RECORD.md` — current agreed plan: formatter
   taxonomy, migration lanes, and the separate render/prose audit.
3. `book/tools/audit/fmt/AUDIT_LEDGER.md` — running validation notes. Add every
   touched LEGO cell here as migration work continues.
4. `book/tools/audit/fmt/NIGHT_RESUME.md` — current state + the remaining
   WS4/WS3/render work.
5. `book/tools/audit/fmt/MIGRATION.md` — rollout board, workstreams, render phases.
6. `book/tools/audit/fmt/ASSESSMENT.md` — the equivalence regimes (byte-identical
   vs glyph-relocation) and the verification gauntlet.

---

## 1. Invariants — do NOT break these
- **Every source edit must keep the chapter executing headlessly** and keep all
  three gates (§2) green. Run them after every batch.
- **Pure formatter relocations must stay byte-identical** (rendered value AND
  visible prose). Verify with `assess_equiv` / by dumping the `_str` value before
  and after. A value-changing edit is allowed ONLY when it is a deliberate,
  documented correctness fix — then you must re-render and read the sentence.
- **Never invent a value-kind glyph in a `suffix=`.** `%`, `×`, `$`, and scale
  letters K/M/B/T are forbidden in `suffix=`; use the typed formatter. A physical
  unit label (`" GB"`, `" ms"`, `" W"`) in `suffix=` is the WS4 target (§4B).
- **No raw `prefix=` in chapter formatter calls.** Approximate and lower-bound
  display markers are now named flags (`approx=True`, `lower_bound=True`) on the
  relevant formatter wrappers. Currency approximation stays
  `fmt_usd(..., approx=True)`.
- **Git:** small commits with clear messages; let pre-commit run; do NOT
  `--amend`, do NOT force-push, do NOT skip hooks. The `prettify-tables` hook may
  re-touch a table AFTER staging and abort the commit — just `git add -A` and
  re-commit (this is normal, not an error; it happened repeatedly last session).
- **Do not change the user's prose grammar/wording on a judgment call.** If a fix
  requires rewording a sentence, splitting an export, or a style ruling, STOP and
  add it to a "needs user decision" list instead of guessing.
- **Imports:** chapters either do a document-level `from mlsysim import *` (then a
  new formatter is available everywhere via the shared kernel) OR use per-cell
  `from mlsysim.fmt import …` selective imports (then you must add the new
  formatter name to each using-cell's import line). Check which the chapter uses.

---

## 2. The three gates (must be green after every batch)
```
# from repo root, PYTHONPATH=mlsysim
python3 book/tools/audit/fmt/fmt_prose_contract.py   --root book/quarto/contents   # expect: 0 violations
python3 book/tools/audit/fmt/audit_prose_semantics.py --root book/quarto/contents  # expect: 0 findings ... CLEAN
python3 book/tools/audit/fmt/codemod_fmt.py queue     --root book/quarto/contents   # expect: by kind: {}
```
Plus the test suite (keep at 100%):
```
python3 -m pytest mlsysim/tests/test_fmt.py book/tests/test_codemod_fmt.py \
  book/tests/test_fmt_prose_contract.py book/tests/test_audit_prose_semantics.py \
  book/tests/test_visible_text.py book/tests/test_fmt_semantic_suffix.py \
  book/tests/test_lego_dead_code.py -q -o addopts=''
```
Gate 1 = glyph ownership (static AST). Gate 2 = rendered-composite semantics
(executes chapters, substitutes values, flags dup unit/glyph, percent-vs-points,
mult-direction "0.5× faster", currency-as-percent, unresolved refs). Gate 3 =
remaining dangerous suffixes by kind. All three already pass right now.

---

## 3. Work items — priority order

### A. User-decision items — DONE
The previously blocked decisions are now resolved.

**A1 — Scale queue: DONE.**
User ruled for the no-space house style (`70K`, `3.5M`, `70B`). All 44 queued
scale sites were migrated to `fmt_count(raw, scale=...)`, plus one manual
`fmt(...) + "B"` blind spot in `fleet_orchestration`. This is a deliberate
style-normalization change, not a byte-identical relocation: examples include
`70 B`→`70B`, `270 K`→`270K`, and `100k`→`100K`. The queue is now empty. The
one-time runner is `run_scale_style_lane.py`; use it only if this lane needs to
be replayed or audited.

**A2 — Four entangled percentage-point sites: DONE.**
User approved the recommendation. `benchmarking.qmd` now uses
`fmt_pp(acc_drop, precision=1, attributive=True)` for the top-1 drop,
hyphenates the hardcoded `1 percentage-point threshold`, uses noun-form
`fmt_pp(edge_drop, precision=1)` for the edge-case drop, and rewords the table
cell to `(drop of 6.8 percentage points)`. Benchmarking HTML was rebuilt and
grepped for the approved wording.

**A3 — Raw formatter prefixes: DONE.**
The 16 remaining chapter-level `prefix="~"` / `prefix="> "` formatter calls were
migrated byte-identically to named `approx=True` / `lower_bound=True` flags.
`fmt`, `fmt_int`, `fmt_qty`, and `fmt_count` now expose those markers, so future
approximate quantities/counts do not need raw string prefixes. The corpus has
zero QMD `prefix=` call sites.

**A4 — Plan-of-record + structured API batch: DONE.**
`PLAN_OF_RECORD.md` and `AUDIT_LEDGER.md` now record the agreed strategy:
semantic formatter APIs plus a separate touched-cell and whole-book inline
render/prose audit. The formatter API now includes structured currency
`scale=`/`per=`, strict count labels/plural overrides, `fmt_rate`, `fmt_time`
with symbol/word style, and typed range helpers (`fmt_qty_range`,
`fmt_time_range`, `fmt_count_range`, `fmt_usd_range`). This batch touched no QMD
LEGO cells. Verification: py_compile PASS, `git diff --check` PASS, focused
pytest suite PASS (157 tests), prose-contract 0, semantic audit 0 findings,
codemod queue empty.

**A5 — Currency denominator relocation: DONE.**
91 chapter call sites moved from `fmt_usd(..., suffix="/...")` to
`fmt_usd(..., per="...")`, removing the `rate_denominator` `suffix=` bucket from
`audit_fmt_usage.py`. `assess_equiv` proved byte-identical exported values and
inline prose for all 10 touched chapters. Verification: `git diff --check` PASS,
focused pytest suite PASS (157 tests), prose-contract 0, semantic audit 0
findings, codemod queue empty.

**A6 — Currency scale/range relocation: DONE.**
The remaining 78 chapter-level `fmt_usd(..., suffix=...)` call sites were moved
to `scale=`, `per=`, checked `marker="*"`, or `fmt_usd_range(...)`. QMD now has
zero `fmt_usd(..., suffix=...)` calls. `assess_equiv` proved byte-identical
values and inline prose for all scale/marker relocations; the only intentional
diff is the H100 table price range changing `~\$25,000-30,000` to
`~\$25,000–30,000` through `fmt_usd_range(..., repeat_symbol=False)`.
Verification: py_compile PASS, `git diff --check` PASS, focused pytest suite
PASS (161 tests), `./book/binder check math` PASS, prose-contract 0, semantic
audit 0 findings, codemod queue empty.

**A7 — QPS rate relocation: DONE.**
26 `suffix=" QPS"` sites moved to `fmt_rate(..., "QPS")`. `fmt_rate` now
defaults to `commas=True` so it matches old `fmt`/`fmt_int` output unless a site
explicitly passes `commas=False`. `assess_equiv` proved byte-identical values
and inline prose for all 5 touched chapters. Verification: py_compile PASS,
`git diff --check` PASS, `./book/binder check math` PASS, focused pytest suite
PASS (161 tests), prose-contract 0, semantic audit 0 findings, codemod queue
empty.

**A8 — GPU count labels: DONE.**
77 `suffix=" GPUs"` sites moved to `fmt_count(..., label="GPU")` with
byte-identical values and inline prose across all 20 touched chapters. The one
singular `suffix=" GPU"` site remains intentionally because it forms
`GPU-days`, not a standalone count noun. `audit_fmt_usage.py` now reports
`fmt_count` calls at 163 and the `count_label` suffix bucket at 41. Verification:
py_compile PASS, `git diff --check` PASS, `./book/binder check math` PASS,
focused pytest suite PASS (161 tests), prose-contract 0, semantic audit 0
findings, codemod queue empty.

**A9 — Remaining direct count labels: DONE.**
40 `tokens`/`nodes`/`layers`/`queries`/`images` suffix sites moved to
`fmt_count(..., label=...)`, byte-identical across all 15 touched chapters. Old
`fmt_int` query sites now round explicitly before `fmt_count`, preserving output
while making the integer boundary visible. The only remaining `count_label`
suffix is the documented `GPU-days` compound. Verification: py_compile PASS,
`git diff --check` PASS, `./book/binder check math` PASS, focused pytest suite
PASS (161 tests), prose-contract 0, semantic audit 0 findings, codemod queue
empty.

**A10 — Formatter comma-default cleanup: TODO.**
User raised a design concern that `commas=` should usually be owned by each
semantic formatter, not repeated at the top-level callsite. The plan of record
now says to preserve explicit `commas=` during byte-identical migration, then
run a cleanup pass that removes redundant `commas=` arguments and leaves only
intentional overrides. Expected defaults: currency/count/rate helpers group
large numbers; compact physical/time quantities, percentages, percentage
points, ratios, and multipliers default to no grouping unless overridden.

**A10b — Unit keyword/source spelling cleanup: TODO.**
User raised that time/quantity/rate helpers should expose the argument as
`unit=`, not as an ambiguous positional string. The plan of record now keeps
`label=` only for count nouns (`fmt_count(label="GPU")`) and treats
`fmt_time(..., "second")` as a unit argument. During current time migrations,
prefer full unit-name strings (`"millisecond"`, `"second"`, `"hour"`) and let
`style="symbol"` or `style="word"` control the rendered suffix. A later API
ergonomics pass should add keyword aliases such as `fmt_time(t, unit="second")`
and lint against new positional unit arguments.

**A11 — Benchmarking millisecond time suffixes: DONE.**
25 `suffix=" ms"` sites in `vol1/benchmarking/benchmarking.qmd` moved to
`fmt_time(..., "ms")`, byte-identical values and inline prose. Explicit
`commas=False` was preserved for now; removing redundant comma overrides belongs
to A10. `audit_fmt_usage.py` now reports `fmt_time` calls at 25 and the
`time_unit` suffix bucket down to 623. Verification: py_compile PASS,
`git diff --check` PASS, `./book/binder check math` PASS, focused pytest suite
PASS (161 tests), prose-contract 0, semantic audit 0 findings, codemod queue
empty.

**A12 — Benchmarking remaining time suffixes: DONE.**
The 7 remaining `benchmarking.qmd` time-label suffixes moved to `fmt_time(...)`:
compact seconds use symbol style, and prose words (`hours`, `seconds`,
`minutes`) use `style="word"` for formatter-owned plural checks. This finished
all `time_unit` suffixes in `vol1/benchmarking/benchmarking.qmd`. Values and
inline prose stayed byte-identical across 240 exports and 102 prose lines.
`audit_fmt_usage.py` now reports `fmt_time` calls at 32 and the `time_unit`
suffix bucket down to 616. Verification: py_compile PASS, `git diff --check`
PASS, `./book/binder check math` PASS, focused pytest suite PASS (161 tests),
prose-contract 0, semantic audit 0 findings, codemod queue empty.

**A13 — ML systems time suffixes: DONE.**
All 24 `time_unit` suffix sites in `vol1/ml_systems/ml_systems.qmd` moved to
`fmt_time(...)`, byte-identical values and inline prose. This batch hardened
`fmt_time(style="word")` for Pint's canonical `year` alias (`a`), used
`allow_negative=True` for the one negative latency-headroom string, and kept
`commas=True` for the one previously grouped millisecond value. `audit_fmt_usage.py`
now reports `fmt_time` calls at 56 and the `time_unit` suffix bucket down to
592. Verification: py_compile PASS, `git diff --check` PASS, `./book/binder
check math` PASS, focused pytest suite PASS (161 tests), prose-contract 0,
semantic audit 0 findings, codemod queue empty.

**A14 — Introduction time suffixes: DONE.**
All 15 `time_unit` suffix sites in `vol1/introduction/introduction.qmd` moved
to `fmt_time(...)`, byte-identical across 91 value exports and 45 prose lines.
Old `fmt_int` duration sites now round explicitly before `fmt_time`, preserving
the old integer display while keeping the precision guard meaningful.
`audit_fmt_usage.py` now reports `fmt_time` calls at 71 and the `time_unit`
suffix bucket down to 577. Verification: py_compile PASS, `git diff --check`
PASS, `./book/binder check math` PASS, focused pytest suite PASS (161 tests),
prose-contract 0, semantic audit 0 findings, codemod queue empty.

**A15 — Hardware acceleration millisecond suffixes: DONE.**
All 6 `time_unit` suffix sites in `vol1/hw_acceleration/hw_acceleration.qmd`
moved to `fmt_time(..., "ms")`, byte-identical across 255 value exports and
140 prose lines. `audit_fmt_usage.py` now reports `fmt_time` calls at 77 and
the `time_unit` suffix bucket down to 571. Verification: py_compile PASS,
`git diff --check` PASS, `./book/binder check math` PASS, focused pytest suite
PASS (161 tests), prose-contract 0, semantic audit 0 findings, codemod queue
empty.

**A16 — Structured quantity denominator cleanup: DONE.**
The only QMD `fmt_qty(..., extra_suffix="/inference")` call moved to structured
`per="inference"` in `vol1/ml_systems/ml_systems.qmd`, byte-identical across
296 value exports and 143 prose lines. `audit_fmt_usage.py` now reports no
`extra_suffix=` calls in QMD.

**A17 — Keyword-unit API ergonomics: TODO.**
User raised that `fmt_time(x, "ms")` would read better and be harder to misuse
as `fmt_time(x, unit="ms")`. Recommendation: after the active byte-identical
suffix lanes, add keyword aliases for unit-bearing semantic helpers
(`fmt_time`, `fmt_qty`, `fmt_rate`, `fmt_time_range`, `fmt_qty_range`), convert
QMD call sites, then lint against new positional unit arguments. Keep
positional compatibility during migration. Add tests for keyword compatibility,
missing/both-unit errors, time-unit validation, rate allowlist validation, and
range endpoint validation through keyword units.

**A18 — ML ops pp suffix gap: DONE.**
Audit found 3 `suffix=" pp"` sites that the semantic suffix checker missed
because it recognized `"percentage points"` but not `"pp"`. These moved to
`fmt_pp(..., style="symbol")` in `vol1/ml_ops/ml_ops.qmd`, byte-identical
across 132 value exports and 75 prose lines. `fmt_semantic_suffix` and its test
now flag `suffix=" pp"` as `pp_in_suffix`; `audit_fmt_usage.py` now classifies
percentage-point suffixes separately and reports `fmt_pp` calls at 21.

**A19 — Scale-word and compound-scale formatter design: DONE.**
`fmt_count` now accepts direct word scales such as `scale="million"` /
`scale="billion"` while preserving compact glyph scales such as `scale="M"` /
`scale="B"`. `fmt_rate` now accepts the same checked scale argument for
counted rates such as `tokens/hour`, and word-scale FLOP prose uses
`fmt_qty(..., unit_label="billion FLOPs")` to preserve visible wording while
still converting through Pint. `fmt_count(..., scale_style="word")` remains for
compatibility, but new QMD should prefer the clearer direct word-scale spelling.

**A20 — ML ops time suffixes: DONE.**
All 19 `time_unit` suffix sites in `vol1/ml_ops/ml_ops.qmd` moved to
`fmt_time(...)`, byte-identical across 132 value exports and 75 prose lines.
Old `fmt_int` duration sites keep intentional integer display through
`precision=0`. `audit_fmt_usage.py` now reports `fmt_time` calls at 96 and the
`time_unit` suffix bucket down to 552; `ml_ops.qmd` has no remaining
`time_unit` suffixes. Verification: py_compile PASS, `git diff --check` PASS,
`./book/binder check math` PASS, focused pytest suite PASS (167 tests),
prose-contract 0, semantic audit 0 findings, codemod queue empty.

**A21 — Data engineering time suffixes: DONE.**
All 21 `time_unit` suffix sites in `vol1/data_engineering/data_engineering.qmd`
moved to `fmt_time(...)`, byte-identical across 203 value exports and 99 prose
lines. `audit_fmt_usage.py` now reports `fmt_time` calls at 116 and the
`time_unit` suffix bucket down to 532; `data_engineering.qmd` has no remaining
`time_unit` suffixes. Verification: py_compile PASS, `git diff --check` PASS,
`./book/binder check math` PASS, focused pytest suite PASS (167 tests),
prose-contract 0, semantic audit 0 findings, codemod queue empty.

**A22 — ML workflow time suffixes: DONE.**
All 10 `time_unit` suffix sites in `vol1/ml_workflow/ml_workflow.qmd` moved to
`fmt_time(...)`, byte-identical across 77 value exports and 36 prose lines.
`audit_fmt_usage.py` now reports `fmt_time` calls at 126 and the `time_unit`
suffix bucket down to 522; `ml_workflow.qmd` has no remaining `time_unit`
suffixes. Verification: py_compile PASS, `git diff --check` PASS,
`./book/binder check math` PASS, focused pytest suite PASS (167 tests),
prose-contract 0, semantic audit 0 findings, codemod queue empty.

**A23 — Corpus time suffix lane: DONE.**
Added `run_time_lane.py` and migrated the remaining 522 exact `time_unit`
suffix sites to `fmt_time(...)`. The lane uses full unit-name strings in QMD
source (`"millisecond"`, `"second"`, `"hour"`, etc.) and accepts a chapter only
when exported values and substituted prose are byte-identical. The only
temporary queue was the Greek-mu vs micro-sign distinction for `μs`; the
formatter now centralizes microsecond output as `μs`, matching the dominant
book source style, and the queued sites then migrated byte-identically. A
source-normalization pass also converted earlier `fmt_time(..., "ms"/"s"/"h")`
calls to full unit names. `audit_fmt_usage.py` now reports `fmt_time` calls at
650 and no remaining `time_unit` suffix bucket. Verification: `git diff
--check` PASS; py_compile PASS; focused pytest suite PASS (171 tests);
`fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `audit_prose_semantics.py` CLEAN across 81
files.

**A24 — WS4 clean Quantity-backed unit batch 2: DONE.**
`run_unit_lane.py --write --all` migrated 99 additional clean
`fmt(q.m_as(UNIT), suffix=" UNIT")` sites to `fmt_qty(q, UNIT)`, all accepted
only when values and substituted prose were byte-identical. `fmt_qty` calls now
stand at 263 and the `physical_unit` suffix bucket is down to 1,258. The unit
lane was hardened to emit `per="token"` instead of legacy
`extra_suffix="/token"`, and QMD again has zero `extra_suffix=` calls. Twenty
Quantity-backed candidates remain intentionally queued because canonical units
would visibly change output, mostly `GB`→`GiB` memory capacity and missing
bandwidth denominators such as `TB`→`TB/s`; handle those as correctness/prose
decisions, not automatic byte-identical relocations. A follow-up
`run_unit_lane.py --write --all` after A27 reconfirmed that all 20 fail the
byte-identical gate, and `.gitignore` now ignores these generated adjudication
queue files. Verification: `git diff
--check` PASS; py_compile PASS; focused pytest suite PASS (171 tests);
`fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `audit_prose_semantics.py` CLEAN across 81
files.

**A25 — Service-rate suffix lane: DONE.**
Added `run_rate_lane.py` and migrated 41 exact service-rate suffix sites
(`tokens/s`, `img/s`, `images/s`, `req/s`, `samples/s`, `FPS`) to
`fmt_rate(...)`, byte-identical across 9 chapters. `fmt_rate` calls now stand
at 67, the `physical_unit` bucket dropped to 1,217, and grep finds no remaining
QMD service-rate `suffix=` calls. `fmt_semantic_suffix` now flags these suffixes
as `rate_in_suffix` so they cannot come back through generic `fmt`.
Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
PASS (174 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue`
`by kind: {}`; `./book/binder check math` PASS; `audit_prose_semantics.py`
CLEAN across 81 files.

**A26 — Compound GPU-day count: DONE.**
The last `count_label` suffix site moved to
`fmt_count(..., label="GPU-day", plural_label="GPU-days")`, and the matching
GPU-hour compound moved to `fmt_count(..., label="GPU-hour",
plural_label="GPU-hours")`. `NASCostCalc.gpu_days_str` intentionally changed
from `22,400 GPU` to `22,400 GPU-days`, while the surrounding footnote dropped
the literal `-days`; `assess_equiv.py` confirmed the substituted prose stayed
byte-identical. `audit_fmt_usage.py` now has no `count_label` bucket.
Verification after A25/A26: `git diff --check` PASS; py_compile PASS; focused
pytest suite PASS (174 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py
queue` `by kind: {}`; `./book/binder check math` PASS;
`audit_prose_semantics.py` CLEAN across 81 files.

**A27 — Exact scale-word suffix lane: DONE.**
Added `fmt_count(..., scale_style="word")` and migrated the 8 exact
`suffix=" million"` / `suffix=" billion"` QMD sites to typed count formatting:
`data_selection` (1), `introduction` (2), `ml_ops` (1), `training` (1),
`network_fabrics` (2), and `sustainable_ai` (1). `assess_equiv.py` confirmed
byte-identical exported values and substituted prose for all six touched
chapters. `audit_fmt_usage.py` now reports no `scale_word` suffix bucket,
`fmt_count` calls at 213, and `physical_unit` suffixes at 1,216. The semantic
suffix gate now flags future exact scale-word suffixes as
`scale_word_in_suffix`. The LEGO dead-code checker was also hardened to ignore
wrapped keyword arguments, after pre-commit exposed the false positive on
`scale_style="word"`. Verification: `git diff --check` PASS; py_compile PASS;
focused pytest suite PASS (176 tests); `fmt_prose_contract.py` 0;
`codemod_fmt.py queue` `by kind: {}`; `./book/binder check math` PASS;
`audit_prose_semantics.py` CLEAN across 81 files. The later compound-scale lane
resolved the source spelling concern: new word-scale QMD should use
`scale="billion"` rather than `scale="B", scale_style="word"`.

**A28 — Residual plain count-label suffix lane: DONE.**
Migrated 17 remaining plain count-noun suffixes (`errors`, `steps`, `photos`,
`requests`, `servers`, `workers`, `stages`, `V100s`, `V100 GPUs`, `link tiers`,
etc.) to `fmt_count(..., label=...)`, byte-identical across 9 chapters:
`benchmarking`, `data_engineering`, `ml_systems`, `model_serving`, `training`,
`conclusion`, `data_storage`, `distributed_training`, and `network_fabrics`.
`audit_fmt_usage.py` now reports `fmt_count` at 230 and direct suffix calls at
1,199. Verification: `git diff --check` PASS; py_compile PASS; focused pytest
suite PASS (176 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue`
`by kind: {}`; `./book/binder check math` PASS; `./book/binder check code
--scope lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**A29 — Word-form micro/millisecond time suffix cleanup: DONE.**
Extended the time codemod to recognize `microsecond(s)` and `millisecond(s)`,
then migrated the three remaining exact word-form time suffixes in
`model_compression` and `responsible_engr` to `fmt_time(..., style="word")`.
`assess_equiv.py` confirmed identical values and prose for both chapters.
`audit_fmt_usage.py` now reports `fmt_time` at 653 and direct suffix calls at
1,196. Verification: `git diff --check` PASS; py_compile PASS; focused pytest
suite PASS (176 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue`
`by kind: {}`; `./book/binder check math` PASS; `audit_prose_semantics.py`
CLEAN across 81 files.

**A30 — Remaining suffix bucket split + epoch label: DONE.**
`audit_fmt_usage.py` now classifies the remaining direct suffix inventory into
actionable sub-buckets instead of reporting everything as `physical_unit`:
1,126 physical units, 19 resource-time labels, 16 unit rates/denominators, 14
compound scale phrases, 12 operation counts, and 8 time compounds. The last
plain count label found in that sweep, `epochs`, moved to
`fmt_count(..., label="epoch", plural_label="epochs")` byte-identically in
`data_engineering`. Verification: `assess_equiv.py` values/prose identical;
`git diff --check` PASS; py_compile PASS; focused pytest subset PASS;
`fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `audit_prose_semantics.py` CLEAN across 81
files.

**A31 — Time denominator suffix lane: DONE.**
Migrated four denominator-style time suffixes to `fmt_time(..., per=...)`:
`μs/op`, `s/hr`, `ms/step`, and `hours/day`. Values and substituted prose were
byte-identical across `frameworks`, `data_engineering`, `introduction`, and
`model_serving`. `audit_fmt_usage.py` now reports `fmt_time` at 657 and
`time_compound` down to 4. The four remaining time compounds are prose/API
decisions (`ms latency`, `ms round-trip`, `ms+`). Verification: `git diff
--check` PASS; py_compile PASS; focused pytest suite PASS (176 tests);
`fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `audit_prose_semantics.py` CLEAN across 81
files.

**A32 — Resource-time count-label lane: DONE.**
Migrated 15 straightforward resource-time suffixes to `fmt_count(..., label=...)`:
`PFLOP-days`, `TPUv4-hours`, `person-years`, `instance-seconds`, `GPU-hours`,
and `GPU-hr`. Values and substituted prose were byte-identical across
`data_engineering`, `ml_systems`, `data_selection`, `responsible_engr`,
`model_serving`, `inference`, and `ops_scale`. `audit_fmt_usage.py` now reports
`fmt_count` at 246 and remaining suffix buckets as: `physical_unit` 1,126,
`unit_rate_or_denominator` 16, `compound_scale` 14, `op_count` 12,
`resource_time` 4, and `time_compound` 4. The four remaining resource-time sites
are the hyphenated attributive forms (`-hour`, `-minute`), which are prose/API
decisions rather than simple plural count labels. Verification: `git diff
--check` PASS; py_compile PASS; focused pytest suite PASS (176 tests);
`fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `audit_prose_semantics.py` CLEAN across 81
files.

**A33 — Exact FLOP-count suffix lane: DONE.**
Migrated 12 exact FLOP-count suffixes (`GFLOPs`, `MFLOPs`, `KFLOPs`, `PFLOPs`)
to `fmt_qty(...)`, using the existing Pint FLOP units rather than adding a new
`fmt_ops` wrapper. Values and substituted prose were byte-identical across
`conclusion`, `frameworks`, `ml_systems`, `ml_workflow`, and `training`.
`audit_fmt_usage.py` now reports no `op_count` suffix bucket and `fmt_qty` at
275. Remaining suffix buckets: `physical_unit` 1,126,
`unit_rate_or_denominator` 16, `compound_scale` 14, `resource_time` 4, and
`time_compound` 4. Word-scale FLOP phrases such as `billion FLOPs` and
`trillion FLOPs` remain in the `compound_scale` bucket because changing them to
`GFLOPs`/`TFLOPs` would alter visible wording. Verification: `git diff --check`
PASS; py_compile PASS; focused pytest suite PASS (176 tests);
`fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `audit_prose_semantics.py` CLEAN across 81
files.

**A34 — Attributive time/resource-time suffix lane: DONE.**
Added `fmt_time(..., style="word", attributive=True)` for hyphenated singular
time noun modifiers such as `1-hour`, `24-hour`, `100,000-hour`, and
`5-minute`. Migrated the four remaining `resource_time` suffix sites
byte-identically across `data_engineering`, `data_selection`, and
`distributed_training`. `audit_fmt_usage.py` now reports no `resource_time`
suffix bucket and `fmt_time` at 661. Remaining suffix buckets:
`physical_unit` 1,126, `unit_rate_or_denominator` 16, `compound_scale` 14, and
`time_compound` 4. Verification: `git diff --check` PASS; py_compile PASS;
focused pytest suite PASS (178 tests); `fmt_prose_contract.py` 0;
`codemod_fmt.py queue` `by kind: {}`; `./book/binder check math` PASS;
`audit_prose_semantics.py` CLEAN across 81 files.

**A35 — Time-compound suffix lane: DONE.**
Cleared the four remaining time-compound suffixes. `ms latency` and
`ms round-trip` now use `fmt_time(...)` for the time value and keep the
descriptive word outside the formatter suffix; the `ms+` site now uses checked
`fmt_time(..., marker="+")`. Values and substituted prose were byte-identical
across `introduction`, `ml_systems`, and `ml_workflow`. `audit_fmt_usage.py` now
reports no `time_compound` suffix bucket and `fmt_time` at 665. Remaining suffix
buckets: `physical_unit` 1,126, `unit_rate_or_denominator` 16, and
`compound_scale` 14. Verification: `git diff --check` PASS; py_compile PASS;
focused pytest suite PASS (179 tests); `fmt_prose_contract.py` 0;
`codemod_fmt.py queue` `by kind: {}`; `./book/binder check math` PASS;
`audit_prose_semantics.py` CLEAN across 81 files.

**A36 — Unit-rate/denominator suffix lane: DONE.**
Added checked `fmt_qty(..., unit_label=...)` for house-style unit labels that
Pint cannot print byte-identically, while preserving dimension conversion
through `display_unit`. Migrated all 16 `unit_rate_or_denominator` suffix sites:
`TFLOP/s per W`, `kg/kWh`, `MWh/household-year`, `FLOP/byte`, `GB/day`,
`GB per day`, `MB/photo`, `KB/patient`, `MWh/year`, and `kJ per hour`.
Values and substituted prose were byte-identical across `hw_acceleration`,
`introduction`, `ml_systems`, `ml_workflow`, `nn_computation`, `training`, and
`edge_intelligence`. `audit_fmt_usage.py` now reports no
`unit_rate_or_denominator` suffix bucket and `fmt_qty` at 291. Remaining suffix
buckets before A37: `physical_unit` 1,126 and `compound_scale` 14. Verification: `git diff
--check` PASS; py_compile PASS; focused pytest suite PASS (181 tests);
`fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `audit_prose_semantics.py` CLEAN across 81
files.

**A37 — Compound-scale suffix lane: DONE.**
Cleared all 14 `compound_scale` suffix sites. `fmt_count` now supports direct
word scales (`scale="million"`, `scale="billion"`) plus
`attributive=True` for `7-billion` noun modifiers; `fmt_rate` supports checked
scaled count rates including `tokens/hour`; and word-scale FLOP phrases moved
to `fmt_qty(..., unit_label="billion FLOPs")` / `"trillion FLOPs"` so Pint still
dimension-checks the value. Most values were byte-identical. Intentional
adjudicated changes: `cost_saving_str` now owns the escaped dollar sign through
`fmt_usd` while substituted prose stayed identical; `200 K parameters` became
the user-approved compact `200K parameters`; and the old floored
`32K tokens` display for 32,768 tokens now renders as `32.8K tokens`.
`audit_fmt_usage.py` now reports no `compound_scale` suffix bucket. Remaining
suffix bucket: `physical_unit` 1,126. Verification: `git diff --check` PASS;
py_compile PASS; focused pytest suite PASS (182 tests); `fmt_prose_contract.py`
0; `codemod_fmt.py queue` `by kind: {}`; `./book/binder check math` PASS;
`audit_prose_semantics.py` CLEAN across 81 files.

**A38 — Appendix bandwidth denominator relocation: DONE.**
Moved 16 appendix bandwidth values from split string/prose units to full
`fmt_qty(..., GB/second|TB/second)` values: 3 in
`appendix_assumptions.qmd` and 13 in `appendix_fleet.qmd`. The rendered prose
and table cells stayed byte-identical because the external `/s` text was
removed wherever the formatter now emits `GB/s` or `TB/s`. `audit_fmt_usage.py`
now reports `fmt_qty` at 309 and physical-unit suffixes down to 1,110.
Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
PASS (182 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue`
`by kind: {}`; `./book/binder check math` PASS; `./book/binder check code
--scope lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**A39 — Network `Gb/s` suffix lane: DONE.**
Migrated all 20 `suffix=" Gb/s"` sites to `fmt_qty(..., Gbps,
unit_label="Gb/s")`, byte-identically across seven chapters:
`appendix_machine`, `data_engineering`, `hw_acceleration`,
`compute_infrastructure`, `distributed_training`, `fleet_orchestration`, and
`network_fabrics`. `audit_fmt_usage.py` now reports no remaining `Gb/s` suffix
sites and physical-unit suffixes down to 1,090. The repeated
`unit_label="Gb/s"` calls should later be replaced by a stock unit display-label
registry so `fmt_qty(..., Gbps)` owns the house style automatically.
Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
PASS (182 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue`
`by kind: {}`; `./book/binder check math` PASS; `./book/binder check code
--scope lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**Backlog note — prose-bound output contract.**
The user raised whether every OUTPUT value consumed by prose should effectively
be a `MarkdownStr`. Typed formatters already return `MarkdownStr`; add a later
gate/design pass to flag prose-bound plain strings or f-strings unless they are
intentional labels/sequences.

**Backlog note — hardware display API.**
The user raised that hardware/model objects should probably own common display
forms for specs they already store: memory capacity, memory bandwidth,
interconnect bandwidth, TDP, and peak FLOP/s. Add a later design pass for
hardware-aware display helpers/accessors so QMD call sites do not have to know
whether a stored value should display as GiB/GB, GB/s, Gb/s, or TB/s.

**A40 — GiB-backed memory capacity lane: DONE.**
Migrated the remaining obvious `m_as(GiB)` + `suffix=" GB"` memory-capacity
sites to `fmt_qty(..., GiB, unit_label="GB")`, byte-identically across
`model_serving`, `appendix_fleet`, `compute_infrastructure`,
`distributed_training`, `fleet_orchestration`, `inference`, and
`performance_engineering`. This preserves the visible `GB` house style while
keeping binary-capacity quantities attached through the formatter.
`audit_fmt_usage.py` now reports physical-unit suffixes down to 1,071.
Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
PASS (182 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue`
`by kind: {}`; `./book/binder check math` PASS; `./book/binder check code
--scope lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**A41 — Direct Quantity physical-unit lane: DONE.**
Added `fmt_qty_int(...)` for checked Pint Quantities that intentionally render
as rounded integers, without weakening `fmt_qty(..., precision=0)`. Migrated 29
direct Quantity-backed physical-unit suffix sites byte-identically across
`introduction`, `model_compression`, `data_engineering`, `ml_systems`,
`model_serving`, `appendix_fleet`, `data_storage`, and
`performance_engineering`. `audit_fmt_usage.py` now reports `fmt_qty_int` at 20
calls, `fmt_qty` at 357, and physical-unit suffixes down to 1,042.
Verification: `assess_equiv.py` values/prose identical for all touched
chapters; `git diff --check` PASS; py_compile PASS; focused pytest suite PASS
(190 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `./book/binder check code --scope
lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**A42 — `vol1/ml_systems` physical-unit lane: DONE.**
Migrated all 30 remaining physical-unit suffix sites in `ml_systems` to typed
quantity formatters, byte-identically. The chapter now has 0 physical-unit
suffixes. One-off labels such as `TOPS peak`, `TOPS derated`, `Mb/s`, and
`KB of detection summaries` are checked `unit_label=` values. `audit_fmt_usage.py`
now reports physical-unit suffixes down to 1,012, `fmt_qty` at 384, and
`fmt_qty_int` at 23. Verification: `assess_equiv.py` values/prose identical for
`ml_systems`; `git diff --check` PASS; py_compile PASS; focused pytest suite
PASS (190 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue`
`by kind: {}`; `./book/binder check math` PASS; `./book/binder check code
--scope lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**A43 — `vol2/backmatter/appendix_fleet` physical-unit lane: DONE.**
Migrated all 28 remaining physical-unit suffix sites in `appendix_fleet` to
typed quantity formatters, byte-identically. The chapter now has 0 `suffix=`
calls. This lane covered checkpoint sizes, bandwidths, memory footprint, peak
FLOP/s, rack/IT/facility power, and PUE overhead examples. The checkpoint
write-bandwidth string is now dimension-checked as `GB/second` but preserves the
existing exported value `100 GB` with prose appending `/s`; this split-rate
shape belongs in the later stock unit/prose-bound output design pass rather than
in a byte-identical cleanup. `audit_fmt_usage.py` now reports physical-unit
suffixes down to 984, `fmt_qty` at 406, and `fmt_qty_int` at 29. Verification:
`assess_equiv.py` values/prose identical for `appendix_fleet`; `git diff
--check` PASS; py_compile PASS; focused pytest suite PASS (190 tests);
`fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `./book/binder check code --scope
lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**A44 — `vol2/collective_communication` physical-unit lane: DONE.**
Migrated all 26 remaining physical-unit suffix sites in `collective_communication`
to typed quantity formatters, byte-identically. The chapter now has 0 `suffix=`
calls. This lane covered gradient/message sizes, critical-message-size examples,
MoE transfer sizes, Ring-vs-Tree crossover sizes, NVLink/InfiniBand bandwidth
recaps, hierarchical AllReduce data volumes, and overlap bucket sizes.
`audit_fmt_usage.py` now reports physical-unit suffixes down to 958 and
`fmt_qty` at 432. Verification: `assess_equiv.py` values/prose identical for
`collective_communication`; `git diff --check` PASS; py_compile PASS; focused
pytest suite PASS (190 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py
queue` `by kind: {}`; `./book/binder check math` PASS; `./book/binder check
code --scope lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81
files.

**A45 — `vol2/fault_tolerance` physical-unit lane: DONE.**
Migrated all 10 remaining physical-unit suffix sites in `fault_tolerance` to
typed quantity formatters, byte-identically. The chapter now has 0 `suffix=`
calls. This lane covered checkpoint component sizes, total checkpoint size,
per-node storage throughput, local NVMe bandwidth, GPT-3 shard size, and
recovery read bandwidth. The rounded per-node NFS throughput moved to
`fmt_qty_int(...)` so the old integer display remains explicit and unit-checked.
`audit_fmt_usage.py` now reports physical-unit suffixes down to 948,
`fmt_qty` at 441, and `fmt_qty_int` at 30. Verification: `assess_equiv.py`
values/prose identical for `fault_tolerance`; `git diff --check` PASS;
py_compile PASS; focused pytest suite PASS (190 tests); `fmt_prose_contract.py`
0; `codemod_fmt.py queue` `by kind: {}`; `./book/binder check math` PASS;
`./book/binder check code --scope lego-dead-code` PASS;
`audit_prose_semantics.py` CLEAN across 81 files.

**A46 — `vol2/backmatter/appendix_communication` physical-unit lane: DONE.**
Migrated all 9 remaining physical-unit suffix sites in `appendix_communication`
to typed quantity formatters, byte-identically. The chapter now has 0 `suffix=`
calls. This lane mostly covered split-rate bandwidth displays that export `GB`
and let prose/table text append `/s`; the values are now checked as
`GB/second` through `fmt_qty(..., unit_label="GB")`. `audit_fmt_usage.py` now
reports physical-unit suffixes down to 939, `fmt_qty` at 449, and `fmt_qty_int`
at 31. Verification: `assess_equiv.py` values/prose identical for
`appendix_communication`; `git diff --check` PASS; py_compile PASS; focused
pytest suite PASS (190 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py
queue` `by kind: {}`; `./book/binder check math` PASS; `./book/binder check
code --scope lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81
files.

**A47 — Tiny Vol2 physical-unit tail: DONE.**
Migrated 7 physical-unit suffix sites across four small Vol2 files:
`appendix_c3` (1), `robust_ai` (1), `conclusion` (2), and
`appendix_reliability` (3). All four touched files now have 0 `suffix=` calls,
and all exported values plus substituted prose are byte-identical. This lane
covered aggregate PFLOP/s, V100 bandwidth, machine power, H100 TFLOP/s,
checkpoint size, and checkpoint write bandwidth. `audit_fmt_usage.py` now
reports physical-unit suffixes down to 932, `fmt_qty` at 456, and
`fmt_qty_int` at 31. Verification: `assess_equiv.py` values/prose identical for
all four files; `git diff --check` PASS; py_compile PASS; focused pytest suite
PASS (190 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue`
`by kind: {}`; `./book/binder check math` PASS; `./book/binder check code
--scope lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**A48 — Small chapter physical-unit cleanup: DONE.**
Migrated 17 physical-unit suffix sites across `vol1/ml_ops` (5),
`vol1/backmatter/appendix_data` (6), and `vol2/security_privacy` (6), leaving
all three files with 0 `suffix=` calls. The lane covered KV-cache capacity,
observability ingest rates, monitoring storage volume, serialization throughput,
data-algebra sizes, and TEE/model memory sizes. All newly formatted values keep
their source quantities attached through `fmt_qty(...)`; split-rate ingest values
now attach `/second` at the output boundary. `audit_fmt_usage.py` now reports
physical-unit suffixes down to 915, `fmt_qty` at 473, and `fmt_qty_int` at 31.
Verification: `assess_equiv.py` values/prose identical for all three files;
`git diff --check` PASS; py_compile PASS; focused pytest suite PASS (190
tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `./book/binder check code --scope
lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**A49 — Assumptions appendix physical-unit cleanup: DONE.**
Migrated 10 physical-unit suffix sites across `vol1/backmatter/appendix_assumptions`
(5) and `vol2/backmatter/appendix_assumptions` (5), leaving both files with
0 `suffix=` calls. The lane covered H100 peak FLOP/s, memory bandwidth,
training memory, H100 memory capacity, A100 TDP, Llama gradient size,
facility power, WUE example power, and AI rack power. `audit_fmt_usage.py` now
reports physical-unit suffixes down to 905, `fmt_qty` at 483, and
`fmt_qty_int` at 31. Verification: `assess_equiv.py` values/prose identical for
both files; `git diff --check` PASS; py_compile PASS; focused pytest suite PASS
(190 tests); `fmt_prose_contract.py` 0; `codemod_fmt.py queue` `by kind: {}`;
`./book/binder check math` PASS; `./book/binder check code --scope
lego-dead-code` PASS; `audit_prose_semantics.py` CLEAN across 81 files.

**A50 — Vol1 introduction/data-selection physical-unit cleanup: DONE.**
Migrated 8 physical-unit suffix sites across `vol1/introduction` (2) and
`vol1/data_selection` (6), leaving both files with 0 `suffix=` calls. The lane
covered GPT-3 training energy, A100 peak FLOP/s, and the split-rate storage
throughput table where the formatter emits `MB` and the table literal supplies
`/s`. `audit_fmt_usage.py` now reports physical-unit suffixes down to 897,
`fmt_qty` at 491, and `fmt_qty_int` at 31. Verification: `assess_equiv.py`
values/prose identical for both files; `git diff --check` PASS; py_compile
PASS; focused pytest suite PASS (190 tests); `fmt_prose_contract.py` 0;
`codemod_fmt.py queue` `by kind: {}`; `./book/binder check math` PASS;
`./book/binder check code --scope lego-dead-code` PASS;
`audit_prose_semantics.py` CLEAN across 81 files.

### B. WS4 — unit-suffix lane (remaining 897 physical-unit suffixes: `GB`/`MB`/`W`/`GB/s`/…)  ← the big one
**Risk: LOW** (a unit label can't cause a 0–1↔0–100 / 100× error). **Effort: HIGH**
and NOT a clean codemod, because ~1,938 of the args are plain floats (e.g.
`weights_gb`), not Pint Quantities, and `fmt_qty` requires a Pint Quantity to
generate the unit. So this is per-site, judgment-bearing source work. Method:

1. **Inventory & bucket** (don't brute-force):
   ```
   python3 book/tools/audit/fmt/audit_fmt_usage.py --root book/quarto/contents --json > /tmp/fmt_usage.json
   ```
   Group by suffix unit and by whether the argument is already a Pint Quantity.
2. **Quantity-backed sites → `fmt_qty` / `fmt_qty_int`** (the clean, preferred case):
   `bw_str = fmt(bw.m_as(GB/second), suffix=" GB/s")` → `bw_str = fmt_qty(bw, GB/second)`.
   `fmt_qty` generates the suffix from the unit (always canonical), dimension-checks,
   and refuses currency. Use `fmt_qty_int(q, UNIT)` only when rounded integer
   display is intentional. Then DROP any duplicate unit the prose was adding.
3. **Plain-float sites** (`fmt(weights_gb, suffix=" GB")` with no Quantity in scope):
   prefer refactoring the source to carry a Quantity and use `fmt_qty`; if that's a
   large refactor, it is acceptable to LEAVE these for now — they are honest unit
   labels, low risk. Do NOT fabricate a Quantity just to satisfy the rule.
4. Go **one chapter at a time**, byte-identical, gate after each, commit per chapter.
   `run_unit_lane.py` now exists for the recurring clean sub-pattern
   `fmt(q.m_as(UNIT), suffix=" <unit>")` → `fmt_qty(q, UNIT)`. It reuses
   `lane_process` from `run_percent_lane.py` and queues plain-float suffix sites.
   Use it for the mechanical Quantity-backed lane, then inspect/refactor any
   queued sites by hand.
5. **Watch the prose:** after a `fmt_qty` migration the unit lives in the string, so
   any unit the prose used to add ("… GB", "… ms") must be deleted. Gate 2
   (`audit_prose_semantics`) will catch the resulting "5 GB GB"-style dups —
   trust it, and extend its `_UNIT` list if you introduce a unit it doesn't know.

### C. WS3 — `MarkdownStr` survivors (~328 sites)  ← medium effort, judgment-heavy
`MarkdownStr` is the escape hatch for genuinely non-numeric labels, sequences, and
compound expressions (see fmt.md §5). Many current uses are legitimate; some hide a
single numeric value that should be typed. Method:
1. Enumerate `MarkdownStr(` sites; classify: **range** (`"5–20 ms"`) → `fmt_range`;
   **single numeric value** wrapped in an f-string → the matching typed formatter;
   **legitimate label/sequence/equation** → leave.
2. Migrate ranges to `fmt_range(lo, hi, kind=…, unit=…)` (owns the en-dash, MIT
   style). Byte-identical-verify (the en-dash + spacing must match). Gate; commit.
3. Leave and briefly justify the genuine escape-hatch survivors.

### D. Phase 3B — PDF / `.tex` render verification (after the lanes above)
HTML render verification is already DONE for all previously-changed chapters. For
any chapter YOU change, do HTML (`./book/binder build html --volN volN/<ch>
--skip-hygiene --skip-validate`, then grep the value in the built HTML under
`book/quarto/_build/html-volN/contents/volN/<ch>/<ch>.html`). Then PDF:
```
python3 book/tools/audit/chapter_pdf_verify.py --vol1 <chapter>   # keeps the .tex
```
Read the kept `.tex`: confirm no literal `%`/`×`/`\$` leaks where a glyph should be,
no "??"/unresolved refs, no overfull-box explosions around changed lines. (Known
benign quirk: `$\times$` inside a figure caption renders fine in the visible
caption but appears as raw `\times` in the HTML `title=` tooltip — this is
pre-existing book-wide Quarto behavior, not a regression.)

### E. Phase 4 — lock it shut (final, once B/C are at an acceptable stopping point)
Flip the opt-in static gate to a global pre-commit blocker so regressions can't
return: `fmt_semantic_suffix` (a.k.a. `./book/binder check math --scope
suffix-semantics`) — set its default to enabled and wire it into pre-commit.
Then run a full-book `audit_lego_html.py` sweep (needs archived HTML via
`book/tools/audit/fmt/render_html.sh vol1` / `vol2`).

---

## 4. The per-change loop (apply to EVERY edit)
1. Read the cell + the prose that references its `_str` exports.
2. Make the typed-formatter edit; add the formatter to the cell's import if the
   chapter uses selective imports.
3. If a unit/glyph moved into the string, delete the now-duplicate unit/glyph from
   the prose.
4. Verify the rendered `_str` value (dump it via `assess_equiv.snapshot_file`):
   byte-identical for relocations; for a deliberate fix, confirm the new value +
   read the surrounding sentence.
5. Run the three gates (§2). Fix anything they flag.
6. Commit (chapter-sized batches). Keep `NIGHT_RESUME.md` current if you want clean
   resumability.

## 5. Definition of done (for the whole migration)
- `codemod_fmt queue` empty.
- Gates 1+2 = 0 across 81 chapters; test suite 100%.
- Every changed chapter HTML- and PDF-verified.
- `fmt_semantic_suffix` flipped to a global blocker (Phase 4).
- WS4/WS3 either migrated or each survivor has a one-line justification.

## 6. Landmines learned last session (save yourself the pain)
- `fmt()` has a **precision guard**: an integer-like value at `precision=1` raises
  "spurious .0", and a non-integer at `precision=0` raises "not integer-like".
  Match the original precision exactly when relocating.
- `fmt_percent` guards ratios to `[0, 1.5]`. For legitimate >150% or signed values
  pass `max_ratio=` / `allow_negative=True` explicitly (don't widen silently).
- `fmt_pp` now has `attributive=True` (hyphenated "N percentage-point") and
  auto singular/plural agreement on the rendered number. Use attributive ONLY when
  the value directly modifies a following noun ("a 5 percentage-point gap").
- The exact-match lanes only matched bare suffixes; **compound suffixes**
  (`"% annually"`, `"×/year"`) and `fmt(...)` calls inside `row.append(...)` (not a
  `_str =` assignment) slip past them — grep for these by hand.
- Class-qualified names matter: two classes can export the same bare `_str` name;
  the contract checker is class-aware, so keep prose refs fully qualified.
