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
  book/tests/test_visible_text.py -q -o addopts=''
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

### B. WS4 — unit-suffix lane (~2,393 sites: `GB`/`ms`/`W`/`GB/s`/…)  ← the big one
**Risk: LOW** (a unit label can't cause a 0–1↔0–100 / 100× error). **Effort: HIGH**
and NOT a clean codemod, because ~1,938 of the args are plain floats (e.g.
`weights_gb`), not Pint Quantities, and `fmt_qty` requires a Pint Quantity to
generate the unit. So this is per-site, judgment-bearing source work. Method:

1. **Inventory & bucket** (don't brute-force):
   ```
   python3 book/tools/audit/fmt/audit_fmt_usage.py --root book/quarto/contents --json > /tmp/fmt_usage.json
   ```
   Group by suffix unit and by whether the argument is already a Pint Quantity.
2. **Quantity-backed sites → `fmt_qty`** (the clean, preferred case):
   `bw_str = fmt(bw.m_as(GB/second), suffix=" GB/s")` → `bw_str = fmt_qty(bw, GB/second)`.
   `fmt_qty` generates the suffix from the unit (always canonical), dimension-checks,
   and refuses currency. Then DROP any duplicate unit the prose was adding.
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
