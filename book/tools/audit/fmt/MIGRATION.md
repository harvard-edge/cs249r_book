# FMT Migration Map

> **Goal.** Every prose value in the book flows from a LEGO OUTPUT cell through a
> **typed, guarded formatter** (`fmt_usd`, `fmt_percent`, `fmt_pp`, `fmt_multiple`,
> `fmt_count`, `fmt_qty`, `fmt_ratio`, `fmt_range`) — never a raw `fmt(prefix=/suffix=)`
> carrying semantic meaning. The migration is **one-time, global, and must ship zero
> wrong numbers.** Authoritative rules: `.claude/rules/fmt.md`.

## Sources of truth (our rules DERIVE from these)

| Source | Location | What it dictates |
|---|---|---|
| **MIT Press copyeditor style sheet** | `~/Desktop/MIT_Press_Feedback/13_style_rules/data/style_sheet.txt` | **Spell out "percent"** in body (`%` ok in tables/eqns); **en-dash + all digits** in ranges (`1992–1993`); **abbreviate** units of measure |
| MIT Pass 03 (percent) | `~/Desktop/MIT_Press_Feedback/03_percent/PLAN.md` | ~2,650 `%`→"percent" annotations; the 3 contexts (prose, inline-py + `%`, string-internal `%`) |
| Existing LEGO verification plan | `~/Desktop/MLSysBook-LEGO-HTML-Verification-Plan.md` | L0–L5 layered audit + per-chapter loop we **extend, not replace** |
| `math.md §6 #14` | AIConfigs rules | multiplier glyph must be LaTeX `$\times$` in prose, never literal `×` |

**Percent target (MIT-grounded, overrides "preserve"):** computed body-prose percents
→ `fmt_percent(style='prose')` → `"71.1 percent"` (matches the hardcoded body prose,
already spelled out); tables/equations/captions → `style='symbol'` → `"71.1%"`.

**Coordination:** MIT Pass 03 percent work also exists on `shashank/feat/mitpress-vol1-copyedit-r1`
(not in `dev`, different convention — glyph in prose). **Decision: typed `fmt_percent` is
canonical**; that branch rebases/defers to it.

## Reuse, do not reinvent (decision: integrate)

The migration runs **inside** the existing audit stack:
`cell_exec.py` (exec), `audit_prose.py` (exec + ref-substitution → composite preview),
`audit_lego_html.py` (**ground truth**: every `{python}` ref vs rendered HTML),
`render_html.sh`, `chapter_html_verify.py`, the focal queues/ledgers in
`book/tools/audit/artifacts/`. New pieces are only: the typed formatters, the
`fmt_semantic_suffix` gate, `assess_equiv.py` (before/after value+prose **diff**, reuses
the above), the paired codemod, and `fmt_range`.

## Why this exists

`fmt(x, suffix=' percent')` and friends let a 0–1 ratio silently render as `8500 percent`.
Typed formatters move the unit *into the function*, where a guard can reject the
impossible (e.g. `fmt_percent` rejects a ratio outside 0–`max_ratio`). The danger is
**semantic** suffixes (percent / × / scale glyphs) that change the *meaning* of a number.
Unit suffixes (`GB`, `ms`, `W`) are only cosmetic — they get migrated too (full purity,
per decision), but **after** the dangerous ones and with far lower risk.

## Baseline (post-`dev`-sync audit — `/tmp/fmt_audit.json`)

| Metric | Count |
|---|---|
| Total fmt-family call sites | 6,611 |
| **Dangerous-semantic `suffix=`** (migrate first) | **870** |
| &nbsp;&nbsp;• percent (`%`,`percent`) → `fmt_percent` | 612 |
| &nbsp;&nbsp;• scale (`K`/`M`/`B`) → `fmt_count` | 152 |
| &nbsp;&nbsp;• multiplier (`×`,`x`) → `fmt_multiple` | 106 |
| Unit `suffix=` (`GB`,`ms`,`W`,…) → `fmt_qty`/`fmt_unit` (Phase 2) | 2,412 |
| Raw `MarkdownStr` (unguarded) → `fmt_range`/typed | 337 |
| `fmt_percent` **used today** | **10** |

The 612-vs-10 gap is the project. Regenerate with:
`python3 book/tools/audit/fmt/audit_fmt_usage.py --root book/quarto/contents --json /tmp/fmt_audit.json`

---

## Phases

### Phase 0 — Foundation
- [x] Commit typed formatters + `fmt_semantic_suffix` checker + audit tooling
- [x] Merge `dev` (the suffix-consolidation pass) into `fmt-fix` — clean
- [x] Re-audit → true baseline (above)
- [x] This ledger (you are here)
- [ ] Keep `fmt_semantic_suffix` **opt-in** during migration (run per-chapter in step 5); flip to a **global pre-commit blocker only at the end** (Phase 2) once the board is all DONE — flipping early would block every chapter commit.
- [x] **Production AST codemod** (`codemod_fmt.py`) — auto-rewrites only *provable* cases; queues ambiguous ones. Lanes: multiplier, percent (`%`/` percent`/`fmt_int`), scale-division.
- [x] **`fmt_range`** typed/guarded helper + tests
- [x] **Prose-unit duplication checker** (`fmt_prose_contract.py`, class-aware) — flags a unit/glyph typed after a ref that already owns it
- [x] **Lane drivers** with per-edit bisect + auto-revert gate (`run_multiplier_lane.py`, `run_percent_lane.py`→generic `lane_process`, `run_scale_lane.py`) plus the deliberate no-space style lane (`run_scale_style_lane.py`)

---

## Progress — live state (DANGEROUS-870 lane)

> **The dangerous-semantic suffixes are migrated corpus-wide.** Every change was
> accepted only by a **byte-identical gate** (rendered values AND visible-prose
> previews identical before/after, via `assess_equiv.snapshot_file`) or, for the
> few glyph-relocations, a transformation-aware gate; anything else was reverted
> and adjudicated. Re-verify any time with the sweeps in "How to re-verify" below.

| Value-kind | Status | How |
|---|---|---|
| **multiplier** (`×`/`x`/spaced/`fmt_int`) | **100% done** | `run_multiplier_lane.py` (byte-identical + `--variants` transform gate); glyph relocated to prose `$\times$` |
| **percent** (`%`/` percent`/`fmt_int`) | **100% done** | `run_percent_lane.py`; `fmt(x,'%')`→`fmt_percent(ratio, style=…)`, strips `*100`, `round(x)/100` for `fmt_int`. 5 signed/>100% sites adjudicated via `fmt_percent(allow_negative=, max_ratio=)` |
| **scale** (`K/M/B/T`, division form) | **clean cases done** (41 sites, 10 ch) | `run_scale_lane.py`; `fmt(x/MILLION,'M')`→`fmt_count(x, scale='M')` |
| scale (pre-scaled / lowercase `k` / spaced / `fmt_int`) | **100% done** | User ruled no-space house style; `run_scale_style_lane.py` migrated 44 queued sites to `fmt_count(raw, scale=…)` and one manual `fmt(...) + "B"` blind spot |

**Real bug caught & fixed by the audit:** `vol2/robust_ai` `acc_drop` (76−50 = 26
percentage *points*) rendered "26 percent" while prose appended "percentage points"
→ "26 percent percentage points". Fixed to a bare number. (commit `bc3729c676`)

**Verification status (whole corpus):** 81/81 chapters execute headlessly; `fmt_prose_contract`
**0 violations**; **`audit_prose_semantics` 0 findings**; multiplier + percent + scale 100% (0 queued);
`codemod_fmt.py queue` empty; targeted `pytest` 129 passing.

**Overnight session (semantic + consistency + render):**
- NEW gate `audit_prose_semantics.py` (+ 7 tests): executes each chapter, substitutes
  LEGO values into prose, normalizes LaTeX→visible, flags duplicated glyph/unit,
  abbr+spelled-word dup, mult-direction ("0.5× faster"), currency-as-percent,
  unresolved refs. Fixed real unit-dup bugs: data_storage "7.6 PB PB"/"PB petabytes",
  network_fabrics "2.56 MW megawatts".
- Killed 4 dangerous glyph-in-suffix stragglers the exact-match lanes missed
  (compound suffixes): "% annually", "×/year", 2× `row.append(fmt(p*100, suffix="%"))`.
- `fmt_pp` made grammatically complete (singular/plural + attributive hyphen mode);
  migrated 14 percentage-point sites to typed fmt_pp byte-identically across 6 chapters;
  fixed 2 attributive hyphen grammar bugs (model_serving). 4 pp sites left for an
  editorial call (documented in NIGHT_RESUME).
- **Phase 3A render verification DONE** for every changed chapter (HTML built + migrated
  value grepped in rendered output). See NIGHT_RESUME for the fig-cap `title=` tooltip
  `\times` note (pre-existing book-wide Quarto behavior; visible caption correct).
- Codex A1 scale-style pass DONE after user ruled for no-space scaled counts:
  migrated 44 queued scale sites to `fmt_count` plus one `fmt(...) + "B"` blind
  spot. This intentionally changes spacing/case only (`70 B`→`70B`, `270 K`→`270K`,
  `100k`→`100K`). HTML built and grepped for all 13 source-changed chapters;
  `codemod_fmt.py queue` is now empty.
- Resume checkpoint: `book/tools/audit/fmt/NIGHT_RESUME.md`.

### Active / next lanes, by priority
- **WS3 — MarkdownStr (337):** ranges → `fmt_range`; justify/replace the rest. *Judgment-heavy.*
- **WS4 — unit suffixes (~2,299): IN PROGRESS.** Added `run_unit_lane.py` for the
  clean Quantity-backed sub-pattern `fmt(q.m_as(UNIT), suffix=" UNIT")` →
  `fmt_qty(q, UNIT)`, accepted only by the existing byte-identical value+prose
  gate. First batch migrated 26 sites across 6 chapters; current dry-run reports
  235 remaining clean candidates across 20 chapters. The larger plain-float set
  (e.g. `weights_gb`) is still not mechanically migratable and should be left or
  refactored source-first.
- **WS2 — precision / spurious-`.0` re-sweep** (`audit_html.py`).
- **WS5 — prose-reference integrity** (`audit_lego_html.py`, ground truth vs rendered HTML).
- **WS6 — per-chapter semantic coherence** (scale queue resolved; continue with WS4/WS3 survivors).
- **Phase 3A/3B render verification**, then **Phase 4 lock** (flip `fmt_semantic_suffix` to a blocker).

### How to re-verify (any agent, from repo root, `PYTHONPATH=mlsysim`)
```
# every chapter still executes + values intact
python3 -c "import sys;sys.path.insert(0,'book/tools/audit/fmt');from pathlib import Path;from assess_equiv import snapshot_file;[print('FAIL',f) for f in Path('book/quarto/contents').rglob('*.qmd') if snapshot_file(f)[2]]"
# glyph-ownership contract (expect no output)
python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents
# rendered-composite semantic scan (expect "0 finding(s) ... CLEAN")
python3 book/tools/audit/fmt/audit_prose_semantics.py --root book/quarto/contents
# remaining dangerous suffixes by kind (expect by kind: {})
python3 book/tools/audit/fmt/codemod_fmt.py queue --root book/quarto/contents
# dry-run any lane to see what's left
python3 book/tools/audit/fmt/run_scale_lane.py --all
```

### Phase 1 — Dangerous-870 rollout (per-chapter SOURCE loop, hardest first)
Walk the board top-down: `training → data_selection → model_serving → benchmarking → …`.
This phase changes **source only** and verifies with the **no-build** gates
(`assess_equiv diff`, `audit_prose`, `fmt_semantic_suffix`). No rendering yet.

### Phase 2 — Unit-suffix purity
Codemod 2,412 unit suffixes → `fmt_qty`/`fmt_unit` (source only, low risk).

### Phase 3 — Render verification (AFTER migration; the "build it and look" phase)
> **Principle: never assume. Build it, open it, read it.** A green source diff is
> necessary but not sufficient — only the rendered artifact proves what ships. Done
> **one chapter at a time**, HTML first (fast, inspectable), then PDF (authoritative
> typeset). This phase runs once a chapter's source migration is complete.

**Stage A — HTML build + look**
1. Build the single chapter:
   `./book/binder build html --vol1 vol1/training --skip-hygiene --skip-validate`
   then archive: `book/quarto/_build/html-audit/vol1/training.html`
   (or batch via `render_html.sh vol1`).
2. **Automated:** `audit_html.py <html>` (spurious `.0`) **and**
   `audit_lego_html.py` (**ground truth** — every `{python}` ref value present in HTML).
3. **Look (mandatory, not optional):** open the HTML in the browser (cursor-ide-browser
   MCP), screenshot each migrated LEGO callout/section, and confirm by eye:
   body percents read "**X percent**", tables read "X%", multipliers show "6×" via
   `$\times$`, **no** doubled glyphs, **no** raw `` `{python}` ``, **no** leaked LaTeX,
   and the numbers tell a coherent story.
4. Gate: HTML scans green **and** the screenshots look right → proceed to Stage B.

**Stage B — PDF / TeX build + screenshots**
1. `python3 book/tools/audit/chapter_pdf_verify.py --vol1 training`
   → `binder build pdf --vol1 training`, **keeps `.tex`**, archives PDF + `.tex` under
   `book/quarto/_build/pdf-audit/`, updates `chapter_pdf_audit.{json,md}`.
2. **Read the `.tex`:** grep the changed refs in the keep-tex — confirm no literal `%`
   leaked into math mode, `$\times$` is intact, ranges use en-dash, no broken macros.
3. **Defect scan:** `binder check pdf` / `pdf_build_verify.py --vol1` (+ scan the Quarto
   render log for overfull boxes / warnings).
4. **Screenshot + look:** `pdftoppm -png -r 150 -f <pg> -l <pg> <pdf> /tmp/pg` on the
   pages with migrated values; open the PNGs and confirm typeset correctness (margins,
   spelled-out percent, `×`, tables, no overflow).
5. Sign off the chapter's PDF ledger row.

**Order across the book:** finish Stage A for a chapter before Stage B; finish the
high-risk chapters (board top) before the tail. The book is shippable only when every
chapter is HTML-verified, PDF-verified, and signed off.

### Phase 4 — Final lock
Full-book `audit_lego_html` sweep + corpus guard sweep; **flip `fmt_semantic_suffix` to a
global pre-commit blocker** (the regression lock).

---

## The per-chapter SOURCE loop (Phase 1/2 — no Quarto build needed)

1. **Baseline values** — `assess_equiv.py baseline --ref HEAD` captures the chapter's
   453-odd `*_str` exports + prose previews on real data (exec, not render).
2. **Codemod provable sites** (`x*100, suffix=' percent'` → `fmt_percent(x, style=…)`;
   `×`→`fmt_multiple`+prose `$\times$`; `K/M/B`→`fmt_count`). Ambiguous percent sites are
   **flagged, never auto-touched**.
3. **Human-resolve the queue** — read each ambiguous cell's compute; normalize **to a 0–1
   ratio at the source** so `fmt_percent` owns the ×100. *(Only judgment step.)*
4. **Fix the prose** — paired with each rewrite: strip now-duplicated glyphs/units; add
   `$\times$` for multipliers; choose percent `style` by context (body=prose, table=symbol).
5. **Value/prose equivalence** — `assess_equiv.py diff` (values **and** prose preview).
   Regime 1 must be byte-identical; Regime 2 identical after visible-text normalization;
   any other change is **adjudicated** (ASSESSMENT §5), never silent.
6. **Static gates green** — `binder check math` (canonical), `fmt_semantic_suffix`,
   `lego-dead-code`, prose-contract checker. (No build.)
7. **Commit** that one chapter's source; mark board status `source-done`.

→ Rendering and visual confirmation happen in **Phase 3** (build it and look), not here.

## Auditability (working back from "we caught the error")
1. **Source guards** — `fmt_percent` throws on out-of-range ratio at render time.
2. **Static gates** — suffix-semantics + canonical block old patterns at commit.
3. **Render-diff** — an unexplained magnitude change is the tripwire.
4. **This board** — the single source of truth for where every chapter stands.

---

## Per-chapter board

Status legend: `pending` → `source-done` (Phase 1/2: edits + `assess_equiv` clean + static gates) → `html-verified` (Phase 3A: built, scanned, screenshot-checked) → `pdf-verified` (Phase 3B: `.tex` read + PDF screenshots) → **`DONE`** (signed off)

Columns from baseline audit: **DGR** = dangerous suffixes (pct+mult+scale), **unit** = unit suffixes (Phase 2), **mds** = raw MarkdownStr.

| DGR | pct | mlt | scl | unit | mds | Chapter | Status |
|---:|---:|---:|---:|---:|---:|---|---|
| 86 | 43 | 23 | 20 | 205 | 21 | vol1/training | pending |
| 66 | 46 | 0 | 20 | 57 | 10 | vol1/data_selection | pending |
| 63 | 44 | 19 | 0 | 199 | 13 | vol1/model_serving | pending |
| 59 | 40 | 14 | 5 | 88 | 13 | vol1/benchmarking | pending |
| 58 | 21 | 16 | 21 | 96 | 5 | vol1/data_engineering | pending |
| 48 | 27 | 1 | 20 | 50 | 16 | vol1/responsible_engr | pending |
| 41 | 33 | 2 | 6 | 133 | 10 | vol2/distributed_training | pending |
| 36 | 19 | 0 | 17 | 70 | 6 | vol2/ops_scale | pending |
| 33 | 28 | 0 | 5 | 39 | 4 | vol1/ml_ops | pending |
| 32 | 31 | 0 | 1 | 65 | 1 | vol2/backmatter/appendix_fleet | pending |
| 30 | 30 | 0 | 0 | 3 | 1 | vol2/backmatter/appendix_c3 | pending |
| 23 | 21 | 0 | 2 | 56 | 5 | vol2/fleet_orchestration | pending |
| 19 | 16 | 0 | 3 | 6 | 0 | vol2/responsible_ai | pending |
| 18 | 17 | 1 | 0 | 33 | 4 | vol1/ml_workflow | pending |
| 18 | 17 | 0 | 1 | 47 | 13 | vol2/fault_tolerance | pending |
| 17 | 17 | 0 | 0 | 53 | 2 | vol2/performance_engineering | pending |
| 16 | 12 | 0 | 4 | 54 | 15 | vol2/sustainable_ai | pending |
| 15 | 14 | 0 | 1 | 75 | 7 | vol2/inference | pending |
| 15 | 8 | 0 | 7 | 19 | 1 | vol2/security_privacy | pending |
| 14 | 8 | 6 | 0 | 41 | 5 | vol1/introduction | pending |
| 13 | 1 | 12 | 0 | 8 | 2 | vol1/conclusion | pending |
| 12 | 11 | 0 | 1 | 16 | 22 | vol2/backmatter/appendix_assumptions | pending |
| 12 | 5 | 4 | 3 | 129 | 69 | vol1/ml_systems | pending |
| 11 | 9 | 2 | 0 | 94 | 4 | vol1/hw_acceleration | pending |
| 11 | 8 | 2 | 1 | 112 | 8 | vol2/data_storage | pending |
| 10 | 10 | 0 | 0 | 60 | 12 | vol1/model_compression | pending |
| 10 | 10 | 0 | 0 | 16 | 1 | vol2/backmatter/appendix_communication | pending |
| 10 | 7 | 0 | 3 | 74 | 4 | vol2/network_fabrics | pending |
| 10 | 6 | 0 | 4 | 52 | 1 | vol1/nn_architectures | pending |
| 10 | 5 | 4 | 1 | 74 | 1 | vol1/frameworks | pending |
| 8 | 8 | 0 | 0 | 174 | 2 | vol2/compute_infrastructure | pending |
| 7 | 5 | 0 | 2 | 22 | 14 | vol1/backmatter/appendix_machine | pending |
| 6 | 6 | 0 | 0 | 14 | 3 | vol2/introduction | pending |
| 6 | 6 | 0 | 0 | 2 | 2 | vol2/robust_ai | pending |
| 5 | 5 | 0 | 0 | 22 | 3 | vol2/backmatter/appendix_reliability | pending |
| 5 | 2 | 0 | 3 | 45 | 8 | vol1/nn_computation | pending |
| 4 | 4 | 0 | 0 | 11 | 2 | vol1/backmatter/appendix_algorithm | pending |
| 3 | 3 | 0 | 0 | 53 | 1 | vol2/collective_communication | pending |
| 3 | 3 | 0 | 0 | 4 | 0 | vol2/conclusion | pending |
| 2 | 2 | 0 | 0 | 27 | 6 | vol2/edge_intelligence | pending |
| 2 | 2 | 0 | 0 | 6 | 17 | vol1/backmatter/appendix_data | pending |
| 2 | 2 | 0 | 0 | 3 | 0 | vol2/backmatter/appendix_inference | pending |
| 1 | 0 | 0 | 1 | 5 | 3 | vol1/backmatter/appendix_assumptions | pending |

**Totals:** 43 files · 870 dangerous · 2,412 unit · 337 MarkdownStr.
