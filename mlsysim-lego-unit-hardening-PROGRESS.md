# MLSysBook LEGO Unit Hardening — Progress

**Last updated:** 2026-06-01 (Phase 9 render + verification gates green)
**Branch:** `fmt-fix`
**Worktree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix`
**Plan spec:** [`mlsysim-lego-unit-hardening-plan.md`](mlsysim-lego-unit-hardening-plan.md)
**Coordinator execution plan:** [`mlsysim-lego-unit-hardening-agent-execution-plan.md`](mlsysim-lego-unit-hardening-agent-execution-plan.md)
**Latest code commit:** `771911d1e0` — Apply LEGO unit discipline across chapters.

**Next action:** Phase **10A** merge workflow: split/commit intended work, leave fmt-thread WIP and generated artifacts out of commits, then merge `dev` into `fmt-fix` and re-run the green gates.

**Resume phrase (new Agent chat):** `Resume LEGO hardening: PROGRESS.md next action. Branch fmt-fix, worktree MLSysBook-fmt-fix.`

---

## Agent protocol (paste into Composer at session start)

Use this block when starting or continuing work — same chat or new Agent window:

```
@mlsysim-lego-unit-hardening-PROGRESS.md
Execute from "Next action" through the end of the **current phase only**
(see Phase 8½ work queue). Do not ask for confirmation between sub-steps.

Rules:
- Work only in /Users/VJ/GitHub/MLSysBook-fmt-fix on branch fmt-fix.
- Do not touch sibling MLSysBook* worktrees or fmt-thread WIP (see Out of scope).
- Commit atomically per sub-step (8½-A1, 8½-A2, …); update this file after each commit.
- Run gates after each sub-step (pytest, pre-commit on touched files).
- If a hook fails: fix once and retry; if still failing, stop and report blocker.
- Stop only on: unrecoverable hook failure, merge conflict, or editorial ambiguity.
- No Phase 9 Quarto HTML/PDF renders unless the user explicitly requests builds.
- Do not commit lego_cells_verify_report.json or other generated audit artifacts.
```

**Session sizing:** one phase per chat when possible (8½-A in one session; 8½-B in 2–4 sessions by file batch). After context feels full, commit, update **Next action**, start fresh chat with resume phrase.

**Do not** run two Agent windows editing the same branch/chapters in parallel.

---

## Where we are (one sentence)

**`.m_as()` migration, Phase 8½ gates (G1–G6), deeper SSoT/quantity-flow/load-Pint pass, and Phase 9 full render verification are done. Remaining work is Phase 10 commit/merge verification.**

**Phase 9A render note:** Single-chapter `quarto render` exits **1** after `verify_rendered_xrefs.py` (scans entire `_build/html-vol1/`, not just the chapter). Treat as **expected** for isolated renders; 9A pass criteria are per-chapter: all cells Done, 0 `{python}` in that chapter's HTML, no traceback in HTML. Full `?@` xref gate is **9C** only.

**Kernel:** Use `-M jupyter:python3` (system Python 3.14 + fmt-fix editable `mlsysim`). Project `mlsysbook` kernel points at main checkout `.venv`.

**Current working doctrine:** MLSysIM is the source of truth; Pint quantities stay attached through calculations; display units are owned by typed/domain formatters. The active static/source queues are now clean, so render verification is the next definition-of-done layer.

---

## Merge-ready gates (Codex 2026-05-31)

All must pass before Phase 10. **Do not trust "lint 0 warnings" until G1 is fixed.**

| ID | Gate | Status | Notes |
|----|------|--------|-------|
| **G1** | L014 linter detects closed-name + `fmt()` | **PASS** | `L014_CLOSED_FMT` regex; regression test added |
| **G2** | `lego-units` baseline reflects real debt | **PASS** | **81 L014** allowlisted in `lego_units_baseline.json` (2026-05-31) |
| **G3** | `book_check_lego_prose_units.py` clean | **PASS** | 81/81 QMD files; checker scoped to closed-export duplicates |
| **G4** | Rate quantities keep dimensions through OUTPUT | **PASS** | Pilot fixed; L011 blocks magnitude-ratio rate reattach |
| **G5** | fmt precision defaults vs guard | **PASS** | `precision=None` → `_resolve_display_precision` on percent/range/pp |
| **G6** | Headless/rendered LEGO exec | **PASS** | 44/44 rendered LEGO chapters; 933/933 LEGO cells verified against HTML |
| **G7** | Phase 9 HTML + PDF renders | **PASS** | Vol I/II PDFs, full Vol I/II HTML, xref scans, and `{python}` leak scans green |
| **G8** | No accidental artifacts in commits | **PASS** | `lego_cells_verify_report.json` restored |

---

## Systematic checklist

### Layer A — mlsysim foundation (Steps 1–10) — DONE

| Step | Item | Status |
|-----:|------|--------|
| 1–10 | Units, aliases, `physics/quantities.py`, domain formatters, fmt fixes | **DONE** |

### Layer A′ — LOAD registry-first — DEFERRED

| Step | Item | Status |
|-----:|------|--------|
| A′-1 … A′-4 | Registry-first LOAD audit | **DONE for current gates** (`registry_sources`, prose literals, quantity-flow, load-Pint clean) |

### Layer B — lint + hooks wired (Steps 11–13) — DONE

| Step | Item | Status |
|-----:|------|--------|
| 11–13 | `lint_lego_units.py`, pre-commit, binder scope | **DONE** (but L014 broken — see G1) |

### Layer C — `.m_as()` migration (Steps 14+) — DONE

| Step | Item | Status |
|-----:|------|--------|
| 14+ | Bulk `.m_as()` → `.to().magnitude` (~1,235) | **DONE** |
| — | Glued cell fences (658 fixes) | **DONE** |
| — | OUTPUT/prose closed-open alignment | **DONE for current gates** (prose-units clean; quantity-flow all-cells clean) |

### Phase 8½ — Gate hardening — DONE

| Step | Item | Status | Command / file |
|-----:|------|--------|----------------|
| **8½-A** | Fix L014 match + regression test | **DONE** | `L014_CLOSED_FMT` in `lint_lego_units.py` |
| **8½-A** | Re-baseline after L014 fix | **DONE** | 81 L014 warnings allowlisted (burn-down deferred) |
| **8½-B** | Prose-units: 17 files → 0 | **DONE** | Checker fix + `sustainable_ai` OUTPUT/prose |
| **8½-B** | Pilot: `sustainable_ai.qmd` | **DONE** | L357 tonnes dup; L2553 table; LifecycleCarbonEstimate fmt_emissions |
| **8½-C** | Rate-quantity audit | **DONE** | `GpuEfficiencyTrajectoryRecap`; L011 added |
| **8½-D** | fmt precision defaults | **DONE** | `fmt_percent`, ranges, `fmt_pp` → auto precision |
| **8½-E** | Discard artifact diff | **DONE** | `git restore lego_cells_verify_report.json` |

### Phase 9 — render verification — DONE

Blocked on Phase 8½ — **cleared** (G1–G6 pass).

| Step | Item | Status |
|-----:|------|--------|
| 9A | HTML every chapter | **DONE** (44/44 LEGO chapters; 0 `{python}` leaks) |
| 9B | PDF every volume | **DONE** (Vol I + Vol II titlepage PDFs, exit 0) |
| 9C | Full volume HTML + PDF | **DONE** (Vol I/II HTML xref verifier clean; PDFs clean) |

### Wave 0–5 — SSoT + quantity-flow hardening — IN PROGRESS

| Wave | Item | Status |
|-----:|------|--------|
| 0 | Coordinator inventories: L014, scalar reattachment, `ureg.*`, source candidates, count scale risks | **DONE** (advisory checker added) |
| 1 | Pilot/source-hardening chapters | **DONE for current gates** |
| 2 | Vol I content chapter batches | **DONE for current gates** |
| 3 | Vol II content chapter batches | **DONE for current gates** |
| 4 | Appendices with LEGO | **DONE for current gates** |
| 5 | Global hardening: burn-down queues, central helpers/lints, full verification | **DONE for current gates** (static/source/headless/render gates clean) |

### Phase 10 — merge — NOT STARTED

| Step | Item | Status |
|-----:|------|--------|
| 10A | Merge `dev` → `fmt-fix` | **NOT STARTED** |
| 10B | Full re-verify | **NOT STARTED** |
| 10C | Merge `fmt-fix` → `dev` | **NOT STARTED** |

---

## Verification gates (honest status)

| Gate | Command | Result | Trust? |
|------|---------|--------|--------|
| Focused pytest | `pytest … test_fmt.py test_units_registry.py test_quantity_formulas.py test_lego_unit_invariants.py test_lint_lego_units.py test_lego_quantity_flow_audit.py -o addopts=` | **173 passed** (Codex) | ✓ |
| Full MLSysIM pytest | `PYTHONPATH=mlsysim pytest mlsysim/ -o addopts=` | **637 passed, 31 skipped** | ✓ |
| `lint_lego_units --fail-on warning` | `python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json` | 0 new warnings (81 baselined) | ✓ |
| `book_check_lego_prose_units` | `python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents` | **81/81 OK** | ✓ |
| `book_check_lego_quantity_flow` | default + `--all-cells` summary | **0 findings** | ✓ |
| `book_check_lego_load_pint` | `python3 book/tools/audit/book_check_lego_load_pint.py book/quarto/contents` | **81/81 OK** | ✓ |
| `book_check_registry_sources` | `python3 book/tools/audit/book_check_registry_sources.py book/quarto/contents` | **81/81 OK** | ✓ |
| `book_check_lego_prose_literals` | `python3 book/tools/audit/book_check_lego_prose_literals.py book/quarto/contents` | **81/81 OK** | ✓ |
| Headless exec | all `{python}` cells | **44 QMD files / 1,099 cells OK** | ✓ |
| Rendered LEGO verifier | `PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_cells.py` | **44/44 chapters; 933/933 LEGO cells PASS** | ✓ |
| fmt_semantic_suffix | per chapter | clean (Codex, sustainable_ai) | ✓ |
| math_canonical | binder | clean at tip | ✓ |
| Quarto render | Phase 9A HTML | **44/44 PASS** (0 `{python}`; xref verify exit 1 expected per isolated chapter) | ✓ |
| Quarto render | Full Vol I HTML | `_build/html-vol1/index.html`; 0 `{python}`, 0 `?@`, xrefs PASS | ✓ |
| Quarto render | Full Vol II HTML | `_build/html-vol2/index.html`; 0 `{python}`, 0 `?@`, xrefs PASS | ✓ |
| Quarto render | Vol I PDF | `_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf` (31 MB); log clean | ✓ |
| Quarto render | Vol II PDF | `_build/pdf-vol2/Machine-Learning-Systems-Vol2.pdf` (65 MB); log clean | ✓ |
| Pre-commit | `pre-commit run --all-files` | **PASS** | ✓ |

---

## Phase 8½ work queue (strict order)

```
8½-A1  Fix L014: line.replace(" ", "") → match =fmt( or re r'=\s*fmt\('     ✓ DONE
8½-A2  Add test_lint_lego_units regression for energy_kwh_str = fmt(...)     ✓ DONE
8½-A3  Re-run lint --fail-on warning; write new baseline (expect L014/L015 debt) ✓ DONE — 81 L014
8½-A4  Triage baseline: burn down or allowlist with dated notes                ✓ DONE — allowlist; burn-down with 8½-B

8½-B1  Pilot sustainable_ai.qmd (prose-units + Codex items)                  ✓ DONE
8½-B2  Fix remaining 16 files from book_check_lego_prose_units.py              ✓ DONE (checker scoped to closed dupes)
8½-B3  Consider wiring prose-units into pre-commit (or baseline)               DEFERRED

8½-C1  Fix GpuEfficiencyTrajectoryRecap (compute_infrastructure ~1815)       ✓ DONE
8½-C2  Grep audit: .magnitude/ patterns that reattach wrong unit             ✓ DONE — 1 hit fixed; 1 defer (distributed_training L470)
8½-C3  Optional: new lint rule for rate dimension loss                       ✓ DONE — L011

8½-D1  Decide fmt_percent / range default precision policy                    ✓ DONE — precision=None → auto
8½-D2  Implement + tests in test_fmt.py                                      ✓ DONE — 120 passed

8½-E   Discard artifact diff (lego_cells_verify_report.json)                 ✓ DONE
```

---

## Phase 9 render log (append as chapters complete)

### 9A HTML — DONE 2026-05-31

**Criteria:** all cells execute; 0 `{python}` in chapter HTML; no traceback in HTML.
**Note:** `quarto render` exits 1 on isolated chapters (post-render `verify_rendered_xrefs.py` scans full `_build/`). Ignore for 9A; xref gate is **9C**.

| Vol | Scope | Chapters | `{python}` leaks | Status |
|-----|-------|:--------:|:----------------:|:------:|
| 1 | content | 16 | 0 | ✓ |
| 1 | appendices | 4 | 0 | ✓ |
| 2 | content | 17 | 0 | ✓ |
| 2 | appendices | 7 | 0 | ✓ |
| **Total** | | **44** | **0** | **PASS** |

**Render command:**
```bash
cd book/quarto
ln -sf config/_quarto-html-vol1.yml _quarto.yml   # or vol2
MPLBACKEND=Agg quarto render contents/vol<N>/<chapter>/<chapter>.qmd --to html -M jupyter:python3
# Output: _build/html-vol<N>/contents/vol<N>/...
```

**Logs:** `/private/tmp/mlsysbook-unit-hardening/phase9a-vol1-batch3.log`, `phase9a-vol2-batch1.log`, appendices logs.

### 9B PDF — DONE

| Vol | Artifact | Size | Exit | Date |
|-----|----------|------|:----:|------|
| 1 | `_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf` | 31 MB | 0 | 2026-05-31 |
| 2 | `_build/pdf-vol2/Machine-Learning-Systems-Vol2.pdf` | 65 MB | 0 | 2026-05-31 |

**Command used:**
```bash
cd book/quarto && ln -sf config/_quarto-pdf-vol1.yml _quarto.yml
test -L index.qmd || ln -sf index-vol1.qmd index.qmd
PYTHONPATH=/Users/VJ/GitHub/MLSysBook-fmt-fix/mlsysim:/Users/VJ/GitHub/MLSysBook-fmt-fix \
MPLBACKEND=Agg quarto render --to titlepage-pdf
```
**Logs:** `/private/tmp/mlsysbook-unit-hardening/vol1-pdf-full.log`, `/private/tmp/mlsysbook-unit-hardening/vol2-pdf-full.log`

### 9C Full volume — DONE

| Vol | HTML artifact | Xrefs | `{python}` leaks | Status |
|-----|---------------|-------|------------------|--------|
| 1 | `_build/html-vol1/index.html` | 0 unresolved | 0 | PASS |
| 2 | `_build/html-vol2/index.html` | 0 unresolved | 0 | PASS |

**Logs:** `/private/tmp/mlsysbook-unit-hardening/vol1-html-full.log`, `/private/tmp/mlsysbook-unit-hardening/vol2-html-full.log`

---

## Codex findings log (2026-05-31)

| # | Severity | Finding | Verified | Fix phase |
|---|----------|---------|----------|-----------|
| 1 | High | L014 false negative (`= fmt(` vs `=fmt(`) | **Yes** — reproduced | 8½-A |
| 2 | High | prose-units fails 17 files | **Yes** | 8½-B |
| 3 | High/Med | TFLOP/s÷W stored as TFLOP/s only | **Yes** — `compute_infrastructure.qmd:1815` | 8½-C |
| 4 | Med | fmt precision defaults vs guard | **Yes** — `fmt_percent` default 1 | 8½-D |
| 5 | Med | sustainable_ai L357 tonnes dup; L2553 math+W | **Yes** | 8½-B pilot |
| 6 | Hygiene | lego_cells_verify_report.json accidental | **Yes** — unstaged | 8½-E |

**Codex assessment we agree with:** architecture direction is right; branch is **not merge-ready** until G1–G3 (+ renders).

---

## Render cleanup categories (for Phase 8½-B / 9A)

When fixing a chapter, apply all three in the same commit as OUTPUT changes:

| Category | Rule | Example |
|----------|------|---------|
| **Duplicate unit (L015)** | Closed export owns unit → bare `{python}` ref | `280 t` + `tonnes CO₂` → ref only |
| **Math vs prose** | No `_str` with unit glyph inside `$...$`/`$$...$$` | Use open magnitude or `_math` export |
| **Name honesty** | `*_kg_str` must render kg (pin `unit=`) or rename | `fmt_emissions(..., unit=kilogram)` |

---

## Prose-units failing files (17 — 2026-05-31 snapshot)

Refresh: `python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents`

| # | File |
|---|------|
| 1 | `vol1/ml_ops/ml_ops.qmd` |
| 2 | `vol1/model_compression/model_compression.qmd` |
| 3 | `vol1/model_serving/model_serving.qmd` |
| 4 | `vol1/nn_computation/nn_computation.qmd` |
| 5 | `vol1/training/training.qmd` |
| 6 | `vol2/backmatter/appendix_assumptions.qmd` |
| 7 | `vol2/backmatter/appendix_c3.qmd` |
| 8 | `vol2/backmatter/appendix_fleet.qmd` |
| 9 | `vol2/backmatter/appendix_reliability.qmd` |
| 10 | `vol2/collective_communication/collective_communication.qmd` |
| 11 | `vol2/compute_infrastructure/compute_infrastructure.qmd` |
| 12 | `vol2/fault_tolerance/fault_tolerance.qmd` |
| 13 | `vol2/fleet_orchestration/fleet_orchestration.qmd` |
| 14 | `vol2/inference/inference.qmd` |
| 15 | `vol2/ops_scale/ops_scale.qmd` |
| 16 | `vol2/robust_ai/robust_ai.qmd` |
| 17 | `vol2/sustainable_ai/sustainable_ai.qmd` ← **pilot first** |

---

## Closure commits (2026-05-31)

| SHA | Subject |
|-----|---------|
| `758e5ddfce` | Clear lego-units lint debt and add closure codemods |
| `f1bbdfd9ed` | Exempt Pint USD in LEGO cells from currency prose check |
| `1270f364d1` | Fix LEGO closure regressions across Vol I and Vol II |
| `89c287556f` | Enable lego-units as a default binder validation scope |

---

## Out of scope / do not commit

| Path | Action |
|------|--------|
| `book/tools/audit/artifacts/lego_cells_verify_report.json` | `git restore` — partial NO_HTML regen |
| `book/tools/audit/fmt/audit_fmt_usage.py` | fmt-thread WIP |
| `book/tools/audit/fmt/fmt_prose_contract.py` | fmt-thread WIP |
| `mlsysim-domain-formatting-plan.md` | separate thread |

---

## Step log

### Closure — 2026-05-31 — pre-build (partial)

- **Commits:** `758e5ddfce` … `89c287556f`
- **Gate:** exec 81/81, pytest — pass
- **Gate:** lint 0 warnings — **not sufficient** (L014 broken)
- **Gate:** prose-units — **fail** (17 files)
- **Status:** DONE for migration; **NOT merge-ready**

### Phase 8½-D — fmt precision defaults — 2026-05-31

- **Policy:** `precision=None` (new default) delegates to `_resolve_display_precision`, matching `fmt_multiple`.
- **Scope:** `fmt_percent`, `fmt_percent_range`, `fmt_qty_range`, `fmt_time_range`, `fmt_pp`.
- **Fixes:** `fmt_percent(0.85)` no longer raises spurious-trailing-zero error; integer-like percentages render without decimals.
- **Gate:** G5 PASS; `test_fmt.py` 120 passed; no chapter OUTPUT changes required.

### Phase 8½-C — rate-quantity integrity — 2026-05-31

- **Fix:** `GpuEfficiencyTrajectoryRecap` — `peak_flops / tdp` with `TFLOPs/second/watt` (matches `ops_scale` `GpuEfficiencyTableRecap`).
- **Audit:** one other `.magnitude/` ratio in `distributed_training.qmd:470` (bytes/bw → ms via `* THOUSAND`); deferred — not TFLOP/s÷W class.
- **Lint:** L011 blocks magnitude-ratio + `* TFLOP/second` without preserved denominator.
- **Gate:** G4 PASS.

### Phase 8½-B — prose-units clean — 2026-05-31

- **Checker:** `book_check_lego_prose_units.py` now flags only closed-export duplicate units (not open `fmt`/`fmt_int` + prose unit).
- **Pilot:** `sustainable_ai.qmd` — removed duplicate tonnes/CO₂; simplified rack-power table row; migrated `LifecycleCarbonEstimate` tonnes exports to `fmt_emissions`.
- **Gate:** G3 PASS — 81/81 QMD files clean.

### Phase 8½-A — L014 fix + honest baseline — 2026-05-31

- **Change:** Replace broken CLOSED_UNIT_SUFFIX loop with `L014_CLOSED_FMT` regex; add `test_l014_closed_name_uses_fmt`; refresh baseline.
- **Baseline triage:** 81 L014 (closed `*_unit_str = fmt(...)`). All allowlisted — burn-down deferred to post-8½-B OUTPUT fixes (same cells often need domain formatter migration).
- **Gate:** G1 PASS, G2 PASS; `pytest test_lint_lego_units.py` 7 passed; lint with baseline 0 new warnings.

### Phase 9A — HTML chapter renders — 2026-05-31

- **Scope:** All 44 vol1+vol2 chapters with LEGO cells (16+4 vol1 content/appendices; 17+7 vol2).
- **Kernel:** `-M jupyter:python3` (fmt-fix editable `mlsysim` on system Python 3.14).
- **Result:** 0 `{python}` leaks across all rendered HTML; all cells executed without ImportError/traceback.
- **Gate:** G7 partial — 9A PASS; 9B/9C remain.
