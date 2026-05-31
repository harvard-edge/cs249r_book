# MLSysBook LEGO Unit Hardening — Progress

**Last updated:** 2026-05-31 (Phase 8½-A complete)
**Branch:** `fmt-fix`
**Worktree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix`
**Plan spec:** [`mlsysim-lego-unit-hardening-plan.md`](mlsysim-lego-unit-hardening-plan.md)
**Last commit:** *(pending — Phase 8½-A)*

**Next action:** Phase **8½-B1** — pilot `sustainable_ai.qmd` via `book_check_lego_prose_units.py` (L357 tonnes dup; L2553 `$...$` + W).

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

**`.m_as()` migration and infra are done, but the branch is not merge-ready — Phase 8½ must fix broken lint gates and OUTPUT/prose contract before Phase 9 renders.**

---

## Merge-ready gates (Codex 2026-05-31)

All must pass before Phase 10. **Do not trust "lint 0 warnings" until G1 is fixed.**

| ID | Gate | Status | Notes |
|----|------|--------|-------|
| **G1** | L014 linter detects closed-name + `fmt()` | **PASS** | `L014_CLOSED_FMT` regex; regression test added |
| **G2** | `lego-units` baseline reflects real debt | **PASS** | **81 L014** allowlisted in `lego_units_baseline.json` (2026-05-31) |
| **G3** | `book_check_lego_prose_units.py` clean | **FAIL** | **17 files** with duplicate units / math-span issues |
| **G4** | Rate quantities keep dimensions through OUTPUT | **PARTIAL** | Pilot: `compute_infrastructure.qmd:1815` TFLOP/s÷W → TFLOP/s only |
| **G5** | fmt precision defaults vs guard | **OPEN** | `fmt_percent(0.85)`, range helpers default `precision=1` |
| **G6** | Headless cell exec | **PASS** | 81/81 `.qmd` files |
| **G7** | Phase 9 HTML + PDF renders | **NOT STARTED** | — |
| **G8** | No accidental artifacts in commits | **WATCH** | `lego_cells_verify_report.json` unstaged partial regen — discard |

---

## Systematic checklist

### Layer A — mlsysim foundation (Steps 1–10) — DONE

| Step | Item | Status |
|-----:|------|--------|
| 1–10 | Units, aliases, `physics/quantities.py`, domain formatters, fmt fixes | **DONE** |

### Layer A′ — LOAD registry-first — DEFERRED

| Step | Item | Status |
|-----:|------|--------|
| A′-1 … A′-4 | Registry-first LOAD audit | **NOT DONE** |

### Layer B — lint + hooks wired (Steps 11–13) — DONE

| Step | Item | Status |
|-----:|------|--------|
| 11–13 | `lint_lego_units.py`, pre-commit, binder scope | **DONE** (but L014 broken — see G1) |

### Layer C — `.m_as()` migration (Steps 14+) — DONE

| Step | Item | Status |
|-----:|------|--------|
| 14+ | Bulk `.m_as()` → `.to().magnitude` (~1,235) | **DONE** |
| — | Glued cell fences (658 fixes) | **DONE** |
| — | OUTPUT/prose closed-open alignment | **NOT DONE** (Phase 8½-B) |

### Phase 8½ — Gate hardening — IN PROGRESS ← **CURRENT**

| Step | Item | Status | Command / file |
|-----:|------|--------|----------------|
| **8½-A** | Fix L014 match + regression test | **DONE** | `L014_CLOSED_FMT` in `lint_lego_units.py` |
| **8½-A** | Re-baseline after L014 fix | **DONE** | 81 L014 warnings allowlisted (burn-down deferred) |
| **8½-B** | Prose-units: 17 files → 0 | **NOT STARTED** | `book_check_lego_prose_units.py book/quarto/contents` |
| **8½-B** | Pilot: `sustainable_ai.qmd` | **NOT STARTED** | L357 tonnes dup; L2553 `$...$` + W |
| **8½-C** | Rate-quantity audit | **NOT STARTED** | Start `compute_infrastructure.qmd:1815` |
| **8½-D** | fmt precision defaults | **NOT STARTED** | `fmt_percent`, `fmt_*_range` |
| **8½-E** | Discard artifact diff | **NOT STARTED** | `git restore lego_cells_verify_report.json` |

### Phase 9 — render verification — NOT STARTED

Blocked on Phase 8½-A/B minimum (renders will otherwise duplicate Codex findings).

| Step | Item | Status |
|-----:|------|--------|
| 9A | HTML every chapter | **NOT STARTED** |
| 9B | PDF every chapter | **NOT STARTED** |
| 9C | Full volume HTML + PDF | **NOT STARTED** |

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
| Focused pytest | `pytest … test_fmt.py test_quantity_formulas.py test_lego_unit_invariants.py test_lint_lego_units.py -o addopts=` | **171 passed** (Codex) | ✓ |
| `lint_lego_units --fail-on warning` | `python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json` | 0 new warnings (81 baselined) | ✓ |
| `book_check_lego_prose_units` | `python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents` | **17 files FAIL** | ✓ trust this |
| Headless exec | all `{python}` cells | **81/81 OK** | ✓ |
| fmt_semantic_suffix | per chapter | clean (Codex, sustainable_ai) | ✓ |
| math_canonical | binder | clean at tip | ✓ |
| Quarto render | Phase 9 | not run | — |

---

## Phase 8½ work queue (strict order)

```
8½-A1  Fix L014: line.replace(" ", "") → match =fmt( or re r'=\s*fmt\('     ✓ DONE
8½-A2  Add test_lint_lego_units regression for energy_kwh_str = fmt(...)     ✓ DONE
8½-A3  Re-run lint --fail-on warning; write new baseline (expect L014/L015 debt) ✓ DONE — 81 L014
8½-A4  Triage baseline: burn down or allowlist with dated notes                ✓ DONE — allowlist; burn-down with 8½-B

8½-B1  Pilot sustainable_ai.qmd (prose-units + Codex items)                  ← NEXT
8½-B2  Fix remaining 16 files from book_check_lego_prose_units.py
8½-B3  Consider wiring prose-units into pre-commit (or baseline)

8½-C1  Fix GpuEfficiencyTrajectoryRecap (compute_infrastructure ~1815)
8½-C2  Grep audit: .magnitude/ patterns that reattach wrong unit
8½-C3  Optional: new lint rule for rate dimension loss

8½-D1  Decide fmt_percent / range default precision policy
8½-D2  Implement + tests in test_fmt.py

── then Phase 9A (HTML pilot chapter → manifest order) ──
```

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

## Phase 9 render log (append as chapters complete)

| Vol | Chapter | HTML | PDF | Notes | Date |
|-----|---------|:----:|:---:|-------|------|
| — | — | — | — | Blocked on Phase 8½ | — |

---

## Step log

### Closure — 2026-05-31 — pre-build (partial)

- **Commits:** `758e5ddfce` … `89c287556f`
- **Gate:** exec 81/81, pytest — pass
- **Gate:** lint 0 warnings — **not sufficient** (L014 broken)
- **Gate:** prose-units — **fail** (17 files)
- **Status:** DONE for migration; **NOT merge-ready**

### Phase 8½-A — L014 fix + honest baseline — 2026-05-31

- **Change:** Replace broken CLOSED_UNIT_SUFFIX loop with `L014_CLOSED_FMT` regex; add `test_l014_closed_name_uses_fmt`; refresh baseline.
- **Baseline triage:** 81 L014 (closed `*_unit_str = fmt(...)`). All allowlisted — burn-down deferred to post-8½-B OUTPUT fixes (same cells often need domain formatter migration).
- **Gate:** G1 PASS, G2 PASS; `pytest test_lint_lego_units.py` 7 passed; lint with baseline 0 new warnings.

### Codex review — 2026-05-31 — plan revision

- **Action:** Added Phase 8½ to plan + PROGRESS; revised merge-ready gates
- **Next:** Phase 8½-A (L014 fix)
