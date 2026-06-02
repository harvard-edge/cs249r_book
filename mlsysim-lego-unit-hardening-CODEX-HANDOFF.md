# MLSysBook LEGO Unit Hardening — Codex Handoff Plan

**Written:** 2026-06-01 (after Phase 9B/9C completion)
**Audience:** Codex (or any agent) picking up from here
**Authoritative trackers:** [`mlsysim-lego-unit-hardening-PROGRESS.md`](mlsysim-lego-unit-hardening-PROGRESS.md), [`mlsysim-lego-unit-hardening-plan.md`](mlsysim-lego-unit-hardening-plan.md)

---

## 0. Resume phrase (paste at session start)

```
@mlsysim-lego-unit-hardening-CODEX-HANDOFF.md
@mlsysim-lego-unit-hardening-PROGRESS.md

Work in /Users/VJ/GitHub/MLSysBook-fmt-fix on branch fmt-fix only.
Continue from "Immediate next actions" below through Phase 10.
Do not stop between sub-steps unless blocked. Do not commit fmt-thread WIP or generated audit artifacts.
```

---

## 1. Executive summary (30 seconds)

| Item | Status |
|------|--------|
| **Goal** | Harden LEGO unit discipline: Pint `.to().magnitude`, domain formatters, lint gates, render truth |
| **Branch** | `fmt-fix` |
| **Worktree** | `/Users/VJ/GitHub/MLSysBook-fmt-fix` |
| **Latest code commit** | `771911d1e0` — Apply LEGO unit discipline across chapters |
| **Migration + Phase 8½** | **DONE** (G1–G6, G8 pass) |
| **Phase 9A HTML** | **DONE** — 44/44 LEGO chapters, 0 `{python}` leaks |
| **Phase 9B PDF** | **DONE** — Vol I + Vol II titlepage PDFs, clean logs |
| **Phase 9C** | **DONE** — full-volume HTML Vol I/II, xrefs clean, 0 `{python}` leaks |
| **Phase 10 merge** | **NOT STARTED** — commit/merge workflow remains |
| **Merge-ready?** | **SOURCE/RENDER GATES GREEN** — needs clean commit split + dev merge verification |

**One sentence:** All source, LEGO, PDF, and full-volume HTML gates are green in `fmt-fix`; next split/commit intended work, exclude fmt-thread WIP/artifacts, then sync and merge to `dev` (Phase 10).

---

## 2. Worktree and branch rules (non-negotiable)

| Rule | Detail |
|------|--------|
| **Stay in** | `/Users/VJ/GitHub/MLSysBook-fmt-fix` only |
| **Branch** | `fmt-fix` — do not create new branches unless user asks |
| **Do not touch** | Sibling `MLSysBook*` worktrees (no `cd`, no `git` there) |
| **Do not commit** | `book/tools/audit/artifacts/lego_cells_verify_report.json` |
| **Do not commit** | fmt-thread WIP: `book/tools/audit/fmt/audit_fmt_usage.py`, `fmt_prose_contract.py` |
| **Do not commit** | `mlsysim-domain-formatting-plan.md` (separate thread) |
| **Atomic commits** | One logical change per commit; `git add <specific-files>` never `-A` |
| **No co-author tags** | No `Co-Authored-By`, no vendor footers in commit messages |
| **Do not** | `gh workflow run *-validate-dev.yml` after push (push already triggers CI) |

---

## 3. What is DONE (do not redo)

### 3.1 Layer A — mlsysim foundation

- Pint unit aliases, `physics/quantities.py`, domain formatters (`fmt_emissions`, `fmt_memory`, etc.)
- `fmt.py`: unified `fmt_*` family, `_resolve_display_precision`, spurious-zero guard
- Single source of truth: all physical numbers trace to MLSysIM registries

### 3.2 Layer B — lint + hooks

- `book/tools/scripts/lint_lego_units.py` — L001–L019 rules
- Pre-commit binder scope `lego-units` default=True (`89c287556f`)
- `book/tools/tests/test_lint_lego_units.py` — regression tests

### 3.3 Layer C — `.m_as()` migration

- ~1,235 `.m_as()` → `.to(<unit>).magnitude` across 81 QMD files
- 658 glued-cell fence fixes
- Headless exec: **81/81 OK**

### 3.4 Phase 8½ — gate hardening (all sub-steps DONE)

| Sub | What | Key commit / file |
|-----|------|-------------------|
| **8½-A** | L014 false negative fix | `5c2d9bfcd2` — `L014_CLOSED_FMT` regex in `lint_lego_units.py`; baseline 81 L014 allowlisted |
| **8½-B** | Prose-units gate | `e6a3636bfa` — checker scoped to closed-export dupes only; `sustainable_ai.qmd` pilot |
| **8½-C** | Rate-quantity integrity | `e7c3f7e34f` — `GpuEfficiencyTrajectoryRecap`; L011 lint rule |
| **8½-D** | fmt precision defaults | `8a4bc970e6` — `fmt_percent`, ranges, `fmt_pp` default `precision=None` |
| **8½-E** | Artifact hygiene | `git restore lego_cells_verify_report.json` |

### 3.5 Phase 9A — HTML per chapter (DONE 2026-05-31)

**44 chapters rendered, 0 `{python}` leaks:**

| Vol | Scope | Count |
|-----|-------|------:|
| 1 | content chapters | 16 |
| 1 | appendices with LEGO | 4 |
| 2 | content chapters | 17 |
| 2 | appendices with LEGO | 7 |

**Output locations:**
- Vol I: `book/quarto/_build/html-vol1/contents/vol1/.../*.html`
- Vol II: `book/quarto/_build/html-vol2/contents/vol2/.../*.html`

**Logs (local temp, may persist):**
- `/private/tmp/mlsysbook-unit-hardening/phase9a-vol1-batch3.log`
- `/private/tmp/mlsysbook-unit-hardening/phase9a-vol2-batch1.log`
- `/private/tmp/mlsysbook-unit-hardening/phase9a-vol1-appendices.log`
- `/private/tmp/mlsysbook-unit-hardening/phase9a-vol2-appendices.log`

---

## 4. Merge-ready gates (G1–G8)

Run these **before claiming merge-ready**. Update PROGRESS.md when status changes.

| ID | Gate | Status | Verify command |
|----|------|--------|----------------|
| **G1** | L014 detects closed-name + `fmt()` | **PASS** | `python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json` |
| **G2** | Baseline honest (81 L014 allowlisted) | **PASS** | Inspect `book/tools/audit/lego_units_baseline.json` |
| **G3** | Prose-units clean | **PASS** | `python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents` → 81/81 OK |
| **G4** | Rate quantities keep dimensions | **PASS** | L011 in lint; `GpuEfficiencyTrajectoryRecap` fixed |
| **G5** | fmt precision defaults | **PASS** | `pytest mlsysim/tests/test_fmt.py -o addopts=` → 120+ pass |
| **G6** | Headless cell exec | **PASS** | LEGO cell verify / exec harness → 81/81 |
| **G7** | Phase 9 renders | **PASS** | Vol I/II PDFs + full HTML + xref/leak scans clean |
| **G8** | No accidental artifacts | **PASS** | `git status` — no staged audit JSON |

**Focused pytest (fast sanity):**
```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
pytest mlsysim/tests/test_fmt.py \
       mlsysim/tests/test_quantity_formulas.py \
       mlsysim/tests/test_lego_unit_invariants.py \
       book/tools/tests/test_lint_lego_units.py \
       -o addopts=
```

---

## 5. Environment setup (read before any render)

### 5.1 Python / Jupyter kernel — CRITICAL

| Kernel | Path | Problem |
|--------|------|---------|
| `mlsysbook` (project default) | `/Users/VJ/GitHub/MLSysBook/.venv/bin/python` | Loads **main checkout** `mlsysim` — lacks fmt-fix changes (e.g. `fmt_memory`) |
| `python3` (override) | System Python 3.14 + fmt-fix editable | **Use this for all renders** |

**Always pass:**
```bash
-M jupyter:python3
```

**Ensure fmt-fix mlsysim is editable-installed on that Python:**
```bash
python3 -c "import mlsysim; print(mlsysim.__file__)"
# Must point to: .../MLSysBook-fmt-fix/mlsysim/mlsysim/...
# If not:
pip install -e /Users/VJ/GitHub/MLSysBook-fmt-fix/mlsysim
```

### 5.2 Quarto config symlinks

From `book/quarto/`:

| Build | Symlink | Format flag |
|-------|---------|---------------|
| Vol I HTML | `ln -sf config/_quarto-html-vol1.yml _quarto.yml` | `--to html` |
| Vol II HTML | `ln -sf config/_quarto-html-vol2.yml _quarto.yml` | `--to html` |
| Vol I PDF | `ln -sf config/_quarto-pdf-vol1.yml _quarto.yml` | **`--to titlepage-pdf`** (NOT `--to pdf`) |
| Vol II PDF | `ln -sf config/_quarto-pdf-vol2.yml _quarto.yml` | **`--to titlepage-pdf`** |

**PDF pitfall (already hit once):** `--to pdf` skips `include-in-header: tex/header-includes.tex`, so `\usepackage{lettrine}` is missing → `\lettrine` undefined control sequence. Always use `titlepage-pdf`.

### 5.3 Book-type PDF requires index symlink

```bash
cd book/quarto
test -L index.qmd || ln -sf index-vol1.qmd index.qmd   # vol1 PDF
# vol2: ln -sf index-vol2.qmd index.qmd
```

### 5.4 Execute environment

PDF config sets `execute.env.PYTHONPATH: "../..:../../mlsysim"`. HTML configs rely on kernel + project. Still set:
```bash
MPLBACKEND=Agg
```

### 5.5 TeX

Quarto uses **TinyTeX** at `~/Library/TinyTeX/`. `lettrine` is installed there. Full TeX Live at `/usr/local/texlive/2026/` is separate — do not assume Quarto uses it.

---

## 6. Phase 9 — remaining work (detailed)

### 6.1 Phase 9A — HTML — ✅ DONE

**Pass criteria (per chapter):**
1. All `{python}` cells show `Done` in log (no ImportError/Traceback)
2. `rg '{python}' chapter.html` → 0 matches
3. `rg -i 'traceback|importerror' chapter.html` → 0 real failures (quiz prose mentioning "traceback" is OK)

**Expected failure (ignore for 9A):**
- Exit code 1 from `scripts/verify_rendered_xrefs.py` post-render hook
- Reason: hook scans **entire** `_build/html-volN/`, not just the chapter rendered
- Unresolved `?@sec-...` refs are normal for isolated chapter renders
- **Full xref gate is 9C only**

**Per-chapter HTML command:**
```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto
ln -sf config/_quarto-html-vol1.yml _quarto.yml   # or vol2
MPLBACKEND=Agg quarto render contents/vol1/introduction/introduction.qmd \
  --to html -M jupyter:python3 2>&1 | tee /private/tmp/mlsysbook-unit-hardening/chapter.log || true
html="_build/html-vol1/contents/vol1/introduction/introduction.html"
rg '{python}' "$html" || echo "OK: no leaks"
```

**Batch script pattern (use `|| true` on quarto — exit 1 is xref noise):**
```bash
set +e
for qmd in contents/vol1/ml_systems/ml_systems.qmd ...; do
  MPLBACKEND=Agg quarto render "$qmd" --to html -M jupyter:python3 >> LOG 2>&1 || true
  html="_build/html-vol1/${qmd%.qmd}.html"
  rg -q '{python}' "$html" && echo "LEAK $qmd" || echo "OK $qmd"
done
```

**Do not use `set -e` with `grep -c` / `rg -c` in post-check loops** — zero-match exit codes kill the loop early (bug already encountered).

---

### 6.2 Phase 9B — PDF — ✅ DONE

#### Reality check: book-type PDF ≠ per-chapter PDF

When you run:
```bash
quarto render contents/vol1/introduction/introduction.qmd --to titlepage-pdf
```
with `_quarto-pdf-vol1.yml` (project `type: book`), Quarto renders **the entire book** (35 files for vol1), not one chapter. The plan's per-chapter PDF wording is aspirational; **practical 9B path:**

**Option A (recommended):** Treat 9B as **full-volume PDF builds** (one per volume) + LaTeX log scan. This satisfies "all chapters PDF-green" because the book PDF includes every chapter.

**Option B:** True per-chapter PDF would need a standalone/minimal `_quarto.yml` without book project type — not set up; do not invent unless user requests.

#### Vol I PDF — done

**Command that was running:**
```bash
mkdir -p /private/tmp/mlsysbook-unit-hardening
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto
ln -sf config/_quarto-pdf-vol1.yml _quarto.yml
test -L index.qmd || ln -sf index-vol1.qmd index.qmd
LOG=/private/tmp/mlsysbook-unit-hardening/vol1-pdf-full.log
MPLBACKEND=Agg quarto render --to titlepage-pdf -M jupyter:python3 >> "$LOG" 2>&1
```

**Check status:**
```bash
# Still running?
ps aux | grep -E 'quarto render|lualatex' | grep -v grep

# Finished?
test -f book/quarto/_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf && ls -lh book/quarto/_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf

# Log tail
tail -30 /private/tmp/mlsysbook-unit-hardening/vol1-pdf-full.log
rg 'ERROR|Undefined control sequence|Traceback|ImportError' /private/tmp/mlsysbook-unit-hardening/vol1-pdf-full.log
```

**Result:** `_build/pdf-vol1/Machine-Learning-Systems-Vol1.pdf`, 31 MB, exit 0, no error patterns in `/private/tmp/mlsysbook-unit-hardening/vol1-pdf-full.log`.

#### Vol II PDF — done

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto
ln -sf config/_quarto-pdf-vol2.yml _quarto.yml
ln -sf index-vol2.qmd index.qmd
LOG=/private/tmp/mlsysbook-unit-hardening/vol2-pdf-full.log
PYTHONPATH=/Users/VJ/GitHub/MLSysBook-fmt-fix/mlsysim:/Users/VJ/GitHub/MLSysBook-fmt-fix \
  MPLBACKEND=Agg quarto render --to titlepage-pdf >> "$LOG" 2>&1
```

**Result:** `_build/pdf-vol2/Machine-Learning-Systems-Vol2.pdf`, 65 MB, exit 0, no error patterns in `/private/tmp/mlsysbook-unit-hardening/vol2-pdf-full.log`.

#### 9B pass criteria

1. PDF artifact exists and is non-trivial size (>10 MB typical)
2. LaTeX log has **no errors** (warnings OK if pre-existing)
3. No cell execution failures in log
4. Spot-check 3–5 pages with LEGO callouts (units render correctly, no raw `{python}`)

**LaTeX log locations:**
- `book/quarto/index.log` (during build)
- Post-render: `scripts/save_latex_log.py` may archive — check `_build/pdf-vol1/` for saved logs

**LaTeX error scan (example):**
```bash
rg '^!' book/quarto/index.log
rg 'Undefined control sequence|LaTeX Error' book/quarto/index.log
```

---

### 6.3 Phase 9C — Full volume builds — ✅ DONE

**Purpose:** Prove complete website + book builds with xref resolution and CI-equivalent checks.

#### 9C-1: Full HTML (both volumes)

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto

# Vol I
ln -sf config/_quarto-html-vol1.yml _quarto.yml
MPLBACKEND=Agg quarto render --to html -M jupyter:python3

# Vol II
ln -sf config/_quarto-html-vol2.yml _quarto.yml
MPLBACKEND=Agg quarto render --to html -M jupyter:python3
```

**Passed:** `scripts/verify_rendered_xrefs.py` reported zero unresolved cross-references for `_build/html-vol1` and `_build/html-vol2`; `rg '{python}'` and `rg '\?@'` returned no HTML matches.

#### 9C-2: Full PDF (both volumes)

Done in 9B.

#### 9C-3: Preflight-style scans

Use `/preflight` skill or manually:

| Check | Command / tool |
|-------|----------------|
| No `{python}` in HTML | `rg '{python}' book/quarto/_build/html-vol1 book/quarto/_build/html-vol2 --glob '*.html'` |
| No unresolved xrefs | `python3 book/quarto/scripts/verify_rendered_xrefs.py` |
| LaTeX errors | Scan saved PDF logs |
| pytest | `pytest mlsysim/ -o addopts=` |
| pre-commit | `pre-commit run --all-files` |

#### 9C-4: CI parity (optional but ideal)

Workflow: `.github/workflows/book-build-container.yml`
Invoked by `book-validate-dev.yml` on push to `dev`. Local 9C green ≈ push confidence.

**Do NOT** manually `gh workflow run validate-dev` after push.

---

## 7. Phase 10 — merge (after G7 green)

### 10A — Sync `dev` into `fmt-fix`

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
git fetch origin dev
git merge --no-ff origin/dev   # resolve conflicts; prefer unit-hardened LEGO cells
```

### 10B — Full re-verify on merged tip

1. `pre-commit run --all-files`
2. `pytest mlsysim/ book/tools/tests/test_lint_lego_units.py -o addopts=`
3. `./book/binder check lego-units` (+ other default scopes)
4. Re-run 9C minimum on conflicted chapters if any
5. One commit per fix; update PROGRESS.md

### 10C — Promote `fmt-fix` → `dev`

Only when 10B green. User validates; push triggers CI.

```bash
# From main checkout or as directed in GIT_WORKFLOW.md
git checkout dev
git merge --no-ff fmt-fix
# User pushes when ready
```

---

## 8. Key files changed in this effort (map for Codex)

| Area | Path | Notes |
|------|------|-------|
| Lint | `book/tools/scripts/lint_lego_units.py` | L014 regex, L011 rate rule |
| Lint tests | `book/tools/tests/test_lint_lego_units.py` | 9 tests |
| Baseline | `book/tools/audit/lego_units_baseline.json` | 81 L014 allowlisted |
| Prose-units | `book/tools/audit/book_check_lego_prose_units.py` | Closed-export dupes only |
| fmt | `mlsysim/mlsysim/fmt.py` | precision=None defaults |
| fmt tests | `mlsysim/tests/test_fmt.py` | 120 tests |
| Pilot chapter | `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd` | fmt_emissions, prose dupes |
| Rate fix | `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd` ~1815 | GpuEfficiencyTrajectoryRecap |
| Rules | `.claude/rules/lego-units.md`, `math.md`, etc. | Authoring discipline |

---

## 9. Deferred / out of scope (do not block merge)

| Item | Notes |
|------|-------|
| **Layer A′** | Registry-first LOAD audit — not started |
| **81 L014 burn-down** | Allowlisted closed `*_unit_str = fmt(...)` → migrate to domain formatters over time |
| **8½-B3** | Wire prose-units into pre-commit — deferred |
| **distributed_training.qmd:470** | bytes/bw `* THOUSAND` → ms — different rate pattern; not TFLOP/s÷W class |
| **fmt-thread WIP** | `audit_fmt_usage.py`, `fmt_prose_contract.py` — separate effort |
| **Prose-units table in PROGRESS** | Stale 17-file snapshot — checker was fixed; re-run shows 81/81 OK |

---

## 10. Known gotchas (save hours)

| # | Gotcha | Fix |
|---|--------|-----|
| 1 | `ImportError: fmt_memory` during render | Use `-M jupyter:python3`, not default `mlsysbook` kernel |
| 2 | `\lettrine` undefined on PDF | Use `--to titlepage-pdf`, not `--to pdf` |
| 3 | Exit code 1 after HTML chapter render | Expected — xref verify scans full build dir |
| 4 | Batch render loop stops after 1 chapter | `set -e` + `grep -c`/`rg -c` exit 1 on zero matches; use `set +e` and `\|\| true` |
| 5 | `--output-dir` on single-file render | Fails: "only when rendering projects" — output goes to `_build/html-volN/` per config |
| 6 | "traceback" in HTML grep | Quiz prose may mention the word — verify context |
| 7 | PDF render triggers full book | Normal for `type: book` config — plan 9B accordingly |
| 8 | `_quarto.yml` symlink | Wrong vol config → wrong output dir / wrong dependencies |
| 9 | Accidental commit of verify report | `git restore book/tools/audit/artifacts/lego_cells_verify_report.json` |

---

## 11. Suggested Codex session plan (ordered, do not skip)

### Session 1 — Phase 10 (~1–2 hours)

1. Merge `dev` → `fmt-fix`
2. Re-run gates on merged tip
3. Merge `fmt-fix` → `dev` when green
4. User push; monitor CI (do not manual workflow dispatch)

---

## 12. Immediate next actions (start here)

```
[x] 1. Verify Vol I PDF artifact and log
[x] 2. Run Vol II titlepage-pdf
[x] 3. Run 9C full HTML Vol I + Vol II with xref verify = 0
[x] 4. Run rendered LEGO verifier: 44/44 chapters, 933/933 cells
[x] 5. pre-commit + pytest green
[ ] 6. Split/commit intended work; exclude fmt-thread WIP and generated artifacts
[ ] 7. Phase 10A merge dev → fmt-fix
[ ] 8. Phase 10B re-verify
[ ] 9. Phase 10C merge fmt-fix → dev (user approval)
```

---

## 13. Uncommitted local state at handoff

```
 M book/tools/audit/fmt/audit_fmt_usage.py      # fmt-thread — do not commit
 M book/tools/audit/fmt/fmt_prose_contract.py   # fmt-thread — do not commit
 M mlsysim-lego-unit-hardening-PROGRESS.md     # session updates — commit when user asks
 M mlsysim-lego-unit-hardening-plan.md         # may need header sync
?? mlsysim-domain-formatting-plan.md           # separate thread
?? mlsysim-lego-unit-hardening-CODEX-HANDOFF.md # this file
```

**Optional commit for handoff:** add only `PROGRESS.md` + `CODEX-HANDOFF.md` with message like: `Document Phase 9A completion and Codex handoff for 9B/9C.`

---

## 14. Acceptance criteria ("we are done")

All must be true:

- [x] G1–G8 all **PASS**
- [x] 44/44 LEGO chapters HTML-green
- [x] Vol I + Vol II PDF build without LaTeX errors
- [x] Full-volume HTML builds with **0** `?@` xref literals
- [x] `pre-commit run --all-files` green
- [x] `pytest mlsysim` green
- [ ] `fmt-fix` merged to `dev` with CI green on push
- [ ] PROGRESS.md shows Phase 10 complete

---

## 15. Reference — 44 LEGO chapters rendered in 9A

### Vol I content (16)
`introduction`, `ml_systems`, `ml_workflow`, `data_engineering`, `nn_computation`, `nn_architectures`, `frameworks`, `training`, `data_selection`, `model_compression`, `hw_acceleration`, `benchmarking`, `model_serving`, `ml_ops`, `responsible_engr`, `conclusion`

### Vol I appendices (4)
`appendix_data`, `appendix_algorithm`, `appendix_machine`, `appendix_assumptions`

### Vol II content (17)
`introduction`, `compute_infrastructure`, `network_fabrics`, `data_storage`, `distributed_training`, `collective_communication`, `fault_tolerance`, `fleet_orchestration`, `performance_engineering`, `inference`, `edge_intelligence`, `ops_scale`, `security_privacy`, `robust_ai`, `sustainable_ai`, `responsible_ai`, `conclusion`

### Vol II appendices (7)
`appendix_dam`, `appendix_c3`, `appendix_fleet`, `appendix_communication`, `appendix_reliability`, `appendix_inference`, `appendix_assumptions`

Paths: `book/quarto/contents/vol{1,2}/<dir>/<dir>.qmd`

---

*End of handoff. Questions about editorial intent → `.claude/rules/lego-units.md` + `book-prose.md`. Questions about git workflow → `.claude/docs/shared/GIT_WORKFLOW.md`.*
