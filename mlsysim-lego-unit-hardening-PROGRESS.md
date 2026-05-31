# MLSysBook LEGO Unit Hardening — Progress

**Branch:** `fmt-fix`
**Worktree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix`
**Plan:** `/Users/VJ/.cursor/plans/lego_unit_hardening_040502db.plan.md`

## Checklist

| Step | Status | Notes |
|------|--------|-------|
| 1–13 — MLSysIM infra + lint | DONE | units, physics, domain formatters, lint_lego_units.py |
| 14+ — QMD migration | DONE | All Vol I + Vol II chapters and appendices |
| Closure — lint + exec + tests | DONE | 0 warnings; baseline empty; L019 blocking |

## Migration summary

- Bulk `.m_as(unit)` → `.to(unit).magnitude` via `migrate_lego_m_as.py` (~1,235 replacements).
- **L019** blocks `.m_as(` in executable LEGO lines (pre-commit error).
- **L017** retired (false positives on closed-auto `fmt_qty` names).
- Warning baseline: `lego_units_baseline.json` — **0 entries** (all warnings cleared).
- Fixed **658** glued cell fences (`)```` → newline + ` ``` `) that broke headless exec.
- Full-book python-cell exec: **ALL OK (81 files)**.
- `book/tools/tests/test_lint_lego_units.py` — 6 tests + full-corpus regression.
- `lego-units` binder scope: **default=True**.

## Deferred (user request: no builds yet)

| Phase | Status |
|-------|--------|
| 9A — HTML render every chapter | pending |
| 9B — PDF render every chapter | pending |
| 9C — full volume HTML/PDF | pending |
| 10 — merge fmt-fix → dev | pending |
