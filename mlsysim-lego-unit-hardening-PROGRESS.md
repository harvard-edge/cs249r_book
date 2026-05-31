# MLSysBook LEGO Unit Hardening — Progress

**Branch:** `fmt-fix`
**Worktree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix`
**Plan:** `/Users/VJ/.cursor/plans/lego_unit_hardening_040502db.plan.md`

## Checklist

| Step | Status | Notes |
|------|--------|-------|
| 1–13 — MLSysIM infra + lint | DONE | units, physics, domain formatters, lint_lego_units.py |
| 14+ — QMD migration | DONE | All Vol I + Vol II chapters and appendices |
| Closure — L019 + baseline | DONE | `.m_as()` blocking error; baseline refreshed (411 warnings) |

## Migration summary

- Bulk `.m_as(unit)` → `.to(unit).magnitude` via `book/tools/scripts/migrate_lego_m_as.py` (~1,235 replacements).
- Remaining `.m_as(` references are **comment-only** (5 files, LEGO header docs).
- **L019** promotes `.m_as(` to pre-commit **error** (default `--fail-on error`).
- Warning baseline: `book/tools/audit/lego_units_baseline.json` (411 entries, post-migration).
- Full-book LEGO exec test: **ALL OK**.

## Next (plan closure phases)

| Phase | Status |
|-------|--------|
| 9A — HTML render every chapter | pending |
| 9B — PDF render every chapter | pending |
| 9C — full volume HTML/PDF + pre-commit + pytest | pending |
| 10 — merge fmt-fix → dev | pending |

Promote additional lint rules (L014, L015, L017, …) incrementally as warning debt is cleared.
