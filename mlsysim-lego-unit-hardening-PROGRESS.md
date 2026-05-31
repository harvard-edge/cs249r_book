# MLSysBook LEGO Unit Hardening — Progress

**Branch:** `fmt-fix`
**Worktree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix`
**Plan:** `/Users/VJ/.cursor/plans/lego_unit_hardening_040502db.plan.md`

## Checklist

| Step | Status | Notes |
|------|--------|-------|
| 1–13 — MLSysIM infra + lint | DONE | units, physics, domain formatters, lint_lego_units.py |
| 14+ — QMD migration | **DONE** | All Vol I + Vol II chapters and appendices |

## Migration summary

- Bulk `.m_as(unit)` → `.to(unit).magnitude` via `book/tools/scripts/migrate_lego_m_as.py` (~1,235 replacements).
- Remaining `.m_as(` references are **comment-only** (5 files, LEGO header docs).
- Manual fixes: `US` (microsecond) vs `USD`, `'B'` (byte) vs `Bparam`, missing `USD`/`kg`/`param` imports, fmt precision guards.
- Full-book LEGO exec test: **ALL OK** (all `{python}` blocks in `book/quarto/contents`).

## All chapters — DONE (0 `.m_as()` in LEGO code)

Vol I: introduction, ml_systems, ml_workflow, data_engineering, nn_computation, nn_architectures, frameworks, training, data_selection, model_compression, hw_acceleration, benchmarking, model_serving, ml_ops, responsible_engr, conclusion, appendices (algorithm, machine, data, assumptions).

Vol II: introduction, compute_infrastructure, data_storage, distributed_training, performance_engineering, network_fabrics, inference, edge_intelligence, ops_scale, fleet_orchestration, collective_communication, fault_tolerance, security_privacy, sustainable_ai, robust_ai, conclusion, appendices (fleet, communication, reliability, assumptions, c3, dam, inference).
