# Repository-level tooling (`tools/`)

Scripts and audit artifacts at the **monorepo root** (not `book/tools/`, which is textbook-specific).

## Layout

| Path | Role |
|------|------|
| [`audit/`](audit/) | Math/HTML/PDF audit scripts for the Quarto book (see `audit/README.md`). |
| [`release-smoke/`](release-smoke/) | Playwright smoke checks for published sites; see `release-smoke/package.json`. Generated reports live under `release-smoke/reports/` (timestamped JSON). Safe to delete old reports when pruning; regenerate by re-running smoke. |
| `phase_b/` … `phase_g/`, `lint_calibration/` | One-off **cleanup and lint calibration manifests** from past passes (JSON/Markdown). Kept for traceability; not imported by CI or application code. Remove or archive only when maintainers agree the history is no longer needed. |
| `validate_playbook.py` | Standalone validation helper; see file docstring for usage. |

## Relationship to `book/tools/`

Use `book/tools/` for BibTeX, git hooks wiring, quarto scripts, and book prose validators. Use **`tools/` here** for cross-cutting audits, release smoke tests, and historical phase outputs.
