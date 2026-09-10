# scripts/

Utility scripts that sit outside the Binder CLI.

Binder is the public command surface for book build/check/fix/format workflows.
If a task is part of routine book automation or pre-commit, prefer
`./binder/binder ...`. Keep this directory for one-off audit helpers and
cross-workflow utilities that do not belong to the book CLI.

## Standalone audit helpers

| Script | Purpose |
|--------|---------|
| `exec_analysis.py` | Execution analysis helper for mlsysim scenarios. |
| `exec_single.py` | Single-scenario execution runner. |
| `audit_blocks.py` | Audits code block structure across chapters. |
| `cross-references/` | Cross-reference audit tooling (see `cross-references/README.md`). |
| `release-smoke/` | Playwright smoke checks for the published sites (`node smoke.mjs <site>`; site keys in `sites.json`, reports under `reports/`). |
| `version/` | Shared release hash/manifest helpers used by site workflows. |

## Usage

Run from the repository root:

```bash
python3 scripts/cross-references/audit_crossrefs.py
```
