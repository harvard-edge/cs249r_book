# scripts/

Utility scripts that sit outside the Binder CLI.

Binder is the public command surface for book build/check/fix/format workflows.
If a task is part of routine book automation or pre-commit, prefer
`./binder/binder ...`. Keep this directory for cross-workflow utilities that do
not belong to the book CLI.

| Folder | Purpose |
|--------|---------|
| `release-smoke/` | Playwright smoke checks for the published sites (`node smoke.mjs <site>`; site keys in `sites.json`, reports under `reports/`). |
| `version/` | Shared release hash/manifest helpers used by site workflows. |
