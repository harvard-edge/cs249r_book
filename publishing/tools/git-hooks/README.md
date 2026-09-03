# MLSysBook Git Hooks

Custom hooks that run pre-commit on **all files** before each commit, matching CI behavior.

## Setup (one-time)

From the repo root:

```bash
./book/tools/git-hooks/setup.sh
```

Or manually: `git config core.hooksPath book/tools/git-hooks`

## What it does

- **pre-commit**: Runs `pre-commit run --all-files` before every commit.
- Ensures local commits pass the same checks as CI.
- If any hook modifies files, the commit is aborted; stage the changes and commit again.

## Revert to default hooks

```bash
git config --unset core.hooksPath
pre-commit install --install-hooks
```
