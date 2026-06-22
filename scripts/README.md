# scripts/

Utility scripts that sit outside the Binder CLI.

Binder is the public command surface for book build/check/fix/format workflows.
If a task is part of routine book automation or pre-commit, prefer
`./book/binder ...`. Keep this directory for one-off audit helpers and
cross-workflow utilities that do not belong to the book CLI.

## Standalone audit helpers

| Script | Purpose |
|--------|---------|
| `figure_audit.py` | Multimodal figure audit via Gemini CLI (compares rendered images against QMD source prose, captions, alt-text). |
| `exec_analysis.py` | Execution analysis helper for mlsysim scenarios. |
| `exec_single.py` | Single-scenario execution runner. |
| `audit_blocks.py` | Audits code block structure across chapters. |
| `cross-references/` | Cross-reference audit tooling (see `cross-references/README.md`). |
| `version/` | Shared release hash/manifest helpers used by site workflows. |

## Prerequisites

- `figure_audit.py` requires the `gemini` CLI installed and authenticated.
- The figure audit assumes the rendered HTML book is available at `https://harvard-edge.github.io/cs249r_book_dev/`.

## Usage

Run from the repository root:

```bash
python3 scripts/figure_audit.py
python3 scripts/cross-references/audit_crossrefs.py
```
