#!/usr/bin/env python3
"""Pre-commit hook guarding vault/corpus.json from direct edits.

Per ARCHITECTURE.md §11.1 (fixes C-2): YAML is the sole authoring surface from
Day 1 of Phase 1. ``corpus.json`` is generated-only; this hook refuses any
commit that touches it unless the commit carries the override trailer.

Install via::

    cp interviews/vault-cli/scripts/pre_commit_corpus_guard.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

Or wire via ``pre-commit`` framework (not configured in Phase 0).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OVERRIDE_TRAILER = "Vault-Override: corpus-json-hand-edit"
GUARDED_PATH = "interviews/vault/corpus.json"


def staged_changes() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def commit_message_has_override() -> bool:
    # During pre-commit, the message lives at .git/COMMIT_EDITMSG in a regular
    # clone, but in a worktree .git is a file pointing at a per-worktree gitdir.
    # Resolve via `git rev-parse --git-path` so this works in both layouts.
    try:
        gp = subprocess.run(
            ["git", "rev-parse", "--git-path", "COMMIT_EDITMSG"],
            capture_output=True,
            text=True,
            check=True,
        )
        msg = Path(gp.stdout.strip())
    except subprocess.CalledProcessError:
        return False
    if not msg.exists():
        return False
    return OVERRIDE_TRAILER in msg.read_text(encoding="utf-8")


def main() -> int:
    changes = staged_changes()
    if GUARDED_PATH not in changes:
        return 0
    if commit_message_has_override():
        sys.stderr.write(
            f"[warning] {GUARDED_PATH} is being edited with override trailer.\n"
        )
        return 0
    sys.stderr.write(
        f"[error] Refusing to commit direct edit of {GUARDED_PATH}.\n"
        f"This file is GENERATED from vault/questions/ YAML per ARCHITECTURE.md §11.1.\n"
        f"If you really need to hand-edit, add this trailer to your commit message:\n\n"
        f"    {OVERRIDE_TRAILER}: <one-line justification>\n\n"
        f"Otherwise: edit the underlying YAML and regenerate with `vault build --local-json`.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
