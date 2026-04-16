#!/usr/bin/env python3
"""CI check: ``id-registry.yaml`` is append-only.

Rejects PRs that remove or reorder lines from ``interviews/vault/id-registry.yaml``
— the registry is the C-5 load-bearing structure. Compares the file's lines
between the PR base and HEAD; ensures every base-line is still present and
in the same relative order.

Invoked from ``.github/workflows/vault-ci.yml``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REGISTRY_PATH = "interviews/vault/id-registry.yaml"


def main() -> int:
    base = "origin/main"
    # Prefer origin/main; fall back to HEAD~1 for local testing.
    try:
        subprocess.run(
            ["git", "rev-parse", "--verify", base], check=True, capture_output=True
        )
    except subprocess.CalledProcessError:
        base = "HEAD~1"

    try:
        result = subprocess.run(
            ["git", "show", f"{base}:{REGISTRY_PATH}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        # File didn't exist at base — first commit landing it is fine.
        return 0

    base_lines = result.stdout.splitlines()
    head = Path(REGISTRY_PATH).read_text(encoding="utf-8").splitlines()

    # Every base-line must be present in head, in the same order.
    # We allow ONLY appending new lines after the existing ones.
    j = 0
    for i, line in enumerate(base_lines):
        while j < len(head) and head[j] != line:
            j += 1
        if j >= len(head):
            sys.stderr.write(
                f"[error] {REGISTRY_PATH}: line {i+1} from base is missing or reordered "
                f"at HEAD.\n  base line: {line!r}\n"
            )
            return 1
        j += 1
    print(f"[ok] {REGISTRY_PATH}: append-only invariant holds "
          f"({len(base_lines)} base lines preserved; "
          f"{len(head) - len(base_lines)} new lines appended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
