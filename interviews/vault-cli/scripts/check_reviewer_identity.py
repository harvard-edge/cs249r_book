#!/usr/bin/env python3
"""CI check: `vault promote --reviewed-by` cannot be spoofed.

Closes Gemini R5-H-3. ARCHITECTURE.md §13 requires the reviewer identity in
a promotion PR to match the committer email; without this check, any
contributor can promote drafts claiming to be someone else.

Runs in CI on every PR touching vault/questions/. For each question newly
added or edited in the PR whose provenance is llm-then-human-edited, ensures
the ``authors`` list contains the git committer's email for at least one of
the commits that touched that file.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from vault_cli.yaml_io import load_file


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True,
    ).stdout


def _base_ref() -> str:
    # Prefer origin/main / origin/dev; fall back to HEAD~1.
    for ref in ("origin/main", "origin/dev", "HEAD~1"):
        try:
            subprocess.run(
                ["git", "rev-parse", "--verify", ref],
                check=True, capture_output=True,
            )
            return ref
        except subprocess.CalledProcessError:
            continue
    return "HEAD~1"


def _changed_yaml_paths(base: str) -> list[Path]:
    out = _git("diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD")
    return [
        Path(p)
        for p in out.splitlines()
        if p.startswith("interviews/vault/questions/") and p.endswith(".yaml")
    ]


def _commit_authors_for_path(path: Path, base: str) -> set[str]:
    out = _git("log", f"{base}..HEAD", "--format=%ae", "--", str(path))
    return {line.strip() for line in out.splitlines() if line.strip()}


def main() -> int:
    base = _base_ref()
    changed = _changed_yaml_paths(base)
    if not changed:
        print("[ok] no vault/questions/ changes in this PR")
        return 0

    failures: list[str] = []
    for path in changed:
        try:
            data = load_file(path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path}: failed to load YAML: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        provenance = data.get("provenance")
        if provenance != "llm-then-human-edited":
            # Only promotions of LLM drafts need reviewer-identity proof.
            continue
        authors = data.get("authors") or []
        if not isinstance(authors, list):
            failures.append(f"{path}: authors is not a list")
            continue
        committer_emails = _commit_authors_for_path(path, base)
        overlap = {a for a in authors if a in committer_emails}
        if not overlap:
            failures.append(
                f"{path}: provenance=llm-then-human-edited but none of the "
                f"authors {authors!r} matches any commit email "
                f"{sorted(committer_emails)!r} that touched this file."
            )

    if failures:
        sys.stderr.write("[error] reviewer-identity check failed:\n")
        for f in failures:
            sys.stderr.write(f"  {f}\n")
        sys.stderr.write(
            "\nIf this is legitimate (e.g., maintainer promoting on behalf "
            "of a vetted contributor), add a commit to the PR from the "
            "reviewer's git-configured identity.\n"
        )
        return 1
    print(f"[ok] reviewer identity verified for {len(changed)} changed file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
