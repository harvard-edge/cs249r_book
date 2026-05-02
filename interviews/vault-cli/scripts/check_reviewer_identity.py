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

import os
import subprocess
import sys
from pathlib import Path

from vault_cli.yaml_io import load_file


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True,
    ).stdout


def _base_ref() -> str:
    """Determine the base ref to diff this PR/push against.

    PR mode: GitHub Actions sets ``GITHUB_BASE_REF`` to the target branch
    name (e.g. ``dev``); we diff against ``origin/<that>``. This is the
    only reliable signal for "what is this PR proposing to add."

    Push mode (no PR): GITHUB_BASE_REF is empty. Diff against ``HEAD~1``
    so we only check files modified by the most recent push, not every
    file that ever differed between branches.

    The previous implementation walked a hard-coded list (origin/main,
    origin/dev, HEAD~1) and returned the first that resolved. On a repo
    where dev is many commits ahead of main, this swept up every YAML
    edit between main and dev on every push to dev, producing 100+
    spurious failures unrelated to the current change. See:
    https://github.com/harvard-edge/cs249r_book/actions/runs/24937063459
    """
    base_env = os.environ.get("GITHUB_BASE_REF", "").strip()
    if base_env:
        candidate = f"origin/{base_env}"
        try:
            subprocess.run(
                ["git", "rev-parse", "--verify", candidate],
                check=True, capture_output=True,
            )
            return candidate
        except subprocess.CalledProcessError:
            pass
    return "HEAD~1"


def _changed_yaml_paths(base: str) -> list[Path]:
    out = _git("diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD")
    return [
        Path(p)
        for p in out.splitlines()
        if p.startswith("interviews/vault/questions/") and p.endswith(".yaml")
    ]


def _commit_author_emails_for_path(path: Path, base: str) -> set[str]:
    """Return commit AUTHOR emails (not committer) for commits touching path.

    Chip R7-L-6: git log %ae is author email. Maintainers rebasing/squashing
    on behalf of contributors preserve author but change committer — so we
    intentionally key on author to keep the chain intact.
    """
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
        commit_author_emails = _commit_author_emails_for_path(path, base)
        overlap = {a for a in authors if a in commit_author_emails}
        if not overlap:
            failures.append(
                f"{path}: provenance=llm-then-human-edited but none of the "
                f"authors {authors!r} matches any commit AUTHOR email "
                f"{sorted(commit_author_emails)!r} that touched this file."
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
