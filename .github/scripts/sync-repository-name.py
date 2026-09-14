#!/usr/bin/env python3
"""
Keep every reference to this repository's name in step with GitHub.

Why this exists
---------------
GitHub's own name for the repository (``github.repository`` in Actions) is
the single source of truth. Workflows and CI scripts read it directly, but
many files cannot: READMEs and badges that GitHub renders verbatim, package
metadata that PyPI publishes, citation files, and page links in the Quarto
sources. Those files spell the name out.

``.github/repository`` records the name those files currently use. When it
differs from GitHub's name, after a rename or a transfer, this script
rewrites every reference and the record. The ``infra-repository-name-sync``
workflow then commits the result to dev.

What gets rewritten
-------------------
Only forms whose context pins them to this repository:

- ``owner/name``, as in clone URLs, issue links, and API paths
- ``owner%2Fname``, the URL-encoded form inside badge URLs
- ``owner.github.io/name``, the project Pages path
- ``YOUR_USERNAME/name``, the fork placeholder in contributor guides
- ``cd name``, stepping into a fresh clone
- ``"projectName": "name"``, in all-contributors configs

A bare name is never matched on its own. A repository name can also be a
brand handle or a domain label, and after a rename to such a word, matching
it bare would rewrite those too. New prose should write ``owner/name``.
Names match as whole words, so ``name_dev``, a different repository, stays
untouched.

Files under ``.github/workflows/`` are never rewritten, because a push made
with GITHUB_TOKEN may not modify them. A reference there fails every run
instead, rename or not; workflows read ``github.repository``.

Usage
-----
  python3 .github/scripts/sync-repository-name.py                   # rewrite if the names differ
  python3 .github/scripts/sync-repository-name.py --check           # report only; exit 1 if stale
  python3 .github/scripts/sync-repository-name.py --repo owner/name # use this name instead of GitHub's
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD = REPO_ROOT / ".github" / "repository"
WORKFLOWS_DIR = ".github/workflows/"
SLUG_RE = re.compile(r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$")

# A qualified reference starts after a non-word character, an encoded slash,
# or a shell default (${VAR:-owner/name}), and ends before a non-word
# character. Hyphens count as word characters because names contain them.
BEFORE = r"(?:(?<![\w-])|(?<=%2F)|(?<=%2f)|(?<=:-))"
AFTER = r"(?![\w-])"


def read_record() -> str:
    for line in RECORD.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    sys.exit(f"error: {RECORD.relative_to(REPO_ROOT)} has no owner/name line")


def github_name(override: str | None) -> str:
    """GitHub's name: --repo, then GITHUB_REPOSITORY, then the checkout's remote via gh."""
    if override:
        return override
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    result = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def reference_forms(old: str, new: str) -> list[tuple[re.Pattern[str], str, bool]]:
    """Each form that names the repository: (pattern, rewrite, names the owner)."""
    owner, name = (re.escape(part) for part in old.split("/"))
    new_owner, new_name = new.split("/")
    return [
        (re.compile(BEFORE + owner + "/" + name + AFTER), f"{new_owner}/{new_name}", True),
        (re.compile(BEFORE + owner + "%2[Ff]" + name + AFTER), f"{new_owner}%2F{new_name}", True),
        (re.compile(BEFORE + owner + r"\.github\.io/" + name + AFTER), f"{new_owner}.github.io/{new_name}", True),
        (re.compile(r"(?<![\w-])(YOUR_USERNAME/)" + name + AFTER), rf"\g<1>{new_name}", False),
        (re.compile(r"(?<![\w-])(cd )" + name + AFTER), rf"\g<1>{new_name}", False),
        (re.compile(r'("projectName": ")' + name + '"'), rf'\g<1>{new_name}"', False),
    ]


def candidate_files(name: str) -> list[str]:
    """Tracked text files that contain the name at all (git grep skips binaries)."""
    result = subprocess.run(
        ["git", "grep", "-l", "-z", "-I", "-F", "-e", name],
        cwd=REPO_ROOT, capture_output=True,
    )
    if result.returncode not in (0, 1):
        sys.exit(f"error: git grep failed: {result.stderr.decode(errors='replace')}")
    return [p for p in result.stdout.decode("utf-8").split("\0") if p]


def read_text(rel: str) -> str | None:
    path = REPO_ROOT / rel
    if path.is_symlink() or not path.is_file():
        return None
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None


def references(slug: str, owner_forms_only: bool = False) -> dict[str, int]:
    """Count references to slug per file. The forms never overlap, so counts add."""
    forms = [f for f in reference_forms(slug, slug) if f[2] or not owner_forms_only]
    counts: dict[str, int] = {}
    for rel in candidate_files(slug.split("/")[1]):
        text = read_text(rel)
        if text is None:
            continue
        count = sum(len(pattern.findall(text)) for pattern, _, _ in forms)
        if count:
            counts[rel] = count
    return counts


def rewrite(old: str, new: str, files: list[str]) -> None:
    forms = reference_forms(old, new)
    for rel in files:
        text = read_text(rel)
        for pattern, replacement, _ in forms:
            text = pattern.sub(replacement, text)
        (REPO_ROOT / rel).write_bytes(text.encode("utf-8"))


def set_outputs(**values: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            for key, value in values.items():
                fh.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="report stale references without rewriting")
    parser.add_argument("--repo", help="owner/name to sync to instead of GitHub's name")
    args = parser.parse_args()

    recorded = read_record()
    current = github_name(args.repo)
    for label, slug in (("recorded", recorded), ("GitHub", current)):
        if not SLUG_RE.match(slug):
            sys.exit(f"error: {label} name {slug!r} is not owner/name")

    found = references(recorded)
    # Check GitHub's name too: a workflow already spelling out a new name
    # would pass today and block every sync after the rename.
    in_workflows: dict[str, int] = {}
    for counts in (found, references(current) if current != recorded else {}):
        for rel, count in counts.items():
            if rel.startswith(WORKFLOWS_DIR):
                in_workflows[rel] = in_workflows.get(rel, 0) + count
    if in_workflows:
        print(f"::error::{WORKFLOWS_DIR} spells out the repository name; read github.repository instead:")
        for rel in sorted(in_workflows):
            print(f"  {rel} ({in_workflows[rel]})")
        set_outputs(changed="false")
        return 1

    if current == recorded:
        print(f"✓ In sync: files use {current}, as GitHub does")
        set_outputs(changed="false")
        return 0

    total = sum(found.values())
    print(f"Repository renamed: {recorded} → {current}")
    print(f"{total} references in {len(found)} files")
    if args.check:
        for rel in sorted(found):
            print(f"  {rel} ({found[rel]})")
        set_outputs(changed="false")
        return 1

    rewrite(recorded, current, sorted(found))
    if read_record() != current:
        sys.exit(f"error: {RECORD.relative_to(REPO_ROOT)} still does not read {current}")
    # A transfer keeps the name, so only the owner-qualified forms must be gone.
    same_name = recorded.split("/")[1] == current.split("/")[1]
    leftover = references(recorded, owner_forms_only=same_name)
    if leftover:
        sys.exit(f"error: references to {recorded} survived the rewrite: {sorted(leftover)}")
    print(f"✓ Rewrote {total} references; files now use {current}")
    set_outputs(changed="true", old=recorded, new=current)
    return 0


if __name__ == "__main__":
    sys.exit(main())
