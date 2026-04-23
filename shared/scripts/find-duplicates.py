#!/usr/bin/env python3
"""find-duplicates.py — surface near-duplicate files across subsites.

Why this exists
---------------
The MLSysBook ecosystem keeps real-file copies of certain shared assets
per subsite (Quarto's resource-copy step preserves symlinks instead of
dereferencing them). Those known mirrors live in
`shared/scripts/sync-mirrors.sh` and are kept identical by that script.

But beyond those known mirrors, smaller pockets of duplication tend to
accumulate over time: a footer template lifted into two subsites, a CSS
partial copy-pasted before we had a canonical theme file, a one-off
script that two builds independently inlined. This auditor finds those
*unintended* duplicates so the next refactor can promote them into the
shared canon (and add an entry to sync-mirrors.sh, or symlink them
where Quarto allows).

What it does
------------
1. Walk a configurable set of roots (default: shared/, book/, kits/,
   labs/, mlsysim/docs/, slides/, instructors/, site/, tinytorch/,
   interviews/staffml/) and hash every file matching the configured
   suffix list.
2. Group files by hash.
3. Report any group with >1 path that is NOT entirely contained in the
   sync-mirrors map (those are intentional duplicates, by design).
4. Exit non-zero if any unintended duplicate group is found.

Output is a human-readable summary plus a JSON report at
.audit/duplicates.json so CI can consume it later.

Limitations
-----------
- Exact-byte hash only; near-duplicates with whitespace differences are
  NOT flagged. That's intentional: the goal is to find true accidental
  copies, not stylistic variations. A future iteration can add a
  normalized-content hash if needed.
- File-size threshold avoids reporting many tiny identical files
  (LICENSE stubs, empty .gitkeep, etc.).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# File suffixes worth auditing. Binary assets (images, PDFs, fonts) are
# NOT scanned — duplicate hash on those is usually intentional (logos,
# icons) and the cleanup story is different from source files.
DEFAULT_SUFFIXES = {
    ".js", ".mjs", ".ts", ".tsx", ".jsx",
    ".scss", ".css",
    ".html", ".yml", ".yaml",
    ".sh", ".py",
}

# Paths to skip entirely. These are either build outputs, vendored
# dependencies, or directories where duplication is structural and not
# fixable at the source-tree level.
DEFAULT_EXCLUDES = [
    re.compile(r"(^|/)_(site|book|build|extensions|freeze|archive)/"),
    re.compile(r"(^|/)node_modules/"),
    re.compile(r"(^|/)\.venv/"),
    re.compile(r"(^|/)\.cache/"),
    re.compile(r"(^|/)htmlcov/"),
    re.compile(r"(^|/)dist/"),
    re.compile(r"(^|/)\.next/"),
    re.compile(r"(^|/)out/"),
    re.compile(r"(^|/)\.git/"),
]

DEFAULT_ROOTS = [
    "shared", "book", "kits", "labs", "mlsysim/docs", "slides",
    "instructors", "site", "tinytorch", "interviews/staffml",
    ".github",
]

# Min size in bytes for a file to participate. Tiny files (<256B) are
# noise; suppress them so the report fits on one screen.
DEFAULT_MIN_SIZE = 256


def known_mirror_groups(repo_root: Path) -> list[set[str]]:
    """Parse `shared/scripts/sync-mirrors.sh` for SYNC_MAP entries.

    Each entry is `canonical|mirror1,mirror2,...`. We treat each entry
    as one *intentional* duplicate group; if a hash collision is fully
    explained by one such group, it's not reported.
    """
    sync_path = repo_root / "shared/scripts/sync-mirrors.sh"
    groups: list[set[str]] = []
    if not sync_path.exists():
        return groups
    in_map = False
    for line in sync_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("SYNC_MAP=("):
            in_map = True
            continue
        if not in_map:
            continue
        if stripped == ")":
            break
        m = re.match(r'"([^"]+)"', stripped)
        if not m:
            continue
        canonical, _, mirrors = m.group(1).partition("|")
        if not mirrors:
            continue
        members = {canonical, *(m.strip() for m in mirrors.split(","))}
        groups.append({m for m in members if m})
    return groups


def is_excluded(path: Path, excludes: list[re.Pattern]) -> bool:
    rel = path.as_posix()
    return any(p.search(rel) for p in excludes)


def hash_file(path: Path, algo: str = "sha1") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(roots: list[str], suffixes: set[str], excludes: list[re.Pattern], min_size: int):
    for root in roots:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        # followlinks=False stops os.walk from descending through symlinked
        # directories (which would re-scan content already walked elsewhere).
        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            rel_dir = Path(dirpath).relative_to(REPO_ROOT)
            if is_excluded(rel_dir, excludes):
                dirnames[:] = []
                continue
            for name in filenames:
                p = Path(dirpath) / name
                # Symlinked files are deliberate aliases for a canonical
                # source — flagging them as "duplicates" hides the real
                # duplicates we want to surface. Skip.
                if p.is_symlink():
                    continue
                if p.suffix.lower() not in suffixes:
                    continue
                rel = p.relative_to(REPO_ROOT)
                if is_excluded(rel, excludes):
                    continue
                try:
                    if p.stat().st_size < min_size:
                        continue
                except OSError:
                    continue
                yield rel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", action="append", help="Repo-relative root to scan (repeatable). Default scans common roots.")
    parser.add_argument("--suffix", action="append", help="File suffix to include (repeatable). Default covers source-y files.")
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE, help=f"Min file size in bytes (default {DEFAULT_MIN_SIZE}).")
    parser.add_argument("--report", default=".audit/duplicates.json", help="Where to write the JSON report.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any unintended duplicate is found.")
    args = parser.parse_args()

    roots = args.root or DEFAULT_ROOTS
    suffixes = {s if s.startswith(".") else f".{s}" for s in (args.suffix or [])} or DEFAULT_SUFFIXES
    known = known_mirror_groups(REPO_ROOT)

    by_hash: dict[str, list[Path]] = defaultdict(list)
    total = 0
    for path in walk_files(roots, suffixes, DEFAULT_EXCLUDES, args.min_size):
        try:
            digest = hash_file(REPO_ROOT / path)
        except OSError:
            continue
        by_hash[digest].append(path)
        total += 1

    unintended = []
    for digest, paths in by_hash.items():
        if len(paths) < 2:
            continue
        path_set = {p.as_posix() for p in paths}
        # Skip if every path is fully covered by one known sync-mirror group.
        if any(path_set <= group for group in known):
            continue
        unintended.append({
            "hash": digest,
            "size": (REPO_ROOT / paths[0]).stat().st_size,
            "paths": sorted(p.as_posix() for p in paths),
        })

    unintended.sort(key=lambda g: (-len(g["paths"]), -g["size"]))

    report_path = REPO_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "scanned_files": total,
        "duplicate_groups": len(unintended),
        "duplicates": unintended,
    }, indent=2))

    print(f"📊 Scanned {total} files across {len(roots)} roots.")
    print(f"📊 Known intentional mirror groups: {len(known)}")
    print(f"📊 Unintended duplicate groups:     {len(unintended)}")
    if unintended:
        print()
        for grp in unintended[:20]:
            print(f"  hash={grp['hash'][:12]}  size={grp['size']:>6}  copies={len(grp['paths'])}")
            for p in grp["paths"]:
                print(f"    - {p}")
        if len(unintended) > 20:
            print(f"  ... and {len(unintended) - 20} more groups (see {args.report})")
    print(f"\nFull report: {args.report}")

    if args.strict and unintended:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
