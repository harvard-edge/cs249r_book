#!/usr/bin/env python3
"""One-shot migration: flat <track>/<id>.yaml -> hierarchical <track>/<area>/<id>.yaml.

Reads each YAML's `track` and `competency_area` fields, then moves the file
to the corresponding hierarchical path. Idempotent — running on an
already-migrated tree is a no-op.

Validates pre-flight that every YAML has a parseable competency_area before
moving anything. Refuses to start on a partial corpus.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

VAULT_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    yaml_files = sorted(QUESTIONS_DIR.rglob("*.yaml"))
    print(f"found {len(yaml_files)} yaml files under {QUESTIONS_DIR}")

    # Pre-flight: parse every YAML and gather (path, track, area)
    plan: list[tuple[Path, Path]] = []
    skipped_already_migrated = 0
    parse_errors: list[tuple[Path, str]] = []

    for path in yaml_files:
        try:
            with open(path) as f:
                doc = yaml.safe_load(f)
        except Exception as exc:
            parse_errors.append((path, f"yaml load failed: {exc}"))
            continue

        track = doc.get("track")
        area = doc.get("competency_area")
        if not track or not area:
            parse_errors.append((path, f"missing track/area: track={track!r} area={area!r}"))
            continue

        rel = path.relative_to(QUESTIONS_DIR)
        parts = rel.parts
        if len(parts) == 3 and parts[0] == track and parts[1] == area:
            skipped_already_migrated += 1
            continue
        if len(parts) == 2 and parts[0] == track:
            new_path = QUESTIONS_DIR / track / area / path.name
            plan.append((path, new_path))
            continue
        parse_errors.append((path, f"unexpected layout: rel={rel}"))

    print(f"  already migrated: {skipped_already_migrated}")
    print(f"  to migrate:       {len(plan)}")
    print(f"  parse errors:     {len(parse_errors)}")

    if parse_errors:
        print("\nERRORS — refusing to migrate. Fix these first:")
        for p, msg in parse_errors[:10]:
            print(f"  {p}: {msg}")
        if len(parse_errors) > 10:
            print(f"  ... and {len(parse_errors) - 10} more")
        return 1

    if not plan:
        print("nothing to do.")
        return 0

    # Show a few sample moves
    print("\nSample moves:")
    for src, dst in plan[:5]:
        print(f"  {src.relative_to(QUESTIONS_DIR.parent)} -> {dst.relative_to(QUESTIONS_DIR.parent)}")

    if args.dry_run:
        print("\n--dry-run set; not moving anything.")
        return 0

    # Group by destination dir to mkdir efficiently
    dest_dirs = {dst.parent for _, dst in plan}
    for d in dest_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Move files. Use shutil.move; git will detect via rename similarity.
    print(f"\nmoving {len(plan)} files...")
    for src, dst in plan:
        if dst.exists():
            print(f"  WARNING: destination exists, skipping: {dst}")
            continue
        shutil.move(str(src), str(dst))

    # Clean up now-empty old <track>/ dirs (they should still hold subdirs only)
    for top in QUESTIONS_DIR.iterdir():
        if not top.is_dir():
            continue
        # Remove empty track dirs (shouldn't happen — they hold area subdirs)
        leftover_files = [p for p in top.iterdir() if p.is_file() and p.name != "LICENSE"]
        if leftover_files:
            print(f"  WARNING: {top.name}/ still has top-level files: {[p.name for p in leftover_files[:3]]}")

    print("✓ migration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
