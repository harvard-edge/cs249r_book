#!/usr/bin/env python3
"""Rename cohort-tagged legacy IDs to clean <track>-NNNN form.

The 2026-04-21 ID_SCHEMES.md retired the cohort-tagged form (cloud-fill-*,
cloud-cell-*, etc.) but kept legacy IDs unchanged because rewriting them
would have broken ~3,100 chain references and external bookmarks.

This script applies the rename anyway, the safe way:

  1. Build a complete old → new ID map (dry-run friendly).
  2. Rename YAML files (`git mv`) and rewrite the `id:` field inside each.
  3. Rewrite every chain reference in chains.json using the same map.
  4. Save the map at vault/docs/id-renames-<date>.yaml for forensic /
     redirect lookup (so /practice?q=<old-id> can fall through to the new
     id rather than 404).

Usage:

    python3 rename_legacy_ids.py --dry-run          # preview the map only
    python3 rename_legacy_ids.py --dry-run --verbose # full diff
    python3 rename_legacy_ids.py                    # apply the rename

The output preview shows: total renames, per-track counts, sample old→new
pairs. After execution, run:

    python3 -m vault_cli.main check --strict       # confirm chain integrity
    npx playwright test tests/practice-smoke        # confirm UI still works
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"
CHAINS_PATH = VAULT_DIR / "chains.json"
RENAMES_DIR = VAULT_DIR / "docs"

CLEAN_ID = re.compile(r"^([a-z]+)-(\d+)$")
COHORT_ID = re.compile(r"^([a-z]+)-([a-z][a-z0-9-]*)-(\d+)$")
TRACKS = ["cloud", "edge", "mobile", "tinyml", "global"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def categorize_ids() -> tuple[dict[str, list[Path]], dict[str, list[Path]], dict[str, int]]:
    """Walk YAML files; classify clean vs cohort. Return per-track lists.

    Returns (clean_by_track, cohort_by_track, max_clean_n_by_track).
    """
    clean_by_track: dict[str, list[Path]] = defaultdict(list)
    cohort_by_track: dict[str, list[Path]] = defaultdict(list)
    max_n: dict[str, int] = defaultdict(int)
    for p in QUESTIONS_DIR.glob("*/*.yaml"):
        m = CLEAN_ID.match(p.stem)
        if m:
            track, n = m.group(1), int(m.group(2))
            clean_by_track[track].append(p)
            max_n[track] = max(max_n[track], n)
            continue
        m2 = COHORT_ID.match(p.stem)
        if m2:
            track = m2.group(1)
            cohort_by_track[track].append(p)
    return clean_by_track, cohort_by_track, max_n


def build_rename_map() -> dict[str, str]:
    """Mint new clean IDs for every cohort-tagged YAML, continuing each
    track's monotonic sequence past its current max."""
    _, cohort_by_track, max_n = categorize_ids()
    rename_map: dict[str, str] = {}
    for track in TRACKS:
        # Sort cohort files for deterministic mapping (alphabetical on stem)
        files = sorted(cohort_by_track.get(track, []), key=lambda p: p.stem)
        next_n = max_n.get(track, 0) + 1
        for p in files:
            old_id = p.stem
            new_id = f"{track}-{next_n:04d}"
            rename_map[old_id] = new_id
            next_n += 1
    return rename_map


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_yaml_rename(rename_map: dict[str, str], dry_run: bool) -> int:
    """Move each YAML and rewrite its `id:` field. Returns count of changes."""
    n_renamed = 0
    for old_id, new_id in rename_map.items():
        # Find the path
        candidates = list(QUESTIONS_DIR.glob(f"*/{old_id}.yaml"))
        if not candidates:
            print(f"  ! {old_id}: file not found", file=sys.stderr)
            continue
        old_path = candidates[0]
        new_path = old_path.parent / f"{new_id}.yaml"
        if dry_run:
            n_renamed += 1
            continue
        # Read + rewrite id field
        d = yaml.safe_load(old_path.read_text(encoding="utf-8"))
        d["id"] = new_id
        # Write to new location, then `git rm` old (can't `git mv` because
        # the file content changes too)
        new_path.write_text(
            yaml.safe_dump(d, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        # Use git rm to preserve history tracking
        subprocess.run(["git", "rm", "--quiet", str(old_path.relative_to(VAULT_DIR.parent.parent))],
                       check=True, cwd=VAULT_DIR.parent.parent)
        subprocess.run(["git", "add", str(new_path.relative_to(VAULT_DIR.parent.parent))],
                       check=True, cwd=VAULT_DIR.parent.parent)
        n_renamed += 1
    return n_renamed


def update_chains(rename_map: dict[str, str], dry_run: bool) -> int:
    """Rewrite chain references in chains.json. Returns count of references updated."""
    if not CHAINS_PATH.exists():
        return 0
    data = json.loads(CHAINS_PATH.read_text(encoding="utf-8"))
    n_updated = 0
    # chains.json is a list of chains, each with 'questions': [{id, level, title, ...}]
    chains = data if isinstance(data, list) else data.get("chains", [])
    for chain in chains:
        for q in chain.get("questions", []):
            old = q.get("id")
            if old in rename_map:
                if not dry_run:
                    q["id"] = rename_map[old]
                n_updated += 1
    if not dry_run and n_updated > 0:
        CHAINS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return n_updated


def save_redirect_index(rename_map: dict[str, str], dry_run: bool) -> Path:
    """Persist the old→new map for forensic / redirect lookup."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = RENAMES_DIR / f"id-renames-{today}.yaml"
    payload = {
        "renamed_at": datetime.now(timezone.utc).isoformat(),
        "reason": "Retire cohort-tagged legacy IDs in favor of clean <track>-NNNN.",
        "count": len(rename_map),
        "renames": dict(sorted(rename_map.items())),
    }
    if not dry_run:
        RENAMES_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview only; no changes.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full rename map (long).")
    args = parser.parse_args()

    clean_by_track, cohort_by_track, max_n = categorize_ids()
    print(f"Clean <track>-NNNN files:  {sum(len(v) for v in clean_by_track.values()):>5}")
    print(f"Cohort-tagged files:       {sum(len(v) for v in cohort_by_track.values()):>5}")
    print()
    print(f"{'Track':<10} {'Clean today':>12} {'Cohort to rename':>18} {'Max N today':>14} {'New range':>20}")
    rename_map = build_rename_map()
    per_track_renames: dict[str, int] = defaultdict(int)
    for old_id, new_id in rename_map.items():
        m = CLEAN_ID.match(new_id)
        if m:
            per_track_renames[m.group(1)] += 1
    for track in TRACKS:
        clean_n = len(clean_by_track.get(track, []))
        cohort_n = per_track_renames.get(track, 0)
        max_t = max_n.get(track, 0)
        end_n = max_t + cohort_n
        new_range = f"{track}-{max_t+1:04d}..{end_n:04d}" if cohort_n else "(no renames)"
        print(f"{track:<10} {clean_n:>12} {cohort_n:>18} {max_t:>14} {new_range:>20}")
    print()
    print(f"Total renames: {len(rename_map)}")

    # Sample 20 for visual sanity
    print("\nSample of 20 rename pairs:")
    for old, new in list(rename_map.items())[:20]:
        print(f"  {old:<30} -> {new}")

    if args.verbose:
        print("\nFull map:")
        for old, new in sorted(rename_map.items()):
            print(f"  {old:<30} -> {new}")

    # Chain reference impact (always preview)
    chain_impact = update_chains(rename_map, dry_run=True)
    print(f"\nChain references that would be remapped: {chain_impact}")

    # Redirect index location
    out = save_redirect_index(rename_map, dry_run=args.dry_run)
    if args.dry_run:
        print(f"\n[dry-run] Redirect index would be saved to: {out}")
    else:
        print(f"\nRedirect index saved to: {out}")

    if args.dry_run:
        print("\n--dry-run: no files were touched.")
        return 0

    # Apply
    print("\nApplying renames…")
    n_renamed = apply_yaml_rename(rename_map, dry_run=False)
    n_chain = update_chains(rename_map, dry_run=False)
    print(f"  YAMLs renamed: {n_renamed}")
    print(f"  Chain refs rewritten: {n_chain}")
    print("\nNext: run `vault check --strict` and `vault build --legacy-json`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
