#!/usr/bin/env python3
"""Repair chain integrity issues in the StaffML question bank.

Two kinds of chain damage accrue over time:

1. **Orphan singletons**: chains where the only remaining member is a
   single question. Sibling questions were deleted/renamed/archived,
   leaving a chain-of-one. By definition a chain has ≥2 members, so the
   remaining question's chain reference is meaningless.
2. **Non-sequential positions**: chains where positions like [0, 2, 3]
   reveal a missing position 1 (a deleted/renamed member left the gap).
   Positions should always be [0..N-1].

Fix strategy:
- Orphans: strip the orphan's chain reference. Delete `chains: []` if empty.
- Non-sequential: collect all members, sort by current position, renumber
  to [0, 1, ..., N-1].

Usage::

    python3 repair_chains.py --dry-run    # report only
    python3 repair_chains.py              # apply
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"


LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6+": 6}


def survey() -> tuple[dict[str, list[tuple[str, int, Path, str, str]]],
                       set[str]]:
    """Return (chain_id -> [(qid, position, yaml_path, status, level)],
              published_chain_ids).

    Includes level so renumbering can restore Bloom-monotonicity (the
    `chain-bloom-monotonic` strict-check rule requires non-decreasing
    levels along position).
    """
    chains: dict[str, list[tuple[str, int, Path, str, str]]] = defaultdict(list)
    published_chains: dict[str, int] = defaultdict(int)
    for p in QUESTIONS_DIR.rglob('*.yaml'):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d or not d.get("chains"):
            continue
        for ref in d["chains"]:
            if isinstance(ref, dict):
                cid = ref.get("id")
                pos = ref.get("position")
                if cid is not None and pos is not None:
                    chains[cid].append((
                        d["id"], pos, p,
                        d.get("status", ""),
                        d.get("level", "L1"),
                    ))
                    if d.get("status") == "published":
                        published_chains[cid] += 1
    # A chain is "published-singleton" if <2 published members.
    pub_singletons = {cid for cid, n in published_chains.items() if n < 2}
    for cid in chains:
        if cid not in published_chains:
            pub_singletons.add(cid)
    return chains, pub_singletons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chains, pub_singletons = survey()
    print(f"Total chains in YAMLs: {len(chains)}")

    # 1. Orphans (in published-only view): chains with <2 published members.
    #    Drop the chain ref from the published member — the chain isn't a
    #    chain anymore from the user's perspective.
    print(f"Published-singleton chains: {len(pub_singletons)}")

    # 2. Renumber chains where any of the following hold:
    #    - positions are not [0, 1, ..., N-1]
    #    - positions duplicate (same value twice)
    #    - levels-by-position are non-monotonic (violates the strict
    #      `chain-bloom-monotonic` rule).
    needs_renumber = {}
    for cid, members in chains.items():
        if len(members) < 2:
            continue
        if cid in pub_singletons:
            continue  # going to be dropped, no point renumbering
        positions = sorted(m[1] for m in members)
        if positions != list(range(len(positions))):
            needs_renumber[cid] = members
            continue
        # Monotonic check on current ordering (sort by current position):
        sorted_by_pos = sorted(members, key=lambda m: m[1])
        ranks = [LEVEL_RANK.get(m[4], 0) for m in sorted_by_pos]
        if ranks != sorted(ranks):
            needs_renumber[cid] = members
    print(f"Chains needing renumber: {len(needs_renumber)}")

    file_plan: dict[Path, dict[str, tuple]] = defaultdict(dict)

    # Singletons: drop ALL chain refs (published + deleted) for chains
    # that no longer have ≥2 published members. The chain ceases to
    # exist; audit trail lives in the registry, not in orphan refs.
    for cid in pub_singletons:
        for qid, pos, fp, status, level in chains.get(cid, []):
            file_plan[fp][cid] = ("drop",)

    # Renumber: every member of an affected chain gets a fresh position
    # ordered by LEVEL ascending (L1 < L6+) so the resulting chain is
    # Bloom-monotonic. Ties broken by old position (preserve curated
    # ordering within a level), then qid (deterministic).
    for cid, members in needs_renumber.items():
        sorted_members = sorted(
            members,
            key=lambda m: (LEVEL_RANK.get(m[4], 0), m[1], m[0]),
        )
        for new_pos, (qid, old_pos, fp, status, level) in enumerate(sorted_members):
            if new_pos != old_pos:
                if file_plan[fp].get(cid, (None,))[0] != "drop":
                    file_plan[fp][cid] = ("renumber", new_pos)

    print(f"\nFiles that need editing: {len(file_plan)}")
    if args.dry_run:
        print("\nSample edits (first 10):")
        for fp, edits in list(file_plan.items())[:10]:
            print(f"  {fp.name}:")
            for cid, action in edits.items():
                print(f"    {cid}: {action}")
        return 0

    # Apply edits
    n_files_changed = 0
    for fp, edits in file_plan.items():
        d = yaml.safe_load(fp.read_text(encoding="utf-8"))
        new_chains = []
        for ref in d.get("chains") or []:
            if not isinstance(ref, dict):
                new_chains.append(ref)
                continue
            cid = ref.get("id")
            action = edits.get(cid)
            if action and action[0] == "drop":
                continue  # remove this chain ref
            if action and action[0] == "renumber":
                ref = {**ref, "position": action[1]}
            new_chains.append(ref)
        if new_chains:
            d["chains"] = new_chains
        else:
            d.pop("chains", None)
        fp.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True),
                      encoding="utf-8")
        n_files_changed += 1

    print(f"\n✓ updated {n_files_changed} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
