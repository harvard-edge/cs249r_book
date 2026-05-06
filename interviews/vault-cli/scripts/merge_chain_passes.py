#!/usr/bin/env python3
"""Merge primary (live) + secondary (lenient-sweep) chains into chains.json.

Phase 1.5 of CHAIN_ROADMAP.md. Inputs:

  --primary <path>    chains.json from the strict pass — entries are
                      backfilled tier="primary" if not already tagged
  --secondary <path>  chains.proposed.lenient.json from the second pass —
                      entries already carry tier="secondary"

Cap-enforcement rules (mirror apply_proposed_chains.py invariants):

  1. A qid can appear in AT MOST 2 chains total across the merged registry.
  2. A qid in 2 chains MUST be L1 or L2 (foundational-anchor exception).

A secondary chain is rejected if accepting it would push ANY of its qids
past those caps. Primary chains are kept verbatim — secondary is the
slack the corpus gets to fill, not the other way around.

Output: chains.json with primary chains first (sorted by chain_id) then
accepted secondaries (sorted by chain_id). Stats printed to stdout: count
kept, count added, count rejected (with the per-rejection reason).

Always run ``apply_proposed_chains.py --proposed chains.json --dry-run``
after this script as the final structural gate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
# AI-pipeline staging lives under _pipeline/ (gitignored). The live chain
# registry is the durable artifact at vault/chains.json.
# See interviews/CLAUDE.md.
PIPELINE_DIR = VAULT_DIR / "_pipeline"
DEFAULT_PRIMARY = VAULT_DIR / "chains.json"
DEFAULT_SECONDARY = PIPELINE_DIR / "chains.proposed.lenient.json"
DEFAULT_OUTPUT = VAULT_DIR / "chains.json"

ANCHOR_LEVELS = frozenset({"L1", "L2"})
MULTI_MEMBERSHIP_CAP = 2


def load_levels() -> dict[str, str]:
    """qid -> Bloom level. Used to gate the L1/L2 anchor exemption."""
    levels: dict[str, str] = {}
    for p in QUESTIONS_DIR.rglob("*.yaml"):
        try:
            with p.open(encoding="utf-8") as f:
                d = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        qid = d.get("id")
        lvl = d.get("level")
        if qid and lvl:
            levels[qid] = lvl
    return levels


def chain_qids(chain: dict) -> list[str]:
    return [m.get("id") for m in chain.get("questions", []) if m.get("id")]


def merge(
    primary: list[dict],
    secondary: list[dict],
    levels: dict[str, str],
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Return (accepted_chains, rejected_secondaries_with_reason).

    accepted_chains contains all primaries (tier-tagged) followed by the
    secondaries that pass the cap rules.
    """
    qid_chain_count: dict[str, int] = defaultdict(int)
    accepted: list[dict] = []

    # Primary first — they always make it in, just stamped with tier.
    for ch in primary:
        entry = dict(ch)
        entry.setdefault("tier", "primary")
        for qid in chain_qids(entry):
            qid_chain_count[qid] += 1
        accepted.append(entry)

    rejected: list[tuple[dict, str]] = []

    for ch in secondary:
        qids = chain_qids(ch)
        if not qids:
            rejected.append((ch, "no qids in chain"))
            continue

        block_reason: str | None = None
        for qid in qids:
            existing = qid_chain_count.get(qid, 0)
            level = levels.get(qid, "?")
            if existing >= MULTI_MEMBERSHIP_CAP:
                block_reason = (
                    f"qid {qid} already in {existing} chain(s); cap is "
                    f"{MULTI_MEMBERSHIP_CAP}"
                )
                break
            if existing >= 1 and level not in ANCHOR_LEVELS:
                block_reason = (
                    f"qid {qid} (level={level}) already in {existing} chain; "
                    f"non-L1/L2 qids capped at 1"
                )
                break

        if block_reason:
            rejected.append((ch, block_reason))
            continue

        entry = dict(ch)
        entry.setdefault("tier", "secondary")
        for qid in qids:
            qid_chain_count[qid] += 1
        accepted.append(entry)

    return accepted, rejected


def report(
    accepted: list[dict],
    rejected: list[tuple[dict, str]],
    n_primary: int,
    n_secondary_in: int,
) -> None:
    n_accepted_secondary = len(accepted) - n_primary
    print(f"primary chains kept:      {n_primary}")
    print(f"secondary chains in:      {n_secondary_in}")
    print(f"secondary chains added:   {n_accepted_secondary}")
    print(f"secondary chains dropped: {len(rejected)}")
    if rejected:
        # Group reasons for a quick read
        by_reason: dict[str, int] = defaultdict(int)
        for _, reason in rejected:
            key = (
                "cap=2 violation" if "cap is 2" in reason
                else "non-anchor double-bind" if "capped at 1" in reason
                else "no qids" if "no qids" in reason
                else "other"
            )
            by_reason[key] += 1
        print()
        print("rejection reasons:")
        for k, v in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"  {k:30s} {v}")
        print()
        print("first 5 rejections (full reason):")
        for ch, reason in rejected[:5]:
            print(f"  {ch.get('chain_id', '?')}: {reason}")
    print(f"\nfinal merged count: {len(accepted)}")


def sort_chains(chains: list[dict]) -> list[dict]:
    """Stable order: by tier (primary first), then chain_id."""
    tier_rank = {"primary": 0, "secondary": 1}
    return sorted(
        chains,
        key=lambda c: (tier_rank.get(c.get("tier", "primary"), 9),
                       c.get("chain_id", "")),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY,
                    help=f"primary chains.json (default: {DEFAULT_PRIMARY})")
    ap.add_argument("--secondary", type=Path, default=DEFAULT_SECONDARY,
                    help=f"secondary chains.proposed.lenient.json "
                         f"(default: {DEFAULT_SECONDARY})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"output chains.json (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="report stats without writing the output")
    args = ap.parse_args()

    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    secondary = json.loads(args.secondary.read_text(encoding="utf-8"))
    levels = load_levels()
    print(f"loaded {len(primary)} primary chains, {len(secondary)} secondary "
          f"candidates, {len(levels)} qid → level entries")

    accepted, rejected = merge(primary, secondary, levels)
    accepted = sort_chains(accepted)

    report(accepted, rejected, n_primary=len(primary),
           n_secondary_in=len(secondary))

    if args.dry_run:
        print("\n--dry-run set; not writing output.")
        return 0

    args.output.write_text(json.dumps(accepted, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(accepted)} chains)")
    print("Now run: python3 interviews/vault-cli/scripts/apply_proposed_chains.py "
          "--proposed interviews/vault/chains.json --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
