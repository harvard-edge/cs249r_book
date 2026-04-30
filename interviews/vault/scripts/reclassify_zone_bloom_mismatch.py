#!/usr/bin/env python3
"""Reclassify questions whose (zone, bloom_level) pair violates the matrix.

The ZONE_BLOOM_AFFINITY matrix in interviews/vault/schema/enums.py is the
canonical "this zone admits these Bloom verbs" rule. Items that violate
the matrix have a self-contradicting classification: zone says one thing,
bloom_level says another.

Per the lint-calibration consensus 2026-04-25, we trust bloom_level as
the canonical truth (Bloom's verbs are a tighter, standardized vocabulary
than the StaffML zones). This script rewrites `zone` to the canonical
zone for the question's bloom_level, using BLOOM_CANONICAL_ZONE.

Usage::

    python3 reclassify_zone_bloom_mismatch.py --dry-run   # report only
    python3 reclassify_zone_bloom_mismatch.py             # apply
    python3 reclassify_zone_bloom_mismatch.py --report-csv /tmp/changes.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"

sys.path.insert(0, str(VAULT_DIR / "schema"))
from enums import BLOOM_CANONICAL_ZONE, ZONE_BLOOM_AFFINITY  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-csv", type=Path, default=None,
                        help="Write a per-item CSV of changes for audit.")
    args = parser.parse_args()

    changes: list[dict[str, str]] = []
    cell_counts: Counter[tuple[str, str, str]] = Counter()  # (old_zone, bloom, new_zone)
    skipped_no_bloom = 0

    for p in sorted(QUESTIONS_DIR.rglob('*.yaml')):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d:
            continue
        zone = d.get("zone")
        bloom = d.get("bloom_level")
        if zone is None:
            continue
        if bloom is None:
            skipped_no_bloom += 1
            continue
        admits = ZONE_BLOOM_AFFINITY.get(zone, set())
        if bloom in admits:
            continue  # valid, no change needed

        # Conflict: rewrite zone to bloom-canonical zone
        new_zone = BLOOM_CANONICAL_ZONE.get(bloom)
        if not new_zone:
            print(f"  ! {d.get('id')}: bloom={bloom!r} not in BLOOM_CANONICAL_ZONE",
                  file=sys.stderr)
            continue

        change = {
            "id": d["id"],
            "track": d["track"],
            "level": d.get("level", "?"),
            "old_zone": zone,
            "new_zone": new_zone,
            "bloom_level": bloom,
            "topic": d.get("topic", "?"),
            "title": d.get("title", "")[:80],
        }
        changes.append(change)
        cell_counts[(zone, bloom, new_zone)] += 1

        if not args.dry_run:
            d["zone"] = new_zone
            p.write_text(
                yaml.safe_dump(d, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

    print(f"Items skipped (no bloom_level): {skipped_no_bloom}")
    print(f"Items with zone-bloom conflict: {len(changes)}")
    print()
    print("Reclassification by (old_zone, bloom_level) → new_zone:")
    print(f"{'old_zone':<14s} {'bloom':<10s} {'→':1s} {'new_zone':<14s} {'count':>6s}")
    print("-" * 50)
    for (old_z, bloom, new_z), n in sorted(cell_counts.items(), key=lambda x: -x[1]):
        print(f"  {old_z:<14s} {bloom:<10s} → {new_z:<14s} {n:>6d}")

    if args.report_csv:
        with open(args.report_csv, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "track", "level", "old_zone",
                               "new_zone", "bloom_level", "topic", "title"],
            )
            writer.writeheader()
            writer.writerows(changes)
        print(f"\nPer-item CSV: {args.report_csv}")

    if args.dry_run:
        print(f"\n[dry-run] would reclassify {len(changes)} items.")
    else:
        print(f"\n✓ reclassified {len(changes)} items.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
