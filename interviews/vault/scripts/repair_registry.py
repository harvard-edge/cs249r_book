#!/usr/bin/env python3
"""Append-only repair of id-registry.yaml.

The registry is an append-only audit log of every ID ever assigned. Two
prior events broke its integrity:

1. Commit ``8a5c3ff3c`` (2026-04-25) renamed 4,754 cohort-tagged IDs
   (e.g. ``cloud-fill-04027``, ``tinyml-cell-13251``) to clean
   ``<track>-NNNN`` form. The YAML files were renamed but the registry
   was never updated, leaving the old IDs as orphaned entries and the
   new IDs unregistered.
2. Subsequent generation runs (e.g. the 320 PASS items from the loop
   at ``_validation_results/coverage_loop/20260425_150712/``) wrote new
   YAMLs without ever appending their IDs to the registry.

This script reconciles by **appending** every disk-YAML ID not currently
in the registry. It does NOT delete the orphan entries — those are the
historical names of renamed questions and remain as audit trail.
A comment block documents the rebuild event so future readers can find
the rationale in the registry file itself.

Usage::

    python3 repair_registry.py --dry-run   # report only
    python3 repair_registry.py             # apply
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"
REGISTRY = VAULT_DIR / "id-registry.yaml"

REBUILD_TAG = "registry-rebuild-2026-04-25"

REBUILD_HEADER = """\
# ─────────────────────────────────────────────────────────────────────────
# Registry rebuild — 2026-04-25
# Reason: commit 8a5c3ff3c renamed 4,754 cohort-tagged IDs to clean
#   <track>-NNNN form, and several generation runs wrote YAMLs without
#   appending to this registry. Both events left disk YAMLs unregistered.
# Action: this block appends every disk-YAML ID not previously in the
#   registry. The earlier orphan entries (e.g. tinyml-exp2-desi-0184,
#   cloud-fill-04027) are the historical names of those questions and
#   remain as audit trail; we do NOT delete them.
# Future: ``vault build`` and ``promote_validated.py`` should append to
#   this registry on every new write so this rebuild is one-time only.
# ─────────────────────────────────────────────────────────────────────────
"""


def _registry_ids(text: str) -> set[str]:
    """Extract every ID currently in the registry."""
    return set(re.findall(r"\{id:\s*([^,]+),", text))


def _disk_ids() -> set[str]:
    return {p.stem for p in QUESTIONS_DIR.glob("*/*.yaml")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be appended; do not write.")
    args = parser.parse_args()

    if not REGISTRY.exists():
        print(f"error: registry not found at {REGISTRY}", file=sys.stderr)
        return 1

    text = REGISTRY.read_text(encoding="utf-8")
    reg_ids = _registry_ids(text)
    disk_ids = _disk_ids()

    missing = sorted(disk_ids - reg_ids)
    extra = sorted(reg_ids - disk_ids)

    print(f"registry IDs:  {len(reg_ids):,}")
    print(f"disk YAML IDs: {len(disk_ids):,}")
    print(f"missing from registry (will be appended): {len(missing):,}")
    print(f"orphan registry entries (kept as history): {len(extra):,}")

    if not missing:
        print("\nNothing to append. Registry is in sync.")
        return 0

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_lines = [
        f"  - {{id: {qid}, created_at: {timestamp}, created_by: {REBUILD_TAG}}}"
        for qid in missing
    ]

    if args.dry_run:
        print(f"\n[dry-run] would append {len(new_lines)} entries.")
        print(f"[dry-run] sample (first 3):")
        for line in new_lines[:3]:
            print(f"  {line}")
        return 0

    # Append: header block + new entries. Preserve trailing newline.
    if not text.endswith("\n"):
        text += "\n"
    text = text + "\n" + REBUILD_HEADER + "\n".join(new_lines) + "\n"
    REGISTRY.write_text(text, encoding="utf-8")
    print(f"\n✓ appended {len(new_lines)} entries to {REGISTRY}")
    print(f"  registry now has {len(reg_ids) + len(new_lines):,} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
