#!/usr/bin/env python3
"""
check_references.py — Bibliography completeness validator.

Scans each volume's QMD chapters for [@citation] keys and verifies every key
is defined in that volume's own references.bib. Reports missing entries and
optionally copies them from the other volume's bib to make each volume
fully standalone.

Usage:
    # Check only (no changes):
    python3 scripts/check_references.py

    # Check and auto-fix by copying missing entries across volumes:
    python3 scripts/check_references.py --fix

Run from book/quarto/ directory.
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOLUMES = {
    "vol1": {
        "chapters_dir": Path("contents/vol1"),
        "bib_file":     Path("contents/vol1/backmatter/references.bib"),
    },
    "vol2": {
        "chapters_dir": Path("contents/vol2"),
        "bib_file":     Path("contents/vol2/backmatter/references.bib"),
    },
}

# Quarto cross-reference prefixes — these are NOT citation keys
CROSSREF_PREFIXES = (
    "sec-", "fig-", "tbl-", "eq-", "lst-", "exm-", "thm-",
    "lem-", "cor-", "prp-", "cnj-", "def-", "rem-", "sol-", "alg-",
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[@([^\]]+)\]")
_KEY_SEP_RE  = re.compile(r"[;,\s]+")


def extract_cited_keys(qmd_path: Path) -> set[str]:
    """Return all bibliography citation keys used in a QMD file."""
    keys: set[str] = set()
    text = qmd_path.read_text(encoding="utf-8")
    for match in _CITATION_RE.finditer(text):
        for raw in _KEY_SEP_RE.split(match.group(1)):
            key = raw.lstrip("@").strip()
            if key and not any(key.startswith(p) for p in CROSSREF_PREFIXES):
                keys.add(key)
    return keys


def parse_bib_entries(bib_path: Path) -> dict[str, str]:
    """Return {key: full_entry_text} for every entry in a .bib file."""
    entries: dict[str, str] = {}
    if not bib_path.exists():
        return entries

    text = bib_path.read_text(encoding="utf-8")
    # Split on entry boundaries — each starts with @Type{
    raw_entries = re.split(r"(?=^@)", text, flags=re.MULTILINE)
    for raw in raw_entries:
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r"@[A-Za-z]+\{([^,]+),", raw)
        if m:
            entries[m.group(1).strip()] = raw
    return entries


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def check_volume(vol_id: str, config: dict, all_bibs: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """
    Returns {missing_key: [qmd_files_that_cite_it]} for citations in this
    volume's chapters that are absent from its own references.bib.
    """
    own_bib = all_bibs[vol_id]
    chapters_dir = config["chapters_dir"]

    missing: dict[str, list[str]] = {}
    for qmd in sorted(chapters_dir.rglob("*.qmd")):
        for key in extract_cited_keys(qmd):
            if key not in own_bib:
                missing.setdefault(key, []).append(str(qmd))
    return missing


def find_entry_in_other_volumes(key: str, vol_id: str, all_bibs: dict[str, dict[str, str]]) -> str | None:
    """Search other volumes' bibs for a missing key."""
    for other_id, bib in all_bibs.items():
        if other_id != vol_id and key in bib:
            return bib[key]
    return None


def append_entries_to_bib(bib_path: Path, entries: list[str]) -> None:
    """Append bib entries to the end of a .bib file."""
    existing = bib_path.read_text(encoding="utf-8")
    with bib_path.open("a", encoding="utf-8") as f:
        if not existing.endswith("\n"):
            f.write("\n")
        f.write("\n")
        for entry in entries:
            f.write(entry + "\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true",
                        help="Copy missing entries from the other volume's bib to make each volume standalone.")
    args = parser.parse_args()

    # Load all bibs upfront
    all_bibs: dict[str, dict[str, str]] = {}
    for vol_id, config in VOLUMES.items():
        bib_path = config["bib_file"]
        if not bib_path.exists():
            print(f"[ERROR] {bib_path} not found — run from book/quarto/", file=sys.stderr)
            return 1
        all_bibs[vol_id] = parse_bib_entries(bib_path)
        print(f"[INFO] {vol_id}/references.bib: {len(all_bibs[vol_id])} entries")

    overall_ok = True

    for vol_id, config in VOLUMES.items():
        print(f"\n{'='*60}")
        print(f"Checking {vol_id} ({config['chapters_dir']}) ...")
        print(f"{'='*60}")

        missing = check_volume(vol_id, config, all_bibs)

        if not missing:
            print(f"  ✅  All citations resolved in {vol_id}/references.bib")
            continue

        overall_ok = False
        not_found_anywhere: list[str] = []
        found_in_other: list[tuple[str, str]] = []

        for key, qmds in sorted(missing.items()):
            entry = find_entry_in_other_volumes(key, vol_id, all_bibs)
            if entry:
                found_in_other.append((key, entry))
                status = "found in other volume"
            else:
                not_found_anywhere.append(key)
                status = "NOT FOUND ANYWHERE"
            print(f"  ⚠️  {key}  [{status}]")
            for q in qmds[:3]:  # limit noise
                print(f"       cited in: {q}")
            if len(qmds) > 3:
                print(f"       ... and {len(qmds) - 3} more file(s)")

        if not_found_anywhere:
            print(f"\n  ❌  {len(not_found_anywhere)} key(s) not found in any volume's bib:")
            for k in not_found_anywhere:
                print(f"       {k}")

        if args.fix and found_in_other:
            bib_path = config["bib_file"]
            entries_to_add = [entry for _, entry in found_in_other]
            append_entries_to_bib(bib_path, entries_to_add)
            # Update our in-memory view so subsequent volumes see the additions
            for key, entry in found_in_other:
                all_bibs[vol_id][key] = entry
            print(f"\n  ✏️   Copied {len(found_in_other)} entr(ies) into {bib_path}")
            if not_found_anywhere:
                print(f"  ⚠️   {len(not_found_anywhere)} key(s) still need manual addition.")
        elif found_in_other and not args.fix:
            print(f"\n  💡  Run with --fix to copy {len(found_in_other)} entr(ies) from other volume(s).")

    print(f"\n{'='*60}")
    if overall_ok:
        print("✅  All volumes are standalone — no missing citations.")
        return 0
    else:
        print("❌  Missing citations found. See above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
