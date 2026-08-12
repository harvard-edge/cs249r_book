#!/usr/bin/env python3
"""
check_references.py — Bibliography completeness validator.

Scans each volume's QMD chapters for [@citation] keys and verifies every key
is defined in the shared book bibliography at contents/references.bib.

Usage:
    # Check only (no changes):
    python3 scripts/check_references.py

    # Check only; --fix is retained for CLI compatibility but does not edit:
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
        "bib_file":     Path("contents/references.bib"),
    },
    "vol2": {
        "chapters_dir": Path("contents/vol2"),
        "bib_file":     Path("contents/references.bib"),
    },
}

# Quarto cross-reference prefixes — these are NOT citation keys
CROSSREF_PREFIXES = (
    "sec-", "fig-", "tbl-", "eq-", "lst-", "exm-", "thm-",
    "lem-", "cor-", "prp-", "cnj-", "def-", "rem-", "sol-", "alg-",
    "algo-", "ch-", "nb-",
)
KNOWN_FALSE_POSITIVE_KEYS = {
    "media", "keyframes", "import", "supports", "page", "font-face",
    "charset", "namespace", "document",
    "grad", "staticmethod", "classmethod", "property", "abstractmethod",
    "dataclass", "cached_property", "wraps",
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"(?<![=,(])@([A-Za-z][\w:.-]*)\b")


def extract_cited_keys(qmd_path: Path) -> set[str]:
    """Return all bibliography citation keys used in a QMD file."""
    keys: set[str] = set()
    text = qmd_path.read_text(encoding="utf-8")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    text = re.sub(r"<!--+.*?-->", "", text, flags=re.S)
    text = re.sub(r"(?m)^[ \t]*```[^\n]*\n.*?\n[ \t]*```[^\n]*$", "", text, flags=re.S)
    text = re.sub(r"`[^`]+`", "", text)
    for match in _CITATION_RE.finditer(text):
        key = match.group(1).rstrip(".,;:)")
        if (
            key
            and key not in KNOWN_FALSE_POSITIVE_KEYS
            and not key.lower().startswith(CROSSREF_PREFIXES)
        ):
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
    volume's chapters that are absent from the shared references.bib.
    """
    own_bib = all_bibs[vol_id]
    chapters_dir = config["chapters_dir"]

    missing: dict[str, list[str]] = {}
    for qmd in sorted(chapters_dir.rglob("*.qmd")):
        for key in extract_cited_keys(qmd):
            if key not in own_bib:
                missing.setdefault(key, []).append(str(qmd))
    return missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true",
                        help="Deprecated no-op; missing shared-bib entries require a reviewed BibTeX addition.")
    args = parser.parse_args()

    shared_bib_path = Path("contents/references.bib")
    if not shared_bib_path.exists():
        print(f"[ERROR] {shared_bib_path} not found — run from book/quarto/", file=sys.stderr)
        return 1
    shared_bib = parse_bib_entries(shared_bib_path)
    all_bibs: dict[str, dict[str, str]] = {vol_id: shared_bib for vol_id in VOLUMES}
    print(f"[INFO] shared references.bib: {len(shared_bib)} entries")

    overall_ok = True

    for vol_id, config in VOLUMES.items():
        print(f"\n{'='*60}")
        print(f"Checking {vol_id} ({config['chapters_dir']}) ...")
        print(f"{'='*60}")

        missing = check_volume(vol_id, config, all_bibs)

        if not missing:
            print(f"  ✅  All citations resolved in shared references.bib")
            continue

        overall_ok = False
        not_found_anywhere: list[str] = []

        for key, qmds in sorted(missing.items()):
            not_found_anywhere.append(key)
            status = "MISSING FROM SHARED BIB"
            print(f"  ⚠️  {key}  [{status}]")
            for q in qmds[:3]:  # limit noise
                print(f"       cited in: {q}")
            if len(qmds) > 3:
                print(f"       ... and {len(qmds) - 3} more file(s)")

        if not_found_anywhere:
            print(f"\n  ❌  {len(not_found_anywhere)} key(s) not found in any volume's bib:")
            for k in not_found_anywhere:
                print(f"       {k}")

        if args.fix:
            print("\n  --fix is no longer automatic with a shared bibliography.")
            print("  Add a reviewed BibTeX entry to contents/references.bib, or update the citation key.")

    print(f"\n{'='*60}")
    if overall_ok:
        print("✅  All book citations resolve against the shared bibliography.")
        return 0
    else:
        print("❌  Missing citations found. See above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
