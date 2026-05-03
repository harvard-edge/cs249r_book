#!/usr/bin/env python3
"""Comprehensive bib/qmd link-integrity check.

Catches three failure modes that the existing per-volume `binder check
refs --scope citations` does NOT cover:

1. **Cross-volume cites** — a `vol1/**/*.qmd` citing a key only in
   `vol2/backmatter/references.bib`. The existing per-volume check
   misses these.
2. **Out-of-tree qmd files** — `interviews/**/*.qmd`, `tinytorch/**/*.qmd`,
   `mlsysim/**/*.qmd`, `periodic-table/**/*.qmd` are skipped by the
   per-volume check entirely.
3. **Orphan bib entries** — keys defined in any `references.bib` but
   not cited by any `.qmd` anywhere. These are wasted maintenance
   surface (every audit re-checks them, every renderer parses them).

The existing check stays where it is; this script is the "belt and
suspenders" sweep that proves no `[@key]` is dangling and no bib entry
is unused.

Usage:
    python3 book/tools/scripts/check_bib_qmd_integrity.py
    python3 book/tools/scripts/check_bib_qmd_integrity.py --json
    python3 book/tools/scripts/check_bib_qmd_integrity.py --orphans-only
    python3 book/tools/scripts/check_bib_qmd_integrity.py --strict     # exit 1 on orphans too

Exit codes:
    0  — every cited key resolves to some bib (default, ignoring orphans)
    1  — at least one cited key is missing from every bib (broken link)
    2  — --strict mode and orphan bib entries exist
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Repo root (this script lives at book/tools/scripts/)
REPO_ROOT = Path(__file__).resolve().parents[3]

# Where bib files live
BIB_FILES = [
    "book/quarto/contents/vol1/backmatter/references.bib",
    "book/quarto/contents/vol2/backmatter/references.bib",
    "interviews/paper/references.bib",
    "tinytorch/paper/references.bib",
    "mlsysim/paper/references.bib",
    "mlsysim/docs/references.bib",
    "periodic-table/paper/references.bib",
]

# Where .qmd source files live
QMD_ROOTS = [
    "book/quarto/contents",
    "interviews",
    "tinytorch",
    "mlsysim",
    "periodic-table",
]

# Companion paper LaTeX sources
TEX_FILES = [
    "interviews/paper/paper.tex",
    "tinytorch/paper/paper.tex",
    "mlsysim/paper/paper.tex",
    "periodic-table/paper/paper.tex",
]

# Excluded path tokens (build artifacts, deps, etc.)
EXCLUDE = ("_build", "_site", "node_modules", ".git", "__pycache__",
           ".venv", "venv", ".aiconfigs-local")

# Quarto cross-reference prefixes — these aren't bibliography citations
NON_CITE_PREFIXES = (
    "sec-", "fig-", "tbl-", "eq-", "lst-", "exr-", "exm-",
    "thm-", "cor-", "cnj-", "def-", "prp-", "rem-", "prf-", "alg-",
)

# Keys that look like CSS / JS / Python decorators / emails — false positives
# (CSS at-rules: @media, @keyframes, @import; emails: @eecs.harvard.edu;
# Python decorators: @grad, @staticmethod; etc.)
KNOWN_FALSE_POSITIVE_KEYS = {
    "media", "keyframes", "import", "supports", "page", "font-face",
    "charset", "namespace", "document",  # CSS at-rules
    "grad", "staticmethod", "classmethod", "property", "abstractmethod",
    "dataclass", "cached_property", "wraps",  # Python decorators
    "eecs",  # email tail
}

# A bib key starts with a letter and contains word chars / colons / hyphens.
# Lookbehind excludes `=@` / `(@` / `,@` patterns that aren't real citations
# (e.g., Python decorators `func(@x)` or matmul notation `A@B`).
CITE_RE = re.compile(r"(?<![=,(])\[?@([A-Za-z][\w:.-]*)\b")

# LaTeX citations: \cite{key}, \citep{key1,key2}, \citet{key}, etc.
TEX_CITE_RE = re.compile(r"\\cite[a-z]*\*?\{([^}]+)\}")

# Single-uppercase-letter "keys" (e.g., A, B, X) are matrix/variable names
# in math-style markdown like `A@B` (matmul). Not bib citations.
SINGLE_LETTER_RE = re.compile(r"^[A-Z](\.[A-Z])*$")


def parse_bib_keys(text: str) -> set[str]:
    """Extract all @entry_type{KEY,...} keys from a bib file."""
    return set(re.findall(r"^@\w+\s*\{\s*([\w:_-]+)\s*,", text, re.M))


def extract_qmd_cites(text: str) -> set[str]:
    """Extract all citation keys from .qmd content, skipping code/yaml/inline-backticks."""
    # Strip YAML frontmatter
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    # Strip fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    # Strip HTML style/script
    text = re.sub(r"<(style|script)\b[^>]*>.*?</\1>", "", text, flags=re.S)
    # Strip inline backticks
    text = re.sub(r"`[^`]+`", "", text)

    keys = set()
    for m in CITE_RE.finditer(text):
        k = m.group(1).rstrip(".,;:)")
        if not k:
            continue
        if k.startswith(NON_CITE_PREFIXES):
            continue
        if k in KNOWN_FALSE_POSITIVE_KEYS:
            continue
        # Skip single uppercase letters or short variable patterns (math notation)
        if SINGLE_LETTER_RE.match(k):
            continue
        # Skip purely-numeric trailing tokens like "1.5"
        if re.match(r"^\d+\.\d+", k):
            continue
        keys.add(k)
    return keys


def extract_tex_cites(text: str) -> set[str]:
    """Extract all citation keys from LaTeX content."""
    # Strip comments
    text = re.sub(r"(?<!\\)%.*", "", text)
    keys = set()
    for m in TEX_CITE_RE.finditer(text):
        # Handle comma-separated lists: \cite{key1, key2}
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                keys.add(k)
    return keys


def discover_qmds() -> list[Path]:
    """All .qmd files under QMD_ROOTS, minus excluded path components."""
    out = []
    for root in QMD_ROOTS:
        p = REPO_ROOT / root
        if not p.exists():
            continue
        for q in p.rglob("*.qmd"):
            if any(part in EXCLUDE for part in q.parts):
                continue
            out.append(q)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON")
    parser.add_argument("--orphans-only", action="store_true",
                        help="Only report orphan bib entries (cited-nowhere)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero on orphan bib entries too")
    parser.add_argument("--show-false-positives", action="store_true",
                        help="Show suppressed false-positive keys")
    args = parser.parse_args()

    # Load all bib keys, tracking which file each lives in
    all_bib_keys: dict[str, list[str]] = defaultdict(list)
    bib_total = 0
    for bp in BIB_FILES:
        path = REPO_ROOT / bp
        if not path.exists():
            print(f"warn: {bp} missing", file=sys.stderr)
            continue
        keys = parse_bib_keys(path.read_text(encoding="utf-8"))
        for k in keys:
            all_bib_keys[k].append(bp)
        bib_total += len(keys)

    # Load all qmd citations, tracking which files reference each key
    all_cites: dict[str, list[str]] = defaultdict(list)
    qmds = discover_qmds()
    for q in qmds:
        try:
            text = q.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        cites = extract_qmd_cites(text)
        rel = str(q.relative_to(REPO_ROOT))
        for k in cites:
            all_cites[k].append(rel)

    # Load all tex citations
    tex_files_found = 0
    for tp in TEX_FILES:
        path = REPO_ROOT / tp
        if not path.exists():
            continue
        tex_files_found += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        cites = extract_tex_cites(text)
        rel = str(path.relative_to(REPO_ROOT))
        for k in cites:
            all_cites[k].append(rel)

    # Compute the two failure sets
    cited_keys = set(all_cites.keys())
    bib_keys = set(all_bib_keys.keys())
    unresolved = sorted(cited_keys - bib_keys)
    orphans = sorted(bib_keys - cited_keys)

    summary = {
        "qmd_files_scanned": len(qmds),
        "tex_files_scanned": tex_files_found,
        "bib_files_scanned": len([1 for bp in BIB_FILES if (REPO_ROOT / bp).exists()]),
        "unique_cited_keys": len(cited_keys),
        "unique_bib_keys": len(bib_keys),
        "bib_total_entries": bib_total,
        "unresolved_count": len(unresolved),
        "orphan_count": len(orphans),
        "unresolved": [
            {"key": k, "cited_in": all_cites[k][:5],
             "extra_files_omitted": max(0, len(all_cites[k]) - 5)}
            for k in unresolved
        ],
        "orphans": [
            {"key": k, "defined_in": all_bib_keys[k]}
            for k in orphans
        ],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return _exit_code(unresolved, orphans, args.strict)

    # Human-readable
    print(f"# Bib / QMD / TeX link integrity report\n")
    print(f"  qmd files scanned:    {summary['qmd_files_scanned']}")
    print(f"  tex files scanned:    {summary['tex_files_scanned']}")
    print(f"  bib files scanned:    {summary['bib_files_scanned']}")
    print(f"  unique cited keys:    {summary['unique_cited_keys']}")
    print(f"  unique bib keys:      {summary['unique_bib_keys']} ({summary['bib_total_entries']} total entries inc. duplicates)")
    print(f"  unresolved (broken):  {summary['unresolved_count']}")
    print(f"  orphans (uncited):    {summary['orphan_count']}")

    if not args.orphans_only:
        print(f"\n## Unresolved citations ({len(unresolved)})")
        if not unresolved:
            print("  ✓ none — every cited key resolves to a bib entry\n")
        else:
            print("  These are real broken links — fix immediately:\n")
            for k in unresolved[:50]:
                src = all_cites[k][:3]
                more = "" if len(all_cites[k]) <= 3 else f" (+{len(all_cites[k]) - 3} more)"
                print(f"    @{k}")
                for s in src:
                    print(f"      cited in: {s}")
                if more: print(f"      {more}")
            if len(unresolved) > 50:
                print(f"    ... ({len(unresolved) - 50} more — see --json for full list)")

    print(f"\n## Orphan bib entries ({len(orphans)})")
    if not orphans:
        print("  ✓ none — every bib entry is cited somewhere\n")
    else:
        print("  These bib entries are defined but cited nowhere — candidates for deletion:\n")
        # Group by which bib file
        by_file = defaultdict(list)
        for k in orphans:
            for bf in all_bib_keys[k]:
                by_file[bf].append(k)
        for bf, keys in sorted(by_file.items()):
            print(f"  {bf}: {len(keys)} orphans")
            for k in keys[:10]:
                print(f"    - {k}")
            if len(keys) > 10:
                print(f"    ... ({len(keys) - 10} more)")
            print()

    return _exit_code(unresolved, orphans, args.strict)


def _exit_code(unresolved: list, orphans: list, strict: bool) -> int:
    if unresolved:
        return 1
    if strict and orphans:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
