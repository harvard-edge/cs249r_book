#!/usr/bin/env python3
"""Comprehensive bib/qmd link-integrity check.

Catches four failure modes that the existing per-volume `binder check
refs --scope citations` does NOT cover:

1. **Unresolved cites** — `[@key]` appears in a `.qmd` but is not
   defined in *any* bib in the repo.
2. **Scope violations** — a `vol1/**/*.qmd` cites a key that *is*
   defined, but only in `vol2/backmatter/references.bib`. The vol1 PDF
   bibliography won't contain it, so the citation renders as `[?]`.
   This is the "cross-volume leak" class — silent under a pooled check
   that just asks "is the key defined somewhere?"
3. **Out-of-tree qmd files** — `interviews/**/*.qmd`, `tinytorch/**/*.qmd`,
   `mlsysim/**/*.qmd`, `periodic-table/**/*.qmd` are skipped by the
   per-volume check entirely.
4. **Orphan bib entries** — keys defined in any `references.bib` but
   not cited by any `.qmd` anywhere. These are wasted maintenance
   surface.

The static scope mapping (SCOPES, below) is the contract: each tree of
`.qmd` files is allowed to resolve only against a specific list of bib
files. Vol1 against vol1's bib, vol2 against vol2's, the four companion
papers each against their own. The shared book-level frontmatter and
backmatter are allowed against either volume's bib.

Usage:
    python3 book/tools/scripts/check_bib_qmd_integrity.py
    python3 book/tools/scripts/check_bib_qmd_integrity.py --json
    python3 book/tools/scripts/check_bib_qmd_integrity.py --orphans-only
    python3 book/tools/scripts/check_bib_qmd_integrity.py --strict       # exit 1 on orphans too
    python3 book/tools/scripts/check_bib_qmd_integrity.py --no-scope-check  # pool all bibs (legacy)

Exit codes:
    0  — every cited key resolves within its scope's allowed bibs
    1  — unresolved citation (in no bib) OR scope violation (in wrong-scope bib)
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

# Static scope mapping: each citation source (qmd or tex) is matched to the
# first scope whose `paths` prefix matches the file's repo-relative path.
# That scope's `bibs` list is the *only* set of bib files allowed to resolve
# the source's citations. A vol1 chapter citing a vol2-only key is a scope
# violation — the vol1 PDF bibliography won't contain that entry, so the
# rendered citation breaks even though the key technically "exists" in the
# repo. Order matters: more-specific prefixes must come before less-specific
# ones (vol1/vol2 before the shared book-level frontmatter/backmatter).
SCOPES = [
    {
        "name": "vol1",
        "paths": ["book/quarto/contents/vol1/"],
        "bibs":  ["book/quarto/contents/vol1/backmatter/references.bib"],
    },
    {
        "name": "vol2",
        "paths": ["book/quarto/contents/vol2/"],
        "bibs":  ["book/quarto/contents/vol2/backmatter/references.bib"],
    },
    {
        # Book-level frontmatter and backmatter are shared between volumes.
        # A citation here may resolve against either volume's bib.
        "name": "book-shared",
        "paths": [
            "book/quarto/contents/frontmatter/",
            "book/quarto/contents/backmatter/",
        ],
        "bibs": [
            "book/quarto/contents/vol1/backmatter/references.bib",
            "book/quarto/contents/vol2/backmatter/references.bib",
        ],
    },
    {
        "name": "interviews",
        "paths": ["interviews/"],
        "bibs":  ["interviews/paper/references.bib"],
    },
    {
        "name": "tinytorch",
        "paths": ["tinytorch/"],
        "bibs":  ["tinytorch/paper/references.bib"],
    },
    {
        "name": "mlsysim",
        "paths": ["mlsysim/"],
        "bibs":  ["mlsysim/paper/references.bib", "mlsysim/docs/references.bib"],
    },
    {
        "name": "periodic-table",
        "paths": ["periodic-table/"],
        "bibs":  ["periodic-table/paper/references.bib"],
    },
]

# Flat list, derived from SCOPES — the union of every bib file referenced
# by any scope. Used for orphan detection and for the legacy --no-scope-check
# pooled mode.
BIB_FILES = sorted({b for s in SCOPES for b in s["bibs"]})

# Where .qmd source files live (scanned recursively)
QMD_ROOTS = [
    "book/quarto/contents",
    "interviews",
    "tinytorch",
    "mlsysim",
    "periodic-table",
]

# Companion paper LaTeX sources. Each falls inside the scope of its
# parent directory (e.g. interviews/paper/paper.tex → "interviews" scope).
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
    # Strip HTML comments BEFORE fence detection. A commented-out figure can
    # legitimately contain a stray ``` (e.g., a leftover code-fence wrapper
    # around a TikZ block) that would otherwise pair with the next real fence
    # and eat the prose in between. Both `<!-- ... -->` and the looser
    # `<!--- ... -->` (Quarto-flavored four-dash open) are handled.
    text = re.sub(r"<!--+.*?-->", "", text, flags=re.S)
    # Strip fenced code blocks. Both fence markers must sit at the start of a
    # line (modulo leading whitespace). The earlier `r"```.*?```"` was buggy:
    # it paired *any* two triple-backticks, including mid-line occurrences,
    # and silently ate prose between unrelated fences. A real Quarto/markdown
    # fence is line-anchored.
    text = re.sub(
        r"(?m)^[ \t]*```[^\n]*\n.*?\n[ \t]*```[^\n]*$",
        "", text, flags=re.S,
    )
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


def find_scope(rel_path: str) -> dict | None:
    """Return the first scope whose path prefix matches `rel_path`, or None.

    `rel_path` is the file's path relative to REPO_ROOT, with forward slashes.
    Order in SCOPES matters: more-specific prefixes (vol1/, vol2/) must come
    before less-specific ones (frontmatter/, backmatter/).
    """
    for scope in SCOPES:
        for prefix in scope["paths"]:
            if rel_path.startswith(prefix):
                return scope
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON")
    parser.add_argument("--orphans-only", action="store_true",
                        help="Only report orphan bib entries (cited-nowhere)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero on orphan bib entries too")
    parser.add_argument("--no-scope-check", action="store_true",
                        help="Pool every bib together (legacy mode); skip "
                             "per-scope vol1/vol2 enforcement")
    parser.add_argument("--show-false-positives", action="store_true",
                        help="Show suppressed false-positive keys")
    args = parser.parse_args()

    # Load all bib keys, tracking which file(s) each lives in.
    # `bib_keys_by_file[bibfile] -> set(keys)` lets us answer scope questions
    # ("is key K defined in bib file B?") without rescanning.
    all_bib_keys: dict[str, list[str]] = defaultdict(list)
    bib_keys_by_file: dict[str, set[str]] = {}
    bib_total = 0
    for bp in BIB_FILES:
        path = REPO_ROOT / bp
        if not path.exists():
            print(f"warn: {bp} missing", file=sys.stderr)
            bib_keys_by_file[bp] = set()
            continue
        keys = parse_bib_keys(path.read_text(encoding="utf-8"))
        bib_keys_by_file[bp] = keys
        for k in keys:
            all_bib_keys[k].append(bp)
        bib_total += len(keys)

    # Per-citation records: for each (file, key) pair, we'll determine its
    # scope and classify it as resolved / scope-violation / unresolved.
    # `all_cites` keeps the cite-key → list-of-files mapping for the
    # legacy pooled view and the orphan computation.
    all_cites: dict[str, list[str]] = defaultdict(list)
    citation_records: list[dict] = []  # each: {file, key, scope}
    files_outside_scope: set[str] = set()

    qmds = discover_qmds()
    for q in qmds:
        try:
            text = q.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        cites = extract_qmd_cites(text)
        rel = str(q.relative_to(REPO_ROOT))
        scope = find_scope(rel)
        if scope is None and cites:
            files_outside_scope.add(rel)
        for k in cites:
            all_cites[k].append(rel)
            citation_records.append({"file": rel, "key": k, "scope": scope})

    # Load all tex citations (same scope-by-path logic)
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
        scope = find_scope(rel)
        if scope is None and cites:
            files_outside_scope.add(rel)
        for k in cites:
            all_cites[k].append(rel)
            citation_records.append({"file": rel, "key": k, "scope": scope})

    cited_keys = set(all_cites.keys())
    bib_keys = set(all_bib_keys.keys())
    orphans = sorted(bib_keys - cited_keys)

    # Classify each citation record. Three buckets:
    #   - resolved        : key is in one of the scope's allowed bibs
    #   - scope_violation : key is defined somewhere, but NOT in scope's bibs
    #   - unresolved      : key is in no bib file at all
    # In --no-scope-check mode, scope_violation collapses into resolved.
    unresolved_records: list[dict] = []
    scope_violation_records: list[dict] = []
    for rec in citation_records:
        k = rec["key"]
        scope = rec["scope"]
        if k not in bib_keys:
            unresolved_records.append(rec)
            continue
        if args.no_scope_check or scope is None:
            continue  # legacy / unscoped: any bib counts
        allowed_bibs = set(scope["bibs"])
        defining_bibs = set(all_bib_keys[k])
        if defining_bibs & allowed_bibs:
            continue  # resolved within scope
        scope_violation_records.append({
            **rec,
            "scope_name": scope["name"],
            "allowed_bibs": sorted(allowed_bibs),
            "found_in_bibs": sorted(defining_bibs),
        })

    # Aggregate unresolved keys (key → list of files), preserving original
    # report shape for backward compatibility.
    unresolved_by_key: dict[str, list[str]] = defaultdict(list)
    for r in unresolved_records:
        if r["file"] not in unresolved_by_key[r["key"]]:
            unresolved_by_key[r["key"]].append(r["file"])
    unresolved = sorted(unresolved_by_key.keys())

    # Aggregate scope violations: group by (scope_name, key) to avoid
    # printing the same leak once per chapter that hit it.
    sv_grouped: dict[tuple[str, str], dict] = {}
    for r in scope_violation_records:
        gk = (r["scope_name"], r["key"])
        if gk not in sv_grouped:
            sv_grouped[gk] = {
                "scope": r["scope_name"],
                "key": r["key"],
                "allowed_bibs": r["allowed_bibs"],
                "found_in_bibs": r["found_in_bibs"],
                "cited_in": [],
            }
        if r["file"] not in sv_grouped[gk]["cited_in"]:
            sv_grouped[gk]["cited_in"].append(r["file"])
    scope_violations = sorted(
        sv_grouped.values(),
        key=lambda v: (v["scope"], v["key"]),
    )

    summary = {
        "mode": "pooled" if args.no_scope_check else "per-scope",
        "qmd_files_scanned": len(qmds),
        "tex_files_scanned": tex_files_found,
        "bib_files_scanned": len([1 for bp in BIB_FILES if (REPO_ROOT / bp).exists()]),
        "files_outside_any_scope": sorted(files_outside_scope),
        "unique_cited_keys": len(cited_keys),
        "unique_bib_keys": len(bib_keys),
        "bib_total_entries": bib_total,
        "unresolved_count": len(unresolved),
        "scope_violation_count": len(scope_violations),
        "orphan_count": len(orphans),
        "unresolved": [
            {"key": k, "cited_in": unresolved_by_key[k][:5],
             "extra_files_omitted": max(0, len(unresolved_by_key[k]) - 5)}
            for k in unresolved
        ],
        "scope_violations": [
            {**v, "extra_files_omitted": max(0, len(v["cited_in"]) - 5),
             "cited_in": v["cited_in"][:5]}
            for v in scope_violations
        ],
        "orphans": [
            {"key": k, "defined_in": all_bib_keys[k]}
            for k in orphans
        ],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return _exit_code(unresolved, scope_violations, orphans, args.strict)

    # Human-readable
    mode_label = "pooled (legacy)" if args.no_scope_check else "per-scope"
    print(f"# Bib / QMD / TeX link integrity report ({mode_label})\n")
    print(f"  qmd files scanned:    {summary['qmd_files_scanned']}")
    print(f"  tex files scanned:    {summary['tex_files_scanned']}")
    print(f"  bib files scanned:    {summary['bib_files_scanned']}")
    print(f"  unique cited keys:    {summary['unique_cited_keys']}")
    print(f"  unique bib keys:      {summary['unique_bib_keys']} ({summary['bib_total_entries']} total entries inc. duplicates)")
    print(f"  unresolved (broken):  {summary['unresolved_count']}")
    if not args.no_scope_check:
        print(f"  scope violations:     {summary['scope_violation_count']}")
    print(f"  orphans (uncited):    {summary['orphan_count']}")

    if files_outside_scope and not args.no_scope_check:
        print(f"\n  warn: {len(files_outside_scope)} citation source(s) match no scope — citations from these are not scope-checked:")
        for f in sorted(files_outside_scope)[:10]:
            print(f"    - {f}")
        if len(files_outside_scope) > 10:
            print(f"    ... ({len(files_outside_scope) - 10} more)")

    if not args.orphans_only:
        print(f"\n## Unresolved citations ({len(unresolved)})")
        if not unresolved:
            print("  ✓ none — every cited key resolves to a bib entry\n")
        else:
            print("  These are real broken links — the key is in NO bib file:\n")
            for k in unresolved[:50]:
                src = unresolved_by_key[k][:3]
                more = "" if len(unresolved_by_key[k]) <= 3 else f" (+{len(unresolved_by_key[k]) - 3} more)"
                print(f"    @{k}")
                for s in src:
                    print(f"      cited in: {s}")
                if more: print(f"      {more}")
            if len(unresolved) > 50:
                print(f"    ... ({len(unresolved) - 50} more — see --json for full list)")

        if not args.no_scope_check:
            print(f"\n## Scope violations ({len(scope_violations)})")
            if not scope_violations:
                print("  ✓ none — every cite resolves within its volume's bib\n")
            else:
                print("  Citation source's scope does not include the bib that defines the key.")
                print("  These render as broken in single-volume PDF builds:\n")
                for v in scope_violations[:50]:
                    print(f"    [{v['scope']}] @{v['key']}")
                    print(f"      allowed: {', '.join(v['allowed_bibs'])}")
                    print(f"      found in: {', '.join(v['found_in_bibs'])}")
                    for s in v["cited_in"][:3]:
                        print(f"      cited in: {s}")
                    if len(v["cited_in"]) > 3:
                        print(f"      (+{len(v['cited_in']) - 3} more files)")
                if len(scope_violations) > 50:
                    print(f"    ... ({len(scope_violations) - 50} more — see --json for full list)")

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

    return _exit_code(unresolved, scope_violations, orphans, args.strict)


def _exit_code(unresolved: list, scope_violations: list, orphans: list, strict: bool) -> int:
    if unresolved or scope_violations:
        return 1
    if strict and orphans:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
