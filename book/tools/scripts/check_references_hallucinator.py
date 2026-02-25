#!/usr/bin/env python3
"""
Validate book bibliography entries with hallucinator.

Uses the hallucinator library (https://github.com/gianlucasb/hallucinator) to check
references from the project's .bib files against academic databases (CrossRef, arXiv,
DBLP, Semantic Scholar, etc.). Helps detect typos, wrong DOIs, or fabricated refs.

Requirements (install separately):
  pip install hallucinator bibtexparser

Usage:
  # From repo root
  python3 book/tools/scripts/check_references_hallucinator.py

  # Specific .bib files
  python3 book/tools/scripts/check_references_hallucinator.py \\
    book/quarto/contents/vol1/backmatter/references.bib

  # Save report
  python3 book/tools/scripts/check_references_hallucinator.py --output report.txt

Optional API keys (env vars) for better coverage and fewer rate limits:
  OPENALEX_KEY  - OpenAlex (openalex.org); free at https://openalex.org/settings/api
  S2_API_KEY    - Semantic Scholar; free, request at https://www.semanticscholar.org/product/api
Without keys, those DBs still work but with stricter rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

try:
    import bibtexparser
except ImportError:
    print("Missing dependency: pip install bibtexparser", file=sys.stderr)
    sys.exit(1)

try:
    from hallucinator import Reference, Validator, ValidatorConfig
except ImportError:
    print("Missing dependency: pip install hallucinator", file=sys.stderr)
    sys.exit(1)


# Default .bib files relative to repo root (from project root)
DEFAULT_BIB_PATHS = [
    "book/quarto/contents/vol1/backmatter/references.bib",
    "book/quarto/contents/vol2/backmatter/references.bib",
]

MIN_TITLE_WORDS = 4  # Skip very short titles (likely false matches)


def _to_ascii(s: str) -> str:
    """Replace non-ASCII chars with ASCII equivalents so hallucinator's Rust code doesn't panic on Unicode."""
    if not s:
        return s
    n = unicodedata.normalize("NFKD", s)
    return n.encode("ascii", "ignore").decode("ascii")


def _normalize_title(raw: str) -> str:
    """Strip braces and collapse whitespace."""
    if not raw:
        return ""
    t = re.sub(r"[\{\}]", "", raw)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _parse_authors(author_field: str) -> list[str]:
    """Parse BibTeX author string into list of family names (or full name if no comma)."""
    if not author_field or not author_field.strip():
        return []
    authors = []
    for part in re.split(r"\s+and\s+", author_field, flags=re.IGNORECASE):
        part = part.strip()
        if not part:
            continue
        # BibTeX often "Last, First" — use Last for matching
        if "," in part:
            family = part.split(",", 1)[0].strip()
        else:
            family = part
        # Drop LaTeX accents/braces for matching
        family = re.sub(r"\\[a-z]+\{([^}]*)\}", r"\1", family)
        family = re.sub(r"[{}\\]", "", family).strip()
        # Hallucinator's Rust code panics on non-ASCII; normalize to ASCII
        family = _to_ascii(family)
        if family:
            authors.append(family)
    return authors[:15]  # hallucinator caps at 15


def _extract_arxiv_id(entry: dict) -> str | None:
    """Get arXiv id from eprint + archiveprefix or from url."""
    ap = (entry.get("archiveprefix") or "").strip().lower()
    eprint = (entry.get("eprint") or "").strip()
    if ap == "arxiv" and eprint:
        return eprint
    url = entry.get("url") or ""
    m = re.search(r"arxiv\.org/abs/(\d+\.\d+v?\d*)", url, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def bib_entries_to_references(bib_path: Path) -> list[tuple[str, Reference]]:
    """Load a .bib file and return [(citation_key, Reference), ...]."""
    with open(bib_path, encoding="utf-8", errors="replace") as f:
        bib_str = f.read()
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    db = bibtexparser.loads(bib_str, parser)

    out = []
    for entry in db.entries:
        key = entry.get("ID", "")
        title = _normalize_title(entry.get("title", ""))
        if not title:
            continue
        if len(title.split()) < MIN_TITLE_WORDS:
            continue
        title = _to_ascii(title)  # avoid Rust Unicode panic
        authors = _parse_authors(entry.get("author", ""))
        doi = (entry.get("doi") or "").strip() or None
        arxiv_id = _extract_arxiv_id(entry)
        ref = Reference(
            title=title,
            authors=authors,
            doi=doi,
            arxiv_id=arxiv_id,
        )
        out.append((key, ref))
    return out


def dedupe_refs(items: list[tuple[str, Reference]]) -> list[tuple[str, Reference]]:
    """Deduplicate by (title, doi, arxiv_id), keeping first citation key."""
    seen: set[tuple[str, str | None, str | None]] = set()
    out = []
    for key, ref in items:
        sig = (ref.title, ref.doi, ref.arxiv_id)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((key, ref))
    return out


_CHILD_SCRIPT = r"""
import json, os, sys
from hallucinator import Reference, Validator, ValidatorConfig
ref_dict = json.loads(sys.argv[1])
ref = Reference(
    ref_dict["title"],
    authors=ref_dict.get("authors") or [],
    doi=ref_dict.get("doi"),
    arxiv_id=ref_dict.get("arxiv_id"),
)
config = ValidatorConfig()
if os.environ.get("OPENALEX_KEY"):
    config.openalex_key = os.environ["OPENALEX_KEY"]
if os.environ.get("S2_API_KEY"):
    config.s2_api_key = os.environ["S2_API_KEY"]
validator = Validator(config)
results = validator.check([ref])
r = results[0]
print(r.status, r.source or "", r.title, sep="\t")
"""


def _validate_resilient(refs: list) -> list:
    """Validate each ref in a subprocess; on crash, record as error and continue."""
    results = []
    for i, ref in enumerate(refs):
        payload = {
            "title": ref.title,
            "authors": ref.authors,
            "doi": ref.doi,
            "arxiv_id": ref.arxiv_id,
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _CHILD_SCRIPT, json.dumps(payload)],
                capture_output=True,
                text=True,
                timeout=90,
                env=os.environ,
            )
        except subprocess.TimeoutExpired:
            results.append(SimpleNamespace(status="error", title=ref.title, source="timeout"))
            icon = "!"
            print(f"  [{i+1}/{len(refs)}] {icon} error (timeout): {ref.title[:60]}...")
            continue
        if proc.returncode != 0 or not proc.stdout.strip():
            results.append(SimpleNamespace(status="error", title=ref.title, source="validator crash"))
            icon = "!"
            print(f"  [{i+1}/{len(refs)}] {icon} error (validator crash): {ref.title[:60]}...")
            continue
        parts = proc.stdout.strip().split("\t", 2)
        status = parts[0] if parts else "error"
        source = (parts[1] or None) if len(parts) > 1 else None
        title_out = parts[2] if len(parts) > 2 else ref.title
        results.append(SimpleNamespace(status=status, title=title_out, source=source))
        icon = {"verified": "+", "not_found": "?", "author_mismatch": "~"}.get(status, " ")
        src = f" ({source})" if source else ""
        print(f"  [{i+1}/{len(refs)}] {icon} {status}: {title_out[:60]}{src}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate .bib references with hallucinator (academic DBs)."
    )
    parser.add_argument(
        "bib_files",
        nargs="*",
        help="Paths to .bib files (default: vol1 and vol2 references.bib)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Write report to FILE",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not deduplicate references across .bib files",
    )
    parser.add_argument(
        "--root",
        default=".",
        metavar="DIR",
        help="Project root (default: current directory)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Validate only first N references (for quick test)",
    )
    parser.add_argument(
        "--no-resilient",
        action="store_true",
        help="Use batch validation (faster but may crash on some DB responses with Unicode)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.bib_files:
        bib_paths = [root / p for p in args.bib_files]
    else:
        bib_paths = [root / p for p in DEFAULT_BIB_PATHS]

    missing = [p for p in bib_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Not found: {p}", file=sys.stderr)
        return 1

    # Collect (key, Reference) from all files
    all_refs: list[tuple[str, Reference]] = []
    for p in bib_paths:
        all_refs.extend(bib_entries_to_references(p))

    if not all_refs:
        print("No references to validate.", file=sys.stderr)
        return 0

    if not args.no_dedupe:
        all_refs = dedupe_refs(all_refs)

    if args.limit is not None:
        all_refs = all_refs[: args.limit]

    refs = [r for _, r in all_refs]
    keys = [k for k, _ in all_refs]
    n = len(refs)
    print(f"Validating {n} references against academic databases...")
    if os.environ.get("OPENALEX_KEY") or os.environ.get("S2_API_KEY"):
        print("(Using OPENALEX_KEY / S2_API_KEY for better coverage)\n")
    else:
        print("(Optional: OPENALEX_KEY, S2_API_KEY for better coverage)\n")

    if not args.no_resilient:
        results = _validate_resilient(refs)
    else:
        config = ValidatorConfig()
        if os.environ.get("OPENALEX_KEY"):
            config.openalex_key = os.environ["OPENALEX_KEY"]
        if os.environ.get("S2_API_KEY"):
            config.s2_api_key = os.environ["S2_API_KEY"]
        validator = Validator(config)

        def progress(event):
            if event.event_type == "result":
                r = event.result
                idx = event.index + 1
                icon = {"verified": "+", "not_found": "?", "author_mismatch": "~"}.get(
                    r.status, " "
                )
                src = f" ({r.source})" if r.source else ""
                print(f"  [{idx}/{event.total}] {icon} {r.status}: {r.title}{src}")

        results = validator.check(refs, progress=progress)

    # Summary
    verified = sum(1 for r in results if r.status == "verified")
    not_found = sum(1 for r in results if r.status == "not_found")
    mismatch = sum(1 for r in results if r.status == "author_mismatch")
    errors = sum(1 for r in results if r.status == "error")

    lines = [
        "",
        "Summary",
        "-------",
        f"  Verified:        {verified}",
        f"  Not found:       {not_found}",
        f"  Author mismatch: {mismatch}",
        f"  Total:           {n}",
    ]
    if errors:
        lines.insert(-1, f"  Error (skipped): {errors}")

    for line in lines:
        print(line)

    # Optional report file
    if args.output:
        report_path = Path(args.output)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Hallucinator reference check report\n")
            f.write("====================================\n\n")
            f.write(f"Sources: {[str(p) for p in bib_paths]}\n")
            f.write("\n".join(lines) + "\n\n")
            f.write("Not found (potential typos or non-indexed):\n")
            for key, r in zip(keys, results):
                if r.status == "not_found":
                    f.write(f"  [{key}] {r.title}\n")
            f.write("\nAuthor mismatch:\n")
            for key, r in zip(keys, results):
                if r.status == "author_mismatch":
                    f.write(f"  [{key}] {r.title}\n")
            errors_list = [(k, r) for k, r in zip(keys, results) if r.status == "error"]
            if errors_list:
                f.write("\nError (validator crash or timeout):\n")
                for key, r in errors_list:
                    f.write(f"  [{key}] {r.title}\n")
        print(f"\nReport written to {report_path}")

    return 0 if not_found == 0 and mismatch == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
