"""
Native reference check: validate .bib entries against academic DBs (hallucinator).

Used by `binder validate references`. Requires: pip install hallucinator bibtexparser
(or install optional extra: pip install -e ".[reference-check]").

Optional env: OPENALEX_KEY, S2_API_KEY. Note: Semantic Scholar allows 1 request/sec;
full runs with S2_API_KEY set will be slow; use --limit for quick tests.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Optional deps; fail at run time with clear message if missing
try:
    import bibtexparser
except ImportError:
    bibtexparser = None  # type: ignore[assignment]

try:
    from hallucinator import Reference, Validator, ValidatorConfig
except ImportError:
    Reference = Validator = ValidatorConfig = None  # type: ignore[assignment,misc]

MIN_TITLE_WORDS = 4

# Default .bib paths relative to repo root (when book_dir.parent.parent is repo root)
DEFAULT_BIB_REL_PATHS = [
    "book/quarto/contents/vol1/backmatter/references.bib",
    "book/quarto/contents/vol2/backmatter/references.bib",
]

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


def _to_ascii(s: str) -> str:
    if not s:
        return s
    n = unicodedata.normalize("NFKD", s)
    return n.encode("ascii", "ignore").decode("ascii")


def _normalize_title(raw: str) -> str:
    if not raw:
        return ""
    t = re.sub(r"[\{\}]", "", raw)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _parse_authors(author_field: str) -> List[str]:
    if not author_field or not author_field.strip():
        return []
    authors = []
    for part in re.split(r"\s+and\s+", author_field, flags=re.IGNORECASE):
        part = part.strip()
        if not part:
            continue
        if "," in part:
            family = part.split(",", 1)[0].strip()
        else:
            family = part
        family = re.sub(r"\\[a-z]+\{([^}]*)\}", r"\1", family)
        family = re.sub(r"[{}\\]", "", family).strip()
        family = _to_ascii(family)
        if family:
            authors.append(family)
    return authors[:15]


def _extract_arxiv_id(entry: dict) -> Optional[str]:
    ap = (entry.get("archiveprefix") or "").strip().lower()
    eprint = (entry.get("eprint") or "").strip()
    if ap == "arxiv" and eprint:
        return eprint
    url = entry.get("url") or ""
    m = re.search(r"arxiv\.org/abs/(\d+\.\d+v?\d*)", url, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _bib_entries_to_references(bib_path: Path) -> List[Tuple[str, Any]]:
    with open(bib_path, encoding="utf-8", errors="replace") as f:
        bib_str = f.read()
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    db = bibtexparser.loads(bib_str, parser)
    out = []
    for entry in db.entries:
        key = entry.get("ID", "")
        title = _normalize_title(entry.get("title", ""))
        if not title or len(title.split()) < MIN_TITLE_WORDS:
            continue
        title = _to_ascii(title)
        authors = _parse_authors(entry.get("author", ""))
        doi = (entry.get("doi") or "").strip() or None
        arxiv_id = _extract_arxiv_id(entry)
        ref = Reference(title=title, authors=authors, doi=doi, arxiv_id=arxiv_id)
        out.append((key, ref))
    return out


def _dedupe_refs(items: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    seen: set = set()
    out = []
    for key, ref in items:
        sig = (ref.title, ref.doi, ref.arxiv_id)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((key, ref))
    return out


def _validate_resilient(keys: List[str], refs: List[Any], console: Optional[Any]) -> List[Any]:
    results = []
    n = len(refs)
    key_w = min(36, max(12, max(len(k) for k in keys) if keys else 12))
    for i, (key, ref) in enumerate(zip(keys, refs)):
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
            if console:
                console.print(f"  [{i+1:>{len(str(n))}}/{n}]  {key[:key_w]:<{key_w}}  ! error (timeout)")
            continue
        if proc.returncode != 0 or not proc.stdout.strip():
            results.append(SimpleNamespace(status="error", title=ref.title, source="validator crash"))
            if console:
                console.print(f"  [{i+1:>{len(str(n))}}/{n}]  {key[:key_w]:<{key_w}}  ! error (crash)")
            continue
        parts = proc.stdout.strip().split("\t", 2)
        status = parts[0] if parts else "error"
        source = (parts[1] or None) if len(parts) > 1 else None
        title_out = parts[2] if len(parts) > 2 else ref.title
        results.append(SimpleNamespace(status=status, title=title_out, source=source))
        if console:
            icon = {"verified": "\u2713", "not_found": "?", "author_mismatch": "~"}.get(status, "!")
            src = f" ({source})" if source else ""
            if status == "verified":
                console.print(f"  [{i+1:>{len(str(n))}}/{n}]  {key[:key_w]:<{key_w}}  {icon} verified{src}")
            else:
                console.print(f"  [{i+1:>{len(str(n))}}/{n}]  {key[:key_w]:<{key_w}}  {icon} {status}{src}")
    return results


def _load_cache(cache_path: Path) -> Dict[str, dict]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_path: Path, updates: Dict[str, dict]) -> None:
    existing = _load_cache(cache_path)
    existing.update(updates)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def run(
    bib_paths: List[Path],
    *,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    dedupe: bool = True,
    resilient: bool = True,
    console: Optional[Any] = None,
    cache_path: Optional[Path] = None,
    skip_verified: bool = False,
    thorough: bool = False,
) -> Tuple[bool, int, List[dict], int]:
    """
    Load .bib files, validate refs against academic DBs, optionally write report.

    cache_path: if set, read/write verification cache (key -> {status, source, date}).
    skip_verified: only validate refs not already verified in cache (ignored if thorough).
    thorough: revalidate all refs and ignore cache for filtering.

    Returns:
        (passed, elapsed_ms, issues, ref_count)
        issues: list of dicts with file, line, code, message, severity for ValidationIssue.
    """
    if bibtexparser is None or Reference is None or Validator is None:
        return (
            False,
            0,
            [
                {
                    "file": "(reference_check)",
                    "line": 0,
                    "code": "references",
                    "message": "Missing deps: pip install hallucinator bibtexparser (or pip install -e \".[reference-check]\")",
                    "severity": "error",
                }
            ],
            0,
        )

    t0 = time.time()
    all_refs: List[Tuple[str, Any]] = []
    for p in bib_paths:
        if not p.exists():
            return (
                False,
                int((time.time() - t0) * 1000),
                [{"file": str(p), "line": 0, "code": "references", "message": f"Not found: {p}", "severity": "error"}],
                0,
            )
        all_refs.extend(_bib_entries_to_references(p))

    if not all_refs:
        return True, int((time.time() - t0) * 1000), [], 0

    if dedupe:
        all_refs = _dedupe_refs(all_refs)
    if limit is not None:
        all_refs = all_refs[:limit]

    # Optionally skip refs already verified in cache
    if skip_verified and not thorough and cache_path:
        cache = _load_cache(cache_path)
        all_refs = [(k, r) for k, r in all_refs if cache.get(k, {}).get("status") != "verified"]

    refs = [r for _, r in all_refs]
    keys = [k for k, _ in all_refs]
    n = len(refs)

    if n == 0:
        # All refs were skipped as already verified
        return True, int((time.time() - t0) * 1000), [], 0

    if console:
        if only_keys is not None:
            console.print(f"Validating {n} references (only keys with issues from report/file)...")
        else:
            console.print(f"Validating {n} references against academic databases...")
        if skip_verified and not thorough and cache_path:
            console.print("(Skipping refs already marked verified in cache)\n")
        if os.environ.get("OPENALEX_KEY") or os.environ.get("S2_API_KEY"):
            console.print("(Using OPENALEX_KEY / S2_API_KEY for better coverage)\n")
        else:
            console.print("(Optional: OPENALEX_KEY, S2_API_KEY for better coverage)\n")

    if resilient:
        results = _validate_resilient(keys, refs, console)
    else:
        config = ValidatorConfig()
        if os.environ.get("OPENALEX_KEY"):
            config.openalex_key = os.environ["OPENALEX_KEY"]
        if os.environ.get("S2_API_KEY"):
            config.s2_api_key = os.environ["S2_API_KEY"]
        validator = Validator(config)
        key_w = min(36, max(12, max(len(k) for k in keys) if keys else 12))

        def progress(event: Any) -> None:
            if event.event_type == "result" and console:
                r = event.result
                idx = event.index + 1
                key = keys[event.index] if event.index < len(keys) else ""
                icon = {"verified": "\u2713", "not_found": "?", "author_mismatch": "~"}.get(r.status, "!")
                src = f" ({r.source})" if r.source else ""
                if r.status == "verified":
                    console.print(f"  [{idx:>{len(str(n))}}/{n}]  {key[:key_w]:<{key_w}}  {icon} verified{src}")
                else:
                    console.print(f"  [{idx:>{len(str(n))}}/{n}]  {key[:key_w]:<{key_w}}  {icon} {r.status}{src}")

        results = validator.check(refs, progress=progress)

    elapsed_ms = int((time.time() - t0) * 1000)
    verified = sum(1 for r in results if r.status == "verified")
    not_found = sum(1 for r in results if r.status == "not_found")
    mismatch = sum(1 for r in results if r.status == "author_mismatch")
    errors = sum(1 for r in results if r.status == "error")
    passed = (not_found == 0 and mismatch == 0 and errors == 0)

    issues: List[dict] = []
    for key, r in zip(keys, results):
        if r.status == "not_found":
            issues.append({"file": key, "line": 0, "code": "references", "message": r.title, "severity": "error"})
        elif r.status == "author_mismatch":
            issues.append({"file": key, "line": 0, "code": "references", "message": f"author_mismatch: {r.title}", "severity": "error"})
        elif r.status == "error":
            issues.append({"file": key, "line": 0, "code": "references", "message": f"error: {r.title}", "severity": "error"})

    if console:
        console.print("")
        console.print("Summary")
        console.print("-------")
        console.print(f"  Verified:        {verified}")
        console.print(f"  Not found:       {not_found}")
        console.print(f"  Author mismatch: {mismatch}")
        if errors:
            console.print(f"  Error (skipped): {errors}")
        console.print(f"  Total:           {n}")
        not_verified = [(k, r) for k, r in zip(keys, results) if r.status in ("not_found", "author_mismatch", "error")]
        if not_verified:
            console.print("")
            console.print("Not verified (review these)")
            console.print("---------------------------")
            for key, r in not_verified:
                title = (r.title or "")[:72] + ("..." if len((r.title or "")) > 72 else "")
                console.print(f"  [{key}]  {r.status}: {title}")

    if output_path is not None:
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Hallucinator reference check report\n")
            f.write("====================================\n\n")
            f.write(f"Sources: {[str(p) for p in bib_paths]}\n\n")
            f.write(f"Verified: {verified}, Not found: {not_found}, Author mismatch: {mismatch}, Error: {errors}, Total: {n}\n\n")
            f.write("Not found (potential typos or non-indexed):\n")
            for key, r in zip(keys, results):
                if r.status == "not_found":
                    f.write(f"  [{key}] {r.title}\n")
            f.write("\nAuthor mismatch:\n")
            for key, r in zip(keys, results):
                if r.status == "author_mismatch":
                    f.write(f"  [{key}] {r.title}\n")
            err_list = [(k, r) for k, r in zip(keys, results) if r.status == "error"]
            if err_list:
                f.write("\nError (validator crash or timeout):\n")
                for key, r in err_list:
                    f.write(f"  [{key}] {r.title}\n")
        if console:
            console.print(f"\nReport written to {output_path}")

    if cache_path is not None:
        now = datetime.now(timezone.utc).isoformat()
        updates = {
            key: {"status": r.status, "source": r.source or "", "date": now}
            for key, r in zip(keys, results)
        }
        _save_cache(Path(cache_path), updates)

    return passed, elapsed_ms, issues, n
