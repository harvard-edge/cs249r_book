#!/usr/bin/env python3
"""
Index audit — produces 4 reports per .claude/rules/index.md plan.

Run from a MLSysBook worktree root:
    python3 .claude/_reviews/index_audit_2026-05-02/audit.py

Outputs (CSV) in this same directory:
    coverage_gaps.csv      — Phase A: under-indexed concepts
    singleton_classifier.csv — Phase B: singleton triage
    subentry_summary.csv   — Phase C: main-entry subentry parallelism
    see_ref_check.csv      — Phase D: |see / |seealso target verification
"""
from __future__ import annotations
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()
CONTENT = ROOT / "book" / "quarto" / "contents"
OUTDIR = ROOT / ".claude" / "_reviews" / "index_audit_2026-05-02"
GLOSSARY = CONTENT / "vol1" / "backmatter" / "glossary" / "glossary.qmd"

INDEX_RE = re.compile(r"\\index\{([^{}]*(?:\{[^{}]*\}[^{}]*)?)\}")
BOLD_RE = re.compile(r"\*\*([^*]{2,80})\*\*")
HEADER_RE = re.compile(r"^#+\s+(.+?)\s*(?:\{[^}]*\})?\s*$")
SEEREF_RE = re.compile(r"\\index\{([^}|]+)\|see(?:also)?\{([^}]+)\}\}")


def get_qmd_files() -> list[Path]:
    return sorted(CONTENT.rglob("*.qmd"))


def extract_all_index_calls() -> tuple[list[str], dict[str, list[Path]]]:
    """Return (all_keys_in_order, key_to_files_map)."""
    all_keys = []
    key_files: dict[str, list[Path]] = defaultdict(list)
    for f in get_qmd_files():
        text = f.read_text(errors="replace")
        for m in INDEX_RE.finditer(text):
            key = m.group(1)
            all_keys.append(key)
            key_files[key].append(f)
    return all_keys, key_files


def parse_key(key: str) -> tuple[str, str | None]:
    """Split 'Headword!subentry' into (headword, subentry); strip @-sort-keys."""
    if "|" in key:  # cross-ref form
        return (key, None)
    parts = key.split("!", 1)
    head = parts[0]
    sub = parts[1] if len(parts) == 2 else None
    # Strip sort key syntax (Foo@Bar -> Bar)
    if "@" in head:
        head = head.split("@", 1)[1]
    return (head, sub)


def normalize(s: str) -> str:
    """Loose normalize for matching — lowercase, strip parens, collapse whitespace."""
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def extract_glossary_terms() -> list[str]:
    """Glossary terms are bold lines `**term**` followed by `: definition` line."""
    if not GLOSSARY.exists():
        return []
    terms = []
    for line in GLOSSARY.read_text().splitlines():
        m = re.match(r"^\*\*([^*]+)\*\*\s*$", line.strip())
        if m:
            terms.append(m.group(1))
    return terms


NARRATIVE_LABEL_PREFIXES = {
    "the ", "step ", "common ", "key ", "use case",
    "archetype ", "scenario ", "example ",
}
NARRATIVE_LABEL_EXACT = {
    "problem", "scenario", "conclusion", "metric", "note", "summary",
    "definition", "step", "example", "answer", "purpose", "goals",
    "the", "and", "or", "aspect", "component", "result", "constraint",
    "technique", "category", "strategy", "cost", "use case", "best for",
    "duration", "rationale", "model type", "parameters", "operation",
    "interconnect", "machine", "power", "time", "compute", "throughput",
    "before", "after", "fix", "bug", "test", "input", "output", "context",
    "failure", "consequence", "physics", "math", "wire", "logic", "totals",
    "total", "without", "with", "key insight", "the problem", "the failure",
    "the consequence", "the math", "the wire", "the logic", "the context",
    "fallacy", "pitfall",
}


def extract_bold_terms_in_body() -> Counter:
    """Bold spans (**Term**) in body prose — count occurrences across the corpus.

    Aggressively filter narrative labels (callout headers, step markers,
    table row labels) so the result reflects substantive concept candidates.
    """
    counts = Counter()
    for f in get_qmd_files():
        # Skip frontmatter/backmatter/parts/glossary/appendix per D2
        s = str(f)
        if any(x in s for x in (
            "frontmatter/", "backmatter/", "/parts/", "/glossary/", "/appendix",
        )):
            continue
        for line in f.read_text(errors="replace").splitlines():
            for m in BOLD_RE.finditer(line):
                term = m.group(1).strip()
                lc = term.lower()

                # Length / markup gates
                if len(term) < 3:
                    continue
                if "$" in term or "\\" in term:
                    continue

                # Narrative-label filters
                if lc in NARRATIVE_LABEL_EXACT:
                    continue
                if any(lc.startswith(p) for p in NARRATIVE_LABEL_PREFIXES):
                    continue

                # Bold spans that end in colon are almost always callout headers
                if term.endswith(":"):
                    continue

                # Parenthetical-only suffix (callout patterns like "X (Quantitative)")
                if re.fullmatch(r"[A-Z][a-zA-Z]+\s*\([^)]+\)", term):
                    continue

                # Skip pure numeric or short-acronym noise
                if re.fullmatch(r"[0-9\-\s.]+", term):
                    continue

                counts[term] += 1
    return counts


def extract_section_headers() -> list[str]:
    """Section headers (## / ### titles) — these are major topic anchors."""
    headers = []
    for f in get_qmd_files():
        s = str(f)
        if any(x in s for x in (
            "frontmatter/", "backmatter/", "/parts/", "/glossary/", "/appendix",
        )):
            continue
        for line in f.read_text(errors="replace").splitlines():
            m = HEADER_RE.match(line)
            if m:
                headers.append(m.group(1).strip())
    return headers


def coverage_gap_report(key_files, glossary_terms, bold_counts, headers, outpath):
    """Phase A — find under-indexed major concepts."""
    # Build a lookup of normalized headwords → count of \index{} uses
    head_uses = Counter()
    for key, files in key_files.items():
        head, sub = parse_key(key)
        head_uses[normalize(head)] += len(files)

    # Build candidate list
    candidates = []
    seen = set()

    for term in glossary_terms:
        n = normalize(term)
        if n in seen:
            continue
        seen.add(n)
        idx_uses = head_uses.get(n, 0)
        # Estimate substantive mentions: bold-count for the term
        bold_uses = bold_counts.get(term, 0)
        # Also try title-cased
        for variant in (term, term.title(), term.upper()):
            bold_uses = max(bold_uses, bold_counts.get(variant, 0))
        candidates.append({
            "candidate": term,
            "source": "glossary",
            "index_main_uses": idx_uses,
            "bold_uses_in_body": bold_uses,
            "gap": max(0, bold_uses - idx_uses),
        })

    # Add high-frequency bold terms not in glossary
    for term, count in bold_counts.most_common(500):
        n = normalize(term)
        if n in seen:
            continue
        seen.add(n)
        if count < 3:
            continue
        idx_uses = head_uses.get(n, 0)
        candidates.append({
            "candidate": term,
            "source": "bold_in_body",
            "index_main_uses": idx_uses,
            "bold_uses_in_body": count,
            "gap": max(0, count - idx_uses),
        })

    # Sort: glossary-source first (high signal), then by gap descending
    candidates.sort(key=lambda x: (
        0 if x["source"] == "glossary" else 1,
        -x["gap"],
        -x["bold_uses_in_body"],
    ))

    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate", "source", "index_main_uses",
                                          "bold_uses_in_body", "gap"])
        w.writeheader()
        w.writerows(candidates)

    print(f"  coverage_gaps.csv: {len(candidates)} rows, top gap = {candidates[0]['gap'] if candidates else 0}")


def singleton_classifier_report(key_files, glossary_terms, headers, outpath):
    """Phase B — classify singletons (full-key occurs once corpus-wide)."""
    glossary_lc = {normalize(t) for t in glossary_terms}
    headers_lc = {normalize(h) for h in headers}

    # Build per-headword stats: total uses, subentry count
    head_subs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    head_total_uses = Counter()
    for key, files in key_files.items():
        if "|see" in key:
            continue
        head, sub = parse_key(key)
        head_total_uses[head] += len(files)
        if sub:
            head_subs[head].append((sub, len(files)))

    # An "established main entry" has ≥2 subentries OR ≥3 total uses
    established_heads = {
        h for h, subs in head_subs.items() if len(subs) >= 2
    } | {h for h, n in head_total_uses.items() if n >= 3}

    # Find singletons (keys that appear exactly once across corpus)
    rows = []
    for key, files in key_files.items():
        if len(files) != 1:
            continue
        head, sub = parse_key(key)
        n_head = normalize(head)
        n_full = normalize(key.replace("!", " "))

        # Classification heuristics
        verdict = "REVIEW"
        reason = ""

        # Cross-ref entries are not singletons-as-data — skip
        if "|see" in key:
            continue

        # DELETE first (cheapest signals)
        if sub:
            sub_lc = sub.lower().split()
            if sub_lc and sub_lc[0] in {"is", "are", "was", "were", "has", "have",
                                         "should", "must", "can", "will", "the",
                                         "a", "an"}:
                verdict = "DELETE"
                reason = "sentence-fragment subentry"

        if verdict == "REVIEW" and not sub and (
            len(head) < 3 or re.fullmatch(r"[0-9$\\\-\s]+", head)
        ):
            verdict = "DELETE"
            reason = "trivial / numeric-only"

        # KEEP: head or full key matches glossary or section header
        if verdict == "REVIEW":
            if n_head in glossary_lc or n_full in glossary_lc:
                verdict = "KEEP"
                reason = "matches glossary term"
            elif n_head in headers_lc or n_full in headers_lc:
                verdict = "KEEP"
                reason = "matches section header"

        # KEEP: Author-name pattern "Lastname, Firstname[!sub]" (per D4)
        if verdict == "REVIEW" and re.match(r"^[A-Z][a-zA-Z]+,\s+[A-Z]", head):
            verdict = "KEEP"
            reason = "author name (D4)"

        # DEMOTE: tightened — head is `Foo Bar`, where Bar is established AND
        # `Foo Bar` itself is NOT established (only this one singleton occurrence).
        # This catches one-off compound entries like "Pipeline Stages" while
        # leaving real terms like "Adam Optimizer" alone (Adam Optimizer has
        # multiple subentries).
        if verdict == "REVIEW":
            words = head.split()
            if (len(words) == 2
                and head not in established_heads
                and head_total_uses[head] == 1
                and n_head not in glossary_lc
            ):
                last_word = words[-1]
                first_word = words[0]
                # Prefer demoting under the LAST word if it's the noun head
                if (last_word in established_heads
                    and last_word not in {"Layer", "Function", "System", "Method"}
                ):
                    proposed_sub = words[0].lower()
                    if sub:
                        proposed_sub = f"{proposed_sub}, {sub}"
                    verdict = "DEMOTE"
                    reason = f"-> {last_word}!{proposed_sub}"
                elif first_word in established_heads and len(first_word) > 3:
                    proposed_sub = words[1].lower()
                    if sub:
                        proposed_sub = f"{proposed_sub}, {sub}"
                    verdict = "DEMOTE"
                    reason = f"-> {first_word}!{proposed_sub}"

        rows.append({
            "key": key,
            "headword": head,
            "subentry": sub or "",
            "file": str(files[0].relative_to(ROOT)),
            "verdict": verdict,
            "reason": reason,
        })

    # Sort by verdict, then key
    rows.sort(key=lambda r: (r["verdict"], r["key"]))

    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "headword", "subentry", "file",
                                          "verdict", "reason"])
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["verdict"] for r in rows)
    print(f"  singleton_classifier.csv: {len(rows)} singletons — "
          f"KEEP={counts['KEEP']} DEMOTE={counts['DEMOTE']} "
          f"DELETE={counts['DELETE']} REVIEW={counts['REVIEW']}")


def subentry_summary_report(key_files, outpath):
    """Phase C — main entries with ≥3 subentries, with grammatical signal."""
    head_to_subs = defaultdict(list)
    for key, files in key_files.items():
        if "|see" in key:
            continue
        head, sub = parse_key(key)
        if sub:
            head_to_subs[head].append((sub, len(files)))

    rows = []
    for head, subs in sorted(head_to_subs.items()):
        if len(subs) < 3:
            continue
        # Check parallelism: how many start with same word-class (heuristic by suffix)
        starts = Counter()
        for s, _ in subs:
            first = s.split()[0] if s.split() else ""
            starts[first] += 1
        # Sentence-fragment count
        fragments = sum(1 for s, _ in subs if s.split() and s.split()[0].lower()
                        in {"is", "are", "was", "were", "has", "have", "should",
                            "must", "can", "will", "the", "a", "an"})
        rows.append({
            "headword": head,
            "subentry_count": len(subs),
            "total_uses": sum(c for _, c in subs),
            "fragment_subs": fragments,
            "subentries_sample": "; ".join(s for s, _ in subs[:8]) + (" ..." if len(subs) > 8 else ""),
        })

    rows.sort(key=lambda r: -r["subentry_count"])

    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["headword", "subentry_count", "total_uses",
                                          "fragment_subs", "subentries_sample"])
        w.writeheader()
        w.writerows(rows)

    print(f"  subentry_summary.csv: {len(rows)} headwords with ≥3 subentries")


def see_ref_check_report(key_files, outpath):
    """Phase D — verify every |see / |seealso target exists."""
    all_heads = set()
    for key in key_files:
        if "|see" in key:
            continue
        head, _ = parse_key(key)
        all_heads.add(head)

    rows = []
    for key in sorted(key_files):
        m = SEEREF_RE.match(r"\\index{" + re.escape(key) + "}") if "|see" in key else None
        if "|see" not in key:
            continue
        # Parse: SOURCE|see{TARGET} or SOURCE|seealso{TARGET}
        m = re.match(r"^([^|]+)\|see(?:also)?\{([^}]+)\}$", key)
        if not m:
            continue
        source, target = m.group(1), m.group(2)
        target_exists = target in all_heads
        rows.append({
            "source": source,
            "target": target,
            "kind": "seealso" if "seealso" in key else "see",
            "target_exists": "yes" if target_exists else "NO",
        })

    rows.sort(key=lambda r: (r["target_exists"], r["source"]))

    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "kind", "target_exists"])
        w.writeheader()
        w.writerows(rows)

    broken = sum(1 for r in rows if r["target_exists"] == "NO")
    print(f"  see_ref_check.csv: {len(rows)} cross-refs — broken={broken}")


def main():
    print(f"Index audit — output dir: {OUTDIR}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Scanning corpus...")
    all_keys, key_files = extract_all_index_calls()
    print(f"  {len(all_keys)} total \\index{{}} calls")
    print(f"  {len(key_files)} unique keys")

    glossary_terms = extract_glossary_terms()
    print(f"  {len(glossary_terms)} glossary terms")

    bold_counts = extract_bold_terms_in_body()
    print(f"  {len(bold_counts)} unique bold spans in body")

    headers = extract_section_headers()
    print(f"  {len(headers)} section headers")

    print()
    print("Generating reports...")
    coverage_gap_report(key_files, glossary_terms, bold_counts, headers,
                        OUTDIR / "coverage_gaps.csv")
    singleton_classifier_report(key_files, glossary_terms, headers,
                                OUTDIR / "singleton_classifier.csv")
    subentry_summary_report(key_files, OUTDIR / "subentry_summary.csv")
    see_ref_check_report(key_files, OUTDIR / "see_ref_check.csv")

    print()
    print(f"Done. CSVs in: {OUTDIR}")


if __name__ == "__main__":
    main()
