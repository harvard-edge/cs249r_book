#!/usr/bin/env python3
"""Every case study must name a source that resolves.

A 2026-09 audit of all 22 case studies in Volume IV against their primary
sources found one that survived as written. Three cited authorities that do not
exist, five described events no primary source records, and ten stated causes
their cited reports contradict.

The manuscript already carried the instruction. The comment

    <!-- INCIDENT: needs a documented, citable case. Do not invent one. -->

appears once per case-study slot, and four boxes were invented directly beneath
it. An instruction in a comment cannot hold. This check is the same requirement
expressed as something the build enforces.

Errors (block the commit)
  missing-bibliography  the volume bibliography is missing or empty
  no-citation           a case study cites nothing
  unresolved-key        it cites a key absent from the volume bibliography
  unverifiable-body     its prose names an archive, registry or docket as its
                        authority without citing anything

Warnings (reported, do not block)
  no-provenance         no Provenance line

Usage:
  ./binder/binder check sources --scope case-studies
  python3 binder/cli/checks/case_study_provenance.py [paths...] [--strict]
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

CALLOUT = "callout-case-study"
REPO = Path(__file__).resolve().parents[3]

# Prose that claims an authority. Harmless with a citation; a fabrication risk
# without one, which is exactly how the three invented archives entered.
AUTHORITY = re.compile(
    r"\b(investigation archive|incident archive|statutory\s+\w+\s+archive|"
    r"investigation archives|incident register|accident register|"
    r"failure analysis report|internal incident log|proprietary incident)",
    re.I,
)

NOT_A_CITEKEY = re.compile(r"^(fig|sec|tbl|eq|lst|thm|def|pri|nbk|exm|cs|ws|chk|psp|lhs)-", re.I)


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    code: str
    message: str
    severity: str  # "error" blocks; "warning" is reported only


def volume_bibliographies(repo: Path) -> list[tuple[Path, Path]]:
    """Map each volume carrying case studies to its designated bibliography file.

    Volumes with dedicated, isolated bibliographies (such as 'vol3' and 'vol4')
    are always mapped to their specific ``references-{vol}.bib`` file. This
    guarantees that a missing dedicated bibliography triggers a missing-bibliography
    error rather than silently falling back to the shared bibliography. Other
    volumes map to their dedicated bibliography if it exists, falling back to
    the project-wide ``references.bib``.

    Args:
        repo: Path to the repository root directory.

    Returns:
        List of (volume_directory, bibliography_file) tuples.
    """
    books_dir = repo / "books" if (repo / "books").is_dir() else repo
    dedicated_volumes = {"vol3", "vol4"}
    pairs = []
    for vol_dir in sorted(books_dir.glob("vol*")):
        if not vol_dir.is_dir():
            continue
        vol_name = vol_dir.name
        if vol_name in dedicated_volumes:
            bib = books_dir / f"references-{vol_name}.bib"
        else:
            cand = books_dir / f"references-{vol_name}.bib"
            bib = cand if cand.exists() else (books_dir / "references.bib")
        pairs.append((vol_dir, bib))
    return pairs or [(books_dir / "vol4", books_dir / "references-vol4.bib")]


def bib_keys(path: Path) -> set[str]:
    """Extract all BibTeX citation keys defined in a bibliography file.

    Args:
        path: Path to the .bib file.

    Returns:
        Set of citation key strings found in the file, or empty set if missing.
    """
    if not path.exists():
        return set()
    return set(re.findall(r"^@\w+\{([^,]+),", path.read_text(errors="replace"), re.M))


def blocks(lines: list[str]):
    """Yield (start_line, body) for each case-study callout.

    Args:
        lines: Lines of the QMD file content.

    Yields:
        Tuple of (start_line_number, callout_body_string).
    """
    for i, line in enumerate(lines):
        if CALLOUT not in line:
            continue
        depth, body = 0, []
        for j in range(i, len(lines)):
            s = lines[j].strip()
            if s.startswith(":::"):
                if re.match(r"^:::+\s*\{", s):
                    depth += 1
                elif re.match(r"^:::+$", s):
                    depth -= 1
                    if depth == 0:
                        body.append(lines[j])
                        break
            body.append(lines[j])
        yield i + 1, "\n".join(body)


def cites(text: str) -> list[str]:
    """Extract citation keys from callout text, excluding non-citation labels.

    Args:
        text: Raw callout text.

    Returns:
        List of citation keys referenced in the text.
    """
    return [
        c for c in re.findall(r"@([A-Za-z][A-Za-z0-9_:+.-]*[A-Za-z0-9])", text)
        if not NOT_A_CITEKEY.match(c)
    ]


def title_of(body: str) -> str:
    """Extract the title attribute from a callout definition line.

    Args:
        body: Callout body text.

    Returns:
        Title string, or '(untitled)' if no title attribute was found.
    """
    m = re.search(r'title="([^"]*)"', body)
    return m.group(1) if m else "(untitled)"


def _rel(path: Path, repo: Path) -> str:
    """Return path relative to repository root for display.

    Args:
        path: Target path.
        repo: Repository root path.

    Returns:
        Relative path string if inside repo, otherwise absolute path string.
    """
    try:
        return str(path.resolve().relative_to(repo))
    except ValueError:
        return str(path)


def collect(paths: list[Path] | None = None, repo: Path = REPO) -> tuple[int, list[Finding]]:
    """Scan case studies; return how many were checked and what was found.

    With ``paths``, only those files inside a case-study volume are scanned;
    otherwise every ``.qmd`` in each volume is.

    Args:
        paths: Optional subset of files to scan.
        repo: Repository root directory.

    Returns:
        Tuple of (checked_callouts_count, list_of_findings).
    """
    findings: list[Finding] = []
    checked = 0

    for root, bib in volume_bibliographies(repo):
        if not root.exists():
            continue
        keys = bib_keys(bib)
        if not keys:
            findings.append(Finding(_rel(bib, repo), 0, "missing-bibliography",
                                    "bibliography missing or empty", "error"))
            continue

        selected = [p for p in (paths or []) if root in p.resolve().parents]
        for f in selected or sorted(root.rglob("*.qmd")):
            if not f.exists():
                continue
            rel = _rel(f, repo)
            for line_no, body in blocks(f.read_text(errors="replace").split("\n")):
                checked += 1
                title = title_of(body)
                used = cites(body)

                if not used:
                    findings.append(Finding(
                        rel, line_no, "no-citation",
                        f"case study {title!r} names no source. Every case study must cite "
                        "a primary source: an investigation report, a regulatory filing, a "
                        "peer-reviewed paper, or a first-party postmortem.",
                        "error",
                    ))
                else:
                    for c in sorted(set(used)):
                        if c not in keys:
                            findings.append(Finding(
                                rel, line_no, "unresolved-key",
                                f"case study {title!r} cites @{c}, which is not in {bib.name}.",
                                "error",
                            ))

                hit = AUTHORITY.search(body)
                if hit and not used:
                    findings.append(Finding(
                        rel, line_no, "unverifiable-body",
                        f"case study {title!r} names {hit.group(0)!r} as its authority but "
                        "cites nothing. Name the document, not the genre of document.",
                        "error",
                    ))

                if not re.search(r"\*\*Provenance:?\*\*|^Provenance:", body, re.M):
                    findings.append(Finding(
                        rel, line_no, "no-provenance",
                        f"{title!r} has no Provenance line.", "warning",
                    ))

    return checked, findings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for case study provenance validation.

    Args:
        argv: Command line arguments.

    Returns:
        0 on success, 1 on failure.
    """
    argv = sys.argv[1:] if argv is None else argv
    strict = "--strict" in argv
    checked, findings = collect([Path(a) for a in argv if not a.startswith("--")])
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]

    for w in warnings:
        print(f"WARN  {w.file}:{w.line} [{w.code}] {w.message}")
    for e in errors:
        print(f"ERROR {e.file}:{e.line} [{e.code}] {e.message}")

    fail = bool(errors) or (strict and bool(warnings))
    print(f"\ncase-study provenance: {checked} checked, "
          f"{len(errors)} error(s), {len(warnings)} warning(s)")
    if fail:
        print("\nA case study without a source that resolves is the one defect this book "
              "cannot ship, because it is the standard the book asks of its readers.")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
