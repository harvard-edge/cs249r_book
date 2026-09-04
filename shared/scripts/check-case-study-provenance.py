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
  no-citation        a case study cites nothing
  unresolved-key     it cites a key absent from the volume bibliography
  unverifiable-body  its prose names an archive, registry or docket as its
                     authority without citing anything

Warnings (reported, do not block)
  no-provenance      no Provenance line

Usage:
  python3 shared/scripts/check-case-study-provenance.py [paths...]
  python3 shared/scripts/check-case-study-provenance.py --strict   # warnings fail too
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CALLOUT = "callout-case-study"
REPO = Path(__file__).resolve().parents[2]

# Volumes and the bibliography each is resolved against.
BIBS = [
    (REPO / "books/vol4", REPO / "books/vol4/references.bib"),
    (REPO / "publishing/quarto/contents/vol4",
     REPO / "publishing/quarto/contents/references-vol4.bib"),
]

# Prose that claims an authority. Harmless with a citation; a fabrication risk
# without one, which is exactly how the three invented archives entered.
AUTHORITY = re.compile(
    r"\b(investigation archive|incident archive|statutory\s+\w+\s+archive|"
    r"investigation archives|incident register|accident register|"
    r"failure analysis report|internal incident log|proprietary incident)",
    re.I,
)

NOT_A_CITEKEY = re.compile(r"^(fig|sec|tbl|eq|lst|thm|def|pri|nbk|exm|cs|ws|chk|psp|lhs)-", re.I)


def bib_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(re.findall(r"^@\w+\{([^,]+),", path.read_text(errors="replace"), re.M))


def blocks(lines: list[str]):
    """Yield (start_line, body) for each case-study callout."""
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
    return [
        c for c in re.findall(r"@([A-Za-z][A-Za-z0-9_:+.-]*[A-Za-z0-9])", text)
        if not NOT_A_CITEKEY.match(c)
    ]


def title_of(body: str) -> str:
    m = re.search(r'title="([^"]*)"', body)
    return m.group(1) if m else "(untitled)"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0

    for root, bib in BIBS:
        if not root.exists():
            continue
        keys = bib_keys(bib)
        if not keys:
            errors.append(f"{bib}: bibliography missing or empty")
            continue

        files = [Path(a) for a in args if root in Path(a).resolve().parents] or sorted(root.rglob("*.qmd"))
        for f in files:
            if not f.exists():
                continue
            for line_no, body in blocks(f.read_text(errors="replace").split("\n")):
                checked += 1
                where = f"{f.relative_to(REPO)}:{line_no}"
                title = title_of(body)
                used = cites(body)

                if not used:
                    errors.append(
                        f"{where} [no-citation] case study {title!r} names no source.\n"
                        f"    Every case study must cite a primary source: an investigation "
                        f"report, a regulatory filing, a peer-reviewed paper, or a "
                        f"first-party postmortem."
                    )
                else:
                    for c in sorted(set(used)):
                        if c not in keys:
                            errors.append(
                                f"{where} [unresolved-key] case study {title!r} cites "
                                f"@{c}, which is not in {bib.name}."
                            )

                hit = AUTHORITY.search(body)
                if hit and not used:
                    errors.append(
                        f"{where} [unverifiable-body] case study {title!r} names "
                        f"{hit.group(0)!r} as its authority but cites nothing.\n"
                        f"    Name the document, not the genre of document."
                    )

                if not re.search(r"\*\*Provenance:?\*\*|^Provenance:", body, re.M):
                    warnings.append(f"{where} [no-provenance] {title!r} has no Provenance line.")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")

    fail = bool(errors) or (strict and bool(warnings))
    print(
        f"\ncase-study provenance: {checked} checked, "
        f"{len(errors)} error(s), {len(warnings)} warning(s)"
    )
    if fail:
        print("\nA case study without a source that resolves is the one defect this book "
              "cannot ship, because it is the standard the book asks of its readers.")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
