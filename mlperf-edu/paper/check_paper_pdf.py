#!/usr/bin/env python3
"""Fail the paper build on unresolved references, placeholders, or PDF defects."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_paper_pdf.py PDF LOG")
    pdf_path = Path(sys.argv[1])
    log_path = Path(sys.argv[2])
    paper_path = Path(__file__).resolve().parent / "paper.tex"
    bib_path = Path(__file__).resolve().parent / "refs.bib"

    reader = PdfReader(pdf_path)
    # Sanity range, not an editorial page budget. It catches a collapsed or
    # runaway build; the upper bound was raised from 14 to 18 in 2026-08 when
    # the workload suite moved to one paragraph per benchmark.
    if not 5 <= len(reader.pages) <= 18:
        raise SystemExit(f"unexpected page count: {len(reader.pages)}")
    page_text = [(page.extract_text() or "").strip() for page in reader.pages]
    for number, text in enumerate(page_text, start=1):
        if len(text) < 180:
            raise SystemExit(
                f"page {number} has too little extractable text ({len(text)} chars)"
            )
    full_text = "\n".join(page_text)
    search_text = re.sub(r"\s+", " ", full_text).replace("- ", "-")
    required = [
        "REVIEW DRAFT",
        "not an official MLCommons benchmark",
        "Committed Reference Evidence",
        "local reviewer handoff",
        "unauthenticated integrity",
        "cross-platform replication",
    ]
    for phrase in required:
        if phrase not in search_text:
            raise SystemExit(f"required review disclosure missing from PDF: {phrase!r}")
    forbidden = [r"\bTODO\b", r"\bTBD\b", r"\bXXX\b", r"\?\?", r"Citation needed"]
    for pattern in forbidden:
        if re.search(pattern, full_text, flags=re.IGNORECASE):
            raise SystemExit(f"placeholder found in PDF: {pattern}")
    stale_evidence_claims = [
        "development-tier five-seed calibrations",
        "clean-commit timing bundles",
        "five-seed bundles do not yet exist",
        "The next milestone is evidence closure",
    ]
    for phrase in stale_evidence_claims:
        if phrase.lower() in full_text.lower():
            raise SystemExit(f"stale evidence claim found in PDF: {phrase!r}")

    source = paper_path.read_text()
    bib = bib_path.read_text()
    cited: set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z*]*\{([^}]+)\}", source):
        cited.update(key.strip() for key in match.group(1).split(","))
    available = set(re.findall(r"@[a-zA-Z]+\{([^,]+),", bib))
    missing = sorted(cited - available)
    if missing:
        raise SystemExit(f"missing bibliography entries: {missing}")

    log = log_path.read_text(errors="replace")
    log_failures = [
        "There were undefined references",
        "Citation `",
        "undefined citations",
        "Overfull \\hbox",
        "Overfull \\vbox",
    ]
    for marker in log_failures:
        if marker in log:
            raise SystemExit(f"LaTeX log contains: {marker}")

    print(
        f"verified {pdf_path}: {len(reader.pages)} pages, "
        f"{sum(len(text) for text in page_text)} extracted characters, "
        f"{len(cited)} citation keys, no placeholders or layout overflows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
