#!/usr/bin/env python3
"""Phase A2 — AU query response coverage.

Reads ``11_au_queries/data/au_queries.json`` (200 highlights), filters down
to entries whose ``content`` actually starts with ``AU:`` / ``Au:`` /
``Comp`` (the true AU/Comp queries; the rest are mis-classified
em-dash/spacing notes), then assigns each query to one of the eight
response categories (A-H) defined in
``book/tools/scripts/mit_press/AU_QUERY_RESPONSES.md``. The verification
contract is: every category that has true queries must have a section
present in the response document.

Output: ``ledgers/vol1-au-queries-coverage.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

INPUT = Path.home() / "Desktop/MIT_Press_Feedback/11_au_queries/data/au_queries.json"
RESPONSE_DOC = Path("/Users/VJ/GitHub/MLSysBook-release-audit/book/tools/scripts/mit_press/AU_QUERY_RESPONSES.md")
OUT_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/ledgers"

CATEGORY_PATTERNS: list[tuple[str, str, str]] = [
    # (category_id, label, regex applied case-insensitively to content)
    ("A-emdash", "Category A: em dashes", r"em[- ]?dash|close up space around the em"),
    ("A-slash", "Category A: slash spacing", r"close up.*slash|\bslash\b"),
    ("A-numunit", "Category A: number-unit spacing", r"add (a )?space between the number"),
    ("A-bib-publisher", "Category A: bibliography publisher", r"please provide (the )?publisher"),
    ("A-bib-page", "Category A: bibliography page number", r"please provide (a |the )?page number"),
    ("A-bib-location", "Category A: bibliography location", r"deleted the publisher location"),
    ("A-bib-info", "Category A: bibliography missing info", r"please provide complete information|please provide.*(year|volume|issue|doi|location|venue)"),
    ("B-layout", "Category B: layout / typesetter", r"move this|next page|widow|two columns|close to the bottom|left[- ]aligned|comp\b"),
    ("C-footnote", "Category C: footnotes", r"footnote|sidenote|footnoe|missing fn"),
    ("D-section", "Category D: section / figure numbers", r"section number|add the correct section|insert (figure|table|section|chapter) (number|name)|add correct figure number|cross[- ]?ref|\?\?"),
    ("E-okdelete", "Category E: ok to delete", r"ok to delete"),
    ("E-toc", "Category E: TOC descriptions", r"section description|part title|don'?t match|toc"),
    ("E-duplicate", "Category E: duplicate definition", r"already defined|already introduced"),
    ("E-appendix", "Category E: appendix layout", r"laid out like|fallacies and pitfalls"),
    ("E-monospace", "Category E: monospace", r"monospace"),
    ("E-minus", "Category E: minus sign", r"minus sign"),
    ("E-vonneumann", "Category E: Von Neumann", r"von neumann"),
    ("E-words", "Category E: are these words", r"supposed to be words"),
    ("F-abbrev", "Category F: abbreviation expansion", r"spell out|first use|abbrev"),
    ("G-cap", "Category G: capitalization", r"lowercase|capital|sentence (style|case)|title case"),
    ("H-percent", "Category H: percent", r"percent|spell out.*%"),
]

CATEGORIES_PRESENT = {
    # The eight category headers actually present in AU_QUERY_RESPONSES.md.
    "A": "Category A: Resolved by Automated Passes",
    "B": "Category B: Layout Notes (Typesetter Handles)",
    "C": "Category C: Footnotes (38 queries)",
    "D": "Category D: Section Numbers (22 queries)",
    "E": "Category E: Decisions Required",
    "F": "Category F: Abbreviation Expansions",
    "G": "Category G: Capitalization",
    "H": "Category H: Percent",
}

AU_PREFIX_RE = re.compile(r"^(au|comp)\b", re.IGNORECASE)


def categorize(content: str) -> tuple[str, str]:
    cl = content.lower()
    for cat_id, label, pat in CATEGORY_PATTERNS:
        if re.search(pat, cl):
            return (cat_id, label)
    return ("UNCATEGORIZED", "uncategorized — manual review")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = json.loads(INPUT.read_text())

    # Identify which categories are actually present in the response doc.
    response_text = RESPONSE_DOC.read_text()
    present_categories = {}
    for letter, header in CATEGORIES_PRESENT.items():
        present_categories[letter] = header in response_text

    results = []
    by_category: dict[str, int] = {}
    addressed_count = 0

    for idx, q in enumerate(queries):
        content = (q.get("content") or "").strip()
        is_au = bool(AU_PREFIX_RE.match(content))
        cat_id, cat_label = categorize(content)
        cat_letter = cat_id.split("-")[0]
        category_section_present = present_categories.get(cat_letter, False)

        addressed = is_au and category_section_present and cat_id != "UNCATEGORIZED"
        if addressed:
            addressed_count += 1

        results.append({
            "query_id": f"au-{idx:04d}",
            "true_au_query": is_au,
            "page": q.get("page"),
            "pdf": q.get("pdf"),
            "content": content,
            "highlighted_text": q.get("highlighted_text"),
            "category": cat_id,
            "category_label": cat_label,
            "category_letter": cat_letter,
            "response_doc_section_present": category_section_present,
            "addressed": addressed,
            "evidence": (
                f"Category section '{CATEGORIES_PRESENT.get(cat_letter, '?')}' present in "
                f"AU_QUERY_RESPONSES.md"
            ) if addressed else (
                "Not a true AU/Comp query (excluded)" if not is_au else
                f"category={cat_id} → no response section"
            ),
        })
        by_category[cat_id] = by_category.get(cat_id, 0) + 1

    true_au = [r for r in results if r["true_au_query"]]
    addressed_true_au = [r for r in true_au if r["addressed"]]

    summary = {
        "source": str(INPUT),
        "response_doc": str(RESPONSE_DOC),
        "total_queries_in_file": len(queries),
        "true_au_or_comp_queries": len(true_au),
        "addressed_among_true_au": len(addressed_true_au),
        "unaddressed_among_true_au": len(true_au) - len(addressed_true_au),
        "category_response_doc_coverage": {
            letter: present_categories[letter] for letter in CATEGORIES_PRESENT
        },
        "queries_by_category": dict(sorted(by_category.items())),
    }

    out_path = OUT_DIR / "vol1-au-queries-coverage.json"
    out_path.write_text(json.dumps({"summary": summary, "queries": results}, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
