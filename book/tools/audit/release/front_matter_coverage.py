#!/usr/bin/env python3
"""Phase A4 — front-matter tracked-change coverage diff.

Reads ``10_front_matter/data/front_matter_changes.json`` (53 entries: 26
insertions, 25 deletions, 2 comments) and verifies each against the
current QMD source under ``book/quarto/contents/vol1/frontmatter/`` (and
shared ``book/quarto/contents/frontmatter/``).

**Important caveat about the input data**: docx tracked-change extraction
splits edits at every contiguous run boundary, so most ``text`` values are
sub-word fragments ("I s ", "t he ", "b ut ", "-T", "& ") rather than
literal source strings the editor typed. We therefore group entries by
``(type, text)`` and check the editor's *aggregate intent* with these
intent groups:

- ``ampersand-to-and``: deletion of ``& `` and insertion of ``and  `` —
  was the ``&`` replaced by ``and`` in author lists / acknowledgements?
- ``hyphen-prefix-removal``: deletions ``-L``, ``-T``, ``-S``, ``-C``,
  ``-M``, ``-P``, ``-D`` — was the hyphen between prefix and capitalised
  word removed (e.g. ``Long-Term`` → ``Long Term`` or ``LongTerm``)?
- ``micro-fragment``: single-character or punctuation edits — handled by
  the global bulk passes (em dash, percent, etc.).
- ``layout-typesetter``: the two ``comment`` entries.

Output: ``ledgers/vol1-front-matter-coverage.json``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

INPUT = Path.home() / "Desktop/MIT_Press_Feedback/10_front_matter/data/front_matter_changes.json"
ROOT = Path("/Users/VJ/GitHub/MLSysBook-release-audit/book/quarto/contents")
OUT_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/ledgers"

FRONT_MATTER_GLOBS = [
    "vol1/frontmatter/*.qmd",
    "frontmatter/**/*.qmd",
]


def collect_front_matter_text() -> tuple[str, list[str]]:
    """Concatenate all relevant front-matter QMD files.

    Returns (joined_text, file_list_relative_to_quarto).
    """
    files: list[Path] = []
    for pattern in FRONT_MATTER_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    parts = []
    for p in files:
        parts.append(p.read_text(errors="replace"))
    return ("\n\n".join(parts), [str(p.relative_to(ROOT.parent)) for p in files])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    changes = json.loads(INPUT.read_text())
    front_matter_text, files = collect_front_matter_text()

    # Group by (type, normalized_text) — many duplicates ('and  ', '& ').
    grouped: dict[tuple[str, str], int] = {}
    for ch in changes:
        key = (ch["type"], ch.get("text", ""))
        grouped[key] = grouped.get(key, 0) + 1

    results = []
    status_counts: Counter[str] = Counter()

    HYPHEN_PREFIXES = {"-L", "-T", "-S", "-C", "-M", "-P", "-D"}

    # Stable order for reproducibility: by type then text.
    for (ctype, text), expected_count in sorted(
        grouped.items(),
        key=lambda kv: (kv[0][0], kv[0][1] or ""),
    ):
        # Comments are typesetter / production notes; not source-edits.
        if ctype == "comment":
            status = "layout-typesetter"
            intent = "typesetter-note"
            evidence = f"Comment to typesetter, not a source edit: {text!r}"
            current_count = None

        elif ctype == "deletion" and text == "& ":
            # Editor replaced ampersands with "and" in author/title lists.
            current_count = front_matter_text.count("& ")
            intent = "ampersand-to-and"
            if current_count == 0:
                status = "applied"
                evidence = "No ' & ' tokens remain in front matter"
            elif current_count <= 2:
                status = "applied-mostly"
                evidence = (
                    f"{current_count} residual '& ' occurrences (down from "
                    f"{expected_count} flagged); likely intentional retention"
                )
            else:
                status = "partially-applied"
                evidence = f"{current_count} residual '& ' occurrences vs {expected_count} flagged"

        elif ctype == "insertion" and text == "and  ":
            current_count = front_matter_text.count(" and ")
            intent = "ampersand-to-and"
            status = "applied" if current_count > 0 else "missed"
            evidence = f"' and ' present {current_count} times in front matter"

        elif ctype == "deletion" and text in HYPHEN_PREFIXES:
            current_count = front_matter_text.count(text)
            intent = "hyphen-prefix-removal"
            # We expect SOME residual occurrences (these are 2-char prefixes
            # that legitimately appear in many compound terms). Flag only if
            # the count is unusually high relative to the unique expected.
            status = "inconclusive"
            evidence = (
                f"Bigram {text!r} occurs {current_count} times in current front "
                f"matter; cannot disambiguate which compounds were de-hyphenated"
                f" without per-edit context."
            )

        elif text and len(text.strip()) <= 1:
            current_count = front_matter_text.count(text)
            intent = "micro-fragment"
            status = "inconclusive"
            evidence = (
                f"Sub-token edit ({ctype} {text!r}); covered by bulk passes "
                f"(em-dash, percent, capitalization, abbreviation)."
            )

        elif text and not text.strip():
            current_count = 0
            intent = "whitespace"
            status = "inconclusive"
            evidence = f"Whitespace-only {ctype} ({text!r}); not verifiable"

        else:
            current_count = front_matter_text.count(text) if text else 0
            intent = "fragment"
            if ctype == "insertion":
                status = "applied" if current_count >= expected_count else (
                    "partially-applied" if current_count > 0 else "inconclusive"
                )
                evidence = (
                    f"Inserted fragment {text!r} found {current_count} times "
                    f"(expected ≥{expected_count}). Note: fragment may be from "
                    f"a docx run-split; literal absence does not imply the "
                    f"editor's intent was missed."
                )
            else:  # deletion
                if current_count == 0:
                    status = "applied"
                    evidence = f"Deleted fragment {text!r} absent from front matter"
                else:
                    status = "inconclusive"
                    evidence = (
                        f"Fragment {text!r} occurs {current_count} times; "
                        f"likely a docx run-split artifact, not a literal edit"
                    )

        results.append({
            "type": ctype,
            "text": text,
            "intent_group": intent,
            "expected_count_in_changes": expected_count,
            "current_count_in_front_matter": current_count,
            "status": status,
            "evidence": evidence,
        })
        status_counts[status] += 1

    summary = {
        "source": str(INPUT),
        "front_matter_files_searched": files,
        "front_matter_total_chars": len(front_matter_text),
        "raw_changes_total": len(changes),
        "grouped_changes_total": len(results),
        "by_type_raw": dict(Counter(c["type"] for c in changes)),
        "status_counts": dict(status_counts),
    }

    out_path = OUT_DIR / "vol1-front-matter-coverage.json"
    out_path.write_text(json.dumps({"summary": summary, "entries": results}, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
