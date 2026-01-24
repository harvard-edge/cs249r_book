#!/usr/bin/env python3
"""
audit_footnotes_cross_chapter.py
--------------------------------
Extracts all footnotes from .qmd files and identifies potential duplicates
across chapters. Helps catch repeated definitions that should be differentiated
or consolidated.

Usage:
    python3 audit_footnotes_cross_chapter.py <search_directory>
    python3 audit_footnotes_cross_chapter.py book/quarto/contents/vol1

Output:
    - List of all footnote terms and where they appear
    - Flagged duplicates (same term, similar content in multiple chapters)
"""

import sys
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


def extract_footnotes(filepath: Path) -> list[tuple[str, str]]:
    """Extract all footnotes from a .qmd file.

    Returns list of (footnote_id, footnote_content) tuples.
    """
    content = filepath.read_text(encoding='utf-8')

    # Match footnote definitions: [^fn-name]: content (may span multiple lines)
    # Footnote ends at blank line or next footnote
    pattern = r'\[\^(fn-[^\]]+)\]:\s*(.+?)(?=\n\n|\n\[\^|\Z)'

    matches = re.findall(pattern, content, re.DOTALL)

    # Clean up content (normalize whitespace)
    return [(fn_id, ' '.join(content.split())) for fn_id, content in matches]


def similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 audit_footnotes_cross_chapter.py <search_directory>")
        sys.exit(1)

    search_dir = Path(sys.argv[1])
    if not search_dir.is_dir():
        print(f"Error: '{search_dir}' is not a valid directory")
        sys.exit(1)

    # Collect all footnotes by ID
    footnotes_by_id: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for qmd_file in sorted(search_dir.rglob('*.qmd')):
        chapter_name = qmd_file.stem
        footnotes = extract_footnotes(qmd_file)

        for fn_id, fn_content in footnotes:
            footnotes_by_id[fn_id].append((chapter_name, fn_content))

    # Report
    print("=" * 70)
    print("CROSS-CHAPTER FOOTNOTE AUDIT")
    print("=" * 70)
    print()

    # Find duplicates (same ID in multiple chapters)
    duplicates = {k: v for k, v in footnotes_by_id.items() if len(v) > 1}

    if not duplicates:
        print("✓ No duplicate footnote IDs found across chapters.")
        print()
        print(f"Total unique footnotes: {len(footnotes_by_id)}")
        return

    print(f"Found {len(duplicates)} footnote IDs appearing in multiple chapters:")
    print()

    high_similarity = []

    for fn_id, occurrences in sorted(duplicates.items()):
        print(f"[^{fn_id}] — appears in {len(occurrences)} chapters:")

        for chapter, content in occurrences:
            # Truncate content for display
            display = content[:100] + "..." if len(content) > 100 else content
            print(f"  • {chapter}: {display}")

        # Check similarity between occurrences
        if len(occurrences) == 2:
            sim = similarity(occurrences[0][1], occurrences[1][1])
            if sim > 0.7:
                high_similarity.append((fn_id, sim, occurrences))
                print(f"  ⚠️  Similarity: {sim:.0%} — may need differentiation")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total unique footnote IDs: {len(footnotes_by_id)}")
    print(f"IDs appearing in multiple chapters: {len(duplicates)}")
    print(f"High-similarity duplicates (>70%): {len(high_similarity)}")

    if high_similarity:
        print()
        print("ACTION NEEDED — These footnotes are very similar across chapters:")
        for fn_id, sim, _ in high_similarity:
            print(f"  • [^{fn_id}] ({sim:.0%} similar)")
        print()
        print("Consider: differentiate by chapter focus, or keep only the first occurrence.")


if __name__ == '__main__':
    main()
