#!/usr/bin/env python3
"""Enrich bibliography entries with missing publisher/journal fields.

Adds publisher fields to @inproceedings and journal fields to @article
entries based on known venue mappings.

Usage:
    python3 fix_bibliography.py --dry-run book/quarto/contents/vol1/backmatter/references.bib
    python3 fix_bibliography.py book/quarto/contents/vol1/backmatter/references.bib
"""
import argparse
import re
import sys
from pathlib import Path

# Known conference → publisher mappings
CONFERENCE_PUBLISHERS = {
    # ACM conferences
    "neurips": "Curran Associates",
    "nips": "Curran Associates",
    "advances in neural information processing": "Curran Associates",
    "icml": "PMLR",
    "international conference on machine learning": "PMLR",
    "iclr": "OpenReview.net",
    "international conference on learning representations": "OpenReview.net",
    "cvpr": "IEEE",
    "computer vision and pattern recognition": "IEEE",
    "iccv": "IEEE",
    "international conference on computer vision": "IEEE",
    "eccv": "Springer",
    "european conference on computer vision": "Springer",
    "aaai": "AAAI Press",
    "annual meeting of the association for computational linguistics": "Association for Computational Linguistics",
    "association for computational linguistics": "Association for Computational Linguistics",
    "conference on computational linguistics": "Association for Computational Linguistics",
    "emnlp": "Association for Computational Linguistics",
    "naacl": "Association for Computational Linguistics",
    "sigmod": "ACM",
    "vldb": "VLDB Endowment",
    "kdd": "ACM",
    "knowledge discovery and data mining": "ACM",
    "isca": "ACM",
    "international symposium on computer architecture": "ACM",
    "micro": "ACM",
    "hpca": "IEEE",
    "asplos": "ACM",
    "osdi": "USENIX Association",
    "sosp": "ACM",
    "nsdi": "USENIX Association",
    "usenix": "USENIX Association",
    "mlsys": "MLSys",
    "machine learning and systems": "MLSys",
    "north american chapter": "Association for Computational Linguistics",
    "interspeech": "ISCA",
    "icassp": "IEEE",
    "chi": "ACM",
    "fat*": "ACM",
    "facct": "ACM",
    "fairness, accountability": "ACM",
    "aistats": "PMLR",
    "artificial intelligence and statistics": "PMLR",
    "sigcomm": "ACM",
    "infocom": "IEEE",
    "dac": "ACM",
    "date": "IEEE",
    "embedded systems": "ACM",
    "sensys": "ACM",
    "mobicom": "ACM",
    "mobisys": "ACM",
    "ieee security": "IEEE",
    "ieee symposium": "IEEE",
    "ieee international": "IEEE",
    "ieee conference": "IEEE",
    "acm conference": "ACM",
    "acm symposium": "ACM",
    # Removed: "proceedings of the" → too generic, causes false matches
}

# Known journal names for articles
KNOWN_JOURNALS = {
    "arxiv": "arXiv preprint",
    "jmlr": "Journal of Machine Learning Research",
    "nature": "Nature",
    "science": "Science",
    "transactions on": "IEEE",
    "communications of the acm": "Communications of the ACM",
}


def find_publisher_for_entry(entry_text: str) -> str | None:
    """Try to determine publisher from booktitle or other fields."""
    # Extract booktitle
    m = re.search(r'booktitle\s*=\s*\{([^}]+)\}', entry_text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    booktitle = m.group(1).lower().replace('\n', ' ')

    # Check specific keywords first (longer/more specific before shorter/generic)
    # Sort by keyword length descending to prioritize specific matches
    sorted_keywords = sorted(CONFERENCE_PUBLISHERS.items(), key=lambda x: len(x[0]), reverse=True)
    for keyword, publisher in sorted_keywords:
        kw_lower = keyword.lower()
        # For short keywords (< 6 chars), require word boundaries to avoid
        # substring matches like "micro" in "microsoft"
        if len(kw_lower) < 6:
            if re.search(r'\b' + re.escape(kw_lower) + r'\b', booktitle):
                return publisher
        else:
            if kw_lower in booktitle:
                return publisher

    return None


def find_journal_for_entry(entry_text: str) -> str | None:
    """Try to determine journal from eprint or other fields."""
    # Check if it's an arXiv preprint
    if re.search(r'eprint\s*=|archiveprefix\s*=\s*\{arxiv\}|arxiv', entry_text, re.IGNORECASE):
        return "arXiv preprint"

    # Check note field for journal info
    m = re.search(r'note\s*=\s*\{([^}]+)\}', entry_text, re.IGNORECASE)
    if m:
        note = m.group(1).lower()
        for keyword, journal in KNOWN_JOURNALS.items():
            if keyword in note:
                return journal

    return None


def add_field_to_entry(entry_text: str, field_name: str, field_value: str) -> str:
    """Add a field to a bib entry, before the closing }."""
    # Find the last } that closes the entry
    # Insert before it
    lines = entry_text.rstrip().rstrip('}').rstrip()
    # Ensure the last line has a comma
    if not lines.rstrip().endswith(','):
        lines = lines.rstrip() + ','
    return f"{lines}\n  {field_name} = {{{field_value}}},\n}}"


def process_bib(filepath: Path, dry_run: bool = False) -> int:
    """Process a bib file. Returns count of entries enriched."""
    text = filepath.read_text(encoding="utf-8")

    # Split into entries
    entries = re.split(r'(?=@\w+\{)', text)
    new_entries = []
    count = 0

    for entry in entries:
        entry_stripped = entry.strip()
        if not entry_stripped:
            new_entries.append(entry)
            continue

        m = re.match(r'@(\w+)\{([^,]+)', entry_stripped)
        if not m:
            new_entries.append(entry)
            continue

        etype = m.group(1).lower()
        key = m.group(2).strip()
        modified = False

        if etype == 'inproceedings' and 'publisher' not in entry.lower():
            publisher = find_publisher_for_entry(entry)
            if publisher:
                entry = add_field_to_entry(entry.rstrip(), 'publisher', publisher)
                if dry_run:
                    print(f"  + {key}: publisher = {{{publisher}}}")
                count += 1
                modified = True

        elif etype == 'article' and 'journal' not in entry.lower():
            journal = find_journal_for_entry(entry)
            if journal:
                entry = add_field_to_entry(entry.rstrip(), 'journal', journal)
                if dry_run:
                    print(f"  + {key}: journal = {{{journal}}}")
                count += 1
                modified = True

        new_entries.append(entry)

    if count > 0 and not dry_run:
        filepath.write_text("".join(new_entries), encoding="utf-8")
        print(f"  {filepath}: {count} entries enriched")

    return count


def main():
    parser = argparse.ArgumentParser(description="Enrich bibliography entries")
    parser.add_argument("path", type=Path, help="Bib file to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview without applying")
    args = parser.parse_args()

    count = process_bib(args.path, dry_run=args.dry_run)
    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"\n{mode}: {count} entries enriched")


if __name__ == "__main__":
    main()
