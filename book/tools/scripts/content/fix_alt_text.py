#!/usr/bin/env python3
"""Apply edited alt-text from MIT Press copy editor docx extraction.

Reads the alt_text_edits.json and updates fig-alt= attributes in QMD files.

Usage:
    python3 fix_alt_text.py --dry-run book/quarto/contents/vol1/
    python3 fix_alt_text.py book/quarto/contents/vol1/
"""
import argparse
import json
import re
import sys
from pathlib import Path

ALT_TEXT_FILE = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "Desktop/MIT_Press_Feedback/07_alt_text/data/alt_text_edits.json"


def load_alt_text_edits(filepath: Path) -> dict[str, str]:
    """Load the alt-text edits, returning {label: alt_text}."""
    with open(filepath) as f:
        data = json.load(f)
    mapping = {}
    skipped = 0
    for entry in data:
        label = entry.get("label", "").strip()
        alt_text = entry.get("alt_text", "").strip()
        if label and alt_text:
            # Clean up whitespace issues from docx extraction
            alt_text = re.sub(r'\s+', ' ', alt_text).strip()

            # Skip entries with OCR corruption (single letter + space + word)
            # e.g. "s witch t ransformer" instead of "Switch Transformer"
            if re.search(r'\b[a-z] [a-z]{2,}', alt_text):
                skipped += 1
                continue

            # Enforce 250 char limit
            if len(alt_text) > 250:
                alt_text = alt_text[:247] + "..."
            mapping[label] = alt_text
    if skipped:
        print(f"Skipped {skipped} entries with OCR corruption from docx extraction")
    return mapping


def process_file(filepath: Path, edits: dict[str, str], dry_run: bool) -> int:
    """Update fig-alt attributes in a single QMD file."""
    text = filepath.read_text(encoding="utf-8")
    count = 0

    for label, new_alt in edits.items():
        # Pattern: fig-alt="..." in a div that has #fig-LABEL or #label
        # The div looks like: ::: {#fig-xxx ... fig-alt="CURRENT ALT TEXT"}
        pattern = rf'({{[^}}]*#{re.escape(label)}\b[^}}]*fig-alt=")([^"]*?)(")'

        def replacer(m):
            nonlocal count
            old_alt = m.group(2)
            if old_alt.strip() != new_alt:
                count += 1
                if dry_run:
                    short_old = old_alt[:60] + "..." if len(old_alt) > 60 else old_alt
                    short_new = new_alt[:60] + "..." if len(new_alt) > 60 else new_alt
                    print(f"  {label}: \"{short_old}\" -> \"{short_new}\"")
                return m.group(1) + new_alt + m.group(3)
            return m.group(0)

        text = re.sub(pattern, replacer, text)

    if count > 0 and not dry_run:
        filepath.write_text(text, encoding="utf-8")

    return count


def main():
    parser = argparse.ArgumentParser(description="Apply edited alt-text")
    parser.add_argument("path", type=Path, help="Directory to process")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--edits", type=Path, default=None,
                        help="Path to alt_text_edits.json")
    args = parser.parse_args()

    edits_path = args.edits or Path("/Users/VJ/Desktop/MIT_Press_Feedback/07_alt_text/data/alt_text_edits.json")
    edits = load_alt_text_edits(edits_path)
    print(f"Loaded {len(edits)} alt-text edits")

    files = sorted(args.path.rglob("*.qmd"))
    total = 0
    for f in files:
        total += process_file(f, edits, dry_run=args.dry_run)

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"\n{mode}: {total} alt-text updates across {len(files)} files")


if __name__ == "__main__":
    main()
