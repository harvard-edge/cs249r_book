#!/usr/bin/env python3
"""
Capitalize cross-reference prefixes at sentence starts.

MIT Press convention: lowercase inline refs ("figure 1", "table 2") with
uppercase only at sentence starts ("Figure 1 shows...").  Quarto supports
this via @Fig- / @Tbl- / @Sec- / @Eq- / @Lst- syntax when the crossref
prefix config is set to lowercase.

This script finds all @fig-/@tbl-/@sec-/@eq-/@lst- references in QMD files,
classifies each as sentence-start or mid-sentence, and optionally applies
the capitalization.

Usage:
    # Dry run (report only):
    python3 capitalize_sentence_start_xrefs.py --dry-run book/quarto/contents/vol1/

    # Apply changes:
    python3 capitalize_sentence_start_xrefs.py book/quarto/contents/vol1/

    # Single file:
    python3 capitalize_sentence_start_xrefs.py book/quarto/contents/vol1/conclusion/conclusion.qmd
"""

import argparse
import re
import sys
from pathlib import Path

XREF_TYPES = ("fig", "tbl", "sec", "eq", "lst")
XREF_RE = re.compile(r"@(fig|tbl|sec|eq|lst)-([a-zA-Z0-9_-]+)")


def is_inside_code_block(lines: list[str], line_idx: int) -> bool:
    """Check if a line is inside a fenced code block."""
    fence_count = 0
    for i in range(line_idx):
        stripped = lines[i].strip()
        if stripped.startswith("```"):
            fence_count += 1
    return fence_count % 2 == 1


def is_yaml_frontmatter(lines: list[str], line_idx: int) -> bool:
    """Check if a line is inside YAML front matter."""
    if line_idx == 0:
        return lines[0].strip() == "---"
    if lines[0].strip() != "---":
        return False
    for i in range(1, line_idx):
        if lines[i].strip() == "---":
            return False
    return True


def is_in_attribute(line: str, match_start: int) -> bool:
    """Check if the match is inside a YAML/attribute string (fig-cap=, title=, etc.)."""
    prefix = line[:match_start]
    attr_patterns = [
        'fig-cap="', 'tbl-cap="', 'lst-cap="', 'fig-alt="', 'tbl-alt="',
        'title="', '#| fig-cap:', '#| tbl-cap:', '#| fig-alt:',
    ]
    for pat in attr_patterns:
        if pat in prefix:
            return True
    return False


def is_in_comment_or_cell_header(line: str) -> bool:
    """Check if the line is a Python comment or cell header."""
    stripped = line.strip()
    return stripped.startswith("#|")


def is_in_footnote_def(line: str) -> bool:
    """Check if the line is a footnote definition [^fn-...]:"""
    return bool(re.match(r'^\s*\[\^fn-', line))


def is_in_div_opener(line: str) -> bool:
    """Check if the ref is inside a ::: {#id ...} div opener."""
    return line.strip().startswith(":::")


def is_in_label_definition(line: str) -> bool:
    """Check if line is a label definition like {#fig-foo} or {#tbl-foo}."""
    return bool(re.search(r'\{#(fig|tbl|sec|eq|lst)-', line))


def get_preceding_context(lines: list[str], line_idx: int, col: int) -> str:
    """Get the text preceding the match on the same line and preceding lines in the paragraph."""
    before_on_line = lines[line_idx][:col].rstrip()

    if before_on_line:
        return before_on_line

    # Look at preceding lines in the same paragraph (until blank line)
    for i in range(line_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            return ""  # blank line = paragraph boundary = sentence start
        return stripped

    return ""  # start of file


def classify_ref(lines: list[str], line_idx: int, col: int) -> str:
    """
    Classify a cross-reference as 'sentence_start' or 'mid_sentence'.

    Returns one of:
        'sentence_start' -- capitalize
        'mid_sentence'   -- keep lowercase
        'skip'           -- inside code/YAML/attribute/comment, don't touch
    """
    line = lines[line_idx]

    # Skip: inside code block
    if is_inside_code_block(lines, line_idx):
        return "skip"

    # Skip: YAML front matter
    if is_yaml_frontmatter(lines, line_idx):
        return "skip"

    # Skip: inside attribute string
    if is_in_attribute(line, col):
        return "skip"

    # Skip: comment or cell header
    if is_in_comment_or_cell_header(line):
        return "skip"

    # Skip: div opener line
    if is_in_div_opener(line):
        return "skip"

    # Footnote definitions: refs inside footnotes are mid-sentence
    # (the footnote text is a continuation, not a new sentence)
    if is_in_footnote_def(line):
        return "mid_sentence"

    # Get what's before the ref on this line
    before_on_line = line[:col].rstrip()

    if not before_on_line:
        # Ref starts the line. Check if this is a paragraph start.
        preceding = get_preceding_context(lines, line_idx, col)
        if not preceding:
            # Paragraph boundary (blank line above or start of file)
            return "sentence_start"
        # Previous line ends with sentence-ending punctuation
        if preceding and preceding[-1] in ".?!":
            return "sentence_start"
        # Previous line ends with closing quote after punctuation
        if len(preceding) >= 2 and preceding[-2] in ".?!" and preceding[-1] in "\"'":
            return "sentence_start"
        # Previous line is a heading
        if preceding.startswith("#"):
            return "sentence_start"
        # Previous line ends with :::
        if preceding.endswith(":::"):
            return "sentence_start"
        # Otherwise it's a continuation line (mid-sentence wrapped)
        return "mid_sentence"

    # Ref is mid-line. Check what's immediately before.
    # Sentence-ending punctuation followed by space
    if re.search(r'[.?!]\s*$', before_on_line):
        return "sentence_start"
    if re.search(r'[.?!][""]\s*$', before_on_line):
        return "sentence_start"

    # After colon + space -- only capitalize if the ref is followed by
    # verb-like text (indicating a sentence), not if it's a bare link
    # or the last thing on the line (e.g., "**Label**: @sec-foo")
    if re.search(r':\s*$', before_on_line):
        after_ref = line[col:]
        ref_match = XREF_RE.match(after_ref)
        if ref_match:
            rest_after_ref = after_ref[ref_match.end():].strip()
            # If the ref is followed by nothing or just punctuation,
            # it's a bare link, not a sentence start
            if not rest_after_ref or rest_after_ref[0] in ".,;:)]\n":
                return "mid_sentence"
            # If followed by words, it's starting a sentence after the colon
            return "sentence_start"
        return "mid_sentence"

    # Everything else is mid-sentence
    return "mid_sentence"


def process_file(filepath: Path, dry_run: bool) -> dict:
    """Process a single QMD file. Returns stats dict."""
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")

    stats = {
        "file": str(filepath),
        "total_refs": 0,
        "sentence_start": 0,
        "mid_sentence": 0,
        "skipped": 0,
        "already_capitalized": 0,
        "changes": [],
    }

    # Find all refs with their positions
    refs = []
    for i, line in enumerate(lines):
        for m in XREF_RE.finditer(line):
            refs.append((i, m.start(), m.group(0), m.group(1), m.group(2)))

    stats["total_refs"] = len(refs)

    # Classify each ref
    changes_to_apply = []  # (line_idx, old_text, new_text, col)
    for line_idx, col, full_ref, ref_type, ref_id in refs:
        classification = classify_ref(lines, line_idx, col)

        if classification == "skip":
            stats["skipped"] += 1
            continue

        if classification == "sentence_start":
            stats["sentence_start"] += 1
            # Check if already capitalized
            if full_ref[1].isupper():
                stats["already_capitalized"] += 1
                continue
            # Need to capitalize
            new_ref = f"@{ref_type[0].upper()}{ref_type[1:]}-{ref_id}"
            changes_to_apply.append((line_idx, full_ref, new_ref, col))
            stats["changes"].append({
                "line": line_idx + 1,
                "old": full_ref,
                "new": new_ref,
                "context": lines[line_idx].strip()[:120],
            })
        else:
            stats["mid_sentence"] += 1

    # Apply changes (process each line, replacing from right to left to preserve positions)
    if not dry_run and changes_to_apply:
        # Group changes by line
        changes_by_line = {}
        for line_idx, old_text, new_text, col in changes_to_apply:
            changes_by_line.setdefault(line_idx, []).append((col, old_text, new_text))

        for line_idx, line_changes in changes_by_line.items():
            # Sort by column position, descending (replace from right to left)
            line_changes.sort(key=lambda x: x[0], reverse=True)
            line = lines[line_idx]
            for col, old_text, new_text in line_changes:
                line = line[:col] + new_text + line[col + len(old_text):]
            lines[line_idx] = line

        filepath.write_text("\n".join(lines), encoding="utf-8")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Capitalize sentence-start cross-references")
    parser.add_argument("paths", nargs="+", help="QMD files or directories to process")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't modify files")
    args = parser.parse_args()

    qmd_files = []
    for p in args.paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".qmd":
            qmd_files.append(path)
        elif path.is_dir():
            qmd_files.extend(sorted(path.rglob("*.qmd")))
        else:
            print(f"Warning: skipping {p}", file=sys.stderr)

    total_changes = 0
    total_refs = 0
    total_sentence_start = 0
    total_mid = 0
    total_skipped = 0

    for qmd in qmd_files:
        stats = process_file(qmd, args.dry_run)
        total_refs += stats["total_refs"]
        total_sentence_start += stats["sentence_start"]
        total_mid += stats["mid_sentence"]
        total_skipped += stats["skipped"]

        if stats["changes"]:
            total_changes += len(stats["changes"])
            rel = qmd.relative_to(Path.cwd()) if qmd.is_relative_to(Path.cwd()) else qmd
            print(f"\n{'[DRY RUN] ' if args.dry_run else ''}{rel}: {len(stats['changes'])} changes")
            for c in stats["changes"]:
                print(f"  L{c['line']:>5}: {c['old']} -> {c['new']}")
                print(f"         {c['context']}")

    print(f"\n{'=' * 60}")
    print(f"Total refs scanned:       {total_refs}")
    print(f"  Sentence-start:         {total_sentence_start}")
    print(f"  Mid-sentence:           {total_mid}")
    print(f"  Skipped (code/YAML/..): {total_skipped}")
    print(f"  Changes {'needed' if args.dry_run else 'applied'}: {total_changes}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
