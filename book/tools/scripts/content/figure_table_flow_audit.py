#!/usr/bin/env python3
"""
Figure & Table Flow Audit Script

Scans Quarto (.qmd) chapters and reports:
1. Where each figure/table is DEFINED (the ::: {#fig-...} or {#tbl-...} block)
2. Where it is FIRST REFERENCED in prose (@fig-... or @tbl-...)
3. The gap (in lines) between definition and first reference
4. Whether the definition comes BEFORE or AFTER its first reference

Flags issues where:
- A figure/table is defined far from its first reference (>20 lines gap)
- A figure/table is referenced BEFORE it is defined (reader sees reference
  before seeing the visual)
- A figure/table is defined but never referenced (orphan)
- A figure/table is referenced but never defined (broken reference, likely
  cross-chapter)

Usage:
    python3 book/tools/scripts/content/figure_table_flow_audit.py [chapter.qmd ...]

    If no files specified, scans all Vol 1 and Vol 2 chapter .qmd files.
"""

import re
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FloatElement:
    """A figure or table element in a chapter."""
    label: str  # e.g., "fig-ai-triad" or "tbl-dam-taxonomy"
    kind: str  # "figure" or "table"
    definition_line: Optional[int] = None
    first_reference_line: Optional[int] = None
    all_reference_lines: list = field(default_factory=list)

    @property
    def gap(self) -> Optional[int]:
        if self.definition_line and self.first_reference_line:
            return self.definition_line - self.first_reference_line
        return None

    @property
    def status(self) -> str:
        if not self.definition_line:
            return "XREF"  # cross-chapter or broken reference
        if not self.first_reference_line:
            return "ORPHAN"  # defined but never referenced
        gap = self.gap
        # gap = definition_line - first_reference_line
        # gap > 0: definition comes AFTER reference (good if small)
        # gap < 0: definition comes BEFORE reference (figure appears early)
        if gap > 30:
            # Definition is well AFTER reference (reader waits too long for figure)
            return "LATE"
        if gap < -5:
            # Definition is way BEFORE reference (figure appears before it's mentioned)
            return "EARLY"
        return "OK"


def scan_chapter(filepath: Path) -> dict[str, FloatElement]:
    """Scan a single .qmd file for figure/table definitions and references."""
    elements: dict[str, FloatElement] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Patterns for DEFINITIONS
    # Quarto fenced div: ::: {#fig-foo ...} or ::: {#tbl-foo ...}
    div_def_pattern = re.compile(r':::\s*\{[^}]*#((?:fig|tbl)-[\w-]+)')
    # Markdown image with label: ![...](...){{#fig-foo ...}
    img_def_pattern = re.compile(r'!\[.*?\]\(.*?\)\s*\{[^}]*#((?:fig|tbl)-[\w-]+)')
    # Table caption with label: : **caption** {#tbl-foo}
    tbl_caption_pattern = re.compile(r'^:\s+.*\{[^}]*#((?:fig|tbl)-[\w-]+)')

    # Pattern for REFERENCES in prose: @fig-foo or @tbl-foo
    ref_pattern = re.compile(r'@((?:fig|tbl)-[\w-]+)')

    # Track whether we're inside a figure/table definition block
    # (to avoid counting self-references in captions/alt-text)
    in_float_block = False
    float_block_label = None
    brace_depth = 0

    # Track code blocks for code-cell figure detection
    in_code_block = False
    code_block_start = 0
    cell_options: dict[str, str] = {}

    for line_num, line in enumerate(lines, start=1):
        stripped = line.rstrip()

        # --- Code-cell figure detection ---
        # Detect code block start: ```{python}, ```{r}, etc.
        if not in_code_block and re.match(r'^```\{(?:python|r|julia|ojs)', stripped):
            in_code_block = True
            code_block_start = line_num
            cell_options = {}
            continue

        # Detect code block end
        if in_code_block and stripped == '```':
            label = cell_options.get('label', '')
            if label.startswith('fig-') or label.startswith('tbl-'):
                kind = "figure" if label.startswith("fig-") else "table"
                if label not in elements:
                    elements[label] = FloatElement(label=label, kind=kind)
                if elements[label].definition_line is None:
                    elements[label].definition_line = code_block_start
            in_code_block = False
            cell_options = {}
            continue

        # Inside a code block — collect cell options (skip prose ref detection)
        if in_code_block:
            opt_match = re.match(r'^#\|\s*([\w-]+):\s*(.+)$', stripped)
            if opt_match:
                key = opt_match.group(1)
                value = opt_match.group(2).strip().strip('"').strip("'")
                cell_options[key] = value
            continue

        # --- Attribute-based definition detection ---
        for pattern in [div_def_pattern, img_def_pattern, tbl_caption_pattern]:
            match = pattern.search(line)
            if match:
                label = match.group(1)
                kind = "figure" if label.startswith("fig-") else "table"
                if label not in elements:
                    elements[label] = FloatElement(label=label, kind=kind)
                if elements[label].definition_line is None:
                    elements[label].definition_line = line_num

                if pattern == div_def_pattern:
                    in_float_block = True
                    float_block_label = label

        # Track block boundaries for fenced divs
        if in_float_block:
            # Count ::: openings and closings
            line_stripped = line.strip()
            if line_stripped.startswith(":::") and not line_stripped.startswith("::: {"):
                # This is a closing :::
                in_float_block = False
                float_block_label = None

        # Check for references in prose
        for match in ref_pattern.finditer(line):
            label = match.group(1)
            kind = "figure" if label.startswith("fig-") else "table"

            # Skip self-references within definition blocks (captions, alt-text)
            if in_float_block and label == float_block_label:
                continue

            # Skip references inside fig-cap or fig-alt strings
            # (these are part of the definition, not prose references)
            if 'fig-cap=' in line or 'fig-alt=' in line:
                continue

            if label not in elements:
                elements[label] = FloatElement(label=label, kind=kind)

            elements[label].all_reference_lines.append(line_num)
            if (elements[label].first_reference_line is None or
                    line_num < elements[label].first_reference_line):
                elements[label].first_reference_line = line_num

    return elements


def format_report(filepath: Path, elements: dict[str, FloatElement]) -> str:
    """Format a report for a single chapter."""
    chapter_name = filepath.stem
    parent_dir = filepath.parent.name

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  {parent_dir}/{chapter_name}.qmd")
    lines.append(f"{'='*80}")

    # Separate by status
    issues = []
    ok_items = []
    xrefs = []

    for label, elem in sorted(elements.items(), key=lambda x: x[1].definition_line or 9999):
        status = elem.status
        if status == "OK":
            ok_items.append(elem)
        elif status == "XREF":
            xrefs.append(elem)
        else:
            issues.append(elem)

    if issues:
        lines.append(f"\n  ISSUES FOUND ({len(issues)}):")
        lines.append(f"  {'─'*74}")
        lines.append(f"  {'Label':<40} {'Status':<8} {'Defined':<10} {'1st Ref':<10} {'Gap':>6}")
        lines.append(f"  {'─'*74}")

        for elem in issues:
            def_str = f"L{elem.definition_line}" if elem.definition_line else "—"
            ref_str = f"L{elem.first_reference_line}" if elem.first_reference_line else "—"
            gap_str = f"{elem.gap:+d}" if elem.gap is not None else "—"

            marker = ""
            if elem.status == "LATE":
                marker = " ⚠️  Figure appears too far AFTER mention (move earlier)"
            elif elem.status == "EARLY":
                marker = " ⚠️  Figure appears BEFORE it's mentioned (move later)"
            elif elem.status == "ORPHAN":
                marker = " ⚠️  Defined but never referenced in prose"

            lines.append(f"  {elem.label:<40} {elem.status:<8} {def_str:<10} {ref_str:<10} {gap_str:>6}{marker}")
    else:
        lines.append(f"\n  ✅ No placement issues found.")

    if ok_items:
        lines.append(f"\n  OK ({len(ok_items)}):")
        lines.append(f"  {'─'*74}")
        for elem in ok_items:
            def_str = f"L{elem.definition_line}" if elem.definition_line else "—"
            ref_str = f"L{elem.first_reference_line}" if elem.first_reference_line else "—"
            gap_str = f"{elem.gap:+d}" if elem.gap is not None else "—"
            lines.append(f"  {elem.label:<40} {'OK':<8} {def_str:<10} {ref_str:<10} {gap_str:>6}")

    if xrefs:
        lines.append(f"\n  CROSS-CHAPTER REFS ({len(xrefs)}):")
        for elem in xrefs:
            ref_lines = ", ".join(f"L{l}" for l in elem.all_reference_lines[:5])
            lines.append(f"  {elem.label:<40} referenced at {ref_lines}")

    # Summary stats
    total = len(elements)
    issue_count = len(issues)
    lines.append(f"\n  Summary: {total} elements | {issue_count} issues | "
                 f"{len(ok_items)} OK | {len(xrefs)} cross-chapter refs")

    return "\n".join(lines)


def format_report_compact(filepath: Path, elements: dict[str, FloatElement]) -> tuple[str, int]:
    """Format a compact report for pre-commit (only issues, no OK items)."""
    chapter_name = filepath.stem
    parent_dir = filepath.parent.name

    issues = [e for e in elements.values() if e.status in ("LATE", "EARLY", "ORPHAN")]

    if not issues:
        return "", 0

    lines = []
    lines.append(f"\n{parent_dir}/{chapter_name}.qmd - {len(issues)} issue(s):")

    for elem in sorted(issues, key=lambda x: x.definition_line or 9999):
        def_str = f"L{elem.definition_line}" if elem.definition_line else "—"
        ref_str = f"L{elem.first_reference_line}" if elem.first_reference_line else "—"

        if elem.status == "LATE":
            lines.append(f"  {elem.label}: defined at {def_str}, first referenced at {ref_str} (figure appears too far AFTER mention)")
        elif elem.status == "EARLY":
            lines.append(f"  {elem.label}: defined at {def_str}, first referenced at {ref_str} (figure appears BEFORE it's mentioned)")
        elif elem.status == "ORPHAN":
            lines.append(f"  {elem.label}: defined at {def_str} but NEVER REFERENCED")

    return "\n".join(lines), len(issues)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit figure and table placement in QMD files"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="QMD files to check. If none provided, scans all vol1/vol2 chapters."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code 1 if any issues found (for pre-commit)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show issues, not OK items (compact output for pre-commit)"
    )

    args = parser.parse_args()

    # Determine files to scan
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        # Find all chapter .qmd files in vol1 and vol2
        base = Path(__file__).resolve().parents[3] / "quarto" / "contents"
        files = []
        for vol in ["vol1", "vol2"]:
            vol_path = base / vol
            if vol_path.exists():
                for qmd in sorted(vol_path.rglob("*.qmd")):
                    # Skip frontmatter, backmatter, parts, index
                    rel = qmd.relative_to(vol_path)
                    skip_dirs = {"frontmatter", "backmatter", "parts"}
                    if any(part in skip_dirs for part in rel.parts):
                        continue
                    if qmd.stem == "index":
                        continue
                    files.append(qmd)

    if not files:
        print("No .qmd files found.")
        sys.exit(1)

    if not args.quiet:
        print(f"Figure & Table Flow Audit")
        print(f"Scanning {len(files)} chapters...\n")

    total_issues = 0
    total_elements = 0

    for filepath in files:
        if not filepath.exists():
            if not args.quiet:
                print(f"  WARNING: {filepath} not found, skipping.")
            continue

        elements = scan_chapter(filepath)
        if not elements:
            continue

        if args.quiet:
            report, issue_count = format_report_compact(filepath, elements)
            if report:
                print(report)
        else:
            report = format_report(filepath, elements)
            print(report)
            issue_count = sum(1 for e in elements.values() if e.status in ("LATE", "EARLY", "ORPHAN"))

        total_elements += len(elements)
        total_issues += issue_count

    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"  GRAND TOTAL: {total_elements} elements across {len(files)} chapters | {total_issues} issues")
        print(f"{'='*80}")
    elif total_issues > 0:
        print(f"\nTotal: {total_issues} figure/table placement issue(s)")
        print("Figures should appear immediately after the paragraph that first references them.")

    # Exit with error if strict mode and issues found
    if args.strict and total_issues > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
