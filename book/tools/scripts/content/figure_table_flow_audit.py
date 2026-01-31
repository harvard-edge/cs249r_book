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
        if gap < -5:
            # Definition is well AFTER reference (reader sees @fig before figure)
            return "LATE"
        if gap > 30:
            # Definition is way BEFORE reference (figure appears pages early)
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

    for line_num, line in enumerate(lines, start=1):
        # Check for definition starts
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
            stripped = line.strip()
            if stripped.startswith(":::") and not stripped.startswith("::: {"):
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
                marker = " ⚠️  Definition AFTER first reference"
            elif elem.status == "EARLY":
                marker = " ⚠️  Definition far BEFORE first reference"
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


def main():
    # Determine files to scan
    if len(sys.argv) > 1:
        files = [Path(f) for f in sys.argv[1:]]
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

    print(f"Figure & Table Flow Audit")
    print(f"Scanning {len(files)} chapters...\n")

    total_issues = 0
    total_elements = 0

    for filepath in files:
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found, skipping.")
            continue

        elements = scan_chapter(filepath)
        if not elements:
            continue

        report = format_report(filepath, elements)
        print(report)

        total_elements += len(elements)
        total_issues += sum(1 for e in elements.values() if e.status in ("LATE", "EARLY", "ORPHAN"))

    print(f"\n{'='*80}")
    print(f"  GRAND TOTAL: {total_elements} elements across {len(files)} chapters | {total_issues} issues")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
