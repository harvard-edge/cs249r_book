#!/usr/bin/env python3
"""
Figure Relocator for Quarto Documents

Safely moves figures closer to where they're first referenced,
respecting footnote definitions that must stay near their markers.

Usage:
    python relocate_figures.py <file.qmd>              # Preview mode (dry run)
    python relocate_figures.py <file.qmd> --apply     # Apply changes
    python relocate_figures.py <file.qmd> --verbose   # Show detailed analysis

Author: MLSysBook Team
"""

import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import copy


@dataclass
class Figure:
    """Represents a figure block in the document."""
    fig_id: str
    start_line: int  # 0-indexed
    end_line: int    # 0-indexed, inclusive
    fig_type: str    # 'div' for ::: blocks, 'markdown' for ![](...){}
    colon_count: int = 0  # Number of colons (3 or 4) for div type
    
    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class FigureReference:
    """Represents a reference to a figure (@fig-xxx)."""
    fig_id: str
    line_num: int  # 0-indexed


@dataclass
class FootnoteDefinition:
    """Represents a footnote definition [^fn-xxx]: ..."""
    fn_id: str
    start_line: int  # 0-indexed
    end_line: int    # 0-indexed, inclusive


@dataclass
class Paragraph:
    """Represents a paragraph (non-empty lines between blank lines)."""
    start_line: int
    end_line: int
    has_footnote_refs: List[str] = field(default_factory=list)


def parse_figures(lines: List[str]) -> List[Figure]:
    """Extract all figure blocks from the document (both div and markdown styles)."""
    figures = []
    
    # Pattern for div-style figures: ::: {#fig-xxx} or :::: {#fig-xxx}
    div_fig_pattern = re.compile(r'^(:{3,4})\s*\{#(fig-[^\s}]+)')
    
    # Pattern for markdown-style figures: ![caption](path){#fig-xxx}
    # Can span multiple lines if caption is long, but the {#fig-xxx} is on the last line
    md_fig_pattern = re.compile(r'!\[.*?\]\(.*?\)\{#(fig-[^\s}]+)')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for div-style figure
        div_match = div_fig_pattern.match(line)
        if div_match:
            colons = div_match.group(1)
            fig_id = div_match.group(2)
            colon_count = len(colons)
            start_line = i
            
            # Find the closing fence (same number of colons, standalone)
            closing_pattern = re.compile(r'^:{' + str(colon_count) + r'}\s*$')
            j = i + 1
            depth = 1
            while j < len(lines):
                # Check for nested fenced divs
                if re.match(r'^:{' + str(colon_count) + r'}\s*\{', lines[j]):
                    depth += 1
                elif closing_pattern.match(lines[j]):
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            
            figures.append(Figure(
                fig_id=fig_id,
                start_line=start_line,
                end_line=j,
                fig_type='div',
                colon_count=colon_count
            ))
            i = j + 1
            continue
        
        # Check for markdown-style figure on this line
        md_match = md_fig_pattern.search(line)
        if md_match:
            fig_id = md_match.group(1)
            # Markdown figures are typically single-line, but may have preceding lines
            # that are part of the same figure (e.g., multi-line captions)
            start_line = i
            # Check if this is a continuation from previous lines (rare but possible)
            # For now, treat as single line
            figures.append(Figure(
                fig_id=fig_id,
                start_line=start_line,
                end_line=i,
                fig_type='markdown'
            ))
            i += 1
            continue
        
        i += 1
    
    return figures


def parse_references(lines: List[str]) -> List[FigureReference]:
    """Extract all figure references from the document."""
    references = []
    # Match @fig-xxx references (not inside figure definitions)
    ref_pattern = re.compile(r'@(fig-[a-zA-Z0-9_-]+)')
    
    for i, line in enumerate(lines):
        # Skip lines that are figure definitions themselves
        if re.match(r'^:{3,4}\s*\{#fig-', line):
            continue
        for match in ref_pattern.finditer(line):
            references.append(FigureReference(
                fig_id=match.group(1),
                line_num=i
            ))
    
    return references


def parse_footnotes(lines: List[str]) -> List[FootnoteDefinition]:
    """Extract all footnote definitions from the document."""
    footnotes = []
    fn_def_pattern = re.compile(r'^\[\^([^\]]+)\]:\s*')
    
    i = 0
    while i < len(lines):
        match = fn_def_pattern.match(lines[i])
        if match:
            fn_id = match.group(1)
            start_line = i
            
            # Footnote continues until blank line or new block
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                # Stop at blank line, new footnote, or block element
                if (next_line.strip() == '' or 
                    fn_def_pattern.match(next_line) or
                    re.match(r'^:{3,}', next_line) or
                    re.match(r'^#', next_line)):
                    break
                j += 1
            
            footnotes.append(FootnoteDefinition(
                fn_id=fn_id,
                start_line=start_line,
                end_line=j - 1
            ))
            i = j
        else:
            i += 1
    
    return footnotes


def find_paragraph_boundary(lines: List[str], line_num: int) -> Tuple[int, int]:
    """Find the start and end of the paragraph containing line_num."""
    # Find start (go backwards to blank line or start)
    start = line_num
    while start > 0 and lines[start - 1].strip() != '':
        # Don't cross block boundaries
        if re.match(r'^:{3,}', lines[start - 1]) or re.match(r'^#', lines[start - 1]):
            break
        start -= 1
    
    # Find end (go forwards to blank line or end)
    end = line_num
    while end < len(lines) - 1 and lines[end + 1].strip() != '':
        # Don't cross block boundaries
        if re.match(r'^:{3,}', lines[end + 1]) or re.match(r'^#', lines[end + 1]):
            break
        end += 1
    
    return start, end


def get_footnote_refs_in_range(lines: List[str], start: int, end: int) -> List[str]:
    """Find all footnote references [^fn-xxx] in a line range."""
    fn_ref_pattern = re.compile(r'\[\^([^\]]+)\](?!:)')
    refs = []
    for i in range(start, end + 1):
        for match in fn_ref_pattern.finditer(lines[i]):
            refs.append(match.group(1))
    return refs


def find_insertion_point(
    lines: List[str],
    ref_line: int,
    footnotes: List[FootnoteDefinition],
    figures: List[Figure]
) -> int:
    """
    Find the safe insertion point for a figure after its first reference.
    
    Rules:
    1. Find the paragraph containing the reference
    2. If footnote definitions follow the paragraph, insert after them
    3. Don't insert inside another figure block
    4. Insert after blank lines for clean formatting
    """
    para_start, para_end = find_paragraph_boundary(lines, ref_line)
    
    # Check for footnote references in this paragraph
    fn_refs = get_footnote_refs_in_range(lines, para_start, para_end)
    
    # Find the last footnote definition that:
    # 1. Is referenced in this paragraph
    # 2. Appears after the paragraph
    last_fn_end = para_end
    for fn in footnotes:
        if fn.fn_id in fn_refs and fn.start_line > para_end:
            last_fn_end = max(last_fn_end, fn.end_line)
    
    # Find insertion point after any footnotes
    insertion = last_fn_end + 1
    
    # Skip any blank lines to find the actual insertion point
    while insertion < len(lines) and lines[insertion].strip() == '':
        insertion += 1
    
    # Make sure we're not inserting inside a figure block
    for fig in figures:
        if fig.start_line <= insertion <= fig.end_line:
            insertion = fig.end_line + 1
    
    return insertion


def calculate_relocations(
    lines: List[str],
    figures: List[Figure],
    references: List[FigureReference],
    footnotes: List[FootnoteDefinition],
    max_distance: int = 50,  # Only relocate if figure is > N lines from reference
    fix_before_ref: bool = False  # Also fix figures that appear before their reference
) -> Dict[str, Tuple[int, int, str]]:
    """
    Calculate which figures should be relocated and where.
    
    Returns dict: fig_id -> (current_line, target_line, reason)
    """
    relocations = {}
    
    # Group references by figure
    refs_by_fig: Dict[str, List[int]] = {}
    for ref in references:
        if ref.fig_id not in refs_by_fig:
            refs_by_fig[ref.fig_id] = []
        refs_by_fig[ref.fig_id].append(ref.line_num)
    
    for fig in figures:
        if fig.fig_id not in refs_by_fig:
            continue  # Unreferenced figure
        
        first_ref_line = min(refs_by_fig[fig.fig_id])
        
        # Calculate ideal insertion point
        target_line = find_insertion_point(lines, first_ref_line, footnotes, figures)
        
        # Check if relocation is needed
        current_distance = fig.start_line - first_ref_line
        
        # Handle figures that appear before their first reference
        if current_distance < 0:
            if fix_before_ref:
                reason = f"RELOCATE: Move from line {fig.start_line + 1} to line {target_line + 1} (ref at line {first_ref_line + 1}, currently {abs(current_distance)} lines BEFORE)"
                relocations[fig.fig_id] = (fig.start_line, target_line, reason)
            else:
                reason = f"ISSUE: Figure appears {abs(current_distance)} lines BEFORE first reference (line {first_ref_line + 1})"
                relocations[fig.fig_id] = (fig.start_line, fig.start_line, reason)
            continue
        
        # Handle figures that are too far after their reference
        if current_distance <= max_distance:
            reason = f"OK: {current_distance} lines after reference (within {max_distance} threshold)"
            relocations[fig.fig_id] = (fig.start_line, fig.start_line, reason)
            continue
            
        # Check if target would actually move the figure closer
        if target_line >= fig.start_line:
            reason = f"SKIP: Target ({target_line + 1}) is at/after current position ({fig.start_line + 1})"
            relocations[fig.fig_id] = (fig.start_line, fig.start_line, reason)
            continue
        
        # This figure should be moved
        reason = f"RELOCATE: Move from line {fig.start_line + 1} to line {target_line + 1} (ref at line {first_ref_line + 1})"
        relocations[fig.fig_id] = (fig.start_line, target_line, reason)
    
    return relocations


def apply_relocations(
    lines: List[str],
    figures: List[Figure],
    relocations: Dict[str, Tuple[int, int, str]]
) -> List[str]:
    """Apply the calculated relocations to produce new document."""
    # Filter to only actual moves
    moves = {fid: (curr, target, reason) 
             for fid, (curr, target, reason) in relocations.items() 
             if curr != target and "RELOCATE" in reason}
    
    if not moves:
        return lines
    
    # Build figure lookup
    fig_by_id = {f.fig_id: f for f in figures}
    
    # Sort moves by current position (descending) to avoid index shifting issues
    sorted_moves = sorted(moves.items(), key=lambda x: -x[1][0])
    
    result = lines.copy()
    
    for fig_id, (curr_line, target_line, reason) in sorted_moves:
        fig = fig_by_id[fig_id]
        
        # Extract figure content (including blank line after for spacing)
        fig_content = result[fig.start_line:fig.end_line + 1]
        
        # Add blank line before and after for proper spacing
        fig_block = [''] + fig_content + ['']
        
        # Remove from current position
        del result[fig.start_line:fig.end_line + 1]
        
        # Adjust target if it was after the removed content
        adjusted_target = target_line
        if target_line > fig.start_line:
            adjusted_target -= fig.line_count
        
        # Insert at target position
        for i, line in enumerate(fig_block):
            result.insert(adjusted_target + i, line)
    
    # Clean up multiple consecutive blank lines
    cleaned = []
    prev_blank = False
    for line in result:
        is_blank = line.strip() == ''
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank
    
    return cleaned


def print_analysis(
    figures: List[Figure],
    references: List[FigureReference],
    footnotes: List[FootnoteDefinition],
    relocations: Dict[str, Tuple[int, int, str]],
    verbose: bool = False
):
    """Print analysis of figures and proposed relocations."""
    print(f"\n{'='*60}")
    print("FIGURE RELOCATION ANALYSIS")
    print(f"{'='*60}\n")
    
    print(f"Found {len(figures)} figures")
    print(f"Found {len(references)} figure references")
    print(f"Found {len(footnotes)} footnote definitions\n")
    
    # Group references by figure
    refs_by_fig: Dict[str, List[int]] = {}
    for ref in references:
        if ref.fig_id not in refs_by_fig:
            refs_by_fig[ref.fig_id] = []
        refs_by_fig[ref.fig_id].append(ref.line_num)
    
    moves_planned = 0
    issues_found = 0
    ok_count = 0
    unreferenced = 0
    
    for fig in figures:
        ref_lines = refs_by_fig.get(fig.fig_id, [])
        first_ref = min(ref_lines) + 1 if ref_lines else None
        
        fig_type_label = f"[{fig.fig_type}]" if verbose else ""
        
        if first_ref:
            print(f"\n{fig.fig_id} {fig_type_label}")
            print(f"  Defined at: line {fig.start_line + 1}-{fig.end_line + 1}")
            print(f"  First reference: line {first_ref}")
            distance = fig.start_line - (first_ref - 1)
            if distance < 0:
                print(f"  Distance: {abs(distance)} lines BEFORE reference")
            else:
                print(f"  Distance: {distance} lines after reference")
            
            if fig.fig_id in relocations:
                curr, target, reason = relocations[fig.fig_id]
                if "RELOCATE" in reason:
                    print(f"  ACTION: {reason}")
                    moves_planned += 1
                elif "ISSUE" in reason:
                    print(f"  {reason}")
                    issues_found += 1
                elif "OK" in reason and verbose:
                    print(f"  Status: {reason}")
                    ok_count += 1
                elif verbose:
                    print(f"  Status: {reason}")
        else:
            # Unreferenced figure - always show as issue
            print(f"\n{fig.fig_id} {fig_type_label}")
            print(f"  Defined at: line {fig.start_line + 1}-{fig.end_line + 1}")
            print(f"  ISSUE: Figure is NEVER REFERENCED in document!")
            unreferenced += 1
    
    total_issues = issues_found + unreferenced
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Figures OK:            {ok_count}")
    print(f"  Before reference:      {issues_found}")
    print(f"  UNREFERENCED:          {unreferenced}")
    print(f"  Will relocate:         {moves_planned}")
    print(f"  ─────────────────────────────")
    print(f"  Total issues:          {total_issues}")
    print(f"{'='*60}\n")
    
    return moves_planned, issues_found, unreferenced


def main():
    parser = argparse.ArgumentParser(
        description="Relocate figures closer to their first reference in Quarto documents"
    )
    parser.add_argument("file", help="Path to .qmd file")
    parser.add_argument("--apply", action="store_true", 
                        help="Apply changes (default is preview/dry-run)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed analysis including skipped figures")
    parser.add_argument("--max-distance", type=int, default=50,
                        help="Only relocate figures more than N lines from reference (default: 50)")
    parser.add_argument("--fix-before-ref", action="store_true",
                        help="Also relocate figures that appear BEFORE their first reference")
    parser.add_argument("--output", "-o", 
                        help="Output file (default: overwrite input when --apply)")
    
    args = parser.parse_args()
    
    # Read the file
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Parse the document
    figures = parse_figures(lines)
    references = parse_references(lines)
    footnotes = parse_footnotes(lines)
    
    # Calculate relocations
    relocations = calculate_relocations(
        lines, figures, references, footnotes, 
        max_distance=args.max_distance,
        fix_before_ref=args.fix_before_ref
    )
    
    # Print analysis
    moves_planned, issues_found, unreferenced = print_analysis(
        figures, references, footnotes, relocations, args.verbose
    )
    
    total_issues = issues_found + unreferenced
    
    if not args.apply:
        print("This was a DRY RUN. Use --apply to make changes.")
        if issues_found > 0 and not args.fix_before_ref:
            print(f"Note: {issues_found} figures appear before their reference.")
            print("      Use --fix-before-ref to also relocate those.")
        if unreferenced > 0:
            print(f"WARNING: {unreferenced} figures are NEVER REFERENCED!")
            print("         These figures should be referenced with @fig-<id> or removed.")
        # Exit with error code if issues found (useful for CI)
        if total_issues > 0:
            sys.exit(1)
        return
    
    if moves_planned == 0:
        print("No relocations needed.")
        if total_issues > 0:
            sys.exit(1)
        return
    
    # Apply relocations
    new_lines = apply_relocations(lines, figures, relocations)
    new_content = '\n'.join(new_lines)
    
    # Write output
    output_path = Path(args.output) if args.output else file_path
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Changes written to: {output_path}")


if __name__ == "__main__":
    main()
