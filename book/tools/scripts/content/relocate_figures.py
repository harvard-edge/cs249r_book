#!/usr/bin/env python3
"""
Element Relocator for Quarto Documents

Safely moves figures, tables, and listings closer to where they're first referenced,
respecting footnote definitions that must stay near their markers.

Usage:
    python relocate_figures.py <file.qmd>              # Preview mode (dry run)
    python relocate_figures.py <file.qmd> --apply     # Apply changes
    python relocate_figures.py <file.qmd> --verbose   # Show detailed analysis
    python relocate_figures.py <file.qmd> --type fig  # Only process figures
    python relocate_figures.py <file.qmd> --type tbl  # Only process tables
    python relocate_figures.py <file.qmd> --type lst  # Only process listings

Author: MLSysBook Team
"""

import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from pathlib import Path
import copy


@dataclass
class Element:
    """Represents a figure, table, or listing block in the document."""
    elem_id: str           # e.g., "fig-example", "tbl-data", "lst-code"
    start_line: int        # 0-indexed
    end_line: int          # 0-indexed, inclusive
    elem_type: str         # 'fig', 'tbl', 'lst'
    block_type: str        # 'div' for ::: blocks, 'markdown' for ![](...){}
                           # 'caption' for table captions, 'code' for lst-label
    colon_count: int = 0   # Number of colons (3 or 4) for div type
    
    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1
    
    @property
    def prefix(self) -> str:
        """Return the element prefix (fig, tbl, lst)."""
        return self.elem_id.split('-')[0]


@dataclass
class ElementReference:
    """Represents a reference to an element (@fig-xxx, @tbl-xxx, @lst-xxx)."""
    elem_id: str
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


def find_table_start(lines: List[str], caption_line: int) -> int:
    """
    Find the start of a grid/pipe table that ends with a caption.
    Tables are preceded by blank lines and consist of |---| style rows.
    """
    # Go backwards from caption to find the table start
    i = caption_line - 1
    
    # Skip any blank lines immediately before caption
    while i >= 0 and lines[i].strip() == '':
        i -= 1
    
    if i < 0:
        return caption_line
    
    # Check if this looks like a table row
    table_row_pattern = re.compile(r'^\s*\|.*\|\s*$')
    
    if not table_row_pattern.match(lines[i]):
        return caption_line  # No table found
    
    # Walk backwards to find the table start
    while i > 0:
        prev_line = lines[i - 1]
        if table_row_pattern.match(prev_line):
            i -= 1
        elif prev_line.strip() == '':
            break  # Found blank line before table
        else:
            break  # Found non-table content
    
    return i


def find_code_block_start(lines: List[str], lst_label_line: int) -> int:
    """
    Find the start of a code block containing #| lst-label.
    Code blocks start with ```{...}
    """
    i = lst_label_line - 1
    code_fence_pattern = re.compile(r'^```')
    
    while i >= 0:
        if code_fence_pattern.match(lines[i]):
            return i
        i -= 1
    
    return lst_label_line  # Fallback


def find_code_block_end(lines: List[str], start_line: int) -> int:
    """Find the closing ``` of a code block."""
    i = start_line + 1
    while i < len(lines):
        if lines[i].strip() == '```':
            return i
        i += 1
    return start_line  # Fallback


def parse_elements(lines: List[str], filter_types: Optional[Set[str]] = None) -> List[Element]:
    """
    Extract all figure, table, and listing blocks from the document.
    
    Handles:
    - Div-style: ::: {#fig-xxx}, ::: {#tbl-xxx}, :::: {#lst-xxx lst-cap="..."}
    - Markdown figures: ![caption](path){#fig-xxx}
    - Table captions: : Caption text {#tbl-xxx}
    - Code listings: #| lst-label: lst-xxx
    """
    elements = []
    if filter_types is None:
        filter_types = {'fig', 'tbl', 'lst'}
    
    # Pattern for div-style elements: ::: {#fig-xxx}, ::: {#tbl-xxx}, :::: {#lst-xxx}
    div_elem_pattern = re.compile(r'^(:{3,4})\s*\{#((?:fig|tbl|lst)-[^\s}]+)')
    
    # Pattern for markdown-style figures: ![caption](path){#fig-xxx}
    md_fig_pattern = re.compile(r'!\[.*?\]\(.*?\)\{#(fig-[^\s}]+)')
    
    # Pattern for table captions: : Caption text {#tbl-xxx}
    tbl_caption_pattern = re.compile(r'^:\s+.+\{#(tbl-[^\s}]+)')
    
    # Pattern for listing labels in code blocks: #| lst-label: lst-xxx
    lst_label_pattern = re.compile(r'^#\|\s*lst-label:\s*(lst-[\w-]+)')
    
    processed_lines: Set[int] = set()
    
    i = 0
    while i < len(lines):
        if i in processed_lines:
            i += 1
            continue
            
        line = lines[i]
        
        # Check for div-style element
        div_match = div_elem_pattern.match(line)
        if div_match:
            colons = div_match.group(1)
            elem_id = div_match.group(2)
            elem_type = elem_id.split('-')[0]
            
            if elem_type in filter_types:
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
                
                elements.append(Element(
                    elem_id=elem_id,
                    start_line=start_line,
                    end_line=j,
                    elem_type=elem_type,
                    block_type='div',
                    colon_count=colon_count
                ))
                for k in range(start_line, j + 1):
                    processed_lines.add(k)
                i = j + 1
                continue
        
        # Check for markdown-style figure
        if 'fig' in filter_types:
            md_match = md_fig_pattern.search(line)
            if md_match:
                elem_id = md_match.group(1)
                elements.append(Element(
                    elem_id=elem_id,
                    start_line=i,
                    end_line=i,
                    elem_type='fig',
                    block_type='markdown'
                ))
                processed_lines.add(i)
                i += 1
                continue
        
        # Check for table caption (: Caption {#tbl-xxx})
        if 'tbl' in filter_types:
            tbl_match = tbl_caption_pattern.match(line)
            if tbl_match:
                elem_id = tbl_match.group(1)
                # Find the start of the table (goes backwards)
                table_start = find_table_start(lines, i)
                
                elements.append(Element(
                    elem_id=elem_id,
                    start_line=table_start,
                    end_line=i,
                    elem_type='tbl',
                    block_type='caption'
                ))
                for k in range(table_start, i + 1):
                    processed_lines.add(k)
                i += 1
                continue
        
        # Check for listing label (#| lst-label: lst-xxx)
        if 'lst' in filter_types:
            lst_match = lst_label_pattern.match(line)
            if lst_match:
                elem_id = lst_match.group(1)
                # Find the code block boundaries
                code_start = find_code_block_start(lines, i)
                code_end = find_code_block_end(lines, code_start)
                
                elements.append(Element(
                    elem_id=elem_id,
                    start_line=code_start,
                    end_line=code_end,
                    elem_type='lst',
                    block_type='code'
                ))
                for k in range(code_start, code_end + 1):
                    processed_lines.add(k)
                i = code_end + 1
                continue
        
        i += 1
    
    return elements


def parse_references(lines: List[str], filter_types: Optional[Set[str]] = None) -> List[ElementReference]:
    """Extract all element references from the document."""
    references = []
    if filter_types is None:
        filter_types = {'fig', 'tbl', 'lst'}
    
    # Build pattern based on filter types
    type_pattern = '|'.join(filter_types)
    ref_pattern = re.compile(rf'@((?:{type_pattern})-[a-zA-Z0-9_-]+)')
    
    # Patterns to skip (element definitions)
    skip_patterns = [
        re.compile(r'^:{3,4}\s*\{#(?:fig|tbl|lst)-'),  # Div definitions
        re.compile(r'\{#(?:fig|tbl|lst)-[^\s}]+\}'),   # Attribute definitions
        re.compile(r'^#\|\s*lst-label:'),              # Listing labels
    ]
    
    for i, line in enumerate(lines):
        # Skip lines that are element definitions
        skip = False
        for pattern in skip_patterns:
            if pattern.search(line):
                skip = True
                break
        
        if skip:
            continue
            
        for match in ref_pattern.finditer(line):
            elem_id = match.group(1)
            elem_type = elem_id.split('-')[0]
            if elem_type in filter_types:
                references.append(ElementReference(
                    elem_id=elem_id,
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
    elements: List[Element]
) -> int:
    """
    Find the safe insertion point for an element after its first reference.
    
    Rules:
    1. Find the paragraph containing the reference
    2. If footnote definitions follow the paragraph, insert after them
    3. Don't insert inside another element block
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
    
    # Make sure we're not inserting inside an element block
    for elem in elements:
        if elem.start_line <= insertion <= elem.end_line:
            insertion = elem.end_line + 1
    
    return insertion


def calculate_relocations(
    lines: List[str],
    elements: List[Element],
    references: List[ElementReference],
    footnotes: List[FootnoteDefinition],
    max_distance: int = 50,  # Only relocate if element is > N lines from reference
    fix_before_ref: bool = False  # Also fix elements that appear before their reference
) -> Dict[str, Tuple[int, int, str]]:
    """
    Calculate which elements should be relocated and where.
    
    Returns dict: elem_id -> (current_line, target_line, reason)
    """
    relocations = {}
    
    # Group references by element
    refs_by_elem: Dict[str, List[int]] = {}
    for ref in references:
        if ref.elem_id not in refs_by_elem:
            refs_by_elem[ref.elem_id] = []
        refs_by_elem[ref.elem_id].append(ref.line_num)
    
    for elem in elements:
        if elem.elem_id not in refs_by_elem:
            continue  # Unreferenced element
        
        first_ref_line = min(refs_by_elem[elem.elem_id])
        
        # Calculate ideal insertion point
        target_line = find_insertion_point(lines, first_ref_line, footnotes, elements)
        
        # Check if relocation is needed
        current_distance = elem.start_line - first_ref_line
        
        # Handle elements that appear before their first reference
        if current_distance < 0:
            if fix_before_ref:
                reason = f"RELOCATE: Move from line {elem.start_line + 1} to line {target_line + 1} (ref at line {first_ref_line + 1}, currently {abs(current_distance)} lines BEFORE)"
                relocations[elem.elem_id] = (elem.start_line, target_line, reason)
            else:
                reason = f"ISSUE: Element appears {abs(current_distance)} lines BEFORE first reference (line {first_ref_line + 1})"
                relocations[elem.elem_id] = (elem.start_line, elem.start_line, reason)
            continue
        
        # Handle elements that are too far after their reference
        if current_distance <= max_distance:
            reason = f"OK: {current_distance} lines after reference (within {max_distance} threshold)"
            relocations[elem.elem_id] = (elem.start_line, elem.start_line, reason)
            continue
            
        # Check if target would actually move the element closer
        if target_line >= elem.start_line:
            reason = f"SKIP: Target ({target_line + 1}) is at/after current position ({elem.start_line + 1})"
            relocations[elem.elem_id] = (elem.start_line, elem.start_line, reason)
            continue
        
        # This element should be moved
        reason = f"RELOCATE: Move from line {elem.start_line + 1} to line {target_line + 1} (ref at line {first_ref_line + 1})"
        relocations[elem.elem_id] = (elem.start_line, target_line, reason)
    
    return relocations


def apply_relocations(
    lines: List[str],
    elements: List[Element],
    relocations: Dict[str, Tuple[int, int, str]]
) -> List[str]:
    """Apply the calculated relocations to produce new document."""
    # Filter to only actual moves
    moves = {eid: (curr, target, reason) 
             for eid, (curr, target, reason) in relocations.items() 
             if curr != target and "RELOCATE" in reason}
    
    if not moves:
        return lines
    
    # Build element lookup
    elem_by_id = {e.elem_id: e for e in elements}
    
    # Sort moves by current position (descending) to avoid index shifting issues
    sorted_moves = sorted(moves.items(), key=lambda x: -x[1][0])
    
    result = lines.copy()
    
    for elem_id, (curr_line, target_line, reason) in sorted_moves:
        elem = elem_by_id[elem_id]
        
        # Extract element content (including blank line after for spacing)
        elem_content = result[elem.start_line:elem.end_line + 1]
        
        # Add blank line before and after for proper spacing
        elem_block = [''] + elem_content + ['']
        
        # Remove from current position
        del result[elem.start_line:elem.end_line + 1]
        
        # Adjust target if it was after the removed content
        adjusted_target = target_line
        if target_line > elem.start_line:
            adjusted_target -= elem.line_count
        
        # Insert at target position
        for i, line in enumerate(elem_block):
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
    elements: List[Element],
    references: List[ElementReference],
    footnotes: List[FootnoteDefinition],
    relocations: Dict[str, Tuple[int, int, str]],
    verbose: bool = False
):
    """Print analysis of elements and proposed relocations."""
    print(f"\n{'='*60}")
    print("ELEMENT RELOCATION ANALYSIS")
    print(f"{'='*60}\n")
    
    # Count by type
    fig_count = sum(1 for e in elements if e.elem_type == 'fig')
    tbl_count = sum(1 for e in elements if e.elem_type == 'tbl')
    lst_count = sum(1 for e in elements if e.elem_type == 'lst')
    
    print(f"Found {len(elements)} elements ({fig_count} figures, {tbl_count} tables, {lst_count} listings)")
    print(f"Found {len(references)} references")
    print(f"Found {len(footnotes)} footnote definitions\n")
    
    # Group references by element
    refs_by_elem: Dict[str, List[int]] = {}
    for ref in references:
        if ref.elem_id not in refs_by_elem:
            refs_by_elem[ref.elem_id] = []
        refs_by_elem[ref.elem_id].append(ref.line_num)
    
    moves_planned = 0
    issues_found = 0
    ok_count = 0
    unreferenced = 0
    
    # Sort elements by type then by line number
    sorted_elements = sorted(elements, key=lambda e: (e.elem_type, e.start_line))
    
    current_type = None
    for elem in sorted_elements:
        # Print type header
        if elem.elem_type != current_type:
            current_type = elem.elem_type
            type_name = {'fig': 'FIGURES', 'tbl': 'TABLES', 'lst': 'LISTINGS'}[current_type]
            print(f"\n--- {type_name} ---")
        
        ref_lines = refs_by_elem.get(elem.elem_id, [])
        first_ref = min(ref_lines) + 1 if ref_lines else None
        
        type_label = f"[{elem.block_type}]" if verbose else ""
        
        if first_ref:
            print(f"\n{elem.elem_id} {type_label}")
            print(f"  Defined at: line {elem.start_line + 1}-{elem.end_line + 1}")
            print(f"  First reference: line {first_ref}")
            distance = elem.start_line - (first_ref - 1)
            if distance < 0:
                print(f"  Distance: {abs(distance)} lines BEFORE reference")
            else:
                print(f"  Distance: {distance} lines after reference")
            
            if elem.elem_id in relocations:
                curr, target, reason = relocations[elem.elem_id]
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
                    ok_count += 1
        else:
            # Unreferenced element - always show as issue
            print(f"\n{elem.elem_id} {type_label}")
            print(f"  Defined at: line {elem.start_line + 1}-{elem.end_line + 1}")
            print(f"  ISSUE: Element is NEVER REFERENCED in document!")
            unreferenced += 1
    
    total_issues = issues_found + unreferenced
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Elements OK:           {ok_count}")
    print(f"  Before reference:      {issues_found}")
    print(f"  UNREFERENCED:          {unreferenced}")
    print(f"  Will relocate:         {moves_planned}")
    print(f"  ─────────────────────────────")
    print(f"  Total issues:          {total_issues}")
    print(f"{'='*60}\n")
    
    return moves_planned, issues_found, unreferenced


def main():
    parser = argparse.ArgumentParser(
        description="Relocate figures, tables, and listings closer to their first reference in Quarto documents"
    )
    parser.add_argument("file", help="Path to .qmd file")
    parser.add_argument("--apply", action="store_true", 
                        help="Apply changes (default is preview/dry-run)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed analysis including skipped elements")
    parser.add_argument("--max-distance", type=int, default=50,
                        help="Only relocate elements more than N lines from reference (default: 50)")
    parser.add_argument("--fix-before-ref", action="store_true",
                        help="Also relocate elements that appear BEFORE their first reference")
    parser.add_argument("--type", "-t", choices=['fig', 'tbl', 'lst', 'all'], default='all',
                        help="Type of elements to process (default: all)")
    parser.add_argument("--output", "-o", 
                        help="Output file (default: overwrite input when --apply)")
    
    args = parser.parse_args()
    
    # Determine which types to filter
    if args.type == 'all':
        filter_types = {'fig', 'tbl', 'lst'}
    else:
        filter_types = {args.type}
    
    # Read the file
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Parse the document
    elements = parse_elements(lines, filter_types)
    references = parse_references(lines, filter_types)
    footnotes = parse_footnotes(lines)
    
    # Calculate relocations
    relocations = calculate_relocations(
        lines, elements, references, footnotes, 
        max_distance=args.max_distance,
        fix_before_ref=args.fix_before_ref
    )
    
    # Print analysis
    moves_planned, issues_found, unreferenced = print_analysis(
        elements, references, footnotes, relocations, args.verbose
    )
    
    total_issues = issues_found + unreferenced
    
    if not args.apply:
        print("This was a DRY RUN. Use --apply to make changes.")
        if issues_found > 0 and not args.fix_before_ref:
            print(f"Note: {issues_found} elements appear before their reference.")
            print("      Use --fix-before-ref to also relocate those.")
        if unreferenced > 0:
            print(f"WARNING: {unreferenced} elements are NEVER REFERENCED!")
            print("         These should be referenced with @fig-<id>, @tbl-<id>, or @lst-<id>, or removed.")
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
    new_lines = apply_relocations(lines, elements, relocations)
    new_content = '\n'.join(new_lines)
    
    # Write output
    output_path = Path(args.output) if args.output else file_path
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Changes written to: {output_path}")


if __name__ == "__main__":
    main()
