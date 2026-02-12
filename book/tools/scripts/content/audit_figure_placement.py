#!/usr/bin/env python3
"""
Audit and optionally fix figure placement in QMD files.

This script identifies figures that are defined far from their first reference
and can optionally relocate them closer to where they're first mentioned.

Usage:
    # Audit only (safe, no changes)
    python audit_figure_placement.py book/quarto/contents/
    
    # Audit a single file
    python audit_figure_placement.py book/quarto/contents/vol1/ml_systems/ml_systems.qmd
    
    # Show detailed analysis
    python audit_figure_placement.py --verbose book/quarto/contents/
    
    # Preview changes (dry run)
    python audit_figure_placement.py --fix --dry-run book/quarto/contents/vol1/ml_systems/ml_systems.qmd
    
    # Apply fixes
    python audit_figure_placement.py --fix book/quarto/contents/vol1/ml_systems/ml_systems.qmd

The script handles two figure syntax types:
1. Markdown images: ![caption](image.png){#fig-label}
2. Quarto div blocks: ::: {#fig-label ...} ... :::
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class FigureType(Enum):
    MARKDOWN_IMAGE = "markdown"
    QUARTO_DIV = "quarto_div"


@dataclass
class FigureDefinition:
    """Represents a figure definition in a QMD file."""
    label: str
    fig_type: FigureType
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed (same as start for single-line figures)
    content: str     # The full figure content including all lines


@dataclass
class FigureReference:
    """Represents a reference to a figure."""
    label: str
    line: int  # 1-indexed


@dataclass
class PlacementIssue:
    """Represents a figure placement issue."""
    figure: FigureDefinition
    first_ref_line: int
    distance: int  # Positive = definition after reference, negative = before
    

# Configuration
DEFAULT_THRESHOLD = 15  # Lines between definition and first reference before flagging
CHAPTERS = [
    "book/quarto/contents/vol1/introduction/introduction.qmd",
    "book/quarto/contents/vol1/ml_systems/ml_systems.qmd",
    "book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd",
    "book/quarto/contents/vol1/data_engineering/data_engineering.qmd",
    "book/quarto/contents/vol1/nn_computation/nn_computation.qmd",
    "book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd",
    "book/quarto/contents/vol1/frameworks/frameworks.qmd",
    "book/quarto/contents/vol1/training/training.qmd",
    "book/quarto/contents/vol1/optimizations/model_compression.qmd",
    "book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd",
    "book/quarto/contents/vol1/data_efficiency/data_efficiency.qmd",
    "book/quarto/contents/vol1/benchmarking/benchmarking.qmd",
    "book/quarto/contents/vol1/model_serving/model_serving.qmd",
    "book/quarto/contents/vol1/ml_ops/ml_ops.qmd",
    "book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd",
    "book/quarto/contents/vol1/conclusion/conclusion.qmd",
    "book/quarto/contents/vol2/introduction/introduction.qmd",
    "book/quarto/contents/vol2/infrastructure/infrastructure.qmd",
    "book/quarto/contents/vol2/infrastructure/networking.qmd",
    "book/quarto/contents/vol2/communication_ops/communication_ops.qmd",
    "book/quarto/contents/vol2/storage/storage.qmd",
    "book/quarto/contents/vol2/distributed_training/distributed_training.qmd",
    "book/quarto/contents/vol2/optimization/optimization.qmd",
    "book/quarto/contents/vol2/inference/inference.qmd",
    "book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd",
    "book/quarto/contents/vol2/ops_scale/ops_scale.qmd",
    "book/quarto/contents/vol2/security_privacy/security_privacy.qmd",
    "book/quarto/contents/vol2/robust_ai/robust_ai.qmd",
    "book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd",
    "book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd",
]


def find_figure_definitions(lines: list[str]) -> list[FigureDefinition]:
    """
    Find all figure definitions in the document.
    
    Handles:
    1. Markdown images: ![caption](image.png){#fig-label}
    2. Quarto div blocks: ::: {#fig-label ...} ... :::
    """
    figures = []
    
    # Pattern for markdown image with figure label
    # Matches: ![...](...)  {#fig-...}  with optional attributes
    markdown_pattern = re.compile(r'!\[.*?\]\(.*?\)\s*\{[^}]*#(fig-[a-zA-Z0-9-]+)[^}]*\}')
    
    # Pattern for Quarto div opening with figure label
    div_open_pattern = re.compile(r'^:::\s*\{[^}]*#(fig-[a-zA-Z0-9-]+)[^}]*\}')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_num = i + 1  # 1-indexed
        
        # Check for markdown image figure
        match = markdown_pattern.search(line)
        if match:
            label = match.group(1)
            figures.append(FigureDefinition(
                label=label,
                fig_type=FigureType.MARKDOWN_IMAGE,
                start_line=line_num,
                end_line=line_num,
                content=line
            ))
            i += 1
            continue
        
        # Check for Quarto div figure
        match = div_open_pattern.match(line)
        if match:
            label = match.group(1)
            start_line = line_num
            # Find the closing :::
            nesting = 1
            j = i + 1
            while j < len(lines) and nesting > 0:
                if lines[j].strip().startswith(':::'):
                    if re.match(r'^:::\s*\{', lines[j].strip()):
                        nesting += 1
                    else:
                        nesting -= 1
                j += 1
            end_line = j  # 1-indexed
            content = '\n'.join(lines[i:j])
            figures.append(FigureDefinition(
                label=label,
                fig_type=FigureType.QUARTO_DIV,
                start_line=start_line,
                end_line=end_line,
                content=content
            ))
            i = j
            continue
        
        i += 1
    
    return figures


def find_figure_references(lines: list[str]) -> list[FigureReference]:
    """Find all figure references (@fig-label) in the document."""
    references = []
    
    # Pattern for figure references
    ref_pattern = re.compile(r'@(fig-[a-zA-Z0-9-]+)')
    
    for i, line in enumerate(lines):
        line_num = i + 1  # 1-indexed
        for match in ref_pattern.finditer(line):
            label = match.group(1)
            references.append(FigureReference(label=label, line=line_num))
    
    return references


def analyze_placement(
    figures: list[FigureDefinition],
    references: list[FigureReference],
    threshold: int = DEFAULT_THRESHOLD
) -> list[PlacementIssue]:
    """
    Analyze figure placement and identify issues.
    
    Returns issues where a figure is defined more than `threshold` lines
    away from its first reference.
    """
    issues = []
    
    # Build a map of label -> first reference line
    first_ref_map: dict[str, int] = {}
    for ref in references:
        if ref.label not in first_ref_map:
            first_ref_map[ref.label] = ref.line
        else:
            first_ref_map[ref.label] = min(first_ref_map[ref.label], ref.line)
    
    for fig in figures:
        if fig.label not in first_ref_map:
            # Figure is defined but never referenced - different issue
            continue
        
        first_ref_line = first_ref_map[fig.label]
        
        # Calculate distance: positive = definition after reference
        # For multi-line figures, use the start line
        distance = fig.start_line - first_ref_line
        
        if abs(distance) > threshold:
            issues.append(PlacementIssue(
                figure=fig,
                first_ref_line=first_ref_line,
                distance=distance
            ))
    
    return issues


def find_ideal_insertion_point(
    lines: list[str],
    ref_line: int,
    fig: FigureDefinition
) -> tuple[int, str]:
    """
    Find the ideal insertion point for a figure near its first reference.
    
    Strategy:
    - Find the paragraph containing the reference
    - Skip over footnotes that follow the paragraph
    - Insert after any callout-definition that follows
    - Respect section boundaries (don't cross ## headers)
    
    Returns:
        (line_number, reason) - 1-indexed line number and explanation
    """
    ref_idx = ref_line - 1  # Convert to 0-indexed
    reason = "after paragraph"
    
    # Find the end of the paragraph containing the reference
    # A paragraph ends with a blank line
    insert_idx = ref_idx + 1
    while insert_idx < len(lines):
        line = lines[insert_idx].strip()
        
        # Stop at blank line (end of paragraph)
        if line == '':
            break
        
        # Don't cross section headers
        if line.startswith('##'):
            return insert_idx + 1, "before section header"
        
        # Don't place inside callouts, code blocks, or other figures
        if line.startswith(':::') or line.startswith('```'):
            break
        
        insert_idx += 1
    
    # Now we're at a blank line (or end of paragraph content)
    # Skip over any footnotes that follow
    footnote_pattern = re.compile(r'^\[\^[^\]]+\]:')
    
    while insert_idx < len(lines):
        line = lines[insert_idx].strip()
        
        # Skip blank lines
        if line == '':
            insert_idx += 1
            continue
        
        # Skip footnote definitions (they can be multi-line)
        if footnote_pattern.match(line):
            reason = "after footnotes"
            # Skip the footnote (may span multiple lines)
            insert_idx += 1
            while insert_idx < len(lines):
                next_line = lines[insert_idx].strip()
                # Footnote continues if indented or empty
                if next_line == '' or lines[insert_idx].startswith('    ') or lines[insert_idx].startswith('\t'):
                    insert_idx += 1
                    continue
                # Check if another footnote follows
                if footnote_pattern.match(next_line):
                    break
                # Otherwise, end of footnotes
                break
            continue
        
        # Check for callout-definition (should stay with the text it defines)
        if line.startswith('::: {.callout-definition'):
            reason = "after definition callout"
            # Skip the entire callout block
            nesting = 1
            insert_idx += 1
            while insert_idx < len(lines) and nesting > 0:
                callout_line = lines[insert_idx].strip()
                if callout_line.startswith(':::'):
                    if re.match(r'^:::\s*\{', callout_line):
                        nesting += 1
                    else:
                        nesting -= 1
                insert_idx += 1
            continue
        
        # Don't cross section headers
        if line.startswith('##'):
            return insert_idx + 1, "before section header"
        
        # Found content that's not a footnote or definition callout
        break
    
    return insert_idx + 1, reason  # Convert back to 1-indexed


def relocate_figure(
    lines: list[str],
    fig: FigureDefinition,
    new_position: int  # 1-indexed line number for insertion
) -> list[str]:
    """
    Relocate a figure to a new position in the document.
    
    Returns a new list of lines with the figure moved.
    """
    # Remove the figure from its current position
    start_idx = fig.start_line - 1  # 0-indexed
    end_idx = fig.end_line  # 0-indexed, exclusive
    
    # Get the figure content (might be multiple lines)
    figure_lines = lines[start_idx:end_idx]
    
    # Remove trailing/leading blank lines from around the figure
    # to avoid leaving gaps
    new_lines = lines[:start_idx]
    
    # Remove blank line before if exists
    while new_lines and new_lines[-1].strip() == '':
        new_lines.pop()
    
    # Skip to after figure
    remaining_start = end_idx
    while remaining_start < len(lines) and lines[remaining_start].strip() == '':
        remaining_start += 1
    
    new_lines.extend(lines[remaining_start:])
    
    # Adjust insertion position if it was after the removed figure
    adjusted_position = new_position - 1  # 0-indexed
    if new_position > fig.end_line:
        # Account for removed lines
        removed_count = end_idx - start_idx
        # Also account for blank lines we removed
        blank_before = fig.start_line - 1 - len(lines[:start_idx])
        blank_after = remaining_start - end_idx
        adjusted_position -= removed_count + blank_before + blank_after
    
    # Ensure we don't go past the end
    adjusted_position = min(adjusted_position, len(new_lines))
    
    # Insert with proper spacing
    result = new_lines[:adjusted_position]
    
    # Add blank line before if needed
    if result and result[-1].strip() != '':
        result.append('\n')
    
    result.extend(figure_lines)
    
    # Add blank line after
    result.append('\n')
    
    result.extend(new_lines[adjusted_position:])
    
    return result


def audit_file(
    filepath: str,
    threshold: int = DEFAULT_THRESHOLD,
    verbose: bool = False
) -> tuple[list[PlacementIssue], list[FigureDefinition], list[FigureReference]]:
    """
    Audit a single file for figure placement issues.
    
    Returns (issues, all_figures, all_references).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    figures = find_figure_definitions(lines)
    references = find_figure_references(lines)
    issues = analyze_placement(figures, references, threshold)
    
    return issues, figures, references


def print_audit_report(
    filepath: str,
    issues: list[PlacementIssue],
    figures: list[FigureDefinition],
    references: list[FigureReference],
    verbose: bool = False
):
    """Print an audit report for a file."""
    rel_path = os.path.relpath(filepath)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"File: {rel_path}")
        print(f"{'='*70}")
        print(f"  Total figures: {len(figures)}")
        print(f"  Total references: {len(references)}")
        print(f"  Placement issues: {len(issues)}")
    
    if not issues:
        if verbose:
            print("  ✅ All figures are well-placed")
        return
    
    if not verbose:
        print(f"\n{rel_path}")
    
    for issue in sorted(issues, key=lambda x: abs(x.distance), reverse=True):
        fig = issue.figure
        direction = "AFTER" if issue.distance > 0 else "BEFORE"
        
        print(f"\n  ⚠️  {fig.label}")
        print(f"      Definition: line {fig.start_line}" + 
              (f"-{fig.end_line}" if fig.end_line != fig.start_line else ""))
        print(f"      First reference: line {issue.first_ref_line}")
        print(f"      Distance: {abs(issue.distance)} lines ({direction} reference)")
        print(f"      Type: {fig.fig_type.value}")
        
        if verbose and fig.fig_type == FigureType.MARKDOWN_IMAGE:
            # Show truncated caption for context
            caption_match = re.search(r'!\[(.*?)\]', fig.content)
            if caption_match:
                caption = caption_match.group(1)[:60]
                if len(caption_match.group(1)) > 60:
                    caption += "..."
                print(f"      Caption: {caption}")


def suggest_placement(
    filepath: str,
    issues: list[PlacementIssue],
    verbose: bool = False
) -> None:
    """
    Show suggested new locations for figures without making changes.
    """
    if not issues:
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    print(f"\n  📋 SUGGESTED RELOCATIONS:")
    
    for issue in sorted(issues, key=lambda x: abs(x.distance), reverse=True):
        fig = issue.figure
        
        # Find ideal insertion point
        ideal_pos, reason = find_ideal_insertion_point(lines, issue.first_ref_line, fig)
        
        current_distance = abs(issue.distance)
        new_distance = abs(ideal_pos - issue.first_ref_line)
        
        if new_distance >= current_distance:
            print(f"\n  {fig.label}:")
            print(f"      ⚠️  Cannot improve: current placement is optimal given constraints")
            continue
        
        print(f"\n  {fig.label}:")
        print(f"      Current: line {fig.start_line} ({current_distance} lines from reference)")
        print(f"      Suggest: line {ideal_pos} ({reason}, {new_distance} lines from reference)")
        
        # Show context
        start = max(0, ideal_pos - 3)
        end = min(len(lines), ideal_pos + 2)
        print(f"      Context:")
        for i in range(start, end):
            marker = ">>>" if i + 1 == ideal_pos else "   "
            line_preview = lines[i][:65] + ("..." if len(lines[i]) > 65 else "")
            print(f"        {marker} {i+1}: {line_preview}")


def fix_file(
    filepath: str,
    issues: list[PlacementIssue],
    dry_run: bool = True,
    verbose: bool = False,
    interactive: bool = False
) -> bool:
    """
    Fix figure placement issues in a file.
    
    Returns True if changes were made (or would be made in dry_run).
    """
    if not issues:
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    rel_path = os.path.relpath(filepath)
    
    # Sort issues by figure position (process from end to start to avoid index shifts)
    sorted_issues = sorted(issues, key=lambda x: x.figure.start_line, reverse=True)
    
    changes_made = []
    
    for issue in sorted_issues:
        fig = issue.figure
        
        # Find ideal insertion point
        ideal_pos, reason = find_ideal_insertion_point(lines, issue.first_ref_line, fig)
        
        # Only move if it's actually closer
        current_distance = abs(issue.distance)
        new_distance = abs(ideal_pos - issue.first_ref_line)
        
        if new_distance >= current_distance:
            if verbose:
                print(f"  ℹ️  Skipping {fig.label}: moving would not improve placement")
            continue
        
        change_desc = (
            f"Move {fig.label} from line {fig.start_line} "
            f"to line {ideal_pos} ({reason}) "
            f"(distance: {current_distance} → {new_distance} lines)"
        )
        
        if dry_run:
            print(f"  📝 Would: {change_desc}")
            # Show context around insertion point
            if verbose:
                start = max(0, ideal_pos - 3)
                end = min(len(lines), ideal_pos + 2)
                print(f"      Context around line {ideal_pos}:")
                for i in range(start, end):
                    marker = ">>>" if i + 1 == ideal_pos else "   "
                    line_preview = lines[i][:60] + ("..." if len(lines[i]) > 60 else "")
                    print(f"        {marker} {i+1}: {line_preview}")
            changes_made.append(change_desc)
            continue
        
        if interactive:
            print(f"\n  🔍 {change_desc}")
            # Show context around insertion point
            start = max(0, ideal_pos - 4)
            end = min(len(lines), ideal_pos + 3)
            print(f"      Context around line {ideal_pos}:")
            for i in range(start, end):
                marker = ">>>" if i + 1 == ideal_pos else "   "
                line_preview = lines[i][:60] + ("..." if len(lines[i]) > 60 else "")
                print(f"        {marker} {i+1}: {line_preview}")
            
            response = input("      Apply this change? [y/N/q]: ").lower()
            if response == 'q':
                print("  Aborted.")
                break
            if response != 'y':
                print(f"      Skipped {fig.label}")
                continue
        
        changes_made.append(change_desc)
        lines = relocate_figure(lines, fig, ideal_pos)
        
        # Re-parse figures for accurate positions on next iteration
        figures = find_figure_definitions(lines)
        references = find_figure_references(lines)
        # Update remaining issues' figure references
        for remaining_issue in sorted_issues:
            for f in figures:
                if f.label == remaining_issue.figure.label:
                    remaining_issue.figure = f
                    break
    
    if not changes_made:
        if verbose:
            print(f"  ℹ️  No beneficial moves found for {rel_path}")
        return False
    
    if not dry_run and changes_made:
        # Write the modified content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  ✅ Applied {len(changes_made)} changes to {rel_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Audit and fix figure placement in QMD files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'path',
        nargs='?',
        help="File or directory to audit. If not provided, audits all known chapters."
    )
    parser.add_argument(
        '--threshold', '-t',
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Distance threshold (in lines) before flagging an issue. Default: {DEFAULT_THRESHOLD}"
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help="Attempt to fix placement issues (requires --dry-run or explicit confirmation)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what changes would be made without modifying files"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Show detailed analysis"
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help="Show only summary statistics"
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help="Interactively confirm each change (requires --fix)"
    )
    parser.add_argument(
        '--suggest',
        action='store_true',
        help="Show suggested new locations for each issue without making changes"
    )
    
    args = parser.parse_args()
    
    # Determine files to process
    files_to_process = []
    
    if args.path:
        path = Path(args.path)
        if path.is_file():
            files_to_process = [str(path)]
        elif path.is_dir():
            files_to_process = [
                str(p) for p in path.rglob('*.qmd')
            ]
        else:
            print(f"Error: Path not found: {args.path}")
            sys.exit(1)
    else:
        # Use default chapter list
        files_to_process = [f for f in CHAPTERS if os.path.exists(f)]
    
    if not files_to_process:
        print("No QMD files found to process.")
        sys.exit(1)
    
    # Validate flag combinations
    if args.interactive and not args.fix:
        print("Error: --interactive requires --fix")
        sys.exit(1)
    
    if args.interactive and args.dry_run:
        print("Error: --interactive and --dry-run cannot be used together")
        sys.exit(1)
    
    # Safety check for --fix without --dry-run (unless interactive)
    if args.fix and not args.dry_run and not args.interactive:
        print("\n⚠️  WARNING: This will modify files!")
        print("Files to be processed:")
        for f in files_to_process:
            print(f"  - {os.path.relpath(f)}")
        print("\n💡 Tip: Use --interactive (-i) to review each change individually")
        response = input("\nProceed with all changes? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Process files
    total_issues = 0
    total_figures = 0
    files_with_issues = 0
    
    print(f"\n🔍 Auditing figure placement (threshold: {args.threshold} lines)")
    print(f"   Processing {len(files_to_process)} file(s)...")
    
    for filepath in files_to_process:
        issues, figures, references = audit_file(filepath, args.threshold, args.verbose)
        
        total_figures += len(figures)
        total_issues += len(issues)
        if issues:
            files_with_issues += 1
        
        if not args.summary:
            print_audit_report(filepath, issues, figures, references, args.verbose)
        
        if args.suggest and issues:
            suggest_placement(filepath, issues, args.verbose)
        elif args.fix and issues:
            fix_file(
                filepath, 
                issues, 
                dry_run=args.dry_run, 
                verbose=args.verbose,
                interactive=args.interactive
            )
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Files processed: {len(files_to_process)}")
    print(f"  Total figures: {total_figures}")
    print(f"  Placement issues: {total_issues}")
    print(f"  Files with issues: {files_with_issues}")
    
    if total_issues > 0 and not args.fix and not args.suggest:
        print(f"\n💡 Options:")
        print(f"   --suggest              Show suggested relocations for manual review")
        print(f"   --fix --dry-run        Preview automatic changes")
        print(f"   --fix --interactive    Apply changes one at a time with confirmation")
    
    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
