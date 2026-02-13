#!/usr/bin/env python3
"""
Check Figure Completeness — Pre-commit Validation

Scans all .qmd files and verifies that every figure has:
  1. A label (fig-ID)
  2. A caption (fig-cap)
  3. Alt-text (fig-alt)

Supports all three Quarto figure syntaxes:
  - Div-based:    ::: {#fig-id fig-cap="..." fig-alt="..."}
  - Markdown img: ![...](path){#fig-id fig-cap="..." fig-alt="..."}
  - Code-cell:    #| label: fig-id / #| fig-cap: ... / #| fig-alt: ...

Usage:
    python3 check_figure_completeness.py -d book/quarto/contents/
    python3 check_figure_completeness.py path/to/chapter.qmd
    python3 check_figure_completeness.py -d book/quarto/contents/ --strict

Options:
    -d DIR          Scan all .qmd files in DIR recursively
    --strict        Exit with non-zero code if any issues found
    --quiet         Only show issues (skip OK figures)
    --alt-only      Only check for missing alt-text
    --cap-only      Only check for missing captions
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FigureInfo:
    """Represents a figure found in a .qmd file."""
    fig_id: str
    line_num: int
    file_path: Path
    has_caption: bool
    has_alt_text: bool
    source_type: str  # 'div', 'markdown', or 'code-cell'

    @property
    def is_complete(self) -> bool:
        return self.has_caption and self.has_alt_text


def _clean_yaml_string(value: str) -> str:
    """Strip surrounding quotes from a YAML-style cell option value."""
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value.strip()


def scan_file(filepath: Path) -> list[FigureInfo]:
    """
    Scan a single .qmd file for all figure definitions across all syntaxes.

    Returns a list of FigureInfo objects.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    figures = []
    seen_ids: dict[str, int] = {}  # fig_id -> index in figures list

    # --- Pass 1: Attribute-based figures (div and markdown) ---
    # Extract just the figure ID from the attribute block, then check the
    # FULL LINE for fig-cap= and fig-alt= presence. This avoids the problem
    # where {#fig-id[^}]*} truncates at LaTeX braces like $5{,}000\times$.
    fig_id_pattern = re.compile(r'\{#(fig-[a-zA-Z0-9_-]+)[\s}]')
    # Markdown image caption: ![non-empty caption text](path)
    # Use a pattern that handles nested brackets (e.g. citations [@key] inside captions)
    md_caption_pattern = re.compile(r'!\[(.+?)\]\(')

    for line_num, line in enumerate(lines, start=1):
        match = fig_id_pattern.search(line)
        if not match:
            continue

        fig_id = match.group(1)

        # Check the FULL LINE for fig-cap="<non-empty>" and fig-alt="<non-empty>"
        has_cap = bool(re.search(r'fig-cap="[^"]+', line))
        has_alt = bool(re.search(r'fig-alt="[^"]+', line))

        # Determine source type and handle markdown caption
        stripped = line.strip()
        if stripped.startswith(':::'):
            source_type = 'div'
        elif '![' in line:
            source_type = 'markdown'
            # Markdown figures carry their caption in ![caption text](path),
            # not in fig-cap="...". Check for non-empty markdown caption.
            md_cap = md_caption_pattern.search(line)
            if md_cap and md_cap.group(1).strip():
                has_cap = True
        else:
            source_type = 'div'

        fig = FigureInfo(
            fig_id=fig_id,
            line_num=line_num,
            file_path=filepath,
            has_caption=has_cap,
            has_alt_text=has_alt,
            source_type=source_type,
        )
        figures.append(fig)
        seen_ids[fig_id] = len(figures) - 1

    # --- Pass 2: Code-cell figures ---
    in_code_block = False
    code_block_start = 0
    cell_options: dict[str, str] = {}

    for line_num, line in enumerate(lines, start=1):
        stripped = line.rstrip()

        # Detect code block start
        if not in_code_block and re.match(r'^```\{(?:python|r|julia|ojs)', stripped):
            in_code_block = True
            code_block_start = line_num
            cell_options = {}
            continue

        # Detect code block end
        if in_code_block and stripped == '```':
            fig_id = cell_options.get('label', '')
            if fig_id.startswith('fig-') and fig_id not in seen_ids:
                cap_val = cell_options.get('fig-cap', '')
                alt_val = cell_options.get('fig-alt', '')

                fig = FigureInfo(
                    fig_id=fig_id,
                    line_num=code_block_start,
                    file_path=filepath,
                    has_caption=bool(cap_val),
                    has_alt_text=bool(alt_val),
                    source_type='code-cell',
                )
                figures.append(fig)
                seen_ids[fig_id] = len(figures) - 1
            in_code_block = False
            cell_options = {}
            continue

        # Inside a code block — collect cell options
        if in_code_block:
            opt_match = re.match(r'^#\|\s*([\w-]+):\s*(.+)$', stripped)
            if opt_match:
                key = opt_match.group(1)
                value = _clean_yaml_string(opt_match.group(2))
                cell_options[key] = value

    return figures


def format_report(figures: list[FigureInfo], quiet: bool = False,
                  check_alt: bool = True, check_cap: bool = True) -> tuple[str, int]:
    """
    Format a report of figure completeness issues.

    Returns (report_string, issue_count).
    """
    issues = []
    ok_count = 0

    for fig in figures:
        missing = []
        if check_cap and not fig.has_caption:
            missing.append('caption')
        if check_alt and not fig.has_alt_text:
            missing.append('alt-text')

        if missing:
            try:
                rel_path = fig.file_path.relative_to(Path.cwd())
            except ValueError:
                rel_path = fig.file_path
            issues.append(
                f"  ❌ {fig.fig_id:<40} L{fig.line_num:<6} [{fig.source_type:<9}] "
                f"missing: {', '.join(missing)}  ({rel_path})"
            )
        else:
            ok_count += 1

    lines = []
    if issues:
        lines.append(f"\n🔍 Figure completeness issues ({len(issues)}):\n")
        lines.extend(issues)
    if not quiet:
        lines.append(f"\n✅ {ok_count} figures OK, {len(issues)} with missing attributes")

    return '\n'.join(lines), len(issues)


def main():
    parser = argparse.ArgumentParser(
        description='Check that all figures have captions and alt-text.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'paths', nargs='*', type=Path,
        help='QMD files to check (if no -d given)'
    )
    parser.add_argument(
        '-d', '--directory', type=Path,
        help='Directory to scan recursively for .qmd files'
    )
    parser.add_argument(
        '--strict', action='store_true',
        help='Exit with non-zero code if any issues found'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Only show issues'
    )
    parser.add_argument(
        '--alt-only', action='store_true',
        help='Only check for missing alt-text'
    )
    parser.add_argument(
        '--cap-only', action='store_true',
        help='Only check for missing captions'
    )

    args = parser.parse_args()

    # Collect files
    files: list[Path] = []
    if args.directory:
        files = sorted(args.directory.rglob('*.qmd'))
    elif args.paths:
        files = [p for p in args.paths if p.suffix == '.qmd']
    else:
        parser.error('Provide either -d DIRECTORY or one or more .qmd file paths')

    if not files:
        print('No .qmd files found.')
        sys.exit(1)

    # Determine what to check
    check_alt = not args.cap_only
    check_cap = not args.alt_only

    all_figures: list[FigureInfo] = []
    for filepath in files:
        if filepath.exists():
            all_figures.extend(scan_file(filepath))

    if not args.quiet:
        print(f"📊 Scanned {len(files)} files, found {len(all_figures)} figures")

    report, issue_count = format_report(
        all_figures, quiet=args.quiet,
        check_alt=check_alt, check_cap=check_cap
    )
    print(report)

    if args.strict and issue_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
