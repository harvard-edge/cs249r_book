#!/usr/bin/env python3
"""
Check for footnotes in forbidden locations that break Quarto builds.

This script validates that footnotes ([^fn-...]) are NOT placed in:
- Table cells (between | | markers)
- Figure captions (fig-cap: "..." or after figures)
- Table captions (tbl-cap: "..." or after tables)
- Inside ::: div blocks (callouts, examples, etc.)

It also checks for inline footnote syntax (^[...]) which should use
proper reference format ([^fn-name]) instead.

These restrictions prevent Quarto rendering errors and build failures.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class ForbiddenFootnoteChecker:
    """Check for footnotes in locations that break Quarto builds."""

    def __init__(self):
        self.errors = []
        self.footnote_pattern = re.compile(r'\[\^fn-[\w-]+\]')
        self.inline_footnote_pattern = re.compile(r'\^\[[^\]]+\]')

    def check_file(self, filepath: Path) -> List[Tuple[int, str, str]]:
        """
        Check a single file for forbidden footnote placements.

        Returns:
            List of (line_number, error_type, context) tuples
        """
        file_errors = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️  Error reading {filepath}: {e}")
            return file_errors

        in_div_block = False
        div_start_line = 0

        for line_num, line in enumerate(lines, 1):
            # Track div blocks
            if line.strip().startswith(':::'):
                if not in_div_block:
                    in_div_block = True
                    div_start_line = line_num
                else:
                    in_div_block = False

            # Check 0: Inline footnotes (^[...]) - should use proper references instead
            # This check runs independently of other checks
            inline_footnotes = self.inline_footnote_pattern.findall(line)
            if inline_footnotes:
                for inline_fn in inline_footnotes:
                    context = line.strip()[:80]
                    file_errors.append((
                        line_num,
                        "INLINE_FOOTNOTE",
                        f"Found inline footnote '{inline_fn}'. Use [^fn-name] reference format instead: {context}"
                    ))

            # Check for footnotes in this line
            footnotes = self.footnote_pattern.findall(line)
            if not footnotes:
                continue

            # Check 1: Footnotes in table cells (between pipes)
            if self._is_table_row(line):
                for footnote in footnotes:
                    # Check if footnote is within table cell content (between | |)
                    if self._footnote_in_table_cell(line, footnote):
                        context = line.strip()[:80]
                        file_errors.append((
                            line_num,
                            "TABLE_CELL",
                            f"Found '{footnote}' in table cell: {context}"
                        ))

            # Check 2: Footnotes in YAML-style captions
            if re.match(r'^\s*(fig-cap|tbl-cap):', line):
                for footnote in footnotes:
                    context = line.strip()[:80]
                    caption_type = "FIGURE" if "fig-cap" in line else "TABLE"
                    file_errors.append((
                        line_num,
                        f"{caption_type}_CAPTION_YAML",
                        f"Found '{footnote}' in {caption_type.lower()} caption: {context}"
                    ))

            # Check 3: Footnotes in markdown-style captions (: **Caption**: text)
            if re.match(r'^:\s*\*\*[^*]+\*\*:', line):
                for footnote in footnotes:
                    context = line.strip()[:80]
                    file_errors.append((
                        line_num,
                        "MARKDOWN_CAPTION",
                        f"Found '{footnote}' in markdown caption: {context}"
                    ))

            # Check 4: Footnotes inside ANY div blocks
            # Div blocks (:::) are used for figures, callouts, examples, etc.
            # Footnotes break Quarto rendering inside these blocks
            if in_div_block and div_start_line != line_num:
                for footnote in footnotes:
                    context = line.strip()[:80]
                    div_line = lines[div_start_line - 1] if div_start_line > 0 else ""
                    div_context = div_line.strip()[:60]
                    file_errors.append((
                        line_num,
                        "DIV_BLOCK",
                        f"Found '{footnote}' inside div block (started line {div_start_line}: {div_context}): {context}"
                    ))

        return file_errors

    def _is_table_row(self, line: str) -> bool:
        """Check if line is a table row (contains | markers)."""
        stripped = line.strip()
        # Must start with a pipe to be a table row (avoids math notation like |x|)
        if not stripped.startswith('|'):
            return False
        # Must have at least two pipes
        if stripped.count('|') < 2:
            return False
        # Exclude separator lines like |---|---|
        if re.match(r'^\|[\s\-:+]+\|', stripped):
            return False
        return True

    def _footnote_in_table_cell(self, line: str, footnote: str) -> bool:
        """Check if footnote appears within table cell content (between pipes)."""
        # Split by pipes and check if footnote is in any cell
        cells = line.split('|')
        for cell in cells:
            if footnote in cell:
                return True
        return False

    def check_directory(self, directory: Path) -> bool:
        """
        Recursively check all .qmd files in directory.

        Returns:
            True if no errors found, False otherwise
        """
        all_errors = []
        qmd_files = sorted(directory.rglob('*.qmd'))

        if not qmd_files:
            print(f"⚠️  No .qmd files found in {directory}")
            return True

        for filepath in qmd_files:
            file_errors = self.check_file(filepath)
            if file_errors:
                all_errors.append((filepath, file_errors))

        if all_errors:
            self._print_errors(all_errors)
            return False
        else:
            print(f"✅ No forbidden footnote placements found in {len(qmd_files)} files")
            return True

    def _print_errors(self, all_errors: List[Tuple[Path, List[Tuple[int, str, str]]]]):
        """Print formatted error messages."""
        print("\n" + "=" * 80)
        print("🚫 FORBIDDEN FOOTNOTE PLACEMENTS DETECTED")
        print("=" * 80)
        print("\nFootnotes CANNOT be placed in:")
        print("  • Table cells (breaks Quarto table rendering)")
        print("  • Figure/table captions (breaks cross-referencing)")
        print("  • Div blocks like callouts (breaks content rendering)")
        print("\nFootnote formatting violations:")
        print("  • Inline footnotes ^[...] (must use [^fn-name] reference format)")
        print("\nSee: tools/scripts/genai/prompt.txt for footnote placement rules")
        print("=" * 80 + "\n")

        for filepath, errors in all_errors:
            try:
                rel_path = filepath.relative_to(Path.cwd()) if filepath.is_absolute() else filepath
            except ValueError:
                # File is outside current directory (e.g., /tmp)
                rel_path = filepath
            print(f"\n📄 {rel_path}")

            # Group by error type
            by_type = {}
            for line_num, error_type, context in errors:
                if error_type not in by_type:
                    by_type[error_type] = []
                by_type[error_type].append((line_num, context))

            for error_type, instances in sorted(by_type.items()):
                print(f"\n  ❌ {error_type}:")
                for line_num, context in instances:
                    print(f"     Line {line_num}: {context}")

        print("\n" + "=" * 80)
        print(f"Total: {sum(len(e) for _, e in all_errors)} forbidden footnote(s) in {len(all_errors)} file(s)")
        print("=" * 80 + "\n")
        print("💡 To fix:")
        print("   1. Move footnote to regular paragraph text before/after the table or caption")
        print("   2. Or convert the footnoted information into inline text")
        print("   3. For tables: Add explanation in text before the table instead")
        print("   4. For inline footnotes ^[...]: Create a proper footnote definition [^fn-name]:")
        print("      and use [^fn-name] as a reference in the text")
        print()


def main():
    """Main entry point for pre-commit hook."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check for footnotes in forbidden locations (tables, captions, divs)"
    )
    parser.add_argument(
        'paths',
        nargs='*',
        help='Files or directories to check (default: quarto/contents/)'
    )
    parser.add_argument(
        '-d', '--directory',
        help='Directory to check recursively',
        default=None
    )

    args = parser.parse_args()

    checker = ForbiddenFootnoteChecker()

    # Determine what to check
    if args.directory:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            sys.exit(1)
        success = checker.check_directory(directory)
    elif args.paths:
        # Check individual files
        all_errors = []
        for path_str in args.paths:
            path = Path(path_str)
            if path.is_file() and path.suffix == '.qmd':
                errors = checker.check_file(path)
                if errors:
                    all_errors.append((path, errors))
            elif path.is_dir():
                if not checker.check_directory(path):
                    sys.exit(1)

        if all_errors:
            checker._print_errors(all_errors)
            sys.exit(1)
        success = True
    else:
        # Default to quarto/contents/
        default_dir = Path('quarto/contents/')
        if not default_dir.exists():
            print(f"❌ Default directory not found: {default_dir}")
            sys.exit(1)
        success = checker.check_directory(default_dir)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
