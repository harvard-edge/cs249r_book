#!/usr/bin/env python3
"""
📝 Comprehensive Footnote Management for Quarto Files

This unified script handles all footnote operations in .qmd files:
- REORGANIZE: Move footnote definitions to immediately after their references
- VALIDATE: Check for undefined references and unused definitions
- CATALOG: Generate comprehensive footnote reports
- REMOVE: Remove all footnotes from files
- CLEAN: Fix validation issues automatically

DESIGN PHILOSOPHY:
- Single tool for all footnote operations
- Clear visual output with emoji indicators
- Fast execution for CI/CD workflows
- Standard -f/-d options like other MLSysBook scripts
- Exit codes: 0 = success, 1 = issues found/errors

Usage:
    python footnote_cleanup.py -d quarto/contents/ --reorganize
    python footnote_cleanup.py -f chapter.qmd --validate
    python footnote_cleanup.py -d quarto/ --catalog --output report.json
    python footnote_cleanup.py -f chapter.qmd --remove --dry-run
    python footnote_cleanup.py -d quarto/ --clean --backup
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class FootnoteManager:
    """Unified footnote management for QMD files."""

    def __init__(self, dry_run: bool = False, backup: bool = False, quiet: bool = False):
        self.dry_run = dry_run
        self.backup = backup
        self.quiet = quiet

        # Regex patterns
        self.footnote_ref_pattern = re.compile(r'\[\^([^]]+)\]')  # [^fn-name] anywhere
        self.footnote_def_pattern = re.compile(r'^\[\^([^]]+)\]:\s*(.+)$', re.MULTILINE)

        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'references_found': 0,
            'definitions_found': 0,
            'issues_found': 0,
            'issues_fixed': 0
        }

    def log(self, message: str, color: str = Colors.ENDC):
        """Log message if not in quiet mode."""
        if not self.quiet:
            print(f"{color}{message}{Colors.ENDC}")

    def find_qmd_files(self, path: str) -> List[Path]:
        """Find all .qmd files in the given path."""
        path_obj = Path(path)

        if path_obj.is_file() and path_obj.suffix == '.qmd':
            return [path_obj]
        elif path_obj.is_dir():
            return sorted(path_obj.rglob('*.qmd'))
        else:
            self.log(f"⚠️  Warning: {path} is not a valid file or directory", Colors.YELLOW)
            return []

    def parse_footnotes(self, content: str) -> Tuple[Dict[str, str], Dict[str, List[int]], List[str]]:
        """
        Parse content to extract footnote definitions, references, and content lines.

        Returns:
            footnote_defs: Dict mapping footnote IDs to their definitions
            footnote_refs: Dict mapping footnote IDs to line numbers where they're referenced
            lines: List of content lines
        """
        lines = content.split('\n')
        footnote_defs = {}
        footnote_refs = defaultdict(list)

        # Find all footnote definitions
        for match in self.footnote_def_pattern.finditer(content):
            footnote_id = match.group(1)
            footnote_content = match.group(2)
            footnote_defs[footnote_id] = footnote_content

        # Find all footnote references and their line numbers
        nested_refs = []  # Track footnotes that reference other footnotes

        for line_num, line in enumerate(lines):
            for match in self.footnote_ref_pattern.finditer(line):
                footnote_id = match.group(1)

                # Check if this match is part of a footnote definition
                # A footnote definition has the pattern [^id]: at the start of the line
                def_match = self.footnote_def_pattern.match(line)
                if def_match and def_match.group(1) == footnote_id:
                    # This is the definition itself, not a reference
                    continue
                elif def_match:
                    # This is a reference inside another footnote definition
                    defining_footnote = def_match.group(1)
                    nested_refs.append({
                        'defining_footnote': defining_footnote,
                        'referenced_footnote': footnote_id,
                        'line_num': line_num + 1
                    })

                footnote_refs[footnote_id].append(line_num)

        # Store nested references for potential warnings
        if hasattr(self, 'nested_refs'):
            self.nested_refs.extend(nested_refs)
        else:
            self.nested_refs = nested_refs

        return footnote_defs, dict(footnote_refs), lines

    def find_paragraph_end(self, lines: List[str], start_line: int) -> int:
        """Find the end of the paragraph containing the given line."""
        for i in range(start_line + 1, len(lines)):
            line = lines[i].strip()

            # Empty line ends paragraph
            if not line:
                return i - 1

            # Heading ends paragraph
            if line.startswith('#'):
                return i - 1

            # Special blocks end paragraph
            if line.startswith(':::') or line.startswith('```') or line.startswith('|'):
                return i - 1

            # Footnote definition ends paragraph
            if self.footnote_def_pattern.match(line):
                return i - 1

        return len(lines) - 1

    def count_footnotes(self, content: str) -> Tuple[int, int]:
        """Count footnote references and definitions in content."""
        footnote_defs, footnote_refs, lines = self.parse_footnotes(content)

        total_refs = sum(len(refs) for refs in footnote_refs.values())
        total_defs = len(footnote_defs)

        return total_refs, total_defs

    def reorganize_footnotes(self, content: str) -> Tuple[str, bool]:
        """Reorganize footnotes to appear after their references."""
        # Count footnotes before reorganization
        original_refs, original_defs = self.count_footnotes(content)

        footnote_defs, footnote_refs, lines = self.parse_footnotes(content)

        if not footnote_defs or not footnote_refs:
            return content, False

        # Check if reorganization is needed
        needs_reorganization = False
        for footnote_id, def_content in footnote_defs.items():
            if footnote_id not in footnote_refs:
                continue

            # Find where this footnote is currently defined
            def_line = None
            for i, line in enumerate(lines):
                if line.startswith(f'[^{footnote_id}]:'):
                    def_line = i
                    break

            if def_line is None:
                continue

            # Find where it should be (after the first reference)
            first_ref_line = min(footnote_refs[footnote_id])
            paragraph_end = self.find_paragraph_end(lines, first_ref_line)

            # If the definition is not immediately after the paragraph, reorganization is needed
            if def_line != paragraph_end + 2:  # +2 for empty line + definition line
                needs_reorganization = True
                break

        if not needs_reorganization:
            return content, False

        # Reorganize footnotes
        new_lines = []
        processed_footnotes = set()
        skip_lines = set()  # Lines to skip (original footnote definitions)

        # Mark original footnote definition lines for removal
        for i, line in enumerate(lines):
            if self.footnote_def_pattern.match(line):
                skip_lines.add(i)

        # Process each line
        for i, line in enumerate(lines):
            # Skip original footnote definition lines
            if i in skip_lines:
                continue

            new_lines.append(line)

            # Check if this line contains footnote references
            refs_in_line = []
            for match in self.footnote_ref_pattern.finditer(line):
                footnote_id = match.group(1)
                if footnote_id in footnote_defs and footnote_id not in processed_footnotes:
                    refs_in_line.append(footnote_id)

            # If this is the end of a paragraph with footnote references, add the definitions
            if refs_in_line:
                paragraph_end = self.find_paragraph_end(lines, i)

                # If we're at the paragraph end, add footnote definitions
                if i == paragraph_end:
                    # Add empty line before footnotes (proper spacing)
                    new_lines.append('')

                    for i, footnote_id in enumerate(refs_in_line):
                        if footnote_id in footnote_defs:
                            footnote_def_line = f'[^{footnote_id}]: {footnote_defs[footnote_id]}'
                            new_lines.append(footnote_def_line)
                            processed_footnotes.add(footnote_id)

                            # Add blank line between footnotes (but not after the last one)
                            if i < len(refs_in_line) - 1:
                                new_lines.append('')

        # Clean up excessive empty lines (more than 2 consecutive)
        final_lines = []
        empty_count = 0

        for line in new_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    final_lines.append(line)
            else:
                empty_count = 0
                final_lines.append(line)

        reorganized_content = '\n'.join(final_lines)

        # Verify footnote counts after reorganization
        new_refs, new_defs = self.count_footnotes(reorganized_content)

        if original_refs != new_refs or original_defs != new_defs:
            self.log(f"⚠️  WARNING: Footnote count mismatch!", Colors.RED)
            self.log(f"   Original: {original_refs} refs, {original_defs} defs", Colors.YELLOW)
            self.log(f"   New: {new_refs} refs, {new_defs} defs", Colors.YELLOW)
            # Return original content to prevent data loss
            return content, False

        return reorganized_content, True

    def validate_footnotes(self, content: str, file_path: Path) -> Tuple[Set[str], Set[str], Set[str], List[Dict]]:
        """
        Validate footnotes and return issues.

        Returns:
            undefined_refs: References without definitions
            unused_defs: Definitions without references
            duplicate_defs: Duplicate definitions
            nested_refs: Footnotes that reference other footnotes
        """
        # Reset nested refs for this file
        self.nested_refs = []

        footnote_defs, footnote_refs, lines = self.parse_footnotes(content)

        # Find undefined references
        undefined_refs = set(footnote_refs.keys()) - set(footnote_defs.keys())

        # Find unused definitions
        unused_defs = set(footnote_defs.keys()) - set(footnote_refs.keys())

        # Find duplicate definitions
        def_counts = defaultdict(int)
        for line in lines:
            match = re.match(r'^\[\^([^]]+)\]:', line)
            if match:
                def_counts[match.group(1)] += 1

        duplicate_defs = {fn_id for fn_id, count in def_counts.items() if count > 1}

        return undefined_refs, unused_defs, duplicate_defs, self.nested_refs

    def clean_footnotes(self, content: str) -> Tuple[str, int]:
        """Clean footnote issues by removing undefined references and unused definitions."""
        footnote_defs, footnote_refs, lines = self.parse_footnotes(content)

        undefined_refs, unused_defs, duplicate_defs, nested_refs = self.validate_footnotes(content, Path("temp"))

        if not undefined_refs and not unused_defs:
            return content, 0

        issues_fixed = 0

        # Remove undefined references from content
        cleaned_content = content
        for ref_id in undefined_refs:
            pattern = rf'\[\^{re.escape(ref_id)}\]'
            cleaned_content = re.sub(pattern, '', cleaned_content)
            issues_fixed += 1

        # Remove unused definitions
        lines = cleaned_content.split('\n')
        cleaned_lines = []
        skip_mode = False

        for i, line in enumerate(lines):
            # Check if this line starts an unused footnote definition
            match = re.match(r'^\[\^([^]]+)\]:', line)
            if match and match.group(1) in unused_defs:
                skip_mode = True
                issues_fixed += 1
                continue

            # If we're in skip mode, check if this line is a continuation
            if skip_mode:
                # Continuation lines start with whitespace (indented)
                if line and (line[0] == ' ' or line[0] == '\t'):
                    continue
                # Empty lines after footnotes are also skipped
                elif not line.strip():
                    # Check if next line exists and is indented (continuation)
                    if i + 1 < len(lines) and lines[i + 1] and (lines[i + 1][0] == ' ' or lines[i + 1][0] == '\t'):
                        continue
                    # Otherwise, end skip mode but still skip this empty line
                    skip_mode = False
                    continue
                else:
                    # Non-indented, non-empty line means footnote is done
                    skip_mode = False

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines), issues_fixed

    def remove_footnotes(self, content: str) -> Tuple[str, int, int]:
        """Remove all footnotes from content."""
        footnote_defs, footnote_refs, lines = self.parse_footnotes(content)

        inline_refs_removed = 0
        definitions_removed = 0

        # Remove inline references
        cleaned_content = content
        for ref_id in footnote_refs:
            pattern = rf'\[\^{re.escape(ref_id)}\]'
            matches = len(re.findall(pattern, cleaned_content))
            cleaned_content = re.sub(pattern, '', cleaned_content)
            inline_refs_removed += matches

        # Remove footnote definitions
        lines = cleaned_content.split('\n')
        cleaned_lines = []
        skip_mode = False

        for i, line in enumerate(lines):
            # Check if this line starts a footnote definition
            if re.match(r'^\[\^[^\]]+\]:', line):
                skip_mode = True
                definitions_removed += 1
                continue

            # If we're in skip mode, check if this line is a continuation
            if skip_mode:
                # Continuation lines start with whitespace (indented)
                if line and (line[0] == ' ' or line[0] == '\t'):
                    continue
                # Empty lines after footnotes are also skipped
                elif not line.strip():
                    # Check if next line exists and is indented (continuation)
                    if i + 1 < len(lines) and lines[i + 1] and (lines[i + 1][0] == ' ' or lines[i + 1][0] == '\t'):
                        continue
                    # Otherwise, end skip mode but still skip this empty line
                    skip_mode = False
                    continue
                else:
                    # Non-indented, non-empty line means footnote is done
                    skip_mode = False

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines), inline_refs_removed, definitions_removed

    def catalog_footnotes(self, content: str, file_path: Path) -> Dict:
        """Catalog all footnotes in the content."""
        footnote_defs, footnote_refs, lines = self.parse_footnotes(content)

        catalog = {
            'file': str(file_path),
            'references': [],
            'definitions': [],
            'stats': {
                'total_references': sum(len(refs) for refs in footnote_refs.values()),
                'unique_references': len(footnote_refs),
                'total_definitions': len(footnote_defs),
                'undefined_references': len(set(footnote_refs.keys()) - set(footnote_defs.keys())),
                'unused_definitions': len(set(footnote_defs.keys()) - set(footnote_refs.keys()))
            }
        }

        # Collect reference details
        for footnote_id, line_numbers in footnote_refs.items():
            for line_num in line_numbers:
                if line_num < len(lines):
                    line_content = lines[line_num]
                    # Get context around the reference
                    start = max(0, line_content.find(f'[^{footnote_id}]') - 50)
                    end = min(len(line_content), line_content.find(f'[^{footnote_id}]') + len(footnote_id) + 50)
                    context = line_content[start:end].strip()

                    catalog['references'].append({
                        'footnote_id': footnote_id,
                        'line': line_num + 1,
                        'context': context,
                        'full_line': line_content.strip()
                    })

        # Collect definition details
        for footnote_id, definition in footnote_defs.items():
            catalog['definitions'].append({
                'footnote_id': footnote_id,
                'definition': definition,
                'referenced': footnote_id in footnote_refs,
                'reference_count': len(footnote_refs.get(footnote_id, []))
            })

        return catalog

    def process_file(self, file_path: Path, operation: str, **kwargs) -> bool:
        """Process a single file with the specified operation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            self.stats['files_processed'] += 1
            modified = False

            if operation == 'reorganize':
                # Count footnotes before reorganization
                original_refs, original_defs = self.count_footnotes(original_content)

                new_content, was_modified = self.reorganize_footnotes(original_content)
                if was_modified:
                    # Verify counts after reorganization
                    new_refs, new_defs = self.count_footnotes(new_content)

                    if not self.dry_run:
                        if self.backup:
                            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                            shutil.copy2(file_path, backup_path)
                            self.log(f"📄 Created backup: {backup_path}", Colors.CYAN)

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)

                    self.log(f"✅ Reorganized footnotes: {file_path} ({original_refs} refs, {original_defs} defs)", Colors.GREEN)
                    self.stats['files_modified'] += 1
                    modified = True
                else:
                    self.log(f"⏭️  No changes needed: {file_path} ({original_refs} refs, {original_defs} defs)", Colors.BLUE)

            elif operation == 'validate':
                undefined_refs, unused_defs, duplicate_defs, nested_refs = self.validate_footnotes(original_content, file_path)

                if undefined_refs or unused_defs or duplicate_defs or nested_refs:
                    self.log(f"❌ Issues found in {file_path}:", Colors.RED)
                    if undefined_refs:
                        self.log(f"   📍 Undefined references: {', '.join(undefined_refs)}", Colors.YELLOW)
                    if unused_defs:
                        self.log(f"   🗑️  Unused definitions: {', '.join(unused_defs)}", Colors.YELLOW)
                    if duplicate_defs:
                        self.log(f"   🔄 Duplicate definitions: {', '.join(duplicate_defs)}", Colors.YELLOW)
                    if nested_refs:
                        self.log(f"   🔗 Nested footnote references:", Colors.YELLOW)
                        for nested in nested_refs:
                            self.log(f"      Line {nested['line_num']}: [^{nested['defining_footnote']}] → [^{nested['referenced_footnote']}]", Colors.YELLOW)

                    self.stats['issues_found'] += len(undefined_refs) + len(unused_defs) + len(duplicate_defs) + len(nested_refs)
                    modified = True
                else:
                    self.log(f"✅ Valid footnotes: {file_path}", Colors.GREEN)

            elif operation == 'clean':
                new_content, issues_fixed = self.clean_footnotes(original_content)
                if issues_fixed > 0:
                    if not self.dry_run:
                        if self.backup:
                            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                            shutil.copy2(file_path, backup_path)

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)

                    self.log(f"🧹 Cleaned {issues_fixed} issues: {file_path}", Colors.GREEN)
                    self.stats['files_modified'] += 1
                    self.stats['issues_fixed'] += issues_fixed
                    modified = True
                else:
                    self.log(f"✅ No issues to clean: {file_path}", Colors.GREEN)

            elif operation == 'remove':
                new_content, inline_refs, definitions = self.remove_footnotes(original_content)
                if inline_refs > 0 or definitions > 0:
                    if not self.dry_run:
                        if self.backup:
                            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                            shutil.copy2(file_path, backup_path)

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)

                    self.log(f"🗑️  Removed {inline_refs} refs, {definitions} defs: {file_path}", Colors.GREEN)
                    self.stats['files_modified'] += 1
                    modified = True
                else:
                    self.log(f"⏭️  No footnotes found: {file_path}", Colors.BLUE)

            elif operation == 'catalog':
                catalog = self.catalog_footnotes(original_content, file_path)
                return catalog

            return modified

        except Exception as e:
            self.log(f"❌ Error processing {file_path}: {e}", Colors.RED)
            return False

    def process_files(self, files: List[Path], operation: str, **kwargs) -> List:
        """Process multiple files with the specified operation."""
        results = []

        for file_path in files:
            result = self.process_file(file_path, operation, **kwargs)
            if operation == 'catalog':
                results.append(result)
            else:
                results.append(result)

        return results

    def print_summary(self, operation: str):
        """Print operation summary."""
        if self.quiet:
            return

        self.log("\n" + "="*60, Colors.HEADER)
        self.log(f"📊 {operation.upper()} SUMMARY", Colors.HEADER)
        self.log("="*60, Colors.HEADER)

        self.log(f"Files processed: {self.stats['files_processed']}")
        self.log(f"Files modified: {self.stats['files_modified']}")

        if operation == 'validate':
            self.log(f"Issues found: {self.stats['issues_found']}")
        elif operation == 'clean':
            self.log(f"Issues fixed: {self.stats['issues_fixed']}")

        if self.dry_run and self.stats['files_modified'] > 0:
            self.log("\n💡 Run without --dry-run to apply changes", Colors.YELLOW)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive footnote management for QMD files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python footnote_cleanup.py -d quarto/contents/ --reorganize --dry-run
  python footnote_cleanup.py -f chapter.qmd --validate
  python footnote_cleanup.py -d quarto/ --catalog --output report.json
  python footnote_cleanup.py -f chapter.qmd --remove --backup
  python footnote_cleanup.py -d quarto/ --clean --quiet
        """
    )

    # File/directory selection (standard MLSysBook pattern)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='Process single QMD file')
    group.add_argument('-d', '--directory', help='Process all QMD files in directory')

    # Operations (mutually exclusive)
    ops = parser.add_mutually_exclusive_group(required=True)
    ops.add_argument('--reorganize', action='store_true',
                     help='Move footnote definitions after their references')
    ops.add_argument('--validate', action='store_true',
                     help='Check for footnote issues (undefined refs, unused defs)')
    ops.add_argument('--catalog', action='store_true',
                     help='Generate comprehensive footnote catalog')
    ops.add_argument('--clean', action='store_true',
                     help='Fix footnote issues automatically')
    ops.add_argument('--remove', action='store_true',
                     help='Remove all footnotes from files')

    # Options
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without modifying files')
    parser.add_argument('--backup', action='store_true',
                       help='Create .bak backup files before modifying')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output for CI/CD workflows')
    parser.add_argument('--output', help='Output file for catalog operation')

    args = parser.parse_args()

    # Determine operation
    if args.reorganize:
        operation = 'reorganize'
    elif args.validate:
        operation = 'validate'
    elif args.catalog:
        operation = 'catalog'
    elif args.clean:
        operation = 'clean'
    elif args.remove:
        operation = 'remove'

    # Initialize manager
    manager = FootnoteManager(dry_run=args.dry_run, backup=args.backup, quiet=args.quiet)

    # Get target path
    target_path = args.file if args.file else args.directory

    if not Path(target_path).exists():
        print(f"❌ Error: Path '{target_path}' does not exist")
        sys.exit(1)

    # Find files
    qmd_files = manager.find_qmd_files(target_path)

    if not qmd_files:
        manager.log("❌ No .qmd files found", Colors.RED)
        sys.exit(1)

    manager.log(f"🔍 Found {len(qmd_files)} .qmd file(s)", Colors.BLUE)

    # Process files
    results = manager.process_files(qmd_files, operation)

    # Handle catalog output
    if operation == 'catalog':
        catalog_data = {
            'operation': 'catalog',
            'files': results,
            'summary': {
                'total_files': len(results),
                'total_references': sum(f['stats']['total_references'] for f in results),
                'total_definitions': sum(f['stats']['total_definitions'] for f in results),
                'files_with_issues': sum(1 for f in results if f['stats']['undefined_references'] > 0 or f['stats']['unused_definitions'] > 0)
            }
        }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(catalog_data, f, indent=2)
            manager.log(f"📄 Catalog saved to: {args.output}", Colors.GREEN)
        else:
            print(json.dumps(catalog_data, indent=2))

    # Print summary
    manager.print_summary(operation)

    # Exit with appropriate code
    if operation == 'validate' and manager.stats['issues_found'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
