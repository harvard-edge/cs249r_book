#!/usr/bin/env python3
"""
Reference Placement Checker for Quarto Documents.

Validates that citations and references are placed in locations where
Pandoc/Quarto can process them. Catches issues where markdown syntax
appears inside raw code blocks where it won't be rendered.

Current checks:
  --citations-in-code    Citations [@key] inside TikZ/LaTeX code blocks
  --citations-in-raw     Citations [@key] inside raw HTML/LaTeX blocks
  --all                  Run all checks (default)

Usage:
    python3 check_references.py [path] [--check-flags]
    python3 check_references.py book/quarto/contents/
    python3 check_references.py --citations-in-code

Exit codes:
    0 - No issues found
    1 - Issues found
    2 - Path error or invalid arguments

Future extensibility:
    Add new check classes and register them in AVAILABLE_CHECKS.
    Each check returns a list of Issue objects for consistent reporting.
"""

import argparse
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Type

DEPRECATION_MSG = (
    "DEPRECATION: use Binder instead of direct script invocation:\n"
    "  ./book/binder validate refs [--path <file-or-dir>]"
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Issue:
    """Represents a reference/citation issue found in a file."""
    file: Path
    line_number: int
    check_name: str
    message: str
    context: str = ""
    severity: str = "error"  # error, warning, info

    def __str__(self) -> str:
        ctx = f"\n    → {self.context[:100]}..." if len(self.context) > 100 else f"\n    → {self.context}" if self.context else ""
        return f"  Line {self.line_number}: {self.message}{ctx}"


@dataclass
class CheckResult:
    """Result of running a check across all files."""
    check_name: str
    description: str
    issues: List[Issue] = field(default_factory=list)
    files_checked: int = 0

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0


# =============================================================================
# Base Check Class
# =============================================================================

class ReferenceCheck(ABC):
    """Base class for all reference checks."""

    name: str = "base_check"
    description: str = "Base check"
    help_text: str = "How to fix these issues"

    @abstractmethod
    def check_file(self, filepath: Path, content: str) -> List[Issue]:
        """
        Check a single file for issues.

        Args:
            filepath: Path to the file being checked
            content: File content as string

        Returns:
            List of Issue objects found in this file
        """
        pass

    def check_files(self, files: List[Path]) -> CheckResult:
        """Run this check across multiple files."""
        result = CheckResult(
            check_name=self.name,
            description=self.description,
            files_checked=len(files)
        )

        for filepath in files:
            try:
                content = filepath.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = filepath.read_text(encoding='utf-8', errors='ignore')

            issues = self.check_file(filepath, content)
            result.issues.extend(issues)

        return result


# =============================================================================
# Check Implementations
# =============================================================================

class CitationsInCodeCheck(ReferenceCheck):
    """
    Check for Quarto citation syntax inside TikZ/LaTeX code blocks.

    Citations like [@key] only work in markdown content, not inside raw
    LaTeX/TikZ code where they render as literal text.
    """

    name = "citations-in-code"
    description = "Citations [@key] inside TikZ/LaTeX code blocks"
    help_text = """How to fix:
  1. Move the citation to the fig-cap attribute in the Quarto div
  2. Or move the citation to prose text after the figure
  3. Or use LaTeX \\cite{key} if bibliography is configured for LaTeX"""

    # Code block classes where citations won't be processed
    PROBLEMATIC_CLASSES = {'tikz', 'latex', 'tex'}

    # Citation pattern: [@key], [@key1; @key2], [-@key], etc.
    CITATION_PATTERN = re.compile(r'\[-?@[a-zA-Z0-9_:-]+(?:;\s*-?@[a-zA-Z0-9_:-]+)*\]')

    # Fenced code block pattern
    FENCED_CODE_PATTERN = re.compile(r'```\{([^}]+)\}(.*?)```', re.DOTALL)

    def _extract_code_class(self, attrs_str: str) -> str:
        """Extract primary code class from attributes."""
        match = re.search(r'\.([a-zA-Z][a-zA-Z0-9_-]*)', attrs_str)
        return match.group(1).lower() if match else "unknown"

    def _line_at_offset(self, content: str, offset: int) -> int:
        """Convert character offset to 1-based line number."""
        return content[:offset].count('\n') + 1

    def check_file(self, filepath: Path, content: str) -> List[Issue]:
        issues = []

        for match in self.FENCED_CODE_PATTERN.finditer(content):
            attrs_str = match.group(1)
            code_content = match.group(2)
            block_start = match.start()

            code_class = self._extract_code_class(attrs_str)

            if code_class not in self.PROBLEMATIC_CLASSES:
                continue

            for cite_match in self.CITATION_PATTERN.finditer(code_content):
                citation = cite_match.group(0)
                opening_len = len(f"```{{{attrs_str}}}")
                cite_offset = block_start + opening_len + cite_match.start()
                line_num = self._line_at_offset(content, cite_offset)

                lines = content.splitlines()
                context = lines[line_num - 1].strip() if 0 < line_num <= len(lines) else ""

                issues.append(Issue(
                    file=filepath,
                    line_number=line_num,
                    check_name=self.name,
                    message=f"{citation} in .{code_class} block",
                    context=context
                ))

        return issues


class CitationsInRawBlocksCheck(ReferenceCheck):
    """
    Check for citations inside raw HTML/LaTeX blocks.

    Raw blocks (```{=html}, ```{=latex}) are passed through without
    markdown processing, so citations won't work.
    """

    name = "citations-in-raw"
    description = "Citations [@key] inside raw HTML/LaTeX blocks"
    help_text = """How to fix:
  1. Move the citation outside the raw block
  2. Or use native HTML/LaTeX citation syntax"""

    CITATION_PATTERN = re.compile(r'\[-?@[a-zA-Z0-9_:-]+(?:;\s*-?@[a-zA-Z0-9_:-]+)*\]')
    RAW_BLOCK_PATTERN = re.compile(r'```\{=(html|latex|tex)\}(.*?)```', re.DOTALL | re.IGNORECASE)

    def _line_at_offset(self, content: str, offset: int) -> int:
        return content[:offset].count('\n') + 1

    def check_file(self, filepath: Path, content: str) -> List[Issue]:
        issues = []

        for match in self.RAW_BLOCK_PATTERN.finditer(content):
            block_type = match.group(1)
            block_content = match.group(2)
            block_start = match.start()

            for cite_match in self.CITATION_PATTERN.finditer(block_content):
                citation = cite_match.group(0)
                cite_offset = block_start + cite_match.start()
                line_num = self._line_at_offset(content, cite_offset)

                lines = content.splitlines()
                context = lines[line_num - 1].strip() if 0 < line_num <= len(lines) else ""

                issues.append(Issue(
                    file=filepath,
                    line_number=line_num,
                    check_name=self.name,
                    message=f"{citation} in raw {block_type} block",
                    context=context
                ))

        return issues


# =============================================================================
# Check Registry
# =============================================================================

# Primary checks - these catch real issues where references won't work
AVAILABLE_CHECKS: Dict[str, Type[ReferenceCheck]] = {
    'citations-in-code': CitationsInCodeCheck,
    'citations-in-raw': CitationsInRawBlocksCheck,
}


# =============================================================================
# Main Functions
# =============================================================================

def find_qmd_files(root: Path) -> List[Path]:
    """Find all .qmd files under root."""
    # Try standard locations
    for subdir in ['quarto/contents', 'book/quarto/contents', '']:
        search_path = root / subdir if subdir else root
        if search_path.exists():
            files = list(search_path.rglob('*.qmd'))
            if files:
                return sorted(files)
    return []


def print_results(results: List[CheckResult], root: Path, verbose: bool = False) -> int:
    """Print check results and return exit code."""
    total_issues = sum(len(r.issues) for r in results)

    if total_issues == 0:
        print("✓ All reference checks passed!")
        if verbose:
            for r in results:
                print(f"  ✓ {r.check_name}: {r.files_checked} files checked")
        return 0

    # Group issues by file for cleaner output
    issues_by_file: Dict[Path, List[Issue]] = {}
    for result in results:
        for issue in result.issues:
            issues_by_file.setdefault(issue.file, []).append(issue)

    print(f"Found {total_issues} reference issue(s):\n")

    for filepath, file_issues in sorted(issues_by_file.items()):
        try:
            rel_path = filepath.relative_to(root)
        except ValueError:
            rel_path = filepath

        print(f"File: {rel_path}")
        for issue in sorted(file_issues, key=lambda x: x.line_number):
            print(str(issue))
        print()

    # Print help text for each check with issues
    print("-" * 60)
    checks_with_issues = {r.check_name: r for r in results if not r.passed}
    for check_name, result in checks_with_issues.items():
        check_class = AVAILABLE_CHECKS.get(check_name)
        if check_class:
            print(f"\n{check_class.help_text}")

    print(f"\nTotal: {total_issues} issue(s) in {len(issues_by_file)} file(s)")
    return 1


def main(argv: List[str] = None) -> int:
    """Main entry point."""
    print(DEPRECATION_MSG, file=sys.stderr)

    parser = argparse.ArgumentParser(
        description="Check for reference and citation issues in Quarto documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s book/quarto/contents/           # Run all checks on directory
  %(prog)s --citations-in-code             # Run specific check only
  %(prog)s file.qmd                        # Check single file

Available checks:
  --citations-in-code   Citations inside TikZ/LaTeX code blocks
  --citations-in-raw    Citations inside raw HTML/LaTeX blocks
  --all                 Run all checks (default)
        """
    )

    parser.add_argument('path', nargs='?', type=Path,
                        help='File or directory to check (default: book root)')
    parser.add_argument('--all', action='store_true',
                        help='Run all checks (default if no specific check selected)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    # Add flag for each check
    for check_name, check_class in AVAILABLE_CHECKS.items():
        parser.add_argument(f'--{check_name}', action='store_true',
                            help=check_class.description)

    args = parser.parse_args(argv)

    # Determine root path
    if args.path:
        root = args.path.resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        root = script_dir.parents[2]  # book/tools/scripts/content -> book

    if not root.exists():
        print(f"Error: Path does not exist: {root}", file=sys.stderr)
        return 2

    # Determine which checks to run
    selected_checks = [
        name for name in AVAILABLE_CHECKS
        if getattr(args, name.replace('-', '_'), False)
    ]

    # Default to all checks if none selected
    if not selected_checks or args.all:
        selected_checks = list(AVAILABLE_CHECKS.keys())

    # Find files
    if root.is_file():
        files = [root]
        root = root.parent
    else:
        files = find_qmd_files(root)

    if not files:
        print(f"No .qmd files found in {root}", file=sys.stderr)
        return 2

    print(f"Running reference checks on {len(files)} file(s)...")
    print(f"Checks: {', '.join(selected_checks)}\n")

    # Run selected checks
    results = []
    for check_name in selected_checks:
        check_class = AVAILABLE_CHECKS[check_name]
        checker = check_class()
        result = checker.check_files(files)
        results.append(result)

    return print_results(results, root, args.verbose)


if __name__ == '__main__':
    sys.exit(main())
