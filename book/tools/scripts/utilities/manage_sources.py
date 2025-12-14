#!/usr/bin/env python3
"""
Source Citation Checker and Cleaner

This script analyzes, validates, and cleans up source citations in QMD files.
Provides comprehensive reporting and automatic cleanup capabilities.

Usage:
    python check_sources.py --analyze
    python check_sources.py --clean
    python check_sources.py --full
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

class SourceChecker:
    """Main class for checking and cleaning source citations"""

    def __init__(self, target_directories=None, target_files=None):
        self.content_dir = Path("contents")
        self.target_directories = target_directories or []
        self.target_files = target_files or []
        self.stats = {
            'academic_citations': 0,
            'company_sources': 0,
            'link_sources': 0,
            'problematic_asterisk': 0,
            'missing_periods': 0,
            'lowercase_sources': 0,
            'double_periods': 0,
            'malformed_citations': 0,
            'total_files': 0,
            'files_with_sources': 0
        }
        self.problems = {
            'asterisk_sources': [],
            'missing_periods': [],
            'lowercase_sources': [],
            'double_periods': [],
            'malformed_citations': [],
            'extra_brackets': []
        }

    def print_status(self, message: str):
        """Print info message in blue"""
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

    def print_success(self, message: str):
        """Print success message in green"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

    def print_warning(self, message: str):
        """Print warning message in yellow"""
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

    def print_error(self, message: str):
        """Print error message in red"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

    def check_environment(self) -> bool:
        """Check if we're in the correct directory"""
        if not self.content_dir.exists():
            self.print_error("Please run this script from the MLSysBook root directory")
            return False
        return True

    def find_qmd_files(self) -> List[Path]:
        """Find QMD files based on target directories/files or all files"""
        if self.target_files:
            # Process specific files
            qmd_files = []
            for file_path in self.target_files:
                path = Path(file_path)
                if path.suffix == '.qmd' and path.exists():
                    qmd_files.append(path)
                else:
                    self.print_warning(f"File not found or not a QMD file: {file_path}")
        elif self.target_directories:
            # Process specific directories
            qmd_files = []
            for dir_path in self.target_directories:
                directory = Path(dir_path)
                if directory.exists() and directory.is_dir():
                    dir_files = list(directory.rglob("*.qmd"))
                    qmd_files.extend(dir_files)
                    self.print_status(f"Found {len(dir_files)} QMD files in {dir_path}")
                else:
                    self.print_warning(f"Directory not found: {dir_path}")
        else:
            # Process all files in contents directory
            qmd_files = list(self.content_dir.rglob("*.qmd"))

        self.stats['total_files'] = len(qmd_files)
        return qmd_files

    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single QMD file for source citations"""
        file_stats = {
            'academic_citations': 0,
            'company_sources': 0,
            'link_sources': 0,
            'problematic_asterisk': 0,
            'missing_periods': 0,
            'problems': []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Count academic citations: Source: [@citation]
            academic_pattern = r'Source: \[@[^\]]*\]'
            academic_matches = re.findall(academic_pattern, content)
            file_stats['academic_citations'] = len(academic_matches)

            # Count company sources: Source: Company (not academic or link)
            company_pattern = r'Source: [A-Za-z][^.@\[]*(?:\.|$)'
            company_matches = re.findall(company_pattern, content)
            # Filter out academic and link sources
            company_matches = [m for m in company_matches if not re.search(r'\[@|\]\(', m)]
            file_stats['company_sources'] = len(company_matches)

            # Count link sources: Source: [text](url)
            link_pattern = r'Source: \[.*?\]\([^)]*\)'
            link_matches = re.findall(link_pattern, content)
            file_stats['link_sources'] = len(link_matches)

            # Find problematic patterns
            self._find_problems_in_content(content, file_path, file_stats)

            return file_stats

        except Exception as e:
            self.print_error(f"Error analyzing {file_path}: {e}")
            return file_stats

    def _find_problems_in_content(self, content: str, file_path: Path, file_stats: Dict):
        """Find problematic patterns in file content"""

        # Find asterisk-wrapped sources
        asterisk_pattern = r'\*[Ss]ource:[^*]*\*'
        asterisk_matches = list(re.finditer(asterisk_pattern, content))
        file_stats['problematic_asterisk'] = len(asterisk_matches)
        for match in asterisk_matches:
            self.problems['asterisk_sources'].append({
                'file': str(file_path),
                'line': content[:match.start()].count('\n') + 1,
                'text': match.group()
            })

        # Find sources without periods
        no_period_pattern = r'Source: [^.]*[^.]$'
        for line_num, line in enumerate(content.split('\n'), 1):
            if re.search(no_period_pattern, line):
                file_stats['missing_periods'] += 1
                self.problems['missing_periods'].append({
                    'file': str(file_path),
                    'line': line_num,
                    'text': line.strip()
                })

        # Find lowercase 'source:'
        lowercase_pattern = r'source:'
        for line_num, line in enumerate(content.split('\n'), 1):
            if re.search(lowercase_pattern, line):
                self.problems['lowercase_sources'].append({
                    'file': str(file_path),
                    'line': line_num,
                    'text': line.strip()
                })

        # Find double periods
        double_period_pattern = r'Source: .*\.\.'
        for line_num, line in enumerate(content.split('\n'), 1):
            if re.search(double_period_pattern, line):
                self.problems['double_periods'].append({
                    'file': str(file_path),
                    'line': line_num,
                    'text': line.strip()
                })

        # Find malformed academic citations (missing brackets)
        malformed_pattern = r'Source: @[^[]'
        for line_num, line in enumerate(content.split('\n'), 1):
            if re.search(malformed_pattern, line):
                self.problems['malformed_citations'].append({
                    'file': str(file_path),
                    'line': line_num,
                    'text': line.strip()
                })

        # Find extra brackets
        extra_brackets_pattern = r'Source: \[\[@'
        for line_num, line in enumerate(content.split('\n'), 1):
            if re.search(extra_brackets_pattern, line):
                self.problems['extra_brackets'].append({
                    'file': str(file_path),
                    'line': line_num,
                    'text': line.strip()
                })

    def analyze_sources(self) -> Dict:
        """Analyze source citations in QMD files"""
        if self.target_files:
            self.print_status(f"🔍 Analyzing source citations in {len(self.target_files)} specific files...")
        elif self.target_directories:
            self.print_status(f"🔍 Analyzing source citations in {len(self.target_directories)} directories...")
        else:
            self.print_status("🔍 Analyzing source citation patterns...")
        print()

        qmd_files = self.find_qmd_files()

        for file_path in qmd_files:
            file_stats = self.analyze_file(file_path)

            # Aggregate stats
            self.stats['academic_citations'] += file_stats['academic_citations']
            self.stats['company_sources'] += file_stats['company_sources']
            self.stats['link_sources'] += file_stats['link_sources']
            self.stats['problematic_asterisk'] += file_stats['problematic_asterisk']
            self.stats['missing_periods'] += file_stats['missing_periods']

            # Count files with sources
            total_sources = (file_stats['academic_citations'] +
                           file_stats['company_sources'] +
                           file_stats['link_sources'] +
                           file_stats['problematic_asterisk'])
            if total_sources > 0:
                self.stats['files_with_sources'] += 1

        # Update problem counts
        self.stats['lowercase_sources'] = len(self.problems['lowercase_sources'])
        self.stats['double_periods'] = len(self.problems['double_periods'])
        self.stats['malformed_citations'] = len(self.problems['malformed_citations'])

        self._print_analysis_results()
        return self.stats

    def _print_analysis_results(self):
        """Print the analysis results"""
        print("📊 Source Citation Summary:")
        print(f"  ✅ Academic citations (Source: [@citation]): {self.stats['academic_citations']}")
        print(f"  ✅ Company sources (Source: Company): {self.stats['company_sources']}")
        print(f"  ✅ Link sources (Source: [text](url)): {self.stats['link_sources']}")
        print(f"  ❌ Problematic asterisk sources (*Source:): {self.stats['problematic_asterisk']}")
        print(f"  ⚠️  Missing periods: {self.stats['missing_periods']}")
        print(f"  ⚠️  Lowercase 'source:': {self.stats['lowercase_sources']}")
        print(f"  ⚠️  Double periods: {self.stats['double_periods']}")
        print(f"  ⚠️  Malformed citations: {self.stats['malformed_citations']}")
        print()

        total_sources = (self.stats['academic_citations'] +
                        self.stats['company_sources'] +
                        self.stats['link_sources'] +
                        self.stats['problematic_asterisk'])
        print(f"📈 Total sources found: {total_sources}")
        print(f"📁 Files with sources: {self.stats['files_with_sources']}/{self.stats['total_files']}")
        print()

    def find_problems(self):
        """Find and display problematic source patterns"""
        if self.target_files or self.target_directories:
            scope = "specific files/directories" if self.target_files or self.target_directories else "all files"
            self.print_status(f"🔍 Searching for problematic source patterns in {scope}...")
        else:
            self.print_status("🔍 Searching for problematic source patterns...")
        print()

        # Show asterisk sources
        if self.problems['asterisk_sources']:
            self.print_warning(f"Found {len(self.problems['asterisk_sources'])} asterisk-wrapped sources:")
            for problem in self.problems['asterisk_sources'][:3]:
                print(f"  📄 {problem['file']}:{problem['line']}")
                print(f"     {problem['text'][:100]}...")
            if len(self.problems['asterisk_sources']) > 3:
                print(f"     ... and {len(self.problems['asterisk_sources']) - 3} more")
            print()

        # Show missing periods
        if self.problems['missing_periods']:
            self.print_warning(f"Found {len(self.problems['missing_periods'])} sources missing periods:")
            for problem in self.problems['missing_periods'][:3]:
                print(f"  📄 {problem['file']}:{problem['line']}")
                print(f"     {problem['text'][:100]}...")
            if len(self.problems['missing_periods']) > 3:
                print(f"     ... and {len(self.problems['missing_periods']) - 3} more")
            print()

        # Show lowercase sources
        if self.problems['lowercase_sources']:
            self.print_warning(f"Found {len(self.problems['lowercase_sources'])} lowercase 'source:' instances:")
            for problem in self.problems['lowercase_sources'][:3]:
                print(f"  📄 {problem['file']}:{problem['line']}")
            if len(self.problems['lowercase_sources']) > 3:
                print(f"     ... and {len(self.problems['lowercase_sources']) - 3} more")
            print()

        # Show double periods
        if self.problems['double_periods']:
            self.print_warning(f"Found {len(self.problems['double_periods'])} sources with double periods:")
            for problem in self.problems['double_periods'][:2]:
                print(f"  📄 {problem['file']}:{problem['line']}")
                print(f"     {problem['text'][:100]}...")
            if len(self.problems['double_periods']) > 2:
                print(f"     ... and {len(self.problems['double_periods']) - 2} more")
            print()

    def perform_cleanup(self) -> int:
        """Perform automatic cleanup of source citations"""
        if self.target_files:
            self.print_status(f"🧹 Performing automatic cleanup on {len(self.target_files)} specific files...")
        elif self.target_directories:
            self.print_status(f"🧹 Performing automatic cleanup in {len(self.target_directories)} directories...")
        else:
            self.print_status("🧹 Performing automatic cleanup...")
        print()

        qmd_files = self.find_qmd_files()
        files_changed = 0

        for file_path in qmd_files:
            original_content = self._read_file(file_path)
            if original_content is None:
                continue

            modified_content = original_content
            file_modified = False

            # Fix asterisk-wrapped sources
            patterns = [
                (r'\*[Ss]ource: (@[^*]*)\*', r'Source: [\1].'),
                (r'\*[Ss]ource: (\[[^\]]*\]\([^)]*\))\*', r'Source: \1.'),
                (r'\*[Ss]ource: ([^*]*)\*', r'Source: \1.'),
            ]

            for pattern, replacement in patterns:
                new_content = re.sub(pattern, replacement, modified_content)
                if new_content != modified_content:
                    modified_content = new_content
                    file_modified = True

            # Fix lowercase 'source:'
            new_content = re.sub(r'source:', 'Source:', modified_content)
            if new_content != modified_content:
                modified_content = new_content
                file_modified = True

            # Add missing periods to company sources
            new_content = re.sub(r'Source: ([^.@\[]*[^.])$', r'Source: \1.', modified_content, flags=re.MULTILINE)
            if new_content != modified_content:
                modified_content = new_content
                file_modified = True

            # Fix academic citations without brackets
            new_content = re.sub(r'Source: @([a-zA-Z0-9][^.]*)\.', r'Source: [@\1].', modified_content)
            if new_content != modified_content:
                modified_content = new_content
                file_modified = True

            # Clean up double periods - handle various formats
            patterns_double_periods = [
                (r'Source: ([^.]*?)\.\.(\s*\{#[^}]*\})', r'Source: \1.\2'),                    # Double periods before table/figure refs
                (r'Source: ([^.]*?)\.\.(\s*\{#[^}]*\s*\.[^}]*\s*\.[^}]*\})', r'Source: \1.\2'), # Double periods before complex table attrs
                (r'Source: (\[[^\]]*\]\([^)]*\))\.\.', r'Source: \1.'),                        # Double periods after markdown links
                (r'Source: ([^.]*)\.\.(\]\([^)]*\)\{#[^}]*\})', r'Source: \1.\2'),             # Double periods before figure closing
                (r'Source: ([^.]*)\.\.(\]\([^)]*\))', r'Source: \1.\2'),                       # Double periods before link closing
                (r'Source: ([^.]*)\.\.', r'Source: \1.'),                                      # General double periods
            ]

            for pattern, replacement in patterns_double_periods:
                new_content = re.sub(pattern, replacement, modified_content)
                if new_content != modified_content:
                    modified_content = new_content
                    file_modified = True

            # Fix double brackets
            new_content = re.sub(r'Source: \[\[@', r'Source: [@', modified_content)
            if new_content != modified_content:
                modified_content = new_content
                file_modified = True

            # Write back if modified
            if file_modified:
                self._write_file(file_path, modified_content)
                files_changed += 1

        self.print_success(f"Cleanup completed! Modified {files_changed} files.")
        return files_changed

    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.print_error(f"Error reading {file_path}: {e}")
            return None

    def _write_file(self, file_path: Path, content: str) -> bool:
        """Write file content safely"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            self.print_error(f"Error writing {file_path}: {e}")
            return False

    def generate_report(self, output_file: str = "source_analysis_report.json") -> bool:
        """Generate detailed JSON report"""
        self.print_status("📋 Generating detailed source report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.stats,
            'problems': self.problems,
            'files_analyzed': self.stats['total_files'],
            'recommendations': self._generate_recommendations()
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.print_success(f"Detailed report saved to: {output_file}")
            return True
        except Exception as e:
            self.print_error(f"Error generating report: {e}")
            return False

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        if self.stats['problematic_asterisk'] > 0:
            recommendations.append("Run cleanup to fix asterisk-wrapped sources")

        if self.stats['missing_periods'] > 0:
            recommendations.append("Add periods to company source citations")

        if self.stats['lowercase_sources'] > 0:
            recommendations.append("Capitalize 'source:' to 'Source:'")

        if self.stats['double_periods'] > 0:
            recommendations.append("Remove double periods from source citations")

        if self.stats['malformed_citations'] > 0:
            recommendations.append("Add brackets to academic citations")

        if not recommendations:
            recommendations.append("All source citations are properly formatted!")

        return recommendations

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Source Citation Checker and Cleaner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_sources.py --analyze                    # Analyze all files
  python check_sources.py --clean -d contents/core     # Clean specific directory
  python check_sources.py --full -f chapter.qmd       # Full analysis on one file
  python check_sources.py --analyze -d contents/core/ml_systems  # Analyze one chapter
        """
    )

    parser.add_argument('-a', '--analyze', action='store_true',
                        help='Analyze current source patterns')
    parser.add_argument('-p', '--problems', action='store_true',
                        help='Find problematic patterns')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Perform automatic cleanup')
    parser.add_argument('-r', '--report', action='store_true',
                        help='Generate detailed JSON report')
    parser.add_argument('-f', '--full', action='store_true',
                        help='Run full analysis (all options)')
    parser.add_argument('-d', '--directories', nargs='+', metavar='DIR',
                        help='Target specific directories (e.g., contents/core/ml_systems)')
    parser.add_argument('--files', nargs='+', metavar='FILE',
                        help='Target specific files (e.g., chapter.qmd)')
    parser.add_argument('--output', default='source_analysis_report.json',
                        help='Output file for report (default: source_analysis_report.json)')

    args = parser.parse_args()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return 1

    # Validate file/directory arguments
    if args.files and args.directories:
        print("Error: Cannot specify both --files and --directories")
        return 1

    checker = SourceChecker(target_directories=args.directories, target_files=args.files)

    if not checker.check_environment():
        return 1

    print("🔍 Source Citation Checker and Cleaner")
    print("=" * 40)

    # Show scope information
    if args.files:
        print(f"📁 Scope: Specific files ({len(args.files)} files)")
    elif args.directories:
        print(f"📁 Scope: Specific directories ({len(args.directories)} directories)")
    else:
        print("📁 Scope: All QMD files in contents/")
    print()

    try:
        if args.full:
            checker.analyze_sources()
            checker.find_problems()
            checker.generate_report(args.output)
        else:
            if args.analyze:
                checker.analyze_sources()
            if args.problems:
                checker.find_problems()
            if args.clean:
                files_changed = checker.perform_cleanup()
                print()
                print("Running post-cleanup analysis...")
                # Re-analyze after cleanup
                checker = SourceChecker(target_directories=args.directories, target_files=args.files)  # Reset state
                checker.analyze_sources()
            if args.report:
                checker.generate_report(args.output)

    except KeyboardInterrupt:
        checker.print_warning("\nOperation cancelled by user")
        return 1
    except Exception as e:
        checker.print_error(f"Unexpected error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
