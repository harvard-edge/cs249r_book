#!/usr/bin/env python3
"""
LaTeX Package Dependency Extractor

This script analyzes files to extract LaTeX package dependencies and generate
a list of required TeX Live packages and collections. It searches through
specified files to find all \\usepackage declarations and TikZ library usage.

The script uses tlmgr (TeX Live package manager) to map LaTeX package names
to their corresponding TeX Live packages and collections, making it easier
to install the correct dependencies for building LaTeX projects.

Usage:
    python update_texlive_packages.py [files...]

Output:
    Creates a package list file containing:
    - TeX Live collections that need to be installed
    - Individual packages not part of collections
    - Packages that couldn't be found (for manual review)

Dependencies:
    - tlmgr (TeX Live package manager)
    - PyYAML for parsing YAML files
"""

import argparse
import logging
import re
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LaTeXPackageExtractor:
    """Extract LaTeX package dependencies from various file types."""

    def __init__(self, quiet: bool = False):
        """
        Initialize the extractor.

        Args:
            quiet: If True, suppress verbose output
        """
        self.quiet = quiet
        self.logger = logging.getLogger(__name__)
        if quiet:
            self.logger.setLevel(logging.WARNING)

    def extract_from_file(self, file_path: Path) -> Tuple[Set[str], bool, bool]:
        """
        Extract LaTeX package dependencies from a file.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Tuple of (packages, has_tikz, has_pgfplots)
        """
        if not file_path.exists():
            self.logger.warning(f"File does not exist: {file_path}")
            return set(), False, False

        self.logger.info(f"Analyzing file: {file_path}")

        # Determine file type and extract accordingly
        if file_path.suffix.lower() in ['.yml', '.yaml']:
            return self._extract_from_yaml(file_path)
        else:
            return self._extract_from_text(file_path)

    def _extract_from_text(self, file_path: Path) -> Tuple[Set[str], bool, bool]:
        """Extract packages from plain text files (TEX, MD, etc.)."""
        packages = set()
        has_tikz = has_pgfplots = False

        try:
            content = file_path.read_text(encoding='utf-8')
            self.logger.debug(f"File size: {len(content)} characters")

            for line_num, line in enumerate(content.splitlines(), 1):
                packages.update(self._extract_packages_from_line(line, line_num))
                has_tikz |= "\\usetikzlibrary" in line
                has_pgfplots |= "\\usepgfplotslibrary" in line

            self.logger.info(f"Extracted {len(packages)} packages from {file_path.name}")
            return packages, has_tikz, has_pgfplots

        except UnicodeDecodeError:
            self.logger.warning(f"Could not decode {file_path} as UTF-8, skipping")
            return set(), False, False
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return set(), False, False

    def _extract_from_yaml(self, file_path: Path) -> Tuple[Set[str], bool, bool]:
        """Extract packages from YAML files by searching for LaTeX content."""
        packages = set()
        has_tikz = has_pgfplots = False

        try:
            content = file_path.read_text(encoding='utf-8')
            yml_data = yaml.safe_load(content)

            # Recursively search for LaTeX content in YAML
            latex_content = self._find_latex_in_yaml(yml_data)

            for line in latex_content:
                packages.update(self._extract_packages_from_line(line))
                has_tikz |= "\\usetikzlibrary" in line
                has_pgfplots |= "\\usepgfplotslibrary" in line

            self.logger.info(f"Extracted {len(packages)} packages from YAML {file_path.name}")
            return packages, has_tikz, has_pgfplots

        except yaml.YAMLError as e:
            self.logger.warning(f"Invalid YAML in {file_path}: {e}")
            return set(), False, False
        except Exception as e:
            self.logger.error(f"Error reading YAML {file_path}: {e}")
            return set(), False, False

    def _find_latex_in_yaml(self, data, path: str = "") -> List[str]:
        """Recursively find LaTeX content in YAML structure."""
        latex_lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and ("\\usepackage" in value or "\\usetikzlibrary" in value):
                    latex_lines.append(value)
                elif isinstance(value, list):
                    latex_lines.extend(self._find_latex_in_yaml(value, current_path))
                elif isinstance(value, dict):
                    latex_lines.extend(self._find_latex_in_yaml(value, current_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                if isinstance(item, str) and ("\\usepackage" in item or "\\usetikzlibrary" in item):
                    latex_lines.append(item)
                else:
                    latex_lines.extend(self._find_latex_in_yaml(item, current_path))

        return latex_lines

    def _extract_packages_from_line(self, line: str, line_num: Optional[int] = None) -> Set[str]:
        """Extract package names from a single line of LaTeX content."""
        packages = set()

        # Remove comments (everything after %)
        line = line.split('%')[0].strip()

        # Extract package names from \\usepackage declarations
        # Regex matches: \usepackage[options]{package1,package2}
        matches = re.findall(r'\\usepackage(?:\[[^\]]*\])?{([^}]+)}', line)

        for match in matches:
            # Split comma-separated packages and clean whitespace
            line_packages = [pkg.strip() for pkg in match.split(',') if pkg.strip()]
            packages.update(line_packages)

            if line_packages and not self.quiet:
                line_info = f"Line {line_num}" if line_num else "Content"
                self.logger.debug(f"{line_info}: Found packages: {line_packages}")

        return packages

class TeXLivePackageMapper:
    """Map LaTeX package names to TeX Live packages and collections."""

    def __init__(self, quiet: bool = False):
        """
        Initialize the mapper.

        Args:
            quiet: If True, suppress verbose output
        """
        self.quiet = quiet
        self.logger = logging.getLogger(__name__)
        if quiet:
            self.logger.setLevel(logging.WARNING)

        # Cache for package lookups to avoid repeated tlmgr calls
        self._package_cache = {}
        self._collection_cache = {}

    def find_package(self, component: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the TeX Live package that provides a given LaTeX component.

        Args:
            component: LaTeX package name (e.g., 'geometry', 'graphicx')

        Returns:
            Tuple of (package_name, collection_name)
        """
        self.logger.debug(f"Looking up package for component: {component}")

        try:
            # Search for the .sty file in TeX Live packages
            cmd = ["tlmgr", "search", "--file", f"/{component}.sty"]
            self.logger.debug(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10  # Reduced timeout to 10 seconds
            )

            if not result.stdout.strip():
                self.logger.debug(f"No tlmgr package found for {component}")
                return None, None

            # Extract package name from tlmgr output
            pkg = result.stdout.split(":")[0].strip()
            self.logger.debug(f"Found tlmgr package: {pkg}")

            # Get detailed information about the package
            info_cmd = ["tlmgr", "info", pkg]
            self.logger.debug(f"Running: {' '.join(info_cmd)}")

            info = subprocess.run(
                info_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10  # Reduced timeout to 10 seconds
            )

            # Extract collection information from package details
            coll_match = re.search(r"collection:\s*(\S+)", info.stdout)

            if coll_match:
                collection = coll_match.group(1)
                self.logger.debug(f"Package {pkg} belongs to collection: {collection}")
                return pkg, collection
            else:
                self.logger.debug(f"Package {pkg} is not part of a collection")
                return pkg, None

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout while looking up {component} (10s)")
            return None, None
        except FileNotFoundError:
            self.logger.error("tlmgr not found. Please install TeX Live.")
            return None, None
        except Exception as e:
            self.logger.error(f"Error looking up {component}: {e}")
            return None, None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract LaTeX package dependencies from files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default files (_quarto.yml, tex/header-includes.tex)
  %(prog)s -f file1.tex file2.yml            # Analyze specific files
  %(prog)s --files custom.tex                # Analyze a single custom file
  %(prog)s --output packages.txt             # Write to custom output file
  %(prog)s --quiet                           # Suppress verbose output
  %(prog)s --dry-run                         # Show what would be done without writing
  %(prog)s --include "*.tex" --include "*.yml"  # Use glob patterns to find files
        """
    )

    parser.add_argument(
        "-f", "--files",
        nargs="+",
        help="Files to analyze (default: _quarto.yml and tex/header-includes.tex)"
    )

    parser.add_argument(
        "--include",
        action="append",
        help="Glob pattern for files to include (can be used multiple times)"
    )

    parser.add_argument(
        "--exclude",
        action="append",
        help="Glob pattern for files to exclude (can be used multiple times)"
    )

    parser.add_argument(
        "--output",
        default="texlive_packages",
        help="Output file path (default: texlive_packages)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output (only show errors and final summary)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing the output file"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 2.0.0"
    )

    return parser.parse_args()

def get_files_to_analyze(args) -> List[Path]:
    """Get list of files to analyze based on arguments."""
    files = []

    # If specific files provided with -f/--files, use them
    if args.files:
        for file_path in args.files:
            path = Path(file_path)
            if path.exists():
                files.append(path)
            else:
                logger.warning(f"File not found: {file_path}")
        return files

    # If include patterns are specified, use glob patterns
    if args.include:
        exclude_patterns = args.exclude or [
            "_build/*", "_site/*", "_book/*",
            "node_modules/*", ".git/*"
        ]

        for pattern in args.include:
            for file_path in Path(".").glob(pattern):
                # Check if file should be excluded
                should_exclude = any(
                    file_path.match(exclude_pattern)
                    for exclude_pattern in exclude_patterns
                )

                if not should_exclude and file_path.is_file():
                    files.append(file_path)

        return sorted(set(files))  # Remove duplicates

    # Default: only look at the two specific files
    default_files = [
        Path("_quarto.yml"),
        Path("tex/header-includes.tex")
    ]

    for file_path in default_files:
        if file_path.exists():
            files.append(file_path)
        else:
            logger.debug(f"Default file not found: {file_path}")

    return files

def main():
    """Main function."""
    args = parse_arguments()

    # Set up logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if not args.quiet:
        logger.info("🚀 Starting LaTeX package dependency extraction...")
        if args.dry_run:
            logger.info("🔍 DRY RUN MODE - No files will be written")

    # Get files to analyze
    files = get_files_to_analyze(args)

    if not files:
        logger.warning("No files found to analyze")
        return 1

    if not args.quiet:
        logger.info(f"📂 Found {len(files)} files to analyze:")
        for file_path in files:
            logger.info(f"   • {file_path}")

    # Initialize extractor and mapper
    extractor = LaTeXPackageExtractor(quiet=args.quiet)
    mapper = TeXLivePackageMapper(quiet=args.quiet)

    # Extract packages from all files
    all_packages = set()
    has_tikz = has_pgfplots = False

    if not args.quiet:
        logger.info("📂 PHASE 1: Extracting packages from files")
        logger.info("-" * 40)

    for i, file_path in enumerate(files, 1):
        if not args.quiet:
            logger.info(f"📄 Processing file {i}/{len(files)}: {file_path}")

        packages, tikz, pgfplots = extractor.extract_from_file(file_path)
        all_packages.update(packages)
        has_tikz |= tikz
        has_pgfplots |= pgfplots

        if not args.quiet:
            logger.info(f"   ✅ Found {len(packages)} packages in {file_path.name}")

    # Add special packages based on TikZ usage
    if has_tikz:
        all_packages.add("pgf")
        if not args.quiet:
            logger.info("➕ Added 'pgf' due to TikZ library usage")
    if has_pgfplots:
        all_packages.add("pgfplots")
        if not args.quiet:
            logger.info("➕ Added 'pgfplots' due to PGFPlots library usage")

    if not args.quiet:
        logger.info(f"📋 Found {len(all_packages)} unique packages:")
        for pkg in sorted(all_packages):
            logger.info(f"   • {pkg}")

    # Map to TeX Live packages
    collections = set()
    explicit_packages = set()
    missing_packages = set()

    if not args.quiet:
        logger.info(f"\n🔍 PHASE 2: Looking up TeX Live packages")
        logger.info("-" * 40)

    sorted_packages = sorted(all_packages)
    for i, package in enumerate(sorted_packages, 1):
        if not args.quiet:
            logger.info(f"🔎 Looking up package {i}/{len(sorted_packages)}: {package}")

        pkg, collection = mapper.find_package(package)

        if collection:
            collections.add(collection)
            if not args.quiet:
                logger.info(f"   ✅ {package} → collection: {collection}")
        elif pkg:
            explicit_packages.add(pkg)
            if not args.quiet:
                logger.info(f"   ✅ {package} → package: {pkg}")
        else:
            missing_packages.add(package)
            if not args.quiet:
                logger.info(f"   ❌ {package} → not found")

    # Add essential collections that are commonly needed
    essential_collections = {
        "collection-fontsrecommended",  # Base 35 PostScript fonts, Latin Modern, Times, etc.
    }

    # Add essential collections to our set
    if not args.quiet and essential_collections - collections:
        logger.info("➕ Adding essential collections automatically:")
        for collection in sorted(essential_collections - collections):
            logger.info(f"   • {collection}")
    collections.update(essential_collections)

    # Generate output
    if not args.dry_run:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("# Auto-generated TeX Live package list\n\n")

            f.write("# Collections:\n")
            for collection in sorted(collections):
                f.write(f"{collection}\n")

            if explicit_packages:
                f.write("\n# Explicit packages (not in collections):\n")
                for pkg in sorted(explicit_packages):
                    f.write(f"{pkg}\n")

            if missing_packages:
                f.write("\n# Not found via tlmgr (check manually):\n")
                for pkg in sorted(missing_packages):
                    f.write(f"# {pkg}\n")

    # Display summary
    if not args.quiet:
        logger.info("✅ FINAL SUMMARY:")
        logger.info(f"   📚 Collections: {len(collections)}")
        for collection in sorted(collections):
            logger.info(f"      • {collection}")
        logger.info(f"   📦 Explicit packages: {len(explicit_packages)}")
        for pkg in sorted(explicit_packages):
            logger.info(f"      • {pkg}")
        logger.info(f"   ❓ Missing/unknown: {len(missing_packages)}")
        for pkg in sorted(missing_packages):
            logger.info(f"      • {pkg}")

        if args.dry_run:
            logger.info("🔍 DRY RUN COMPLETED - No files were written")
        else:
            logger.info(f"🎉 Successfully wrote {args.output}")
    else:
        # Quiet mode: only show essential information
        print(f"Collections: {len(collections)}, Packages: {len(explicit_packages)}, Missing: {len(missing_packages)}")
        if not args.dry_run:
            print(f"Output written to: {args.output}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
