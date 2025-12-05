#!/usr/bin/env python3
"""
EPUB Validator Script

Validates EPUB files for common issues including:
- XML parsing errors (double-hyphen in comments)
- CSS variable issues (--variable syntax)
- Malformed HTML/XHTML
- Missing required files
- Structural validation

Uses epubcheck (official EPUB validator) if available, with custom checks for project-specific issues.

Installation:
    # Install epubcheck (recommended)
    brew install epubcheck  # macOS
    # OR download from: https://github.com/w3c/epubcheck/releases

Usage:
    python3 validate_epub.py <path_to_epub_file>
    python3 validate_epub.py quarto/_build/epub/Machine-Learning-Systems.epub
    python3 validate_epub.py --quick <path_to_epub_file>  # Skip epubcheck
"""

import sys
import zipfile
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
import tempfile
import shutil
import subprocess
import json


class EPUBValidator:
    """Validates EPUB files for common issues."""

    def __init__(self, epub_path: str, use_epubcheck: bool = True):
        self.epub_path = Path(epub_path)
        self.errors: List[Tuple[str, str, str]] = []  # (severity, category, message)
        self.warnings: List[Tuple[str, str, str]] = []
        self.temp_dir = None
        self.use_epubcheck = use_epubcheck

    def validate(self) -> bool:
        """Run all validation checks. Returns True if no errors found."""
        print(f"🔍 Validating EPUB: {self.epub_path.name}\n")

        if not self.epub_path.exists():
            self._add_error("CRITICAL", "File", f"EPUB file not found: {self.epub_path}")
            return False

        # Run epubcheck first if available
        if self.use_epubcheck:
            self._run_epubcheck()

        # Extract EPUB to temp directory
        self.temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(self.epub_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
        except zipfile.BadZipFile:
            self._add_error("CRITICAL", "Structure", "Invalid ZIP/EPUB file")
            return False

        # Run custom validation checks (project-specific)
        print("\n📋 Running custom validation checks...")
        self._check_mimetype()
        self._check_container_xml()
        self._check_css_variables()
        self._check_xml_comments()
        self._check_common_xhtml_errors()
        self._check_xhtml_validity()
        self._check_opf_structure()

        # Print results
        self._print_results()

        # Cleanup
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)

        return len(self.errors) == 0

    def _add_error(self, severity: str, category: str, message: str):
        """Add an error to the list."""
        self.errors.append((severity, category, message))

    def _add_warning(self, severity: str, category: str, message: str):
        """Add a warning to the list."""
        self.warnings.append((severity, category, message))

    def _run_epubcheck(self):
        """Run epubcheck validator if available."""
        print("🔧 Running epubcheck (official EPUB validator)...\n")

        try:
            # Try to run epubcheck
            result = subprocess.run(
                ['epubcheck', '--json', '-', str(self.epub_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                print("✅ epubcheck: PASS\n")
                return

            # Parse JSON output
            try:
                output = json.loads(result.stdout) if result.stdout else {}
                messages = output.get('messages', [])

                error_count = 0
                warning_count = 0

                for msg in messages:
                    severity = msg.get('severity', 'INFO')
                    message_text = msg.get('message', 'Unknown error')
                    locations = msg.get('locations', [])

                    location_str = ""
                    if locations:
                        loc = locations[0]
                        path = loc.get('path', '')
                        line = loc.get('line', '')
                        col = loc.get('column', '')
                        location_str = f"{path}:{line}:{col}" if line else path

                    full_message = f"{location_str}: {message_text}" if location_str else message_text

                    if severity == 'ERROR' or severity == 'FATAL':
                        self._add_error("ERROR", "epubcheck", full_message)
                        error_count += 1
                    elif severity == 'WARNING':
                        self._add_warning("WARNING", "epubcheck", full_message)
                        warning_count += 1

                print(f"❌ epubcheck found {error_count} errors, {warning_count} warnings\n")

            except json.JSONDecodeError:
                # Fallback to text parsing
                if result.stderr:
                    print(f"⚠️  epubcheck output (text mode):\n{result.stderr}\n")
                    self._add_warning("WARNING", "epubcheck", "Could not parse JSON output")

        except FileNotFoundError:
            print("⚠️  epubcheck not found. Install with: brew install epubcheck")
            print("   Skipping official EPUB validation.\n")
        except subprocess.TimeoutExpired:
            self._add_error("ERROR", "epubcheck", "Validation timed out after 120 seconds")
        except Exception as e:
            print(f"⚠️  Could not run epubcheck: {e}\n")

    def _check_mimetype(self):
        """Check for valid mimetype file."""
        mimetype_path = Path(self.temp_dir) / "mimetype"
        if not mimetype_path.exists():
            self._add_error("ERROR", "Structure", "Missing mimetype file")
            return

        content = mimetype_path.read_text().strip()
        if content != "application/epub+zip":
            self._add_error("ERROR", "Structure", f"Invalid mimetype: {content}")

    def _check_container_xml(self):
        """Check for valid META-INF/container.xml."""
        container_path = Path(self.temp_dir) / "META-INF" / "container.xml"
        if not container_path.exists():
            self._add_error("ERROR", "Structure", "Missing META-INF/container.xml")
            return

        try:
            tree = ET.parse(container_path)
            root = tree.getroot()
            # Check for rootfile element
            rootfiles = root.findall(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
            if not rootfiles:
                self._add_error("ERROR", "Structure", "No rootfile found in container.xml")
        except ET.ParseError as e:
            self._add_error("ERROR", "XML", f"Invalid container.xml: {e}")

    def _check_css_variables(self):
        """Check CSS files for problematic CSS custom properties."""
        print("📝 Checking CSS files for CSS variables...")

        css_files = list(Path(self.temp_dir).rglob("*.css"))

        for css_file in css_files:
            rel_path = css_file.relative_to(self.temp_dir)
            content = css_file.read_text()

            # Check for CSS variable declarations (--variable-name)
            var_declarations = re.findall(r'^\s*(--[\w-]+)\s*:', content, re.MULTILINE)
            if var_declarations:
                self._add_error("ERROR", "CSS",
                    f"{rel_path}: Found CSS variable declarations: {', '.join(var_declarations[:5])}")

            # Check for CSS variable usage (var(--variable-name))
            var_usage = re.findall(r'var\((--[\w-]+)\)', content)
            if var_usage:
                self._add_error("ERROR", "CSS",
                    f"{rel_path}: Found CSS variable usage: {', '.join(set(var_usage[:5]))}")

            # Count total double-hyphens (for reference)
            double_hyphen_count = content.count('--')
            if double_hyphen_count > 0:
                # Check if they're only in comments
                without_comments = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                double_hyphens_in_code = without_comments.count('--')

                if double_hyphens_in_code > 0:
                    self._add_warning("WARNING", "CSS",
                        f"{rel_path}: Found {double_hyphens_in_code} double-hyphens outside comments")
                else:
                    print(f"  ✓ {rel_path}: {double_hyphen_count} double-hyphens (all in comments)")

    def _check_xml_comments(self):
        """Check for XML comment violations (double-hyphen in comments)."""
        print("\n📝 Checking for XML comment violations...")

        xml_files = list(Path(self.temp_dir).rglob("*.xhtml")) + \
                    list(Path(self.temp_dir).rglob("*.xml")) + \
                    list(Path(self.temp_dir).rglob("*.opf"))

        # Pattern to find comments with double-hyphens inside them
        # XML spec prohibits -- inside comments
        comment_pattern = re.compile(r'<!--.*?--.*?-->', re.DOTALL)

        for xml_file in xml_files:
            rel_path = xml_file.relative_to(self.temp_dir)
            try:
                content = xml_file.read_text()
                matches = comment_pattern.findall(content)

                if matches:
                    # Find line numbers
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if '--' in line and '<!--' in content[:content.index(line) if line in content else 0]:
                            self._add_error("ERROR", "XML",
                                f"{rel_path}:{i}: Comment contains '--' (double-hyphen)")
            except Exception as e:
                self._add_warning("WARNING", "XML", f"{rel_path}: Could not check comments: {e}")

    def _check_common_xhtml_errors(self):
        """Check for common XHTML/XML errors that plague EPUB files."""
        print("\n📝 Checking for common XHTML errors...")

        xhtml_files = list(Path(self.temp_dir).rglob("*.xhtml"))

        for xhtml_file in xhtml_files:
            rel_path = xhtml_file.relative_to(self.temp_dir)
            try:
                content = xhtml_file.read_text()
                lines = content.split('\n')

                for i, line in enumerate(lines, 1):
                    # Check for unclosed tags (common patterns)
                    if '<br>' in line and '<br/>' not in line and '<br />' not in line:
                        self._add_warning("WARNING", "XHTML",
                            f"{rel_path}:{i}: Use self-closing <br/> instead of <br>")

                    if '<img ' in line and not '/>' in line[line.index('<img '):]:
                        self._add_warning("WARNING", "XHTML",
                            f"{rel_path}:{i}: <img> tag should be self-closing")

                    if '<hr>' in line and '<hr/>' not in line and '<hr />' not in line:
                        self._add_warning("WARNING", "XHTML",
                            f"{rel_path}:{i}: Use self-closing <hr/> instead of <hr>")

                    # Check for unescaped ampersands (except entities)
                    if '&' in line:
                        # Simple check for unescaped &
                        if re.search(r'&(?![a-zA-Z]+;|#\d+;|#x[0-9a-fA-F]+;)', line):
                            self._add_warning("WARNING", "XHTML",
                                f"{rel_path}:{i}: Possibly unescaped ampersand (&)")

                    # Check for <  > without proper escaping
                    if re.search(r'<(?![a-zA-Z/!?])', line):
                        self._add_warning("WARNING", "XHTML",
                            f"{rel_path}:{i}: Possibly unescaped < character")

                    # Check for attributes without quotes
                    if re.search(r'<\w+[^>]*\s+\w+=\w+[^"\']', line):
                        self._add_warning("WARNING", "XHTML",
                            f"{rel_path}:{i}: Attribute values should be quoted")

            except Exception as e:
                self._add_warning("WARNING", "XHTML",
                    f"{rel_path}: Could not check for common errors: {e}")

    def _check_xhtml_validity(self):
        """Check XHTML files for basic validity."""
        print("\n📝 Checking XHTML validity...")

        xhtml_files = list(Path(self.temp_dir).rglob("*.xhtml"))

        for xhtml_file in xhtml_files:
            rel_path = xhtml_file.relative_to(self.temp_dir)
            try:
                # Try to parse as XML (XHTML should be well-formed XML)
                ET.parse(xhtml_file)
                print(f"  ✓ {rel_path}: Valid XHTML")
            except ET.ParseError as e:
                self._add_error("ERROR", "XHTML", f"{rel_path}: Parse error - {e}")

    def _check_opf_structure(self):
        """Check OPF file structure."""
        print("\n📝 Checking OPF structure...")

        opf_files = list(Path(self.temp_dir).rglob("*.opf"))

        if not opf_files:
            self._add_error("ERROR", "Structure", "No OPF file found")
            return

        for opf_file in opf_files:
            rel_path = opf_file.relative_to(self.temp_dir)
            try:
                tree = ET.parse(opf_file)
                root = tree.getroot()

                # Check for required elements
                namespaces = {'opf': 'http://www.idpf.org/2007/opf'}

                metadata = root.find('.//opf:metadata', namespaces)
                manifest = root.find('.//opf:manifest', namespaces)
                spine = root.find('.//opf:spine', namespaces)

                if metadata is None:
                    self._add_error("ERROR", "OPF", f"{rel_path}: Missing metadata element")
                if manifest is None:
                    self._add_error("ERROR", "OPF", f"{rel_path}: Missing manifest element")
                if spine is None:
                    self._add_error("ERROR", "OPF", f"{rel_path}: Missing spine element")
                else:
                    print(f"  ✓ {rel_path}: Valid OPF structure")

            except ET.ParseError as e:
                self._add_error("ERROR", "OPF", f"{rel_path}: Parse error - {e}")

    def _print_results(self):
        """Print validation results."""
        print("\n" + "="*70)
        print("📊 VALIDATION RESULTS")
        print("="*70)

        if not self.errors and not self.warnings:
            print("\n✅ SUCCESS: No issues found!")
            print(f"   {self.epub_path.name} is valid")
            return

        if self.errors:
            print(f"\n❌ ERRORS FOUND: {len(self.errors)}")
            print("-" * 70)
            for severity, category, message in self.errors:
                print(f"  [{severity}] [{category}] {message}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            print("-" * 70)
            for severity, category, message in self.warnings:
                print(f"  [{severity}] [{category}] {message}")

        print("\n" + "="*70)
        if self.errors:
            print("❌ VALIDATION FAILED")
        else:
            print("✅ VALIDATION PASSED (with warnings)")
        print("="*70)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 validate_epub.py [--quick] <path_to_epub_file>")
        print("\nOptions:")
        print("  --quick    Skip epubcheck validation (faster, custom checks only)")
        print("\nExamples:")
        print("  python3 validate_epub.py quarto/_build/epub/Machine-Learning-Systems.epub")
        print("  python3 validate_epub.py --quick quarto/_build/epub/Machine-Learning-Systems.epub")
        sys.exit(1)

    # Parse arguments
    use_epubcheck = True
    epub_path = None

    for arg in sys.argv[1:]:
        if arg == '--quick':
            use_epubcheck = False
        elif not epub_path:
            epub_path = arg

    if not epub_path:
        print("Error: No EPUB file specified")
        sys.exit(1)

    validator = EPUBValidator(epub_path, use_epubcheck=use_epubcheck)

    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
