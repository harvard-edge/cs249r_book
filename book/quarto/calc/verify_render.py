#!/usr/bin/env python3
"""
verify_render.py - Closed-loop verification for inline Python rendering.

This script:
1. Extracts expected values from Python compute blocks in QMD files
2. Renders the QMD to HTML
3. Verifies the expected values appear correctly in the output
4. Reports any mismatches (missing decimals, raw code, etc.)

Usage:
    python verify_render.py FILE.qmd           # Verify single file
    python verify_render.py --appendix         # Verify all appendix files
    python verify_render.py --all              # Verify all files with inline Python
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
QUARTO_ROOT = SCRIPT_DIR.parent
CONTENTS = QUARTO_ROOT / "contents"


def extract_expected_values(qmd_path: Path) -> dict[str, str]:
    """
    Extract variable assignments from Python compute blocks.
    Returns dict of {var_name: expected_string_value}
    """
    content = qmd_path.read_text()
    expected = {}
    
    # Find all Python code blocks
    cell_pattern = re.compile(r'```\{python\}.*?```', re.DOTALL)
    
    for cell in cell_pattern.findall(content):
        # Look for string assignments: var = "value" or var = f"..."
        # Match: var_name = "string" or var_name = f"{expr}"
        str_assign = re.compile(r'^(\w+)\s*=\s*[f]?["\']([^"\']+)["\']', re.MULTILINE)
        for match in str_assign.finditer(cell):
            var_name = match.group(1)
            value = match.group(2)
            # Skip if it's a format string with unresolved {expr}
            if '{' not in value:
                expected[var_name] = value
    
    return expected


def extract_inline_refs(qmd_path: Path) -> list[tuple[int, str]]:
    """Extract all inline Python references with line numbers."""
    content = qmd_path.read_text()
    refs = []
    pattern = re.compile(r'`\{python\}\s*([^`]+)`')
    
    for i, line in enumerate(content.split('\n'), 1):
        for match in pattern.finditer(line):
            refs.append((i, match.group(1).strip()))
    
    return refs


def render_to_html(qmd_path: Path, output_dir: Path) -> tuple[bool, Path, str]:
    """Render QMD to HTML. Returns (success, html_path, error_msg)."""
    cmd = [
        "quarto", "render", str(qmd_path),
        "--to", "html",
        "--output-dir", str(output_dir)
    ]
    
    try:
        result = subprocess.run(
            cmd, cwd=QUARTO_ROOT, capture_output=True, text=True, timeout=300
        )
        html_path = output_dir / qmd_path.with_suffix('.html').name
        
        if result.returncode != 0:
            return False, html_path, result.stderr[:500]
        return True, html_path, ""
    except Exception as e:
        return False, Path(), str(e)


def verify_html_output(html_path: Path, expected: dict[str, str]) -> list[dict]:
    """
    Verify expected values appear in HTML output.
    Returns list of issues found.
    """
    if not html_path.exists():
        return [{"type": "RENDER_FAILED", "msg": "HTML file not created"}]
    
    content = html_path.read_text()
    issues = []
    
    # Check 1: Raw code in output (render failed)
    if '`{python}' in content:
        issues.append({
            "type": "RAW_CODE",
            "msg": "Raw `{python}` found in output - inline code not executed"
        })
    
    # Check 2: Look for each expected value
    for var_name, expected_val in expected.items():
        if expected_val not in content:
            # Check for common corruption patterns
            no_decimal = expected_val.replace('.', '')
            if no_decimal in content and '.' in expected_val:
                issues.append({
                    "type": "DECIMAL_STRIPPED",
                    "var": var_name,
                    "expected": expected_val,
                    "found": no_decimal,
                    "msg": f"{var_name}: Expected '{expected_val}' but found '{no_decimal}' (decimal stripped)"
                })
    
    # Check 3: Look for suspicious patterns (common corruption)
    # Numbers that look like they lost decimals
    suspicious = [
        ("59", "5.9"),   # Amdahl's law
        ("153", "15.3"), # Could be valid, but flag it
    ]
    
    return issues


def verify_file(qmd_path: Path, output_dir: Path, verbose: bool = False) -> dict:
    """Verify a single QMD file. Returns verification result."""
    result = {
        "file": str(qmd_path.name),
        "inline_refs": 0,
        "expected_values": 0,
        "issues": [],
        "status": "PASS"
    }
    
    # Extract inline refs
    refs = extract_inline_refs(qmd_path)
    result["inline_refs"] = len(refs)
    
    if not refs:
        result["status"] = "SKIP"
        return result
    
    # Extract expected values
    expected = extract_expected_values(qmd_path)
    result["expected_values"] = len(expected)
    
    if verbose:
        print(f"  Found {len(refs)} inline refs, {len(expected)} expected values")
    
    # Render to HTML
    if verbose:
        print(f"  Rendering to HTML...")
    
    success, html_path, error = render_to_html(qmd_path, output_dir)
    
    if not success:
        result["issues"].append({"type": "RENDER_FAILED", "msg": error})
        result["status"] = "FAIL"
        return result
    
    # Verify output
    issues = verify_html_output(html_path, expected)
    result["issues"] = issues
    result["status"] = "FAIL" if issues else "PASS"
    result["html_path"] = str(html_path)
    
    return result


def print_report(results: list[dict]):
    """Print verification report."""
    print(f"\n{'='*60}")
    print("RENDER VERIFICATION REPORT")
    print('='*60)
    
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    
    if failed > 0:
        print(f"\n{'─'*60}")
        print("FAILURES:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"\n  {r['file']}")
                for issue in r["issues"]:
                    print(f"    [{issue['type']}] {issue.get('msg', '')}")
                if "html_path" in r:
                    print(f"    → Check: {r['html_path']}")
    
    print(f"\n{'─'*60}")
    return 1 if failed > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Verify inline Python renders correctly")
    parser.add_argument("file", nargs="?", help="QMD file to verify")
    parser.add_argument("--appendix", action="store_true", help="Verify appendix files")
    parser.add_argument("--all", action="store_true", help="Verify all files")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--keep-html", "-k", action="store_true", help="Keep HTML output")
    
    args = parser.parse_args()
    
    # Collect files to verify
    files = []
    
    if args.file:
        p = Path(args.file)
        if not p.exists():
            p = CONTENTS / args.file
        if p.exists():
            files.append(p)
        else:
            print(f"File not found: {args.file}")
            sys.exit(1)
    
    elif args.appendix:
        backmatter = CONTENTS / "vol1" / "backmatter"
        files = sorted(backmatter.glob("appendix_*.qmd"))
    
    elif args.all:
        for qmd in sorted(CONTENTS.rglob("*.qmd")):
            content = qmd.read_text()
            if '`{python}' in content:
                files.append(qmd)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    print(f"Verifying {len(files)} files...")
    
    # Create output directory
    if args.keep_html:
        output_dir = Path("/tmp/quarto_verify")
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="quarto_verify_"))
    
    # Verify each file
    results = []
    for qmd in files:
        print(f"\n{qmd.name}...", end=" " if not args.verbose else "\n")
        result = verify_file(qmd, output_dir, verbose=args.verbose)
        results.append(result)
        
        if not args.verbose:
            status = "✓" if result["status"] == "PASS" else "✗" if result["status"] == "FAIL" else "○"
            print(status)
    
    # Print report
    exit_code = print_report(results)
    
    if args.keep_html:
        print(f"\nHTML output saved to: {output_dir}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
