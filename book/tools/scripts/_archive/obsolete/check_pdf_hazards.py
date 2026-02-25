#!/usr/bin/env python3
r"""
Check for markup patterns that break PDF/LaTeX rendering in QMD files.

Catches non-Python rendering hazards:
  1. Quad asterisks (malformed bold/italic): ****text**
  2. Footnotes inside table cells: | text[^fn-xxx] |
  3. Non-standard arithmetic intensity units (should be FLOPs/byte)

Note: Inline Python checks are handled separately by check_inline_python.py.

Usage:
  python check_pdf_hazards.py quarto/contents/vol1/
  python check_pdf_hazards.py quarto/contents/vol1/training/training.qmd

Exit codes:
  0 - No issues found
  1 - Issues found
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Issue:
    """A single PDF rendering hazard."""
    line_num: int
    check: str
    severity: str
    message: str
    context: str


CHECKS = {
    'quad_asterisks': {
        'regex': re.compile(r'\*{4,}'),
        'severity': 'warning',
        'message': 'Quad asterisks — likely malformed bold/italic',
    },
    'footnote_in_table': {
        'regex': re.compile(r'^\|.*\[\^fn-[^\]]+\].*\|'),
        'severity': 'warning',
        'message': 'Footnote in table cell — may break PDF rendering',
    },
    'inconsistent_arith_units': {
        'regex': re.compile(r'(?:Ops/Byte|Ops/byte|ops/byte|FLOPS/byte|FLOPS per byte|FLOPs per byte)'),
        'severity': 'warning',
        'message': 'Non-standard arithmetic intensity unit — should be "FLOPs/byte"',
    },
    'tflops_vs_TFLOPS': {
        # TFLOPs in prose context (throughput rate) should be TFLOPS
        # Matches "TFLOPs" when followed by rate-indicating words, NOT inside code blocks
        'regex': re.compile(
            r'\b(?:TFLOPs|PFLOPs|GFLOPs)\b'
            r'(?=\s+(?:peak|sustained|effective|theoretical|FP\d|without|dense|on\s))'
        ),
        'severity': 'warning',
        'message': 'TFLOPs/PFLOPs/GFLOPs used for throughput rate — should be TFLOPS/PFLOPS/GFLOPS (ops/second)',
    },
    'flops_mfu_casing': {
        # "Model FLOPS Utilization" should be "Model FLOPs Utilization"
        'regex': re.compile(r'Model FLOPS Utilization|model FLOPS utilization'),
        'severity': 'warning',
        'message': '"Model FLOPS Utilization" should be "Model FLOPs Utilization" (MFU measures operations, not rate)',
    },
}


def check_file(filepath: Path) -> list[Issue]:
    """Check a single QMD file for PDF rendering hazards."""
    lines = filepath.read_text().split('\n')
    issues: list[Issue] = []

    in_code_block = False
    for i, line in enumerate(lines):
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        for check_name, check_info in CHECKS.items():
            for match in check_info['regex'].finditer(line):
                ctx = match.group(0)
                if len(ctx) > 60:
                    ctx = ctx[:57] + '...'
                issues.append(Issue(
                    line_num=i + 1,
                    check=check_name,
                    severity=check_info['severity'],
                    message=check_info['message'],
                    context=ctx,
                ))

    return issues


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Check for markup patterns that break PDF rendering',
    )
    parser.add_argument('paths', nargs='*', help='Files or directories to check')
    parser.add_argument('--errors-only', '-e', action='store_true')
    parser.add_argument('--summary', '-s', action='store_true')

    args = parser.parse_args()
    if not args.paths:
        parser.print_help()
        return 0

    all_issues: list[tuple[Path, Issue]] = []

    for path_str in args.paths:
        path = Path(path_str)
        files = [path] if path.is_file() else sorted(path.rglob('*.qmd'))
        for f in files:
            if f.suffix != '.qmd':
                continue
            for issue in check_file(f):
                all_issues.append((f, issue))

    if args.errors_only:
        all_issues = [(p, i) for p, i in all_issues if i.severity == 'error']

    files_hit = sorted(set(p for p, _ in all_issues))
    errors = sum(1 for _, i in all_issues if i.severity == 'error')
    warnings = sum(1 for _, i in all_issues if i.severity == 'warning')

    if args.summary:
        print(f"Files with issues: {len(files_hit)}")
        print(f"Errors: {errors}  Warnings: {warnings}")
    else:
        cur = None
        for filepath, issue in all_issues:
            if filepath != cur:
                if cur is not None:
                    print()
                print(f"{filepath}:")
                cur = filepath
            icon = '❌' if issue.severity == 'error' else '⚠️'
            print(f"  {icon} L{issue.line_num}: {issue.message}")
            print(f"     → {issue.context}")

    if all_issues:
        print(f"\n{'❌' if errors else '⚠️'} {errors} error(s), {warnings} warning(s) in {len(files_hit)} file(s).")
        return 1

    print("✓ No PDF rendering hazards found.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
