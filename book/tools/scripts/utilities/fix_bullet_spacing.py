#!/usr/bin/env python3
"""
Fix bullet list spacing in QMD files.

Ensures there's a blank line before bullet lists start.
Pattern: Text ending with colon followed directly by bullet should have blank line.
"""

import re
import sys
from pathlib import Path


def fix_bullet_spacing(content: str) -> tuple[str, int]:
    """
    Fix bullet lists that are missing blank line before them.
    
    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        result.append(line)
        
        # Check if this line ends with colon (intro text) and next line is bullet
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            # Line ends with : (but not in code block markers or URLs)
            if (line.rstrip().endswith(':') and 
                not line.strip().startswith('```') and
                not line.strip().startswith('#|') and
                not '://' in line and
                not line.strip().startswith('def ') and
                not line.strip().startswith('class ') and
                # Next line is a bullet
                (next_line.startswith('*   ') or 
                 next_line.startswith('-   ') or
                 next_line.startswith('- ') or
                 re.match(r'^\d+\.  ', next_line))):
                # Add blank line after this line (will be inserted before next)
                result.append('')
                fixes += 1
    
    return '\n'.join(result), fixes


def process_file(filepath: Path, dry_run: bool = False) -> int:
    """Process a single file. Returns number of fixes."""
    content = filepath.read_text()
    fixed_content, fixes = fix_bullet_spacing(content)
    
    if fixes > 0:
        print(f"{filepath}: {fixes} fix(es)")
        if not dry_run:
            filepath.write_text(fixed_content)
    
    return fixes


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fix bullet list spacing in QMD files')
    parser.add_argument('paths', nargs='*', default=['.'], help='Files or directories to process')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Show what would be fixed without making changes')
    args = parser.parse_args()
    
    total_fixes = 0
    
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.qmd':
            total_fixes += process_file(path, args.dry_run)
        elif path.is_dir():
            for qmd_file in path.rglob('*.qmd'):
                total_fixes += process_file(qmd_file, args.dry_run)
    
    print(f"\nTotal: {total_fixes} fix(es)" + (" (dry run)" if args.dry_run else ""))
    return 0 if total_fixes == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
