#!/usr/bin/env python3
"""
Remove all footnotes from Quarto markdown (.qmd) files.

This script removes:
1. Inline footnote references like [^fn-name]
2. Footnote definitions like [^fn-name]: Definition text...
3. Multi-line footnote definitions that are indented
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def remove_inline_footnotes(text: str) -> str:
    """Remove inline footnote references like [^fn-name] from text."""
    # Pattern matches [^anything-here] where 'anything-here' doesn't contain ]
    pattern = r'\[^\^[^\]]+\]'
    return re.sub(pattern, '', text)


def remove_footnote_definitions(lines: List[str]) -> List[str]:
    """Remove footnote definitions from a list of lines."""
    cleaned_lines = []
    skip_mode = False
    
    for i, line in enumerate(lines):
        # Check if this line starts a footnote definition
        if re.match(r'^\[\^[^\]]+\]:', line):
            skip_mode = True
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
        
        # Keep this line
        cleaned_lines.append(line)
    
    return cleaned_lines


def process_qmd_file(file_path: Path) -> Tuple[bool, int, int]:
    """
    Process a single .qmd file to remove footnotes.
    
    Returns:
        Tuple of (was_modified, inline_refs_removed, definitions_removed)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
        
        # Count footnotes before processing
        inline_refs_before = len(re.findall(r'\[^\^[^\]]+\]', content))
        definitions_before = len([l for l in lines if re.match(r'^\[\^[^\]]+\]:', l)])
        
        # Remove inline references
        content_no_inline = remove_inline_footnotes(content)
        
        # Remove definitions (work with lines)
        lines_no_inline = content_no_inline.splitlines()
        cleaned_lines = remove_footnote_definitions(lines_no_inline)
        
        # Reconstruct content
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Only write if there were changes
        if cleaned_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
                # Ensure file ends with newline
                if cleaned_content and not cleaned_content.endswith('\n'):
                    f.write('\n')
            return True, inline_refs_before, definitions_before
        
        return False, 0, 0
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0, 0


def find_qmd_files(root_dir: Path) -> List[Path]:
    """Find all .qmd files in the directory tree."""
    return sorted(root_dir.rglob('*.qmd'))


def main():
    """Main function to process all .qmd files."""
    # Determine root directory
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        # Default to quarto directory
        root_dir = Path('/Users/VJ/GitHub/MLSysBook/quarto')
    
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist")
        sys.exit(1)
    
    print(f"Scanning for .qmd files in: {root_dir}")
    
    # Find all .qmd files
    qmd_files = find_qmd_files(root_dir)
    
    if not qmd_files:
        print("No .qmd files found")
        return
    
    print(f"Found {len(qmd_files)} .qmd files")
    print("-" * 60)
    
    # Process each file
    total_modified = 0
    total_inline_refs = 0
    total_definitions = 0
    
    for qmd_file in qmd_files:
        relative_path = qmd_file.relative_to(root_dir.parent)
        was_modified, inline_refs, definitions = process_qmd_file(qmd_file)
        
        if was_modified:
            total_modified += 1
            total_inline_refs += inline_refs
            total_definitions += definitions
            print(f"✓ {relative_path}")
            print(f"  - Removed {inline_refs} inline references")
            print(f"  - Removed {definitions} footnote definitions")
        else:
            print(f"  {relative_path} (no footnotes found)")
    
    # Summary
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Files processed: {len(qmd_files)}")
    print(f"  Files modified: {total_modified}")
    print(f"  Total inline references removed: {total_inline_refs}")
    print(f"  Total footnote definitions removed: {total_definitions}")
    
    if total_modified > 0:
        print(f"\n✓ Successfully removed all footnotes from {total_modified} files")
    else:
        print("\n✓ No footnotes found in any files")


if __name__ == "__main__":
    main()