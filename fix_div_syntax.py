#!/usr/bin/env python3
"""
Script to fix malformed fenced div syntax in Quarto files.
Converts ::: {layout-narrow} to :::: {layout-narrow} and matching closing tags.
"""

import os
import re
import glob

def fix_div_syntax(file_path):
    """Fix div syntax in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match the problematic div structure
    pattern = r'(::: \{layout-narrow\}\n::: \{\.column-margin\}\n.*?\n:::)\n\n(\\noindent\n!\[.*?\]\(.*?\)\n\n)(:::)'
    
    def replace_func(match):
        # Replace opening ::: with ::::
        opening_part = match.group(1).replace('::: {layout-narrow}', ':::: {layout-narrow}')
        opening_part = opening_part.replace('::: {.column-margin}', ':::: {.column-margin}')
        opening_part = opening_part.replace(':::', '::::')  # Fix the closing of column-margin
        
        middle_part = match.group(2)
        
        # Replace closing ::: with ::::
        closing_part = '::::'
        
        return opening_part + '\n\n' + middle_part + closing_part
    
    # Apply the fix
    new_content = re.sub(pattern, replace_func, content, flags=re.DOTALL)
    
    # Check if changes were made
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    """Fix div syntax in all affected files."""
    # Find all .qmd files in the contents directory
    pattern = '/Users/VJ/GitHub/MLSysBook/quarto/contents/**/*.qmd'
    files = glob.glob(pattern, recursive=True)
    
    fixed_files = []
    
    for file_path in files:
        if fix_div_syntax(file_path):
            fixed_files.append(file_path)
            print(f"Fixed: {file_path}")
    
    if fixed_files:
        print(f"\nFixed {len(fixed_files)} files:")
        for file_path in fixed_files:
            print(f"  - {os.path.basename(file_path)}")
    else:
        print("No files needed fixing.")

if __name__ == "__main__":
    main()
