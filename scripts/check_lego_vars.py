#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

def check_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    errors = []
    
    # Find all python blocks containing "# ┌── LEGO"
    blocks = re.findall(r'```{python}.*?# ┌── LEGO.*?```', content, flags=re.DOTALL)
    
    for block in blocks:
        class_match = re.search(r'class\s+([a-zA-Z0-9_]+)', block)
        if not class_match:
            continue
        class_name = class_match.group(1)

        output_match = re.search(r'# ┌── 4\. OUTPUT.*?(?=\n(?:# ┌──|\Z|```))', block, flags=re.DOTALL)
        if not output_match:
            errors.append(f"Structure Error: Missing '# ┌── 4. OUTPUT' section in class '{class_name}'. Every LEGO block must export variables in this section.")
            continue
        
        output_section = output_match.group(0)
        vars_assigned = re.findall(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', output_section, flags=re.MULTILINE)
        
        for var in vars_assigned:
            if var.startswith('_') or var == 'RS': # Skip private/intermediate vars and aliases like RS
                continue
                
            full_ref = f"{class_name}.{var}"
            
            # If the exact `Class.var` is missing
            if full_ref not in content:
                # Check if it's used with an alias, e.g., `RS.var_name`
                alias_pattern = r'\b[a-zA-Z0-9_]+\.' + re.escape(var) + r'\b'
                if not re.search(alias_pattern, content):
                    # Check if it's an intermediate variable used to build ANOTHER string in the OUTPUT section.
                    if output_section.count(var) <= 1:
                        errors.append(f"Dead Code: Variable '{full_ref}' is exported but never used in the prose.")

    return errors

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check for unused LEGO variables.")
    parser.add_argument('files', nargs='*', help='Files to check')
    args = parser.parse_args()
    
    files = args.files
    if not files:
        base_dir = Path("book/quarto/contents")
        if base_dir.exists():
            files = list(base_dir.rglob("*.qmd"))
        else:
            files = list(Path(".").rglob("*.qmd"))
        
    all_errors = False
    for f in files:
        errs = check_file(f)
        if errs:
            all_errors = True
            print(f"\n❌ {f}")
            for e in errs:
                print(f"  - {e}")
                
    if all_errors:
        print("\nFix these by deleting the unused variables from the # ┌── 4. OUTPUT section, or using them in the text. Ensure every LEGO block has a 4. OUTPUT section.")
        sys.exit(1)
    else:
        print("✅ LEGO variable checks passed! No dead code or missing sections found.")

if __name__ == '__main__':
    main()
