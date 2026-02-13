import os
import re
import glob
import sys

def audit_element(vol_path, element_type):
    """
    element_type: 'tbl' or 'lst'
    """
    print(f"Auditing {element_type.upper()} in: {vol_path}\n")
    
    prefix = element_type
    
    # Regex patterns
    # 1. Code chunk labels: #| label: tbl-name or #| label: lst-name
    code_label_pattern = re.compile(r'#|\s*label:\s*' + prefix + r'-([\w-]+)')
    
    # 2. Markdown/Div definitions
    if element_type == 'tbl':
        # Markdown table caption: : Caption text {#tbl-name}
        md_caption_pattern = re.compile(r':\s*.*\{#'+ prefix + r'-([\w-]+)\}')
        # Div syntax: ::: {#tbl-name}
        div_id_pattern = re.compile(r':::\s*\{.*#' + prefix + r'-([\w-]+).*\}')
        # Combined list for tables
        def_patterns = [code_label_pattern, md_caption_pattern, div_id_pattern]
    else: # lst
        # Code fence: ```python {#lst-name}
        fence_pattern = re.compile(r'```.*\{.*#' + prefix + r'-([\w-]+).*\}')
        # Combined list for listings
        def_patterns = [code_label_pattern, fence_pattern]

    # Reference pattern: @tbl-name or @lst-name
    ref_pattern = re.compile(r'@' + prefix + r'-([\w-]+)')

    qmd_files = glob.glob(os.path.join(vol_path, "**/*.qmd"), recursive=True)
    
    audit_report = {}

    for fpath in qmd_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        rel_path = os.path.relpath(fpath, vol_path)
        defined_ids = {} # id -> line_num
        referenced_ids = [] # (id, line_num, context)
        
        # Scan lines
        for i, line in enumerate(lines):
            # Check definitions
            for pat in def_patterns:
                m = pat.search(line)
                if m:
                    defined_ids[m.group(1)] = i + 1
                    break 
            
            # Check references
            for m_ref in ref_pattern.finditer(line):
                elem_id = m_ref.group(1)
                # Get context: previous line + current line + next line
                start_ctx = max(0, i - 1)
                end_ctx = min(len(lines), i + 2)
                context = "".join([l.strip() + " " for l in lines[start_ctx:end_ctx]])
                referenced_ids.append({
                    'id': elem_id,
                    'line': i + 1,
                    'context': context
                })

        if defined_ids or referenced_ids:
            audit_report[rel_path] = {
                'definitions': defined_ids,
                'references': referenced_ids
            }

    # Analysis & Reporting
    print("-" * 60)
    print(f"{element_type.upper()} AUDIT REPORT")
    print("-" * 60)
    
    total_unreferenced = 0
    
    for fname, data in audit_report.items():
        defs = set(data['definitions'].keys())
        refs = set(r['id'] for r in data['references'])
        
        # 1. Unreferenced
        unref = defs - refs
        if unref:
            print(f"\n[UNREFERENCED] {fname}:")
            for item in unref:
                print(f"  - {prefix}-{item} (Line {data['definitions'][item]})")
                total_unreferenced += 1
        
    print(f"\nTotal Unreferenced {element_type.upper()}: {total_unreferenced}")
    print("-" * 60)
    
    # 2. Reference Context Sample
    print(f"\nREFERENCE CONTEXT SAMPLE ({element_type.upper()}):")
    for fname, data in audit_report.items():
        if not data['references']: continue
        print(f"\nFile: {fname}")
        for ref in data['references']:
            # Only print context if it's a local definition (to avoid noise from cross-chapter refs)
            # or if we want to check everything. Let's check everything for narrative quality.
            print(f"  Line {ref['line']} (@{prefix}-{ref['id']}): ...{ref['context'][:200]}...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 audit_tables_listings.py [tbl|lst]")
    else:
        audit_element("quarto/contents/vol1", sys.argv[1])
