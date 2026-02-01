import os
import re
import glob

def audit_figures(vol_path):
    print(f"Auditing Volume 1 in: {vol_path}\n")
    
    # regex patterns
    # Matches: {#fig-name} or { #fig-name ... }
    img_attr_pattern = re.compile(r'\{.*#fig-([\w-]+).*\}')
    # Matches: #| label: fig-name
    code_label_pattern = re.compile(r'#\|\s*label:\s*fig-([\w-]+)')
    # Matches: ::: {#fig-name ... }
    div_id_pattern = re.compile(r':::\s*\{.*#fig-([\w-]+).*\}')
    
    # Matches: @fig-name
    ref_pattern = re.compile(r'@fig-([\w-]+)')

    qmd_files = glob.glob(os.path.join(vol_path, "**/*.qmd"), recursive=True)
    
    audit_report = {}

    for fpath in qmd_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        rel_path = os.path.relpath(fpath, vol_path)
        defined_figs = {} # id -> line_num
        referenced_figs = [] # (id, line_num, context)
        
        # Pass 1: Find definitions
        for i, line in enumerate(lines):
            # Check for image/div attributes
            m_attr = img_attr_pattern.search(line)
            if m_attr:
                defined_figs[m_attr.group(1)] = i + 1
                continue
                
            # Check for code labels
            m_code = code_label_pattern.search(line)
            if m_code:
                defined_figs[m_code.group(1)] = i + 1
                continue
                
            # Check for div ids
            m_div = div_id_pattern.search(line)
            if m_div:
                defined_figs[m_div.group(1)] = i + 1
                continue

        # Pass 2: Find references and context
        for i, line in enumerate(lines):
            for m_ref in ref_pattern.finditer(line):
                fig_id = m_ref.group(1)
                # Get context: previous line + current line + next line
                start_ctx = max(0, i - 1)
                end_ctx = min(len(lines), i + 2)
                context = "".join([l.strip() + " " for l in lines[start_ctx:end_ctx]])
                referenced_figs.append({
                    'id': fig_id,
                    'line': i + 1,
                    'context': context
                })

        if defined_figs or referenced_figs:
            audit_report[rel_path] = {
                'definitions': defined_figs,
                'references': referenced_figs
            }

    # Analysis
    print("-" * 60)
    print("FIGURE AUDIT REPORT")
    print("-" * 60)
    
    total_unreferenced = 0
    
    for fname, data in audit_report.items():
        defs = set(data['definitions'].keys())
        refs = set(r['id'] for r in data['references'])
        
        # 1. Unreferenced Figures
        unref = defs - refs
        if unref:
            print(f"\n[UNREFERENCED] {fname}:")
            for fig in unref:
                print(f"  - fig-{fig} (Line {data['definitions'][fig]})")
                total_unreferenced += 1
        
        # 2. Undefined References (Broken Links)
        # Note: References might point to other chapters, so this is just a warning
        undef = refs - defs
        # if undef:
        #     print(f"\n[EXTERNAL/UNDEFINED REF] {fname}:")
        #     for fig in undef:
        #         print(f"  - @fig-{fig}")

    print(f"\nTotal Unreferenced Figures: {total_unreferenced}")
    print("-" * 60)
    
    # 3. Reference Quality Check (Sampling)
    print("\nREFERENCE CONTEXT SAMPLE (Check for explanation quality):")
    for fname, data in audit_report.items():
        if not data['references']: continue
        print(f"\nFile: {fname}")
        for ref in data['references']:
            if ref['id'] in data['definitions']: # Only check local refs for now
                print(f"  Line {ref['line']} (@fig-{ref['id']}): ...{ref['context'][:200]}...")

if __name__ == "__main__":
    audit_figures("quarto/contents/vol1")
