
import os
import re
import glob

def audit_narrative(vol_path, prefix):
    print(f"Auditing {prefix.upper()} Narrative Context in: {vol_path}\n")
    
    ref_pattern = re.compile(r'@' + prefix + r'-([\w-]+)')
    qmd_files = glob.glob(os.path.join(vol_path, "**/*.qmd"), recursive=True)
    
    results = []
    for fpath in sorted(qmd_files):
        rel_path = os.path.relpath(fpath, vol_path)
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            for m_ref in ref_pattern.finditer(line):
                elem_id = m_ref.group(1)
                # Capture 2 lines before and 2 lines after
                start_ctx = max(0, i - 2)
                end_ctx = min(len(lines), i + 3)
                context = "".join([l.strip() + " " for l in lines[start_ctx:end_ctx]])
                results.append(f"### `@{prefix}-{elem_id}` ({rel_path}:{i+1})\n> {context}\n")
    
    return results

if __name__ == "__main__":
    tbl_results = audit_narrative("quarto/contents/vol1", "tbl")
    lst_results = audit_narrative("quarto/contents/vol1", "lst")
    
    with open("tools/scripts/tbl_lst_narrative_audit.md", "w", encoding='utf-8') as f:
        f.write("# Tables and Listings Narrative Audit\n\n")
        f.write("## Tables\n\n")
        f.write("\n".join(tbl_results))
        f.write("\n\n## Listings\n\n")
        f.write("\n".join(lst_results))
    
    print("Report generated: tools/scripts/tbl_lst_narrative_audit.md")
