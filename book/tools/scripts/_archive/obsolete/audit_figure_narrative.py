
import os
import re
import glob

def audit_figure_context(vol_path, output_file):
    print(f"Auditing Figure Context in: {vol_path}\n")
    
    ref_pattern = re.compile(r'@fig-([\w-]+)')
    qmd_files = glob.glob(os.path.join(vol_path, "**/*.qmd"), recursive=True)
    
    with open(output_file, 'w', encoding='utf-8') as report:
        report.write("# Figure Narrative Audit Report\n\n")
        
        for fpath in sorted(qmd_files):
            rel_path = os.path.relpath(fpath, vol_path)
            
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            file_refs = []
            for i, line in enumerate(lines):
                for m_ref in ref_pattern.finditer(line):
                    fig_id = m_ref.group(1)
                    # Capture 2 lines before and 2 lines after for context
                    start_ctx = max(0, i - 2)
                    end_ctx = min(len(lines), i + 3)
                    
                    context_lines = [l.strip() for l in lines[start_ctx:end_ctx] if l.strip()]
                    context_block = "\n> ".join(context_lines)
                    
                    file_refs.append(f"### `@fig-{fig_id}` (Line {i+1})\n\n> {context_block}\n")

            if file_refs:
                report.write(f"## {rel_path}\n\n")
                report.write("\n".join(file_refs))
                report.write("\n" + "-"*40 + "\n\n")

    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    audit_figure_context("quarto/contents/vol1", "tools/scripts/figure_narrative_audit.md")
