import os
import re
import glob

def audit_all_captions(vol_path):
    print(f"Auditing ALL Captions (Fig, Tbl, Lst) in: {vol_path}\n")
    
    # regex to find fig-cap, tbl-cap, lst-cap in various syntaxes
    # 1. Quarto attribute syntax: fig-cap="...", tbl-cap="...", lst-cap="..."
    # 2. Markdown table caption syntax: : Caption {#tbl-name}
    # 3. Code block attributes
    
    qmd_files = glob.glob(os.path.join(vol_path, "**/*.qmd"), recursive=True)
    
    issues = []
    
    # Combined regex for any caption attribute
    attr_pattern = re.compile(r'(fig|tbl|lst)-cap\s*[:=]\s*"(.*?)"', re.DOTALL)
    # Regex for markdown table captions
    md_tbl_pattern = re.compile(r'\n:\s*(.*?)\s*\{#tbl-([\w-]+)\}', re.DOTALL)

    for fpath in sorted(qmd_files):
        rel_path = os.path.relpath(fpath, vol_path)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check Attribute Captions
        for m in attr_pattern.finditer(content):
            cap_text = m.group(2).replace('\n', ' ').strip()
            if not is_high_standard(cap_text):
                issues.append({'file': rel_path, 'type': m.group(1), 'text': cap_text})

        # Check Markdown Table Captions
        for m in md_tbl_pattern.finditer(content):
            cap_text = m.group(1).replace('\n', ' ').strip()
            if not is_high_standard(cap_text):
                issues.append({'file': rel_path, 'type': 'tbl', 'text': cap_text})

    print("-" * 60)
    print("CAPTION QUALITY REPORT")
    print("-" * 60)
    for issue in issues:
        print(f"[{issue['type'].upper()}] {issue['file']}: {issue['text'][:80]}...")
    
    print(f"\nTotal violations: {len(issues)}")

def is_high_standard(text):
    # 1. Format Check: Starts with **Title*:
    # Note: Using : because the user specifically asked for it. 
    # Current book often uses . inside bold or . after bold.
    if not re.match(r'^\*\*.*?\*\*\s*:', text):
        return False
    # 2. Depth Check: Word count > 12 (approx 2 sentences)
    if len(text.split()) < 12:
        return False
    return True

if __name__ == "__main__":
    audit_all_captions("quarto/contents/vol1")
