
import os
import re
import glob

def audit_captions(vol_path):
    print(f"Auditing Figure Captions in: {vol_path}\n")
    
    # Regex to capture fig-cap inside image syntax or div syntax
    # Matches: fig-cap="Content" or fig-cap: "Content"
    # Handling multiline strings in regex is tricky, doing line-by-line parsing is safer given Quarto format
    
    qmd_files = glob.glob(os.path.join(vol_path, "**/*.qmd"), recursive=True)
    
    weak_captions = []
    
    for fpath in sorted(qmd_files):
        rel_path = os.path.relpath(fpath, vol_path)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Regex for fig-cap attributes
        # Looking for fig-cap="... content ..." or fig-cap: "... content ..."
        # Only capturing double-quoted captions for now as they are standard
        matches = re.finditer(r'fig-cap\s*[:=]\s*"(.*?)"', content, re.DOTALL)
        
        for m in matches:
            cap_text = m.group(1).replace('\n', ' ').strip()
            # Check for **Title**: pattern
            has_bold_title = re.match(r'\*\*.*?\*\*\s*[:.]', cap_text)
            
            # Heuristic for "Teaching Quality": Length
            word_count = len(cap_text.split())
            
            if not has_bold_title or word_count < 15:
                weak_captions.append({
                    'file': rel_path,
                    'text': cap_text,
                    'issue': []
                })
                if not has_bold_title:
                    weak_captions[-1]['issue'].append("Missing **Bold Title**")
                if word_count < 15:
                    weak_captions[-1]['issue'].append(f"Too Short ({word_count} words)")

    print("-" * 60)
    print("CAPTION AUDIT REPORT (Potential Weakness)")
    print("-" * 60)
    
    for item in weak_captions:
        print(f"\nFile: {item['file']}")
        print(f"Issues: {', '.join(item['issue'])}")
        print(f"Caption: {item['text'][:100]}...")

    print(f"\nTotal Weak Captions Found: {len(weak_captions)}")

if __name__ == "__main__":
    audit_captions("quarto/contents/vol1")
