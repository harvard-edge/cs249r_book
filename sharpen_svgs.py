import os
import glob
import re

svg_files = glob.glob('book/quarto/contents/vol2/**/*.svg', recursive=True)

rx_ry_pattern = re.compile(r'\s*\b(rx|ry)=["\'][0-9.]+["\']')

modifications = 0
files_changed = 0

for fpath in svg_files:
    if "/_" in fpath or fpath.startswith('_'): continue
    
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    parts = content.split('<rect ')
    new_parts = [parts[0]]
    
    changed = False
    for part in parts[1:]:
        end_idx = part.find('>')
        if end_idx != -1:
            tag_content = part[:end_idx]
            rest = part[end_idx:]
            
            new_tag_content, num_subs = rx_ry_pattern.subn('', tag_content)
            if num_subs > 0:
                new_tag_content = re.sub(r'\s+', ' ', new_tag_content).replace(' />', '/>').replace(' >', '>')
                changed = True
                modifications += num_subs
                
            new_parts.append(new_tag_content + rest)
        else:
            new_parts.append(part)
            
    new_content = '<rect '.join(new_parts)
    
    if changed:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        files_changed += 1

print(f"Removed {modifications} rounded corners (rx/ry) from {files_changed} SVG files in Volume 2.")
