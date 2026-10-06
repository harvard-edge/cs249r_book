import re
from collections import defaultdict

with open('09_data_selection.qmd', 'r') as f:
    lines = f.readlines()

sections = []
current_section = None
current_count = 0
figures = []
tables = []
current_fig_table = None

for i, line in enumerate(lines):
    header_match = re.match(r'^(#{1,4})\s+([A-Za-z].*)', line)
    if header_match:
        if current_section:
            sections.append((current_section, current_count))
        current_section = header_match.group(2).strip()
        current_count = 1
    elif current_section:
        current_count += len(line.split())
        
    # Check for figures or tables
    if "@fig-" in line or "fig-" in line or "Figure" in line:
        # crude check
        pass

if current_section:
    sections.append((current_section, current_count))

print("Section Word Counts:")
for sec, count in sections:
    if count > 500:
        print(f"[{count} words] {sec}")
    else:
        print(f"[{count} words] {sec}")
