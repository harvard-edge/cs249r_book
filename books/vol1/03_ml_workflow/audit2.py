import re
from collections import defaultdict

with open("03_ml_workflow.qmd", "r") as f:
    content = f.read()
    lines = content.split('\n')

# Find labels
labels = set(re.findall(r'\{#(fig-[^}]+|tbl-[^}]+)\}', content))
# Find Quarto div labels
labels.update(re.findall(r':::\s*\{#(fig-[^\s}]+|tbl-[^\s}]+)', content))
# Look for cross-references
refs = set(re.findall(r'@(fig-[^\s,.;:\]\)]+|tbl-[^\s,.;:\]\)]+)', content, re.IGNORECASE))
refs = {r.lower() for r in refs}

print("=== Visuals Check ===")
for label in sorted(labels):
    if label.lower() not in refs:
        print(f"[ORPHANED] {label}")
    else:
        print(f"[REFERENCED] {label}")

# Find long sections
print("\n=== Section Lengths (potential bloat) ===")
current_section = "Intro"
section_lines = 0
for line in lines:
    if line.startswith("#"):
        if section_lines > 150:
            print(f"Long section: {current_section} ({section_lines} lines)")
        match = re.search(r'^(#+)\s+(.+?)(?:\s+\{#.*\})?$', line)
        if match:
            current_section = match.group(2)
        section_lines = 0
    else:
        section_lines += 1

if section_lines > 150:
    print(f"Long section: {current_section} ({section_lines} lines)")

# Find long lists
print("\n=== Long Lists (potential bloat) ===")
list_count = 0
current_list_length = 0
for line in lines:
    if re.match(r'^\s*-\s+', line):
        current_list_length += 1
    elif line.strip() == "":
        continue # empty lines between list items
    elif current_list_length > 0 and not re.match(r'^\s+', line):
        if current_list_length >= 7:
            print(f"Found a list with {current_list_length} items near line.")
        current_list_length = 0

