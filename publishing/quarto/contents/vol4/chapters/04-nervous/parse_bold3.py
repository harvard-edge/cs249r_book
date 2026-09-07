import re

with open('/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/04-nervous/04-nervous.qmd', 'r') as f:
    lines = f.readlines()

in_table = False
in_caption = False
in_code = False
in_yaml = False
in_header = False

for i, line in enumerate(lines):
    if line.startswith('```'):
        in_code = not in_code
        continue
    if in_code:
        continue
    if line.startswith('|'):
        continue # table
    if line.strip().startswith(':'): # table caption
        continue
    if line.startswith('#'):
        continue
    
    # find bold terms
    matches = re.findall(r'\*\*([^*]+)\*\*', line)
    for m in matches:
        # Check if it has title casing and isn't at the start of a sentence or an abbreviation
        if any(w.istitle() for w in m.split()):
            print(f"Line {i+1}: {line.strip()} -> {m}")

