import re

with open('01_introduction.qmd', 'r') as f:
    lines = f.readlines()

current_section = "Start"
counts = {current_section: 0}
for line in lines:
    m = re.match(r'^(#+ [A-Z].*)', line)
    if m:
        current_section = m.group(1).strip()
        counts[current_section] = 0
    else:
        # crude word count
        counts[current_section] += len(line.split())

for sec, words in counts.items():
    if words > 500:
        print(f"{sec}: {words} words")
    elif words > 200:
        print(f"{sec}: {words} words (medium)")
