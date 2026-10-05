import re

with open("books/vol1/01_introduction/01_introduction.qmd") as f:
    lines = f.readlines()

sections = []
current_section = None
for i, line in enumerate(lines):
    if re.match(r"^## ", line):
        if current_section:
            current_section["end"] = i - 1
            sections.append(current_section)
        current_section = {"title": line.strip(), "start": i}

if current_section:
    current_section["end"] = len(lines) - 1
    sections.append(current_section)

for s in sections:
    print(f"{s['title']} ({s['start']}-{s['end']})")
