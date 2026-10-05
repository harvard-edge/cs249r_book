import re

with open('01_introduction.qmd', 'r') as f:
    text = f.read()

# Remove yaml header
text = re.sub(r'^---.*?^---', '', text, flags=re.MULTILINE|re.DOTALL)

# Remove python chunks
text = re.sub(r'```\{python\}.*?```', '', text, flags=re.MULTILINE|re.DOTALL)
# Remove tikz chunks
text = re.sub(r'```\{\.tikz\}.*?```', '', text, flags=re.MULTILINE|re.DOTALL)

# Now count words per section
lines = text.split('\n')
current_section = "Start"
counts = {current_section: 0}
for line in lines:
    m = re.match(r'^(#+ [A-Z].*)', line)
    if m:
        current_section = m.group(1).strip()
        counts[current_section] = 0
    elif not line.startswith('#'):
        counts[current_section] += len(line.split())

for sec, words in counts.items():
    if words > 500:
        print(f"{sec}: {words} words")
    elif words > 200:
        print(f"{sec}: {words} words (medium)")
