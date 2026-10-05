import re

with open('01_introduction.qmd', 'r') as f:
    text = f.read()

# Match either #| label: fig-name or {#fig-name ...} or ![](...){#fig-name}
labels = set(re.findall(r'#\|\s*label:\s*(fig-[a-zA-Z0-9-]+)', text))
labels.update(re.findall(r'\{#(fig-[a-zA-Z0-9-]+)[ \}]', text))

print(f"Found {len(labels)} defined figures: {labels}")

# Find references (case insensitive to catch @Fig-...)
refs = set([r.lower() for r in re.findall(r'@(fig-[a-zA-Z0-9-]+)', text, re.IGNORECASE)])

print(f"Found {len(refs)} referenced figures: {refs}")

orphans = [l for l in labels if l.lower() not in refs]
missing = [r for r in refs if r not in [l.lower() for l in labels]]

print(f"Orphaned figures (defined but not referenced): {orphans}")
print(f"Missing figures (referenced but not defined): {missing}")

