import re

with open('01_introduction.qmd', 'r') as f:
    text = f.read()

# Find figure definitions
# Either #| label: fig-name or {#fig-name} or ![](...){#fig-name}
labels = set(re.findall(r'#\|\s*label:\s*(fig-[a-zA-Z0-9-]+)', text))
labels.update(re.findall(r'\{#(fig-[a-zA-Z0-9-]+)\}', text))

print(f"Found {len(labels)} figures: {labels}")

# Find references
refs = set(re.findall(r'@(fig-[a-zA-Z0-9-]+)', text))

orphans = labels - refs
missing = refs - labels

print(f"Orphaned figures (defined but not referenced): {orphans}")
print(f"Missing figures (referenced but not defined): {missing}")

