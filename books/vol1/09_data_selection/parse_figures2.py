import re

with open('09_data_selection.qmd', 'r') as f:
    text = f.read()

# Find figure labels. They might be in code chunks as `#| label: fig-XYZ` or `{#fig-XYZ}`
labels = re.findall(r'label:\s*(fig-[a-zA-Z0-9_-]+|tbl-[a-zA-Z0-9_-]+)', text)
labels += re.findall(r'\{#(fig-[a-zA-Z0-9_-]+|tbl-[a-zA-Z0-9_-]+)\}', text)
# Just look for any ID-like fig- or tbl- definition:
all_labels = set(labels)

print(f"Found {len(all_labels)} total labels (figures and tables).")

orphaned = []
for lbl in all_labels:
    # Check if there is an '@' reference to it
    if text.count(f"@{lbl}") == 0:
        orphaned.append(lbl)

print("\nOrphaned visuals (no @reference found in text):")
for o in orphaned:
    print(f" - {o}")

