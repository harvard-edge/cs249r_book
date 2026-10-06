import re

with open('10_model_compression.qmd', 'r') as f:
    text = f.read()

# Find definitions
defs = re.findall(r'\{#(fig-[a-zA-Z0-9-]+|tbl-[a-zA-Z0-9-]+)[ \}]', text)

for d in defs:
    # case insensitive search for @d
    count = len(re.findall(f'@{d}', text, re.IGNORECASE))
    if count == 0:
        print(f"Orphaned: {d}")
    else:
        print(f"Referenced {count} times: {d}")
