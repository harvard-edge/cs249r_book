import re

with open("12_benchmarking.qmd", "r") as f:
    text = f.read()

# Quarto explicit labels: ::: {#fig-id} or {#fig-id} or #| label: fig-id
fig_defs = set(re.findall(r'\{#(fig-[a-zA-Z0-9-]+)[ \}]', text, re.IGNORECASE) + re.findall(r'#\|\s*label:\s*(fig-[a-zA-Z0-9-]+)', text, re.IGNORECASE))
tbl_defs = set(re.findall(r'\{#(tbl-[a-zA-Z0-9-]+)[ \}]', text, re.IGNORECASE) + re.findall(r'#\|\s*label:\s*(tbl-[a-zA-Z0-9-]+)', text, re.IGNORECASE))

# Normalize definitions to lowercase
fig_defs = {x.lower() for x in fig_defs}
tbl_defs = {x.lower() for x in tbl_defs}

# Find all references in text: @fig-my-figure or @tbl-my-table
refs = re.findall(r'@(fig-[a-zA-Z0-9-]+|tbl-[a-zA-Z0-9-]+)', text, re.IGNORECASE)
refs_set = {x.lower() for x in refs}

unreferenced = (fig_defs | tbl_defs) - refs_set
missing_defs = refs_set - (fig_defs | tbl_defs)

print(f"Defined figures: {len(fig_defs)}")
print(f"Defined tables: {len(tbl_defs)}")

print("\nUnreferenced (Orphaned) Visuals:")
for v in sorted(unreferenced):
    print("  - " + v)

print("\nReferenced but not defined (Missing):")
for v in sorted(missing_defs):
    print("  - " + v)

