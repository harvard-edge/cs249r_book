import re

with open("03_ml_workflow.qmd", "r") as f:
    content = f.read()

# Find all figures and tables
figures = re.findall(r'#\| label: (fig-[^\s]+)', content)
tables = re.findall(r'#\| label: (tbl-[^\s]+)', content)
markdown_figures = re.findall(r'!\[.*?\]\(.*?\)\{.*?(#fig-[^\s\}]+).*?\}', content)
markdown_tables = re.findall(r'\|.*\|\n\|[-:| ]+\|\n(?:\|.*\|\n)+.*?(#tbl-[^\s\}]+)', content)

all_figures = set(figures + [f.lstrip('#') for f in markdown_figures])
all_tables = set(tables + [t.lstrip('#') for t in markdown_tables])

print("Figures found:")
for fig in all_figures: print(f" - {fig}")
print("Tables found:")
for tbl in all_tables: print(f" - {tbl}")

# Check for references
print("\nChecking for orphaned visuals...")
orphaned_figs = []
for fig in all_figures:
    # Need to check if there is a reference like @fig-...
    ref_pattern = f"@{fig}"
    if content.count(ref_pattern) == 0:
        orphaned_figs.append(fig)

for tbl in all_tables:
    ref_pattern = f"@{tbl}"
    if content.count(ref_pattern) == 0:
        print(f"Orphaned table: {tbl}")

for fig in orphaned_figs:
    print(f"Orphaned figure: {fig}")

# Check for encyclopedic bloat (e.g. very long bulleted lists)
lists = re.findall(r'(?:^\s*-\s+.*\n){10,}', content, re.MULTILINE)
if lists:
    print("\nPossible encyclopedic bloat (long lists found):")
    for i, lst in enumerate(lists):
        print(f"List {i+1} (starts with): {lst.strip().split(chr(10))[0]}")

