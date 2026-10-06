import re

with open("05_nn_computation/05_nn_computation.qmd", "r") as f:
    lines = f.readlines()

print("## Headings")
for i, line in enumerate(lines):
    if line.startswith("#"):
        print(f"L{i+1}: {line.strip()}")

print("\n## Figures and Tables")
for i, line in enumerate(lines):
    if re.search(r'fig-', line) or re.search(r'tbl-', line) or re.search(r'!\[', line) or '::: {#fig-' in line or '::: {#tbl-' in line:
        if 'fig-env' in line or 'fig-cap' in line or 'tbl-cap' in line or '![' in line or '::: {#' in line:
            print(f"L{i+1}: {line.strip()[:100]}")
