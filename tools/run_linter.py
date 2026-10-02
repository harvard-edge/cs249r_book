import yaml
from pathlib import Path

with open("books/shared/_partials/notation_registry.yml") as f:
    registry = yaml.safe_load(f)

banned = {var["latex"].strip(): var["macro"].strip() for var in registry["variables"] if "\\text{" in var["latex"] or "_" in var["latex"]}

issues = 0
for qmd in Path('books').rglob('*.qmd'):
    if "frontmatter" in str(qmd) or "_partials" in str(qmd):
        continue
    text = qmd.read_text()
    for i, line in enumerate(text.split('\n')):
        if "$" not in line and "\\[" not in line and "\\(" not in line:
            continue
        for latex, macro in banned.items():
            if latex in line:
                issues += 1
                if issues <= 10:
                    print(f"{qmd}:{i+1} found {latex} -> should be {macro}")

print(f"Total issues found: {issues}")
