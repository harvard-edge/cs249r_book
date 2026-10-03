import glob
import yaml

with open("mlsysim/mlsysim/notation/data/symbols.yaml") as f:
    data = yaml.safe_load(f)

custom_macros = {s["macro"] for s in data["symbols"]}

files = sorted(glob.glob("books/vol*/**/*.qmd", recursive=True))

count = 0
for filepath in files:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_python = False
    for i, line in enumerate(lines):
        if line.strip().startswith("```{python}"):
            in_python = True
        elif line.strip().startswith("```") and in_python:
            in_python = False
            
        if in_python:
            for m in custom_macros:
                if m in line:
                    count += 1
                    print(f"{filepath}:{i+1}: {line.strip()} (Macro: {m})")

print(f"Total found: {count}")
