import sys
import glob
import re
import yaml

with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

custom_macros = {v["macro"] for v in data["variables"]}

files = glob.glob("books/vol4/**/*.qmd", recursive=True)

for filepath in files:
    with open(filepath) as f:
        lines = f.readlines()
        
    in_python = False
    for i, line in enumerate(lines):
        if line.startswith("```{python}"):
            in_python = True
        elif line.startswith("```") and in_python:
            in_python = False
            
        if in_python:
            for m in custom_macros:
                if m in line:
                    print(f"Found in {filepath}:{i+1}: {line.strip()} (Macro: {m})")
