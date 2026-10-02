#!/usr/bin/env python3
import yaml
import re
import os
import sys
from pathlib import Path

# Always resolve paths relative to the project root (parent of tools dir)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

registry_path = PROJECT_ROOT / "books/shared/_includes/notation_registry.yml"
with open(registry_path) as f:
    data = yaml.safe_load(f)

# 1. Generate macros
macros = set()
tex_lines = []
for var in data["variables"]:
    macro = var["macro"]
    latex = var["latex"]
    if macro not in macros:
        tex_lines.append(f"\\newcommand{{{macro}}}{{{latex}}}")
        macros.add(macro)

with open(PROJECT_ROOT / "books/shared/_includes/_macros.tex", "w") as out:
    out.write("% AUTO-GENERATED NOTATION MACROS\n")
    out.write("\n".join(tex_lines) + "\n")

with open(PROJECT_ROOT / "books/shared/_includes/_macros.html", "w") as out:
    out.write("<!-- AUTO-GENERATED NOTATION MACROS -->\n")
    out.write("<div style=\"display: none;\">\n$$\n")
    out.write("\n".join(tex_lines) + "\n")
    out.write("$$\n</div>\n")

print(f"Generated {len(macros)} macros.")

# 2. Inject tables into Markdown files
files = [
    "books/vol1/frontmatter/_notation_degradation.qmd",
    "books/vol2/frontmatter/_notation_scaling.qmd",
    "books/vol3/frontmatter/_notation_agentic.qmd",
    "books/vol4/frontmatter/_notation_physical.qmd",
    "books/shared/_includes/_notation_core_head.qmd",
    "books/shared/_includes/_notation_core_tail.qmd"
]

for filepath in files:
    full_path = PROJECT_ROOT / filepath
    try:
        with open(full_path, "r") as f:
            content = f.read()
    except Exception as e:
        continue

    out = []
    lines = content.split('\n')
    i = 0
    current_category = None
    
    while i < len(lines):
        line = lines[i]
        
        m = re.match(r'^##\s+(.+?)(?:\{.*\}|)$', line)
        if m:
            cat = m.group(1).strip()
            cat = re.sub(r'\{#.*\}$', '', cat).strip()
            current_category = cat
            
        if line.startswith('|') and '| **Symbol**' in line:
            # We found the start of a table!
            is_disambig = "Disambiguation" in current_category if current_category else False
            if is_disambig:
                out.append("| **Symbol** | **Canonical Convention** | **Colliding Meaning** | **Disambiguation Rule** |")
            else:
                out.append("| **Symbol** | **Definition** | **Unit** | **Notes** |")
            out.append("|:---|:---|:---|:---|")
            
            cat_vars = [v for v in data['variables'] if v.get('category') == current_category]
            cat_vars.sort(key=lambda x: x["macro"])
            
            for var in cat_vars:
                symbol = f"**${var['latex']}$**"
                defn = f"**{var['definition']}**"
                unit = var.get('unit', '')
                notes = var.get('notes', '').replace('\n', ' ')
                out.append(f"| {symbol} | {defn} | {unit} | {notes} |")
                
            # Skip the old table lines
            while i < len(lines) and (lines[i].startswith('|') or lines[i].strip() == ''):
                i += 1
            out.append("")
            continue
            
        out.append(line)
        i += 1
        
    with open(full_path, "w") as f:
        f.write("\n".join(out).strip() + "\n")
    print(f"Injected tables into {filepath}")
