import yaml
import os
from collections import defaultdict

with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

vols = defaultdict(list)
for var in data["variables"]:
    vols[var.get("volume", "core")].append(var)

def write_qmd(vol_key, vars_list):
    cats = defaultdict(list)
    for var in vars_list:
        cats[var.get("category", "General Notation")].append(var)
        
    out = []
    out.append(f"<!-- AUTO-GENERATED NOTATION TABLES FOR {vol_key.upper()} -->\n")
    
    for cat in sorted(cats.keys()):
        out.append(f"## {cat} {{.unlisted}}\n")
        
        is_disambig = "Disambiguation" in cat
        if is_disambig:
            out.append("::: {tbl-colwidths=\"[12,18,18,52]\"}\n")
            out.append("| **Symbol** | **Canonical Convention** | **Colliding Meaning** | **Disambiguation Rule** |")
        else:
            out.append("::: {tbl-colwidths=\"[15,20,15,50]\"}\n")
            out.append("| **Symbol** | **Definition** | **Unit** | **Notes** |")
            
        out.append("|:---|:---|:---|:---|")
        
        cat_vars = sorted(cats[cat], key=lambda x: x["macro"])
        for var in cat_vars:
            symbol = f"**${var['latex'].replace('$', '')}$**"
            defn = f"**{var['definition']}**"
            unit = var.get('unit', '')
            notes = var.get('notes', '').replace('\n', ' ')
            out.append(f"| {symbol} | {defn} | {unit} | {notes} |")
            
        out.append("\n:::\n")
        
    return "\n".join(out)

for vol_key, vars_list in vols.items():
    if vol_key.startswith("vol"):
        path = f"books/{vol_key}/frontmatter/_notation_auto.qmd"
        if os.path.exists(f"books/{vol_key}/frontmatter/"):
            with open(path, "w") as f:
                f.write(write_qmd(vol_key, vars_list))
            print(f"Generated {path}")
