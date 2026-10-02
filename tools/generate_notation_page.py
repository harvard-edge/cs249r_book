import yaml
import os
from collections import defaultdict

with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

# Group variables by volume
# (Assuming the 'volume' field exists, e.g., 'vol4', 'core', 'vol1', etc.)
vols = defaultdict(list)
for var in data["variables"]:
    vols[var.get("volume", "core")].append(var)

def write_qmd(vol_key, vars_list):
    # Sort alphabetically by macro
    vars_list.sort(key=lambda x: x["macro"])
    
    out = []
    out.append(f"# Notation Guide ({vol_key.upper()})\n")
    out.append("This is an auto-generated notation guide. Do not edit this manually.\n")
    out.append("| Symbol | Macro | Definition |")
    out.append("|:---|:---|:---|")
    
    for var in vars_list:
        latex = var['latex'].replace('\\', '\\\\') # escape for markdown
        macro = f"`{var['macro']}`"
        # Render the latex inside math delimiters
        symbol = f"${var['latex']}$"
        defn = var['definition']
        out.append(f"| {symbol} | {macro} | {defn} |")
        
    out.append("\n")
    
    return "\n".join(out)

for vol_key, vars_list in vols.items():
    if vol_key.startswith("vol"):
        path = f"books/{vol_key}/frontmatter/notation.qmd"
        # Just writing it to stdout for now to show the user, or we can save it.
        if os.path.exists(f"books/{vol_key}/frontmatter/"):
            with open(path, "w") as f:
                f.write(write_qmd(vol_key, vars_list))
            print(f"Generated {path}")

