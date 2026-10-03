import re
import glob
import yaml
import sys

files = [
    "books/shared/_includes/_notation_core_head.qmd",
    "books/shared/_includes/_notation_core_tail.qmd",
    "books/vol1/frontmatter/_notation_degradation.qmd",
    "books/vol2/frontmatter/_notation_scaling.qmd",
    "books/vol3/frontmatter/_notation_agentic.qmd",
    "books/vol4/frontmatter/_notation_physical.qmd",
]

registry = {"variables": []}

def to_camel(s):
    # Remove parens, non-alphanumeric
    s = re.sub(r'[^a-zA-Z0-9 ]', '', s)
    words = s.split()
    if not words: return "Var"
    return "".join(w.capitalize() for w in words)

for fname in files:
    vol = "core"
    if "vol1" in fname: vol = "vol1"
    elif "vol2" in fname: vol = "vol2"
    elif "vol3" in fname: vol = "vol3"
    elif "vol4" in fname: vol = "vol4"
    
    with open(fname, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith("|") and not line.startswith("|:--") and not "Symbol" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                sym_raw = parts[1]
                desc_raw = parts[2]
                sym = re.sub(r'\*\*\$(.+?)\$\*\*', r'\1', sym_raw)
                sym = re.sub(r'\$(.+?)\$', r'\1', sym)
                desc = desc_raw.replace('**', '').strip()
                
                # Check if it has multiple symbols
                syms = [s.strip() for s in sym.split(',') if s.strip()]
                for s in syms:
                    if not s: continue
                    macro = "\\" + to_camel(desc)
                    if len(syms) > 1:
                        suf = re.search(r'_\{?\\text\{([^}]+)\}\}?', s)
                        if suf:
                            macro += suf.group(1).capitalize()
                        
                    registry["variables"].append({
                        "id": macro.replace('\\', ''),
                        "macro": macro,
                        "latex": s,
                        "definition": desc,
                        "volume": vol
                    })

with open("books/shared/_includes/notation_registry.yml", "w") as f:
    yaml.dump(registry, f, sort_keys=False, allow_unicode=True)
