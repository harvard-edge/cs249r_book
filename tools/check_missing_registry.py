import yaml
import re

with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

known_defs = {re.sub(r'[^a-z0-9]', '', str(v['definition']).lower()) for v in data['variables']}
known_latex = {v['latex'].replace('\\\\', '\\') for v in data['variables']}

files = [
    "books/shared/_includes/_notation_core_head.qmd",
    "books/shared/_includes/_notation_core_tail.qmd",
    "books/vol1/frontmatter/_notation_degradation.qmd",
    "books/vol2/frontmatter/_notation_scaling.qmd",
    "books/vol3/frontmatter/_notation_agentic.qmd",
    "books/vol4/frontmatter/_notation_physical.qmd"
]

missing = []

for filepath in files:
    try:
        with open(filepath) as f:
            content = f.read()
    except:
        continue
        
    for line in content.split('\n'):
        if line.startswith('|') and not line.startswith('|:---') and not line.startswith('| **Symbol'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 3:
                symbol = parts[0]
                defn = parts[1].replace('**', '')
                defn_norm = re.sub(r'[^a-z0-9]', '', defn.lower())
                
                if defn_norm not in known_defs:
                    # check if the symbol contains a known latex
                    found = False
                    for lat in known_latex:
                        if lat in symbol:
                            found = True
                            break
                    if not found:
                        missing.append((symbol, defn, filepath))

for s, d, f in missing:
    print(f"Missing in registry: {s} | {d} (from {f})")
