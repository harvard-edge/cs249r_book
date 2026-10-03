import yaml
import re

with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

def normalize(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

registry_map = {}
for var in data['variables']:
    registry_map[normalize(var['definition'])] = var

files = [
    "books/vol1/frontmatter/_notation_degradation.qmd",
    "books/vol2/frontmatter/_notation_scaling.qmd",
    "books/vol3/frontmatter/_notation_agentic.qmd",
    "books/vol4/frontmatter/_notation_physical.qmd"
]

for filepath in files:
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        continue

    # We will rebuild the file line by line.
    # When we are inside a table, we skip the lines and just output the fresh table
    # based on the variables belonging to the current category.
    
    out = []
    lines = content.split('\n')
    in_table = False
    current_category = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        m = re.match(r'^##\s+(.+?)(?:\{.*\}|)$', line)
        if m:
            cat = m.group(1).strip()
            cat = re.sub(r'\{#.*\}$', '', cat).strip()
            current_category = cat
            
        if line.startswith('|') and '| **Symbol**' in line:
            # We found the start of a table!
            in_table = True
            
            # Generate the new table
            is_disambig = "Disambiguation" in current_category if current_category else False
            if is_disambig:
                out.append("| **Symbol** | **Canonical Convention** | **Colliding Meaning** | **Disambiguation Rule** |")
            else:
                out.append("| **Symbol** | **Definition** | **Unit** | **Notes** |")
            out.append("|:---|:---|:---|:---|")
            
            # Find all variables for this category in the file's volume
            # Wait, the category name is exactly matched.
            cat_vars = [v for v in data['variables'] if v.get('category') == current_category]
            cat_vars.sort(key=lambda x: x["macro"])
            
            for var in cat_vars:
                symbol = f"**${var['latex'].replace('$', '')}$**"
                defn = f"**{var['definition']}**"
                unit = var.get('unit', '')
                notes = var.get('notes', '').replace('\n', ' ')
                out.append(f"| {symbol} | {defn} | {unit} | {notes} |")
                
            # Skip the old table lines
            while i < len(lines) and (lines[i].startswith('|') or lines[i].strip() == ''):
                i += 1
            # We skipped the empty line after table too, so append an empty line
            out.append("")
            continue
            
        out.append(line)
        i += 1
        
    with open(filepath, "w") as f:
        f.write("\n".join(out))
    print(f"Injected tables into {filepath}")
