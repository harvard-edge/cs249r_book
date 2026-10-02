import re

files = [
    "books/vol1/frontmatter/_notation_degradation.qmd",
    "books/vol2/frontmatter/_notation_scaling.qmd",
    "books/vol3/frontmatter/_notation_agentic.qmd",
    "books/vol4/frontmatter/_notation_physical.qmd",
    "books/shared/_includes/_notation_core_head.qmd",
    "books/shared/_includes/_notation_core_tail.qmd"
]

def slugify(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

for filepath in files:
    try:
        with open(filepath, "r") as f:
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
            
        # Detect table start (ignoring table colwidths since it's just formatting)
        if line.startswith('::: {tbl-colwidths'):
            out.append(line)
            # Find the actual table
            i += 1
            while i < len(lines) and lines[i].strip() == '':
                out.append(lines[i])
                i += 1
                
            if i < len(lines) and lines[i].startswith('|'):
                # It's a table!
                slug = slugify(current_category) if current_category else "general"
                # Determine relative path from file to tables
                # frontmatter is 2 dirs deep (books/vol1/frontmatter), shared/_includes/tables is ../../shared/_includes/tables
                # shared/_includes is 1 dir deep relative to books/shared/_includes? No, it's in the same dir!
                if "shared/_includes" in filepath:
                    include_path = f"tables/_table_{slug}.qmd"
                else:
                    include_path = f"../../shared/_includes/tables/_table_{slug}.qmd"
                    
                out.append(f"{{{{< include {include_path} >}}}}")
                
                # skip until table ends
                while i < len(lines) and (lines[i].startswith('|') or lines[i].strip() == ''):
                    i += 1
                continue
                
        out.append(line)
        i += 1
        
    with open(filepath, "w") as f:
        f.write("\n".join(out))
    print(f"Replaced tables in {filepath}")
