import yaml
import re
import subprocess

with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

def normalize(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

registry_map = {}
for var in data['variables']:
    registry_map[normalize(var['definition'])] = var

files = [
    ("books/shared/_partials/_notation_core_head.qmd", "books/shared/_includes/_notation_core_head.qmd"),
    ("books/shared/_partials/_notation_core_tail.qmd", "books/shared/_includes/_notation_core_tail.qmd"),
    ("books/vol1/frontmatter/_notation_degradation.qmd", "books/vol1/frontmatter/_notation_degradation.qmd"),
    ("books/vol2/frontmatter/_notation_scaling.qmd", "books/vol2/frontmatter/_notation_scaling.qmd"),
    ("books/vol3/frontmatter/_notation_agentic.qmd", "books/vol3/frontmatter/_notation_agentic.qmd"),
    ("books/vol4/frontmatter/_notation_physical.qmd", "books/vol4/frontmatter/_notation_physical.qmd")
]

current_category = "General"

for dev_path, local_path in files:
    try:
        content = subprocess.check_output(['git', 'show', f'origin/dev:{dev_path}']).decode('utf-8')
    except:
        try:
            with open(local_path) as f:
                content = f.read()
        except:
            continue
            
    for line in content.split('\n'):
        # Matches ## 1. Some Title {#id .unlisted}
        m = re.match(r'^##\s+(.*?)(?:\s*\{#.*\})?$', line)
        if m:
            cat = m.group(1).strip()
            # Remove trailing {#sec... .unlisted} if not caught by regex
            cat = re.sub(r'\{#.*\}$', '', cat).strip()
            current_category = cat
            
        if line.startswith('|') and not line.startswith('|:---') and not line.startswith('| **Symbol'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 3: 
                symbol = parts[0]
                definition = parts[1].replace('**', '')
                unit = parts[2]
                notes = parts[3] if len(parts) >= 4 else ""
                
                key = normalize(definition)
                if key in registry_map:
                    registry_map[key]['unit'] = unit
                    registry_map[key]['notes'] = notes
                    registry_map[key]['category'] = current_category
                else:
                    for var in data['variables']:
                        latex_clean = var['latex'].replace('\\\\', '\\')
                        if latex_clean in symbol:
                            var['unit'] = unit
                            var['notes'] = notes
                            var['category'] = current_category
                            break
                            
with open("books/shared/_includes/notation_registry.yml", 'w') as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)
