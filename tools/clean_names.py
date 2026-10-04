import yaml
import re

with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

for var in data["variables"]:
    # clean definition
    desc = var["definition"]
    desc = re.sub(r'\(.*?\)', '', desc) # remove parentheticals
    desc = desc.split(';')[0] # take first part before semicolon
    desc = desc.split(',')[0] # take first part before comma
    desc = desc.split(' as in ')[0]
    desc = desc.replace("vs.", "")
    desc = re.sub(r'\$.*?\$', '', desc) # remove math
    desc = desc.strip()
    
    # generate camel
    words = re.sub(r'[^a-zA-Z0-9 ]', '', desc).split()
    if not words:
        words = ["Var"]
    macro = "".join(w.capitalize() for w in words)
    
    # handle suffixes from latex
    latex = var["latex"]
    suf = re.search(r'_\{?\\text\{([^}]+)\}\}?', latex)
    if suf:
        suf_str = "".join(w.capitalize() for w in re.sub(r'[^a-zA-Z0-9 ]', '', suf.group(1)).split())
        if suf_str.lower() not in macro.lower():
            macro += suf_str
            
    # handle simple subscritps
    suf2 = re.search(r'_([a-zA-Z0-9]+)', latex)
    if suf2 and not suf:
        suf_str = suf2.group(1).capitalize()
        if suf_str.lower() not in macro.lower():
            macro += suf_str

    var["macro"] = "\\" + macro

# Deduplicate macro names
seen = {}
for var in data["variables"]:
    m = var["macro"]
    if m in seen:
        seen[m] += 1
        var["macro"] = f"{m}{seen[m]}"
    else:
        seen[m] = 1

with open("books/shared/_includes/notation_registry.yml", "w") as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)
