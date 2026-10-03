import yaml
import glob
import re

with open("books/shared/_includes/notation_registry.yml") as f:
    data = yaml.safe_load(f)

# Filter safe variables to auto-replace
safe_vars = []
for var in data["variables"]:
    latex = var["latex"]
    macro = var["macro"]
    # Safe to replace if it has a \text subscript, or is > 5 chars and has _, or is \text{BW}
    if "\\text{" in latex or ("_" in latex and len(latex) > 4):
        # Escape for regex
        latex_esc = re.escape(latex)
        # We need to match it when it's surrounded by math boundaries or similar, but
        # a simple string replace is safe for things like D_{\text{vol}}
        safe_vars.append((latex, macro))

print(f"Found {len(safe_vars)} safe variables for automatic replacement.")
safe_vars.sort(key=lambda x: len(x[0]), reverse=True) # Replace longer strings first

files = glob.glob("books/vol*/**/*.qmd", recursive=True) + glob.glob("books/shared/**/*.qmd", recursive=True)

replaced_count = 0
for fname in files:
    with open(fname, "r") as f:
        content = f.read()
    
    orig = content
    for latex, macro in safe_vars:
        # replace exact string
        content = content.replace(latex, macro)
    
    if content != orig:
        with open(fname, "w") as f:
            f.write(content)
        replaced_count += 1

print(f"Updated {replaced_count} .qmd files.")
