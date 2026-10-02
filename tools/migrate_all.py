import yaml
import glob
import re
import os

with open("books/shared/_partials/notation_registry.yml") as f:
    data = yaml.safe_load(f)

# Sort variables by length descending to replace longest first
variables = sorted(data["variables"], key=lambda x: len(x["latex"]), reverse=True)

files = glob.glob("books/vol*/**/*.qmd", recursive=True)

def replace_in_math(text, latex, macro):
    # This regex is tricky. We'll find all math blocks $...$, $$...$$, \(...\), \[...\]
    # and only replace inside them.
    # Actually, a simpler way is to just use word boundaries for the LaTeX string.
    # We must ensure we don't replace 's_t' inside 's_{t+1}'.
    # We will just split the text into math and non-math segments.
    
    # regex for inline math: \$[^$]+\$
    # regex for display math: \$\$[^$]+\$\$
    pattern = re.compile(r'(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\))', re.DOTALL)
    
    segments = pattern.split(text)
    
    # Escape latex for regex
    # We want to match exactly the latex string.
    # If it ends with a letter (e.g. \pi), we need \b.
    # If it ends with a symbol (e.g. s_t), we need to ensure it's not followed by _, ^, or alphanumeric.
    
    escaped_latex = re.escape(latex.strip())
    
    # Negative lookbehind to ensure we don't match inside a larger word/command
    # Negative lookahead to ensure we don't match if it's part of a larger word/subscript
    regex = re.compile(r'(?<![a-zA-Z0-9_\\])' + escaped_latex + r'(?![a-zA-Z0-9_^{])')
    
    for i in range(1, len(segments), 2):
        segments[i] = regex.sub(macro.replace("\\", "\\\\"), segments[i])
        
    return "".join(segments)

replaced_count = 0
for fname in files:
    if "frontmatter" in fname or "backmatter" in fname or "_partials" in fname:
        continue
    with open(fname, "r") as f:
        content = f.read()
    
    orig = content
    for var in variables:
        latex = var["latex"].strip()
        macro = var["macro"].strip()
        content = replace_in_math(content, latex, macro)
    
    if content != orig:
        with open(fname, "w") as f:
            f.write(content)
        replaced_count += 1

print(f"Updated {replaced_count} files.")
