import re

with open("system-anatomy.svg", "r") as f:
    content = f.read()

# Remove the text about 161 lines
content = re.sub(r'<text x="12" y="300"[^>]+>.*?</text>\n', '', content)

# Remove the Backward-compat shim block
shim_block = r'''\s*<!-- Backward-compat shim -->\n\s*<rect x="12" y="312"[^>]+/>\n\s*<text x="72" y="327"[^>]+>.*?</text>\n\s*<line x1="132" y1="323"[^>]+/>\n'''
content = re.sub(shim_block, '', content)

with open("system-anatomy.svg", "w") as f:
    f.write(content)
