import re

with open('/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/04-nervous/04-nervous.qmd', 'r') as f:
    text = f.read()

# Let's find all bold terms that don't start at the beginning of a line (or right after a list bullet)
for i, line in enumerate(text.split('\n')):
    # skip headers, lists, captions, tables, blocks
    if line.startswith(('#', '|', ':', '>', '- ', '* ', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '```')): continue
    if not line.strip(): continue
    
    # find bold terms inside the line
    matches = re.findall(r'(?<!^)\*\*([^*]+)\*\*', line)
    for m in matches:
        if any(w.istitle() for w in m.split()) and not m.isupper():
            print(f"Line {i+1}: {m}")

