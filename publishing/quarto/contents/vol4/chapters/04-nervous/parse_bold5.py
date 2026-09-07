import re

with open('/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/04-nervous/04-nervous.qmd', 'r') as f:
    text = f.read()

for i, line in enumerate(text.split('\n')):
    if line.startswith(('#', '|', ':', '>', '- ', '* ', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '```')): continue
    if not line.strip(): continue
    
    # match 2 or 3 asterisks
    matches = re.findall(r'(?<!^)\*{2,3}([^*]+)\*{2,3}', line)
    for m in matches:
        if any(w.istitle() for w in m.split()) and not m.isupper():
            print(f"Line {i+1}: {m}")
