import re

with open('/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/04-nervous/04-nervous.qmd', 'r') as f:
    text = f.read()

matches = re.findall(r'(\*\*.*?\*\*(?:\\index{[^}]+})+|(\*\*\*.*?\*\*\*(?:\\index{[^}]+})+))', text)
for m in matches:
    print(m[0] if m[0] else m[1])

# Also check for terms in neuroprose that are bold but maybe not indexed yet?
