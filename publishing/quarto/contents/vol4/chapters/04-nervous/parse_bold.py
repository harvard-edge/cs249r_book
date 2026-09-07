import re

with open('/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/04-nervous/04-nervous.qmd', 'r') as f:
    text = f.read()

# find all **term**\index{...} and **term** without index, maybe.
# Actually, the rule says:
# Audit for bold terms (index terms) ensuring they are strictly sentence case (or lowercase) inside the main neuroprose, matching the style from Volume I (e.g., `**model compression**\\index{Model compression!definition}`).
# Do not capitalize bold terms unless they are proper nouns or abbreviations.

matches = re.findall(r'\*\*([^*]+)\*\*(?:\\index{[^}]+})?', text)
for m in matches:
    print(m)
