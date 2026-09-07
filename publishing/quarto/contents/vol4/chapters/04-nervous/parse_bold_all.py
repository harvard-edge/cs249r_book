import re

with open('/Users/VJ/GitHub/MLSysBook-vol4-deep-audit/publishing/quarto/contents/vol4/chapters/04-nervous/04-nervous.qmd', 'r') as f:
    text = f.read()

# Let's extract all bold terms `**...**` or `***...***`
matches = re.findall(r'(?<!\*)\*\*(?!\*)(.*?)(?<!\*)\*\*(?!\*)', text)
matches3 = re.findall(r'\*\*\*(.*?)\*\*\*', text)

for m in set(matches + matches3):
    if any(w.istitle() for w in m.split()) and not m.isupper():
        print(f"Match: {m}")
