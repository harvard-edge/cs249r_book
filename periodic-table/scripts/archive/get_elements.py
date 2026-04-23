import re
with open('periodic-table/index.html', 'r') as f:
    content = f.read()

m = re.search(r'(const elements = \[.*?\];)', content, re.DOTALL)
if m:
    with open('current_elements.txt', 'w') as out:
        out.write(m.group(1))
