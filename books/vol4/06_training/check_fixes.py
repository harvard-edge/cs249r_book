import re

file_path = "/private/tmp/MLSysBook-dev-notation/books/vol4/06_training/06_training.qmd"
with open(file_path, "r") as f:
    text = f.read()

def check_pattern(pattern, name):
    if re.search(pattern, text):
        print(f"FAILED: {name} still found!")
    else:
        print(f"PASSED: {name} fixed.")

check_pattern(r' \(@sec-(evaluation|perception|memory|intent|planning|enforcement|placement|intervention|verification|release|frontier)[^\)]*\)', "Inter-chapter paren")
check_pattern(r'@sec-(evaluation|perception|memory|intent|planning|enforcement|placement|intervention|verification|release|frontier)', "Inter-chapter direct")
check_pattern(r'Chapters?\s+[789]|Chapters?\s+1[0-7]', "Chapter N")

print("Checking intra-chapter (should be none BEFORE their definition):")
sections = set(re.findall(r'\{#(sec-training-[a-zA-Z0-9_-]+)\}', text))
for sec in sections:
    # simple text search to see where defined vs used
    def_pos = text.find("{#" + sec + "}")
    uses = [m.start() for m in re.finditer(f"@{sec}", text)]
    for u in uses:
        if u < def_pos:
            print(f"FAILED: Intra-chapter forward reference to {sec}")

