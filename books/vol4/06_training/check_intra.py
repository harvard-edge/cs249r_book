import re

sections = set()
with open("/private/tmp/MLSysBook-dev-notation/books/vol4/06_training/06_training.qmd", "r") as f:
    for i, line in enumerate(f):
        # find section definitions
        for match in re.finditer(r'\{#([a-zA-Z0-9_-]+)\}', line):
            sections.add(match.group(1))
            
        # find references
        for match in re.finditer(r'@(sec-[a-zA-Z0-9_-]+)', line):
            sec = match.group(1)
            if sec.startswith("sec-training-") and sec not in sections:
                print(f"Line {i+1}: Forward reference to {sec}")
