import re
import glob

fpath = glob.glob("books/vol2/07_*/07_*.qmd")[0]
with open(fpath, 'r') as f:
    text = f.read()

fi_regex = re.compile(r'(## Fault Injection Tools and Frameworks .*?\n)(?=## )', re.DOTALL)
debug_regex = re.compile(r'(## Distributed Debugging and Observability .*?\n)(?=## )', re.DOTALL)

fi_match = fi_regex.search(text)
debug_match = debug_regex.search(text)

if fi_match and debug_match:
    fi_start, fi_end = fi_match.span()
    debug_start, debug_end = debug_match.span()
    
    if fi_start < debug_start:
        # fi comes first, then debug. Let's swap.
        # Wait, the text between fi_end and debug_start is empty since they are consecutive H2s (presumably).
        # We can just extract both and swap their order.
        new_text = text[:fi_start] + debug_match.group(1) + fi_match.group(1) + text[debug_end:]
        with open(fpath, 'w') as f:
            f.write(new_text)
        print("Ch7 swapped")
    else:
        print("Already swapped?")
else:
    print("Match failed")
