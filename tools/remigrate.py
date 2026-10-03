import glob
import re

num_map = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}

files = glob.glob("books/vol*/**/*.qmd", recursive=True) + glob.glob("books/shared/**/*.qmd", recursive=True)

replaced_count = 0
for fname in files:
    with open(fname, "r") as f:
        content = f.read()
    
    orig = content
    # Find all macros like \Something2 and replace with \SomethingTwo
    def replacer(match):
        macro = match.group(0)
        new_macro = ""
        for char in macro:
            if char in num_map:
                new_macro += num_map[char]
            else:
                new_macro += char
        return new_macro
        
    content = re.sub(r'\\[a-zA-Z]+[0-9]+', replacer, content)
    
    if content != orig:
        with open(fname, "w") as f:
            f.write(content)
        replaced_count += 1

print(f"Fixed {replaced_count} files with bad numbered macros.")
