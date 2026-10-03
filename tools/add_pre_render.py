import glob
import re

for filepath in glob.glob("books/config/_quarto-*.yml"):
    with open(filepath, "r") as f:
        content = f.read()
        
    if "pre-render:" in content:
        continue
        
    # Find the project: block and add pre-render
    # It might have post-render: or other fields
    
    # Simple regex to insert after project:
    content = re.sub(
        r'^(project:\n)', 
        r'\1  pre-render:\n    - ../tools/build_notation.py\n', 
        content, 
        flags=re.MULTILINE
    )
    
    with open(filepath, "w") as f:
        f.write(content)
