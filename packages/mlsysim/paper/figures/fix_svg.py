import re
import sys
import glob

def make_sharp(m):
    path_d = m.group(1)
    if 'C' not in path_d:
        return m.group(0)
    
    # Extract all numbers from the path
    numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', path_d)]
    if not numbers:
        return m.group(0)
    
    xmin = min(numbers[::2])
    xmax = max(numbers[::2])
    ymin = min(numbers[1::2])
    ymax = max(numbers[1::2])
    
    new_d = f"M{xmin},{ymin} L{xmax},{ymin} L{xmax},{ymax} L{xmin},{ymax} z"
    return f'd="{new_d}"'

for filepath in glob.glob("*.svg"):
    with open(filepath, "r") as f:
        content = f.read()

    new_content = re.sub(r'd="([^"]+)"', make_sharp, content)

    with open(filepath, "w") as f:
        f.write(new_content)
