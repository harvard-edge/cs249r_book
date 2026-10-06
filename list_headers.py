import re
import sys

def get_header_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    headers = []
    for i, line in enumerate(lines):
        if line.startswith('#'):
            match = re.match(r'^(#+)\s+(.*)', line)
            if match:
                headers.append((i, len(match.group(1)), line.strip()))
    return headers

for chapter in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
    filepath = f"books/vol2/{chapter:02d}_*/{chapter:02d}_*.qmd"
    import glob
    files = glob.glob(filepath)
    if files:
        print(f"--- Chapter {chapter} ---")
        headers = get_header_lines(files[0])
        for h in headers:
            if h[1] <= 2:
                print(f"{h[0]}: {h[2]}")
