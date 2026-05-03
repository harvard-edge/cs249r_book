import os
import re
import sys

directory = '/Users/VJ/GitHub/MLSysBook-notation-audit/book/quarto/contents/'
violations = 0

def check_file(filepath):
    global violations
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Iron Law efficiency
    matches = re.finditer(r'R_\{\\text\{peak\}\}\s*\\cdot\s*\\eta(?![_\{])', content)
    for match in matches:
        print(f"[Violation] Bare \\eta in Iron Law compute term: {filepath}")
        violations += 1

    # 2. Little's Law
    matches = re.finditer(r'L\s*=\s*\\lambda\s*\\cdot?\s*W', content)
    for match in matches:
        print(f"[Violation] Little's Law uses L and W: {filepath}")
        violations += 1
        
    # 3. Efficiency(N)
    matches = re.finditer(r'\\?t?e?x?t?\{?Efficiency\}?\(N\)', content)
    for match in matches:
        print(f"[Violation] Scaling Efficiency uses Efficiency(N): {filepath}")
        violations += 1

    # 4. Unit spacing
    matches = re.finditer(r'(`\{python\}[^`]+`)(ms|GB|MB|KB|TB|W|Gbps|percent)\b', content)
    for match in matches:
        print(f"[Violation] Missing space before unit {match.group(2)} after python chunk: {filepath}")
        violations += 1
        
    # 5. Hardware Balance
    matches = re.finditer(r'Hardware Balance \(\$B\$\)', content, re.IGNORECASE)
    for match in matches:
        print(f"[Violation] Hardware Balance uses $B$ instead of $B_{{hw}}: {filepath}")
        violations += 1
        
    # 6. Lowercase b for batch
    # We look for "batch size b" or "batch size of b"
    matches = re.finditer(r'batch size (?:of )?\$?b\$?\b', content, re.IGNORECASE)
    for match in matches:
        if match.group(0)[-1] == 'b' or match.group(0)[-2:] == 'b$': # basic filter
            print(f"[Violation] Lowercase b for batch size: {filepath}")
            violations += 1

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.qmd'):
            check_file(os.path.join(root, file))

if violations == 0:
    print("AUDIT PASSED: 0 violations found.")
    sys.exit(0)
else:
    print(f"AUDIT FAILED: {violations} violations found.")
    sys.exit(1)
