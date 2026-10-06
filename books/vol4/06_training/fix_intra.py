import re

with open("/private/tmp/MLSysBook-dev-notation/books/vol4/06_training/06_training.qmd", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if line.startswith("How much statistical confidence can twenty successful consecutive trials provide for a policy checkpoint"):
        continue # delete the last paragraph
    if "It analyzes an illustrative benchmark campaign beginning with twenty nominal door trials" in line:
        continue
    if "It then evaluates `latch-bc-recovery-02` through rigorous nominal regression" in line:
        continue
    
    # Fix remaining forward references in line
    line = re.sub(r' \(@sec-training-[a-zA-Z0-9_-]+\)', ' (detailed later)', line)
    line = re.sub(r'@sec-training-[a-zA-Z0-9_-]+', 'a later section', line)
    
    # Also fix @Sec-evaluation which is leftover
    line = re.sub(r'@Sec-[a-zA-Z0-9_-]+', 'A later chapter', line)
    
    new_lines.append(line)

with open("/private/tmp/MLSysBook-dev-notation/books/vol4/06_training/06_training.qmd", "w") as f:
    f.writelines(new_lines)
