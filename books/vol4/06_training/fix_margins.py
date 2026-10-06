import re

file_path = "/private/tmp/MLSysBook-dev-notation/books/vol4/06_training/06_training.qmd"
with open(file_path, "r") as f:
    lines = f.readlines()

out_lines = []
skip = False
for line in lines:
    if "↪" in line and any(stage in line for stage in ["intervention stage", "placement stage", "enforcement stage", "verification stage"]):
        continue # delete the line
    out_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(out_lines)
