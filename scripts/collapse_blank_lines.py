import sys

for filepath in sys.argv[1:]:
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    blank = False

    for line in lines:
        if line.strip() == "":
            if not blank:
                new_lines.append(line)
            blank = True
        else:
            new_lines.append(line)
            blank = False

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
