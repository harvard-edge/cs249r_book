import yaml
import os

files_to_fix = [
    "interviews/vault/questions/edge/edge-0022.yaml",
    "interviews/vault/questions/edge/edge-0100.yaml",
    "interviews/vault/questions/edge/edge-0162.yaml",
    "interviews/vault/questions/edge/edge-0335.yaml",
    "interviews/vault/questions/edge/edge-0477.yaml",
    "interviews/vault/questions/tinyml/tinyml-0616.yaml"
]

def manual_fix_edge_0022():
    path = "interviews/vault/questions/edge/edge-0022.yaml"
    with open(path, 'r') as f:
        lines = f.readlines()
    # Line 14: realistic_solution: PoE delivers ...
    # It has a colon. Let's wrap it in |
    new_lines = []
    for line in lines:
        if line.startswith("  realistic_solution:"):
            content = line.split(":", 1)[1].strip()
            new_lines.append("  realistic_solution: |\n")
            new_lines.append(f"    {content}\n")
        else:
            new_lines.append(line)
    with open(path, 'w') as f:
        f.writelines(new_lines)

def manual_fix_edge_0100():
    path = "interviews/vault/questions/edge/edge-0100.yaml"
    with open(path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.startswith("  realistic_solution:"):
            content = line.split(":", 1)[1].strip()
            new_lines.append("  realistic_solution: |\n")
            new_lines.append(f"    {content}\n")
        else:
            new_lines.append(line)
    with open(path, 'w') as f:
        f.writelines(new_lines)

def manual_fix_edge_0162():
    path = "interviews/vault/questions/edge/edge-0162.yaml"
    with open(path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.startswith("  realistic_solution:"):
            content = line.split(":", 1)[1].strip()
            # It was single quoted with backslash-quote. Let's use |
            content = content.strip("'").replace("\\'", "'")
            new_lines.append("  realistic_solution: |\n")
            new_lines.append(f"    {content}\n")
        else:
            new_lines.append(line)
    with open(path, 'w') as f:
        f.writelines(new_lines)

def manual_fix_edge_0335():
    path = "interviews/vault/questions/edge/edge-0335.yaml"
    with open(path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.startswith("  realistic_solution:"):
            content = line.split(":", 1)[1].strip()
            new_lines.append("  realistic_solution: |\n")
            new_lines.append(f"    {content}\n")
        else:
            new_lines.append(line)
    with open(path, 'w') as f:
        f.writelines(new_lines)

def manual_fix_edge_0477():
    path = "interviews/vault/questions/edge/edge-0477.yaml"
    with open(path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.startswith("  realistic_solution:"):
            content = line.split(":", 1)[1].strip()
            content = content.strip("'").replace("\\'", "'")
            new_lines.append("  realistic_solution: |\n")
            new_lines.append(f"    {content}\n")
        else:
            new_lines.append(line)
    with open(path, 'w') as f:
        f.writelines(new_lines)

def manual_fix_tinyml_0616():
    path = "interviews/vault/questions/tinyml/tinyml-0616.yaml"
    with open(path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        # Line 33 and 34 start with - `
        if line.strip().startswith("- `"):
            new_lines.append(line.replace("- `", "- '`").replace(" finishes.", " finishes.'").replace(" wakeup.", " wakeup.'"))
        elif line.startswith("  realistic_solution:"):
            content = line.split(":", 1)[1].strip()
            content = content.strip("'")
            new_lines.append("  realistic_solution: |\n")
            new_lines.append(f"    {content}\n")
        else:
            new_lines.append(line)
    with open(path, 'w') as f:
        f.writelines(new_lines)

manual_fix_edge_0022()
manual_fix_edge_0100()
manual_fix_edge_0162()
manual_fix_edge_0335()
manual_fix_edge_0477()
manual_fix_tinyml_0616()
print("Surgically fixed the remaining 6 YAML load errors.")
