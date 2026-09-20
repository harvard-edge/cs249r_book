#!/usr/bin/env python3
"""
Refine Chapter 15:
- Remove redundant ASCII draft blocks where SVGs exist
- Tag all untagged code blocks with ```text
- Ensure zero untagged code blocks
"""

from pathlib import Path
file_path = str(Path(__file__).resolve().parent.parent / "books/vol3/15_multi_agent/15_multi_agent.qmd")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines(keepends=True)
out = []
i = 0
n = len(lines)
in_block = False

while i < n:
    line = lines[i]
    if line.startswith("```"):
        tag = line.strip()[3:]
        if not in_block:
            in_block = True
            # peek inside block
            block_lines = []
            j = i + 1
            while j < n and not lines[j].startswith("```"):
                block_lines.append(lines[j])
                j += 1
            block_content = "".join(block_lines)

            # Check for redundant ASCII before capability-attenuation-tree
            if "Root Orchestrator (A_0)" in block_content and "Compiler Worker (A_2)" in block_content:
                # skip this block
                j += 1 # skip closing ```
                while j < n and lines[j].strip() == "":
                    j += 1
                i = j
                in_block = False
                continue
            else:
                if not tag:
                    out.append("```text\n")
                else:
                    out.append(line)
                i += 1
                continue
        else:
            in_block = False
            out.append(line)
            i += 1
            continue
    else:
        out.append(line)
        i += 1

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(out)

print("Updated Chapter 15.")
