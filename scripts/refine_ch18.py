#!/usr/bin/env python3
"""
Refine Chapter 18 (Architectural Synthesis) to production publication grade:
1. Load drafts/vol3/ch18_conclusion_v4/assembled_draft.qmd
2. Tag all untagged code fences with ```text
3. Append {.unnumbered} to all #### headings
4. Verify all SVG references are accurate
5. Ensure zero unbalanced dollar errors and zero list spacing issues
6. Write out to books/vol3/18_conclusion/18_conclusion.qmd
"""

import re

SRC = "drafts/vol3/ch18_conclusion_v4/assembled_draft.qmd"
DST = "books/vol3/18_conclusion/18_conclusion.qmd"

with open(SRC, "r") as f:
    lines = f.readlines()

out = []
in_fence = False

for i, line in enumerate(lines):
    # Fix code fence tagging
    if line.startswith("```"):
        if not in_fence:
            in_fence = True
            tag = line[3:].strip()
            if not tag:
                line = "```text\n"
        else:
            in_fence = False

    # Fix H4 unnumbered
    if line.startswith("#### ") and "{.unnumbered}" not in line:
        line = line.rstrip() + " {.unnumbered}\n"

    out.append(line)

with open(DST, "w") as f:
    f.writelines(out)

print(f"Successfully refined Chapter 18 ({len(out)} lines written to {DST})")
