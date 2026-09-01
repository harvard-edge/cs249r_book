#!/usr/bin/env python3
"""
Remove all ::: {.callout-question} ... ::: blocks from all .qmd files in the book.
"""
import glob
import os
import re

BOOK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

pattern = re.compile(r'::: \{\.callout-question\}[\s\S]*?:::\n*', re.MULTILINE)

qmd_files = glob.glob(os.path.join(BOOK_DIR, "**", "*.qmd"), recursive=True)

for qmd_path in qmd_files:
    with open(qmd_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "callout-question" in content:
        new_content = pattern.sub("", content)
        with open(qmd_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Cleaned callout-question from {os.path.relpath(qmd_path, BOOK_DIR)}")

print("All callout-question blocks removed.")
